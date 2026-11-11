"""PLUS 全流程串联脚本(修正回测设计版)

一条命令跑完整条链:
    起止两期图 → expansion 扩张图 → leas 概率波段 → neighborhood_weight 权重
    → markov 需求 → cars 模拟 → validation Kappa → 汇总报告

回测设计(2026-10-08 修正):
    用起期/止期两期训练(如 2003/2010),markov 预测"验证年"(如 2013)的需求,
    cars 从止期图出发模拟到验证年,和验证年真实图算 Kappa。
    之前手动实验的错误:用了 2033 的预测需求却对比 2013 真实图,导致 Kappa 偏低。

用法:
    python run_pipeline.py \
      --early <起期图> --late <止期图> --real <验证年真实图> \
      --features <驱动因子文件夹> --output <输出目录> \
      --early-year 2003 --late-year 2005 --predict-year 2013 \
      [--policy <约束图>] [--matrix "行1;行2;..."]

注意:预测年必须满足"整数倍周期"规则(预测年 = 终止年 + N×(终止年−起始年)),
2003→2005 推 2013(步长 2,4 个周期)是有效示例;2003→2010 推 2013 会被拒绝。
"""

import os
import sys
import json
import argparse

import numpy as np

# 让本脚本能直接 import 同目录的其他模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import force_utf8, get_plus_backend, load_project
from run_expansion import run_expansion
from run_leas import run_leas
from run_neighborhood_weight import compute_weights
from run_markov import run_markov_demands, parse_markov_csv, format_cars_demands, valid_predict_years
from run_linear import run_linear, predict
from run_cars import run_cars
from run_validation import run_validation, read_metric

force_utf8()

# 需求方法的报告标签(WB 2026-11-05:real 曾被误标成 markov;GM(1,1) 已删,2026-11-09)
DEMAND_LABELS = {
    "markov": "markov 马尔可夫链",
    "linear": "linear 线性回归",
    "real": "真实验证年面积(诊断模式)",
}
# 注意:不再有硬编码的 DEFAULT_TRANSITION_MATRIX(WB 审查 P0-2:旧常量是
# "湿地体系"假设,与 CLCD 语义冲突)。矩阵单一来源 = transition_matrix 配置:
# "default" → build_default_matrix(按语义名锁森林/建设用地),或用户显式 N×N。

# 官方示例模板的"不可逆类别"(依据:PLUS 教程 Parameterfile/TransitionMatrix_0.csv
# 与教程 GUI 运行记录:落叶林行 0,1,0...、城镇用地行 0,0,0,1,0...、常绿林行只允许自转或转草地)
IRREVERSIBLE_CLASSES = {"森林", "落叶林", "常绿林", "建设用地", "城镇用地"}

# 需要提示"默认全开放"的水域类(不默认锁,只告警:锁水会改变已定稿的历史数字)
WATER_CLASSES = {"水域", "水体", "湿地", "湖泊", "水库"}


# 类名同义族:同一族内任意写法互认。
# 起因(2026-11-11):官方模板写死"建设用地",而 CLCD 重分类后的类名是"不透水面"
# → 默认矩阵没锁住建设用地(城市可退化成耕地),matrix_restrictions 写"建设用地"也报类别名不存在。
SYNONYM_GROUPS = (
    ("建设用地", "不透水面", "城镇用地", "城乡建设用地", "建成区", "人工表面", "城镇工矿用地"),
    ("森林", "林地", "落叶林", "常绿林", "针叶林", "阔叶林", "混交林", "有林地"),
    ("灌木", "灌丛", "灌木林", "疏林"),
    ("草地", "牧草地", "草原", "高覆盖度草地", "中覆盖度草地", "低覆盖度草地"),
    ("水域", "水体", "河流", "湖泊", "水库", "河渠", "水面"),
    ("耕地", "农田", "水田", "旱地", "农业用地", "水浇地"),
    ("裸地", "未利用地", "裸土", "裸岩", "沙地", "荒漠"),
    ("冰雪", "冰川", "雪地", "永久冰雪", "永久性冰川雪地"),
)


def _canon(name):
    """把类名归一化到同义族首名(族外原样返回),用于跨体系比对"""
    s = str(name).strip()
    for grp in SYNONYM_GROUPS:
        for alias in grp:
            if s == alias or s in _aliases(alias):
                return grp[0]
    return s


def _aliases(name):
    """类名别名集合:'建设用地/裸地'(0 像元类自动归并后的合并名)→ {'建设用地','裸地'}"""
    return {a.strip() for a in str(name).split("/") if a.strip()}


def _matches_any(name, targets):
    if _aliases(name) & set(targets):
        return True
    canon_targets = {_canon(t) for t in targets}
    return any(_canon(a) in canon_targets for a in _aliases(name))


def _class_index(names, target):
    """按别名族+同义族在类名列表里找目标类的下标,找不到返回 -1"""
    canon_target = _canon(target)
    for i, nm in enumerate(names):
        if target == nm or target in _aliases(nm):
            return i
        if any(_canon(a) == canon_target for a in _aliases(nm)):
            return i
    return -1


def build_default_matrix(class_names):
    """按官方示例模板生成 N×N 默认转换矩阵(内置默认,用户无需自己上传)

    模板哲学(官方示例归纳):森林、建设用地不可逆(行 = 仅对角 1);
    其余类别(耕地/草地/水域/裸地)全开放(行 = 全 1)。
    地区性小修改用 project.json 的 matrix_restrictions 做(开工前调查实情/政策后改)。

    类名匹配按别名:自动归并产生的 '建设用地/裸地' 同样能命中不可逆规则
    (2026-11-07 修:此前精确匹配,归并后类名带斜杠 → 静默退化成全开放)。
    """
    n = len(class_names)
    names = [str(class_names.get(str(i + 1), class_names.get(i + 1, str(i + 1))))
             for i in range(n)]
    rows = []
    locked = []
    for i, name in enumerate(names):
        if _matches_any(name, IRREVERSIBLE_CLASSES):
            row = ["0"] * n
            row[i] = "1"
            locked.append(name)
        else:
            row = ["1"] * n
        rows.append(",".join(row))
    matrix = ";".join(rows)
    print(f"  内置默认矩阵(官方模板,{n} 类):")
    for name, row in zip(names, rows):
        print(f"    {name}: {row}")
    if not locked:
        print("  ⚠️ 默认矩阵未锁任何类(全开放):分类体系里没有森林/建设用地等不可逆类")
        print("     若研究区确有林地/城镇,请检查类名是否为语义名(而非 类别N)")
    water_open = [nm for nm in names
                  if _matches_any(nm, WATER_CLASSES) and nm not in locked]
    if water_open:
        print(f"  ⚠️ 水域类默认全开放:{water_open} —— 水域可被转为任何类,")
        print("     大流域城市会因此出现水面凭空缩小;确需保护请在 matrix_restrictions 里禁转")
    return matrix


def apply_matrix_restrictions(matrix_str, class_names, restrictions):
    """对矩阵做针对性小修改(开工前调查该地实情/权威政策后的修改)

    restrictions: [{"from": "水域", "to": "建设用地", "allow": false}, ...]
    allow=true 表示强制允许(覆盖模板里的 0),false 表示禁止(覆盖模板里的 1)。
    修改原因必须写进 project.json 的 data_notes,保证可追溯。
    """
    rows = [r.split(",") for r in matrix_str.split(";")]
    names = [str(class_names.get(str(i + 1), class_names.get(i + 1, str(i + 1))))
             for i in range(len(rows))]

    for r in restrictions:
        f, t = r.get("from"), r.get("to")
        fi, ti = _class_index(names, f), _class_index(names, t)
        if fi < 0 or ti < 0:
            raise ValueError(f"限制规则类别名不在 class_names 里: {r}(可选: {names})")
        rows[fi][ti] = "1" if r.get("allow", True) else "0"
        print(f"  矩阵修改: {names[fi]}→{names[ti]} = {'允许' if rows[fi][ti] == '1' else '禁止'}")
    return ";".join(",".join(row) for row in rows)


def lock_classes(matrix_str, class_names, locked):
    """把指定类别整行锁死:只能自转,不能转成任何其他类(如水域锁定不转换)。

    locked: ["水域", ...] 类名(支持别名匹配);用于前期问答里用户选"水域锁定不转换"。
    修改依据写进 project.json 的 data_notes。
    """
    rows = [r.split(",") for r in matrix_str.split(";")]
    names = [str(class_names.get(str(i + 1), class_names.get(i + 1, str(i + 1))))
             for i in range(len(rows))]
    for name in locked:
        i = _class_index(names, name)
        if i < 0:
            raise ValueError(f"锁定类别名不在 class_names 里: {name}(可选: {names})")
        n = len(rows)
        rows[i] = ["0"] * n
        rows[i][i] = "1"
        print(f"  🔒 锁定类别: {names[i]}(只能自转,不转换)")
    return ";".join(",".join(row) for row in rows)


def step(title):
    """打印步骤分隔行"""
    print(f"\n{'=' * 60}\n【{title}】\n{'=' * 60}")


def run_pipeline(early, late, real, features, output_dir, early_year,
                 late_year, predict_year, policy="", matrix="default",
                 plus_backend=None, demand_method="markov", history=None,
                 history_years=None, interpolate=False, data_notes="",
                 validation_mode="out", diagnose=False, reuse_artifacts=False,
                 sequence=None, sim_start_early=False):
    """执行全流程,返回最终报告字典(任一步失败返回 None)

    demand_method:
        "markov" —— 马尔可夫链预测需求(引擎只支持"整数周期"年份,
                    预测年 = 终止年 + N×(终止年−起始年);配 interpolate=True 时,
                    目标年不在步长上会取前后可达年份的需求分段插值)
        "linear" —— 线性回归预测需求(任意年份均可,如 2028/2030;
                    公式与官方引擎手动结果逐系数一致,已验证)
    real:验证真实图;做未来预测(无真实图)时传 None,自动跳过 Kappa 验证。
    history:linear 模式可给全部历史图(按时间顺序),不给则用 [early, late]。
    validation_mode:
        "out"(默认)= 样本外:expansion/LEAS/权重用 early→late 训练,从 late 模拟到 predict。
        "in"= 样本内:expansion/LEAS/权重用 early→predict(全周期)训练,
              CARS 仍从 late 出发模拟到 predict,验证 predict 真实图。
              用途是证明"管线本身能复现变化"(官方 0.255 口径),不是论文主数字。
    diagnose:回测时额外跑一次"需求=真实面积"的诊断 CARS(复用已算出的概率/权重),
              用于把 FoM 损失拆成数量误差与位置误差。仅回测有效。
    """
    plus_backend = plus_backend or get_plus_backend()
    os.makedirs(output_dir, exist_ok=True)
    # 样本内/样本外必须产出到不同文件名:否则 run_leas 的"波段已存在就复用"会把
    # 另一口径的波段当成本口径结果(2026-11-07 武汉实测:in 口径直接复用了 out 的波段)
    tag = "_in" if (validation_mode == "in" and real) else ""
    base = os.path.join(output_dir, f"pipeline{tag}")

    # 训练终点:样本内口径用验证年(全周期),样本外口径用 late
    train_late = real if (validation_mode == "in" and real) else late
    if train_late != late:
        step(f"【样本内验证口径】扩张/概率/权重改用 {early_year}→{predict_year} 全周期训练")
        print(f"  训练终点 = {predict_year} 年图(CARS 仍从 {late_year} 年图出发)")

    # ⓪ 建模前预警:markov 引擎对"起始年 0 像元的类"会算 0/0=NaN 并污染全表
    # (2026-11-01 鄂州实测:裸地 2010 年 0 像元 → markov 全表 NaN)。跑任何步骤前先拦住,
    # 处理办法:用 reclassify_lulc.py 把小类合并进相邻类(如 裸地→耕地)。
    if demand_method == "markov":
        import rasterio as _rio
        with _rio.open(early) as _src:
            _vals, _ = np.unique(_src.read(1), return_counts=True)
        # 排除背景/无数据哨兵值:0(重分类约定)与 255(PLUS convert 模块的统一空值)
        present = {int(v) for v in _vals if int(v) not in (0, 255)}
        if present:
            missing = [c for c in range(1, max(present) + 1) if c not in present]
            if missing:
                print(f"❌ 起始年图 {os.path.basename(early)} 中 0 像元类别: {missing}")
                print("   markov 引擎会把 0/0 算成 NaN 并污染全部需求量(实测),建模中止")
                print("   处理:重分类时把这些小类合并进相邻类(如 裸地→耕地),或从分类体系中剔除")
                return None

    # ① expansion:起止两期 → 扩张图(样本内口径时终点 = 验证年)
    import glob as _glob
    reuse = bool(reuse_artifacts)
    expansion_out = base + "_expansion.tif"
    old_bands = sorted(_glob.glob(os.path.join(output_dir, f"DevPotential_bt{tag}_band_*.tif")))
    bands = None
    if reuse and os.path.exists(expansion_out) and old_bands:
        # 网格校验:复用前必须确认波段与模拟起点图逐像素对齐,否则整幅错位
        import rasterio as _r
        with _r.open(late) as _s0, _r.open(old_bands[0]) as _s1:
            if (_s0.transform, _s0.width, _s0.height) != (_s1.transform, _s1.width, _s1.height):
                print("⚠️ 已有概率波段与模拟起点图网格不一致,改为重训")
                reuse = False
    if reuse and os.path.exists(expansion_out) and old_bands:
        step("① ② 复用已算好的扩张图与发展概率(未重训 LEAS)")
        print(f"  复用:{expansion_out}")
        print(f"  复用:{len(old_bands)} 个概率波段(仅重跑需求 + CARS)")
        bands = old_bands
    else:
        step("① 提取土地扩张 expansion")
        expansion_out = run_expansion(early, train_late, expansion_out, plus_backend)
        if expansion_out is None:
            return None

        # 【导师 2026-11-10 实测】LEAS 按"扩张图最大类别编码"决定概率波段数:
        # 末位类在整个训练期从未有转入斑块 → 波段少 1 → CARS 类别数错位 → 0xC0000005 崩溃。
        # 官方《教程一》5.8 已记载此问题。这里只检测告警、不改数据:缺末位类必崩,提前拦下。
        import rasterio as _rio
        with _rio.open(expansion_out) as _s:
            _exp = {int(v) for v in np.unique(_s.read(1)) if int(v) not in (0, 255)}
        with _rio.open(late) as _s:
            _lulc = {int(v) for v in np.unique(_s.read(1)) if int(v) not in (0, 255)}
        _miss = sorted(_lulc - _exp)
        if _miss and max(_miss) == max(_lulc):
            print(f"❌ 扩张图缺末位类 {_miss}:LEAS 概率波段将少 1 个,CARS 会报 0xC0000005 崩溃")
            print("   官方解法(《教程一》5.8):重编码,把'有扩张的类'放末位,再重跑扩张提取;")
            print("   或确认扩张用了全周期(首年→末年),让末位类有机会保留少量转入斑块。")
            print("   详见 skill 包 FAQ.md F2。建模中止,处理后再跑。")
            return None
        if _miss:
            print(f"⚠️ 扩张图缺类别 {_miss}(非末位,不会崩溃):"
                  f"这些类得不到发展概率、模拟期基本不会扩张")

        # ② leas:扩张图 + 驱动因子 → 概率波段
        step("② LEAS 发展概率")
        prob_base = os.path.join(output_dir, f"DevPotential_bt{tag}.tif")
        bands = run_leas(expansion_out, features, prob_base, plus_backend)
        if bands is None:
            return None

    # ③ neighborhood_weight:扩张图 → 权重串
    step("③ 邻域权重")
    # 类数按扩张图实际编码推算(此前硬编码 7:0 像元类自动归并后只剩 6 类时,
    # 权重串长度与矩阵/概率波段数不一致 → CARS 参数单错位)
    import rasterio as _rio
    with _rio.open(expansion_out) as _s:
        _vals = np.unique(_s.read(1))
    n_cls = int(max([int(v) for v in _vals if int(v) not in (0, 255)] or [0]))
    if n_cls < 2:
        print(f"❌ 扩张图有效类别数异常(n_cls={n_cls})")
        return None
    weights, _, _ = compute_weights(expansion_out, n_cls)
    if weights is None:
        print("❌ 无法计算邻域权重(扩张图没有有效像元),停止")
        return None
    weight_str = ",".join(f"{w:.7f}" for w in weights)
    print(f"  权重串: {weight_str}")

    # ④ 需求预测:markov(整数周期)/ linear(任意年份)——官方教程两法
    # 抽成 _calc_demand(year):连续预测(sequence)要对多期分别求需求,公式只拟合一次
    hist_images = history or [early, late]
    hist_years = history_years or [early_year, late_year]
    _formulas = None
    if demand_method == "linear":
        step("④ 线性回归需求预测(任意年份)")
        # 外推步数必须以"历史序列最后一年"为基准,不能用 late(回测时 late 只是模拟起点):
        # 用 predict-late 会把外推年数多算 → 公式外推到负值 → 需求量全 0
        base_year = max(hist_years)
        n_ahead = predict_year - base_year
        if n_ahead < 0:
            print(f"❌ 预测年 {predict_year} 早于历史末年 {base_year}")
            return None
        print(f"  线性拟合:历史 {hist_years} → 公式(可外推任意年)")
        _formulas = run_linear(hist_images, n_ahead, plus_backend)
        if _formulas is None:
            return None
    elif demand_method == "real":
        step("④ 需求 = 真实验证年面积(诊断模式:只测空间分配,数量不作弊给模型)")
        if not real:
            print("❌ --demand real 只能用于回测(必须给验证年真实图)")
            return None

    def _calc_demand(year):
        """算 year 的需求量字符串;返回 None 表示失败"""
        nonlocal _formulas
        if demand_method == "linear":
            dm = predict(_formulas, year)
            vals = [dm[t] for t in sorted(dm)]
            # 任务10:下降类线性外推可能穿零(负需求),截断到 0(该类预测消失,不进入 CARS)
            if any(v < 0 for v in vals):
                neg_idx = [i + 1 for i, v in enumerate(vals) if v < 0]
                vals = [max(0, v) for v in vals]
                print(f"  ⚠️ 类别 {neg_idx} 线性外推穿零,负需求截断为 0(该类预测消失)")
            # 总量守恒校正(文献标准"数量校正"):引擎线性拟合与严格 OLS 有细微差异
            # (WB 固原实测 +1.2%),按有效像元总量比例缩放,保证 Σ需求 = 总面积
            import rasterio as _rio2
            with _rio2.open(late) as _s2:
                _v2, _c2 = np.unique(_s2.read(1), return_counts=True)
            _total = sum(int(c) for v, c in zip(_v2, _c2) if int(v) not in (0, 255))
            _sv = sum(vals)
            if _sv > 0:
                vals = [int(v * _total / _sv + 0.5) for v in vals]
            d = format_cars_demands(year, vals)
            print(f"  {year} 年需求(linear,守恒校正后): {d}")
        elif demand_method == "real":
            import rasterio as _rio
            with _rio.open(real) as _s:
                _vals, _cnt = np.unique(_s.read(1), return_counts=True)
            counts = {int(v): int(c) for v, c in zip(_vals, _cnt) if int(v) not in (0, 255)}
            n_cls = len(matrix.split(";"))
            vals = [counts.get(i + 1, 0) for i in range(n_cls)]
            d = format_cars_demands(year, vals)
            print(f"  {year} 年需求(真实面积诊断): {d}")
        else:  # markov
            result = run_markov_demands(early, late, early_year, late_year, year,
                                        plus_backend, interpolate=interpolate)
            if result is None or year not in result:
                print(f"❌ 预测年 {year} 不在 markov 引擎输出里(步长 {late_year - early_year} 年)")
                print(f"   连续预测请改用 --demand linear(任意年),markov 只支持整数周期")
                return None
            d = format_cars_demands(year, result[year])
            print(f"  {year} 年需求: {d}")
        return d

    demands = _calc_demand(predict_year)
    if demands is None:
        return None

    # CARS 模拟起点(用户 2026-11-11 定):回测口径下 LEAS 一律学全周期(in);
    # 出发年份由口径定——推荐口径从训练首年(2000)出发模拟整 20 年,
    # "中间年出发"对照口径仍从 late(2010)出发;预测从末年向后。
    sim_start = early if (validation_mode == "in" and real and sim_start_early) else late
    if sim_start != late:
        print(f"  CARS 起点 = {early_year} 年图(推荐口径:整 20 年模拟,训练/测试不分离)")

    # 矩阵-需求一致性检查(2026-11-04 重庆实测):矩阵锁死的类(行仅对角 1)若需求比现状少,
    # 引擎无法缩减它,该类会在邻域效应下继续扩张、挤占其他类配额(森林超发 245 万实况)
    def _warn_matrix_conflict(demand_str, current_map):
        try:
            import rasterio as _rio
            vals = [int(x) for x in demand_str.split(",")[1:]]
            with _rio.open(current_map) as _s:
                _vv, _cc = np.unique(_s.read(1), return_counts=True)
            cur = {int(a): int(b) for a, b in zip(_vv, _cc) if int(a) not in (0, 255)}
            rows = [r.split(",") for r in matrix.split(";")]
            for i, row in enumerate(rows):
                if row.count("1") == 1 and row[i] == "1" and i < len(vals):
                    if vals[i] < cur.get(i + 1, 0):
                        print(f"⚠️ 矩阵-需求矛盾:类别 {i + 1} 被矩阵锁死(不可转出),"
                              f"但需求 {vals[i]} < 当前 {cur.get(i + 1, 0)};"
                              f"引擎无法缩减锁死类,建议放开 matrix_restrictions 或接受偏差")
        except Exception:
            pass  # 一致性检查失败不阻断主流程

    _warn_matrix_conflict(demands, sim_start)

    # ④b 需求值守卫:0/负需求会让 CARS 空转并输出"看起来正常"的全 0 结果(2026-11-07 实测)
    _vals_dem = [int(x) for x in str(demands).split(",")[1:] if str(x).strip().lstrip("-").isdigit()]
    if _vals_dem:
        import rasterio as _rio2
        with _rio2.open(sim_start) as _s2:
            _v2, _c2 = np.unique(_s2.read(1), return_counts=True)
        _cur = {int(a): int(b) for a, b in zip(_v2, _c2) if int(a) not in (0, 255)}
        _tot_d, _tot_c = sum(_vals_dem), sum(_cur.values())
        if min(_vals_dem) < 0 or _tot_d <= 0:
            print(f"❌ 需求量异常(含负值):{demands}")
            print("   多为需求公式外推越界或历史期数不足;")
            print("   请改用 --demand markov,或补足历史年份。已中止,避免空转出假结果。")
            return None
        if _tot_c and abs(_tot_d - _tot_c) / _tot_c > 0.5:
            print(f"⚠️ 需求总量与模拟起点差异 {(_tot_d - _tot_c) / _tot_c:+.0%}"
                  f"({_tot_d:,} vs {_tot_c:,}),请确认需求方法适用性")

    # ⑤ cars:止期图 + 概率 + 权重 + 需求 + 矩阵 → 模拟目标年
    # 2026-11-10 机理改:连续预测的中间期排在"末年→目标年"之间,链式必须按时间
    # 升序(late → 中间期… → 目标年);目标年(链尾)的输出仍是主模拟产物名,
    # 下游报告/出图按 predict 年找主模拟不受影响。
    step("⑤ CARS 模拟")
    sim_base = os.path.join(output_dir, f"simulation_bt_{predict_year}.tif")
    chain = sorted([y for y in (sequence or []) if y < predict_year] + [predict_year])
    simulated = sim_start   # 链式起点 = 模拟起点图(每期从上一期结果继续)
    seq_sims = []
    for y in chain:
        if len(chain) > 1:
            step("⑤ CARS 模拟目标年(链尾)" if y == predict_year
                 else f"⑤+ 链式模拟 {y} 年(末年→目标年之间的中间期)")
        d_y = demands if y == predict_year else _calc_demand(y)
        if d_y is None:
            return None
        _warn_matrix_conflict(d_y, simulated)
        sim_base_y = (sim_base if y == predict_year
                      else os.path.join(output_dir, f"simulation_bt_{y}.tif"))
        sims_y = run_cars(simulated, bands, sim_base_y, policy, weight_str,
                          matrix, d_y, plus_backend)
        if sims_y is None:
            return None
        simulated = sims_y[0]
        seq_sims.append({"year": y, "sim": simulated, "demands": d_y})
    if len(seq_sims) > 1:
        print(f"  ✅ 连续预测完成,共 {len(seq_sims)} 期: "
              f"{', '.join(str(s['year']) for s in seq_sims)}(目标年 {predict_year} 在最后)")

    # ⑥ validation:模拟 vs 真实 → Kappa(未来预测没有真实图,跳过)
    kappa = overall = kappa_csv = metrics = None
    if real:
        step("⑥ Kappa/FoM 验证")
        # Python 主路径(WB 审查 P0-4:引擎 validate 会挂起);引擎版仅在显式要求时用
        from run_validation import python_metrics
        metrics = python_metrics(simulated, real, sim_start)
        kappa, overall = metrics["Kappa"], metrics["OA"]
        kappa_csv = None
        if os.environ.get("PLUS_VALIDATE_ENGINE", "").lower() == "plus":
            kappa_csv = run_validation(simulated, real, sim_start, plus_backend=plus_backend)
            if kappa_csv is None:
                print("⚠️ 引擎版验证失败,保留 Python 版结果")
        print(f"  Kappa = {kappa:.4f}  总体精度 = {overall:.2%}  FoM = {metrics['FoM']:.4f}")
        print(f"  变化四格:命中 A={metrics['A']:,} 漏报 B={metrics['B']:,} "
              f"假警报 C={metrics['C']:,} 错类 D={metrics['D']:,}(10% 采样口径)")
    else:
        step("⑥ 验证跳过(未来预测,无真实图可比)")

    # ⑥b 误差分解:需求换成真实验证年面积,只保留位置误差,再与主口径相减得数量误差
    diag = None
    if real and diagnose and demand_method != "real":
        step("⑥b 误差分解(诊断口径:需求 = 真实验证年面积)")
        import rasterio as _rio
        with _rio.open(real) as _s:
            _v, _c = np.unique(_s.read(1), return_counts=True)
        counts = {int(a): int(b) for a, b in zip(_v, _c) if int(a) not in (0, 255)}
        vals_diag = [counts.get(i + 1, 0) for i in range(n_cls)]
        demands_diag = format_cars_demands(predict_year, vals_diag)
        print(f"  诊断需求(真实面积):{demands_diag}")
        diag_stem = os.path.join(output_dir, f"simulation_diag_{predict_year}.tif")
        # 诊断 CARS 与主口径同起点(sim_start),否则"数量误差贡献"两个口径起点不同不可比
        sims_d = run_cars(sim_start, bands, diag_stem, policy, weight_str, matrix,
                          demands_diag, plus_backend)
        if sims_d:
            from run_validation import python_metrics
            m_d = python_metrics(sims_d[0], real, sim_start)
            diag = {"simulated": sims_d[0], "metrics": m_d,
                    "demands": demands_diag,
                    "fom_gain": m_d["FoM"] - metrics["FoM"],
                    "loc_loss": 1.0 - m_d["FoM"]}
            print(f"  诊断 FoM = {m_d['FoM']:.4f}(主口径 {metrics['FoM']:.4f})")
            print(f"  → 数量误差贡献 = {diag['fom_gain']:+.4f};位置误差损失 = {diag['loc_loss']:.4f}")
        else:
            print("⚠️ 诊断口径 CARS 失败,跳过误差分解")


    # ⑦ 报告
    step("⑦ 汇总报告")
    report = {
        "early": early, "late": late, "real": real,
        "early_year": early_year, "late_year": late_year,
        "predict_year": predict_year,
        "expansion": expansion_out, "bands": bands,
        "weights": weight_str, "demands": demands,
        "simulated": simulated, "kappa": kappa, "overall": overall,
        "metrics": metrics, "diagnose": diag,
        "validation_mode": validation_mode,
        "sequence": seq_sims,
        "n_classes": n_cls,
        "output_dir": output_dir,
    }
    report_path = os.path.join(output_dir, "报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        kind = "回测报告" if real else "预测报告"
        f.write(f"# PLUS 全流程{kind}({late_year}→{predict_year})\n\n")
        train_end = predict_year if validation_mode == "in" else late_year
        f.write(f"- 训练期:{early_year}–{train_end}"
                f"{'(含目标年,样本内)' if validation_mode == 'in' else ''},目标年:{predict_year}\n")
        f.write(f"- 需求预测:{DEMAND_LABELS.get(demand_method, demand_method)}\n")
        # 矩阵口径留痕:回测与预测必须同矩阵,否则精度报告无法校验预测配置
        import hashlib as _hl
        _rows = matrix.split(";")
        _locked = [i + 1 for i, r in enumerate(_rows) if r.split(",").count("1") == 1]
        _mh = _hl.md5(matrix.replace(" ", "").encode()).hexdigest()[:8]
        f.write(f"- 转移矩阵:行锁定类编码 {_locked}(仅自转),指纹 {_mh}\n")
        f.write(f"- 验证口径:{'样本内(扩张/概率用全周期训练)' if validation_mode == 'in' else '样本外(默认)'}\n")
        if kappa is not None:
            f.write(f"- **Kappa = {kappa:.4f},总体精度 = {overall:.2%},"
                    f"FoM = {metrics['FoM']:.4f}**\n\n")
        else:
            f.write("- 验证:无(未来预测,没有真实图可比)\n\n")
        # 误差分解:数量 vs 位置(仅回测 + 开了诊断口径时才有)
        if metrics and diag:
            f.write("## 误差分解(数量误差 vs 位置误差)\n\n")
            f.write("| 口径 | 需求量来源 | FoM | 说明 |\n|---|---|---|---|\n")
            f.write(f"| 主口径 | {DEMAND_LABELS.get(demand_method, demand_method)} | "
                    f"{metrics['FoM']:.4f} | 数量 + 位置误差并存 |\n")
            f.write(f"| 诊断口径 | 真实验证年面积 | {diag['metrics']['FoM']:.4f} | 数量完全正确,只剩位置误差 |\n\n")
            f.write(f"- **数量误差贡献 = {diag['fom_gain']:+.4f} FoM**"
                    f"(把需求换成真实面积后 FoM 的提升量)\n")
            f.write(f"- **位置误差损失 = {diag['loc_loss']:.4f}**"
                    f"(1 − 诊断 FoM;即使数量完全正确,位置也只能做对这么多)\n")
            f.write(f"- 诊断口径变化四格:命中 {diag['metrics']['A']:,} / 漏报 {diag['metrics']['B']:,} / "
                    f"假警报 {diag['metrics']['C']:,} / 错类 {diag['metrics']['D']:,}\n")
            f.write("- 两项贡献不严格相加等于 1−主口径 FoM(交叉项),作数量级参考\n\n")
        f.write("## 各步骤输出\n\n")
        f.write(f"1. 扩张图:{expansion_out}\n")
        f.write(f"2. 概率波段:{os.path.dirname(bands[0])}({len(bands)} 个 band)\n")
        f.write(f"3. 邻域权重:{weight_str}\n")
        f.write(f"4. {predict_year} 年需求:{demands}\n")
        f.write(f"5. 模拟图:{simulated}\n")
        if len(seq_sims) > 1:
            f.write("\n## 连续预测序列\n\n")
            f.write("| 年份 | 需求 | 模拟图 |\n|---|---|---|\n")
            for s in seq_sims:
                f.write(f"| {s['year']} | {s['demands']} | {s['sim']} |\n")
        if diag:
            f.write(f"6. 诊断口径模拟图:{diag['simulated']}\n")
        if kappa_csv:
            f.write(f"7. Kappa 报告:{kappa_csv}(已归档同目录)\n")
        if kappa is not None:
            f.write(f"\n> 注:Kappa/FoM 按 10% 有效像元随机采样计算,重复运行有 ±0.002 波动。\n")
        if data_notes:
            f.write(f"\n## 数据处理特殊情况说明\n\n{data_notes}\n")
    print(f"  ✅ 报告已保存: {report_path}")

    # 机器可读结果(run_full 的汇总报告 / 验收脚本读这个,避免解析 md)
    def _jsonable(o):
        if hasattr(o, "item"):
            return float(o)
        return str(o)
    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_jsonable)
    print(f"\n{'=' * 60}")
    print(f"  全流程完成!{late_year}→{predict_year} {kind}:")
    if kappa is not None:
        print(f"  Kappa = {kappa:.4f}  总体精度 = {overall:.2%}")
    else:
        print(f"  模拟图: {simulated}")
    print(f"{'=' * 60}")
    return report


def main():
    parser = argparse.ArgumentParser(description="PLUS 全流程串联(通用:任何研究区一套机制)")
    parser.add_argument("--project", default=None,
                        help="项目配置 JSON(数据路径/年份/矩阵/地类名;未给的参数从配置取默认值)")
    parser.add_argument("--early", default=None, help="训练起始年土地利用图")
    parser.add_argument("--late", default=None, help="训练终止年土地利用图(模拟起点)")
    parser.add_argument("--real", default=None, help="验证年真实土地利用图(回测用;未来预测可不给)")
    parser.add_argument("--features", default=None, help="驱动因子文件夹")
    parser.add_argument("--output", default=None, help="输出目录")
    parser.add_argument("--early-year", type=int, default=None, help="起始年份")
    parser.add_argument("--late-year", type=int, default=None, help="终止年份")
    parser.add_argument("--predict-year", type=int, default=None, help="验证/预测年份")
    parser.add_argument("--demand", default=None, choices=["markov", "linear", "real"],
                        help="需求预测方法:markov(整数周期)/ linear(任意年份,官方两法)/ real(诊断:真实验证年面积,只测分配)")
    parser.add_argument("--validation", default=None, choices=["out", "in"],
                        help="验证口径:out=样本外(默认,论文主数字);in=样本内(扩张/概率用全周期训练,证明管线正确性)")
    parser.add_argument("--diagnose", action="store_true",
                        help="回测时额外跑一次'需求=真实面积'的诊断 CARS,输出数量/位置误差分解")
    parser.add_argument("--reuse-artifacts", action="store_true",
                        help="复用输出目录里已有的扩张图与概率波段(只重跑需求+CARS,用于多需求口径对比)")
    parser.add_argument("--interpolate", action="store_true",
                        help="markov 模式:目标年不在步长上时,用前后可达年份的需求分段插值(如 2031 取 2030/2035 插值)")
    parser.add_argument("--history", nargs="+", default=None,
                        help="linear 模式的全部历史图(按时间顺序);不给则用 early+late 两期")
    parser.add_argument("--history-years", nargs="+", type=int, default=None,
                        help="linear 模式各历史图的年份(与 --history 一一对应)")
    parser.add_argument("--policy", default=None, help="policy 约束图(可空)")
    parser.add_argument("--matrix", default=None, help="转移矩阵,行间分号")
    parser.add_argument("--plus-backend", default=None, help="引擎目录(默认按配置自动解析)")
    args = parser.parse_args()

    # 项目配置提供默认值,命令行参数优先(通用化的核心:任何研究区一套机制)
    proj = load_project(args.project)
    def pick(attr, key):
        v = getattr(args, attr)
        return v if v is not None else proj.get(key)
    args.early = pick("early", "early")
    args.late = pick("late", "late")
    args.real = pick("real", "real")
    args.features = pick("features", "features")
    args.output = pick("output", "output_dir")
    args.early_year = pick("early_year", "early_year")
    args.late_year = pick("late_year", "late_year")
    args.predict_year = pick("predict_year", "predict_year")
    args.demand = pick("demand", "demand") or "markov"
    args.interpolate = args.interpolate or bool(proj.get("interpolate", False))
    args.validation = pick("validation", "validation_mode") or "out"
    args.diagnose = args.diagnose or bool(proj.get("diagnose", False))
    args.reuse_artifacts = args.reuse_artifacts or bool(proj.get("reuse_artifacts", False))
    args.sequence = proj.get("sequence")  # 连续预测追加年份列表(任务9.1)
    if args.demand == "real" and not args.real:
        parser.error("--demand real 只能用于回测(必须给 --real 验证年真值图)")
    args.history = args.history or proj.get("history")
    args.history_years = args.history_years or proj.get("history_years")
    args.policy = pick("policy", "policy") or ""
    args.matrix = pick("matrix", "transition_matrix") or "default"
    # 矩阵解析:支持 "default"(内置官方模板)、显式矩阵串;两者都可叠加针对性修改
    class_names = proj.get("class_names") or {}
    restrictions = proj.get("matrix_restrictions") or []
    if args.matrix == "default":
        if not class_names:
            parser.error("transition_matrix=default 需要项目配置提供 class_names")
        args.matrix = build_default_matrix(class_names)
    if restrictions:
        if not class_names:
            parser.error("matrix_restrictions 需要项目配置提供 class_names")
        args.matrix = apply_matrix_restrictions(args.matrix, class_names, restrictions)
    # 整类锁定(前期问答:水域是否锁定不转换)——在 restriction 之后叠加,锁死优先
    locked = proj.get("locked_classes") or []
    if locked:
        if not class_names:
            parser.error("locked_classes 需要项目配置提供 class_names")
        args.matrix = lock_classes(args.matrix, class_names, locked)
    for name in ("early", "late", "features", "output", "early_year", "late_year", "predict_year"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} 必填(或在 --project 配置里提供 {name})")
    if args.project:
        print(f"已加载项目配置: {os.path.abspath(args.project)}")

    # 整数周期规则前置检查(只约束 markov 且未开插值;linear 任意年份不检查)
    if args.demand == "markov" and not args.interpolate:
        step = args.late_year - args.early_year
        if step <= 0 or (args.predict_year - args.late_year) % step != 0:
            print(f"❌ 预测年 {args.predict_year} 不符合 markov 引擎规则")
            print(f"  规则:预测年 = 终止年 + N×(终止年−起始年)")
            valid = [args.late_year + n * step for n in range(6)]
            print(f"  当前起止 {args.early_year}→{args.late_year}(步长 {step})可预测:{valid}")
            print(f"  要预测任意年份:加 --interpolate(分段插值)或 --demand linear")
            sys.exit(1)

    # 路径全部转绝对路径(引擎工作目录是 plus-backend,相对路径会写错地方)
    args.early = os.path.abspath(args.early)
    args.late = os.path.abspath(args.late)
    if args.real:
        args.real = os.path.abspath(args.real)
    args.features = os.path.abspath(args.features)
    args.output = os.path.abspath(args.output)
    if args.policy:
        args.policy = os.path.abspath(args.policy)
    if args.history:
        args.history = [os.path.abspath(p) for p in args.history]

    if run_pipeline(args.early, args.late, args.real, args.features, args.output,
                    args.early_year, args.late_year, args.predict_year,
                    args.policy, args.matrix, args.plus_backend,
                    demand_method=args.demand, history=args.history,
                    history_years=args.history_years,
                    interpolate=args.interpolate,
                    data_notes=proj.get("data_notes", ""),
                    validation_mode=args.validation,
                    diagnose=args.diagnose,
                    reuse_artifacts=args.reuse_artifacts,
                    sequence=args.sequence,
                    sim_start_early=bool(proj.get("sim_start_early", False))) is None:
        sys.exit(1)  # 失败给非零退出码,调用方(脚本/AI)靠它判断成败


if __name__ == "__main__":
    main()
