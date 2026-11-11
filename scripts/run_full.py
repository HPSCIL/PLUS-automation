"""plus-automation 一站式全流程:从"一句话"到"精度报告"

一条命令串起全部环节:
    ① 下载土地利用(CLCD,行政区名/GeoJSON 边界,分省包兜底,全国网格对齐)
    ② 下载驱动因子(官方同款 17 项,自动对齐到参考栅格)
    ③ 语义重分类(内置官方图例 + 显式映射;0 像元类自动成因分析)
    ④ 检查 + 转换(PLUS convert,编码连续校验)
    ⑤ 建模(expansion → LEAS → 需求 → CARS → Kappa)
    ⑥ FoM 验证 + 汇总报告

用法:
    python run_full.py --region 鄂州市 --years 2010,2015,2020 --predict 2020   # 回测
    python run_full.py --region 鄂州市 --years 2010,2015,2020 --predict 2030   # 预测
    python run_full.py --region 鄂州市 --years 2010,2015,2020 --predict 2030 --mapping 映射.json

设计约定(与三城验收一致):
    - 回测:predict == 末年 → early=首年, late=次年, real=末年(默认矩阵)
    - 预测:predict > 末年 → early=首年, late=末年, real 空(默认矩阵)
    - 需求:官方两法——2 期 → markov(非整周期自动插值);≥3 期 → linear
    - 矩阵:官方模板(森林/建设用地不可逆);区域化修改由 AI/用户编辑生成的
      project.json(matrix_restrictions)后再确认
"""

import os
import sys
import json
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import force_utf8

force_utf8()

PY = sys.executable
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), "plus_skill_GIS_data")  # 默认输出根目录 = 用户主目录下(跨机通用)


def run(cmd, timeout=14400):
    """执行子命令,失败退出(带完整输出)

    子命令输出只在控制台打印尾部 2000 字符(便于阅读),完整输出落盘到
    <data_dir>/logs/,避免矩阵提示/归并依据/引擎告警这类关键信息被截断丢掉。
    """
    print(f"\n▶ {' '.join(cmd[:3])} ...")
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if LOG_DIR:
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            name = os.path.splitext(os.path.basename(cmd[1] if len(cmd) > 1 else "step"))[0]
            stamp = time.strftime("%H%M%S")
            with open(os.path.join(LOG_DIR, f"{stamp}_{name}.log"), "w",
                      encoding="utf-8") as f:
                f.write(" ".join(cmd) + "\n" + "=" * 60 + "\n" + out)
        except OSError:
            pass
    print(out[-2000:])
    if r.returncode != 0:
        print(f"❌ 命令失败(exit {r.returncode}): {' '.join(cmd[:4])}")
        sys.exit(1)
    return out


def resolve_folder(region):
    """研究区文件夹名:与 download_lulc 约定一致(中文名→adcode,纯 ASCII→原名)

    旧实现靠"扫描输出根目录里含 lulc_ 的子目录"反推,多研究区共存时会取到
    最后一个匹配目录(可能是别的研究区),导致下游读到错研究区的数据。
    这里改为按研究区确定性推导(WorkBuddy 2026-11-05 修复)。
    """
    if os.path.isfile(region):
        name = os.path.splitext(os.path.basename(region))[0]
        return name if all(ord(c) < 128 for c in name) else "region"
    cache_dir = os.path.join(SCRIPTS, "cache")
    adcode = ""
    try:
        from download_lulc import _cached_adcode, resolve_adcode, _full_adcode
        adcode = _cached_adcode(region, cache_dir)
        if not adcode:
            adcode = _full_adcode(resolve_adcode(region) or "")
    except Exception as e:
        print(f"  ⚠️ 行政区划解析失败({e})")
    return adcode if (adcode and any(ord(c) > 127 for c in region)) else region


def script(name):
    return os.path.join(SCRIPTS, name)


# ---------------- 阶段计时(汇总报告要写"每步输入/输出/耗时") ----------------
TIMING = []
LOG_DIR = None  # 子命令完整日志目录(定位到 data_dir 后设置)


def tick(label, t0, inputs="", outputs=""):
    import time
    dt = time.time() - t0
    TIMING.append({"stage": label, "seconds": dt, "inputs": inputs, "outputs": outputs})
    print(f"⏱ {label} 用时 {dt/60:.1f} min")
    return time.time()


# ---------------- 面积统计 CSV(逐类 km²) ----------------
def write_area_stats(data_dir, out_name, class_names, years, predict, sim, seq_sims):
    """生成逐类面积统计 CSV(单位 km²):历史各年真实 + 主模拟 + 连续预测各期。

    列 = 地类 + [各历史年, 主模拟年, 各追加模拟年];末行合计;末列标单位。
    面积 = 有效像元数 × 像元面积(从栅格 res 反推,不写死 30m)。
    """
    import rasterio
    import numpy as np
    import csv

    def _counts(path):
        with rasterio.open(path) as s:
            a = s.read(1)
            px_km2 = abs(s.res[0]) * abs(s.res[1]) / 1e6
            vals, cnts = np.unique(a, return_counts=True)
        return {int(k): int(n) for k, n in zip(vals, cnts) if int(k) not in (0, 255)}, px_km2

    cols = []  # [(列名, {code: 像元数}, 每像元 km²)]
    for y in years:
        p = os.path.join(data_dir, f"lulc_{y}_rc_uc.tif")
        if os.path.exists(p):
            c, px = _counts(p)
            cols.append((str(y), c, px))
    if os.path.exists(sim):
        c, px = _counts(sim)
        cols.append((f"{predict}模拟", c, px))
    for s in seq_sims:
        if s["year"] == predict:
            continue
        if os.path.exists(s["sim"]):
            c, px = _counts(s["sim"])
            cols.append((f"{s['year']}模拟", c, px))

    codes = sorted({c for _, cnt, _ in cols for c in cnt})
    out = os.path.join(data_dir, out_name, "面积统计.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["地类"] + [name for name, _, _ in cols])
        for code in codes:
            name = class_names.get(str(code), f"类别{code}")
            w.writerow([name] + [round(cnt.get(code, 0) * px, 2) for _, cnt, px in cols])
        w.writerow(["合计"] + [round(sum(cnt.values()) * px, 2) for _, cnt, px in cols])
    print(f"✅ 面积统计(km²): {out}")
    return out


# ---------------- 0 像元类自动归并(老师规则:有规则匹配就自动执行) ----------------
def parse_zero_advice(rc_out):
    """从 reclassify_lulc 的成因分析输出里解析 {类编码: (建议并入语义名, 依据)}

    输出格式(analyze_zero_classes):
        - 类别 6:1234 个像元位于 (114.8E, 30.4N)
          轨迹:起始年为 {...},末期 {...}
          成因: ...
          ✅ 建议: 并入建设用地
          依据: ...
    """
    import re
    blocks = re.split(r"\n(?=- 类别 \d+:)", rc_out)
    advice = {}
    for b in blocks:
        m = re.match(r"- 类别 (\d+):", b.strip())
        if not m:
            continue
        cls = int(m.group(1))
        s = re.search(r"✅ 建议: 并入(\S+)", b)
        r = re.search(r"依据: (.+)", b)
        if s:
            advice[cls] = (s.group(1).strip(), (r.group(1).strip() if r else "内置规则"))
    return advice


# 规则建议里的类名与数据图例可能不同名(CLCD 的城镇类叫"不透水面",规则写"建设用地")
SEMANTIC_ALIASES = {
    "建设用地": ("不透水面", "建设用地", "城镇用地", "建成区", "人工表面"),
    "耕地": ("耕地", "水田", "旱地", "农田"),
    "草地": ("草地", "牧草地", "稀疏草地"),
    "森林": ("森林", "林地", "落叶林", "常绿林"),
}


def resolve_target(target_sem, keys):
    """规则给出的类名 → 映射里实际存在的类名(先精确,再按别名族匹配)"""
    if target_sem in keys:
        return target_sem
    for cand in SEMANTIC_ALIASES.get(target_sem, ()):
        if cand in keys:
            return cand
    return None


def auto_merge_zero_classes(rc_out, mapping_path, mapping, data_dir):
    """起始年 0 像元类:规则命中就自动归并并重跑重分类;规则未命中返回 None

    多个 0 像元类必须基于**原始编码**一次性归并(逐个归并会重排编码,
    第二个类的编码已经变了 → 会归并到错的类)。返回 (新 mapping, 说明行列表)。
    """
    advice = parse_zero_advice(rc_out)
    if not advice:
        return None, ["成因分析输出里没有可解析的'✅ 建议',需人工处置"]
    orig = {k: int(v) for k, v in mapping.items() if not str(k).startswith("_")}
    alias, logs = {}, []
    for cls, (target_sem, basis) in sorted(advice.items()):
        resolved = resolve_target(target_sem, orig.keys())
        if resolved is None:
            return None, [f"类别 {cls} 建议并入 {target_sem},但映射里没有同类"
                          f"({', '.join(orig.keys())})"] + logs
        target_code = orig[resolved]
        if target_code in advice:
            return None, [f"类别 {cls} 建议并入的 {resolved} 本身也是 0 像元类,规则冲突"] + logs
        alias[cls] = target_code
        merged_sems = [s for s, c in orig.items() if c == cls]
        note = f"类别{cls}({'/'.join(merged_sems)}) → {resolved}"
        if resolved != target_sem:
            note += f"(规则建议'{target_sem}',映射中对应'{resolved}')"
        logs.append(f"{note};依据:{basis}")
        print(f"  ✅ 自动归并 {note}(依据:{basis})")
    # 一次性重映射 + 编码重排为 1..n
    merged = {s: alias.get(c, c) for s, c in orig.items()}
    codes = sorted(set(merged.values()))
    remap = {c: i + 1 for i, c in enumerate(codes)}
    merged = {s: remap[c] for s, c in merged.items()}
    # 备份原映射 + 写回(带 _notes,reclassify 会写进 重分类说明.md)
    bak = mapping_path.replace(".json", "_before_merge.json")
    if os.path.exists(mapping_path) and not os.path.exists(bak):
        with open(mapping_path, encoding="utf-8") as f:
            old = f.read()
        with open(bak, "w", encoding="utf-8") as f:
            f.write(old)
        print(f"  原映射已备份:{bak}")
    out = dict(merged)
    out["_notes"] = "0 像元类自动归并(规则命中,非人工):\n" + "\n".join(logs)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return merged, logs


def main():
    # ⓪ 环境自检(缺依赖/引擎时给修复指引并退出,不带着问题往下跑)
    from common import env_check
    if not env_check():
        sys.exit(1)

    parser = argparse.ArgumentParser(description="plus-automation 一站式全流程")
    parser.add_argument("--region", required=True, help="行政区划名(如 鄂州市)或 GeoJSON 边界路径")
    parser.add_argument("--years", required=True, help="逗号分隔年份,≥2 期,如 2010,2015,2020")
    parser.add_argument("--predict", type=int, required=True, help="预测/验证目标年份")
    parser.add_argument("--output", default=None, help="输出根目录(默认 D:\\plus_skill_GIS_data\\<研究区>)")
    parser.add_argument("--mapping", default=None, help="重分类映射 JSON(不给则按类别并集自动生成恒等映射)")
    parser.add_argument("--demand", default="auto",
                        choices=["auto", "markov", "linear", "real"],
                        help="需求方法:auto=2期markov,≥3期linear(官方两法);"
                             "real=真实验证年面积(仅回测,诊断口径,不作预测报告)")
    parser.add_argument("--validation", default="out", choices=["out", "in"],
                        help="验证口径:out=样本外(默认,论文主数字);"
                             "in=样本内(扩张/概率/权重用全周期训练,证明管线正确性)")
    parser.add_argument("--diagnose", dest="diagnose", action="store_true", default=True,
                        help="回测时自动附误差分解(数量/位置),默认开")
    parser.add_argument("--no-diagnose", dest="diagnose", action="store_false",
                        help="关闭误差分解(省一次 CARS 时间约 30%%)")
    parser.add_argument("--threads", type=int, default=4, help="下载线程数")
    parser.add_argument("--data-dir", default=None,
                        help="直接指定已有数据目录(复用缓存,跳过文件夹推导与重复下载)")
    parser.add_argument("--lulc-source", default="clcd", choices=["clcd", "worldcover"],
                        help="土地利用数据源:clcd(30m 年度,默认)/ worldcover"
                             "(ESA 10m,仅 2020/2021 两期)")
    parser.add_argument("--lulc-res", type=int, default=30, choices=[10, 30],
                        help="土地利用图分辨率:30m(默认)/ 10m(仅 WorldCover,"
                             "同区域像元数 ≈ 9 倍,建模慢数倍,适合小区域)")
    parser.add_argument("--drivers-dir", default=None,
                        help="用户自供驱动因子文件夹(跳过自动下载;df_*.tif 命名,"
                             "网格/投影不一致自动重采样对齐,体检通过后复制进任务目录)")
    parser.add_argument("--matrix-restrictions", default=None,
                        help="区域化矩阵修改(JSON 数组或 .json 文件路径),如 "
                             "'[{\"from\":\"水域\",\"to\":\"建设用地\",\"allow\":false,"
                             "\"依据\":\"长江保护法第25条\"}]';由 AI 检索研究区信息后自动生成")
    parser.add_argument("--resume", action="store_true",
                        help="(断点已自动生效,此参数保留仅为兼容旧命令)强制重跑某阶段请"
                             "删除 data_dir 下 .stage_*.done")
    parser.add_argument("--reuse-model", action="store_true",
                        help="复用已有扩张图与概率波段,只重跑需求+CARS(多需求口径对比时用,省 LEAS 时间)")
    parser.add_argument("--osm", default="auto", choices=["auto", "overpass", "pbf"],
                        help="OSM 渠道")
    parser.add_argument("--sequence", default="none",
                        help="连续预测间隔(年):末年之后每隔 N 年一期,排到目标年,目标年总是"
                             "最后一期(2020→2030 --sequence 5 → 2025、2030;间隔不整除时"
                             "中间期取整、目标年补最后);none=只出目标年一张图(默认)")
    parser.add_argument("--backtest", default="none", choices=["none", "real", "predict", "train"],
                        help="回测选择:none=默认;real=验证年真实面积诊断(推荐);"
                             "predict=验证年预测面积回测;train=训练期年份(全周期)")
    parser.add_argument("--water", default="open", choices=["open", "lock"],
                        help="水域是否锁定不转换:open=不锁定(自然发展,默认);"
                             "lock=锁定(水域只能自转,不转出,保护大江大河湖泊水库)")
    args = parser.parse_args()

    years = sorted(int(y) for y in args.years.split(",") if y.strip())
    if len(years) < 2:
        print("❌ --years 至少两期")
        sys.exit(1)
    if args.predict < years[-1]:
        print(f"❌ --predict {args.predict} 不能早于最后一年 {years[-1]}")
        sys.exit(1)
    if args.lulc_source == "worldcover" and any(y not in (2020, 2021) for y in years):
        print("❌ WorldCover 只有 2020(v100)/2021(v200)两期;"
              "需要更早年份请改用 --lulc-source clcd")
        sys.exit(1)
    if args.lulc_res == 10 and args.lulc_source != "worldcover":
        print("❌ 10m 分辨率只有 ESA WorldCover 提供(仅 2020/2021);"
              "请用 --lulc-source worldcover --lulc-res 10")
        sys.exit(1)
    if args.lulc_res == 10:
        print("⚠️ 10m 分辨率:同区域像元数 ≈ 30m 的 9 倍,CARS 耗时数倍,仅建议小区域使用")
    # 数据源降级说明(① 下载时可能被赋值:CLCD 失败 → WorldCover 顶替),重分类说明读取后合并
    downgrade_note = ""

    # 连续预测序列(2026-11-10 机理改):中间期 = 末年之后每隔 interval 年一期,
    # 一直排到目标年;目标年 = 主模拟年,是序列最后一期(不在 sequence 列表里,
    # 主模拟自身就是它)。间隔不整除时中间期取整、目标年补在最后。
    # 例:2020→2030 间隔 5 → 中间期 [2025],序列 [2025, 2030];间隔 3 → [2023, 2026, 2029, 2030]。
    sequence = None
    if args.sequence and args.sequence.lower() != "none":
        try:
            interval = int(args.sequence.strip())
        except Exception:
            print("❌ --sequence 应为一个间隔年数(如 5);旧版'间隔,份数'格式已废弃"
                  "(2026-11-10 机理改为:末年→目标年之间排期,目标年总是最后一期)")
            sys.exit(1)
        if interval <= 0:
            print("❌ --sequence 间隔必须 > 0")
            sys.exit(1)
        if args.predict <= years[-1]:
            print("❌ --sequence 仅用于预测(目标年必须晚于末年);回测请用 --backtest")
            sys.exit(1)
        late = years[-1]
        span = args.predict - late
        mids = [late + interval * k for k in range(1, span // interval + 1)
                if late + interval * k < args.predict]
        if not mids:
            print(f"  ⚠️ 间隔 {interval} ≥ 跨度 {span} 年,中间期为空 → 等价于只出目标年单期")
        else:
            sequence = mids
            print(f"  连续预测序列:{mids + [args.predict]}"
                  f"(间隔 {interval} 年,共 {len(mids) + 1} 期,目标年 {args.predict} 在最后)")

    # 任务9.3 回测选择:映射到现有 demand/validation 机制
    if args.backtest == "real":
        if args.predict != years[-1]:
            print("❌ --backtest real 需要目标年 = 末年(回测)")
            sys.exit(1)
        args.demand = "real"
        args.diagnose = False
        # 用户 2026-11-11 定:真实面积口径 LEAS 也学全周期(与推荐口径一致),
        # 只是 CARS 从中间年(late,如 2010)出发——与推荐口径对照出发年份的影响
        args.validation = "in"
        print("  --backtest real:真实面积、CARS 从中间年出发(LEAS 学全周期,对照口径)")
    elif args.backtest == "train":
        if args.predict != years[-1]:
            print("❌ --backtest train 需要目标年 = 末年(回测)")
            sys.exit(1)
        args.validation = "in"
        # 用户 2026-11-11 定:推荐的全周期回测 = LEAS 用全周期 + CARS 从首年图出发、
        # 用真实验证年面积(数量不作弊)模拟整 20 年窗口;主口径即真实面积,不再跑第二次诊断
        args.demand = "real"
        args.diagnose = False
        print("  --backtest train:全周期回测(推荐口径)——LEAS 用全周期训练,"
              "CARS 从首年图出发、用真实面积模拟到验证年(训练/测试不分离)")
    elif args.backtest == "predict":
        # 预测法推 A = 现有回测默认(early=首年, late=次年, 预测法推末年)
        if args.predict != years[-1]:
            print("❌ --backtest predict 需要目标年 = 末年(回测)")
            sys.exit(1)
        # 用户 2026-11-11 定:预测面积口径同样学 20 年全周期、CARS 从首年(2000)出发,
        # 与推荐口径只差"面积来源"(预测 vs 真实),论文主数字口径
        args.validation = "in"
        print("  --backtest predict:预测面积、从首年出发(全周期学 20 年,论文主数字口径)")

    # 数据目录:纯 ASCII(PLUS 下游对路径编码敏感),中文研究区按 adcode 命名
    # P1-1 修复(WB 审查):download_lulc 的 --output 是"最终数据目录"而不是根目录,
    # 必须先推导文件夹名、拼出完整目录再传参,否则数据会散落在根目录且下游找不到
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
        os.makedirs(data_dir, exist_ok=True)
    else:
        out_root = args.output or DEFAULT_ROOT
        folder = resolve_folder(args.region)
        data_dir = os.path.join(out_root, folder)
        # ① 土地利用下载。30m 渠道双数据源(用户 2026-11-10 定):
        # 主 = CLCD;CLCD 失败且年份恰为 2020/2021 → 自动降级 WorldCover 10m 重采样 30m 顶替
        print("=" * 60 + "\n【① 土地利用下载】\n" + "=" * 60)
        t1 = time.time()
        cmd = [PY, script("download_lulc.py"), "--region", args.region,
               "--years", args.years, "--output", data_dir,
               "--threads", str(args.threads),
               # download_lulc 的渠道名是 ncdc-ftp/http/worldcover;"clcd" 只是本脚本对外的
               # 选项名(2026-11-10 修:WorldCover 改造后默认 30m 主渠道被传成 --source clcd,
               # argparse 直接报 invalid choice,导致默认路径完全不可用)
               "--source", "ncdc-ftp" if args.lulc_source == "clcd" else args.lulc_source,
               "--res", str(args.lulc_res)]
        if args.lulc_source == "clcd":
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=14400)
            print((r.stdout or "")[-1500:] + (r.stderr or "")[-500:])
            if r.returncode != 0:
                if all(y in (2020, 2021) for y in years):
                    print("⚠️ CLCD 下载失败;自动降级:ESA WorldCover 10m 重采样 30m 顶替")
                    print("   (WorldCover 仅 2020/2021 两期,本次年份恰好可用)")
                    args.lulc_source = "worldcover"  # 后续重分类/报告全按 worldcover 走
                    downgrade_note = ("【数据源降级】CLCD 下载失败,自动改用 ESA WorldCover "
                                      "10m 重采样 30m(该产品仅 2020/2021 两期)。\n")
                    run([PY, script("download_lulc.py"), "--region", args.region,
                         "--years", args.years, "--output", data_dir,
                         "--threads", str(args.threads), "--source", "worldcover",
                         "--res", "30"])
                else:
                    print("❌ CLCD 下载失败;WorldCover 只有 2020/2021 两期,"
                          f"无法顶替所需年份 {years}")
                    print("   请检查网络后重试,或用 --manual 手动提供数据")
                    sys.exit(1)
        else:
            run(cmd)
        tick("① 土地利用下载", t1, inputs=args.region,
             outputs=f"lulc_{{{','.join(str(y) for y in years)}}}.tif")
    global LOG_DIR
    LOG_DIR = os.path.join(data_dir, "logs")
    print(f"  数据目录: {data_dir}")

    # 断点标记:耗时段(①②⑤)完成后写 .stage_N.done,**自动生效**(2026-11-10 用户定:
    # 不再要求记着加 --resume;导师曾实测"有标记但没加参数 → 重新联网下载卡死")。
    # 关键:标记内容存"本次配置指纹",配置(需求方法/矩阵/年份/验证口径)一变就失效重跑,
    # 否则换个 --demand 仍会跳过建模,拿到的还是上一次的结果(2026-11-07 实测踩坑)。
    # 想强制重跑某阶段:删除 data_dir 下对应 .stage_*.done 文件。
    def stage_done(n, key=""):
        p = os.path.join(data_dir, f".stage_{n}.done")
        if not os.path.exists(p):
            return False
        if not key:
            print(f"  ♻️ 断点标记 {os.path.basename(p)} 存在,自动跳过本阶段(强制重跑请删除该文件)")
            return True
        with open(p, encoding="utf-8") as f:
            old = f.read().strip()
        # 路径比较要归一化:斜杠方向/大小写差异不该让断点失效
        match = os.path.normcase(os.path.abspath(old)) == os.path.normcase(os.path.abspath(key)) \
            if os.sep in old or "/" in old else old == key
        if match:
            print(f"  ♻️ 断点标记 {os.path.basename(p)} 配置指纹匹配,自动跳过本阶段"
                  f"(强制重跑请删除该文件)")
        else:
            print(f"  🔄 断点标记 {os.path.basename(p)} 配置指纹不匹配(参数已变),本阶段重跑")
        return match

    def mark_done(n, key="ok"):
        with open(os.path.join(data_dir, f".stage_{n}.done"), "w", encoding="utf-8") as f:
            f.write(key)

    # ② 驱动因子:自动下载官方同款 17 项 / 用户自供文件夹体检复制(--drivers-dir)
    t2 = time.time()
    ref_tif = os.path.join(data_dir, f"lulc_{years[0]}.tif")  # 网格检查也要用,--resume 时不能缺
    drivers_dir = os.path.join(data_dir, "drivers")
    if args.drivers_dir:
        print("=" * 60 + "\n【② 驱动因子(用户自供,体检+复制,不下载)】\n" + "=" * 60)
        run([PY, script("download_drivers.py"), "--region", args.region,
             "--ref", ref_tif, "--output", drivers_dir,
             "--use-existing", os.path.abspath(args.drivers_dir)])
        mark_done("2", ref_tif)
        tick("② 驱动因子(用户自供)", t2, inputs=args.drivers_dir,
             outputs=f"{len([x for x in os.listdir(drivers_dir) if x.startswith('df_')])} 个 df_*.tif")
    elif not stage_done("2", ref_tif):
        print("=" * 60 + "\n【② 驱动因子下载(官方同款 17 项)】\n" + "=" * 60)
        run([PY, script("download_drivers.py"), "--region", args.region,
             "--ref", ref_tif, "--output", drivers_dir, "--osm", args.osm])
        mark_done("2", ref_tif)
        tick("② 驱动因子下载", t2, inputs=f"参考图 {os.path.basename(ref_tif)}",
             outputs=f"{len([x for x in os.listdir(drivers_dir) if x.startswith('df_')])} 个 df_*.tif")
    else:
        drivers_dir = os.path.join(data_dir, "drivers")
        print("♻️ 阶段②已跳过(断点自动生效)")

    # ③ 重分类(自动映射:类别并集→连续编码;0 像元类自动成因分析)
    t3 = time.time()
    print("=" * 60 + "\n【③ 语义重分类】\n" + "=" * 60)
    if args.mapping:
        mapping_path = os.path.abspath(args.mapping)
    else:
        # 自动映射:各期类别并集 → 连续编码(语义名取内置 CLCD 图例)
        import rasterio as _rio
        import numpy as _np
        from reclassify_lulc import BUILTIN_LEGENDS
        union = set()
        for y in years:
            with _rio.open(os.path.join(data_dir, f"lulc_{y}.tif")) as s:
                vals = _np.unique(s.read(1))
            union |= {int(v) for v in vals if int(v) != 0}
        legend = BUILTIN_LEGENDS[args.lulc_source]
        # 按语义名去重后再编号:WorldCover 多个源编码共用同一语义
        # (70 冰雪→裸地、95 红树林→湿地、100 苔藓→草地)时,直接按编码枚举
        # 会产生同名键覆盖 → 目标编码不连续 → PLUS 报错(2026-11-10 加 WorldCover 时发现)
        sem = {}
        for c in sorted(union):
            name = legend.get(c, f"类别{c}")
            sem.setdefault(name, c)
        mapping = {name: i + 1 for i, name in enumerate(sem)}
        mapping_path = os.path.join(data_dir, "mapping_auto.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"  自动映射(类别并集→连续编码): {mapping}")
    rc_cmd = [PY, script("reclassify_lulc.py"),
              *[os.path.join(data_dir, f"lulc_{y}.tif") for y in years],
              "--source", args.lulc_source, "--mapping", mapping_path, "--output-dir", data_dir]
    rc_out = run(rc_cmd, timeout=1800)
    # "起始年 0 像元类":markov 会算 0/0=NaN 并污染全表(实测)。
    # 老师规则(2026-11-07):成因分析命中内置规则 → 自动归并并继续跑;未命中才中止要人工。
    if "起始年" in rc_out and "0 像元类别" in rc_out:
        print("\n⚠️ 检测到'起始年 0 像元类'(markov NaN 风险 + 编码错位),进入自动归并判断")
        new_map, logs = auto_merge_zero_classes(rc_out, mapping_path, mapping, data_dir)
        if new_map is None:
            print("\n❌ 无法自动归并(规则未命中),需人工处置:")
            for line in logs:
                print(f"   - {line}")
            print("   请按 重分类说明.md 的建议改映射,用 --mapping 显式映射后重跑。")
            sys.exit(1)
        mapping = new_map
        rc_cmd[rc_cmd.index("--mapping") + 1] = mapping_path
        rc_out = run(rc_cmd, timeout=1800)
        if "起始年" in rc_out and "0 像元类别" in rc_out:
            print("\n❌ 自动归并后起始年仍有 0 像元类(可能归并目标本身也缺席),需人工处置")
            sys.exit(1)
        print("  ✅ 自动归并完成,重分类已按新映射重跑")

    # 极小类告警(不强制):<1000 像元(≈0.9 km²)CARS 难以生成斑块、F1 无统计意义
    import rasterio as _rio
    import numpy as _np
    with _rio.open(os.path.join(data_dir, f"lulc_{years[-1]}_rc.tif")) as s:
        _vals, _cnt = _np.unique(s.read(1), return_counts=True)
    _tiny = {int(v): int(c) for v, c in zip(_vals, _cnt)
             if int(v) not in (0, 255) and 0 < int(c) < 1000}
    if _tiny:
        print(f"\n⚠️ 极小类告警(末期 {years[-1]}):{_tiny}")
        print("   像元数 <1000(≈0.9 km²)的类难以生成独立斑块,精度指标也无统计意义")
        print("   建议:在映射里并入相邻类(不强制,由用户决定)")

    tick("③ 语义重分类", t3, inputs=f"{len(years)} 期原始裁切图",
         outputs="mapping_auto.json + lulc_*_rc.tif + 重分类说明.md")

    # ④ 检查 + 转换
    t4 = time.time()
    print("=" * 60 + "\n【④ 检查转换(PLUS convert)】\n" + "=" * 60)
    for y in years:
        run([PY, script("check_landuse.py"),
             os.path.join(data_dir, f"lulc_{y}_rc.tif")], timeout=1800)
    tick("④ 检查转换", t4, inputs="lulc_*_rc.tif", outputs="lulc_*_rc_uc.tif")

    # 网格一致性检查:所有参与建模的图必须逐像素对齐
    # (缓存里其他年份可能是旧网格裁切,混用会错位;只校验本次用到的年份)
    import rasterio as _rio
    _grids = {}
    for y in years:
        with _rio.open(os.path.join(data_dir, f"lulc_{y}_rc_uc.tif")) as s:
            _grids[y] = (s.transform, s.width, s.height)
    if len(set(_grids.values())) > 1:
        print("❌ 各年份网格不一致(缓存中混有旧网格裁切):")
        for y, g in _grids.items():
            print(f"   {y}: {g[1]}×{g[2]} {g[0].c:.2f},{g[0].f:.2f}")
        print("   请删除旧网格年份的 lulc_*_rc_uc.tif 后重跑(脚本会按新口径重裁)")
        sys.exit(1)
    with _rio.open(ref_tif) as s:
        _ref_grid = (s.transform, s.width, s.height)
    if _ref_grid != _grids[years[0]]:
        print("❌ 驱动因子参考图与建模图网格不一致,请重跑驱动因子下载")
        sys.exit(1)
    print("  ✓ 全链网格一致")

    # ⑤ 生成项目配置(必须先写配置,再建模——07 版此处顺序错乱,proj_path 未定义即被使用)
    if args.predict == years[-1]:
        early, late, real = years[0], years[1], years[-1]
        out_name = f"model_bt{args.predict}"
    else:
        early, late, real = years[0], years[-1], ""
        out_name = f"model_{args.predict}"
    if args.demand == "real" and not real:
        print("❌ --demand real 只能用于回测(目标年必须等于末年)")
        sys.exit(1)
    with open(os.path.join(data_dir, "重分类说明.md"), encoding="utf-8") as f:
        notes = (downgrade_note + f.read()[:800])
    # 类数:直接用转换后的图读类别数
    with _rio.open(os.path.join(data_dir, f"lulc_{years[-1]}_rc_uc.tif")) as s:
        vals = _np.unique(s.read(1))
    classes = [int(v) for v in vals if int(v) not in (0, 255)]
    n_cls = len(classes)
    # 类名:统一从映射文件反查语义名——自动映射路径也写语义名,
    # 否则默认矩阵按"森林/建设用地"匹配不到"类别N"会静默退化成全开放(P0-1,WB 审查)
    with open(mapping_path, encoding="utf-8") as f:
        _m = json.load(f)
    _m.pop("_notes", None)
    _rev = {}
    for _sem, _code in _m.items():
        _rev.setdefault(int(_code), []).append(_sem)
    names = {str(c): "/".join(_rev.get(c, [f"类别{c}"])) for c in classes}
    if all(n.startswith("类别") for n in names.values()):
        print("⚠️ 类名无法反查语义名,默认矩阵将退化为全开放——请检查映射文件")
    del _m, _rev

    if args.demand == "auto":
        # 官方两法(2026-11-09 定稿,GM(1,1) 已删):有效历史 ≥3 期 → linear;2 期 → markov。
        # 有效历史 = 回测时排除目标年(防近似作弊),预测时全部年份。
        # 不能用 len(years) 直接判:回测排除目标年后 linear 可能只剩 2 期(引擎 2 期拟合全 0)。
        hist_count = len(years) - 1 if args.predict == years[-1] else len(years)
        demand = "markov" if hist_count < 3 else "linear"
        print(f"  auto 需求方法:{demand}(官方两法,有效历史 {hist_count} 期)")

    else:
        demand = args.demand
    # 数据源标注(2026-11-10 导师实测:汇总报告模板写死"CLCD 30m",自有数据也被写成 CLCD)。
    # 以"CLCD 下载报告"为证据自动判断,写进 project → make_report.py 按此描述数据源。
    if args.lulc_source == "worldcover":
        # WorldCover 渠道直接标 worldcover,不走 CLCD 报告文件推断(报告里写的是 ESA)
        data_source = "worldcover"
        print("  数据源判定:ESA WorldCover(10m,已重采样 30m)")
    else:
        _clcd_ev = [os.path.join(data_dir, p) for p in
                    ("CLCD下载报告.md", "下载报告.md",
                     os.path.join("logs", "子报告", "CLCD下载报告.md"))]
        data_source = "clcd" if any(os.path.isfile(p) for p in _clcd_ev) else "user"
        print(f"  数据源判定:{'CLCD 自动下载' if data_source == 'clcd' else '用户自有数据'}"
              + ("(依据:存在 CLCD 下载报告)" if data_source == "clcd"
                 else "(依据:未见 CLCD 下载报告;若实为 CLCD,请核对下载报告是否在 logs\\子报告\\)"))
    # 区域化矩阵修改(2026-11-10 用户定:AI 自动检索研究区信息+文献政策后生成,
    # 经此参数一次传入;规则与依据进 proj → 报告 3b 节自动说明改了什么)
    restrictions = []
    if args.matrix_restrictions:
        raw = args.matrix_restrictions
        if os.path.isfile(raw):
            with open(raw, encoding="utf-8") as f:
                restrictions = json.load(f)
        else:
            try:
                restrictions = json.loads(raw)
            except Exception:
                print("❌ --matrix-restrictions 解析失败(需 JSON 数组或 .json 文件路径)")
                sys.exit(1)
        if not isinstance(restrictions, list):
            print("❌ --matrix-restrictions 必须是 JSON 数组")
            sys.exit(1)
        print(f"  区域化矩阵修改 {len(restrictions)} 条:"
              + "; ".join(f"{r.get('from')}→{r.get('to')}="
                          f"{'禁止' if not r.get('allow', True) else '允许'}"
                          for r in restrictions))
    proj = {
        "_说明": f"{args.region} 全流程{'回测' if real else '预测'}(plus-automation 自动生成)",
        "data_dir": data_dir,
        "early": os.path.join(data_dir, f"lulc_{early}_rc_uc.tif"),
        "late": os.path.join(data_dir, f"lulc_{late}_rc_uc.tif"),
        "real": os.path.join(data_dir, f"lulc_{real}_rc_uc.tif") if real else "",
        "features": drivers_dir,
        "policy": "",
        "early_year": early, "late_year": late, "predict_year": args.predict,
        "demand": demand,
        "validation_mode": args.validation,
        "diagnose": bool(args.diagnose and real and demand != "real"),
        "reuse_artifacts": bool(args.reuse_model),
        "sequence": sequence,
        "transition_matrix": "default",
        "matrix_restrictions": restrictions,
        # 推荐口径(train)与预测面积口径(predict)的 CARS 都从训练首年出发(用户 2026-11-11 定)
        "sim_start_early": bool(args.backtest in ("train", "predict")),
        "class_names": names,
        "data_source": data_source,
        "lulc_source": args.lulc_source,
        "lulc_res": args.lulc_res,
        "drivers_source": "user" if args.drivers_dir else "download",
        "data_notes": notes,
        "output_dir": os.path.join(data_dir, out_name),
    }
    # 水域锁定(前期问答:水域是否锁定不转换);锁定则水域行只允许自转
    if args.water == "lock":
        water_names = [n for n in names.values()
                       if any(a in n for a in ("水域", "水体", "湿地", "湖泊", "水库"))]
        if water_names:
            proj["locked_classes"] = water_names  # 用实际类名(lock_classes 按别名再匹配一次)
            proj["data_notes"] = (notes + "\n水域锁定不转换(前期问答用户选择):"
                                  "水域只能自转,不转为耕地/建设用地等其他类。")
            print(f"  🔒 水域锁定不转换(命中类:{water_names})")
        else:
            print("⚠️ --water lock 但分类体系里没有水域类,忽略锁定")
    if demand == "linear":
        # 历史序列:回测时排除目标年(防近似作弊),预测时用全部年份。
        # linear 至少 3 期有效历史:2 期拟合外推会穿零/全 0(2026-11-09 实测引擎行为)
        hist_years = [y for y in years if y != args.predict] \
            if args.predict == years[-1] else years
        need = 3
        if len(hist_years) < need:
            print(f"❌ linear 需要 ≥{need} 期有效历史(当前 {len(hist_years)} 期);"
                  f"2 期请用 --demand markov,或补历史年份")
            sys.exit(1)
        proj["history"] = [os.path.join(data_dir, f"lulc_{y}_rc_uc.tif") for y in hist_years]
        proj["history_years"] = hist_years
        _excl = " 目标年已排除(防近似作弊)" if args.predict in years else ""
        print(f"  需求历史序列({demand}):{hist_years}{_excl}")
    elif demand == "markov" and (
            (args.predict - late) % (late - early) != 0
            or (sequence and any((y - late) % (late - early) != 0 for y in sequence))):
        # 2026-11-11 实测:目标年恰在步长上(如 2040→2060),但序列中间期(2030/2050)脱格时,
        # 旧逻辑不开插值 → 链式模拟报"markov 引擎不支持非整数周期年"。任一中间期脱格都要开。
        proj["interpolate"] = True
        print(f"⚠️ markov 目标年或序列中间期不在步长上,启用分段插值")
    proj_path = os.path.join(data_dir, f"project_full_{args.predict}.json")
    # 区域化转换矩阵(FAQ F5):若同名 project json 已存在,继承其中人工/AI 写的
    # matrix_restrictions —— 旧实现每次无条件覆盖,"先跑一次→改 json→再跑"的编辑必丢
    # (2026-11-10 修)。约束模板可以只写 matrix_restrictions 这一个字段,其余字段仍按本次参数重建。
    if os.path.exists(proj_path):
        try:
            with open(proj_path, encoding="utf-8") as f:
                _prev = json.load(f)
        except Exception as _e:
            _prev = {}
            print(f"  ⚠️ 既有项目配置读取失败({_e}),按默认矩阵继续")
        _rest = _prev.get("matrix_restrictions") or []
        if _rest:
            proj["matrix_restrictions"] = _rest
            _desc = ";".join(
                f"{r.get('from')}→{r.get('to')}{'允许' if r.get('allow', True) else '禁止'}"
                f"({r.get('依据', '未注明依据')})" for r in _rest)
            _line = f"区域化转换矩阵(前期问答用户选'区域化修改'):{_desc}"
            if "区域化转换矩阵" not in (proj.get("data_notes") or ""):
                proj["data_notes"] = (proj.get("data_notes") or "") + "\n" + _line
            print(f"  🧩 继承既有区域化矩阵 {len(_rest)} 条:{_desc}")
    with open(proj_path, "w", encoding="utf-8") as f:
        json.dump(proj, f, ensure_ascii=False, indent=2)
    print(f"  项目配置已生成: {proj_path}")
    print(f"  验证口径: {args.validation}(样本{'内' if args.validation == 'in' else '外'})"
          f"{'; 误差分解: 开' if proj['diagnose'] else ''}")

    # ⑤ 建模
    t5 = time.time()
    import hashlib
    cfg_key = hashlib.md5(json.dumps(proj, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()
    if stage_done("5", cfg_key):
        print("♻️ 阶段⑤已跳过(断点自动生效)")
    else:
        print("=" * 60 + "\n【⑤ 建模(expansion→LEAS→需求→CARS→精度)】\n" + "=" * 60)
        run([PY, script("run_pipeline.py"), "--project", proj_path], timeout=43200)
        mark_done("5", cfg_key)
    tick("⑤ 建模(expansion→LEAS→需求→CARS→精度)", t5,
         inputs=f"{n_cls} 类 × {len(years)} 期 + {len(os.listdir(drivers_dir))} 个因子",
         outputs=os.path.join(data_dir, out_name))
    sim = os.path.join(data_dir, out_name, f"simulation_bt_{args.predict}Simulation_1.tif")


    # ⑥ 精度(读建模阶段落盘的 report.json;缺失才回退调 run_validation)
    print("=" * 60 + "\n【⑥ 精度汇总 + 误差分解】\n" + "=" * 60)
    t6 = time.time()
    rep = {}
    rep_path = os.path.join(data_dir, out_name, "report.json")
    if os.path.exists(rep_path):
        with open(rep_path, encoding="utf-8") as f:
            rep = json.load(f)
    m = rep.get("metrics") or {}
    diag = rep.get("diagnose") or {}
    fom_line = "(预测,无真实图,跳过精度)"
    if real and m:
        fom_line = (f"Kappa {m['Kappa']:.4f} / OA {m['OA']*100:.2f}% / "
                    f"FoM {m['FoM']:.4f}")
        print(f"  {fom_line}")
        if diag.get("metrics"):
            print(f"  诊断口径(真实面积需求)FoM = {diag['metrics']['FoM']:.4f}")
            print(f"  数量误差贡献 {diag['fom_gain']:+.4f};位置误差损失 {diag['loc_loss']:.4f}")
    elif real:
        r = subprocess.run(
            [PY, script("run_validation.py"), "--simulated", sim,
             "--real", proj["real"], "--start", proj["late"], "--fom"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=3600)
        for line in (r.stdout + r.stderr).splitlines():
            if "FoM =" in line:
                fom_line = "FoM " + line.split("FoM =")[-1].strip()
    tick("⑥ 精度汇总(含误差分解)", t6, inputs="模拟图 + 真实图", outputs=rep_path)

    # ⑦ 自动出图:模拟结果(及回测的真实图)→ 论文级 JPG
    print("=" * 60 + "\n【⑦ 结果自动出图(JPG,300dpi)】\n" + "=" * 60)
    t7 = time.time()
    name_str = ",".join(f"{c}={n}" for c, n in sorted(
        ((int(k), v) for k, v in proj["class_names"].items())))
    figs = []
    for label, tif in ([("模拟", sim)] +
                       ([("真实", proj["real"])] if real else [])):
        out_jpg = os.path.join(data_dir, out_name,
                               f"{'simulation' if label == '模拟' else 'real'}_{args.predict}.jpg")
        rr = subprocess.run(
            [PY, script("landuse_map.py"), tif, "--output", out_jpg,
             "--names", name_str],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=1800)
        ok = rr.returncode == 0 and os.path.exists(out_jpg)
        figs.append((label, out_jpg if ok else None))
        print(f"  {'✅' if ok else '❌'} {label}图: {out_jpg if ok else (rr.stdout + rr.stderr)[-200:]}")
    fig_line = "; ".join(f"{l}: {p or '失败'}" for l, p in figs)

    # ⑦+ 连续预测:多期模拟图并排对比(任务9.1)
    seq_fig = None
    seq_sims = rep.get("sequence") or []
    if len(seq_sims) > 1:
        from landuse_map import make_sequence_map
        pairs = [(s["sim"], s["year"]) for s in seq_sims]
        _seq_last = seq_sims[-1]["year"]
        seq_fig = os.path.join(data_dir, out_name,
                               f"sequence_{args.predict}.png" if _seq_last == args.predict
                               else f"sequence_{args.predict}_{_seq_last}.png")
        try:
            # 整图大标题(2026-11-10 用户要求):"某市 首年-末年 年土地利用对比"
            _seq_title = (f"{args.region} {seq_sims[0]['year']}-{seq_sims[-1]['year']}"
                          f" 年土地利用对比")
            make_sequence_map(pairs, seq_fig, names=name_str, dpi=150, title=_seq_title)
            fig_line += f"; 多期对比: {seq_fig}"
        except Exception as e:
            print(f"⚠️ 多期对比图失败: {e}")
            seq_fig = None

        # 目标年单幅成图:连续预测时任务目标年是序列末年,不能只出主模拟年那一张。
        # 2026-11-10 机理改后序列末年 = 主模拟年(如 2020→2030 间隔 5 → 2025、2030),
        # 主模拟图 simulation_2030.jpg 就是目标年图,不再重复渲染。
        _final_year = seq_sims[-1]["year"]
        if _final_year == args.predict:
            print(f"  ℹ️ 序列末年 = 目标年 {_final_year},主模拟图即目标年图")
        else:
            _final_jpg = os.path.join(data_dir, out_name, f"simulation_{_final_year}.jpg")
            _rf = subprocess.run(
                [PY, script("landuse_map.py"), seq_sims[-1]["sim"], "--output", _final_jpg,
                 "--names", name_str],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=1800)
            if _rf.returncode == 0 and os.path.exists(_final_jpg):
                fig_line += f"; 目标年({_final_year}): {_final_jpg}"
                print(f"  ✅ 目标年图({_final_year}): {_final_jpg}")
            else:
                print(f"  ⚠️ 目标年图生成失败: {(_rf.stdout + _rf.stderr)[-200:]}")
    tick("⑦ 出图", t7, inputs=sim, outputs="; ".join(p for _, p in figs if p))

    # ⑦++ 面积统计 CSV(逐类 km²):历史各年真实 + 主模拟 + 连续预测各期
    area_csv = None
    try:
        area_csv = write_area_stats(data_dir, out_name, proj["class_names"],
                                    years, args.predict, sim, seq_sims)
    except Exception as e:
        print(f"⚠️ 面积统计生成失败: {e}")

    # ⑧ 汇总报告(唯一出口):由 make_report 统一生成,分步 md 报告归档到 logs/子报告/
    # 被跳过的阶段(复用缓存/--resume)也要出现在耗时表里,否则清单不完整
    _done = {t["stage"][0] for t in TIMING}
    if "①" not in _done:
        TIMING.insert(0, {"stage": "① 土地利用下载", "seconds": 0.0,
                          "inputs": "复用已有裁切图(--data-dir)", "outputs": "-"})
    if "②" not in _done:
        TIMING.insert(1, {"stage": "② 驱动因子下载", "seconds": 0.0,
                          "inputs": "复用已有因子(断点自动生效)", "outputs": "-"})
    try:  # 耗时落盘,便于日后只重生报告
        with open(os.path.join(data_dir, out_name, "timing.json"), "w", encoding="utf-8") as _tf:
            json.dump(TIMING, _tf, ensure_ascii=False, indent=2)
    except Exception:
        pass
    from make_report import build_report
    report = build_report(data_dir, out_name, proj, args.predict, args.region,
                          timings=TIMING, fig_line=fig_line, area_csv=area_csv,
                          backtest=args.backtest, mapping=args.mapping, water=args.water)
    print(f"\n✅ 全流程完成!汇总报告: {report}")
    print(f"   精度: {fom_line}")

    # ⑨ 清理临时残留(.part/.parts.json/引擎中间图 simulation_xxxite_N.tif)
    leftover = []
    for root, _, names in os.walk(data_dir):
        for nm in names:
            is_engine_tmp = nm.startswith("simulation") and "ite_" in nm and "Simulation" not in nm
            if nm.endswith((".part", ".parts.json")) or is_engine_tmp:
                leftover.append(os.path.join(root, nm))
    if leftover:
        for p in leftover:
            try:
                os.remove(p)
            except OSError:
                pass
        print(f"🧹 清理临时残留 {len(leftover)} 个(.part/.tmp/引擎中间图)")


if __name__ == "__main__":
    main()
