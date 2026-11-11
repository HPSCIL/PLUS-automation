#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""合并式汇总报告生成器(plus-automation 唯一报告出口)

设计原则:
1. 一个研究区只产出**一份** Markdown 报告: `<输出目录>/汇总报告.md`
2. 下载报告 / 驱动因子报告 / 重分类说明 / 建模报告 全部归档到 `logs/子报告/`,
   正文只摘取要点,避免用户在一堆 md 里翻找
3. 目标年以"连续预测序列的末年份"为准(--predict 2030 --sequence 5,2 → 目标 2040),
   不再把主模拟年当成任务目标年
4. 可独立运行:不重跑建模也能重生报告
   `python scripts/make_report.py --data-dir D:/plus_skill_GIS_data/371700 --project project_full_2030.json`

输入(缺哪个就跳过哪节,不报错):
  项目配置 project_full_*.json / 建模产物 report.json / 面积统计.csv / 各子报告 md / timing.json
"""
import csv
import json
import os
import shutil
import sys

# 控制台编码防护:中文 Windows 默认 GBK,打印 ✅ 等符号会崩(2026-11-10 郑州报告重生实测)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import force_utf8

force_utf8()

SUBDIR = "子报告"          # 归档子目录(logs/子报告)
REPORT_NAME = "汇总报告.md"

# 子报告 → (简称, 所在目录 data|out, 相对路径, 归档后的文件名, 一句话说明)
SUB_REPORTS = [
    ("CLCD 下载报告", "data", "下载报告.md", "CLCD下载报告.md", "数据来源、期次、裁切范围与校验结果"),
    ("驱动因子报告", "data", os.path.join("drivers", "驱动因子报告.md"), "驱动因子报告.md",
     "17 项因子来源与网格对齐"),
    ("重分类说明", "data", "重分类说明.md", "重分类说明.md", "语义映射、0 像元归并、特殊情况"),
    ("建模过程报告", "out", "报告.md", "建模过程报告.md", "扩张/概率/需求/CARS 逐步输出"),
]


# ---------------- 小工具 ----------------
def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _num(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"


def _signed(x, nd=2):
    try:
        v = float(x)
        return f"{v:+,.{nd}f}"
    except Exception:
        return "—"


def _pct(cur, base):
    try:
        cur, base = float(cur), float(base)
        if abs(base) < 1e-9:
            return "—"
        return f"{(cur - base) / base * 100:+.1f}%"
    except Exception:
        return "—"


def _year_of(col):
    """'2040模拟' → 2040;'2020' → 2020"""
    digits = "".join(ch for ch in str(col) if ch.isdigit())
    return int(digits) if digits else None


def read_area_table(csv_path):
    """读 面积统计.csv → (列名列表, 年份列表, {类名: {列名: 值}})"""
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    except Exception:
        return None
    if len(rows) < 2:
        return None
    cols = rows[0][1:]
    table = {}
    order = []
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        name = r[0]
        vals = {}
        for c, v in zip(cols, r[1:]):
            try:
                vals[c] = float(v)
            except Exception:
                vals[c] = 0.0
        if name == "合计":
            table["_total"] = vals
        else:
            table[name] = vals
            order.append(name)
    return order, cols, table


def archive_sub_reports(data_dir, out_dir):
    """把分步 md 报告归档到 logs/子报告/,返回归档清单 [(简称, 原路径, 归档路径)]"""
    dest_root = os.path.join(data_dir, "logs", SUBDIR)
    moved = []
    for label, base, rel, dst_name, _desc in SUB_REPORTS:
        root = out_dir if base == "out" else data_dir
        src = os.path.join(root, rel)
        if not os.path.exists(src):
            continue
        try:
            os.makedirs(dest_root, exist_ok=True)
            dst = os.path.join(dest_root, dst_name)
            shutil.move(src, dst)
            moved.append((label, src, dst))
        except Exception as e:
            print(f"  ⚠️ 归档失败 {src}: {e}")
    return moved


def _data_source(proj, data_dir):
    """数据源判定:project.data_source 优先;旧项目(无此字段)按 CLCD 下载报告推断。

    2026-11-10 导师实测:模板曾写死"CLCD 30m",自有数据也被写成 CLCD,故参数化。
    """
    ds = proj.get("data_source")
    if not ds:
        ev = [os.path.join(data_dir, p) for p in
              ("CLCD下载报告.md", "下载报告.md",
               os.path.join("logs", "子报告", "CLCD下载报告.md"))]
        ds = "clcd" if any(os.path.isfile(p) for p in ev) else "user"
    return ds


# ---------------- 正文各部分 ----------------
def _sec_meta(proj, predict, target, seq, region, backtest, water, mapping, factors):
    L = []
    L.append("## 1. 任务与参数\n")
    L.append("| 项目 | 设置 |")
    L.append("|---|---|")
    L.append(f"| 研究区 | {region} |")
    _ds = _data_source(proj, proj.get("data_dir") or "")
    _wc = ("ESA WorldCover 10m 全球土地覆盖(v100/v200,原生 10m;CC BY 4.0)"
           if proj.get("lulc_res") == 10
           else "ESA WorldCover 10m 全球土地覆盖(v100/v200,重采样 30m;CC BY 4.0)")
    _src = {"clcd": "CLCD 30m 逐年数据(中科院, Yang & Huang 2021)",
            "worldcover": _wc,
            "user": "用户自有土地利用数据(各年份 tif 由用户提供)"}
    L.append(f"| 数据源 | {_src.get(_ds, _src['user'])} |")
    L.append(f"| 训练期 | {proj.get('early_year')}–{proj.get('late_year')} |")
    if seq:
        yrs = " / ".join(str(s["year"]) for s in seq)
        L.append(f"| 预测年份 | {yrs}(链式,逐期从上一期结果继续) |")
        L.append(f"| **目标年** | **{target}** |")
    else:
        L.append(f"| **目标年** | **{target}** |")
    L.append(f"| 面积需求方法 | {proj.get('demand')}"
             + ("(马尔可夫链,非周期年按分段插值)" if proj.get("demand") == "markov" else "") + " |")
    _bt = {"none": "否(纯预测)", "real": "真实面积诊断(数量给定,只测位置)",
           "predict": "预测面积回测", "train": "训练期全周期(样本内)"}.get(backtest, backtest)
    L.append(f"| 回测 | {_bt} |")
    L.append(f"| 验证口径 | {'样本内' if proj.get('validation_mode') == 'in' else '样本外'} |")
    locked = proj.get("locked_classes") or []
    L.append(f"| 水域处理 | {'锁定不转出(' + '、'.join(locked) + ')' if locked else '不锁定'} |")
    L.append(f"| 转换矩阵 | 官方模板(森林/建设用地不可逆)" + (
        ";区域化修改 " + "; ".join(f"{r.get('from')}→{r.get('to')}" for r in
                                (proj.get("matrix_restrictions") or []))
        if proj.get("matrix_restrictions") else "") + " |")
    L.append(f"| 重分类 | {'自定义映射' if mapping else '自动(类别并集→连续编码)'} |")
    _drv = ("用户自供(体检网格一致+非全0)" if proj.get("drivers_source") == "user"
            else "官方同款 17 项自动下载:地形/气候/人口/距离/土壤")
    L.append(f"| 驱动因子 | {factors} 项({_drv}) |")
    L.append(f"| 数据目录 | `{proj.get('data_dir') or ''}` |")
    L.append("")
    return L


def _history_years(data_dir):
    """从裁切文件反推历史期次(lulc_YYYY_rc_uc.tif),不依赖项目配置"""
    ys = []
    try:
        for f in os.listdir(data_dir):
            if f.startswith("lulc_") and f.endswith("_rc_uc.tif"):
                y = f[5:9]
                if y.isdigit():
                    ys.append(int(y))
    except Exception:
        pass
    return sorted(ys)


def _sec_data(data_dir, proj, factors):
    L = []
    L.append("## 2. 数据与驱动因子\n")
    # 栅格规格:取自末年裁切图
    late_tif = proj.get("late") or ""
    spec = ""
    try:
        import rasterio
        if late_tif and os.path.exists(late_tif):
            with rasterio.open(late_tif) as s:
                proj_name = (s.crs.to_dict().get("proj") if s.crs else "") or ""
                crs_txt = "Albers 等积投影(aea)" if proj_name.lower() == "aea" else (proj_name or "—")
                spec = (f"{s.width}×{s.height} 像元,{abs(s.res[0]):.0f} m 分辨率,{crs_txt}")
    except Exception:
        pass
    hys = _history_years(data_dir)
    L.append(f"- 基础栅格:{spec or '见子报告'}")
    L.append(f"- 土地利用期次:{'、'.join(str(y) for y in hys) if hys else '见子报告'}"
             f"(已按研究区边界裁切、全链统一网格)")
    if proj.get("drivers_source") == "user":
        L.append(f"- 驱动因子:{factors} 项(用户自供,已体检:与参考图网格一致、非全 0;"
                 f"因子含义与来源以用户说明为准)")
    else:
        L.append(f"- 驱动因子:{factors} 项,含高程/坡度/坡向、年均气温/降水、人口、"
                 f"到各级道路与铁路/水域/政府驻地距离、土壤")
        L.append("- 因子已知短板:气候数据为 2.5° 粗分辨率(NCEP/CMAP);OSM 道路与水域在国内有遗漏,"
                 "距离因子精度受其完整性限制")
    L.append("")
    return L


def _sec_matrix(proj):
    """## 3b. 转换矩阵规则:基线 + 改动 + 依据(2026-11-10 用户要求:报告必须说明改了什么)

    官方模板基线 = 森林/建设用地不可逆;区域化修改/用户自定义在此列出每条差异与依据,
    水域锁定也在此按矩阵视角说明。
    """
    L = ["## 3b. 转换矩阵规则\n"]
    locked = proj.get("locked_classes") or []
    rests = proj.get("matrix_restrictions") or []
    L.append("- 基线:官方模板——森林、建设用地不可逆(形成后不再转出),其余类别开放互转")
    if locked:
        L.append(f"- 水域锁定:{'、'.join(locked)} 只能自转、不转出(前期问答用户选择)")
    if rests:
        L.append("- **区域化修改**(相对官方模板的改动,共 %d 条):" % len(rests))
        for r in rests:
            allow = "禁止" if not r.get("allow", True) else "允许"
            line = f"  - {allow} {r.get('from', '?')} → {r.get('to', '?')}"
            if r.get("依据"):
                line += f";依据:{r.get('依据')}"
            L.append(line)
    else:
        L.append("- 无区域化修改(使用官方模板)")
    L.append("")
    return L


def _sec_reclass(data_dir, proj):
    """从重分类说明/建模报告中摘取映射与特殊情况(文件可能已归档,按归档路径找)"""
    txt = _read(os.path.join(data_dir, "logs", SUBDIR, "重分类说明.md"))
    L = ["## 3. 重分类与特殊处理\n"]
    if not txt:
        return L + ["- 详见归档的《重分类说明》。\n"]
    # 抽取 json 映射块
    mapping = None
    try:
        start = txt.index("{")
        end = txt.index("}", start) + 1
        mapping = json.loads(txt[start:end])
    except Exception:
        mapping = None
    if mapping:
        L.append("| 语义类 | 建模编码 |")
        L.append("|---|---|")
        for k, v in mapping.items():
            L.append(f"| {k} | {v} |")
        L.append("")
    # 只摘「特殊情况:」那一条,不要把整篇说明连代码块一起吞进来
    note = []
    hit = False
    for ln in txt.splitlines():
        if ln.startswith("## 特殊情况"):
            hit = True
            rest = ln.split("特殊情况", 1)[1].lstrip(":： ")
            if rest:
                note.append(rest)
            continue
        if hit:
            if ln.startswith("## ") or ln.startswith("# "):
                break
            s = ln.strip()
            if not s or s.startswith("```"):
                continue
            note.append(s)
    if note:
        L.append("**特殊情况**:" + " ".join(note) + "\n")
    locked = proj.get("locked_classes") or []
    if locked:
        L.append(f"**水域锁定**:{'、'.join(locked)} 整行锁定,只能自转、不得转为其他地类;"
                 "其他地类仍可转入水域(故水域面积可能继续增长)。\n")
    return L


def _find_backtest(data_dir, exclude_out):
    """在同研究区目录里找一份带精度指标的回测结果(供预测报告引用可靠性)"""
    best = None
    try:
        entries = os.listdir(data_dir)
    except Exception:
        return None
    for d in entries:
        if d == exclude_out:
            continue
        rj = _load_json(os.path.join(data_dir, d, "report.json"))
        if not rj or not rj.get("metrics"):
            continue
        # 回测的 sim 目录名通常带 bt / 目标年早于预测年
        score = 1 if "bt" in d.lower() else 0
        if best is None or score > best[0]:
            best = (score, d, rj)
    return (best[1], best[2]) if best else None


def _sec_accuracy(rj, backtest, proj, bt=None):
    L = ["## 4. 建模与精度\n"]
    m = (rj or {}).get("metrics")
    dg = (rj or {}).get("diagnose") or {}
    if not m:
        L.append("本次为**未来情景预测**(目标年在历史数据之后),没有真实图可比,"
                 "因此本报告不出 Kappa/FoM。")
        if bt:
            d, brj = bt
            bm = brj["metrics"]
            L.append(f"同一研究区、同一套参数的回测结果(目录 `{d}`)可作为可靠性参照:\n")
            L.append("| 指标 | 回测值 | 说明 |")
            L.append("|---|---|---|")
            L.append(f"| Kappa | {bm.get('Kappa', 0):.4f} | 总体一致性 |")
            L.append(f"| 总体精度 OA | {_num((bm.get('OA') or 0) * 100)}% | 未变化像元占比高,易偏高 |")
            L.append(f"| FoM | {bm.get('FoM', 0):.4f} | 变化像元的定位能力 |")
            A, B, C, D = bm.get("A"), bm.get("B"), bm.get("C"), bm.get("D")
            if None not in (A, B, C, D):
                L.append(f"| 变化四格(10% 采样) | 命中 {A:,} / 漏报 {B:,} / "
                         f"假警报 {C:,} / 错类 {D:,} | — |")
            L.append("")
            L.append("> 该回测为**真实面积诊断口径**(需求量直接取验证年真实面积),"
                     "数量完全正确,因此其 FoM 只反映位置分配能力,是 CARS 能力的上界;"
                     "完整口径(含数量预测)的 FoM 通常更低。\n")
        else:
            L.append("如需可靠性指标,可加 `--predict 末年份 --backtest real` 单独回测。\n")
        return L
    ka, oa, fom = m.get("Kappa"), m.get("OA"), m.get("FoM")
    grade = "优" if (ka or 0) >= 0.8 else ("良" if (ka or 0) >= 0.6 else "一般")
    lv = "较强" if (fom or 0) >= 0.12 else ("中等" if (fom or 0) >= 0.05 else "偏弱")
    L.append("| 指标 | 数值 | 说明 |")
    L.append("|---|---|---|")
    L.append(f"| Kappa | {ka:.4f} | 总体一致性({grade}) |")
    L.append(f"| 总体精度 OA | {_num((oa or 0) * 100, 2)}% | 未变化像元占比高,易偏高 |")
    L.append(f"| FoM | {fom:.4f} | 变化像元的定位能力({lv}) |")
    A, B, C, D = m.get("A"), m.get("B"), m.get("C"), m.get("D")
    if None not in (A, B, C, D):
        L.append(f"| 变化四格(10% 采样) | 命中 {A:,} / 漏报 {B:,} / 假警报 {C:,} / 错类 {D:,} | — |")
    L.append("")
    if dg.get("metrics"):
        d = dg["metrics"]
        L.append("**误差分解(数量 vs 位置)**\n")
        L.append("| 口径 | 需求量来源 | FoM |")
        L.append("|---|---|---|")
        L.append(f"| 主口径 | {proj.get('demand')} 预测 | {fom:.4f} |")
        L.append(f"| 诊断口径 | 真实验证年面积 | {d.get('FoM', 0):.4f} |")
        L.append("")
        L.append(f"- 数量误差贡献 **{dg.get('fom_gain', 0):+.4f} FoM**;"
                 f"位置误差损失 **{dg.get('loc_loss', 0):.4f}**")
        L.append("")
    if backtest == "real" or proj.get("demand") == "real":
        L.append("> 本次回测为**真实面积诊断口径**:需求量直接取验证年真实面积,"
                 "数量完全正确,因此 FoM 只反映位置分配能力,是 CARS 能力的上界,"
                 "不能等同于「含数量预测的完整精度」。\n")
    L.append("> 判读提醒:Kappa/OA 高不代表预测可信——变化像元通常不到 10%,"
             "90% 的未变化像元正确就能把 OA 撑到 90% 以上。看变化看 FoM。\n")
    return L


def _sec_area(order, cols, table, target, late_year, early_year):
    L = ["## 5. 面积统计与变化\n"]
    if not order:
        return L + ["- 面积统计缺失(面积统计.csv 未生成)。\n"]
    yrs = [(c, _year_of(c)) for c in cols]
    yrs = sorted([(c, y) for c, y in yrs if y], key=lambda t: t[1])
    L.append("单位:km²(按 30m 像元反算)\n")
    L.append("| 地类 | " + " | ".join(str(y) for _, y in yrs) +
             " | " + f"{late_year}→{target} 变化 | 变化率 |")
    L.append("|---" * (len(yrs) + 3) + "|")
    tgt_col = next((c for c, y in yrs if y == target), None)
    base_col = next((c for c, y in yrs if y == late_year), None)
    for name in order:
        vals = table.get(name, {})
        cells = [_num(vals.get(c, 0)) for c, _ in yrs]
        dv = (vals.get(tgt_col, 0) - vals.get(base_col, 0)) if (tgt_col and base_col) else None
        L.append(f"| {name} | " + " | ".join(cells) +
                 f" | {_signed(dv) if dv is not None else '—'} | "
                 f"{_pct(vals.get(tgt_col, 0), vals.get(base_col, 0)) if (tgt_col and base_col) else '—'} |")
    if "_total" in table:
        tv = table["_total"]
        L.append("| **合计** | " + " | ".join(_num(tv.get(c, 0)) for c, _ in yrs) + " | — | — |")
    L.append("")

    # —— 变化要点(自动生成) ——
    L.append("**变化要点**\n")
    if tgt_col and base_col:
        chg = []
        for name in order:
            v = table.get(name, {})
            b, t = v.get(base_col, 0), v.get(tgt_col, 0)
            chg.append((t - b, name, b, t))
        chg.sort(key=lambda x: -abs(x[0]))
        for dv, name, b, t in chg[:3]:
            if abs(dv) < 1e-6:
                continue
            ny = max(target - late_year, 1)
            L.append(f"- **{name}** {_num(b)} → {_num(t)} km²"
                     f"({'增' if dv > 0 else '减'} {_num(abs(dv))} km²,"
                     f"{_pct(t, b)}),年均 {_signed(dv / ny)} km²/年")
        # 与历史速率对比
        if early_year:
            ey = next((c for c, y in yrs if y == early_year), None)
            if ey:
                for nm in order:
                    v = table.get(nm, {})
                    h = (v.get(base_col, 0) - v.get(ey, 0)) / max(late_year - early_year, 1)
                    p = (v.get(tgt_col, 0) - v.get(base_col, 0)) / max(target - late_year, 1)
                    if abs(h) > 1 and abs(p) > 1 and (h * p) > 0:
                        tag = "放缓" if abs(p) < abs(h) else "加速"
                        L.append(f"- {nm}变化速率由历史 {_signed(h)} km²/年 变为 "
                                 f"预测 {_signed(p)} km²/年({tag})")
                        break
        L.append("")
    return L


def _sec_products(out_dir, target, seq, fig_line, area_csv, report_path):
    L = ["## 6. 产物清单\n"]
    L.append("| 类型 | 文件 |")
    L.append("|---|---|")
    if seq:
        for s in seq:
            L.append(f"| {s['year']} 年模拟栅格 | `{s['sim']}` |")
    figs = []
    for f in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if not f.lower().endswith((".jpg", ".png")):
            continue
        if "sequence" in f:
            tag = "多期对比图"
        elif str(target) in f:
            tag = f"**目标年({target})成图**"
        elif "real" in f.lower():
            tag = "真实验证图(对照)"
        else:
            tag = "主模拟年成图"
        figs.append((tag, f))
    for tag, f in figs:
        L.append(f"| {tag} | `{os.path.join(out_dir, f)}` |")
    if area_csv and os.path.exists(area_csv):
        L.append(f"| 面积统计表 | `{area_csv}` |")
    L.append(f"| **本汇总报告** | `{report_path}` |")
    L.append("")
    if fig_line:
        L.append(f"> 出图日志:{fig_line}\n")
    return L


def _sec_limits():
    L = ["## 7. 已知限制\n"]
    L.append("1. **FoM 才是位置能力的真实指标,Kappa 不是**——变化像元稀少,OA 天然偏高。")
    L.append("2. 气候因子为 2.5° 粗分辨率;人口默认 2020 年;OSM 道路/水域在国内有遗漏。")
    L.append("3. 参数取官方手册默认(邻域 3 / 扩散 0.1 / 斑块 0.2 / 种子 0.1),未做调参寻优。")
    L.append("4. 水域锁定只禁止水域转出,不禁止其他类转入;若需冻结水域面积,"
             "还要在转换矩阵里禁止转入,且需重跑。")
    L.append("5. 面积极小的类(末期 <1000 像元,约 0.9 km²)难以生成独立斑块,"
             "其模拟结果与精度贡献都无统计意义,建议合并。")
    L.append("6. 预测结果依赖历史转移概率的稳定性;若未来出现重大政策/工程扰动"
             "(如大规模区划调整、重大水利工程),情景会偏离。")
    L.append("7. **引擎非确定性(实测)**:PLUS 引擎的 LEAS 随机森林未固定随机种子,同样输入重复运行"
             "会得到不同概率波段;CARS 结果随之小幅波动。同一研究区同参数连跑两次的实测:"
             "大类面积一致(耕地差 <0.01%),末期小类(数百像元)面积可差 25% 以上,"
             "逐像元位置也不同。故**大类结论稳定,小类数字与栅格不可逐位复现**,"
             "论文复现性表述宜写\"统计意义上可复现\",不承诺 bit-level 一致;小类一律不作结论。")
    L.append("")
    return L


def _sec_appendix(timings, moved, data_dir):
    L = ["## 附录\n"]
    if timings:
        L.append("**A. 分步耗时**\n")
        L.append("| 阶段 | 耗时 |")
        L.append("|---|---|")
        for t in timings:
            sec = t.get("seconds", 0) or 0
            L.append(f"| {t.get('stage', '')} | {sec / 60:.1f} min |")
        L.append("")
    else:
        L.append("**A. 分步耗时**:本次为报告重生成(未重跑建模),耗时分步日志见 "
                 f"`{os.path.join(data_dir, 'logs')}`。\n")
    # 子报告:本次归档的 + 之前已归档的都列上(报告可重复生成,不能只报本次移动的)
    dest_root = os.path.join(data_dir, "logs", SUBDIR)
    listed = []
    seen = set()
    for label, _src, dst in moved:
        listed.append((label, dst))
        seen.add(os.path.basename(dst))
    if os.path.isdir(dest_root):
        for f in sorted(os.listdir(dest_root)):
            if f.endswith(".md") and f not in seen:
                listed.append((f[:-3], os.path.join(dest_root, f)))
    if listed:
        L.append("**B. 子报告**(已归档,正文只摘录要点)\n")
        for label, dst in listed:
            L.append(f"- {label}:`{dst}`")
        L.append("")
    return L


# ---------------- 主入口 ----------------
def build_report(data_dir, out_name, proj, predict, region,
                 timings=None, fig_line="", area_csv=None,
                 backtest="none", mapping=None, water="open", archive=True):
    out_dir = os.path.join(data_dir, out_name) if out_name else data_dir
    rj = _load_json(os.path.join(out_dir, "report.json"))
    seq = (rj or {}).get("sequence") or []
    target = seq[-1]["year"] if seq else predict
    late_year = proj.get("late_year")
    early_year = proj.get("early_year")
    factors = 0
    ddir = os.path.join(data_dir, "drivers")
    if os.path.isdir(ddir):
        factors = len([x for x in os.listdir(ddir)
                       if x.startswith("df_") and x.endswith(".tif")])

    # 面积表:优先读已有 CSV(含历史上所有期),缺了就不出这节
    order, cols, table = None, None, None
    _at = read_area_table(area_csv or os.path.join(out_dir, "面积统计.csv"))
    if _at:
        order, cols, table = _at

    moved = archive_sub_reports(data_dir, out_dir) if archive else []
    # 预测报告引用了回测结果 → 回测目录自带的那份报告已并入本报告第 4 节,一并归档,
    # 保证一个研究区只有一份 汇总报告.md 在明面上
    bt = _find_backtest(data_dir, out_name)
    if archive and bt and not (rj or {}).get("metrics"):
        dest_root = os.path.join(data_dir, "logs", SUBDIR)
        for fn, newname, label in (
                ("汇总报告.md", f"{bt[0]}汇总(已并入主报告).md", "回测汇总"),
                ("报告.md", f"{bt[0]}过程报告.md", "回测建模过程")):
            src = os.path.join(data_dir, bt[0], fn)
            if not os.path.exists(src):
                continue
            try:
                os.makedirs(dest_root, exist_ok=True)
                dst = os.path.join(dest_root, newname)
                shutil.move(src, dst)
                moved.append((label, src, dst))
            except Exception as e:
                print(f"  ⚠️ 归档失败 {src}: {e}")

    report_path = os.path.join(out_dir, REPORT_NAME)
    class_names = proj.get("class_names") or {}

    L = []
    L.append(f"# {region} {target} 年土地利用模拟 · 汇总报告\n")
    _ds = _data_source(proj, data_dir)
    _head_src = {"clcd": "CLCD 30m ", "worldcover": "ESA WorldCover 10m ",
                 "user": "用户提供的 "}
    head = (f"以 {_head_src.get(_ds, _head_src['user'])}{early_year}–{late_year} 三期土地利用数据为训练期,"
            f"PLUS 模型(扩张分析 + LEAS 随机森林 + "
            f"{'马尔可夫链' if proj.get('demand') == 'markov' else '线性回归'}需求 + CARS 斑块模拟)")
    if seq:
        head += f"链式推演 {target} 年土地利用格局(中间期 {' / '.join(str(s['year']) for s in seq[:-1])})"
    else:
        head += f"推演 {target} 年土地利用格局"
    L.append(head + "。\n")
    m = (rj or {}).get("metrics")
    if m:
        L.append(f"> 精度:{'Kappa %.4f' % m['Kappa'] if m.get('Kappa') is not None else ''}"
                 f"{' / OA %.2f%%' % (m['OA'] * 100) if m.get('OA') is not None else ''}"
                 f"{' / FoM %.4f' % m['FoM'] if m.get('FoM') is not None else ''}"
                 f"({'真实面积诊断口径' if backtest == 'real' else '回测口径'})")
    else:
        if bt:
            bm = bt[1]["metrics"]
            L.append(f"> 精度:纯预测无真实图可比;可靠性参照同参数 2020 回测 "
                     f"Kappa {bm.get('Kappa', 0):.4f} / OA {(bm.get('OA') or 0) * 100:.2f}% / "
                     f"FoM {bm.get('FoM', 0):.4f}(真实面积诊断口径,只代表位置能力)")
        else:
            L.append("> 精度:纯预测,无真实图可比;如需可靠性指标请补跑 `--backtest real`")
    L.append("")

    L += _sec_meta(proj, predict, target, seq, region, backtest, water, mapping, factors)
    L += _sec_data(data_dir, proj, factors)
    L += _sec_reclass(data_dir, proj)
    L += _sec_matrix(proj)
    L += _sec_accuracy(rj, backtest, proj, bt)
    L += _sec_area(order, cols, table, target, late_year, early_year)
    L += _sec_products(out_dir, target, seq, fig_line, area_csv, report_path)
    L += _sec_limits()
    L += _sec_appendix(timings, moved, data_dir)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L).replace("\n\n\n", "\n\n"))
    print(f"  ✅ 唯一汇总报告: {report_path}")
    if moved:
        print(f"  📦 子报告已归档 {len(moved)} 份 → {os.path.join(data_dir, 'logs', SUBDIR)}")
    return report_path


def _auto_project(data_dir):
    """没给 --project 时,挑 data_dir 下最新的 project_full_*.json"""
    cands = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
             if f.startswith("project_full_") and f.endswith(".json")]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def main():
    import argparse
    p = argparse.ArgumentParser(description="生成唯一汇总报告(可独立运行,不重跑建模)")
    p.add_argument("--data-dir", required=True, help="研究区数据目录")
    p.add_argument("--project", default=None, help="项目配置 json;不填自动挑最新的 project_full_*.json")
    p.add_argument("--region", default=None, help="研究区名;默认取目录名")
    p.add_argument("--timing", default=None, help="分步耗时 json(可选)")
    p.add_argument("--no-archive", action="store_true", help="不归档子报告")
    a = p.parse_args()

    data_dir = os.path.abspath(a.data_dir)
    proj_path = a.project or _auto_project(data_dir)
    if not proj_path or not os.path.exists(proj_path):
        print(f"❌ 找不到项目配置 {proj_path}")
        sys.exit(1)
    proj = _load_json(proj_path) or {}
    proj.setdefault("data_dir", data_dir)
    out_name = os.path.basename(proj.get("output_dir") or f"model_{proj.get('predict_year')}")
    region = a.region or proj.get("region") or os.path.basename(data_dir)
    timings = _load_json(a.timing) if a.timing else _load_json(
        os.path.join(data_dir, out_name, "timing.json"))
    predict = int(proj.get("predict_year") or 0)

    build_report(data_dir, out_name, proj, predict, region,
                 timings=timings, area_csv=os.path.join(data_dir, out_name, "面积统计.csv"),
                 backtest=proj.get("backtest", "none"),
                 mapping=proj.get("mapping"), water="lock" if proj.get("locked_classes") else "open",
                 archive=not a.no_archive)


if __name__ == "__main__":
    main()
