"""土地利用图语义重分类:数据源编码 → 官方图例语义 → 研究分类体系

为什么需要这个脚本:
    不同数据源的编码含义不同!同样是"5",CLCD 是水域,而 PLUS 官方教程数据
    是裸地。直接拿编码建模会把地类搞错。本脚本强制走三步,杜绝"猜码":

        1. 查内置图例:源编码 → 语义名(如 CLCD 5 → 水域)
        2. 用户映射:语义名 → 研究分类体系编码(如 水域 → 4)
        3. 数据里有但映射里没有的类别 → 直接报错,逼用户显式决策
           (合并小类就写成 裸地→耕地 这种映射,不会悄悄丢数据)

用法:
    python reclassify_lulc.py <tif...> --source clcd --mapping 映射.json [--output-dir 目录]

映射.json 格式(语义名 → 目标编码):
    {"耕地": 1, "森林": 2, "草地": 3, "水域": 4, "裸地": 5, "不透水面": 6}
"""

import os
import sys
import json
import argparse

import numpy as np
import rasterio

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

# ==================== 0 像元类成因分析与处理建议 ====================
# "起始年某类 0 像元"会触发 markov 全表 NaN。处理方法不能一刀切,
# 必须:①分析该地信息(位置+前后年份轨迹)②按文献惯例给建议 ③输出说明存档。
# 规则表(2026-11-01 文献检索整理;只有匹配到规则才给默认建议,否则要求人工判断):
ZERO_CLASS_RULES = [
    # (前驱语义集合, 后继语义集合, 成因, 建议归宿, 依据)
    ({"水域"}, {"不透水面"},
     "填湖/开发区工地:建设用地开发的中间态(裸土施工期)",
     "并入建设用地",
     "本地轨迹 水→裸→建;武汉地区 PLUS 研究惯例'裸地极少时并入建设用地'(自然资源遥感 2026; MDPI Land 2026)"),
    ({"耕地"}, {"耕地"},
     "休耕/轮作:耕地管理的中间态",
     "并入耕地",
     "CLCD 裸地↔耕地存在混淆(ESSD 2021 混淆矩阵);休耕地并入耕地的农学惯例"),
    ({"草地"}, {"草地"},
     "分类噪声:裸地-草地混淆(CLCD 在部分区域把 >70% 裸地误分为草地)",
     "并入草地",
     "水土保持学报 2024:CLCD 裸地 71.53% 被误分为草地"),
]


def analyze_zero_classes(tifs, legend, mapping, zero_classes, out_dir):
    """对起始年 0 像元的类自动分析:位置 + 前后年份轨迹 → 成因 → 文献建议

    返回 {类编码: 分析文本}(供 重分类说明.md 存档)。
    """
    # 读全部原始图(轨迹分析用源编码,不用重分类后的编码)
    arrays = {}
    transform0 = None
    src_crs = None
    for tif in tifs:
        with rasterio.open(tif) as s:
            arrays[os.path.basename(tif)] = s.read(1)
            if transform0 is None:
                transform0 = s.transform
                src_crs = s.crs

    from rasterio.warp import transform as warp_transform
    notes = {}
    names = list(arrays)
    for cls in zero_classes:
        # 目标编码 cls 对应的源编码(可能有多个源类映射过来)
        src_codes = [code for code, name in legend.items()
                     if code != 0 and mapping.get(name) == cls]
        # 找该类有像元的年份(用于取坐标),以及起始年这些位置的编码(前驱)
        present_years = [n for n in names if np.isin(arrays[n], src_codes).any()]
        if not present_years:
            notes[cls] = f"类别 {cls} 三期均无像元(纯缺席,可整体剔除)。"
            continue
        ref = present_years[0]
        mask = np.isin(arrays[ref], src_codes)
        rows, cols = np.where(mask)
        xs = transform0.c + (cols + 0.5) * transform0.a
        ys = transform0.f + (rows + 0.5) * transform0.e
        try:
            lons, lats = warp_transform(src_crs, "EPSG:4326", xs, ys)
        except Exception:
            lons, lats = xs, ys
        # warp_transform 返回 list(不是 ndarray),必须转数组才能 .mean()
        lons, lats = np.asarray(lons, dtype="float64"), np.asarray(lats, dtype="float64")
        # 前驱:起始年(第一张图)在这些位置上的编码分布
        pred_codes = arrays[names[0]][rows, cols]
        pred_sem = _semantic_distribution(pred_codes, legend)
        succ_codes = arrays[names[-1]][rows, cols]
        succ_sem = _semantic_distribution(succ_codes, legend)

        # 匹配内置规则(前驱主要语义 ∈ 前驱集合 且 后继主要语义 ∈ 后继集合)
        pred_main = max(pred_sem, key=pred_sem.get) if pred_sem else "无"
        succ_main = max(succ_sem, key=succ_sem.get) if succ_sem else "无"
        rule = next((r for r in ZERO_CLASS_RULES
                     if pred_main in r[0] and succ_main in r[1]), None)
        loc = (f"({lons.mean():.4f}E, {lats.mean():.4f}N)" if len(lons)
               else "(无像元)")
        text = (f"- 类别 {cls}:{len(rows)} 个像元位于 {loc}\n"
                f"  轨迹:起始年为 {pred_sem},末期 {succ_sem}\n")
        if rule:
            text += (f"  成因: {rule[2]}\n"
                     f"  ✅ 建议: {rule[3]}\n"
                     f"  依据: {rule[4]}\n")
        else:
            text += "  ⚠️ 未匹配内置规则,需人工判断处理方式(不允许默认合并)\n"
        notes[cls] = text
        print("\n" + text)
    return notes


def _semantic_distribution(codes, legend):
    """编码数组 → {语义名: 像元数}"""
    from collections import Counter
    return {legend.get(int(c), f"编码{c}"): n for c, n in Counter(codes.tolist()).items()}


def write_reclassify_report(out_dir, notes, tifs, mapping):
    """重分类说明.md:问题/成因/处理选择存档(供最终建模报告引用)"""
    lines = ["# 重分类与特殊情况说明", "",
             "## 输入", ""]
    for t in tifs:
        lines.append(f"- {os.path.basename(t)}")
    lines += ["", f"## 用户映射(语义 → 目标编码)", "", f"```json", "",
              json.dumps(mapping, ensure_ascii=False, indent=2), "```"]
    if notes:
        lines += ["", "## 特殊情况:起始年 0 像元类(markov NaN 风险)", ""]
        for cls, text in notes.items():
            lines += [text, ""]
    else:
        lines += ["", "## 特殊情况:无(各类在起始年均存在)", ""]
    path = os.path.join(out_dir, "重分类说明.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✅ 重分类说明已保存: {path}")
    return path
# ==================== 内置数据源官方图例 ====================
# 只收录"已核实"的图例(来源写在注释里);加新数据源前必须先去官方文档核对编码含义,
# 不要凭记忆填——编码搞错比没有图例更危险(2026-11-01 教训:曾把 CLCD 6/9 写反)
BUILTIN_LEGENDS = {
    # CLCD(China Land Cover Dataset,Yang & Huang 2021,ESSD 13:3907-3925)
    # 图例核实自论文正文:9 类 + 0 背景
    "clcd": {
        0: "背景", 1: "耕地", 2: "森林", 3: "灌木", 4: "草地", 5: "水域",
        6: "冰雪", 7: "裸地", 8: "不透水面", 9: "湿地",
    },
    # PLUS 官方教程数据(本实验室 project.json 记录:wh2003_refy_uc 等 7 类)
    # 1 草地 / 2 落叶林 / 3 耕地 / 4 城镇用地 / 5 裸地 / 6 水域 / 7 常绿林
    "plus_tutorial": {
        0: "背景", 1: "草地", 2: "落叶林", 3: "耕地", 4: "城镇用地",
        5: "裸地", 6: "水域", 7: "常绿林",
    },
    # ESA WorldCover v100/v200(10m 全球产品,官方 esa-worldcover.org,CC BY 4.0)
    # 11 类编码核实自官方产品说明(2026-11-10);按用户决策默认并类:
    # 70 冰雪→裸地、95 红树林→湿地、100 苔藓地衣→草地(多编码同名会自然合并到同一目标编码)
    "worldcover": {
        0: "背景", 10: "森林", 20: "灌木", 30: "草地", 40: "耕地",
        50: "不透水面", 60: "裸地", 70: "裸地", 80: "水域",
        90: "湿地", 95: "湿地", 100: "草地",
    },
    # Esri Sentinel-2 10m Land Cover(官方 Living Atlas 条目 cfcb7609de5f478eb7666240902d4d3d
    # 描述核实,2026-11-11):9 类、2017-2025 逐年;10 云为遮挡无效→背景;
    # 9 冰雪并入裸地、11 牧场并入草地(与 WorldCover 并类口径一致)
    "esri": {
        0: "背景", 1: "水域", 2: "森林", 4: "湿地", 5: "耕地",
        7: "不透水面", 8: "裸地", 9: "裸地", 10: "背景", 11: "草地",
    },
    # TODO 待核实后加入:GlobeLand30 / CNLUCC / GLC_FCS30 / MODIS …
}


def load_mapping(path):
    """读映射文件 {语义名: 目标编码};支持顶层 "_notes" 键(特殊情况说明,存档用)"""
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    assert isinstance(m, dict), "映射文件必须是 JSON 对象"
    notes = m.pop("_notes", "")  # 取出说明,不参与映射
    return m, notes


def reclassify(src_path, legend, mapping, out_path):
    """源图 → 目标编码图(逐像元查表)"""
    with rasterio.open(src_path) as src:
        a = src.read(1)
        # 统计源编码分布,顺便找出"数据里有但映射里没有"的类别
        src_codes, counts = np.unique(a, return_counts=True)
        unmapped = []
        for code in src_codes:
            if int(code) == 0 and legend.get(int(code)) == "背景":
                continue  # 背景 0 固定映射为 0
            name = legend.get(int(code))
            if name is None:
                unmapped.append(f"编码 {int(code)} 不在该数据源的内置图例里(请先核实数据源图例)")
            elif name not in mapping:
                unmapped.append(f"语义 [{name}](源编码 {int(code)},{counts[list(src_codes).index(code)]} 像元)不在映射文件里")
        if unmapped:
            print("❌ 以下类别没有映射,请显式决策(合并就写 裸地→耕地 这类映射):")
            for u in unmapped:
                print(f"   - {u}")
            raise RuntimeError("存在未映射类别,已中止(宁可报错,不悄悄丢数据)")

        # 逐像元重编码:源编码 → 语义 → 目标编码;0 背景 → 0
        out = np.zeros_like(a)
        for code in src_codes:
            name = legend.get(int(code))
            target = 0 if (int(code) == 0 or name == "背景") else mapping[name]
            out[a == code] = target

        # 检查输出编码连续性(PLUS 要求从 1 开始连续)
        dst_codes = np.unique(out)
        expect = list(range(1, int(dst_codes.max()) + 1))
        actual = [c for c in dst_codes if c != 0]
        if actual != expect:
            print(f"⚠️ 目标编码不连续: 实际 {actual},期望 {expect}(PLUS 会报错)")
            print("   常见原因:某一年的原始图缺少某类(该类在这一年 0 像元)→ 编码留空")
            print("   若属'起始年 0 像元类',run_full 会自动归并后重跑本步,可忽略此提示")

        profile = src.profile.copy()
        profile.update(compress="deflate")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)

        print(f"✅ {os.path.basename(src_path)} → {os.path.basename(out_path)}")
        for code, cnt in zip(src_codes, counts):
            name = legend.get(int(code), f"未知{code}")
            print(f"   {name:>6s}(源编码 {int(code)}): {cnt} 像元")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="土地利用语义重分类(数据源图例 → 研究分类体系)")
    parser.add_argument("tifs", nargs="+", help="要重分类的 tif(可多个,如三期图)")
    parser.add_argument("--source", required=True, choices=list(BUILTIN_LEGENDS),
                        help="数据源(内置图例): " + ",".join(BUILTIN_LEGENDS))
    parser.add_argument("--mapping", required=True, help="映射 JSON 文件:{语义名: 目标编码}")
    parser.add_argument("--output-dir", default=None, help="输出目录(默认与输入同目录,文件名加 _rc 后缀)")
    args = parser.parse_args()

    legend = BUILTIN_LEGENDS[args.source]
    mapping, mapping_notes = load_mapping(args.mapping)

    print(f"数据源: {args.source}  内置图例(已核实):")
    for code, name in legend.items():
        if code != 0:
            print(f"   {code} = {name}")
    print(f"用户映射(语义 → 目标编码): {mapping}\n")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.tifs[0]))
    series_counts = {}  # {图名: {编码: 像元数}}(多期图时用来做"0 像元类"预警)
    for tif in args.tifs:
        stem, ext = os.path.splitext(os.path.basename(tif))
        out_path = os.path.join(out_dir, stem + "_rc" + ext)
        reclassify(tif, legend, mapping, out_path)
        with rasterio.open(out_path) as s:
            vals, cnt = np.unique(s.read(1), return_counts=True)
        series_counts[os.path.basename(tif)] = dict(zip(map(int, vals), map(int, cnt)))

    # 多期图预警:PLUS 的 markov 引擎对"起始年 0 像元的类"会算出 0/0=NaN 并污染全表
    # (2026-11-01 鄂州实测踩坑:裸地 2010 年 0 像元 → 全表 NaN)。建模前必须拦住。
    notes = {}
    if len(series_counts) > 1:
        all_classes = sorted({c for d in series_counts.values() for c in d if c != 0})
        print("\n=== 多期图类别存在性预警(markov 要求起始年每类 ≥1 像元)===")
        names = list(series_counts)
        zero_early = []
        for i, name in enumerate(names):
            d = series_counts[name]
            zero = [c for c in all_classes if d.get(c, 0) == 0]
            tag = "起始年" if i == 0 else "后续年"
            if zero:
                extra = ("起始年 0 像元会触发 markov 全表 NaN,必须合并/剔除!"
                         if i == 0 else "仅后续年缺失不影响 markov,但需求会预测出 0")
                print(f"  警告 {name}({tag}):0 像元类别 {zero} —— {extra}")
                if i == 0:
                    zero_early = zero
            else:
                print(f"  {name}({tag}):各类均存在 ✓")

        # 起始年 0 像元类 → 自动启动"该地信息收集 + 成因分析"流程:
        # 位置 + 前后年份轨迹 → 匹配内置文献规则 → 给建议(没匹配到不允许默认处理)
        if zero_early:
            print("\n=== 0 像元类成因自动分析(位置 + 轨迹 + 文献建议)===")
            notes = analyze_zero_classes(args.tifs, legend, mapping, zero_early, out_dir)

    # 重分类说明存档(问题/成因/处理选择,供最终建模报告引用)
    if mapping_notes:
        notes["_mapping_notes"] = "- " + mapping_notes.replace("\n", "\n- ")
    write_reclassify_report(out_dir, notes, args.tifs, mapping)


if __name__ == "__main__":
    main()
