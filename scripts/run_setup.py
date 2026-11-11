"""PLUS skill 首次使用引导:数据核对与一致性检查

用户在对话里粘贴"所有数据所在文件夹"的路径后,本脚本负责:
    1. 扫描文件夹里所有 tif(递归)
    2. 智能分类:哪些像土地利用图(名字含年份/landuse/lulc)、哪些像驱动因子
    3. 一致性检查:行列数、投影是否统一(以出现最多的组合为基准)
    4. 格式检查:dtype 是否 uint8、编码是否从 1 连续
    5. 输出核对报告:齐了什么、缺了什么、什么不对

用法:
    python run_setup.py --data-dir <用户粘贴的数据文件夹路径> [--save-report]

之后 AI 按报告结果向用户收"决策参数"(转移矩阵/驱动因子取舍/policy/年份),
收集齐了写配置,再开工跑流程。这就是安装后的第一步。
"""

import os
import sys
import re
import glob
import argparse
from collections import Counter

import numpy as np
import rasterio

# 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 【参数】无效值约定(PLUS 协议:0=背景 255=NoData -128=int8空值)
NODATA_VALUES = (0, 255, -128)

# 名字里含这些关键词的,更可能是土地利用图(而不是驱动因子)
LULC_KEYWORDS = ("lulc", "landuse", "land_use", "refy", "lucc", "landcover")
YEAR_RE = re.compile(r"(19\d\d|20\d\d)")


def classify(rel_path):
    """按完整路径(含文件夹名)猜文件类型:landuse / factor / policy / other

    只按文件名猜会漏:驱动因子常放在 dringfactor 这类文件夹里,
    而文件本身叫 wh_df_dem.tif 这种看不出类型的名字。
    优先级:文件夹级信号(factor/drive)最强——dringfactor 里的 water 距离因子
    应该算因子,不能因为名字带 water 就被标成约束图(审查发现过的坑)。
    """
    low = rel_path.lower()
    if "factor" in low or "drive" in low or "driving" in low or "_df_" in low:
        return "factor/驱动因子(可能)"
    if "water" in low or "policy" in low or "planning" in low:
        return "policy/约束(可能)"
    if any(k in low for k in LULC_KEYWORDS):
        return "landuse/土地利用(可能)"
    return "other/其他"


def scan_and_check(data_dir):
    """扫描 + 检查,返回报告 dict"""
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.tif"), recursive=True))
    print(f"扫描到 {len(files)} 个 tif 文件:\n")

    infos = []
    for f in files:
        rel = os.path.relpath(f, data_dir)
        try:
            with rasterio.open(f) as src:
                info = {
                    "path": f, "rel": rel,
                    "shape": (src.height, src.width),
                    # 无投影的栅格要单独标记成问题,不能和正常投影混进同一个 "None" 字符串
                    "crs": str(src.crs) if src.crs is not None else None,
                    "dtype": src.dtypes[0],
                    "kind": classify(rel),
                }
                # 年份从文件名抓
                m = YEAR_RE.search(os.path.basename(f))
                info["year"] = m.group(1) if m else None
                infos.append(info)
        except Exception as e:
            print(f"  ⚠️ 打不开(跳过): {rel} — {e}")

    if not infos:
        print("❌ 一个 tif 都没读成功,检查路径或文件")
        return None

    # 一致性基准:出现最多的 (行列, 投影) 组合
    shape_c = Counter(i["shape"] for i in infos)
    crs_c = Counter(i["crs"] for i in infos)
    ref_shape = shape_c.most_common(1)[0][0]
    ref_crs = crs_c.most_common(1)[0][0]

    print(f"\n基准(出现最多):行列 {ref_shape[0]}×{ref_shape[1]},投影 {ref_crs}")
    print(f"\n{'文件':<55} {'类型':<22} {'年份':<6} {'行列':<14} {'类型':<8} {'一致性'}")
    print("-" * 120)

    lulc_files, factor_files = [], []
    for i in infos:
        shape_ok = i["shape"] == ref_shape
        crs_ok = i["crs"] is not None and i["crs"] == ref_crs
        if shape_ok and crs_ok:
            mark = "✅"
        else:
            mark = "❌" + ("行列不同" if not shape_ok else "") + ("无投影" if i["crs"] is None else ("投影不同" if not crs_ok else ""))
        year = i["year"] or ""
        print(f"{i['rel']:<55} {i['kind']:<22} {year:<6} "
              f"{str(i['shape'][0])+'×'+str(i['shape'][1]):<14} {i['dtype']:<8} {mark}")
        if i["kind"].startswith("landuse"):
            lulc_files.append(i)
        elif i["kind"].startswith("factor"):
            factor_files.append(i)

    # 结论
    print("\n" + "=" * 60)
    print("核对结论")
    print("=" * 60)
    problems = []

    lulc_by_year = sorted(lulc_files, key=lambda i: i["year"] or "9999")
    print(f"\n1) 土地利用图:{len(lulc_files)} 个")
    if not lulc_files:
        problems.append("没有找到土地利用图(名字含 lulc/landuse/refy 的 tif)")
    else:
        for i in lulc_by_year:
            tag = f"{i['year']} 年" if i["year"] else "年份未知"
            # 真读像素值做格式检查(docstring 承诺的检查,审查发现之前根本没实现):
            # dtype 必须 uint8;有效编码必须从 1 开始且连续
            try:
                with rasterio.open(i["path"]) as src:
                    vals = src.read(1)
                classes = np.unique(vals[~np.isin(vals, NODATA_VALUES)])
                classes = classes[classes >= 1]
                if i["dtype"] != "uint8":
                    problems.append(f"{i['rel']} 不是 uint8({i['dtype']}),需要先转换")
                if len(classes) == 0:
                    problems.append(f"{i['rel']} 没有有效地类(全是无效值)")
                else:
                    mn, mx = int(classes.min()), int(classes.max())
                    expected = set(range(mn, mx + 1))
                    if mn != 1:
                        problems.append(f"{i['rel']} 地类编码不是从 1 开始(最小 {mn})")
                    if expected != set(int(c) for c in classes):
                        missing = expected - set(int(c) for c in classes)
                        problems.append(f"{i['rel']} 地类编码不连续,缺 {sorted(missing)}")
                print(f"   - {i['rel']}  [{tag}] 有效编码 {[int(c) for c in classes][:12]}")
            except Exception as e:
                problems.append(f"{i['rel']} 像素值读取失败: {e}")
        if len(lulc_files) < 2:
            problems.append("土地利用图少于两期(模拟至少需要起止两期)")

    print(f"\n2) 驱动因子:{len(factor_files)} 个")
    if not factor_files:
        problems.append("没有找到驱动因子文件夹(tif 文件名含 factor/drive)")
    else:
        print(f"   - 共 {len(factor_files)} 个因子文件")

    print(f"\n3) 一致性:基准行列 {ref_shape[0]}×{ref_shape[1]}")
    bad = [i for i in infos if i["shape"] != ref_shape or i["crs"] != ref_crs]
    if bad:
        problems.append(f"{len(bad)} 个文件行列/投影与基准不一致(见上表 ❌)")
    else:
        print("   ✅ 全部文件行列/投影一致")
    # 无投影的栅格单独拦:PLUS 读不了没坐标系的图
    no_crs = [i for i in infos if i["crs"] is None]
    if no_crs:
        problems.append(f"{len(no_crs)} 个文件没有投影信息(需在 QGIS/ArcGIS 里定义坐标系后重新导出)")
        for i in no_crs:
            print(f"   ⚠️ 无投影: {i['rel']}")

    if problems:
        print("\n" + "=" * 60)
        print("需要用户处理的问题")
        print("=" * 60)
        for p in problems:
            print(f"  ❌ {p}")
    else:
        print("\n  ✅ 数据齐备且一致,可以收决策参数开工")

    return {"infos": infos, "lulc": lulc_by_year, "factors": factor_files,
            "ref_shape": ref_shape, "ref_crs": ref_crs, "problems": problems}


def main():
    parser = argparse.ArgumentParser(description="PLUS 首次使用引导:数据核对与一致性检查")
    parser.add_argument("--data-dir", required=True, help="用户粘贴的数据文件夹路径(递归扫描)")
    parser.add_argument("--save-report", action="store_true", help="把报告保存到数据目录")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"❌ 路径不存在或不是文件夹: {args.data_dir}")
        sys.exit(1)

    args.data_dir = os.path.abspath(args.data_dir)
    report = scan_and_check(args.data_dir)
    if report is None:
        sys.exit(1)  # 数据门禁失败给非零退出码
    if args.save_report:
        out = os.path.join(args.data_dir, "数据核对报告.md")
        lines = ["# PLUS 数据核对报告", "",
                 f"- 数据目录:{args.data_dir}", f"- 基准行列:{report['ref_shape'][0]}×{report['ref_shape'][1]}",
                 f"- 基准投影:{report['ref_crs']}", "",
                 "## 文件清单", ""]
        for i in report["infos"]:
            lines.append(f"- {i['rel']} | {i['kind']} | {i['dtype']} | {i['shape'][0]}×{i['shape'][1]}")
        lines += ["", "## 问题", ""]
        lines += [f"- {p}" for p in (report["problems"] or ["- 无"])]
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n✅ 报告已保存: {out}")
    if report["problems"]:
        sys.exit(1)  # 有问题时不输出"开工"信号,调用方按退出码拦截


if __name__ == "__main__":
    main()
