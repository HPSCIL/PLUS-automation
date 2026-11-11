import os
import sys
import argparse
import subprocess

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import force_utf8, get_plus_backend, run_engine

force_utf8()

# 引擎目录由 common 解析:环境变量 PLUS_BACKEND > config.json > 内置默认
PLUS_BACKEND = get_plus_backend()


def build_expansion_tmp(early_lulc, late_lulc, output_change):
    """构造"提取土地扩张"的操作单(PLUS 引擎规定的文本格式)

    标签是引擎自己定义的(官方教程 4.2 节"提取土地扩张"):
        <Input number>        输入图像个数
        <Input LULC series>   输入图像路径列表(起期在前、止期在后,顺序不能反)
        <Output change>       输出扩张图路径
    """
    return (
        "<Input number>\n"
        "2\n"
        "<Input LULC series>\n"
        f"{early_lulc}\n"
        f"{late_lulc}\n"
        "<Output change>\n"
        f"{output_change}\n"
    )


def run_expansion(early_lulc, late_lulc, output_change,
                  plus_backend=PLUS_BACKEND, timeout=600):
    """调用引擎跑 expansion,返回实际输出文件路径(失败返回 None)

    和 convert 一个套路:写操作单 → 跑 bat → 查输出。
    注意:引擎会把输出重命名为 <你起的名字>_landuse_1to2.tif。
    """
    # 步骤 1:删掉可能存在的旧输出。
    # 实测:引擎内部失败时打印错误但返回码仍是 0、且不写文件——
    # 不先删的话,上一次运行的旧文件会被误当成这次的结果。
    stem, ext = os.path.splitext(output_change)
    actual = stem + "_landuse_1to2" + ext
    for p in (actual, output_change):
        if os.path.exists(p):
            os.remove(p)
            print(f"🧹 已清理旧输出: {p}")

    # 步骤 2:写操作单 + 跑 bat(公共样板,含失败检测)
    ok, _, _ = run_engine("expansion.bat", "PLUS_Expansion.tmp",
                          build_expansion_tmp(early_lulc, late_lulc, output_change),
                          plus_backend, timeout, ("failed", "nan(ind)"))
    if not ok:
        return None

    # 步骤 3:查输出(旧文件已删,存在即本次产物)。引擎惯例:输出重命名为 <名字>_landuse_1to2.tif
    if os.path.exists(actual):
        print(f"✅ 扩张图生成: {actual}")
        return actual
    # 兜底:也许没有重命名,直接用了原路径
    if os.path.exists(output_change):
        print(f"✅ 扩张图生成(未重命名): {output_change}")
        return output_change
    print(f"❌ 输出文件不存在: {actual}")
    return None


def check_expansion(tif_path):
    """检查扩张图格式:uint8;值 0=没变,1~7=扩张成的地类,255=NoData"""
    with rasterio.open(tif_path) as src:
        a = src.read(1)
        print(f"  行列: {src.height}x{src.width}  类型: {src.dtypes[0]}")
        print(f"  唯一值: {np.unique(a)}")
        ok = src.dtypes[0] == "uint8"
        print("  ✅ 格式合规" if ok else "  ⚠️ 类型不是 uint8")
    return ok


def compare_with_standard(actual, standard):
    """逐像素和手动标准答案对比,返回一致率(1.0 = 完全一致)"""
    with rasterio.open(actual) as s1, rasterio.open(standard) as s2:
        a, b = s1.read(1), s2.read(1)
    if a.shape != b.shape:
        print(f"❌ 行列不一致: 自动 {a.shape} vs 标准 {b.shape}")
        return 0.0
    same = (a == b).sum()
    total = a.size
    rate = same / total
    print(f"  一致像元: {same:,}/{total:,} = {rate:.4%}")
    if rate < 1.0:
        diff = a != b
        print(f"  差异像元: {diff.sum():,},前 5 个差异位置(自动→标准):")
        ys, xs = np.where(diff)
        for i in range(min(5, diff.sum())):
            y, x = ys[i], xs[i]
            print(f"    行{y},列{x}: {a[y, x]} → {b[y, x]}")
    return rate


def main():
    parser = argparse.ArgumentParser(
        description="提取土地扩张(expansion):两期土地利用图 → 扩张图")
    parser.add_argument("--early", required=True, help="起始期土地利用图路径")
    parser.add_argument("--late", required=True, help="结束期土地利用图路径")
    parser.add_argument("--output", required=True, help="输出路径(引擎会自动加 _landuse_1to2 后缀)")
    parser.add_argument("--plus-backend", default=PLUS_BACKEND, help="引擎目录,默认用内置路径")
    parser.add_argument("--compare", default=None, help="手动标准答案路径,给了就逐像素对比")
    args = parser.parse_args()

    # 路径全部转成绝对路径:引擎的工作目录是 plus-backend,相对路径会写错地方
    args.early = os.path.abspath(args.early)
    args.late = os.path.abspath(args.late)
    args.output = os.path.abspath(args.output)
    if args.compare:
        args.compare = os.path.abspath(args.compare)

    print(f"起始期: {args.early}")
    print(f"结束期: {args.late}")
    actual = run_expansion(args.early, args.late, args.output, args.plus_backend)
    if actual is None:
        sys.exit(1)  # 失败要给非零退出码,调用方(脚本/AI)靠它判断成败
    check_expansion(actual)
    if args.compare:
        print(f"\n和手动标准答案逐像素对比: {args.compare}")
        rate = compare_with_standard(actual, args.compare)
        if rate == 1.0:
            print("✅ 完全一致!自动流程 = 手动流程,expansion 模块拿下")
        else:
            print(f"⚠️ 有差异,一致率 {rate:.4%},需要排查")


if __name__ == "__main__":
    main()
