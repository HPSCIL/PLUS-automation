import os
import sys
import argparse

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import force_utf8, NODATA_VALUES as BACKGROUND_VALUES

force_utf8()

# 【参数】背景值约定:算邻域权重时排除这些值(0=没变化的像元,255=NoData,-128=int8空值)
# 值统一来自 common.NODATA_VALUES(全局唯一一份,不再各文件各写)


def compute_weights(expansion_raster, num_classes):
    """从扩张图计算 CARS 邻域权重:每类扩张像元数 / 总扩张像元数

    原理(官方教程):扩张图里,值为 k 的像元 = 扩张成第 k 类的地方。
    各扩张类占比就是 CARS 里该类"随机种子扩张权重"的依据。
    纯 Python 计算,不调引擎,没有随机性,结果完全确定。
    """
    with rasterio.open(expansion_raster) as src:
        data = src.read(1)

    # 只保留有效的地类编码(排除背景和越界值)
    valid = data[~np.isin(data, BACKGROUND_VALUES)]
    valid = valid[(valid >= 1) & (valid <= num_classes)]

    total = int(len(valid))
    if total == 0:
        print("❌ 扩张图里没有有效像元")
        return None, None, 0

    # 数每个类有多少个像元(1..num_classes)
    counts = np.bincount(valid.astype(int), minlength=num_classes + 1)
    class_counts = [int(counts[k]) for k in range(1, num_classes + 1)]
    weights = [counts[k] / total for k in range(1, num_classes + 1)]
    return weights, class_counts, total


def main():
    parser = argparse.ArgumentParser(
        description="neighborhood_weight:扩张图 → CARS 邻域权重")
    parser.add_argument("--expansion", required=True, help="扩张图路径(expansion 的输出)")
    parser.add_argument("--classes", type=int, default=7, help="地类数,默认 7")
    parser.add_argument("--precision", type=int, default=7, help="权重小数位数,默认 7(与手动痕迹一致)")
    parser.add_argument("--compare", default=None,
                        help="手动标准答案的权重字符串,如 \"0.0333542,0.0720101,...\"")
    args = parser.parse_args()

    args.expansion = os.path.abspath(args.expansion)  # 路径归一化

    print(f"扩张图: {args.expansion}")
    weights, counts, total = compute_weights(args.expansion, args.classes)
    if weights is None:
        sys.exit(1)  # 失败给非零退出码

    print(f"\n  各类扩张像元数: {counts}")
    print(f"  总扩张像元数: {total:,}")
    print(f"\n  各类权重(占比):")
    for k in range(1, args.classes + 1):
        print(f"    地类 {k}: {counts[k-1]:>9,} / {total:,} = {weights[k-1]:.{args.precision}f}")

    # 拼成 CARS 操作单要的格式
    weight_str = ",".join(f"{w:.{args.precision}f}" for w in weights)
    print(f"\n  供 CARS 使用的权重串:")
    print(f"  {weight_str}")

    if args.compare:
        manual = args.compare.strip()
        print(f"\n=== 和手动标准答案对比 ===")
        print(f"  手动: {manual}")
        print(f"  自动: {weight_str}")
        if manual == weight_str:
            print("  ✅ 完全一致!neighborhood_weight 模块拿下")
        else:
            # 可能差在最后一位小数(浮点精度),逐位对比
            m = [float(x) for x in manual.split(",")]
            print("  字符串不完全一致,数值对比(前 6 位小数):")
            all_close = True
            for k in range(args.classes):
                diff = abs(m[k] - weights[k])
                flag = "✅" if diff < 1e-6 else "⚠️"
                if diff >= 1e-6:
                    all_close = False
                print(f"    地类 {k+1}: 手动 {m[k]:.7f} vs 自动 {weights[k]:.7f} 差 {diff:.2e} {flag}")
            if all_close:
                print("  ✅ 数值一致(差异 < 1e-6,仅最后一位小数的舍入不同)——模块拿下")


if __name__ == "__main__":
    main()
