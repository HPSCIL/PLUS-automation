"""LEAS 驱动因子贡献度排序对比:手动 vs 自动

手动(Parameterfile\Contribution0~6.csv)和自动(归档目录的 Contribution*.csv)
各算一套"因子重要性排序",对比两套排序的一致性:
    - 每类 top-5 因子重合度
    - 每类 Spearman 秩相关系数(排序整体一致性)
这是 leas"统计等价"验证的第三根支柱(前两根:概率相关性、OOB 精度同量级)。

用法:
    python compare_contributions.py --manual <手动Contribution目录> --auto <自动归档目录> [--classes 7]
"""

import os
import sys
import glob
import csv
import argparse

# 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_contribution(csv_path):
    """读一个 ContributionN.csv,返回 {因子名: 贡献值}"""
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    factors, contrib = None, None
    for row in rows:
        if not row:
            continue
        key = row[0].strip()
        if key == "Factors":
            factors = [c.strip() for c in row[1:] if c.strip()]
        elif key.startswith("Contribution"):
            contrib = []
            for v in row[1:]:
                if not v.strip():
                    continue  # 行尾逗号产生的空字段,跳过而不是当 0
                try:
                    contrib.append(float(v.replace(",", "").strip()))
                except ValueError:
                    continue
    if not factors or not contrib or len(factors) != len(contrib):
        return None
    return dict(zip(factors, contrib))


def rank_names(contrib_dict):
    """按贡献值从大到小排,返回因子名列表"""
    return [name for name, _ in
            sorted(contrib_dict.items(), key=lambda kv: -kv[1])]


def spearman(a, b):
    """两套完整排序的 Spearman 秩相关系数(a、b 为同名列表)"""
    n = len(a)
    ra = {name: i for i, name in enumerate(a)}
    rb = {name: i for i, name in enumerate(b)}
    common = [name for name in ra if name in rb]
    d2 = sum((ra[name] - rb[name]) ** 2 for name in common)
    m = len(common)
    if m < 2:
        return None
    return 1 - 6 * d2 / (m * (m * m - 1))


def main():
    parser = argparse.ArgumentParser(description="LEAS 贡献度排序对比(手动 vs 自动)")
    parser.add_argument("--manual", required=True, help="手动 Contribution*.csv 目录(如 GUI 的 Parameterfile)")
    parser.add_argument("--auto", required=True, help="自动归档目录")
    parser.add_argument("--classes", type=int, default=7, help="地类数,默认 7")
    parser.add_argument("--top", type=int, default=5, help="top-N 重合度,N 默认 5")
    args = parser.parse_args()

    print(f"{'类':>4}  {'top{0}重合'.format(args.top):>12}  {'Spearman':>10}  手动 top5 / 自动 top5")
    print("-" * 100)
    overlaps, rhos = [], []
    for i in range(args.classes):
        mf = os.path.join(args.manual, f"Contribution{i}.csv")
        af_candidates = sorted(glob.glob(os.path.join(args.auto, f"Contribution{i}.csv")))
        if not af_candidates:
            # 自动归档可能编号错位,尝试任何匹配
            af_candidates = sorted(glob.glob(os.path.join(args.auto, "Contribution*.csv")))
        if not os.path.exists(mf) or not af_candidates:
            print(f"  {i:>4}  缺文件(手动:{os.path.basename(mf)},自动:{len(af_candidates)} 个)")
            continue
        af = af_candidates[i % len(af_candidates)]
        cm = load_contribution(mf)
        ca = load_contribution(af)
        if not cm or not ca:
            print(f"  {i:>4}  解析失败")
            continue
        rm, ra = rank_names(cm), rank_names(ca)
        top_m, top_a = set(rm[:args.top]), set(ra[:args.top])
        overlap = len(top_m & top_a) / args.top
        rho = spearman(rm, ra)
        overlaps.append(overlap)
        if rho is not None:
            rhos.append(rho)
        print(f"  {i:>4}  {overlap:>11.0%}  {rho:>10.3f}  {'、'.join(x.split('.')[0] for x in rm[:args.top])} / {'、'.join(x.split('.')[0] for x in ra[:args.top])}")
    print("-" * 100)
    if overlaps:
        print(f"  平均 top-{args.top} 重合度: {sum(overlaps)/len(overlaps):.0%}")
    if rhos:
        print(f"  平均 Spearman: {sum(rhos)/len(rhos):.3f}")
    if overlaps and sum(overlaps)/len(overlaps) >= 0.8 and rhos and sum(rhos)/len(rhos) >= 0.9:
        print("  ✅ 贡献度排序高度一致——leas 统计等价的第三根支柱成立")
    elif overlaps:
        print("  ⚠️ 排序有差异,需要看具体类别(可能因子本身不重要时排序噪声大)")


if __name__ == "__main__":
    main()
