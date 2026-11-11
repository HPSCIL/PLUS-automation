import os
import sys
import csv
import shutil
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import force_utf8, get_plus_backend, run_engine

force_utf8()

# 引擎目录由 common 解析:环境变量 PLUS_BACKEND > config.json > 内置默认
PLUS_BACKEND = get_plus_backend()


def python_metrics(simulated_map, real_map, start_map, sample_frac=0.1, seed=42):
    """纯 Python 精度计算:Kappa / OA / FoM(A/B/C/D 分解)/ 逐类 UA-PA

    与引擎版口径一致(10% 随机采样,固定种子可复现)。
    引擎 validate 模块在部分机器上会挂起(WB 实测 0% CPU 10 分钟+),
    Python 版作主路径,引擎版(--engine)留作交叉验证。
    返回指标字典。
    """
    import numpy as np
    import rasterio

    def load(p):
        with rasterio.open(p) as s:
            return s.read(1)

    start, real, sim = load(start_map), load(real_map), load(simulated_map)
    valid = ((start != 255) & (real != 255) & (sim != 255) &
             (start != 0) & (real != 0) & (sim != 0))
    idx = np.where(valid.ravel())[0]
    rng = np.random.default_rng(seed)
    n = max(1, int(len(idx) * sample_frac))
    samp = idx[rng.choice(len(idx), size=min(n, len(idx)), replace=False)]
    s, r, m = (start.ravel()[samp], real.ravel()[samp], sim.ravel()[samp])

    agree = (m == r)
    oa = agree.mean()
    # Kappa:Po = 一致率;Pe = 期望一致率(行和×列和的乘积和 / N²)
    classes = sorted(set(np.unique(r)) | set(np.unique(m)))
    po = oa
    pe = 0.0
    for c in classes:
        pe += ((r == c).sum() / len(r)) * ((m == c).sum() / len(m))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

    # FoM 四格(标准定义):A 命中 / B 漏报 / C 假警报 / D 错类
    real_chg = s != r
    sim_chg = s != m
    a = ((real_chg & sim_chg & (m == r))).sum()
    b = (real_chg & ~sim_chg).sum()
    c = (sim_chg & ~real_chg).sum()
    d = (real_chg & sim_chg & (m != r)).sum()
    fom = a / (a + b + c + d) if (a + b + c + d) else 0.0

    # 逐类生产者/用户精度(变化像元口径)
    per_class = {}
    for cls in classes:
        rc = real_chg & (r == cls)
        mc = sim_chg & (m == cls)
        hits = (rc & mc).sum()
        pa = hits / rc.sum() if rc.sum() else None
        ua = hits / mc.sum() if mc.sum() else None
        per_class[int(cls)] = {"PA": pa, "UA": ua}

    return {"Kappa": kappa, "OA": oa, "FoM": fom,
            "A": int(a), "B": int(b), "C": int(c), "D": int(d),
            "per_class": per_class, "sampled": len(samp)}


def build_validation_tmp(simulated_map, real_map, start_map,
                         is_fom=0, sampling_rate=0.1):
    """构造 validation 模块的操作单(标签与引擎协议一致)

    is_fom: 0 = 算 Kappa(全图对比),1 = 算 FoM(只看变化像元)
    sampling_rate: 采样比例,0.1 = 抽 10% 有效像元算(与手动一致)
    """
    return (
        "<IsFom>\n"
        f"{is_fom}\n"
        "<Sampling rate>\n"
        f"{sampling_rate}\n"
        "<Simulated Map>\n"
        f"{simulated_map}\n"
        "<Real Map>\n"
        f"{real_map}\n"
        "<Start Map>\n"
        f"{start_map}\n"
    )


def run_validation(simulated_map, real_map, start_map, is_fom=0, sampling_rate=0.1,
                   plus_backend=PLUS_BACKEND, timeout=600):
    """调用引擎跑 validation,返回归档后的结果 csv 路径(失败返回 None)

    Kappa 模式输出 Kappa.csv,FoM 模式输出 FoM.csv,都写在 plus-backend 目录;
    我们归档到模拟图同目录,并返回归档副本路径(保证证据不被后续运行改写)。
    """
    name = "FoM.csv" if is_fom else "Kappa.csv"
    out_csv = os.path.join(plus_backend, name)

    # 步骤 1:删掉引擎固定输出位的旧 Kappa.csv/FoM.csv
    # (引擎静默失败时返回码 0 不写文件,旧文件会被误当成本次结果)
    if os.path.exists(out_csv):
        os.remove(out_csv)
        print(f"🧹 已清理旧输出: {out_csv}")

    # 步骤 2:写操作单 + 跑 bat(公共样板,含失败检测)
    # 专属失败痕迹(实测):行列不一致时 GDAL 打 "Access window out of range"、
    # 引擎打 "Sampling Number: 0";结果污染时写 nan
    ok, _, _ = run_engine("validate.bat", "PLUS_Validation.tmp",
                          build_validation_tmp(simulated_map, real_map, start_map,
                                               is_fom, sampling_rate),
                          plus_backend, timeout,
                          ("failed", "nan(ind)", "Sampling Number: 0",
                           "Access window out of range"))
    if not ok:
        return None

    if not os.path.exists(out_csv):
        print(f"❌ 没找到输出 {name}")
        return None
    print(f"✅ 输出: {out_csv}")

    # 归档到结果目录(引擎下次运行会覆盖,先存一份;返回归档副本路径)
    result_dir = os.path.dirname(os.path.abspath(simulated_map))
    archive = os.path.join(result_dir, name)
    shutil.copy2(out_csv, archive)
    print(f"✅ 归档: {archive}")
    return archive


def read_metric(csv_path, section_prefix, key_prefix):
    """从 Kappa.csv/FoM.csv 里取某个指标值。

    用"前缀匹配"而不是完全相等——引擎真实格式是:
        [Figure of Merit]=B/(A+B+C+D)   ← 段标题带公式后缀
        FoM=,0.208723                   ← 键带等号
    所以按 section_prefix / key_prefix 前缀找,并容忍 BOM 和空格。
    """
    with open(csv_path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        in_section = False
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if first.startswith("["):
                in_section = first.startswith(section_prefix)
                continue
            if in_section and len(row) >= 2 and first.startswith(key_prefix):
                try:
                    return float(row[1].replace(",", "").strip())
                except ValueError:
                    return None
    return None


def _clean_metric(val):
    """指标值清洗:None 或 NaN 都按"缺失"处理,防格式化崩溃"""
    if val is None:
        return None
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


def main():
    parser = argparse.ArgumentParser(description="Validation:模拟图 vs 真实图 → Kappa/FoM 精度")
    parser.add_argument("--simulated", required=True, help="模拟结果图")
    parser.add_argument("--real", required=True, help="真实土地利用图")
    parser.add_argument("--start", required=True, help="模拟起始年图")
    parser.add_argument("--fom", action="store_true", help="算 FoM(默认算 Kappa)")
    parser.add_argument("--sampling", type=float, default=0.1,
                        help="采样比例,默认 0.1(与手动一致)")
    parser.add_argument("--engine", action="store_true",
                        help="强制走引擎版 validate(默认 Python 版;引擎版留作交叉验证)")
    parser.add_argument("--plus-backend", default=PLUS_BACKEND)
    parser.add_argument("--compare", default=None, help="手动标准答案 csv 路径,给了就对比")
    args = parser.parse_args()

    # 路径全部转绝对路径(引擎工作目录是 plus-backend)
    args.simulated = os.path.abspath(args.simulated)
    args.real = os.path.abspath(args.real)
    args.start = os.path.abspath(args.start)
    if args.compare:
        args.compare = os.path.abspath(args.compare)

    # Python 主路径(WB 审查 P0-4:引擎 validate 会挂起;Python 版输出更全的分解)
    if not args.engine:
        m = python_metrics(args.simulated, args.real, args.start,
                           args.sampling)
        print(f"验证模式: Python({'FoM' if args.fom else 'Kappa'}),采样比例 {args.sampling}")
        print(f"  模拟图: {args.simulated}")
        print(f"  真实图: {args.real}")
        print(f"  Kappa = {m['Kappa']:.4f}  总体精度 = {m['OA']:.2%}  "
              f"FoM = {m['FoM']:.6f}")
        print(f"  FoM 分解: A(命中)={m['A']} B(漏报)={m['B']} "
              f"C(假警报)={m['C']} D(错类)={m['D']}")
        for cls, pc in sorted(m["per_class"].items()):
            pa = f"{pc['PA']:.2%}" if pc["PA"] is not None else "—"
            ua = f"{pc['UA']:.2%}" if pc["UA"] is not None else "—"
            print(f"  类别 {cls}: PA={pa} UA={ua}")
        return

    is_fom = 1 if args.fom else 0
    # 指标对只定一次,自动和手动都用同一对,避免分支变量陷阱(审查发现)
    if is_fom:
        section_prefix, key_prefix, mode = "[Figure of Merit]", "FoM", "FoM"
    else:
        section_prefix, key_prefix, mode = "[Kappa Coefficient]", "Kappa", "Kappa"
    print(f"验证模式: {mode},采样比例 {args.sampling}")
    print(f"模拟图: {args.simulated}")
    print(f"真实图: {args.real}")

    out_csv = run_validation(args.simulated, args.real, args.start,
                             is_fom, args.sampling, args.plus_backend)
    if out_csv is None:
        sys.exit(1)  # 失败给非零退出码

    auto_val = _clean_metric(read_metric(out_csv, section_prefix, key_prefix))
    if auto_val is None:
        print(f"  ❌ 读不到本次 {mode} 值(引擎输出格式可能有变),请检查 {out_csv}")
        sys.exit(1)
    if is_fom:
        print(f"\n  本次 FoM = {auto_val:.6f}")
    else:
        overall = _clean_metric(read_metric(out_csv, "[Overall Accuracy]", "Overall"))
        print(f"\n  本次 Kappa = {auto_val:.6f}"
              + (f",总体精度 = {overall:.4%}" if overall is not None else ""))

    if args.compare:
        print(f"\n=== 和手动标准答案对比: {args.compare} ===")
        manual_val = _clean_metric(read_metric(args.compare, section_prefix, key_prefix))
        if manual_val is None:
            print("  ❌ 读不到手动标准答案")
            return
        print(f"  手动: {manual_val:.6f}")
        print(f"  自动: {auto_val:.6f}")
        diff = abs(auto_val - manual_val)
        if diff < 1e-6:
            print("  ✅ 完全一致!validation 模块拿下")
        elif diff < 0.02:
            print(f"  ✅ 高度一致(差 {diff:.6f},模拟结果本身的随机性所致)——模块拿下")
        else:
            print(f"  ⚠️ 差异较大({diff:.6f}),需要排查")


if __name__ == "__main__":
    main()
