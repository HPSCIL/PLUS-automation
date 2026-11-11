"""一键回归验证:改完代码跑这个,确认没改坏

三样快速检查(子进程跑,退出码为准):
    1. expansion 复现:自动 vs 手动标准答案,必须 100%
    2. 数据门禁:对干净数据(转换后的 uint8 目录)必须放行 exit 0
    3. markov 快速检查:合法年份组合必须产出需求
加 --full 追加"全流程回测"(约 5 分钟,改引擎相关代码时才需要)。

用法:
    python verify.py [--full]
"""

import os
import sys
import subprocess

PY = sys.executable  # 用当前解释器(换机可移植,不硬编码 venv 路径)
HERE = os.path.dirname(os.path.abspath(__file__))
# 数据目录:--data-dir 指定,否则回退到示例项目(武汉演示数据,换研究区务必给 --data-dir)
DATA = None
OUT = None

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def run(name, args, expect=0, timeout=600):
    """跑一条检查,按退出码判成败"""
    print(f"\n{'='*60}\n检查: {name}\n{'='*60}")
    # 子脚本强制 UTF-8 输出,这里解码也必须用 UTF-8(否则 GBK 控制台会解码崩)
    r = subprocess.run([PY] + args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    tail = ((r.stdout or "") + (r.stderr or ""))[-800:]
    print(tail.strip())
    ok = r.returncode == expect
    print(f"{'✅ 通过' if ok else '❌ 失败'}(退出码 {r.returncode},预期 {expect})")
    return ok


def prepare_clean_data():
    """组装一个'干净数据集'文件夹(转换好的 LULC + 驱动因子),供门禁回归用"""
    import shutil
    clean = os.path.join(OUT, "clean_data")
    lulc_dir = os.path.join(clean, "LULCs")
    os.makedirs(lulc_dir, exist_ok=True)
    for year in ("2003", "2005", "2010"):
        src = os.path.join(DATA, "result2026.9.15", f"wh{year}_refy_uc.tif")
        dst = os.path.join(lulc_dir, f"wh{year}_refy_uc.tif")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    factors_dst = os.path.join(clean, "dringfactor")
    if not os.path.isdir(factors_dst):
        shutil.copytree(os.path.join(DATA, "PLUS_test_data", "dringfactor"), factors_dst)
    return clean


def main():
    global DATA, OUT
    full = "--full" in sys.argv
    argv = [a for a in sys.argv[1:] if not a.startswith("--data-dir")]
    dd = None
    if "--data-dir" in sys.argv:
        dd = sys.argv[sys.argv.index("--data-dir") + 1]
    # 回退到项目配置或示例数据(示例路径仅用于开发环境自检)
    if dd:
        DATA = dd
    else:
        try:
            from common import load_project
            proj = load_project()
            DATA = proj.get("data_dir") or r"D:\app\PLUS\PLUS_V1.4.1\PLUS_test_data"
        except Exception:
            proj = {}
            DATA = r"D:\app\PLUS\PLUS_V1.4.1\PLUS_test_data"
        print(f"(未给 --data-dir,使用默认数据目录: {DATA})")
    # 各检查项的输入:项目配置给就用配置,否则用示例数据布局(仅开发自检)
    E = proj.get("early") or os.path.join(DATA, "result2026.9.15", "wh2003_refy_uc.tif")
    L = proj.get("late") or os.path.join(DATA, "result2026.9.15", "wh2005_refy_uc.tif")
    R = proj.get("real") or os.path.join(DATA, "result2026.9.9", "wh2013_refy_uc.tif")
    F = proj.get("features") or os.path.join(DATA, "PLUS_test_data", "dringfactor")
    global EARLY, LATE, REAL, FEATURES
    EARLY, LATE, REAL, FEATURES = E, L, R, F
    OUT = os.path.join(DATA, "result_verify")
    os.makedirs(OUT, exist_ok=True)
    results = []

    # 1) expansion 复现(有手动标准答案时 100% 对照;没有则只验证能产出)
    compare_arg = [os.path.join(DATA, "result2026.9.9",
                                "Land_expansion(2003_2013)_landuse_1to2.tif")]
    if not os.path.exists(compare_arg[0]):
        compare_arg = []
    results.append(run(
        "expansion 产出验证(有标准答案则 100% 对照)",
        [os.path.join(HERE, "run_expansion.py"),
         "--early", EARLY, "--late", REAL or LATE,
         "--output", f"{OUT}\\verify_expansion.tif"] + compare_arg))

    # 2) 数据门禁对数据目录放行
    clean_dir = prepare_clean_data()
    results.append(run(
        "数据门禁(数据目录应 exit 0)",
        [os.path.join(HERE, "run_setup.py"),
         "--data-dir", clean_dir]))

    # 3) markov 合法年份快速检查
    results.append(run(
        "markov 快速检查(应产出需求)",
        [os.path.join(HERE, "run_markov.py"),
         "--start-map", EARLY, "--end-map", LATE,
         "--start-year", str(proj.get("early_year", 2003)),
         "--end-year", str(proj.get("late_year", 2005)),
         "--predict-year", str(proj.get("predict_year", 2013))],
        timeout=300))

    if full:
        results.append(run(
            "全流程回测(约 5 分钟)",
            [os.path.join(HERE, "run_pipeline.py"),
             "--early", f"{DATA}\\result2026.9.15\\wh2003_refy_uc.tif",
             "--late", f"{DATA}\\result2026.9.15\\wh2005_refy_uc.tif",
             "--real", f"{DATA}\\result2026.9.9\\wh2013_refy_uc.tif",
             "--features", f"{DATA}\\PLUS_test_data\\dringfactor\\",
             "--output", f"{OUT}\\backtest",
             "--early-year", "2003", "--late-year", "2005", "--predict-year", "2013",
             "--policy", f"{DATA}\\PLUS_test_data\\water\\wh_open_water13.tif"],
            timeout=900))

    print(f"\n{'='*60}\n结果汇总: {sum(results)}/{len(results)} 通过\n{'='*60}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
