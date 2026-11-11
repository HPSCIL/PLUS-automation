"""plus-automation 冒烟测试:最小案例走通全流程,断言产物齐全 + 指标落在合理区间

目的:改动任何下载/裁切/重分类/建模代码后,5 分钟内确认"整条链没坏",
而不是等到跑完一个市域(30+ 分钟)才发现产物缺失。

用法:
    python smoke_test.py                                    # 用默认小案例(鄂州 3 期)
    python smoke_test.py --data-dir <已有数据目录> --years 2010,2015,2020 --predict 2020
    python smoke_test.py --region 鄂州市                     # 从下载开始(需网络,较慢)

判定(全部满足才 PASS):
    产物 7 类齐全 + Kappa ∈ (0.30, 1.0] + FoM ∈ (0, 0.80) + 报告含"产物清单"
耗时上限默认 600 s(可用 --max-minutes 改)
"""
import os
import sys
import json
import time
import argparse
import subprocess

PY = sys.executable
SCRIPTS = os.path.dirname(os.path.abspath(__file__))

# 控制台编码防护:中文 Windows 默认 GBK,打印 ✅ 等符号会崩(2026-11-10 实测复现)
sys.path.insert(0, SCRIPTS)
from common import force_utf8

force_utf8()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="鄂州市")
    ap.add_argument("--years", default="2010,2015,2020")
    ap.add_argument("--predict", type=int, default=2020)
    ap.add_argument("--data-dir", default=None,
                    help="已有数据目录(推荐:跳过下载,只测建模链)")
    ap.add_argument("--max-minutes", type=float, default=10.0)
    a = ap.parse_args()

    years = [int(y) for y in a.years.split(",") if y.strip()]
    cmd = [PY, os.path.join(SCRIPTS, "run_full.py"),
           "--region", a.region, "--years", a.years, "--predict", str(a.predict)]
    if a.data_dir:
        cmd += ["--data-dir", os.path.abspath(a.data_dir), "--resume"]

    print(f"▶ 冒烟测试:{a.region} {years} → {a.predict}")
    print(f"  命令:{' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = time.time() - t0
    log = (r.stdout or "") + (r.stderr or "")
    print(log[-1500:])

    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"  {'✅' if ok else '❌'} {name}{(' — ' + detail) if detail else ''}")

    check("进程退出码 0", r.returncode == 0, f"exit={r.returncode}")
    if r.returncode != 0:
        print("\n❌ 冒烟测试失败(流程中断)")
        sys.exit(1)

    data_dir = os.path.abspath(a.data_dir) if a.data_dir else None
    if not data_dir:
        # 从日志里抓 "数据目录: xxx"
        for line in log.splitlines():
            if line.strip().startswith("数据目录:"):
                data_dir = line.split(":", 1)[1].strip()
    check("定位到数据目录", bool(data_dir) and os.path.isdir(data_dir or ""), data_dir or "-")

    out_name = f"model_bt{a.predict}" if a.predict == years[-1] else f"model_{a.predict}"
    model_dir = os.path.join(data_dir, out_name)

    # 1 裁切图  2 因子  3 概率波段  4 模拟图  5 报告  6 图  7 report.json
    lulc = [f for f in os.listdir(data_dir) if f.startswith("lulc_") and f.endswith(".tif")
            and "_rc" not in f]
    check("① 裁切图", len(lulc) >= len(years), f"{len(lulc)} 个")
    drv = os.path.join(data_dir, "drivers")
    ndrv = len([f for f in os.listdir(drv) if f.startswith("df_")]) if os.path.isdir(drv) else 0
    check("② 驱动因子", ndrv >= 10, f"{ndrv} 个 df_*.tif")
    bands = [f for f in os.listdir(model_dir) if f.startswith("DevPotential_bt_band_")]
    check("③ 概率波段", len(bands) >= 2, f"{len(bands)} 个")
    sim = [f for f in os.listdir(model_dir)
           if f.startswith("simulation_bt_") and "Simulation_1" in f]
    check("④ 模拟图", bool(sim), sim[0] if sim else "缺失")
    # 报告改造(2026-11-10)后:建模过程报告归档到 logs/子报告/,model 目录只留汇总报告.md
    _proc_report = os.path.join(data_dir, "logs", "子报告", "建模过程报告.md")
    check("⑤ 报告(汇总报告.md + 归档建模过程报告)",
          os.path.exists(os.path.join(model_dir, "汇总报告.md"))
          and os.path.exists(_proc_report))
    jpg = [f for f in os.listdir(model_dir) if f.endswith(".jpg")]
    check("⑥ 成图 JPG", len(jpg) >= 1, f"{len(jpg)} 张")
    rep_path = os.path.join(model_dir, "report.json")
    rep = {}
    if os.path.exists(rep_path):
        with open(rep_path, encoding="utf-8") as f:
            rep = json.load(f)
    check("⑦ report.json", bool(rep))

    # 指标区间(冒烟只查"有没有算出来、量级对不对",不锁具体值)
    m = rep.get("metrics") or {}
    kappa, fom = m.get("Kappa"), m.get("FoM")
    check("Kappa 落在 (0.30, 1.0]",
          isinstance(kappa, (int, float)) and 0.30 < float(kappa) <= 1.0,
          f"{kappa if kappa is None else round(float(kappa), 4)}")
    check("FoM 落在 (0, 0.80)",
          isinstance(fom, (int, float)) and 0 < float(fom) < 0.80,
          f"{fom if fom is None else round(float(fom), 4)}")
    summary = os.path.join(model_dir, "汇总报告.md")
    if os.path.exists(summary):
        txt = open(summary, encoding="utf-8").read()
        check("汇总报告含分步耗时/产物清单", "分步耗时" in txt and "产物清单" in txt)

    # 目录干净:.part/.parts.json 不应残留(引擎中间图由 ⑨ 清理)
    junk = [f for f in os.listdir(data_dir) if f.endswith((".part", ".parts.json"))]
    check("无 .part 残留", not junk, ", ".join(junk[:3]))

    check(f"耗时 < {a.max_minutes:g} min", dt < a.max_minutes * 60, f"{dt/60:.1f} min")

    failed = [c for c in checks if not c[1]]
    print(f"\n{'=' * 50}")
    if failed:
        print(f"❌ 冒烟测试未通过:{len(failed)}/{len(checks)} 项失败")
        for name, _, detail in failed:
            print(f"   - {name} {detail}")
        sys.exit(1)
    print(f"✅ 冒烟测试通过({len(checks)} 项全过,{dt/60:.1f} min)")
    print(f"   Kappa={float(kappa):.4f} FoM={float(fom):.4f} 产物目录={model_dir}")


if __name__ == "__main__":
    main()
