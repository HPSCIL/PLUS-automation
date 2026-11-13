#!/usr/bin/env python
"""answers.json → run_full 执行器(2026-11-13 试验版,新文件)

读 ask_questions.py 校验通过的 answers.json,逐项翻译成 run_full 命令行参数并执行。
AI 全程只搬运答案,不在此处做任何"决策"。

用法:
    python run_ask.py                       # 执行 answers.json
    python run_ask.py --dry-run             # 只打印将执行的命令,不跑
"""
import json
import os
import sys
import subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ANSWERS = os.path.join(os.path.dirname(SCRIPTS), "answers.json")


def build_cmd(answers):
    """answers.json → run_full CLI 参数(逐项机械映射,无决策)"""
    cmd = [sys.executable, os.path.join(SCRIPTS, "run_full.py"),
           "--region", str(answers["region"]),
           "--years", ",".join(str(y) for y in answers["years"]),
           "--predict", str(answers["predict"]),
           "--lulc-source", str(answers["lulc_source"]),
           "--lulc-res", str(answers.get("lulc_res", 30)),
           "--demand", str(answers["demand"])]
    if answers.get("do_backtest"):
        cmd += ["--backtest", str(answers["backtest"])]
    seq = answers.get("sequence") or 0
    if seq:
        cmd += ["--sequence", str(seq)]
    if str(answers.get("water_lock") or "").startswith("lock"):
        cmd += ["--water", "lock"]
    mres = answers.get("matrix_restrictions")
    if mres:
        cmd += ["--matrix-restrictions", json.dumps(mres, ensure_ascii=False)]
    if answers.get("output_dir"):
        cmd += ["--output", str(answers["output_dir"])]
    return cmd


def main():
    if not os.path.exists(ANSWERS):
        print(f"❌ 找不到 {ANSWERS}")
        print("   请先由 AI 运行 ask_questions.py --dump 弹窗收集答案,"
              "再用 ask_questions.py --finalize 校验生成本文件")
        sys.exit(1)
    with open(ANSWERS, encoding="utf-8") as f:
        answers = json.load(f)
    cmd = build_cmd(answers)
    print("▶ 将执行:", " ".join(cmd))
    if "--dry-run" in sys.argv:
        print("(dry-run,不执行)")
        return
    r = subprocess.run(cmd)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
