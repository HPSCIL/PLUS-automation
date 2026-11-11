"""开发版 → 安装版同步:改完代码跑这个,不再手工 cp(防漂移)

同步到两个平台:
    - Claude Code:用户目录下的 .claude\\skills
    - WorkBuddy:  用户目录下的 .workbuddy\\skills
覆盖:plus-model 的模块脚本 + common.py(每个 skill 一份),以及三个老 skill 的脚本。

用法:
    python sync.py [--check] [--platform claude|workbuddy|both]
    --platform 默认 both;开发期新功能未确认前用 --platform claude(先不动 WorkBuddy)
"""

import os
import sys
import shutil

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEV_MODEL = r"D:\plus_skill\plus-model\scripts"
# 两个平台的技能根目录(顺序无关)
PLATFORMS = [
    r"C:\Users\26271\.claude\skills",
    r"C:\Users\26271\.workbuddy\skills",
]

# (开发版文件, 技能名)
MODULE_PAIRS = [
    ("run_expansion.py", "plus-expansion"),
    ("run_leas.py", "plus-leas"),
    ("run_cars.py", "plus-cars"),
    ("run_markov.py", "plus-markov"),
    ("run_neighborhood_weight.py", "plus-neighborhood-weight"),
    ("run_validation.py", "plus-validation"),
    ("run_linear.py", "plus-linear"),
]
# 主 skill(plus-model)本身:全脚本 + SKILL.md + 项目配置
# 注意:run_pipeline.py 是同目录 import 这 7 个模块的(sys.path.insert + from run_*),
#       plus-model 自己的 scripts 里必须各有一份 —— 漏同步会让安装版跑旧逻辑。
#       config.json 不同步(两平台引擎路径不同:dev/Claude 与 WorkBuddy 各一份)。
MODEL_FILES = [
    "common.py", "run_pipeline.py", "run_setup.py", "verify.py",
    "compare_contributions.py",
    "project.json", "project_wuhan2035.json", "run_wuhan_2030.py",
    "run_expansion.py", "run_leas.py", "run_cars.py", "run_markov.py",
    "run_neighborhood_weight.py", "run_validation.py", "run_linear.py",
    "sync.py",
]
OLD_SKILL_PAIRS = [
    (r"D:\plus_skill\plus_data_check\scripts\check_landuse.py", "plus-data-check"),
    (r"D:\plus_skill\plus_data_check\scripts\reclassify_lulc.py", "plus-data-check"),
    (r"D:\plus_skill\plus-landuse-stats\scripts\landuse_stats.py", "plus-landuse-stats"),
    (r"D:\plus_skill\plus-map-figure\scripts\landuse_map.py", "plus-map-figure"),
]
# 整目录 skill:整个 scripts 文件夹 + SKILL.md 一起同步
FULL_SKILLS = [
    (r"D:\plus_skill\plus-data-download", "plus-data-download"),
    (r"D:\plus_skill\plus-driver-download", "plus-driver-download"),
    (r"D:\plus_skill\plus-automation", "plus-automation"),
]


def same(src, dst):
    try:
        with open(src, "rb") as f1, open(dst, "rb") as f2:
            return f1.read() == f2.read()
    except FileNotFoundError:
        return False


def collect_pairs(platform="both"):
    """列出所有 (源, 目标) 同步对(platform: claude / workbuddy / both)"""
    pairs = []
    for inst in PLATFORMS:
        if platform == "claude" and "workbuddy" in inst:
            continue
        if platform == "workbuddy" and ".claude" in inst:
            continue
        for fname, skill in MODULE_PAIRS:
            pairs.append((os.path.join(DEV_MODEL, fname),
                          os.path.join(inst, skill, "scripts", fname)))
        for fname in MODEL_FILES:
            pairs.append((os.path.join(DEV_MODEL, fname),
                          os.path.join(inst, "plus-model", "scripts", fname)))
        for fname in ("SKILL.md",):
            pairs.append((os.path.join(DEV_MODEL, "..", fname),
                          os.path.join(inst, "plus-model", fname)))
        for skill in [s for _, s in MODULE_PAIRS]:
            pairs.append((os.path.join(DEV_MODEL, "common.py"),
                          os.path.join(inst, skill, "scripts", "common.py")))
        for src, skill in OLD_SKILL_PAIRS:
            pairs.append((src, os.path.join(inst, skill, "scripts", os.path.basename(src))))
        # 老 skill 的 SKILL.md(dev 版在 D:\plus_skill 下,安装版在技能根目录)
        pairs.append((r"D:\plus_skill\plus_data_check\SKILL.md",
                      os.path.join(inst, "plus-data-check", "SKILL.md")))
        for dev_dir, skill in FULL_SKILLS:
            for f in os.listdir(os.path.join(dev_dir, "scripts")):
                if f.endswith(".py"):
                    pairs.append((os.path.join(dev_dir, "scripts", f),
                                  os.path.join(inst, skill, "scripts", f)))
            # 依赖清单一并同步(换机安装必带)
            req = os.path.join(dev_dir, "requirements.txt")
            if os.path.exists(req):
                pairs.append((req, os.path.join(inst, skill, "requirements.txt")))
            # config.json 一并同步(引擎路径覆盖配置,换机必带)
            cfg_src = os.path.join(dev_dir, "scripts", "config.json")
            if os.path.exists(cfg_src):
                pairs.append((cfg_src, os.path.join(inst, skill, "scripts", "config.json")))
            pairs.append((os.path.join(dev_dir, "SKILL.md"),
                          os.path.join(inst, skill, "SKILL.md")))
            # 根目录级文档与 docs/(说明书 PDF + 速查)一并同步
            # ——故障排查协议引用这些文件,漏同步会让安装版查不到说明书(2026-11-10)
            # 注意:各 skill 根目录文档并不齐全(如 plus-data-download 没有 FAQ.md),
            # 不存在的直接跳过,否则同步中断(2026-11-10 实测踩坑)
            for rootfile in ("FAQ.md", "安装说明.md", "数据来源与版权说明.md", ".gitignore"):
                src_root = os.path.join(dev_dir, rootfile)
                if os.path.exists(src_root):
                    pairs.append((src_root, os.path.join(inst, skill, rootfile)))
            docs_dir = os.path.join(dev_dir, "docs")
            if os.path.isdir(docs_dir):
                for f in os.listdir(docs_dir):
                    pairs.append((os.path.join(docs_dir, f),
                                  os.path.join(inst, skill, "docs", f)))
    return pairs


def main():
    check_only = "--check" in sys.argv
    platform = "both"
    if "--platform" in sys.argv:
        platform = sys.argv[sys.argv.index("--platform") + 1]
    changed = [(s, d) for s, d in collect_pairs(platform) if not same(s, d)]

    if not changed:
        print("✅ 开发版与两个平台的安装版完全一致,无需同步")
        return
    for src, dst in changed:
        print(f"{'🔍 差异' if check_only else '📦 同步'}: {os.path.basename(src)} → {dst}")
    if check_only:
        print(f"\n共 {len(changed)} 处差异(用 python sync.py 同步)")
        return
    for src, dst in changed:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    print(f"\n✅ 已同步 {len(changed)} 个文件(平台:{platform})")


if __name__ == "__main__":
    main()
