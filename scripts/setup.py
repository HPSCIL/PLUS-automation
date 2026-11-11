"""PLUS skill 安装器(阶段 2 可移植改造):一步建好自己的运行时

干三件事:
    1. 检查 Python 依赖(rasterio/numpy)
    2. 从任意一个"引擎来源"复制引擎文件(PLUS.exe + dll)到本项目的 plus-runtime
    3. 生成 9 个 bat 启动器 + 写 config.json(所有脚本自动从这里读引擎路径)

用法:
    python setup.py --from-backend <现有 plus-backend 目录>     ← 从已有引擎目录复制
    python setup.py --check                                      ← 只检查当前配置/依赖

引擎来源说明:PLUS.exe 是官方软件自带的计算引擎(官方 GUI 安装包内含),
本安装器只负责"复制 + 生成官方协议的一行 bat + 写配置",不含任何私有代码。
"""

import os
import sys
import shutil
import argparse

# 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_config, save_config, generate_bats,
                    DEFAULT_PLUS_BACKEND, check_ascii_path)

# 引擎运行时需要的最小文件集:cpp 目录(PLUS.exe + dll)整体复制
CPP_DIR = "cpp"
REQUIRED_ENGINE_FILES = ["PLUS.exe"]


def check_deps():
    """检查 Python 依赖"""
    missing = []
    for mod in ("rasterio", "numpy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"❌ 缺少依赖: {', '.join(missing)}")
        print(f"   安装命令: pip install {' '.join(missing)}")
        return False
    print("✅ Python 依赖齐全(rasterio/numpy)")
    return True


def build_runtime(backend_source, target=None):
    """从来源目录复制引擎 + 生成 bat + 写配置"""
    target = target or DEFAULT_PLUS_BACKEND
    src_cpp = os.path.join(backend_source, CPP_DIR)
    if not os.path.exists(src_cpp):
        print(f"❌ 来源目录里没有 {CPP_DIR}\\ 文件夹: {src_cpp}")
        print(f"  这个目录应该是 plus-backend 结构(含 cpp\\PLUS.exe 和各模块 bat)")
        return False
    exe = os.path.join(src_cpp, "PLUS.exe")
    if not os.path.exists(exe):
        print(f"❌ 来源目录里没有 PLUS.exe: {exe}")
        return False

    check_ascii_path(target)  # PLUS 路径必须纯英文

    # 1) 复制引擎(cpp 目录整体,含 dll)
    dst_cpp = os.path.join(target, CPP_DIR)
    if os.path.exists(dst_cpp):
        shutil.rmtree(dst_cpp)
    shutil.copytree(src_cpp, dst_cpp)
    print(f"✅ 引擎文件已复制: {dst_cpp}")

    # 2) 生成 9 个 bat(官方协议的一行启动器)
    made = generate_bats(target)
    print(f"✅ 已生成 {len(made)} 个 bat: {', '.join(made)}")

    # 3) 写配置(所有脚本自动从这里读引擎路径)
    save_config({"plus_backend": target})
    print(f"✅ 配置已写入: config.json → {target}")
    return True


def main():
    parser = argparse.ArgumentParser(description="PLUS skill 安装器:建自己的运行时")
    parser.add_argument("--from-backend", default=None,
                        help="来源引擎目录(plus-backend 结构,含 cpp\\PLUS.exe)")
    parser.add_argument("--target", default=None, help="目标目录,默认 D:\\plus_skill\\plus-runtime\\plus-backend")
    parser.add_argument("--check", action="store_true", help="只检查依赖和当前配置")
    args = parser.parse_args()

    print("=== PLUS skill 安装器 ===\n")
    if args.check or args.from_backend is None:
        ok = check_deps()
        cfg = load_config()
        print(f"当前配置引擎目录: {cfg.get('plus_backend', '(未配置,用内置默认 ' + DEFAULT_PLUS_BACKEND + ')')}")
        if args.check:
            return

    if args.from_backend:
        src = os.path.abspath(args.from_backend)
        target = os.path.abspath(args.target) if args.target else DEFAULT_PLUS_BACKEND
        print(f"来源: {src}")
        print(f"目标: {target}\n")
        if not check_deps():
            sys.exit(1)
        if not build_runtime(src, target):
            sys.exit(1)
        print("\n=== 安装完成 === 现在可以直接跑 run_pipeline.py / 各模块脚本了")


if __name__ == "__main__":
    main()
