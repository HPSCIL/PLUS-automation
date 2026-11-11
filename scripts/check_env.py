"""环境自检独立入口:python check_env.py —— 检查系统/Python/依赖/引擎并给修复指引"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import force_utf8, env_check

force_utf8()
sys.exit(0 if env_check() else 1)
