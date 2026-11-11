"""plus-model 公共模块:引擎路径解析、配置、引擎执行样板、bat 生成器

阶段 2"可移植改造"的核心:所有脚本从这里拿引擎路径和共用样板,
不再每个文件各写一份(此前 12 处硬编码 + 5 份拷贝已漂移,审查发现)。
"""

import os
import sys
import json
import subprocess

# 内置默认引擎目录:skill 自带的运行时(相对路径,换机可移植;setup.py 会把引擎复制到这里)
# 阶段 2"可移植改造":不再硬编码 D:\plus_skill,引擎装到 <skill>/plus-runtime/plus-backend
DEFAULT_PLUS_BACKEND = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "plus-runtime", "plus-backend")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# 【参数】无效值约定(PLUS 协议:0=背景 255=NoData -128=int8空值),全局唯一一份
NODATA_VALUES = (0, 255, -128)


def force_utf8():
    """强制 UTF-8 输出(部分终端 GBK 会崩),stdout 和 stderr 都管"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---- 项目配置(通用化的核心:用户的研究区决策落盘,脚本读配置取默认值)----

def load_project(project_path=None):
    """读项目配置 JSON。找不到返回空 dict。

    项目配置 = 用户研究区的所有决策:数据目录、各期图、驱动因子、
    policy、年份、转移矩阵、地类名称。写一次,以后所有命令复用,
    任何研究区(武汉/深圳/...)一套机制。
    """
    path = project_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "project.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_project(cfg, project_path=None):
    """写项目配置 JSON(幂等,重复写覆盖)"""
    path = project_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "project.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return path


def get_plus_backend():
    """引擎目录解析顺序:环境变量 PLUS_BACKEND > config.json > skill 内置 runtime > 内置默认。

    换设备时:优先用 skill 自带的 runtime\\plus-backend(私有打包时内置;
    GitHub 公开仓库不含此目录,见安装说明)——也可以改 config.json 或设环境变量。
    """
    env = os.environ.get("PLUS_BACKEND")
    if env:
        return env
    cfg = load_config()
    if cfg.get("plus_backend"):
        return cfg["plus_backend"]
    # skill 内置引擎(可选:私有交付包含 runtime 目录;公开仓库没有时跳过)
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime",
                         "plus-backend")
    if (os.path.exists(os.path.join(local, "PLUS.exe")) or
            os.path.exists(os.path.join(local, "cpp", "PLUS.exe"))):
        return os.path.abspath(local)
    return DEFAULT_PLUS_BACKEND


def env_check():
    """⓪ 环境自检:系统 / Python 版本 / 依赖 / 引擎,缺什么给什么修复指令。

    供 run_full 第一步调用(也可单独 python check_env.py)。
    返回 True=就绪;False=有问题(已打印修复指引)。
    """
    import platform
    ok = True
    print("=" * 60 + "\n【⓪ 环境自检】\n" + "=" * 60)
    if platform.system() != "Windows":
        print("❌ 仅支持 Windows(PLUS 引擎为 Windows 程序)")
        ok = False
    else:
        print("✅ 系统: Windows")
    v = sys.version_info
    # 3.13 支持(2026-11-10 实测):rasterio 1.4.4/numpy 2.4.6/scipy 1.17.1 均有 cp313 轮子,
    # 真实栅格读写(6386×5778 CLCD)+ matplotlib 出图全通过;引擎为外部 exe,与解释器版本无关。
    if (3, 10) <= v[:2] <= (3, 13):
        print(f"✅ Python {v.major}.{v.minor}.{v.micro}")
    else:
        print(f"❌ Python {v.major}.{v.minor}(需 3.10~3.13),下载: https://www.python.org/downloads/")
        ok = False
    import importlib.util
    missing = [m for m in ("rasterio", "numpy", "scipy", "shapely", "requests", "osmium")
               if importlib.util.find_spec(m) is None]
    if missing:
        print(f"❌ 缺少依赖: {missing}")
        print(f"   修复: {sys.executable} -m pip install -r requirements.txt")
        ok = False
    else:
        print("✅ 依赖: rasterio/numpy/scipy/shapely/requests/osmium 齐全")
    try:
        be = get_plus_backend()
        # 引擎可执行文件位置:plus-backend/cpp/PLUS.exe(官方目录结构,bat 在根目录)
        exe = os.path.join(be, "PLUS.exe")
        if not os.path.exists(exe):
            exe = os.path.join(be, "cpp", "PLUS.exe")
        if os.path.exists(exe):
            print(f"✅ 引擎: {be}")
        else:
            print(f"❌ 引擎缺失({be} 下无 PLUS.exe)")
            print("   修复: 私有包已内置 runtime\\plus-backend;从 GitHub 安装的见 安装说明.md 第 2 步")
            ok = False
    except Exception as e:
        print(f"❌ 引擎解析失败: {e}")
        ok = False
    print("✅ 环境全部就绪" if ok else "⚠️ 请按上述提示修复后重跑")
    return ok


def check_ascii_path(path):
    """PLUS 对路径编码敏感:非 ASCII 路径要提前拦住,而不是等引擎深处崩溃"""
    bad = {ch for ch in str(path) if ord(ch) > 127}
    if bad:
        raise ValueError(f"路径含非英文字符 {bad},PLUS 可能崩溃: {path}")


def _pid_alive(pid):
    """Windows 下检查进程是否存活(单实例锁用)"""
    import ctypes
    try:
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
    except Exception:
        pass
    return False


def _engine_lock_acquire(plus_backend):
    """引擎单实例锁(WB 审查 P1-2):所有模块共用固定 tmp 文件与输出路径,
    并发运行会互相覆盖参数单。锁文件存 PID,进程存活则拒绝并发。
    返回 (True, None) 或 (False, 提示)"""
    lock = os.path.join(plus_backend, ".engine.lock")
    if os.path.exists(lock):
        try:
            with open(lock, encoding="utf-8") as f:
                pid = int(f.read().strip())
            if _pid_alive(pid):
                return False, f"引擎正被 PID {pid} 占用(拒绝并发,防参数单互踩)"
            os.remove(lock)  # 死锁残留
        except Exception:
            try:
                os.remove(lock)
            except Exception:
                pass
    with open(lock, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    return True, None


def _engine_lock_release(plus_backend):
    lock = os.path.join(plus_backend, ".engine.lock")
    try:
        if os.path.exists(lock):
            os.remove(lock)
    except Exception:
        pass


def run_engine(bat_name, tmp_name, tmp_content, plus_backend,
               timeout=600, fail_markers=("failed", "nan(ind)")):
    """引擎执行样板(所有模块共用的唯一实现):写操作单 → 跑 bat → 检测失败。

    返回 (ok, stdout, stderr)。
    注意"静默失败"陷阱:PLUS.exe 内部出错时返回码仍是 0、只打印错误不写文件,
    所以除了返回码,还要扫输出里的专属失败痕迹。
    不能用裸 "Error" 当标记——CARS 的正常输出里就有 "Error: 10516" 这类迭代指标。
    """
    bat_path = os.path.join(plus_backend, bat_name)
    if not os.path.exists(bat_path):
        print(f"❌ 找不到 {bat_path}")
        print(f"  引擎目录配置为: {plus_backend}")
        print(f"  请运行 python setup.py --from-backend <引擎目录> 配置,或检查环境变量 PLUS_BACKEND")
        return False, "", ""

    tmp_path = os.path.join(plus_backend, tmp_name)
    # 单实例锁:并发运行时拒绝(WB 审查 P1-2)
    ok_lock, busy = _engine_lock_acquire(plus_backend)
    if not ok_lock:
        print(f"❌ {busy}")
        return False, "", busy
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(tmp_content)
    print(f"✅ 已写入操作单: {tmp_path}")

    try:
        result = subprocess.run(
            [bat_path], cwd=plus_backend, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL,  # bat 里有 pause("按任意键继续"),没这行在部分环境会挂起(WorkBuddy 实测)
            timeout=timeout,
        )
        # 引擎控制台输出留档(如 CARS 的逐 epoch 错误数/停止原因,排查必备;
        # 2026-11-04 重庆 CARS 反常调查就靠手动抓这份日志破案)
        try:
            log_dir = os.path.join(plus_backend, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, os.path.splitext(bat_name)[0] + ".log")
            with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
                lf.write(result.stdout or "")
                if result.stderr:
                    lf.write("\n--- stderr ---\n" + result.stderr)
        except Exception:
            pass  # 日志写失败不影响主流程
    except subprocess.TimeoutExpired:
        print(f"❌ 超时({timeout}s)")
        _engine_lock_release(plus_backend)
        return False, "", f"Timeout after {timeout}s"
    finally:
        _engine_lock_release(plus_backend)

    stdout, stderr = result.stdout or "", result.stderr or ""
    if result.returncode != 0 or any(m in stdout + stderr for m in fail_markers):
        print(f"❌ 引擎运行失败(返回码 {result.returncode})")
        if stdout:
            print(stdout[-1500:])
        if stderr:
            print(stderr[-1500:])
        return False, stdout, stderr
    return True, stdout, stderr


# 各模块的 bat 与 tmp 对应关系(官方协议:一行启动器 + 固定 tmp 名)
# bat 内容是这个模板,可以由我们生成,不是谁的私有代码。
# 注意:markov 模块的官方启动器多了 -o 指定输出路径,其余模块无额外参数。
BAT_TEMPLATE = '@echo off\ncd /d "%~dp0"\ncpp\\PLUS.exe -t {module} -i {tmp}{extra}\n'
BAT_SPECS = {
    "convert": ("convert.bat", "PLUS_Convert.tmp", ""),
    "expansion": ("expansion.bat", "PLUS_Expansion.tmp", ""),
    "leas": ("leas.bat", "PLUS_LEAS.tmp", ""),
    "markov": ("markov.bat", "PLUS_Markov.tmp", " -o cpp/output/markov.csv"),
    "cars": ("cars.bat", "PLUS_CARS.tmp", ""),
    "linear": ("linear.bat", "PLUS_Linear.tmp", ""),
    "diverse": ("diverse.bat", "PLUS_Diverse.tmp", ""),
    "validate": ("validate.bat", "PLUS_Validation.tmp", ""),
}


def generate_bats(backend_dir):
    """在引擎目录生成全部 bat 启动器(内容由官方调用协议决定,我们自产)"""
    made = []
    for module, (bat_name, tmp_name, extra) in BAT_SPECS.items():
        p = os.path.join(backend_dir, bat_name)
        with open(p, "w", encoding="ascii") as f:
            f.write(BAT_TEMPLATE.format(module=module, tmp=tmp_name, extra=extra))
        made.append(bat_name)
    return made
