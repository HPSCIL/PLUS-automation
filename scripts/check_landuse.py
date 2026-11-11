import os
import subprocess
import argparse

import numpy as np
import rasterio

# PLUS 约定的"无效值"，检查地类编码时必须排除：
#   0    = 背景值
#   255  = NoData（空值），PLUS 转换工具输出的统一空值
#   -128 = int8 数据类型的 NoData（带符号 8 位整数的最小值）
# 【参数】无效值约定(PLUS 协议,一般不调;改这里会同时影响统计/出图脚本的约定)
NODATA_VALUES = (0, 255, -128)

# plus-backend 的位置：PLUS 无界面引擎目录
# 里面有 cpp\PLUS.exe 和各模块的 .bat 文件
# 注意：这个路径必须全英文（PLUS 对路径编码敏感，中文路径会崩）
# 【参数】引擎路径(阶段2已配置化:统一走 common 的 env > config.json > 相对运行时)
from common import force_utf8, get_plus_backend

# 控制台编码防护:中文 Windows 默认 GBK,打印 ✅ 等符号会崩(2026-11-10 实测复现)
force_utf8()
PLUS_BACKEND = get_plus_backend()


def check_path(tif_path):
    """检查文件路径是否合法，返回 True/False"""
    print(f"正在检查路径: {tif_path}")

    # 检查非英文字符：PLUS 要求纯英文（ASCII）路径
    # ord(ch) > 127 表示这个字符不是标准英文字母、数字或符号
    bad_chars = {ch for ch in tif_path if ord(ch) > 127}
    if bad_chars:
        print(f"❌ 警告：路径中包含非英文字符 {bad_chars}，PLUS 可能崩溃！")
    else:
        print("✅ 路径无中文等特殊字符。")

    # 检查空格
    if " " in tif_path:
        print("❌ 警告：路径中包含空格，PLUS 可能崩溃！")
    else:
        print("✅ 路径无空格。")

    # 检查文件是否存在
    if not os.path.exists(tif_path):
        print("❌ 文件不存在，请检查路径！")
        return False
    else:
        print("✅ 文件存在。")
        return True


def check_tif_format(tif_path):
    """读取 tif 文件，检查格式是否符合 PLUS 要求。

    返回一个字典（info），包含检查结果，方便后面的代码决定要不要转换。
    """
    print(f"\n开始读取 tif 文件...")
    info = {"ok": False, "dtype": None, "classes": []}

    try:
        with rasterio.open(tif_path) as src:
            # with 语句：打开文件，用完自动关闭（不用手动 close）

            print(f"✅ 成功打开 tif 文件")
            print(f"  行列数: {src.height} 行 × {src.width} 列")
            print(f"  波段数: {src.count}")
            print(f"  分辨率: {src.res}")
            print(f"  坐标系: {src.crs}")
            print(f"  数据类型: {src.dtypes[0]}")
            info["dtype"] = src.dtypes[0]

            # 检查数据类型
            if src.dtypes[0] == 'uint8':
                print("✅ 是 unsigned char 格式（PLUS 要求）")
            else:
                print(f"⚠️ 不是 unsigned char，当前是 {src.dtypes[0]}，需要转换")

            # 读取数据，检查地类编码
            print(f"\n  正在读取像素值...")
            data = src.read(1)  # 读第一个波段（波段序号从 1 开始）

            unique_vals = np.unique(data)
            # 关键：先排除 0、255、-128 这些"无效值"，再检查编码
            # 之前把 255 当成普通地类，才会误报"缺少 8~254"
            classes = unique_vals[~np.isin(unique_vals, NODATA_VALUES)]
            info["classes"] = [int(v) for v in classes]

            print(f"  地类唯一值（已排除无效值）: {info['classes']}")
            if len(classes) == 0:
                print("❌ 没有找到有效地类，文件可能全是空值")
                return info

            min_val = int(classes.min())
            max_val = int(classes.max())

            # 检查 1：编码从 1 开始
            if min_val == 1:
                print("✅ 地类编码从 1 开始")
            else:
                print(f"⚠️ 地类编码不是从 1 开始（当前最小={min_val}）")

            # 检查 2：编码连续（1,2,3... 中间不能缺）
            expected = set(range(min_val, max_val + 1))
            actual = set(info["classes"])
            if expected == actual:
                print(f"✅ 编码从 {min_val} 到 {max_val} 连续，共 {len(classes)} 类")
            else:
                missing = expected - actual
                print(f"⚠️ 编码不连续，缺少值: {sorted(missing)}")

            info["ok"] = True

    except Exception as e:
        print(f"❌ 读取失败: {e}")

    return info


def _convert_tif_rasterio(input_path, output_path):
    """PLUS convert 的等价快速实现(rasterio),返回 (ok, 原因)

    引擎产物规格(与引擎逐像元比对 0 差异,WB 2026-11-05 济南实测):
    uint8、无压缩、nodata=255、原 0 背景改 255。
    边界策略:多波段/非 uint8/含负值 → 不适用,由调用方回退引擎版。
    """
    import numpy as np
    with rasterio.open(input_path) as src:
        if src.count != 1:
            return False, f"多波段({src.count})"
        if src.dtypes[0] != "uint8":
            return False, f"dtype {src.dtypes[0]}"
        arr = src.read(1)
        if int(arr.min()) < 0:
            return False, "含负值"
        out = np.where(arr == 0, 255, arr).astype("uint8")
        prof = src.profile.copy()
        for k in ("compress", "blockxsize", "blockysize", "tiled", "interleave"):
            prof.pop(k, None)
        prof.update(dtype="uint8", nodata=255, count=1)
        with rasterio.open(output_path, "w", **prof) as dst:
            dst.write(out, 1)
    return True, ""


def convert_tif(input_path, output_path, plus_backend=PLUS_BACKEND, timeout=600):  # 【参数】timeout=600 转换超时秒数,超大栅格可调大
    """把 tif 转成 PLUS 兼容格式（unsigned char）。

    默认走 rasterio 快速通道(0→255,uint8,nodata=255,无压缩):
    - 与引擎产物逐像元一致(WB 济南实测 0 差异)
    - 引擎版对中等栅格会卡死(WB 实测 6000×7000 挂起 0% CPU)
    - 环境变量 PLUS_CONVERT_ENGINE=plus 强制走引擎版(交叉验证用)
    - 快速通道不适用(多波段/非 uint8/负值)时自动回退引擎版

    引擎版原理（PLUS 官方引擎只认这一种无人值守用法）：
        1. 把输入/输出路径写进文本文件 PLUS_Convert.tmp
        2. 运行 convert.bat，它内部执行 PLUS.exe 读取 tmp 并干活
        3. 检查输出文件是否真的生成了
    返回 True/False。
    """
    print(f"\n开始转换: {input_path}")

    # 前置检查：plus-backend 必须存在（里面有 PLUS.exe 和 convert.bat）
    # 快速通道调度:默认 rasterio,PLUS_CONVERT_ENGINE=plus 强制引擎
    if os.environ.get("PLUS_CONVERT_ENGINE", "").lower() != "plus":
        try:
            ok, why = _convert_tif_rasterio(input_path, output_path)
            if ok:
                print(f"✅ 转换完成(快速通道): {output_path}")
                return True
            print(f"⚠️ 快速通道不适用({why}),回退引擎版")
        except Exception as e:
            print(f"⚠️ 快速通道异常({e}),回退引擎版")

    bat_path = os.path.join(plus_backend, "convert.bat")
    if not os.path.exists(bat_path):
        print(f"❌ 找不到 {bat_path}，请检查 plus-backend 路径")
        return False

    # 步骤 1：写 tmp 文件。格式是 PLUS 固定的：
    #   <Input number>       输入文件个数
    #   <Input LULC series>  输入文件路径列表
    #   <Output LULC series> 输出文件路径列表
    tmp_content = (
        "<Input number>\n"
        "1\n"
        "<Input LULC series>\n"
        f"{input_path}\n"
        "<Output LULC series>\n"
        f"{output_path}\n"
    )
    tmp_path = os.path.join(plus_backend, "PLUS_Convert.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(tmp_content)
    print(f"✅ 已写入 {tmp_path}")

    # 步骤 2：运行 convert.bat（工作目录必须是 plus-backend）
    # 注意：PLUS.exe 结束时可能触发 "按任意键继续"(pause)，不屏蔽 stdin 会永久挂起；
    # 超时必须杀整棵进程树，否则孙进程占着管道，communicate() 永远等不到 EOF
    # (WorkBuddy 2026-11-05 实测死锁;其余模块走 common.run_engine 统一入口,此处独漏)
    proc = subprocess.Popen(
        [bat_path],
        cwd=plus_backend,            # 在 plus-backend 目录里执行
        stdout=subprocess.PIPE,      # 把程序的输出抓回来，不刷屏
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,    # 关键：屏蔽 stdin，pause 立即返回
        text=True,
        encoding="utf-8", errors="replace",
    )
    try:
        out, err = proc.communicate(timeout=timeout)  # 超时保护，防止卡死
    except subprocess.TimeoutExpired:
        print(f"❌ 转换超时（超过 {timeout} 秒），杀整棵进程树")
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, timeout=60)
        try:
            proc.communicate(timeout=60)
        except Exception:
            pass
        return False

    if proc.returncode != 0:
        # returncode 是程序退出码：0 = 成功，非 0 = 出错
        print(f"❌ 转换失败，返回码 {proc.returncode}")
        if out:
            print(out)
        if err:
            print(err)
        return False

    # 步骤 3：检查输出文件是否真的生成了
    if os.path.exists(output_path):
        print(f"✅ 转换完成: {output_path}")
        return True
    else:
        print(f"❌ bat 跑完了但输出文件不存在: {output_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="检查土地利用 tif 是否符合 PLUS 要求，检查通过后自动调用 PLUS convert 转换"
    )
    parser.add_argument("tif_path", help="要检查的 tif 文件路径")
    parser.add_argument("--plus-backend", default=PLUS_BACKEND,
                        help="plus-backend 引擎目录（默认用脚本内置的路径）")
    parser.add_argument("--output", default="",
                        help="转换输出路径；不填则默认放在输入同目录，文件名加 _uc 后缀")
    args = parser.parse_args()

    # 第一步：检查路径
    if not check_path(args.tif_path):
        return  # return 直接结束 main，程序退出

    # 第二步：检查格式
    info = check_tif_format(args.tif_path)
    if not info["ok"]:
        print("\n❌ 无法读取文件，停止。")
        return

    # 第三步：转换。默认输出路径 = 输入同目录 + _uc 后缀
    if not args.output:
        stem, ext = os.path.splitext(args.tif_path)
        args.output = stem + "_uc" + ext
    if not convert_tif(args.tif_path, args.output, args.plus_backend):
        return

    # 第四步：转换后再检查一次输出，确认达标
    print("\n--- 转换后复查 ---")
    check_tif_format(args.output)

    print("\n--- 全部完成 ---")


if __name__ == "__main__":
    main()
