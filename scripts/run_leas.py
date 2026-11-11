import os
import sys
import re
import shutil
import argparse
import glob

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import force_utf8, get_plus_backend, run_engine

force_utf8()

# 引擎目录由 common 解析:环境变量 PLUS_BACKEND > config.json > 内置默认
PLUS_BACKEND = get_plus_backend()


# 【参数】LEAS 模型参数(当前=默认值/对齐用户 9.9 手动设置;后期参数调整阶段统一改这里)
def build_leas_tmp(input_lulc, feature_folder, output_probability,
                   sampling_rate=0.01, m_try=16, n_trees=20, thread_count=1):
    """构造 LEAS 模块的操作单(PLUS 引擎规定的文本格式)

    默认参数与用户 2026-09-09 手动跑的设置完全一致
    (从 GUI 的 LEASparameters.tmp 痕迹里挖出来的):
        采样率 0.01、mTry 16、树 20、线程 1

    标签含义(官方教程 4.3 节):
        <Input LULC>           扩张图(上一模块 expansion 的输出)
        <Input Featrue folder> 驱动因子文件夹(注意标签里 Featrue 是引擎原文拼写)
        <Output probability>   输出概率图基础名(引擎会生成 <名字>_band_1..N.tif)
        <Input sampling rate>  训练采样比例(0.01 = 抽 1% 像元训练随机森林)
        <mTry>                 随机森林每棵树随机挑的特征数
        <Input the number of trees> 随机森林的树数量
        <Input thread count>   计算线程数(1 = 单线程,结果最容易复现)
    """
    return (
        "<Input LULC>\n"
        f"{input_lulc}\n"
        "<Input Featrue folder>\n"
        f"{feature_folder}\n"
        "<Output probability>\n"
        f"{output_probability}\n"
        "<Is net exit?>\n"
        "0\n"
        "<Input sampling rate>\n"
        f"{sampling_rate}\n"
        "<mTry>\n"
        f"{m_try}\n"
        "<Input the number of trees>\n"
        f"{n_trees}\n"
        "<Is balance?>\n"
        "0\n"  # 【参数】类别平衡采样:0=随机分布(手册:未勾选),1=每类采样一致;改会改变随机性来源
        "<Input thread count>\n"
        f"{thread_count}\n"
        "<High precision>\n"
        "0\n"  # 【参数】高精度模式:0=关,1=开
        "<Update Number>\n"
        "0\n"  # 【参数】更新编号(增量训练用,常规流程保持 0)
        "<Update Variable>\n"
        "0\n"  # 【参数】更新变量(同上)
    )


def run_leas(input_lulc, feature_folder, output_probability,
             plus_backend=PLUS_BACKEND, timeout=7200, **params):
    """调用引擎跑 LEAS,返回概率波段文件列表(失败返回 None)

    套路不变:写操作单 → 跑 bat → 查输出。
    输出有两类:
        1. 概率波段:<名字>_band_1..N.tif(N = 地类数)
        2. 附属文件:accuracy_record_rf.txt(随机森林精度)、
           imageminmax.txt(归一化信息)、Contribution*.csv(各驱动因子贡献度)、
           TrainData*.csv(训练数据)——引擎会写进 plus-backend 目录,
           我们顺手把它们归档到输出文件夹旁边,留作证据。
    """
    stem, ext = os.path.splitext(output_probability)

    # 步骤 0(幂等复用):波段文件已全部存在且非空时直接复用,不再重训。
    # 场景:大区域训练被超时杀掉时,引擎可能已写完波段(2026-11-04 重庆 231M 像元实测),
    # 重跑不应白白重训几十分钟。
    existing = sorted(glob.glob(glob.escape(stem) + "_band_*" + ext),
                      key=lambda p: int(re.search(r"_band_(\d+)", p).group(1)))
    if existing and all(os.path.getsize(p) > 1 << 20 for p in existing):
        # 校验波段网格与当前输入图一致:网格变化时(如数据重新对齐后)旧波段必须作废,
        # 否则 CARS 会因概率波段与 LULC 行列不符直接报 "Probability Error"
        # (2026-11-05 武汉复刻重跑实测)
        try:
            with rasterio.open(input_lulc) as _s:
                _lh, _lw = _s.height, _s.width
            with rasterio.open(existing[0]) as _b:
                _dims_ok = (_b.height == _lh and _b.width == _lw)
        except Exception:
            _dims_ok = False
        if _dims_ok:
            print(f"♻️ 检测到 {len(existing)} 个完整概率波段,直接复用(跳过重训)")
            archive_leas_outputs(output_probability, plus_backend)
            return existing
        print("⚠️ 旧概率波段与当前输入图网格不一致,作废重训")

    # 步骤 1:删掉旧的概率波段(引擎静默失败时返回码 0 不写文件,
    # 旧波段会被误当成本次结果;换类数重跑时旧 band_6/7 也会混进来)
    old_bands = glob.glob(glob.escape(stem) + "_band_*" + ext)
    for p in old_bands:
        os.remove(p)
    if old_bands:
        print(f"🧹 已清理旧概率波段 {len(old_bands)} 个")

    # 步骤 2:写操作单 + 跑 bat(公共样板,含失败检测;随机森林训练可能几分钟)
    print("  运行中(随机森林训练,请稍候)...")
    ok, _, _ = run_engine("leas.bat", "PLUS_LEAS.tmp",
                          build_leas_tmp(input_lulc, feature_folder, output_probability, **params),
                          plus_backend, timeout, ("failed", "nan(ind)"))
    if not ok:
        return None

    # 步骤 3:查输出(概率波段)。自然排序:band_1,band_2,...,band_10
    # 不能用普通 sorted():字典序会把 band_10 排在 band_2 前面,类别就错位了
    def band_num(p):
        m = re.search(r"_band_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    bands = sorted(glob.glob(glob.escape(stem) + "_band_*" + ext), key=band_num)
    if not bands:
        print(f"❌ 没有找到概率波段文件: {stem}_band_*{ext}")
        return None
    print(f"✅ 生成 {len(bands)} 个概率波段: {os.path.basename(bands[0])} ... {os.path.basename(bands[-1])}")

    # 步骤 4:归档附属文件到输出文件夹(留证据,也避免下次运行被覆盖)
    archive_leas_outputs(output_probability, plus_backend)

    return bands


def archive_leas_outputs(output_probability, plus_backend):
    """把 LEAS 附属文件(精度/归一化信息/贡献度)归档到输出文件夹"""
    out_dir = os.path.dirname(os.path.abspath(output_probability))
    for name in ("accuracy_record_rf.txt", "imageminmax.txt"):
        src = os.path.join(plus_backend, name)
        if os.path.exists(src):
            dst = os.path.join(out_dir, name)
            shutil.copy2(src, dst)
            print(f"✅ 归档: {name}")
    contribs = glob.glob(os.path.join(plus_backend, "Contribution*.csv"))
    for csv_name in contribs:
        dst = os.path.join(out_dir, os.path.basename(csv_name))
        shutil.copy2(csv_name, dst)
    if contribs:
        print(f"✅ 归档: Contribution*.csv ({len(contribs)} 个)")
    else:
        print("⚠️ 没有找到 Contribution*.csv,跳过归档")


def compare_bands(auto_bands, manual_prefix, n_classes=None):
    """逐像素对比自动和手动概率波段,返回每个波段的一致率。

    注意:LEAS 有随机森林,若引擎每次训练随机性不同,
    一致率可能达不到 100%——那时看各波段一致率是否足够高(如 >99%)。
    """
    n = n_classes or len(auto_bands)
    rates = []
    print(f"\n{'波段':>4}  {'一致像元':>14}  {'一致率':>10}")
    print("-" * 34)
    for i in range(1, n + 1):
        manual = f"{manual_prefix}_band_{i}.tif"
        if i - 1 >= len(auto_bands):
            print(f"  {i:>4}  自动波段缺失(只有 {len(auto_bands)} 个)")
            rates.append(0.0)
            continue
        auto = auto_bands[i - 1]
        if not os.path.exists(manual):
            print(f"  {i:>4}  手动文件不存在: {manual}")
            rates.append(0.0)
            continue
        with rasterio.open(auto) as s1, rasterio.open(manual) as s2:
            a, b = s1.read(1), s2.read(1)
        if a.shape != b.shape:
            print(f"  {i:>4}  ❌ 行列不一致: 自动 {a.shape} vs 手动 {b.shape},跳过")
            rates.append(0.0)
            continue
        same = (a == b).sum()
        total = a.size
        rate = same / total
        rates.append(rate)
        print(f"  {i:>4}  {same:>13,}/{total:,}  {rate:>9.4%}")
    overall = np.mean(rates)
    print("-" * 34)
    print(f"  平均一致率: {overall:.4%}")
    if overall == 1.0:
        print("✅ 完全一致!自动 = 手动,leas 模块拿下(说明引擎随机种子固定)")
    elif overall > 0.99:
        print(f"✅ 高度一致({overall:.4%})——随机森林略有随机性,但结果基本可复现")
    else:
        print(f"⚠️ 一致率偏低({overall:.4%}),需要排查参数或引擎随机性")
    return overall


def main():
    parser = argparse.ArgumentParser(description="LEAS:扩张图+驱动因子 → 发展概率图")
    parser.add_argument("--input", required=True, help="扩张图(expansion 的输出)")
    parser.add_argument("--features", required=True, help="驱动因子文件夹路径")
    parser.add_argument("--output", required=True, help="输出概率图基础名(引擎生成 <名字>_band_1..N.tif)")
    parser.add_argument("--sampling", type=float, default=0.01, help="采样率,默认 0.01(与手动一致)")
    parser.add_argument("--mtry", type=int, default=16, help="mTry,默认 16(与手动一致)")
    parser.add_argument("--trees", type=int, default=20, help="树数量,默认 20(与手动一致)")
    parser.add_argument("--threads", type=int, default=1, help="线程数,默认 1(与手动一致)")
    parser.add_argument("--plus-backend", default=PLUS_BACKEND)
    parser.add_argument("--compare-prefix", default=None,
                        help="手动标准答案的前缀,如 ...\\DevPotential,给了就逐波段对比")
    parser.add_argument("--n-classes", type=int, default=7, help="地类数,默认 7")
    args = parser.parse_args()

    # 路径全部转绝对路径(引擎工作目录是 plus-backend,相对路径会写错地方)
    args.input = os.path.abspath(args.input)
    args.features = os.path.abspath(args.features)
    args.output = os.path.abspath(args.output)
    if args.compare_prefix:
        args.compare_prefix = os.path.abspath(args.compare_prefix)

    print(f"扩张图: {args.input}")
    print(f"驱动因子: {args.features}")
    bands = run_leas(args.input, args.features, args.output, args.plus_backend,
                     sampling_rate=args.sampling, m_try=args.mtry,
                     n_trees=args.trees, thread_count=args.threads)
    if bands is None:
        sys.exit(1)  # 失败给非零退出码
    if args.compare_prefix:
        print(f"\n和手动标准答案逐像素对比: {args.compare_prefix}_band_1~{args.n_classes}.tif")
        compare_bands(bands, args.compare_prefix, args.n_classes)


if __name__ == "__main__":
    main()
