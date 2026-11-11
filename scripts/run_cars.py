import os
import sys
import re
import shutil
import argparse
import glob

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import force_utf8, get_plus_backend, run_engine, NODATA_VALUES

force_utf8()

# 引擎目录由 common 解析:环境变量 PLUS_BACKEND > config.json > 内置默认
PLUS_BACKEND = get_plus_backend()

# 【参数】CARS 模拟参数默认值:全部对齐用户 9.9 手动设置(从 GUI 的 CARSparameters.tmp 痕迹挖出)
DEFAULT_NEIGHBORHOOD = 3
DEFAULT_PATCH_GENERATION = 0.2
DEFAULT_EXPANSION_COEFF = 0.1
DEFAULT_SEED_PERCENTAGE = 0.1
DEFAULT_THREADS = 8  # 【参数】CARS 线程数:官方引擎支持多线程;大区域(如重庆 2.3 亿像元)单线程慢到不可用,实测 8 线程快数倍
DEFAULT_YEARS = 1


def build_cars_tmp(input_lulc, probability_paths, output_simulation,
                   policy_path, neighborhood_weights, transition_matrix,
                   yearly_demands, input_classes=None, neighborhood=None,
                   thread_count=None, patch_generation=None,
                   expansion_coefficient=None, seed_percentage=None,
                   years=None):
    """构造 CARS 模块的操作单(标签与官方教程/GUI 痕迹文件完全一致)

    输入来源说明:
        probability_paths  ← LEAS 模块输出的概率波段(个数 = 地类数)
        neighborhood_weights ← neighborhood_weight 模块算的权重
        yearly_demands    ← markov 模块预测的各地类需求
        transition_matrix ← 转移矩阵(研究决策输入)
    参数默认值全部取自本文件的 DEFAULT_* 常量(单一调参入口)。
    """
    # 默认值统一走常量,避免 CLI 和管线各用一套(审查发现的隐患)
    if input_classes is None:
        input_classes = len(probability_paths)  # 地类数 = 概率波段数,不用手填
    if neighborhood is None:
        neighborhood = DEFAULT_NEIGHBORHOOD
    if thread_count is None:
        thread_count = DEFAULT_THREADS
    if patch_generation is None:
        patch_generation = DEFAULT_PATCH_GENERATION
    if expansion_coefficient is None:
        expansion_coefficient = DEFAULT_EXPANSION_COEFF
    if seed_percentage is None:
        seed_percentage = DEFAULT_SEED_PERCENTAGE
    if years is None:
        years = DEFAULT_YEARS
    matrix = transition_matrix.replace(";", "\n")  # 行之间可用分号或换行
    return (
        "<Input classes>\n"
        f"{input_classes}\n"
        "<Input LULC>\n"
        f"{input_lulc}\n"
        "<Input Probability Folder>\n"
        + "\n".join(probability_paths) + "\n"
        "<Output simulation>\n"
        f"{output_simulation}\n"
        "<Input Policy>\n"
        f"{policy_path}\n"
        "<Input Neighborhood>\n"
        f"{neighborhood}\n"
        "<How many years>\n"
        f"{years}\n"
        "<Input thread count>\n"
        f"{thread_count}\n"
        "<Patch generation>\n"
        f"{patch_generation}\n"
        "<Expansion coefficient>\n"
        f"{expansion_coefficient}\n"
        "<Neighborhood Weight>\n"
        f"{neighborhood_weights}\n"
        "<Transition matrix>\n"
        f"{matrix}\n"
        "<Years and corresponding demands>\n"
        f"{yearly_demands}\n"
        "<Percentage of seeds>\n"
        f"{seed_percentage}\n"
        "<Development type exist>\n"
        "0\n"
        "<Development type>\n"
        "0\n"
        "<Development weight>\n"
        "0.5\n"
    )


def run_cars(input_lulc, probability_paths, output_simulation,
             policy_path, neighborhood_weights, transition_matrix,
             yearly_demands, plus_backend=PLUS_BACKEND, timeout=14400,
             **params):
    """调用引擎跑 CARS,返回模拟结果文件列表(失败返回 None)

    套路不变:写操作单 → 跑 bat → 查输出。
    输出惯例:<名字>Simulation_1.tif 这类文件名。
    CARS 含随机种子(Percentage of seeds),结果有随机性,验证用统计对比。
    """
    stem, ext = os.path.splitext(output_simulation)

    # 步骤 1:删掉旧的模拟结果(引擎静默失败时返回码 0 不写文件,
    # 旧 Simulation_*.tif 会被误当成本次结果;换参数重跑时旧文件也会混进来)
    old_sims = glob.glob(glob.escape(stem) + "Simulation_*" + ext)
    for p in old_sims:
        os.remove(p)
    if old_sims:
        print(f"🧹 已清理旧模拟结果 {len(old_sims)} 个")

    # 步骤 2:写操作单 + 跑 bat(公共样板,含失败检测)
    # 注意:CARS 正常输出里就有 "Error: 10516, Derror:0" 这类迭代指标,
    # 失败标记必须用它的专属失败字样(实测:失败时打 "Land Use Error")
    print("  运行中(CARS 斑块模拟,可能跑几分钟,请稍候)...")
    ok, _, _ = run_engine("cars.bat", "PLUS_CARS.tmp",
                          build_cars_tmp(input_lulc, probability_paths,
                                         output_simulation, policy_path,
                                         neighborhood_weights, transition_matrix,
                                         yearly_demands, **params),
                          plus_backend, timeout, ("failed", "Land Use Error", "nan(ind)"))
    if not ok:
        return None

    # 自然排序(Simulation_1 < Simulation_2 < ... < Simulation_10,不用字典序)
    def sim_num(p):
        m = re.search(r"Simulation_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    outs = sorted(glob.glob(glob.escape(stem) + "Simulation_*" + ext), key=sim_num)
    if not outs:
        print(f"❌ 没有找到模拟结果: {stem}Simulation_*{ext}")
        return None
    for o in outs:
        print(f"✅ 模拟结果: {o}")

    # 归档引擎生成的转移矩阵等附属文件
    out_dir = os.path.dirname(os.path.abspath(output_simulation))
    tms = glob.glob(os.path.join(plus_backend, "TransitionMatrix*.csv"))
    for csv_name in tms:
        shutil.copy2(csv_name, os.path.join(out_dir, os.path.basename(csv_name)))
    if tms:
        print(f"✅ 归档: TransitionMatrix*.csv ({len(tms)} 个)")
    else:
        print("⚠️ 没有找到 TransitionMatrix*.csv,跳过归档")
    return outs


def compare_simulation(auto_path, manual_path):
    """对比自动模拟和手动模拟:总体一致率 + 各地类一致率"""
    with rasterio.open(auto_path) as s1, rasterio.open(manual_path) as s2:
        a, b = s1.read(1), s2.read(1)
    if a.shape != b.shape:
        print(f"❌ 行列不一致: {a.shape} vs {b.shape}")
        return
    total = a.size
    same = (a == b).sum()
    print(f"  总体一致率: {same:,}/{total:,} = {same/total:.4%}")
    print(f"  各地类一致率(自动→手动混淆概览):")
    for v in sorted(set(np.unique(a)) | set(np.unique(b))):
        if v in NODATA_VALUES:  # 0=背景 255=NoData -128=int8空值,不算地类
            continue
        # a 来自自动图(s1=auto_path),b 来自手动图(s2=manual_path)——标签不能写反
        auto_mask, manual_mask = (a == v), (b == v)
        both = (auto_mask & manual_mask).sum()
        print(f"    地类 {int(v):>3}: 自动像元 {auto_mask.sum():>9,}, 手动像元 {manual_mask.sum():>9,}, 一致 {both:>9,}")


def main():
    parser = argparse.ArgumentParser(
        description="CARS:概率+需求+权重+转移矩阵 → 未来土地利用模拟图")
    parser.add_argument("--lulc", required=True, help="起始年份土地利用图")
    parser.add_argument("--probability", nargs="+", required=True, help="概率波段路径(按地类顺序)")
    parser.add_argument("--output", required=True, help="输出基础名(引擎生成 <名字>Simulation_1.tif)")
    parser.add_argument("--policy", default="", help="policy 约束图(可空)")
    parser.add_argument("--weights", required=True, help="邻域权重,逗号分隔小数")
    parser.add_argument("--matrix", required=True, help="转移矩阵,行间用分号分隔")
    parser.add_argument("--demands", required=True, help="需求,格式 年数,需求1,需求2,...")
    parser.add_argument("--classes", type=int, default=None, help="地类数,默认=概率波段数")
    parser.add_argument("--neighborhood", type=int, default=DEFAULT_NEIGHBORHOOD,
                        help=f"邻域大小,默认 {DEFAULT_NEIGHBORHOOD}(与手动一致)")
    parser.add_argument("--patch-generation", type=float, default=DEFAULT_PATCH_GENERATION,
                        help=f"斑块生成阈值,默认 {DEFAULT_PATCH_GENERATION}(与手动一致)")
    parser.add_argument("--expansion-coefficient", type=float, default=DEFAULT_EXPANSION_COEFF,
                        help=f"扩张系数,默认 {DEFAULT_EXPANSION_COEFF}(与手动一致)")
    parser.add_argument("--seed-percentage", type=float, default=DEFAULT_SEED_PERCENTAGE,
                        help=f"随机种子比例,默认 {DEFAULT_SEED_PERCENTAGE}(与手动一致)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help=f"线程数,默认 {DEFAULT_THREADS}(与手动一致)")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS,
                        help=f"模拟年数,默认 {DEFAULT_YEARS}(与手动一致)")
    parser.add_argument("--plus-backend", default=PLUS_BACKEND)
    parser.add_argument("--compare", default=None, help="手动标准答案路径,给了就对比")
    args = parser.parse_args()

    # 路径全部转绝对路径(引擎工作目录是 plus-backend,相对路径会写错地方)
    args.lulc = os.path.abspath(args.lulc)
    args.probability = [os.path.abspath(p) for p in args.probability]
    args.output = os.path.abspath(args.output)
    if args.policy:
        args.policy = os.path.abspath(args.policy)
    if args.compare:
        args.compare = os.path.abspath(args.compare)

    print(f"起始图: {args.lulc}")
    print(f"概率波段: {len(args.probability)} 个")
    outs = run_cars(args.lulc, args.probability, args.output, args.policy,
                    args.weights, args.matrix, args.demands, args.plus_backend,
                    input_classes=args.classes, neighborhood=args.neighborhood,
                    thread_count=args.threads,
                    patch_generation=args.patch_generation,
                    expansion_coefficient=args.expansion_coefficient,
                    seed_percentage=args.seed_percentage, years=args.years)
    if outs is None:
        sys.exit(1)  # 失败给非零退出码
    if args.compare:
        print(f"\n和手动标准答案对比: {args.compare}")
        compare_simulation(outs[0], args.compare)


if __name__ == "__main__":
    main()
