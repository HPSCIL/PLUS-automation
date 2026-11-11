import os
import sys
import csv
import argparse
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import force_utf8, get_plus_backend, run_engine

force_utf8()

# 引擎目录由 common 解析:环境变量 PLUS_BACKEND > config.json > 内置默认
PLUS_BACKEND = get_plus_backend()


def build_markov_tmp(start_map, end_map, start_year, end_year, predict_year):
    """构造 markov 模块的操作单(标签与引擎协议一致)

    含义:用起止两期图之间的转移规律(马尔可夫链),预测 predict_year 年的
    各地类需求量。markov 是纯确定性计算,没有随机性。
    """
    return (
        "<StartMap>\n"
        f"{start_map}\n"
        "<EndMap>\n"
        f"{end_map}\n"
        "<Start Year>\n"
        f"{start_year}\n"
        "<End Year>\n"
        f"{end_year}\n"
        "<Predict Year>\n"
        f"{predict_year}\n"
    )


def valid_predict_years(early_year, late_year, predict_year):
    """整数周期规则(实测发现的引擎规则):预测年 = 终止年 + N×(终止年−起始年)"""
    step = late_year - early_year
    if step <= 0:
        return False
    return (predict_year - late_year) % step == 0


def run_markov_demands(start_map, end_map, start_year, end_year, predict_year,
                       plus_backend=PLUS_BACKEND, timeout=600, interpolate=False):
    """预测某年各地类需求量;目标年不在步长上时用"前后可达年份插值"

    返回 {predict_year: [d1..dN]}(各像元数,半进位取整)。

    插值方法(用户研究决策,2026-11-01):
        目标年 = 终止年 + N×步长 + 余数时,取前后两个可达年份(下界/上界)的
        马尔可夫需求量,把两者差值按年数等分,目标年 = 下界 + 余数份。
        例:2010→2015(步长 5)预测 2031:下界 2030(3 周期)、上界 2035(4 周期),
        demand(2031) = demand(2030) + 1/5×(demand(2035) − demand(2030))。
    """
    if valid_predict_years(start_year, end_year, predict_year) or not interpolate:
        # 常规路径:目标年恰好在步长上
        csv_path = run_markov(start_map, end_map, start_year, end_year,
                              predict_year, plus_backend, timeout)
        if csv_path is None:
            return None
        demands_map, _ = parse_markov_csv(csv_path)
        if predict_year not in demands_map:
            return None
        return {predict_year: demands_map[predict_year]}

    # 插值路径:目标年不在步长上,先算前后两个可达年份
    step = end_year - start_year
    n = (predict_year - end_year) // step          # 向下取整的周期数
    lower_year = end_year + n * step               # 下界可达年
    upper_year = lower_year + step                 # 上界可达年
    frac = (predict_year - lower_year) / step      # 目标年在下界后的份数(0~1)

    print(f"\n🔢 插值需求:{predict_year} 不在步长上,取 {lower_year}/{upper_year} 分段插值")
    lower_csv = run_markov(start_map, end_map, start_year, end_year,
                           lower_year, plus_backend, timeout)
    if lower_csv is None:
        return None
    lower_d, _ = parse_markov_csv(lower_csv)
    upper_csv = run_markov(start_map, end_map, start_year, end_year,
                           upper_year, plus_backend, timeout)
    if upper_csv is None:
        return None
    upper_d, _ = parse_markov_csv(upper_csv)
    if lower_year not in lower_d or upper_year not in upper_d:
        print(f"❌ 插值失败:输出里没有 {lower_year} 或 {upper_year}")
        return None

    lo, hi = lower_d[lower_year], upper_d[upper_year]
    # 每类独立插值,半进位取整(和引擎 csv 解析同款舍入约定)
    interp = [int(lo[i] + frac * (hi[i] - lo[i]) + 0.5) for i in range(len(lo))]
    print(f"  {lower_year} 年需求: {lo}")
    print(f"  {upper_year} 年需求: {hi}")
    print(f"  插值系数: {frac:.4f} → {predict_year} 年需求: {interp}")
    return {predict_year: interp}


def run_markov(start_map, end_map, start_year, end_year, predict_year,
               plus_backend=PLUS_BACKEND, timeout=600):
    """调用引擎跑 markov,返回输出 markov.csv 的路径(失败返回 None)

    套路不变:写操作单 → 跑 bat → 查输出。
    输出位置:引擎通常写到 plus-backend/cpp/output/markov.csv。
    """
    # 步骤 1:删掉引擎固定输出位的旧 markov.csv
    # (引擎静默失败时不写文件,旧文件会被误当成本次结果)
    fixed_out = os.path.join(plus_backend, "cpp", "output", "markov.csv")
    for p in (fixed_out, os.path.join(plus_backend, "markov.csv")):
        if os.path.exists(p):
            os.remove(p)
            print(f"🧹 已清理旧输出: {p}")

    # 步骤 2:写操作单 + 跑 bat(公共样板,含失败检测)
    ok, _, _ = run_engine("markov.bat", "PLUS_Markov.tmp",
                          build_markov_tmp(start_map, end_map, start_year, end_year, predict_year),
                          plus_backend, timeout, ("failed", "nan(ind)"))
    if not ok:
        return None

    # 找输出文件(旧文件已删,存在即本次产物;位置可能因版本而异,都查一遍)
    candidates = [
        fixed_out,
        os.path.join(plus_backend, "markov.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"✅ markov 输出: {c}")
            return c
    found = glob.glob(os.path.join(plus_backend, "cpp", "output", "markov*.csv"))
    if found:
        print(f"✅ markov 输出: {found[0]}")
        return found[0]
    print("❌ 没找到 markov 输出文件")
    return None


def parse_markov_csv(csv_path):
    """解析引擎的 markov.csv,返回 {predict_year: [d1..d7]} 和转换矩阵总数行"""
    demands = {}
    matrix_total = None
    with open(csv_path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        section = None
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if first.startswith("["):
                section = first
                continue
            if section == "[Conversion Matrix]" and first == "total":
                matrix_total = [int(float(v) + 0.5) for v in row[1:] if v.strip()]
            if section == "[Predict amount]" and first.isdigit():
                # 引擎输出带小数(如 14527.597564),按"四舍五入(半进位)"取整成像元数
                # 注意:Python 的 round() 是银行家舍入(x.5 向偶),会和 GUI 差 1 像元,这里用 +0.5 取整
                demands[int(first)] = [int(float(v) + 0.5) for v in row[1:] if v.strip()]
    return demands, matrix_total


def format_cars_demands(predict_year, values):
    """转成 CARS 操作单里 <Years and corresponding demands> 的格式"""
    return f"1,{','.join(str(v) for v in values)}"


def compare_markov(auto_csv, manual_csv, predict_year, end_year=None):
    """和手动标准答案(MakovChain.csv)对比:转换矩阵 + 预测需求量"""
    auto_d, auto_t = parse_markov_csv(auto_csv)
    manual_d, manual_t = parse_markov_csv(manual_csv)

    year_label = f"实际 {end_year} 年" if end_year else "终止年"
    print(f"\n=== 对比 1:转换矩阵 total 行({year_label}各地类面积)===")
    if manual_t and auto_t:
        same = auto_t == manual_t
        print(f"  手动: {manual_t}")
        print(f"  自动: {auto_t}")
        print("  ✅ 完全一致" if same else "  ⚠️ 不一致")

    print(f"\n=== 对比 2:{predict_year} 年预测需求量 ===")
    if predict_year in manual_d and predict_year in auto_d:
        m, a = manual_d[predict_year], auto_d[predict_year]
        same = m == a
        print(f"  手动: {m}")
        print(f"  自动: {a}")
        print("  ✅ 完全一致!markov 模块拿下" if same else "  ⚠️ 不一致,需要排查")
        print(f"\n  转成 CARS 需求格式: {format_cars_demands(predict_year, a)}")
        return same
    else:
        print(f"  ❌ 预测年 {predict_year} 不在输出里(手动:{list(manual_d)},自动:{list(auto_d)})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Markov:两期图+年份 → 未来需求量(可插值)")
    parser.add_argument("--start-map", required=True, help="起始年土地利用图")
    parser.add_argument("--end-map", required=True, help="结束年土地利用图")
    parser.add_argument("--start-year", type=int, required=True, help="起始年份")
    parser.add_argument("--end-year", type=int, required=True, help="结束年份")
    parser.add_argument("--predict-year", type=int, required=True, help="预测目标年份")
    parser.add_argument("--interpolate", action="store_true",
                        help="目标年不在步长上时,用前后可达年份的需求分段插值(如 2031 取 2030/2035 插值)")
    parser.add_argument("--plus-backend", default=PLUS_BACKEND)
    parser.add_argument("--compare", default=None, help="手动标准答案 MakovChain.csv 路径")
    args = parser.parse_args()

    # 整数周期规则前置检查:目标年在步长上直接跑;不在步长上 → 有 --interpolate 走插值,否则报错
    on_step = valid_predict_years(args.start_year, args.end_year, args.predict_year)
    if not on_step and not args.interpolate:
        step = args.end_year - args.start_year
        print(f"❌ 预测年 {args.predict_year} 不符合引擎规则")
        print(f"  规则:预测年 = 终止年 + N×(终止年−起始年)(N=0,1,2,...)")
        if step > 0:
            valid = [args.end_year + n * step for n in range(6)]
            print(f"  当前起止 {args.start_year}→{args.end_year}(步长 {step})可预测:{valid}")
        print(f"  或加 --interpolate 用前后可达年份分段插值")
        sys.exit(1)

    # 路径全部转绝对路径(引擎工作目录是 plus-backend)
    args.start_map = os.path.abspath(args.start_map)
    args.end_map = os.path.abspath(args.end_map)
    if args.compare:
        args.compare = os.path.abspath(args.compare)

    print(f"起止: {args.start_year} → {args.end_year},预测: {args.predict_year}"
          + ("(分段插值)" if not on_step else ""))
    if not on_step:
        # 插值路径:两次引擎运行 + 按年数等分
        result = run_markov_demands(args.start_map, args.end_map, args.start_year,
                                    args.end_year, args.predict_year,
                                    args.plus_backend, interpolate=True)
        if result is None:
            sys.exit(1)
        print(f"\n  转成 CARS 需求格式: {format_cars_demands(args.predict_year, result[args.predict_year])}")
        return

    auto_csv = run_markov(args.start_map, args.end_map, args.start_year,
                          args.end_year, args.predict_year, args.plus_backend)
    if auto_csv is None:
        sys.exit(1)
    if args.compare:
        print(f"\n和手动标准答案对比: {args.compare}")
        compare_markov(auto_csv, args.compare, args.predict_year, args.end_year)


if __name__ == "__main__":
    main()
