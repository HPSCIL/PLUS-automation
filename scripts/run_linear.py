import os
import sys
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import force_utf8, get_plus_backend, run_engine

force_utf8()

# 引擎目录由 common 解析:环境变量 PLUS_BACKEND > config.json > 内置默认
PLUS_BACKEND = get_plus_backend()


def build_linear_tmp(predict_amount, image_paths):
    """构造 linear 模块的操作单(线性回归需求预测)

    <Image Amount>     历史期土地利用图个数
    <Predict Amount>   要预测的未来年数(从最后一期起往后数)
    <Images Path>      按时间顺序排列的历史图路径
    与 markov 不同:linear 没有"整数周期"限制,可以预测任意年数!
    """
    return (
        "<Image Amount>\n"
        f"{len(image_paths)}\n"
        "<Predict Amount>\n"
        f"{predict_amount}\n"
        "<Images Path>\n"
        + "\n".join(image_paths) + "\n"
    )


FORMULA_RE = re.compile(r"result\(Type (\d+)\):\s*y = ([\-\d.]+)\.x \+ ([\-\d.]+)")


def parse_formulas(stdout):
    """从引擎标准输出解析每类的回归公式,返回 {type: (斜率, 截距)}"""
    formulas = {}
    for line in stdout.splitlines():
        m = FORMULA_RE.search(line)
        if m:
            formulas[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return formulas


def predict(formulas, year):
    """用公式预测某年的各地类像元数(四舍五入成整数)"""
    return {t: int(round(s * year + b)) for t, (s, b) in formulas.items()}


def run_linear(image_paths, predict_amount, plus_backend=PLUS_BACKEND, timeout=600):
    """调用引擎跑 linear,返回解析出的公式 {type: (斜率, 截距)}(失败返回 None)

    输出在引擎的标准输出里(这个模块不落结果文件)。
    """
    ok, stdout, stderr = run_engine("linear.bat", "PLUS_Linear.tmp",
                                    build_linear_tmp(predict_amount, image_paths),
                                    plus_backend, timeout, ("failed", "nan(ind)"))
    if not ok:
        return None
    formulas = parse_formulas(stdout)
    if not formulas:
        print("❌ 标准输出里没有解析到回归公式")
        print(stdout[-1500:])
        return None
    return formulas


def main():
    parser = argparse.ArgumentParser(
        description="Linear:多期历史图线性回归 → 任意年份需求量(无整数周期限制)")
    parser.add_argument("--images", nargs="+", required=True, help="历史期土地利用图(按时间顺序)")
    parser.add_argument("--predict-amount", type=int, required=True, help="预测未来年数")
    parser.add_argument("--plus-backend", default=PLUS_BACKEND)
    parser.add_argument("--compare", default=None, help="手动标准答案 FitFormula.csv,给了就对比公式")
    parser.add_argument("--predict-year", type=int, default=None,
                        help="按公式算某年的各地类需求(如 2028)")
    args = parser.parse_args()

    # 路径归一化
    args.images = [os.path.abspath(p) for p in args.images]
    if args.compare:
        args.compare = os.path.abspath(args.compare)

    print(f"历史图: {len(args.images)} 期,预测 {args.predict_amount} 年")
    formulas = run_linear(args.images, args.predict_amount, args.plus_backend)
    if formulas is None:
        sys.exit(1)

    print(f"\n解析出 {len(formulas)} 类回归公式:")
    for t in sorted(formulas):
        s, b = formulas[t]
        print(f"  Type {t}: y = {s:.2f}·x + {b:.2f}")

    if args.compare:
        print(f"\n=== 和手动标准答案对比: {args.compare} ===")
        manual = parse_formulas(open(args.compare, encoding="utf-8", errors="replace").read())
        if not manual:
            print("  ❌ 读不到手动公式")
            sys.exit(1)
        max_diff = 0.0
        for t in sorted(set(manual) | set(formulas)):
            ms, mb = manual.get(t, (0, 0))
            as_, ab = formulas.get(t, (0, 0))
            d = max(abs(ms - as_), abs(mb - ab))
            max_diff = max(max_diff, d)
            flag = "✅" if d < 0.05 else "⚠️"
            print(f"  Type {t}: 手动 y={ms:.2f}x+{mb:.2f} vs 自动 y={as_:.2f}x+{ab:.2f} 差 {d:.2f} {flag}")
        if max_diff < 0.05:
            print("  ✅ 公式一致(差异 < 0.05,系数显示舍入所致)——linear 模块拿下")
        else:
            print(f"  ⚠️ 公式有差异(最大 {max_diff:.2f}),需要排查")

    if args.predict_year:
        print(f"\n=== 预测 {args.predict_year} 年需求量 ===")
        pred = predict(formulas, args.predict_year)
        for t in sorted(pred):
            print(f"  Type {t}: {pred[t]:,} 像元")
        print(f"  CARS 需求格式: 1,{','.join(str(pred[t]) for t in sorted(pred))}")


if __name__ == "__main__":
    main()
