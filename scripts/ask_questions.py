#!/usr/bin/env python
"""引导问答标准器(2026-11-13 试验版,新文件,不替代现有流程)

解决"执行 AI 自由发挥弹窗"问题:所有问题的选项文字、标注、分支联动
全部在本文件写死。AI 的职责 = 搬运,不是发挥。

三种模式:
  1. --dump
     输出标准问题清单 JSON(问题文字、选项逐字、分支规则)。
     AI 读它 → 用 AskUserQuestion 逐字弹窗 → 收集用户答案。
  2. --finalize '<答案 JSON 字符串>' 或 --finalize-file 文件
     校验答案(分支联动/年份范围/选项合法性)→ 通过则写 answers.json。
     不合法 → 打印具体错误,exit 1,AI 必须回头改答案重新问。
  3. --dry-run answers.json
     仅展示 answers.json 将翻译成的 run_full 参数(不执行)。

答案 JSON 示例:
{
  "lulc_source": "esri",            # esri | clcd | worldcover | cnlucc | stanford | user
  "lulc_res": 30,                   # 30 | 10
  "years": [2017, 2025],
  "demand": "markov",               # markov | linear
  "do_backtest": true,
  "backtest": "train",              # train(真实面积推荐) | predict(预测面积,需三期)
  "predict": 2035,
  "sequence": 5,                    # 0 = 不要连续预测;>0 = 间隔年数
  "water_lock": "",                 # 开放题:AI 检索后填,如 "lock:该地有黄河干流..."
  "matrix_choice": "regional",      # regional(区域化修改,推荐) | default | custom
  "matrix_restrictions": [],        # regional 时 AI 检索后填 [{"from","to","allow","依据"}]
  "output_dir": "",                 # 空 = 默认(用户主目录)
  "lulc_dir": "",                   # cnlucc/user/stanford 必填:用户提供 tif 的文件夹路径
  "lulc_legend_json": ""            # user/stanford 必填:{"编码":"语义名"} 分类图例
}
"""
import json
import sys
import os

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ESRI_YEARS = (2017, 2025)
CLCD_YEARS = (1990, 2022)
WC_YEARS = (2020, 2021)
# CNLUCC(RESDC)13 期(非逐年)
CNLUCC_YEARS = {1980, 1990, 1995, 2000, 2005, 2008, 2010, 2013, 2015, 2018, 2020, 2023, 2025}
# 2026-11-17 图例库扩容:内置图例数据源年份范围(闭区间)
BUILTIN_LEGEND_SOURCE_RANGES = {
    "globeland30": (2000, 2020),   # 仅 2000/2010/2020 三期(此处只校验区间)
    "glc_fcs30": (1985, 2022),
    "modis_igbp": (2001, 2023),    # 500m 粗分辨率
    "esa_cci": (1992, 2020),
    "from_glc": (2015, 2019),      # 2015(30m)/2017/2019(10m)
    "globcover": (2005, 2009),     # 2005-06/2009 两期
    "modis_umd": (2001, 2023),
}


def questions():
    """标准问题清单(选项文字逐字写死,AI 弹窗必须照抄)"""
    return {
        "_flow": (
            "弹窗流程(必须严格遵守):Q1 和 Q1b 是**两个独立问题,都无条件必弹**,"
            "Q1b 第一选项'不用了'供第一问已选前 3 项的用户跳过——不存在条件分支,"
            "因此不存在漏弹(2026-11-13 用户定)。"
            "Q4 选'要' → 紧接弹 Q5。Q7 为开放题,AI 先检索再填理由。"
        ),
        "Q1_data_source": {
            "question": ("用哪个土地利用数据源?(Esri = 免注册,年份 2017-2025,"
                         "30m 为 10m 自动聚合、10m 为原生;CLCD = 需自行注册提供账号,1990-2022;"
                         "其他数据源里:WorldCover = 免注册仅 2020-2021、Stanford = 需开代理、自己的数据 = 用户提供)"),
            "options": [
                "Esri 30m(免注册,推荐)",
                "Esri 10m(免注册)",
                "CLCD 30m(需注册)",
                "其他数据源",
            ],
        },
        "Q1b_data_source_more": {
            "question": "补充数据源(第二问;第一问选了前 3 项的用户选第 1 项'不用了'即可)",
            "options": [
                "不用了,就用第一问选的",
                "WorldCover 30m/10m 2020-2021(免注册;仅 2020/2021 两期)",
                "Stanford 派生数据(需开代理;数据经代理下载后提供文件夹)",
                "我自己的多年份 tif(紧接弹输入框要文件夹路径;若知道数据源名"
                "[如 CNLUCC/GlobeLand30]一并告诉 AI,命中内置图例库则免图例表)",
            ],
        },
        "Q2_years": {
            "question": "用哪几期历史数据建模?(跨度越长,变化样本越足)",
            "branch_esri": [
                "2017、2025 两期(推荐,跨度越长变化样本越足)",
                "2017、2021、2025 三期",
                "其他年份(输入框,逗号分隔)",
            ],
            "branch_clcd": [
                "2000、2020 两期(推荐,跨度越长变化样本越足)",
                "2000、2010、2020 三期",
                "其他年份(输入框,逗号分隔)",
            ],
            "branch_worldcover": [
                "2020、2021 两期(推荐,该数据源仅此两期)",
                "其他年份(输入框,逗号分隔)",
            ],
        },
        "Q3_demand": {
            "question": "未来各地类面积怎么推算?",
            "options": [
                "Markov 马氏链(推荐;两期历史的标准方法)",
                "Linear 线性回归(三期历史时更合适)",
            ],
        },
        "Q4_backtest_need": {
            "question": "要不要先做回测精度检验?",
            "options": [
                "要,出 Kappa/FoM(推荐;回测用历史某年做验证)",
                "不要,纯预测(目标年在末年之后,无真实图可比)",
            ],
        },
        "Q5_backtest_mode": {
            "question": "回测用哪种口径?(验证年 = 末年;[首年]/[末年] 按所选历史期数填)",
            "options": [
                "用 [末年] 年真实面积、从 [首年] 年图模拟到 [末年](推荐)",
                "用预测的面积、从 [首年] 年图模拟到 [末年](需三期历史)",
            ],
        },
        "Q6_sequence": {
            "question": "需要中间的过渡年份图吗?",
            "options_template": (
                "不要,只出 {target} 一张(推荐) / "
                "要(每个候选必须写成'每 N 年一期,共 X 期:完整年份列表',禁止只写'每 N 年一期')"
            ),
        },
        "Q8_matrix": {
            "question": ("地类转换矩阵用哪种?(水域锁不锁定 = 矩阵规则的一部分,在此一并判断:"
                         "AI 检索发现该地水域重要 → 矩阵规则自动加'水域→其他 禁止'并附依据;"
                         "缺水城市/水源保护区必须双向锁定[禁转出+禁转入],防历史概率灌水虚增,"
                         "郑州 2026-11-13 实测只禁转出虚增 32%;水域不重要 → 不锁并说明理由)"),
            "options": [
                "区域化修改(AI 自动定)(推荐;AI 收集该地文献/政策/实地情况后自动定规则)",
                "默认模板(森林/建设用地不可逆,其余开放)",
                "我自己定规则(用户给出具体规则)",
            ],
        },
        "Q9_output": {
            "question": "成果存在哪?(必须显示完整路径,不许只写'默认目录')",
            "options_template": (
                "默认目录(推荐):C:\\Users\\<用户名>\\plus_skill_GIS_data\\<研究区名> / "
                "我自己指定(紧接弹输入框要路径)"
            ),
        },
    }


def validate(answers):
    """校验答案,返回 (ok, 错误列表)"""
    errs = []
    src = answers.get("lulc_source")
    years = answers.get("years") or []
    if src not in ("esri", "clcd", "worldcover", "cnlucc", "stanford", "user",
                   "globeland30", "glc_fcs30", "modis_igbp", "esa_cci",
                   "from_glc", "globcover", "modis_umd"):
        errs.append(f"lulc_source 非法: {src}")
        return False, errs
    if src == "esri" and any(not (ESRI_YEARS[0] <= y <= ESRI_YEARS[1]) for y in years):
        errs.append(f"Esri 年份必须 {ESRI_YEARS[0]}-{ESRI_YEARS[1]},给了 {years}")
    if src == "clcd" and any(not (CLCD_YEARS[0] <= y <= CLCD_YEARS[1]) for y in years):
        errs.append(f"CLCD 年份必须 {CLCD_YEARS[0]}-{CLCD_YEARS[1]},给了 {years}")
    if src == "worldcover" and any(y not in WC_YEARS for y in years):
        errs.append(f"WorldCover 只有 {list(WC_YEARS)},给了 {years}")
    if src == "cnlucc" and any(y not in CNLUCC_YEARS for y in years):
        errs.append(f"CNLUCC 只有 13 期 {sorted(CNLUCC_YEARS)},给了 {years}")
    if src in BUILTIN_LEGEND_SOURCE_RANGES:
        lo, hi = BUILTIN_LEGEND_SOURCE_RANGES[src]
        if any(not (lo <= y <= hi) for y in years):
            errs.append(f"{src} 年份范围 {lo}-{hi},给了 {years}")
    if src in ("cnlucc", "user", "stanford") or src in BUILTIN_LEGEND_SOURCE_RANGES:
        if not answers.get("lulc_dir"):
            errs.append("选了这个数据源必须给 lulc_dir(用户提供数据的文件夹路径)")
    if src in ("user", "stanford") and not answers.get("lulc_legend_json"):
        errs.append("user/stanford 数据源必须给 lulc_legend_json(分类图例,格式 {\"编码\":\"语义名\"})")
    if len(years) < 2:
        errs.append("历史期数至少 2 期")
    res = answers.get("lulc_res")
    if src in ("esri", "worldcover") and res not in (10, 30):
        errs.append(f"{src} 的 lulc_res 必须 10 或 30,给了 {res}")
    if src == "clcd" and res != 30:
        errs.append("CLCD 只有 30m")
    if answers.get("demand") not in ("markov", "linear"):
        errs.append("面积方案只允许 markov / linear(auto 已删除)")
    bt = answers.get("backtest")
    if answers.get("do_backtest") and bt not in ("train", "predict"):
        errs.append("回测口径只允许 train(真实面积推荐)/ predict(预测面积)")
    if bt == "predict" and len(years) < 3:
        errs.append("预测面积口径需三期历史,当前不足三期")
    seq = answers.get("sequence") or 0
    if seq and (not isinstance(seq, int) or seq <= 0):
        errs.append("sequence 必须为正整数间隔年数")
    if answers.get("matrix_choice") == "regional" and not answers.get("matrix_restrictions"):
        errs.append("选了区域化修改,matrix_restrictions 不能为空(AI 检索后必填)")
    return (not errs), errs


def main():
    if "--dump" in sys.argv:
        print(json.dumps(questions(), ensure_ascii=False, indent=2))
        return
    if "--finalize" in sys.argv:
        raw = sys.argv[sys.argv.index("--finalize") + 1]
        answers = json.loads(raw)
        ok, errs = validate(answers)
        if not ok:
            print("❌ 答案校验失败:")
            for e in errs:
                print(f"   - {e}")
            sys.exit(1)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "answers.json"), "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
        print("✅ 答案校验通过,已写 answers.json")
        return
    if "--finalize-file" in sys.argv:
        # 文件方式(2026-11-17 补:docstring 早已承诺,实现缺失;AI 手拼 JSON 字符串
        # 容易转义出错,文件更稳)
        path = sys.argv[sys.argv.index("--finalize-file") + 1]
        with open(path, encoding="utf-8") as f:
            answers = json.load(f)
        ok, errs = validate(answers)
        if not ok:
            print("❌ 答案校验失败:")
            for e in errs:
                print(f"   - {e}")
            sys.exit(1)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "answers.json"), "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
        print("✅ 答案校验通过,已写 answers.json")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
