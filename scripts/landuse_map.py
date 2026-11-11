import os
import sys
import csv
import math
import argparse

import numpy as np
import rasterio

import matplotlib

# 强制输出用 UTF-8:不然在部分终端里中文和 ✅ 会报编码错误
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass  # 个别环境不支持 reconfigure,忽略即可

matplotlib.use("Agg")  # 无界面模式:不弹窗口,直接把图存成文件
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle

# 与 plus-data-check / plus-landuse-stats 相同的约定
# 【参数】无效值约定(PLUS 协议,与其他脚本保持一致,一般不调)
NODATA_VALUES = (0, 255, -128)

# 中文字体:Windows 自带微软雅黑,设了它中文才能正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示,不变成方框

# 按地类名称的语义配色(土地利用制图的通用惯例):
# 草地浅绿、落叶林深绿、耕地金黄、城镇用地红、裸地棕灰、水域蓝、常绿林墨绿
# 【参数】语义配色表:名称→颜色(RGB 0~1)。加新地类名、改某个颜色都在这里
# 名称依据:PLUS 官方《教程一》表 1(20250531 版)"示例数据描述":
#   1:草地 2:落叶林 3:耕地 4:城镇用地 5:裸地 6:水域 7:常绿林
NAME_COLORS = {
    "草地": (0.62, 0.80, 0.35),
    "落叶林": (0.13, 0.47, 0.21),
    "耕地": (0.98, 0.85, 0.40),
    "城镇用地": (0.85, 0.22, 0.22),
    "裸地": (0.72, 0.65, 0.55),
    "水域": (0.15, 0.45, 0.75),
    "常绿林": (0.10, 0.35, 0.15),
}

# ---- 字号规则(HPSCIL 规范:字体保持一致、字号足够清晰)----
# 规范没给具体字号,这里定规则:字号不写死,按底部空白区大小自动推算——
# 空白大字号就大,空白小就压缩,但限定在 8~12 之间保证可读。
# 全图统一:图例文字 = 基准,图例标题/比例尺 = 基准+1,指北针 = 基准+1
# 【参数】字号自适应范围:空白大字号大,限 8~12;想整体放大/缩小改这两个数
FONT_MIN, FONT_MAX = 8, 12

# 兜底配色:没有名称信息时按编码顺序配(tab20 前 12 色,区分度大)
# 【参数】兜底配色:没名称时按编码顺序取色,改顺序/换颜色在这里
CLASS_COLORS = [
    (0.122, 0.467, 0.706),  # 蓝
    (1.000, 0.498, 0.055),  # 橙
    (0.172, 0.627, 0.173),  # 绿
    (0.839, 0.153, 0.157),  # 红
    (0.580, 0.404, 0.741),  # 紫
    (0.549, 0.337, 0.294),  # 棕
    (0.890, 0.466, 0.761),  # 粉
    (0.498, 0.498, 0.498),  # 灰
    (0.737, 0.741, 0.133),  # 黄绿
    (0.090, 0.745, 0.812),  # 青
    (0.900, 0.630, 0.000),  # 金黄
    (0.000, 0.000, 0.000),  # 黑
]


def parse_names(text):
    """把 '1=耕地,2=林地,3=草地' 这种文字解析成 {编码: 名称} 字典。

    这是给"对话直给"用的:用户说"1 是耕地、2 是林地",转成这个格式就行。
    """
    mapping = {}
    for part in text.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        code_s, name = part.split("=", 1)
        try:
            mapping[int(float(code_s.strip()))] = name.strip()
        except ValueError:
            continue
    return mapping


def load_labels(labels_path):
    """读"编码,名称[,颜色]"对照表(CSV),返回 (名称字典, 颜色字典)。

    CSV 格式:第一行列名(编码,名称),第三列"颜色"可选,
    填十六进制(#FF0000)或 matplotlib 颜色名(green)。
    Excel 存的 CSV 也能读(带 BOM 头不碍事)。
    """
    names = {}
    colors = {}
    with open(labels_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                code = int(float(row["编码"]))
            except (KeyError, ValueError):
                continue  # 空行或坏行,跳过
            name = row.get("名称", "").strip()
            names[code] = name if name else str(code)
            color = row.get("颜色", "").strip()
            if color:
                colors[code] = color
    print(f"  已加载地类名称表: {labels_path} ({len(names)} 条)")
    return names, colors


def build_color_table(classes, max_class, labels, label_colors):
    """给 1..max_class 每个编码建颜色表,返回 (cmap, norm, color_table)。

    配色优先级:
        1. 用户名称表里指定了颜色(第三列)→ 用它
        2. 名称在 NAME_COLORS 语义表里(草地绿、水体蓝…)→ 用语义色
        3. 都没有 → 按编码顺序的兜底色
    缺的编码(如 8~14)用浅灰占位,保证颜色区间数和颜色数对得上。
    """
    color_table = []
    for k in range(1, max_class + 1):
        if k not in classes:
            color_table.append((0.95, 0.95, 0.95))  # 缺的编码:浅灰占位
            continue
        if label_colors and k in label_colors:
            color_table.append(label_colors[k])  # 用户指定(hex 字符串也能直接用)
            continue
        name = (labels or {}).get(k, "")
        if name in NAME_COLORS:
            color_table.append(NAME_COLORS[name])  # 语义配色
        else:
            color_table.append(CLASS_COLORS[(k - 1) % len(CLASS_COLORS)])  # 兜底
    cmap = ListedColormap(color_table)
    # BoundaryNorm:规定"编码 1 用什么色、2 用什么色……"
    # 边界写成 [1, 2, ..., max+1],即编码 k 落在区间 [k, k+1) 内
    boundaries = list(range(1, max_class + 2))
    norm = BoundaryNorm(boundaries, max_class)
    return cmap, norm, color_table


def draw_map(map_ax, valid, cmap, norm):
    """把栅格画到地图轴。地图轴无边框,图框由外层 frame_ax 提供。"""
    map_ax.imshow(valid, cmap=cmap, norm=norm, interpolation="nearest")
    # nearest:像素级放大不模糊
    map_ax.set_xticks([])  # 去掉坐标轴刻度,地图不需要
    map_ax.set_yticks([])
    for spine in map_ax.spines.values():
        spine.set_visible(False)  # 地图自己不带框,图框画在外层


def compute_legend_font(n_classes, band_in):
    """按底部空白区高度反推图例字号,返回 (字号, 列数)。

    空白区大 → 字号大;空白区小 → 压缩字号;下限 8 保证可读。
    列数按类数分档(≤4 单列、5~8 双列、>8 三列),多列减少行数、省高度,
    让图例始终能放进框内底部空白区(图例与比例尺都在框内)。
    每行高度按 字号×1.6 磅(行距)估算,再加一点内边距。
    """
    if n_classes <= 4:
        ncol = 1
    elif n_classes <= 8:
        ncol = 2
    else:
        ncol = 3
    rows = math.ceil(n_classes / ncol) + 1  # +1 是"图例"标题行
    font = int((band_in - 0.25) * 72 / (rows * 1.6))
    return max(FONT_MIN, min(FONT_MAX, font)), ncol


def _nice_round(v):
    """取整到 1/2/5×10^n 序列里最接近 v 的值(比例尺长度取整用)"""
    if v <= 0:
        return 5
    exp = 10 ** math.floor(math.log10(v))
    cands = [m * exp for m in (1, 2, 5, 10)]
    return min(cands, key=lambda c: abs(c - v))


def add_scale_bar(ax, pixel_size, w_px, is_geographic, img_frac, font, x0=None,
                  y0=0.07, bar_h=0.016):
    """按 HPSCIL 制图规范画比例尺:2 格、数字取整,画在图框内底部。

    条长按"轴宽的固定目标比例"反推真实公里数(再取整),而不是直接取
    map_width/4:否则大研究区(如济南 100km 宽)会选中一个只占轴宽 4% 的小条,
    三个刻度数字(0/中间值/最大值)挤成一团(2026-11-09 实测)。
    y0:条所在的轴分数纵坐标(默认 0.07;多期对比图比例尺单开一行放图例下面时传 0.02)。
    bar_h:条高(轴百分比;多期对比图用户嫌小,传 0.032 放大 2 倍)。
    """
    if is_geographic:
        print("⚠️ 经纬度坐标系,比例尺按度数算不可靠,已跳过(建议先投影)")
        return
    map_width_km = w_px * pixel_size / 1000

    # 目标条长 = 轴宽的 18%(视觉舒适、刻度文字分得开);反推真实公里并取整
    TARGET_FRAC = 0.18
    target_km = TARGET_FRAC / img_frac * map_width_km
    bar_km = _nice_round(target_km)
    half = bar_km / 2

    # 条长:公里 → 像元 → 地图宽度比例 → 再乘地图占轴的比例
    full_frac = bar_km * 1000 / pixel_size / w_px * img_frac
    half_frac = half * 1000 / pixel_size / w_px * img_frac
    # 兜底:条太短(<6% 轴宽)时文字仍会挤,直接把条拉伸到 12% 并放弃精确长度标注
    if full_frac < 0.06:
        full_frac = 0.12
        half_frac = 0.06

    # 条居中放置:x0 缺省按条长居中
    if x0 is None:
        x0 = (1.0 - full_frac) / 2

    y0, h = y0, bar_h  # 框内底部;条高(轴百分比坐标)
    # 两格:前半白、后半黑(制图常用样式),黑边
    ax.add_patch(Rectangle((x0, y0), half_frac, h, transform=ax.transAxes,
                           facecolor="white", edgecolor="black", linewidth=0.8))
    ax.add_patch(Rectangle((x0 + half_frac, y0), half_frac, h, transform=ax.transAxes,
                           facecolor="black", edgecolor="black", linewidth=0.8))
    # 三个刻度数字:0(左对齐)、中间值(居中)、最大值带单位(右对齐)
    # 字号比图例小一号,刻度文字宽度远小于条长时才不挤
    sfont = max(7, font - 1)
    ax.text(x0, y0 - 0.02, "0", transform=ax.transAxes, ha="left", va="top",
            fontsize=sfont)
    ax.text(x0 + half_frac, y0 - 0.02, f"{half:g}", transform=ax.transAxes,
            ha="center", va="top", fontsize=sfont)
    ax.text(x0 + full_frac, y0 - 0.02, f"{bar_km:g} km", transform=ax.transAxes,
            ha="right", va="top", fontsize=sfont)
    print(f"  比例尺: 0—{half:g}—{bar_km:g} km(条长 {full_frac*100:.0f}% 轴宽)")


def add_north_arrow(ax, font, fx=0.965):
    """指北针:N 字母 + 向上箭头,画在图框内右上角的专属窄边里。

    地图与框四周留统一间距,右上角窄边附近小范围无图,不遮挡。
    """
    ax.annotate("", xy=(fx, 0.95), xytext=(fx, 0.885),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax.text(fx, 0.973, "N", transform=ax.transAxes, ha="center", va="center",
            fontsize=font, fontweight="bold")


def add_legend(ax, classes, color_table, labels, font, ncol, x=0.02, outside=False):
    """图例画在图框内底部空白区左侧;超过 4 类自动两列合并,省高度。

    outside=True:图例移到图框**下方外部**(不压地图)——近方形研究区或地类很多时用,
    靠 bbox_inches='tight' 自动扩展画布包含图例。
    规范要求:同种类图例集中放置、符号大小一致、文字对齐。
    """
    legend_patches = []
    for i, c in enumerate(classes):
        code = int(c)
        name = (labels or {}).get(code, "")
        text = f"{code} {name}" if name else f"地类 {code}"
        legend_patches.append(Patch(facecolor=color_table[code - 1], label=text))
    if outside:
        fig = ax.figure
        fig.legend(handles=legend_patches, loc="upper center",
                   bbox_to_anchor=(0.5, 0.02), title="图例",
                   frameon=False, ncol=ncol, borderpad=0.3, labelspacing=0.4,
                   columnspacing=1.0, fontsize=font, title_fontsize=font + 1)
    else:
        ax.legend(handles=legend_patches, loc="lower left",
                  bbox_to_anchor=(x, 0.03), title="图例",
                  frameon=False, ncol=ncol, borderpad=0.3, labelspacing=0.4,
                  columnspacing=1.0, fontsize=font,
                  title_fontsize=font + 1)


def make_landuse_map(tif_path, output_path, title=None, dpi=300,
                     labels=None, labels_path=None, names=None, inset=None,
                     legend_outside=False):
    """读土地利用 tif,画带图例/比例尺/指北针的专题图,存成 PNG。

    布局:黑边图框包住一切;地图在框内上方(必要时略微缩小让位),
    图例、比例尺、指北针集中放在框内底部的空白区,不遮挡地图。

    参数:
        tif_path     输入的 tif 路径
        output_path  输出的 PNG 路径
        title        图片标题(HPSCIL 规范要求论文图不带图名,默认不画)
        dpi          清晰度,300 是论文配图标准
        names        对话直给的名称 "1=耕地,2=林地,...",优先级最高
        labels       地类名称字典 {编码: 名称}
        labels_path  名称对照表 CSV 路径(自动找输入同名的 _labels.csv)
        inset        局部图范围"列min,行min,列max,行max"(像元坐标)
    """
    print(f"开始出图: {tif_path}")

    if not os.path.exists(tif_path):
        print(f"❌ 文件不存在: {tif_path}")
        return False

    # 名称来源优先级:对话直给(--names)> 名称表文件(--labels)
    # > 自动找 tif 同目录同名 _labels.csv > 没有(图例显示"地类 编码")
    label_colors = {}
    if names and labels is None:
        labels = parse_names(names)
        print(f"  已解析用户给出的名称: {labels}")
    if labels is None and labels_path is None:
        auto = os.path.splitext(tif_path)[0] + "_labels.csv"
        if os.path.exists(auto):
            labels_path = auto
    if labels is None and labels_path:
        if not os.path.exists(labels_path):
            print(f"❌ 名称表不存在: {labels_path}")
            return False
        labels, label_colors = load_labels(labels_path)

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        pixel_size = abs(src.res[0])  # 每个像元的边长(单位:米,投影坐标系下)
        is_geographic = src.crs is not None and src.crs.is_geographic

    # 把无效值"掩"掉:掩掉的位置画成白色,不参与配色
    mask = np.isin(data, NODATA_VALUES)
    valid = np.ma.masked_where(mask, data)

    # 找出有效地类
    classes = np.unique(valid.compressed())
    if len(classes) == 0:
        print("❌ 文件里没有有效地类,无法出图")
        return False
    classes = classes[classes >= 1]  # 保险:排除 0 及负数
    max_class = int(classes.max())
    print(f"  地类: {[int(c) for c in classes]}")
    # 图例默认在框内底部空白区(与比例尺一起);只有显式 --legend-outside 才移到框外
    if legend_outside:
        print("  图例外置模式(--legend-outside,仅显式指定时启用)")

    cmap, norm, color_table = build_color_table(classes, max_class, labels, label_colors)
    h_px, w_px = data.shape  # 行(高)、列(宽)

    # 图框内布局(单位:英寸):
    #   地图与框四周留统一间距 gap_in(不紧挨、也不离太远);
    #   底部空白 band_in 按地图高度比例定(18%,限 1.4~2.8 英寸),放图例+比例尺;
    #   上限 2.8 保证 7~9 类图例(2 列)也装得下,不压地图、不外置;
    #   右侧留 strip_in 窄边放指北针(窄边里没有图)
    # 【参数】图框内布局尺寸(单位:英寸),想调整图感就动这里
    mw_in = 9.2                                             # 地图显示宽度
    mh_in = mw_in * h_px / w_px                             # 地图高度(按栅格比例)
    gap_in = 0.25                                           # 地图与框四周的间距
    band_in = min(2.8, max(1.4, 0.18 * mh_in))              # 底部空白区高(图高18%,限1.4~2.8)
    strip_in = 0.9                                          # 右上指北针窄边宽
    frame_w = mw_in + gap_in + strip_in
    frame_h = mh_in + gap_in + band_in
    # 字号按空白区大小推算(空白大字号大),全图统一
    legend_font, ncol = compute_legend_font(len(classes), band_in)
    # 比例尺长度换算:地图占图框宽度的比例
    map_frac = mw_in / frame_w
    print(f"  图例字号: {legend_font}(按空白区自动推算),{ncol} 列")

    def build_frame(fig, rect):
        """建一个带黑边图框的外层轴,内部放地图轴,返回 (frame_ax, map_ax)"""
        frame_ax = fig.add_axes(rect)
        frame_ax.set_xticks([])
        frame_ax.set_yticks([])
        for spine in frame_ax.spines.values():  # 图框:要素都画在框内
            spine.set_linewidth(1.0)
        # 地图轴:嵌在框内,与上/左/右框边保持 gap_in 间距,底部留空白区
        map_ax = frame_ax.inset_axes(
            [gap_in / frame_w, band_in / frame_h, mw_in / frame_w, mh_in / frame_h])
        return frame_ax, map_ax

    if inset:
        # ---- 整体+局部 双联图 ----
        try:
            cmin, rmin, cmax, rmax = [int(v) for v in inset.split(",")]
        except ValueError:
            print(f"❌ --inset 格式不对: {inset},应为 列min,行min,列max,行max")
            return False
        if not (0 <= cmin < cmax <= data.shape[1] and 0 <= rmin < rmax <= data.shape[0]):
            print(f"❌ 局部范围超出图像 ({data.shape[1]} 列 × {data.shape[0]} 行)")
            return False

        inset_sq = 4.0  # 【参数】局部图面板边长(英寸)
        fig_w = frame_w + inset_sq + 1.3
        fig_h = frame_h + 0.3
        fig = plt.figure(figsize=(fig_w, fig_h))

        # 整体图:左侧,框内放地图+要素
        frame_ax, map_ax = build_frame(
            fig, [0.15 / fig_w, 0.15 / fig_h, frame_w / fig_w, frame_h / fig_h])
        # 局部图:右侧,正方形面板,只有图没有要素
        ax_inset = fig.add_axes(
            [(0.15 + frame_w + 0.5) / fig_w, (fig_h - inset_sq - 0.15) / fig_h,
             inset_sq / fig_w, inset_sq / fig_h])

        draw_map(map_ax, valid, cmap, norm)
        # 红框标出局部图所在位置(坐标是像元,行在前列在后)
        map_ax.add_patch(Rectangle((cmin, rmin), cmax - cmin, rmax - rmin,
                                   fill=False, edgecolor="red", linewidth=1.5))

        sub = valid[rmin:rmax, cmin:cmax]
        draw_map(ax_inset, sub, cmap, norm)
        for spine in ax_inset.spines.values():  # 局部图面板有自己的图框
            spine.set_visible(True)
            spine.set_linewidth(1.0)

        # 要素画在整体图框内:图例左、比例尺中(底部),指北针右上窄边
        add_legend(frame_ax, classes, color_table, labels, legend_font, ncol,
                   outside=legend_outside)
        add_scale_bar(frame_ax, pixel_size, w_px, is_geographic, map_frac, legend_font + 1)
        add_north_arrow(frame_ax, legend_font + 1, 1 - strip_in / 2 / frame_w)

        if title:
            fig.suptitle(title)
        print(f"  局部范围: 列 {cmin}~{cmax},行 {rmin}~{rmax}")
    else:
        # ---- 单幅图 ----
        fig = plt.figure(figsize=(frame_w + 0.3, frame_h + 0.3))
        frame_ax, map_ax = build_frame(
            fig, [0.15 / (frame_w + 0.3), 0.15 / (frame_h + 0.3),
                  frame_w / (frame_w + 0.3), frame_h / (frame_h + 0.3)])

        draw_map(map_ax, valid, cmap, norm)
        add_legend(frame_ax, classes, color_table, labels, legend_font, ncol,
                   outside=legend_outside)
        add_scale_bar(frame_ax, pixel_size, w_px, is_geographic, map_frac, legend_font + 1)
        add_north_arrow(frame_ax, legend_font + 1, 1 - strip_in / 2 / frame_w)
        if title:
            frame_ax.set_title(title)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)  # 画完关闭画布,释放内存
    print(f"✅ 图片已保存: {output_path} (dpi={dpi})")
    return True


def make_sequence_map(tif_year_pairs, output_path, names=None, dpi=150, title=None):
    """多期模拟图并排对比(任务9.1):整图大标题 + 每期年份小标题 + 底部共享图例/比例尺。

    参数:
        tif_year_pairs  [(tif_path, year), ...],按时间顺序
        output_path     输出 PNG 路径
        names           "1=耕地,2=森林,..." 或 None(图例显示"地类 编码")
        dpi             150(对比图可低于单图 300)
        title           整图大标题,如"郑州市 2030-2036 年土地利用对比"(2026-11-10 用户要求;
                        放顶部居中,与右上指北针同排不重叠)
    """
    if not tif_year_pairs:
        print("❌ 多期对比图没有可画的图")
        return False
    labels = parse_names(names) if names else None

    # 全期并集类别 → 统一色表(保证同一地类各期同色)
    classes = set()
    for tif, _ in tif_year_pairs:
        with rasterio.open(tif) as s:
            d = s.read(1)
        classes |= {int(c) for c in np.unique(d) if int(c) not in NODATA_VALUES and int(c) >= 1}
    classes = sorted(classes)
    if not classes:
        print("❌ 多期图里没有有效地类")
        return False
    max_class = max(classes)
    cmap, norm, color_table = build_color_table(classes, max_class, labels, {})

    n = len(tif_year_pairs)
    # 高度 4.8(顶部含大标题 + 指北针两行专属留白,做成显式画布高度;
    # bbox_inches="tight" 会在底部扩展图例时改变保存比例,不能用 rect 顶部留白硬撑)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 4.8), dpi=dpi, squeeze=False)
    axes = axes[0]
    for ax, (tif, year) in zip(axes, tif_year_pairs):
        with rasterio.open(tif) as s:
            d = s.read(1)
        valid = np.ma.masked_where(np.isin(d, NODATA_VALUES), d)
        ax.imshow(valid, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{year} 年", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

    # 比例尺 + 指北针(2026-11-10 用户要求:多期对比图与单期图一样要有;
    # 制图规范 5.3:多幅并列图共享 1 个比例尺和 1 个指北针,且不能覆盖任何内容)。
    # 布局:子图区 rect=[0, 0.13, 1, 0.90](4.4in 高画布上底部 0.57in、顶部 0.44in 留白);
    # 比例尺在底部留白(图例上方一行),指北针在顶部留白;overlay 轴在 tight_layout
    # **之后**创建,避免布局引擎告警。所有装饰元素都不与地图/标题/图例重叠。
    with rasterio.open(tif_year_pairs[0][0]) as s0:
        _res = abs(s0.transform.a)
        _wpx = s0.width
        _geo = bool(s0.crs.is_geographic) if s0.crs else True

    handles = [Patch(facecolor=color_table[c - 1],
                     label=(labels or {}).get(c, "") and f"{c} {(labels or {}).get(c, '')}"
                            or f"地类 {c}") for c in classes]
    # 底部两行(2026-11-10 用户定):图例在上,比例尺单独一行放图例下面;
    # 用户 2026-11-11 要求比例尺变小 → 条高 0.024、刻度字 10(整体元素缩小)
    if title:
        # y=0.86:用户 2026-11-11 微调——"图名再下降 3 行"
        fig.suptitle(title, fontsize=14, y=0.86)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(classes), 8),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.10))
    fig.tight_layout(rect=[0, 0.13, 1, 0.86])

    oax = fig.add_axes([0, 0, 1, 1], frameon=False)  # 整图覆盖轴,只用来画比例尺
    oax.set_xticks([])
    oax.set_yticks([])
    oax.set_axis_off()
    # img_frac = 单幅地图占整图宽度的比例(从 tight_layout 后的实际 bbox 取)。
    # 2026-11-10 修正:此前传 1.0,条长 ≈ 标注值的 3 倍(用户发现"比例尺只是图标")。
    _img_frac = float(axes[0].get_position().width)
    add_scale_bar(oax, _res, _wpx, _geo, img_frac=_img_frac, font=10, x0=0.05,
                  y0=0.025, bar_h=0.024)
    # 经典四星指北针:上实心黑、下空心白,细边线宽 0.8,大小与粗细解耦。
    # (用户 2026-11-10:annotate 的 "-|>" 三角头太丑、放大又太粗 → 换制图规范菱形星)
    from matplotlib.patches import Polygon as _MplPoly
    _cx, _cy, _L, _Wd = 0.985, 0.92, 0.05, 0.007   # 中心/半高/半宽(figure 分数)
    fig.add_artist(_MplPoly([(_cx, _cy + _L), (_cx + _Wd, _cy), (_cx - _Wd, _cy)],
                            closed=True, facecolor="black", edgecolor="black",
                            lw=0.8, transform=fig.transFigure))
    fig.add_artist(_MplPoly([(_cx, _cy - _L), (_cx + _Wd, _cy), (_cx - _Wd, _cy)],
                            closed=True, facecolor="white", edgecolor="black",
                            lw=0.8, transform=fig.transFigure))
    fig.text(_cx, _cy + _L + 0.018, "N", ha="center", va="center",
             fontsize=13, fontweight="bold")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 多期对比图已保存: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="土地利用 tif 出图(图例/比例尺/指北针都在图框内)")
    parser.add_argument("tif_path", help="要出图的 tif 文件路径")
    parser.add_argument("--output", default="", help="输出 PNG 路径;不填则放在输入同目录,文件名加 _map 后缀")
    parser.add_argument("--title", default=None, help="图片标题;规范要求论文图不带图名,不填就不画")
    parser.add_argument("--dpi", type=int, default=300, help="清晰度,默认 300(论文标准)")  # 【参数】输出 dpi,论文标准 300
    parser.add_argument("--labels", default=None, help="地类名称对照表 CSV 路径(列名:编码,名称,颜色可选)")
    parser.add_argument("--names", default=None, help="对话直给名称,如 \"1=耕地,2=林地,3=草地\",优先级最高")
    parser.add_argument("--inset", default=None, help="局部图范围 列min,行min,列max,行max,给了就出整体+局部双联图")
    parser.add_argument("--legend-outside", action="store_true",
                        help="图例移到图框下方外部(不压地图;近方形研究区/地类多时用;>6 类自动外置)")
    args = parser.parse_args()

    if not args.output:
        stem, ext = os.path.splitext(args.tif_path)
        args.output = stem + "_map.png"

    make_landuse_map(args.tif_path, args.output, args.title, args.dpi,
                     labels_path=args.labels, names=args.names, inset=args.inset,
                     legend_outside=args.legend_outside)


if __name__ == "__main__":
    main()
