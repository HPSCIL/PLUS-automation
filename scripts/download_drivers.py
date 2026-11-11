"""驱动因子自动下载(PLUS 建模用):边界 + 参考栅格 → 各因子下载/计算 → 对齐输出

因子清单(核心 8 类、10 个栅格):

    地形(3):dem(30m SRTM,CGIAR 官方镜像)、slope 坡度、aspect 坡向 —— 后两者本地计算
    气候(2):tavg 年均温、prec 年降水 —— CHELSA v2.1(1km,1981-2010 平均)
    人口(1):pop 人口密度 —— WorldPop 100m(默认 2020)
    距离(4):dist_highway(高速+铁路)、dist_main(国道+省道)、dist_road(全部道路)、
             dist_water(河流+湖泊) —— OSM 矢量 + 本地欧氏距离计算

流程:
    1. 定位研究区(行政区划名自动取边界+纠偏,带本地缓存)
    2. 读参考栅格(土地利用图),所有因子最后重投影+重采样到与它完全相同的网格
    3. 逐个因子:下载(大文件用"窗口读取"或 Range 分段并行)→ 对齐 → df_*.tif
    4. 驱动因子报告.md

用法:
    python download_drivers.py --region 武汉市 --ref D:\\...\\lulc_2010.tif
    python download_drivers.py --region D:\\边界.geojson --ref D:\\...\\lulc_2010.tif
    python download_drivers.py --region 武汉市 --ref D:\\...\\lulc_2010.tif --factors dem,pop
    python download_drivers.py --region 武汉市 --ref D:\\...\\lulc_2010.tif --dry-run

已知限制(说明书同步声明):
    - 渠道可达性随网络波动(内置重试+断点续传,--dry-run 可定位断的渠道)
    - 人口默认 2020(WorldPop 有 2000-2020 逐年,--pop-year 可改)
    - OSM 道路/水域在国内有遗漏,距离栅格精度受限于 OSM 完整性
    - 土壤/灯光/GDP 不在核心清单(需要时单独下载)
"""

import os
import sys
import json
import time
import shutil
import argparse
import threading

import requests
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcj02_wgs84 import convert_geojson

# 强制 UTF-8 输出 + 行缓冲(日志每行立即可见,AI 监控不滞后)
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

# ---- 渠道常量(与 plus-data-download 同款边界逻辑,但缓存目录各自独立)----
ADCODE_URL = "https://unpkg.com/china-division/dist/pcas-code.json"
BOUNDARY_URL = "https://geo.datav.aliyun.com/areas_v3/bound/{adcode}.json"
# 【参数】DEM:CGIAR SRTM v4(30m 全球,5°×5° 一个 zip;xx 从西往东 1-72,yy 从北往南 1-24)
CGIAR_ZIP_URL = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_{xx:02d}_{yy:02d}.zip"
# 【参数】气候:NCEP/CMAP 月均再分析(PSL/NOAA 公开,无需账号,国内可达,支持 Range)
# 分辨率 2.5°(约 280km),较粗但完全自动,是 PLUS 论文经典源;
# WorldClim(1km)/CHELSA(1km)在国内常不可达(2026-10-28 实测),可达后可直接替换本表
NCEP_URLS = {
    "tavg": ("https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface/air.mon.mean.nc", "air"),
    "prec": ("https://downloads.psl.noaa.gov/Datasets/cmap/enh/precip.mon.mean.nc", "precip"),
}
# 【参数】气候平均期:默认 1991-2020(30 年)
CLIM_START, CLIM_END = 1991, 2020
# 【参数】人口:WorldPop 100m 全球逐年(中国区,UN 调整版)
# 注意:data.worldpop.org 不支持 Range(2026-10-28 实测),只能整包流式下载(约 600MB)
WORLDPOP_URL = ("https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/"
                "{year}/BSGM/CHN/chn_ppp_{year}_UNadj_constrained.tif")
# 【参数】OSM:Overpass API 公共服务器(主站 + 镜像轮询,某个抽风自动切换)
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]
# 【参数】OSM 兜底:openstreetmap.fr 镜像的中国分省包(Overpass 全挂时用)
# 湖北省包实测 59MB;下载一次后本地提取,不再依赖任何在线查询
PBF_URL_TEMPLATE = ("https://download.openstreetmap.fr/extracts/asia/china/"
                    "{province}-latest.osm.pbf")
# 省级行政区中文名 → pinyin(fr 镜像的包名,2026-10-28 对照目录实测;
# 注意三个例外:inner_mongolia、hong_kong、tibet;镜像无台湾包)
PROVINCE_PINYIN = {
    "北京市": "beijing", "天津市": "tianjin", "河北省": "hebei", "山西省": "shanxi",
    "内蒙古自治区": "inner_mongolia", "辽宁省": "liaoning", "吉林省": "jilin",
    "黑龙江省": "heilongjiang", "上海市": "shanghai", "江苏省": "jiangsu",
    "浙江省": "zhejiang", "安徽省": "anhui", "福建省": "fujian", "江西省": "jiangxi",
    "山东省": "shandong", "河南省": "henan", "湖北省": "hubei", "湖南省": "hunan",
    "广东省": "guangdong", "广西壮族自治区": "guangxi", "海南省": "hainan",
    "重庆市": "chongqing", "四川省": "sichuan", "贵州省": "guizhou",
    "云南省": "yunnan", "西藏自治区": "tibet", "陕西省": "shaanxi",
    "甘肃省": "gansu", "青海省": "qinghai", "宁夏回族自治区": "ningxia",
    "新疆维吾尔自治区": "xinjiang", "香港特别行政区": "hong_kong",
    "澳门特别行政区": "macau",
}

DEFAULT_OUTPUT_ROOT = r"D:\plus_skill_GIS_data"

# 各因子的输出文件名 + 说明(报告用)
FACTOR_INFO = {
    # —— 地形/气候/人口(核心)——
    "dem": ("df_dem.tif", "DEM 高程(m)", "CGIAR SRTM v4, 30m"),
    "slope": ("df_slope.tif", "坡度(度)", "本地计算(基于 DEM)"),
    "aspect": ("df_aspect.tif", "坡向(0-360°,正北 0)", "本地计算(基于 DEM)"),
    "tavg": ("df_tavg.tif", "年均温(℃)", "NCEP 再分析, 2.5°, 1991-2020 平均"),
    "prec": ("df_prec.tif", "年降水(mm/年)", "CMAP 增强降水, 2.5°, 1991-2020 平均"),
    "pop": ("df_pop.tif", "人口密度(人/像元)", "WorldPop 100m"),
    # —— 距离类(官方示例同款细分,全部来自 OSM)——
    "dist_motorway": ("df_dist_motorway.tif", "距高速公路(m)", "OSM highway=motorway + 欧氏距离"),
    "dist_trunk": ("df_dist_trunk.tif", "距主干道(m)", "OSM highway=trunk + 欧氏距离"),
    "dist_primary": ("df_dist_primary.tif", "距国道(m)", "OSM highway=primary + 欧氏距离"),
    "dist_secondary": ("df_dist_secondary.tif", "距省道(m)", "OSM highway=secondary + 欧氏距离"),
    "dist_tertiary": ("df_dist_tertiary.tif", "距三级道路(m)", "OSM highway=tertiary + 欧氏距离"),
    "dist_railway": ("df_dist_railway.tif", "距铁路(m)", "OSM railway=rail/subway + 欧氏距离"),
    "dist_hsrail": ("df_dist_hsrail.tif", "距高铁站(m)", "OSM 火车站点(highspeed=yes 优先)+ 欧氏距离"),
    "dist_gov": ("df_dist_gov.tif", "距政府驻地(m)", "OSM 政府机构点/面 + 欧氏距离"),
    "dist_historic": ("df_dist_historic.tif", "距历史遗址(m)", "OSM historic=* + 欧氏距离"),
    "dist_water": ("df_dist_water.tif", "距水域(m)", "OSM 河流+湖泊 + 欧氏距离"),
    # —— 旧合并版(保留兼容:高速+铁路/国省道合并/全部道路)——
    "dist_highway": ("df_dist_highway.tif", "距高速/铁路(m,合并版)", "OSM + 欧氏距离"),
    "dist_main": ("df_dist_main.tif", "距国道/省道(m,合并版)", "OSM + 欧氏距离"),
    "dist_road": ("df_dist_road.tif", "距全部道路(m,合并版)", "OSM + 欧氏距离"),
    # —— 社会经济代理 ——
    "soil": ("df_soil.tif", "土壤黏粒含量 0-5cm(g/kg,土壤类型代理)", "ISRIC SoilGrids 1km"),
    "nightlight": ("df_nightlight.tif", "夜间灯光亮度(GDP 代理)", "VIIRS 年合成(EOG Mines)"),
}

# 默认因子清单 = 官方示例同款 16 项(地形3+气候2+人口1+距离9+土壤1)
DEFAULT_FACTORS = ("dem,slope,aspect,tavg,prec,pop,"
                   "dist_motorway,dist_trunk,dist_primary,dist_secondary,dist_tertiary,"
                   "dist_railway,dist_hsrail,dist_gov,dist_historic,dist_water,soil")

# Overpass 查询模板:{bbox} = "南,西,北,东"(WGS84)
OVER_QUERIES = {
    "motorway": '[out:json][timeout:300];way["highway"="motorway"]({bbox});out geom;',
    "trunk": '[out:json][timeout:300];way["highway"="trunk"]({bbox});out geom;',
    "primary": '[out:json][timeout:300];way["highway"="primary"]({bbox});out geom;',
    "secondary": '[out:json][timeout:300];way["highway"="secondary"]({bbox});out geom;',
    "tertiary": '[out:json][timeout:300];way["highway"="tertiary"]({bbox});out geom;',
    "railway": '[out:json][timeout:300];way["railway"~"rail|subway"]({bbox});out geom;',
    # 高铁站:全部火车站(高铁站无统一标签,距离到"最近火车站"即可)
    "hsrail": ('[out:json][timeout:300];(node["railway"="station"]({bbox});'
               'way["railway"="station"]({bbox}););out geom;'),
    # 政府驻地:政府办公楼/驻地(点或面)
    "gov": ('[out:json][timeout:300];(node["amenity"~"townhall|government"]({bbox});'
            'node["office"="government"]({bbox});'
            'way["amenity"~"townhall|government"]({bbox});'
            'way["office"="government"]({bbox}););out geom;'),
    "historic": ('[out:json][timeout:300];(node["historic"]({bbox});way["historic"]({bbox});'
                 'relation["historic"]({bbox}););out geom;'),
    # 旧合并版(兼容)
    "highway": ('[out:json][timeout:300];(way["highway"~"motorway|trunk"]({bbox});'
                'way["railway"~"rail|subway"]({bbox}););out geom;'),
    "main": ('[out:json][timeout:300];way["highway"~"primary|secondary"]({bbox});out geom;'),
    "road": '[out:json][timeout:300];way["highway"]({bbox});out geom;',
    "water": ('[out:json][timeout:300];(way["natural"="water"]({bbox});'
              'relation["natural"="water"]({bbox});'
              'way["waterway"~"river|canal|stream"]({bbox}););out geom;'),
}


# ==================== 边界(与 plus-data-download 同款,缓存独立) ====================

def _get_retry(url, attempts=3, timeout=60):
    """GET 请求带重试:校园网/家用 DNS 偶尔抽风,第 1 次解析失败第 2 次常成功"""
    last = None
    for i in range(attempts):
        try:
            return requests.get(url, timeout=timeout)
        except requests.exceptions.ConnectionError as e:
            last = e
            if i < attempts - 1:
                print(f"  ⚠️ 连接失败({e}),{attempts - i - 1} 次重试机会,5 秒后重试")
                time.sleep(5)
    raise last


_PCAS_CACHE = None  # 行政区划表缓存(2MB,一次运行只拉一次)


def _pcas_table():
    """行政区划表(带缓存):省/市/县三级"""
    global _PCAS_CACHE
    if _PCAS_CACHE is None:
        r = _get_retry(ADCODE_URL)
        r.raise_for_status()
        _PCAS_CACHE = json.loads(r.text)
    return _PCAS_CACHE


def resolve_adcode(name):
    """行政区划名 → adcode(公开的 pcas-code 表,含省市县三级)"""
    print(f"  解析行政区划: {name}")
    data = _pcas_table()
    for prov in data:
        if prov.get("name") == name:
            return prov["code"]
        for city in prov.get("children", []):
            if city.get("name") == name:
                return city["code"]
            for dist in city.get("children", []):
                if dist.get("name") == name:
                    return dist["code"]
    return None


def resolve_province_pinyin(name):
    """行政区划名 → 所在省的 pinyin(fr 镜像包名);不在中国/查不到返回 None"""
    data = _pcas_table()
    for prov in data:
        if prov.get("name") == name:
            return PROVINCE_PINYIN.get(name)
        for city in prov.get("children", []):
            if city.get("name") == name:
                return PROVINCE_PINYIN.get(prov["name"])
            for dist in city.get("children", []):
                if dist.get("name") == name:
                    return PROVINCE_PINYIN.get(prov["name"])
    return None


def _full_adcode(code):
    """pcas-code 表的省/市码是简码(42/4201),DataV 边界接口要 6 位全码"""
    if len(code) == 2:
        return code + "0000"
    if len(code) == 4:
        return code + "00"
    return code


def _cached_adcode(name, cache_dir):
    """从边界缓存文件名 boundary_<名称>_<adcode>.geojson 里找回 adcode(断网也能用)"""
    prefix = f"boundary_{name}_"
    if os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            if f.startswith(prefix) and f.endswith(".geojson"):
                adcode = f[len(prefix):-len(".geojson")]
                return adcode if adcode != "unknown" else ""
    return ""


def fetch_boundary_geojson(name, adcode="", cache_path=None):
    """行政区划名 → 边界 GeoJSON(WGS84,已纠偏);本地缓存优先"""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            print(f"  使用本地边界缓存: {os.path.basename(cache_path)}")
            return json.load(f)
    if not adcode:
        adcode = _full_adcode(resolve_adcode(name) or "")
    if not adcode:
        print(f"❌ 找不到行政区划编码: {name}(试试写全称,如'武汉市')")
        return None
    print(f"  获取边界: {BOUNDARY_URL.format(adcode=adcode)}")
    r = _get_retry(BOUNDARY_URL.format(adcode=adcode))
    r.raise_for_status()
    gj = r.json()
    if gj.get("features"):
        gj = convert_geojson(gj)  # GCJ-02 → WGS84
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(gj, f, ensure_ascii=False)
        return gj
    print("❌ 边界数据为空")
    return None


def geojson_bounds(geojson):
    """GeoJSON → (西, 南, 东, 北) 边界框(WGS84)"""

    def walk(geom):
        t = geom.get("type")
        if t == "Polygon":
            yield from ((x, y) for ring in geom["coordinates"] for x, y in ring)
        elif t == "MultiPolygon":
            yield from ((x, y) for poly in geom["coordinates"] for ring in poly for x, y in ring)
        elif t == "LineString":
            yield from ((x, y) for x, y in geom["coordinates"])

    xs, ys = [], []
    for f in geojson["features"]:
        for x, y in walk(f["geometry"]):
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


# ==================== HTTP Range 多线程分段下载(断点续传) ====================

def ranged_http_download(url, dest, threads=4):
    """HTTP 多线程分段下载(断点续传 + 段缓存),返回本地路径

    原理和 FTP 版分段下载一致,但 HTTP 的 Range 头能同时指定起点和终点,
    每段收满即完、不多不少,合并时无需砍越界字节。
    """
    h = requests.head(url, allow_redirects=True, timeout=60)
    h.raise_for_status()
    total = int(h.headers.get("Content-Length", 0))
    if not total:
        raise RuntimeError("服务器没给 Content-Length,无法分段")
    print(f"  远端文件: {url.rsplit('/', 1)[-1]} ({total / 1048576:.0f} MB)")

    done = os.path.getsize(dest) if os.path.exists(dest) else 0
    if done >= total:
        print(f"  ✅ 本地已有完整文件,跳过: {dest}")
        return dest
    if done > 0:
        print(f"  断点续传:从 {done / 1048576:.0f} MB 继续")

    remaining = total - done
    n = max(1, min(threads, remaining // (8 * 1048576) + 1))
    seg = remaining // n
    print(f"  并行分段:{n} 线程 × 每段 {seg / 1048576:.0f} MB")

    # 分段参数存档:主文件变化时旧段作废
    meta_path = dest + ".parts.json"
    meta = {"done": done, "total": total, "seg": seg, "n": n}
    old_meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                old_meta = json.load(f)
        except Exception:
            old_meta = None
    if old_meta != meta:
        for old in [dest + f".part{i}" for i in range(64)]:
            if os.path.exists(old):
                os.remove(old)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    progress = [0] * n

    def _worker(i):
        """第 i 段:Range 下载 [start, end],收满即完(最多重试 3 次)"""
        start = done + i * seg
        end = total - 1 if i == n - 1 else done + (i + 1) * seg - 1
        want = end - start + 1
        part = dest + f".part{i}"
        if os.path.exists(part) and os.path.getsize(part) == want:
            print(f"  第 {i + 1} 段已有完整缓存,直接复用")
            progress[i] = want
            return
        for attempt in range(3):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=600) as r:
                    r.raise_for_status()
                    got = 0
                    with open(part, "wb") as f:
                        for chunk in r.iter_content(1 << 16):
                            f.write(chunk)
                            got += len(chunk)
                            progress[i] = got
                if got == want:
                    return
                print(f"\n  ⚠️ 第 {i + 1} 段不完整({got}/{want}),重试 {attempt + 1}/3")
            except Exception as e:
                print(f"\n  ⚠️ 第 {i + 1} 段出错: {e},重试 {attempt + 1}/3")
        raise RuntimeError(f"第 {i + 1} 段重试 3 次仍失败")

    workers = []
    for i in range(n):
        th = threading.Thread(target=_worker, args=(i,), daemon=True)
        th.start()
        workers.append(th)

    t0 = time.time()
    while any(th.is_alive() for th in workers):
        time.sleep(1)
        got = sum(progress)
        el = max(time.time() - t0, 1e-6)
        speed = got / 1048576 / el
        pct = (done + got) / total * 100
        print(f"\r  {(done + got) / 1048576:6.0f}/{total / 1048576:6.0f} MB "
              f"({pct:5.1f}%)  {speed:5.2f} MB/s", end="")

    for th in workers:
        th.join()

    # 校验:HTTP Range 精确到字节,每段必须分毫不差
    for i in range(n):
        start = done + i * seg
        end = total - 1 if i == n - 1 else done + (i + 1) * seg - 1
        want = end - start + 1
        p = dest + f".part{i}"
        if not os.path.exists(p) or os.path.getsize(p) != want:
            raise RuntimeError(f"第 {i + 1} 段字节数不符")
    with open(dest, "ab" if done > 0 else "wb") as out:
        for i in range(n):
            with open(dest + f".part{i}", "rb") as f:
                shutil.copyfileobj(f, out)  # 逐块拼接(避免大文件一次读进内存)
            os.remove(dest + f".part{i}")
    if os.path.exists(meta_path):
        os.remove(meta_path)
    print(f"\n  ✅ 下载完成: {dest}({n} 线程分段)")
    return dest


# ==================== DEM(CGIAR SRTM) ====================


def fetch_dem(bounds_wgs84, workdir):
    """DEM:CGIAR SRTM 30m —— 下载研究区所在的 5°×5° 瓦片包,裁出窗口

    每个 zip 只有 ~36MB(一个合并 tif 压缩后),直接整包下载最省事;
    研究区跨多个瓦片包时逐个处理,最后拼合。
    """
    x0, y0, x1, y1 = bounds_wgs84  # 西, 南, 东, 北
    # 5°×5° 瓦片编号:xx 从西往东(1-72),yy 从北往南(1-24);
    # 研究区四边各自落在哪个包,就下载这些包(边界恰好压线时多下相邻包,宁可多下不可漏)
    x_first = int((x0 + 180) // 5) + 1
    x_last = int((x1 + 180) // 5) + 1
    y_first = int((60 - y1) // 5) + 1
    y_last = int((60 - y0) // 5) + 1
    zips = {(xx, yy) for xx in range(x_first, x_last + 1)
            for yy in range(y_first, y_last + 1)}

    import zipfile
    from rasterio.windows import from_bounds, Window
    from rasterio.merge import merge

    dem_parts = []
    for xx, yy in sorted(zips):
        zip_url = CGIAR_ZIP_URL.format(xx=xx, yy=yy)
        local_zip = os.path.join(workdir, f"srtm_{xx:02d}_{yy:02d}.zip")
        print(f"  下载 SRTM 瓦片包 srtm_{xx:02d}_{yy:02d}(约 36MB)")
        ranged_http_download(zip_url, local_zip)
        member = f"srtm_{xx:02d}_{yy:02d}.tif"
        with zipfile.ZipFile(local_zip) as zf:
            zf.extract(member, workdir)
        tif = os.path.join(workdir, member)
        # 裁出研究区窗口(瓦片包是 5°×5°,研究区通常只占一小块)
        part = os.path.join(workdir, f"dem_part_{xx}_{yy}.tif")
        with rasterio.open(tif) as src:
            win = from_bounds(max(x0, src.bounds.left), max(y0, src.bounds.bottom),
                              min(x1, src.bounds.right), min(y1, src.bounds.top),
                              src.transform)
            win = win.round_offsets().round_lengths(op="ceil").intersection(
                Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                continue  # 这个包与研究区不相交
            data = src.read(1, window=win)
            profile = src.profile.copy()
            # SRTM 的 NoData 哨兵值是 -32768,元数据里常缺失,必须显式写进 nodata,
            # 否则对齐重采样时会把 -32768 当有效高程
            profile.update(height=data.shape[0], width=data.shape[1],
                           transform=src.window_transform(win), compress="deflate",
                           nodata=-32768 if src.nodata is None else src.nodata)
            with rasterio.open(part, "w", **profile) as dst:
                dst.write(data, 1)
        dem_parts.append(part)
        os.remove(tif)  # 5°×5° 大 tif 用完即删,只留小窗口

    if not dem_parts:
        raise RuntimeError("DEM 瓦片包与研究区没有交集")
    dem_path = os.path.join(workdir, "dem_raw.tif")
    if len(dem_parts) == 1:
        shutil.move(dem_parts[0], dem_path)
        return dem_path
    merged, merged_transform = merge(dem_parts)
    with rasterio.open(dem_parts[0]) as src:
        profile = src.profile.copy()
        profile.update(height=merged.shape[1], width=merged.shape[2],
                       transform=merged_transform, compress="deflate")
        with rasterio.open(dem_path, "w", **profile) as dst:
            dst.write(merged[0], 1)
    return dem_path

# ==================== 气候 / 人口 ====================

def plain_http_download(url, dest, total):
    """单线程流式下载(服务器不支持 Range 时的兜底,中断只能重头来)"""
    done = 0
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1048576:6.0f}/{total / 1048576:6.0f} MB "
                          f"({done / total * 100:5.1f}%)", end="")
    print()
    return dest


def smart_http_download(url, dest, threads=4):
    """自动选策略:服务器支持 Range → 多线程分段;不支持 → 单线程流式"""
    if os.path.exists(dest):
        return dest  # 已有文件不再下(调用方判断是否完整)
    h = requests.head(url, allow_redirects=True, timeout=60)
    h.raise_for_status()
    total = int(h.headers.get("Content-Length", 0))
    probe = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    supports = probe.status_code == 206  # 206 Partial Content = 支持 Range
    probe.close()
    if supports:
        return ranged_http_download(url, dest, threads)
    print(f"  远端文件: {url.rsplit('/', 1)[-1]} ({total / 1048576:.0f} MB)")
    print("  ⚠️ 服务器不支持 Range,单线程流式下载(中断只能重头来)")
    return plain_http_download(url, dest, total)


def netcdf_month_indices(src, start_year, end_year):
    """从 netCDF 的 time 维里找 [start-01, end-12] 的下标区间

    GDAL 把时间维数值存在 'NETCDF_DIM_time_VALUES' 元数据(形如 "{1297320,...}"),
    单位在 'time#units'(形如 "hours since 1800-01-01 00:00:0.0")。
    注意:数值不是年份,必须按 units 换算成真实日期再找下标。
    """
    from datetime import datetime, timedelta
    tags = src.tags() or {}
    raw = tags.get("NETCDF_DIM_time_VALUES", "")
    units = tags.get("time#units", "")
    if raw and units and " since " in units:
        unit, base_str = units.split(" since ", 1)
        try:
            base = datetime.strptime(base_str.split(" ")[0], "%Y-%m-%d")
            vals = [float(v) for v in raw.strip("{}").split(",") if v.strip()]
            mult = {"hours": 3600, "days": 86400, "seconds": 1}.get(unit, 0)
            dates = [base + timedelta(seconds=v * mult) for v in vals]
            i0 = next(i for i, d in enumerate(dates) if d.year >= start_year and d.month == 1)
            i1 = next(i for i, d in enumerate(dates) if d.year > end_year)
            return i0, i1
        except Exception:
            pass  # 解析失败走下面的兜底
    # 兜底:没有时间元数据时按 NCEP 惯例推(1948-01 起每月一条)
    vals = [1948 + i / 12 for i in range(src.count)]
    i0 = next(i for i, v in enumerate(vals) if v >= start_year)
    i1 = next(i for i, v in enumerate(vals) if v > end_year)
    return i0, i1


def fetch_climate(bounds_wgs84, workdir):
    """年均温 + 年降水:NCEP/CMAP 月均 → 30 年气候平均 → 裁研究区窗口

    NCEP 文件不大(39/68MB),整包多线程下载后本地处理;
    时间维是"每月一条"(1948-01 起),按元数据定位 1991-2020 的切片。
    """
    paths = {}
    for name, (url, var) in NCEP_URLS.items():
        dest = os.path.join(workdir, f"ncep_{name}.nc")
        print(f"  下载 {name}({url.rsplit('/', 1)[-1]})")
        smart_http_download(url, dest)
        sub = f'NETCDF:"{dest}":{var}'
        with rasterio.open(sub) as src:
            i0, i1 = netcdf_month_indices(src, CLIM_START, CLIM_END)
            months = src.read(indexes=range(i0 + 1, i1 + 1))  # (月数, 行, 列)
            units = (src.tags().get(f"{var}#units", "") or "").lower()
            profile = src.profile.copy()
        # 缺失值填充为 -9.969e36(_FillValue),必须先屏蔽再求平均,
        # 否则 30 年平均会被填充值污染
        months = np.where(months < -1e30, np.nan, months)
        mean = np.nanmean(months, axis=0)
        mean = np.nan_to_num(mean, nan=0.0)  # 全 NaN 的像元(研究区外)置 0
        # 按元数据里的单位换算成"℃"和"mm/年"
        if name == "tavg":
            if "degc" not in units and "celsius" not in units:
                mean = mean - 273.15  # 开尔文 → 摄氏度(元数据已标 degC 时不转)
        else:
            if "kg" in units:        # kg/m²/s → mm/年
                mean = mean * 86400 * 365.25
            elif "day" in units:     # mm/天 → mm/年(CMAP 实测此单位)
                mean = mean * 365.25
            elif "month" in units:   # mm/月 → mm/年
                mean = mean * 12
            # 单位已是 mm/年或识别不了 → 原样保留
        # 写临时单波段 tif(保留源网格),交给 align_to_ref 重采样对齐
        # 注意:netCDF 驱动的 profile 有两个坑——driver 是 "netCDF"(只读)要改 GTiff;
        # CRS 是 None(有地理变换但不带坐标系),必须显式补 WGS84,
        # 否则重投影会把"度"当"米",输出全 0
        tmp = os.path.join(workdir, f"{name}_clim.tif")
        profile.update(driver="GTiff", height=mean.shape[0], width=mean.shape[1],
                       count=1, dtype="float32", compress="deflate",
                       crs="EPSG:4326")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(mean.astype("float32"), 1)
        paths[name] = tmp
    return paths["tavg"], paths["prec"]


def fetch_pop(bounds_wgs84, workdir, year):
    """人口密度(WorldPop 100m)——整包下载(服务器不支持 Range,约 600MB)"""
    url = WORLDPOP_URL.format(year=year)
    dest = os.path.join(workdir, "pop_full.tif")
    smart_http_download(url, dest)
    return dest


# ==================== 土壤 / 夜间灯光 ====================

# 【参数】土壤:ISRIC SoilGrids(全球 1km,公开无需账号)。黏粒含量(0-5cm)作土壤类型代理
# (SoilGrids 无直接"土壤类型"分类层,常规做法用质地/有机碳等属性代理;实测 .vrt 可达)
SOILGRIDS_VRT = "https://files.isric.org/soilgrids/latest/data/clay/clay_0-5cm_mean.vrt"
# 【参数】夜间灯光:VIIRS 年合成(EOG Mines,公开无需账号)——GDP 代理;
# 服务器不稳定且文件大(~1GB),默认清单不含,需要时显式 --factors nightlight
NIGHTLIGHT_URL = ("https://eogdata.mines.edu/nighttime_light/annual/v22/{year}/"
                  "VNL_v22_npp_{year}_global_vcmslcfg_c202205302300.average_masked.dat.tif.gz")


def fetch_soil(bounds_wgs84, workdir):
    """土壤黏粒含量(SoilGrids 1km)——远程 VRT 按窗口读取(只传研究区小块)"""
    os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
    x0, y0, x1, y1 = bounds_wgs84
    vurl = "/vsicurl/" + SOILGRIDS_VRT
    out_path = os.path.join(workdir, "soil_raw.tif")
    with rasterio.open(vurl) as src:
        from rasterio.windows import from_bounds, Window
        from rasterio.warp import transform_bounds
        # SoilGrids 原生是 Interrupted Goode Homolosine 投影,
        # 必须先把 WGS84 边界转到源的 CRS 再开窗(直接开会得到空窗)
        bx = transform_bounds("EPSG:4326", src.crs, x0, y0, x1, y1)
        win = from_bounds(*bx, src.transform)
        win = win.round_offsets().round_lengths(op="ceil").intersection(
            Window(0, 0, src.width, src.height))
        data = src.read(1, window=win)
        profile = src.profile.copy()
        profile.update(driver="GTiff", height=data.shape[0], width=data.shape[1],
                       transform=src.window_transform(win), count=1,
                       dtype="float32", compress="deflate")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data.astype("float32"), 1)
    return out_path


def fetch_nightlight(bounds_wgs84, workdir, year=2020):
    """夜间灯光(VIIRS 年合成,GDP 代理)——整包下载后裁窗口

    文件是 .tif.gz:gzip 不能随机读,先用 GDAL 的 /vsigzip/ 尝试远程窗口读
    (GDAL 自带可寻址 gzip 读取),失败则整包下载解压。
    """
    import gzip
    url = NIGHTLIGHT_URL.format(year=year)
    x0, y0, x1, y1 = bounds_wgs84
    out_path = os.path.join(workdir, "nightlight_raw.tif")
    try:
        vurl = "/vsigzip//vsicurl/" + url
        with rasterio.open(vurl) as src:
            from rasterio.windows import from_bounds, Window
            win = from_bounds(x0, y0, x1, y1, src.transform)
            win = win.round_offsets().round_lengths(op="ceil").intersection(
                Window(0, 0, src.width, src.height))
            data = src.read(1, window=win)
            profile = src.profile.copy()
            profile.update(driver="GTiff", height=data.shape[0], width=data.shape[1],
                           transform=src.window_transform(win), count=1,
                           dtype="float32", compress="deflate")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data.astype("float32"), 1)
        return out_path
    except Exception as e:
        print(f"  ⚠️ 远程窗口读取失败({e}),回退整包下载(约 1GB,耗时较长)")
        gz_path = os.path.join(workdir, f"nightlight_{year}.tif.gz")
        smart_http_download(url, gz_path)
        tif_path = gz_path[:-3]
        with gzip.open(gz_path, "rb") as gz, open(tif_path, "wb") as f:
            shutil.copyfileobj(gz, f)
        # 裁窗口(与 DEM 同款逻辑)
        from rasterio.windows import from_bounds, Window
        with rasterio.open(tif_path) as src:
            win = from_bounds(x0, y0, x1, y1, src.transform)
            win = win.round_offsets().round_lengths(op="ceil").intersection(
                Window(0, 0, src.width, src.height))
            data = src.read(1, window=win)
            profile = src.profile.copy()
            profile.update(driver="GTiff", height=data.shape[0], width=data.shape[1],
                           transform=src.window_transform(win), count=1,
                           dtype="float32", compress="deflate")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data.astype("float32"), 1)
        os.remove(tif_path)  # 解压出的全球大图用完即删
        return out_path


# ==================== OSM 距离因子 ====================

def overpass_query(query, attempts=3):
    """向 Overpass API 发查询:主站+镜像轮询(某个抽风自动换,每个最多试 attempts 轮)"""
    last_err = None
    for i in range(attempts):
        for base in OVERPASS_URLS:
            try:
                r = requests.post(base, data={"data": query}, timeout=90)
                if r.status_code == 200:
                    data = r.json()
                    # 过载时服务器会返回 200 + remark + 空结果(如 "runtime error: Query timed out"),
                    # 必须识别并当作失败,否则会静默产出全 0 距离栅格
                    if data.get("remark"):
                        last_err = f"{base} remark: {data['remark'][:80]}"
                        print(f"  ⚠️ {last_err},换下一个镜像")
                        continue
                    return data
                last_err = f"{base} 返回 {r.status_code}"
                print(f"  ⚠️ {last_err},换下一个镜像")
            except Exception as e:
                last_err = f"{base} {type(e).__name__}"
                print(f"  ⚠️ {last_err},换下一个镜像")
        if i < attempts - 1:
            print(f"  第 {i + 1} 轮全部失败,10 秒后重试")
            time.sleep(10)
    raise RuntimeError(f"Overpass 查询多次失败(最后错误: {last_err})")


# ==================== OSM 兜底:分省 pbf 本地提取 ====================

_PBF_CACHE = {}  # (pbf_path, bounds) → {kind: geoms},一次提取 4 类共用


def fetch_osm_pbf(province, workdir):
    """下载分省 OSM 包(fr 镜像,湖北实测 59MB),返回本地路径"""
    url = PBF_URL_TEMPLATE.format(province=province)
    dest = os.path.join(workdir, f"{province}-latest.osm.pbf")
    if not os.path.exists(dest):
        print(f"  下载分省 OSM 包: {province}({url.rsplit('/', 1)[-1]})")
        smart_http_download(url, dest)
    return dest


def extract_pbf(pbf_path, bounds_wgs84):
    """从 pbf 提取各类要素(道路细分/铁路/站点/政府/遗址/水域),按研究区范围过滤

    一次遍历全包:线要素按标签分桶;点要素(火车站/政府/遗址)直接入桶;
    水域的 relation(多边湖如东湖)在遍历后按成员 way 拼装。
    返回 {"motorway": [...], "hsrail": [...], ...},元素是 GeoJSON 几何(线/点)。
    """
    import osmium

    key = (pbf_path, bounds_wgs84)
    if key in _PBF_CACHE:
        return _PBF_CACHE[key]

    x0, y0, x1, y1 = bounds_wgs84
    way_coords = {}      # 水域相关 way:id → 坐标(供 relation 拼装)
    water_rel_ids = []   # natural=water 的 relation 的成员 way id
    out = {k: [] for k in ("motorway", "trunk", "primary", "secondary", "tertiary",
                           "railway", "hsrail", "gov", "historic", "water",
                           "highway", "main", "road")}
    hsrail_all = []      # 全部火车站(highspeed=yes 筛选用)

    def in_pt(c):
        """点粗筛:在研究区范围内"""
        return x0 <= c[0] <= x1 and y0 <= c[1] <= y1

    def in_box(c):
        """线粗筛:几何的包围盒与研究区相交才保留"""
        xs = [p[0] for p in c]
        ys = [p[1] for p in c]
        return not (max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1)

    class Handler(osmium.SimpleHandler):
        def node(self, n):
            # 点要素:火车站、政府驻地、历史遗址(距点距离即距设施距离)
            t = n.tags
            kind = None
            if t.get("railway") == "station":
                kind = "hsrail"
            elif t.get("amenity") in ("townhall", "government") or \
                    t.get("office") == "government":
                kind = "gov"
            elif "historic" in t:
                kind = "historic"
            if kind is None:
                return
            try:
                c = (n.location.lon, n.location.lat)
            except osmium.InvalidLocationError:
                return
            if in_pt(c):
                out[kind].append({"type": "Point", "coordinates": c})
                if kind == "hsrail" and t.get("highspeed") == "yes":
                    hsrail_all.append({"type": "Point", "coordinates": c})

        def way(self, w):
            t = w.tags
            kinds = []
            hw = t.get("highway")
            if hw is not None:
                kinds.append("road")                  # 合并版:全部道路
                if hw == "motorway":
                    kinds += ["motorway", "highway"]  # 高速(合并版"高速级"也含)
                elif hw == "trunk":
                    kinds += ["trunk", "highway"]
                elif hw == "primary":
                    kinds += ["primary", "main"]
                elif hw == "secondary":
                    kinds += ["secondary", "main"]
                elif hw == "tertiary":
                    kinds.append("tertiary")
            if t.get("railway") in ("rail", "subway"):
                kinds += ["railway", "highway"]       # 铁路并入合并版"高速级"
            if t.get("railway") == "station":
                kinds.append("hsrail")
            if t.get("amenity") in ("townhall", "government") or \
                    t.get("office") == "government":
                kinds.append("gov")
            if "historic" in t:
                kinds.append("historic")
            if t.get("natural") == "water" or \
                    t.get("waterway") in ("river", "canal", "stream"):
                kinds.append("water")
            if not kinds:
                return
            try:
                coords = [(n.location.lon, n.location.lat) for n in w.nodes]
            except osmium.InvalidLocationError:
                return  # 部分 way 引用了提取范围外的节点(无坐标),跳过
            if len(coords) < 2:
                return
            if "water" in kinds:
                way_coords[w.id] = coords            # 存下坐标供 relation 拼装
            if not in_box(coords):
                return
            for k in kinds:
                out[k].append({"type": "LineString", "coordinates": coords})

        def relation(self, r):
            # 多边水域(如武汉东湖):记录成员 way,遍历结束后拼装
            if r.tags.get("natural") == "water":
                for m in r.members:
                    if m.type == "w" and m.role == "outer":
                        water_rel_ids.append(m.ref)

    h = Handler()
    h.apply_file(pbf_path, locations=True)
    # relation 拼装:成员 way 直接作为独立线要素(用于距离计算足够)
    for wid in water_rel_ids:
        c = way_coords.get(wid)
        if c and len(c) >= 2 and in_box(c):
            out["water"].append({"type": "LineString", "coordinates": c})
    # 高铁站:优先只用 highspeed=yes 的站;该区没有标记时退回到全部火车站
    if hsrail_all:
        out["hsrail"] = hsrail_all
    for kind in out:
        print(f"  pbf {kind}: {len(out[kind])} 条要素")
    _PBF_CACHE[key] = out
    return out


def osm_geoms(kind, bounds_wgs84, workdir=None, province=None, mode="auto"):
    """按类型取 OSM 要素:Overpass 优先(或 --osm 指定渠道),失败转分省 pbf 本地提取"""
    x0, y0, x1, y1 = bounds_wgs84
    if mode != "pbf":
        try:
            bbox = f"{y0},{x0},{y1},{x1}"  # Overpass bbox 顺序:南,西,北,东
            data = overpass_query(OVER_QUERIES[kind].format(bbox=bbox))
            geoms = []
            for el in data.get("elements", []):
                if el.get("type") == "node":
                    # 点要素(火车站/政府/遗址):直接取 lon/lat
                    geoms.append({"type": "Point",
                                  "coordinates": [el["lon"], el["lat"]]})
                    continue
                pts = el.get("geometry")
                if not pts or len(pts) < 2:
                    continue
                # Overpass 的 geometry 是 [lat, lon] 对;GeoJSON 要 [lon, lat]
                coords = [[p["lon"], p["lat"]] for p in pts]
                if len(coords) >= 2:
                    geoms.append({"type": "LineString", "coordinates": coords})
            if not geoms:
                # 0 要素必须报错:否则距离栅格会静默产出"全 0"垃圾数据,比失败更危险
                raise RuntimeError(f"Overpass {kind} 返回 0 要素(服务器可能限流或空库)")
            print(f"  OSM {kind}: {len(geoms)} 条要素(Overpass)")
            return geoms
        except RuntimeError:
            if mode == "overpass":
                raise
            print("  Overpass 不可用,改用分省 pbf 本地提取")
    if not province or not workdir:
        raise RuntimeError("OSM 渠道不可用且没有分省包兜底(研究区不在中国或没给省名)")
    pbf = fetch_osm_pbf(province, workdir)
    return extract_pbf(pbf, bounds_wgs84)[kind]


def distance_from_geoms(geoms, ref_profile):
    """矢量 → 栅格化 → 欧氏距离(米)

    原理:把要素画到与参考栅格同网格的 0/1 图(1=有要素),再用距离变换
    算每个像元到最近要素的"像元个数",乘以像元大小(米)就是距离栅格。
    注意:OSM 几何是 WGS84,而参考网格一般是投影坐标,必须先做几何重投影。
    """
    h, w = ref_profile["height"], ref_profile["width"]
    t = ref_profile["transform"]
    if ref_profile["crs"] is not None and not ref_profile["crs"].is_geographic:
        from rasterio.warp import transform_geom
        geoms = [transform_geom("EPSG:4326", ref_profile["crs"], g) for g in geoms]
    mask = rasterize(geoms, out_shape=(h, w), transform=t, all_touched=True,
                     fill=0, default_value=1).astype(np.uint8)
    res = abs(t.a)  # 投影坐标系的像元大小(米)
    dist = distance_transform_edt(mask == 0) * res
    return dist.astype("float32")


# ==================== 坡度坡向 / 对齐 ====================

def slope_aspect(dem, res):
    """坡度(度)+ 坡向(0-360,正北 0,顺时针)"""
    dzdx = np.gradient(dem, res, axis=1)  # 东西方向高程变化率(米/米)
    dzdy = np.gradient(dem, res, axis=0)  # 南北方向
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.mod(np.degrees(np.arctan2(dzdx, dzdy)), 360)
    return slope.astype("float32"), aspect.astype("float32")


def align_to_ref(src_path, ref_profile, out_path, nodata=None, src_nodata=None):
    """把任意投影/分辨率的栅格重投影+重采样到参考栅格网格(行列完全一致)

    PLUS 要求所有因子与土地利用图逐像素对齐,这一步自动完成。
    src_nodata:源数据的无数据值(有些数据源元数据里没写,需要显式传入,
    否则 -32768/-99999 这类哨兵值会混进结果)。
    """
    with rasterio.open(src_path) as s:
        arr = np.zeros((ref_profile["height"], ref_profile["width"]), dtype="float32")
        sn = src_nodata if src_nodata is not None else s.nodata
        reproject(
            source=rasterio.band(s, 1),
            destination=arr,
            src_transform=s.transform, src_crs=s.crs,
            dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
            resampling=Resampling.bilinear, src_nodata=sn,
        )
        # GDAL 双线性重采样有个经典行为:采样窗口内 4 个源像元全是 nodata 时,
        # 输出会被直接写成 nodata 值本身(而不是跳过)。必须事后把这些像元换回 0,
        # 与土地利用图"背景=0"的约定一致,否则下游 PLUS 会读到哨兵值。
        if sn is not None:
            arr[arr == sn] = 0.0
    return write_array(arr, ref_profile, out_path, nodata=nodata)


def make_ref_profile(ref_path):
    """读参考栅格,提取对齐用的网格参数"""
    with rasterio.open(ref_path) as src:
        return {
            "driver": "GTiff",
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "count": 1,
        }


def write_array(arr, profile, out_path, nodata=None):
    """把内存数组写成与参考栅格同网格的 tif"""
    p = profile.copy()
    p.update(dtype="float32", count=1, compress="deflate", nodata=nodata)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(arr.astype("float32"), 1)
    return out_path


# ==================== 报告 / 主流程 ====================

def write_report(out_dir, region_name, ref_path, results, pop_year, failures):
    lines = ["# 驱动因子下载报告", "",
             f"- 研究区: {region_name}",
             f"- 对齐基准(参考栅格): {ref_path}",
             f"- 人口年份: {pop_year}  气候平均期: {CLIM_START}-{CLIM_END}", "",
             "## 输出文件", ""]
    for fac, (fname, desc, source) in FACTOR_INFO.items():
        if fac not in results:
            continue
        ok, extra = results[fac]
        lines.append(f"- {fname}:{'✅' if ok else '❌'} {desc}({source}){extra}")
    lines += ["", "## 提醒", "",
              "- 所有因子已重投影+重采样到与参考栅格完全相同的行列/投影,可直接进 plus-model",
              "- 气候数据分辨率 2.5°(约 280km),在 30m 网格上表现为平滑场,属 PLUS 论文通行做法",
              "- 距离类因子精度受限于 OSM 数据在国内的完整性(乡村道路/小水体可能有遗漏)",
              "- 人口年份与模拟年份不一致是通行做法,如需其他年份用 --pop-year 重下"]
    with open(os.path.join(out_dir, "驱动因子报告.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def check_user_drivers(folder, ref_tif, ref_profile, out_dir):
    """用户自供驱动因子体检 + 自动对齐(2026-11-10 加:因子与土地利用图一样支持"自己提供")

    体检三项,不过即报错返回 False(调用方 exit 1),绝不悄悄改用户数据:
      1) 至少 1 个 df_*.tif;
      2) 每个因子可读;
      3) 有效数据非全 0 / 非全同值(排除源 nodata 哨兵后)——全 0/常数因子没有
         区分能力,疑似合成占位(导师 2026-11-10 写死禁止,见 FAQ.md F6)。
    网格不一致**自动处理**(用户 2026-11-10 要求,与下载渠道同待遇):
      重投影 + 双线性重采样对齐到参考图网格(align_to_ref,下载因子同款逻辑),
      对齐后的文件再查一次全同值,通过才写入 out_dir。原件一律不动。
    """
    import rasterio
    import numpy as np
    import shutil
    files = sorted(f for f in os.listdir(folder)
                   if f.startswith("df_") and f.lower().endswith(".tif"))
    if not files:
        print(f"❌ {folder} 里没有 df_*.tif 因子文件")
        print("   命名约定:因子文件必须以 df_ 开头、.tif 结尾(如 df_gdp.tif)")
        return False
    with rasterio.open(ref_tif) as r:
        ref_grid = (r.transform, r.width, r.height, str(r.crs))
    print(f"用户自供因子 {len(files)} 个,体检 + 自动对齐中(参考图 {os.path.basename(ref_tif)})...")
    os.makedirs(out_dir, exist_ok=True)
    ok = True
    for f in files:
        p = os.path.join(folder, f)
        with rasterio.open(p) as s:
            g = (s.transform, s.width, s.height, str(s.crs))
            arr = s.read(1)
            sn = s.nodata
        # 有效值检查:排除源数据声明的 nodata 哨兵(-32768/-99999 之类),再判全同值
        valid = arr[arr != sn] if sn is not None else arr
        if valid.size == 0 or float(np.nanmin(valid)) == float(np.nanmax(valid)):
            print(f"  ❌ {f}:有效数据全图同值({float(np.nanmin(valid)):.4g})"
                  f"——全 0/常数因子没有区分能力,疑似占位数据")
            print(f"      导师 2026-11-10 定:禁止合成占位因子,请提供真实数据(见 FAQ.md F6)")
            ok = False
            continue
        dst = os.path.join(out_dir, f)
        if g == ref_grid:
            shutil.copy2(p, dst)     # 网格一致:直接复制,原件不动
            print(f"  ✅ {f}:网格一致,直接复制(值域 "
                  f"[{float(np.nanmin(valid)):.4g}, {float(np.nanmax(valid)):.4g}])")
            continue
        # 网格不一致 → 自动重采样对齐(与下载渠道同款:重投影 + 双线性)
        print(f"  🔧 {f}:网格与参考图不一致(因子 {g[1]}×{g[2]} vs 参考 "
              f"{ref_grid[1]}×{ref_grid[2]}),自动重采样对齐")
        align_to_ref(p, ref_profile, dst, src_nodata=sn)
        with rasterio.open(dst) as s2:      # 对齐后再查一次全同值,防对齐事故
            a2 = s2.read(1)
        if a2.size == 0 or float(np.nanmin(a2)) == float(np.nanmax(a2)):
            print(f"  ❌ {f}:对齐后有效数据全图同值,无法使用(请检查源因子)")
            os.remove(dst)
            ok = False
            continue
        print(f"  ✅ {f}:已自动对齐并写入(值域 "
              f"[{float(np.nanmin(a2)):.4g}, {float(np.nanmax(a2)):.4g}])")
    if not ok:
        return False
    print(f"  ✅ 体检完成:{len(files)} 个因子已就绪(原件未动)")
    return True


def main():
    parser = argparse.ArgumentParser(description="驱动因子下载:边界 + 参考栅格 → 各因子 → 对齐输出")
    parser.add_argument("--region", required=True,
                        help="行政区划名称(如 武汉市)或 GeoJSON 边界文件路径")
    parser.add_argument("--ref", required=True, help="参考栅格(土地利用 tif,所有因子对齐到它的网格)")
    parser.add_argument("--factors", default=DEFAULT_FACTORS,
                        help="逗号分隔因子清单,可选: " + ",".join(FACTOR_INFO))
    parser.add_argument("--output", default=None, help="输出目录(默认 D:\\plus_skill_GIS_data\\<研究区名>\\drivers)")
    parser.add_argument("--pop-year", type=int, default=2020, help="人口数据年份(WorldPop 2000-2020)")
    parser.add_argument("--threads", type=int, default=4, help="多线程分段下载的线程数")
    parser.add_argument("--osm", default="auto", choices=["auto", "overpass", "pbf"],
                        help="OSM 渠道:auto=Overpass 优先失败转 pbf;overpass=只用在线查询;pbf=直接下分省包本地提取")
    parser.add_argument("--dry-run", action="store_true", help="只校验渠道/边界/参考栅格,不下载")
    parser.add_argument("--use-existing", default=None,
                        help="用户自供因子文件夹:只体检(网格一致/可读/非全0)并复制到输出目录,不下载")
    args = parser.parse_args()

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    bad = [f for f in factors if f not in FACTOR_INFO]
    if bad:
        print(f"❌ 未知因子: {bad}(可选: {','.join(FACTOR_INFO)})")
        sys.exit(1)

    # 输出文件夹名必须纯 ASCII(PLUS 下游对路径编码敏感)
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    if os.path.isfile(args.region):
        region_name = os.path.splitext(os.path.basename(args.region))[0]
        folder = region_name if all(ord(c) < 128 for c in region_name) else "region"
        adcode = ""
    else:
        region_name = args.region
        os.makedirs(cache_dir, exist_ok=True)
        adcode = _cached_adcode(region_name, cache_dir)
        if not adcode:
            try:
                adcode = _full_adcode(resolve_adcode(region_name) or "")
            except Exception as e:
                print(f"  ⚠️ 行政区划表获取失败({e})")
        folder = adcode if (adcode and any(ord(c) > 127 for c in region_name)) else region_name
    out_dir = args.output or os.path.join(DEFAULT_OUTPUT_ROOT, folder, "drivers")
    workdir = os.path.join(out_dir, "_raw")  # 下载的原始数据(裁切对齐后可删)
    os.makedirs(workdir, exist_ok=True)
    print(f"研究区: {args.region}(文件夹 {folder})  因子: {factors}")
    print(f"输出: {out_dir}")

    # 1) 定位研究区 + 参考栅格
    if os.path.isfile(args.region):
        with open(args.region, encoding="utf-8") as f:
            geojson = json.load(f)
        print(f"✅ 边界文件已加载(假定 WGS84): {args.region}")
    else:
        cache_path = os.path.join(cache_dir, f"boundary_{region_name}_{adcode or 'unknown'}.geojson")
        geojson = fetch_boundary_geojson(region_name, adcode, cache_path)
        if geojson is None:
            sys.exit(1)
        print(f"✅ 边界已获取并纠偏(WGS84),要素 {len(geojson.get('features', []))} 个")
    bounds_wgs84 = geojson_bounds(geojson)
    print(f"  边界范围(WGS84): 西 {bounds_wgs84[0]:.3f} 南 {bounds_wgs84[1]:.3f} "
          f"东 {bounds_wgs84[2]:.3f} 北 {bounds_wgs84[3]:.3f}")

    # 所在省(OSM 兜底用:Overpass 全挂时下载该省 pbf 本地提取)
    province_pinyin = None
    if not os.path.isfile(args.region):
        try:
            province_pinyin = resolve_province_pinyin(region_name)
            if province_pinyin:
                print(f"  所在省: {province_pinyin}(OSM 兜底渠道)")
        except Exception:
            province_pinyin = None

    if not os.path.isfile(args.ref):
        print(f"❌ 参考栅格不存在: {args.ref}")
        sys.exit(1)
    ref_profile = make_ref_profile(args.ref)
    ref_res = abs(ref_profile["transform"].a)
    print(f"✅ 参考栅格: {ref_profile['width']}×{ref_profile['height']}, "
          f"{ref_profile['crs']}, 像元 {ref_res:.1f} m")

    # 1.5) 用户自供因子:体检 + 自动对齐(网格不一致直接重采样,不要求用户先对齐)
    if args.use_existing:
        if not check_user_drivers(args.use_existing, args.ref, ref_profile, out_dir):
            sys.exit(1)
        print("\n✅ 用户自供因子体检通过(未下载任何数据)")
        return

    # 2) 干跑:只校验各渠道可达性
    if args.dry_run:
        print("\n=== 渠道校验(不下载)===")
        checks = []
        for name, (url, var) in NCEP_URLS.items():
            code = _probe(url)
            checks.append((f"气候 {name}(NCEP/CMAP)", code))
        checks.append(("DEM(CGIAR SRTM)", _probe(CGIAR_ZIP_URL.format(xx=59, yy=6))))
        checks.append(("人口(WorldPop)", _probe(WORLDPOP_URL.format(year=args.pop_year))))
        checks.append(("OSM(Overpass)", _probe(OVERPASS_URLS[0], head=False)))
        for label, code in checks:
            print(f"  {'✅' if code else '❌'} {label}: {code or '不可达'}")
        sys.exit(0)

    # 3) 逐因子:下载/计算 → 对齐 → 写 df_*.tif
    results = {}
    dem_arr = None  # 对齐后的 DEM 数组缓存(slope/aspect 共用,只下载一次)
    cache = {}      # 下载/查询结果缓存(气候一次取两变量,OSM 按类型缓存几何)
    failures = []
    for fac in factors:
        print(f"\n=== 因子 {fac} ===")
        fname, desc, source = FACTOR_INFO[fac]
        out_path = os.path.join(out_dir, fname)
        try:
            if fac in ("dem", "slope", "aspect"):
                # 三个因子共享一次 DEM 下载:先对齐到参考网格,坡度坡向在 30m 网格上算
                if dem_arr is None:
                    print("  下载 DEM(CGIAR SRTM 30m)")
                    raw = fetch_dem(bounds_wgs84, workdir)
                    dem_out = os.path.join(out_dir, FACTOR_INFO["dem"][0])
                    align_to_ref(raw, ref_profile, dem_out)
                    with rasterio.open(dem_out) as d:
                        dem_arr = d.read(1)
                    dem_arr = np.where(dem_arr < -30000, 0.0, dem_arr)  # 兜底屏蔽哨兵值
                    results["dem"] = (True, "")
                if fac == "slope":
                    slope, _ = slope_aspect(dem_arr, ref_res)
                    write_array(slope, ref_profile, out_path)
                    results[fac] = (True, "")
                elif fac == "aspect":
                    _, aspect = slope_aspect(dem_arr, ref_res)
                    write_array(aspect, ref_profile, out_path)
                    results[fac] = (True, "")
            elif fac in ("tavg", "prec"):
                if "climate" not in cache:
                    print("  下载气候月均数据(NCEP/CMAP)")
                    cache["climate"] = fetch_climate(bounds_wgs84, workdir)
                tavg_raw, prec_raw = cache["climate"]
                raw = tavg_raw if fac == "tavg" else prec_raw
                align_to_ref(raw, ref_profile, out_path)
                results[fac] = (True, "")
            elif fac == "pop":
                raw = fetch_pop(bounds_wgs84, workdir, args.pop_year)
                # WorldPop 的无数据值是 -99999(元数据里通常没写,显式传入)
                align_to_ref(raw, ref_profile, out_path, src_nodata=-99999)
                results[fac] = (True, "")
            elif fac == "soil":
                raw = fetch_soil(bounds_wgs84, workdir)
                align_to_ref(raw, ref_profile, out_path)
                results[fac] = (True, "")
            elif fac == "nightlight":
                raw = fetch_nightlight(bounds_wgs84, workdir)
                align_to_ref(raw, ref_profile, out_path)
                results[fac] = (True, "")
            elif fac.startswith("dist_"):
                kind = fac.split("_", 1)[1]  # highway / main / road / water
                if kind not in cache:
                    print(f"  查询 OSM({kind})")
                    cache[kind] = osm_geoms(kind, bounds_wgs84, workdir,
                                            province_pinyin, args.osm)
                arr = distance_from_geoms(cache[kind], ref_profile)
                write_array(arr, ref_profile, out_path)
                results[fac] = (True, "")
            print(f"  ✅ 输出: {out_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ {fac} 失败: {e}")
            results[fac] = (False, f"(失败: {e})")
            failures.append(fac)

    # 4) 报告 + 退出码
    write_report(out_dir, region_name, args.ref, results, args.pop_year, failures)
    print(f"\n✅ 驱动因子报告已保存: {os.path.join(out_dir, '驱动因子报告.md')}")
    if failures:
        print(f"⚠️ 失败的因子: {failures}")
        sys.exit(1)


def _probe(url, head=True):
    """渠道可达性探测:返回 HTTP 状态码字符串;不可达返回空串"""
    try:
        r = (requests.head if head else requests.get)(url, allow_redirects=True, timeout=30)
        return f"HTTP {r.status_code}"
    except Exception:
        return ""


if __name__ == "__main__":
    main()
