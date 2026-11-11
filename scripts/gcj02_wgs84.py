"""GCJ-02(火星坐标)→ WGS84 纠偏(公开算法,无需密钥)

国内行政区划边界数据源(阿里 DataV 等)常用 GCJ-02 加密坐标,
而卫星数据(CLCD 等)是 WGS84。下载裁切前必须纠偏,否则错位几百米。
算法是公开的坐标偏移公式,纯 Python 实现。
"""

import math

_PI = math.pi
_A = 6378245.0            # 长半轴
_EE = 0.00669342162296594323  # 偏心率平方


def _out_of_china(lng, lat):
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * _PI) + 40.0 * math.sin(y / 3.0 * _PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * _PI) + 320 * math.sin(y * _PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * _PI) + 40.0 * math.sin(x / 3.0 * _PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * _PI) + 300.0 * math.sin(x / 30.0 * _PI)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lng, lat):
    """单个点纠偏"""
    if _out_of_china(lng, lat):
        return lng, lat
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * _PI
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * _PI)
    dlng = (dlng * 180.0) / (_A / sqrtmagic * math.cos(radlat) * _PI)
    return lng - dlng, lat - dlat


def convert_geojson(geojson_obj):
    """把 GeoJSON 里所有坐标从 GCJ-02 纠偏成 WGS84(原地改对象)"""
    def walk(coord):
        if isinstance(coord, (list, tuple)) and len(coord) >= 2 and \
           all(isinstance(v, (int, float)) for v in coord[:2]):
            coord[0], coord[1] = gcj02_to_wgs84(coord[0], coord[1])
        elif isinstance(coord, list):
            for c in coord:
                walk(c)
    for feature in geojson_obj.get("features", []):
        walk(feature.get("geometry", {}).get("coordinates", []))
    return geojson_obj
