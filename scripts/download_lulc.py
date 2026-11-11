"""土地利用图自动下载(CLCD 30m):行政区划名/边界文件 → 下载多年 → 裁切到研究区

流程:
    1. 定位研究区:行政区划名(自动取边界+GCJ-02 纠偏)或用户给 GeoJSON 边界
    2. 下载:CLCD 官方发布源(30m 年度中国全图),流式下载+进度;失败提示手动兜底
    3. 裁切:按边界裁出研究区,输出 lulc_<年份>.tif
    4. 报告:数据源/年份/行列/投影/编码说明

用法:
    python download_lulc.py --region 武汉市 --years 2010,2020 [--output 目录] [--threads 4] [--dry-run]
    python download_lulc.py --region D:\\边界.geojson --years 2010,2020
    python download_lulc.py --region 武汉市 --years 2010 --manual --manual-files "D:\\CLCD_2010.tif"  # 手动兜底

已知限制(说明书同步声明):
    - 不做行列/投影对齐(下载+裁切后需自行统一,plus-data-check 可检查)
    - CLCD 编码原样保留(1~9 类),映射到研究区分类体系是用户研究决策
    - 下载源改版/网络问题时,用 --manual 手动下载兜底
"""

import os
import sys
import json
import time
import shutil
import ftplib
import argparse
import threading

import requests
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcj02_wgs84 import convert_geojson

# 强制 UTF-8 输出 + 行缓冲:重定向到日志文件时,不攒 8KB 才刷盘,
# 每行立即可见(AI 看日志时不会以为进程卡死)
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

# 【参数】行政区划名称→adcode 表(公开源,和 cn-dem 同款)
ADCODE_URL = "https://unpkg.com/china-division/dist/pcas-code.json"
# 【参数】边界数据源(阿里 DataV,GCJ-02 坐标,需纠偏)
BOUNDARY_URL = "https://geo.datav.aliyun.com/areas_v3/bound/{adcode}.json"
# 【参数】CLCD 下载 URL 模板(官方 Zenodo 记录 records/8176941,文件名 *_albert.tif;
# 国内直连常超时,默认渠道不可用时用 --url 覆盖或 --manual 手动兜底)
CLCD_URL_TEMPLATE = "https://zenodo.org/records/8176941/files/CLCD_v01_{year}_albert.tif?download=1"

# 【参数】ESA WorldCover 官方分发(AWS S3 欧区,2026-11-10 实测国内可达,
# 单线程 ~100KB/s,多线程分段并行可提速数倍):3°×3° 分幅 tile,命名 N{lat}E{lon};
# v100=2020,v200=2021(该产品只有这两年)。许可 CC BY 4.0。
WORLDCOVER_TILE_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
                       "{ver}/{year}/map/ESA_WorldCover_10m_{year}_{ver}_"
                       "N{lat:02d}E{lon:03d}_Map.tif")

DEFAULT_OUTPUT_ROOT = os.path.join(os.path.expanduser("~"), "plus_skill_GIS_data")  # 默认输出根目录 = 用户主目录下(跨机通用)


# ---- NCDC FTP 渠道(CLCD 国内官方镜像,实测可达;账号在 ftp_config.json)----

def load_ftp_config():
    """NCDC FTP 账号:环境变量 PLUS_FTP_CONFIG > ~/.plus-full/ftp_config.json > 脚本目录

    skill 包内不放凭据(P1-5);脚本目录那份只用于开发机本地调试。
    """
    cands = []
    env = os.environ.get("PLUS_FTP_CONFIG")
    if env:
        cands.append(env)
    cands.append(os.path.join(os.path.expanduser("~"), ".plus-full", "ftp_config.json"))
    cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ftp_config.json"))
    for p in cands:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        "找不到 NCDC FTP 账号配置。请任选其一:\n"
        "  1) 设置环境变量 PLUS_FTP_CONFIG=<配置文件路径>\n"
        "  2) 放到 ~/.plus-full/ftp_config.json\n"
        "  3) 在 scripts/ 下放 ftp_config.json(仅限本机开发,勿提交)\n"
        "格式:{\"host\":..., \"port\":2121, \"username\":..., \"password\":...}")


def _connect_retry(cfg, attempts=3):
    """建 FTP_TLS 连接带重试:晚上网络/服务器偶尔抽风,TLS 握手超时第 2 次常成功"""
    last = None
    for i in range(attempts):
        try:
            t = ftplib.FTP_TLS()
            t.connect(cfg["host"], cfg.get("port", 2121), timeout=60)
            t.login(cfg["username"], cfg["password"])
            t.prot_p()
            t.voidcmd("TYPE I")
            return t
        except Exception as e:
            last = e
            if i < attempts - 1:
                print(f"  ⚠️ FTP 连接失败({e}),{attempts - i - 1} 次重试机会,10 秒后重试")
                time.sleep(10)
    raise last


def ftp_download(remote_name, local_path, cfg=None, resume=True):
    """从 NCDC FTP 下载文件(进度显示 + 断点续传),返回本地路径"""
    cfg = cfg or load_ftp_config()
    try:
        ftp = _connect_retry(cfg)
        total = ftp.size(remote_name)
    except Exception as e:
        # FTP 暂不可达时,若本地已有缓存文件则直接复用(文件在下载时已逐字节校验过)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 50 * 1048576:
            print(f"  ⚠️ FTP 暂不可达({e}),本地缓存存在,视为完整直接使用: {local_path}")
            return local_path
        raise
    print(f"  远端文件: {remote_name}({total / 1048576:.0f} MB)")

    done = 0
    if resume and os.path.exists(local_path):
        done = os.path.getsize(local_path)
        if done >= total:
            print(f"  ✅ 本地已有完整文件,跳过: {local_path}")
            ftp.quit()
            return local_path

    mode = "ab" if done > 0 else "wb"
    if done > 0:
        ftp.sendcmd(f"REST {done}")  # 断点续传;注意:必须 sendcmd,不能 voidcmd——
        # REST 成功返回 350(不是 2xx),voidcmd 会误判为错误直接抛异常
        print(f"  断点续传:从 {done / 1048576:.0f} MB 继续")

    def cb(chunk):
        nonlocal done
        done += len(chunk)
        pct = done / total * 100
        print(f"\r  {done / 1048576:6.0f}/{total / 1048576:6.0f} MB ({pct:5.1f}%)", end="")

    with open(local_path, mode) as f:
        ftp.retrbinary(f"RETR {remote_name}", lambda c: (f.write(c), cb(c))[1])
    print()
    ftp.quit()
    print(f"  ✅ 下载完成: {local_path}")
    return local_path


# 省级行政区中文名 → pinyin(分省包成员命名用,对照 2026-11-04 实测成员列表;
# 注意镜像用了一些非标准拼写如 ganshu,匹配不到时兜底全部解压)
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


def resolve_province_pinyin(adcode):
    """adcode → 所在省 pinyin(用 pcas 表按前 2 位找省名再查拼音)"""
    if not adcode:
        return None
    prov_code = adcode[:2]
    try:
        r = _get_retry(ADCODE_URL)
        r.raise_for_status()
        data = json.loads(r.text)
        for prov in data:
            if str(prov.get("code", "")).startswith(prov_code):
                return PROVINCE_PINYIN.get(prov["name"])
    except Exception as e:
        print(f"  ⚠️ 省名解析失败({e})")
    return None


def ftp_province_zip_fallback(year, cfg, adcode, out_dir):
    """全国图缺失年份的兜底:下载分省 zip,解压出研究区所在省

    NCDC 镜像个别年份(实测 2005)没有全国 tif,只有
    CLCD_v01_<年>_albert_province.zip(34 个省级 tif 打包,约 780MB)。
    解压出研究区所在省(adcode 前 2 位)的成员;adcode 未知时全部解压后拼合。
    """
    import zipfile
    import re
    remote = f"CLCD_v01_{year}_albert_province.zip"
    zip_path = os.path.join(out_dir, f"CLCD_{year}_province.zip")
    print(f"  下载分省包: {remote}")
    ftp_download_parallel(remote, zip_path, cfg)

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
        members = []
        if adcode:
            # 成员是拼音命名(CLCD_v01_2005_albert_hubei.tif):adcode → 省名 → 拼音
            pinyin = resolve_province_pinyin(adcode)
            if pinyin:
                members = [n for n in names if pinyin in n.lower()]
            if not members:
                # 拼音也匹配不到(如镜像的非标准拼写):全部解压后拼合兜底
                print(f"  ⚠️ 没匹配到省拼音成员({pinyin}),全部解压拼合(耗时较长)")
                members = names
        else:
            members = names
        tif_path = os.path.join(out_dir, f"CLCD_{year}_raw.tif")
        if len(members) == 1:
            with zf.open(members[0]) as src, open(tif_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            # 多成员(adcode 未知):全部解压后拼合成一张
            parts = []
            for m in members:
                p = os.path.join(out_dir, os.path.basename(m))
                with zf.open(m) as src, open(p, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                parts.append(p)
            from rasterio.merge import merge
            merged, merged_t = merge(parts)
            with rasterio.open(parts[0]) as s:
                profile = s.profile.copy()
                profile.update(height=merged.shape[1], width=merged.shape[2],
                               transform=merged_t, compress="deflate")
                with rasterio.open(tif_path, "w", **profile) as dst:
                    dst.write(merged[0], 1)
            for p in parts:
                os.remove(p)
    print(f"  ✅ 分省包提取完成: {tif_path}")
    return tif_path


def ftp_download_parallel(remote_name, local_path, cfg=None, threads=4, resume=True):
    """多线程分段并行下载:每个线程一条独立 FTP 连接,各下自己的一段

    原理(FTP 协议的知识点):
        REST 命令只能设"起始偏移",不能设"结束位置"——所以每个连接
        从自己的起点开始收,收满本段字节数就主动断开(故意断尾,
        服务器端只是记一条日志,不影响别人)。
        主线程等所有线程收完后,按顺序把各段 .part 临时文件合并成最终文件。
        断点续传照常生效:本地已有的部分视为"已完成前缀",线程只下剩余部分。

    NCDC FTP 单连接实测限速 ~60KB/s,4 线程约 240KB/s(4 倍)。
    """
    cfg = cfg or load_ftp_config()

    def _connect():
        """每条线程各开一条独立 FTP 连接(FTP 连接不是线程安全的,必须各用各的)"""
        t = ftplib.FTP_TLS()
        t.connect(cfg["host"], cfg.get("port", 2121), timeout=60)
        t.login(cfg["username"], cfg["password"])
        t.prot_p()
        t.voidcmd("TYPE I")
        return t

    # 查远端大小用带重试的连接(查完即断,不占会话);线程自己的连接由段重试兜底
    try:
        ftp = _connect_retry(cfg)
        total = ftp.size(remote_name)
        ftp.quit()
    except Exception as e:
        # FTP 暂不可达时,若本地已有缓存文件则直接复用(文件在下载时已逐字节校验过)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 50 * 1048576:
            print(f"  ⚠️ FTP 暂不可达({e}),本地缓存存在,视为完整直接使用: {local_path}")
            return local_path
        raise
    print(f"  远端文件: {remote_name}({total / 1048576:.0f} MB)")

    # 断点续传:本地已有部分 = 已完成前缀,线程只下剩余部分
    done = os.path.getsize(local_path) if (resume and os.path.exists(local_path)) else 0
    if done >= total:
        print(f"  ✅ 本地已有完整文件,跳过: {local_path}")
        return local_path
    if done > 0:
        print(f"  断点续传:从 {done / 1048576:.0f} MB 继续")

    # 分段参数存档:主文件又下过一部分的话,旧的 .part 与新分段错位,必须作废重下
    remaining = total - done
    # 每段至少 8MB:小文件不必开一堆无谓的连接
    n = max(1, min(threads, remaining // (8 * 1048576) + 1))
    seg = remaining // n  # 每段字节数(最后一段拿剩余全部)
    print(f"  并行分段:{n} 线程 × 每段 {seg / 1048576:.0f} MB")

    meta_path = local_path + ".parts.json"
    meta = {"done": done, "total": total, "seg": seg, "n": n}
    old_meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                old_meta = json.load(f)
        except Exception:
            old_meta = None
    if old_meta != meta:
        # 分段参数与上次不同(例如主文件又下过一部分),旧段作废,重下
        for old in [local_path + f".part{i}" for i in range(64)]:
            if os.path.exists(old):
                os.remove(old)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    progress = [0] * n  # 每段已收字节(多线程写同一列表的不同下标,安全)

    def _worker(i):
        """下载第 i 段:从起点收满本段字节数就断(每段最多重试 3 次)"""
        start = done + i * seg                       # 本段起点(绝对偏移)
        want = remaining - i * seg if i == n - 1 else seg  # 本段应收字节
        part = local_path + f".part{i}"              # 本段的临时文件
        # 上次中断留下的完整分段直接复用(不必重下)
        if os.path.exists(part):
            size = os.path.getsize(part)
            if want <= size <= want + (1 << 16):
                print(f"  第 {i + 1} 段已有完整缓存,直接复用")
                progress[i] = want
                return
        for attempt in range(3):
            t = sock = None
            try:
                t = _connect()
                t.sendcmd(f"REST {start}")           # 从本段起点开始(REST 返回 350,必须 sendcmd)
                sock = t.transfercmd(f"RETR {remote_name}")
                got = 0
                with open(part, "wb") as f:
                    while got < want:
                        chunk = sock.recv(1 << 16)   # 一次收 64KB
                        if not chunk:
                            break                    # 服务器提前断开 → 重试
                        f.write(chunk)
                        got += len(chunk)
                        progress[i] = got
                if got >= want:
                    return                           # 本段收满,成功
                print(f"\n  ⚠️ 第 {i + 1} 段中断({got}/{want} 字节),重试 {attempt + 1}/3")
            except Exception as e:
                print(f"\n  ⚠️ 第 {i + 1} 段出错: {e},重试 {attempt + 1}/3")
            finally:
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass
                if t:
                    try:
                        t.quit()
                    except Exception:
                        pass
        raise RuntimeError(f"第 {i + 1} 段重试 3 次仍失败")

    workers = []
    for i in range(n):
        th = threading.Thread(target=_worker, args=(i,), daemon=True)
        th.start()
        workers.append(th)

    # 主线程只负责刷新进度条(每 1 秒),不碰数据
    t0 = time.time()
    while any(th.is_alive() for th in workers):
        time.sleep(1)
        got = sum(progress)
        el = max(time.time() - t0, 1e-6)
        speed = got / 1048576 / el  # MB/s(本次续传部分)
        pct = (done + got) / total * 100
        print(f"\r  {(done + got) / 1048576:6.0f}/{total / 1048576:6.0f} MB ({pct:5.1f}%)  {speed:5.2f} MB/s", end="")

    # 全部线程结束后 join:哪个线程抛了异常,这里原样抛给上层
    for th in workers:
        th.join()

    # 校验每段字节数:允许比 want 多至多 64KB——每段最后一"口"收满时会
    # 越过段边界,多出的字节属于下一段开头,合并时砍掉即可,不影响正确性
    for i in range(n):
        want = remaining - i * seg if i == n - 1 else seg
        p = local_path + f".part{i}"
        if not os.path.exists(p):
            raise RuntimeError(f"第 {i + 1} 段文件缺失")
        size = os.path.getsize(p)
        if not (want <= size <= want + (1 << 16)):
            raise RuntimeError(f"第 {i + 1} 段字节数异常({size},预期 {want}~{want + 65536})")
    # 按顺序拼接:每段只取应得的 want 字节,砍掉越界部分(不能重复写入)
    with open(local_path, "ab" if done > 0 else "wb") as out:
        for i in range(n):
            want = remaining - i * seg if i == n - 1 else seg
            with open(local_path + f".part{i}", "rb") as f:
                out.write(f.read(want))             # 只取本段应得字节
            os.remove(local_path + f".part{i}")     # 合并完即删临时段
    if os.path.exists(meta_path):
        os.remove(meta_path)                        # 分段存档随任务结束删除
    print(f"\n  ✅ 下载完成: {local_path}({n} 线程分段,校验通过)")
    return local_path


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


def resolve_adcode(name):
    """行政区划名 → adcode(用公开的 pcas-code 表,含省市县三级)"""
    print(f"  解析行政区划: {name}")
    r = _get_retry(ADCODE_URL)
    r.raise_for_status()
    data = json.loads(r.text)
    # 表结构:[{"code":"420000","name":"湖北省","children":[...]}, ...]
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


def _full_adcode(code):
    """pcas-code 表的省/市码是简码(42/4201),DataV 边界接口要 6 位全码"""
    if len(code) == 2:
        return code + "0000"
    if len(code) == 4:
        return code + "00"
    return code


def fetch_boundary_geojson(name, adcode="", cache_path=None):
    """行政区划名 → 边界 GeoJSON(WGS84,已纠偏)

    cache_path:本地缓存文件。边界行政数据不会变,成功过一次就存盘,
    以后重跑直接读缓存,不依赖 DataV 接口(网络抽风时也能跑)。
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            print(f"  使用本地边界缓存: {os.path.basename(cache_path)}")
            return json.load(f)
    if not adcode:
        adcode = _full_adcode(resolve_adcode(name) or "")
    if not adcode:
        print(f"❌ 找不到行政区划编码: {name}(试试写全称,如'武汉市')")
        return None
    url = BOUNDARY_URL.format(adcode=adcode)
    print(f"  获取边界: {url}")
    r = _get_retry(url)
    r.raise_for_status()
    gj = r.json()
    if gj.get("features"):
        gj = convert_geojson(gj)  # GCJ-02 → WGS84
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(gj, f, ensure_ascii=False)  # 存缓存,下次离线也能跑
        return gj
    print("❌ 边界数据为空")
    return None


def _cached_adcode(name, cache_dir):
    """从边界缓存文件名 boundary_<名称>_<adcode>.geojson 里找回 adcode

    行政区划表接口断了也能定输出文件夹(网络恢复前不阻塞下载)。
    """
    prefix = f"boundary_{name}_"
    if os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            if f.startswith(prefix) and f.endswith(".geojson"):
                adcode = f[len(prefix):-len(".geojson")]
                return adcode if adcode != "unknown" else ""
    return ""


def load_boundary_file(path):
    """用户提供的边界文件(GeoJSON)"""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    # 用户文件假定已是 WGS84(若是 GCJ-02 需自行转换,说明书写明)
    return gj


def download_file(url, dest, timeout=3600):
    """流式下载 + 进度显示(大文件友好)"""
    print(f"  下载: {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {done / 1048576:6.1f}/{total / 1048576:6.1f} MB ({pct:5.1f}%)", end="")
        print()
    return dest


def http_range_download(url, dest, threads=4):
    """HTTP Range 分段并行下载(WorldCover S3 专用;单线程 ~100KB/s,分段并行可 3~4 倍)

    每段独立分片文件,全部成功后按序合并;分片按 Range 长度截断
    (服务端多给/少给都砍齐——多线程下载的老坑,CLCD FTP 分段同款教训)。
    """
    h = requests.head(url, timeout=60)
    h.raise_for_status()
    total = int(h.headers.get("content-length", 0))
    if total <= 0:
        raise RuntimeError(f"HEAD 拿不到文件大小: {url}")
    part = (total + threads - 1) // threads
    parts = [f"{dest}.part{i}" for i in range(threads)]
    errs = []

    def worker(i):
        start = i * part
        if start >= total:      # 文件小于线程数时跳过空段
            return
        end = min((i + 1) * part, total) - 1
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                             timeout=600, stream=True)
            r.raise_for_status()
            with open(parts[i], "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
            with open(parts[i], "r+b") as f:    # 砍齐到 Range 长度,防越界字节
                f.truncate(end - start + 1)
            print(f"  段 {i + 1}/{threads} 完成")
        except Exception as e:
            errs.append((i, str(e)))
            print(f"  ❌ 段 {i + 1}/{threads} 失败: {e}")

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise RuntimeError(f"{len(errs)}/{threads} 段下载失败: {errs[:2]}")
    with open(dest, "wb") as out:
        for p in parts:
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(p)
    print(f"  ✅ 合并完成: {dest} ({os.path.getsize(dest) / 1048576:.0f} MB)")
    return dest


def worldcover_tile_ids(minx, miny, maxx, maxy):
    """边界(WGS84)→ 覆盖的 WorldCover 3° 分幅编号列表 [(lat0, lon0), ...]

    分幅规则:每幅 3°×3°,编号取西南角(N{lat0}E{lon0},中国境内全为 N/E)。
    边界恰好压在 3 的倍数上时只取界内一侧(用 while < 上界,不取贴边空幅)。
    """
    import math
    ids = []
    lat0 = int(math.floor(miny / 3.0)) * 3
    while lat0 < maxy:
        lon0 = int(math.floor(minx / 3.0)) * 3
        while lon0 < maxx:
            ids.append((lat0, lon0))
            lon0 += 3
        lat0 += 3
    return ids


def worldcover_download(year, geojson, out_dir, threads=4, dry_run=False,
                        ref_grid=None, res=30):
    """ESA WorldCover 某年下载:分幅 tile → 逐幅投影 → 裁切掩膜

    输出 <out_dir>/lulc_<year>.tif(与 CLCD 同命名,下游全链无感知)。
    res=30:重采样 30m(用户 2026-11-10 决策,30m 渠道的降级数据源);
    res=10:保持 10m 原生(10m 渠道)。重投影到 CLCD 同款 Albers 等积投影
    (中央经线 105/标准纬线 25,47),类别数据用 mode(众数)重采样。
    ref_grid:第一年裁切后返回 (transform, w, h),第二年传入以保证两期逐像素对齐。
    返回 ((h, w, crs 文本), ref_grid)。
    """
    from rasterio.warp import reproject, Resampling, transform_bounds
    from rasterio.crs import CRS

    ver = {2020: "v100", 2021: "v200"}.get(year)
    if not ver:
        raise RuntimeError(f"WorldCover 只有 2020(v100)/2021(v200)两期,不支持 {year}")
    minx, miny, maxx, maxy = bounds_wgs84_geom(geojson)
    tiles = worldcover_tile_ids(minx, miny, maxx, maxy)
    print(f"  覆盖分幅({len(tiles)} 幅,3°×3°): {tiles}")
    raw_dir = os.path.join(out_dir, "worldcover_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # 1) 下载分幅 tile(已下载过且大小正常则复用)
    paths = []
    for lat0, lon0 in tiles:
        url = WORLDCOVER_TILE_URL.format(ver=ver, year=year, lat=lat0, lon=lon0)
        dest = os.path.join(raw_dir, f"wc_{year}_{ver}_N{lat0:02d}E{lon0:03d}.tif")
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000_000:
            print(f"  复用已下载 tile: {os.path.basename(dest)}")
            paths.append(dest)
            continue
        if dry_run:
            r = requests.head(url, timeout=60)
            size = int(r.headers.get("content-length", 0))
            print(f"  干跑: {os.path.basename(dest)} ← HTTP {r.status_code}, {size / 1048576:.0f} MB")
            paths.append(dest)
            continue
        print(f"  下载: {os.path.basename(dest)}")
        http_range_download(url, dest, threads=threads)
        paths.append(dest)
    if dry_run:
        return None, None

    # 2) 目标网格:按研究区边界算 Albers 网格 + 每边 100 像元余量(不需要覆盖整幅 tile,
    #    目标数组只要研究区大小,计算量小一个数量级);第二年用第一年的 ref_grid 强制对齐
    dst_crs = CRS.from_dict({"proj": "aea", "lat_1": 25, "lat_2": 47, "lat_0": 0,
                             "lon_0": 105, "x_0": 0, "y_0": 0, "datum": "WGS84", "units": "m"})
    if ref_grid is None:
        import math
        bx0, by0, bx1, by1 = transform_bounds("EPSG:4326", dst_crs, minx, miny, maxx, maxy)
        pad = 100 * res      # 每边 100 像元余量,防边界贴边缺数据
        dst_transform = rasterio.Affine(res, 0, bx0 - pad, 0, -res, by1 + pad)
        dst_w = int(math.ceil((bx1 - bx0 + 2 * pad) / res))
        dst_h = int(math.ceil((by1 - by0 + 2 * pad) / res))
    else:
        dst_transform, dst_w, dst_h = ref_grid

    # 3) 逐幅重投影到目标网格(类别数据用 mode 众数)。分幅间不重叠,但 rasterio.reproject
    #    对 ndarray destination 是"全量覆盖写"(界外区域写 0)——对同一数组连续调用会把
    #    先前幅的结果冲掉(2026-11-10 郑州真下载实测抓出:只剩最后一幅的数据)。
    #    所以每幅投影到独立数组,再用"非 0 才覆盖"合并(0 是背景哨兵,类别编码 10~100)。
    full = np.zeros((dst_h, dst_w), dtype="uint8")
    for p in paths:
        with rasterio.open(p) as s:
            part = np.zeros((dst_h, dst_w), dtype="uint8")
            reproject(rasterio.band(s, 1), part,
                      src_transform=s.transform, src_crs=s.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=Resampling.mode)
        before = int((full != 0).sum())
        full = np.where(part != 0, part, full)
        print(f"  已投影: {os.path.basename(p)}(本幅新增有效像元 "
              f"{int((full != 0).sum()) - before:,})")
    # 写出全幅中间件(供裁切步骤读窗口;裁完删除)
    full_tif = os.path.join(raw_dir, f"wc_{year}_{ver}_aea{res}_full.tif")
    with rasterio.open(full_tif, "w", driver="GTiff", height=dst_h, width=dst_w,
                       count=1, dtype="uint8", crs=dst_crs,
                       transform=dst_transform, compress="deflate") as dst:
        dst.write(full, 1)

    # 4) 按边界裁切 + 掩膜(复用统一网格裁切:窗口与输出网格按全幅网格算,
    #    数据读窗口,边界外置 0——与 CLCD 各期对齐口径一致)
    out_path = os.path.join(out_dir, f"lulc_{year}.tif")
    (h, w), crs = clip_to_boundary(full_tif, geojson, out_path,
                                   ref_grid=(dst_transform, dst_w, dst_h))
    os.remove(full_tif)   # 中间件即用即删,不留垃圾
    print(f"  ✅ WorldCover {year} 裁切完成: {out_path} ({h}×{w}, {res}m Albers)")
    return (h, w, str(crs)), (dst_transform, dst_w, dst_h)


def clip_to_boundary(tif_path, geojson, out_path, ref_grid=None):
    """按边界裁切(保持原始投影/分辨率,不重采样)

    ref_grid:(transform, width, height) 全国网格参照。分省包省图与全国图存在
    亚像元错位(2026-11-05 审计实测:武汉 2005 省图与全国图差 ~2-6m,导致各期
    边缘错位、变化检测类指标被伪变化毒害)。给参照时:窗口与输出网格按全国
    网格计算,数据从源图对应窗口读取(最近邻快照到标准网格,保证各期逐像素一致)。
    """
    with rasterio.open(tif_path) as src:
        geom = [f["geometry"] for f in geojson["features"]]
        if src.crs is not None and not src.crs.is_geographic:
            from rasterio.warp import transform_geom
            geom = [transform_geom("EPSG:4326", src.crs, g) for g in geom]
        if ref_grid is not None:
            # 统一网格裁切:窗口按全国网格算。源图与全国网格相同时直接读窗口;
            # 不同时(分省包省图,原点相差可达数百公里,索引不能直接套用)→
            # 用重采样把源数据"快照"到全国网格(最近邻,同一投影仅平移微调)。
            from rasterio.windows import from_bounds, Window
            from rasterio.warp import transform_bounds, reproject, Resampling
            ref_t, ref_w, ref_h = ref_grid
            x0, y0, x1, y1 = bounds_wgs84_geom(geojson)
            bx = transform_bounds("EPSG:4326", src.crs, x0, y0, x1, y1)
            win = from_bounds(*bx, ref_t)
            win = win.round_offsets().round_lengths(op="ceil").intersection(
                Window(0, 0, ref_w, ref_h))
            # 窗口变换 = 参照网格 × 平移(col_off, row_off)
            # 注意不能写 ref_t * (col, row):Affine 与 2 元组相乘返回的是坐标点
            out_transform = ref_t * rasterio.Affine.translation(win.col_off, win.row_off)
            if src.transform == ref_t and src.width == ref_w and src.height == ref_h:
                out_img = src.read(1, window=win)
            else:
                out_img = np.zeros((win.height, win.width), dtype=src.dtypes[0])
                reproject(source=rasterio.band(src, 1), destination=out_img,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=out_transform, dst_crs=src.crs,
                          resampling=Resampling.nearest, dst_nodata=0)
            if out_img.ndim == 3:  # 部分读取路径返回 (1, h, w),归一化到 2D
                out_img = out_img[0]
            h, w = out_img.shape
            # 边界掩膜:窗口矩形会带进周边行政区,必须按边界多边形把界外置 0
            # (与旧版 rio_mask 的口径一致;2026-11-05 审计发现漏掩膜导致地类数虚高)
            from rasterio.features import rasterize as rio_rasterize
            mask = rio_rasterize(geom, out_shape=(h, w), transform=out_transform,
                                 fill=0, default_value=1)
            out_img = np.where(mask == 1, out_img, 0)
        else:
            out_img, out_transform = rio_mask(src, geom, crop=True, all_touched=True)
            if out_img.ndim == 3:
                out_img = out_img[0]   # rio_mask 返回 (1,H,W),归一化到 2D(WB 固原实测必现 bug)
            h, w = out_img.shape
        meta = src.meta.copy()
        meta.update(height=h, width=w,
                    transform=out_transform, compress="deflate")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(out_img, 1)  # 显式带波段索引(2D 数组不带索引在部分路径报维度错)
    return (h, w), str(src.crs)


def bounds_wgs84_geom(geojson):
    """GeoJSON(WGS84)→ (西, 南, 东, 北)"""
    xs, ys = [], []

    def walk(g):
        t = g.get("type")
        if t == "Polygon":
            for ring in g["coordinates"]:
                for x, y in ring:
                    xs.append(x), ys.append(y)
        elif t == "MultiPolygon":
            for poly in g["coordinates"]:
                for ring in poly:
                    for x, y in ring:
                        xs.append(x), ys.append(y)

    for f in geojson["features"]:
        walk(f["geometry"])
    return min(xs), min(ys), max(xs), max(ys)


def write_report(out_dir, region, years, infos, source="ncdc-ftp"):
    src_desc = ("ESA WorldCover(10m 全球产品,v100=2020 / v200=2021,已重采样 30m;"
                "官方 esa-worldcover.org,许可 CC BY 4.0)"
                if source == "worldcover"
                else "CLCD(30m,年度,武汉大学/官方发布)")
    lines = [f"# 土地利用数据下载报告", "",
             f"- 研究区: {region}", f"- 数据源: {src_desc}",
             f"- 年份: {', '.join(str(y) for y in years)}", "",
             "## 输出文件", ""]
    for y in years:
        p = os.path.join(out_dir, f"lulc_{y}.tif")
        if os.path.exists(p):
            h, w, crs = infos.get(y, ("?", "?", "?"))
            lines.append(f"- lulc_{y}.tif : {h}×{w}, {crs}")
        else:
            lines.append(f"- lulc_{y}.tif : ❌ 下载失败")
    rem = ("- WorldCover 原始编码(10/20/30/.../100 共 11 类)原样保留,"
           "语义重分类由后续重分类步骤按内置图例处理"
           if source == "worldcover"
           else "- CLCD 编码(1~9 类)原样保留,重分类到研究区体系需自行处理")
    lines += ["", "## 提醒", "",
              "- 输出未做行列/投影对齐(与驱动因子不一致时请先统一,可用 plus-data-check 检查)",
              rem,
              "- 边界已从 GCJ-02 纠偏为 WGS84"]
    if source == "worldcover":
        lines += ["- WorldCover 10m → 30m 重采样:重投影到 Albers 等积投影"
                  "(中央经线 105,标准纬线 25/47),类别重采样用众数(mode)",
                  "- 多年份网格已强制对齐(第二年套第一年网格),两期逐像素一致"]
    with open(os.path.join(out_dir, "下载报告.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="土地利用图下载(CLCD):行政区/边界 → 多年 → 裁切")
    parser.add_argument("--region", required=True,
                        help="行政区划名称(如 武汉市)或 GeoJSON 边界文件路径")
    parser.add_argument("--years", required=True, help="逗号分隔年份,如 2010,2020")
    parser.add_argument("--output", default=None, help="输出目录(默认 D:\\plus_skill_GIS_data\\<研究区名>)")
    parser.add_argument("--url", default=None, help="自定义 CLCD 下载 URL 模板(含 {year} 占位符)")
    parser.add_argument("--manual", action="store_true", help="手动下载兜底模式")
    parser.add_argument("--manual-files", default=None,
                        help="手动模式:逗号分隔 文件路径:年份,如 D:\\a.tif:2010,D:\\b.tif:2020")
    parser.add_argument("--source", default="ncdc-ftp",
                        choices=["ncdc-ftp", "http", "worldcover"],
                        help="下载渠道:默认 ncdc-ftp(CLCD 国内官方镜像,账号在 ftp_config.json);"
                             "worldcover=ESA WorldCover 10m(仅 2020/2021)")
    parser.add_argument("--res", type=int, default=30, choices=[30, 10],
                        help="WorldCover 输出分辨率:30(重采样 30m,默认)/ 10(保持 10m 原生)")
    parser.add_argument("--threads", type=int, default=4,
                        help="FTP 并行分段数,默认 4(1=单线程;服务器限速时多线程能快数倍)")
    parser.add_argument("--dry-run", action="store_true", help="只解析边界/校验渠道,不下载")
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    if not years:
        print("❌ --years 至少给一个年份")
        sys.exit(1)
    if args.source == "worldcover":
        bad = [y for y in years if y not in (2020, 2021)]
        if bad:
            print(f"❌ WorldCover 只有 2020(v100)/2021(v200)两期,不支持年份 {bad}")
            sys.exit(1)

    # 输出文件夹名必须纯 ASCII(PLUS 下游对路径编码敏感):中文名改用 adcode
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    if os.path.isfile(args.region):
        region_name = os.path.splitext(os.path.basename(args.region))[0]
        folder = region_name if all(ord(c) < 128 for c in region_name) else "region"
        adcode = ""
    else:
        region_name = args.region
        # 先看本地缓存里有没有这个区的边界(顺带找回 adcode,断网也能定文件夹)
        os.makedirs(cache_dir, exist_ok=True)
        adcode = _cached_adcode(region_name, cache_dir)
        if not adcode:
            try:
                adcode = _full_adcode(resolve_adcode(region_name) or "")
            except Exception as e:
                print(f"  ⚠️ 行政区划表获取失败({e})")
        folder = adcode if (adcode and any(ord(c) > 127 for c in region_name)) else region_name
    out_dir = args.output or os.path.join(DEFAULT_OUTPUT_ROOT, folder)
    os.makedirs(out_dir, exist_ok=True)
    print(f"研究区: {args.region}(文件夹名 {folder})  年份: {years}  输出: {out_dir}")
    if any(ord(c) > 127 for c in out_dir):
        print("⚠️ 输出路径含非英文字符,PLUS 下游可能崩溃,建议 --output 指定纯英文目录")

    # 1) 定位研究区
    if os.path.isfile(args.region):
        geojson = load_boundary_file(args.region)
        print(f"✅ 边界文件已加载(假定 WGS84): {args.region}")
    else:
        cache_path = os.path.join(cache_dir, f"boundary_{region_name}_{adcode or 'unknown'}.geojson")
        geojson = fetch_boundary_geojson(region_name, adcode, cache_path)
        if geojson is None:
            sys.exit(1)
        print(f"✅ 边界已获取并纠偏(WGS84),要素 {len(geojson.get('features', []))} 个")

    template = args.url or CLCD_URL_TEMPLATE

    # 全国网格参照:分省包省图与全国图存在亚像元错位(2026-11-05 审计实测
    # 武汉 2005 省图与全国图差 ~2-6m),裁切统一按全国网格快照,保证各期逐像素对齐。
    # 仅 CLCD 渠道使用(WorldCover 有自己的 30m 重投影网格,见 worldcover_download)
    ref_grid = None
    wc_grid = None   # WorldCover 第一年的 30m 网格,第二年强制对齐用
    for f in os.listdir(out_dir) if args.source != "worldcover" else []:
        if f.startswith("CLCD_") and f.endswith("_raw.tif"):
            try:
                with rasterio.open(os.path.join(out_dir, f)) as s:
                    if s.width > 100000:  # 全国图(16 万列量级)
                        ref_grid = (s.transform, s.width, s.height)
                        print(f"  全国网格参照: {f}({s.width}×{s.height})")
                        break
            except Exception:
                pass

    # 2) 下载 + 裁切(或手动/干跑)
    infos = {}
    for y in years:
        print(f"\n=== {y} 年 ===")
        tif = None
        if args.manual:
            # 手动兜底:用户给 文件路径:年份 列表,建"年份→路径"映射
            # 注意用 rsplit:Windows 路径盘符带冒号(D:\...),只能从最后一个冒号切
            mapping = {}
            for kv in (args.manual_files or "").split(","):
                if ":" in kv:
                    path, year = kv.rsplit(":", 1)
                    mapping[year.strip()] = path.strip()
            tif = mapping.get(str(y))
            if not tif or not os.path.exists(tif):
                print(f"  ❌ 手动模式下 {y} 年没有给文件路径")
                continue
            print(f"  使用手动文件: {tif}")
        else:
            if args.source == "worldcover":
                # ESA WorldCover:下载分幅 → 合并 → 重投影 30m → 裁切,一步完成;
                # 第二年用第一年的网格强制对齐(ref_grid 传 wc_grid),保证两期逐像素一致
                try:
                    info, wc_grid = worldcover_download(
                        y, geojson, out_dir, args.threads,
                        dry_run=args.dry_run,
                        ref_grid=wc_grid if wc_grid is not None else None,
                        res=args.res)
                    if args.dry_run:
                        continue
                    infos[y] = info
                except Exception as e:
                    print(f"  ❌ WorldCover {y} 年失败: {e}")
                continue
            if args.source == "ncdc-ftp":
                # 主渠道:NCDC FTP(国内官方镜像,账号配置一次性)
                cfg = load_ftp_config()
                remote = cfg["file_template"].format(year=y)
                if args.dry_run:
                    ftp = ftplib.FTP_TLS()
                    ftp.connect(cfg["host"], cfg.get("port", 2121), timeout=30)
                    ftp.login(cfg["username"], cfg["password"])
                    ftp.prot_p()
                    ftp.voidcmd("TYPE I")
                    print(f"  干跑:远端存在 {remote},大小 {ftp.size(remote)/1048576:.0f} MB")
                    ftp.quit()
                    continue
                dest = os.path.join(out_dir, f"CLCD_{y}_raw.tif")
                try:
                    # 先试多线程分段(默认 4 线程);服务器拒绝多连接时自动回退单线程续传
                    if args.threads <= 1:
                        tif = ftp_download(remote, dest, cfg)
                    else:
                        try:
                            tif = ftp_download_parallel(remote, dest, cfg, threads=args.threads)
                        except Exception as e:
                            print(f"  ⚠️ 多线程下载失败({e}),回退单线程续传(断点不丢)")
                            tif = ftp_download(remote, dest, cfg)
                except ftplib.error_perm as e:
                    # 兜底:全国图缺失的年份(实测 2005 在 NCDC 镜像上没有全国图,
                    # 只有分省包)→ 下载分省 zip,解压出研究区所在省
                    print(f"  ⚠️ 全国图不可得({e}),改用分省包兜底")
                    tif = ftp_province_zip_fallback(y, cfg, adcode, out_dir)
                    if tif is None:
                        continue
                except Exception as e:
                    print(f"  ❌ FTP 下载失败: {e}")
                    print(f"  兜底:手动下载 {y} 年数据后,用 --manual --manual-files \"<路径>:{y}\" 重跑")
                    continue
            else:
                # 备用渠道:HTTP URL 模板(默认 Zenodo,国内常不可达)
                url = template.format(year=y)
                if args.dry_run:
                    try:
                        r = requests.head(url, timeout=60, allow_redirects=True)
                        print(f"  干跑:URL 可访问({r.status_code}) → {url[:80]}...")
                    except Exception as e:
                        print(f"  ⚠️ 干跑:URL 访问失败({e}) → 建议 --source ncdc-ftp")
                    continue
                dest = os.path.join(out_dir, f"CLCD_{y}_raw.tif")
                try:
                    tif = download_file(url, dest)
                except Exception as e:
                    print(f"  ❌ 下载失败: {e}")
                    print(f"  兜底:手动下载 {y} 年数据后,用 --manual --manual-files \"<路径>:{y}\" 重跑")
                    continue

        # 3) 裁切
        try:
            out_path = os.path.join(out_dir, f"lulc_{y}.tif")
            (h, w), crs = clip_to_boundary(tif, geojson, out_path, ref_grid)
            infos[y] = (h, w, crs)
            print(f"  ✅ 裁切完成: {out_path} ({h}×{w}, {crs})")
        except Exception as e:
            print(f"  ❌ 裁切失败: {e}")

    # 4) 报告(干跑只验证渠道与分幅,不写报告、不判失败)
    if not args.dry_run:
        write_report(out_dir, region_name, years, infos, source=args.source)
        print(f"\n✅ 下载报告已保存: {os.path.join(out_dir, '下载报告.md')}")

        # 任一年份失败 → 退出码 1(让调用方/AI 能直接判断成败,不必解析报告文件)
        missing = [y for y in years if y not in infos]
        if missing:
            print(f"⚠️ 以下年份未成功: {missing}")
            sys.exit(1)
    else:
        print("\n(干跑模式:仅验证渠道与分幅,未实际下载)")


if __name__ == "__main__":
    main()
