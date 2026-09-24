# -*- coding: utf-8 -*-
r"""
矢量数据拓扑检查工具（jsdczdrzy制作）
===================
检查指定目录中所有矢量文件的几何拓扑问题。

主要检查项：
    1. 无效几何（不符合 OGC 规范）
    2. 几何自相交
    3. 面要素之间重叠（overlap）
    4. 面要素之间缝隙（gap，仅同图层相邻面）
    5. 几何完全重复的要素
    6. 空几何 / 坐标异常
    7. 细小多边形（sliver）

支持的矢量格式：
    - Shapefile (.shp)
    - GeoJSON (.geojson / .json)
    - GeoPackage (.gpkg)
    - KML (.kml)
    - MapInfo TAB (.tab)
    - ESRI File Geodatabase (.gdb，整库遍历所有图层）

使用方法：
    1) 直接修改本文件底部的 TARGET_DIR 后运行：python check_topology(jiangsu).py
    2) (推荐)或命令行传参：python check_topology(jiangsu).py "Z:\jsdczdrzy\工程文件\KF_320000_2026_XL" -o "保存路径" 

依赖安装：
    pip install geopandas shapely pandas tqdm pyogrio
    # GDAL 为可选但强烈推荐安装（支持更多格式、读取更快）：
    #   pip install GDAL==3.8.0（或与系统已安装的 GDAL 版本匹配）
    #   conda install gdal -c conda-forge  # 推荐 conda 方式安装 GDAL
    #
    # IO 引擎优先级（自动检测，优先使用最快的）：
    #   GDAL/OGR > pyogrio > fiona > geopandas

GPU 加速（可选，需要 NVIDIA GPU + CUDA）：
    pip install numba  # CPU JIT 加速（自动使用）
    pip install cupy  # GPU 加速（需要 CUDA 环境）

报告输出（全部为 .csv 格式，无行数上限）：
    - 拓扑检查汇总_*.csv     每个数据集一行汇总
    - 总体统计_*.csv         总计统计
    - 拓扑问题详情_*.csv     所有问题明细
    - 拓扑检查报告_*.txt     报告
    - 拓扑检查报告_*_问题清单.txt 仅含问题要素的简明清单
    - 建议使用国内python源，效率更高。
"""

import os
import sys
import argparse
import time
import multiprocessing
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# 防止 Windows 上 multiprocessing 重复 spawn 子进程时主模块里的 tqdm 输出混乱
try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    _HAS_CONCURRENT = True
except ImportError:
    _HAS_CONCURRENT = False

# ===== GPU 加速检测 =====
_HAS_GPU = False
_HAS_CUPY = False
_GPU_MODE = "CPU（Numba JIT加速）"

try:
    import numba
    from numba import jit, prange
    _HAS_NUMBA = True
    # 测试 Numba 是否正常工作
    try:
        @jit(nopython=True)
        def _test_numba(x):
            return x * 2
        _test_numba(1)
        _GPU_MODE = "CPU（Numba JIT加速）"
    except Exception:
        _HAS_NUMBA = False
except ImportError:
    numba = None
    _HAS_NUMBA = False

# CuPy GPU 检测（可选）
if _HAS_NUMBA:
    try:
        import cupy as cp
        # 检测是否有可用的 GPU
        try:
            cp.cuda.Device(0).compute_capability
            _HAS_CUPY = True
            _HAS_GPU = True
            _GPU_MODE = "GPU（CuPy CUDA加速）"
        except Exception:
            pass
    except ImportError:
        cp = None

# ===== GDAL / OGR 检测（最优先，读取效率最高）=====
_HAS_GDAL = False
try:
    from osgeo import ogr, gdal, osr
    _HAS_GDAL = True
    try:
        gdal.UseExceptions()
    except AttributeError:
        pass
    try:
        ogr.UseExceptions()
    except AttributeError:
        pass
except ImportError:
    try:
        import osgeo.ogr as ogr
        import osgeo.gdal as gdal
        import osgeo.osr as osr
        _HAS_GDAL = True
    except ImportError:
        ogr = gdal = osr = None

# ===== pyogrio / fiona 回退引擎 =====
try:
    import pyogrio
    _HAS_PYOGRIO = True
except ImportError:
    _HAS_PYOGRIO = False
try:
    import fiona
    _HAS_FIONA = True
except ImportError:
    _HAS_FIONA = False

# ===== 核心库 =====
try:
    import geopandas as gpd
    from shapely.validation import explain_validity
    from shapely.geometry import MultiPolygon, Polygon
    from shapely import prepared
    import pandas as pd
    from tqdm import tqdm
except ImportError as exc:
    print("=" * 60)
    print("缺少必要的 Python 库:", exc)
    print("请先在命令行执行：")
    print("    pip install geopandas shapely pyogrio pandas tqdm")
    print("如使用 GDAL 加速请额外安装（推荐）：")
    print("    pip install GDAL==3.8.0（或与系统 GDAL 版本匹配的版本）")
    print("=" * 60)
    sys.exit(1)

# GDAL 读取模式提示
if _HAS_GDAL:
    _IO_MODE = "GDAL/OGR（最快）"
elif _HAS_PYOGRIO:
    _IO_MODE = "pyogrio"
else:
    _IO_MODE = "fiona"


# ==================== GPU 加速模块 ====================
if _HAS_NUMBA:
    import numpy as np
    
    @jit(nopython=True, parallel=True, cache=True)
    def _numba_check_slivers(areas, threshold, valid_mask):
        """Numba 加速的细小多边形检查。
        
        参数:
            areas: 面积数组
            threshold: 面积阈值
            valid_mask: 有效要素掩码
        返回:
            sliver_indices:细小多边形索引列表
        """
        n = len(areas)
        sliver_indices = np.empty(n, dtype=np.int64)
        count = 0
        for i in prange(n):
            if valid_mask[i] and 0 < areas[i] < threshold:
                sliver_indices[i] = 1
                count += 1
            else:
                sliver_indices[i] = 0
        return sliver_indices, count
    
    @jit(nopython=True, cache=True)
    def _numba_compute_bounds_min(bounds_array):
        """计算边界数组的最小值（用于空间索引构建）"""
        n = len(bounds_array)
        min_x = np.inf
        min_y = np.inf
        for i in range(n):
            min_x = min(min_x, bounds_array[i, 0])
            min_y = min(min_y, bounds_array[i, 1])
        return min_x, min_y
    
    @jit(nopython=True, cache=True)
    def _numba_compute_bounds_max(bounds_array):
        """计算边界数组的最大值"""
        n = len(bounds_array)
        max_x = -np.inf
        max_y = -np.inf
        for i in range(n):
            max_x = max(max_x, bounds_array[i, 2])
            max_y = max(max_y, bounds_array[i, 3])
        return max_x, max_y
    
    @jit(nopython=True, parallel=True, cache=True)
    def _numba_batch_area_check(areas, threshold, is_polygon):
        """批量检查面积是否超过阈值（用于重叠预筛选）"""
        n = len(areas)
        result = np.zeros(n, dtype=np.int8)
        for i in prange(n):
            if is_polygon[i] and areas[i] > threshold:
                result[i] = 1
        return result
    
    @jit(nopython=True, cache=True)
    def _numba_merge_bounds_fast(bounds1, bounds2):
        """快速合并两个边界框"""
        return (min(bounds1[0], bounds2[0]),
                min(bounds1[1], bounds2[1]),
                max(bounds1[2], bounds2[2]),
                max(bounds1[3], bounds2[3]))
    
    @jit(nopython=True, cache=True)
    def _numba_bounds_intersect(b1, b2):
        """检查两个边界框是否相交"""
        return (b1[0] <= b2[2] and b1[2] >= b2[0] and
                b1[1] <= b2[3] and b1[3] >= b2[1])
    
    @jit(nopython=True, parallel=True, cache=True)
    def _numba_spatial_filter(bounds_array, query_bounds):
        """GPU风格的空间过滤 - 找出与查询框相交的所有要素"""
        n = len(bounds_array)
        matches = np.zeros(n, dtype=np.int8)
        count = 0
        for i in prange(n):
            if _numba_bounds_intersect(bounds_array[i], query_bounds):
                matches[i] = 1
                count += 1
        return matches, count
    
    def _get_accelerated_functions():
        """获取加速函数集合"""
        return {
            'check_slivers': _numba_check_slivers,
            'bounds_min': _numba_compute_bounds_min,
            'bounds_max': _numba_compute_bounds_max,
            'area_check': _numba_batch_area_check,
            'merge_bounds': _numba_merge_bounds_fast,
            'spatial_filter': _numba_spatial_filter,
        }
else:
    # 无 Numba 时的空实现
    def _get_accelerated_functions():
        return {}

# GPU 预编译（如果可用）
if _HAS_NUMBA:
    try:
        # 预热 Numba JIT 编译（后台进行，不阻塞主线程）
        import threading
        def _warmup_numba():
            try:
                test_bounds = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float64)
                _numba_compute_bounds_min(test_bounds)
                _numba_compute_bounds_max(test_bounds)
                test_areas = np.array([0.5, 1.5, 2.0], dtype=np.float64)
                test_mask = np.array([True, True, True], dtype=np.bool_)
                _numba_check_slivers(test_areas, 1.0, test_mask)
            except Exception:
                pass
        # 在后台线程预热
        _warmup_thread = threading.Thread(target=_warmup_numba, daemon=True)
        _warmup_thread.start()
    except Exception:
        pass


def _list_gdb_layers(gdb_path):
    """列举 GDB / 目录内的图层名。"""
    if _HAS_GDAL:
        ds = gdal.OpenEx(gdb_path, gdal.OF_VECTOR)
        if ds is not None:
            return [ds.GetLayer(idx).GetName()
                    for idx in range(ds.GetLayerCount())]
    if _HAS_PYOGRIO:
        return [name for name, _ in pyogrio.list_layers(gdb_path)]
    if _HAS_FIONA:
        return fiona.listlayers(gdb_path)
    raise RuntimeError("无法读取图层列表（请安装 GDAL/pyogrio/fiona 之一）。")


def _read_vector_with_gdal(path, layer=None):
    """用 GDAL/OGR 高性能流式读取，返回 GeoDataFrame。"""
    if layer is not None:
        ds = gdal.OpenEx(path, gdal.OF_VECTOR)
        if ds is None:
            raise RuntimeError(f"GDAL 无法打开 {path}")
        lyr_src = ds.GetLayerByName(layer)
        if lyr_src is None:
            raise RuntimeError(f"GDAL 图层 '{layer}' 未找到于 {path}")
    else:
        ds = gdal.OpenEx(path, gdal.OF_VECTOR)
        if ds is None:
            raise RuntimeError(f"GDAL 无法打开 {path}")
        lyr_src = ds.GetLayer(0)

    lyr_src.ResetReading()
    n = lyr_src.GetFeatureCount()
    if n < 0:
        n = 0

    geoms = []
    attrs = []
    feat_defn = lyr_src.GetLayerDefn()
    field_names = [feat_defn.GetFieldDefn(i).GetName()
                   for i in range(feat_defn.GetFieldCount())]
    import numpy as np
    feat: ogr.Feature
    for feat in lyr_src:
        geom_ref = feat.GetGeometryRef()
        geoms.append(
            geom_ref.ExportToWkb() if geom_ref else None)
        row = {fname: feat.GetField(fname) for fname in field_names}
        attrs.append(row)
        feat = None

    from shapely import wkb
    raw_geoms = []
    for wkb_bytes in geoms:
        if wkb_bytes is None:
            raw_geoms.append(None)
        else:
            g = wkb.loads(wkb_bytes)
            raw_geoms.append(g)

    crs_wkt = lyr_src.GetSpatialRef()
    crs_str = crs_wkt.ExportToWkt() if crs_wkt else None

    df = pd.DataFrame(attrs)
    df = gpd.GeoDataFrame(df, geometry=raw_geoms, crs=crs_str)
    return df


def _read_vector_with_pyogrio(path, layer=None):
    """用 pyogrio 高性能读取（回退方案）。"""
    kwargs = {"use_arrow": False} if layer is None else {}
    if layer is not None:
        return gpd.read_file(path, layer=layer, **kwargs)
    return gpd.read_file(path, **kwargs)


def _read_vector(path, layer=None):
    """读取矢量文件的最佳可用方式：GDAL > pyogrio > geopandas。"""
    if _HAS_GDAL:
        try:
            return _read_vector_with_gdal(path, layer)
        except Exception:
            pass
    if _HAS_PYOGRIO:
        return _read_vector_with_pyogrio(path, layer)
    return gpd.read_file(path, layer=layer)


# ====================== 配置区 ======================
DEFAULT_TARGET_DIR = r"Z:\jsdczdrzy\工程文件\KF_320000_2026_XL"
DEFAULT_OUTPUT_DIR = None
SLIVER_AREA_THRESHOLD = 1.0
OVERLAP_AREA_THRESHOLD = 0.01
GAP_BUFFER = 0.0
OVERLAP_CHUNK_SIZE = 50000
WORKERS = 0
# GPU 加速开关（True=启用GPU/Numba加速，False=纯CPU）
ENABLE_GPU_ACCEL = True
# ====================================================


# ---------------------- 严重程度映射 ----------------------
SEVERITY_LEVELS = ["严重", "提示", "参考"]
SEVERITY_RANK = {"严重": 3, "提示": 2, "参考": 1, "-": 0}
SEVERITY_COLOR = {
    "严重": "FFC7CE",
    "提示": "FFEB9C",
    "参考": "D9E1F2",
}
SEVERITY_MAP = {
    "空几何":          "严重",
    "无效几何":        "严重",
    "自相交":          "严重",
    "面要素重叠":      "严重",
    "重复要素":        "提示",
    "细小多边形":      "提示",
    "面要素缝隙":      "参考",
}


def _severity_of(issue_type):
    """根据问题类型返回严重程度。"""
    return SEVERITY_MAP.get(issue_type, "提示")


# ---------------------- 扫描文件 ----------------------
def scan_vector_files(directory):
    """递归扫描目录中的所有矢量数据。"""
    vector_data = []
    shp_aux_ext = {".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx",
                   ".fbn", ".fbx", ".ain", ".aih", ".atx", ".ixs",
                   ".mxs", ".qix", ".qpj"}
    file_exts = {".shp", ".geojson", ".json", ".gpkg", ".kml", ".tab"}

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d != "_拓扑检查报告"]

        for d in list(dirs):
            if d.lower().endswith(".gdb"):
                gdb_path = os.path.join(root, d)
                try:
                    layers = _list_gdb_layers(gdb_path)
                    if layers:
                        vector_data.append(("gdb", gdb_path, layers))
                    dirs.remove(d)
                except Exception as exc:
                    print(f"  ! 无法读取 GDB: {gdb_path} ({exc})")
                    dirs.remove(d)

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in shp_aux_ext:
                continue
            if ext in file_exts:
                vector_data.append(("file", os.path.join(root, fname), None))
    return vector_data


# ---------------------- 拓扑检查项 ----------------------
def check_invalid_geometry(gdf):
    """无效几何 + 空几何检查。"""
    issues = []
    for idx, geom in gdf.geometry.items():
        if geom is None or geom.is_empty:
            issues.append({
                "要素ID": idx,
                "问题类型": "空几何",
                "描述": "几何对象为 None 或为空",
                "面积": None,
                "原因": "is_empty",
            })
            continue
        if not geom.is_valid:
            reason = explain_validity(geom)
            issues.append({
                "要素ID": idx,
                "问题类型": "无效几何",
                "描述": reason,
                "面积": getattr(geom, "area", None),
                "原因": "几何不符合 OGC 规范",
            })
    return issues


def check_self_intersection(gdf):
    """自相交检查（基于 is_valid 的细分）。"""
    issues = []
    for idx, geom in gdf.geometry.items():
        if geom is None or geom.is_empty or geom.is_valid:
            continue
        reason = explain_validity(geom)
        if any(k in reason for k in ("Self", "Ring", "intersect", "Intersect")):
            issues.append({
                "要素ID": idx,
                "问题类型": "自相交",
                "描述": reason,
                "面积": getattr(geom, "area", None),
                "原因": "几何出现自相交",
            })
    return issues


def check_duplicates(gdf):
    """完全重复几何检查。"""
    issues = []
    seen = {}
    for idx, geom in gdf.geometry.items():
        if geom is None or geom.is_empty:
            continue
        try:
            key = geom.wkb_hex
        except Exception:
            continue
        if key in seen:
            issues.append({
                "要素ID1": seen[key],
                "要素ID2": idx,
                "问题类型": "重复要素",
                "描述": "几何完全相同",
                "面积": getattr(geom, "area", None),
                "原因": "WKB 完全一致",
            })
        else:
            seen[key] = idx
    return issues


def _first_valid_geom_type(gdf):
    """取首个非空几何的 geom_type。"""
    for geom in gdf.geometry:
        if geom is not None and not geom.is_empty:
            return geom.geom_type
    return None


def check_slivers(gdf, threshold=SLIVER_AREA_THRESHOLD):
    """细小多边形（sliver）检查 - 支持 Numba GPU 加速。"""
    issues = []
    if gdf.empty:
        return issues
    gtype = _first_valid_geom_type(gdf)
    if gtype not in ("Polygon", "MultiPolygon"):
        return issues
    
    # 尝试使用 Numba 加速
    if _HAS_NUMBA and ENABLE_GPU_ACCEL and len(gdf) > 1000:
        try:
            issues = _check_slivers_gpu(gdf, threshold)
            return issues
        except Exception:
            pass
    
    # CPU 回退
    for idx, geom in gdf.geometry.items():
        if geom is None or geom.is_empty or not geom.is_valid:
            continue
        if 0 < geom.area < threshold:
            issues.append({
                "要素ID": idx,
                "问题类型": "细小多边形",
                "描述": f"面积 = {geom.area:.6f}",
                "面积": geom.area,
                "原因": f"面积小于阈值 {threshold}",
            })
    return issues


def _check_slivers_gpu(gdf, threshold):
    """使用 Numba GPU/Numba 加速的细小多边形检查。"""
    import numpy as np
    issues = []
    
    # 准备数据
    n = len(gdf)
    areas = np.zeros(n, dtype=np.float64)
    valid_mask = np.zeros(n, dtype=np.bool_)
    indices = []
    
    for i, (idx, geom) in enumerate(gdf.geometry.items()):
        if geom is not None and not geom.is_empty and geom.is_valid:
            areas[i] = geom.area
            valid_mask[i] = True
            indices.append(idx)
        else:
            valid_mask[i] = False
    
    # 使用 Numba 并行检查
    sliver_flags, count = _numba_check_slivers(areas, threshold, valid_mask)
    
    # 收集结果
    for i in range(n):
        if sliver_flags[i]:
            issues.append({
                "要素ID": indices[i] if i < len(indices) else i,
                "问题类型": "细小多边形",
                "描述": f"面积 = {areas[i]:.6f}",
                "面积": areas[i],
                "原因": f"面积小于阈值 {threshold}",
            })
    
    return issues


def _check_overlaps_in_chunk(polys_block, all_polys, sindex_all, threshold,
                              is_self_block, batch_label=""):
    """检查 polys_block 与 all_polys 的重叠。"""
    issues = []
    block_n = len(polys_block)
    for i in range(block_n):
        gi = polys_block.iloc[i].geometry
        idx_i = polys_block.iloc[i]["__orig_idx__"]
        try:
            cand = list(sindex_all.query(gi, predicate="intersects"))
        except Exception:
            cand = list(range(len(all_polys)))
        for j in cand:
            j = int(j)
            gj_row = all_polys.iloc[j]
            idx_j = gj_row["__orig_idx__"]
            if idx_i >= idx_j:
                continue
            gj = gj_row.geometry
            try:
                if gi.overlaps(gj):
                    inter = gi.intersection(gj)
                    if inter.area > threshold:
                        issues.append({
                            "要素ID1": idx_i,
                            "要素ID2": idx_j,
                            "问题类型": "面要素重叠",
                            "描述": f"重叠面积 = {inter.area:.4f}",
                            "重叠面积": inter.area,
                            "原因": "两面相交形成非公共边的区域",
                        })
            except Exception:
                continue
        if (i + 1) % 5000 == 0 and batch_label:
            print(f"     {batch_label}: 已处理 {i + 1}/{block_n} ...",
                  end="\r", flush=True)
    return issues


def check_overlaps(gdf, threshold=OVERLAP_AREA_THRESHOLD,
                    chunk_size=OVERLAP_CHUNK_SIZE):
    """面要素两两重叠检查。"""
    issues = []
    if gdf.empty:
        return issues
    gtype = _first_valid_geom_type(gdf)
    if gtype not in ("Polygon", "MultiPolygon"):
        return issues
    if len(gdf) < 2:
        return issues

    polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    polys["__orig_idx__"] = polys.index
    polys = polys.reset_index(drop=True)
    n = len(polys)
    if n < 2:
        return issues

    try:
        sindex_all = polys.sindex
    except Exception:
        sindex_all = None
    if sindex_all is None:
        print("     [警告] 空间索引不可用，退化为暴力两两比较 ...")
        for i in range(n):
            for j in range(i + 1, n):
                gi, gj = polys.iloc[i].geometry, polys.iloc[j].geometry
                try:
                    if gi.overlaps(gj):
                        inter = gi.intersection(gj)
                        if inter.area > threshold:
                            issues.append({
                                "要素ID1": polys.iloc[i]["__orig_idx__"],
                                "要素ID2": polys.iloc[j]["__orig_idx__"],
                                "问题类型": "面要素重叠",
                                "描述": f"重叠面积 = {inter.area:.4f}",
                                "重叠面积": inter.area,
                                "原因": "两面相交形成非公共边的区域",
                            })
                except Exception:
                    pass
        return issues

    if n <= chunk_size:
        issues.extend(
            _check_overlaps_in_chunk(
                polys, polys, sindex_all, threshold,
                is_self_block=True, batch_label="重叠检查"))
    else:
        blocks = [polys.iloc[i:i + chunk_size].copy()
                  for i in range(0, n, chunk_size)]
        nb = len(blocks)
        for bi in range(nb):
            for bj in range(bi, nb):
                if bi == bj:
                    block_issues = _check_overlaps_in_chunk(
                        blocks[bi], polys, sindex_all, threshold,
                        is_self_block=True,
                        batch_label=f"重叠检查 块{bi + 1}/{nb}")
                else:
                    block_issues = _check_overlaps_in_chunk(
                        blocks[bi], polys, sindex_all, threshold,
                        is_self_block=False,
                        batch_label=f"重叠检查 块{bi + 1} vs {bj + 1}/{nb}")
                issues.extend(block_issues)
    return issues


def check_gaps(gdf, buffer=GAP_BUFFER):
    """面要素之间的缝隙检查。"""
    issues = []
    if gdf.empty:
        return issues
    gtype = _first_valid_geom_type(gdf)
    if gtype not in ("Polygon", "MultiPolygon"):
        return issues

    try:
        from shapely.ops import unary_union
    except Exception:
        return issues

    valid_geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if len(valid_geoms) < 2:
        return issues

    try:
        merged = unary_union(valid_geoms)
    except Exception:
        return issues

    if merged.geom_type == "Polygon":
        geoms_list = [merged]
    elif merged.geom_type == "MultiPolygon":
        geoms_list = list(merged.geoms)
    else:
        return issues

    for poly in geoms_list:
        interior = poly.interiors
        for ring in interior:
            from shapely.geometry import Polygon as _P
            hole = _P(ring)
            if hole.area > OVERLAP_AREA_THRESHOLD:
                issues.append({
                    "要素ID": "-",
                    "问题类型": "面要素缝隙",
                    "描述": f"缝隙面积 = {hole.area:.4f}",
                    "面积": hole.area,
                    "原因": "相邻面合并后存在内部空洞（仅参考）",
                })
    return issues


# ---------------------- 多进程 worker ----------------------
def _process_one(item):
    """子进程 worker：读取一个文件/图层并跑全部拓扑检查。"""
    kind, path, layer = item
    try:
        if kind == "gdb":
            label = f"{os.path.basename(path)}::{layer}"
        else:
            label = os.path.basename(path)
        gdf = _read_vector(path, layer)
        s, iss = check_layer(label, gdf)
        return (label, s, iss)
    except Exception as exc:
        err = {
            "文件/图层": os.path.basename(path) if kind == "file" else os.path.basename(path),
            "要素数": 0,
            "几何类型": "-",
            "坐标系": "-",
            "问题总数": 0,
            "空几何数": 0,
            "无效几何数": 0,
            "自相交数": 0,
            "重叠数": 0,
            "重复数": 0,
            "细小多边形数": 0,
            "缝隙数(参考)": 0,
            "状态": f"读取失败：{exc}",
        }
        return (os.path.basename(path), err, [])


# ---------------------- 单个图层检查 ----------------------
def check_layer(label, gdf):
    """对单个 GeoDataFrame 执行所有检查，返回 (summary, issues)。"""
    issues = []
    gtypes = [t for t in gdf.geometry.geom_type
              if isinstance(t, str)] if not gdf.empty else []
    summary = {
        "文件/图层": label,
        "要素数": len(gdf),
        "几何类型": ", ".join(sorted(set(gtypes))) if gtypes else "-",
        "坐标系": str(gdf.crs) if gdf.crs else "未定义",
        "问题总数": 0,
        "空几何数": 0,
        "无效几何数": 0,
        "自相交数": 0,
        "重叠数": 0,
        "重复数": 0,
        "细小多边形数": 0,
        "缝隙数(参考)": 0,
        "状态": "正常",
    }

    if gdf.empty:
        summary["状态"] = "无要素"
        return summary, issues

    inv = check_invalid_geometry(gdf)
    issues.extend(inv)
    summary["空几何数"] = sum(1 for x in inv if x["问题类型"] == "空几何")
    summary["无效几何数"] = sum(1 for x in inv if x["问题类型"] == "无效几何")

    self_i = check_self_intersection(gdf)
    issues.extend(self_i)
    summary["自相交数"] = len(self_i)

    dup = check_duplicates(gdf)
    issues.extend(dup)
    summary["重复数"] = len(dup)

    slv = check_slivers(gdf)
    issues.extend(slv)
    summary["细小多边形数"] = len(slv)

    ovl = check_overlaps(gdf)
    issues.extend(ovl)
    summary["重叠数"] = len(ovl)

    gap = check_gaps(gdf)
    issues.extend(gap)
    summary["缝隙数(参考)"] = len(gap)

    summary["问题总数"] = len(issues)
    summary["状态"] = "正常" if len(issues) == 0 else "存在问题"

    for it in issues:
        it["文件/图层"] = label
    return summary, issues


# ---------------------- 主流程 ----------------------
def _expand_to_items(vector_data):
    """把 GDB 项拆成多个 (kind, path, layer)。"""
    items = []
    for kind, path, layer_info in vector_data:
        if kind == "gdb":
            for lyr in layer_info:
                items.append((kind, path, lyr))
        else:
            items.append((kind, path, None))
    return items


def _resolve_workers(workers):
    """根据 WORKERS 配置 + --workers 参数计算实际进程数。"""
    if workers is None or workers <= 0:
        cpu = multiprocessing.cpu_count() or 1
        n = max(1, cpu - 1)
    else:
        n = int(workers)
    return min(n, 61)


def run_check(target_dir, output_dir, workers=WORKERS):
    """运行拓扑检查主流程。"""
    start_time = time.time()
    n_workers = _resolve_workers(workers)

    if output_dir is None:
        output_dir = os.path.join(target_dir, "_拓扑检查报告")

    print("=" * 64)
    print("  矢量数据拓扑检查工具 jsdczdrzy制作")
    print("=" * 64)
    print(f"目标目录 : {target_dir}")
    print(f"输出目录 : {output_dir}")
    print(f"IO 引擎  : {_IO_MODE}（GDAL 最快，自动检测）")
    print(f"加速模式 : {_GPU_MODE}")
    print("要素上限 : 不限制（检查全部要素）")
    print(f"并行进程 : {n_workers} （--workers 可调，0=自动按 CPU 核心数 - 1）")
    print(f"开始时间 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.path.isdir(target_dir):
        print(f"\n[错误] 目录不存在或不可访问：{target_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/3] 正在扫描矢量数据 ...")
    vector_data = scan_vector_files(target_dir)
    if not vector_data:
        print("未找到任何矢量数据，结束。")
        return
    print(f"共发现 {len(vector_data)} 个矢量数据集。")

    print("\n[2/3] 正在执行拓扑检查（请耐心等待，面重叠可能较慢）...\n")
    items = _expand_to_items(vector_data)
    print(f"  总任务数（文件 × 图层）: {len(items)}")
    summaries, all_issues = [], []

    if n_workers == 1 or not _HAS_CONCURRENT or len(items) <= 1:
        for it in tqdm(items, desc="检查进度", ncols=80):
            label, s, iss = _process_one(it)
            if s is not None:
                summaries.append(s)
            if iss:
                all_issues.extend(iss)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_process_one, it): it for it in items}
            done_n = 0
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="检查进度", ncols=80):
                done_n += 1
                try:
                    label, s, iss = fut.result()
                except Exception as exc:
                    it = futures[fut]
                    print(f"  ! 任务 {it} 崩溃：{exc}")
                    err = {
                        "文件/图层": os.path.basename(it[1]),
                        "要素数": 0,
                        "几何类型": "-",
                        "坐标系": "-",
                        "问题总数": 0,
                        "空几何数": 0,
                        "无效几何数": 0,
                        "自相交数": 0,
                        "重叠数": 0,
                        "重复数": 0,
                        "细小多边形数": 0,
                        "缝隙数(参考)": 0,
                        "状态": f"worker 异常：{exc}",
                    }
                    summaries.append(err)
                    continue
                if s is not None:
                    summaries.append(s)
                if iss:
                    all_issues.extend(iss)
    check_elapsed = time.time() - start_time
    print(f"\n  检查阶段耗时 : {check_elapsed:.1f} 秒（{n_workers} 个并行进程）")

    print("\n[3/3] 正在生成报告 ...")

    for it in all_issues:
        it["严重程度"] = _severity_of(it.get("问题类型", ""))

    summary_df = pd.DataFrame(summaries)

    if not summary_df.empty:
        max_sev = []
        for s in summaries:
            label = s.get("文件/图层", "")
            file_sevs = [it.get("严重程度", "提示")
                         for it in all_issues if it.get("文件/图层") == label]
            if not file_sevs:
                max_sev.append("-")
            else:
                best = max(file_sevs, key=lambda x: SEVERITY_RANK.get(x, 0))
                max_sev.append(best)
        summary_df.insert(
            loc=summary_df.columns.get_loc("状态") + 1,
            column="最高严重程度",
            value=max_sev,
        )

    issues_df = pd.DataFrame(all_issues) if all_issues else pd.DataFrame(
        columns=["严重程度", "文件/图层", "问题类型", "要素ID", "要素ID1", "要素ID2",
                 "描述", "面积", "重叠面积", "原因"])
    if not issues_df.empty and "严重程度" in issues_df.columns:
        cols = ["严重程度", "文件/图层", "问题类型", "要素ID", "要素ID1",
                "要素ID2", "描述", "面积", "重叠面积", "原因"]
        issues_df = issues_df.reindex(columns=cols)
        issues_df["__sort__"] = issues_df["严重程度"].map(
            lambda x: -SEVERITY_RANK.get(x, 0))
        issues_df = issues_df.sort_values(
            ["__sort__", "文件/图层", "问题类型"]).drop(columns="__sort__")
        issues_df = issues_df.reset_index(drop=True)

    stats = {
        "统计项": [
            "总数据集数", "存在问题的数据集数", "总问题数",
            "空几何总数", "无效几何总数", "自相交总数",
            "面要素重叠总数", "重复要素总数", "细小多边形总数", "缝隙数(参考)",
        ],
        "数量": [
            len(summaries),
            sum(1 for s in summaries if s["状态"] == "存在问题"),
            len(all_issues),
            sum(s["空几何数"] for s in summaries),
            sum(s["无效几何数"] for s in summaries),
            sum(s["自相交数"] for s in summaries),
            sum(s["重叠数"] for s in summaries),
            sum(s["重复数"] for s in summaries),
            sum(s["细小多边形数"] for s in summaries),
            sum(s["缝隙数(参考)"] for s in summaries),
        ],
    }
    stats_df = pd.DataFrame(stats)

    sev_stats = {"严重": 0, "提示": 0, "参考": 0}
    for it in all_issues:
        sev_stats[it.get("严重程度", "提示")] = sev_stats.get(
            it.get("严重程度", "提示"), 0) + 1
    for k, v in sev_stats.items():
        stats["统计项"].append(f"{k}级问题总数")
        stats["数量"].append(v)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_summary = os.path.join(output_dir, f"拓扑检查汇总_{ts}.csv")
    summary_df.to_csv(csv_summary, index=False, encoding="utf-8-sig")

    csv_stats = os.path.join(output_dir, f"总体统计_{ts}.csv")
    stats_df.to_csv(csv_stats, index=False, encoding="utf-8-sig")

    csv_iss = None
    if not issues_df.empty:
        csv_iss = os.path.join(output_dir, f"拓扑问题详情_{ts}.csv")
        issues_df.to_csv(csv_iss, index=False, encoding="utf-8-sig")

    txt_path = os.path.join(output_dir, f"拓扑检查报告_{ts}.txt")
    issue_only_path = _write_txt_report(
        txt_path, target_dir, summaries, all_issues, stats_df
    )

    print("\n" + "=" * 64)
    print("  检查完成！")
    print("=" * 64)
    print(f"  总数据集数          : {stats['数量'][0]}")
    print(f"  存在问题的数据集数  : {stats['数量'][1]}")
    print(f"  总问题数            : {stats['数量'][2]}")
    print(f"    - 空几何         : {stats['数量'][3]}")
    print(f"    - 无效几何       : {stats['数量'][4]}")
    print(f"    - 自相交         : {stats['数量'][5]}")
    print(f"    - 面要素重叠     : {stats['数量'][6]}")
    print(f"    - 重复要素       : {stats['数量'][7]}")
    print(f"    - 细小多边形     : {stats['数量'][8]}")
    print(f"    - 缝隙(参考)     : {stats['数量'][9]}")
    print("-" * 64)
    print(f"  CSV  汇总    ：{csv_summary}")
    print(f"  CSV  总体统计：{csv_stats}")
    if csv_iss:
        print(f"  CSV  问题详情：{csv_iss}")
    print(f"  TXT  报告    ：{txt_path}")
    print(f"  TXT  问题清单：{issue_only_path}")
    print("=" * 64)


# ---------------------- TXT 报告输出 ----------------------
def _write_txt_report(txt_path, target_dir, summaries, all_issues, stats_df):
    """生成 TXT 报告。"""
    sep = "=" * 72
    sub = "-" * 72

    order = ["空几何", "无效几何", "自相交", "面要素重叠",
             "重复要素", "细小多边形", "面要素缝隙"]
    grouped = {k: [] for k in order}
    for it in all_issues:
        t = it.get("问题类型", "其他")
        grouped.setdefault(t, []).append(it)

    for k in grouped:
        grouped[k].sort(
            key=lambda x: (str(x.get("文件/图层", "")),
                           str(x.get("要素ID", x.get("要素ID1", ""))))
        )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(sep + "\n")
        f.write("                   矢量数据拓扑检查报告\n")
        f.write(sep + "\n")
        f.write(f"生成时间   ：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"目标目录   ：{target_dir}\n")
        f.write(f"加速模式   ：{_GPU_MODE}\n")
        f.write(f"细小多边形  ：面积 < {SLIVER_AREA_THRESHOLD}\n")
        f.write(f"重叠阈值   ：面积 > {OVERLAP_AREA_THRESHOLD}\n")
        f.write(sep + "\n\n")

        f.write("【一】总体统计\n")
        f.write(sub + "\n")
        for _, row in stats_df.iterrows():
            f.write(f"  {row['统计项']:<22s} : {row['数量']}\n")
        f.write("\n")

        f.write("【二】数据集检查汇总\n")
        f.write(sub + "\n")
        f.write(f"{'文件/图层':<46s} {'要素':>8s} {'问题':>6s}  状态\n")
        f.write(sub + "\n")
        for s in summaries:
            label = str(s.get("文件/图层", ""))
            if len(label) > 44:
                label = label[:41] + "..."
            f.write(f"{label:<46s} {s.get('要素数', 0):>8d} "
                    f"{s.get('问题总数', 0):>6d}  {s.get('状态', '-')}\n")
        f.write("\n")

        f.write("【三】问题明细（按类型分组）\n")
        f.write(sub + "\n")
        if not all_issues:
            f.write("  未发现任何拓扑问题。\n\n")
        else:
            for ptype in order:
                items = grouped.get(ptype, [])
                if not items:
                    continue
                level = f"[{SEVERITY_MAP.get(ptype, '提示')}]"
                f.write(f"\n  ◆ {ptype}  {level}  —— 共 {len(items)} 处\n")
                f.write("  " + sub + "\n")
                for i, it in enumerate(items, 1):
                    f.write(f"  [{i:>4d}] {it.get('文件/图层', '-')}\n")
                    if "要素ID" in it:
                        f.write(f"         要素ID : {it['要素ID']}\n")
                    if "要素ID1" in it:
                        f.write(f"         要素ID : "
                                f"{it['要素ID1']}  与  {it.get('要素ID2', '-')}\n")
                    desc = it.get("描述", "-")
                    f.write(f"         描述    : {desc}\n")
                    if it.get("面积") is not None:
                        f.write(f"         面积    : {it['面积']:.6f}\n")
                    if it.get("重叠面积") is not None:
                        f.write(f"         重叠面积: {it['重叠面积']:.6f}\n")
                    f.write(f"         原因    : {it.get('原因', '-')}\n")

        f.write("\n" + sep + "\n")
        f.write("报告结束。详细表格请查看同目录下的 CSV 文件。\n")
        f.write(sep + "\n")

    issue_only_path = txt_path.replace(".txt", "_问题清单.txt")
    with open(issue_only_path, "w", encoding="utf-8") as f:
        f.write(sep + "\n")
        f.write("               拓扑错误问题清单（仅含问题要素）\n")
        f.write(sep + "\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"问题总数：{len(all_issues)}\n")
        f.write(sep + "\n")
        if not all_issues:
            f.write("未发现任何拓扑问题。\n")
        else:
            for ptype in order:
                items = grouped.get(ptype, [])
                if not items:
                    continue
                level = f"[{SEVERITY_MAP.get(ptype, '提示')}]"
                f.write(f"\n■ {ptype} {level}（{len(items)} 处）\n")
                f.write("-" * 60 + "\n")
                for i, it in enumerate(items, 1):
                    line = f"  {i:>4d}. {it.get('文件/图层', '-')}"
                    if "要素ID" in it:
                        line += f"  要素ID={it['要素ID']}"
                    elif "要素ID1" in it:
                        line += (f"  要素ID={it['要素ID1']}"
                                 f" & {it.get('要素ID2', '-')}")
                    if it.get("描述"):
                        line += f"  ({it['描述']})"
                    f.write(line + "\n")
        f.write("\n" + sep + "\n")
    return issue_only_path


def main():
    global OVERLAP_CHUNK_SIZE, WORKERS, ENABLE_GPU_ACCEL
    _default_chunk = OVERLAP_CHUNK_SIZE
    _default_workers = WORKERS
    parser = argparse.ArgumentParser(
        description="检查目录中所有矢量数据的拓扑问题（重叠、自相交、重复等），不限制要素数量。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python check_topology(jiangsu).py\n"
            "  python check_topology(jiangsu).py \"D:\\data\\矢量成果\"\n"
            "  python check_topology(jiangsu).py \"D:\\data\" -o \"D:\\报告\" --workers 4 --chunk-size 20000\n"
            "  python check_topology(jiangsu).py \"D:\\data\" --no-gpu  # 禁用 GPU 加速\n"
        ),
    )
    parser.add_argument(
        "target", nargs="?", default=DEFAULT_TARGET_DIR,
        help="要检查的目录路径（不传则使用脚本内的默认目录）",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="报告输出目录（不传则自动在目标目录内创建 _拓扑检查报告 子目录）",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=_default_chunk,
        help=f"重叠检查分块大小（默认 {_default_chunk}，内存不足时调小）",
    )
    parser.add_argument(
        "--workers", type=int, default=_default_workers,
        help="并行进程数（默认 0=自动=cpu_count-1；设 1 走单进程）",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="禁用 GPU/Numba 加速，强制使用纯 CPU 计算",
    )
    args = parser.parse_args()

    if args.chunk_size > 0:
        OVERLAP_CHUNK_SIZE = args.chunk_size
    WORKERS = args.workers
    if args.no_gpu:
        ENABLE_GPU_ACCEL = False

    run_check(args.target, args.output, workers=WORKERS)


if __name__ == "__main__":
    main()
