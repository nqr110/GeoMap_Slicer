import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import voronoi_diagram, unary_union
import random

# --- 步骤1: 读取金华市边界 ---
# 将您的文件路径替换为实际路径
gdf_jinhua = gpd.read_file("金华市.txt")
# 确保坐标系是经纬度 (EPSG:4326)
if gdf_jinhua.crs != "EPSG:4326":
    gdf_jinhua = gdf_jinhua.to_crs("EPSG:4326")

# 获取金华市的几何体 (MultiPolygon 或 Polygon)
jinhua_geom = gdf_jinhua.geometry.unary_union  # 合并所有部分
jinhua_bounds = jinhua_geom.bounds  # (minx, miny, maxx, maxy)

# --- 步骤2: 在边界内生成27个随机点 ---
num_blocks = 27
random_points = []
while len(random_points) < num_blocks:
    # 在包围盒内随机生成一个点
    x = random.uniform(jinhua_bounds[0], jinhua_bounds[2])
    y = random.uniform(jinhua_bounds[1], jinhua_bounds[3])
    point = Point(x, y)
    # 检查点是否在金华市边界内
    if jinhua_geom.contains(point):
        random_points.append(point)

# --- 步骤3 & 4: 创建并裁剪Voronoi图 ---
# 为了确保Voronoi图覆盖整个区域，我们使用一个更大的包围盒
# 将随机点转换为GeoSeries
gdf_points = gpd.GeoDataFrame(geometry=random_points, crs="EPSG:4326")
# 创建一个远大于金华市的矩形作为Voronoi图的边界
voronoi_bbox = box(jinhua_bounds[0] - 1, jinhua_bounds[1] - 1, 
                   jinhua_bounds[2] + 1, jinhua_bounds[3] + 1)  # 扩大1度
# 生成Voronoi图
voronoi_polygons = voronoi_diagram(unary_union(gdf_points.geometry), 
                                   envelope=voronoi_bbox)

# 将Voronoi多边形转换为GeoDataFrame
gdf_voronoi = gpd.GeoDataFrame(geometry=list(voronoi_polygons.geoms), crs="EPSG:4326")
# 用金华市边界裁剪Voronoi图
gdf_blocks = gpd.overlay(gdf_voronoi, gdf_jinhua, how='intersection')

# --- 步骤5: 扰动边缘 (使边缘不笔直) ---
# 这是一个简化的扰动方法：对每个多边形的顶点进行微小的随机移动
def perturb_polygon(polygon, max_perturbation=0.001):
    """
    对多边形的坐标进行随机扰动。
    max_perturbation: 最大扰动距离（度，约100米）
    """
    if polygon.geom_type == 'Polygon':
        exterior_coords = list(polygon.exterior.coords)
        perturbed_exterior = []
        for x, y in exterior_coords:
            # 添加随机偏移 (-max_perturbation 到 +max_perturbation)
            dx = random.uniform(-max_perturbation, max_perturbation)
            dy = random.uniform(-max_perturbation, max_perturbation)
            perturbed_exterior.append((x + dx, y + dy))
        
        # 处理内部环（如果有）
        interiors = []
        for interior in polygon.interiors:
            interior_coords = list(interior.coords)
            perturbed_interior = []
            for x, y in interior_coords:
                dx = random.uniform(-max_perturbation, max_perturbation)
                dy = random.uniform(-max_perturbation, max_perturbation)
                perturbed_interior.append((x + dx, y + dy))
            interiors.append(Polygon(perturbed_interior))
        
        return Polygon(perturbed_exterior, interiors)
    
    elif polygon.geom_type == 'MultiPolygon':
        # 如果裁剪后是MultiPolygon，对每个部分进行扰动
        perturbed_polys = []
        for poly in polygon.geoms:
            perturbed_poly = perturb_polygon(poly, max_perturbation)
            if perturbed_poly.is_valid:
                perturbed_polys.append(perturbed_poly)
        return unary_union(perturbed_polys) # 或者返回 MultiPolygon(perturbed_polys)
    
    return polygon

# 对每个区块应用扰动
gdf_blocks['geometry'] = gdf_blocks['geometry'].apply(lambda geom: perturb_polygon(geom, max_perturbation=0.0005))

# --- 步骤6: 保存结果 ---
# 重置索引并添加一个区块ID
gdf_blocks = gdf_blocks.reset_index(drop=True)
gdf_blocks['block_id'] = range(1, len(gdf_blocks) + 1)
gdf_blocks['name'] = gdf_blocks['block_id'].apply(lambda x: f"区块{x}")

# 保存为新的GeoJSON文件
gdf_blocks.to_file("金华市_27个随机区块.geojson", driver='GeoJSON')

print("分割完成！结果已保存为 '金华市_27个随机区块.geojson'")