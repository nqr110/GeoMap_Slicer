import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import voronoi_diagram, unary_union
import random
import math
from sklearn.cluster import KMeans

# --- 步骤1: 读取金华市边界 ---
# 将您的文件路径替换为实际路径
gdf_jinhua = gpd.read_file("金华市.txt")
# 确保坐标系是经纬度 (EPSG:4326)
if gdf_jinhua.crs != "EPSG:4326":
    gdf_jinhua = gdf_jinhua.to_crs("EPSG:4326")

# 获取金华市的几何体 (MultiPolygon 或 Polygon)
jinhua_geom = gdf_jinhua.geometry.union_all()  # 使用新的方法替代deprecated的unary_union
jinhua_bounds = jinhua_geom.bounds  # (minx, miny, maxx, maxy)

# --- 步骤2: 在边界内生成106个更均匀分布的点 ---
num_blocks = 106

def generate_uniform_points_advanced(geometry, bounds, num_points, max_iterations=50):
    """
    使用高级算法生成更均匀分布的点
    1. 首先生成大量随机点
    2. 使用K-means聚类找到均匀分布的中心点
    3. 迭代优化点的位置
    """
    # 步骤1: 生成大量随机点作为候选点
    candidate_points = []
    attempts = 0
    max_attempts = num_points * 20  # 生成20倍数量的候选点
    
    while len(candidate_points) < max_attempts and attempts < max_attempts * 2:
        x = random.uniform(bounds[0], bounds[2])
        y = random.uniform(bounds[1], bounds[3])
        point = Point(x, y)
        
        if geometry.contains(point):
            candidate_points.append([x, y])
        
        attempts += 1
    
    if len(candidate_points) < num_points:
        # 如果候选点不够，直接使用随机点
        return generate_simple_random_points(geometry, bounds, num_points)
    
    # 步骤2: 使用K-means聚类找到均匀分布的中心点
    candidate_array = np.array(candidate_points)
    kmeans = KMeans(n_clusters=num_points, random_state=42, n_init=10)
    cluster_centers = kmeans.fit_predict(candidate_array)
    
    # 获取聚类中心
    centers = kmeans.cluster_centers_
    
    # 步骤3: 将聚类中心转换为Point对象，并确保在边界内
    final_points = []
    for center in centers:
        point = Point(center[0], center[1])
        if geometry.contains(point):
            final_points.append(point)
        else:
            # 如果中心点不在边界内，找到最近的边界内点
            nearest_point = find_nearest_valid_point(center, geometry, bounds)
            if nearest_point:
                final_points.append(nearest_point)
    
    # 如果聚类方法没有生成足够的点，补充随机点
    while len(final_points) < num_points:
        additional_point = generate_simple_random_points(geometry, bounds, 1)[0]
        if additional_point not in final_points:
            final_points.append(additional_point)
    
    return final_points[:num_points]

def generate_simple_random_points(geometry, bounds, num_points):
    """简单的随机点生成方法作为备选"""
    points = []
    attempts = 0
    max_attempts = num_points * 10
    
    while len(points) < num_points and attempts < max_attempts:
        x = random.uniform(bounds[0], bounds[2])
        y = random.uniform(bounds[1], bounds[3])
        point = Point(x, y)
        
        if geometry.contains(point):
            points.append(point)
        
        attempts += 1
    
    return points

def find_nearest_valid_point(center, geometry, bounds, max_attempts=100):
    """找到最近的边界内点"""
    center_point = Point(center[0], center[1])
    
    # 在中心点周围搜索
    for radius in np.linspace(0.01, 0.1, 10):
        for angle in np.linspace(0, 2*np.pi, 20):
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            if bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
                point = Point(x, y)
                if geometry.contains(point):
                    return point
    
    return None

# 生成均匀分布的点
random_points = generate_uniform_points_advanced(jinhua_geom, jinhua_bounds, num_blocks)

# --- 步骤3 & 4: 创建并裁剪Voronoi图 ---
# 为了确保Voronoi图覆盖整个区域，我们使用一个更大的包围盒
# 将随机点转换为GeoSeries
gdf_points = gpd.GeoDataFrame(geometry=random_points, crs="EPSG:4326")
# 创建一个远大于金华市的矩形作为Voronoi图的边界
voronoi_bbox = box(jinhua_bounds[0] - 1, jinhua_bounds[1] - 1, 
                   jinhua_bounds[2] + 1, jinhua_bounds[3] + 1)  # 扩大1度
# 生成Voronoi图
voronoi_polygons = voronoi_diagram(gdf_points.geometry.union_all(), 
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
gdf_blocks.to_file("金华市_106个随机区块.geojson", driver='GeoJSON')

# 计算更准确的面积统计（使用投影坐标系）
gdf_blocks_projected = gdf_blocks.to_crs("EPSG:3857")  # Web Mercator投影
areas = gdf_blocks_projected.geometry.area / 1000000  # 转换为平方公里

print(f"分割完成！生成了{len(gdf_blocks)}个区块，结果已保存为 '金华市_106个随机区块.geojson'")
print(f"区块面积统计（平方公里）：")
print(f"  最小面积: {areas.min():.4f} km²")
print(f"  最大面积: {areas.max():.4f} km²")
print(f"  平均面积: {areas.mean():.4f} km²")
print(f"  面积标准差: {areas.std():.4f} km²")
print(f"  面积变异系数: {(areas.std() / areas.mean() * 100):.2f}%")