import streamlit as st
import time
import datetime
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import os
import json
import networkx as nx
from shapely.geometry import LineString, Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from shapely import buffer

# ========================== 全局配置：汉化CSS ==========================
st.set_page_config(page_title="无人机航线规划系统", layout="wide")

st.markdown('''
<style>
.leaflet-tooltip,.leaflet-draw-tooltip,.leaflet-control-zoom-in[title],.leaflet-control-zoom-out[title],.leaflet-draw-buttons button[title] {display:none!important;visibility:hidden!important;}
.leaflet-control-zoom-in::after {content:"放大地图";position:absolute;left:45px;top:0;background:#222;color:#fff;padding:3px 8px;font-size:12px;border-radius:3px;white-space:nowrap;z-index:999999;}
.leaflet-control-zoom-out::after {content:"缩小地图";position:absolute;left:45px;top:0;background:#222;color:#fff;padding:3px 8px;font-size:12px;border-radius:3px;white-space:nowrap;z-index:999999;}
.leaflet-draw-draw-polygon::after {content:"绘制多边形";position:absolute;left:45px;top:0;background:#222;color:#fff;padding:3px 8px;font-size:12px;border-radius:3px;white-space:nowrap;z-index:999999;}
.leaflet-draw-draw-rectangle::after {content:"绘制矩形";position:absolute;left:45px;top:0;background:#222;color:#fff;padding:3px 8px;font-size:12px;border-radius:3px;white-space:nowrap;z-index:999999;}
.leaflet-draw-draw-circle::after {content:"绘制圆形";position:absolute;left:45px;top:0;background:#222;color:#fff;padding:3px 8px;font-size:12px;border-radius:3px;white-space:nowrap;z-index:999999;}
.leaflet-draw-draw-marker::after {content:"添加标记点";position:absolute;left:45px;top:0;background:#222;color:#fff;padding:3px 8px;font-size:12px;border-radius:3px;white-space:nowrap;z-index:999999;}
.leaflet-draw-edit-edit::after {content:"编辑图层";position:absolute;left:45px;top:0;background:#222;color:#fff;padding:3px 8px;font-size:12px;border-radius:3px;white-space:nowrap;z-index:999999;}
.leaflet-draw-edit-remove::after {content:"删除图层";position:absolute;left:45px;top:0;background:#222;color:#fff;padding:3px 8px;font-size:12px;border-radius:3px;white-space:nowrap;z-index:999999;}
.leaflet-control-attribution {display:none!important;}
.stButton>button {border-radius:4px!important;}
</style>
''', unsafe_allow_html=True)

# ========================== 基础全局参数 ==========================
CONFIG_DIR = r"D:\wrj\3Dwrj"
CONFIG_FILE = os.path.join(CONFIG_DIR, "障碍物配置.json")
VERSION = "v18.0 可视图Dijkstra全局最优避障版"
DEFAULT_SAFE_RADIUS = 5

# ========================== 坐标系转换 ==========================
def wgs84_to_gcj02(lat, lon):
    a = 6378245.0
    ee = 0.00669342162296594323
    dLat = transform_lat(lon - 105.0, lat - 35.0)
    dLon = transform_lon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * np.pi
    magic = np.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = np.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * np.pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * np.cos(radLat) * np.pi)
    mgLat = lat + dLat
    mgLon = lon + dLon
    return round(mgLat, 6), round(mgLon, 6)

def gcj02_to_wgs84(lat, lon):
    g_lat, g_lon = wgs84_to_gcj02(lat, lon)
    d_lat = g_lat - lat
    d_lon = g_lon - lon
    wgs_lat = lat - d_lat
    wgs_lon = lon - d_lon
    return round(wgs_lat, 6), round(wgs_lon, 6)

def transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.fabs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(y * np.pi) + 40.0 * np.sin(y / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (160.0 * np.sin(y / 12.0 * np.pi) + 320 * np.sin(y * np.pi / 30.0)) * 2.0 / 3.0
    return ret

def transform_lon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.fabs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(x * np.pi) + 40.0 * np.sin(x / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (150.0 * np.sin(x / 12.0 * np.pi) + 300.0 * np.sin(x / 30.0 * np.pi)) * 2.0 / 3.0
    return ret

def latlon_to_meter(lat1, lon1, lat2, lon2):
    R = 6371000
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def meter_to_latlon_offset(lat, meter):
    lat_offset = meter / 111319.9
    lon_offset = meter / (111319.9 * np.cos(np.radians(lat)))
    return lat_offset, lon_offset

# 计算整条航线总里程
def calc_route_length(pts):
    dist = 0.0
    for i in range(len(pts)-1):
        lat1, lon1 = pts[i]
        lat2, lon2 = pts[i+1]
        dist += latlon_to_meter(lat1, lon1, lat2, lon2)
    return round(dist, 2)

# 二阶贝塞尔平滑曲线
def smooth_curve(points, seg_num=18):
    smooth_pts = []
    for i in range(len(points)-1):
        p0 = np.array(points[i])
        p1 = np.array(points[i+1])
        mid = (p0 + p1) / 2
        for t in np.linspace(0, 1, seg_num):
            lat = (1-t)**2 * p0[0] + 2*(1-t)*t * mid[0] + t**2 * p1[0]
            lon = (1-t)**2 * p0[1] + 2*(1-t)*t * mid[1] + t**2 * p1[1]
            smooth_pts.append((round(lat,6), round(lon,6)))
    return smooth_pts

# ========================== 障碍物配置持久化 ==========================
def ensure_config_dir():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)

def save_obstacles_to_file():
    ensure_config_dir()
    save_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_data = {
        "版本": VERSION,
        "保存时间": save_time,
        "障碍物总数": len(st.session_state.obstacle_polygons),
        "障碍物列表": []
    }
    for idx, obs in enumerate(st.session_state.obstacle_polygons):
        obs_data = {
            "编号": idx + 1,
            "坐标": obs,
            "高度(米)": st.session_state.obstacle_heights.get(idx, 50),
            "创建时间": st.session_state.obstacle_create_time.get(idx, save_time)
        }
        save_data["障碍物列表"].append(obs_data)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    return save_data

def load_obstacles_from_file():
    ensure_config_dir()
    if not os.path.exists(CONFIG_FILE):
        st.warning("配置文件不存在，请先保存配置")
        return None
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            load_data = json.load(f)
        new_polygons = []
        new_heights = {}
        new_create_time = {}
        for idx, obs in enumerate(load_data["障碍物列表"]):
            new_polygons.append(obs["坐标"])
            new_heights[idx] = obs.get("高度(米)", 50)
            new_create_time[idx] = obs.get("创建时间", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.session_state.obstacle_polygons = new_polygons
        st.session_state.obstacle_heights = new_heights
        st.session_state.obstacle_create_time = new_create_time
        st.session_state.last_drawing_id = None
        st.session_state.map_rerun_key += 1
        st.rerun()
        return load_data
    except Exception as e:
        st.error(f"加载失败：{str(e)}")
        return None

# ========================== 新增：可视图核心工具函数 ==========================
def point_latlon_to_xy(pt):
    """(lat,lon)转shapely Point(x=lon,y=lat)"""
    return Point(pt[1], pt[0])

def is_line_safe(p1, p2, safe_obstacle_union):
    """两点连线不与安全缓冲区相交=可视"""
    line = LineString([point_latlon_to_xy(p1), point_latlon_to_xy(p2)])
    return not line.intersects(safe_obstacle_union)

def build_visibility_shortest_path(start, end, safe_obs_union):
    """构建可视图 + Dijkstra求解全局最短无碰撞路径"""
    # 提取所有障碍物顶点
    verts = [start, end]
    if isinstance(safe_obs_union, MultiPolygon):
        polys = list(safe_obs_union.geoms)
    else:
        polys = [safe_obs_union]
    for poly in polys:
        coords = list(poly.exterior.coords)[:-1]
        for (x, y) in coords:
            verts.append((y, x)) # xy转latlon
    # 构建图
    G = nx.Graph()
    node_idx = {}
    for i, v in enumerate(verts):
        node_idx[v] = i
        G.add_node(i, pos=v)
    # 遍历所有点对，建立可视边
    n = len(verts)
    for i in range(n):
        p_i = verts[i]
        for j in range(i+1, n):
            p_j = verts[j]
            if is_line_safe(p_i, p_j, safe_obs_union):
                dist = latlon_to_meter(p_i[0], p_i[1], p_j[0], p_j[1])
                G.add_edge(node_idx[p_i], node_idx[p_j], weight=dist)
    # Dijkstra求起点到终点最短路径
    start_id = node_idx[start]
    end_id = node_idx[end]
    try:
        path_ids = nx.dijkstra_path(G, start_id, end_id, weight="weight")
        raw_path = [verts[pid] for pid in path_ids]
        return smooth_curve(raw_path, seg_num=18)
    except nx.NetworkXNoPath:
        # 无通路兜底
        mid = ((start[0]+end[0])/2, (start[1]+end[1])/2)
        return smooth_curve([start, mid, end])

# ========================== 核心规划函数：多算法对比输出 ==========================
def generate_routes(start, end, obstacle_list, obstacle_heights, fly_height, safe_radius):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end
    # 1. 直飞参考线
    mid_point = ((start[0]+end[0])/2, (start[1]+end[1])/2)
    routes["方案1：直接飞越航线"] = smooth_curve([start, mid_point, end], seg_num=12)
    # 无障碍物直接返回直飞
    if not obstacle_list:
        routes["综合最优航线"] = routes["方案1：直接飞越航线"]
        return routes
    # 最大障碍物高度
    max_obs_height = max([obstacle_heights.get(i,50) for i in range(len(obstacle_list))])
    # 障碍物多边形合并
    raw_polys = []
    for coords in obstacle_list:
        raw_polys.append(Polygon([(lon,lat) for lat,lon in coords]))
    merged_raw = unary_union(raw_polys)
    # 生成安全缓冲区（所有航线必须远离该区域）
    center_lat = np.mean([p[0] for obs in obstacle_list for p in obs])
    buf_deg = max(meter_to_latlon_offset(center_lat, safe_radius))
    safe_obstacle = merged_raw.buffer(buf_deg, join_style="round", quad_segs=6)
    # 高度足够：飞越最优
    if fly_height > max_obs_height:
        routes["综合最优航线"] = routes["方案1：直接飞越航线"]
        return routes
    # 2. 可视图全局最短绕行（标准最优方案）
    vis_short = build_visibility_shortest_path(start, end, safe_obstacle)
    routes["方案2：可视图全局最短绕行(理论最优)"] = vis_short
    # 3. 旧版偏移绕行（对比用）
    center_point = Point(np.mean([p[1] for obs in obstacle_list for p in obs]), np.mean([p[0] for obs in obstacle_list for p in obs]))
    lat_off, lon_off = meter_to_latlon_offset(center_lat, safe_radius)
    offset_scale = 9.2
    left_way = (center_point.y + lat_off * offset_scale, center_point.x - lon_off * offset_scale)
    right_way = (center_point.y - lat_off * offset_scale, center_point.x + lon_off * offset_scale)
    routes["方案3：左侧固定偏移绕行"] = smooth_curve([start, left_way, end])
    routes["方案3：右侧固定偏移绕行"] = smooth_curve([start, right_way, end])
    # 对比所有平面绕行里程，选出综合最优
    plane_candidates = [
        ("方案2：可视图全局最短绕行(理论最优)", routes["方案2：可视图全局最短绕行(理论最优)"]),
        ("方案3：左侧固定偏移绕行", routes["方案3：左侧固定偏移绕行"]),
        ("方案3：右侧固定偏移绕行", routes["方案3：右侧固定偏移绕行"])
    ]
    min_dist = float("inf")
    best_name = ""
    best_path = None
    for name, pts in plane_candidates:
        d = calc_route_length(pts)
        if d < min_dist:
            min_dist = d
            best_name = name
            best_path = pts
    routes[f"综合最优航线({best_name})"] = best_path
    return routes

# ========================== 全局状态初始化 ==========================
if 'current_page' not in st.session_state:
    st.session_state.current_page = "航线规划"
if 'input_coord_system' not in st.session_state:
    st.session_state.input_coord_system = "GCJ-02(高德/百度)"
if 'df_history' not in st.session_state:
    st.session_state.df_history = pd.DataFrame(columns=["时间", "序号"])
if 'last_received' not in st.session_state:
    st.session_state.last_received = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'obstacle_polygons' not in st.session_state:
    st.session_state.obstacle_polygons = []
if 'obstacle_heights' not in st.session_state:
    st.session_state.obstacle_heights = {}
if 'obstacle_create_time' not in st.session_state:
    st.session_state.obstacle_create_time = {}
if 'last_drawing_id' not in st.session_state:
    st.session_state.last_drawing_id = None
if 'flight_height' not in st.session_state:
    st.session_state.flight_height = 10
if 'safe_radius' not in st.session_state:
    st.session_state.safe_radius = DEFAULT_SAFE_RADIUS
if 'current_route_points' not in st.session_state:
    st.session_state.current_route_points = [(32.2323, 118.749), (32.2344, 118.749)]
if 'map_rerun_key' not in st.session_state:
    st.session_state.map_rerun_key = 0
if 'all_routes' not in st.session_state:
    st.session_state.all_routes = {}

# ========================== 左侧导航栏 ==========================
with st.sidebar:
    st.subheader("🧭 导航")
    st.session_state.current_page = st.radio("", ["航线规划", "飞行监控"], index=0, label_visibility="collapsed")
    st.divider()
    st.subheader("⚙️ 坐标系设置")
    st.session_state.input_coord_system = st.radio("", ["WGS-84", "GCJ-02(高德/百度)"], index=1, label_visibility="collapsed")
    st.divider()
    st.subheader("📊 系统状态")
    st.success("✅ 起点A已设置")
    st.success("✅ 终点B已设置")
    if st.button("🔄 强制刷新地图", type="secondary", use_container_width=True):
        st.session_state.map_rerun_key += 1
        st.rerun()

# ========================== 航线规划主页面 ==========================
if st.session_state.current_page == "航线规划":
    st.header("🗺️ 航线规划 | 多算法对比择优")
    col_map, col_control = st.columns([2, 1])
    with col_control:
        st.subheader("⚙️ 控制面板")
        st.markdown("#### 📍 起点A")
        input_a_lat = st.number_input("纬度", value=32.2323, format="%.4f", key="a_lat")
        input_a_lon = st.number_input("经度", value=118.749, format="%.3f", key="a_lon")
        if st.button("✅ 设置A点", use_container_width=True):
            st.success("起点A已更新！地图将刷新")
            st.session_state.map_rerun_key += 1
            st.rerun()
        st.divider()
        st.markdown("#### 📍 终点B")
        input_b_lat = st.number_input("纬度 ", value=32.2344, format="%.4f", key="b_lat")
        input_b_lon = st.number_input("经度 ", value=118.749, format="%.3f", key="b_lon")
        if st.button("✅ 设置B点", use_container_width=True):
            st.success("终点B已更新！地图将刷新")
            st.session_state.map_rerun_key += 1
            st.rerun()
        st.divider()
        st.markdown("#### ✈️ 飞行参数")
        st.session_state.flight_height = st.slider(
            "无人机飞行高度(米)", 1, 200, st.session_state.flight_height, key="flight_h",
            on_change=lambda: st.session_state.update({"map_rerun_key": st.session_state.map_rerun_key + 1})
        )
        st.session_state.safe_radius = st.number_input(
            "安全距离(米)", value=st.session_state.safe_radius, min_value=1, key="safe_r",
            on_change=lambda: st.session_state.update({"map_rerun_key": st.session_state.map_rerun_key + 1})
        )
        st.caption("提示：飞行高度≤障碍物高度自动对比3种平面绕行，选出全局最短")
        st.divider()
        st.markdown("#### 🚀 障碍物配置")
        if st.session_state.obstacle_polygons:
            st.caption(f"已配置 {len(st.session_state.obstacle_polygons)} 个障碍物 | 画完自动刷新")
            for idx in range(len(st.session_state.obstacle_polygons)):
                with st.expander(f"障碍物 {idx+1}", expanded=True):
                    st.session_state.obstacle_heights[idx] = st.slider(
                        "障碍物高度(米)", 1, 200, value=st.session_state.obstacle_heights.get(idx, 50),
                        key=f"h_{idx}", on_change=lambda: st.session_state.update({"map_rerun_key": st.session_state.map_rerun_key + 1})
                    )
                    if st.button(f"🗑️ 删除障碍物 {idx+1}", key=f"del_{idx}", use_container_width=True):
                        st.session_state.obstacle_polygons.pop(idx)
                        if idx in st.session_state.obstacle_heights:
                            del st.session_state.obstacle_heights[idx]
                        if idx in st.session_state.obstacle_create_time:
                            del st.session_state.obstacle_create_time[idx]
                        st.session_state.last_drawing_id = None
                        st.session_state.map_rerun_key += 1
                        st.success(f"障碍物 {idx+1} 已删除！")
                        st.rerun()
        else:
            st.info("🖌️ 请在地图上圈选障碍物区域（画完自动刷新）")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            if st.button("💾 保存", type="primary", use_container_width=True):
                save_obstacles_to_file()
                st.success("配置已保存！")
        with c2:
            if st.button("📂 加载", use_container_width=True):
                load_obstacles_from_file()
        with c3:
            if st.button("🗑️ 清空", use_container_width=True):
                st.session_state.obstacle_polygons.clear()
                st.session_state.obstacle_heights.clear()
                st.session_state.obstacle_create_time.clear()
                st.session_state.last_drawing_id = None
                st.session_state.map_rerun_key += 1
                st.success("所有障碍物已清空！")
                st.rerun()
        with c4:
            if st.button("🚀 部署", type="primary", use_container_width=True):
                st.success("航线已部署！")
        st.divider()
        # 坐标转换
        if st.session_state.input_coord_system == "WGS-84":
            a_lat,a_lon = wgs84_to_gcj02(input_a_lat,input_a_lon)
            b_lat,b_lon = wgs84_to_gcj02(input_b_lat,input_b_lon)
        else:
            a_lat,a_lon = input_a_lat,input_a_lon
            b_lat,b_lon = input_b_lat,input_b_lon
        start_pt = (a_lat,a_lon)
        end_pt = (b_lat,b_lon)
        # 执行多算法路径规划
        st.session_state.all_routes = generate_routes(
            start_pt, end_pt,
            st.session_state.obstacle_polygons,
            st.session_state.obstacle_heights,
            st.session_state.flight_height,
            st.session_state.safe_radius
        )
        # 展示各方案里程对比
        st.markdown("#### 📏 各方案里程对比")
        for name, pts in st.session_state.all_routes.items():
            st.write(f"{name}：{calc_route_length(pts)} m")
        # 航线选择
        st.markdown("#### 🧭 航线选择")
        route_keys = list(st.session_state.all_routes.keys())
        default_idx = route_keys.index([k for k in route_keys if "综合最优航线" in k][0])
        selected_route = st.radio("当前激活航线", route_keys, index=default_idx, key="route_sel",
            on_change=lambda: st.session_state.update({"map_rerun_key": st.session_state.map_rerun_key + 1}))
        st.session_state.current_route_points = st.session_state.all_routes[selected_route]
    # 地图渲染区域
    with col_map:
        st.subheader("🗺️ 地图")
        st.caption("⚫直飞越障 | 🟦可视图全局最优 | 🟥左偏移 | 🟧右偏移 | 加粗蓝=综合最优")
        map_placeholder = st.empty()
        def render_map():
            center_lat = (a_lat + b_lat) / 2
            center_lon = (a_lon + b_lon) / 2
            m = folium.Map(
                [center_lat, center_lon], zoom_start=17,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri卫星地图"
            )
            # 起止标记
            folium.Marker([a_lat, a_lon], popup="起点A", icon=folium.Icon(color="red", icon="flag")).add_to(m)
            folium.Marker([b_lat, b_lon], popup="终点B", icon=folium.Icon(color="green", icon="flag")).add_to(m)
            # 绘制障碍物本体
            for idx, poly in enumerate(st.session_state.obstacle_polygons):
                folium.Polygon(
                    poly, color="#FF0000", fill=True, fill_color="#FF0000", fill_opacity=0.4, weight=3,
                    popup=f"障碍物 {idx+1} | 高度：{st.session_state.obstacle_heights.get(idx,50)}米"
                ).add_to(m)
            # 航线配色规则
            style_map = {
                "方案1：直接飞越航线": {"color":"#333333", "weight":3, "opacity":0.6},
                "方案2：可视图全局最短绕行(理论最优)": {"color":"#0066FF", "weight":4, "opacity":0.8},
                "方案3：左侧固定偏移绕行": {"color":"#FF2222", "weight":4, "opacity":0.8},
                "方案3：右侧固定偏移绕行": {"color":"#FF9922", "weight":4, "opacity":0.8},
            }
            # 绘制所有备选航线
            for name, pts in st.session_state.all_routes.items():
                if "综合最优航线" not in name:
                    s = style_map[name]
                    folium.PolyLine(pts, popup=f"{name} 里程:{calc_route_length(pts)}m",** s).add_to(m)
            # 综合最优加粗高亮
            best_key = [k for k in list(st.session_state.all_routes.keys()) if "综合最优航线" in k][0]
            best_pts = st.session_state.all_routes[best_key]
            folium.PolyLine(best_pts, color="#0033FF", weight=7, opacity=1, popup=f"【综合最优】{best_key}").add_to(m)
            # 绘图工具
            draw = Draw(
                export=False, position="topleft",
                draw_options={"polyline":False,"polygon":{"allowIntersection":False},"rectangle":{},"circle":{},"marker":{},"circlemarker":False},
                edit_options={"edit":{},"remove":{}}
            )
            draw.add_to(m)
            # 渲染地图
            with map_placeholder:
                map_data = st_folium(m, width=1000, height=700, returned_objects=["last_active_drawing"], key=f"map_{st.session_state.map_rerun_key}")
            # 新增障碍物逻辑
            if map_data and map_data.get("last_active_drawing"):
                draw_id = str(map_data["last_active_drawing"]["geometry"]["coordinates"])
                if draw_id != st.session_state.last_drawing_id:
                    st.session_state.last_drawing_id = draw_id
                    g_type = map_data["last_active_drawing"]["geometry"]["type"]
                    if g_type == "Polygon":
                        coords = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
                        poly_coords = [[lat, lon] for lon, lat in coords]
                        if poly_coords not in st.session_state.obstacle_polygons:
                            st.session_state.obstacle_polygons.append(poly_coords)
                            new_id = len(st.session_state.obstacle_polygons)-1
                            st.session_state.obstacle_heights[new_id] = 50
                            st.session_state.obstacle_create_time[new_id] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.map_rerun_key +=1
                            st.success(f"✅ 障碍物{new_id+1}添加完成，默认高度50m")
                            st.rerun()
        render_map()
# ========================== 飞行监控页面 ==========================
elif st.session_state.current_page == "飞行监控":
    st.header("📡 飞行监控（心跳包实时展示）")
    c1,c2,c3 = st.columns(3)
    with c1: start = st.button("▶️ 启动飞行", type="primary")
    with c2: pause = st.button("⏸️ 暂停飞行")
    with c3: reset = st.button("🔄 重置数据")
    if start: st.session_state.is_running = True
    if pause: st.session_state.is_running = False
    if reset:
        st.session_state.df_history = pd.DataFrame(columns=["时间","序号"])
        st.session_state.is_running = False
        st.rerun()
    status = st.empty()
    chart_area = st.empty()
    list_area = st.empty()
    while st.session_state.is_running:
        now_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        seq = len(st.session_state.df_history)+1
        new_row = pd.DataFrame({"时间":[now_time],"序号":[seq]})
        st.session_state.df_history = pd.concat([st.session_state.df_history,new_row], ignore_index=True)
        chart_area.line_chart(st.session_state.df_history, x="时间", y="序号", color="#39ff14")
        list_area.dataframe(st.session_state.df_history.tail(10), hide_index=True, height=400)
        status.success(f"✅ 飞行运行正常 | 心跳序号：{seq}")
        st.session_state.last_received = time.time()
        time.sleep(1)
    if st.session_state.last_received and not st.session_state.is_running:
        elapsed = time.time() - st.session_state.last_received
        if elapsed > 3 and len(st.session_state.df_history) > 0:
            status.error("🚨 连接异常！超过3秒未收到心跳包！")
        else:
            status.warning("⏸️ 飞行已暂停")
