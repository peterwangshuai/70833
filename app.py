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
CONFIG_DIR = r"D:\17166\Documents\作业"          # 修改为您的目录
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

def calc_route_length(pts):
    dist = 0.0
    for i in range(len(pts)-1):
        lat1, lon1 = pts[i]
        lat2, lon2 = pts[i+1]
        dist += latlon_to_meter(lat1, lon1, lat2, lon2)
    return round(dist, 2)

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
    """保存到固定文件（障碍物配置.json）"""
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

def load_obstacles_from_file(file_path=None):
    """从指定文件加载，默认为固定文件"""
    if file_path is None:
        file_path = CONFIG_FILE
    ensure_config_dir()
    if not os.path.exists(file_path):
        st.warning(f"文件不存在：{file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
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

# ========================== 沿凸包边界生成左右绕行路径 ==========================
def build_edge_path(start, end, polygon, direction='clockwise'):
    coords = list(polygon.exterior.coords)[:-1]
    if len(coords) < 3:
        return smooth_curve([start, end])
    pts_latlon = [(y, x) for x, y in coords]
    def dist_to_point(p, target):
        return (p[0]-target[0])**2 + (p[1]-target[1])**2
    start_idx = min(range(len(pts_latlon)), key=lambda i: dist_to_point(pts_latlon[i], start))
    end_idx = min(range(len(pts_latlon)), key=lambda i: dist_to_point(pts_latlon[i], end))
    if direction == 'clockwise':
        if start_idx >= end_idx:
            indices = list(range(start_idx, end_idx-1, -1))
        else:
            indices = list(range(start_idx, -1, -1)) + list(range(len(pts_latlon)-1, end_idx-1, -1))
    else:
        if start_idx <= end_idx:
            indices = list(range(start_idx, end_idx+1))
        else:
            indices = list(range(start_idx, len(pts_latlon))) + list(range(0, end_idx+1))
    path = [start]
    for i in indices:
        path.append(pts_latlon[i])
    path.append(end)
    return smooth_curve(path)

# ========================== 核心规划函数（只生成可行方案） ==========================
def generate_routes(start, end, obstacle_list, obstacle_heights, fly_height, safe_radius):
    routes = {}
    if not obstacle_list:
        mid_point = ((start[0]+end[0])/2, (start[1]+end[1])/2)
        routes["直接飞越航线"] = smooth_curve([start, mid_point, end], seg_num=12)
        routes["综合最优航线"] = routes["直接飞越航线"]
        return routes

    max_obs_height = max([obstacle_heights.get(i,50) for i in range(len(obstacle_list))])
    if fly_height > max_obs_height:
        mid_point = ((start[0]+end[0])/2, (start[1]+end[1])/2)
        routes["直接飞越航线"] = smooth_curve([start, mid_point, end], seg_num=12)
        routes["综合最优航线"] = routes["直接飞越航线"]
        return routes

    # 高度不足，只生成绕行
    raw_polys = []
    for coords in obstacle_list:
        raw_polys.append(Polygon([(lon, lat) for lat, lon in coords]))
    merged_raw = unary_union(raw_polys)
    center_lat = np.mean([p[0] for obs in obstacle_list for p in obs])
    buf_deg = max(meter_to_latlon_offset(center_lat, safe_radius))
    safe_obstacle = merged_raw.buffer(buf_deg, join_style="round", quad_segs=6)
    hull = safe_obstacle.convex_hull
    if hull.geom_type == 'Polygon':
        routes["左侧绕行（顺时针贴边）"] = build_edge_path(start, end, hull, 'clockwise')
        routes["右侧绕行（逆时针贴边）"] = build_edge_path(start, end, hull, 'counterclockwise')
    else:
        center_point = Point(np.mean([p[1] for obs in obstacle_list for p in obs]), np.mean([p[0] for obs in obstacle_list for p in obs]))
        lat_off, lon_off = meter_to_latlon_offset(center_lat, safe_radius)
        offset_scale = 3.0
        left_way = (center_point.y + lat_off * offset_scale, center_point.x - lon_off * offset_scale)
        right_way = (center_point.y - lat_off * offset_scale, center_point.x + lon_off * offset_scale)
        routes["左侧绕行（备份）"] = smooth_curve([start, left_way, end])
        routes["右侧绕行（备份）"] = smooth_curve([start, right_way, end])

    candidates = [(name, routes[name]) for name in routes.keys() if "绕行" in name]
    if candidates:
        best_name, best_path = min(candidates, key=lambda x: calc_route_length(x[1]))
        routes["综合最优航线"] = best_path
        routes["综合最优名称"] = best_name
    else:
        mid_point = ((start[0]+end[0])/2, (start[1]+end[1])/2)
        routes["直接飞越航线"] = smooth_curve([start, mid_point, end], seg_num=12)
        routes["综合最优航线"] = routes["直接飞越航线"]
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
    st.session_state.flight_height = 5
if 'safe_radius' not in st.session_state:
    st.session_state.safe_radius = DEFAULT_SAFE_RADIUS
if 'current_route_points' not in st.session_state:
    st.session_state.current_route_points = [(32.234097, 118.749413), (32.235200, 118.749800)]  # 南京科技职业学院内
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
        st.markdown("#### 📍 起点A（南京科技职业学院内）")
        input_a_lat = st.number_input("纬度", value=32.234097, format="%.6f", key="a_lat")
        input_a_lon = st.number_input("经度", value=118.749413, format="%.6f", key="a_lon")
        if st.button("✅ 设置A点", use_container_width=True):
            st.success("起点A已更新！地图将刷新")
            st.session_state.map_rerun_key += 1
            st.rerun()
        st.divider()
        st.markdown("#### 📍 终点B（南京科技职业学院内）")
        input_b_lat = st.number_input("纬度 ", value=32.235200, format="%.6f", key="b_lat")
        input_b_lon = st.number_input("经度 ", value=118.749800, format="%.6f", key="b_lon")
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
        st.caption("提示：飞行高度≤障碍物高度时自动选取左右绕行最短路径")
        st.divider()
        st.markdown("#### 🚀 障碍物配置")
        if st.session_state.obstacle_polygons:
            st.caption(f"已配置 {len(st.session_state.obstacle_polygons)} 个障碍物 | 画完自动刷新")
            for idx in range(len(st.session_state.obstacle_polygons)):
                with st.expander(f"障碍物 {idx+1}", expanded=True):
                    st.session_state.obstacle_heights[idx] = st.slider(
                        "障碍物高度(米)", 1, 200,
                        value=st.session_state.obstacle_heights.get(idx, 50),
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

        # ====== 修改后的保存/下载/加载区域 ======
        st.markdown("---")
        # 第一行：保存到目录 + 下载JSON
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 保存到目录", type="primary", use_container_width=True):
                save_obstacles_to_file()
                st.success(f"已保存到 {CONFIG_FILE}")
        with col2:
            # 下载按钮
            if st.session_state.obstacle_polygons:
                data = {
                    "版本": VERSION,
                    "保存时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "障碍物总数": len(st.session_state.obstacle_polygons),
                    "障碍物列表": [
                        {
                            "编号": i+1,
                            "坐标": obs,
                            "高度(米)": st.session_state.obstacle_heights.get(i, 50),
                            "创建时间": st.session_state.obstacle_create_time.get(i, "")
                        }
                        for i, obs in enumerate(st.session_state.obstacle_polygons)
                    ]
                }
                json_str = json.dumps(data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 下载JSON",
                    data=json_str,
                    file_name=f"障碍物_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.download_button(
                    label="📥 下载JSON",
                    data="{}",
                    file_name="障碍物_空.json",
                    mime="application/json",
                    use_container_width=True,
                    disabled=True
                )

        # 第二行：加载区域（显示文件列表 + 加载按钮）
        ensure_config_dir()
        json_files = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]
        if json_files:
            selected_file = st.selectbox("选择要加载的障碍物文件", json_files, key="select_obs_file")
            if st.button("📂 加载选中", use_container_width=True):
                load_path = os.path.join(CONFIG_DIR, selected_file)
                load_obstacles_from_file(load_path)
        else:
            st.info("目录下暂无障碍物JSON文件，请先保存或下载")

        # 清空和部署按钮（放在同一行）
        col3, col4 = st.columns(2)
        with col3:
            if st.button("🗑️ 清空所有", use_container_width=True):
                st.session_state.obstacle_polygons.clear()
                st.session_state.obstacle_heights.clear()
                st.session_state.obstacle_create_time.clear()
                st.session_state.last_drawing_id = None
                st.session_state.map_rerun_key += 1
                st.success("所有障碍物已清空！")
                st.rerun()
        with col4:
            if st.button("🚀 部署", type="primary", use_container_width=True):
                st.success("航线已部署！")
        st.divider()

        # 坐标转换及路径规划
        if st.session_state.input_coord_system == "WGS-84":
            a_lat,a_lon = wgs84_to_gcj02(input_a_lat,input_a_lon)
            b_lat,b_lon = wgs84_to_gcj02(input_b_lat,input_b_lon)
        else:
            a_lat,a_lon = input_a_lat,input_a_lon
            b_lat,b_lon = input_b_lat,input_b_lon

        start_pt = (a_lat,a_lon)
        end_pt = (b_lat,b_lon)

        st.session_state.all_routes = generate_routes(
            start_pt, end_pt,
            st.session_state.obstacle_polygons,
            st.session_state.obstacle_heights,
            st.session_state.flight_height,
            st.session_state.safe_radius
        )

        st.markdown("#### 📏 各方案里程对比")
        for name, pts in st.session_state.all_routes.items():
            if name not in ["综合最优航线", "综合最优名称"]:
                st.write(f"{name}：{calc_route_length(pts)} m")

        st.markdown("#### 🧭 航线选择")
        route_keys = [k for k in st.session_state.all_routes.keys() if k not in ["综合最优航线", "综合最优名称"]]
        if route_keys:
            selected_route = st.radio("当前激活航线", route_keys, index=0, key="route_sel",
                                      on_change=lambda: st.session_state.update({"map_rerun_key": st.session_state.map_rerun_key + 1}))
            st.session_state.current_route_points = st.session_state.all_routes[selected_route]
        else:
            st.info("无备选方案，请添加障碍物或调整高度")

    # 地图渲染
    with col_map:
        st.subheader("🗺️ 地图")
        has_bypass = any("绕行" in k for k in st.session_state.all_routes.keys())
        if has_bypass:
            st.caption("🟥左绕行（顺） | 🟧右绕行（逆） | 加粗蓝=综合最优")
        else:
            st.caption("⚫直接飞越航线（综合最优）")
        map_placeholder = st.empty()

        def render_map():
            center_lat = (a_lat + b_lat) / 2
            center_lon = (a_lon + b_lon) / 2
            m = folium.Map(
                [center_lat, center_lon], zoom_start=17,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri卫星地图"
            )
            folium.Marker([a_lat, a_lon], popup="起点A", icon=folium.Icon(color="red", icon="flag")).add_to(m)
            folium.Marker([b_lat, b_lon], popup="终点B", icon=folium.Icon(color="green", icon="flag")).add_to(m)

            for idx, poly in enumerate(st.session_state.obstacle_polygons):
                folium.Polygon(
                    poly, color="#FF0000", fill=True, fill_color="#FF0000", fill_opacity=0.4, weight=3,
                    popup=f"障碍物 {idx+1} | 高度：{st.session_state.obstacle_heights.get(idx,50)}米"
                ).add_to(m)

            style_map = {
                "直接飞越航线": {"color":"#333333", "weight":3, "opacity":0.6},
                "左侧绕行（顺时针贴边）": {"color":"#FF2222", "weight":4, "opacity":0.8},
                "右侧绕行（逆时针贴边）": {"color":"#FF9922", "weight":4, "opacity":0.8},
                "左侧绕行（备份）": {"color":"#FF6666", "weight":4, "opacity":0.8},
                "右侧绕行（备份）": {"color":"#FFAA66", "weight":4, "opacity":0.8},
            }
            for name, pts in st.session_state.all_routes.items():
                if name in ["综合最优航线", "综合最优名称"]:
                    continue
                s = style_map.get(name, {"color":"#888888", "weight":3, "opacity":0.6})
                folium.PolyLine(pts, popup=f"{name} 里程:{calc_route_length(pts)}m",** s).add_to(m)

            if "综合最优航线" in st.session_state.all_routes:
                best_pts = st.session_state.all_routes["综合最优航线"]
                best_label = st.session_state.all_routes.get("综合最优名称", "综合最优")
                folium.PolyLine(best_pts, color="#0033FF", weight=7, opacity=1, popup=f"【综合最优】{best_label}").add_to(m)

            draw = Draw(
                export=False, position="topleft",
                draw_options={"polyline":False,"polygon":{"allowIntersection":False},"rectangle":{},"circle":{},"marker":{},"circlemarker":False},
                edit_options={"edit":{},"remove":{}}
            )
            draw.add_to(m)

            with map_placeholder:
                map_data = st_folium(m, width=1000, height=700, returned_objects=["last_active_drawing"], key=f"map_{st.session_state.map_rerun_key}")

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

# ========================== 飞行监控页面（带预留接口） ==========================
elif st.session_state.current_page == "飞行监控":
    st.header("📡 飞行监控（心跳包实时展示）")

    # ---------- 预留函数接口 ----------
    def mavlink_data_receive():
        """
        预留：持续监听 UDP:14550 接收 MAVLink 数据包。
        后续用 pymavlink 实现，yield 原始消息。
        """
        # from pymavlink import mavutil
        # master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
        # master.wait_heartbeat()
        # while st.session_state.is_running:
        #     msg = master.recv_match(type=['ATTITUDE', 'LOCAL_POSITION_NED', 'GLOBAL_POSITION_INT'])
        #     if msg:
        #         yield msg
        pass

    def data_parse(raw_msg):
        """
        预留：解析 MAVLink 消息，提取位置(经纬度)、高度、姿态。
        返回 dict，例如：{'lat': 32.23, 'lon': 118.74, 'alt': 10.5, 'roll': 0.1, ...}
        """
        # 示例解析结构（待实现）
        return {}

    def ui_refresh(parsed_data):
        """
        预留：将解析后的数据推送到前端显示（比如更新 st.text 或 st.metric）。
        """
        # st.session_state.real_time_info = parsed_data
        pass

    # ---------- UI 控件 ----------
    c1, c2, c3 = st.columns(3)
    with c1: start = st.button("▶️ 启动飞行", type="primary")
    with c2: pause = st.button("⏸️ 暂停飞行")
    with c3: reset = st.button("🔄 重置数据")

    if start:
        st.session_state.is_running = True
        # 预留：启动后台 UDP 接收线程（需用 threading，避免阻塞主线程）
        # import threading
        # if not st.session_state.mavlink_thread_running:
        #     thread = threading.Thread(target=mavlink_data_receive, daemon=True)
        #     thread.start()
        #     st.session_state.mavlink_thread_running = True

    if pause:
        st.session_state.is_running = False
        # 预留：停止线程或关闭连接

    if reset:
        st.session_state.df_history = pd.DataFrame(columns=["时间", "序号"])
        st.session_state.is_running = False
        st.rerun()

    status = st.empty()
    chart_area = st.empty()
    list_area = st.empty()

    # ---------- 模拟心跳显示（后续替换为真实数据） ----------
    while st.session_state.is_running:
        # ===== 这一段以后要改成接收真实 MAVLink 数据 =====
        # 目前是模拟数据，保留作为演示
        now_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        seq = len(st.session_state.df_history) + 1
        new_row = pd.DataFrame({"时间": [now_time], "序号": [seq]})
        st.session_state.df_history = pd.concat([st.session_state.df_history, new_row], ignore_index=True)

        # 显示图表和表格
        chart_area.line_chart(st.session_state.df_history, x="时间", y="序号", color="#39ff14")
        list_area.dataframe(st.session_state.df_history.tail(10), hide_index=True, height=400)
        status.success(f"✅ 飞行运行正常 | 心跳序号：{seq}")

        # 这里可以调用 ui_refresh(real_data) 显示真实数据
        # real_data = data_parse(接收到的消息)
        # ui_refresh(real_data)

        st.session_state.last_received = time.time()
        time.sleep(1)

    # 异常检测（仍为占位）
    if st.session_state.last_received and not st.session_state.is_running:
        elapsed = time.time() - st.session_state.last_received
        if elapsed > 3 and len(st.session_state.df_history) > 0:
            status.error("🚨 连接异常！超过3秒未收到心跳包！")
        else:
            status.warning("⏸️ 飞行已暂停")
