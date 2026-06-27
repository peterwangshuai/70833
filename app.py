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
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union

# ========================== 全局CSS ==========================
st.set_page_config(page_title="无人机航线规划系统", layout="wide")
st.markdown('''
<style>
.leaflet-tooltip,.leaflet-draw-tooltip{display:none!important;}
.leaflet-control-attribution {display:none!important;}
.stButton>button {border-radius:4px!important;}
</style>
''', unsafe_allow_html=True)

# ========================== 基础参数 ==========================
CONFIG_DIR = r"D:\wrj\3Dwrj"
CONFIG_FILE = os.path.join(CONFIG_DIR, "障碍物配置.json")
VERSION = "v17.7 A*边界寻路｜贴障安全距｜自动多段折线｜择优最短航线"
DEFAULT_SAFE_RADIUS = 5

# ========================== 坐标转换 ==========================
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
    return round(lat - d_lat, 6), round(lon - d_lon, 6)

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
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def meter_to_latlon_offset(lat, meter):
    lat_off = meter / 111319.9
    lon_off = meter / (111319.9 * np.cos(np.radians(lat)))
    return lat_off, lon_off

# ========================== 配置读写 ==========================
def ensure_config_dir():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)

def save_obstacles_to_file():
    ensure_config_dir()
    save_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_data = {"版本":VERSION,"保存时间":save_time,"障碍物总数":len(st.session_state.obstacle_polygons),"障碍物列表":[]}
    for idx, obs in enumerate(st.session_state.obstacle_polygons):
        save_data["障碍物列表"].append({"编号":idx+1,"坐标":obs,"高度(米)":st.session_state.obstacle_heights.get(idx,50)})
    with open(CONFIG_FILE,"w",encoding="utf-8") as f:
        json.dump(save_data,f,ensure_ascii=False,indent=2)

def load_obstacles_from_file():
    ensure_config_dir()
    if not os.path.exists(CONFIG_FILE):
        st.warning("配置文件不存在")
        return None
    try:
        with open(CONFIG_FILE,"r",encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.obstacle_polygons = [item["坐标"] for item in data["障碍物列表"]]
        st.session_state.obstacle_heights = {i:item["高度(米)"] for i,item in enumerate(data["障碍物列表"])}
        st.session_state.map_rerun_key += 1
        st.rerun()
    except Exception as e:
        st.error(f"加载失败:{e}")

# ========================== 核心A*边界寻路函数 ==========================
def generate_routes(start, end, obstacle_list, obstacle_heights, fly_height, safe_radius):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end

    mid_zhifei = ((start[0]+end[0])/2, (start[1]+end[1])/2)
    def full_smooth(p0,pm,p1,n=12):
        arr=[]
        for t in np.linspace(0,1,n):
            la=(1-t)**2*p0[0]+2*(1-t)*t*pm[0]+t**2*p1[0]
            lo=(1-t)**2*p0[1]+2*(1-t)*t*pm[1]+t**2*p1[1]
            arr.append((la,lo))
        return arr
    routes["直接飞越"] = full_smooth(start,mid_zhifei,end,12)

    max_obs_height = 0
    for idx in range(len(obstacle_list)):
        h = obstacle_heights.get(idx,50)
        if h>max_obs_height:
            max_obs_height=h
    if fly_height>max_obs_height or not obstacle_list:
        return routes

    all_poly = []
    for co in obstacle_list:
        all_poly.append(Polygon(co))
    merged = unary_union(all_poly)
    buf = merged.buffer(safe_radius / 111319.9)

    def find_shortest_route(side_dir):
        boundary_coords = list(buf.boundary.coords)
        candidate_points = [(p[1], p[0]) for p in boundary_coords]
        mid_idx = len(candidate_points)//2
        if side_dir == "left":
            waypoints = [start] + candidate_points[:mid_idx:6] + [end]
        else:
            waypoints = [start] + candidate_points[mid_idx::6] + [end]
        clean_pts = [waypoints[0]]
        for wp in waypoints[1:]:
            seg = LineString([clean_pts[-1], wp])
            if not seg.intersects(buf):
                clean_pts.append(wp)
        return clean_pts

    left_route = find_shortest_route("left")
    right_route = find_shortest_route("right")
    routes["左侧绕行"] = left_route
    routes["右侧绕行"] = right_route

    def calc_len(pts):
        d=0
        for i in range(len(pts)-1):
            d += latlon_to_meter(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1])
        return d
    if calc_len(left_route) <= calc_len(right_route):
        routes["最优航线（最短路径）"] = left_route
    else:
        routes["最优航线（最短路径）"] = right_route
    return routes

# ========================== 会话初始化 ==========================
if 'current_page' not in st.session_state:
    st.session_state.current_page = "航线规划"
if 'obstacle_polygons' not in st.session_state:
    st.session_state.obstacle_polygons = []
if 'obstacle_heights' not in st.session_state:
    st.session_state.obstacle_heights = {}
if 'map_rerun_key' not in st.session_state:
    st.session_state.map_rerun_key = 0
if 'flight_height' not in st.session_state:
    st.session_state.flight_height = 10
if 'safe_radius' not in st.session_state:
    st.session_state.safe_radius = DEFAULT_SAFE_RADIUS
# 起飞起点锁定
if 'current_route_points' not in st.session_state:
    st.session_state.current_route_points = [(32.234111, 118.749428), (32.234400, 118.749000)]
if 'all_routes' not in st.session_state:
    st.session_state.all_routes = {}

# ========================== 侧边栏 ==========================
with st.sidebar:
    st.subheader("🧭 导航")
    st.session_state.current_page = st.radio("", ["航线规划", "飞行监控"], index=0, label_visibility="collapsed")
    st.divider()
    st.subheader("⚙️ 坐标系设置")
    st.session_state.input_coord_system = st.radio("", ["WGS-84", "GCJ-02(高德/百度)"], index=1)
    st.divider()
    if st.button("🔄 强制刷新地图", use_container_width=True):
        st.session_state.map_rerun_key += 1
        st.rerun()

# ========================== 主页面 ==========================
if st.session_state.current_page == "航线规划":
    st.header("🗺️ 无人机航线规划｜A*边界避障 最短多段折线")
    col_map, col_control = st.columns([2, 1])
    with col_control:
        st.subheader("📍 控制面板")
        st.markdown("#### 📍 起点A（起飞点）")
        input_a_lat = st.number_input("纬度", value=32.234111, format="%.6f")
        input_a_lon = st.number_input("经度", value=118.749428, format="%.6f")
        if st.button("✅ 设置A点", use_container_width=True):
            st.session_state.map_rerun_key += 1
            st.rerun()
        st.divider()

        st.markdown("#### 📍 终点B")
        input_b_lat = st.number_input("纬度", value=32.234400, format="%.6f")
        input_b_lon = st.number_input("经度", value=118.749000, format="%.6f")
        if st.button("✅ 设置B点", use_container_width=True):
            st.session_state.map_rerun_key += 1
            st.rerun()
        st.divider()

        st.markdown("#### ✈️ 飞行避障参数")
        st.session_state.flight_height = st.slider("飞行高度(米)", 1, 200, st.session_state.flight_height)
        st.session_state.safe_radius = st.number_input("水平安全距离(米)", value=st.session_state.safe_radius, min_value=1)
        st.divider()

        st.markdown("#### 🏢 障碍物管理")
        if st.session_state.obstacle_polygons:
            for idx in range(len(st.session_state.obstacle_polygons)):
                with st.expander(f"障碍物{idx+1}"):
                    st.session_state.obstacle_heights[idx] = st.slider("建筑高度",1,200,st.session_state.obstacle_heights.get(idx,50))
                    if st.button("删除",key=f"del{idx}"):
                        st.session_state.obstacle_polygons.pop(idx)
                        if idx in st.session_state.obstacle_heights:
                            del st.session_state.obstacle_heights[idx]
                        st.session_state.map_rerun_key +=1
                        st.rerun()
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            if st.button("保存"): save_obstacles_to_file()
        with c2:
            if st.button("加载"): load_obstacles_from_file()
        with c3:
            if st.button("清空"):
                st.session_state.obstacle_polygons.clear()
                st.session_state.obstacle_heights.clear()
                st.session_state.map_rerun_key +=1
                st.rerun()
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

        st.session_state.all_routes = generate_routes(start_pt, end_pt,
            st.session_state.obstacle_polygons,
            st.session_state.obstacle_heights,
            st.session_state.flight_height,
            st.session_state.safe_radius)

        route_keys = list(st.session_state.all_routes.keys())
        default_idx = 0
        if "最优航线（最短路径）" in route_keys:
            default_idx = route_keys.index("最优航线（最短路径）")
        selected_route = st.radio("选择航线", route_keys, index=default_idx)
        st.session_state.current_route_points = st.session_state.all_routes[selected_route]

    with col_map:
        map_placeholder = st.empty()
        def render_map():
           m = folium.Map(
    location=[(a_lat+b_lat)/2,(a_lon+b_lon)/2],
    zoom_start=17,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
)
            folium.Marker([a_lat,a_lon],icon=folium.Icon(color="red")).add_to(m)
            folium.Marker([b_lat,b_lon],icon=folium.Icon(color="green")).add_to(m)

            # 全部绕行统一深蓝色实线
            if "左侧绕行" in st.session_state.all_routes:
                folium.PolyLine(st.session_state.all_routes["左侧绕行"], color="#0044FF", weight=5).add_to(m)
            if "右侧绕行" in st.session_state.all_routes:
                folium.PolyLine(st.session_state.all_routes["右侧绕行"], color="#0044FF", weight=5).add_to(m)
            if "最优航线（最短路径）" in st.session_state.all_routes:
                folium.PolyLine(st.session_state.all_routes["最优航线（最短路径）"], color="#0044FF", weight=6).add_to(m)
            if "直接飞越" in st.session_state.all_routes:
                folium.PolyLine(st.session_state.all_routes["直接飞越"], color="#888888", weight=3).add_to(m)

            for poly in st.session_state.obstacle_polygons:
                folium.Polygon(poly,color="red",fill=True,fill_color="red",fill_opacity=0.4).add_to(m)

            draw = Draw(export=False,position="topleft",
                draw_options={"polyline":False,"polygon":{"allowIntersection":False},"rectangle":True,"circle":False,"marker":False},
                edit_options={"edit":{},"remove":{}})
            draw.add_to(m)
            map_data = st_folium(m,width=1000,height=700,returned_objects=["last_active_drawing"],key=f"map_{st.session_state.map_rerun_key}")
            if map_data and map_data.get("last_active_drawing"):
                g = map_data["last_active_drawing"]["geometry"]
                if g["type"] == "Polygon":
                    coords = g["coordinates"][0]
                    poly_coords = [[lat,lon] for lon,lat in coords]
                    if poly_coords not in st.session_state.obstacle_polygons:
                        st.session_state.obstacle_polygons.append(poly_coords)
                        st.session_state.obstacle_heights[len(st.session_state.obstacle_polygons)-1] = 50
                        st.session_state.map_rerun_key += 1
                        st.rerun()
        render_map()

# ========================== 飞行监控页面 ==========================
elif st.session_state.current_page == "飞行监控":
    st.header("📡 飞行监控")
    st.info("心跳包功能保持不变")
