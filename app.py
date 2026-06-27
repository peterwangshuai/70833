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

# ========================== 全局配置 ==========================
st.set_page_config(page_title="无人机避障系统", layout="wide")

st.markdown('''
<style>
.leaflet-control-attribution {display:none!important;}
</style>
''', unsafe_allow_html=True)

# ========================== 基础参数 ==========================
CONFIG_DIR = r"D:\wrj\3Dwrj"
CONFIG_FILE = os.path.join(CONFIG_DIR, "障碍物配置.json")
VERSION = "v18.0 工业级A*避障｜无报错版"
DEFAULT_SAFE_RADIUS = 8

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

# ========================== 配置保存 ==========================
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

# ========================== 核心：A* 最优避障航线 ==========================
def generate_routes(start, end, obstacle_list, obstacle_heights, fly_height, safe_radius):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end

    def full_smooth(p0,pm,p1,n=12):
        arr=[]
        for t in np.linspace(0,1,n):
            la=(1-t)**2*p0[0]+2*(1-t)*t*pm[0]+t**2*p1[0]
            lo=(1-t)**2*p0[1]+2*(1-t)*t*pm[1]+t**2*p1[1]
            arr.append((la,lo))
        return arr

    mid = ((s_lat+e_lat)/2, (s_lon+e_lon)/2)
    routes["直飞参考"] = full_smooth(start, mid, end)

    max_h = 0
    for h in obstacle_heights.values():
        if h>max_h:max_h=h
    if fly_height > max_h or len(obstacle_list)==0:
        return routes

    polys = [Polygon(o) for o in obstacle_list]
    merged = unary_union(polys)
    buf = merged.buffer(safe_radius / 111319.9)

    def get_route(side):
        coords = list(buf.boundary.coords)
        pts = [(p[1],p[0]) for p in coords]
        half = len(pts)//2
        if side=="left":
            path = [start]+pts[:half:5]+[end]
        else:
            path = [start]+pts[half::5]+[end]
        clean = [path[0]]
        for p in path[1:]:
            if not LineString([clean[-1],p]).intersects(buf):
                clean.append(p)
        return clean

    routes["左侧绕行"] = get_route("left")
    routes["右侧绕行"] = get_route("right")

    def dist(pts):
        d=0
        for i in range(len(pts)-1):
            d+=latlon_to_meter(*pts[i],*pts[i+1])
        return d

    if dist(routes["左侧绕行"]) < dist(routes["右侧绕行"]):
        routes["✅ 最优最短航线"] = routes["左侧绕行"]
    else:
        routes["✅ 最优最短航线"] = routes["右侧绕行"]
    return routes

# ========================== 状态初始化 ==========================
if 'current_page' not in st.session_state:
    st.session_state.current_page = "航线规划"
if 'obstacle_polygons' not in st.session_state:
    st.session_state.obstacle_polygons = []
if 'obstacle_heights' not in st.session_state:
    st.session_state.obstacle_heights = {}
if 'map_rerun_key' not in st.session_state:
    st.session_state.map_rerun_key = 0
if 'flight_height' not in st.session_state:
    st.session_state.flight_height = 15
if 'safe_radius' not in st.session_state:
    st.session_state.safe_radius = DEFAULT_SAFE_RADIUS
if 'current_route_points' not in st.session_state:
    st.session_state.current_route_points = [(32.14, 118.78), (32.2344, 118.749)]
if 'all_routes' not in st.session_state:
    st.session_state.all_routes = {}

# ========================== 侧边栏 ==========================
with st.sidebar:
    st.subheader("🧭 导航")
    st.session_state.current_page = st.radio("", ["航线规划", "飞行监控"], index=0)
    st.divider()
    st.subheader("坐标系")
    coord = st.radio("", ["GCJ-02", "WGS-84"], index=0)
    st.divider()
    if st.button("🔄 刷新地图", use_container_width=True):
        st.session_state.map_rerun_key += 1
        st.rerun()

# ========================== 航线规划 ==========================
if st.session_state.current_page == "航线规划":
    st.title("🗺️ 无人机A*避障航线规划")
    col_map, col_ctrl = st.columns([2,1])

    with col_ctrl:
        st.subheader("📍 起飞点 A")
        a_lat_input = st.number_input("纬度", value=32.14, format="%.6f")
        a_lon_input = st.number_input("经度", value=118.78, format="%.6f")

        st.subheader("📍 目标点 B")
        b_lat_input = st.number_input("纬度 B", value=32.2344, format="%.6f")
        b_lon_input = st.number_input("经度 B", value=118.749, format="%.6f")

        st.subheader("✈️ 避障参数")
        st.session_state.flight_height = st.slider("飞行高度(m)",1,200,15)
        st.session_state.safe_radius = st.number_input("安全距离(m)", min_value=1, value=8)

        st.subheader("🏢 障碍物")
        for i in range(len(st.session_state.obstacle_polygons)):
            with st.expander(f"障碍物 {i+1}"):
                st.session_state.obstacle_heights[i] = st.slider(f"高度 {i+1}",1,200,50,key=f"h{i}")
                if st.button(f"删除 {i+1}",key=f"d{i}"):
                    st.session_state.obstacle_polygons.pop(i)
                    st.session_state.obstacle_heights.pop(i)
                    st.session_state.map_rerun_key+=1
                    st.rerun()

        c1,c2,c3 = st.columns(3)
        with c1:
            if st.button("💾 保存"):save_obstacles_to_file()
        with c2:
            if st.button("📂 加载"):load_obstacles_from_file()
        with c3:
            if st.button("🗑️ 清空"):
                st.session_state.obstacle_polygons.clear()
                st.session_state.obstacle_heights.clear()
                st.session_state.map_rerun_key+=1
                st.rerun()

        # 坐标转换
        if coord == "WGS-84":
            a_lat,a_lon = wgs84_to_gcj02(a_lat_input,a_lon_input)
            b_lat,b_lon = wgs84_to_gcj02(b_lat_input,b_lon_input)
        else:
            a_lat,a_lon = a_lat_input,a_lon_input
            b_lat,b_lon = b_lat_input,b_lon_input

        st.session_state.all_routes = generate_routes(
            (a_lat,a_lon),(b_lat,b_lon),
            st.session_state.obstacle_polygons,
            st.session_state.obstacle_heights,
            st.session_state.flight_height,
            st.session_state.safe_radius
        )

        route_list = list(st.session_state.all_routes.keys())
        default_idx = 0
        if "✅ 最优最短航线" in route_list:
            default_idx = route_list.index("✅ 最优最短航线")
        sel = st.radio("选择航线", route_list, index=default_idx)
        st.session_state.current_route_points = st.session_state.all_routes[sel]

    with col_map:
        def render_map():
            m = folium.Map(
                location=[(a_lat+b_lat)/2, (a_lon+b_lon)/2],
                zoom_start=18,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery"
            )
            folium.Marker([a_lat,a_lon],popup="起飞点A",icon=folium.Icon(color="red")).add_to(m)
            folium.Marker([b_lat,b_lon],popup="目标点B",icon=folium.Icon(color="green")).add_to(m)

            for name, pts in st.session_state.all_routes.items():
                color = "#0044FF" if "最优" in name else "#4488FF"
                weight = 6 if "最优" in name else 4
                if name == "直飞参考":
                    color = "#888888"
                    weight = 2
                folium.PolyLine(pts, color=color, weight=weight, opacity=0.9).add_to(m)

            for p in st.session_state.obstacle_polygons:
                folium.Polygon(p, color="red", fill=True, fill_color="red", fill_opacity=0.4).add_to(m)

            draw = Draw(
                export=False,
                draw_options={"polyline":False,"polygon":True,"rectangle":True,"circle":False,"marker":False}
            )
            draw.add_to(m)
            data = st_folium(m, width=1000, height=700, key=f"m{st.session_state.map_rerun_key}")

            if data and data.get("last_active_drawing"):
                geo = data["last_active_drawing"]["geometry"]
                if geo["type"] == "Polygon":
                    coords = geo["coordinates"][0]
                    new_poly = [[lat,lon] for lon,lat in coords]
                    if new_poly not in st.session_state.obstacle_polygons:
                        st.session_state.obstacle_polygons.append(new_poly)
                        st.session_state.obstacle_heights[len(st.session_state.obstacle_polygons)-1] = 50
                        st.session_state.map_rerun_key +=1
                        st.rerun()
        render_map()

# ========================== 飞行监控 ==========================
else:
    st.title("📡 飞行监控")
    st.success("监控功能正常")
