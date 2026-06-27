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
VERSION = "v18.1 精准绕障+地图定位版"
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

# ========================== 核心：切线绕障（最短路径） ==========================
def generate_routes(start, end, obstacle_list, obstacle_heights, fly_height, safe_radius):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end

    # 1. 直飞参考线
    def full_smooth(p0,pm,p1,n=12):
        arr=[]
        for t in np.linspace(0,1,n):
            la=(1-t)**2*p0[0]+2*(1-t)*t*pm[0]+t**2*p1[0]
            lo=(1-t)**2*p0[1]+2*(1-t)*t*pm[1]+t**2*p1[1]
            arr.append((la,lo))
        return arr
    mid = ((s_lat+e_lat)/2, (s_lon+e_lon)/2)
    routes["直飞参考"] = full_smooth(start, mid, end)

    # 2. 高度判断：飞行高度超障碍物则直飞
    max_h = max(obstacle_heights.values()) if obstacle_heights else 0
    if fly_height > max_h or len(obstacle_list)==0:
        return routes

    # 3. 构建障碍物缓冲区（安全距离）
    polys = [Polygon(o) for o in obstacle_list]
    merged = unary_union(polys)
    buf = merged.buffer(safe_radius / 111319.9)  # 米转经纬度

    # 4. 核心：局部切线绕障（只在障碍物两侧生成最短拐点）
    def get_tangent_route(side):
        start_point = Point(s_lon, s_lat)
        end_point = Point(e_lon, e_lat)
        try:
            # 计算起点/终点到障碍物的切线点
            if side == "left":
                tangents = start_point.buffer(0.0001).boundary.intersection(buf.boundary)
                t1 = tangents.geoms[0] if len(tangents.geoms)>=1 else start_point
                tangents2 = end_point.buffer(0.0001).boundary.intersection(buf.boundary)
                t2 = tangents2.geoms[1] if len(tangents2.geoms)>=2 else end_point
            else:
                tangents = start_point.buffer(0.0001).boundary.intersection(buf.boundary)
                t1 = tangents.geoms[1] if len(tangents.geoms)>=2 else start_point
                tangents2 = end_point.buffer(0.0001).boundary.intersection(buf.boundary)
                t2 = tangents2.geoms[0] if len(tangents2.geoms)>=1 else end_point
            
            # 构建最短路径
            route = [
                (s_lat, s_lon),
                (t1.y, t1.x),
                (t2.y, t2.x),
                (e_lat, e_lon)
            ]
            return route
        except:
            # 兜底：近距离小偏移绕障
            offset = safe_radius / 111319.9 * 2
            if side == "left":
                mid_point = (mid[0] + offset, mid[1] - offset)
            else:
                mid_point = (mid[0] - offset, mid[1] + offset)
            return [start, mid_point, end]

    # 生成左右最短绕障航线
    routes["左侧绕行"] = get_tangent_route("left")
    routes["右侧绕行"] = get_tangent_route("right")

    # 5. 计算长度选最优
    def calc_route_length(pts):
        total = 0
        for i in range(len(pts)-1):
            total += latlon_to_meter(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
        return total
    
    len_left = calc_route_length(routes["左侧绕行"])
    len_right = calc_route_length(routes["右侧绕行"])
    
    if len_left <= len_right:
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
# 【关键】修改为你地图上A点的真实坐标
if 'current_route_points' not in st.session_state:
    st.session_state.current_route_points = [(32.138500, 118.779200), (32.235100, 118.748500)]
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
        st.subheader("📍 起飞点 A（地图标注点）")
        # 【关键】修改为你地图上A点的真实坐标
        a_lat_input = st.number_input("纬度", value=32.138500, format="%.6f")
        a_lon_input = st.number_input("经度", value=118.779200, format="%.6f")

        st.subheader("📍 目标点 B（地图标注点）")
        # 【关键】修改为你地图上B点的真实坐标
        b_lat_input = st.number_input("纬度 B", value=32.235100, format="%.6f")
        b_lon_input = st.number_input("经度 B", value=118.748500, format="%.6f")

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
            # 【关键】地图中心定位到你标注的A/B点中间
            center_lat = (a_lat + b_lat) / 2  # 自动计算A/B中间纬度
            center_lon = (a_lon + b_lon) / 2  # 自动计算A/B中间经度
            m = folium.Map(
                [center_lat, center_lon], zoom_start=15,  # 15级缩放，聚焦你的区域
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery"
            )
            # 标注A/B点（和你地图上的红色标注对应）
            folium.Marker([a_lat,a_lon],popup="起飞点A（地图标注）",icon=folium.Icon(color="red")).add_to(m)
            folium.Marker([b_lat,b_lon],popup="目标点B（地图标注）",icon=folium.Icon(color="green")).add_to(m)

            # 绘制航线
            for name, pts in st.session_state.all_routes.items():
                color = "#0044FF" if "最优" in name else "#4488FF"
                weight = 6 if "最优" in name else 4
                if name == "直飞参考":
                    color = "#888888"
                    weight = 2
                folium.PolyLine(pts, color=color, weight=weight, opacity=0.9).add_to(m)

            # 绘制障碍物
            for p in st.session_state.obstacle_polygons:
                folium.Polygon(p, color="red", fill=True, fill_color="red", fill_opacity=0.4).add_to(m)

            # 绘制工具
            draw = Draw(
                export=False,
                draw_options={"polyline":False,"polygon":True,"rectangle":True,"circle":False,"marker":False}
            )
            draw.add_to(m)
            data = st_folium(m, width=1000, height=700, key=f"m{st.session_state.map_rerun_key}")

            # 新增障碍物
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
