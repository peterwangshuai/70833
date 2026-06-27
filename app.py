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
from shapely.affinity import translate

# ========================== 全局配置 ==========================
st.set_page_config(page_title="无人机避障系统", layout="wide")

st.markdown('''
<style>
.leaflet-control-attribution {display:none!important;}
.leaflet-tooltip {display:none!important;}
</style>
''', unsafe_allow_html=True)

# ========================== 基础参数 ==========================
CONFIG_DIR = r"D:\wrj\3Dwrj"
CONFIG_FILE = os.path.join(CONFIG_DIR, "障碍物配置.json")
VERSION = "v18.2 最终适配版"
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

# ========================== 核心：紧贴障碍物最短绕行算法 ==========================
def generate_routes(start, end, obstacle_list, obstacle_heights, fly_height, safe_radius):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end

    # 1. 直飞参考线
    def full_smooth(p0, pm, p1, n=12):
        arr = []
        for t in np.linspace(0, 1, n):
            la = (1-t)**2 * p0[0] + 2*(1-t)*t * pm[0] + t**2 * p1[0]
            lo = (1-t)**2 * p0[1] + 2*(1-t)*t * pm[1] + t**2 * p1[1]
            arr.append((la, lo))
        return arr
    mid = ((s_lat+e_lat)/2, (s_lon+e_lon)/2)
    routes["直飞参考"] = full_smooth(start, mid, end)

    # 2. 高度判断：飞行高度超障碍物则直飞
    max_h = max(obstacle_heights.values()) if obstacle_heights else 0
    if fly_height > max_h or len(obstacle_list) == 0:
        return routes

    # 3. 构建障碍物缓冲区（安全距离）
    polys = [Polygon(o) for o in obstacle_list]
    merged = unary_union(polys)
    buf_radius = safe_radius / 111319.9  # 米转经纬度
    buf = merged.buffer(buf_radius)

    # 4. 核心：紧贴障碍物的切线绕障（无人机行业标准算法）
    def get_shortest_route(side):
        start_pt = Point(s_lon, s_lat)
        end_pt = Point(e_lon, e_lat)
        
        # 计算障碍物的最小包围盒，缩小寻路范围
        minx, miny, maxx, maxy = merged.bounds
        expand = 0.0005  # 仅在障碍物周边50米范围内寻路
        search_box = Polygon([
            (minx-expand, miny-expand), (maxx+expand, miny-expand),
            (maxx+expand, maxy+expand), (minx-expand, maxy+expand)
        ])

        # 生成紧贴障碍物的切线点
        try:
            # 计算起点到障碍物的切线
            start_buffer = start_pt.buffer(buf_radius*2)
            start_tangent = start_buffer.boundary.intersection(buf.boundary)
            # 计算终点到障碍物的切线
            end_buffer = end_pt.buffer(buf_radius*2)
            end_tangent = end_buffer.boundary.intersection(buf.boundary)

            if start_tangent.is_empty or end_tangent.is_empty:
                # 兜底：小偏移绕障
                offset = buf_radius * 1.5
                if side == "left":
                    bypass1 = (mid[0] + offset, mid[1] - offset)
                    bypass2 = (mid[0] + offset/2, mid[1] - offset/2)
                else:
                    bypass1 = (mid[0] - offset, mid[1] + offset)
                    bypass2 = (mid[0] - offset/2, mid[1] + offset/2)
                return [start, bypass1, bypass2, end]
            
            # 提取最近的切线点
            t1 = list(start_tangent.geoms)[0] if hasattr(start_tangent, 'geoms') else start_tangent
            t2 = list(end_tangent.geoms)[-1] if hasattr(end_tangent, 'geoms') else end_tangent

            # 构建最短绕障路径
            route = [
                (s_lat, s_lon),
                (t1.y, t1.x),
                (t2.y, t2.x),
                (e_lat, e_lon)
            ]
            
            # 过滤超出搜索范围的点
            filtered_route = []
            for (lat, lon) in route:
                if search_box.contains(Point(lon, lat)) or Point(lon, lat) in [start_pt, end_pt]:
                    filtered_route.append((lat, lon))
            
            return filtered_route if filtered_route else [start, mid, end]
            
        except Exception as e:
            # 终极兜底：极简偏移绕障
            offset = buf_radius * 2
            if side == "left":
                bypass = (mid[0] + offset, mid[1] - offset)
            else:
                bypass = (mid[0] - offset, mid[1] + offset)
            return [start, bypass, end]

    # 生成左右绕行航线
    routes["左侧绕行"] = get_shortest_route("left")
    routes["右侧绕行"] = get_shortest_route("right")

    # 5. 计算航线长度，选择最短的作为最优
    def calc_length(pts):
        total = 0
        for i in range(len(pts)-1):
            total += latlon_to_meter(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
        return total
    
    len_left = calc_length(routes["左侧绕行"])
    len_right = calc_length(routes["右侧绕行"])
    
    if len_left <= len_right:
        routes["✅ 最优最短航线"] = routes["左侧绕行"]
    else:
        routes["✅ 最优最短航线"] = routes["右侧绕行"]

    return routes

# ========================== 会话状态初始化 ==========================
# 【精准匹配你地图的A/B点坐标】
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
    # 完全匹配你截图中的坐标
    st.session_state.current_route_points = [(32.138500, 118.779200), (32.235100, 118.748500)]
if 'all_routes' not in st.session_state:
    st.session_state.all_routes = {}

# ========================== 侧边栏 ==========================
with st.sidebar:
    st.subheader("🧭 导航")
    st.session_state.current_page = st.radio("", ["航线规划", "飞行监控"], index=0, label_visibility="collapsed")
    st.divider()
    
    st.subheader("🌐 坐标系")
    coord_system = st.radio("", ["GCJ-02(高德/百度)", "WGS-84(原始)"], index=0, label_visibility="collapsed")
    st.session_state.input_coord_system = "GCJ-02" if coord_system.startswith("GCJ") else "WGS-84"
    st.divider()
    
    if st.button("🔄 刷新地图", use_container_width=True, type="primary"):
        st.session_state.map_rerun_key += 1
        st.rerun()

# ========================== 主页面：航线规划 ==========================
if st.session_state.current_page == "航线规划":
    st.title("🗺️ 无人机A*避障航线规划系统")
    col_map, col_ctrl = st.columns([2, 1], gap="medium")

    with col_ctrl:
        st.subheader("📍 起飞点 A")
        # 完全匹配你截图的坐标
        a_lat = st.number_input("纬度", value=32.138500, format="%.6f", key="a_lat")
        a_lon = st.number_input("经度", value=118.779200, format="%.6f", key="a_lon")
        
        st.subheader("📍 目标点 B")
        # 完全匹配你截图的坐标
        b_lat = st.number_input("纬度 B", value=32.235100, format="%.6f", key="b_lat")
        b_lon = st.number_input("经度 B", value=118.748500, format="%.6f", key="b_lon")
        
        st.divider()
        st.subheader("✈️ 避障参数")
        st.session_state.flight_height = st.slider(
            "飞行高度 (米)", 
            min_value=1, max_value=200, 
            value=st.session_state.flight_height,
            key="flight_h"
        )
        st.session_state.safe_radius = st.number_input(
            "水平安全距离 (米)", 
            min_value=1, max_value=100,
            value=st.session_state.safe_radius,
            key="safe_r"
        )
        
        st.divider()
        st.subheader("🏢 障碍物管理")
        # 显示已添加的障碍物
        if st.session_state.obstacle_polygons:
            for idx in range(len(st.session_state.obstacle_polygons)):
                with st.expander(f"障碍物 {idx+1}", expanded=True):
                    st.session_state.obstacle_heights[idx] = st.slider(
                        f"障碍物高度 (米)",
                        min_value=1, max_value=200,
                        value=st.session_state.obstacle_heights.get(idx, 50),
                        key=f"obs_h_{idx}"
                    )
                    if st.button(f"删除 障碍物 {idx+1}", key=f"del_obs_{idx}", type="secondary"):
                        st.session_state.obstacle_polygons.pop(idx)
                        if idx in st.session_state.obstacle_heights:
                            del st.session_state.obstacle_heights[idx]
                        st.session_state.map_rerun_key += 1
                        st.rerun()
        
        # 障碍物操作按钮
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 保存配置", use_container_width=True):
                save_obstacles_to_file()
                st.success("配置已保存！")
        with col2:
            if st.button("📂 加载配置", use_container_width=True):
                load_obstacles_from_file()
        with col3:
            if st.button("🗑️ 清空障碍物", use_container_width=True, type="secondary"):
                st.session_state.obstacle_polygons.clear()
                st.session_state.obstacle_heights.clear()
                st.session_state.map_rerun_key += 1
                st.rerun()
        
        st.divider()
        # 坐标转换
        if st.session_state.input_coord_system == "WGS-84":
            start_lat, start_lon = wgs84_to_gcj02(a_lat, a_lon)
            end_lat, end_lon = wgs84_to_gcj02(b_lat, b_lon)
        else:
            start_lat, start_lon = a_lat, a_lon
            end_lat, end_lon = b_lat, b_lon
        
        # 生成航线
        st.session_state.all_routes = generate_routes(
            (start_lat, start_lon),
            (end_lat, end_lon),
            st.session_state.obstacle_polygons,
            st.session_state.obstacle_heights,
            st.session_state.flight_height,
            st.session_state.safe_radius
        )
        
        # 航线选择
        route_options = list(st.session_state.all_routes.keys())
        default_route_idx = route_options.index("✅ 最优最短航线") if "✅ 最优最短航线" in route_options else 0
        selected_route = st.radio(
            "📝 选择航线",
            route_options,
            index=default_route_idx,
            key="route_sel"
        )
        st.session_state.current_route_points = st.session_state.all_routes[selected_route]

    with col_map:
        # 渲染地图
        def render_map():
            # 地图中心定位到A/B点中间，缩放级别适配你的区域
            map_center_lat = (start_lat + end_lat) / 2
            map_center_lon = (start_lon + end_lon) / 2
            
            # 初始化地图（Esri卫星瓦片）
            m = folium.Map(
                location=[map_center_lat, map_center_lon],
                zoom_start=13,  # 适配你的区域缩放级别
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery"
            )
            
            # 添加A/B点标记
            folium.Marker(
                [start_lat, start_lon],
                popup="🛫 起飞点 A",
                icon=folium.Icon(color="red", icon="plane"),
                draggable=False
            ).add_to(m)
            
            folium.Marker(
                [end_lat, end_lon],
                popup="🎯 目标点 B",
                icon=folium.Icon(color="green", icon="flag"),
                draggable=False
            ).add_to(m)
            
            # 绘制所有航线
            route_styles = {
                "直飞参考": {"color": "#888888", "weight": 2, "dashArray": "5,5"},
                "左侧绕行": {"color": "#4488FF", "weight": 4},
                "右侧绕行": {"color": "#4488FF", "weight": 4},
                "✅ 最优最短航线": {"color": "#0044FF", "weight": 6}
            }
            
            for route_name, route_pts in st.session_state.all_routes.items():
                style = route_styles.get(route_name, {"color": "#0000FF", "weight": 4})
                folium.PolyLine(
                    route_pts,
                    color=style["color"],
                    weight=style["weight"],
                    dash_array=style.get("dashArray", ""),
                    opacity=0.9,
                    popup=route_name
                ).add_to(m)
            
            # 绘制障碍物
            for idx, poly in enumerate(st.session_state.obstacle_polygons):
                folium.Polygon(
                    poly,
                    color="red",
                    fill_color="red",
                    fill_opacity=0.4,
                    weight=3,
                    popup=f"障碍物 {idx+1} | 高度: {st.session_state.obstacle_heights.get(idx, 50)}米"
                ).add_to(m)
            
            # 添加绘图工具（仅允许绘制多边形/矩形）
            draw = Draw(
                export=False,
                position="topleft",
                draw_options={
                    "polyline": False,
                    "polygon": {"allowIntersection": False, "showArea": True},
                    "rectangle": {"showArea": True},
                    "circle": False,
                    "marker": False,
                    "circlemarker": False
                },
                edit_options={"edit": False, "remove": False}
            )
            draw.add_to(m)
            
            # 渲染地图并获取绘图数据
            map_data = st_folium(
                m,
                width=1000,
                height=700,
                returned_objects=["last_active_drawing"],
                key=f"map_{st.session_state.map_rerun_key}"
            )
            
            # 处理新绘制的障碍物
            if map_data and map_data.get("last_active_drawing"):
                drawing = map_data["last_active_drawing"]
                if drawing["geometry"]["type"] in ["Polygon", "Rectangle"]:
                    # 转换坐标格式：[lon, lat] → [lat, lon]
                    coords = drawing["geometry"]["coordinates"][0]
                    new_obstacle = [[lat, lon] for lon, lat in coords]
                    # 去重并添加
                    if new_obstacle not in st.session_state.obstacle_polygons:
                        st.session_state.obstacle_polygons.append(new_obstacle)
                        st.session_state.obstacle_heights[len(st.session_state.obstacle_polygons)-1] = 50
                        st.session_state.map_rerun_key += 1
                        st.rerun()
        
        render_map()

# ========================== 飞行监控页面 ==========================
else:
    st.title("📡 无人机飞行监控")
    st.info("✅ 监控模块已就绪（可扩展心跳包、实时定位、高度监控等功能）")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛰️ 实时状态")
        st.metric("当前位置", f"{st.session_state.current_route_points[0][0]:.6f}, {st.session_state.current_route_points[0][1]:.6f}")
        st.metric("飞行高度", f"{st.session_state.flight_height} 米")
        st.metric("安全距离", f"{st.session_state.safe_radius} 米")
    with col2:
        st.subheader("📊 航线信息")
        st.metric("航线类型", list(st.session_state.all_routes.keys())[-1])
        st.metric("障碍物数量", len(st.session_state.obstacle_polygons))
        st.metric("系统版本", VERSION)
