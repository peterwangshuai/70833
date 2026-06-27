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
from shapely.ops import unary_union, linemerge
from shapely import offset_curve, buffer

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
VERSION = "v19.0 可视图平滑避障优化版"
DEFAULT_SAFE_RADIUS = 8

# ========================== 坐标系转换（保留原有） ==========================
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

# ========================== 配置保存（无修改） ==========================
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

# ========================== 新增工具函数：几何辅助 ==========================
def smooth_bezier(points, seg_num=15):
    """贝塞尔平滑路径点"""
    smooth_pts = []
    for i in range(len(points)-1):
        p0 = np.array(points[i])
        p1 = np.array(points[i+1])
        mid = (p0 + p1) / 2
        for t in np.linspace(0, 1, seg_num):
            la = (1-t)**2 * p0[0] + 2*(1-t)*t * mid[0] + t**2 * p1[0]
            lo = (1-t)**2 * p0[1] + 2*(1-t)*t * mid[1] + t**2 * p1[1]
            smooth_pts.append((round(la,6), round(lo,6)))
    return smooth_pts

def line_collision_check(line_points, safe_buffer):
    """检测路径是否与安全缓冲区碰撞"""
    line = LineString([(lon,lat) for lat,lon in line_points])
    return line.intersects(safe_buffer)

def calc_route_total_len(pts):
    """计算航线总米数"""
    total = 0.0
    for i in range(len(pts)-1):
        lat1, lon1 = pts[i]
        lat2, lon2 = pts[i+1]
        total += latlon_to_meter(lat1, lon1, lat2, lon2)
    return round(total, 2)

# ========================== 重构核心：可视图平滑避障算法（完全重写） ==========================
def generate_routes(start, end, obstacle_list, obstacle_heights, fly_height, safe_radius):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end
    start_pt = Point(s_lon, s_lat)
    end_pt = Point(e_lon, e_lat)

    # 1. 生成平滑直飞参考线
    raw_straight = [start, end]
    routes["直飞参考"] = smooth_bezier(raw_straight)
    straight_line = LineString([(lon,lat) for lat,lon in raw_straight])

    # 无障碍物直接返回直飞
    if len(obstacle_list) == 0:
        return routes

    # 2. 构建障碍物+安全缓冲区
    polys = []
    for coords in obstacle_list:
        poly = Polygon([(lon, lat) for lat, lon in coords])
        poly = buffer(poly, 0.0) # 修复拓扑自相交
        polys.append(poly)
    merged_obstacle = unary_union(polys)
    buf_meter = safe_radius
    buf_lat, buf_lon = meter_to_latlon_offset((s_lat+e_lat)/2, buf_meter)
    buf_deg = max(buf_lat, buf_lon)
    safe_buffer = merged_obstacle.buffer(buf_deg, join_style="round", quad_segs=12)

    # 3. 高度判断：飞行高度超过所有障碍物，直接直飞
    max_obs_h = max(obstacle_heights.values())
    if fly_height > max_obs_h:
        routes["✅ 最优越障航线"] = routes["直飞参考"]
        return routes

    # 4. 直飞碰撞检测，无碰撞则最优直飞
    if not straight_line.intersects(safe_buffer):
        routes["✅ 最优直飞航线"] = routes["直飞参考"]
        return routes

    # ====================== 可视图构建：生成左右平滑绕行 ======================
    # 提取障碍物缓冲边界顶点
    buffer_bound = safe_buffer.boundary
    verts = list(buffer_bound.coords)
    vis_points = [(p[1], p[0]) for p in verts] # (lat, lon)

    def build_smooth_detour(offset_side=1):
        """
        offset_side=1 左侧绕行，-1右侧绕行
        沿缓冲区边缘生成平滑贴边路径
        """
        try:
            # 起点到缓冲边界偏移曲线
            seg_start = LineString([(s_lon, s_lat), (e_lon, e_lat)])
            offset_dist = buf_deg * offset_side
            offset_line = offset_curve(buffer_bound, offset_dist, quad_segs=12, join_style="round")
            if offset_line.is_empty:
                raise Exception("偏移曲线为空")
            # 提取绕行轮廓点
            detour_coords = list(offset_line.coords)
            detour_latlon = [(p[1], p[0]) for p in detour_coords]
            # 构建完整路径：起点→绕行轮廓→终点
            raw_path = [start] + detour_latlon + [end]
            # 平滑插值
            smooth_path = smooth_bezier(raw_path, seg_num=10)
            # 过滤重复点
            clean_path = []
            prev = None
            for p in smooth_path:
                if p != prev:
                    clean_path.append(p)
                    prev = p
            return clean_path
        except Exception as err:
            # 兜底圆弧绕行
            mid_lat = (s_lat + e_lat) / 2
            mid_lon = (s_lon + e_lon) / 2
            off_lat, off_lon = meter_to_latlon_offset(mid_lat, safe_radius * 1.8)
            if offset_side == 1:
                bypass_pt = (mid_lat + off_lat, mid_lon - off_lon)
            else:
                bypass_pt = (mid_lat - off_lat, mid_lon + off_lon)
            raw = [start, bypass_pt, end]
            return smooth_bezier(raw)

    # 生成左右绕行航线
    left_route = build_smooth_detour(offset_side=1)
    right_route = build_smooth_detour(offset_side=-1)
    routes["左侧平滑绕行"] = left_route
    routes["右侧平滑绕行"] = right_route

    # 5. 计算两种绕行里程，选择平面最优
    len_left = calc_route_total_len(left_route)
    len_right = calc_route_total_len(right_route)
    plane_opt = left_route if len_left <= len_right else right_route
    plane_opt_name = "左侧平滑绕行" if len_left <= len_right else "右侧平滑绕行"
    routes["✅ 平面最优绕行"] = plane_opt

    # 6. 对比「平面绕行」vs「爬升越障」，生成综合最优
    climb_len = calc_route_total_len(routes["直飞参考"])
    plane_len = calc_route_total_len(plane_opt)
    climb_cost = climb_len * 0.8 + max_obs_h - fly_height # 爬升高度代价
    plane_cost = plane_len * 1.0

    if climb_cost < plane_cost:
        routes["🏔️ 综合最优(越障)"] = routes["直飞参考"]
    else:
        routes["🏔️ 综合最优(绕行)"] = plane_opt

    return routes

# ========================== 会话状态初始化 ==========================
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
    st.title("🗺️ 无人机可视图平滑避障航线规划系统")
    col_map, col_ctrl = st.columns([2, 1], gap="medium")

    with col_ctrl:
        st.subheader("📍 起飞点 A")
        a_lat = st.number_input("纬度", value=32.138500, format="%.6f", key="a_lat")
        a_lon = st.number_input("经度", value=118.779200, format="%.6f", key="a_lon")

        st.subheader("📍 目标点 B")
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
        if st.session_state.obstacle_polygons:
            for idx in range(len(st.session_state.obstacle_polygons)):
                with st.expander(f"障碍物 {idx+1}", expanded=False):
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

        # 航线选择+里程展示
        route_options = list(st.session_state.all_routes.keys())
        # 自动选中综合最优航线
        opt_key = None
        for k in route_options:
            if "综合最优" in k:
                opt_key = k
                break
        default_idx = route_options.index(opt_key) if opt_key else 0
        selected_route = st.radio(
            "📝 选择航线",
            route_options,
            index=default_idx,
            key="route_sel"
        )
        st.session_state.current_route_points = st.session_state.all_routes[selected_route]

        # 每条航线里程展示
        st.divider()
        st.subheader("📏 航线里程一览")
        for name, pts in st.session_state.all_routes.items():
            dist = calc_route_total_len(pts)
            st.write(f"{name}：{dist} 米")

    with col_map:
        def render_map():
            map_center_lat = (start_lat + end_lat) / 2
            map_center_lon = (start_lon + end_lon) / 2

            m = folium.Map(
                location=[map_center_lat, map_center_lon],
                zoom_start=13,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery"
            )

            # A/B标记
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

            # 绘制障碍物安全缓冲区（新增）
            if st.session_state.obstacle_polygons:
                polys = []
                for coords in st.session_state.obstacle_polygons:
                    poly = Polygon([(lon, lat) for lat, lon in coords])
                    polys.append(poly)
                merged = unary_union(polys)
                buf_lat, buf_lon = meter_to_latlon_offset(map_center_lat, st.session_state.safe_radius)
                buf_deg = max(buf_lat, buf_lon)
                safe_buf = merged.buffer(buf_deg, join_style="round")
                # 绘制安全缓冲虚线
                folium.Polygon(
                    locations=[(p[1], p[0]) for p in safe_buf.exterior.coords],
                    color="orange", fill=False, dash_array="8,8", weight=2,
                    popup="安全隔离缓冲区"
                ).add_to(m)

            # 绘制障碍物本体
            for idx, poly_coords in enumerate(st.session_state.obstacle_polygons):
                h = st.session_state.obstacle_heights.get(idx, 50)
                folium.Polygon(
                    poly_coords,
                    color="red", fill_color="red", fill_opacity=0.4, weight=3,
                    popup=f"障碍物 {idx+1} | 高度:{h}m"
                ).add_to(m)

            # 航线样式
            route_styles = {
                "直飞参考": {"color": "#888888", "weight": 2, "dashArray": "5,5"},
                "左侧平滑绕行": {"color": "#4488FF", "weight": 4},
                "右侧平滑绕行": {"color": "#4488FF", "weight": 4},
                "✅ 平面最优绕行": {"color": "#0066FF", "weight": 5},
                "🏔️ 综合最优(越障)": {"color": "#00AA00", "weight": 6},
                "🏔️ 综合最优(绕行)": {"color": "#0044FF", "weight": 6},
                "✅ 最优直飞航线": {"color": "#00AA00", "weight": 6},
                "✅ 最优越障航线": {"color": "#00AA00", "weight": 6},
            }
            for route_name, route_pts in st.session_state.all_routes.items():
                style = route_styles.get(route_name, {"color": "#0000FF", "weight": 4})
                folium.PolyLine(
                    route_pts,
                    color=style["color"],
                    weight=style["weight"],
                    dash_array=style.get("dashArray", ""),
                    opacity=0.9,
                    popup=f"{route_name} 距离:{calc_route_total_len(route_pts)}m"
                ).add_to(m)

            # 绘图工具
            draw = Draw(
                export=False, position="topleft",
                draw_options={
                    "polyline": False, "polygon": {"allowIntersection": False, "showArea": True},
                    "rectangle": {"showArea": True}, "circle": False, "marker": False, "circlemarker": False
                },
                edit_options={"edit": False, "remove": False}
            )
            draw.add_to(m)

            map_data = st_folium(
                m, width=1000, height=700,
                returned_objects=["last_active_drawing"],
                key=f"map_{st.session_state.map_rerun_key}"
            )

            # 新增障碍物
            if map_data and map_data.get("last_active_drawing"):
                drawing = map_data["last_active_drawing"]
                if drawing["geometry"]["type"] in ["Polygon", "Rectangle"]:
                    coords = drawing["geometry"]["coordinates"][0]
                    new_obs = [[lat, lon] for lon, lat in coords]
                    if new_obs not in st.session_state.obstacle_polygons:
                        st.session_state.obstacle_polygons.append(new_obs)
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
        st.metric("当前航线", selected_route if "selected_route" in locals() else "无")
        st.metric("航线总长", f"{calc_route_total_len(st.session_state.current_route_points)} 米")
        st.metric("障碍物数量", len(st.session_state.obstacle_polygons))
        st.metric("系统版本", VERSION)
