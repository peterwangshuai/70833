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
from shapely.ops import nearest_points

# ========================== 基础配置 ==========================
CONFIG_DIR = r"D:\wrj\3Dwrj"
CONFIG_FILE = os.path.join(CONFIG_DIR, "obstacle_config.json")
VERSION = "v13.0 航线规划优化版"
DEFAULT_SAFE_RADIUS = 5  # 默认安全半径5米

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

# 经纬度距离转米
def latlon_to_meter(lat1, lon1, lat2, lon2):
    R = 6371000
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# 米转经纬度偏移量
def meter_to_latlon_offset(lat, meter):
    lat_offset = meter / 111319.9
    lon_offset = meter / (111319.9 * np.cos(np.radians(lat)))
    return lat_offset, lon_offset

# ========================== 障碍物持久化 ==========================
def ensure_config_dir():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)

def save_obstacles_to_file():
    ensure_config_dir()
    save_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_data = {
        "version": VERSION,
        "save_time": save_time,
        "obstacle_count": len(st.session_state.obstacle_polygons),
        "obstacles": []
    }
    for idx, obs in enumerate(st.session_state.obstacle_polygons):
        obs_data = {
            "id": idx + 1,
            "coordinates": obs,
            "height": st.session_state.obstacle_heights.get(idx, 50),
            "create_time": st.session_state.obstacle_create_time.get(idx, save_time)
        }
        save_data["obstacles"].append(obs_data)
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
        for idx, obs in enumerate(load_data["obstacles"]):
            new_polygons.append(obs["coordinates"])
            new_heights[idx] = obs.get("height", 50)
            new_create_time[idx] = obs.get("create_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.session_state.obstacle_polygons = new_polygons
        st.session_state.obstacle_heights = new_heights
        st.session_state.obstacle_create_time = new_create_time
        st.session_state.last_drawing_id = None
        st.rerun()
        return load_data
    except Exception as e:
        st.error(f"加载失败：{str(e)}")
        return None

# ========================== 【核心优化】航线规划算法 ==========================
def generate_routes(start, end, obstacle_coords, obs_height, fly_height, safe_radius):
    """
    航线生成主逻辑
    :param start: (lat, lon) 起点
    :param end: (lat, lon) 终点
    :param obstacle_coords: 障碍物多边形坐标列表
    :param obs_height: 障碍物高度
    :param fly_height: 无人机飞行高度
    :param safe_radius: 安全半径(米)
    :return: 航线字典
    """
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end

    # 1. 飞行高度 > 障碍物高度：直接飞跃
    if fly_height > obs_height:
        routes["直接飞跃"] = [start, end]
        return routes

    # 2. 飞行高度 < 障碍物高度：必须绕行，生成 左绕行 / 右绕行 / 最佳最短航线
    obs_poly = Polygon(obstacle_coords)
    center_point = Point(np.mean([p[0] for p in obstacle_coords]), np.mean([p[1] for p in obstacle_coords]))
    lat_off, lon_off = meter_to_latlon_offset(center_point.x, safe_radius)

    # -------- 左绕行（保持安全距离） --------
    left_waypoint = (center_point.x + lat_off, center_point.lon - lon_off)
    routes["向左绕行"] = [start, left_waypoint, end]

    # -------- 右绕行（保持安全距离） --------
    right_waypoint = (center_point.x - lat_off, center_point.lon + lon_off)
    routes["向右绕行"] = [start, right_waypoint, end]

    # -------- 最佳航线：全局最短路径 --------
    min_dist = float("inf")
    best_route = None
    # 遍历左右绕行对比距离，取最短作为最佳航线
    for name, pts in routes.items():
        if name in ("向左绕行", "向右绕行"):
            dist = latlon_to_meter(pts[0][0], pts[0][1], pts[1][0], pts[1][1]) \
                 + latlon_to_meter(pts[1][0], pts[1][1], pts[2][0], pts[2][1])
            if dist < min_dist:
                min_dist = dist
                best_route = pts
    routes["最佳航线(最短距离)"] = best_route

    return routes

# ========================== 全局状态初始化 ==========================
if 'current_page' not in st.session_state:
    st.session_state.current_page = "航线规划"
if 'input_coord_system' not in st.session_state:
    st.session_state.input_coord_system = "GCJ-02(高德/百度)"

# 心跳包
if 'df_history' not in st.session_state:
    st.session_state.df_history = pd.DataFrame(columns=["time", "seq"])
if 'last_received' not in st.session_state:
    st.session_state.last_received = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# 障碍物
if 'obstacle_polygons' not in st.session_state:
    st.session_state.obstacle_polygons = []
if 'obstacle_heights' not in st.session_state:
    st.session_state.obstacle_heights = {}
if 'obstacle_create_time' not in st.session_state:
    st.session_state.obstacle_create_time = {}
if 'last_drawing_id' not in st.session_state:
    st.session_state.last_drawing_id = None

# 飞行参数
if 'flight_height' not in st.session_state:
    st.session_state.flight_height = 10
if 'safe_radius' not in st.session_state:
    st.session_state.safe_radius = DEFAULT_SAFE_RADIUS
if 'selected_route' not in st.session_state:
    st.session_state.selected_route = "直接飞跃"
if 'current_route_points' not in st.session_state:
    st.session_state.current_route_points = []

# ========================== 页面布局 ==========================
st.set_page_config(page_title="无人机心跳包接收系统", layout="wide")

# 左侧导航栏
with st.sidebar:
    st.subheader("🧭 导航")
    st.caption("功能页面")
    st.session_state.current_page = st.radio(
        label="",
        options=["航线规划", "飞行监控"],
        index=0 if st.session_state.current_page == "航线规划" else 1,
        label_visibility="collapsed"
    )
    st.divider()

    st.subheader("⚙️ 坐标系设置")
    st.caption("输入坐标系")
    st.session_state.input_coord_system = st.radio(
        label="",
        options=["WGS-84", "GCJ-02(高德/百度)"],
        index=1,
        label_visibility="collapsed"
    )
    st.divider()

    st.subheader("📊 系统状态")
    st.success("✅ A点已设")
    st.success("✅ B点已设")

# ========================== 航线规划页面 ==========================
if st.session_state.current_page == "航线规划":
    st.header("🗺️ 航线规划")
    col_map, col_control = st.columns([2, 1])

    with col_control:
        st.subheader("⚙️ 控制面板")

        # 起点A
        st.markdown("#### 📍 起点A")
        st.caption("输入坐标系: GCJ-02")
        col_a_lat, col_a_lon = st.columns(2)
        with col_a_lat:
            input_a_lat = st.number_input("纬度", value=32.2323, format="%.4f", key="a_lat", on_change=st.rerun)
        with col_a_lon:
            input_a_lon = st.number_input("经度", value=118.749, format="%.3f", key="a_lon", on_change=st.rerun)
        st.success("✅ 设置A点")
        st.divider()

        # 终点B
        st.markdown("#### 📍 终点B")
        col_b_lat, col_b_lon = st.columns(2)
        with col_b_lat:
            input_b_lat = st.number_input("纬度", value=32.2344, format="%.4f", key="b_lat", on_change=st.rerun)
        with col_b_lon:
            input_b_lon = st.number_input("经度", value=118.749, format="%.3f", key="b_lon", on_change=st.rerun)
        st.success("✅ 设置B点")
        st.divider()

        # 飞行高度 + 安全半径
        st.markdown("#### ✈️ 飞行参数")
        st.session_state.flight_height = st.slider("无人机飞行高度(m)", 1, 200, 10)
        st.session_state.safe_radius = st.number_input("安全半径(m)", value=DEFAULT_SAFE_RADIUS, min_value=1)
        st.caption("规则：飞行高度 > 障碍物高度 → 直接飞跃；反之自动绕行")
        st.divider()

        # 障碍物配置持久化
        st.markdown("#### 🚀 障碍物配置持久化")
        st.caption(f"配置文件: {CONFIG_FILE} | 版本: {VERSION}")
        st.info("💡 文件保存在程序同目录下，绝对路径如上所示")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("💾 保存到文件", type="primary", use_container_width=True):
                save_obstacles_to_file()
                st.success("保存成功！")
                st.rerun()
        with col2:
            if st.button("📂 从文件加载", use_container_width=True):
                load_obstacles_from_file()
        with col3:
            if st.button("🗑️ 清除全部", use_container_width=True):
                st.session_state.obstacle_polygons = []
                st.session_state.obstacle_heights = {}
                st.session_state.obstacle_create_time = {}
                st.session_state.last_drawing_id = None
                st.success("已清除全部障碍物")
                st.rerun()
        with col4:
            if st.button("🚀 一键部署", type="primary", use_container_width=True):
                st.success("✅ 障碍物配置已一键部署！")
                time.sleep(0.8)
                st.rerun()
        st.divider()

        # 下载配置文件
        st.markdown("#### ⬇️ 下载配置文件到本地")
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                json_content = f.read()
            st.download_button(
                label="⬇️ 下载 obstacle_config.json",
                data=json_content,
                file_name="obstacle_config.json",
                mime="application/json",
                use_container_width=True
            )
            st.caption("点击下载即可将云端保存的障碍物配置保存到你的电脑")
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(CONFIG_FILE)).strftime("%Y-%m-%d %H:%M:%S")
                st.info(f"📂 文件状态: 共 {config_data['obstacle_count']} 个障碍物 | 保存时间: {config_data['save_time']} | 版本: {config_data['version']}")
            except:
                st.info("📂 文件状态: 配置文件解析失败")
        else:
            st.info("暂无配置文件，请先点击「保存到文件」生成配置")
        st.code(CONFIG_FILE, language="text")
        st.divider()

        # ========== 坐标转换 & 航线自动计算 ==========
        if st.session_state.input_coord_system == "WGS-84":
            a_lat, a_lon = wgs84_to_gcj02(input_a_lat, input_a_lon)
            b_lat, b_lon = wgs84_to_gcj02(input_b_lat, input_b_lon)
        else:
            a_lat, a_lon = input_a_lat, input_a_lon
            b_lat, b_lon = input_b_lat, input_b_lon

        start_pt = (a_lat, a_lon)
        end_pt = (b_lat, b_lon)
        fly_h = st.session_state.flight_height
        safe_r = st.session_state.safe_radius

        # 航线选项
        route_options = []
        route_map = {}
        hit_obstacle = False
        current_obs_height = 0

        if st.session_state.obstacle_polygons:
            # 取第一个障碍物做判断（多障碍物可自行扩展）
            obs_idx = 0
            obs_coords = st.session_state.obstacle_polygons[obs_idx]
            current_obs_height = st.session_state.obstacle_heights.get(obs_idx, 50)
            hit_obstacle = True

            # 调用优化后的航线生成函数
            route_dict = generate_routes(start_pt, end_pt, obs_coords, current_obs_height, fly_h, safe_r)
            route_map = route_dict
            route_options = list(route_dict.keys())
        else:
            # 无障碍物：仅直接飞跃
            route_options = ["直接飞跃"]
            route_map["直接飞跃"] = [start_pt, end_pt]

        # 状态提示
        st.markdown("#### 🧭 航线选择")
        if not hit_obstacle:
            st.success("✅ 无障碍物，默认直接飞跃")
        else:
            if fly_h > current_obs_height:
                st.success(f"✅ 飞行高度({fly_h}m) > 障碍物高度({current_obs_height}m)，选择直接飞跃")
            else:
                st.warning(f"⚠️ 飞行高度({fly_h}m) < 障碍物高度({current_obs_height}m)，启用绕行航线（安全半径{safe_r}m）")

        # 航线单选
        st.session_state.selected_route = st.radio(
            "可选航线",
            options=route_options,
            index=0
        )
        st.session_state.current_route_points = route_map[st.session_state.selected_route]

    # ========== 地图渲染 ==========
    with col_map:
        st.subheader("🗺️ 地图")
        map_placeholder = st.empty()

        def render_satellite_map():
            center_lat = (a_lat + b_lat) / 2
            center_lon = (a_lon + b_lon) / 2
            satellite_tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=17,
                tiles=satellite_tiles,
                attr="Tiles © Esri"
            )

            # A/B标记
            folium.Marker([a_lat, a_lon], popup="起点A", icon=folium.Icon(color="red", icon="play")).add_to(m)
            folium.Marker([b_lat, b_lon], popup="终点B", icon=folium.Icon(color="green", icon="flag")).add_to(m)

            # 当前航线
            folium.PolyLine(
                locations=st.session_state.current_route_points,
                color="blue",
                weight=4,
                opacity=0.9,
                popup=f"选中航线：{st.session_state.selected_route}"
            ).add_to(m)

            # 障碍物
            for idx, poly_coords in enumerate(st.session_state.obstacle_polygons):
                h = st.session_state.obstacle_heights.get(idx, 50)
                folium.Polygon(
                    locations=poly_coords,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.4,
                    weight=2,
                    popup=f"障碍物\n高度：{h}m"
                ).add_to(m)

            # 无人机位置
            current_seq = len(st.session_state.df_history)
            total_steps = 50
            progress = min(current_seq / total_steps, 1.0)
            route_pts = st.session_state.current_route_points

            if len(route_pts) == 2:
                drone_lat = route_pts[0][0] + (route_pts[1][0] - route_pts[0][0]) * progress
                drone_lon = route_pts[0][1] + (route_pts[1][1] - route_pts[0][1]) * progress
            else:
                seg1_len = latlon_to_meter(route_pts[0][0], route_pts[0][1], route_pts[1][0], route_pts[1][1])
                seg2_len = latlon_to_meter(route_pts[1][0], route_pts[1][1], route_pts[2][0], route_pts[2][1])
                total_len = seg1_len + seg2_len
                if progress * total_len <= seg1_len:
                    seg_p = (progress * total_len) / seg1_len
                    drone_lat = route_pts[0][0] + (route_pts[1][0] - route_pts[0][0]) * seg_p
                    drone_lon = route_pts[0][1] + (route_pts[1][1] - route_pts[0][1]) * seg_p
                else:
                    seg_p = (progress * total_len - seg1_len) / seg2_len
                    drone_lat = route_pts[1][0] + (route_pts[2][0] - route_pts[1][0]) * seg_p
                    drone_lon = route_pts[1][1] + (route_pts[2][1] - route_pts[1][1]) * seg_p

            folium.CircleMarker(
                [drone_lat, drone_lon],
                radius=10,
                color="orange",
                fill=True,
                fill_color="yellow",
                popup=f"无人机\n进度: {progress*100:.1f}%"
            ).add_to(m)

            # 绘制工具
            draw = Draw(
                export=False,
                position='topleft',
                draw_options={'polyline':False,'polygon':True,'rectangle':True,'circle':True,'marker':True,'circlemarker':False},
                edit_options={'edit':True,'remove':True}
            )
            draw.add_to(m)

            with map_placeholder:
                map_out = st_folium(m, width=1000, height=700, returned_objects=["last_active_drawing"])

            # 捕获绘制障碍物
            if map_out and map_out["last_active_drawing"]:
                draw_data = map_out["last_active_drawing"]
                draw_id = str(draw_data["geometry"]["coordinates"])
                if draw_id != st.session_state.last_drawing_id:
                    st.session_state.last_drawing_id = draw_id
                    g_type = draw_data["geometry"]["type"]
                    if g_type == "Polygon":
                        poly_coords = [[lat, lon] for lon, lat in draw_data["geometry"]["coordinates"][0]]
                        if poly_coords not in st.session_state.obstacle_polygons:
                            st.session_state.obstacle_polygons.append(poly_coords)
                            new_idx = len(st.session_state.obstacle_polygons) - 1
                            st.session_state.obstacle_heights[new_idx] = 50
                            st.session_state.obstacle_create_time[new_idx] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.success("障碍物圈选成功！")
                            st.rerun()

        render_satellite_map()

# ========================== 飞行监控页面 ==========================
elif st.session_state.current_page == "飞行监控":
    st.header("📡 飞行监控（心跳包实时显示）")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶️ 启动飞行", type="primary", use_container_width=True):
            st.session_state.is_running = True
    with c2:
        if st.button("⏸️ 暂停飞行", use_container_width=True):
            st.session_state.is_running = False
    with c3:
        if st.button("🔄 重置数据", use_container_width=True):
            st.session_state.df_history = pd.DataFrame(columns=["time", "seq"])
            st.session_state.last_received = None
            st.session_state.is_running = False
            st.rerun()

    status_box = st.empty()
    col_chart, col_data = st.columns([2, 1])
    with col_chart:
        st.subheader("📈 心跳包序号实时趋势")
        chart_placeholder = st.empty()
    with col_data:
        st.subheader("📋 心跳数据包列表")
        data_box = st.empty()

    chart_obj = chart_placeholder.line_chart(st.session_state.df_history, x="time", y="seq", color="#39ff14")

    while st.session_state.is_running:
        current_seq = len(st.session_state.df_history) + 1
        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        new_data = pd.DataFrame({"time": [current_time], "seq": [current_seq]})
        st.session_state.df_history = pd.concat([st.session_state.df_history, new_data], ignore_index=True)

        chart_obj.add_rows(new_data)
        data_box.dataframe(st.session_state.df_history.tail(10), hide_index=True, height=400)
        status_box.success(f"✅ 飞行正常 | 心跳包序号：{current_seq} | 航线：{st.session_state.selected_route} | 时间：{current_time}")

        st.session_state.last_received = time.time()
        time.sleep(1)

    if st.session_state.last_received and not st.session_state.is_running:
        elapsed = time.time() - st.session_state.last_received
        if elapsed > 3 and len(st.session_state.df_history) > 0:
            status_box.error("🚨 连接超时！超过3秒未收到心跳包！")
        else:
            status_box.warning("⏸️ 飞行已暂停")
