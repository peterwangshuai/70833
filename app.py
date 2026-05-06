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

# ========================== 固定配置 ==========================
CONFIG_DIR = r"D:\wrj\3Dwrj"
CONFIG_FILE = os.path.join(CONFIG_DIR, "obstacle_config.json")
VERSION = "v14.0 仿界面布局版"

# ========================== 坐标系转换核心算法 ==========================
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

# ========================== 障碍物持久化工具函数 ==========================
def ensure_config_dir():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)

def save_obstacles_to_file():
    ensure_config_dir()
    save_data = {
        "version": VERSION,
        "save_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "obstacle_count": len(st.session_state.obstacle_polygons),
        "obstacles": []
    }
    for idx, obs in enumerate(st.session_state.obstacle_polygons):
        obs_data = {
            "id": idx + 1,
            "coordinates": obs,
            "height": st.session_state.obstacle_heights.get(idx, 50),
            "create_time": st.session_state.obstacle_create_time.get(idx, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
        return load_data
    except Exception as e:
        st.error(f"加载失败：{str(e)}")
        return None

# ========================== 避障航线生成核心函数 ==========================
def line_polygon_intersection(line_start, line_end, polygon_coords):
    line = LineString([line_start, line_end])
    polygon = Polygon(polygon_coords)
    return line.intersects(polygon)

def generate_detour_routes(a_point, b_point, obstacle_poly, obstacle_height, flight_height, safe_radius=5):
    a_lat, a_lon = a_point
    b_lat, b_lon = b_point
    original_line = LineString([[a_lon, a_lat], [b_lon, b_lat]])
    obstacle = Polygon(obstacle_poly)
    
    # 高度足够，直接飞跃
    if flight_height > obstacle_height:
        return {
            "原航线（直接飞跃）": [[a_lat, a_lon], [b_lat, b_lon]],
            "左绕行": None,
            "右绕行": None,
            "最佳航线": [[a_lat, a_lon], [b_lat, b_lon]]
        }
    
    # 高度不足，生成绕行航线
    obstacle_bounds = obstacle.bounds
    center_lon = (obstacle_bounds[0] + obstacle_bounds[2]) / 2
    center_lat = (obstacle_bounds[1] + obstacle_bounds[3]) / 2
    center_point = Point(center_lon, center_lat)
    
    safe_degree = safe_radius * 0.0000089932
    obstacle_radius = max(center_point.distance(Point(p[1], p[0])) for p in obstacle_poly) + safe_degree
    
    dx = b_lon - a_lon
    dy = b_lat - a_lat
    line_length = np.sqrt(dx**2 + dy**2)
    left_dx = -dy / line_length * obstacle_radius
    left_dy = dx / line_length * obstacle_radius
    right_dx = dy / line_length * obstacle_radius
    right_dy = -dx / line_length * obstacle_radius
    
    # 左/右/最佳航线
    left_route = [[a_lat, a_lon], [center_lat + left_dy, center_lon + left_dx], [b_lat, b_lon]]
    right_route = [[a_lat, a_lon], [center_lat + right_dy, center_lon + right_dx], [b_lat, b_lon]]
    left_length = LineString([[p[1], p[0]] for p in left_route]).length
    right_length = LineString([[p[1], p[0]] for p in right_route]).length
    best_route = left_route if left_length < right_length else right_route
    
    return {
        "原航线（直接飞跃）": [[a_lat, a_lon], [b_lat, b_lon]],
        "左绕行": left_route,
        "右绕行": right_route,
        "最佳航线": best_route
    }

# ========================== 全局状态初始化 ==========================
# 页面状态
if 'current_page' not in st.session_state:
    st.session_state.current_page = "航线规划"
if 'input_coord_system' not in st.session_state:
    st.session_state.input_coord_system = "GCJ-02(高德/百度)"

# 心跳包状态
if 'df_history' not in st.session_state:
    st.session_state.df_history = pd.DataFrame(columns=["time", "seq"])
if 'last_received' not in st.session_state:
    st.session_state.last_received = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# 障碍物状态
if 'obstacle_polygons' not in st.session_state:
    st.session_state.obstacle_polygons = []
if 'obstacle_heights' not in st.session_state:
    st.session_state.obstacle_heights = {}
if 'obstacle_create_time' not in st.session_state:
    st.session_state.obstacle_create_time = {}
if 'last_drawing_id' not in st.session_state:
    st.session_state.last_drawing_id = None

# 飞行参数状态
if 'flight_height' not in st.session_state:
    st.session_state.flight_height = 50
if 'safe_radius' not in st.session_state:
    st.session_state.safe_radius = 5
if 'selected_route' not in st.session_state:
    st.session_state.selected_route = "原航线（直接飞跃）"
if 'current_route_points' not in st.session_state:
    st.session_state.current_route_points = []

# ========================== 页面基础配置 ==========================
st.set_page_config(page_title="无人机心跳包接收系统", layout="wide")

# ========================== 左侧全局导航栏（完全仿照截图） ==========================
with st.sidebar:
    st.subheader("🧭 导航")
    # 1. 功能页面切换（单选，和截图一致）
    st.caption("功能页面")
    st.session_state.current_page = st.radio(
        label="",
        options=["航线规划", "飞行监控"],
        index=0 if st.session_state.current_page == "航线规划" else 1,
        label_visibility="collapsed"
    )

    st.divider()

    # 2. 坐标系设置（和截图完全一致）
    st.subheader("⚙️ 坐标系设置")
    st.caption("输入坐标系")
    st.session_state.input_coord_system = st.radio(
        label="",
        options=["WGS-84", "GCJ-02(高德/百度)"],
        index=1,
        label_visibility="collapsed"
    )

    st.divider()

    # 3. 系统状态（和截图一致，绿色勾选框）
    st.subheader("📊 系统状态")
    st.success("✅ A点已设")
    st.success("✅ B点已设")

# ========================== 主区域：根据选中页面切换内容 ==========================
# -------------------------- 页面1：航线规划（仅此处显示地图，和要求一致） --------------------------
if st.session_state.current_page == "航线规划":
    st.header("🗺️ 航线规划")
    # 主区域分栏：左列设置，右列地图（和截图一致）
    col_setting, col_map = st.columns([1, 2])

    # 左侧设置栏
    with col_setting:
        # A/B点坐标设置
        st.subheader("📍 A/B点坐标设置")
        input_a_lat = st.number_input("A点纬度", value=32.2322, format="%.6f")
        input_a_lon = st.number_input("A点经度", value=118.7490, format="%.6f")
        input_b_lat = st.number_input("B点纬度", value=32.2343, format="%.6f")
        input_b_lon = st.number_input("B点经度", value=118.7490, format="%.6f")

        # 坐标系自动转换
        if st.session_state.input_coord_system == "WGS-84":
            a_lat, a_lon = wgs84_to_gcj02(input_a_lat, input_a_lon)
            b_lat, b_lon = wgs84_to_gcj02(input_b_lat, input_b_lon)
            st.caption("已自动转换为GCJ-02坐标用于地图显示")
        else:
            a_lat, a_lon = input_a_lat, input_a_lon
            b_lat, b_lon = input_b_lat, input_b_lon

        st.divider()

        # 飞行参数设置
        st.subheader("✈️ 飞行参数设置")
        st.session_state.flight_height = st.slider("无人机飞行高度(m)", 10, 200, st.session_state.flight_height)
        st.session_state.safe_radius = st.number_input("无人机安全半径(m)", value=st.session_state.safe_radius, min_value=1)

        st.divider()

        # 障碍物管理
        st.subheader("🏢 障碍物管理")
        if st.session_state.obstacle_polygons:
            st.caption(f"已保存 {len(st.session_state.obstacle_polygons)} 个障碍物")
            for idx, poly in enumerate(st.session_state.obstacle_polygons):
                with st.expander(f"障碍物 {idx+1} 配置", expanded=False):
                    current_height = st.session_state.obstacle_heights.get(idx, 50)
                    new_height = st.slider(f"障碍物高度(m)", 1, 200, current_height, key=f"height_{idx}")
                    if new_height != current_height:
                        st.session_state.obstacle_heights[idx] = new_height
                        st.rerun()
                    if st.button("删除该障碍物", key=f"del_obs_{idx}"):
                        st.session_state.obstacle_polygons.pop(idx)
                        if idx in st.session_state.obstacle_heights:
                            del st.session_state.obstacle_heights[idx]
                        if idx in st.session_state.obstacle_create_time:
                            del st.session_state.obstacle_create_time[idx]
                        st.rerun()
        else:
            st.caption("暂无障碍物，请在右侧地图上圈选")

        if st.button("🗑️ 清除全部障碍物", use_container_width=True):
            st.session_state.obstacle_polygons = []
            st.session_state.obstacle_heights = {}
            st.session_state.obstacle_create_time = {}
            st.session_state.last_drawing_id = None
            st.rerun()

        st.divider()

        # 避障航线选择
        st.subheader("🧭 避障航线选择")
        route_options = ["原航线（直接飞跃）"]
        has_collision = False
        all_routes = {}

        if st.session_state.obstacle_polygons:
            for idx, obs_poly in enumerate(st.session_state.obstacle_polygons):
                obs_height = st.session_state.obstacle_heights.get(idx, 50)
                if line_polygon_intersection([a_lat, a_lon], [b_lat, b_lon], obs_poly) and st.session_state.flight_height <= obs_height:
                    has_collision = True
                    all_routes = generate_detour_routes(
                        [a_lat, a_lon], [b_lat, b_lon], obs_poly, obs_height,
                        st.session_state.flight_height, st.session_state.safe_radius
                    )
                    route_options = list(all_routes.keys())
                    break
        
        if has_collision:
            st.warning("⚠️ 检测到航线碰撞，需选择绕行航线")
        else:
            st.success("✅ 航线无碰撞，可直接飞跃")

        st.session_state.selected_route = st.radio(
            "选择飞行航线",
            options=route_options,
            index=0
        )

        if all_routes and st.session_state.selected_route in all_routes:
            st.session_state.current_route_points = all_routes[st.session_state.selected_route]
        else:
            st.session_state.current_route_points = [[a_lat, a_lon], [b_lat, b_lon]]

        st.divider()

        # 配置持久化
        st.subheader("🚀 配置持久化")
        col_save, col_load = st.columns(2)
        with col_save:
            if st.button("💾 保存配置", type="primary", use_container_width=True):
                save_data = save_obstacles_to_file()
                st.success(f"保存成功！共{save_data['obstacle_count']}个障碍物")
        with col_load:
            if st.button("📂 加载配置", use_container_width=True):
                load_data = load_obstacles_from_file()
                if load_data:
                    st.success(f"加载成功！共{load_data['obstacle_count']}个障碍物")
        
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                json_content = f.read()
            st.download_button(
                label="⬇️ 下载配置文件",
                data=json_content,
                file_name="obstacle_config.json",
                mime="application/json",
                use_container_width=True
            )

    # 右侧地图显示（仅航线规划页显示，完全符合要求）
    with col_map:
        st.subheader("🌍 卫星地图")
        map_placeholder = st.empty()

        # 地图渲染函数
        def render_satellite_map():
            center_lat = (a_lat + b_lat) / 2
            center_lon = (a_lon + b_lon) / 2

            # 卫星底图，和截图一致
            satellite_tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=17,
                tiles=satellite_tiles,
                attr="Tiles © Esri"
            )

            # A/B点标记
            folium.Marker([a_lat, a_lon], popup="起点A", icon=folium.Icon(color="red", icon="play")).add_to(m)
            folium.Marker([b_lat, b_lon], popup="终点B", icon=folium.Icon(color="green", icon="flag")).add_to(m)

            # 选中的航线
            folium.PolyLine(
                locations=st.session_state.current_route_points,
                color="blue",
                weight=4,
                opacity=0.9,
                popup=f"选中航线：{st.session_state.selected_route}"
            ).add_to(m)

            # 障碍物渲染
            for idx, poly_coords in enumerate(st.session_state.obstacle_polygons):
                obs_height = st.session_state.obstacle_heights.get(idx, 50)
                folium.Polygon(
                    locations=poly_coords,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.4,
                    weight=2,
                    popup=f"障碍物\n高度：{obs_height}m"
                ).add_to(m)

            # 无人机实时位置
            current_seq = len(st.session_state.df_history)
            total_steps = 50
            progress = min(current_seq / total_steps, 1.0)

            route_points = st.session_state.current_route_points
            if len(route_points) == 2:
                drone_lat = route_points[0][0] + (route_points[1][0] - route_points[0][0]) * progress
                drone_lon = route_points[0][1] + (route_points[1][1] - route_points[0][1]) * progress
            else:
                seg1_length = np.sqrt((route_points[1][0]-route_points[0][0])**2 + (route_points[1][1]-route_points[0][1])**2)
                seg2_length = np.sqrt((route_points[2][0]-route_points[1][0])**2 + (route_points[2][1]-route_points[1][1])**2)
                total_length = seg1_length + seg2_length
                if progress * total_length <= seg1_length:
                    seg_progress = (progress * total_length) / seg1_length
                    drone_lat = route_points[0][0] + (route_points[1][0] - route_points[0][0]) * seg_progress
                    drone_lon = route_points[0][1] + (route_points[1][1] - route_points[0][1]) * seg_progress
                else:
                    seg_progress = (progress * total_length - seg1_length) / seg2_length
                    drone_lat = route_points[1][0] + (route_points[2][0] - route_points[1][0]) * seg_progress
                    drone_lon = route_points[1][1] + (route_points[2][1] - route_points[1][1]) * seg_progress

            folium.CircleMarker(
                [drone_lat, drone_lon],
                radius=10,
                color="orange",
                fill=True,
                fill_color="yellow",
                popup=f"无人机\n进度: {progress*100:.1f}%"
            ).add_to(m)

            # 绘制工具栏（和截图完全一致，包含多边形、矩形、圆形、标记、撤销、删除）
            draw = Draw(
                export=False,
                position='topleft',
                draw_options={
                    'polyline': False,
                    'polygon': True,
                    'rectangle': True,
                    'circle': True,
                    'marker': True,
                    'circlemarker': False,
                },
                edit_options={'edit': True, 'remove': True}
            )
            draw.add_to(m)

            # 渲染地图并捕获圈选数据
            with map_placeholder:
                map_output = st_folium(
                    m,
                    width=1000,
                    height=700,
                    returned_objects=["last_active_drawing"]
                )

            # 捕获新圈选的障碍物
            if map_output and map_output["last_active_drawing"]:
                drawing = map_output["last_active_drawing"]
                drawing_id = str(drawing["geometry"]["coordinates"])
                
                if drawing_id != st.session_state.last_drawing_id:
                    st.session_state.last_drawing_id = drawing_id
                    geom_type = drawing["geometry"]["type"]
                    # 处理多边形/矩形/圆形，统一转为多边形
                    if geom_type == "Polygon":
                        poly_coords = [[lat, lon] for lon, lat in drawing["geometry"]["coordinates"][0]]
                        if poly_coords not in st.session_state.obstacle_polygons:
                            st.session_state.obstacle_polygons.append(poly_coords)
                            new_idx = len(st.session_state.obstacle_polygons) - 1
                            st.session_state.obstacle_heights[new_idx] = 50
                            st.session_state.obstacle_create_time[new_idx] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.success("障碍物圈选成功！")
                            st.rerun()

        # 执行地图渲染
        render_satellite_map()

# -------------------------- 页面2：飞行监控（仅此处显示心跳包，和要求一致） --------------------------
elif st.session_state.current_page == "飞行监控":
    st.header("📡 飞行监控（心跳包实时显示）")
    # 控制按钮
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

    # 状态显示
    status_box = st.empty()
    # 布局：左折线图，右数据列表
    col_chart, col_data = st.columns([2, 1])
    with col_chart:
        st.subheader("📈 心跳包序号实时趋势")
        chart_placeholder = st.empty()
    with col_data:
        st.subheader("📋 心跳数据包列表")
        data_box = st.empty()

    # 初始化折线图
    if len(st.session_state.df_history) > 0:
        chart_obj = chart_placeholder.line_chart(st.session_state.df_history, x="time", y="seq", color="#39ff14")
    else:
        chart_obj = chart_placeholder.line_chart(pd.DataFrame(columns=["time", "seq"]), x="time", y="seq")

    # 实时心跳循环
    while st.session_state.is_running:
        current_seq = len(st.session_state.df_history) + 1
        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        new_data = pd.DataFrame({"time": [current_time], "seq": [current_seq]})

        st.session_state.df_history = pd.concat([st.session_state.df_history, new_data], ignore_index=True)
        
        # 更新心跳折线图
        chart_obj.add_rows(new_data)
        # 更新数据列表
        data_box.dataframe(st.session_state.df_history.tail(10), hide_index=True, height=400)
        # 更新状态
        status_box.success(f"✅ 飞行正常 | 心跳包序号：{current_seq} | 航线：{st.session_state.selected_route} | 时间：{current_time}")

        # 超时检测计时
        st.session_state.last_received = time.time()
        time.sleep(1)

    # 3秒超时报警逻辑
    if st.session_state.last_received and not st.session_state.is_running:
        elapsed = time.time() - st.session_state.last_received
        if elapsed > 3 and len(st.session_state.df_history) > 0:
            status_box.error("🚨 连接超时！超过3秒未收到心跳包！")
        else:
            status_box.warning("⏸️ 飞行已暂停")
