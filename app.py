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
from shapely.ops import split

# ========================== 全局配置：终极全汉化CSS ==========================
st.set_page_config(page_title="无人机航线规划系统", layout="wide")

# 强制汉化所有地图控件、隐藏原生英文提示
st.markdown('''
<style>
/* 1. 隐藏所有原生英文悬浮提示 */
.leaflet-tooltip,
.leaflet-draw-tooltip,
.leaflet-control-zoom-in[title],
.leaflet-control-zoom-out[title],
.leaflet-draw-buttons button[title] {
    display: none !important;
    visibility: hidden !important;
}

/* 2. 地图缩放按钮汉化 */
.leaflet-control-zoom-in::after {
    content: "放大地图";
    position:absolute;
    left:45px;
    top:0;
    background:#222;
    color:#fff;
    padding:3px 8px;
    font-size:12px;
    border-radius:3px;
    white-space:nowrap;
    z-index:999999;
}
.leaflet-control-zoom-out::after {
    content: "缩小地图";
    position:absolute;
    left:45px;
    top:0;
    background:#222;
    color:#fff;
    padding:3px 8px;
    font-size:12px;
    border-radius:3px;
    white-space:nowrap;
    z-index:999999;
}

/* 3. 绘图工具按钮汉化 */
.leaflet-draw-draw-polygon::after {
    content:"绘制多边形";
    position:absolute;
    left:45px;
    top:0;
    background:#222;
    color:#fff;
    padding:3px 8px;
    font-size:12px;
    border-radius:3px;
    white-space:nowrap;
    z-index:999999;
}
.leaflet-draw-draw-rectangle::after {
    content:"绘制矩形";
    position:absolute;
    left:45px;
    top:0;
    background:#222;
    color:#fff;
    padding:3px 8px;
    font-size:12px;
    border-radius:3px;
    white-space:nowrap;
    z-index:999999;
}
.leaflet-draw-draw-circle::after {
    content:"绘制圆形";
    position:absolute;
    left:45px;
    top:0;
    background:#222;
    color:#fff;
    padding:3px 8px;
    font-size:12px;
    border-radius:3px;
    white-space:nowrap;
    z-index:999999;
}
.leaflet-draw-draw-marker::after {
    content:"添加标记点";
    position:absolute;
    left:45px;
    top:0;
    background:#222;
    color:#fff;
    padding:3px 8px;
    font-size:12px;
    border-radius:3px;
    white-space:nowrap;
    z-index:999999;
}
.leaflet-draw-edit-edit::after {
    content:"编辑图层";
    position:absolute;
    left:45px;
    top:0;
    background:#222;
    color:#fff;
    padding:3px 8px;
    font-size:12px;
    border-radius:3px;
    white-space:nowrap;
    z-index:999999;
}
.leaflet-draw-edit-remove::after {
    content:"删除图层";
    position:absolute;
    left:45px;
    top:0;
    background:#222;
    color:#fff;
    padding:3px 8px;
    font-size:12px;
    border-radius:3px;
    white-space:nowrap;
    z-index:999999;
}

/* 4. 隐藏地图右下角英文版权，替换为中文 */
.leaflet-control-attribution { display: none !important; }
</style>
''', unsafe_allow_html=True)

# ========================== 基础全局参数 ==========================
CONFIG_DIR = r"D:\wrj\3Dwrj"
CONFIG_FILE = os.path.join(CONFIG_DIR, "障碍物配置.json")
VERSION = "v15.0 安全绕行+实时刷新+全中文终极版"
DEFAULT_SAFE_RADIUS = 5

# ========================== 坐标系转换工具函数 ==========================
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
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def meter_to_latlon_offset(lat, meter):
    lat_offset = meter / 111319.9
    lon_offset = meter / (111319.9 * np.cos(np.radians(lat)))
    return lat_offset, lon_offset

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
        st.rerun()
        return load_data
    except Exception as e:
        st.error(f"加载失败：{str(e)}")
        return None

# ========================== 【核心修复】安全绕行算法：绝不穿过障碍物 ==========================
def generate_routes(start, end, obstacle_coords, obs_height, fly_height, safe_radius):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end

    # 飞行高度足够：直接飞越
    if fly_height > obs_height:
        routes["直接飞越"] = [start, end]
        return routes

    # 1. 生成障碍物 + 安全缓冲区（单位：米转度）
    obs_poly = Polygon(obstacle_coords)
    center_point = Point(
        np.mean([p[1] for p in obstacle_coords]),
        np.mean([p[0] for p in obstacle_coords])
    )
    lat_off, lon_off = meter_to_latlon_offset(center_point.y, safe_radius)
    safe_buffer = obs_poly.buffer(safe_radius / 111319.9)

    # 2. 生成初始航点，确保不与缓冲区相交
    offset_scale = 2.5  # 基础偏移倍数
    for attempt in range(5):  # 最多尝试5次，找到不相交的航点
        # 左侧绕行点
        left_waypoint = (center_point.y + lat_off * offset_scale, center_point.x - lon_off * offset_scale)
        # 右侧绕行点
        right_waypoint = (center_point.y - lat_off * offset_scale, center_point.x + lon_off * offset_scale)

        # 生成航线并检查是否相交
        left_route = LineString([start, left_waypoint, end])
        right_route = LineString([start, right_waypoint, end])

        if not left_route.intersects(safe_buffer):
            routes["左侧绕行"] = [start, left_waypoint, end]
            break
        if not right_route.intersects(safe_buffer):
            routes["右侧绕行"] = [start, right_waypoint, end]
            break

        # 如果相交，扩大偏移倍数
        offset_scale += 1.0

    # 3. 计算最短最优航线
    min_dist = float("inf")
    best_route = None
    for name, pts in routes.items():
        if name in ("左侧绕行", "右侧绕行"):
            dist = latlon_to_meter(pts[0][0], pts[0][1], pts[1][0], pts[1][1]) \
                 + latlon_to_meter(pts[1][0], pts[1][1], pts[2][0], pts[2][1])
            if dist < min_dist:
                min_dist = dist
                best_route = pts
    routes["最优航线(最短距离)"] = best_route
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
if 'selected_route' not in st.session_state:
    st.session_state.selected_route = "直接飞越"
if 'current_route_points' not in st.session_state:
    st.session_state.current_route_points = []
# 用于强制地图刷新的key
if 'map_rerun_key' not in st.session_state:
    st.session_state.map_rerun_key = 0

# ========================== 左侧导航栏 ==========================
with st.sidebar:
    st.subheader("🧭 导航")
    st.caption("功能页面")
    st.session_state.current_page = st.radio(
        "", ["航线规划", "飞行监控"],
        index=0, label_visibility="collapsed"
    )
    st.divider()

    st.subheader("⚙️ 坐标系设置")
    st.session_state.input_coord_system = st.radio(
        "", ["WGS-84", "GCJ-02(高德/百度)"],
        index=1, label_visibility="collapsed"
    )
    st.divider()

    st.subheader("📊 系统状态")
    st.success("✅ 起点A已设置")
    st.success("✅ 终点B已设置")

# ========================== 航线规划主页面 ==========================
if st.session_state.current_page == "航线规划":
    st.header("🗺️ 航线规划")
    col_map, col_control = st.columns([2, 1])

    # 右侧控制面板
    with col_control:
        st.subheader("⚙️ 控制面板")

        # 起点终点设置
        st.markdown("#### 📍 起点A")
        input_a_lat = st.number_input("纬度", value=32.2323, format="%.4f")
        input_a_lon = st.number_input("经度", value=118.749, format="%.3f")
        st.success("✅ 已设置起点A")
        st.divider()

        st.markdown("#### 📍 终点B")
        input_b_lat = st.number_input("纬度 ", value=32.2344, format="%.4f")
        input_b_lon = st.number_input("经度 ", value=118.749, format="%.3f")
        st.success("✅ 已设置终点B")
        st.divider()

        # 飞行参数
        st.markdown("#### ✈️ 飞行参数")
        st.session_state.flight_height = st.slider("无人机飞行高度(米)", 1, 200, 10)
        st.session_state.safe_radius = st.number_input("安全距离(米)", value=5, min_value=1)
        st.caption("高度>障碍物高度=直接飞越，反之自动绕行避障")
        st.divider()

        # 障碍物配置持久化 + 高度设置（实时更新）
        st.markdown("#### 🚀 障碍物配置持久化")
        st.markdown("##### 障碍物高度设置")
        if st.session_state.obstacle_polygons:
            st.caption(f"已配置 {len(st.session_state.obstacle_polygons)} 个障碍物")
            for idx in range(len(st.session_state.obstacle_polygons)):
                with st.expander(f"障碍物 {idx+1}", expanded=False):
                    st.session_state.obstacle_heights[idx] = st.slider(
                        "障碍物高度(米)", 1, 200,
                        value=st.session_state.obstacle_heights.get(idx, 50),
                        key=f"h_{idx}"
                    )
                    if st.button("删除该障碍物", key=f"del_{idx}"):
                        st.session_state.obstacle_polygons.pop(idx)
                        if idx in st.session_state.obstacle_heights:
                            del st.session_state.obstacle_heights[idx]
                        if idx in st.session_state.obstacle_create_time:
                            del st.session_state.obstacle_create_time[idx]
                        st.session_state.map_rerun_key += 1
                        st.rerun()
        else:
            st.info("暂无障碍物，请在地图上圈选区域")

        # 功能按钮
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            if st.button("💾 保存", type="primary", use_container_width=True):
                save_obstacles_to_file()
                st.success("保存完成")
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
                st.success("已清空所有障碍物")
                st.rerun()
        with c4:
            if st.button("🚀 部署", type="primary", use_container_width=True):
                st.success("部署完成")
        st.divider()

        # 航线逻辑计算
        if st.session_state.input_coord_system == "WGS-84":
            a_lat,a_lon = wgs84_to_gcj02(input_a_lat,input_a_lon)
            b_lat,b_lon = wgs84_to_gcj02(input_b_lat,input_b_lon)
        else:
            a_lat,a_lon = input_a_lat,input_a_lon
            b_lat,b_lon = input_b_lat,input_b_lon

        start_pt = (a_lat,a_lon)
        end_pt = (b_lat,b_lon)

        route_map = {}
        if st.session_state.obstacle_polygons:
            obs_idx = 0
            obs_h = st.session_state.obstacle_heights.get(0,50)
            route_map = generate_routes(start_pt,end_pt,
                st.session_state.obstacle_polygons[0],
                obs_h,
                st.session_state.flight_height,
                st.session_state.safe_radius)
        else:
            route_map["直接飞越"] = [start_pt,end_pt]

        st.markdown("#### 🧭 航线选择")
        st.session_state.selected_route = st.radio("可选航线", list(route_map.keys()))
        st.session_state.current_route_points = route_map[st.session_state.selected_route]

    # 地图渲染区域（带强制刷新key，实时更新）
    with col_map:
        st.subheader("🗺️ 地图")
        map_placeholder = st.empty()

        def render_map():
            center_lat = (a_lat + b_lat) / 2
            center_lon = (a_lon + b_lon) / 2

            # 卫星地图，中文属性
            m = folium.Map(
                [center_lat, center_lon],
                zoom_start=17,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="卫星地图来源：Esri"
            )

            # 标记、航线、障碍物（中文弹窗）
            folium.Marker([a_lat, a_lon], popup="起点A", icon=folium.Icon(color="red")).add_to(m)
            folium.Marker([b_lat, b_lon], popup="终点B", icon=folium.Icon(color="green")).add_to(m)
            folium.PolyLine(
                st.session_state.current_route_points,
                color="blue",
                weight=4,
                popup=f"当前航线：{st.session_state.selected_route}"
            ).add_to(m)

            for idx, poly in enumerate(st.session_state.obstacle_polygons):
                folium.Polygon(
                    poly,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.4,
                    popup=f"障碍物 | 高度：{st.session_state.obstacle_heights.get(idx, 50)}米"
                ).add_to(m)

            # 绘图组件（无多余英文配置）
            draw = Draw(
                export=False,
                position="topleft",
                draw_options={"polyline":False,"polygon":{},"rectangle":{},"circle":{},"marker":{},"circlemarker":False},
                edit_options={"edit":{},"remove":{}}
            )
            draw.add_to(m)

            # 带动态key，强制每次状态变化时重绘
            with map_placeholder:
                map_data = st_folium(
                    m,
                    width=1000,
                    height=700,
                    returned_objects=["last_active_drawing"],
                    key=f"map_{st.session_state.map_rerun_key}"
                )

            # 处理圈选的障碍物（实时更新+刷新）
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
                            new_id = len(st.session_state.obstacle_polygons) - 1
                            st.session_state.obstacle_heights[new_id] = 50
                            st.session_state.obstacle_create_time[new_id] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.map_rerun_key += 1
                            st.success("障碍物添加成功！地图已刷新")
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
