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
VERSION = "v17.5 多段折线｜安全距=最小贴边｜最短路径自动最优"
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
        st.session_state.map_rerun_key += 1
        st.rerun()
        return load_data
    except Exception as e:
        st.error(f"加载失败：{str(e)}")
        return None

# ========================== 核心：多段折线生成函数 ==========================
def generate_routes(start, end, obstacle_list, obstacle_heights, fly_height, safe_radius):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end

    # 直飞参考平滑线
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

    # 合并障碍物+安全缓冲区（航线与建筑最小距离=safe_radius）
    all_poly = []
    for co in obstacle_list:
        all_poly.append(Polygon(co))
    merged = unary_union(all_poly)
    buf = merged.buffer(safe_radius / 111319.9)

    # 递归生成【三段多折航线：起点→拐点1→拐点2→终点】
    def get_multi_polyline(side_offset):
        clat = np.mean([p[0] for obs in obstacle_list for p in obs])
        clon = np.mean([p[1] for obs in obstacle_list for p in obs])
        cpoint = Point(clon,clat)
        lat_off,lon_off = meter_to_latlon_offset(clat,safe_radius)
        mid_anchor = (cpoint.y + lat_off*side_offset, cpoint.x - lon_off*side_offset)
        # 双拐点，3段折线
        p1 = mid_anchor
        p2 = ((p1[0]+e_lat)/2.1, (p1[1]+e_lon)/1.9)
        poly_pts = [start, p1, p2, end]
        check_line = LineString(poly_pts)
        if not check_line.intersects(buf):
            return poly_pts
        else:
            return get_multi_polyline(side_offset+1.8)

    left_line_pts = get_multi_polyline(7.2)
    right_line_pts = get_multi_polyline(-7.2)

    routes["左侧绕行"] = left_line_pts
    routes["右侧绕行"] = right_line_pts

    # 计算里程，择优最短为最优航线
    def calc_total_len(pts):
        dist=0
        for i in range(len(pts)-1):
            dist += latlon_to_meter(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1])
        return dist
    len_left = calc_total_len(left_line_pts)
    len_right = calc_total_len(right_line_pts)

    if len_left <= len_right:
        routes[f"最优航线（左侧最短）"] = left_line_pts
    else:
        routes[f"最优航线（右侧最短）"] = right_line_pts

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
    st.header("🗺️ 航线规划")
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
            "安全距离(米)=航线距建筑最小间距", value=st.session_state.safe_radius, min_value=1, key="safe_r",
            on_change=lambda: st.session_state.update({"map_rerun_key": st.session_state.map_rerun_key + 1})
        )
        st.caption("提示：三段多段折线｜贴障绕行｜左右自动择优最短为最优｜全深蓝色实线")
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

        st.markdown("#### 🧭 航线选择")
        route_keys = list(st.session_state.all_routes.keys())

        if len(route_keys) == 1:
            max_obs_h = 0
            for idx in range(len(st.session_state.obstacle_polygons)):
                max_obs_h = max(max_obs_h, st.session_state.obstacle_heights.get(idx, 50))
            st.warning(f"⚠️ 仅显示直接飞越：飞行高度({st.session_state.flight_height}米) > 障碍物最大高度({max_obs_h}米)")

        default_idx = 0
        if any("最优航线" in k for k in route_keys):
            best_key = [k for k in route_keys if "最优航线" in k][0]
            default_idx = route_keys.index(best_key)
        elif "左侧绕行" in route_keys:
            default_idx = route_keys.index("左侧绕行")
        elif "右侧绕行" in route_keys:
            default_idx = route_keys.index("右侧绕行")

        selected_route = st.radio(
            "当前激活航线", route_keys, index=default_idx, key="route_sel",
            on_change=lambda: st.session_state.update({"map_rerun_key": st.session_state.map_rerun_key + 1})
        )
        st.session_state.current_route_points = st.session_state.all_routes[selected_route]

    with col_map:
        st.subheader("🗺️ 地图（实时刷新）")
        st.caption("🟦深蓝色三段多折线｜贴障最小距离=安全距离｜里程最短自动优选｜⚫灰色=直飞参考线")
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

            # 全部绕行统一深蓝色实线
            if "左侧绕行" in st.session_state.all_routes:
                folium.PolyLine(st.session_state.all_routes["左侧绕行"], color="#0044FF", weight=5, opacity=0.8).add_to(m)
            if "右侧绕行" in st.session_state.all_routes:
                folium.PolyLine(st.session_state.all_routes["右侧绕行"], color="#0044FF", weight=5, opacity=0.8).add_to(m)
            if any("最优航线" in k for k in st.session_state.all_routes.keys()):
                best_route_key = [k for k in st.session_state.all_routes.keys() if "最优航线" in k][0]
                folium.PolyLine(st.session_state.all_routes[best_route_key], color="#0044FF", weight=5, opacity=1.0).add_to(m)
            # 直飞灰色参考
            if "直接飞越" in st.session_state.all_routes:
                folium.PolyLine(st.session_state.all_routes["直接飞越"], color="#808080", weight=3, opacity=0.5).add_to(m)

            for idx, poly in enumerate(st.session_state.obstacle_polygons):
                folium.Polygon(
                    poly, color="#FF0000", fill=True, fill_color="#FF0000", fill_opacity=0.4,
                    popup=f"障碍物 {idx+1} | 高度：{st.session_state.obstacle_heights.get(idx,50)}米", weight=3
                ).add_to(m)

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
