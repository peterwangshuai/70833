import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from shapely import buffer, offset_curve
import numpy as np

# -------------------------- 页面全局样式 --------------------------
st.set_page_config(page_title="无人机可视图平滑避障航线规划系统", layout="wide")
st.markdown("""
<style>
    .stApp {background-color: #121212; color: white;}
    .leaflet-control-attribution {display:none!important;}
    div[data-testid="stSidebar"] {background-color: #1e1e1e;}
    .stButton>button {color:white;}
</style>
""", unsafe_allow_html=True)

# -------------------------- 坐标转换工具（GCJ02 <-> WGS84） --------------------------
def wgs84_to_gcj02(lat, lon):
    a = 6378245.0
    ee = 0.00669342162296594323
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
    return round(lat - (g_lat - lat), 6), round(lon - (g_lon - lon), 6)

def meter_to_latlon_offset(lat, meter):
    lat_off = meter / 111319.9
    lon_off = meter / (111319.9 * np.cos(np.radians(lat)))
    return lat_off, lon_off

def latlon_to_meter(lat1, lon1, lat2, lon2):
    R = 6371000
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# -------------------------- 路径平滑函数 --------------------------
def smooth_bezier(points, seg_num=12):
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

# -------------------------- 核心避障算法：增加空障碍物容错 --------------------------
def plan_avoidance_routes(start, end, obstacle_polys, fly_height, safe_meter, obs_height_list):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end
    start_pt = Point(s_lon, s_lat)
    end_pt = Point(e_lon, e_lat)
    straight_line = LineString([(s_lon, s_lat), (e_lon, e_lat)])
    routes["直飞航线(越障备选)"] = smooth_bezier([start, end])

    # 无障碍物，直接返回直飞
    if len(obstacle_polys) == 0:
        routes["最优航线"] = routes["直飞航线(越障备选)"]
        return routes, None

    # 合并障碍物 + 生成安全缓冲区
    poly_list = []
    for coords in obstacle_polys:
        poly = Polygon([(lon, lat) for lat, lon in coords])
        poly_list.append(poly)
    merged_obs = unary_union(poly_list)
    buf_lat, buf_lon = meter_to_latlon_offset((s_lat+e_lat)/2, safe_meter)
    buf_deg = max(buf_lat, buf_lon)
    safe_zone = merged_obs.buffer(buf_deg, join_style="round", quad_segs=12)

    # 判断高度：可以直接飞越障碍物
    if fly_height > max(obs_height_list):
        routes["最优航线"] = routes["直飞航线(越障备选)"]
        return routes, safe_zone

    # 判断直飞是否碰撞安全区
    if not straight_line.intersects(safe_zone):
        routes["最优航线"] = routes["直飞航线(越障备选)"]
        return routes, safe_zone

    # 生成左侧绕行（西边空地）
    try:
        left_offset = offset_curve(safe_zone.boundary, buf_deg, join_style="round", quad_segs=12)
        left_coords = [(p[1], p[0]) for p in left_offset.coords]
        left_path = [start] + left_coords + [end]
        left_smooth = smooth_bezier(left_path)
        routes["左侧绕行(西侧空地)"] = left_smooth
    except:
        mid_lat = (s_lat+e_lat)/2
        mid_lon = (s_lon+e_lon)/2
        off_lat, off_lon = meter_to_latlon_offset(mid_lat, safe_meter*2)
        bypass = (mid_lat + off_lat, mid_lon - off_lon)
        routes["左侧绕行(西侧空地)"] = smooth_bezier([start, bypass, end])

    # 生成右侧绕行（东边马路）
    try:
        right_offset = offset_curve(safe_zone.boundary, -buf_deg, join_style="round", quad_segs=12)
        right_coords = [(p[1], p[0]) for p in right_offset.coords]
        right_path = [start] + right_coords + [end]
        right_smooth = smooth_bezier(right_path)
        routes["右侧绕行(东侧马路)"] = right_smooth
    except:
        mid_lat = (s_lat+e_lat)/2
        mid_lon = (s_lon+e_lon)/2
        off_lat, off_lon = meter_to_latlon_offset(mid_lat, safe_meter*2)
        bypass = (mid_lat - off_lat, mid_lon + off_lon)
        routes["右侧绕行(东侧马路)"] = smooth_bezier([start, bypass, end])

    # 选出最短绕行作为最优航线
    def get_len(pts):
        total = 0
        for i in range(len(pts)-1):
            lat1, lon1 = pts[i]
            lat2, lon2 = pts[i+1]
            total += latlon_to_meter(lat1, lon1, lat2, lon2)
        return total
    len_left = get_len(routes["左侧绕行(西侧空地)"])
    len_right = get_len(routes["右侧绕行(东侧马路)"])
    if len_left <= len_right:
        routes["最优航线"] = routes["左侧绕行(西侧空地)"]
    else:
        routes["最优航线"] = routes["右侧绕行(东侧马路)"]
    return routes, safe_zone

# -------------------------- 会话状态初始化 --------------------------
if "page" not in st.session_state:
    st.session_state.page = "航线规划"
if "obs_polygons" not in st.session_state:
    st.session_state.obs_polygons = []
if "obs_heights" not in st.session_state:
    st.session_state.obs_heights = []
if "map_key" not in st.session_state:
    st.session_state.map_key = 0
if "start_lat" not in st.session_state:
    st.session_state.start_lat = 32.232300
    st.session_state.start_lon = 118.749000
    st.session_state.end_lat = 32.234400
    st.session_state.end_lon = 118.749000
if "fly_h" not in st.session_state:
    st.session_state.fly_h = 10
if "safe_dist" not in st.session_state:
    st.session_state.safe_dist = 5

# -------------------------- 左侧侧边栏（复刻截图布局） --------------------------
with st.sidebar:
    st.subheader("🧭 导航")
    page_sel = st.radio("", ["航线规划", "飞行监控"], index=0, label_visibility="collapsed")
    st.session_state.page = page_sel
    st.divider()

    st.subheader("🌐 坐标系设置")
    coord_sel = st.radio("", ["WGS-84", "GCJ-02(高德/百度)"], index=1, label_visibility="collapsed")
    st.divider()

    st.subheader("📊 系统状态")
    st.success("✅ 起点A已设置")
    st.success("✅ 终点B已设置")
    if st.button("🔵 强制刷新地图", use_container_width=True):
        st.session_state.map_key += 1
        st.rerun()

# -------------------------- 主界面：地图 + 右侧控制面板 --------------------------
if st.session_state.page == "航线规划":
    st.title("🗺️ 无人机可视图平滑避障航线规划系统")
    col_map, col_ctrl = st.columns([2, 1])

    with col_ctrl:
        st.header("⚙️ 控制面板")
        st.divider()
        # 起点A
        st.subheader("📍 起点A")
        st.session_state.start_lat = st.number_input("纬度", value=st.session_state.start_lat, format="%.6f")
        st.session_state.start_lon = st.number_input("经度", value=st.session_state.start_lon, format="%.6f")
        st.checkbox("设置A点", value=True, disabled=True)
        st.divider()
        # 终点B
        st.subheader("📍 终点B")
        st.session_state.end_lat = st.number_input("纬度 B", value=st.session_state.end_lat, format="%.6f")
        st.session_state.end_lon = st.number_input("经度 B", value=st.session_state.end_lon, format="%.6f")
        st.checkbox("设置B点", value=True, disabled=True)
        st.divider()
        # 避障参数
        st.subheader("✈️ 避障参数")
        st.session_state.fly_h = st.slider("飞行高度 (米)", min_value=1, max_value=200, value=st.session_state.fly_h)
        st.session_state.safe_dist = st.number_input("水平安全距离 (米)", min_value=1, max_value=50, value=st.session_state.safe_dist)

    with col_map:
        # 坐标转换
        if coord_sel == "WGS-84":
            g_start_lat, g_start_lon = wgs84_to_gcj02(st.session_state.start_lat, st.session_state.start_lon)
            g_end_lat, g_end_lon = wgs84_to_gcj02(st.session_state.end_lat, st.session_state.end_lon)
        else:
            g_start_lat, g_start_lon = st.session_state.start_lat, st.session_state.start_lon
            g_end_lat, g_end_lon = st.session_state.end_lat, st.session_state.end_lon

        # 执行路径规划
        route_dict, safe_buffer = plan_avoidance_routes(
            start=(g_start_lat, g_start_lon),
            end=(g_end_lat, g_end_lon),
            obstacle_polys=st.session_state.obs_polygons,
            fly_height=st.session_state.fly_h,
            safe_meter=st.session_state.safe_dist,
            obs_height_list=st.session_state.obs_heights
        )

        # 初始化卫星地图
        map_center = ((g_start_lat+g_end_lat)/2, (g_start_lon+g_end_lon)/2)
        m = folium.Map(
            location=map_center,
            zoom_start=15,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri 卫星影像"
        )

        # 起止点标记
        folium.Marker([g_start_lat, g_start_lon], icon=folium.Icon(color="green", icon="plane")).add_to(m)
        folium.Marker([g_end_lat, g_end_lon], icon=folium.Icon(color="red", icon="flag")).add_to(m)

        # 【修复】只有存在缓冲区才绘制虚线框
        if safe_buffer is not None:
            folium.Polygon(
                locations=[(p[1], p[0]) for p in safe_buffer.exterior.coords],
                color="blue", fill=False, dash_array="6,6", weight=3
            ).add_to(m)

        # 绘制障碍物本体
        for coords in st.session_state.obs_polygons:
            folium.Polygon(
                locations=coords,
                color="red", fill_color="red", fill_opacity=0.5, weight=2
            ).add_to(m)

        # 多航线分色绘制
        route_style = {
            "左侧绕行(西侧空地)": {"color":"#c82423", "weight":4},
            "右侧绕行(东侧马路)": {"color":"#d2691e", "weight":4},
            "直飞航线(越障备选)": {"color":"#808080", "weight":3, "dashArray":"4,4"},
            "最优航线": {"color":"#0066ff", "weight":5}
        }
        for name, pts in route_dict.items():
            style = route_style[name]
            folium.PolyLine(pts,** style).add_to(m)

        # 绘图工具
        from folium.plugins import Draw
        draw = Draw(
            position="topleft",
            draw_options={
                "polygon": {"allowIntersection":False},
                "rectangle": True,
                "polyline":False, "marker":False, "circle":False
            },
            edit_options={"edit":True, "remove":True}
        )
        draw.add_to(m)

        # 接收绘制的障碍物
        map_out = st_folium(m, width=1050, height=720, key=f"map_{st.session_state.map_key}")
        if map_out and map_out.get("last_active_drawing"):
            geo = map_out["last_active_drawing"]["geometry"]
            if geo["type"] in ["Polygon", "Rectangle"]:
                new_coords = [[lat, lon] for lon, lat in geo["coordinates"][0]]
                st.session_state.obs_polygons.append(new_coords)
                st.session_state.obs_heights.append(30)
                st.session_state.map_key += 1
                st.rerun()

# -------------------------- 飞行监控页面 --------------------------
else:
    st.title("📡 无人机飞行监控")
    st.info("实时位置、航线跟踪、心跳包接收模块可在此扩展")
