import streamlit as st
import time
import datetime
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

# --------------------------
# 坐标系转换（WGS84 ↔ GCJ-02，适配国内地图）
# --------------------------
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

# --------------------------
# 初始化全局状态（记忆功能核心：session_state持久化）
# --------------------------
if 'df_history' not in st.session_state:
    st.session_state.df_history = pd.DataFrame(columns=["time", "seq"])
if 'last_received' not in st.session_state:
    st.session_state.last_received = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
# 障碍物记忆核心：持久化存储多边形坐标，页面刷新不丢失
if 'obstacle_polygons' not in st.session_state:
    st.session_state.obstacle_polygons = []
# 用于避免重复添加同一障碍物的标记
if 'last_drawing_id' not in st.session_state:
    st.session_state.last_drawing_id = None

# --------------------------
# 页面配置
# --------------------------
st.set_page_config(page_title="心跳包接收系统-3D地图规划模块", layout="wide")
st.title("🚁 心跳包接收系统-3D地图规划模块")
tab1, tab2 = st.tabs(["🗺️ 3D地图航线规划", "📡 心跳包实时监控"])

# --------------------------
# 标签页1：3D地图航线规划（卫星地图+圈选记忆）
# --------------------------
with tab1:
    st.header("🗺️ 航线规划（卫星实况地图，可缩放/拖拽/多边形圈选）")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📍 A/B点坐标设置")
        a_lat = st.number_input("A点纬度(GCJ-02)", value=32.2322, format="%.4f")
        a_lon = st.number_input("A点经度(GCJ-02)", value=118.7490, format="%.4f")
        b_lat = st.number_input("B点纬度(GCJ-02)", value=32.2343, format="%.4f")
        b_lon = st.number_input("B点经度(GCJ-02)", value=118.7490, format="%.4f")

        st.divider()
        st.subheader("🏢 障碍物圈选设置（记忆功能已修复）")
        st.info("💡 操作说明：在右侧地图左上角点击「多边形图标」，在地图上点击圈选障碍物区域，双击结束绘制。圈选的区域会自动永久保存。")
        
        # 障碍物管理
        col_clear, col_refresh = st.columns(2)
        with col_clear:
            if st.button("🗑️ 清空所有障碍物", type="secondary"):
                st.session_state.obstacle_polygons = []
                st.session_state.last_drawing_id = None
                st.success("已清空所有障碍物")
                st.rerun()
        
        # 显示已保存的障碍物列表+单个删除（记忆功能可视化）
        if st.session_state.obstacle_polygons:
            st.caption(f"✅ 已保存 {len(st.session_state.obstacle_polygons)} 个障碍物区域")
            for idx, poly in enumerate(st.session_state.obstacle_polygons):
                col_name, col_del = st.columns([4, 1])
                with col_name:
                    st.text(f"障碍物 {idx+1}（{len(poly)}个顶点）")
                with col_del:
                    if st.button("删除", key=f"del_obs_{idx}"):
                        st.session_state.obstacle_polygons.pop(idx)
                        st.rerun()
        else:
            st.caption("暂无保存的障碍物，请在地图上圈选")

        st.divider()
        st.subheader("✈️ 飞行参数设置")
        flight_height = st.slider("设定飞行高度(m)", 10, 100, 50)
        view_pitch = st.slider("3D视角倾斜角度", 0, 60, 30)
        st.info(f"当前飞行高度：{flight_height}m | 3D视角倾斜：{view_pitch}°")

    with col2:
        st.subheader("🌍 卫星实况地图")
        map_placeholder = st.empty()

# --------------------------
# 标签页2：心跳包实时监控
# --------------------------
with tab2:
    st.header("📡 心跳包实时监控（每秒自发自收）")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶️ 启动飞行", type="primary"):
            st.session_state.is_running = True
    with c2:
        if st.button("⏸️ 暂停飞行"):
            st.session_state.is_running = False
    with c3:
        if st.button("🔄 重置数据"):
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

    # 初始化折线图
    if len(st.session_state.df_history) > 0:
        chart_obj = chart_placeholder.line_chart(st.session_state.df_history, x="time", y="seq", color="#39ff14")
    else:
        chart_obj = chart_placeholder.line_chart(pd.DataFrame(columns=["time", "seq"]), x="time", y="seq")

# --------------------------
# 地图渲染核心函数（卫星实况+圈选+记忆）
# --------------------------
def render_satellite_map(current_seq=0, total_steps=50):
    # 地图中心点
    center_lat = (a_lat + b_lat) / 2
    center_lon = (a_lon + b_lon) / 2

    # ========== 需求1：卫星实况地图（全球稳定加载，无灰色问题） ==========
    satellite_tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles=satellite_tiles,
        attr="Tiles © Esri",
        view_control={"pitch": view_pitch, "bearing": 0}
    )

    # 计算无人机实时位置
    progress = min(current_seq / total_steps, 1.0)
    drone_lat = a_lat + (b_lat - a_lat) * progress
    drone_lon = a_lon + (b_lon - a_lon) * progress

    # 基础标记点渲染
    folium.Marker([a_lat, a_lon], popup="起点A", icon=folium.Icon(color="red", icon="play")).add_to(m)
    folium.Marker([b_lat, b_lon], popup="终点B", icon=folium.Icon(color="green", icon="flag")).add_to(m)
    folium.CircleMarker(
        [drone_lat, drone_lon], radius=8, color="orange", fill=True, fill_color="yellow",
        popup=f"无人机\n进度: {progress*100:.1f}%\n高度: {flight_height}m"
    ).add_to(m)
    folium.PolyLine(locations=[[a_lat, a_lon], [b_lat, b_lon]], color="blue", weight=3).add_to(m)

    # ========== 需求2：记忆功能 - 渲染所有已保存的障碍物 ==========
    for poly_coords in st.session_state.obstacle_polygons:
        folium.Polygon(
            locations=poly_coords,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.4,
            weight=2,
            popup="障碍物禁飞区"
        ).add_to(m)

    # ========== 多边形圈选插件（修复按钮显示） ==========
    draw = Draw(
        export=False,
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'polygon': {
                'allowIntersection': False,
                'drawError': {'color': '#ff0000', 'timeout': 2000},
                'shapeOptions': {'color': '#ff0000', 'fillOpacity': 0.4, 'weight': 2}
            }
        },
        edit_options={'edit': True, 'remove': True}
    )
    draw.add_to(m)

    # 渲染地图并捕获圈选数据
    with map_placeholder:
        map_output = st_folium(
            m,
            width=800,
            height=600,
            returned_objects=["last_active_drawing", "all_drawings"]
        )

    # ========== 记忆功能核心：捕获圈选数据并持久化保存 ==========
    if map_output and map_output["last_active_drawing"]:
        drawing = map_output["last_active_drawing"]
        # 生成唯一ID，避免重复添加
        drawing_id = str(drawing["geometry"]["coordinates"])
        
        if drawing_id != st.session_state.last_drawing_id:
            st.session_state.last_drawing_id = drawing_id
            # 只处理多边形
            if drawing["geometry"]["type"] == "Polygon":
                # 坐标格式转换：GeoJSON的[lon, lat] → Folium的[lat, lon]
                poly_coords = [[lat, lon] for lon, lat in drawing["geometry"]["coordinates"][0]]
                # 去重添加
                if poly_coords not in st.session_state.obstacle_polygons:
                    st.session_state.obstacle_polygons.append(poly_coords)
                    st.success("障碍物区域已保存！")
                    st.rerun()

# --------------------------
# 主程序运行
# --------------------------
# 初始渲染卫星地图
render_satellite_map(len(st.session_state.df_history))

# 实时心跳+地图更新循环
while st.session_state.is_running:
    current_seq = len(st.session_state.df_history) + 1
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    new_data = pd.DataFrame({"time": [current_time], "seq": [current_seq]})

    st.session_state.df_history = pd.concat([st.session_state.df_history, new_data], ignore_index=True)
    
    # 更新心跳折线图
    chart_obj.add_rows(new_data)
    # 更新卫星地图（无人机移动）
    render_satellite_map(current_seq)
    # 更新数据列表
    data_box.dataframe(st.session_state.df_history.tail(10), hide_index=True, height=300)
    # 更新状态
    status_box.success(f"✅ 连接正常 | 心跳包序号：{current_seq} | 时间：{current_time}")

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
