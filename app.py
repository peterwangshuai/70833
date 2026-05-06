import streamlit as st
import time
import datetime
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

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
# 初始化全局状态
# --------------------------
if 'df_history' not in st.session_state:
    st.session_state.df_history = pd.DataFrame(columns=["time", "seq"])
if 'last_received' not in st.session_state:
    st.session_state.last_received = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'obstacles' not in st.session_state:
    st.session_state.obstacles = []  # 存储障碍物坐标

# --------------------------
# 页面配置 + 双标签页
# --------------------------
st.set_page_config(page_title="南京科技职业学院无人机监控", layout="wide")
st.title("🚁 南京科技职业学院 无人机智能化应用Demo")
tab1, tab2 = st.tabs(["🗺️ 校园航线规划", "📡 飞行监控（心跳包）"])

# --------------------------
# 标签页1：航线规划（南京科技职业学院专属坐标）
# --------------------------
with tab1:
    st.header("🗺️ 校园航线规划（卫星底图，可缩放/拖拽）")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📍 A/B点坐标设置（校园内）")
        # 南京科技职业学院 校园内默认坐标（GCJ-02），A点教学楼，B点操场，中间有建筑障碍物
        a_lat = st.number_input("A点纬度(GCJ-02)", value=32.2322, format="%.4f")
        a_lon = st.number_input("A点经度(GCJ-02)", value=118.7490, format="%.4f")
        b_lat = st.number_input("B点纬度(GCJ-02)", value=32.2343, format="%.4f")
        b_lon = st.number_input("B点经度(GCJ-02)", value=118.7490, format="%.4f")

        st.divider()
        st.subheader("🏢 障碍物设置")
        obstacle_lat = st.number_input("障碍物纬度", value=32.2332, format="%.4f")
        obstacle_lon = st.number_input("障碍物经度", value=118.7490, format="%.4f")
        if st.button("添加障碍物"):
            st.session_state.obstacles.append((obstacle_lat, obstacle_lon))
            st.success("障碍物已添加！")
        if st.button("清空障碍物"):
            st.session_state.obstacles = []
            st.info("已清空所有障碍物")

        st.divider()
        st.subheader("飞行参数")
        flight_height = st.slider("设定飞行高度(m)", 10, 100, 50)
        st.info(f"当前飞行高度：{flight_height}m")

    with col2:
        st.subheader("🌍 南京科技职业学院 卫星地图")
        map_placeholder = st.empty()

# --------------------------
# 标签页2：飞行监控（心跳包 + 实时图表 + 超时报警）
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
# 修复版卫星地图渲染函数（ArcGIS稳定底图，云端必现）
# --------------------------
def render_satellite_map(current_seq=0, total_steps=50):
    # 地图中心点：南京科技职业学院
    center_lat = (a_lat + b_lat) / 2
    center_lon = (a_lon + b_lon) / 2

    # ArcGIS全球卫星影像底图（全球稳定加载，无空白问题，校园清晰可见）
    satellite_tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,  # 校园级放大，清晰看到建筑
        tiles=satellite_tiles,
        attr="Tiles © Esri"
    )

    # 计算无人机实时位置（沿AB线移动）
    progress = min(current_seq / total_steps, 1.0)
    drone_lat = a_lat + (b_lat - a_lat) * progress
    drone_lon = a_lon + (b_lon - a_lon) * progress

    # 1. 起点A（红色标记）
    folium.Marker(
        [a_lat, a_lon],
        popup="起点A（南京科技职业学院）",
        icon=folium.Icon(color="red", icon="play")
    ).add_to(m)

    # 2. 终点B（绿色标记）
    folium.Marker(
        [b_lat, b_lon],
        popup="终点B（南京科技职业学院）",
        icon=folium.Icon(color="green", icon="flag")
    ).add_to(m)

    # 3. 无人机（黄色圆点，随心跳移动）
    folium.CircleMarker(
        [drone_lat, drone_lon],
        radius=8,
        popup=f"无人机\n进度: {progress*100:.1f}%\n高度: {flight_height}m",
        color="orange",
        fill=True,
        fill_color="yellow"
    ).add_to(m)

    # 4. 飞行航线（蓝色线）
    folium.PolyLine(
        locations=[[a_lat, a_lon], [b_lat, b_lon]],
        color="blue",
        weight=3,
        opacity=0.8
    ).add_to(m)

    # 5. 障碍物（黑色方块标记）
    for obs_lat, obs_lon in st.session_state.obstacles:
        folium.CircleMarker(
            [obs_lat, obs_lon],
            radius=12,
            popup="校园障碍物",
            color="black",
            fill=True,
            fill_color="black"
        ).add_to(m)

    # 渲染地图到页面
    with map_placeholder:
        st_folium(m, width=800, height=600, returned_objects=[])

# --------------------------
# 主程序运行
# --------------------------
# 初始渲染校园卫星地图
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
    status_box.success(f"✅ 飞行正常 | 包序号：{current_seq} | 时间：{current_time}")

    # 超时检测计时
    st.session_state.last_received = time.time()
    time.sleep(1)

# 3秒超时报警逻辑
if st.session_state.last_received and not st.session_state.is_running:
    elapsed = time.time() - st.session_state.last_received
    if elapsed > 3 and len(st.session_state.df_history) > 0:
        status_box.error("🚨 连接超时！3秒未收到心跳包！")
    else:
        status_box.warning("⏸️ 飞行已暂停")
