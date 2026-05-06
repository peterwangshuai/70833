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
    st.session_state.obstacles = []  # 存储障碍物：[(纬度, 经度), ...] 全自主管理

# --------------------------
# 页面配置（项目名精准修改）
# --------------------------
st.set_page_config(page_title="心跳包接收系统-3D地图规划模块", layout="wide")
st.title("🚁 心跳包接收系统-3D地图规划模块")
# 双标签页贴合项目功能
tab1, tab2 = st.tabs(["🗺️ 3D地图航线规划", "📡 心跳包实时监控"])

# --------------------------
# 标签页1：3D地图航线规划（核心模块）
# --------------------------
with tab1:
    st.header("🗺️ 校园航线规划（卫星底图，可缩放/拖拽/3D视角）")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📍 A/B点坐标设置（南京科技职业学院校园内）")
        # 校园内默认坐标，A点教学楼，B点操场，中间可设置障碍物
        a_lat = st.number_input("A点纬度(GCJ-02)", value=32.2322, format="%.4f")
        a_lon = st.number_input("A点经度(GCJ-02)", value=118.7490, format="%.4f")
        b_lat = st.number_input("B点纬度(GCJ-02)", value=32.2343, format="%.4f")
        b_lon = st.number_input("B点经度(GCJ-02)", value=118.7490, format="%.4f")

        st.divider()
        # ========== 障碍物全自主设置核心优化 ==========
        st.subheader("🏢 障碍物自主设置")
        obstacle_lat = st.number_input("障碍物纬度", value=32.2332, format="%.4f")
        obstacle_lon = st.number_input("障碍物经度", value=118.7490, format="%.4f")
        
        # 新增障碍物按钮
        col_add, col_clear = st.columns(2)
        with col_add:
            if st.button("✅ 添加障碍物", type="primary"):
                st.session_state.obstacles.append((obstacle_lat, obstacle_lon))
                st.success("障碍物添加成功！")
        with col_clear:
            if st.button("🗑️ 清空全部"):
                st.session_state.obstacles = []
                st.info("已清空所有障碍物")

        # 已添加障碍物列表 + 单个删除（完全自主管理）
        if st.session_state.obstacles:
            st.caption("已添加障碍物列表")
            for idx, (obs_lat, obs_lon) in enumerate(st.session_state.obstacles):
                col_obs, col_del = st.columns([4, 1])
                with col_obs:
                    st.text(f"障碍物{idx+1}：{obs_lat}, {obs_lon}")
                with col_del:
                    if st.button("删除", key=f"del_{idx}"):
                        st.session_state.obstacles.pop(idx)
                        st.rerun()
        # ==============================================

        st.divider()
        st.subheader("✈️ 飞行参数设置")
        flight_height = st.slider("设定飞行高度(m)", 10, 100, 50)
        view_pitch = st.slider("3D视角倾斜角度", 0, 60, 30)
        st.info(f"当前飞行高度：{flight_height}m | 3D视角倾斜：{view_pitch}°")

    with col2:
        st.subheader("🌍 南京科技职业学院 3D卫星地图")
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
# 3D卫星地图渲染函数（稳定加载+3D视角）
# --------------------------
def render_3d_map(current_seq=0, total_steps=50):
    # 地图中心点：南京科技职业学院
    center_lat = (a_lat + b_lat) / 2
    center_lon = (a_lon + b_lon) / 2

    # ArcGIS全球稳定卫星底图（无空白问题，校园清晰）
    satellite_tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    # 创建地图，支持3D倾斜视角
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles=satellite_tiles,
        attr="Tiles © Esri",
        # 3D视角核心参数
        view_control={"pitch": view_pitch, "bearing": 0}
    )

    # 计算无人机实时位置
    progress = min(current_seq / total_steps, 1.0)
    drone_lat = a_lat + (b_lat - a_lat) * progress
    drone_lon = a_lon + (b_lon - a_lon) * progress

    # 标记点渲染
    folium.Marker([a_lat, a_lon], popup="起点A", icon=folium.Icon(color="red", icon="play")).add_to(m)
    folium.Marker([b_lat, b_lon], popup="终点B", icon=folium.Icon(color="green", icon="flag")).add_to(m)
    folium.CircleMarker(
        [drone_lat, drone_lon], radius=8, color="orange", fill=True, fill_color="yellow",
        popup=f"无人机\n进度: {progress*100:.1f}%\n高度: {flight_height}m"
    ).add_to(m)
    folium.PolyLine(locations=[[a_lat, a_lon], [b_lat, b_lon]], color="blue", weight=3).add_to(m)

    # 自主设置的障碍物渲染
    for obs_lat, obs_lon in st.session_state.obstacles:
        folium.CircleMarker(
            [obs_lat, obs_lon], radius=12, color="black", fill=True, fill_color="black",
            popup=f"障碍物\n坐标：{obs_lat}, {obs_lon}"
        ).add_to(m)

    # 渲染地图到页面
    with map_placeholder:
        st_folium(m, width=800, height=600, returned_objects=[])

# --------------------------
# 主程序运行
# --------------------------
# 初始渲染3D地图
render_3d_map(len(st.session_state.df_history))

# 实时心跳+地图更新循环
while st.session_state.is_running:
    current_seq = len(st.session_state.df_history) + 1
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    new_data = pd.DataFrame({"time": [current_time], "seq": [current_seq]})

    st.session_state.df_history = pd.concat([st.session_state.df_history, new_data], ignore_index=True)
    
    # 实时更新心跳折线图
    chart_obj.add_rows(new_data)
    # 实时更新3D地图（无人机移动）
    render_3d_map(current_seq)
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
