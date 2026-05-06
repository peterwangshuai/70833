import streamlit as st
import time
import datetime
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# --------------------------
# 坐标系转换工具（WGS84 ↔ GCJ-02）
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
if 'is_timeout' not in st.session_state:
    st.session_state.is_timeout = False

# --------------------------
# 页面配置 + 双标签页
# --------------------------
st.set_page_config(page_title="无人机综合监控系统", layout="wide")
st.title("🚁 无人机综合监控系统")
tab1, tab2 = st.tabs(["🗺️ 航线规划", "📡 飞行监控"])

# --------------------------
# 标签页1：航线规划（2D地图 + 坐标转换）
# --------------------------
with tab1:
    st.header("🗺️ 2D航线规划（支持缩放/拖拽）")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📍 A/B点坐标设置")
        a_lat = st.number_input("A点纬度(WGS84)", value=39.904200, format="%.6f")
        a_lon = st.number_input("A点经度(WGS84)", value=116.407400, format="%.6f")
        b_lat = st.number_input("B点纬度(WGS84)", value=31.230400, format="%.6f")
        b_lon = st.number_input("B点经度(WGS84)", value=121.473700, format="%.6f")

        st.divider()
        st.subheader("🔄 坐标系转换结果")
        a_gcj_lat, a_gcj_lon = wgs84_to_gcj02(a_lat, a_lon)
        b_gcj_lat, b_gcj_lon = wgs84_to_gcj02(b_lat, b_lon)
        st.info(f"A点(GCJ-02)：{a_gcj_lat}, {a_gcj_lon}")
        st.info(f"B点(GCJ-02)：{b_gcj_lat}, {b_gcj_lon}")

    with col2:
        st.subheader("🌍 二维地图视图")
        map_placeholder = st.empty()

# --------------------------
# 标签页2：飞行监控（心跳包 + 实时图表）
# --------------------------
with tab2:
    st.header("📡 心跳包实时监控")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶️ 启动飞行", type="primary"):
            st.session_state.is_running = True
            st.session_state.is_timeout = False
    with c2:
        if st.button("⏸️ 暂停飞行"):
            st.session_state.is_running = False
    with c3:
        if st.button("🔄 重置数据"):
            st.session_state.df_history = pd.DataFrame(columns=["time", "seq"])
            st.session_state.last_received = None
            st.session_state.is_running = False
            st.session_state.is_timeout = False
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
# 2D地图渲染函数（Folium）
# --------------------------
def render_folium_map(current_seq=0, total_steps=50):
    center_lat = (a_lat + b_lat) / 2
    center_lon = (a_lon + b_lon) / 2

    # 纯二维地图
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='OpenStreetMap')

    # 无人机实时位置
    progress = min(current_seq / total_steps, 1.0)
    drone_lat = a_lat + (b_lat - a_lat) * progress
    drone_lon = a_lon + (b_lon - a_lon) * progress

    # 添加标记
    folium.Marker([a_lat, a_lon], popup="起点 A", icon=folium.Icon(color="red")).add_to(m)
    folium.Marker([b_lat, b_lon], popup="终点 B", icon=folium.Icon(color="green")).add_to(m)
    folium.CircleMarker([drone_lat, drone_lon], radius=10, color="orange", fill=True, fill_color="yellow", popup="无人机").add_to(m)
    folium.PolyLine(locations=[[a_lat, a_lon], [b_lat, b_lon]], color="blue", weight=3).add_to(m)

    # 显示地图
    with map_placeholder:
        st_folium(m, width=700, height=500)

# --------------------------
# 主程序运行
# --------------------------
render_folium_map(len(st.session_state.df_history))

# 实时心跳+地图更新循环
while st.session_state.is_running:
    current_seq = len(st.session_state.df_history) + 1
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    new_data = pd.DataFrame({"time": [current_time], "seq": [current_seq]})

    st.session_state.df_history = pd.concat([st.session_state.df_history, new_data], ignore_index=True)
    
    chart_obj.add_rows(new_data)
    render_folium_map(current_seq)
    data_box.dataframe(st.session_state.df_history.tail(10), hide_index=True, height=300)
    status_box.success(f"✅ 飞行正常 | 包序号：{current_seq} | 时间：{current_time}")

    st.session_state.last_received = time.time()
    time.sleep(1)

# 超时检测
if st.session_state.last_received and not st.session_state.is_running:
    elapsed = time.time() - st.session_state.last_received
    if elapsed > 3 and len(st.session_state.df_history) > 0:
        status_box.error("🚨 连接超时！3秒未收到心跳包！")
    else:
        status_box.warning("⏸️ 飞行已暂停")
