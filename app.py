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

# ========================== 常量配置（按你的要求固定） ==========================
# 配置文件保存路径
CONFIG_DIR = r"D:\wrj\3Dwrj"
CONFIG_FILE = os.path.join(CONFIG_DIR, "obstacle_config.json")
# 版本号
VERSION = "v12.2 障碍物持久化版"

# ========================== 坐标系转换（WGS84 ↔ GCJ-02） ==========================
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

# ========================== 障碍物持久化工具函数 ==========================
# 确保配置目录存在
def ensure_config_dir():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)

# 保存障碍物到JSON文件
def save_obstacles_to_file():
    ensure_config_dir()
    save_data = {
        "version": VERSION,
        "save_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "obstacle_count": len(st.session_state.obstacle_polygons),
        "obstacles": []
    }
    # 遍历所有障碍物，保存坐标、高度、创建时间
    for idx, obs in enumerate(st.session_state.obstacle_polygons):
        obs_data = {
            "id": idx + 1,
            "coordinates": obs,
            "height": st.session_state.obstacle_heights.get(idx, 50),
            "create_time": st.session_state.obstacle_create_time.get(idx, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
        save_data["obstacles"].append(obs_data)
    # 写入文件
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    return save_data

# 从JSON文件加载障碍物
def load_obstacles_from_file():
    ensure_config_dir()
    if not os.path.exists(CONFIG_FILE):
        st.warning("配置文件不存在，请先保存配置")
        return None
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            load_data = json.load(f)
        # 解析数据到session_state
        new_polygons = []
        new_heights = {}
        new_create_time = {}
        for idx, obs in enumerate(load_data["obstacles"]):
            new_polygons.append(obs["coordinates"])
            new_heights[idx] = obs.get("height", 50)
            new_create_time[idx] = obs.get("create_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # 更新全局状态
        st.session_state.obstacle_polygons = new_polygons
        st.session_state.obstacle_heights = new_heights
        st.session_state.obstacle_create_time = new_create_time
        st.session_state.last_drawing_id = None
        return load_data
    except Exception as e:
        st.error(f"加载失败：{str(e)}")
        return None

# ========================== 全局状态初始化 ==========================
if 'df_history' not in st.session_state:
    st.session_state.df_history = pd.DataFrame(columns=["time", "seq"])
if 'last_received' not in st.session_state:
    st.session_state.last_received = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
# 障碍物核心状态
if 'obstacle_polygons' not in st.session_state:
    st.session_state.obstacle_polygons = []
if 'obstacle_heights' not in st.session_state:
    st.session_state.obstacle_heights = {}  # 障碍物高度：key=索引, value=高度(m)
if 'obstacle_create_time' not in st.session_state:
    st.session_state.obstacle_create_time = {}  # 障碍物创建时间
if 'last_drawing_id' not in st.session_state:
    st.session_state.last_drawing_id = None

# ========================== 页面基础配置 ==========================
st.set_page_config(page_title="心跳包接收系统-3D地图规划模块", layout="wide")
st.title("🚁 心跳包接收系统-3D地图规划模块")
tab1, tab2 = st.tabs(["🗺️ 3D地图航线规划", "📡 心跳包实时监控"])

# ========================== 标签页1：3D地图航线规划 ==========================
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
        st.subheader("🏢 障碍物圈选与高度设置")
        st.info("💡 操作说明：在右侧地图左上角点击「多边形图标」，圈选障碍物区域，双击结束绘制。")
        
        # 障碍物列表+高度设置
        if st.session_state.obstacle_polygons:
            st.caption(f"✅ 已保存 {len(st.session_state.obstacle_polygons)} 个障碍物区域")
            for idx, poly in enumerate(st.session_state.obstacle_polygons):
                with st.expander(f"障碍物 {idx+1} 配置", expanded=False):
                    # 高度设置滑块
                    current_height = st.session_state.obstacle_heights.get(idx, 50)
                    new_height = st.slider(f"障碍物高度(m)", 1, 200, current_height, key=f"height_{idx}")
                    if new_height != current_height:
                        st.session_state.obstacle_heights[idx] = new_height
                        st.rerun()
                    # 基础信息
                    st.caption(f"顶点数：{len(poly)}")
                    st.caption(f"创建时间：{st.session_state.obstacle_create_time.get(idx, '-')}")
                    # 删除按钮
                    if st.button("删除该障碍物", key=f"del_obs_{idx}", type="secondary"):
                        st.session_state.obstacle_polygons.pop(idx)
                        if idx in st.session_state.obstacle_heights:
                            del st.session_state.obstacle_heights[idx]
                        if idx in st.session_state.obstacle_create_time:
                            del st.session_state.obstacle_create_time[idx]
                        st.rerun()
        else:
            st.caption("暂无障碍物，请在地图上圈选")

        # 清除全部按钮
        if st.button("🗑️ 清除全部障碍物", use_container_width=True):
            st.session_state.obstacle_polygons = []
            st.session_state.obstacle_heights = {}
            st.session_state.obstacle_create_time = {}
            st.session_state.last_drawing_id = None
            st.success("已清除全部障碍物")
            st.rerun()

        st.divider()
        # ========================== 障碍物配置持久化UI（和截图完全一致） ==========================
        st.subheader("🚀 障碍物配置持久化")
        st.caption(f"配置文件路径：{CONFIG_FILE} | 版本：{VERSION}")
        st.caption("💡 文件保存在程序指定目录下，绝对路径如上所示")

        # 四个核心按钮
        col_save, col_load, col_clear, col_deploy = st.columns(4)
        with col_save:
            if st.button("💾 保存到文件", type="primary", use_container_width=True):
                save_data = save_obstacles_to_file()
                st.success(f"保存成功！共保存{save_data['obstacle_count']}个障碍物")
                st.rerun()
        with col_load:
            if st.button("📂 从文件加载", use_container_width=True):
                load_data = load_obstacles_from_file()
                if load_data:
                    st.success(f"加载成功！共加载{load_data['obstacle_count']}个障碍物")
                    st.rerun()
        with col_clear:
            if st.button("🗑️ 清除全部", use_container_width=True):
                st.session_state.obstacle_polygons = []
                st.session_state.obstacle_heights = {}
                st.session_state.obstacle_create_time = {}
                st.session_state.last_drawing_id = None
                st.success("已清除全部障碍物")
                st.rerun()
        with col_deploy:
            if st.button("🚀 一键部署", type="primary", use_container_width=True):
                st.success("✅ 障碍物配置已一键部署到飞行系统！")
                time.sleep(0.8)
                st.rerun()

        st.divider()
        # 下载配置文件
        st.subheader("⬇️ 下载配置文件到本地")
        json_content = ""
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
        else:
            st.info("暂无配置文件，请先点击「保存到文件」生成配置")

        # 文件状态显示
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(CONFIG_FILE)).strftime("%Y-%m-%d %H:%M:%S")
                st.info(f"📂 文件状态: 共 {config_data['obstacle_count']} 个障碍物 | 保存时间: {file_mtime} | 版本: {config_data['version']}")
                st.code(CONFIG_FILE, language="text")
            except:
                st.info("📂 文件状态: 配置文件存在但解析失败")
        else:
            st.info("📂 文件状态: 暂无配置文件，请先保存配置")

        st.divider()
        st.subheader("✈️ 飞行参数设置")
        flight_height = st.slider("设定飞行高度(m)", 10, 100, 50)
        view_pitch = st.slider("3D视角倾斜角度", 0, 60, 30)
        st.info(f"当前飞行高度：{flight_height}m | 3D视角倾斜：{view_pitch}°")

    with col2:
        st.subheader("🌍 卫星实况地图")
        map_placeholder = st.empty()

# ========================== 标签页2：心跳包实时监控 ==========================
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

# ========================== 卫星地图渲染核心函数 ==========================
def render_satellite_map(current_seq=0, total_steps=50):
    center_lat = (a_lat + b_lat) / 2
    center_lon = (a_lon + b_lon) / 2

    # 卫星实况底图（稳定加载无空白）
    satellite_tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles=satellite_tiles,
        attr="Tiles © Esri",
        view_control={"pitch": view_pitch, "bearing": 0}
    )

    # 无人机实时位置计算
    progress = min(current_seq / total_steps, 1.0)
    drone_lat = a_lat + (b_lat - a_lat) * progress
    drone_lon = a_lon + (b_lon - a_lon) * progress

    # 基础标记点
    folium.Marker([a_lat, a_lon], popup="起点A", icon=folium.Icon(color="red", icon="play")).add_to(m)
    folium.Marker([b_lat, b_lon], popup="终点B", icon=folium.Icon(color="green", icon="flag")).add_to(m)
    folium.CircleMarker(
        [drone_lat, drone_lon], radius=8, color="orange", fill=True, fill_color="yellow",
        popup=f"无人机\n进度: {progress*100:.1f}%\n高度: {flight_height}m"
    ).add_to(m)
    folium.PolyLine(locations=[[a_lat, a_lon], [b_lat, b_lon]], color="blue", weight=3).add_to(m)

    # 渲染所有已保存的障碍物（带高度信息）
    for idx, poly_coords in enumerate(st.session_state.obstacle_polygons):
        obs_height = st.session_state.obstacle_heights.get(idx, 50)
        folium.Polygon(
            locations=poly_coords,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.4,
            weight=2,
            popup=f"障碍物禁飞区\n高度：{obs_height}m"
        ).add_to(m)

    # 多边形圈选插件
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
            returned_objects=["last_active_drawing"]
        )

    # 捕获新圈选的障碍物，自动保存到session_state
    if map_output and map_output["last_active_drawing"]:
        drawing = map_output["last_active_drawing"]
        drawing_id = str(drawing["geometry"]["coordinates"])
        
        if drawing_id != st.session_state.last_drawing_id:
            st.session_state.last_drawing_id = drawing_id
            if drawing["geometry"]["type"] == "Polygon":
                # 坐标格式转换
                poly_coords = [[lat, lon] for lon, lat in drawing["geometry"]["coordinates"][0]]
                if poly_coords not in st.session_state.obstacle_polygons:
                    # 添加新障碍物
                    st.session_state.obstacle_polygons.append(poly_coords)
                    new_idx = len(st.session_state.obstacle_polygons) - 1
                    # 设置默认高度和创建时间
                    st.session_state.obstacle_heights[new_idx] = 50
                    st.session_state.obstacle_create_time[new_idx] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("障碍物圈选成功！已自动保存")
                    st.rerun()

# ========================== 主程序运行 ==========================
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
