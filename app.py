import streamlit as st
import time
import datetime
import pandas as pd

# --------------------------
# 1. 初始化 Session State (保持页面刷新后的状态)
# --------------------------
if 'heartbeats' not in st.session_state:
    st.session_state.heartbeats = []  # 存储数据: [{'seq': 1, 'time': '12:00:00'}, ...]
if 'last_time' not in st.session_state:
    st.session_state.last_time = None  # 记录上一次收到包的时间戳
if 'is_running' not in st.session_state:
    st.session_state.is_running = False # 控制模拟开关
if 'timeout' not in st.session_state:
    st.session_state.timeout = False    # 超时状态标记

# --------------------------
# 2. 页面 UI 布局
# --------------------------
st.set_page_config(page_title="无人机心跳监控", layout="wide")
st.title("🚁 无人机心跳包模拟系统")

# 控制面板
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("▶️ 开始模拟", type="primary"):
        st.session_state.is_running = True
        st.session_state.timeout = False
with col2:
    if st.button("⏸️ 停止模拟"):
        st.session_state.is_running = False
with col3:
    if st.button("🔄 重置数据"):
        st.session_state.heartbeats = []
        st.session_state.last_time = None
        st.session_state.timeout = False
        st.session_state.is_running = False

# --------------------------
# 3. 核心逻辑：心跳生成与超时检测
# --------------------------
placeholder = st.empty() # 占位符，用于动态显示状态

if st.session_state.is_running:
    # --- A. 生成心跳包 (自发自收) ---
    new_seq = len(st.session_state.heartbeats) + 1
    new_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3] # 精确到毫秒
    
    # 模拟发送并接收
    st.session_state.heartbeats.append({
        "序号": new_seq,
        "时间": new_time
    })
    
    # 记录接收时间
    st.session_state.last_time = time.time()
    
    # --- B. 模拟 1秒 间隔 ---
    time.sleep(1)
    
    # 刷新页面
    st.rerun()

# --- C. 超时检测 (3秒规则) ---
if st.session_state.last_time is not None:
    elapsed = time.time() - st.session_state.last_time
    if elapsed > 3 and st.session_state.is_running:
        st.session_state.timeout = True
        st.session_state.is_running = False # 超时自动停止

# --------------------------
# 4. 状态显示与可视化
# --------------------------

# 超时警告
if st.session_state.timeout:
    st.error("❌ 连接超时！超过 3 秒未收到心跳包！", icon="🚨")

# 分为两列展示图表和数据
chart_col, data_col = st.columns([2, 1])

with chart_col:
    st.subheader("📈 心跳包序号随时间变化")
    if st.session_state.heartbeats:
        df = pd.DataFrame(st.session_state.heartbeats)
        # 绘制折线图
        st.line_chart(df, x="时间", y="序号", color="#FF4B4B")
    else:
        st.info("请点击开始模拟以生成数据")

with data_col:
    st.subheader("📋 数据包列表")
    if st.session_state.heartbeats:
        # 显示最新的 10 条数据，倒序排列
        df = pd.DataFrame(st.session_state.heartbeats)
        st.dataframe(df.sort_index(ascending=False).head(10), hide_index=True, use_container_width=True)
        
        # 显示统计
        st.caption(f"总计收到包数: {len(st.session_state.heartbeats)}")
