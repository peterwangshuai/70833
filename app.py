import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np

from drone_simulator import DroneHeartbeatSimulator

# 页面配置
st.set_page_config(
    page_title="无人机心跳监控系统",
    page_icon="🚁",
    layout="wide"
)

# 初始化session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = DroneHeartbeatSimulator()
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# 标题
st.title("🚁 无人机心跳监控系统")
st.markdown("---")

# 侧边栏控制面板
with st.sidebar:
    st.header("控制面板")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ 启动监控", use_container_width=True):
            if not st.session_state.is_running:
                st.session_state.simulator.start()
                st.session_state.is_running = True
                st.success("心跳监控已启动")
                st.rerun()
    
    with col2:
        if st.button("⏹️ 停止监控", use_container_width=True):
            if st.session_state.is_running:
                st.session_state.simulator.stop()
                st.session_state.is_running = False
                st.warning("心跳监控已停止")
                st.rerun()
    
    st.markdown("---")
    st.header("监控参数")
    timeout_threshold = st.number_input(
        "超时阈值（秒）",
        min_value=1,
        max_value=10,
        value=3,
        help="连续多少秒未收到心跳包判定为超时"
    )
    
    st.markdown("---")
    st.header("关于")
    st.info("""
    - 心跳频率：1次/秒
    - 超时时间：3秒
    - 实时监控连接状态
    - 自动记录所有心跳数据
    """)

# 主显示区域
# 状态显示
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.is_running:
        # 检查连接状态
        is_connected, timeout_count = st.session_state.simulator.get_connection_status()
        
        if is_connected:
            st.metric(
                label="连接状态",
                value="🟢 在线",
                delta="正常连接"
            )
        else:
            st.metric(
                label="连接状态",
                value="🔴 离线",
                delta=f"超时 {timeout_count} 次",
                delta_color="inverse"
            )
    else:
        st.metric(
            label="连接状态",
            value="⚪ 未启动",
            delta="点击启动按钮"
        )

with col2:
    if st.session_state.is_running:
        latest = st.session_state.simulator.get_latest_heartbeat()
        if latest:
            st.metric(
                label="最新心跳序号",
                value=f"#{latest['seq']}",
                delta=f"时间: {latest['timestamp']}"
            )
        else:
            st.metric(
                label="最新心跳序号",
                value="等待中..."
            )
    else:
        st.metric(
            label="最新心跳序号",
            value="---"
        )

with col3:
    if st.session_state.is_running:
        data_count = len(st.session_state.simulator.get_heartbeat_data())
        st.metric(
            label="接收心跳总数",
            value=f"{data_count} 个",
            delta=f"{(data_count / (time.time() - st.session_state.last_update)):.1f}个/秒" if data_count > 0 else "0个/秒"
        )
    else:
        st.metric(
            label="接收心跳总数",
            value="0 个"
        )

st.markdown("---")

# 创建两列布局
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 心跳序号时间序列图")
    
    # 获取数据
    if st.session_state.is_running:
        heartbeat_data = st.session_state.simulator.get_heartbeat_data()
        
        if heartbeat_data:
            # 转换为DataFrame
            df = pd.DataFrame(heartbeat_data)
            
            # 创建图表
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=("心跳序号变化趋势", "心跳间隔时间"),
                vertical_spacing=0.12,
                row_heights=[0.6, 0.4]
            )
            
            # 第一张图：心跳序号随时间变化
            fig.add_trace(
                go.Scatter(
                    x=df['full_datetime'],
                    y=df['seq'],
                    mode='lines+markers',
                    name='心跳序号',
                    line=dict(color='#00BFFF', width=2),
                    marker=dict(size=6, color='#1E88E5'),
                    hovertemplate='序号: %{y}<br>时间: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 计算心跳间隔
            if len(df) > 1:
                time_diffs = []
                for i in range(1, len(df)):
                    diff = (df['full_datetime'].iloc[i] - df['full_datetime'].iloc[i-1]).total_seconds()
                    time_diffs.append(diff)
                
                diff_df = pd.DataFrame({
                    'time': df['full_datetime'].iloc[1:],
                    'interval': time_diffs
                })
                
                # 第二张图：心跳间隔时间
                fig.add_trace(
                    go.Scatter(
                        x=diff_df['time'],
                        y=diff_df['interval'],
                        mode='lines+markers',
                        name='心跳间隔',
                        line=dict(color='#FF6B6B', width=2),
                        marker=dict(size=6, color='#FF5252'),
                        fill='tozeroy',
                        fillcolor='rgba(255, 107, 107, 0.2)',
                        hovertemplate='间隔: %{y:.2f}秒<br>时间: %{x}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # 添加阈值线
                fig.add_hline(
                    y=1.0, 
                    line_dash="dash", 
                    line_color="orange",
                    annotation_text="期望间隔(1秒)",
                    row=2, col=1
                )
                
                fig.add_hline(
                    y=3.0, 
                    line_dash="dot", 
                    line_color="red",
                    annotation_text="超时阈值(3秒)",
                    row=2, col=1
                )
            
            # 更新布局
            fig.update_layout(
                height=600,
                showlegend=True,
                hovermode='x unified',
                title_text="无人机心跳监控数据",
                title_x=0.5
            )
            
            fig.update_xaxes(title_text="时间", row=2, col=1)
            fig.update_yaxes(title_text="心跳序号", row=1, col=1)
            fig.update_yaxes(title_text="间隔时间(秒)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示超时告警信息
            is_connected, timeout_count = st.session_state.simulator.get_connection_status()
            if not is_connected:
                st.error(f"⚠️ 连接超时！已超过 {timeout_threshold} 秒未收到心跳包")
            
        else:
            st.info("等待接收心跳数据...")
    else:
        st.info("请点击左侧「启动监控」按钮开始接收心跳数据")

with col_right:
    st.subheader("📋 实时数据列表")
    
    if st.session_state.is_running:
        heartbeat_data = st.session_state.simulator.get_heartbeat_data()
        
        if heartbeat_data:
            # 显示最近10条数据
            recent_data = heartbeat_data[-10:]
            
            # 创建显示用的DataFrame
            display_df = pd.DataFrame(recent_data[::-1])  # 倒序显示最新的在上面
            display_df = display_df[['seq', 'timestamp']]
            display_df.columns = ['序号', '时间']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400,
                hide_index=True
            )
            
            # 统计数据
            st.markdown("---")
            st.subheader("📈 统计信息")
            
            total = len(heartbeat_data)
            latest = heartbeat_data[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("总接收数", total)
                st.metric("当前序号", latest['seq'])
            with col2:
                _, timeout_count = st.session_state.simulator.get_connection_status()
                st.metric("超时次数", timeout_count)
                if total > 0:
                    st.metric("丢包率", f"{(timeout_count / total * 100):.1f}%" if timeout_count > 0 else "0%")
            
        else:
            st.info("暂无数据")
    else:
        st.info("监控未启动")

# 自动刷新（每0.5秒）
if st.session_state.is_running:
    time.sleep(0.5)
    st.rerun()

# 底部信息
st.markdown("---")
st.caption("🚁 无人机心跳监控系统 | 心跳频率: 1次/秒 | 超时检测: 3秒")
