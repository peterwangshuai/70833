import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ===========================================
# 页面配置
# ===========================================
st.set_page_config(
    page_title="心跳包监控仪表板",
    page_icon="❤️",
    layout="wide"
)

# ===========================================
# 自定义样式 - GitHub 风格深色/浅色自适应
# ===========================================
st.markdown("""
<style>
    /* 主容器背景柔和 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 标题样式 */
    h1 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* 卡片样式 */
    .metric-card {
        background-color: rgba(27, 31, 35, 0.05);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(27, 31, 35, 0.15);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    
    /* 暗色模式适配 */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: rgba(255, 255, 255, 0.07);
            border-color: rgba(255, 255, 255, 0.15);
        }
    }
</style>
""", unsafe_allow_html=True)

# ===========================================
# 辅助函数：生成模拟心跳包数据
# ===========================================
def generate_heartbeat_data(duration_seconds=120, sample_rate_hz=2):
    """
    生成模拟的心跳包序号时间序列数据
    
    Args:
        duration_seconds: 数据时长（秒）
        sample_rate_hz: 每秒采样点数（模拟包发送频率）
    
    Returns:
        pd.DataFrame: 包含时间戳和序号的数据框
    """
    total_points = duration_seconds * sample_rate_hz
    # 生成均匀的时间戳（从当前时间往前推 duration_seconds 秒）
    end_time = datetime.now()
    start_time = end_time - timedelta(seconds=duration_seconds)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=total_points)
    
    # 心跳包序号：基础线性增长 + 随机微小抖动 + 偶尔的丢包重传模拟（序号跳跃）
    base_seq = np.arange(1, total_points + 1)
    # 添加轻微噪声模拟网络延迟乱序（但序号整体递增）
    noise = np.random.normal(0, 0.3, total_points).cumsum() * 0.2
    # 模拟偶尔的序号跳跃（例如重传或突发）
    jumps = np.zeros(total_points)
    num_jumps = np.random.randint(2, 6)
    jump_indices = np.random.choice(total_points - 1, num_jumps, replace=False)
    for idx in jump_indices:
        jumps[idx:] += np.random.randint(3, 15)
    
    sequence = base_seq + noise + jumps
    sequence = np.maximum.accumulate(sequence)  # 确保序号非递减（心跳包序号逻辑上单调递增）
    
    # 确保整数序号（保留一位小数模拟精度，展示更真实）
    sequence = np.round(sequence, 1)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'heartbeat_seq': sequence
    })
    return df

# ===========================================
# 主应用界面
# ===========================================
def main():
    # 头部区域
    col_title, col_github = st.columns([3, 1])
    with col_title:
        st.title("❤️ 心跳包实时监控")
        st.markdown("基于 **Streamlit** 构建的轻量级数据看板 | 展示心跳包序号随时间的变化趋势")
    with col_github:
        st.markdown("""
        <div style="text-align: right; padding-top: 1rem;">
            <a href="https://github.com" target="_blank" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/GitHub-源码仓库-181717?style=flat&logo=github" alt="GitHub">
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 侧边栏控制区
    with st.sidebar:
        st.header("⚙️ 数据控制面板")
        st.markdown("可调整模拟数据的时长与采样频率，刷新图表以观察不同场景下的心跳包行为。")
        
        duration = st.slider(
            "数据时长 (秒)",
            min_value=30,
            max_value=300,
            value=120,
            step=10,
            help="模拟心跳包历史数据的时间跨度"
        )
        
        sample_rate = st.slider(
            "采样频率 (包/秒)",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="每秒发送的心跳包数量，影响数据密度"
        )
        
        st.markdown("---")
        st.markdown("### 📊 数据特征")
        st.info("""
        - 序号总体单调递增  
        - 包含随机抖动模拟网络波动  
        - 偶尔出现序号跳跃（重传/突发场景）
        """)
        
        regenerate = st.button("🔄 重新生成数据", use_container_width=True)
    
    # 生成或更新数据
    if 'df' not in st.session_state or regenerate:
        with st.spinner("正在生成心跳包数据..."):
            st.session_state.df = generate_heartbeat_data(
                duration_seconds=duration,
                sample_rate_hz=sample_rate
            )
    df = st.session_state.df
    
    # 关键指标卡片 (KPI)
    col1, col2, col3, col4 = st.columns(4)
    
    # 计算指标
    latest_seq = df['heartbeat_seq'].iloc[-1]
    first_seq = df['heartbeat_seq'].iloc[0]
    total_packets = len(df)
    time_span_sec = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    avg_rate = total_packets / time_span_sec if time_span_sec > 0 else 0
    seq_growth = latest_seq - first_seq
    
    with col1:
        st.metric(
            label="最新心跳序号",
            value=f"{latest_seq:.1f}",
            delta=f"+{seq_growth:.1f}",
            help="最新的心跳包序号及从起始到现在的总增长量"
        )
    with col2:
        st.metric(
            label="总心跳包数量",
            value=f"{total_packets:,}",
            delta=None,
            help="时间范围内捕获的心跳包总数"
        )
    with col3:
        st.metric(
            label="平均发送频率",
            value=f"{avg_rate:.2f} 包/秒",
            delta=None,
            help="实际数据包平均速率"
        )
    with col4:
        st.metric(
            label="时间跨度",
            value=f"{time_span_sec:.0f} 秒",
            delta=None,
            help="数据显示的时间范围"
        )
    
    st.divider()
    
    # 核心折线图 - 使用 Plotly 实现交互
    st.subheader("📈 心跳包序号时间序列")
    
    # 构建 Plotly 图表
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['heartbeat_seq'],
        mode='lines+markers',
        name='心跳序号',
        line=dict(color='#e25555', width=2.5),
        marker=dict(size=3, color='#ff7f7f', opacity=0.7),
        hovertemplate='时间: %{x|%Y-%m-%d %H:%M:%S}<br>序号: %{y:.1f}<extra></extra>'
    ))
    
    # 优化布局
    fig.update_layout(
        title=dict(
            text="心跳包序号随时间演化趋势",
            font=dict(size=18, weight='bold'),
            x=0.02,
            xanchor='left'
        ),
        xaxis=dict(
            title="时间戳",
            titlefont=dict(size=12),
            gridcolor='lightgray',
            showgrid=True,
            gridwidth=0.5,
            rangeslider=dict(visible=True, thickness=0.05),  # 添加范围滑块方便局部放大
            type='date'
        ),
        yaxis=dict(
            title="心跳包序号",
            titlefont=dict(size=12),
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
            tickformat='.1f'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=500,
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # 添加辅助线（平滑趋势线，可选）
    # 计算移动平均平滑曲线展示整体趋势
    window = max(5, int(len(df) * 0.03))  # 动态窗口
    df['smooth'] = df['heartbeat_seq'].rolling(window=window, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['smooth'],
        mode='lines',
        name=f'趋势平滑 (窗口={window})',
        line=dict(color='#2c7bb6', width=2, dash='dash'),
        opacity=0.6,
        hoverinfo='skip'
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 附加分析区域: 序号增量分布直方图
    st.subheader("📊 序号增量分析")
    
    # 计算相邻心跳包的序号差值（增量）
    df['delta'] = df['heartbeat_seq'].diff().fillna(0)
    # 过滤掉零值（理论上心跳包序号每次应增加，但可能有极小抖动为负？由于用了累加最大值，非负，但可展示实际增量分布）
    delta_pos = df[df['delta'] > 0]['delta'].values
    
    col_hist, col_stats = st.columns([2, 1])
    
    with col_hist:
        # 使用 Plotly 绘制直方图
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=delta_pos,
            nbinsx=30,
            marker_color='#58a6ff',
            opacity=0.7,
            name='序号增量'
        ))
        hist_fig.update_layout(
            title="相邻心跳包序号增量分布",
            xaxis_title="序号增量",
            yaxis_title="频次",
            template='plotly_white',
            height=350,
            bargap=0.05
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    
    with col_stats:
        st.markdown("### 📐 统计摘要")
        if len(delta_pos) > 0:
            stats_data = {
                "均值 (Mean)": f"{np.mean(delta_pos):.3f}",
                "中位数 (Median)": f"{np.median(delta_pos):.3f}",
                "标准差 (Std)": f"{np.std(delta_pos):.3f}",
                "最小值 (Min)": f"{np.min(delta_pos):.3f}",
                "最大值 (Max)": f"{np.max(delta_pos):.3f}",
                "95% 分位数": f"{np.percentile(delta_pos, 95):.3f}"
            }
            for key, value in stats_data.items():
                st.markdown(f"**{key}:** `{value}`")
        else:
            st.info("无有效增量数据（可能数据点过少）")
        
        st.markdown("---")
        st.caption("💡 注：理想心跳包序号每次增加1，实际受网络抖动、重传等影响增量会波动。此处模拟了轻微的噪声与突发跳跃。")
    
    # 数据表折叠展示
    with st.expander("📋 查看原始数据表 (前/后 10 行)", expanded=False):
        st.dataframe(
            df[['timestamp', 'heartbeat_seq', 'delta']].round(2),
            use_container_width=True,
            height=300,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("时间戳"),
                "heartbeat_seq": st.column_config.NumberColumn("心跳序号", format="%.1f"),
                "delta": st.column_config.NumberColumn("序号增量", format="%.2f")
            }
        )
        st.caption(f"总共 {len(df)} 条心跳记录 | 数据生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===========================================
if __name__ == "__main__":
    main()
