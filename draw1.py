import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# ==========================================
# 1. 准备数据 (保持不变)
# ==========================================
def generate_mock_metrics():
    # 模拟 S3-S27 (Percentage Era)
    seasons_pct = np.repeat(np.arange(3, 28), 10)
    weeks_pct = np.tile(np.arange(1, 11), 25)
    n_pct = len(seasons_pct)
    
    margins_pct = np.random.beta(5, 2, n_pct) * 0.15 
    feas_pct = np.ones(n_pct)
    corr_pct = np.random.normal(0.6, 0.1, n_pct)
    
    df_pct = pd.DataFrame({
        'season': seasons_pct, 'week': weeks_pct, 'era': 'Percentage',
        'metric_val': margins_pct, 'metric_type': 'Safety Margin',
        'feasibility': feas_pct, 'correlation': corr_pct
    })

    # 模拟 S1-2, S28+ (Rank Era)
    seasons_rnk = np.concatenate([np.repeat([1,2], 10), np.repeat(np.arange(28, 35), 10)])
    weeks_rnk = np.tile(np.arange(1, 11), len(seasons_rnk)//10)
    n_rnk = len(seasons_rnk)
    
    hit_rates = np.random.beta(2, 3, n_rnk)
    # 制造黑天鹅
    hit_rates[15] = 0.0001 
    hit_rates[55] = 0.00005
    
    df_rnk = pd.DataFrame({
        'season': seasons_rnk, 'week': weeks_rnk, 'era': 'Rank',
        'metric_val': hit_rates, 'metric_type': 'Hit Probability',
        'feasibility': (hit_rates > 0).astype(int), 
        'correlation': np.random.normal(0.4, 0.2, n_rnk)
    })
    
    return pd.concat([df_pct, df_rnk])

df = generate_mock_metrics()

# ==========================================
# 2. 绘图 A: 雷达图 (保持不变，重新生成以确保一致)
# ==========================================
def plot_radar_comparison_final(df):
    stats = df.groupby('era')[['feasibility', 'metric_val', 'correlation']].mean()
    
    def normalize_stability(row):
        if row.name == 'Percentage':
            return min(row['metric_val'] / 0.15, 1.0)
        else:
            return min(row['metric_val'] / 0.8, 1.0)

    stats['Stability'] = stats.apply(normalize_stability, axis=1)
    stats['Feasibility'] = stats['feasibility']
    stats['Independence'] = 1 - stats['correlation'] 
    
    categories = ['Feasibility\n(Math Logic)', 'Stability\n(Robustness)', 'Independence\n(Fan Agency)']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    values_rank = stats.loc['Rank', ['Feasibility', 'Stability', 'Independence']].values.flatten().tolist()
    values_rank += values_rank[:1]
    ax.plot(angles, values_rank, linewidth=2.5, linestyle='solid', label='Rank Era (Chaotic)', color='#e74c3c')
    ax.fill(angles, values_rank, '#e74c3c', alpha=0.15)
    
    values_pct = stats.loc['Percentage', ['Feasibility', 'Stability', 'Independence']].values.flatten().tolist()
    values_pct += values_pct[:1]
    ax.plot(angles, values_pct, linewidth=2.5, linestyle='solid', label='Percentage Era (Stable)', color='#3498db')
    ax.fill(angles, values_pct, '#3498db', alpha=0.15)
    
    plt.xticks(angles[:-1], [])
    ax.text(angles[0], 1.15, categories[0], ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333')
    ax.text(angles[1], 1.15, categories[1], ha='left', va='top', fontsize=12, fontweight='bold', color='#333333')
    ax.text(angles[2], 1.15, categories[2], ha='right', va='top', fontsize=12, fontweight='bold', color='#333333')

    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
    plt.ylim(0, 1.1)
    ax.spines['polar'].set_visible(False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=11)
    plt.title("Structural Profile Comparison:\nPercentage vs. Rank Eras", size=15, weight='bold', y=1.1)
    plt.tight_layout()
    plt.savefig('Chart_A_Radar_Final.png', dpi=300, bbox_inches='tight') 
    print("生成 Chart A Final")

# ==========================================
# 3. 绘图 B: 时序图 (修复遮挡问题)
# ==========================================
def plot_timeline_final(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9)) 
    plt.subplots_adjust(hspace=0.4)
    
    # --- Top: Percentage Era ---
    data_pct = df[df['era'] == 'Percentage']
    data_pct['time_idx'] = range(len(data_pct))
    
    # 获取数据最大值，用于设置Y轴上限
    max_val = data_pct['metric_val'].max()
    
    ax1.fill_between(data_pct['time_idx'], 0, data_pct['metric_val'], color='#3498db', alpha=0.5, label='Safety Margin')
    ax1.plot(data_pct['time_idx'], data_pct['metric_val'], color='#2980b9', lw=1.2)
    
    ax1.set_title("Percentage Era: Elimination Safety Margin (Stability)", fontsize=13, fontweight='bold', loc='left', pad=15)
    ax1.set_ylabel("Margin Magnitude", fontsize=11)
    ax1.axhline(0, color='black', lw=1, ls='--')
    
    # === 关键修改：设置 Y 轴上限，留出头部空间 ===
    # 将上限设为最大值的 1.3 倍，确保文字有地方放
    ax1.set_ylim(0, max_val * 1.35)
    
    # === 关键修改：调整文字位置和背景 ===
    ax1.annotate("Consistent Positive Margin\n(High Stability)", 
                 xy=(20, 0.02),             # 箭头指向的数据点 (低位)
                 xytext=(30, max_val * 1.15), # 文字位置 (高位，悬浮在数据之上)
                 arrowprops=dict(facecolor='#2980b9', shrink=0.05, alpha=0.7, width=2, headwidth=8),
                 fontsize=10, color='#205E8E', fontweight='bold',
                 # 添加白色半透明背景框，防止万一重叠也能看清
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
    
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xticks([]) # 隐藏X刻度

    # --- Bottom: Rank Era ---
    data_rnk = df[df['era'] == 'Rank']
    data_rnk['time_idx'] = range(len(data_rnk))
    
    ax2.bar(data_rnk['time_idx'], data_rnk['metric_val'], color='#e74c3c', alpha=0.6, width=1.0, label='Reconstruction Prob')
    
    black_swans = data_rnk[data_rnk['metric_val'] < 0.001]
    # 将黑天鹅标记稍微抬高一点，或者统一高度
    scatter_y = 0.08 # 统一在底部上方一点展示
    ax2.scatter(black_swans['time_idx'], [scatter_y]*len(black_swans), 
                color='black', marker='v', s=80, zorder=5, label='Black Swan Events')
    
    for idx, row in black_swans.iterrows():
        ax2.annotate("Anomaly!", 
                     xy=(row['time_idx'], scatter_y), 
                     xytext=(row['time_idx'], scatter_y + 0.2), # 文字向上引出
                     arrowprops=dict(arrowstyle="->", color='black'),
                     ha='center', color='black', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.6))

    ax2.set_title("Rank Era: Historical Reconstruction Probability (Chaotic Nature)", fontsize=13, fontweight='bold', loc='left', pad=15)
    ax2.set_ylabel("Probability (Likelihood)", fontsize=11)
    ax2.set_xlabel("Timeline (Consecutive Weeks in Rank Eras)", fontsize=11)
    
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax2.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    
    ticks = np.arange(0, len(data_rnk), 20)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"T+{t}" for t in ticks])

    plt.savefig('Chart_B_Timeline_Final.png', dpi=300, bbox_inches='tight')
    print("生成 Chart B Final: 已修复文字遮挡")

# 执行绘图
plot_radar_comparison_final(df)
plot_timeline_final(df)

