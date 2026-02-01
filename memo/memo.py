import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ==========================================
# 0. 风格设置 (商业汇报风格)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

# 颜色定义：使用直观的红绿配色
COLOR_RISK = '#E74C3C'    # 红色 (坏情况/危险)
COLOR_SAFE = '#27AE60'    # 绿色 (好情况/安全)

# ==========================================
# 1. 图表 1: The "Talent Loss" Tracker (人才流失追踪)
# ==========================================
def plot_figure_1_talent_loss():
    weeks = 10
    x = np.arange(1, weeks + 1)
    
    # 模拟数据
    # Current System: 天才流失严重，数值不断攀升
    y_current = np.cumsum([0.2, 0.8, 1.5, 2.0, 2.5, 3.0, 3.2, 3.5, 3.5, 3.5])
    # New Plan: 几乎留住了所有天才，数值保持低位
    y_new = np.cumsum([0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 绘制曲线
    ax.plot(x, y_current, color=COLOR_RISK, linewidth=3, linestyle='--', marker='o', 
            label='Current Rules (Losing Best Dancers)')
    ax.plot(x, y_new, color=COLOR_SAFE, linewidth=4, linestyle='-', marker='D', 
            label='New Plan (Keeping Best Dancers)')

    # 填充差距区域
    ax.fill_between(x, y_current, y_new, color=COLOR_RISK, alpha=0.1)
    
    # 添加通俗易懂的标注 (Call-out)
    ax.annotate('We lost a potential Winner here!', xy=(3, y_current[2]), xytext=(3.5, 1.3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5), fontsize=10)

    # 图表设置
    ax.set_title("Are We Eliminating the Wrong People?", loc='left', pad=15)
    ax.set_xlabel("Week of Show")
    ax.set_ylabel("Accumulated Loss of Talent")
    ax.set_yticks([]) # 隐藏具体数学数值，强调趋势
    ax.legend(frameon=True, loc='upper left')
    
    plt.tight_layout()
    # plt.savefig('Figure1_TalentLoss.png') # 如果需要保存
    plt.show()

# ==========================================
# 2. 图表 2: The "Danger Zone" Simulation (危险区模拟)
# ==========================================
def plot_figure_2_danger_zone():
    weeks = 10
    
    # 模拟一个"高人气但低分"选手(如Bobby Bones)的状态
    # 0 = Safe, 1 = Danger Zone (Bottom 3)
    
    # 旧规则：全是绿色，一直安全
    status_old = np.array([[0]*weeks]) 
    
    # 新规则：经常掉入红色危险区
    status_new = np.array([[0, 0, 0, 1, 1, 0, 1, 1, 1, 1]])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    
    cmap = sns.color_palette([COLOR_SAFE, COLOR_RISK]) 

    # Plot 1: 旧规则
    sns.heatmap(status_old, ax=ax1, cmap=cmap, cbar=False, linewidths=2, linecolor='white')
    ax1.set_title("Old Rules: The Viral Star is Untouchable (Boring)", loc='left', fontsize=11)
    ax1.set_yticks([])
    
    # Plot 2: 新规则
    sns.heatmap(status_new, ax=ax2, cmap=cmap, cbar=False, linewidths=2, linecolor='white')
    ax2.set_title("New Plan: The Viral Star faces the 'Danger Zone' (Exciting!)", loc='left', fontsize=11)
    ax2.set_yticks([])
    ax2.set_xlabel("Week of Show")
    ax2.set_xticks(np.arange(weeks) + 0.5)
    ax2.set_xticklabels(np.arange(1, weeks + 1))

    # 自定义图例
    legend_elements = [
        Patch(facecolor=COLOR_SAFE, label='Safe'),
        Patch(facecolor=COLOR_RISK, label='In Danger (Judges Decide)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.957), ncol=2, frameon=False)

    plt.suptitle("The 'Viral Star' Test", y=0.99, fontweight='bold', fontsize=14)
    plt.tight_layout()
    # plt.savefig('Figure2_DangerZone.png') # 如果需要保存
    plt.show()

# 运行绘图函数
plot_figure_1_talent_loss()
plot_figure_2_danger_zone()