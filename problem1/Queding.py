import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据加载与合并
# ==========================================
# 加载两个模型结果
df1 = pd.read_csv('model1_result.csv') # Percentage Era (S3-27)
df2 = pd.read_csv('model2_result.csv') # Rank Eras (S1-2, S28+)

# 统一列名以进行合并
# Model 1 (Point Estimates)
df1_clean = df1[['season', 'week', 'name', 'est_fan_share', 'is_eliminated']].copy()
df1_clean.columns = ['season', 'week', 'celebrity_name', 'fan_share', 'is_eliminated']
df1_clean['uncertainty'] = 0.0  # Percentage时代模型给出点估计，视作无区间宽度
df1_clean['System'] = 'Percentage System (S3-S27)'
df1_clean['Model'] = 'Model 1 (Optimization)'

# Model 2 (Interval Estimates)
df2_clean = df2[['season', 'week', 'celebrity_name', 'estimated_fan_share', 'uncertainty_95ci', 'is_eliminated']].copy()
df2_clean.columns = ['season', 'week', 'celebrity_name', 'fan_share', 'uncertainty', 'is_eliminated']
# 区分 S1-2 和 S28+
df2_clean['System'] = df2_clean['season'].apply(lambda x: 'Rank System (S1-S2)' if x <= 2 else 'Rank + Save (S28+)')
df2_clean['Model'] = 'Model 2 (Simulation)'

# 合并所有数据
df_all = pd.concat([df1_clean, df2_clean], ignore_index=True)

# 过滤掉 Week 1 (通常无淘汰或数据噪音较大)
df_all = df_all[df_all['week'] > 1]

# ==========================================
# 2. 绘图：全周期确定性概览
# ==========================================
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# 我们按赛季分组计算平均不确定性，展示宏观趋势
season_stats = df_all.groupby(['season', 'System'])['uncertainty'].mean().reset_index()

# 定义颜色映射
palette = {
    'Rank System (S1-S2)': '#e74c3c',       # 红
    'Percentage System (S3-S27)': '#3498db',# 蓝
    'Rank + Save (S28+)': '#e67e22'         # 橙
}

# 绘制柱状图/散点图组合
# 1. 绘制背景区域：标示出不同的时代
plt.axvspan(0.5, 2.5, color='#e74c3c', alpha=0.1, label='Rank Era')
plt.axvspan(2.5, 27.5, color='#3498db', alpha=0.1, label='Percentage Era')
plt.axvspan(27.5, 34.5, color='#e67e22', alpha=0.1, label='Rank+Save Era')

# 2. 绘制每个赛季的平均不确定性 (Mean Uncertainty Width)
sns.scatterplot(
    data=season_stats, 
    x='season', 
    y='uncertainty', 
    hue='System', 
    palette=palette, 
    s=100, 
    edgecolor='black',
    zorder=10
)

# 3. 添加连接线以显示趋势
plt.plot(season_stats['season'], season_stats['uncertainty'], color='gray', alpha=0.5, linestyle='--')

# 4. 图表装饰
plt.title('Certainty of Fan Vote Estimates Across 34 Seasons', fontsize=16, fontweight='bold')
plt.xlabel('Season', fontsize=12)
plt.ylabel('Average Uncertainty Interval Width (95% CI)', fontsize=12)
plt.xlim(0, 35)
plt.ylim(-0.02, 0.4) # 留一点空间

# 5. 关键注释
plt.annotate('High Uncertainty\n(Rank System loses information)', 
             xy=(2, 0.3), xytext=(5, 0.35),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('High Precision\n(Percentage System preserves magnitude)', 
             xy=(15, 0.01), xytext=(10, 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('Return to Uncertainty\n(Reintroduction of Ranks)', 
             xy=(28, 0.25), xytext=(20, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.legend(loc='upper right', title='Scoring System', fontsize=9, title_fontsize=10, markerscale=0.8)
plt.tight_layout()

plt.savefig('combined_certainty_analysis.png', dpi=300)
print("综合分析图表已保存为: combined_certainty_analysis.png")
plt.show()