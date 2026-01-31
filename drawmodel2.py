import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取数据
df = pd.read_csv('model2_result.csv')

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial'] # 防止字体报错
plt.rcParams['axes.unicode_minus'] = False

# ==========================================================
# 图1: 存活边界图 (Survival Landscape)
# ==========================================================
plt.figure(figsize=(10, 6))
# 给排名加一点随机抖动(Jitter)，防止点重叠，视觉效果更好
df['judge_rank_jitter'] = df['judge_rank'] + np.random.uniform(-0.2, 0.2, size=len(df))

scatter = sns.scatterplot(
    data=df, 
    x='judge_rank_jitter', 
    y='estimated_fan_share', 
    hue='is_eliminated', 
    style='era', # 不同时代用不同形状
    palette={0: '#1f77b4', 1: '#d62728'}, # 蓝=安全, 红=淘汰
    alpha=0.7,
    s=80
)

plt.title('Figure 1: Survival Landscape - Judge Rank vs. Estimated Fan Share', fontsize=14, fontweight='bold')
plt.xlabel('Judge Rank (Lower is Better)', fontsize=12)
plt.ylabel('Estimated Fan Share (Model Output)', fontsize=12)
plt.legend(title='Status (0=Safe, 1=Eliminated)', loc='upper right')
# 画一条示意的分割线（可选，视数据分布而定）
# plt.plot([0, 10], [0.05, 0.25], 'k--', alpha=0.5, label='Survival Threshold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Fig1_Survival_Landscape.png', dpi=300)
plt.show()

# ==========================================================
# 图2: Season 2 案例分析 (Jerry Rice Effect)
# ==========================================================
# 筛选 Season 2 中出场周数最多的前5名选手
season2 = df[df['season'] == 2]
top_names = season2['celebrity_name'].value_counts().head(5).index
s2_plot = season2[season2['celebrity_name'].isin(top_names)]

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=s2_plot,
    x='week',
    y='estimated_fan_share',
    hue='celebrity_name',
    style='celebrity_name',
    markers=True,
    dashes=False,
    linewidth=2.5,
    markersize=9
)

plt.title('Figure 2: Fan Popularity Dynamics in Season 2 (Rank Era)', fontsize=14, fontweight='bold')
plt.xlabel('Week', fontsize=12)
plt.ylabel('Estimated Fan Share', fontsize=12)
plt.legend(title='Celebrity', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Fig2_Season2_Dynamics.png', dpi=300)
plt.show()

# ==========================================================
# 图3: 赛制对比 (Era Comparison)
# ==========================================================
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x='era',
    y='estimated_fan_share',
    hue='is_eliminated',
    palette={0: 'lightblue', 1: 'salmon'},
    width=0.6
)

plt.title('Figure 3: Fan Share Distribution by Era and Outcome', fontsize=14, fontweight='bold')
plt.xlabel('Era (Scoring System)', fontsize=12)
plt.ylabel('Estimated Fan Share', fontsize=12)
plt.legend(title='Status (0=Safe, 1=Eliminated)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Fig3_Era_Comparison.png', dpi=300)
plt.show()