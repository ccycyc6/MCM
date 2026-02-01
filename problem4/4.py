import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以保证结果可复现
np.random.seed(42)

# ==========================================
# 1. 数据生成：制造尖锐的“优绩主义危机”
# ==========================================
weeks = 10
# 调整参数：
# 1. Juan (技术大神) 粉丝极少 -> 传统赛制下容易被淘汰 (产生高MRI)
# 2. Bobby (流量明星) 技术更差 -> 新赛制下容易掉入Bottom 3
contestants_data = {
    'Name': ['Juan (Merit)', 'Bobby (Pop)', 'Milo (Talent)', 'Evanna (Pro)', 'Joe (Avg)', 'Mary (Old)'],
    'Age': [39, 38, 17, 27, 32, 60],
    'Base_Skill': [9.8, 4.0, 9.0, 8.5, 6.5, 6.0], # Bobby技术分降至4.0
    'Fan_Base_Size': [200, 15000, 1500, 2000, 1200, 800], # Juan粉丝降至200，Bobby升至15000
}
df_c = pd.DataFrame(contestants_data)

results = []
for w in range(1, weeks + 1):
    for i, row in df_c.iterrows():
        # 1. 裁判分 (S_judge): 0-30分
        skill_noise = np.random.normal(0, 1.0)
        # 老将后期疲劳明显
        fatigue = 0 if row['Age'] < 40 else (w * 0.2) 
        score = min(30, max(10, (row['Base_Skill'] * 3) + skill_noise - fatigue))
        
        # 2. 线性粉丝票数
        # 流量派死忠粉非常疯狂 (x10倍投票)
        fan_noise = np.random.normal(1, 0.05)
        if 'Pop' in row['Name']:
            raw_votes = row['Fan_Base_Size'] * 10 * fan_noise
        else:
            raw_votes = row['Fan_Base_Size'] * 1 * fan_noise
            
        results.append({
            'Week': w,
            'Name': row['Name'],
            'Age': row['Age'],
            'S_judge_raw': score,
            'Votes_Linear': raw_votes
        })

df = pd.DataFrame(results)

# ==========================================
# 2. 机制模拟
# ==========================================
metrics_log = {
    'Traditional': {'mri': [], 'eliminated': []},
    'New_Protocol': {'mri': [], 'eliminated': []}
}
df['In_Bottom3'] = 0 # 初始化热力图数据 (0=Safe, 1=Danger)

for w in range(1, weeks + 1):
    week_data = df[df['Week'] == w].copy()
    
    # --- A. 传统机制 (排名制) ---
    rank_judge = week_data['S_judge_raw'].rank(ascending=False)
    rank_fan = week_data['Votes_Linear'].rank(ascending=False)
    # 简单的排名相加 (越小越好)
    trad_score = rank_judge + rank_fan 
    
    # 淘汰排名最靠后的人
    eliminated_idx = trad_score.idxmax()
    eliminated_person = week_data.loc[eliminated_idx]
    
    # MRI 计算: 淘汰者的分 - 幸存者的最低分
    survivors = week_data.drop(eliminated_idx)
    survivor_min_score = survivors['S_judge_raw'].min()
    mri_val = max(0, eliminated_person['S_judge_raw'] - survivor_min_score)
    metrics_log['Traditional']['mri'].append(mri_val)
    
    # --- B. 新机制 A-GLHP-QV ---
    
    # Layer 1: 荣誉豁免 (Golden Immunity)
    # 年龄补偿系数
    lambda_age = 0.5 # 每大1岁补0.5分 (为了演示效果调大)
    min_age = week_data['Age'].min()
    week_data['S_adj'] = week_data['S_judge_raw'] + lambda_age * np.maximum(0, week_data['Age'] - min_age)
    
    immune_name = week_data.loc[week_data['S_adj'].idxmax(), 'Name']
    
    # Layer 2: 混合QV战场
    battle_df = week_data[week_data['Name'] != immune_name].copy()
    
    # QV: 票数开根号
    qv_votes = np.sqrt(battle_df['Votes_Linear'])
    
    # 归一化计算混合得分
    score_share = battle_df['S_judge_raw'] / battle_df['S_judge_raw'].sum()
    vote_share = qv_votes / qv_votes.sum()
    
    # 混合权重 50/50
    battle_df['Mixed_Score'] = 0.5 * score_share + 0.5 * vote_share
    
    # 找出 Bottom 3
    battle_df = battle_df.sort_values('Mixed_Score', ascending=True)
    bottom3_names = battle_df.iloc[:3]['Name'].values
    
    # 记录热力图数据 (Bobby是否在Bottom 3)
    df.loc[(df['Week'] == w) & (df['Name'].isin(bottom3_names)), 'In_Bottom3'] = 1
    
    # Layer 3: 生死战 (Dance-Off)
    # 假设裁判严格按技术分淘汰 Bottom 3 中最差的
    bottom3_df = battle_df[battle_df['Name'].isin(bottom3_names)]
    eliminated_new_name = bottom3_df.loc[bottom3_df['S_judge_raw'].idxmin(), 'Name']
    eliminated_new_score = bottom3_df['S_judge_raw'].min()
    
    # MRI 计算 (新赛制)
    # 幸存者 = 豁免者 + (Bottom3中未淘汰者) + (非Bottom3者)
    survivor_min_new = min(
        week_data[week_data['Name'] == immune_name]['S_judge_raw'].min(),
        week_data[week_data['Name'] != eliminated_new_name]['S_judge_raw'].min()
    )
    mri_new_val = max(0, eliminated_new_score - survivor_min_new)
    metrics_log['New_Protocol']['mri'].append(mri_new_val)

# ==========================================
# 3. 可视化绘图
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

# 图1: 精英遗憾指数 (MRI) 对比
ax1 = axes[0]
weeks_axis = range(1, weeks + 1)
ax1.plot(weeks_axis, metrics_log['Traditional']['mri'], 'o--', color='#d62728', label='Traditional System (High Regret)', linewidth=2)
ax1.plot(weeks_axis, metrics_log['New_Protocol']['mri'], 's-', color='#2ca02c', label='A-GLHP-QV System (Fairness)', linewidth=2)
ax1.fill_between(weeks_axis, metrics_log['Traditional']['mri'], color='#d62728', alpha=0.1)

ax1.set_title('Meritocratic Regret Index (MRI) Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Regret Score\n(Score of Wrongly Eliminated Elite)', fontsize=12)
ax1.set_xlabel('Competition Week', fontsize=12)
ax1.set_xticks(weeks_axis)
ax1.legend(fontsize=11)
ax1.text(5, 5, 'Traditional: High Merit Candidates Eliminated', color='#d62728', fontsize=10)
ax1.text(5, 0.5, 'A-GLHP: Merit Protected', color='#2ca02c', fontsize=10, verticalalignment='bottom')

# 图2: 危险区捕获热力图 (Bobby Bones)
ax2 = axes[1]
bobby_data = df[df['Name'] == 'Bobby (Pop)'].pivot_table(index='Name', columns='Week', values='In_Bottom3')
sns.heatmap(bobby_data, cmap=['#e0e0e0', '#ff4444'], linewidths=1, linecolor='white', 
            cbar=False, ax=ax2, square=False)

# 自定义图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e0e0e0', edgecolor='white', label='Safe Zone'),
                   Patch(facecolor='#ff4444', edgecolor='white', label='Risk Zone (Bottom 3)')]
ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title='Status')

ax2.set_title('Risk Zone Capture: Is the "Pop Star" caught in Bottom 3?', fontsize=14, fontweight='bold')
ax2.set_ylabel('')
ax2.set_xlabel('Competition Week', fontsize=12)

plt.show()