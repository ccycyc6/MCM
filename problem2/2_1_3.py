import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_policy_conflict_map(file_path):
    # 1. 加载并清洗数据
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['judge_share', 'est_fan_share'])
    
    conflict_points = []
    
    # 2. 遍历每周，寻找冲突
    # 我们只关心“结果不一样”的那些周
    for (season, week), group in df.groupby(['season', 'week']):
        n = len(group)
        if n < 2: continue
        
        g = group.copy()
        
        # --- 计算排名 (1 is Best) ---
        g['rank_j'] = g['judge_share'].rank(ascending=False, method='min')
        g['rank_f'] = g['est_fan_share'].rank(ascending=False, method='min')
        
        # --- 模拟 Percentage System (淘汰总分最低者) ---
        # Score = J% + V% (越高越好)
        g['score_pct'] = g['judge_share'] + g['est_fan_share']
        elim_pct_name = g.loc[g['score_pct'].idxmin(), 'name']
        
        # --- 模拟 Rank System (淘汰总分最高者) ---
        # Score = Rank J + Rank V (越低越好)
        g['score_rank_sys'] = g['rank_j'] + g['rank_f']
        # 淘汰 Rank 总和最大的人。如果有平局，通常取裁判分更低的人，这里简化为 idxmax
        elim_rank_name = g.loc[g['score_rank_sys'].idxmax(), 'name']
        
        # --- 记录冲突 ---
        if elim_pct_name != elim_rank_name:
            # 找到具体的两个人
            # 1. 在 Percentage 下死掉的人 (Red Point: Saved by Rank)
            p_red = g[g['name'] == elim_pct_name].iloc[0]
            # 2. 在 Rank 下死掉的人 (Blue Point: Saved by Percentage)
            p_blue = g[g['name'] == elim_rank_name].iloc[0]
            
            conflict_points.append({'type': 'red', 'data': p_red})
            conflict_points.append({'type': 'blue', 'data': p_blue})

    # 3. 绘图
    plt.figure(figsize=(10, 8), dpi=120)
    
    # 提取坐标
    red_x = [p['data']['rank_j'] for p in conflict_points if p['type'] == 'red']
    red_y = [p['data']['rank_f'] for p in conflict_points if p['type'] == 'red']
    
    blue_x = [p['data']['rank_j'] for p in conflict_points if p['type'] == 'blue']
    blue_y = [p['data']['rank_f'] for p in conflict_points if p['type'] == 'blue']
    
    # 添加随机抖动 (Jitter) 以防点重叠
    jitter = 0.2
    red_x = np.array(red_x) + np.random.uniform(-jitter, jitter, len(red_x))
    red_y = np.array(red_y) + np.random.uniform(-jitter, jitter, len(red_y))
    blue_x = np.array(blue_x) + np.random.uniform(-jitter, jitter, len(blue_x))
    blue_y = np.array(blue_y) + np.random.uniform(-jitter, jitter, len(blue_y))
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_policy_conflict_map(file_path):
    # 1. 加载并清洗数据
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['judge_share', 'est_fan_share'])
    
    conflict_points = []
    
    # 2. 遍历每周，寻找冲突
    # 我们只关心“结果不一样”的那些周
    for (season, week), group in df.groupby(['season', 'week']):
        n = len(group)
        if n < 2: continue
        
        g = group.copy()
        
        # --- 计算排名 (1 is Best) ---
        g['rank_j'] = g['judge_share'].rank(ascending=False, method='min')
        g['rank_f'] = g['est_fan_share'].rank(ascending=False, method='min')
        
        # --- 模拟 Percentage System (淘汰总分最低者) ---
        # Score = J% + V% (越高越好)
        g['score_pct'] = g['judge_share'] + g['est_fan_share']
        elim_pct_name = g.loc[g['score_pct'].idxmin(), 'name']
        
        # --- 模拟 Rank System (淘汰总分最高者) ---
        # Score = Rank J + Rank V (越低越好)
        g['score_rank_sys'] = g['rank_j'] + g['rank_f']
        # 淘汰 Rank 总和最大的人。如果有平局，通常取裁判分更低的人，这里简化为 idxmax
        elim_rank_name = g.loc[g['score_rank_sys'].idxmax(), 'name']
        
        # --- 记录冲突 ---
        if elim_pct_name != elim_rank_name:
            # 找到具体的两个人
            # 1. 在 Percentage 下死掉的人 (Red Point: Saved by Rank)
            p_red = g[g['name'] == elim_pct_name].iloc[0]
            # 2. 在 Rank 下死掉的人 (Blue Point: Saved by Percentage)
            p_blue = g[g['name'] == elim_rank_name].iloc[0]
            
            conflict_points.append({'type': 'red', 'data': p_red})
            conflict_points.append({'type': 'blue', 'data': p_blue})

    # 3. 绘图
    plt.figure(figsize=(10, 8), dpi=120)
    
    # 提取坐标
    red_x = [p['data']['rank_j'] for p in conflict_points if p['type'] == 'red']
    red_y = [p['data']['rank_f'] for p in conflict_points if p['type'] == 'red']
    
    blue_x = [p['data']['rank_j'] for p in conflict_points if p['type'] == 'blue']
    blue_y = [p['data']['rank_f'] for p in conflict_points if p['type'] == 'blue']
    
    # 添加随机抖动 (Jitter) 以防点重叠
    jitter = 0.2
    red_x = np.array(red_x) + np.random.uniform(-jitter, jitter, len(red_x))
    red_y = np.array(red_y) + np.random.uniform(-jitter, jitter, len(red_y))
    blue_x = np.array(blue_x) + np.random.uniform(-jitter, jitter, len(blue_x))
    blue_y = np.array(blue_y) + np.random.uniform(-jitter, jitter, len(blue_y))
    
# ... (前面的代码保持不变)

    # 绘制散点
    plt.scatter(red_x, red_y, c='#d62728', s=100, alpha=0.7, edgecolors='white', 
                label='Saved by Rank, Eliminated by Pct\n(Rank System Proteges)')
    plt.scatter(blue_x, blue_y, c='#1f77b4', s=100, alpha=0.7, edgecolors='white', 
                label='Saved by Pct, Eliminated by Rank\n(Percentage System Proteges)')
    
    # 设置坐标轴 (翻转，使得 1 在右上方)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    
    plt.title('The Policy Conflict Map: Rank vs. Percentage', fontsize=16, fontweight='bold')
    plt.xlabel('Judge Rank (1 is Best)', fontsize=12)
    plt.ylabel('Fan Vote Rank (1 is Best)', fontsize=12)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 添加注释区域
    plt.text(0.3, 0.8, 'Top Contenders\n(Both Safe)', ha='right', va='top', fontsize=14, color='green', alpha=0.5)
    
    plt.tight_layout()
    
    # 【关键修改】保存图片而不是仅仅显示
    plt.savefig('policy_conflict_map.png', dpi=300)
    print("图表已保存为 'policy_conflict_map.png'")
    # plt.show() # 如果在 Jupyter Notebook 中可以保留这行

# 使用说明：
# 确保你的 CSV 文件名正确，例如 'model1_result(1).csv'
plot_policy_conflict_map('model1_result.csv')