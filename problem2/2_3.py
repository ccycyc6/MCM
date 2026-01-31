import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def generate_decision_radar_chart_optimized(file_path):
    # 1. 读取数据
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['judge_share', 'est_fan_share'])
    
    # 定义四种待对比的方案
    methods = ['Rank', 'Percentage', 'Rank + Save', 'Pct + Save']
    
    # 初始化统计字典
    metrics = {m: {'regret_sum': 0.0, 'fan_corr_sum': 0.0, 'stability_count': 0, 'weeks_count': 0} for m in methods}
    
    # 2. 遍历每一周数据进行全量模拟
    for (season, week), group in df.groupby(['season', 'week']):
        n = len(group)
        if n < 3: continue 
        
        g = group.copy()
        
        # --- 预计算基准数据 ---
        min_judge_share = g['judge_share'].min()
        ideal_loser_names = g[g['judge_share'] == min_judge_share]['name'].tolist()
        
        g['rank_j'] = g['judge_share'].rank(ascending=False, method='min')
        g['rank_f'] = g['est_fan_share'].rank(ascending=False, method='min')
        
        # --- 定义四种方法的模拟逻辑 ---
        def get_method_outcome(method_name):
            if 'Rank' in method_name:
                scores = g['rank_j'] + g['rank_f']
                final_ranks = scores.rank(ascending=True, method='min')
                sorted_g = g.assign(temp_score=scores).sort_values('temp_score', ascending=False)
                bottom_2 = sorted_g.iloc[:2]
                
                if 'Save' in method_name:
                    b1 = bottom_2.iloc[0]
                    b2 = bottom_2.iloc[1]
                    if b1['judge_share'] < b2['judge_share']:
                        elim_name = b1['name']
                    elif b2['judge_share'] < b1['judge_share']:
                        elim_name = b2['name']
                    else:
                        elim_name = b1['name'] 
                else:
                    elim_name = sorted_g.iloc[0]['name']
                    
            elif 'Percentage' in method_name or 'Pct' in method_name:
                scores = g['judge_share'] + g['est_fan_share']
                final_ranks = scores.rank(ascending=False, method='min')
                sorted_g = g.assign(temp_score=scores).sort_values('temp_score', ascending=True)
                bottom_2 = sorted_g.iloc[:2]
                
                if 'Save' in method_name:
                    b1 = bottom_2.iloc[0]
                    b2 = bottom_2.iloc[1]
                    if b1['judge_share'] < b2['judge_share']:
                        elim_name = b1['name']
                    elif b2['judge_share'] < b1['judge_share']:
                        elim_name = b2['name']
                    else:
                        elim_name = b1['name']
                else:
                    elim_name = sorted_g.iloc[0]['name']
            
            return elim_name, final_ranks

        # --- 3. 计算三个维度的原始指标 ---
        for m in methods:
            elim_name, final_ranks = get_method_outcome(m)
            
            elim_row = g[g['name'] == elim_name].iloc[0]
            regret = max(0, elim_row['judge_share'] - min_judge_share)
            metrics[m]['regret_sum'] += regret
            
            corr = final_ranks.corr(g['rank_f'], method='spearman')
            if pd.isna(corr): corr = 0
            metrics[m]['fan_corr_sum'] += corr
            
            if elim_name in ideal_loser_names:
                metrics[m]['stability_count'] += 1
                
            metrics[m]['weeks_count'] += 1

    # 4. 数据汇总与归一化
    final_stats = []
    for m in methods:
        count = metrics[m]['weeks_count']
        if count == 0: continue
        
        avg_regret = metrics[m]['regret_sum'] / count
        avg_corr = metrics[m]['fan_corr_sum'] / count
        stability_rate = metrics[m]['stability_count'] / count
        
        final_stats.append({
            'Method': m,
            'Avg_Regret': avg_regret,
            'Avg_Fan_Corr': avg_corr,
            'Stability_Rate': stability_rate
        })
    
    stat_df = pd.DataFrame(final_stats)
    
    # 归一化处理
    max_regret = stat_df['Avg_Regret'].max()
    stat_df['Score_Fairness'] = 1 - (stat_df['Avg_Regret'] / (max_regret * 1.05))
    stat_df['Score_FanImpact'] = stat_df['Avg_Fan_Corr'].clip(lower=0)
    stat_df['Score_Stability'] = stat_df['Stability_Rate']
    
    # 5. 绘图 (带非线性视觉优化)
    categories = ['Fairness\n(Meritocracy)', 'Fan Impact\n(Democracy)', 'Stability\n(Predictability)']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] 
    
    plt.figure(figsize=(10, 10), dpi=150)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    
    plt.xticks(angles[:-1], categories, color='black', size=19, fontweight='bold')
    
    # --- 核心修改：非线性视觉缩放 (Non-linear Visual Scaling) ---
    # 使用平方根变换 (Sqrt Transform) 来放大低数值区域
    # 变换公式: r_plot = sqrt(r_data)
    # 这使得面积与数值成正比 (Area ~ Value)，而不仅仅是半径
    
    def transform_scale(x):
        return np.power(x, 0.5) # 平方根变换
    
    # 自定义网格线位置
    # 原始值: 0.2, 0.4, 0.6, 0.8, 1.0
    # 绘图位置: sqrt(0.2)=0.45, sqrt(0.4)=0.63, ...
    tick_values = [0.2, 0.4, 0.6, 0.8]
    tick_locs = [transform_scale(x) for x in tick_values]
    tick_labels = ["0.2", "0.4", "0.6", "0.8"]
    
    # 应用自定义网格
    ax.set_rlabel_position(0)
    plt.yticks(tick_locs, tick_labels, color="grey", size=15)
    plt.ylim(0, 1.0) # 视觉上限设为1.0 (因为 sqrt(1)=1)
    
    colors = {
        'Rank': '#ff7f0e',        # 橙色
        'Percentage': '#1f77b4',  # 蓝色
        'Rank + Save': '#d62728', # 红色
        'Pct + Save': '#2ca02c'   # 绿色
    }
    
    for i, row in stat_df.iterrows():
        # 获取原始数据 (0-1)
        raw_values = [row['Score_Fairness'], row['Score_FanImpact'], row['Score_Stability']]
        
        # 应用变换到绘图数据
        plot_values = [transform_scale(v) for v in raw_values]
        plot_values += plot_values[:1] 
        
        method_name = row['Method']
        c = colors.get(method_name, 'black')
        
        ax.plot(angles, plot_values, linewidth=2, linestyle='solid', label=method_name, color=c)
        ax.fill(angles, plot_values, color=c, alpha=0.1) 
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=19, title="Voting Mechanisms")
    plt.title('Decision Radar: Structural Trade-offs (Visual Optimized)', size=16, y=1.1, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('decision_radar_chart_optimized.png')
    print("美化后的雷达图已保存为 'decision_radar_chart_optimized.png'")
    
    return stat_df

# 调用函数 (确保文件名正确)
generate_decision_radar_chart_optimized('model1_result.csv')