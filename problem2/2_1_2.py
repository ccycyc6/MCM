import pandas as pd
import numpy as np

def analyze_mechanism_bias(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['judge_share', 'est_fan_share'])
    
    bias_metrics = []
    
    for (season, week), group in df.groupby(['season', 'week']):
        n = len(group)
        if n <= 3: continue # 样本太少不计算相关性
        
        g = group.copy()
        
        # 1. 准备基础排名 (1 = Best)
        g['rank_j'] = g['judge_share'].rank(ascending=False, method='min') # 裁判输入
        g['rank_f'] = g['est_fan_share'].rank(ascending=False, method='min') # 观众输入
        
        # 2. 计算两种机制下的最终排名 (1 = Best)
        # Percentage System
        score_pct = g['judge_share'] + g['est_fan_share']
        g['final_rank_pct'] = score_pct.rank(ascending=False, method='min')
        
        # Rank System
        score_rank_sys = g['rank_j'] + g['rank_f']
        g['final_rank_rank_sys'] = score_rank_sys.rank(ascending=True, method='min') # 数值越小排名越靠前
        
        # 3. 计算相关性 (Spearman)
        # 越高代表最终结果越符合该方的输入
        # Percentage System 的相关性
        corr_pct_vs_fan = g['final_rank_pct'].corr(g['rank_f'], method='spearman')
        corr_pct_vs_judge = g['final_rank_pct'].corr(g['rank_j'], method='spearman')
        
        # Rank System 的相关性
        corr_rank_vs_fan = g['final_rank_rank_sys'].corr(g['rank_f'], method='spearman')
        corr_rank_vs_judge = g['final_rank_rank_sys'].corr(g['rank_j'], method='spearman')
        
        bias_metrics.append({
            'season': season,
            'week': week,
            'pct_fan_bias': corr_pct_vs_fan,
            'pct_judge_bias': corr_pct_vs_judge,
            'rank_fan_bias': corr_rank_vs_fan,
            'rank_judge_bias': corr_rank_vs_judge
        })
        
    bias_df = pd.DataFrame(bias_metrics)
    
    print("\n机制偏向性分析 (平均相关系数):")
    print("------------------------------------------------")
    print(f"【百分比制】 对 观众 的相关性: {bias_df['pct_fan_bias'].mean():.4f}")
    print(f"【百分比制】 对 裁判 的相关性: {bias_df['pct_judge_bias'].mean():.4f} (Winner: Judges)")
    print("------------------------------------------------")
    print(f"【排名制】   对 观众 的相关性: {bias_df['rank_fan_bias'].mean():.4f} (Winner: Fans)")
    print(f"【排名制】   对 裁判 的相关性: {bias_df['rank_judge_bias'].mean():.4f}")
    
    return bias_df

# 运行代码
bias_df = analyze_mechanism_bias('model1_result.csv')