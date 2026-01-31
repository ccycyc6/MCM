import pandas as pd
import numpy as np

def analyze_policy_divergence(file_path):
    # 1. 加载数据 (使用 Model 1 的结果，因为它包含具体的份额估计)
    df = pd.read_csv(file_path)
    
    # 确保没有空值
    df = df.dropna(subset=['judge_share', 'est_fan_share'])
    
    results = []
    
    # 按赛季和周进行遍历
    for (season, week), group in df.groupby(['season', 'week']):
        n = len(group)
        if n < 2: continue  # 跳过只有1个人的情况
        
        g = group.copy()
        
        # ----------------------------------------
        # 策略 1: 百分比制 (Percentage Approach)
        # ----------------------------------------
        # 假设：分数越高越好。淘汰总分最低者。
        g['score_pct'] = g['judge_share'] + g['est_fan_share']
        elim_pct_name = g.loc[g['score_pct'].idxmin(), 'name']
        
        # ----------------------------------------
        # 策略 2: 排名制 (Rank Approach)
        # ----------------------------------------
        # 转换份额为排名 (1 = 份额最高/最好)
        # method='min' 意味着如果有两个人并列第一，他们都是 Rank 1，下一名是 Rank 3
        g['rank_j'] = g['judge_share'].rank(ascending=False, method='min')
        g['rank_f'] = g['est_fan_share'].rank(ascending=False, method='min')
        
        # 总排名分 = 裁判排名 + 观众排名
        # 注意：在排名制中，数值越小越好 (1+1=2)。
        # 淘汰：数值最大者 (最差)。
        g['score_rank_sys'] = g['rank_j'] + g['rank_f']
        
        # 找到得分最高(最差)的人。如果有平局，通常由裁判分更低的人淘汰，这里简化为取其中之一
        elim_rank_name = g.loc[g['score_rank_sys'].idxmax(), 'name']
        
        # ----------------------------------------
        # 记录结果
        # ----------------------------------------
        results.append({
            'season': season,
            'week': week,
            'contestants_count': n,
            'eliminated_under_pct': elim_pct_name,
            'eliminated_under_rank': elim_rank_name,
            'is_outcome_different': (elim_pct_name != elim_rank_name)
        })
        
    res_df = pd.DataFrame(results)
    
    # 输出统计信息
    divergence_rate = res_df['is_outcome_different'].mean()
    print(f"分析完成: 共模拟 {len(res_df)} 场淘汰赛。")
    print(f"决策差异率 (Divergence Rate): {divergence_rate:.2%}")
    
    return res_df

# 运行代码
divergence_df = analyze_policy_divergence('model1_result.csv')
divergence_df.to_csv('q2_a_1.csv', index=False)