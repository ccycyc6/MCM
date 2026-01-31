import pandas as pd
import numpy as np

def analyze_controversy_and_save(file_path):
    # 1. 加载数据
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['judge_share', 'est_fan_share'])
    
    # 争议选手名单 (根据题目要求)
    targets = ['Jerry Rice', 'Billy Ray Cyrus', 'Bristol Palin', 'Bobby Bones']
    
    results = []
    
    # 2. 遍历每一周，进行全场模拟
    for (season, week), group in df.groupby(['season', 'week']):
        # 只要该周有争议选手在场，就进行分析
        target_in_week = group[group['name'].isin(targets)]
        if target_in_week.empty:
            continue
            
        n = len(group)
        if n < 2: continue
        
        g = group.copy()
        
        # --- A. 模拟 Rank System ---
        g['rank_j'] = g['judge_share'].rank(ascending=False, method='min')
        g['rank_f'] = g['est_fan_share'].rank(ascending=False, method='min')
        g['score_rank'] = g['rank_j'] + g['rank_f']
        # 谁被淘汰？(Rank Sum 最大者)
        elim_rank_name = g.loc[g['score_rank'].idxmax(), 'name']
        
        # --- B. 模拟 Percentage System ---
        g['score_pct'] = g['judge_share'] + g['est_fan_share']
        # 谁被淘汰？(Total Pct 最小者)
        # 找出 Bottom 2 用于 "Judges' Save"
        # 排序：分数从低到高
        sorted_g = g.sort_values('score_pct', ascending=True)
        bottom_1_name = sorted_g.iloc[0]['name'] # 倒数第一
        bottom_2_name = sorted_g.iloc[1]['name'] if n > 1 else None # 倒数第二
        bottom_2_set = {bottom_1_name, bottom_2_name} - {None}
        
        # --- C. 模拟 Judges' Save (法官拯救机制) ---
        # 逻辑：如果是 Bottom 2，且裁判分比对手低，则被淘汰。
        # 假设 Judges' Save 是基于 Percentage System 的 Bottom 2 运行的 (现代赛制)
        
        # 对每一个争议选手进行判定
        for _, row in target_in_week.iterrows():
            name = row['name']
            
            # 1. 现实情况 (Reality)
            # 注意：数据中的 is_eliminated 可能只是最终结果，我们需要看他是否活过了这一周
            # 如果这周他没被淘汰，is_eliminated == 0
            survived_reality = (row['is_eliminated'] == 0)
            
            # 2. Rank 机制结果
            # 如果他是 Rank 分最高的，他就会死
            killed_by_rank = (name == elim_rank_name)
            
            # 3. Pct 机制结果
            # 如果他是 Pct 分最低的，他就会死
            killed_by_pct = (name == bottom_1_name)
            
            # 4. Judges' Save 机制结果
            killed_by_save = False
            save_comment = "Safe (Not in Btm 2)"
            
            if name in bottom_2_set:
                # 找到 Bottom 2 里的对手
                opponent_name = [x for x in bottom_2_set if x != name][0]
                opponent_row = g[g['name'] == opponent_name].iloc[0]
                
                # 比较裁判分
                if row['judge_share'] < opponent_row['judge_share']:
                    killed_by_save = True
                    save_comment = f"Eliminated vs {opponent_name} (Judge Score Lower)"
                else:
                    save_comment = f"Saved vs {opponent_name} (Judge Score Higher)"
            
            results.append({
                'Season': season,
                'Week': week,
                'Name': name,
                'Judge Rank': row['judge_share'], # 显示原始份额用于参考
                'Fan Rank Est': row['est_fan_share'],
                'Reality': 'Survived' if survived_reality else 'Eliminated',
                'Method_Rank': 'Eliminated' if killed_by_rank else 'Survived',
                'Method_Pct': 'Eliminated' if killed_by_pct else 'Survived',
                'Method_Save': 'Eliminated' if killed_by_save else 'Survived',
                'Save_Details': save_comment
            })

    # 转化为 DataFrame
    res_df = pd.DataFrame(results)
    
    # 筛选出有趣的行：即现实中活下来了，但被新机制杀死的行
    interesting_df = res_df[
        (res_df['Reality'] == 'Survived') & 
        ((res_df['Method_Rank'] == 'Eliminated') | 
         (res_df['Method_Pct'] == 'Eliminated') | 
         (res_df['Method_Save'] == 'Eliminated'))
    ]
    
    return res_df, interesting_df

# 运行并显示
all_res, key_findings = analyze_controversy_and_save('model1_result.csv')
print(key_findings[['Season', 'Week', 'Name', 'Method_Rank', 'Method_Pct', 'Method_Save']])