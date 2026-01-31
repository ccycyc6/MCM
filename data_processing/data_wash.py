import pandas as pd
import numpy as np

# 1. 读取你刚刚生成的清洗数据
df = pd.read_csv('processed_dwts_model_input.csv')

# 2. 数据分流：只取 Percentage 时代的 S3-S27
# 这就是 Model 1.1 的数据全集
df_model1 = df[df['era'] == 'Percentage'].copy()

# 3. 初始化结果容器
# 我们需要把算出来的 fan_share 存回去，现在先新建一列空着
df_model1['estimated_fan_share'] = np.nan

# 4. 核心循环结构：按赛季 -> 按周次遍历
# 优化是“每周独立求解”的，但需要“上周的数据”做先验
seasons = sorted(df_model1['season'].unique())

for season in seasons:
    season_data = df_model1[df_model1['season'] == season]
    weeks = sorted(season_data['week'].unique())
    
    # 初始化：上一周的粉丝投票结果 (Prior)
    # 在赛季开始前，它是 None
    previous_week_fan_share = None 
    
    for week in weeks:
        # 获取本周的数据切片
        current_week_df = season_data[season_data['week'] == week]
        
        # ==========================================
        # 这里就是你需要提取的“模型输入”
        # ==========================================
        
        # 1. 参赛选手名字 (用于索引)
        contestants = current_week_df['celebrity_name'].values
        
        # 2. 已知量：裁判份额向量 J (Vector J)
        J = current_week_df['judge_share'].values
        
        # 3. 约束目标：谁被淘汰了? (Mask)
        # 找到 is_eliminated == 1 的索引
        eliminated_indices = np.where(current_week_df['is_eliminated'].values == 1)[0]
        
        # 4. 准备先验向量 (Prior Vector)
        # 如果是第1周，或者没有上周数据 -> 用裁判份额代替 (假设粉丝是理性的)
        if previous_week_fan_share is None:
            v_prior = J 
        else:
            # 关键：由于每周有人淘汰，人员名单变了！
            # 你需要从上周的结果中，提取出“本周还在的人”的份额，并重新归一化
            # 这里需要一个匹配逻辑 (Matching Logic)
            # (将在下一步建模代码中详细实现)
            v_prior = J # 暂时占位，建模时细化
            
        # ==========================================
        # 至此，你拥有了求解第 (Season, Week) 所需的一切：
        # J (裁判分), eliminated_indices (谁必须分低), v_prior (历史惯性)
        # 下一步就是把它们扔进 scipy.optimize
        # ==========================================
        
        print(f"准备就好 Season {season} Week {week} 的数据: {len(contestants)} 人参赛")
        
        # [Placeholder] 这里将插入 Step 3 的优化求解代码
        # solution = solve_optimization(J, eliminated_indices, v_prior)
        
        # [Placeholder] 更新 previous_week_fan_share 供下一次循环使用
        # previous_week_fan_share = solution