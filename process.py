import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings

# 忽略一些运行时的数学警告（如 log(0)）
warnings.filterwarnings('ignore')

# ==============================================================================
# 模块 1: 核心优化求解器 (The Optimization Engine)
# ==============================================================================
def solve_fan_distribution(J, is_eliminated, v_prior, alpha=0.1, beta=2.0, delta=0.001):
    """
    针对 Percentage Era 的单周求解函数
    
    参数:
        J (array): 裁判份额向量 (Judge Share)
        is_eliminated (array): 0/1 标记
        v_prior (array): 动量先验向量
        alpha: 最大熵权重 (惩罚极端值)
        beta: 动量权重 (惩罚剧烈波动)
        delta: 安全边际 (Margin)
    """
    n = len(J)
    x0 = v_prior # 初始猜测点
    
    # 1. 定义目标函数 (Loss Function)
    def objective(x):
        # 加上 1e-9 防止 log(0)
        # 最大熵项 (最小化负熵): sum(x * log(x))
        entropy_loss = np.sum(x * np.log(x + 1e-9))
        # 动量项: sum((x - prior)^2)
        momentum_loss = np.sum((x - v_prior)**2)
        
        return alpha * entropy_loss + beta * momentum_loss

    # 2. 定义约束条件
    cons = []
    
    # (A) 等式约束: 粉丝票总和为 1
    cons.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # (B) 不等式约束: 淘汰逻辑
    # 优化策略：我们不需要 N*M 个约束，只需要 "淘汰者最高分" < "幸存者最低分" - delta
    # 但由于可能有多个淘汰者，我们遍历每一个淘汰者，要求他必须低于所有幸存者的最低分
    
    idx_elim = np.where(is_eliminated == 1)[0]
    idx_surv = np.where(is_eliminated == 0)[0]
    
    if len(idx_elim) > 0 and len(idx_surv) > 0:
        # 为了更稳健，我们建立 "Pairwise Constraints"
        for e in idx_elim:
            for s in idx_surv:
                def constraint_func(x, e_idx=e, s_idx=s):
                    score_s = J[s_idx] + x[s_idx]
                    score_e = J[e_idx] + x[e_idx]
                    # 幸存者 - 淘汰者 - 边际 >= 0
                    return score_s - score_e - delta
                cons.append({'type': 'ineq', 'fun': constraint_func})
    
    # 3. 边界 (0% - 100%)
    bounds = [(0.0, 1.0) for _ in range(n)]
    
    # 4. 求解
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                      options={'maxiter': 100, 'ftol': 1e-6})
    
    # 5. 结果处理
    if result.success:
        return result.x / np.sum(result.x), True, result.fun
    else:
        # 求解失败 (可能是争议极大)，返回先验值作为保底
        return v_prior / np.sum(v_prior), False, 0.0

# ==============================================================================
# 模块 2: 主流程控制 (Data Flow & Loop)
# ==============================================================================
def run_model_pipeline(input_file):
    print(f"正在读取数据: {input_file} ...")
    try:
        df = pd.read_csv(input_file)
    except:
        print("错误：找不到输入文件。请确保第一步生成的 'processed_dwts_model_input.csv' 存在。")
        return None

    # 1. 筛选 Percentage 时代 (S3 - S27)
    df_model = df[df['era'] == 'Percentage'].copy()
    
    output_records = [] # 存储结果
    
    # 2. 按赛季遍历
    seasons = sorted(df_model['season'].unique())
    
    for season in seasons:
        print(f"正在处理 Season {season}...")
        season_data = df_model[df_model['season'] == season]
        weeks = sorted(season_data['week'].unique())
        
        # 初始化上一周的记忆 (Name -> Share)
        last_week_memory = {} 
        
        for week in weeks:
            current_df = season_data[season_data['week'] == week].copy()
            
            # --- 准备输入向量 ---
            contestants = current_df['celebrity_name'].values
            J = current_df['judge_share'].values
            is_elim = current_df['is_eliminated'].values
            
            # --- 构建 v_prior (动量先验) ---
            prior_list = []
            
            if len(last_week_memory) == 0:
                # Case A: 第1周 (无记忆) -> 假设 Prior = Judge Share
                v_prior = J
                prior_source = "Judge_Share"
            else:
                # Case B: 有记忆 -> 人员对齐
                prior_source = "History"
                for i, name in enumerate(contestants):
                    if name in last_week_memory:
                        prior_list.append(last_week_memory[name])
                    else:
                        # 极端情况：新加入的选手 (Wildcard)，给平均分
                        prior_list.append(1.0 / len(contestants))
                
                v_prior = np.array(prior_list)
                
                # 【关键】重新归一化 (Renormalize)
                # 因为上周有人淘汰，剩下的人加起来 < 1，必须放大
                if np.sum(v_prior) > 0:
                    v_prior = v_prior / np.sum(v_prior)
                else:
                    v_prior = J # 防止异常
            
            # --- 运行优化模型 ---
            solved_v, success, final_loss = solve_fan_distribution(
                J, is_elim, v_prior, 
                alpha=0.05,  # 熵权重 (调大则票数更平均)
                beta=5.0,    # 动量权重 (调大则更像上周)
                delta=0.001  # 0.1% 的分差
            )
            
            # --- 更新记忆 (供下周使用) ---
            last_week_memory = dict(zip(contestants, solved_v))
            
            # --- 存储数据 ---
            for i, name in enumerate(contestants):
                record = {
                    'season': season,
                    'week': week,
                    'celebrity_name': name,
                    'judge_share': J[i],
                    'is_eliminated': is_elim[i],
                    'prior_fan_share': v_prior[i],
                    'estimated_fan_share': solved_v[i], # 这就是我们要的结果
                    'optimization_success': success,
                    'loss_value': final_loss,
                    'implied_total_score': J[i] + solved_v[i] # 用于验证淘汰
                }
                output_records.append(record)
    
    # 3. 生成最终表格
    result_df = pd.DataFrame(output_records)
    print("\n建模完成！")
    return result_df

# ==============================================================================
# 执行代码
# ==============================================================================
if __name__ == "__main__":
    # 请确保文件名与你第一步保存的一致
    input_csv = 'processed_dwts_model_input.csv'
    
    final_df = run_model_pipeline(input_csv)
    
    if final_df is not None:
        # 保存结果
        final_df.to_csv('dwts_model1_results.csv', index=False)
        print("结果已保存至 'dwts_model1_results.csv'")
        
        # --- 简单验证 ---
        # 看看 Season 5 Week 9 (Jennie Garth 被淘汰那周) 的结果
        # 题目中说她 Judge 分数不错，但因为粉丝分低走了
        check = final_df[(final_df['season'] == 5) & (final_df['week'] == 9)]
        if not check.empty:
            print("\n验证案例 (Season 5 Week 9):")
            print(check[['celebrity_name', 'judge_share', 'estimated_fan_share', 'is_eliminated', 'implied_total_score']].sort_values('implied_total_score'))