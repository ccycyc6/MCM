import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit # 引入 Sigmoid 函数
import warnings

# 忽略数学计算中的警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 核心求解器：平滑非线性优化 (Sigmoid-based Smooth Solver)
# ==============================================================================
def solve_sigmoid_engine(
    J,              # 裁判份额向量
    is_eliminated,  # 淘汰状态掩码
    v_last_week,    # 上周结果 (动量)
    sympathy_boost=1.3, 
    sigmoid_k=25.0, # 越高则越接近硬阈值
    alpha=0.05,     # 熵权重
    beta=2.5,       # 社会学拟合权重
    gamma=1.0,      # 动量权重
    delta=0.001     # 淘汰安全边际
):
    n = len(J)
    avg_share = 1.0 / n
    
    # --- Step 1: 计算 Pity Score (可怜系数) ---
    # 逻辑：分数相对于平均分越低，Pity Score 越高 (0->1)
    threshold = avg_share
    # 归一化得分差异
    score_diff = (threshold - J) / (np.std(J) + 1e-9) 
    pity_scores = expit(sigmoid_k * score_diff)
    
    # --- Step 2: 构建动态目标 (Target Blending) ---
    # 混合“跟随裁判(Herding)”与“同情救助(Sympathy)”
    target_herding = J
    target_sympathy = np.ones(n) * avg_share * sympathy_boost
    
    v_target = (1 - pity_scores) * target_herding + pity_scores * target_sympathy
    v_target = v_target / np.sum(v_target) # 归一化
    
    # --- Step 3: 构建 U 型置信度权重 ---
    # 极端分数（极高或极低）预测信心强，中间分数允许模型根据约束摆动
    confidence = 2 * np.abs(pity_scores - 0.5) 
    weights_vector = 1.0 + 5.0 * confidence
    
    # --- Step 4: 准备动量 ---
    if v_last_week is None:
        v_momentum = v_target
        current_gamma = 0
    else:
        v_momentum = v_last_week
        current_gamma = gamma

    # --- Step 5: 定义平滑目标函数 ---
    x0 = v_target 
    def objective(x):
        loss_entropy = np.sum(x * np.log(x + 1e-9))
        loss_sociology = np.sum(weights_vector * (x - v_target)**2)
        loss_momentum = np.sum((x - v_momentum)**2)
        return alpha * loss_entropy + beta * loss_sociology + current_gamma * loss_momentum

    # --- Step 6: 淘汰约束 ---
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    idx_elim = np.where(is_eliminated == 1)[0]
    idx_surv = np.where(is_eliminated == 0)[0]
    
    if len(idx_elim) > 0 and len(idx_surv) > 0:
        for e in idx_elim:
            for s in idx_surv:
                def constraint_func(x, e_idx=e, s_idx=s):
                    return (J[s_idx] + x[s_idx]) - (J[e_idx] + x[e_idx]) - delta
                cons.append({'type': 'ineq', 'fun': constraint_func})
                
    # --- Step 7: 执行优化 ---
    bounds = [(0.0, 1.0) for _ in range(n)]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 100, 'ftol': 1e-6})
    
    if res.success:
        return res.x / np.sum(res.x), pity_scores, True
    else:
        return v_target, pity_scores, False

# ==============================================================================
# 2. 自动化流水线 (The Pipeline)
# ==============================================================================
def run_advanced_pipeline(input_csv):
    df = pd.read_csv(input_csv)
    df_model = df[df['era'] == 'Percentage'].copy()
    
    output_records = []
    for season in sorted(df_model['season'].unique()):
        print(f"Processing Season {season}...")
        season_data = df_model[df_model['season'] == season]
        last_week_memory = {}
        
        for week in sorted(season_data['week'].unique()):
            curr = season_data[season_data['week'] == week]
            names = curr['celebrity_name'].values
            J = curr['judge_share'].values
            is_elim = curr['is_eliminated'].values
            
            # 动量对齐
            v_prior = np.array([last_week_memory.get(n, 1.0/len(names)) for n in names])
            v_prior /= np.sum(v_prior)
            
            # 调用高级引擎
            solved_v, pities, success = solve_sigmoid_engine(J, is_elim, v_prior)
            
            # 更新记忆
            last_week_memory = dict(zip(names, solved_v))
            
            # 存入结果
            for i, name in enumerate(names):
                output_records.append({
                    'season': season, 'week': week, 'name': name,
                    'judge_share': J[i], 'pity_score': pities[i],
                    'est_fan_share': solved_v[i], 'success': success,
                    'total_implied': J[i] + solved_v[i]
                })
                
    return pd.DataFrame(output_records)

if __name__ == "__main__":
    results = run_advanced_pipeline('processed_dwts_model_input.csv')
    results.to_csv('final_fan_vote_estimation.csv', index=False)
    print("Done! Check 'final_fan_vote_estimation.csv'.")