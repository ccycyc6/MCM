import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_consistency(model1_path, model2_path):
    print("正在生成一致性评估报告...\n")
    report = {}
    
    # =======================================================
    # 1. 评估 Model 1 (Percentage / Optimization)
    # =======================================================
    try:
        df1 = pd.read_csv(model1_path)
        print(f"--- Model 1 (Percentage Era) 评估 ---")
        
        # 指标 A: 求解成功率 (能否找到满足淘汰约束的解?)
        # 检查 'optimization_success' 列
        if 'optimization_success' in df1.columns:
            success_rate = df1['optimization_success'].mean()
            print(f"1. 数学自洽性 (求解成功率): {success_rate:.2%}")
            report['M1_Success_Rate'] = success_rate
        
        # 指标 B: 约束满足度验证 (Constraint Satisfaction)
        # 验证: 被淘汰者的 implied_total_score 是否确实小于幸存者的最低分?
        violation_count = 0
        total_weeks = 0
        
        for (season, week), group in df1.groupby(['season', 'week']):
            eliminated = group[group['is_eliminated'] == 1]
            survivors = group[group['is_eliminated'] == 0]
            
            if not eliminated.empty and not survivors.empty:
                total_weeks += 1
                max_elim_score = eliminated['implied_total_score'].max()
                min_surv_score = survivors['implied_total_score'].min()
                
                # 如果淘汰者的分比幸存者高 (或相等，考虑浮点误差)，就是违背了一致性
                if max_elim_score > min_surv_score - 1e-5:
                    violation_count += 1
        
        consistency_score = 1 - (violation_count / total_weeks) if total_weeks > 0 else 1.0
        print(f"2. 淘汰逻辑一致性 (Constraint Consistency): {consistency_score:.2%}")
        report['M1_Constraint_Consistency'] = consistency_score

        # 指标 C: 社会学拟合度 (Loss Value)
        # Loss 越低，说明结果越符合"羊群+同情"的假设
        avg_loss = df1['loss_value'].mean()
        print(f"3. 社会学解释力 (Avg Loss): {avg_loss:.4f} (越低越好)")
        
    except Exception as e:
        print(f"Model 1 评估失败: {e}")

    print("\n" + "="*30 + "\n")

    # =======================================================
    # 2. 评估 Model 2 (Rank / Monte Carlo)
    # =======================================================
    try:
        df2 = pd.read_csv(model2_path)
        print(f"--- Model 2 (Rank Era) 评估 ---")
        
        # 指标 A: 有效率 (Valid Rate / Likelihood)
        # 代表历史结果发生的概率。如果 valid_rate 太低，说明模型觉得这个结果很离谱
        avg_valid_rate = df2[df2['is_eliminated']==1]['valid_rate'].mean()
        print(f"1. 历史复现概率 (Avg Valid Rate): {avg_valid_rate:.2%}")
        report['M2_Avg_Likelihood'] = avg_valid_rate
        
        # 极端案例分析: 有多少周是"黑天鹅" (Rate < 0.1%)
        black_swans = df2[(df2['is_eliminated']==1) & (df2['valid_rate'] < 0.001)]
        num_black_swans = black_swans[['season', 'week']].drop_duplicates().shape[0]
        print(f"   > 极难解释的黑天鹅周数: {num_black_swans}")

        # 指标 B: 确定性 (Certainty / CI Width)
        # 95% CI 越窄，说明模型越确定
        avg_ci = df2['uncertainty_95ci'].mean()
        print(f"2. 估计确定性 (Avg 95% CI Width): {avg_ci:.4f} (越低越好)")
        report['M2_Avg_CI_Width'] = avg_ci
        
    except Exception as e:
        print(f"Model 2 评估失败: {e}")

    return report

# 假设文件名如下 (请根据你实际保存的文件名修改)
# 如果你还没运行生成代码，请先运行之前的建模代码
m1_file = 'model1_result.csv' # 或者 'dwts_model1_nonlinear_results.csv'
m2_file = 'model2_result.csv'  # 或者 'dwts_model2_rank_sigmoid_results.csv'

# 执行评估
# 注意：这需要你本地有这两个文件。如果没有，请先运行上面的生成代码。
import os
if os.path.exists(m1_file) and os.path.exists(m2_file):
    metrics = evaluate_model_consistency(m1_file, m2_file)
else:
    print("找不到结果文件，无法进行评估。请先运行模型生成 CSV。")