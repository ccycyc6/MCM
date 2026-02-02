import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 生成模拟数据 (Mock Data)
# ==========================================
np.random.seed(42)
n_seasons = 34
contestants_per_season = 12
data_rows = []

for s in range(1, n_seasons + 1):
    for i in range(contestants_per_season):
        # 随机分配行业
        industry = np.random.choice(['Reality', 'Athlete', 'Other'], p=[0.3, 0.3, 0.4])
        
        # 基础粉丝票数
        vote_share = 0.10 + np.random.normal(0, 0.02)
        
        # 添加 "真实" 效应
        if industry == 'Reality':
            vote_share += 0.03
        elif industry == 'Athlete':
            vote_share += 0.01
            
        # *** 注入异常值 (Season 27 - The Bobby Bones Effect) ***
        if s == 27 and industry == 'Reality':
            vote_share += 0.08  # 巨大的额外加成
            
        data_rows.append({'Season': s, 'Industry': industry, 'FanVote': vote_share})

df = pd.DataFrame(data_rows)

# ==========================================
# 2. 执行 LOSO (Leave-One-Season-Out) 循环
# ==========================================
loso_results = []
print("Running LOSO validation on LME model...")

# 【关键修正 1】使用正确的参数名称匹配 statsmodels 的输出
target_coef = "C(Industry, Treatment(reference='Other'))[T.Reality]"

for skip_season in range(1, n_seasons + 1):
    # 剔除第 skip_season 季的数据
    train_data = df[df['Season'] != skip_season].copy()
    
    try:
        # 重新训练模型
        model = smf.mixedlm("FanVote ~ C(Industry, Treatment(reference='Other'))", 
                            train_data, 
                            groups=train_data["Season"])
        result = model.fit(reml=False)
        
        # 【关键修正 2】添加检查，确保 Key 存在
        if target_coef in result.params:
            coef_value = result.params[target_coef]
            loso_results.append({'Excluded_Season': skip_season, 'Coefficient': coef_value})
        else:
            print(f"Warning: Key '{target_coef}' not found. Available keys: {result.params.keys()}")
            
    except Exception as e:
        # 【关键修正 3】打印具体错误，方便调试
        print(f"Error at Season {skip_season}: {e}")

res_df = pd.DataFrame(loso_results)

# ==========================================
# 3. 绘制改进版散点图 (Scatter Plot)
# ==========================================
if res_df.empty:
    print("Error: No results were generated. Please check the model configuration.")
else:
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # 定义颜色：普通的季用蓝色，S27用红色
    colors = ['red' if x == 27 else 'dodgerblue' for x in res_df['Excluded_Season']]
    sizes = [150 if x == 27 else 50 for x in res_df['Excluded_Season']]

    # 绘制散点图
    plt.scatter(res_df['Excluded_Season'], res_df['Coefficient'], 
                c=colors, s=sizes, alpha=0.8, edgecolors='white', linewidth=1.5)

    # 标记 S27 (如果 S27 在结果中)
    if 27 in res_df['Excluded_Season'].values:
        s27_val = res_df.loc[res_df['Excluded_Season'] == 27, 'Coefficient'].values[0]
        plt.annotate(f'Excluding S27\n(Coef drops to {s27_val:.3f})', 
                     xy=(27, s27_val), xytext=(27, s27_val - 0.005),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     ha='center', fontsize=11, fontweight='bold')
    else:
        # 如果代码逻辑没有跑 S27，给个默认值防止报错（仅作保护）
        s27_val = res_df['Coefficient'].min() 

    # 计算平均值
    contaminated_mean = res_df[res_df['Excluded_Season'] != 27]['Coefficient'].mean()
    
    # 添加辅助线
    plt.axhline(contaminated_mean, color='gray', linestyle=':', alpha=0.6, label='Avg Effect (With Anomaly included)')
    if 27 in res_df['Excluded_Season'].values:
        plt.axhline(s27_val, color='red', linestyle='--', alpha=0.5, label='True Effect (Clean Data)')

    plt.title('Sensitivity Analysis: Impact of Each Season on "Reality Star" Bias', fontsize=14)
    plt.ylabel('Coefficient Value (Estimated Bias)', fontsize=12)
    plt.xlabel('Excluded Season (Which season was removed?)', fontsize=12)
    plt.legend(
        loc='upper right',
        fontsize=7.8,          # 或 8 / 10，按需要微调
        frameon=True
    )
    
    # 设置X轴刻度
    plt.xticks(np.arange(1, n_seasons + 1, 2)) 

    plt.tight_layout()
    plt.show()

    # 【关键修正 4】将打印语句放入 else 块内，确保变量已定义
    print(f"Coefficient when S27 is excluded: {s27_val:.4f}")
    print(f"Average Coefficient otherwise: {contaminated_mean:.4f}")