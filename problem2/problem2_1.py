import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 通用模拟引擎 (Simulation Engine)
# ==============================================================================
class DualSimulationEngine:
    def __init__(self, df_data, source_name):
        self.df = df_data.copy()
        self.source_name = source_name
        self._preprocess()

    def _preprocess(self):
        # 确保 judge_share 存在
        if 'judge_share' not in self.df.columns:
            if 'judge_rank' in self.df.columns:
                self.df['judge_share'] = 1.0 / (self.df['judge_rank'] + 1)
                sums = self.df.groupby(['season', 'week'])['judge_share'].transform('sum')
                self.df['judge_share'] = self.df['judge_share'] / sums
            else:
                # 最后的补救：假设 judge_score 列存在
                # 但通常 model1_result.csv 肯定有 judge_share
                pass
    
    def run_sensitivity(self, weight_range=np.arange(0.0, 1.01, 0.05)):
        results = []
        for w_fan in weight_range:
            w_judge = 1.0 - w_fan
            
            total = 0
            rank_ot = 0
            pct_ot = 0
            
            for (s, w), group in self.df.groupby(['season', 'week']):
                if len(group) < 2: continue
                
                # 基准：裁判想淘汰谁
                judge_worst_idx = group['judge_share'].idxmin()
                judge_worst_name = group.loc[judge_worst_idx, 'celebrity_name']
                
                # --- Rank Method ---
                r_judge = group['judge_share'].rank(ascending=False, method='min')
                r_fan = group['estimated_fan_share'].rank(ascending=False, method='min')
                # 淘汰 Weighted Rank Sum 最大者
                rank_elim_name = group.loc[(w_judge * r_judge + w_fan * r_fan).idxmax(), 'celebrity_name']
                
                # --- Percent Method ---
                p_judge = group['judge_share']
                p_fan = group['estimated_fan_share']
                # 淘汰 Weighted Score Sum 最小者
                pct_elim_name = group.loc[(w_judge * p_judge + w_fan * p_fan).idxmin(), 'celebrity_name']
                
                total += 1
                if rank_elim_name != judge_worst_name: rank_ot += 1
                if pct_elim_name != judge_worst_name: pct_ot += 1
            
            results.append({
                'Source': self.source_name,
                'Fan_Weight': w_fan,
                'Rank_Overturn_Rate': rank_ot / total if total > 0 else 0,
                'Percent_Overturn_Rate': pct_ot / total if total > 0 else 0
            })
        return pd.DataFrame(results)

# ==============================================================================
# 主执行逻辑
# ==============================================================================
def run_full_cross_validation(file_m1, file_m2):
    all_results = []
    
    # 1. 跑 Model 1 数据 (Percentage Era Source)
    try:
        df1 = pd.read_csv(file_m1)
        print(f">>> Running simulation on Model 1 Data ({len(df1)} rows)...")
        engine1 = DualSimulationEngine(df1, "Data Source: Model 1 (Optimization)")
        res1 = engine1.run_sensitivity()
        all_results.append(res1)
    except Exception as e:
        print(f"Skipping Model 1: {e}")

    # 2. 跑 Model 2 数据 (Rank Era Source)
    try:
        df2 = pd.read_csv(file_m2)
        print(f">>> Running simulation on Model 2 Data ({len(df2)} rows)...")
        engine2 = DualSimulationEngine(df2, "Data Source: Model 2 (Monte Carlo)")
        res2 = engine2.run_sensitivity()
        all_results.append(res2)
    except Exception as e:
        print(f"Skipping Model 2: {e}")
        
    if not all_results:
        print("No data to plot.")
        return

    # 3. 合并绘图
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 绘图：两张子图，分别展示两个数据源下的结果
    g = sns.FacetGrid(final_df, col="Source", height=6, aspect=1.2)
    
    def plot_lines(data, **kwargs):
        sns.lineplot(data=data, x='Fan_Weight', y='Rank_Overturn_Rate', 
                     label='Rank Method', color='#4c72b0', linewidth=2.5)
        sns.lineplot(data=data, x='Fan_Weight', y='Percent_Overturn_Rate', 
                     label='Percentage Method', color='#c44e52', linewidth=2.5)
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)

    g.map_dataframe(plot_lines)
    g.set_axis_labels("Nominal Fan Weight", "Judge Overturn Rate")
    g.set_titles("{col_name}")
    
    plt.savefig('dual_source_sensitivity.png')
    print("图表已生成: dual_source_sensitivity.png")
    
    # 输出关键对比数据 (at w=0.2)
    print("\n[Key Data Points at Weight 0.2]")
    print(final_df[final_df['Fan_Weight'].between(0.19, 0.21)])

if __name__ == "__main__":
    # 请确保这里的文件名是你真实生成的文件名
    f1 = 'model1_result.csv' 
    f2 = 'model2_result.csv' # 或 'final_model2_with_certainty_metrics.csv'
    
    run_full_cross_validation(f1, f2)