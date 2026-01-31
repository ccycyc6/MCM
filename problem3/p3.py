import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 模块 1: 特征工厂 (Feature Factory)
# ==============================================================================
class DWTSFeatureFactory:
    def __init__(self, filepath):
        try:
            self.raw_df = pd.read_csv(filepath)
        except:
            self.raw_df = pd.read_excel(filepath)
        
        # 标准化列名
        self.raw_df.columns = [c.strip().lower().replace(' ', '_').replace('/', '_') for c in self.raw_df.columns]

    def _map_industry(self, ind):
        """行业归类逻辑"""
        ind = str(ind).lower()
        if pd.isna(ind): return 'Other'
        if any(x in ind for x in ['actor', 'actress', 'movie', 'tv', 'star', 'disney']): return 'Actor'
        if any(x in ind for x in ['singer', 'musician', 'rapper', 'pop', 'song']): return 'Musician'
        if any(x in ind for x in ['athlete', 'nfl', 'nba', 'olympian', 'player']): return 'Athlete'
        if any(x in ind for x in ['reality', 'bachelor', 'housewife', 'survivor']): return 'RealityStar'
        if any(x in ind for x in ['host', 'personality', 'anchor']): return 'Host'
        return 'Other'

    def process(self, fan_estimates_df=None):
        df = self.raw_df.copy()

        # 1. 基础特征清洗
        df['industry_group'] = df['celebrity_industry'].apply(self._map_industry)
        
        # 模拟/补充性别 (实际需外部数据)
        np.random.seed(42)
        if 'gender' not in df.columns:
            df['is_female'] = np.random.randint(0, 2, len(df))
        
        # 2. 关键：提取职业舞伴历史能力 (防泄露)
        # 必须确保按时间顺序遍历
        df = df.sort_values(['season', 'week1_judge1_score'])
        
        partner_history = {} # {partner_name: [list of past ranks]}
        pro_avg_ranks = []
        pro_is_new = []      # 新增变量：是否新人
        
        # 全局平均排名 (用于填充)
        global_avg_rank = 7.0 

        for idx, row in df.iterrows():
            p = row['ballroom_partner']
            
            # --- 步骤 A: 读取历史 (作为当前的 Input) ---
            if p in partner_history and len(partner_history[p]) > 0:
                # 是老手
                avg_val = np.mean(partner_history[p])
                is_new_val = 0
            else:
                # 是新人
                avg_val = global_avg_rank
                is_new_val = 1
            
            pro_avg_ranks.append(avg_val)
            pro_is_new.append(is_new_val)
            
            # --- 步骤 B: 更新历史 (作为未来的 Input) ---
            # 只有当本行数据包含最终排名时，才将其加入历史库
            try:
                # 假设 placement 列存在且为数值
                rank = float(row['placement'])
                if not pd.isna(rank):
                    if p not in partner_history: partner_history[p] = []
                    partner_history[p].append(rank)
            except:
                pass
        
        df['pro_avg_rank'] = pro_avg_ranks
        df['pro_is_new'] = pro_is_new

        # 3. Target 1: 评委分 (Z-Score by Season)
        judge_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
        df['raw_judge_avg'] = df[judge_cols].mean(axis=1)
        # 按赛季分组标准化，消除早期赛季打分偏高/偏低的影响
        df['y_judge_std'] = df.groupby('season')['raw_judge_avg'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

        # 4. Target 2: 粉丝份额 (Logit Transform)
        if fan_estimates_df is not None:
            # TODO: Merge fan_estimates_df here based on Name/Season
            pass
        else:
            # [模拟] 制造一个假数据用于演示
            # 假设粉丝份额在 0.05 到 0.25 之间
            df['estimated_fan_share'] = 0.15 + 0.05 * df['y_judge_std'] + np.random.normal(0, 0.02, len(df))
        
        # Logit 变换前的安全截断 (避免 log(0) 或 log(1))
        # 假设 fan_share 是 0-1 之间的值
        epsilon = 1e-4
        df['fan_share_clipped'] = df['estimated_fan_share'].clip(epsilon, 1-epsilon)
        df['y_fan_logit'] = np.log(df['fan_share_clipped'] / (1 - df['fan_share_clipped']))

        return df

# ==============================================================================
# 模块 2: 诊断模块 (Diagnostics)
# ==============================================================================
def run_diagnostics(df):
    print("\n[Module 2] Running Diagnostics (VIF Check)...")
    
    # 选择进入模型的变量
    features = ['y_judge_std', 'celebrity_age_during_season', 'industry_group', 
                'is_female', 'pro_avg_rank', 'pro_is_new']
    model_df = df[features].dropna()
    
    # 对数值变量进行简单标准化，方便VIF计算
    scaler = StandardScaler()
    model_df[['age_scaled', 'pro_rank_scaled']] = scaler.fit_transform(
        model_df[['celebrity_age_during_season', 'pro_avg_rank']]
    )
    
    # 使用 patsy 生成设计矩阵 (自动处理 One-Hot)
    formula = "y_judge_std ~ age_scaled + C(industry_group) + is_female + pro_rank_scaled + C(pro_is_new)"
    y, X = dmatrices(formula, model_df, return_type='dataframe')
    
    # 计算 VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # 格式化输出
    print(vif_data.round(2))
    
    # 自动判定
    max_vif = vif_data['VIF'].max()
    if max_vif < 5:
        print(f"-> 诊断通过: 最大 VIF ({max_vif}) < 5，多重共线性风险低。")
    else:
        print(f"-> 警告: 存在高 VIF ({max_vif})，请检查相关变量。")
    
    return model_df

# ==============================================================================
# 模块 3: 核心建模 (LME Core)
# ==============================================================================
def run_lme_core(df):
    print("\n[Module 3] Running LME Core Models...")
    
    # 准备数据 (确保无空值)
    cols = ['y_judge_std', 'y_fan_logit', 'celebrity_age_during_season', 
            'industry_group', 'is_female', 'pro_avg_rank', 'pro_is_new', 'ballroom_partner']
    data = df[cols].dropna()
    
    # 标准化 Predictors (让系数具有可比性)
    scaler = StandardScaler()
    data[['age_scaled', 'pro_rank_scaled']] = scaler.fit_transform(
        data[['celebrity_age_during_season', 'pro_avg_rank']]
    )

    # 公式定义: 固定效应 + 随机效应
    # 注意: pro_is_new 是 0/1 变量，视为 Categorical
    formula_common = " ~ age_scaled + C(industry_group) + is_female + pro_rank_scaled + C(pro_is_new)"
    
    # --- 模型 A: 评委 (Standardized Score) ---
    print("--- Fitting Judge Model ---")
    model_judge = smf.mixedlm(
        "y_judge_std" + formula_common, 
        data, 
        groups=data["ballroom_partner"]
    ).fit()
    print(model_judge.summary().tables[1])
    
    # --- 模型 B: 粉丝 (Logit Transformed) ---
    print("\n--- Fitting Fan Model (Logit) ---")
    model_fan = smf.mixedlm(
        "y_fan_logit" + formula_common, 
        data, 
        groups=data["ballroom_partner"]
    ).fit()
    print(model_fan.summary().tables[1])
    
    return model_judge, model_fan

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    # 1. 工厂生产数据
    # 请替换为你的文件名
    input_file = '2026_MCM_Problem_C_Data.csv.xlsx - 2026_MCM_Problem_C_Data.csv' 
    
    factory = DWTSFeatureFactory(input_file)
    df_clean = factory.process() # 如果有粉丝数据df，传入 process(fan_estimates_df)
    
    # 2. 诊断
    df_for_model = run_diagnostics(df_clean)
    
    # 3. 建模
    m_judge, m_fan = run_lme_core(df_clean)
    
    print("\n[Success] 模型运行完毕。请查阅上方系数表进行对比分析。")