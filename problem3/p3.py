import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 模块 1: 特征工厂 (Feature Factory) - 数据清洗与构建
# ==============================================================================
class DWTSAnalytics:
    def __init__(self, raw_data_path):
        try:
            self.df = pd.read_csv(raw_data_path)
        except:
            self.df = pd.read_excel(raw_data_path)
        
        # 标准化列名
        self.df.columns = [c.strip().lower().replace(' ', '_').replace('/', '_') for c in self.df.columns]

    def _map_industry(self, ind):
        """将复杂的行业归类为几大组，便于统计"""
        ind = str(ind).lower()
        if pd.isna(ind): return 'Other'
        if any(x in ind for x in ['actor', 'actress', 'movie', 'tv', 'star']): return 'Actor'
        if any(x in ind for x in ['singer', 'musician', 'pop', 'song']): return 'Musician'
        if any(x in ind for x in ['athlete', 'nfl', 'nba', 'olympian']): return 'Athlete'
        if any(x in ind for x in ['reality', 'bachelor', 'housewife']): return 'RealityStar'
        return 'Other'

    def process_data(self, fan_data_df=None):
        """
        fan_data_df: 包含 ['season', 'celebrity_name', 'estimated_fan_share'] 的 DataFrame
        """
        df = self.df.copy()
        
        # 1. 行业特征
        df['industry_group'] = df['celebrity_industry'].apply(self._map_industry)
        
        # 2. 性别特征 (模拟，实际请补充)
        np.random.seed(42)
        if 'gender' not in df.columns:
            df['is_female'] = np.random.randint(0, 2, len(df))
            
        # 3. 构建职业舞伴的历史能力 (Partner Ability) - 防泄露
        df = df.sort_values(['season', 'week1_judge1_score'])
        partner_history = {} 
        pro_avg_ranks = []
        pro_is_new = [] # 新手标记
        
        for idx, row in df.iterrows():
            p = row['ballroom_partner']
            # 获取历史数据
            if p in partner_history and partner_history[p]:
                avg_rank = np.mean(partner_history[p])
                is_new = 0
            else:
                avg_rank = 7.0 # 默认填充
                is_new = 1
            
            pro_avg_ranks.append(avg_rank)
            pro_is_new.append(is_new)
            
            # 更新历史
            try:
                if not pd.isna(row['placement']):
                    if p not in partner_history: partner_history[p] = []
                    partner_history[p].append(float(row['placement']))
            except: pass
            
        df['pro_avg_rank'] = pro_avg_ranks
        df['pro_is_new'] = pro_is_new

        # 4. 目标变量 Y1: 评委分 (Z-Score)
        judge_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
        df['raw_judge_avg'] = df[judge_cols].mean(axis=1)
        # 按赛季标准化
        df['y_judge_std'] = df.groupby('season')['raw_judge_avg'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )
        
        # 5. 目标变量 Y2: 粉丝份额 (Logit变换)
        if fan_data_df is not None:
            # 真实合并逻辑:
            # df = pd.merge(df, fan_data_df, on=['season', 'celebrity_name'], how='left')
            # 暂时用模拟数据代替展示：
            pass
            
        if 'estimated_fan_share' not in df.columns:
            # 模拟数据 (请替换为真实数据)
            df['estimated_fan_share'] = 0.1 + 0.05 * df['y_judge_std'] + np.random.normal(0, 0.02, len(df))
            
        # Logit 变换
        eps = 1e-3
        clipped_share = df['estimated_fan_share'].clip(eps, 1-eps)
        df['y_fan_logit'] = np.log(clipped_share / (1 - clipped_share))
        
        return df

# ==============================================================================
# 模块 2: 诊断与建模核心 (Analytics Core)
# ==============================================================================
def run_analysis_report(df):
    # 准备建模数据
    features = ['celebrity_age_during_season', 'industry_group', 'is_female', 
                'pro_avg_rank', 'pro_is_new']
    targets = ['y_judge_std', 'y_fan_logit']
    
    data = df[features + targets + ['ballroom_partner']].dropna()
    
    # 标准化数值变量 (为了比较系数大小)
    scaler = StandardScaler()
    data['age_scaled'] = scaler.fit_transform(data[['celebrity_age_during_season']])
    data['pro_rank_scaled'] = scaler.fit_transform(data[['pro_avg_rank']])
    
    # ---------------------------------------------------------
    # 1. 多重共线性诊断 (VIF)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 1: 多重共线性诊断 (Multicollinearity Check)")
    print("="*50)
    
    # 构造设计矩阵
    formula_x = "age_scaled + C(industry_group) + is_female + pro_rank_scaled + C(pro_is_new)"
    y_dummy, X = dmatrices(f"y_judge_std ~ {formula_x}", data, return_type='dataframe')
    
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_df.round(2))
    
    if vif_df['VIF'].max() < 5:
        print("\n[结论] VIF均小于5，变量间独立性良好，模型系数可信。")
    else:
        print("\n[警告] 存在VIF>5的变量，需在论文中讨论共线性影响。")

    # ---------------------------------------------------------
    # 2. 建立 LME 双模型
    # ---------------------------------------------------------
    formula_lme = f" ~ {formula_x}"
    
    print("\n" + "="*50)
    print("STEP 2: 评委模型结果 (Judge Model)")
    print("="*50)
    model_judge = smf.mixedlm("y_judge_std" + formula_lme, data, groups=data["ballroom_partner"]).fit()
    print(model_judge.summary().tables[1])
    
    print("\n" + "="*50)
    print("STEP 3: 粉丝模型结果 (Fan Model)")
    print("="*50)
    model_fan = smf.mixedlm("y_fan_logit" + formula_lme, data, groups=data["ballroom_partner"]).fit()
    print(model_fan.summary().tables[1])

    return model_judge, model_fan

# ==============================================================================
# 执行入口
# ==============================================================================
if __name__ == "__main__":
    # 请修改为你的文件名
    FILE_PATH = '2026_MCM_Problem_C_Data.csv'
    
    analytics = DWTSAnalytics(FILE_PATH)
    # 如果有粉丝数据，传入: analytics.process_data(fan_df)
    df_clean = analytics.process_data() 
    
m1, m2 = run_analysis_report(df_clean)