

import pandas as pd
import numpy as np
import re
from scipy.special import expit  # Sigmoid 函数
from tqdm import tqdm

# ==============================================================================
# 1. 数据预处理 (Data Preprocessing)
# ==============================================================================
def load_and_process_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except:
        try:
            df = pd.read_excel(filepath)
        except:
            print("Error: 无法读取数据文件。")
            return pd.DataFrame()

    if df.empty: return df

    # 标准化列名
    df.columns = [c.strip().lower().replace(' ', '_').replace('/', '_') for c in df.columns]

    # 1. 解析结果，提取 Last Active Week
    def parse_res(row):
        res = str(row['results']).lower()
        match = re.search(r'week (\d+)', res)
        if match: return int(match.group(1))
        if any(x in res for x in ['winner', 'place', 'runner', 'finalist']): return 99
        return 0
    df['last_active_week'] = df.apply(parse_res, axis=1)

    # 2. 修正决赛周数 (Max Week)
    def get_max_week(season_df):
        cols = [c for c in season_df.columns if 'judge' in c and 'score' in c]
        valid = season_df[cols].dropna(how='all', axis=1).columns
        weeks = [int(re.search(r'week(\d+)', c).group(1)) for c in valid if re.search(r'week(\d+)', c)]
        return max(weeks) if weeks else 10
    season_max = df.groupby('season').apply(get_max_week)
    
    def fix_week(row):
        return season_max.loc[row['season']] if row['last_active_week'] == 99 else row['last_active_week']
    df['last_active_week'] = df.apply(fix_week, axis=1)

    # 3. 关键：区分 S28+ 的赛制标签
    def get_era(s):
        if s <= 2: return 'Rank'            # S1-S2: 纯排名，最后一名直接淘汰
        elif s <= 27: return 'Percentage'   # S3-S27: 百分比制
        else: return 'Rank + Save'          # S28+: 排名制 + 裁判救人 (Bottom 2)
    df['era'] = df['season'].apply(get_era)

    # 4. 转换长表 (Melt)
    id_vars = ['season', 'celebrity_name', 'last_active_week', 'era', 'placement', 'results']
    score_cols = [c for c in df.columns if 'week' in c and 'judge' in c and 'score' in c]
    df_long = pd.melt(df, id_vars=id_vars, value_vars=score_cols, var_name='wj', value_name='score')
    df_long['week'] = df_long['wj'].str.extract(r'week(\d+)').astype(float)
    df_long['score'] = pd.to_numeric(df_long['score'], errors='coerce')
    df_long = df_long[(df_long['score'] > 0) & (df_long['week'] <= df_long['last_active_week'])].copy()

    # 5. 聚合计算 Rank 和 Share
    weekly = df_long.groupby(['season', 'week', 'celebrity_name', 'era'])['score'].sum().reset_index()
    weekly.rename(columns={'score': 'raw_score'}, inplace=True)
    
    # 计算裁判份额 (用于 Sigmoid 同情分计算)
    weekly['judge_share'] = weekly['raw_score'] / weekly.groupby(['season', 'week'])['raw_score'].transform('sum')
    
    # 计算裁判排名 (Rank制必须项: 1=最高分)
    # method='min' 意味着如果有并列第一，则两个都是1，下一个是3
    weekly['judge_rank'] = weekly.groupby(['season', 'week'])['raw_score'].rank(method='min', ascending=False)
    
    # 找回淘汰标记
    res_map = df_long[['season', 'celebrity_name', 'results']].drop_duplicates()
    weekly = pd.merge(weekly, res_map, on=['season', 'celebrity_name'], how='left')

    def check_elim(row):
        res = str(row['results']).lower()
        if f"week {int(row['week'])}" in res and ('eliminated' in res or 'withdrew' in res):
            return 1
        return 0
    weekly['is_eliminated'] = weekly.apply(check_elim, axis=1)
    
    return weekly

# ==============================================================================
# 2. 核心模型: Rank Sigmoid Model (含 S28+ 特殊处理)
# ==============================================================================

def solve_rank_sigmoid_model(
    judge_ranks,        # 裁判排名 (1=Best)
    judge_shares,       # 裁判份额 (用于计算同情先验)
    eliminated_mask,    # 淘汰状态
    era_type='Rank',    # 赛制类型：区分 'Rank' 和 'Rank + Save'
    scale=10.0,         # 全局 Scale
    n_samples=50000,
    # --- Sigmoid 参数 ---
    sympathy_boost=1.3, 
    sigmoid_k=25.0
):
    n = len(judge_shares)
    avg_share = 1.0 / n
    
    # --- Step 1: 构造 Sigmoid 同情先验 ---
    # 同情逻辑：分数越低，Pity Score 越高
    threshold = avg_share
    score_diff = (threshold - judge_shares) / (np.std(judge_shares) + 1e-9)
    pity_scores = expit(sigmoid_k * score_diff)
    
    target_herding = judge_shares
    target_sympathy = np.ones(n) * avg_share * sympathy_boost
    
    # 合成先验 pi
    pi = (1 - pity_scores) * target_herding + pity_scores * target_sympathy
    if np.sum(pi) > 0: pi = pi / np.sum(pi)
    else: pi = np.ones(n) / n
    
    # --- Step 2: 蒙特卡洛模拟 ---
    alpha = pi * scale
    raw_samples = np.random.dirichlet(alpha, size=n_samples)
    
    # --- Step 3: 计算总排名 ---
    # 粉丝票数转排名：票数越高 -> 排名数值越小(1)
    # argsort(-x) 降序排列，argsort(argsort) 得到 rank index
    fan_ranks = np.argsort(np.argsort(-raw_samples, axis=1), axis=1) + 1
    
    # 总排名 = 裁判排名 + 粉丝排名
    # 注意：这里数值越大代表表现越差 (Rank 1 + Rank 1 = 2 Best; Rank 10 + Rank 10 = 20 Worst)
    total_ranks = judge_ranks + fan_ranks
    
    # --- Step 4: 过滤器 (核心区别) ---
    actual_elim_idx = np.where(eliminated_mask == 1)[0]
    if len(actual_elim_idx) == 0: 
        return pi, np.zeros(n), 1.0 # 无淘汰发生
    
    target_idx = actual_elim_idx[0]
    
    if era_type == 'Rank':
        # === S1-S2 逻辑 ===
        # 规则：Lowest Combined Score 被淘汰
        # 模拟：找到每行中 Total Rank 数值最大的人
        # Tie-breaker: 现实中如果总分平局，裁判分低的走。这里简化为直接看总分最大。
        sim_worst = np.argmax(total_ranks, axis=1)
        valid_mask = (sim_worst == target_idx)
        
    elif era_type == 'Rank + Save':
        # === S28+ 逻辑 ===
        # 规则：Bottom 2 进入裁判拯救环节
        # 模拟：只要目标落入 Bottom 2 (倒数两名)，即视为“有效样本”
        # (因为只要进了 Bottom 2，就有被裁判淘汰的可能性，而实际上他也确实被淘汰了)
        
        # argpartition 快速找到数值最大的两个索引 (即最差的两个排名)
        # 结果的最后两列就是 Bottom 2 的索引
        bottom2_indices = np.argpartition(total_ranks, -2, axis=1)[:, -2:]
        
        # 检查 target_idx 是否在 bottom2_indices 的每一行中
        # any(axis=1) 表示只要那两个里有一个是 target，就 True
        valid_mask = np.any(bottom2_indices == target_idx, axis=1)
    
    else:
        # 默认回退到 Rank 逻辑
        sim_worst = np.argmax(total_ranks, axis=1)
        valid_mask = (sim_worst == target_idx)

    # --- Step 5: 统计结果 ---
    valid_samples = raw_samples[valid_mask]
    n_valid = len(valid_samples)
    
    if n_valid < 10:
        return pi, np.zeros(n), 0.0
    
    median_share = np.median(valid_samples, axis=0)
    lower = np.percentile(valid_samples, 2.5, axis=0)
    upper = np.percentile(valid_samples, 97.5, axis=0)
    ci_width = upper - lower
    rate = n_valid / n_samples
    
    return median_share, ci_width, rate

# ==============================================================================
# 3. 奇偶交叉验证 (Cross Validation)
# ==============================================================================
def cross_validate_rank_model(df_input):
    print("\n[Validation] 开始 Rank 模型的奇偶交叉验证...")
    # 筛选 Rank 时代 (包含 S1-2 和 S28+)
    df_rank = df_input[df_input['era'].isin(['Rank', 'Rank + Save'])].copy()
    
    seasons = df_rank['season'].unique()
    odd_seasons = [s for s in seasons if s % 2 != 0]
    even_seasons = [s for s in seasons if s % 2 == 0]
    
    sets = {'Odd Seasons (Train)': odd_seasons, 'Even Seasons (Test)': even_seasons}
    
    for name, season_list in sets.items():
        total_ll = 0
        count = 0
        for s in season_list:
            s_data = df_rank[df_rank['season'] == s]
            for w in s_data['week'].unique():
                curr = s_data[s_data['week'] == w]
                if curr['is_eliminated'].sum() == 0: continue
                
                # 快速验证 (n_samples=2000)
                _, _, rate = solve_rank_sigmoid_model(
                    curr['judge_rank'].values, 
                    curr['judge_share'].values, 
                    curr['is_eliminated'].values, 
                    era_type=curr['era'].iloc[0], # 传入正确的赛制
                    scale=10.0,
                    n_samples=2000
                )
                
                total_ll += np.log(rate + 1e-9)
                count += 1
        
        avg_ll = total_ll / count if count > 0 else -np.inf
        print(f"  > {name} Avg Log-Likelihood: {avg_ll:.4f}")
    
    print("[Validation] 完成。Log-Likelihood 越高且两者越接近，说明模型越好。")

# ==============================================================================
# 4. 主执行流程
# ==============================================================================
if __name__ == "__main__":
    # 1. 读取和预处理
    input_file = '2026_MCM-ICM_Problems/2026_MCM_Problem_C_Data.csv'
    weekly_data = load_and_process_data(input_file)
    
    if not weekly_data.empty:
        # 2. 验证模型稳健性
        cross_validate_rank_model(weekly_data)
        
        # 3. 全量运行
        print("\n[Model Run] 开始全量计算 (Scale=10, Sigmoid, S28+ Logic)...")
        output_records = []
        
        # 只取 Rank 相关赛制
        df_target = weekly_data[weekly_data['era'].isin(['Rank', 'Rank + Save'])].copy()
        
        for season in tqdm(sorted(df_target['season'].unique())):
            s_data = df_target[df_target['season'] == season]
            # 获取该赛季赛制类型 (Rank 或 Rank + Save)
            season_era = s_data['era'].iloc[0]
            
            for week in sorted(s_data['week'].unique()):
                curr = s_data[s_data['week'] == week]
                
                names = curr['celebrity_name'].values
                j_ranks = curr['judge_rank'].values
                j_shares = curr['judge_share'].values
                is_elim = curr['is_eliminated'].values
                
                if np.sum(is_elim) == 0:
                    est_v, ci, rate = j_shares, np.zeros_like(j_shares), 1.0
                else:
                    est_v, ci, rate = solve_rank_sigmoid_model(
                        j_ranks, j_shares, is_elim, 
                        era_type=season_era, # 确保传入正确的赛制
                        scale=10.0,
                        n_samples=50000
                    )
                
                for i, name in enumerate(names):
                    output_records.append({
                        'season': season, 
                        'week': week, 
                        'celebrity_name': name,
                        'era': season_era,
                        'judge_rank': j_ranks[i], 
                        'is_eliminated': is_elim[i],
                        'estimated_fan_share': est_v[i],
                        'uncertainty_95ci': ci[i],
                        'valid_rate': rate
                    })
        
        final_df = pd.DataFrame(output_records)
        final_df.to_csv('dwts_model2_rank_final_results.csv', index=False)
        print("结果已保存至 'dwts_model2_rank_final_results.csv'")