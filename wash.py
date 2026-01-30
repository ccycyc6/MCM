import pandas as pd
import numpy as np
import re

# ==========================================
# 1. 数据加载与清洗 (Load & Clean)
# ==========================================
file_path = '2026_MCM-ICM_Problems/2026_MCM_Problem_C_Data.csv'

# 尝试读取数据 (兼容CSV格式)
try:
    df = pd.read_csv(file_path)
    print("成功读取 CSV 文件")
except:
    df = pd.read_excel(file_path)
    print("成功读取 Excel 文件")

# 标准化列名：转小写，去空格，统一格式
df.columns = [c.strip().lower().replace(' ', '_').replace('/', '_') for c in df.columns]

# ==========================================
# 2. 解析核心状态：最后活跃周 & 赛制
# ==========================================

# 解析 "results" 列，提取数字
def parse_last_week(row):
    res = str(row['results']).lower()
    # 提取 "Eliminated Week X" 或 "Withdrew Week X"
    match = re.search(r'week (\d+)', res)
    if match:
        return int(match.group(1))
    
    # 标记决赛选手 (Winner, Runner Up等)，暂时标记为 99
    if any(x in res for x in ['winner', 'place', 'runner', 'finalist']):
        return 99 
    
    return 0 # 异常数据

df['last_active_week'] = df.apply(parse_last_week, axis=1)

# 动态计算每个赛季的最大周数 (用于修正决赛选手的活跃时间)
def get_season_max_week(season_df):
    # 找到该赛季所有含有 'score' 的列
    score_cols = [c for c in season_df.columns if 'judge' in c and 'score' in c]
    # 找到非全空的列，提取其中最大的周数
    valid_cols = season_df[score_cols].dropna(how='all', axis=1).columns
    weeks = [int(re.search(r'week(\d+)', c).group(1)) for c in valid_cols if re.search(r'week(\d+)', c)]
    return max(weeks) if weeks else 10

# 计算每个赛季的实际持续周数
season_max_weeks = df.groupby('season').apply(get_season_max_week)

# 修正决赛选手的 Last Week (将其设为该赛季的最后一面)
def fix_finalist_week(row):
    if row['last_active_week'] == 99:
        return season_max_weeks.loc[row['season']]
    return row['last_active_week']

df['last_active_week'] = df.apply(fix_finalist_week, axis=1)

# 赛制打标 (Era Tagging)
def get_era(season):
    if season <= 2: return 'Rank'
    elif season <= 27: return 'Percentage'
    else: return 'Rank + Save' # S28+

df['era'] = df['season'].apply(get_era)

# ==========================================
# 3. 数据重塑 (Melt to Long Format)
# ==========================================
# 保留ID列
id_vars = ['season', 'celebrity_name', 'last_active_week', 'era', 'placement', 'results']
# 提取分数相关列
score_cols = [c for c in df.columns if 'week' in c and 'judge' in c and 'score' in c]

# 宽表转长表
df_long = pd.melt(df, id_vars=id_vars, value_vars=score_cols, 
                  var_name='week_judge', value_name='score')

# 从列名中提取 Week 和 Judge 编号
df_long['week'] = df_long['week_judge'].str.extract(r'week(\d+)').astype(float)
df_long['judge'] = df_long['week_judge'].str.extract(r'judge(\d+)').astype(float)

# ==========================================
# 4. 指示函数应用 (Indicator Function)
# ==========================================
# 转换分数为数值型
df_long['score'] = pd.to_numeric(df_long['score'], errors='coerce')

# 核心过滤逻辑：
# 1. 分数必须 > 0 (排除已淘汰或缺席的 0 分)
# 2. 当前周必须 <= 该选手的最后活跃周 (排除淘汰后的无效记录)
df_long['is_valid'] = (df_long['score'] > 0) & (df_long['week'] <= df_long['last_active_week'])

# 只保留有效数据
df_clean = df_long[df_long['is_valid']].copy()

# ==========================================
# 5. 指标聚合与计算 (Aggregation)
# ==========================================
# 第一步：计算选手当周的裁判总分 (Raw Score)
# 注意：有些周可能有3个或4个裁判，这里直接求和
weekly_scores = df_clean.groupby(
    ['season', 'week', 'celebrity_name', 'era', 'last_active_week', 'placement', 'results']
)['score'].sum().reset_index()
weekly_scores.rename(columns={'score': 'raw_judge_score'}, inplace=True)

# 第二步：计算当周所有选手的总分池 (用于计算 Share)
season_week_totals = weekly_scores.groupby(['season', 'week'])['raw_judge_score'].transform('sum')

# 第三步：计算关键指标
# 1. Judge Share (裁判份额)
weekly_scores['judge_share'] = weekly_scores['raw_judge_score'] / season_week_totals

# 2. Judge Rank (裁判排名) - 降序排列 (分数越高，排名越前，Rank 1 = 最高分)
# 使用 method='min' 处理并列情况 (如两个第1名，下一个是第3名)
weekly_scores['judge_rank'] = weekly_scores.groupby(['season', 'week'])['raw_judge_score'].rank(method='min', ascending=False)

# 3. 标记当周被淘汰者 (Target Variable)
# 逻辑：如果是最后活跃周，且结果含有 "Eliminated" 或 "Withdrew"
def check_elimination(row):
    if row['week'] == row['last_active_week']:
        res = str(row['results']).lower()
        if 'eliminated' in res or 'withdrew' in res or 'quit' in res:
            return 1
    return 0

weekly_scores['is_eliminated'] = weekly_scores.apply(check_elimination, axis=1)

# ==========================================
# 6. 输出结果
# ==========================================
print("数据预处理完成！")
print(f"最终数据形状: {weekly_scores.shape}")
print(weekly_scores.head(10))

# 保存为 CSV (可以直接用于后续建模)
weekly_scores.to_csv('processed_dwts_model_input.csv', index=False)