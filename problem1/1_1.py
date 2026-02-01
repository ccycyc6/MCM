import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据准备
# ==========================================
df1 = pd.read_csv('model1_result.csv')
df2 = pd.read_csv('model2_result.csv')

# 选择展示的周次
# Model 1 示例: Season 19, Week 5
week_m1 = df1[(df1['season'] == 19) & (df1['week'] == 5)].copy()

# Model 2 示例: Season 32, Week 6
week_m2 = df2[(df2['season'] == 32) & (df2['week'] == 6)].copy()

# ==========================================
# 2. 通用饼图绘制函数（方案 1：无阴影 + 清晰边缘）
# ==========================================
def create_pie_chart(data, value_col, name_col, title, filename):
    plt.figure(figsize=(10, 8))

    # 按份额大小排序（顺时针更美观）
    data = data.sort_values(value_col, ascending=False)

    labels = []
    explode = []

    for _, row in data.iterrows():
        name = row[name_col]
        is_elim = row['is_eliminated']

        if is_elim == 1:
            labels.append(f"{name}\n(Eliminated ❌)")
            explode.append(0.08)   # 轻微突出
        else:
            labels.append(name)
            explode.append(0)

    # 柔和、学术风格配色
    colors = sns.color_palette('pastel', len(data))

    # 绘制饼图（不使用 shadow）
    wedges, texts, autotexts = plt.pie(
        data[value_col],
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        colors=colors,
        explode=explode,
        textprops={'fontsize': 12},
        wedgeprops={
            'edgecolor': 'white',   # 清晰分隔
            'linewidth': 1.5
        }
    )

    # 百分比文字样式
    plt.setp(autotexts, size=11, weight='bold', color='black')

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# ==========================================
# 3. 生成图表
# ==========================================
create_pie_chart(
    week_m1,
    'est_fan_share',
    'name',
    f'Model 1 Estimated Fan Votes\n'
    f'(Season {week_m1.iloc[0]["season"]}, Week {week_m1.iloc[0]["week"]})',
    'Pie_Chart_Model1.png'
)

create_pie_chart(
    week_m2,
    'estimated_fan_share',
    'celebrity_name',
    f'Model 2 Estimated Fan Votes\n'
    f'(Season {week_m2.iloc[0]["season"]}, Week {week_m2.iloc[0]["week"]})',
    'Pie_Chart_Model2.png'
)
