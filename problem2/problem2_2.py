import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 设置学术风格
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

# 数据构造 (基于之前的反事实模拟结果)
survival_data = [
    # Jerry Rice (S2)
    {'Name': 'Jerry Rice\n(Season 2)', 'Method': 'Rank Method (Actual)', 'Weeks': 8, 'Status': 'Finalist (Survived)'},
    {'Name': 'Jerry Rice\n(Season 2)', 'Method': 'Percentage Method', 'Weeks': 8, 'Status': 'Finalist (Survived)'},
    {'Name': 'Jerry Rice\n(Season 2)', 'Method': 'Judge Save', 'Weeks': 7, 'Status': 'Eliminated W7'},

    # Billy Ray Cyrus (S4)
    {'Name': 'Billy Ray Cyrus\n(Season 4)', 'Method': 'Rank Method (Actual)', 'Weeks': 8, 'Status': 'Semifinalist'},
    {'Name': 'Billy Ray Cyrus\n(Season 4)', 'Method': 'Percentage Method', 'Weeks': 1, 'Status': 'Eliminated W1'},
    {'Name': 'Billy Ray Cyrus\n(Season 4)', 'Method': 'Judge Save', 'Weeks': 5, 'Status': 'Eliminated W5'},

    # Bristol Palin (S11)
    {'Name': 'Bristol Palin\n(Season 11)', 'Method': 'Rank Method (Actual)', 'Weeks': 10, 'Status': 'Finalist (Survived)'},
    {'Name': 'Bristol Palin\n(Season 11)', 'Method': 'Percentage Method', 'Weeks': 10, 'Status': 'Finalist (Survived)'},
    {'Name': 'Bristol Palin\n(Season 11)', 'Method': 'Judge Save', 'Weeks': 9, 'Status': 'Eliminated W9'},

    # Bobby Bones (S27)
    {'Name': 'Bobby Bones\n(Season 27)', 'Method': 'Percentage Method (Actual)', 'Weeks': 9, 'Status': 'Winner'},
    {'Name': 'Bobby Bones\n(Season 27)', 'Method': 'Rank Method', 'Weeks': 9, 'Status': 'Finalist'},
    {'Name': 'Bobby Bones\n(Season 27)', 'Method': 'Judge Save', 'Weeks': 9, 'Status': 'Eliminated W9'}, 
]

df_fig1 = pd.DataFrame(survival_data)

# 绘图设置
fig1, ax1 = plt.subplots(figsize=(11, 7)) # 加宽一点
y_pos = np.arange(len(df_fig1['Name'].unique()))
bar_height = 0.25
names = df_fig1['Name'].unique()

# 绘制条形图
for i, name in enumerate(names):
    subset = df_fig1[df_fig1['Name'] == name]
    
    # 1. Rank Method (Blue)
    row1 = subset[subset['Method'].str.contains('Rank')]
    if not row1.empty:
        weeks = row1.iloc[0]['Weeks']
        ax1.barh(i + bar_height, weeks, height=bar_height, color='#3498db', label='Rank Method' if i==0 else "")
        ax1.text(weeks + 0.2, i + bar_height, f"{row1.iloc[0]['Status']}", va='center', fontsize=10, color='#2980b9', fontweight='bold')

    # 2. Percentage Method (Red)
    row2 = subset[subset['Method'].str.contains('Percentage')]
    if not row2.empty:
        weeks = row2.iloc[0]['Weeks']
        ax1.barh(i, weeks, height=bar_height, color='#e74c3c', label='Percentage Method' if i==0 else "")
        ax1.text(weeks + 0.2, i, f"{row2.iloc[0]['Status']}", va='center', fontsize=10, color='#c0392b', fontweight='bold')
        
    # 3. Judge Save (Black Marker X)
    row3 = subset[subset['Method'] == 'Judge Save']
    if not row3.empty:
        weeks = row3.iloc[0]['Weeks']
        # 只有当 Save 导致比 Actual 更早淘汰时才画 X
        actual_weeks = max(row1.iloc[0]['Weeks'] if not row1.empty else 0, row2.iloc[0]['Weeks'] if not row2.empty else 0)
        
        # 特殊处理：Bobby Bones 虽然走到最后，但在 Save 规则下会在 W9 被杀
        should_plot = (weeks < actual_weeks) or (name == 'Bobby Bones\n(Season 27)')
        
        if should_plot:
            ax1.scatter(weeks, i + bar_height/2, color='black', marker='X', s=180, zorder=10, label='Judge Save Triggered' if i==0 else "")
            # 标注文字
            if i == 0: # 只在第一个画解释箭头，避免乱
                ax1.annotate('Judge Save "Kill"', xy=(weeks, i + bar_height/2), xytext=(weeks-1, i + bar_height + 0.4),
                             arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center', fontsize=10, fontweight='bold')

# 轴设置
ax1.set_yticks(y_pos + bar_height/2)
ax1.set_yticklabels(names)
ax1.set_xlabel('Weeks Survived in Competition', fontweight='bold')
ax1.set_title('Counterfactual History: The "Judge Save" Veto', fontsize=16, pad=30)
ax1.set_xlim(0, 12) # 留出右侧空间给文字

# 图例设置 (放在顶部，不遮挡)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig('Figure_Destiny_Fixed.png', dpi=300)
plt.show()