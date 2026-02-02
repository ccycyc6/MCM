import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import kendalltau

# ==========================================
# 1. 准备数据 (保持逻辑不变)
# ==========================================
pi = np.array([0.25, 0.22, 0.20, 0.18, 0.15])      # Judge Preference
v_prev = np.array([0.24, 0.21, 0.22, 0.17, 0.16])  # Momentum

def objective_function(v, alpha, beta, gamma, pi, v_prev):
    epsilon = 1e-9
    entropy = np.sum(v * np.log(v + epsilon))
    social = np.sum((v - pi)**2)
    momentum = np.sum((v - v_prev)**2)
    return alpha * entropy + beta * social + gamma * momentum

def solve_votes(alpha, beta, gamma):
    n = len(pi)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, 1) for _ in range(n)]
    res = minimize(objective_function, np.ones(n)/n, args=(alpha, beta, gamma, pi, v_prev),
                   method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

# ==========================================
# 2. 网格搜索 (Grid Search)
# ==========================================
# 提高一点分辨率，让图看起来更细腻
param_range = np.linspace(0.1, 3.0, 30) 
tau_matrix = np.zeros((len(param_range), len(param_range)))

v_baseline = solve_votes(1.0, 1.0, 1.0)

for i, gamma in enumerate(param_range):
    for j, beta in enumerate(param_range):
        v_new = solve_votes(1.0, beta, gamma)
        tau, _ = kendalltau(v_baseline, v_new)
        tau_matrix[i, j] = tau

# ==========================================
# 3. 绘制美化版热力图 (Aesthetic Plotting)
# ==========================================
plt.figure(figsize=(10, 8), dpi=120) # 提高DPI，清晰度更高
sns.set(style="white", font_scale=1.1) # 调整字体大小

# 翻转矩阵以匹配坐标轴方向
plot_data = np.flipud(tau_matrix)

# 使用 RdYlBu_r 配色 (红=低相关, 蓝=高相关)
# 这是一个非常经典的学术配色
ax = sns.heatmap(plot_data, 
                 xticklabels=5, # 减少刻度标签，避免拥挤
                 yticklabels=5, 
                 cmap="RdYlBu", # 注意：这里用 RdYlBu，如果想蓝是高，红是低，通常不需要 _r，视版本而定，通常 RdYlBu 是 红->蓝
                 vmin=0.6, vmax=1.0, # 设置显示范围，突显差异
                 cbar_kws={'label': r"Kendall's Rank Correlation ($\tau$)"},
                 linewidths=0, # 去掉格子线，让渐变更平滑
                 rasterized=True) # 栅格化，防止矢量图过大

# 重新设置刻度标签 (因为只显示了部分)
x_labels = np.round(np.linspace(0.1, 3.0, 6), 1)
y_labels = np.round(np.linspace(0.1, 3.0, 6), 1) # Y轴是从上到下的，需要注意

# 手动设置漂亮的坐标轴标签
ax.set_xticks(np.linspace(0, 29, 6))
ax.set_xticklabels(x_labels)
ax.set_yticks(np.linspace(0, 29, 6))
ax.set_yticklabels(np.flip(y_labels)) # 翻转Y轴标签以匹配 flipud

plt.xlabel(r'Social Fit Weight ($\beta$)', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel(r'Momentum Weight ($\gamma$)', fontsize=14, fontweight='bold', labelpad=10)
plt.title('Sensitivity Phase Diagram: Structural Stability\n(Model I: Percentage Era)', fontsize=16, pad=20)

# 标记基准点 (Baseline)
# 计算基准点在 30x30 网格中的位置
# 1.0 在 0.1-3.0 的位置大约是 30 * (1.0-0.1)/(3.0-0.1)
idx_pos = 30 * (1.0 - 0.1) / (2.9)
# 因为 Y 轴被 flipud 了，原来的 idx 变成了 (29 - idx)
plt.plot(idx_pos, 29 - idx_pos, 'o', markersize=12, 
         markerfacecolor='none', markeredgecolor='black', markeredgewidth=2, 
         label='Baseline (1.0, 1.0)')

# 添加图例
plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.show()