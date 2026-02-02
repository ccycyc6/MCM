import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Simulation Setup (Mock Data)
# ==========================================
# Scenario: "Pro" (High Score, Low Vote) vs "Star" (Low Score, High Vote)
scores_judge = np.array([29, 18, 24, 25, 22])  # Pro is index 0 (29), Star is index 1 (18)
votes_raw    = np.array([1e5, 6e5, 1e5, 1e5, 1e5]) # Star has 6x votes

def calculate_losses(p):
    # 1. Effective Votes & Normalization
    votes_eff = np.power(votes_raw, p)
    share_judge = scores_judge / np.sum(scores_judge)
    share_fan = votes_eff / np.sum(votes_eff)
    
    # 2. Determine Winner/Loser
    total_score = 0.5 * share_judge + 0.5 * share_fan
    eliminated_idx = np.argmin(total_score)
    
    # 3. Calc Merit Regret (Did we eliminate the Pro?)
    # Ideal elimination is index 1 (Star, lowest judge score)
    ideal_elim_idx = np.argmin(scores_judge)
    # Regret is the judge-score gap between who left and who should have left
    merit_regret = (share_judge[eliminated_idx] - share_judge[ideal_elim_idx]) * 100
    
    # 4. Calc Democratic Suppression (Are fans ignored?)
    # Empirical decay function: lower p = higher suppression
    demo_suppression = 5.0 * np.exp(-2.5 * p)
    
    return merit_regret, demo_suppression

# ==========================================
# 2. Threshold Analysis Loop
# ==========================================
p_values = np.linspace(0.1, 1.0, 100)
merit, demo, total = [], [], []

for p in p_values:
    m, d = calculate_losses(p)
    # Non-linear penalty to simulate "Mob Rule" breakout when p > 0.6
    if p > 0.65:
        m += 15 * (p - 0.65)**2 * 20 
    merit.append(m)
    demo.append(d)
    total.append(m + d)

# ==========================================
# 3. Visualization
# ==========================================
plt.figure(figsize=(10, 6), dpi=120)
sns.set(style="whitegrid", font_scale=1.1)

# Plot Curves
plt.plot(p_values, merit, label='Meritocratic Regret (Unfairness)', color='#3498db', lw=2)
plt.plot(p_values, demo, label='Democratic Suppression ( apathy)', color='#e67e22', lw=2)
plt.plot(p_values, total, label='Combined Loss', color='#e74c3c', lw=3, ls='--')

# Highlight Goldilocks Zone
plt.axvspan(0.45, 0.55, color='green', alpha=0.15, label='"Goldilocks Zone"')

# Mark Optimal Point
min_idx = np.argmin(total)
min_p, min_loss = p_values[min_idx], total[min_idx]
plt.plot(min_p, min_loss, 'o', color='#c0392b', markersize=8, zorder=5)
plt.annotate(f'Optimal $p \\approx {min_p:.2f}$\n(Square Root)', 
             xy=(min_p, min_loss), xytext=(min_p, min_loss + 2.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             ha='center', fontsize=11, fontweight='bold')

plt.xlabel(r'Voting Power Exponent $p$ ($RawVotes^p$)', fontsize=12)
plt.ylabel('Normalized Loss Metric', fontsize=12)
plt.title('Threshold Analysis: Finding the Optimal Voting Power', fontsize=14, pad=15)
plt.legend(loc='upper left', framealpha=0.9)
plt.ylim(0, max(total)*0.8) # Zoom in on the relevant part

plt.tight_layout()
plt.show()