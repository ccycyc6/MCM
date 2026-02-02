1.2
 Task 2: Compare the impact of Rank-Based and Percentage-Based aggregation methods on
competitive outcomes and historical controversies.
 任务2：比较基于排名与基于百分比的聚合方法对竞赛结果及历史争议的影响。
##这里还给出了基于分析，模型的更佳选择如何

 Task 3: Analyze the influence of celebrity demographics, industry background, and professional
任务3：分析名人人口统计特征、行业背景及专业搭档对评委评分与粉丝支持度的影响。
##demographics这个单词不当


p4 figure 1 de 图a和图b的具体解释
   figure 2 解释缺失（看不懂）
##两图位置不变
Figure 1_1_1:
Figure 1_1_1: The Mechanism of Sympathy Bias and Reconstructed Fan Distribution.
(a) The correlation between Judge Share ($J$) and Estimated Fan Share ($F$). (b)The Sigmoid-based "Pity Score" activation function. 
Figure 1_1_2:
Figure 1_1_1:The "Rational" Zone (Lower-Right Trend)What we see: The data points form a linear cluster. As Judge Score ($J$) increases, Fan Share ($F$) rises proportionally.2. The "Emotional" Zone (Upper-Left Tail) At low judge scores ($J < 0.15$), the fan share deviates upwards from the trend line.


Figure 1_1_4和1_1_5的位置应该并列放在3_2_2的末尾，且带上解释（1_1_5的解释在3.3的头上）

Figure 1_2_1:(直接放在3.3.1的末尾)
Figure 1_2_1:Distribution of Estimated Fan Shares by Outcome in Rank-Based Systems. The boxplots contrast the fan share distributions for Safe (0) and Eliminated (1) contestants. The "Rank" era exhibits significant variance and extreme outliers, visually confirming the high volatility and susceptibility to "Black Swan" fan interventions compared to the compressed distribution in the "Rank + Save" era.


删去1_2_2及其解释


Figure 1_3_1:(直接放在3.3.2的末尾)
Figure 1_3_1:Percentage Era (Blue): Percentage Era (Blue): Preserving continuous magnitudes enables tight convergence and structural robustness, reflected by high stability.Rank Era (Red): Information loss from ranking collapses stability and independence, expands uncertainty, and yields a chaotic profile.

Figure 1_3_2:(在Figure 1_3_1下面)
Percentage Era: Percentage Era: Continuous scoring preserves magnitude, yielding stable, predictable behavior.Rank Era: Ordinal ranking loses information, producing volatility and black-swan anomalies characteristic of chaotic systems.

Figure 1_4_1:(在1_3_2下面)
Temporal Evolution of Estimation Uncertainty. The visualization highlights a structural break in model precision driven by voting rules. The Percentage System (Seasons 3-27) enables near-perfect parameter recovery (near-zero CI width) by preserving magnitude information. Conversely, the Rank System (Seasons 1-2, 28+) introduces significant high uncertainty (wide intervals) due to information loss inherent in compressing continuous scores into ordinal ranks.



3.3.3部分的文字全部改成以下内容
We quantify the certainty of fan vote estimates through the width of 95% credible intervals (CI) derived from posterior sampling. This certainty is not uniformly distributed across all contestants or weeks, but exhibits significant heterogeneity driven by factors including scoring systems and elimination status.
Quantitative analysis reveals: Under percentage scoring (Seasons 3–27), estimates achieve near-perfect certainty (confidence interval width $\approx 0$). The magnitude-preserving property of consecutive scores imposes stringent equivalence constraints, forcing the optimization process to converge to a unique solution. In stark contrast, the ranking system (Seasons 1-2 and post-Season 28) introduced substantial uncertainty, with confidence interval widths typically ranging from [insert minimum value] to [insert maximum value] (e.g., 0.20–0.40). This reflects the inherent information loss characteristic of ordinal data, which expands the feasible solution space.
Furthermore, for each contestant and week, the “boundary constraint effect” causes fluctuations in certainty within a single week. As shown in Figure 8, eliminated contestants' uncertainty intervals are often narrower than those of safe contestants. The strict elimination condition ($Score_{elim} < Score_{safe}$) imposes a hard upper bound on the potential vote share, effectively compressing the solution space for lower-ranked contestants. In contrast, safe contestants retain a broader feasible range of vote shares.


p7 2_1_2.py是啥，直接改成“通过模拟。。。”

4.1 整体全部重构，参考2_1.docs,4.1包括的图有2_1_1,2_1_2

4.2"在模拟代码2 2 1.py中"改成“我们的模拟结果显示”

去除整个4.2.2部分



Figure 2_3(在4.3.1后面)
Figure 2_3: Decision Radar showing the structural trade-offs of voting mechanisms. The chart visually validates the superiority of the "Pct + Save" (Green) approach. While the pure Percentage Method (Blue) maximizes Fan Impact ("Democracy") at the cost of Stability, and Rank-based methods (Orange/Red) sacrifice Fan engagement for Predictability, the Pct + Save mechanism achieves the optimal balance. It expands the "Fairness" and "Stability" axes significantly—acting as the "Circuit Breaker"—while retaining a higher degree of Fan Impact than rank-based alternatives, thus preventing the "dampening" effect on audience participation.

去掉整个4.3.2



注意一下全文各处的双引号

task3我先不看了

任务四的标题改成“建立评分新机制”

6, "（评委）与"民主制"（粉丝）之间的矛盾，我们提出一种创新机制：自适应黄金锁混合协议结合二次投票（A-GLHP-QV）。"重复了

流程图需要美化，且放在6.1.3后面，6.2前面

"• 解读：黄金豁免与淘汰赛的组合机制确保被淘汰选手始终是风险组中技术实力最弱者，从而消除了系统的"遗憾"。"这里的解读改成：说明






1



















