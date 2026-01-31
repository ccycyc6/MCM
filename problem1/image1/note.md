1_3_1.png  在一致性指标
The Structural Trade-off (配合雷达图)"As illustrated in Figure [X], the two voting systems exhibit fundamentally opposing structural characteristics. 
The Percentage Era (Blue profile) prioritizes Stability, achieving a nearly perfect stability score due to the weighted-sum logic that preserves the magnitude of score differences. 
In stark contrast, the Rank Era (Red profile) sacrifices stability for Independence. By compressing scores into ordinal ranks, the system amplifies fan agency, allowing for outcomes that diverge significantly from judges' intent, albeit at the cost of increased chaotic volatility."


1_3_2.png  在一致性指标
Temporal Robustness (配合时序图)"Our temporal analysis (Figure [Y]) further validates this dichotomy. 
In the Percentage Era (top panel), the model maintains a consistent positive 'Safety Margin' between survivors and eliminated contestants, confirming the system's robustness. However, the Rank Era (bottom panel) is characterized by high stochasticity.
Notably, the model successfully identified two 'Black Swan' anomalies (marked with triangles), where the elimination outcomes were statistically improbable ($P < 0.1\%$), suggesting the influence of external factors beyond numerical performance."

1_1_1.png  模型1的自适应同情系统 (Adaptive Sympathy System)
Visualization of the Adaptive Pity-Momentum Mechanism.
(a) The relationship between raw judge shares and pity scores across different stages of the competition (color-coded by the number of contestants). The shifting curves demonstrate the model's ability to adapt its "sympathy threshold" based on the varying average judge share ($1/N$).
(b) The mechanism validation showing the underlying Sigmoid function. By normalizing the judge scores ($z$-score), all data points collapse onto a single theoretical curve, effectively distinguishing between the "Sympathy Region" (where fans rescue underdogs) and the "Herding Region" (where fans validate high scores).

1_1_2.png  模型1模型“拟合效果”的核心图表，它揭示了裁判评分与粉丝投票之间的非线性博弈关系
Decoupling Sympathy Bias from Elimination Constraints. (a) The Sympathy Effect: The scatter plot displays the estimated fan share versus judge share, color-coded by the Pity Score. The text labels (with white backgrounds) clearly demarcate the "Sympathy Zone" in the upper-left (where fans boost underdogs) and the "Herding Zone" in the lower-right (where fans agree with judges). (b) Elimination Outcome: The same data classified by competition result. The red dashed box highlights the "Danger Zone" where both judge and fan scores are low, leading to elimination. Note how the annotation is positioned to avoid obscuring the dense cluster of eliminated contestants (red crosses).

1_4_1.png 确定性分析图
Analysis of Model Certainty. (Left) Evaluation of a specific week (Season 28, Week 2) reveals that the model constraints are tightest for the eliminated contestant (Mary Wilson), resulting in high estimation certainty (narrow error bar), whereas safe contestants exhibit larger degrees of freedom. (Right) Comparing the Rank System (S1-S2) with the Percentage System (S3+), we observe no significant change in the distribution of uncertainty intervals. This confirms that the variation in certainty is driven by the binary outcome (Elimination vs. Safety) rather than the scoring algorithm itself.

1_1_3.png model1的动量轨迹图
 Temporal Evolution of Fan Support (Season 27 Case Study).The line chart tracks the estimated fan share ($F_t$) over the course of the season.The Momentum Effect: The smooth trajectories validate the model's "Momentum Hypothesis," suggesting that fan bases grow or shrink gradually rather than fluctuating randomly.The Bobby Bones Anomaly: Despite consistently lower judge scores, Bobby Bones (Red) maintains a dominant and stable fan share throughout the season, securing his victory.The Shock Elimination: Juan Pablo Di Pace (Gold), despite perfection from judges, shows a stagnant fan trend, leading to his elimination in Week 8 when the "Sympathy Vote" for others likely outpaced his "Validation Vote."

1_2_1.png model2的存活图
The Survival Landscape of Contestants. The scatter plot illustrates the relationship between Judge Rank (x-axis) and the model-estimated Fan Share (y-axis). Blue points represent contestants who advanced (Safe), while red points indicate those who were eliminated. The distinct separation between the two groups outlines a "survival frontier," demonstrating the compensatory mechanism where lower judge rankings require significantly higher fan support to avoid elimination.

1_2_2.png
裁判排名 (Judge Rank) 与 模型估算的粉丝份额 (Estimated Fan Share) 的关系图。

Figure 1_2_2: Correlation Analysis between Judge Ranks and Reconstructed Fan Shares in Season 2. Outliers (e.g., Jerry Rice) demonstrate the decoupling of judge evaluation and popularity.
趋势性 (General Trend)： 总体来看，裁判排名靠前（X轴数值小）的选手，粉丝份额通常也较高（分布在左上角），说明大众审美与专业评判在大部分情况下是一致的。
异常值 (The Anomalies - "The Jerry Rice Effect")： 注意图中被标注出的 Jerry Rice 和 Master P。他们的点位于 右上角 或 右中区域。
含义： 他们的裁判排名很差（数值大，甚至接近垫底），但你的模型反推算出他们的粉丝份额极高（Y轴高），这解释了为什么他们没有被淘汰。
作用： 这是对你模型准确性的极佳证明。你的模型成功捕捉到了导致Rank制度崩塌的“罪魁祸首”。
不确定性 (Uncertainty Bars)： 图中的灰色误差棒（Error Bars）展示了 uncertainty_95ci。
学术加分点： 在论文中强调这一点，说明你的模型不是盲目输出一个数值，而是考虑了信息熵或概率分布，体现了建模的严谨性。

