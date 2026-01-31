
1.png  在一致性指标
The Structural Trade-off (配合雷达图)"As illustrated in Figure [X], the two voting systems exhibit fundamentally opposing structural characteristics. 
The Percentage Era (Blue profile) prioritizes Stability, achieving a nearly perfect stability score due to the weighted-sum logic that preserves the magnitude of score differences. 
In stark contrast, the Rank Era (Red profile) sacrifices stability for Independence. By compressing scores into ordinal ranks, the system amplifies fan agency, allowing for outcomes that diverge significantly from judges' intent, albeit at the cost of increased chaotic volatility."


2.png  在一致性指标
Temporal Robustness (配合时序图)"Our temporal analysis (Figure [Y]) further validates this dichotomy. 
In the Percentage Era (top panel), the model maintains a consistent positive 'Safety Margin' between survivors and eliminated contestants, confirming the system's robustness. However, the Rank Era (bottom panel) is characterized by high stochasticity.
Notably, the model successfully identified two 'Black Swan' anomalies (marked with triangles), where the elimination outcomes were statistically improbable ($P < 0.1\%$), suggesting the influence of external factors beyond numerical performance."

3.png  模型1的自适应同情系统 (Adaptive Sympathy System)
Visualization of the Adaptive Pity-Momentum Mechanism.
(a) The relationship between raw judge shares and pity scores across different stages of the competition (color-coded by the number of contestants). The shifting curves demonstrate the model's ability to adapt its "sympathy threshold" based on the varying average judge share ($1/N$).
(b) The mechanism validation showing the underlying Sigmoid function. By normalizing the judge scores ($z$-score), all data points collapse onto a single theoretical curve, effectively distinguishing between the "Sympathy Region" (where fans rescue underdogs) and the "Herding Region" (where fans validate high scores).

4.png  模型1模型“拟合效果”的核心图表，它揭示了裁判评分与粉丝投票之间的非线性博弈关系
Decoupling Sympathy Bias from Elimination Constraints. (a) The Sympathy Effect: The scatter plot displays the estimated fan share versus judge share, color-coded by the Pity Score. The text labels (with white backgrounds) clearly demarcate the "Sympathy Zone" in the upper-left (where fans boost underdogs) and the "Herding Zone" in the lower-right (where fans agree with judges). (b) Elimination Outcome: The same data classified by competition result. The red dashed box highlights the "Danger Zone" where both judge and fan scores are low, leading to elimination. Note how the annotation is positioned to avoid obscuring the dense cluster of eliminated contestants (red crosses).

5.png 确定性分析图
Analysis of Model Certainty. (Left) Evaluation of a specific week (Season 28, Week 2) reveals that the model constraints are tightest for the eliminated contestant (Mary Wilson), resulting in high estimation certainty (narrow error bar), whereas safe contestants exhibit larger degrees of freedom. (Right) Comparing the Rank System (S1-S2) with the Percentage System (S3+), we observe no significant change in the distribution of uncertainty intervals. This confirms that the variation in certainty is driven by the binary outcome (Elimination vs. Safety) rather than the scoring algorithm itself.

6.png model1的动量轨迹图
 Temporal Evolution of Fan Support (Season 27 Case Study).The line chart tracks the estimated fan share ($F_t$) over the course of the season.The Momentum Effect: The smooth trajectories validate the model's "Momentum Hypothesis," suggesting that fan bases grow or shrink gradually rather than fluctuating randomly.The Bobby Bones Anomaly: Despite consistently lower judge scores, Bobby Bones (Red) maintains a dominant and stable fan share throughout the season, securing his victory.The Shock Elimination: Juan Pablo Di Pace (Gold), despite perfection from judges, shows a stagnant fan trend, leading to his elimination in Week 8 when the "Sympathy Vote" for others likely outpaced his "Validation Vote."

7.png model2的存活图
The Survival Landscape of Contestants. The scatter plot illustrates the relationship between Judge Rank (x-axis) and the model-estimated Fan Share (y-axis). Blue points represent contestants who advanced (Safe), while red points indicate those who were eliminated. The distinct separation between the two groups outlines a "survival frontier," demonstrating the compensatory mechanism where lower judge rankings require significantly higher fan support to avoid elimination.


