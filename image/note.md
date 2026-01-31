
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
Model Verification - Decoupling Sympathy Bias from Elimination Constraints. (a) The Sympathy Effect: The scatter plot displays the estimated fan share versus judge share, color-coded by the Pity Score. The text labels (with white backgrounds) clearly demarcate the "Sympathy Zone" in the upper-left (where fans boost underdogs) and the "Herding Zone" in the lower-right (where fans agree with judges). (b) Elimination Outcome: The same data classified by competition result. The red dashed box highlights the "Danger Zone" where both judge and fan scores are low, leading to elimination. Note how the annotation is positioned to avoid obscuring the dense cluster of eliminated contestants (red crosses).


