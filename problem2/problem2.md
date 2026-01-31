8.png "逆袭"案例轨迹图Jerry Rice
Temporal Evolution of Estimated Fan Share in Season 2. This time-series plot tracks the estimated popularity trajectories of the top 5 contestants throughout Season 2. The highlighted curve for Jerry Rice shows a sustained high fan share despite his consistently low judge scores. This contrasts with other contestants who relied more heavily on judge performance.


9.png 第二问第一小问灵敏度分析
Sensitivity Analysis of Overturn Rates against Fan Weight ($w$).
The red solid line represents the Percentage Method, while the blue dashed line represents the Rank Method. The plot reveals a significant "Structural Bias Gap" at $w=0.2$. Despite fans holding only a 20% nominal weight, the Percentage Method amplifies their influence to a 41.4% overturn rate, whereas the Rank Method restricts it to a linear 26.4%. This demonstrates the non-linear "Amplifier Effect" inherent in the percentage-based aggregation.

10.png 第二问第二小问方差对比
Distribution Variance Contrast between Judge Shares and Fan Shares. The violin plots illustrate the fundamental statistical disparity between the two data sources. Judge Shares (left, blue) are tightly concentrated around the mean with low variance, acting as a "Low-Energy" component. In contrast, Fan Shares (right, red) exhibit high dispersion and extreme outliers ("High-Energy"). The Percentage Method preserves this magnitude difference, allowing the high-variance fan signal to dominate the low-variance judge signal.

11.png “压路机”和“放大器”的理论区别
Conceptual Comparison of Aggregation Mechanisms. The blue dashed line illustrates the Rank Method acting as an "Equalizer": once a contestant surpasses another, the system reward is fixed regardless of the margin, effectively "killing" excess variance. The red solid line depicts the Percentage Method acting as an "Amplifier": system rewards scale linearly with raw performance gaps. The "Viral Zone" indicates the scenario where massive fan support creates a reward magnitude unattainable under the rank-based system.


12.png 命运分岔图
见problem2_2.md

13.png 死亡陷阱图
见problem2_2.md
