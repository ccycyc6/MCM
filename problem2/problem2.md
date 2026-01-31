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









第一张图：命运分岔图 (The Destiny Timeline)
一句话总结： 数学公式决定了你是“神”还是“鬼”。
想象一下，我们把 Billy Ray Cyrus (第4季) 扔进三个平行宇宙里：
蓝色宇宙（排名法宇宙 - Rank Method）：
规则：只要不垫底就行。
情况：Billy 跳得烂（裁判分倒数第一），但他粉丝多。在排名法里，不管他跳得有多烂，只要不是全场最低分（或者粉丝够多），他就能混过去。
结果：他像穿了防弹衣一样，一直混到了第8周。图上的蓝条很长，这就是因为排名法掩盖了他“跳得烂”的事实。
红色宇宙（百分比法宇宙 - Percentage Method）：
规则：分差如实记录。
情况：他跳得烂，比如只拿了10分，别人拿了30分。这个20分的巨大深坑，粉丝票数再多也填不平。
结果：防弹衣被扒了。在第1周他就因为分差太大，直接被淘汰了。图上的红条极短，这就是“照妖镜”。
黑色标记宇宙（裁判拯救宇宙 - Judge Save）：
规则：倒数两名由裁判说了算。
情况：Bobby Bones (第27季) 这种超级流量王，粉丝多到在蓝红宇宙里都能夺冠。但是，只要他在第9周不小心掉进了倒数两名。
结果：裁判一看：“又是你这个跳得最烂的？” 直接淘汰。那个黑色的X，就是裁判手里的枪，专门在最后关头把混子干掉。
这张图证明了： 这些人的命不是天注定的，是算法注定的。



第二张图：死亡陷阱图 (The Trap Door)
一句话总结： 平时粉丝罩着你，一旦脚滑（掉进倒数两名），裁判就弄死你。
你看那个图里的坐标系：
往右走：裁判越讨厌你（跳得越烂）。
往上走：粉丝越喜欢你。
1. 那些蓝点点（安全区）：
这些争议选手（比如 Bobby Bones）平时都躲在右上角。
虽然他们在最右边（裁判讨厌），但因为他们在最上面（粉丝喜欢），两边一中和，总分还凑合。
这时候，粉丝是他们的保护伞。裁判看着生气，但干不掉他们。
2. 那个红叉叉（死亡陷阱）：
触发条件：某一周，他们粉丝投票稍微少了一点，或者别人发挥太好，导致他们滑落到了倒数两名 (Bottom Two)。
陷阱机制：这时候规则变了——粉丝的保护伞瞬间消失（不看粉丝票了）。
必死结局：
保护伞一撤，裁判一看：“哟，你不在最右边（倒数第一）蹲着呢吗？”
二选一的时候，裁判肯定留那个跳得好的，把这个跳得烂的（右边的）淘汰掉。
这就叫**“陷阱门”**：平时看着没事，只要地板（粉丝票）一抽走，立马掉下去摔死