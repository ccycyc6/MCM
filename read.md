数据清理部分
前三点改成
To ensure the robustness of our mathematical models, we performed a rigorous data preprocessing
pipeline. This process transforms raw voting records into a structured format suitable for both optimization and simulation tasks.The raw dataset contains missing entries (e.g., “N/A”) for weeks where contestants did not perform. We strictly filtered these incomplete records to maintain the integrity of the weekly scoring matrix, ensuring that each time step functions as a closed competitive system. This prevents data gaps from introducing noise that could destabilize the entropy minimization model. Additionally, textual result descriptions were mapped to a standardized binary indicator $I_{elim} \in \{0,1\}$.
To ensure comparability across seasons with varying scoring scales (e.g., 30-point vs. 40-point caps), we transformed raw scores into a relative Judge Share ($J_{i,t}$):
$$J_{i,t} = \frac{Score_{i,t}}{\sum_{j=1}^{N_t} Score_{j,t}}$$
where $Score_{i,t}$ is the raw score of contestant $i$ in week $t$. This normalization mitigates numerical artifacts arising from historical rule changes, isolating the judges' relative preference structure—the primary signal driving the "Herding" effect.
We segmented the dataset into the "Percentage Era" (Seasons 3–27) and "Rank Era" (Seasons 1–2, 28+) based on the structural dichotomy of historical voting rules, creating a feature column era for data routing. This segmentation ensures mathematical compatibility, allowing us to apply the appropriate solver—continuous Gradient Descent or discrete Combinatorial approaches—tailored to the specific topology of each aggregation mechanism.Data Overview: The final processed dataset covers 34 seasons, providing a consistent timeline
of judge shares, elimination statuses, and era labels for all modeling tasks.



Figure 1_1_1_a:
Sigmoid-based "Underdog Protection Mechanism."
The model maps Normalized Judge Share ($J_i$) to Pity Score ($P_i$). When $J_i$ falls below the dynamic average threshold ($1/N$), $P_i$ rapidly saturates to 1.0 (upper-left). This simulates fans concentrating votes to rescue at-risk contestants, while high performers receive negligible sympathy boosts.
Figure 1_1_1_b:
Sigmoid-Based Voting Dynamics.The model utilizes a steep Sigmoid function ($\text{sigmoid\_k}=25.0$) to simulate a non-linear behavioral switch based on normalized underperformance ($z_i$).Sympathy Region ($z_i > 0, P_i \to 1$): Significant underperformance triggers the "Underdog Effect," maximizing pity scores to simulate audience rescue behavior.Herding Region ($z_i < 0, P_i \to 0$): Superior performance suppresses sympathy, aligning fan votes with judges' evaluations to reflect the "Herding Effect."

Figure 1_1_1_a,1_1_1_b:
Fan Vote Dynamics and Validation.
(a) The Sympathy Effect: The non-linear trend confirms our Sigmoid model successfully captures the "underdog" phenomenon, where low judge scores ($J$) trigger a compensatory boost in estimated fan support ($F$).
(b) Survival Logic: Validation shows eliminations (red $\times$) are concentrated in the "Danger Zone" (Low $J$ & $F$). Conversely, high fan intensity acts as a safety net, rescuing technically weaker contestants from elimination.



3.3.1第二段说到成功识别黑天鹅异常（黑园圈标记）





















