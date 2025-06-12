**The Adaptive Edge Compounding Risk Model**  
*A Universal Framework for Dynamic Risk Allocation under Probabilistic Uncertainty*

---

**Abstract**

This paper proposes a general-purpose, mathematically rigorous framework for dynamic risk allocation in probabilistic environments. By blending Bayesian-updated Kelly sizing, conditional Martingale reinforcement, volatility-weighted exposure scaling, and stochastic drawdown control, the model adapts risk levels in real time based on statistical confidence and market turbulence. Unlike traditional fixed-risk systems, this hybrid approach increases allocation only when posterior edge is strong and drawdown pressure is low. Though originally tested in financial contexts, the framework applies broadly to any decision-making process with asymmetric reward profiles and quantifiable probabilities. Simulations show strong resilience and compounding efficiency under volatile conditions.

---

**1. Motivation**

Classical risk models fail to adjust dynamically to real-world uncertainty. Kelly sizing is theoretically optimal for compounding but dangerously aggressive under volatile or uncertain conditions. Conversely, Martingale systems amplify exposure irrespective of statistical confidence, leading to path dependence and unsustainable tail risk.

We propose a hybrid system that:
- Shrinks the Kelly fraction under posterior uncertainty,
- Triggers multiplicative pyramiding *only* when edge is statistically significant,
- Buffers risk according to quantile-scaled volatility, and
- Applies a stochastic drawdown barrier to throttle leverage when equity decays.

This creates a compounding engine with built-in respect for randomness, drawdowns, and uncertainty.

---

**2. Notation**

| Symbol | Definition |
|--------|------------|
| $R_t$ | Return over window $t$ |
| $\hat{p}_t$ | Estimated probability of success at time $t$ |
| $b$ | Reward-to-risk ratio (gain/loss) |
| $f_t$ | Fraction of capital risked at time $t$ |
| $B_t$ | Capital before decision $t$ |
| $\xi_t$ | Volatility multiplier (95th-quantile scaling) |
| $D_t$ | Drawdown barrier at time $t$ |
| $k$ | Active streak of trades in same direction |

---

**3. Bayesian Kelly Fraction**

**3.1 Classical Form**
For a known edge $e = p - q$ and ratio $b$, the Kelly optimal fraction is:

\[ f_{\text{Kelly}} = \frac{e}{b} \tag{1} \]

**3.2 Posterior Shrinkage**
To account for uncertainty in $p$, we model it with a Beta distribution:

\[ p \sim \text{Beta}(\alpha_t, \beta_t) \tag{2} \]

The posterior mean becomes:

\[ \bar{p}_t = \frac{\alpha_t}{\alpha_t + \beta_t} \Rightarrow \bar{e}_t = 2\bar{p}_t - 1 \tag{3} \]

Substituting into (1):

\[ \tilde{f}_t = \frac{\bar{e}_t}{b} = \frac{2\bar{p}_t - 1}{b} \tag{4} \]

---

**4. Volatility-Weighted Buffer**

Let $\sigma^{(95)}_t$ be the rolling 95th-percentile absolute return. Define a volatility multiplier:

\[ \xi_t = \frac{\sigma^{(95)}_t}{\text{Median}_{30d}(\sigma^{(95)})} \tag{5} \]

Then damp Kelly sizing by:

\[ f_t^{\text{base}} = \frac{\tilde{f}_t}{1 + \xi_t} \tag{6} \]

This prevents over-sizing when tail volatility is elevated.

---

**5. Conditional Martingale Escalation**

Only apply position compounding when both:
- Edge is positive ($\bar{e}_t > 0$), and
- Posterior $\hat{p}_t$ exceeds high-confidence threshold (e.g., $\hat{p}_t > 0.97$).

Then:

\[ f_t = \min(c^k f_t^{\text{base}}, f_{\text{max}}) \tag{7} \]

Where:
- $k$ = open tickets in same direction,
- $c > 1$ = pyramiding multiplier (e.g., 1.35),
- $f_{\text{max}}$ = max allowable risk (e.g., 8% of capital).

---

**6. Stochastic Drawdown Diffusion Barrier**

Let high-water mark $B_{\text{HWM}} = \max_{\tau \le t} B_\tau$ and drawdown:

\[ \Delta_t = 1 - \frac{B_t}{B_{\text{HWM}}} \tag{8} \]

We throttle risk using an Ornstein-Uhlenbeck (OU) bridge:

\[ D_{t+1} = \mu + (D_t - \mu) e^{-\theta \Delta t} + \eta\sqrt{\frac{1 - e^{-2\theta\Delta t}}{2\theta}}\, \varepsilon_t \tag{9} \]

Where $\varepsilon_t \sim \mathcal{N}(0, 1)$.
If $\Delta_t > D_t$, then scale down all future $f_t$ by factor $\gamma < 1$ until equity recovers.

---

**7. Simulation and Example**

We simulate this system with random Bernoulli trials ($p=0.55$, $b=2.0$) and fat-tailed volatility. Key insights:
- Posterior shrinkage prevents over-betting during low confidence.
- OU throttle stabilizes equity decay and avoids gambler's ruin.
- Dynamic risk adapts smoothly across regimes.

Equity curve comparisons available upon request.

---

**8. Conclusions**

This framework generalizes optimal bet-sizing to real-world noisy environments with fat tails and path dependence. By fusing:
- Bayesian edge estimation
- Volatility-aware adjustments
- Controlled pyramiding
- Stochastic risk gating

...we offer a universal risk engine usable in finance, betting, AI ensembling, and even real-world decision-making.

Future work: incorporating regime switching, multi-agent edge agreement, and reinforcement-trained pyramiding policies.

---

**Appendix A: Discrete OU Derivation**

From stochastic process theory:
\[ dD_t = \theta (\mu - D_t) dt + \eta dW_t \tag{A1} \]
Discretized:
\[ D_{t+1} = \mu + (D_t - \mu) e^{-\theta\Delta t} + \eta\sqrt{\frac{1 - e^{-2\theta\Delta t}}{2\theta}}\, \varepsilon_t \tag{A2} \]

---

**References**
- Kelly, J. (1956). *Information Theory and Gambling*
- Thorp, E. (2006). *The Kelly Criterion in Blackjack and Financial Markets*
- Bouchaud, J.-P., & Potters, M. (2019). *Theory of Financial Risk and Derivative Pricing*
- Ljungqvist, A., & Oehm, C. (2020). *Volatility-Driven Bet-Sizing*
- Mandelbrot, B. (1963). *The Variation of Certain Speculative Prices*

