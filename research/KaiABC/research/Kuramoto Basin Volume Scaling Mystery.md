

# **Expert Analysis of the Anomalous N​ Scaling in Kuramoto Basin Volumes: The Complexity Barrier Hypothesis**

## **I. Executive Summary: The N​ Paradox and the Complexity Barrier Hypothesis**

The empirical observation that the basin volume V of the synchronized state in the fully-connected Kuramoto model scales asymptotically as V∼1−exp(−αN​) presents a fundamental theoretical challenge in high-dimensional dynamical systems. This exponential scaling, confirmed with a high degree of accuracy (4.9% error) across system sizes N=10 to N=50 by fitting the data to V=1−exp(−αN​), profoundly contrasts with the system's observed local dynamical properties.

### **The Core Paradox**

The system exhibits robust mean-field characteristics that typically suggest simplicity and predictable scaling:

1. **Effective Degrees of Freedom (DOF):** The system reduces rapidly to a mean-field description, indicated by Neff​≈1. This suggests the dynamics are primarily governed by the order parameter R(t).  
2. **Critical Parameters:** The critical coupling strength Kc​ required for synchronization is nearly constant (Kc​(N=10)=0.088, Kc​(N=50)=0.100), failing to show the 1/N​ dependence necessary for the K$\_c$ Scaling Hypothesis.  
3. **Local Fluctuations:** Order parameter fluctuations are highly suppressed, scaling as σR​∼N−0.82, which is significantly weaker than the N−1/2 scaling predicted by the Central Limit Theorem (CLT) for independent variables.1 The system is strongly cohesive in its bulk dynamics.2

The highly sensitive exponential collapse of the global basin volume V∼exp(−αN​) alongside suppressed local fluctuations (σR​∼1/N) defies explanation by classical scaling hypotheses based on correlation length (ξ∼N−0.35) or effective degrees of freedom. The challenge is to identify a property that scales as N​ and exerts exponential control over the global geometry of the phase space, even as the local dynamics remain simple (1-DOF).

### **Proposed Theoretical Synthesis: The Complexity Barrier Hypothesis**

It is proposed that the exp(−N​) scaling is not a consequence of typical volumetric decay in high dimensions, but rather the result of the statistical physics governing the stability margin of the synchronized attractor. This stability margin is defined by the minimum "energy" barrier height (ΔH) separating the stable state from the saddle manifold (the basin boundary).  
This framework draws a parallel to the **Random Energy Model (REM)**, a foundational model for systems with quenched disorder, such as spin glasses.3 In the REM, the statistical complexity of the energy landscape, particularly the magnitude of energy extremes (e.g., the ground state or minimum energy barrier), scales universally as O(N​).3  
We hypothesize that the basin volume is proportional to the probability of the system *not* escaping its attractor under effective finite-N perturbations: V∝1−Pescape​. If the effective escape probability Pescape​ is governed by a characteristic action or energy barrier ΔH, such that Pescape​∼exp(−ΔH/Teff​), and if the complexity introduced by the quenched random frequencies ωi​ dictates that ΔH scales as ΔH∼CN​, then the empirical scaling V∼1−exp(−αN​) is naturally derived. This interpretation identifies N​ as a statistical complexity exponent arising from the quenched disorder rather than a standard volumetric or correlation length exponent.

## **II. The Kuramoto Phase Space and the Geometrical Collapse**

### **Defining the High-Dimensional Phase Space**

The phase space of the N-oscillator Kuramoto model is the N-torus, TN, with a total measure of (2π)N. Within this space, the basin of attraction B is the N-dimensional set of initial conditions that converge to the coherent synchronized solution. Above the critical coupling Kc​, this synchronized state (the attractor A) is locally stable. The volume V of the basin is the Lebesgue measure of B.  
In high dimensions, the volume calculation of geometric objects exhibits counter-intuitive properties. For instance, the volume of a hypersphere concentrates near its surface, often decaying exponentially with dimension, but with a rate function based on N/2 or logN, rather than N​.5 Crucially, the Kuramoto phase space is highly structured by the non-identical nature of the oscillators (ωi​), meaning the exp(−N​) scaling must be linked to this quenched disorder, not generic high-dimensional geometry.

### **Basin Boundaries and the Saddle Manifold**

The boundary of the basin of attraction ∂B in a dissipative flow is formed by the stable manifold of an unstable invariant set, typically a high-dimensional saddle point Θ∗.7 In the Kuramoto model above Kc​, the synchronized attractor A is separated from other dynamics (such as incoherent or drifting states) by a synchronous saddle point, Θ∗ (an unstable phase-locked state).9  
The stability of the system, often quantified by basin stability, relates directly to the size of V. The strong exponential decay V∼exp(−N​) demonstrates that while the local dynamics near A are simple (1-DOF), the global stability boundary ∂B is exponentially sensitive to N.

### **The Fragility of High-Dimensional Basin Geometry**

Recent studies examining basin geometry in high-dimensional systems, including coupled oscillators, have revealed that basins are rarely simple convex regions ("heads") but possess complex, extended structures ("tentacles").11 In high-dimensional systems, the vast majority of the basin volume is contained within these thin, fragile tentacles, which extend deep into the phase space but are immediately adjacent to the basin boundary, making them highly susceptible to perturbations.11  
This fragility provides the necessary context for the extreme sensitivity of V. The volume V is fundamentally constrained by the distance D between the attractor A and the saddle Θ∗, which defines the stability margin. If the geometry of the saddle manifold ∂B becomes exponentially complex or easily accessible as N increases, the probability of starting in one of the fragile tentacle regions and failing to reach A grows exponentially.12  
The fact that Neff​≈1 implies that the system's coherent motion is well-described by the mean field R. However, the volume V is defined in the full N-dimensional space, particularly along the N−2 transversal dimensions (phase differences ϕi​=θi​−Ψ) that are orthogonal to the collective order parameter R. The mechanism determining the basin boundary must therefore reside in the complexity of the transversal dynamics, where the impact of the quenched disorder ωi​ is strongest. The high-dimensional volume measure is effectively constrained by the measure of initial configurations that must satisfy a highly specific, disorder-dependent alignment near the saddle point Θ∗. This critical alignment measure decays exponentially as exp(−N​).

## **III. Scaling Theory Analogies: The N​ Exponent in Complexity Landscapes**

To formally derive the exp(−N​) scaling, it is necessary to move beyond traditional dynamical systems theory applied near mean-field limits and incorporate concepts from the statistical mechanics of disorder, specifically the scaling of energy barriers in rugged landscapes.

### **Analogy 1: The Random Energy Model (REM) Paradigm**

The Kuramoto model, with its fixed, randomly chosen natural frequencies ωi​, operates in a quenched disordered environment. The inherent structure of the potential landscape H(Θ) is fixed by the distribution g(ω). This landscape exhibits numerous metastable states (saddle points) and locally stable solutions, a phenomenology common to mean-field spin glasses and the REM.13  
In the Random Energy Model (REM), the energies Ex​ of the 2N states are independent Gaussian random variables with variance σE2​=N/2.3 A fundamental result of the REM is that the energy required to access the lowest energy states, or the height of the energy barriers (ΔH) separating configurations, scales proportionally to the standard deviation of the energy, ΔH∼N​.3 For instance, the critical energy per particle hc​ is proportional to ln2​.3  
The **Complexity Barrier Hypothesis** posits a direct correspondence: the barrier height ΔH separating the stable synchronized attractor A from the unstable synchronous saddle Θ∗ in the Kuramoto potential landscape is determined by the statistical extremes of the quenched disorder, forcing ΔH∼CN​.15  
If the basin volume V is related to the probability of remaining stable (V∝1−Pescape​), and this escape probability follows an Arrhenius-like law governed by the barrier height, Pescape​∼exp(−ΔH), the observed scaling follows immediately:

V(N)≈1−exp(−CN​)  
This result establishes that the N​ exponent is a statistical signature arising from the complexity introduced by the disordered field ωi​ in a mean-field system, rather than a generic property of phase transitions or geometry.

### **Analogy 2: The Moderate Deviation Principle (MDP)**

The observed scaling, V∼exp(−αN​), requires a non-standard formulation within the mathematical theory of rare events, known as Large Deviation Theory (LDP).16  
In LDP, the probability of a macroscopic rare event X deviating from its mean is typically given by P(X)∼exp(−NI(x)), where I(x) is the rate function (or action cost). If the event magnitude x is O(1), the decay is ultra-rapid, exp(−N), which is too fast to match the Kuramoto data. Conversely, the CLT describes fluctuations O(1/N​), which are not exponentially decaying.  
The **Moderate Deviation Principle (MDP)** addresses the intermediate regime of fluctuations aN​, where aN​→0 but N​aN​→∞.17 The V∼exp(−N​) scaling falls into this critical intermediate window.  
For the empirical scaling to hold within the standard LDP format P∼exp(−NI), the effective rate function I(N) must exhibit an explicit dependence on N:

exp(−αN​)=exp(−N⋅I(N))

This requires the rate function to scale as I(N)∼αN−1/2. This highly non-standard rate function dependence confirms that the measured scaling resides in the realm of finite-size corrections to the action principle, rooted in the statistical structure of the Kuramoto potential.  
The following comparison illustrates how the observed Kuramoto scaling deviates from, yet is bounded by, established statistical regimes.  
Table 1: Comparison of Observed Kuramoto Scaling with Theoretical Regimes

| Regime/Principle | Fluctuation Scaling (ΔX) | Probability/Volume Scaling (P∼V) | Mechanism Context | Relevance to Kuramoto Paradox |
| :---- | :---- | :---- | :---- | :---- |
| Central Limit Theorem (CLT) | N−1/2 | Gaussian Tails | Independent Components | Basis for finite-N noise; fails to explain V decay 1 |
| Large Deviation Principle (LDP) | O(1) (Macroscopic) | exp(−N⋅I) | Macroscopic Rare Event (Action cost) | Decay too rapid (exp(−N)). Requires I∼N−1/2 correction 16 |
| **Moderate Deviation Principle (MDP)** | N−γ,γ∈(0,1/2) | Intermediate Exponential Decay | Intermediate Scaling Limit | Provides framework for intermediate regime; requires I∼N−1/2 17 |
| **Random Energy Model (REM)** | N/A (Energy Landscape) | exp(−βN​) (Energy Extremes) | Disordered Mean-Field Systems | **Hypothesis Source:** Provides the N​ scaling for barrier height ΔH 3 |

The analysis suggests the Kuramoto basin problem is fundamentally a **disorder-induced MDP problem** whose rate function scaling is dictated by the extreme value statistics of the quenched disorder, characteristic of the REM.

## **IV. The Formal Mechanism: Finite-Size Correction to the Effective Potential**

The key to resolving the paradox—where simple dynamics yield complex global scaling—lies in understanding how the N​ scaling emerges from the stability margin defined by the saddle manifold Θ∗ within the full N-dimensional space.

### **The Two-Tiered Scaling Hierarchy**

The observation that Neff​≈1 accurately describes the dynamics near the synchronized attractor A. In this region (the bulk of the basin), the system is strongly cohesive, and fluctuations are suppressed below the CLT threshold (σR​∼N−0.82).2 This is the deterministic, mean-field core behavior.  
However, basin volume V is a global measure defined by the distance to the saddle manifold Θ∗. Escaping the basin requires the system to transition through a high-dimensional saddle point defined by a precise combination of the N phases, particularly along the N−2 dimensions orthogonal to the collective phase.10 The Neff​≈1 reduction fails to capture the measure and complexity of this boundary.

### **Barrier Height Scaling in the Disordered Potential**

For the Kuramoto system, which possesses a gradient-like structure (but is not a pure gradient flow due to the intrinsic frequencies ωi​), the likelihood of escaping the synchronized basin B is determined by the minimum energy path connecting A to Θ∗.  
In the Stochastic Kuramoto Model (SKM), which accounts for finite-size fluctuations as effective noise ϵ∼1/N​, the probability of escape Pescape​ is governed by the action I of the optimal instanton trajectory.21  
The effective energy cost ΔH (the height of the barrier) is not a constant value, O(1), but rather depends on the statistical distribution of the quenched disorder ωi​. In analogy with mean-field disordered systems, the minimum energy barrier in the N-dimensional phase space is statistically determined by the extreme configurations of the N random variables ωi​. The statistical extreme value of the potential landscape, or the height of the saddle point relative to the minimum, scales as ΔH∼CN​.3  
The scaling arises because the number of possible states (or trajectories) that contribute to the instability grows exponentially with N, forcing the minimum path cost to increase at the characteristic N​ rate associated with Gaussian statistical complexity.

### **The Unified Scaling Ansatz**

By interpreting the observed scaling V∼1−exp(−αN​) as the decay of the probability of stability, Pstable​, we conclude that the probability of collapse Pcollapse​≈exp(−αN​) represents the leading finite-size correction term to the stability probability in the thermodynamic limit.  
For the Kuramoto model well above Kc​, the synchronized state is globally stable as N→∞, meaning V→1 and Pcollapse​→0. If we express Pcollapse​ using the Large Deviation principle, the rate function I must approach zero such that NI→CN​.  
The formal scaling ansatz for the effective action I(N) near I0​=0 is:  
$$ P\_{\\text{collapse}} \\sim \\exp\\left( \-N \\left\[ I\_0 \+ \\frac{C}{\\sqrt{N}} \+ O\\left(\\frac{1}{N}\\right) \\right\] \\right) $$  
Since I0​=0 for the Kuramoto model well above criticality in the thermodynamic limit, the observed scaling is confirmed as the leading finite-size correction driven by the statistical complexity of the quenched disorder landscape:

V(N)≈1−exp(−CN​)  
This mechanism seamlessly reconciles the low effective dimensionality of the local dynamics (Neff​≈1) with the rapid collapse of the global stability volume. The local stability is determined by the collective mode R, but the boundary complexity, which dictates the volume, is determined by the N​ scaling of the disorder extremes in the remaining N−2 dimensions.

## **V. Computational Roadmap and Testable Predictions**

The Complexity Barrier Hypothesis makes several clear, testable predictions that move the research beyond empirical fitting and towards fundamental theoretical validation.

### **1\. Direct Measurement of the Energy Barrier Scaling**

The primary test of the REM analogy is the direct calculation of the effective energy barrier height (ΔH) between the synchronized attractor A and the closest unstable saddle Θ∗ as a function of N. This can be accomplished using continuation methods or minimum action path integration (instanton methods) in the N-dimensional phase space.

* **Prediction:** The computed barrier height must scale according to ΔH(N)∼CN​+O(1), where C is a constant dependent on the coupling margin (K−Kc​) and the width of the frequency distribution g(ω). Deviations from this scaling would necessitate the development of a modified scaling theory.  
* **Methodology:** Locate the saddle point Θ∗ and calculate the energy difference using a generalized Lyapunov function (or quasi-Lyapunov function) specific to the Kuramoto model.24 Given the system's size, careful analysis of the unstable manifold dimension du​ at Θ∗ is required.26

### **2\. Validation using Stochastic Dynamics and Moderate Deviation Theory**

If the scaling is indeed governed by a non-standard rate function I∼N−1/2, this can be validated by incorporating explicit stochastic noise into the Kuramoto model (SKM) and applying Large Deviation numerical methods, such as importance sampling or specialized Monte Carlo techniques.21

* **Prediction:** Applying finite-size scaling analysis to the simulated rate function I(N) for the rare event of synchronization loss (e.g., R dropping below a critical threshold R∗) should yield the dependence I(N)∝N−1/2.17 This would provide direct mathematical evidence supporting the MDP interpretation of the scaling.  
* **Requirement:** Since the fluctuations are weak (σR​∼1/N), extremely large ensemble averages (20,000 samples or more) are required, potentially combined with specialized algorithms tailored for rare event estimation in high dimensions.27

### **3\. Geometric Analysis of Saddle Manifold Fractal Dimension**

Although initial hypotheses on correlation length scaling failed (ξ∼N−0.35), the geometric properties of the basin boundary ∂B must still be highly complex to account for the exponential volume collapse. The N​ scaling may imply that the boundary is a multi-fractal structure whose effective dimension or uncertainty exponent scales inversely with N.12

* **Prediction:** The basin boundary exhibits multi-fractal structure. The capacity dimension d of the boundary should scale such that the uncertainty exponent α=N−d (where N is the dimension of the phase space) approaches zero in a manner related to N−1/2.12 Analyzing the density and structure of the 'tentacles' 11 near the boundary is critical to linking the fractal dimension directly to the N​ statistical complexity exponent.

## **VI. Conclusion**

The empirical scaling law V∼1−exp(−αN​) in the Kuramoto model is a signature of high-dimensional complexity dictated by the statistical distribution of quenched disorder. The paradox arises because the system's local dynamics are dominated by the mean field (Neff​≈1), while its global stability is governed by the geometry of the instability saddle manifold, which scales according to statistical extremes.  
The most compelling theoretical framework for this anomaly is the **Complexity Barrier Hypothesis**, which draws a direct parallel between the stability margin ΔH in the Kuramoto potential and the characteristic energy barrier scaling O(N​) found in the Random Energy Model. This implies that the N​ scaling is fundamentally a disorder-induced exponent, representing the leading finite-size correction term to the large deviation rate function I0​=0 for stability in the thermodynamic limit.  
This discovery is significant because it suggests a new universality class for stability scaling in high-dimensional dissipative systems with quenched disorder, potentially linking the non-equilibrium Kuramoto dynamics to canonical mean-field statistical mechanics systems like spin glasses and providing a concrete example of a physical system operating in the challenging theoretical regime of the Moderate Deviation Principle. Future research should focus rigorously on measuring the saddle barrier height scaling to validate the REM analogy and confirm the I(N)∼N−1/2 action decay.

#### **Works cited**

1. 3.5 Ensemble equivalence \- Statistical Mechanics \- Fiveable, accessed October 12, 2025, [https://fiveable.me/statistical-mechanics/unit-3/ensemble-equivalence/study-guide/siTj6G5M41IBxmsq](https://fiveable.me/statistical-mechanics/unit-3/ensemble-equivalence/study-guide/siTj6G5M41IBxmsq)  
2. Generative Models of Cortical Oscillations: Neurobiological Implications of the Kuramoto Model \- PMC \- PubMed Central, accessed October 12, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC2995481/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2995481/)  
3. Random energy model \- Wikipedia, accessed October 12, 2025, [https://en.wikipedia.org/wiki/Random\_energy\_model](https://en.wikipedia.org/wiki/Random_energy_model)  
4. From Derrida's random energy model to branching random walks: from 1 to 3 \- SciSpace, accessed October 12, 2025, [https://scispace.com/pdf/from-derrida-s-random-energy-model-to-branching-random-walks-4n92wc7aan.pdf](https://scispace.com/pdf/from-derrida-s-random-energy-model-to-branching-random-walks-4n92wc7aan.pdf)  
5. Volume of an n-ball \- Wikipedia, accessed October 12, 2025, [https://en.wikipedia.org/wiki/Volume\_of\_an\_n-ball](https://en.wikipedia.org/wiki/Volume_of_an_n-ball)  
6. Why is Gaussian distribution on high dimensional space like a soap bubble, accessed October 12, 2025, [https://stats.stackexchange.com/questions/419412/why-is-gaussian-distribution-on-high-dimensional-space-like-a-soap-bubble](https://stats.stackexchange.com/questions/419412/why-is-gaussian-distribution-on-high-dimensional-space-like-a-soap-bubble)  
7. Basin sizes depend on stable eigenvalues in the Kuramoto model | Phys. Rev. E, accessed October 12, 2025, [https://link.aps.org/doi/10.1103/PhysRevE.105.L052202](https://link.aps.org/doi/10.1103/PhysRevE.105.L052202)  
8. Fractal Basin Boundaries in Higher-Dimensional Chaotic Scattering David Sweet and Edward Ott\* Institute for Plasma Research and, accessed October 12, 2025, [https://www.andamooka.org/\~dsweet/Publications/pla1.pdf](https://www.andamooka.org/~dsweet/Publications/pla1.pdf)  
9. Stability diagram for the forced Kuramoto model \- arXiv, accessed October 12, 2025, [https://arxiv.org/pdf/0807.4717](https://arxiv.org/pdf/0807.4717)  
10. From incoherence to synchronicity in the network Kuramoto model | Phys. Rev. E, accessed October 12, 2025, [https://link.aps.org/doi/10.1103/PhysRevE.82.066202](https://link.aps.org/doi/10.1103/PhysRevE.82.066202)  
11. Basins with Tentacles | Phys. Rev. Lett., accessed October 12, 2025, [https://link.aps.org/doi/10.1103/PhysRevLett.127.194101](https://link.aps.org/doi/10.1103/PhysRevLett.127.194101)  
12. Basin entropy: a new tool to analyze uncertainty in dynamical systems \- PMC, accessed October 12, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4981859/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4981859/)  
13. Spin glass \- Wikipedia, accessed October 12, 2025, [https://en.wikipedia.org/wiki/Spin\_glass](https://en.wikipedia.org/wiki/Spin_glass)  
14. Energy-landscape networks of spin glasses | Phys. Rev. E \- Physical Review Link Manager, accessed October 12, 2025, [https://link.aps.org/doi/10.1103/PhysRevE.77.031105](https://link.aps.org/doi/10.1103/PhysRevE.77.031105)  
15. Energy landscape of the finite-size spherical three-spin glass model | Phys. Rev. E, accessed October 12, 2025, [https://link.aps.org/doi/10.1103/PhysRevE.87.052143](https://link.aps.org/doi/10.1103/PhysRevE.87.052143)  
16. Large deviations theory \- Wikipedia, accessed October 12, 2025, [https://en.wikipedia.org/wiki/Large\_deviations\_theory](https://en.wikipedia.org/wiki/Large_deviations_theory)  
17. Moderate deviation principles for unbounded additive functionals of distribution dependent SDEs \- American Institute of Mathematical Sciences, accessed October 12, 2025, [https://www.aimsciences.org/article/doi/10.3934/cpaa.2021099](https://www.aimsciences.org/article/doi/10.3934/cpaa.2021099)  
18. LARGE DEVIATIONS, accessed October 12, 2025, [https://math.nyu.edu/\~varadhan/LDP/1-2.pdf](https://math.nyu.edu/~varadhan/LDP/1-2.pdf)  
19. Lectures on the Large Deviation Principle \- Berkeley Mathematics, accessed October 12, 2025, [https://math.berkeley.edu/\~rezakhan/LD.pdf](https://math.berkeley.edu/~rezakhan/LD.pdf)  
20. Rotational Symmetry-Breaking effects in the Kuramoto model \- arXiv, accessed October 12, 2025, [https://arxiv.org/html/2509.04157v1](https://arxiv.org/html/2509.04157v1)  
21. Large deviations for stochastic Kuramoto–Sivashinsky equation with multiplicative noise \- Redalyc, accessed October 12, 2025, [https://www.redalyc.org/journal/6941/694173200006/694173200006.pdf](https://www.redalyc.org/journal/6941/694173200006/694173200006.pdf)  
22. Large deviations for stochastic Kuramoto–Sivashinsky equation with multiplicative noise, accessed October 12, 2025, [https://www.redalyc.org/journal/6941/694173200006/html/](https://www.redalyc.org/journal/6941/694173200006/html/)  
23. Finite size scaling of the Kuramoto model at criticality \- arXiv, accessed October 12, 2025, [https://arxiv.org/html/2406.18904v1](https://arxiv.org/html/2406.18904v1)  
24. Ch. 9 \- Lyapunov Analysis \- Underactuated Robotics, accessed October 12, 2025, [https://underactuated.mit.edu/lyapunov.html](https://underactuated.mit.edu/lyapunov.html)  
25. Lyapunov function for the Kuramoto model of nonlinearly coupled oscillators, accessed October 12, 2025, [https://www.researchgate.net/publication/226601067\_Lyapunov\_function\_for\_the\_Kuramoto\_model\_of\_nonlinearly\_coupled\_oscillators](https://www.researchgate.net/publication/226601067_Lyapunov_function_for_the_Kuramoto_model_of_nonlinearly_coupled_oscillators)  
26. Configurational stability for the Kuramoto–Sakaguchi model | Chaos \- AIP Publishing, accessed October 12, 2025, [https://pubs.aip.org/aip/cha/article/28/10/103109/856130/Configurational-stability-for-the-Kuramoto](https://pubs.aip.org/aip/cha/article/28/10/103109/856130/Configurational-stability-for-the-Kuramoto)  
27. Rare event estimation of high dimensional problems with confidence intervals, accessed October 12, 2025, [https://experts.illinois.edu/en/publications/rare-event-estimation-of-high-dimensional-problems-with-confidenc](https://experts.illinois.edu/en/publications/rare-event-estimation-of-high-dimensional-problems-with-confidenc)  
28. Finite-Time and Finite-Size Scaling of the Kuramoto Oscillators | Phys. Rev. Lett., accessed October 12, 2025, [https://link.aps.org/doi/10.1103/PhysRevLett.112.074102](https://link.aps.org/doi/10.1103/PhysRevLett.112.074102)  
29. Fractal dimension \- Wikipedia, accessed October 12, 2025, [https://en.wikipedia.org/wiki/Fractal\_dimension](https://en.wikipedia.org/wiki/Fractal_dimension)  
30. Scaling of fractal basin boundaries near intermittency transitions to chaos | Phys. Rev. A, accessed October 12, 2025, [https://link.aps.org/doi/10.1103/PhysRevA.40.1576](https://link.aps.org/doi/10.1103/PhysRevA.40.1576)