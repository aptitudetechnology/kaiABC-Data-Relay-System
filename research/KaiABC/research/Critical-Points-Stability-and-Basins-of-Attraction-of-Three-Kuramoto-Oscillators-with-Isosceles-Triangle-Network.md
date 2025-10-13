Critical Points, Stability, and Basins of Attraction of Three Kuramoto Oscillators with Isosceles Triangle Network



Authors: achieve the best HTML results from your LaTeX submissions by following these best practices.
License: arXiv.org perpetual non-exclusive license
arXiv:2407.11314v2 [math.DS] 26 Jul 2024
Critical Points, Stability, and Basins of Attraction of Three Kuramoto Oscillators with Isosceles Triangle Network
Xiaoxue Zhao School of Data Science and Department of Mathematics, City University of Hong Kong, Hong Kong SAR School of Mathematics, Harbin Institute of Technology, Heilongjiang, China Xiang Zhou Corresponding author: xizhou@cityu.edu.hk School of Data Science and Department of Mathematics, City University of Hong Kong, Hong Kong SAR
Abstract

We investigate the Kuramoto model with three oscillators interconnected by an isosceles triangle network. The characteristic of this model is that the coupling connections between the oscillators can be either attractive or repulsive. We list all critical points and investigate their stability. We furthermore present a framework studying convergence towards stable critical points under special coupled strengths. The main tool is the linearization and the monotonicity arguments of oscillator diameter.

Keywords: Kuramoto model; critical point; stability; basin of attraction.
1 Introduction

Synchronized collective phenomena are extensively studied in complex physical and biological systems. One of the most successful models in the interpretation of the phenomena is the Kuramoto model [9], which has a wide range of applications [1, 11]. Flame flickering is a common phenomenon observed in combustion systems, where flames exhibit oscillatory behavior due to various instabilities. Understanding and controlling these oscillations are crucial for improving combustion efficiency and reducing emissions. The Kuramoto model can be applied to flame flickering dynamics by modeling the flames as coupled oscillators[2, 8, 3]. Each flame element is considered an oscillator with the interactions modeled as coupling terms. We are interested in using the Kuramoto model as a powerful theoretic framework for understanding the synchronization and phase dynamics of flame flickering.

The literature [2] has conceptually proposed the Kuramoto model that involves three oscillators interconnected through an isosceles triangle network:
	
{θ˙1⁢(t)=K23⁢sin⁡(θ3⁢(t)−θ1⁢(t))+K13⁢sin⁡(θ2⁢(t)−θ1⁢(t))θ˙2⁢(t)=K23⁢sin⁡(θ3⁢(t)−θ2⁢(t))+K13⁢sin⁡(θ1⁢(t)−θ2⁢(t))θ˙3⁢(t)=K23⁢sin⁡(θ1⁢(t)−θ3⁢(t))+K23⁢sin⁡(θ2⁢(t)−θ3⁢(t)).
		(1)

The flame states are modelled as oscillators θi⁢(t)∈ℝ,i=1,2,3. Here the coupling strengths between θ1-θ3 and between θ2-θ3 are designed as the same K2, so that all three-body coupling strengths are assumed by only two nonzero real numbers K1 and K2. Our interest lies in the theoretical analysis of the synchronous dynamics of this system. It is evident that if Θ⁢(t)=(θ1⁢(t),θ2⁢(t),θ3⁢(t)) is a solution of (1), then for any constant c∈ℝ, Θ⁢(t)+c⁢(1,1,1) is also a solution. This is called translational invariance of the solution.

The dynamical system (1) is a gradient system Θ˙=−∇V⁢(Θ) with the energy function V⁢(Θ)=−13⁢[K2⁢cos⁡(θ3−θ1)+K1⁢cos⁡(θ2−θ1)+K2⁢cos⁡(θ2−θ3)]. It is easy to see that ∇V⁢(Θ) is continuous and ∇2V⁢(Θ) is continuous and uniformly bounded on ℝ3, then ∇V⁢(Θ) is globally Lipschitz in Θ on ℝ3 for any time t≥0. Hence, this gradient system with any initial data Θ⁢(0) has a unique solution over [0,∞). Calling Θ∗∈ℝ3 a critical point of system (1) refers to ∇V⁢(Θ∗)=0. Li and Xue [10, Theorem 1.1] have proved that limt→∞Θ⁢(t)=Θ∗∈{Θ∗∣∇V⁢(Θ∗)=0} and limt→∞Θ˙⁢(t)=0 for any initial data Θ⁢(0).

There have been some similar studies on system (1), see [4, 13, 14]. These results assume identical and positive coupling strengths (K1=K2>0). However, for flame dynamics, there is competition for limited resources among combustion units, such as oxygen or combustible material. This means that the interactions between these units can lead to negative coupling forces. The study of negative coupling forces can restrict the propagation speed, spread range, or combustion efficiency of a flame. As far as we know, there is no existing study on the dynamics (1) with non-identical and negative coupling strengths, which is the main focus of our research.

Contributions. In this paper, we will perform rigorous analysis for system (1) and the main results are twofold. Firstly, we compute all critical points and analyze their stability, based on the linearization (see Lemma 1 and Proposition 1). Our results indicate that for certain coupling strengths, stable critical points can coexist, implying richer dynamical properties compared to those described in reference [14]. Secondly, we analyze the basins of attraction of critical points. Specifically, we focus on a special case: K1=−K2<0, where two stable critical points coexist. We provide a framework for determining the basins of attraction of these two stable critical points using novel inequalities (see Theorem 1).

Organization of paper. In Section 2, we identify the formation and stability of critical points of system (1). In Section 3, we prove the convergence of the solution and give estimates on the basins of critical points. Section 4 is devoted to be a brief summary of this paper.
2 Critical points and stability

This section concerns the formation and stability of critical points, which plays a central role in systems theory. The following lemma lists all critical points in the range of [0,2⁢π)3 for system (1). The critical points discussed here are considered in an equivalence class sense. In other words, if Θ∗ is a critical point of the system (1), then Θ∗+2⁢L⁢π,L∈ℝ3 is also a critical point. This will not be reiterated further.
Lemma 1.

For any K1,K2∈ℝ∖{0}, there are four critical points of system (1):
	
Θ1∗=(0,π,0),Θ2∗=(0,π,π),Θ3∗=(0,0,0),Θ4∗=(0,0,π).
	

In addition, if |K22⁢K1|≤1, then there are two more critical points:
	
Θ5∗=(0,2⁢arccos⁡(−K22⁢K1),arccos⁡(−K22⁢K1)),
	

and
	
Θ6∗=(0,2⁢π−2⁢arccos⁡(−K22⁢K1),2⁢π−arccos⁡(−K22⁢K1)).
	

Proof.

The results can be easily obtained by setting the right-hand side of system (1) to zero.

The proposition below gives the stability of critical points, based on linearization.
Proposition 1.

The linear stability of critical points {Θi∗}i=16 of system (1) are as follows (Θi∗,i=5,6 only exist under the condition |K22⁢K1|≤1):

    (1)

    Θ1∗ and Θ2∗ are unstable for any K1,K2∈ℝ∖{0};
    (2)

    Θ3∗ is stable and all others are unstable if K2>0 and 2⁢K1+K2>0;
    (3)

    Θ4∗ is stable and all others are unstable if K2<0 and 2⁢K1−K2>0;
    (4)

    Θ5∗ and Θ6∗ are stable and all four others are unstable if 2⁢K1+K2<0 and 2⁢K1−K2<0.

Proof.

The Jacobian matrix JΘ∗ at Θ∗=(θ1∗,θ2∗,θ3∗) is
	
JΘ∗=13⁢[∗K1⁢cos⁡(θ2∗−θ1∗)K2⁢cos⁡(θ3∗−θ1∗)K1⁢cos⁡(θ1∗−θ2∗)∗K2⁢cos⁡(θ3∗−θ2∗)K2⁢cos⁡(θ1∗−θ3∗)K2⁢cos⁡(θ2∗−θ3∗)∗]
	

where Ji⁢i=−∑j=1,j≠i3Ji⁢j, are marked in “∗”. JΘ∗ has a simple eigenvalue 0, due to the shift invariance. So, as long as the remaining two eigenvalues are negative, the critical point Θ∗ is stable.The calculations of following eigenvalues and eigenvectors are trivial verification.

    (a)

    The eigenvalues and corresponding eigenvectors of Θ1∗ and Θ2∗ are λ1=0, λ2=K1+K12+3⁢K223, λ3=K1−K12+3⁢K223, and
    	
    ν1=[111],ν2=[−K1+K12+2⁢K22K2−1K1+K12+2⁢K22K2−12],ν3=[−K1−K12+2⁢K22K2−1K1−K12+2⁢K22K2−12].
    	
    (b)

    The eigenvalues and corresponding eigenvectors of Θ3∗ are λ1=0,λ4=−2⁢K1+K23, λ5=−K2, and ν1=[1 1 1]T, ν4=[−1 1 0]T, ν5=[−12−12⁢ 1]T.
    (c)

    The eigenvalues and corresponding eigenvectors of Θ4∗ are λ1=0, λ6=K2−2⁢K13, λ7=K2, and ν1,ν4,ν5.
    (d)

    The eigenvalues and corresponding eigenvectors of Θ5∗ and Θ6∗ are λ1=0, λ8=4⁢K12−K226⁢K1, λ9=K222⁢K1, and ν1,ν4,ν5.

Figure 1 demonstrates the above results.
Refer to caption
(a) K1>0
Refer to caption
(b) K1<0
Figure 1: The six critical points {Θi∗}i=16 (marked in six different colors) with dependence on the value of coupling strengths. The plot is the potential function V of the critical points in terms of the ratio K2/K1. The stability of each critical point is depicted by the pattern of the lines: the solid lines represent stability and the dashed lines represent instability.
3 Analysis of basins of attraction

From Figure 1, the critical points Θ1∗ and Θ2∗ are unstable for any K1 and K2, so the basins of attraction are Lebesgue zero-measure sets. For (K1,K2)∈((0,∞)×(0,∞))∪((−∞,0)×(−2⁢K1,∞)), Θ3∗ is the only stable point and the solution of system (1) starting from almost any initial value will converge to Θ3∗. The analysis for Θ4∗ is the same as that of Θ3∗ under the condition (K1,K2)∈((0,∞)×(−∞,0))∪((−∞,0)×(−∞,2⁢K1)). For (K1,K2)∈(−∞,0)×(2⁢K1,−2⁢K1), the critical points Θ5∗ and Θ6∗ are both stable, while all the others are unstable.

We will focus on the theoretic analysis of basins of attraction of co-existing stable points Θ5∗ and Θ6∗ when (K1,K2)∈(−∞,0)×(2⁢K1,−2⁢K1), i.e., K1<0 and |K2|<2⁢|K1|. We recognize that the ratio |K2|/|K1| might lead to substantial alterations in the shape and size of basins of attraction, making it difficult to accurately estimate by theory. However, utilizing the phase diameter function approach, a sufficient condition for the initial condition within one of the basins is available, provided that a specific assumption of K1=−K2 is met. On this assumption,
	
Θ5∗=(0,2⁢π3,π3),Θ6∗=(0,4⁢π3,5⁢π3).
	

Our primary analytical result is Theorem 1, derived in the content that follows. In the absence of the K1=−K2 assumption, there are difficulties within this approach because the estimate in Lemma 2 becomes nontrivial or perhaps impossible.

We consider Θ5∗ first since the case for Θ6∗ is similar. Performing the translation Θ~:=Θ−Θ5∗, i.e., θ~1=θ1,θ~2=θ2−2⁢π3,θ~3=θ3−π3, we can rewrite the system (1) as
	
{θ~˙1⁢(t)=K23⁢sin⁡(θ~3⁢(t)−θ~1⁢(t)+π3)−K13⁢sin⁡(θ~2⁢(t)−θ~1⁢(t)−π3)θ~˙2⁢(t)=K23⁢sin⁡(θ~3⁢(t)−θ~2⁢(t)−π3)−K13⁢sin⁡(θ~1⁢(t)−θ~2⁢(t)+π3)θ~˙3⁢(t)=K23⁢sin⁡(θ~1⁢(t)−θ~3⁢(t)−π3)+K23⁢sin⁡(θ~2⁢(t)−θ~3⁢(t)+π3)
		(2)

and the problem at hand is transformed into estimating an attraction domain of (0,0,0).

The challenge in studying system (2) lies in the fact that the signs before π3, referred to as the phase-lag or frustration [7, 5, 6], can be both positive and negative. There are studies that focus on two situations: no frustration and consistently positive frustration[4, 7, 5, 14, 6]. The commonly used method is to construct a phase diameter function, estimate its derivative, and apply the Grönwall’s inequality to prove that the phase diameter function will exponentially decay to zero. This paper adopts the same approach, but we provide new estimation inequalities for the frustration. Consider the phase diameter function defined by
	
𝒟⁢(Θ~⁢(t)):=maxi,j=1,2,3⁡(θ~i⁢(t)−θ~j⁢(t)),
	

which is non-negative, continuous and piece-wise differentiable with respect to time t. Let Mt=argmax⁡{θ~˙i⁢(t)∣i∈argmaxl=1,2,3⁢θ~l} and mt=argmin⁡{θ~˙i⁢(t)∣i∈argminl=1,2,3⁢θ~l}, then by [12, Lemma 2.2], the upper Dini derivative D+⁢𝒟⁢(Θ~⁢(t)) along the system (2) is given by
	
D+⁢𝒟⁢(Θ~⁢(t)):=lim suph↓0𝒟⁢(Θ~⁢(t+h))−𝒟⁢(Θ~⁢(t))h=maxi′∈Mt,j′∈mt⁡(θ~˙i′⁢(t)−θ~˙j′⁢(t)).
	

At each t, there are six possible combinations for the indices (i′ and j′ ) in the upper Dini derivative of the phase diameter function. We discuss each of them below.
Lemma 2.

Assume K1=−K2<0. Let t1,t2 be any two real number such that 0≤t1<t2.

    (1)

    If (i′,j′)≡(3,1) or (1,2) or (2,3), for all t∈(t1,t2), then
    	
    θ~˙i′⁢(t)−θ~˙j′⁢(t)=−2⁢K23⁢cos⁡(𝒟⁢(Θ~)2+π6)⁢[2⁢sin⁡(𝒟⁢(Θ~)2+π6)−cos⁡(θ~k−θ~i′+θ~j′2)];
    	
    (2)

    If (i′,j′)≡(2,1) or (3,2) or (1,3), for all t∈(t1,t2), then
    	
    θ~˙i′⁢(t)−θ~˙j′⁢(t)=−2⁢K23⁢[sin⁡(𝒟⁢(Θ~)−π3)+cos⁡(θ~k−θ~i′+θ~j′2)⁢sin⁡(𝒟⁢(Θ~)2+π3)],
    	

where k≠i′,j′.
Proof.

We only need to prove the result for (i′,j′)=(3,1), and the proofs for the rest are similar. By the definition of i′ and j′, 𝒟⁢(Θ~)=θ~3−θ~1. Then
	θ~˙3−θ~˙1 	
	=K23⁢sin⁡(θ~1−θ~3−π3)+K23⁢sin⁡(θ~2−θ~3+π3)−K23⁢sin⁡(θ~3−θ~1+π3)−K23⁢sin⁡(θ~2−θ~1−π3) 	
	=−2⁢K23⁢sin⁡(𝒟⁢(Θ~)+π3)+2⁢K23⁢cos⁡(θ~2−θ~32+θ~2−θ~12)⁢sin⁡(−𝒟⁢(Θ~)2+π3). 	

Using sin⁡(−𝒟⁢(Θ~)2+π3)=cos⁡(𝒟⁢(Θ~)2+π6) and sin⁡2⁢x=2⁢sin⁡x⁢cos⁡x, we obtain that
	θ~˙3−θ~˙1 	=−2⁢K23⁢sin⁡(𝒟⁢(Θ~)+π3)+2⁢K23⁢cos⁡(θ~2−θ~32+θ~2−θ~12)⁢cos⁡(𝒟⁢(Θ~)2+π6) 	
		=−2⁢K23⁢cos⁡(𝒟⁢(Θ~)2+π6)⁢[2⁢sin⁡(𝒟⁢(Θ~)2+π6)−cos⁡(θ~2−θ~32+θ~2−θ~12)]. 	

The following lemma states that the boundedness of the phase diameter implies the exponential decay for the proper upper bound 2⁢π3.
Lemma 3.

Assume K1=−K2<0. For any T∈(0,∞], if supt∈[0,T)𝒟⁢(Θ~⁢(t))<2⁢π3 in system (2), then there exists λ>0 such that 𝒟⁢(Θ~⁢(t))≤𝒟⁢(Θ~⁢(0))⁢e−λ⁢t, for any t∈[0,T).
Proof.

Let δ be an arbitrary positive number such that δ≤2⁢π3−supt∈[0,T)𝒟⁢(Θ~⁢(t)), then we shall give the estimation for D+⁢𝒟⁢(Θ~⁢(t)) under this condition.

For any t∈[0,T), if D+⁢𝒟⁢(Θ~⁢(t)) corresponds to Case (1) in Lemma 2, then by equations cos⁡(𝒟⁢(Θ~)2+π6)≥sin⁡δ2, 2⁢sin⁡(x2+π6)≥x5+1,∀x∈[0,π], and cos⁡(θ~k−θ~i′+θ~j′2)≤1,k≠i′,j′, we obtain that at time t,
	
D+⁢𝒟⁢(Θ~⁢(t))≤−2⁢K215⁢(sin⁡δ2)⁢𝒟⁢(Θ~⁢(t)).
	

If D+⁢𝒟⁢(Θ~⁢(t)) corresponds to Case (2) in Lemma 2, then by cos⁡(θ~k−θ~i′+θ~j′2)≥cos⁡𝒟⁢(Θ~)2,k≠i′,j′, and sin⁡(x−π3)+cos⁡x2⁢sin⁡(x2+π3)≥x5,∀x∈[0,π], we obtain that at time t,
	
D+⁢𝒟⁢(Θ~⁢(t))≤−2⁢K215⁢𝒟⁢(Θ~⁢(t)).
	

Hence, for any t∈[0,T), D+⁢𝒟⁢(Θ~⁢(t))≤−2⁢K215⁢(sin⁡δ2)⁢𝒟⁢(Θ~⁢(t)) always holds. The desired result can be obtained by choosing λ=2⁢K215⁢sin⁡δ2 and applying the Grönwall’s inequality.
Remark 1.

To apply Lemma 2, we need the max-min indices i′,j′ to remain constant over an interval (t1,t2). For strict rigor in the above proof, we can partition [0,T) into a countable of sub-intervals to satisfy such conditions by the discontinuous location of the index.
Proposition 2.

Assume K1=−K2<0. If the initial configurations of system (2) satisfy 𝒟⁢(Θ~⁢(0))<2⁢π3, then there exists λ^>0 such that 𝒟⁢(Θ~⁢(t))≤𝒟⁢(Θ~⁢(0))⁢e−λ^⁢t, for any t≥0.
Proof.

Select an arbitrary positive number δ^ such that δ^<2⁢π3−𝒟⁢(Θ~⁢(0)). Define the set 𝒯:={T>0∣𝒟⁢(Θ~⁢(t))<𝒟⁢(Θ~⁢(0))+δ^,∀t∈[0,T)}. Obviously, 𝒯 is not a empty set and T∗:=sup𝒯 is well defined. We claim that T∗=∞. If this is not true, i.e., T∗<∞, then
	
𝒟⁢(Θ~⁢(t))<𝒟⁢(Θ~⁢(0))+δ^,∀t∈[0,T∗)and𝒟⁢(Θ~⁢(T∗))=𝒟⁢(Θ~⁢(0))+δ^.
		(3)

By the first assertion of (3) and Lemma 3, we have that
	
supt∈[0,T∗)𝒟⁢(Θ~⁢(t))≤𝒟⁢(Θ~⁢(0))+δ^<2⁢π3,
	

and there exists a constant λ^>0 such that 𝒟⁢(Θ~⁢(t))≤𝒟⁢(Θ~⁢(0))⁢e−λ^⁢t, t∈[0,T∗). We let t→T∗−, and then 𝒟(Θ~(T∗)))≤𝒟(Θ~(0))e−λ^⁢T∗≤𝒟(Θ~(0)), which contradicts the second assertion of (3). So, we conduce that T∗=∞. Hence, supt≥0𝒟⁢(Θ~⁢(t))≤𝒟⁢(Θ~⁢(0))+δ^<2⁢π3. Applying Lemma 3 again, the desired result is obtained.

We now provide the main result of this paper.
Theorem 1.

Assume K1=−K2<0.

    (1)

    If the initial configurations of system (1) satisfy
    	
    −π<θ1⁢(0)−θ3⁢(0)<π3,−π3<θ2⁢(0)−θ3⁢(0)<π,−4⁢π3<θ1⁢(0)−θ2⁢(0)<0,
    		(4)

    then the solution Θ⁢(t) converges exponentially fast to the synchronization mode Θ5∗.
    (2)

    If the initial configurations of system (1) satisfy
    	
    −7⁢π3<θ1⁢(0)−θ3⁢(0)<−π,−π<θ2⁢(0)−θ3⁢(0)<π3,−2⁢π<θ1⁢(0)−θ2⁢(0)<−2⁢π3,
    		(5)

    then the solution Θ⁢(t) converges exponentially fast to the synchronization mode Θ6∗.

Proof.

Part (1): Based on Proposition 2 and equations θ1=θ~1,θ2=θ~2+2⁢π3,θ3=θ~3+π3, we obtain the first assertion. The proof of Part (2) is similar to (1).
4 Conclusion

We discussed theoretical findings on the stability and the attraction regions of critical points in Kuramoto oscillators, which relate to the nonlinear dynamics of three flames interacting in an isosceles configuration. The parameter of coupling strength can take negative values. The co-existence of stable oscillators was analyzed with the adequate conditions for their attraction basins. These findings enhance our qualitative understanding of the complex behaviors in Kuramoto oscillators, even in a minimal system comprising only three nodes.
Acknowledgements

This work is partially supported by Hong Kong RGC GRF grants 11308121, 11318522 and 11308323. Zhao acknowledges the supported of the Hong Kong Scholars Scheme (Grant No. XJ2023001), the Natural Science Foundation of China (Grant No. 12201156), the China Postdoctoral Science Foundation (Grant No. 2021M701013). We thank Peng Zhang at City University of Hong Kong for introducing us the background of flame oscillation.
References

    [1]

T.-M. Antonsen and R.-T. Faghih, et al. External periodic driving of large systems of globally coupled phase oscillators. Chaos, 18(3):037112, 2008.
[2]
Y. Chi, Z. Hu, T. Yang, and P. Zhang. Synchronization modes of triple flickering buoyant diffusion flames: Experimental identification and model interpretation. Phys. Rev. E, 109(2):024211, 2024.
[3]
A. Gergely and B. Sándor, et al. Flickering candle flames and their collective behavior. Sci. Rep., 10(1):21305, 2020.
[4]
S.-Y. Ha and M.-J. Kang. On the basin of attractors for the unidirectionally coupled Kuramoto model in a ring. SIAM J. Appl. Math., 72(5):1549–1574, 2012.
[5]
S.-Y. Ha, Y. Kim, and Z. Li. Asymptotic synchronous behavior of Kuramoto type models with frustrations. Netw. Heterog. Media, 9(1):33–64, 2014.
[6]
S.-Y. Ha, Y. Kim, and Z. Li. Large-time dynamics of Kuramoto oscillators under the effects of inertia and frustration. SIAM J. Appl. Dyn. Syst., 13(1):466–492, 2014.
[7]
C.-H. Hsia, C.-Y. Jung, B. Kwon, and Y. Ueda. Synchronization of Kuramoto oscillators with time-delayed interactions and phase lag effect. J. Differ. Equ., 268(12):7897–7939, 2020.
[8]
H. Kitahata and J. Taguchi, et al. Oscillation and synchronization in the combustion of candles. J. Phys. Chem. A, 113(29):8164–8168, 2009.
[9]
Y. Kuramoto. Chemical Turbulence. Springer, 1984.
[10]
Z. Li and X. Xue. Convergence of analytic gradient-type systems with periodicity and its applications in Kuramoto models. Appl. Math. Lett., 90:194–201, 2019.
[11]
Z. Li and X. Zhao. Synchronization in adaptive Kuramoto oscillators for power grids with dynamic voltages. Nonlinearity, 33(12):6624, 2020.
[12]
Z. Lin, B. Francis, and M. Maggiore. State agreement for continuous-time coupled nonlinear systems. SIAM J. Control Optim., 46(1):288–307, 2007.
[13]
J.-A. Rogge and D. Aeyels. Stability of phase locking in a ring of unidirectionally coupled oscillators. J. Phys. A: Math. Gen., 37(46):11135, 2004.
[14]

    X. Zhao, Z. Li, and X. Xue. Formation, stability and basin of phase-locking for Kuramoto oscillators bidirectionally coupled in a ring. Netw. Heterog. Media, 13(2):323–337, 2018.

