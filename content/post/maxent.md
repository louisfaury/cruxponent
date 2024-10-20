+++
author = "Louis Faury"
title = "Control in Entropy Regularized MDPs"
date = "2024-10-14"
+++

This post introduces the Maximum Entropy (MaxEnt) framework for Reinforcement Learning. The ambition is to see how "standard" control algorithms 
like value or policy iteration translate to the MaxEnt setting, and describe their convergence properties. We will also discuss how said
control approach influenced the design of some popular modern algorithms such as the Soft Actor Critic (SAC).
<!--more-->


## Problem formulation

Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, p, r)$ an MDP. 
For simplicity we will assume throughout that $\mathcal{M}$ is finite: $\vert \mathcal{S} \vert \times \vert \mathcal{A}\vert <+\infty$.
Let's briefly remind ourselves of the "classical" discounted objective for solving MDPs (see [this post](/post/mdp_basics/) for a complete refresher).
With $\lambda \in [0, 1)$, recall that the discounted cumulative reward objective writes:
$$
\tag{1}
v\_\lambda^\star := \max\_{\pi\in\mathcal{S}^{\text{MD}}}\mathbb{E}^\pi\left[\sum\_{t\geq 1} \lambda^{t-1} r(s\_t, a\_t) \right] \\; .
$$
Above, $\mathcal{S}^\text{MD}$ denotes the set of stationary, Markovian, deterministic policies.
In this post we will only be concerned with stationary policies; we will profusely conflate the notion of policy and decision-rule. 

### Entropy regularisation
For entropy-regularised MDPs the objective is augmented with an entropy penalty. Formally, with $\alpha \in \mathbb{R}^+$:
$$
\tag{2}
v\_{\lambda, \alpha}^\star := \max\_{\pi \in \mathcal{S}^\text{MR}}\mathbb{E}^\pi\left[\sum\_{t\geq 1} \lambda^{t-1} \\{r(s\_t, a\_t) + \alpha \mathcal{H}(\pi(\cdot\vert s\_t))\\}\right]\\;,
$$
where $\mathcal{H}(\pi(\cdot\vert s\_t)) = -\sum\_{a\in\mathcal{A}} \pi(a\vert s\_t)\log\pi(a\vert s\_t)$ is the entropy of $\pi(\cdot\vert s\_t)$.
Notice how we now optimise over the set of randomised stationary policies $\mathcal{S}^\text{MR}$; unlike the classical
MDP case, we can now longer anticipate that the optimal policy is deterministic.
Below, we will denote $\pi^\star\_\alpha$ the maximum-entropy optimal policy.
Notice that (2) can be exactly rewritten as an optimisation over a new MDP with a modified reward function:
$$
\tag{3}
v\_{\lambda, \alpha}^\star := \max\_{\pi \in \mathcal{S}^\text{MR}}\mathbb{E}^\pi\left[\sum\_{t\geq 1} \lambda^{t-1} \\{r(s\_t, a\_t) - \alpha \log\pi(a\_t\vert s\_t)\\}\right]\\;.
$$
The entropy-regularised objective promotes policies that trade-off visiting highly rewarding
(state, action) pairs with visiting states where the policy has kept its fair share of randomness. 
It is natural to ask _why_ we would be more interested in solving (2) than (1). 
Ultimately, we are only concerned about the cumulative reward captured by (1).
However, entropy regularisation in MDPs comes with several practical benefits. 
The obvious one is that it creates a natural framework to promote (shallow) exploration by encouraging the visit
of states for which the policy is still fairly random. 
Further, a max-entropy policy can also keep track of several alternative ways to maximise the discounted reward; this is
often an argument that it is more robust than its deterministic counterpart. 
Several research groups have observed that such policies generalise better under environmental shift  {{< ref link="ql">}} [1]{{< /ref>}} and
are easier to fine-tune to new environments or cross a sim-to-real gap  {{< ref link="sac">}} [2]{{< /ref>}}.
There is also an argument to be made for stability; by having policies off the edges of the simplex
$\Delta(\mathcal{A}$), policy improvement objectives remain smoother—easing optimisation from a numerical standpoint. 


{{< infoblock>}}
$\quad$ Observe that $\alpha$ should be adapted to the reward signal's scale.
In practice, it can be dynamically adjusted so we directly enforce an entropy constraint.
{{< /infoblock >}}

### Additional notations
Our objective in this blog-post is to understand what control algorithms (similar to value and policy iteration in MDPs)
look like in entropy-regularised MDPs.
We will need a couple of additional notations.
For any policy $\pi\in\mathcal{S}^\text{MR}$ we will write its _soft_ value function, for any $s\in\mathcal{S}$:
$$
v\_{\lambda, \alpha}^\pi(s) := \mathbb{E}\_s^\pi\left[\sum\_{t\geq 1} \lambda^{t-1} \\{r(s\_t, a\_t) + \alpha \mathcal{H}(\pi(\cdot\vert s\_t))\\}\right]\\; .
$$
As in classical MDP, the state-action value function writes $q\_{\lambda, \alpha}^\pi(s, a) = r(s, a) + \mathbb{E}_{s^\prime\sim p(\cdot\vert s, a)}\[v\_{\lambda, \alpha}^\pi(s^\prime)\]$, 
the optimal soft q-value function $q\_{\lambda, \alpha}^\star$ being naturally defined via this identity. 


## Soft Value Iteration
To mirror our MDP control post, we will start by deriving the entropy-regularised version of the Value Iteration (VI) algorithm -- called Soft Value Iteration (SVI).
We will now go through the usual "dance": introduce an operator $\mathcal{T}\_{\lambda, \alpha}^\star$, show that the soft-optimal value function is a fixed-point
of said operator and finally that said operator is contracting. 
As in the "classical" VI case, this will be enough to prove convergence of the sequence $q\_{t+1} = \mathcal{T}\_{\lambda, \alpha}^\star(q\_{t})$ to $q\_{\lambda, \alpha}^\star$.

### Soft-Bellman Operator


The soft Bellman operator $\mathcal{T}\_{\lambda, \alpha}^\star: \mathbb{R}^{\mathcal{S}\times\mathcal{A}} \mapsto \mathbb{R}^{\mathcal{S}\times\mathcal{A}}$ writes 
for any $q\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}$ and $s, a \in \mathcal{S}\times\mathcal{A}$:

$$
\begin{aligned}
\mathcal{T}\_{\lambda, \alpha}^\star(q)(s, a) = r(s, a) + \lambda\alpha \mathbb{E}\_{s^\prime \sim p(\cdot\vert s, a)} \left[\log \sum_{a^\prime \in \mathcal{A}} \exp(q(s^\prime, a^\prime)/\alpha)\right]\\; .
\end{aligned}
$$
To make sense of this expression it is useful to compare it to the "hard" Bellman operator: $\mathcal{T}\_{\lambda}^\star(q)(s, a) = r(s, a) + \lambda \mathbb{E}\_{s^\prime} \left[\max\_{a^\prime\in\mathcal{A}} q(s^\prime, a^\prime)\right]$.
This is where the hard-soft distinction appears. 
It is easy to see that as $\alpha\to 0$, $\alpha \log \sum_{a^\prime \in \mathcal{A}} \exp(q(s^\prime, a^\prime)/\alpha) \to \max\_{a^\prime\in\mathcal{A}} q(s^\prime, a^\prime)$.
On the other hand, as $\alpha\to \infty$ we end up averaging the q-values over the action space. 
In other words, $\mathcal{T}\_{\lambda, \alpha}^\star$ performs a soft-maximum with a temperature $\alpha$. 
### Algorithm 
As in the non-regularised setting, the algorithm is a simple fixed-point iteration:

{{< pseudocode title="Soft qVI" >}} 
$\textbf{init } q_0\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}, \text{ max. iteration T}\\$
$\textbf{for } t = 0, \ldots, T-1:\\$
$\qquad\textbf{for }  s\in\mathcal{S}, \, a\in\mathcal{A}:$
$$
    q_{t+1}(s, a) = r(s,a) + \lambda\alpha \sum_{s^\prime\in\mathcal{S}} p(s^\prime\vert s, a) \log \sum_{a^\prime \in \mathcal{A}} \exp(q(s^\prime, a^\prime)/\alpha)
$$
$\qquad\textbf{end for}\\$
$\textbf{end for}\\$
$\textbf{return } q_T$
{{< /pseudocode >}}

### Convergence
Proving convergence of the Soft qVI algorithm to the soft-optimal q-function follows the proof mechanism of the non-regularised case. 
It starts by claim (and proving) that $q\_{\lambda, \alpha}^\star$ is a fixed-point of $\mathcal{T}\_{\lambda, \alpha}^\star$.

{{< boxed title="Soft-Bellman fixed-point" >}}
$\qquad \qquad \qquad \qquad\qquad \quad\;\text{For all } (s, a)\in\mathcal{S}\times\mathcal{A}$:
$$
\mathcal{T}_{\lambda, \alpha}^\star(q_{\lambda, \alpha}^\star)(s, a) = q_{\lambda, \alpha}^\star(s, a)\; .
$$
{{< /boxed >}}


An important tool that we will use throughout this post is the soft-improvement update. 
Consider some policy $\pi\in\mathcal{S}^\text{MR}$ and define $\forall s, a \in\mathcal{S}\times\mathcal{A}$:
$$
\tag{4}
\pi^\prime(a\vert s) = \exp(q\_{\lambda, \alpha}^\pi(s, a) /\alpha - z\_{\lambda, \alpha}^\pi(s)/\alpha) \\;,
$$
with $ z\_{\lambda, \alpha}^\pi(s) = \alpha \log \sum_{a\in\mathcal{A}} \exp(q\_{\lambda, \alpha}^\pi(s, a)/\alpha)$ only
plays the part of a normalising constant. 
You can indeed also write (1) through the more concise
$
\pi^\prime(\cdot\vert s) \propto \exp(q\_{\lambda, \alpha}^\pi(s, \cdot) / \alpha)\; .
$
Through those equivalent definitions of $\pi^\prime$, we have introduced a policy soft-improvement operator. 
Indeed, for all $s, a \in\mathcal{S}\times\mathcal{A}$ we have:
$$q\_{\lambda, \alpha}^{\pi^\prime}(s, a) \geq q\_{\lambda, \alpha}^\pi(s, a)\\;.$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Let's prove that the operator (1) brings soft-improvement.
Observe that for any $\pi^{\prime\prime}$ and $s\in\mathcal{S}$ we have:
$$
\text{KL}(\pi^{\prime\prime}(\cdot\vert s) ||\pi^\prime(\cdot\vert s)) = -\mathcal{H}(\pi^{\prime\prime}(\cdot\vert s)) + \frac{1}{\alpha} \left(z\_{\lambda, \alpha}^\pi(s) - \sum\_{a\in\mathcal{A}} \pi^{\prime\prime}(a\vert s) q\_{\lambda, \alpha}^\pi(s, a)\right)\\; .
$$
from which we obtain that 
$$
\tag{5}
\pi^\prime \in \argmax\_{\pi^{\prime\prime}\in\mathcal{S}^\text{MR}} \mathcal{H}(\pi^{\prime\prime}(\cdot\vert s)) + \frac{1}{\alpha} \sum\_{a\in\mathcal{A}} \pi^{\prime\prime}(a\vert s) q\_{\lambda, \alpha}^\pi(s, a)\\;,$$
for any $s\in\mathcal{S}$.
Further, by expanding the state value function, we obtain that for any $s\in\mathcal{S}$:
$$
\begin{aligned}
v\_{\lambda, \alpha}^\pi(s) &:= \alpha\mathcal{H}(\pi(\cdot\vert s)) + \sum\_{a\in\mathcal{A}} \pi(a\vert s) q\_{\lambda, \alpha}^\pi(s, a) \\;,\\\
&\leq \alpha\mathcal{H}(\pi^\prime(\cdot\vert s)) + \sum\_{a\in\mathcal{A}} \pi^\prime(a\vert s) q\_{\lambda, \alpha}^\pi(s, a) \\;,&(\text{by (5)}) \\\
&= \alpha\mathcal{H}(\pi^\prime(\cdot\vert s)) + \sum\_{a\in\mathcal{A}} \pi^\prime(a\vert s) \left[r(s, a) + \lambda\sum\_{s^\prime\in\mathcal{S}}p(s^\prime\vert s, a) v\_{\lambda, \alpha}^\pi(s^\prime)\right]\\;, \\\
&= \mathbb{E}\_s^{\pi^\prime}\left[r(s, a\_1) + \alpha\mathcal{H}({\pi^\prime}(\cdot\vert s))\right] + \lambda \mathbb{E}\_{s}^a \left[v\_{\lambda, \alpha}^\pi(s\_2)\right]\\;, \\\
&\leq  \mathbb{E}\_s^{\pi^\prime}\left[r(s, a) + \alpha\mathcal{H}({\pi^\prime}(\cdot\vert s)\right] + \lambda  \mathbb{E}\_s^{\pi^\prime}\left[r(s\_2, a\_2) + \alpha\mathcal{H}(\pi^\prime(\cdot\vert s\_2)\right] + \lambda^2 \mathbb{E}\_{s}^a \left[v\_{\lambda, \alpha}^\pi(s\_3)\right] \\;, &(\text{repeat})\\\
&\leq \ldots \\;, &(\text{unroll})\\\
&\leq \mathbb{E}\_s^{\pi^\prime}\left[\sum\_{t\geq 1} \lambda^{t-1}\\{ r(s\_t, a\_t) + \alpha \mathcal{H}(\pi^\prime(\cdot\vert s\_t))\\}\right] = v\_{\lambda, \alpha}^{\pi^\prime}(s)\\; , 
\end{aligned}
$$
which concludes the proof.
{{% /toggle_block %}}


This tool is useful to prove the announced claim. Let $\pi\_\alpha^\star$ the soft-optimal policy. Since it cannot be soft-improved (by definition)
it must be a fixed point to (1). Hence, for all $s, a\in\mathcal{S}\times\mathcal{A}$:
$$
\tag{6}
\pi\_\alpha^\star(a\vert s) = \exp(q\_{\lambda, \alpha}^{\pi\_\alpha^\star}(s, a) /\alpha - z\_{\lambda, \alpha}^{\pi\_\alpha^\star}(s)/\alpha)\\;, 
$$
This is enough to yield:
$$
\begin{aligned}
q\_{\lambda, \alpha}^{\pi\_\alpha^\star}(s, a) &= r(s, a) + \lambda \mathbb{E}\_{s^\prime\sim p(\cdot \vert s, a)}\mathbb{E}\_{a^\prime\sim \pi\_{\lambda, \alpha}^\star(\cdot\vert s^\prime)} \left[q\_{\lambda, \alpha}^{\pi\_\alpha^\star}(s^\prime, a^\prime)- \alpha\log \pi\_{\lambda, \alpha}^\star(a^\prime\vert s^\prime)\right]\\:, &(\text{by def.})\\\
&= r(s, a) + \lambda \mathbb{E}\_{s^\prime\sim p(\cdot \vert s, a)} [z\_{\lambda, \alpha}^{\pi_\alpha^\star}(s^\prime)] \\;, &(\text{by (6)})\\\
&= r(s, a) + \lambda\alpha \mathbb{E}\_{s^\prime\sim p(\cdot \vert s, a)} \log \sum_{a^\prime\in\mathcal{A}} \exp(q\_{\lambda, \alpha}^{\pi\_\alpha^\star}(s^\prime, a^\prime)/\alpha)\\; .
\end{aligned}
$$
In other words, $q\_{\lambda, \alpha}^{\pi\_\alpha^\star}$ is a fixed-point of $\mathcal{T}\_{\lambda, \alpha}^{\star}$; for all $s, a\in\mathcal{S}\times\mathcal{A}$:
$$
q\_{\lambda, \alpha}^{\pi\_\alpha^\star}(s, a) = \mathcal{T}\_{\lambda, \alpha}^{\star}(q\_{\lambda, \alpha}^{\pi\_\alpha^\star})(s, a) \\; .
$$

As in the classical control case, what we have left to do is show that $\mathcal{T}\_{\lambda, \alpha}^{\star}$ is a _contracting_ operator.
This is actually quite straight-forward: we can replicate the classical VI proof by incorporating the entropy inside the reward function -- see (3).
The convergence of Soft-qVI to $q\_{\lambda, \alpha}^\star$ is then a formality (a replication of the Banach fixed-point theorem's proof). 
Hence:

{{< boxed title="Convergence of Soft qVI" >}}
$\qquad \qquad \qquad \qquad\qquad \quad\;\text{Let } \{q_t\}_{t\geq 1} \text{ the sequence maintained by Soft-qVI. Then:}$
$$
\lim_{t\to\infty} q_{t} = q_{\lambda, \alpha}^\star\; .
$$
{{< /boxed >}}
Observe that the optimal policy $\pi\_\alpha^\star$ can be retrieve from $q\_{\lambda, \alpha}^\star$ via the 
improvement operator (4).


## Soft Policy Iteration
Most of the groundwork needed to introduce and analyse Soft Policy Iteration (SPI) is now already done! 
Mirroring tge Policy Iteration algorithm, we will alternate between soft policy evaluation and soft policy improvement. Soft-policy evaluation is fairly straight-forward. By the identity:
$$
q\_{\lambda, \alpha}^\pi(s, a) = r(s, a) + \mathbb{E}\_{s^\prime\sim p(\cdot\vert s, a)} \mathbb{E}\_{a^\prime\sim \pi(\cdot\vert s^\prime)}\left[q\_{\lambda, \alpha}^\pi(s^\prime, a^\prime) - \alpha \log \pi(a^\prime\vert s^\prime) \right] \\; ,
$$
we can fall back to the classical policy evaluation framework and compute $q\_{\lambda, \alpha}^\pi$ by solving a linear system.
Alternatively, one can write the equivalent soft-policy operator $\mathcal{T}_{\lambda, \alpha}^\pi$ 
and go through a fixed-point algorithm to compute $q\_{\lambda, \alpha}^\pi$ (after showing that $\mathcal{T}\_{\lambda, \alpha}^\pi$ is contracting and that $q\_{\lambda, \alpha}^\pi$ is its fixed-point).
Soft policy improvement was already covered in (4); any stationary policy can be soft-improved by a policy $\pi^\prime$ where:
$$
\tag{7}
\pi^\prime(\cdot\vert s) \propto \exp(q\_{\lambda, \alpha}^\pi(s, \cdot)) \\; .
$$



### Algorithm

{{< pseudocode title="Soft Policy Iteration" >}} 
$\textbf{init } q_0\in\mathbb{R}^{\mathcal{S}\times \mathcal{A}}, \text{ max. iteration }T\\$
$\textbf{for } t = 0, \ldots, T-1\\$
$\quad\text{\color{YellowGreen}\# Soft Policy Improvement}\\$
$\qquad\textbf{for }  s\in\mathcal{S}, \, a\in\mathcal{A}:$
$$
    \pi_{t+1}(s, a) = \propto \exp(q_t (s, a))
$$
$\qquad\textbf{end for}\\$
$\quad\text{\color{YellowGreen}\# Soft Policy Evaluation}\\$
$\qquad \text{Compute } q_{t+1} = q_{\lambda, \alpha}^{\pi_{t+1}}\text{ by solving the fixed-point equation:}\\$
$$
    \forall (s, a), \quad q_{t+1}(s, a) = r(s, a) + \mathbb{E}_{s^\prime\sim p(\cdot\vert s, a)} \mathbb{E}_{a^\prime\sim \pi(\cdot\vert s^\prime)}\left[q_{t+1}(s^\prime, a^\prime) - \alpha \log \pi(a^\prime\vert s^\prime) \right] \; .
$$
$\textbf{end for}\\$
$\textbf{return } \pi_T = (d_T, d_T, \ldots)$
{{< /pseudocode >}}


### Convergence
Similarly to the classical PI, proving the convergence of SPI uses a monotonicity argument. 

{{< boxed title="Monotonic soft improvement" >}}
$\qquad \qquad \qquad \qquad\qquad \qquad \quad \text{ Let } \{\pi_t\}_{t\geq 1} \text{ the policy sequence maintained by SPI. Then for all } t\geq 1$:
$$
\forall(s, a)\in\mathcal{S}\times\mathcal{A}, \; q_{\lambda, \alpha}^{\pi_{t+1}}(s, a) \geq q_{\lambda, \alpha}^{\pi_{t}}(s, a)\;.
$$
{{< /boxed >}}

We already proved this result when proving the convergence of the Soft-qVI algorithm! 
Hence $\{q_{\lambda, \alpha}^{\pi_{t}}\}$ is a increasing, bounded sequence. It therefore converges; further, 
it must converge to $q\_\lambda^\star$ -- or else it could be further improved. That does it for the convergence of SPI!


## RL algorithms
Like qVI can be turned into the practical RL algorithm that is Q-learning, Soft qVI can go through similar stochastic and functional
approximation hoops (see [here](post/rl_landscape_vb)) to move from the control to the RL realm. 
The most salient difficulty comes from the inner-sum over the action space $\sum\_a \exp q\_t(s, a)$ present in the
soft Bellman operator $\mathcal{T}\_{\lambda, \alpha}^\star$. 
To avoid the computational cost of computing this term when the action space gets large, {{< ref link="ql">}} [1]{{< /ref>}}
suggest to approach it via importance sampling. Together with the design of a sampling network to achieve 
approach $a \sim \exp(q\_t(s, \cdot)/\alpha)/\eta$
this yields their Soft Q-learning algorithm. 

Similarly, the Soft Actor Critic {{< ref link="ql">}} [2]{{< /ref>}} is the RL counterpart of the control algorithm that is Soft (Generalised) Policy Iteration. 
Apart from many tricks destined to stabilise RL training with neural networks, the main idea is to "project" the policy improvement step 
back to the parametric class $\Pi$ covered by the chosen neural network architecture:
$$
\pi_{t+1}(\cdot\vert s) \in \argmin_{\Pi} \text{KL} \left (\pi(\cdot\vert s) \vert\vert \exp(q_{\lambda, \alpha}^{\pi_t}(s, \cdot) /\alpha)/\eta\right)\\; .
$$

Finally, the entropy-regularised objective (2) can easily be turned into a entropy-constrained one:
$$
\begin{aligned}
v\_{\lambda, \alpha}^\star := \max\_{\pi \in \mathcal{S}^\text{MR}}&\mathbb{E}^\pi\left[\sum\_{t\geq 1} \lambda^{t-1} \\{r(s\_t, a\_t)\\}\right]\\;,\\\
&\text{s.t } \mathbb{E}^\pi[\mathcal{H}(\pi(\cdot\vert s\_t))] \geq \beta\\; .
\end{aligned}
$$
Being imposed by direct considerations over physical concept (the actual entropy) the parameter $\beta$ is easier to
decide on (and does not have to be chosen depending on the reward's scale). 
Concretely, training doesn't change much—but for a dynamically updated value of $\alpha$, depending on current entropy values.

## References
<div id="ql"></div>
[1] Haarnoja & al, 2017. Reinforcement Learning with Deep Energy-Based Policies.

<div id="sac"></div>
[2] Haarnoja & al, 2019. Soft Actor-Critic Algorithms and Applications.


<div id="geist"></div>
[3] Geist & al, 2019. A Theory of Regularized Markov Decision Processes.<br>
<br>
<div id="vieillard"></div>
[4] Vieillard & al, 2021. Leverage the Average: an Analysis of KL Regularization in Reinforcement Learning.