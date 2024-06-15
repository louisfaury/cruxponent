+++
author = "Louis Faury"
title = "Multi-Agent Dynamical Systems (2/3)"
date = "2023-10-21"
+++

This second blog-post of the multi-agent series is dedicated to _solution concepts_: how we can characterize and compare different joint policies. 
We will cover Pareto optimality, the definition and existence of Nash equilibrium,
as well as the minimax theorem in two-agents zero-sum games.

<!--more-->

In stark contrast with the single-agent case ($n=1$), calling some joint policy "optimal" in a multi-agent context ($n>1$) is ambiguous at best.
In a multi-agent setting, an outside observer considers the whole _joint_ policy; that is, 
the collection of _marginal_ agent-centric controllers.
While each agent tries to maximise its own return or utility, asking said observer (having no _a-priori_ preference over agents) to rank different joint policies is an ill-defined problem.

{{< infoblock>}}
$\quad$
We will adopt the notations of the first post and some notations from our MDP series. Further, for the sake of simplicity, we assume finite games. That is, 
the state, action and observation spaces are all discrete and finite:
$\vert\mathcal{S}\vert \times \vert\mathcal{A}\vert \times \vert\Omega\vert < \infty$.
{{< /infoblock >}}


## Ranking joint policies
Consider a general-sum partially observable stochastic game $\mathcal{M} = (n, \mathcal{S}, \mathcal{A}, \Omega, p, q, \bm{r})$.
For the sake of simplicity, we assume here that agents pursue a discounted reward objective. 
Formally, this means that the value of some joint policy $\bm{\pi}$ in the eyes of some agent $i$ is:
$$
v\_i(\bm{\pi}) := \mathbb{E}^{\bm{\pi}}\_\nu\left\[\sum\_{t\geq 1} \lambda^{t-1} r^i(s\_t, \bm{a}\_t)\right\]\\; ,
$$
where $\nu$ is some initial state distribution. 
The joint value function $\bm{v}\_\lambda(\bm{\pi})$ is simply the collection of the agents marginal values:
$
\bm{v}(\bm{\pi}) := (v\_1(\bm{\pi}), \ldots, v\_n(\bm{\pi}))\in\mathbb{R}^n\\; .
$
For $\bm{\pi}\_1, \bm{\pi}\_2$ two given joint policies, there is no _a-priori_
way to establish a strict order between the two: $\bm{v}(\bm{\pi}\_1)$ and $\bm{v}(\bm{\pi}\_2)$ are not comparable.

The only non-ambiguous case concerns cooperative games (shared reward), where, by construction, for every joint $\bm{\pi}$ one has:
$
v\_1(\bm{\pi}) = \ldots = v\_n(\bm{\pi})\\; .
$
The problem of raking policies collapses back to the single-agent case, since one can now look for an optimal policy w.r.t to 
the coalition's value. 


## Pareto Optimality 
Even if, in general, we cannot provide a strict order over joint values, we can still establish some partial ordering.
Concretely, this means that some situations can clearly be preferred over others: for instance, when one comes with a joint
improvement over another. This is called _Pareto domination_.

{{< boxed title="Pareto dominance" >}}
$\qquad\qquad\qquad\qquad \;\text{ Let } \bm{\pi}_1 \text{and }\bm{\pi}_2\text{ two joint policies. We say that }
\bm{\pi}_1\text{ Pareto-dominates }
\bm{\pi}_2\text{ if:}$
$$
\forall i \in \{1, \ldots, n\}, \; v_i(\pmb{\pi}_1) \geq v_i(\pmb{\pi}_2)\; .
$$
{{< /boxed >}}

A Pareto-dominated policy can be jointly improved; any agent's return can be increased, without
hindering the return of the others. 
Any policy that is not Pareto-dominated by another is called _Pareto-optimal_. 
For any such policy, the return of an agent can no longer be improved without having another see its value drop. 

It is easy to check that every finite game admits a Pareto-optimal policy. Often, there will exist several Pareto-optimal policies, not all of them being useful or relevant. 
For instance, in a zero-sum game, every joint policy is Pareto-optimal! (This is a direct result of the reward structure.) At the opposite end of the spectrum where lay cooperative games, all Pareto-optimal policies have the same value
(and are therefore completely equivalent). This visible variability behind the interpretation of Pareto-optimality testifies 
of this concept's relative weakness. It is useful to trim out some dull policies, but not much more. Game-theorists have 
therefore somewhat let go of the concept to develop stronger agent-centric solution concepts.


## Best-Response
A _best-response_ policy is not a solution concept _per se_ as it only concerns marginal policies.
The idea will however be quite useful to introduce the Nash equilibrium, a proper solution concept.
To understand what a best-response is, let's fix some joint policy $\bm{\pi}$ and put ourselves in the shoes of some agent $i$.
If the joint policy $\bm{\pi}\_{-i}$ of the other agents is known and fixed, then agent $i$'s path is straight-forward. 
To maximize its return, it should follow a _best-response_ to $\bm{\pi}\_{-i}$: a
marginal policy $\pi\_i$ which maximizes $v\_i$ assuming $\bm{\pi}\_{-i}$ will not change. Below, we will explicitly
split $\bm{\pi} = (\pi\_i, \bm{\pi}\_{-i})$.

{{<boxed title="Best-Response" >}}
$\qquad \qquad \qquad \quad \text{Fix } i\in\{1, \ldots, n\} \text{ and let }\bm{\pi} = (\pi_i, \bm{\pi}_{-i})$.
$\text{The best-response operator } \mathcal{T}_i^{\text{br}}\text{ is defined as:}$
$$
\mathcal{T}_i^{\text{br}}(\bm{\pi}_{-i}) = \argmax_{\pi_i} v_i(\pi_i, \bm{\pi}_{-i})
$$
{{< /boxed >}}
Observe that a policy $\bm{\pi}\_{-i}$ can admit several best-responses. The best-response operator
$\mathcal{T}\_i^\text{br}$ is therefore a set-valued function. Given some joint policy $\bm{\pi}$ we define the joint best-response set as:
$$
\mathcal{T}^{\text{br}}(\bm{\pi}) := \mathcal{T}\_1^{\text{br}}(\bm{\pi}\_{-1}) \times \ldots \times \mathcal{T}\_n^{\text{br}}(\bm{\pi}\_{-n})\\; .
$$


## Nash Equilibrium
The concept of Nash equilibrium is relatively straight-forward, yet arguably one of the most influential in game theory.
A joint policy
$\bm{\pi}$ is a Nash equilibrium if each marginal policy $\pi\_i$ is a best-response to the remaining $\bm{\pi}\_{-i}$.
Intuitively, this comes with plenty of stability: for any agent, even if he knew the 
strategies of the others, he would have no advantage to change his. 
As a result, no agents have _a-priori_ some interest to deviate from the current joint strategy.

{{<boxed title="Nash Equilibrium" >}}
$\qquad \qquad \qquad \qquad \; \bm{\pi}\text{ is a Nash-equilibrium if:}$
$$
\forall i\in\{1, \ldots, n\}, \; \pi_{-i} \in \mathcal{T}^\text{br}_i(\bm{\pi}_{-i})\; .
$$
{{< /boxed >}}

In other words, a Nash-equilibrium policy $\bm{\pi}$ is such that $
\bm{\pi} \in \mathcal{T}^\text{br}(\bm{\pi}) \\; .
$

{{< warningblock >}}
$\quad$ To ease exposition, we will now restrict our attention to simple matrix games
$\mathcal{M} = (n, \mathcal{A}, \bm{r})$. The Nash equilibrium is a fairly portable concept,
and most of the results below can be generalised to more complex games (although, often, at the price
of some simplifying hypothesis).
{{< /warningblock >}}

The first question that pops into our mind is whether such policies exist. 
That's the point of [Nash's theorem](https://en.wikipedia.org/wiki/Nash_equilibrium#:~:text=Proof%20of-,existence,-%5Bedit%5D),
which we will state and prove shortly. Before that, let's establish a few useful results.
Below, we denote $\Pi\_i = \Delta(\mathcal{A}\_i)$ the policy space for agent $i$, and
$\bm{\Pi} = \prod\_{i=1}^n \Pi\_i$ the joint-policy space.
Our first claim establishes that there always exists a deterministic best-response policy.

<ins>**Claim 1.**</ins> For every $\bm{\pi}\in\bm{\Pi}$ and $i\in\\{1, \ldots n\\}$ there exists
a deterministic policy in $\mathcal{T}^\text{br}_i(\bm{\pi}\_{-i})$.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Let's start by re-writing the value-function for agent $i$'s standpoint. Fix $\bm{\pi}\_{-i}$, and
for every $\pi\_i\in\Pi\_i$:
$$
\begin{aligned}
v\_i(\pi\_i, \bm{\pi}\_{-i}) &= \mathbb{E}^{\bm{\pi}}\left[ r^i(\bm{a})\right] \\;, &(\text{where } \bm{\pi} = (\pi\_i, \bm{\pi}\_{-i}))\\\
&= \sum\_{\bm{a}\in\mathcal{A}} \bm{\pi}(\bm{a})r^i(\bm{a}) \\;,\\\
&=  \sum\_{a\_i\in\mathcal{A}\_i}\sum\_{ \bm{a}\_{-i}\in\mathcal{A}\_{-i}} \bm{\pi}(a\_i, \bm{a}\_{-i})r^i(a\_i, \bm{a}\_{-i}) \\;,\\\
&\overset{(i)}{=}  \sum\_{a\_i\in\mathcal{A}\_i} \pi\_i(a\_i)\sum\_{ \bm{a}\_{-i}\in\mathcal{A}\_{-i}} \bm{\pi}\_{-i}(\bm{a}\_{-i})r^i(a\_i, \bm{a}\_{-i}) \\;, \\\
&\overset{(ii)}{=}  \sum\_{a\_i\in\mathcal{A}\_i} \pi\_i(a\_i) \tilde{r}^i(a\_i)\\;,
\end{aligned}
$$
where we used in $(i)$ that $\bm{\pi}(\bm{a}) = \prod\_{k=1}^n \pi\_k(a\_k)$ and defined in $(ii)$ the expected
reward associated to $a\_i$ when the other players keep their policies: $\tilde{r}^i(a\_i) = \mathbb{E}\_{\bm{a}\_{-i}\sim\bm{\pi}\_{-i}}[r^i(a\_i, \bm{a}\_{-i})] $.
Letting $a\_i^\star = \max\_{a\_i \in \mathcal{A}\_i}\tilde{r}^i(a\_i)$ we have that:
$$
\begin{aligned}
v\_i(a\_i^\star, \bm{\pi}\_{-i}) &= \tilde{r}^i(a\_i^\star) \\;, \\\
&\geq  \sum\_{a\_i\in\mathcal{A}\_i} \pi\_i(a\_i) \tilde{r}^i(a\_i)\\;, \\\
&= v\_i(\pi\_i, \bm{\pi}\_{-i})\\;,
\end{aligned}
$$
for every $\pi\_i\in\Pi\_i$. Therefore, the deterministic policy playing $a\_i^\star$ is a best-response to $\bm{\pi}\_{-i}$.
{{% /toggle_block %}}

Our second claim gives us a tool to identify some best-response stochastic policies. More precisely, it 
establishes that if some policy $\pi\_i$ cannot be improved by any deterministic policy, then it is a best-response. To formalise this, 
let us introduce the coefficients:
$$
\alpha\_{ij}(\bm{\pi}) = \max(0, v\_i(a\_j, \bm{\pi}\_{-i}) - v\_i(\bm{\pi})), \qquad j\in\\{1, \ldots, \vert \mathcal{A}\_i\vert\\}\\; .
$$

<ins>**Claim 2.**</ins> For every $i\in\\{1, \ldots n\\}$ and $\bm{\pi}=(\pi\_i, \bm{\pi}\_{-i})\in\bm{\Pi}$, if $\max\_{j\in\\{1, \ldots, n\\}} \alpha\_{ij}(\bm{\pi}) = 0$ then $\pi\_i \in\mathcal{T}\_i^\text{br}(\bm{\pi}\_{-i})$. 

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
This is a direct consequence of the first claim. Indeed, if $\max\_{j\in\\{1, \ldots, n\\}} \alpha\_{ij}(\bm{\pi}) = 0$ then
in particular:
$$v\_i(a\_i^\star, \bm{\pi}\_{-i}) - v\_i(\bm{\pi})=0\\; ,$$
where $a\_i^\star$ is the best-response action to $\bm{\pi}\_{-i}$ (see the proof of Claim 1). Therefore:
$$
v\_i(\pi\_i, \bm{\pi}\_{-i}) = v\_i(a\_i^\star, \bm{\pi}\_{-i}) = \max\_{\pi\_i'\in\Pi\_i}v\_i(\pi\_i', \bm{\pi}\_{-i})\\;,
$$
and $\pi\_i\in$ is a best-response to $\bm{\pi}\_{-i}$.
{{% /toggle_block %}}

We are now ready to claim and prove Nash's theorem.

{{< boxed title="Nash Theorem" >}}
$\qquad \qquad \qquad \quad \text{For every game with a finite amount of player ($n<\infty$), each with a with finite number of }$
$\text{ actions }
(\vert \mathcal{A}\vert < \infty)
\text{, there exists at least one Nash equilibrium.}$
{{< /boxed >}}

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof makes heavy use of the [Brouwer fixed-point theorem](https://en.wikipedia.org/wiki/Brouwer_fixed-point_theorem)--which we will not prove here.
Consider the following mapping:
$$
\begin{aligned}
f \\; : \\; \bm{\Pi} &\mapsto \bm{\Pi}\\\
\bm{\pi} &\mapsto \bm{\pi}' = f(\bm{\pi})
\end{aligned}
$$
where for any $i\in\\{1, \ldots, n\\}$ and $j\in\\{1, \ldots, \vert \mathcal{A}\_i\vert\\}$:
$$
\pi\_i'(a\_j) = \frac{\pi\_{i}(a\_j) + \alpha\_{ij}(\bm\pi)}{1 + \sum\_{k\in\\{1, \ldots, \vert \mathcal{A}\_i\vert\\}} \alpha\_{ik}(\bm\pi)}
$$
If you look up the definition of $\alpha\_{ij}(\bm\pi)$, you'll see that the function $f$ moves each marginal policy of $\bm\pi$
towards an improved version of themselves, assuming all others stay fixed. 
Clearly, $f$ is continuous, and its domain $\bm{\Pi}$ is a convex and compact set.
By Brouwer's fixed-point theorem, $f$ has therefore at least one fixed-point. Let $\bm{\pi}$ be one such fixed-point, that is, such that
$\bm\pi = f(\bm\pi)$.
Then, it is fairly easy to show that for every $i\in\\{1, \ldots, n\\}$ and $j\in\\{1, \ldots, \vert \mathcal{A}\_i\vert\\}$:
$$
\alpha\_{ij}(\bm\pi) = 0 \\; .
$$
(Simply establish a contradiction argument assuming that the above is false.) Simply applying our second claim yields that for every
$i\in\\{1, \ldots, n\\}$:
$$
    \pi\_i \in\mathcal{T}\_i^\text{br}(\bm{\pi}\_{-i}) \\;.
$$
Therefore, $\bm\pi$ is a Nash equilibrium. (One can further show that all Nash equilibria are fixed-point of $f$).
{{% /toggle_block %}}

A game can have several, distinct Nash equilibria--each with different expected returns for each agent. There exists
many variation to this solution concept (_e.g._, $\varepsilon$-Nash, correlated equilibrium) which we will not cover here.
Their point is to describe equilibria that are compatible with rounding errors, extra information, ..


## Maximin and Minimax
This section is concerned with two-players ($n=2$) zero-sum games.
The maximin and minimax concepts seamlessly generalise to other games, but we will keep the focus on two-player zero-sum games for simplicity.

- The _maximin_ strategy for some agent $i$ consists in the following rationale: follow a best-response policy, assuming that the other agent is acting
to minimize my return. In other words, play a policy that maximize agent $i$'s pay-off, assuming the other agent is out to 
cause him the greatest harm.
The maximin policy for agent $i$ is formally defined defined as:
$$
\bar{\pi}\_i \in \argmax\_{\pi_i} \min\_{\pi\_{-i}} v\_i(\pi\_i, \pi\_{-i})\\;,
$$
and the maximin value as $\max\_{\pi\_i} \min\_{\pi\_{-i}} v\_i(\pi\_i, \pi\_{-i})$.
The maximin value is also sometimes referred to as the _security level_. It is the minimum value
agent $i$ can guarantee, regardless of its opponent's strategy.

- The _minimax_ strategy sees things the other way around. When agent $i$ follows a minimax strategy, 
it abandons the maximisation of its own return in order to minimize its opponent's value. 
Formally, the minimax policy for agent $i$ is defined as:
$$
\pi\_i \in \argmin\_{\pi\_i} \max\_{\pi\_{-i}} v\_{-i}(\pi\_i, \pi\_{-i})\\;,
$$
and the minimax value as $\min\_{\pi_i} \max\_{\pi\_{-i}} v\_i(\pi\_i, \pi\_{-i})$.
The minimax value is the maximum return agent $i$ can force on its opponent.


A maximin (resp. minimax) solution concept is a joint policy $\bm\pi$ where each player follows a maximin
(resp. minimax) strategy.  In two-player zero-sum games, the two concepts blend with one another to yield the so-called
minimax solution.

{{< boxed title="Minimax solution" >}}
$\qquad \qquad \qquad \qquad \; \text{In a zero-sum game, a joint policy }\bm\pi=(\pi_1, \pi_2) \text{ is a minimax solution if:}$
$$
\begin{aligned}
    v_1(\bm\pi) &= \max_{\pi_1} \min_{\pi_2} v_1(\pi_1, \pi_2)\;, \\
&= \min_{\pi_1}\max_{\pi_2}v_1(\pi_1, \pi_2) \;, 
\\&= -v_2(\bm\pi) \; .
\end{aligned}
$$
{{< /boxed >}}

In a minimax solution, the minimax and maximin values of agent $i$ coincide.
Such a solution always exists in finite games. Below, we claim and prove the Minimax Theorem, 
which establishes this result for matrix games. The minimax solution concept has tight links with
the Nash equilibrium; actually, in a two-player zero-sum games, every Nash equilibrium is a minimax solution (see the proof).


{{< boxed title="Minimax Theorem" >}}
$\qquad \qquad \qquad \qquad \quad \text{Any finite zero-sum matrix game with two agents admits a minimax solution.}$
{{< /boxed >}}


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
By Nash's theorem, there exists at least one Nash equilibrium $\bm\pi=(\pi\_1, \pi\_2)$.
Let's denote $\bar{v}^1$ (resp. $\underline{v}^1$) the maximum (resp. minimax) value of $\bm\pi$
in the eyes of the first player. We will show that $v\_1(\bm\pi) = \bar{v}^1$; the proof of
$v\_1(\bm\pi) = \underline{v}^1$ is similar and omitted here. We will have that
$v\_1(\bm\pi) = \bar{v}^1=\underline{v}^1$, proving the announced result.

Proving that $v\_1(\bm\pi) \leq \bar{v}^1$ is done by simple manipulations:
$$
\begin{aligned}
v\_1(\bm\pi) &= -v\_2(\bm\pi) \\;, &(\text{zero-sum game})\\\
&= -\max\_{\pi\_2'}v\_2(\pi\_1, \pi\_2')\\;,  &(\text{Nash equilibrium})\\\
&=  -\max\_{\pi\_2'}-v\_1(\pi\_1, \pi\_2')\\;,  &(\text{zero-sum game})\\\
&=  \min\_{\pi\_2'}v\_1(\pi\_1, \pi\_2')\\;, \\\
&\leq \max\_{\pi\_1'}\min\_{\pi\_2'}v\_1(\pi\_1', \pi\_2') = \bar{v}^1\\;,
\end{aligned}
$$
Now, let's assume that $v\_1(\bm\pi)<\bar{v}^1$. This would contradict the fact that $\bm\pi$ is a 
Nash equilibrium. Denote $\bm\pi^\star = (\pi\_1^\star, \pi\_2^\star)$ the maximin policy. Then:
$$
\begin{aligned}
v\_1(\bm\pi)&<\bar{v}^1 \\;, \\\
&= v\_1(\pi\_1^\star, \pi\_2^\star)\\;, \\\
&= \min\_{\pi\_2'}v\_1(\pi\_1^\star, \pi\_2')\\;, \\\
&\leq v\_1(\pi\_1^\star, \pi\_2) \\;.
\end{aligned}
$$
Therefore, $v\_1(\pi\_1, \pi\_2) < v\_1(\pi\_1^\star, \pi\_2)$ which contradicts the fact that 
$\pi\_1$ is a best-response to $\pi\_2$ and that $\bm\pi$ is a Nash equilibrium. 
Therefore, we obtain as announced that $v\_1(\bm\pi)=\bar{v}^1$.


{{% /toggle_block %}}


## Resources
This blog post material was taken from [\[Shoham and  Leyton-Brown, Chapter 3\]](http://www.masfoundations.org/mas.pdf).
