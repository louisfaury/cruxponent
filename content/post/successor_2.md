+++
author = "Louis Faury"
title = "Successor States and Representations (2/3)"
date = "2025-05-03"
+++

In this second post of this series, we take a break from successor measures to focus on successor _features_.
We will first review the use of a generalised policy improvement mechanism 
that can efficiently leverage the successor features of existing policies to 
enable zero-shot transfer to new tasks.
We will discuss some generalisation to universal successor features approximations, 
allowing direct zero-shot control.

<!--more-->

{{< infoblock>}}
$\quad$ The first post of this series can be found
<a href="../successor_1" style="text-decoration:none; color:#0074aa;" ">here</a>.
{{< /infoblock >}}

We have introduced in an earlier post the successor measure $m\_\lambda^\pi$
of a stationary policy $\pi$, as $\forall s, s^\prime\in\mathcal{S}$:

$$
\tag{1}
m\_\lambda^\pi(s^\prime\vert s) = \sum\_{t\geq 1} \lambda^{t-1}\mathbb{P}^\pi(s\_t=s^\prime\vert s\_1=s)\\;.
$$
We focused our original discussion on the efficient estimation / learning of $m\_\lambda^\pi$
in a reward-free MDP,
and how it could be used for cheap evaluation of $\pi$ under any _a-posteriori_ specified reward.
We postponed any _control_ consideration; it should be, for now, rather blurry 
how successor measures really help us find optimal policies in a zero-shot manner.
This post aims at providing a smooth introduction to this question by focusing on the less general
(but still quite interesting!) successor features.

As usual, we will work with finite MDPs for simplicity.
We will denote $\mathcal{M} = (\mathcal{S}, \mathcal{A}, p)$ some
reward-free MDP, with $\mathrm{S}=\vert\mathcal{S}\vert$ and $\mathrm{A}=\vert\mathcal{A}\vert$.
It can be further specified by any reward function
$r\in\mathbb{R}^{\mathrm{S}\times\mathrm{A}}$. We will follow the usual
notations on this blog, and interchangeably go from index to
vectorial (in bold) notations.

## Successor Features


{{< infoblock>}}
$\quad$ Many results in this section are consequences of more general properties
inherited from successor measures.
For the sake of completeness, we will
re-derive them from first principles.
{{< /infoblock >}}

### Policy evaluation
Let $\phi : \mathcal{S}\times\mathcal{A}\mapsto\mathbb{R}^d$ 
some given feature map.
For any stationary policy $\pi$, we define
its successor features:
$$
\tag{2}
\psi\_\lambda^\pi(s, a) := \mathbb{E}\_{s, a}^\pi\Big[\sum\_{t\geq 1} \lambda^{t-1}\phi(s\_t, a\_t)\Big]\\;,
$$
for any state-action pair $(s, a)\in\mathcal{S}\times\mathcal{A}$.
Observe that successor features are only concerned by the dynamics of $\mathcal{M}$.
They become useful when considering rewards that are _linear_ in $\phi$.
Concretely, we will now consider reward signals $r\in\mathbb{R}^{\mathrm{S}\times\mathrm{A}}$ where
for any $(s, a)\in \mathcal{S}\times\mathcal{A}$:
$$
\tag{3}
    r(s, a) = \theta^\top\phi(s, a)\\;,
$$
for some $\theta\in\mathbb{R}^d$.
For any MDP $\mathcal{M}\_\theta$ yielded by the speciation of $\mathcal{M}$ with rewards parametrised by $\theta$,
there is a direct link between the values of policies and their successor features.

{{< boxed title="Successor features" >}}
$\qquad\qquad\qquad\qquad\quad 
\text{The value of any policy } \pi \text{ is linear in its successor features; for all }
(s, a)\in\mathcal{S}\times\mathcal{A}$:
$$
\tag{4}
q_\lambda^\pi(s, a)= \theta^\top \psi_\lambda^\pi(s, a)\;.
$$
{{< /boxed >}}

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
For any $(s, a)\in\mathcal{S}\times\mathcal{A}$:
$$
\begin{aligned}
q\_\lambda^\pi(s, a) &= \mathbb{E}\_{s, a}^\pi\left[\sum\_{t\geq 1}\lambda^{t-1} r(s\_t, a\_t)\right]\\;,\\\
&= \theta^\top \mathbb{E}\_{s, a}^\pi\left[\sum\_{t\geq 1}\lambda^{t-1} \phi(s\_t, a\_t)\right]\\;.\\\
&= \theta^\top \psi(s, a)\\;.
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

This has direct consequences for policy evaluation.
Successor features can
be used to evaluate $\pi$ given _any_ reward signal checking (3) without having
to experience said rewards—only at the cost of a scalar product.


### Bellman equations
Successor features check some Bellman-like equations.
Observe that successor features map $\mathcal{S}\times\mathcal{A}$ to $\mathbb{R}^d$, 
hence they belong to $\mathbb{R}^{d\times(\mathrm{S}\times\mathrm{A})}$.
Introduce the Bellman operator
$\mathcal{T}\_\lambda^{\pi, \phi} : \mathbb{R}^{d\times(\mathrm{S}\times\mathrm{A})} \mapsto \mathbb{R}^{d\times(\mathrm{S}\times\mathrm{A})}$
where for any function $\psi\in\mathbb{R}^{d\times(\mathrm{S}\times\mathrm{A})}$ and state-action pair $(s, a)$:
$$
\mathcal{T}\_\lambda^{\pi, \phi}(\psi)(s, a) = \phi(s, a) + \lambda \sum\_{s^\prime,a^\prime} p(s^\prime\vert s, a)\pi(a^\prime\vert s^\prime)\psi(s^\prime, a^\prime)\\;.
$$
This operator checks the usual properties of Bellman operator; it is contracting, and 
$\psi\_\lambda^\pi$ is a fixed-point.

{{< boxed title="Bellman equations" >}}
$\qquad\qquad\qquad\qquad\quad
\text{For any stationary policy }\pi,\text{ its successor features }
\psi_\lambda^\pi \text{ are the only fixed point of }\mathcal{T}_\lambda^{\pi, \phi}.$
$$
\tag{5}
\mathcal{T}_\lambda^{\pi, \phi}(\psi_\lambda^\pi)=\psi_\lambda^\pi\;.
$$
{{< /boxed >}}

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}

That $\psi\_\lambda^\pi$ is a fixed-point of $\mathcal{T}\_\lambda^{\pi, \phi}$
is straight-forward and left as an exercise here.
We will show that $\mathcal{T}\_\lambda^{\pi, \phi}$ is contracting under the $\ell\_\infty$ norm.
Observe that for any $\psi, \psi^\prime\in\mathbb{R}^{d\times(\mathrm{S}\times\mathrm{A})}$ and 
$s, a\in\mathcal{S}\times\mathcal{A}$:
$$
\begin{aligned}
\mathcal{T}\_\lambda^{\pi, \phi}(\psi)(s, a) - \mathcal{T}\_\lambda^{\pi, \phi}(\psi^\prime)(s, a)
&= \lambda \sum\_{s^\prime,a^\prime} p(s^\prime\vert s, a) \pi(a^\prime\vert s^\prime)(\psi(s^\prime, a^\prime) - \psi^\prime(s^\prime, a^\prime))
\end{aligned}
$$
which directly yields that $\\| \mathcal{T}\_\lambda^{\pi, \phi}(\psi) - \mathcal{T}\_\lambda^{\pi, \phi}(\psi^\prime)\\|\_\infty \leq 
\lambda \\| \psi-\psi^\prime\\|\_\infty$. 
Following the usual contracting operator logic, this proves that $\psi\_\lambda^\pi$
is the unique fixed-point of $\mathcal{T}\_\lambda^{\pi, \phi}$.

<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

In short, the implications are that successor features can be learned via classical
dynamic programming mechanisms. When coupled with stochastic approximation (sampling) and 
function approximations (say, neural networks)
all the usual tricks (Bellman error minimisation, replay buffer, target networks, etc.) are valid.
In deep reinforcement learning terms, it is safe to think of learning successor features to be roughly
similar as implementing one of the many variants of DQN (except here for policy evaluation, not control).


## Transfer
Below, we follow {{< ref link="features">}} [1]{{< /ref>}} and apply successor features
to the transfer problem. Concretely, we assume access to some pre-trained policies
$\\{\pi\_1, \ldots, \pi\_n\\}$. For instance, each can be a (near-)optimal policy for
the reward $r\_i(s, a) = \theta\_i^\top\phi(s, a)$.
As we emphasized earlier, evaluating each $\pi\_i$ on $\mathcal{M}\_\theta$ is immediate:
$$
q\_\lambda^{\pi\_i}(s, a) = \theta^\top \psi\_\lambda^{\pi\_i}(s, a).
$$
A naive (but effective) way to use this for our advantage is to perform this evaluation
at the starting state and greedily follow to the highest valued policy.
The authors of {{< ref link="features">}} [1]{{< /ref>}} propose to do better, 
by extending policy improvement to several base policies; concretely, to form $\pi$ such that for all $s\in\mathcal{S}$:
$$
\tag{6}
\pi(s) \in \argmax_{a\in\mathcal{A}}\max\_{i=1, \ldots, n} \theta^\top \psi\_\lambda^{\pi\_i}(s, a)\\;.
$$
This guarantees that $\pi$ dominates any of the $\pi\_i$ on $\mathcal{M}\_\theta$.
If the base policies $\\{\pi\_i\\}\_i$ can be indeed "stitched" together to yield 
a performant policy on $\mathcal{M}\_\theta$, transfer will be succesful.


{{< boxed title="Transfer" >}}
$\qquad\qquad\;
\text{Let } \pi\text{ given by (6)}. \text{ Then, for any }i\in\{1, \ldots, n\} \text{ and }s, a\in\mathcal{S}\times\mathcal{A}:$
$$
q_\lambda^{\pi}(s, a) \geq q_\lambda^{\pi_i}(s, a)\;.
$$
{{< /boxed >}}


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Observe that by definition of $\pi$, for any $a\in\mathcal{A}$:
$$
\tag{a}
\max\_{i, \ldots, n} q\_\lambda^{\pi\_i}(s, \pi(s)) \geq \max\_{i, \ldots, n} q\_\lambda^{\pi\_i}(s, a) \\;.
$$
For simplicity, we will assume each $\pi\_i$ to be deterministic--which checks out if they are each
optimal policy for some $\theta\_i$. For any $s, a\in\mathcal{S}\times\mathcal{A}$:
$$
\begin{aligned}
\max\_{i\in\\{1, \ldots, n\\}}q\_\lambda^{\pi\_i}(s, a) &= \max\_{i\in\\{1, \ldots, n\\}}
\left\\{ r(s, a) + \lambda\sum\_{s^\prime}p(s^\prime\vert s, a)q\_\lambda^{\pi\_i}\left(s^\prime, \pi\_i(s^\prime)\right)\right\\}\\;, &(q\_\lambda^{\pi\_i} = \mathcal{T}\_\lambda^{\pi\_i}q\_\lambda^{\pi\_i}) \\\
&\leq r(s, a) + \lambda\sum\_{s^\prime}p(s^\prime\vert s, a)\max\_{i\in\\{1, \ldots, n\\}}q\_\lambda^{\pi\_i}\left(s^\prime, \pi\_i(s^\prime)\right)\\;, \\\
&\leq r(s, a) +  \lambda\sum\_{s^\prime}p(s^\prime\vert s, a)\max\_{i\in\\{1, \ldots, n\\}}q\_\lambda^{\pi\_i}\left(s^\prime, \pi(s^\prime)\right)\\;, &(\text{by (a)})
\end{aligned}
$$
We obtain the desired claim by repeating over the last line;
$$
\begin{aligned}
\max\_{i\in\\{1, \ldots, n\\}}q\_\lambda^{\pi\_i}(s, a) &\leq r(s, a) +  \lambda\sum\_{s^\prime}p(s^\prime\vert s, a)\max\_{i\in\\{1, \ldots, n\\}}q\_\lambda^{\pi\_i}\left(s^\prime, \pi(s^\prime)\right)\\;,\\\
&\leq r(s, a) +  \lambda\sum\_{s^\prime}p(s^\prime\vert s, a) \left(r(s^\prime, \pi(s^\prime)) + \lambda \sum\_{s^{\prime\prime}} p(s^{\prime\prime}\vert s^\prime, \pi(s^\prime))\max\_{i\in\\{1, \ldots, n\\}}q\_\lambda^{\pi\_i}(s^{\prime\prime}, \pi(s^{\prime\prime}))\right)\\;,\\\
&\leq \ldots\\;,\\\
&\leq \mathbb{E}\_s^\pi\left[\sum\_{t\geq 1}\lambda^{t-1}r(s\_t, a\_t)\right]\\;,\\\
&= q\_\lambda^\pi(s, a)\\;.
\end{aligned}
$$

<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}


## Universal SFs
The previous section leverages successor features' fast policy evaluation, which
indirectly unlocks fast (multi-) policy improvement and, as a result, transfer.
In this section, we follow {{< ref link="universal">}} [2]{{< /ref>}} and rather
focus on how we can directly train some zero-shot capabilities.
We'd be more excited to directly learn to predict:
$$
\psi\_\lambda^{\pi^\star}(s, a) = \mathbb{E}\_{s, a}^{\pi^\star}\left[ \sum\_{t\geq 1}\lambda^{t-1}\phi(s\_t, a\_t)\right]\\;,
$$
the successor feature of the optimal policy $\pi^\star$, and _without_ any interaction with $\mathcal{M}\_\theta$, of course.
Doing so will require learning on a variety of _tasks_ (several values of $\theta$),
so we can hope to generalise zero-shot to new ones.
Below, we denote $\pi\_\theta^\star$ the optimal value for the MDP $\mathcal{M}\_\theta$.
Let the universal successor feature be:
$$
\begin{aligned}
\psi\\, : \\,\mathcal{S}\times\mathcal{A}\times\mathbb{R}^d &\mapsto \mathbb{R}^d \\;,\\\
(s, a, \theta) &\mapsto \psi\_\lambda^{\pi\_\theta^\star}(s, a)\\;.
\end{aligned}
$$

It also checks a Bellman equation. Indeed, for all $s, a\in\mathcal{S}\times\mathcal{A}$, and for all $\theta\in\mathbb{R}^d$

$$
\psi(s, a, \theta) = \phi(s, a) + \sum\_{s^\prime}p(s^\prime\vert s, a)\psi(s^\prime, \argmax_{a^\prime}\theta^\top\psi(s^\prime, a^\prime, \theta), \theta)\\;.
$$
Finally, observe that the optimal policy $\pi\_\theta^\star$ for $\mathcal{M}\_\theta$ can readily be retrieved from 
the universal successor feature; because of its ties to state-action value function, we have that 
$
\pi\_\theta^\star(s) \in \argmax\_{a\in\mathcal{A}} \theta^\top \psi(s, a, \theta)\\;
$.

{{% toggle_block background-color="#CBE4FE" title="About universal value functions" default-display="none"%}}
At this point, the proximity to universal value function approximators {{< ref link="uvfa">}} [2]{{< /ref>}} is quite striking.
As will soon become quite clear (hopefully), an advantage of universal SF approximators comes from the
decoupling of reward and dynamics, as it allows learning for many values of $\theta$ from a single trajectory, 
without synthetizing different rewards.
{{% /toggle_block %}}

We provide below some pseudo-code to illustrate the learning of the universal successor features. 
Using standard "tricks" (replay buffer, target networks, etc.) is recommended, but skipped here for clarity.
The point is to illustrate the main features that arise when training $\psi$.
This naive algorithm iterates through tasks by sampling $\theta\in\mathbb{R}^d$. 
Because we decoupled dynamics from rewards, one can train $\psi$ for several values of 
$\theta$ _even_ when following a single task!
We actually use the sampled task only to guide our exploration policy, as we will pick up
action in an (almost) greedy fashion w.r.t $\theta^\top\psi(s, a, \theta)$--a.k.a the
(approximated) q-values of the optimal policy for the MDP parametrised by $\theta$.


{{< pseudocode title="$\texttt{Learning universal SFs}$" >}} 
$\textbf{init } \text{learning rate } \alpha, \text{ number of tasks to train } K, \text{ spread } \varepsilon, \text{ episode length } T\\$
$\textbf{while } \text{has not converged}\\$
$\quad \text{ sample task }\theta\in\mathbb{R}^d\\$
$\quad \text{ sample which policies to train for } \theta_1, \ldots, \theta_n\overset{\text{i.i.d}}{\sim}\mathcal{N}(\theta, \varepsilon \mathbf{I}_d)\\$
$\quad\text{\color{CadetBlue}\# collect episode}\\$
$\quad\textbf{for } t=1,\ldots, T\\$
$\quad\quad\text{observe } s_t\\$
$\quad\quad\text{\color{CadetBlue}\# $\varepsilon$-greedy exploration policy with (6)}\\$
$\quad\quad\text{pick } a_t \leftarrow \argmax_a\max_{k=1, \ldots, K} \theta^\top \psi(s, a, \theta_k) \text{ with proba } 1-\varepsilon, \text{ else } a_t\leftarrow \mathcal{U}(\mathcal{A})\\$
$\quad\quad\text{observe } s_{t+1}\\$
$\quad\quad\textbf{for } k=1,\ldots, K\\$
$\quad\quad\quad\text{\color{CadetBlue}\# form the regression targets}\\$
$$
t_k \leftarrow \phi(s_t, a_t) + \lambda\psi(s_{t+1}, \argmax_a \theta_k^\top \psi(s_{t+1}, a, \theta_k), \theta_k)\;.
$$
$\quad\quad\textbf{end for}\\$
$\quad\quad\text{\color{CadetBlue}\# gradient step}\\$
$$
\psi \leftarrow \psi - \alpha \nabla_\psi \sum_{k=1}^K \left\|\psi(s_t, a_t, \theta_k) - t_k\right\|_2^2\;.
$$
$\quad\textbf{end for}\\$
$\textbf{end while}\\$
$\textbf{return } \psi$
{{< /pseudocode >}}



## Feature selection

So far, we have assumed that the feature map $\phi$ and the reward vector $\theta$ is known.
The former assumption is checked when state featurisation is "easy"; _e.g._ by building event detectors
for picking up a relevant object, moving into a speficif direction, etc. Once we have a 
feature map, estimating $\theta$ for a given reward signal can anively be achieved by regression on some buffered data.

However, in general, designing $\phi$ from scratch is arguably hard.
It is desirable to extract it directly from data, _e.g._ some trajectories collected arbitrarily,
or via some pre-trained policies.
Proving a comprehensive list of self-supervised criterion to learn $\phi$ is beyond the scope
of this post--also, such a list can be found directly in {{< ref link="touati">}} [4]{{< /ref>}}.
There, successor features are compared to "forward-backward" representations, which we studied in the 
[first post](../successor_1) of this series. The link between the two approaches is the next post's topic!


## References

<div id="features"></div> [1] Successor Features for Transfer in Reinforcement Learning, Barreto et al. 2016.
<div id="universal"></div> [2] Universal Successor Features Approximators, Borsa et al. 2019.
<div id="uvfa"></div> [3] Universal Value Function Approximators, Schaul et al. 2015.
<div id="touati"></div> [4] Does Zero-Shot Reinforcement Learning Exist? Touati et al. 2023.

<br>
<br>
and references from the previous post of this series.