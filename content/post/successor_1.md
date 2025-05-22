+++
author = "Louis Faury"
title = "Successor States and Representations (1/3)"
date = "2025-02-24"
+++

The main promise of unsupervised RL is test-time adaptation to newly specified reward functions.
This requires a systemic untangling of reward and dynamics
in traditional RL tools.
In this first post of a short series, we see how this can be done via the concept 
of successor states and successor representations. 
We will focus on policy evaluation, leaving control for a follow-up.

<!--more-->

One over-arching theme in unsupervised RL is the identification of a device
that compactly summarises a dynamical system. 
It is to be used downstream for generating a near-optimal policy,
but for rewards that are _a-posteriori_ specified. 
It can be argued that said device can simply be a model of
the dynamical system (although such models are seldom compact).
In this post, and in contrast with model-based approaches, 
we will focus on test-time adaptation to new rewards
_without_ any state synthesis nor explicit planning. 


## Preliminaries

{{< warningblock>}}
$\quad$ We will heavily use the MDP notations defined in <a href="../mdp_basics" style="text-decoration:none; color:#0074aa;" ">this post</a>.
{{< /warningblock >}}

We consider a stationary Markov Decision Process $\mathcal{M}=(\mathcal{S}, \mathcal{A}, p, r)$.
We will assume that $\mathcal{M}$ is finite, _i.e_
that $\mathrm{S}\times\mathrm{A}<\infty$
where $\mathrm{S} = \vert \mathcal{S} \vert$ and $\mathrm{A} = \vert \mathcal{A} \vert$.
We solve $\mathcal{M}$ under a discounted reward criterion; for some $\lambda\in[0, 1)$:
$$
v\_\lambda^\pi(s) := \mathbb{E}\_s^\pi\left[\sum\_{t\geq 1}\lambda^{t-1}r(s\_t, a\_t)\right]\\;.
$$

Throughout this post, we will focus on policy evaluation more than control.
Hence, we index everything under some stationary policy $\pi\in\mathcal{S}^\text{MR}$
at hand to focus on the specific induced Markov reward process. 
For instance, we will use $r\_\pi(s) := \sum\_{a\in\mathcal{A}} r(s, a)\pi(a\vert s)$.
and $p\_\pi(s^\prime\vert s) := \sum\_{a\in\mathcal{A}}p(s^\prime\vert s, a)\pi(a\vert s)$
the expected reward and transition under $\pi$, respectively.
Given the MDP's finite nature, we will make heavy use of vectorial notations. 
Formally, for any $f:\mathcal{S}\mapsto\mathbb{R}$ and $F:\mathcal{S}\times\mathcal{S}\mapsto\mathbb{R}$
we use $\mathbf{f}\in\mathbb{R}^\mathrm{S}$ and $\mathbf{F}\in\mathbb{R}^{\mathrm{S}\times\mathrm{S}}$, respectively, 
for the associated vector notations. For instance, for any $s, s^\prime\in\mathcal{S}$ we have
$[\mathbf{r}\_\pi]\_s=r\_\pi(s)$ and $[\mathbf{P}\_\pi]\_{ss^\prime}=p\_\pi(s^\prime\vert s)$.
In a [previous post](/post/mdp_basics/) we showed that this allowed for the compact identity:
$$
\tag{1}
\mathbf{v}\_\lambda^\pi = \sum\_{t\geq 1} \lambda^{t-1}\mathbf{P}\_\pi^{t-1} \mathbf{r}\_\pi,
$$


{{< infoblock>}}
$\quad$ The finiteness assumption is made for building intuition.
We will relax it by the end of this post.
{{< /infoblock >}}


## Successor states

### Successor measure
To compactly represent the dynamics $p$ of $\mathcal{M}$, we first need to disentangle
it from the rewards in the value function $v\_\lambda^\pi$. 
This can be done via the introduction of the discounted state occupancy measure, which represents
the cumulative discounted time spent in some state when starting from another; $\forall s, s^\prime\in\mathcal{S}$:
$$
m\_\lambda^\pi(s^\prime\vert s) := \sum\_{t\geq 1} \lambda^{t-1} \mathbb{P}^\pi(s\_t=s^\prime \vert s\_1 = s)\\;.
$$
The value function indeed writes as $v\_\lambda^\pi(s) = \sum\_{s^\prime\in\mathcal{S}} m\_\lambda^\pi(s^\prime\vert s)r_\pi(s^\prime)$.
The discounted state occupancy measure $m\_\lambda^\pi$ captures the dynamics 
of the stochastic process driven by $\pi$, while the reward function is isolated in $r\_\pi$.
{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Observe that for any $s\in\mathcal{S}$:
$$
\begin{aligned}
v\_\lambda^\pi(s) &:= \mathbb{E}\_s^\pi\left[\sum\_{t\geq 1}\lambda^{t-1}r(s\_t, a\_t)\right]\\;, \\\
&=  \mathbb{E}\_s^\pi\left[\sum\_{t\geq 1}\sum\_{s^\prime\in\mathcal{S}}\sum\_{a^\prime\in\mathcal{A}}\lambda^{t-1}r(s^\prime, a^\prime)\mathbf{1}[s\_t=s^\prime, a\_t=a^\prime]\right]\\;,\\\
&= \sum\_{s^\prime\in\mathcal{S}}\sum\_{a^\prime\in\mathcal{A}}\sum\_{t\geq 1}\lambda^{t-1}r(s^\prime, a^\prime)\mathbb{E}\_s^\pi\left[\mathbf{1}[s\_t=s^\prime, a\_t=a^\prime]\right]\\;,\\\
&= \sum\_{s^\prime\in\mathcal{S}}\sum\_{a^\prime\in\mathcal{A}}\sum\_{t\geq 1}\lambda^{t-1}r(s^\prime, a^\prime)\mathbb{P}^\pi\left(s\_t=s^\prime, a\_t=a^\prime\middle\vert s\_1=s\right)\\;,\\\
&= \sum\_{s^\prime\in\mathcal{S}}\sum\_{a^\prime\in\mathcal{A}}\sum\_{t\geq 1}\lambda^{t-1}r(s^\prime, a^\prime)\mathbb{P}\_s^\pi\left(s\_t=s^\prime\middle\vert s\_1=s\right)\pi(a^\prime\vert s)\\;,\\\
&= \sum\_{s^\prime\in\mathcal{S}}\sum\_{t\geq 1}\lambda^{t-1}\mathbb{P}\_s^\pi\left(s\_t=s^\prime\middle\vert s\_1=s\right)r\_\pi(s^\prime)\\;,\\\
&= \sum\_{s^\prime\in\mathcal{S}} m\_\lambda^\pi(s^\prime\vert s)r_\pi(s^\prime)\\;.
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

This result is actually immediate when glancing at (1) -- it is exactly what was proved above. 
Introducing the matrix version of $m\_\lambda^\pi$:
$$
\tag{2}
\mathbf{M}\_\lambda^\pi := \sum\_{t\geq 1} \lambda^{t-1}\mathbf{P}\_\pi^{t-1}\\;,
$$
which checks $[\mathbf{M}\_\lambda^\pi]_{ss^\prime}=m\_\lambda^\pi(s^\prime\vert s)$ for any $s, s^\prime\in\mathcal{S}$, we have 
$\mathbf{v}\_\lambda^\pi = \mathbf{M}\_\lambda^\pi \mathbf{r}\_\pi$.
In others words, policy evaluation is as simple as materialising the matrix $\mathbf{M}\_\lambda^\pi$ 
and the reward vector $\mathbf{r}\_\pi$, and going through a simple matrix vector multiplication. 
There is nothing very deep about this; we already know that policy evaluation in a finite MDP
is not much more than solving a linear system, since
$
\mathbf{v}\_\lambda^\pi = (\mathbf{I}\_{d} - \lambda\mathbf{P}\_\pi)^{-1} \mathbf{r}\_\pi\\;.
$

Below, we will ignore the triviality of policy evaluation in finite MDPs
and stick with our finiteness assumption to gain some intuition. The identity:
$$
\mathbf{v}\_\lambda^\pi = \mathbf{M}\_\lambda^\pi \mathbf{r}\_\pi
$$
has an interesting ring to it: given $\mathbf{M}\_\lambda^\pi$, 
we can easily evaluate $\pi$ on the fly, for any new reward function, and via a simple matrix-vector product.
That is a promising feature on our test-time reward specification trip!


### Successor features
In a similar fashion, some work {{< ref link="features">}} [1]{{< /ref>}} aim to represent sucessor _features_
rather than states.
Concretely, let's assume that it exists $\phi:\mathbb{R}^{\mathrm{S}} \mapsto \mathbb{R}^d$ and $\theta\in\mathbb{R}^d$
such that for all $s, a\in\mathcal{S}\times\mathcal{A}$ we have $r(s\_t, a\_t) = r(s\_t) = \theta^\top\phi(s\_t)$.
One can therefore re-write the value function at $s\in\mathcal{S}$ of any stationary $\pi$ as:
$$
\begin{aligned}
v\_\lambda^\pi(s) &= \mathbb{E}\_s^\pi\left[\sum\_{t=1}^T \lambda^{t-1}\phi(s\_t)\right]^\top \theta\\; , \\\
\end{aligned}
$$
or, equivalently,
$
\mathbf{v}\_\lambda ^\pi = \Phi\_\lambda^\pi\theta
$
where $[\Phi\_\lambda^\pi]\_s=\sum\_{s^\prime\in\mathcal{S}} \phi(s^\prime) \sum\_{t=1}^t \lambda^{t-1}\mathbb{P}(s\_t=s^\prime\vert s\_1=s)$
 represents the discounted mean feature occupied by the stochastic process starting at $s$. 
Observe how the state representation supersedes this approach, as the discounted mean feature can be simply
retrieved via matrix-matrix multiplication; for any $s\in\mathcal{S}$ we have
$
\Phi\_\lambda^\pi = \mathbf{M}\_\lambda^\pi \Phi
$
where $\Phi = (\phi(s\_1), \phi(s\_2), \ldots)^\top$. As a summary, we'll retrieve a successor feature representation via:
$$
\mathbf{v}\_\lambda ^\pi  =  \mathbf{M}\_\lambda^\pi \Phi\theta\\; .
$$

## Learning

### Bellman equations

Let's forget for a moment the obvious equality $\mathbf{M}\_\lambda^\pi=(\mathbf{I}\_{d} - \lambda\mathbf{P}\_\pi)^{-1}$
and consider how we could characterise the discounted occupancy measure $\mathbf{M}\_\lambda^\pi$ in a fashion that 
would allow for some efficient learning.
That could be useful for large state-spaces, where matrix inversion would turn out too expensive.
We clearly have in mind finding some Bellman-like equations for $\mathbf{M}\_\lambda^\pi$. 
Turns out, such equations exist and are quite similar to the ones we write for value functions:

$$
\tag{3}
\mathbf{M}\_\lambda^\pi = \mathbf{I}_d + \lambda \mathbf{P}\_\pi \mathbf{M}\_\lambda^\pi\\; .
$$

In state-indexed notations, this would write that for any $s\in\mathcal{S}$:
$$
\tag{4}
m\_\lambda^\pi(s^\prime\vert s) = 1[s=s^\prime] + \lambda \sum\_{s^{\prime\prime}\in\mathcal{S}}m(s^\prime\vert s^{\prime\prime})p(s^{\prime\prime}\vert s)\\; .
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
We will prove the vector notation result, which implies the state-indexed one.
(The latter can be retrieved on its own using the standard techniques.)
$$
\begin{aligned}
\mathbf{M}\_\lambda^\pi &= \sum\_{t\geq 1} \lambda^{t-1}\mathbf{P}\_\pi^t\\;,\\\
&= \mathbf{I}_d + \sum\_{t\geq 2} \lambda^{t-1}\mathbf{P}\_\pi^{t-1}\\;,\\\
&= \mathbf{I}_d + \lambda\mathbf{P}\_\pi\sum\_{t\geq 2} \lambda^{t-2}\mathbf{P}\_\pi^{t-2}\\;,\\\
&= \mathbf{I}_d + \lambda\mathbf{P}\_\pi\sum\_{t\geq 1} \lambda^{t-1}\mathbf{P}\_\pi^{t-1}\\;,\\\
&= \mathbf{I}_d + \lambda \mathbf{P}\_\pi \mathbf{M}\_\lambda^\pi\\; .
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

We can rewrite (3) using the forward successor Bellman operator:
$$
\begin{aligned}
\mathcal{T}\_\lambda^{\pi, \rightarrow} : \mathbb{R}^{\mathrm{S}\times\mathrm{S}} &\mapsto \mathbb{R}^{\mathrm{S}\times\mathrm{S}}\\;,\\\
 \mathbf{M} &\mapsto \mathbf{I}\_d + \lambda \mathbf{P}\_\pi\mathbf{M} \\;.
\end{aligned}
$$
$\mathbf{M}\_\lambda^\pi$ is a fixed-point of $\mathcal{T}\_\lambda^{\pi, \rightarrow}$; we have $\mathcal{T}\_\lambda^{\pi, \rightarrow}(\mathbf{M}\_\lambda^\pi)=\mathbf{M}\_\lambda^\pi$. 
Ressorting to the now usual dance, we can guarantee that it is actually the _unique_ fixed-point, thanks to 
the contractive property of $\mathcal{T}\_\lambda^{\pi, \rightarrow}$ with respect to the operator norm $|||\cdot|||\_\infty$
(where $|||\mathbf{M}|||\_\infty = \sup\_{\\|\mathbf{v}\\|\_\infty=1} \\|\mathbf{M}\mathbf{v}\\|\_\infty)$.
This property also opens the path for learning $\mathbf{M}\_\lambda^\pi$ via 
a fixed-point iteration algorithm.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
For any $\mathbf{M}, \mathbf{M}^\prime\in\mathbb{R}^{\mathrm{S}\times\mathrm{S}}$ we have:
$$
\begin{aligned}
\|||\mathcal{T}\_\lambda^{\pi, \rightarrow}(\mathbf{M}) - \mathcal{T}\_\lambda^{\pi, \rightarrow}(\mathbf{M^\prime})\|||\_\infty
&= \lambda \||| \mathbf{P}\_\pi(\mathbf{M}-\mathbf{M}^\prime) \|||\_\infty \\;,\\\
&\leq  \lambda \||| \mathbf{P}\_\pi \|||\_\infty \||| \mathbf{M}-\mathbf{M}^\prime \|||\_\infty \\;, &(\text{operator norm is sub-multiplicative})\\\
&\leq \lambda \|||\mathbf{M}-\mathbf{M}^\prime \|||\_\infty \\;, &( \||| \mathbf{P}\_\pi \|||\_\infty\leq 1)\\\
\end{aligned}
$$
which proves the claim. The last line uses a bound on the operator norm of $\mathbf{P}\_\pi$, 
which is essentially a consequence of $\mathbf{P}\_\pi$ being a right-stochastic matrix:
$$
\begin{aligned}
\||| \mathbf{P}\_\pi \|||\_\infty &= \sup\_{\\|\mathbf{v}\\|\_\infty=1} \\|\mathbf{P}\_\pi\mathbf{v}\\|\_\infty\\;,\\\
&=  \sup\_{\\|\mathbf{v}\\|\_\infty=1} \sup\_{s\in\mathcal{S}}\left\vert\sum\_{s^\prime\in\mathcal{S}} \mathbb{P}^\pi(s\_t=s^\prime\vert s\_1=s)v(s^\prime)\right\vert\\;,\\\
&=  \sup\_{\\|\mathbf{v}\\|\_\infty=1} \sup\_{s\in\mathcal{S}}\sum\_{s^\prime\in\mathcal{S}} \mathbb{P}^\pi(s\_t=s^\prime\vert s\_1=s)\vert v(s^\prime)\vert\\;,\\\
&\leq \sup\_{\\|\mathbf{v}\\|\_\infty=1} \||\mathbf{v}\||\_\infty \sup\_{s\in\mathcal{S}}\sum\_{s^\prime\in\mathcal{S}} \mathbb{P}^\pi(s\_t=s^\prime\vert s\_1=s)\\;,\\\
&= \sup\_{\\|\mathbf{v}\||\_\infty=1} \||\mathbf{v}\||\_\infty\\;, &(\sum\_{s^\prime\in\mathcal{S}} \mathbb{P}^\pi(s\_t=s^\prime\vert s\_1=s)=1 \\; \forall s\in\mathcal{S})\\\
&= 1\\;.
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
The state-indexed Bellman equations (4) for the occupancy measure echoes the classical
Bellman equations we write for policy evaluation; for all $s\in\mathcal{S}$:
$$
v\_\lambda^\pi(s) = r\_\pi(s) + \lambda \sum\_{s^\prime\in\mathcal{S}} p(s^\prime\vert s)v\_\lambda^\pi(s^\prime)\\; .
$$
We can see that for every $s\in\mathcal{S}$, the identity (4) can be rewritten as a policy evaluation equation,
but under a reward $r\_\pi(s) = 1[s=s^\prime]$.
In other words, $m\_\lambda^\pi(s^\prime\vert s)$ can be identified with the value of $\pi$ under a goal-conditioned 
reward which is only non-zero at $s^\prime$.
{{% /toggle_block %}}

It is clear from (2) that $\mathbf{M}\_\lambda^\pi$ and $\mathbf{P}\_\pi$ commute.
Looking at the forward successor Bellman operator, we can immediately deduce that
$\mathbf{M}\_\lambda^\pi$ is also a fixed-point of a _backward_ Bellman operator: 
$$
\begin{aligned}
\mathcal{T}\_\lambda^{\pi, \leftarrow} : \mathbb{R}^{\mathrm{S}\times\mathrm{S}} &\mapsto \mathbb{R}^{\mathrm{S}\times\mathrm{S}}\\;,\\\
 \mathbf{M} &\mapsto \mathbf{I}\_d + \lambda \mathbf{M}\mathbf{P}\_\pi \\;.
\end{aligned}
$$
In other words $
\mathbf{M}\_\lambda^\pi = \mathbf{I}_d + \lambda  \mathbf{M}\_\lambda^\pi\mathbf{P}\_\pi\\; .
$
Similarly to the forward case, we can show that the backward successor Bellman operator $\mathcal{T}\_\lambda^{\pi, \leftarrow}$
contracts under $\|||\cdot\|||\_\infty$.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
For any $\mathbf{M}, \mathbf{M}^\prime\in\mathbb{R}^{\mathrm{S}\times\mathrm{S}}$ we have:
$$
\begin{aligned}
\|||\mathcal{T}\_\lambda^{\pi, \leftarrow}(\mathbf{M}) - \mathcal{T}\_\lambda^{\pi, \leftarrow}(\mathbf{M^\prime})\|||\_\infty
&= \lambda \||| (\mathbf{M}-\mathbf{M}^\prime)\mathbf{P}\_\pi \|||\_\infty \\;,\\\
&\leq  \lambda \||| \mathbf{M}-\mathbf{M}^\prime \|||\_\infty \||| \mathbf{P}\_\pi \|||\_\infty \\;, &(\text{operator norm is sub-multiplicative})\\\
&\leq \lambda \|||\mathbf{M}-\mathbf{M}^\prime \|||\_\infty \\;. &( \||| \mathbf{P}\_\pi \|||\_\infty\leq 1)\\\
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}} 

{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
Unlike in the forward case, the backward Bellman operator does not have any straight-forward
ties to the "traditional" Bellman policy evaluation operator for value functions.
{{% /toggle_block %}}

### Fixed-point iterates
The fixed-point characterisations of $\mathbf{M}\_\lambda^\pi$ via contractive maps
allow up to write fixed-points iterations for computing $\mathbf{M}\_\lambda^\pi$.
Concretely, we can use two different protocols, relying on 
$\mathcal{T}\_\lambda^{\pi, \rightarrow}$ and $\mathcal{T}\_\lambda^{\pi, \leftarrow}$, respectively:
$$
\begin{aligned}
\mathbf{M}\_{k+1}^{\rightarrow} &= \mathcal{T}\_\lambda^{\pi, \rightarrow}(\mathbf{M}\_k^{\rightarrow})=\mathbf{I}_d + \lambda\mathbf{P}\_\pi \mathbf{M}\_k^{\rightarrow}\\;, \\\
\mathbf{M}\_{k+1}^{\leftarrow} &= \mathcal{T}\_\lambda^{\pi, \leftarrow}(\mathbf{M}\_k^{\leftarrow})=\mathbf{I}_d + \lambda\mathbf{M}\_k^{\leftarrow}\mathbf{P}\_\pi \\;. \\\
\end{aligned}
$$
By the contractive nature of each operator, both $\\{\mathbf{M}\_{k}^{\rightarrow}\\}\_k$ and 
$\\{\mathbf{M}\_{k}^{\leftarrow}\\}\_k$ converge to $\mathbf{M}\_\lambda^\pi$.


{{< infoblock>}}
$\quad$ The inherent value of those updates in the finite case is limited. They, however,
provide the necessary ground for stochastic approximation variants, which will be useful in continuous conditions.
{{< /infoblock >}}

One can alternate between $\mathcal{T}\_\lambda^{\pi, \rightarrow}$ and $\mathcal{T}\_\lambda^{\pi, \leftarrow}$
during fixed-point iterations. 
This has been shown to accelerate convergence of $\mathbf{M}\_\lambda^\pi$ {{< ref link="states">}} [2]{{< /ref>}} 
by reducing the dimension of the subspace where its convergence is at its slowest.

## Factorisation

### Forward-Backward
Learning $\mathbf{M}\_\lambda^\pi$ involves estimating $\mathrm{S}^2$ parameters, and in its current form,
does not allow for much interpretation or generalisation across states.
As a first step to alleviate those issues, we can learn a _factorised_ version of $\mathbf{M}\_\lambda^\pi$.
Concretely, we assume a low-rank structure for the state occupancy matrix;
let $\mathbf{F}\_\lambda^\pi$ and $\mathbf{B}\_\lambda^\pi \in\mathbb{R}^{d\times \mathrm{S}}$ with $d<\mathrm{S}$
such that:
$$
\tag{5}
\mathbf{M}\_\lambda^\pi = (\mathbf{F}\_\lambda^\pi)^\top \mathbf{B}\_\lambda^\pi\\; . 
$$

We can interpret $\mathbf{F}\_\lambda^\pi$ and $\mathbf{B}\_\lambda^\pi$ as mappings
from $\mathcal{S}$ to $\mathbb{R}^d$, effectively embedding each state into a more compact representation. 
Each entry in $\mathbf{M}\_\lambda^\pi$ is now obtained by the scalar products between the columns of the forward 
$\mathbf{F}\_\lambda^\pi$ and backward $\mathbf{B}\_\lambda^\pi$ matrices.
Beyond providing us with tools to compactly represent states, this paves the way for using successor states in a control
fashion (for a later post).

The forward and backward matrices laid out in (5) also check some fixed-point equations (by simply replacing them
in $\mathbf{M}\_\lambda^\pi$'s fixed-point characterisation).
However, hoping to approximate them via fixed-point iterations is somewhat futile (for instance, it is clear
that their definition via (5) is not unique).
Below, we let go of theoretical guarantees and follow {{< ref link="states">}} [2]{{< /ref>}}; we will instead
illustrate how to compute _gradient_ updates for $\mathbf{F}\_\lambda^\pi$ based on Bellman residual minimisation.
(Similar computations hold for $\mathbf{B}\_\lambda^\pi$). To reduce clutter, we will now use the shorthand 
$\mathbf{F}:=\mathbf{F}\_\lambda^\pi$ and $\mathbf{B}:=\mathbf{B}\_\lambda^\pi$.

### TD-like updates
Assume $\mathbf{B}$ fixed. The forward successor Bellman equations for $\mathbf{F}$ writes 
$
\mathbf{F}^\top\mathbf{B} = \mathbf{I}_d + \lambda\mathbf{P}\_\pi\mathbf{F}^\top\mathbf{B}\\; .
$
A reasonable value for $\mathbf{F}$ minimises the Bellman residual:
$$
\min\_{\mathbf{F}\in\mathbb{R}^{d\times\mathrm{S}}} \left\\| \mathbf{F}^\top\mathbf{B} -  (\mathbf{I}_d + \lambda\mathbf{P}\_\pi\mathbf{F}^\top\mathbf{B})\right\\|\_2^2\\; ,
$$
where the matrix $\ell\_2$-norm is $\\|\mathbf{A}\\|\_2^2 = \text{Tr}(\mathbf{A}\mathbf{A}^\top)$.
This program, however, does not admit a closed-form. 
Instead, we can follow some now well-established ideas, and update $\mathbf{F}$ along the gradients of the objective:
$
J(\mathbf{F}) :=  \left\\| \mathbf{F}^\top\mathbf{B} -  (\mathbf{I}_d + \lambda\mathbf{P}\_\pi\mathbf{F}^\top\mathbf{B})\_\bot\right\\|\_2^2\\;,
$
where $\bot$ in the stop-gradient operator (_a la_ target networks). This yields the update rule:
$$
\tag{6}
\delta \mathbf{F} \propto \mathbf{B} - \mathbf{B}\mathbf{B}^\top\mathbf{F}( \mathbf{I}\_d-  \lambda\mathbf{P}\_\pi^\top)\\; .
$$
Turns out, (6) yields the basis for learning $\mathbf{F}$ under more intricate conditions, such as _e.g._ continuous states.
{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
A simple re-write of the objective yields:
$$
\begin{aligned}
J(\mathbf{F}) &= \text{Tr}(\mathbf{B}^\top\mathbf{F}\mathbf{F}^\top\mathbf{B}) - 2\text{Tr}(\mathbf{F}^\top\mathbf{B}(\mathbf{I}_d+\mathbf{B}^\top\mathbf{F}\_\bot^\top\mathbf{P}\_\pi^\top))+\square\\;,
\end{aligned}
$$
where $\square$ denotes constants (w.r.t. $\mathbf{F}$), and where we used $\text{Tr}(\mathbf{A})=\text{Tr}(\mathbf{A^\top})$.
From this, remembering the definition of the [Gateau derivative](https://en.wikipedia.org/wiki/Gateaux_derivative), 
that $\text{Tr}(\mathbf{A}\mathbf{B})=\text{Tr}(\mathbf{B}\mathbf{A})$ and realising that
$(\mathbf{A},\mathbf{B})\mapsto\text{Tr}(\mathbf{A}\mathbf{B}^\top)$ defines an inner-product over matrices
yields the announced result.
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

A similar update rule can be computed for $\mathbf{B}$; likewise, their backward counterpart 
can be obtained by minimising the backward Bellman successor residuals. 
This yields four different update rules for alternatively updating $\mathbf{F}$ and $\mathbf{B}$,
which can be used interchangeably. {{< ref link="states">}} [2]{{< /ref>}}


## Generic state-spaces
Let us now revisit our finiteness assumption and consider a continuous state-space
$\mathcal{S}=\mathbb{R}^{\mathrm{S}}$.
Representing the discounted state occupancy measure is no longer on the table.
Let us re-define this measure for this new reality. For any measurable subset $S\subseteq\mathcal{S}$
and starting point $s\in\mathcal{S}$, the occupancy measure $m\_\lambda^\pi$ writes:
$$
m\_\lambda^\pi(S\vert s) := \sum\_{t\geq 1} \lambda^{t-1} \mathbb{P}^\pi(s\_t\in S \vert s\_1 = s)\\;.
$$
Learning a measure directly is non-trivial. Instead, we can learn its density (or Radon-Nikodym derivative) 
with respect to a reference probability measure $\rho$. Using infinitesimal notations:
$$
m\_\lambda^\pi(ds^\prime) := \tilde m\_\lambda^\pi(s^\prime\vert s)\rho(ds^\prime) \\;.
$$
We can learn $\tilde m\_\lambda^\pi(s^\prime\vert s)$ directly, which is interpreted as a 
similarity metric between states. It will be assumed, though, that one can 
sample from $\rho$ -- for instance, if $\rho$ is the steady-state distribution induced by $\pi$. 
The forward-backward factorisation will now write
$
m\_\lambda^\pi(s^\prime\vert s) = f\_\lambda^\pi(s)^\top b\_\lambda^\pi(s^\prime)
$
where $f\_\lambda^\pi$ and $b\_\lambda^\pi$ map $\mathcal{S}$ to $\mathbb{R}^d$.
Most results detailed for the finite case can be generalised, under this set-up,
to continuous condition. Learning rules for $m\_\lambda^\pi$, $f\_\lambda^\pi$ and
$b\_\lambda^\pi$ can be retrieved by following a similar logic.
 
## References
I got introduced to successor features reading:
<div id="features"></div>
[1] Successor features for transfer in reinforcement learning. Barreto, Andre, et al, 2017.

<br>
<br>

which led me to the following, which this post is a condensed summary of:
<div id="states"></div>
[2] Learning successor states and goal-dependent values. Blier, Leonard et al, 2021.