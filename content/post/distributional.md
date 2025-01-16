+++
author = "Louis Faury"
title = "Distributional Dynamic Programming"
date = "2024-12-23"
+++

Control in MDPs is often focused on the expected return criterion, which has the good taste to follow 
the Bellman dynamic programing equations we all love and cherish. 
In this post, we see how those equations generalise beyond the expected return to the full return _distribution_.
We will cover the distributional Bellman equations for policy evaluation, and see them in action 
in practical distributional dynamic programming algorithm based on quantiles.

<!--more-->


{{< infoblock>}}
$\quad$ Throughout, we will use the MDP notations defined in <a href="../mdp_basics" style="text-decoration:none; color:#0074aa;" ">this post</a>.
{{< /infoblock >}}


Let us consider a finite MDP  $\mathcal{M} = (\mathcal{S}, \mathcal{A}, r, p)$ and
$\pi:\mathcal{S}\mapsto\Delta(\mathcal{A})$ some stationary policy.
We denote $R = \sum\_{t\geq 1} \lambda^{t-1} r(s\_t, a\_t)$ the return associated with
the stochastic process obtained when running $\pi$ on $\mathcal{M}$. 
When evaluating $\pi$ under a _discounted_ objective, we score $\pi$ via the expected value of $R$–_a.k.a_ the value-function:
$$
\tag{1}
\begin{aligned}
v^\pi(s) := \mathbb{E}^\pi\left[R\\,\middle\vert\\, s\_1=s\right]\\; .
\end{aligned}
$$
The expectation is _convenient_ -- most control and RL algorithms heavily rely on 
its underlying properties, captured by the classical Bellman equations.
It, however, does average out many subtleties of the reward process collected under $\pi$. 
Such nuances can be quite relevant when one decides to _e.g._ adopt & risk-averse profile.
In this post, we follow the distributional RL literature
and set out to characterise and model the entire distribution $\mu\_\lambda^\pi$ of the return. 

<br>
{{< image src="/return_distrib.png" width="560px" align="center" caption="Two different return distributions with the same mean.">}}
<br>

{{< warningblock>}}
$\quad$ In this post we are only concerned about policy evaluation, and leave control for another day.
{{< /warningblock >}}

## Preliminaries
### Definitions
For any stationary $\pi$ the return distribution $\mu\_\lambda^\pi$ maps $\mathcal{S}$ to $\Delta(\mathbb{R})$, the space of distributions over 
the real line.
For concreteness, fix $s\in\mathcal{S}$ and denote for an instant $\mu = \mu\_\lambda^\pi(s)$; then for any $\mathcal{R}\subseteq\mathbb{R}$:
$$
\tag{2}
\mu(\mathcal{R}) := \mathbb{P}^\pi\left(R \in \mathcal{R} \\, \middle\vert \\, s\_1=s\right)\\; .
$$

{{< infoblock>}}
$\quad$ We will often use $\mu$ to denote either a distribution (member of $\Delta(\mathbb{R})$) or a return
distribution (member of $\Delta(\mathbb{R})^\mathcal{S}$) in an effort to reduce clutter. Context shall
be clear enough to avoid confusions.
{{< /infoblock >}}

We will denote $R(s)$ the return obtained by conditioning $s\_1=s$ and following the reward process
induced by $\pi$ (we drop the dependency to $\pi$ to reduce clutter.) 
A simple decomposition of the return tells us that for any transition $(s, a, s')$:
$$
\tag{3}
R(s) \overset{d}{=} r(s,a) + \lambda R(s^\prime)\\;,
$$
where the notation $\overset{d}{=}$ indicates an equality of distribution between the left and right hand sides.
This is an equality (in distribution) between two random variables; it does not get us through the finish line as 
it is not a Bellman-like equation for the return distribution $\mu\_\lambda^\pi$. 
To get there, we will need a few extra tools that we detail below—without much context for now.


### Push-forward operator
Let $f:\mathbb{R}\mapsto \mathbb{R}$ a Borel-measurable function.
The notation $f\_\sharp$ denotes the push-forward operator by $f$:
$$
\begin{aligned}
f\_\sharp : \Delta(\mathbb{R}) &\mapsto \Delta(\mathbb{R}) \\;,\\\ 
\mu &\mapsto f\_\sharp\mu
\end{aligned}
$$
where $f\_\sharp\mu(\mathcal{R}) = \mu(f^{-1}(\mathcal{R}))$ for any $\mathcal{R}\subseteq\mathbb{R}$.


### Wasserstein distance
We will equip $\Delta(\mathbb{R})$ with some Wasserstein metric. 
For $p>1$ the $p$-Wasserstein distance between $\mu$ and $\nu$
is generically defined via the coupling $\Gamma(\mu, \nu)$ -- the set of joint distributions which
marginals are $\mu$ and $\nu$:
$$
\omega\_p(\mu, \nu) = \inf\_{\gamma\in\Gamma(\mu, \nu)} \mathbb{E}_{(X,Y)\sim\gamma} \left[\vert X-Y\vert^p\right]^{1/p}\\; .
$$
For distribution on the real line this simplifies to involve only the (inverse) cumulative distribution functions:
$$
\omega\_p(\mu, \nu) = \left(\int\_{0}^1 \left\vert F\_{\mu}^{-1}(z) - F\_{\nu}^{-1}(z)\right\vert ^pdz\right)^{1/p}\\; .
$$



## Distributional Bellman equations
Our road to devising algorithms that can compute $\mu\_\lambda^\pi$ is going to ressemble
the "classical" one, that we follow to compute the optimal value function for an MDP -- the
tools will be quite different, though.
Up to some hoops we will have to jump through along the way, the protocol will be to:
1. show that 
$\mu\_\lambda^\pi$ is the fixed-point to some contractive operator,
2. compute it via some fixed-point iterations.

{{< warningblock>}}
$\quad$ This section assumes some background with classical fixed-point properties (see e.g. <a href="https://en.wikipedia.org/wiki/Banach_fixed-point_theorem" style="text-decoration:none; color:#0074aa;" ">here</a>).
{{< /warningblock >}}


### A fixed-point equation
The fixed-point characterisation of $\mu\_\lambda^\pi$ is the distributional equivalent of (3). For any $s, a\in\mathcal{S}\times\mathcal{A}$
let us introduce an operator which push-forward will prove useful:
$$
\begin{aligned}
T\_\lambda^{s, a} : \mathbb{R} &\mapsto \mathbb{R} \\;, \\\
z &\mapsto r(s, a) + \lambda z \\; .
\end{aligned}
$$
The push-forward $(T\_\lambda^{s, a})\_\sharp$ manifestly allows us to represent (distributionally) 
stepping one step ahead in the reward process.
Without further due, we will now state the desired result and move forward with a proof. 

{{< boxed title="Distributional Bellman Equation" >}}
$\qquad \qquad \qquad\qquad\qquad\qquad\qquad\quad \text{ For any } s\in\mathcal{S}:$
$$
\tag{4}
\mu_\lambda^\pi(s) = \sum_{a\in\mathcal{A}}\sum_{s^\prime\in\mathcal{S}} \pi(a\vert s)p(s^\prime\vert s, a)(T_\lambda^{s, a})_\sharp
\mu_\lambda^\pi(s^\prime)\;.
$$
{{< /boxed >}}

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
For any $s\in\mathcal{S}$ and $r\in\mathbb{R}$:
$$
\begin{aligned}
\mu\_\lambda^\pi(s)([-\infty, r[) &= \mathbb{P}^\pi(R\_t(s) \leq r )\\, \\\
&= \mathbb{P}^\pi(R\leq r\vert s\_1 = s)\\;, \\\
&= \sum\_{a\in\mathcal{A}}\pi(a\vert s) \mathbb{P}^\pi(R\leq r\vert s\_1 = s, a\_1=a)\\;,\\\
&= \sum\_{a\in\mathcal{A}}\pi(a\vert s) \sum\_{s^\prime\in\mathcal{S}}p(s^\prime\vert s, a)\mathbb{P}^\pi\left(r(s, a) + \lambda R(s\_2)\leq r\vert s\_1 = s, a\_1=a, s\_2=s^\prime\right)\\;,\\\
&= \sum\_{a\in\mathcal{A}}\pi(a\vert s) \sum\_{s^\prime\in\mathcal{S}}p(s^\prime\vert s, a)\mathbb{P}^\pi\left(R(s^\prime)\leq (r-r(s, a)/\lambda)\right)\\;,\\\
\end{aligned}
$$
where we applied the law of total probility a few time, and recognise $(T\_\lambda^{s, a})\_\sharp\mu(s^\prime)([-\infty, r[)$ on the 
last line. As a result, we have $\mu\_\lambda^\pi(s)([-\infty, r[) = \sum\_{a\in\mathcal{A}}\pi(a\vert s) \sum\_{s^\prime\in\mathcal{S}}p(s^\prime\vert s, a)
(T\_\lambda^{s, a})\_\sharp\mu(s^\prime)([-\infty, r[)$ for any $r\in\mathbb{R}$.
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}


We will sometimes write (4) as $\mu\_\lambda^\pi(s) = \mathbb{E}^\pi\_s\left[(T\_\lambda^{s, a})\_\sharp
\mu_\lambda^\pi(s^\prime)\right]$, or equivalently use the distributional 
operator $\mathcal{T}\_\text{d}\Delta:(\mathbb{R})^\mathcal{S} \mapsto \Delta(\mathbb{R})^\mathcal{S}$
where $\mathcal{T}\_\text{d}^\pi(\mu)(s)=\sum_{a\in\mathcal{A}}\sum\_{s^\prime\in\mathcal{S}} \pi(a\vert s)p(s^\prime\vert s, a)(T\_\lambda^{s, a})_\sharp
\mu\_\lambda(s^\prime)$ for any $s\in\mathcal{S}$, so that:
$$
\tag{5}
\mu\_\lambda^\pi = \mathcal{T}\_\text{d}^\pi(\mu\_\lambda^\pi)\\; .
$$


### The distributional Bellman operator
Beyond the intellectual satisfaction, the identities (4–5) are only useful if they can be turned into an algorithm to
compute $\mu\_\lambda^\pi$. 
In this section, we make the first step towards this goal by characterising the contractive properties of the Bellman 
distributional operator $\mathcal{T}\_\text{d}^\pi$.
Contraction will happen under a metric that extends $\omega\_p$ to $\Delta(\mathbb{R})^\mathcal{S}$.
Concretely, we define:
$$
\begin{aligned}
W\_p : \Delta(\mathbb{R})^\mathcal{S}\times \Delta(\mathbb{R})^\mathcal{S} &\mapsto\mathbb{R} \\;, \\\
(\mu, \nu) &\mapsto \max\_{s\in\mathcal{S}} \omega\_p(\mu(s), \nu(s)) \\; ,
\end{aligned}
$$
which is easily proven to be a valid metric over $ \Delta(\mathbb{R})^\mathcal{S}$.
The following claim establishes that
$\mathcal{T}\_\text{d}^\pi $ is a $\lambda$-contraction under $W\_p$.
It is then a classical fixed-point result that $\mu\_\lambda^\pi$ is the unique solution to (5).


{{< boxed title="$\mathcal{T}_\text{d}^\pi$ contracts" >}}
$\qquad \qquad\qquad \text{ For any } \mu, \nu \in\Delta(\mathbb{R})^\mathcal{S}:$
$$
W_p(\mathcal{T}_\text{d}^\pi\mu, \mathcal{T}_\text{d}^\pi\nu)  \leq \lambda W_p(\mu, \nu)\;.
$$
{{< /boxed >}}


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof uses the general coupling definition for the p-Wasserstein metric $\omega\_p$.
Let $\mu, \nu\in\Delta(\mathbb{R})^\mathcal{S}$ and 
for every $s\in\mathcal{S}$ denote $\gamma(s)$ the coupling that achieves the minimum value for 
the Wasserstein distance; $\omega(\mu(s), \nu(s)) = \mathbb{E}\_{(X,Y)\sim\gamma(s)}[\vert X-Y\vert^p]^{1/p}$.


Define a collection of random variables $(R\_1(s), R\_2(s))\_{s\in\mathcal{S}}$ where 
$(R\_1(s), R\_2(s)) \sim \gamma(s)$ for any $s\in\mathcal{S}$.
We will define yet another set of random variables; letting $(s, a, s^\prime)$ be some random
transition along $\pi$:
$$
\begin{aligned}
\tilde R\_1(s) &\overset{d}{=} r(s, a) + \lambda R\_1(s^\prime) \\;, \\\
\tilde R\_2(s) &\overset{d}{=} r(s, a) + \lambda R\_2(s^\prime) \\;. \\\
\end{aligned}
$$
Observe that $R\_1$ (resp $R\_2$) is distributed according to $\mathcal{T}\_\text{d}^\pi\mu$ (resp $\mathcal{T}\_\text{d}^\pi\nu$)
hence for any $s\in\mathcal{S}$, $\tilde\gamma(s) = (\tilde R\_1(s), \tilde R\_2(s))$ is a valid
$(\mathcal{T}\_\text{d}^\pi\mu, \mathcal{T}\_\text{d}^\pi\nu)$-coupling. Finally for any $s\in\mathcal{S}$:
$$
\begin{aligned}
\omega\_p(\mathcal{T}\_\text{d}^\pi\mu(s), \mathcal{T}\_\text{d}^\pi\nu(s)) &\leq  \mathbb{E}\_{(X,Y)\sim\tilde\gamma(s)}[\vert X-Y\vert^p]^{1/p} \\;,&\text{(def)}\\\
&= \mathbb{E}\left[\vert \tilde R\_1(s) - \tilde R\_2(s)\vert ^p\right]^{1/p} \\;, \\\
&= \lambda \mathbb{E}\_s\mathbb{E}\left[\vert R\_1(s^\prime) - R\_2(s^\prime)\vert ^p\right]^{1/p} \\;, \\\
&\leq \lambda \max\_{s^\prime\in\mathcal{S}} \mathbb{E}\left[\vert R\_1(s^\prime) - R\_2(s^\prime)\vert ^p\right]^{1/p} \\;, \\\
&\leq \lambda\max\_{s^\prime\in\mathcal{S}}\omega\_p(\mu(s^\prime), \nu(s^\prime)) \\;, \\\
&= \lambda W\_p(\mu, \nu) \\; .
\end{aligned}
$$
Taking the maximum over $s\in\mathcal{S}$ in the left-hand side delivers the result.
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

It also promises us an algorithmic way forward by ensuring the convergence of the fixed point iteration $\mu\_{t+1} = \mathcal{T}\_\text{d}^\pi \mu$
towards $\mu\_\lambda^\pi$, in the p-Wasserstein sense at least since
$
\lim\_{t\to\infty} W\_p(\mu\_t, \mu\_\lambda^\pi) = 0\\; .
$

## Distributional Dynamic Programming

<br>
{{< image src="/categorical_repr.png" width="560px" align="center" caption="A member $\mu$ of the empirical family $\mathcal{F}$ and its cumulative distribution function.">}}
<br>

### Empirical representation
The previous section might convince us that we can naively follow the
fixed-point iteration
$
\mu\_{t+1} = \mathcal{T}\_\text{d}^\pi \mu
$
to compute some good approximation of $\mu\_\lambda^\pi$.
This is conceptually true, but concretely it is not a trivial operation since $\mu\_t$,
a member of $\Delta(\mathbb{R})^\mathcal{S}$, cannot be _a priori_ be represented within the finite memory of a computer.
One way forward is to maintain distributions that are weighted Dirac combs -- in other
words, members of the family $\mathcal{F}$ of empirical distributions:
$$
\mathcal{F}^\mathcal{S} := \Big\\{ \mu(s) = \sum\_{k=1}^{K(s)} p\_k(s) \delta\_{r\_k(s)}, \\;
\sum\_{k=1}^{K(s)} p\_k(s) = 1, r\_k(s)\in\mathbb{R}, \\, K(s)\in\mathbb{N}\Big\\}^\mathcal{S} \subset \Delta(\mathbb{R})^\mathcal{S}\\; .
$$
Observe that for any $\mu\in\mathcal{F}^\mathcal{S}$ we have $\mathcal{T}\_\text{d}^\pi\mu\in\mathcal{F}^\mathcal{S}$.
In other words, $\mathcal{F}$ is _closed_ under $\mathcal{T}\_\text{d}^\pi$ -- any fixed-point iteration
starting in $\mathcal{F}^\mathcal{S}$ will not stray away from it.
We detail below some pseudocode detailing a single fixed-point iteration
when maintaining empirical distributions, represented via
the pair $(\mathbf{r}, \mathbf{p})$:
$$
\begin{aligned}
\mathbf{r} := \left[\\{r\_k(s)\\}\_{k=1}^{K(s)}\right]\_{s=1}^{S}\\;, \\\
\mathbf{p} := \left[\\{p\_k(s)\\}\_{k=1}^{K(s)}\right]\_{s=1}^{S}\\;.
\end{aligned}
$$

{{< pseudocode title="$\texttt{Empirical distributional fixed-point iteration}$" >}} 
$\textbf{input } \text{atoms } \mathbf{r} \text{, weights }\mathbf{p} \\$
$\textbf{init } \text{atoms } \tilde{\mathbf{r}}= \emptyset, \text{, weights }\tilde{\mathbf{p}}=\emptyset \\$
$\textbf{for } s\in\mathcal{S}\\$
$\quad\textbf{for } a\in\mathcal{A}\\$
$\quad\quad\textbf{for } s^\prime\in\mathcal{S}\\$
$\quad\quad\quad \tilde{\mathbf{r}} \leftarrow \tilde{\mathbf{r}} \cup \{r(s, a) + \lambda r_k(s^\prime)\}_{k=1}^{K(s^\prime)}\\$
$\quad\quad\quad \tilde{\mathbf{p}} \leftarrow \tilde{\mathbf{p}} \cup \{\pi(s\vert a)p(s^\prime\vert s, a)p_k(s^\prime)\}_{k=1}^{K(s^\prime)}\\$
$\quad\quad\textbf{end for}\\$
$\quad\textbf{end for}\\$
$\textbf{end for}\\$
$\textbf{return } (\tilde{\mathbf{r}}, \tilde{\mathbf{p}})\;.$
{{< /pseudocode >}}
<br> 

Repeating this protocol ensures convergence (in the $W\_p$ sense) to $\mu\_\lambda^\pi$.
However, this comes at a cost: the memory needed to maintain $\mu\_t$ grows as $\mathcal{O}((SA)^t)$!
This is clearly unacceptable but for prohibitively small values of $t$ -- rendering the 
practical influence of this approach, well, void. 


### Quantile representation

In this section we study an alternative representation allowing for 
constant memory usage—at the price, inevitably, of some precision.
Concretely, we aim to maintain return distributions $\mu$ in:
$$
\mathcal{F}\_q^\mathcal{S} := \Big\\{\mu(s) = \frac{1}{m} \sum\_{k=1}^K \delta\_{r\_k(s)},\\, K\in\mathbb{N}, \\, r\_k(s)\in\mathbb{R} \\,\forall k\Big\\}\\;.
$$
This _quantile_ representation essentially indexes $K$ atoms of equal weight for each state, 
therefore upholding a more reasonable $\mathcal{O}(KS)$ memory cost.
This comes at a price: in particular, that $\mathcal{F}\_q^\mathcal{S}$ is no longer closed 
under $\mathcal{T}\_\text{d}^\pi$. 
To maintain fixed point iterates in $\mathcal{F}\_q^\mathcal{S}$ we must go through a _projection_
step after each application of the distributional Bellman operator. 
It makes sense for $\mathcal{P}_q$ to be a projection mapping according to some 
well-chosen metric. Below, we choose the $\omega\_1$ distance; formally, for 
$\mu\in\Delta(\mathbb{R})^\mathcal{S}$ and any $s\in\mathcal{S}$:
$$
(\mathcal{P}\_q \mu)(s) \in \text{arg min}\_{\zeta\in \mathcal{F}\_q} \omega\_1(\mu(s), \zeta)\\; .
$$

<br>
{{< image src="/quantile_repr.png" width="560px" align="center" caption="The quantile projection of a Gaussian mixture $\mu$.">}}
<br>

Conveniently enough, this projection happens to have a closed-form.
For 
$\mu\in\Delta(\mathbb{R})^\mathcal{S}$ and $s\in\mathcal{S}$
$$
\tag{6}
(\mathcal{P}\_q \mu)(s) = \frac{1}{m} \sum\_{k=1}^K \delta\_{F\_{\mu(s)}^{-1}(\frac{2k-1}{2K})}\\; .
$$
In plain words, the atoms of the projected distributions 
are given by evenly space quantiles of $\mu(s)$.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof relies on the following intermediary result.

**Fact.** For any $\mu\in\Delta(\mathbb{R})$ and $0\leq a \leq b \leq 1$, the minimum of 
$
\int\_a^b \vert F\_\mu^{-1}(y) - z \vert dy
$
is attained in $z=F\_\mu^{-1}(\frac{a+b}{2})$.

With this in mind; for any $\nu\in\mathcal{F}_q$
we write 
$$\nu = \frac{1}{K} \sum\_{k=1}^K \delta\_{r\_k}\text{ with } r\_1\leq r\_2\leq \ldots \leq r\_K \\; .
$$
Observe that $F\_\nu$ is piece-wise constant with $F\_\nu(z) = \frac{1}{K}\sum\_{k=1}^K \mathbf{1}[z\geq r\_k]$.
From there, one can easily obtain that $F\_\nu^{-1}$ is also piece-wise constant, 
and $F\_\nu^{-1}(y) = \sum\_{k=1}^K r\_k\mathbf{1}[\frac{k-1}{K} \leq y \leq \frac{k}{K}]$. Therefore:
$$
\begin{aligned}
\omega\_1(\mu, \nu) &= \int\_{0}^1 \left\vert F\_\mu^{-1}(y) -  F\_\nu^{-1}(y)\right\vert dz \\;, &(\text{def})\\\
&= \int\_{0}^1 \Big\vert F\_\mu^{-1}(y) -  \sum\_{k=1}^K r\_k\mathbf{1}\left[\frac{k-1}{K} \leq y \leq \frac{k}{K}\right]\Big\vert dy \\;, \\\
&= \int\_{0}^1 \sum\_{k=1}^K \vert r\_k - F\_\mu^{-1}(y)\vert\mathbf{1}\left[\frac{k-1}{K} \leq y \leq \frac{k}{K}\right]dy\\;, \\\
&= \sum\_{k=1}^K\int\_{\frac{k-1}{K}}^{\frac{k}{K}} \vert r\_k - F\_\mu^{-1}(y)\vert dy\\; .
\end{aligned}
$$
As a result:
$$
\begin{aligned}
\min\_{\nu\in\mathcal{F}\_q} \omega\_1(\mu, \nu) &= \min\_{r\_{1:K}}\sum\_{k=1}^K\int\_{\frac{k-1}{K}}^{\frac{k}{K}} \vert r\_k - F\_\mu^{-1}(y)\vert dy\\; , \\\
&= \sum\_{k=1}^K\min\_{r\_k}\int\_{\frac{k-1}{K}}^{\frac{k}{K}} \vert r\_k - F\_\mu^{-1}(y)\vert dy\\; , \\\
\end{aligned}
$$
Hence, the $\omega\_1$-projection on $\mathcal{F}_q$ is attained by $\nu$ which atoms
satisfy $r\_k \in \argmin \int\_{\frac{k-1}{K}}^{\frac{k}{K}} \vert r\_k - F\_\mu^{-1}(y)\vert dy$.
By the fact claimed as a prelude to this proof, this means that for each $k\in\\{1, \ldots, K\\}$ we have:
$$
r\_k = F\_\mu^{-1}\Big(\frac{1}{2}\big(\frac{k-1}{K}+\frac{k}{K}\big)\Big) =  F\_\mu^{-1}\Big(\frac{2k-1}{2K}\Big)\\; .
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}


### Quantile Dynamic Programming 
The quantile Dynamic Programming procedure is essentially a combination of the 
distributional Bellman operator with the quantile projection.
Concretely, it followes the iterative protocol:
$$
\tag{7}
\mu\_{t+1} = \mathcal{P}_q \mathcal{T}\_\text{d}^\pi\mu\_t\\; ,
$$
where $\mathcal{P}_q$ is defined in (6). In plain words, after each application of 
the Bellman distributional operator, we project the resulting distribution back to $\mathcal{F}\_q^\mathcal{S}$.
It is straight forward to adapt the pseudocode given above to obtain a working and computationally feasible algorithm.
We still have two questions to answer, however.
1. Does the sequence $\\{\mu\_t\\}\_t$ produced by (7) converge?
2. If yes, what is the approximation error to $\mu\_\lambda^\pi$?

The first question is answered by the affirmative. Indeed, one can show that in the $\mathcal{P}_q$ is 
non-expansive in the $W\_\infty$-sense. As a result,  $\mathcal{P}_q \circ \mathcal{T}\_\text{d}$ contracts 
and convergence to the unique fixed-point of $\mathcal{P}_q \circ \mathcal{T}\_\text{d}$ is guaranteed.


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
For any $\mu, \nu \in \Delta(\mathbb{R})$ we have:
$$
\begin{aligned}
\omega\_\infty(\mathcal{P}_q \mu, \mathcal{P}_q \nu) = \sup\_{y\in[0, 1]} \left\vert F^{-1}\_{\mathcal{P}_q\mu}(y) - F^{-1}\_{\mathcal{P}_q\nu}(y)\right\vert\\; .
\end{aligned}
$$
We will re-use part of the last proof, which basically proves that:

$$
F^{-1}\_{\mathcal{P}_q \mu}(y) = \sum\_{k=1}^K F\_{\mu}^{-1}\big(\frac{2k-1}{2K}\big) \mathbf{1}\Big[\frac{k-1}{K} \leq y \leq \frac{k}{K}\Big]\\; ,
$$
and similarly for $\nu$. Hence;
$$
\begin{aligned}
\omega\_\infty(\mathcal{P}_q \mu, \mathcal{P}_q \nu) &= \max\_{k=1, \ldots, K}\left\vert F\_{\mu}^{-1}\big(\frac{2k-1}{2K}\big) - F\_{\nu}^{-1}\big(\frac{2k-1}{2K}\big)\right\vert \\; , \\\
&\leq \sup\_{y\in[0, 1]} \left\vert F\_{\mu}^{-1}(z) - F\_{\nu}^{-1}(z)\right\vert \\; ,\\\
&= \omega\_{\infty}(\mu, \nu) \\; .
\end{aligned}
$$
Finally:
$$
\omega\_\infty(\mathcal{P}_q\mathcal{T}\_\text{d}^\pi\mu, \mathcal{P}_q\mathcal{T}\_\text{d}^\pi\nu) \leq 
\omega\_\infty(\mathcal{T}\_\text{d}^\pi\mu, \mathcal{T}\_\text{d}^\pi\nu) \leq
\lambda \omega\_\infty(\mu, \nu)\\; .
$$

<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

The second question basically wonders if said fixed point could be anywhere near what we _actually_
want to compute, which is $\mu\_\lambda^\pi$. The result, that we will not prove here (but check the reference below!)
has a very similar flavour to the ones we get when studying approximated dynamic programming:
$$
W\_\infty(\mu\_\infty, \mu\_\lambda^\pi) \leq \frac{W\_\infty(\mathcal{P}\_q \mu\_\lambda^\pi, \mu\_\lambda^\pi)}{1-\lambda}\\;.
$$
Essentially, the error bound depends on how well $\mu\_\lambda^\pi$ is approximated by its projection on $\mathcal{F}\_q^\mathcal{S}$.

## References
This post is basically a condensed version of chapters 1–5 of the excellent book:

<div id="disrl"></div>
[1] Distributional Reinforcement Learning. Bellemare M, Dabney W, Rowland M, 2024.