+++
author = "Louis Faury"
title = "Control in (Generally) Regularised MDPs"
date = "2025-12-05"
+++
This post explores the theory of regularised MDPs beyond entropic regularisation (which we covered in an older post).
We will introduce convex regularisation of the classical Bellman operators and study the induced regularised policy
iteration algorithms. On the way, we will tie some links with several popular algorithms.
This post is mostly a good excuse to refresh some convex optimisation classics.
<!--more-->

## Warm-up

We covered entropic regularisation in [this post](../maxent). It was motivated by a modification of the discounted return
involving a policy's differential entropy in its valuation. 
There, we introduced the Bellman operators associated to this objective, along with the value/policy iteration scheme that ensues.
Below, we repeat this operation by working directly at the operator level. 
We will cover a collection of convex-regularised Bellman operators, each inducing
their own set of control algorithms – entropic regularisation being a special case.

{{< warningblock>}}
$\quad$ We will use indexed and vectorial notations interchangeably– see <a href="../mdp_basics" style="text-decoration:none; color:#0074aa;" ">here</a> for some background.
{{< /warningblock >}}

Let $\mathcal{M}=(\mathcal{S}, \mathcal{A}, p, r)$ be some finite MDP that we study under a discounted criterion.
For any $v\in\mathbb{R}^\mathrm{S}$ denote $q\in\mathbb{R}^{\mathrm{S}\times\mathrm{A}}$ the associated $q$-values. We will denote
$q(s)$ the associated vector in $\mathbb{R}^\mathrm{A}$, which entry $a$ is $q(s, a)$. Further, let $\Delta\_\mathrm{A}$ the simplex over $\mathcal{A}$. 
For any stationary policy $\pi\in\Delta\_{\mathrm{A}}^\mathrm{S}$ and $v\in\mathbb{R}^\mathrm{S}$,
the Bellman evaluation and optimality operators write, 
respectively,
$
\mathcal{T}\_\lambda^\pi(v) := r\_\pi + \lambda \mathbf{P}\_\pi v
$
and
$
\mathcal{T}\_\lambda^\star(v) := \max\_\pi\mathcal{T}\_\lambda^\pi(v).
$
Rewriting the former using state-indexed notations reveals a linear structure w.r.t the policy and the $q$-values:
$$
\mathcal{T}\_\lambda^\pi(v)(s) = \sum\_{a} \pi(a\vert s)\big[r(s, a) + \sum\_{s^\prime\in\mathcal{S}} r(s^\prime\vert s, a)v(s^\prime)\big]
= \pi(s)^\top q(s)\\;.\tag{1}
$$
Similarly, the optimality operator exists as the solution of a linear program via $\mathcal{T}\_\lambda^\star(v)(s) = \max\_{\pi} \pi(s)^\top q(s)$.


## Regularised MDPs

Throughout, $\Omega:\Delta\_\mathrm{A}\mapsto\mathbb{R}$ refers to some strongly convex function,
that we assume to be smoothly differentiable for simplicity.
Concretely, $\Omega$
is lower-bounded by a quadratic form; it exists some $\alpha>0$ such that for any $y$ and $x\in\Delta\_\mathrm{A}$ we have
$
\Omega(y) > \Omega(x) + \nabla\Omega(x)^\top(y-x) + \alpha\\|y-x\\|^2\\;.
$
### Convex conjugate

We will start with some convex optimisation refreshers. The convex conjugate $\Omega^\star:\mathbb{R}^\mathrm{A}\mapsto\mathbb{R}$
is defined as:
$$
\Omega^\star(y) := \max\_{x\in\mathbb{R}^\mathrm{A}} \left\\{ x^\top y - \Omega(x)\right\\}\\;. \tag{2}
$$

The convex conjugate is sometimes called the Legendre-Fenchel transformation.
There are several great resources online to gain intuition on this object–
here, we will focus directly on some key properties. 
First, observe that the definition (2) is valid thanks to $\Omega$'s strong convexity:
the maximum exists and is unique. 
Second, 
$\Omega^\star$ is convex and differentiable. Finally, we have a useful characterisation 
of $\nabla\Omega^\star$; for any $y\in\mathbb{R}^d$ :
$$
\nabla\Omega^\star(y) = \argmax_{x\in\mathbb{R}^\mathrm{A}} \left\\{ x^\top y - \Omega(x)\right\\}\\;. \tag{3}
$$
{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
By being a maximum of linear (and therefore convex) functions, $\Omega^\star$ is convex.
Fix $y,y^\prime\in\mathbb{R}^\mathrm{A}$
and denote ${x_\star} = \argmax_{x} \\{ x^\top y- \Omega(x)\\}$.
As a result $\Omega^\star(y)={x_\star}^\top y - \Omega({x_\star})$.
By definition of $\Omega^\star$:
$$
\begin{aligned}
\Omega^\star(y^\prime) &\geq {x_\star}^\top y^\prime - \Omega({x_\star}) \\;, \\\
&= {x_\star}^\top (y^\prime-y) + {x_\star}^\top y - \Omega({x_\star})\\;, \\\
&= \Omega^\star(y) + {x_\star}^\top (y^\prime-y)\\;, &(\text{def. of } {x_\star})
\end{aligned}
$$
which is the characterisation of the subgradient $\partial \Omega^\star(y)$; hence we proved
that $x_\star \in \partial\Omega^\star(y)$. Reversing the argument, one gets that 
$\partial\Omega^\star(y)=\\{x_\star\\}$, proving that $\Omega^\star$ is differentiable
and $\nabla\Omega^\star(y) = \argmax_{x} x^\top y- \Omega(x)$.
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}
{{< infoblock>}}
$\quad$ There are countless excellent resources to gain intuition on convex conjugates.
We do not cover them here, in an effort to keep the focus on its applications for our purposes.
{{< /infoblock >}}


### Regularised operators

We can now introduce a _regularised_ Bellman operator involving $\Omega$.
This might seem somewhat arbitrary for now–nonetheless, for $\pi\in\Delta_\mathrm{A}^\mathrm{S}$ some stationary policy
and $v\in\mathbb{R}^\mathrm{S}$, define:
$$
\mathcal{T}\_\Omega^\pi(v)(s) := \pi(s)^\top q(s)-\Omega(\pi(s))\\;.\\\
\tag{4}
$$

This operator retains the original's desirable properties, such as linearity, monotonicity, and, most importantly, contraction.
It therefore admits a unique fixed-point, which can be retrieved via fixed-point iterations. 
We denote $v\_\Omega^\pi$ this fixed-point, which from now on will stand as a regularised value-function.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Contraction and monotonicity are immediate because inherited from the original Bellman evaluation operator.
Indeed for all $v, v^\prime\in\mathbb{R}^\mathrm{S}$, observe that:
$$
\mathcal{T}\_\Omega^\pi(v) - \mathcal{T}\_\Omega^\pi(v^\prime)  =
     \mathcal{T}\_\lambda^\pi(v) - \mathcal{T}\_\lambda^\pi(v^\prime)\\;.
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

{{% toggle_block background-color="#CBE4FE" title="Examples" default-display="none"%}}
For instance, take $\Omega(\pi(s))=-\mathcal{H}(\pi(s))$ be the negative entropy. 
The resulting operator is the one studied
[here](../maxent/#:~:text=improvement%20operator%20(4).-,Soft%20Policy%20Iteration,-Most%20of%20the)
for entropic regularisation.
Another valid regulariser is $\Omega(\pi(s)) = \text{KL}(\pi(s)\\| \pi\_0(s))$, for some reference $\pi\_0$.
{{% /toggle_block %}}

The regularised Bellman optimality operator naturally follows, and it involves the convex conjugate $\Omega^\star$.
It is defined by symmetry to the unregularised case, where
$\mathcal{T}\_\lambda^\star(v) = \max\_\pi \mathcal{T}\_\lambda^\pi(v)$. For any $v\in\mathbb{R}^\mathrm{S}$
and $s\in\mathcal{S}$:
$$
\tag{5}
\mathcal{T}\_\Omega^\star(v)(s) := \max\_{\pi} \big\\{ \pi(s)^\top q(s) - \Omega(\pi(s)) \big\\} = \Omega^\star(q(s))\\;.
$$
Because $\mathcal{T}\_\Omega^\pi$ had the same properties as $\mathcal{T}\_\lambda^\pi$, 
we can repeat most of the argument developed
[here](../mdp_basics_2/#:~:text=Bellman%20prediction%20equations-,Optimal%20Control,-We%20can%20now)
to prove that it also is a contracting mapping. 
We will denote $v\_\Omega^\star$ its unique fixed-point, and call that object the
optimal regularised value. 
Thanks to (3), given some $v\in\mathbb{R}^\mathrm{S}$ (and its associated $q\in\mathbb{R}^{\mathrm{S}\times\mathrm{A}}$),
the greedy regularised policy, _a.k.a_ the one which attains the maximum in (5), is provided by the
convex conjugate's gradient. 
Generalising from the non-regularised case, the optimal $\Omega$-regularised policy 
is defined as the only $v\_\Omega^\star$-improving policy; concretely, 
$\pi\_\Omega^\star(s) := \nabla\Omega^\star(q\_\Omega^\star(s))$ for every $s\in\mathcal{S}$.

Finally, it is reasonable to wonder what is the impact of regularisation on actual performance, 
as measured by discounted return. This can be directly quantified; with $u\_\Omega$ (resp. $\ell\_\Omega$) is $\Omega$'s upper (resp. lower) bound:
$$
v\_\lambda^\star - \frac{u\_\Omega - \ell\_\Omega}{1-\lambda}- \leq v\_\lambda^{\pi^\star\_\Omega} \leq v\_\lambda^\star \\;.
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Below, we overload notations and write $\Omega(\pi)\in\mathbb{R}^\mathrm{S}$ such that $\[\Omega(\pi)\](s) = \Omega(\pi(s))$.
Now, we can write that for any stationary $\pi$ and $v\in\mathbb{R}^\mathrm{S}$:
$$
\begin{aligned}
\mathcal{T}\_\Omega^\pi(v) &= \mathcal{T}\_\lambda^\pi(v) - \Omega(\pi)\\;, \\\
&\geq \mathcal{T}\_\lambda^\pi(v) - u\_\Omega \mathbf{e}\\;,
\end{aligned}
$$
where the inequality holds coordinate-wise, and $\mathbf{e}\in\mathbb{R}^\mathrm{S}$ is a vector which every entry is 1.
Therefore, by the monotonicity of $\mathcal{T}\_\Omega^\pi$:
$$
\begin{aligned}
(\mathcal{T}\_\Omega^\pi)^2(v) &\geq \mathcal{T}\_\Omega^\pi\mathcal{T}\_\lambda^\pi(v) - \mathcal{T}\_\Omega^\pi u\_\Omega \mathbf{e}\\;, \\\
&\geq (\mathcal{T}\_\lambda^\pi)^2(v) - u\_\Omega \mathbf{e} - \lambda u\_\Omega\mathbf{e}\\;,
\end{aligned}
$$
and by induction one show that for all $n\in\mathbb{N}$ we have 
$(\mathcal{T}\_\Omega^\pi)^n (v) \geq (\mathcal{T}\_\lambda^\pi)^n(v) - u\_\Omega \mathbf{e}\sum\_{k\leq n} \lambda^i$.
Letting $n\to\infty$ proves, thanks to fixed-point characterisations, that $v\_\Omega^\pi \geq v\_\lambda^\pi - u\_\Omega/(1-\lambda)$.
Repeating this for $\ell\_\Omega$ one gets:
$$
v\_\lambda^\pi - u\_\Omega/(1-\lambda) \leq v\_\Omega^\pi \leq v\_\lambda^\pi + \ell\_\Omega/(1-\lambda) \\;.
$$
Finally, we get that:
$$
\begin{aligned}
v\_\lambda^\star &= v\_\lambda^{\pi^\star} \\;, \\\
&\leq v\_\Omega^{\pi^\star} + u\_\Omega/(1-\lambda)\\;,\\\
&\leq v\_\Omega^{\pi^\star\_\Omega} + u\_\Omega/(1-\lambda)\\;, &(\text{def. of } \pi^\star\_\Omega)\\\
&\leq v\_\Omega^{\pi^\star} + u\_\Omega/(1-\lambda) - \ell\_\Omega/(1-\lambda)
\end{aligned}
$$
which concludes the proof.
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}


### Algorithms

{{< infoblock>}}
$\quad$ A refresher on control algorithms in MDPs can be found <a href="../mdp_basics_3" style="text-decoration:none; color:#0074aa;" ">here</a>.
{{< /infoblock >}}

#### Regularised Policy Iteration
Armed with regularised Bellman operators, we can walk again the route for computing
$v\_\Omega^\star$ and/or $\pi\_\Omega^\star$. By the contractive nature of such 
operators, the story is going to be quite similar to the non-regularised case.
For instance, a regularised value iteration   writes 
$v\_{\Omega}^{k+1} = \mathcal{T}\_\Omega^\star(v\_{\Omega}^k)$ and this process will 
converge to $v\_\Omega^\star$.
Similarly, a regularised policy iteration scheme executes
$\pi\_{k+1} = \nabla\Omega^\star(v\_{\pi\_k})$ – and will converge to $\pi\_\Omega^\star$.
We do not provide proof for said claims; one can directly extend the ones developed in the 
unregularised setting. Finally, a regularised generalised policy iteration protocol will look like, 
for some $n\in\mathbb{N}$:
$$
\left\\{
\begin{aligned}
v\_{k+1} &= \underbrace{\mathcal{T}\_\Omega^{\pi\_k}\circ\mathcal{T}\_\Omega^{\pi\_k} \circ \ldots \mathcal{T}\_\Omega^{\pi\_k}}_{n}(v\_k)\\;,\\\
\pi\_{k+1} &= \nabla\Omega^\star(v\_{k+1})\\;.
\end{aligned}\right.
$$


{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
This is quite powerful; for any convex $\Omega$, we just got a functioning control algorithm out of the box and with the usual convergence guarantees.
{{% /toggle_block %}}


#### With approximations

Beyond classical control algorithms, some modern approaches
can be reformulated to fall within the family of regularised algorithms.
The most famous examples are derived from entropic regularisation–that is, 
$\Omega(p) = \sum\_{a\in\mathcal{A}} p_a\log p\_a$
and $\Omega^\star(p) = \log[\sum\_{a\in\mathcal{A}}\exp(p_a)]$ being the smoothed maximum.
For instance, the soft q-learning of {{< ref link="sql">}} [2]{{< /ref>}}
can be seen as a extension of the $\Omega$-regularised value iteration algorithm
(extend to support stochastic approximation and function approximation).
There, the main difficulty comes from the evaluation of $\Omega^\star$ for 
continuous action spaces (it involves an integral over $\mathcal{A}$).
Similarly, the soft actor-critic of {{< ref link="sac">}} [3]{{< /ref>}}
can be seen as the generalised policy-iteration scheme with the entropic regulariser.


## Extension

### Mirror descent
Let us take a short break to go through some additional convex optimisation refreshers. In what follows, for any 
$x, y\in\mathbb{R}^\mathrm{A}$ we denote $d\_\Omega(x \\|\\ y)$ the Bregman divergence of $x$ wrt $y$
associated to $\Omega$. That is:
$$
d\_\Omega(x \\|\\ y) = \Omega(x) - \Omega(y) - \nabla\Omega(y)^\top(x-y)\\;,
$$
which is the difference between the value at $x$ of $\Omega$
and its first-order approximation around $y$.
Observe that by strong convexity of $\Omega$, the Bregman divergence is a positive, strongly convex function of $x$.
It equals 0 if and only if $x=y$ but (in general) is not symmetric and breaks the triangle inequality, so
it is not a distance.

{{% toggle_block background-color="#CBE4FE" title="Examples" default-display="none"%}}
When $\Omega(x)=\\|x\\|^2$, the associated Bregman divergence is the $\ell\_2$ norm.
Similarly, one can check that the KL-divergence is associated with the negative entropy 
$
\Omega(x) = \sum\_i x\_i\log(x\_i) \\;.
$
{{% /toggle_block %}}


The mirror descent algorithm is a generalisation of the gradient descent algorithm, 
but instead of minimising a quadratic lower-bound to the original objective, it 
leverages a Bregman divergence surrogate. Concretely, the mirror descent update
trades-off between a linear approximation of some objective function $f:\mathbb{R}^\mathrm{A}\mapsto\mathbb{R}$
around the current iterate $x\_t$ and the Bregman divergence associated with $\Omega$ and wrt $x\_t$:
$$
\tag{6}
x\_{t+1} := \min\_x \big\\{ f(x\_t) + \nabla f(x\_t)^\top(x-x\_t) + d\_\Omega(x\\|x\_t)\big\\}\\;.
$$
Observe that when $\Omega = \\|\cdot\\|^2$ we retrieve the gradient descent update.
The point of (6) is to allow the update's regulariser to better capture the geometry
induced by $f$ and its feasible set.

{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
To illustrate that last point observe that if $\Omega\equiv f$, a single mirror-descent update leads
us straight to a minimum of $f$, that is $x\_{t+1} = \argmin\_x f(x)$. 
That is of course somewhat silly (if we knew how to compute this update, we would not be using 
an iterative algorithm to minimise $f$) but illustrate that with the 'right' regulariser, mirro descent
can considerably speed up the optimisation of $f$.
{{% /toggle_block %}}

{{< infoblock>}}
$\quad$ This is called the proximal view of mirror descent. There is also a 
fascinating mirror-map view which generalises gradient descent beyond Hillbert spaces
(for which the representer theorem holds).
{{< /infoblock >}}



### Dynamic regularisation
We so far pursued _static_ regularisers $\Omega(\pi)$ which, in spite of potential benefits
(_e.g._ additional stability when applied to policy optimisation), do bias the fixed point away from
the original discounted objective's solution. 
In an effort to maintain satisfying asymptotic performance,
it is tempting to adjust the regularisation dynamically. We can, for instance, encourage the
_next_ iterate of a (generalised) policy improvement step to remain 'close' to the current
policy. Proximity can be measured by, _e.g_, the Bregman divergence $d\_\Omega(\cdot\\|\pi\_t)$, 
and plugged in the regularised operators introduced in the previous section.
For instance, the improvement step resembles a mirror descent update, as for all $s\in\mathcal{S}$:
$$
\pi\_{t+1}(s) \in \argmax\_\pi \big\\{\pi(s)^\top q(s) - d\_\Omega(\pi(s)\\|\pi\_t(s))\big\\}\\;.
$$


Several modern algorithms fall into the related family of regularised approaches – such as PPO and MPO,
which both uses a Kullback-Leibler divergence as a regulariser (and, of course, diverge from each other on many other 
aspects – such as policy evaluation).



## Reference
This entire blog post is essentially my retranscription of the excellent paper:

[1] A Theory of Regularized Markov Decision Processes. Geist et al, 2019.

Other references include:

<div id="sql"></div>
[2] Reinforcement Learning with Deep Energy-Based Policies. Haarnoja et al, 2017.


<div id="sac"></div>
[3] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. Haarnoja et al, 2018.

<div id=""></div>
