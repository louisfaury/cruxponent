+++
author = "Louis Faury"
title = "The Linear Quadratic Regulator"
date = "2023-06-26"
+++


This post is concerned with the Linear Quadratic Regulator (LQR) in discrete-time.
The LQR stands as somewhat of a singularity in optimal control theory: the (only?) non-trivial control
problem in continuous state and action space for which a closed-form solution is known. 

<!--more-->

## Deterministic LQR

Let $d\_x, d\_u \in\mathbb{N}$. We are concerned with a _continuous_ control problem: let 
$\mathcal{X} = \mathbb{R}^{d\_x}$ its state space and $\mathcal{U} = \mathbb{R}^{d\_u}$
its input or control space. The LQR presumes that the system's dynamics are _linear_:
$$
x\_{n+1} = A x\_n + B u\_n \\; ,
$$
where $A \in\mathbb{R}^{d\_x\times d\_x}$  and $B \in\mathbb{R}^{d\_x \times d\_u}$. 

{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
Above, the dynamics are assumed stationary, mainly for simplicity. In some cases (that we will specify below)
having non-stationary dynamics:
$$
x\_{n+1} = A\_n x\_n + B\_n u\_n \\;, 
$$
comes with virtually no added complexity when it comes to solving an LQR problem. 
{{% /toggle_block %}}

### Finite-horizon problem
Let $N\in\mathbb{N}$. Under the aforementioned dynamics, we evaluate a sequence of control $(u\_0, \ldots, u\_{N-1}) \in \mathcal{U}^N$
via a finite-horizon, quadratic criterion:
$$
\mathcal{V}(x\_0, u\_0, \ldots, u\_{N-1}) = \sum\_{n=0}^{N-1} \left\\{ x\_n^{\top} Q\_n x\_n + u\_n^\top R\_n u\_n \right\\} + x\_N^\top Q\_N x\_N \\; , 
$$
where $Q\_n \in\mathbb{R}^{d\_x\times d\_x}$ and $R\_n \in\mathbb{R}^{d\_u\times d\_u}$ for $n\in\mathbb{N}$ are both _symmetric_ matrices.
We will be looking for controls that makes this criterion _small_. For our problem to be well-defined, 
we will further ask that $Q\_n$'s are [positive semi-definite](https://en.wikipedia.org/wiki/Definite_matrix) and $R\_t$'s are positive definite. 
Briefly, this ensures that our criterion is bounded-below and that minimizing $v(\cdot)$ makes some sense. That's good new, since this is exactly what we now set out to do:
$$
\\text{ find }(u\_0^\star, \ldots, u\_{N-1}^\star) \in \argmin_{u\_{\_0}, \ldots, u\_{\_{N-1}}} \mathcal{V}(x\_0, u\_0, \ldots, u\_{N-1}) \\; .
\tag{1}
$$
In other words, we want to find an _optimal_ control for a deterministic, finite-time LQR problem. 

#### Just a large optimization problem

It's easy to realize that (1) is just one large optimization problem - and a quadratic one nonetheless!
Indeed, observe that under the linear dynamics:
$$
\begin{aligned}
    x\_1 &= Ax\_0 + Bu\_1 \\;, \\\
    x\_2 &= Ax\_1 + Bu\_1 \\;, \\\
         &= A^2 x\_0 + ABu\_1 + Bu\_1 \\;, \\\
    \vdots \\\
    x\_N &= Ax\_{N-1} + Bu\_{N-1} \\;, \\\
         &= A^N x\_0 + A^{N-1}Bu\_1 + \ldots + u\_{N-1} \\; .
\end{aligned}
$$
With a bit of re-writing, we basically just said that:
$$
\begin{pmatrix} x\_0 \\\ x\_1 \\\ \vdots \\\ x\_{N} \end{pmatrix} =
\underbrace{\begin{pmatrix} I\_{d_x} \\\ A \\\ \vdots  \\\ A^N\end{pmatrix}}\_{\mathbf{H}} x\_0 + 
\underbrace{\begin{pmatrix} 0 & \ldots & \ldots& \ldots \\\ B & 0 & \ldots& \ldots \\\ AB & B & 0& \ldots  \\\ \vdots \\\ A^{N-1}B & A^{N-2}B & \ldots & B \end{pmatrix}}\_{\mathbf{G}}
\begin{pmatrix} u\_0 \\\ \vdots \\\ u\_{N-1} \end{pmatrix}
$$

By denoting $X = \begin{pmatrix} x\_0^\top & \ldots & x\_{N}^\top \end{pmatrix}^\top$ and
$U = \begin{pmatrix} u\_0^\top & \ldots & u\_{N-1}^\top \end{pmatrix}^\top$ we therefore have that:
$$
X= \mathbf{G} U + \mathbf{H} x\_0 \\; ,
$$


Let's now rewrite our objective. It is relatively easy to realize that:
$$
\begin{aligned}
\mathcal{V}(x\_0, U) &= \sum\_{n=0}^{N-1} \left\\{ x\_n^{\top} Q\_n x\_n + u\_n^\top R\_n u\_n \right\\} + x\_N^\top Q\_N x\_N \\; ,\\\
&= X^\top
\overbrace{\begin{pmatrix}
Q\_0 &   & \large{0} \\\
& \ddots & \\\
\large{0} &  & Q\_N
\end{pmatrix}}^{\mathbf{\tilde{Q}}} X  + 
U^\top
\overbrace{\begin{pmatrix}
R\_0 &   & \large{0} \\\
& \ddots & \\\
\large{0} &  & R\_{N-1}
\end{pmatrix}}^{\mathbf{\tilde{R}}} U \\\
&= X^\top {\mathbf{\tilde{Q}}}X + U^\top {\mathbf{\tilde{R}}} U \\\
&= x\_0^\top \mathbf{H}^\top {\mathbf{\tilde{Q}}} \mathbf{H}x\_0+ 2 U \mathbf{G}^\top {\mathbf{\tilde{Q}}}\mathbf{H}x\_0 + 
U^\top ({ \mathbf{G}^\top {\mathbf{\tilde{Q}}} \mathbf{G} + \mathbf{\tilde{R}}}) U \\; ,
\end{aligned}
$$
where we last used the identity $X= \mathbf{G} U + \mathbf{H} x\_0$. 

Notice how $\mathcal{V}$
is _quadratic_ in U, and how $\lim_{\\| U \\| \to \infty} \mathcal{V}(x\_0, U) = +\infty$ (coercitivity) since
$\mathbf{G}^\top {\mathbf{\tilde{Q}}} \mathbf{G} + \mathbf{\tilde{R}} \succ 0$.
As a result, solving $\min\_U \mathcal{V}(x\_0, U)$ boils down to solving a (somewhat large)
linear system: 
$$
\left(\mathbf{G}^\top {\mathbf{\tilde{Q}}} \mathbf{G} + \mathbf{\tilde{R}}\right) U^\star = - G^\top \mathbf{\tilde{Q}} H x\_0 \\; .
$$
This can be easier said than done; indeed, the size of this linear system grows linearly with $N$ -- hence solving it
will cost us roughly $\mathcal{O}(N^3)$ operations, which quickly becomes prohibitive. 


#### Dynamic Programming
When we think about it, we have only described so far some finite horizon MDP. It is therefore only
reasonable to give Dynamic Programing (DP) a shot, hoping for a more efficient solution. 

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
What we have described so far is nothing more than a continuous MDP, evaluated under a finite-criterion.
As a result, the DP approach to the LQR will be exactly the DP solution of a finite-horizon MDP. The only twist is 
that even though the MDP is continuous, we will have an analytical form for the tail value functions.
{{% /toggle_block %}}

In the following, for any $x\in\mathbb{X}$ and $t \in [T]$ let:
$$
\mathcal{V}^\star\_t(x) := \min\_{u\_{_t}, \ldots, u\_{\_{T-1}}} \sum\_{k=t}^{T-1} 
\left\\{x\_k^\top Q\_k x\_k + u\_k^\top R\_k u\_k  \right\\} + x\_T^\top Q\_T x\_T\\; \quad \text{ with } x\_t = x\\;,
$$
the value function associated to the $t$-tail sub-problem. 
One of the LQR fundamental result is that each $\mathcal{V}^\star\_t(\cdot)$ is a _quadratic_ function which 
coefficient can be computed in a backward fashion. Formally, for any $t\in[T]$ we have 
$
\mathcal{V}^\star\_t(x) = x^\top P\_t x 
$, 
where:
$$
\begin{aligned}
P\_T &= Q\_T \succeq 0 \\\
\text{ for }t<T, \\; P\_t &= Q\_t + A^\top P\_{t+1}A - A^\top P\_{t+1}B\left(R\_t + B^\top P\_{t+1} B\right)^{-1} B^\top P\_{t+1}A \succeq 0\\;.
\end{aligned}
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="expanded"%}}
The proof is done by induction. Initialisation is trivial as it holds by the very definition of $\mathcal{V}\_T^\star(\cdot)$.
Let's now assume that the property holds at some round $t+1$. From the definition of $\mathcal{V}^\star\_t(x)$:
$$
\begin{aligned}
\mathcal{V}^\star\_t(x) &=  \min\_{u\_{_t}} \left\\{ x^\top Q\_t x + u\_t^\top R\_t u\_t  + 
 \min\_{u\_{_t+1}, \ldots, u\_{\_{T-1}}}  \sum\_{k=t+1}^{T-1} 
\left\\{x\_k^\top Q\_k x\_k + u\_k^\top R\_k u\_k  \right\\} + x\_T^\top Q\_T x\_T \right\\} \\;,\\\
&= x^\top Q\_t x + \min\_{u\_{_t}} \left\\{ u\_t^\top R\_t u\_t + \mathcal{V}\_{t+1}^\star(Ax + Bu\_t ) \right\\} \\;,\\\
&\overset{(i)}{=} x^\top Q\_t x + \min\_{u\_{_t}} \left\\{ u\_t^\top R\_t u\_t + (Ax + B u\_t)P\_{t+1}(Ax + B u\_t)\right\\} \\;,\\\
&= x^\top (Q\_t + A^\top P\_{t+1}A) x + \min\_{u\_{_t}} \left(u\_t^\top (R\_t + B^\top P\_{t+1} B)u + 2 u\_t^\top B^\top P\_{t+1}Ax\right)\\;,
\end{aligned}
$$
where $(i)$ uses the induction hypothesis.
The minimum is attained for $u\_t^\star = (R\_t + B^\top P\_{t+1} B)^{-1}B^\top P\_{t+1}Ax$. Replacing in the above yields the desired claim.
We are now only left with proving that $P\_t$ is p.s.d. It is symmetric by construction since by
the induction hypothesis $P\_{t+1}$ is symmetric. Further:
$$
\begin{aligned}
P\_t &= Q\_t + A^\top P\_{t+1}A - A^\top P\_{t+1}B\left(R\_t + B^\top P\_{t+1} B\right)^{-1} B^\top P\_{t+1}A \\;, \\\
& \overset{(i)}{\succeq}  A^\top P\_{t+1}A - A^\top P\_{t+1}B\left(R\_t + B^\top P\_{t+1} B\right)^{-1} B^\top P\_{t+1}A \\;, \\\
&\overset{(ii)}{=} A^\top P\_{t+1}^{1/2}\left(I\_{d\_{\_x}} - P\_{t+1}^{1/2}B^\top\left(R\_t + B^\top P\_{t+1} B\right)^{-1} B^\top P\_{t+1}^{1/2}\right) P\_{t+1}^{1/2} A \\;, \\\
&\overset{(iii)}{\succ} A^\top P\_{t+1}^{1/2}\left(I\_{d\_{\_x}} - P\_{t+1}^{1/2}B^\top B^{-1} P\_{t+1}^{-1} B^{-\top}) B^\top P\_{t+1}^{1/2}\right) P\_{t+1}^{1/2} A \\;, \\\
&= 0\_{d\_{\_x}}
\end{aligned}
$$
where $(i)$ uses $Q\_t \succeq 0$, $(ii)$ the fact that  $P\_{t+1}\succeq 0$ (hence its square-root exists)
and $(iii)$ uses $R\_t \succ 0$. 
{{% /toggle_block %}}


Left: optimal open-loop control and strategy. 
By definition:
$$
\mathcal{V}^\star\_0(x\_0) = \min\_U \mathcal{V}(x\_0, U) \\; .
$$
### Steady-state problem

## Stochastic LQR