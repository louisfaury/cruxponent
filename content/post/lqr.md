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

{{% toggle_block background-color="#FAD7A0" title="Proof"%}}
The proof is done by induction. Initialisation is trivial as it holds by the very definition of $\mathcal{V}\_T^\star(\cdot)$.
Let's now assume that the property holds at some round $t+1$. From the definition of $\mathcal{V}^\star\_t(x)$:
$$
\begin{aligned}
\mathcal{V}^\star\_t(x) &=  \min\_{u\_{_t}} \left\\{ x^\top Q\_t x + u\_t^\top R\_t u\_t  + 
 \min\_{u\_{\_{t+1}}, \ldots, u\_{\_{T-1}}}  \sum\_{k=t+1}^{T-1} 
\left\\{x\_k^\top Q\_k x\_k + u\_k^\top R\_k u\_k  \right\\} + x\_T^\top Q\_T x\_T \right\\} \\;,\\\
&= x^\top Q\_t x + \min\_{u\_{_t}} \left\\{ u\_t^\top R\_t u\_t + \mathcal{V}\_{t+1}^\star(Ax + Bu\_t ) \right\\} \\;,\\\
&\overset{(i)}{=} x^\top Q\_t x + \min\_{u\_{_t}} \left\\{ u\_t^\top R\_t u\_t + (Ax + B u\_t)P\_{t+1}(Ax + B u\_t)\right\\} \\;,\\\
&= x^\top (Q\_t + A^\top P\_{t+1}A) x + \min\_{u\_{_t}} \left(u\_t^\top (R\_t + B^\top P\_{t+1} B)u + 2 u\_t^\top B^\top P\_{t+1}Ax\right)\\;,
\end{aligned}
$$
where $(i)$ uses the induction hypothesis.
The minimum is attained for $u\_t^\star = -(R\_t + B^\top P\_{t+1} B)^{-1}B^\top P\_{t+1}Ax$. Replacing in the above yields the desired claim.
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


But what good is this property when it comes to find an optimal control sequence? It turns out that the sequence 
$(u\_1^\star, \ldots, u\_{N-1}^\star)$ where each $u\_t^\star = (R\_t + B^\top P\_{t+1} B)^{-1}B^\top P\_{t+1}Ax\_t$ solves  the Dynamic Programming objective (cf. proof above) is optimal. 
In other words, playing:
$$u\_t = K\_t x\_t \quad \text{ where }\quad  K\_t = -(R\_t + B^\top P\_{t+1} B)^{-1}B^\top P\_{t+1}A
$$
at every round $t\in\[T\]$ yields an optimal behavior. 

{{% toggle_block background-color="#FAD7A0" title="Proof" %}}
Let $U^\star = (u\_1^\star, \ldots, u\_{N-1}^\star)$ the controls given by the DP protocol.
Notice that:
$$
\begin{aligned}
\mathcal{V}(x\_0, U^\star) &= x\_0^\top Q\_0 x\_0 + \left\\{ (u\_1^\star)^\top R\_1 u\_1^\star + \mathcal{V}\_{1}^\star(Ax\_0 + Bu\_0 ) \right\\}\\;, &\\\
&\overset{(i)}{=} x\_0^\top Q\_0 x\_0 + \min\_{u}\left\\{ u^\top R\_0 u + \mathcal{V}\_{1}^\star(Ax\_0 + Bu ) \right\\}\\;,& \\\
&\leq x\_0^\top Q\_0 x\_0 + u\_1^\top R\_0 u\_0 + \mathcal{V}\_{1}^\star(Ax\_0 + Bu\_0 )\\;, &\text{ for any } u\_0 \\\
&\overset{(ii)}{\leq} x\_0^\top Q\_0 x\_0 + u\_0^\top R\_0 u\_0 +  x\_1^\top Q\_1 x\_1 + u\_1^\top R\_1 u\_1 + \mathcal{V}\_{2}^\star(Ax\_2 + Bu\_2 )\\;, \\\
&\leq \ldots \\;, \\\
&\leq \mathcal{V}(x\_0, U) & \text{for any } U \\; . 
\end{aligned}
$$
Above, we used in $(i)$ the very definition of $u\_t^\star$. The steps $(ii)$ and latter are obtained by
repeating the preceding argument. 
{{% /toggle_block %}}


In the previous approach, we computed optimal controls
$(u\_1^\star, \ldots, u\_{N-1}^\star)$ before-hand. This is called an _open loop_ approach. 
We just found out that each $u\_t^\star$ can be expressed directly as a function
of $x\_t$. We can therefore wait to observe the state $x\_t$ before deciding on our command (_closed loop_).
This is an _action vs. strategy_ distinction.
In deterministic systems the two are equivalent; however, as we'll see shortly, in stochastic settings the open loop approach can be 
arbitrarily sub-optimal. 


Finally, observe how the computation cost's is now being trimmed down to $\mathcal{O}(T)$ -- one of DP main's advantage, 
along with its portability to stochastic settings. 

### Steady-state problem
Let's now part with the finite horizon problem and consider _steady state_ settings. The idea is to
build stabilizing control ($x\_t \to 0$), evaluated under an infinite horizon. We now evaluate a control sequence 
$U = (u\_0, \ldots, u\_t, \ldots)$ under the following criterion:
$$
\mathcal{V}\_\infty(x\_0, U) = \sum_{t=0}^\infty x\_t Q x + u\_t R u\_t \\; .
$$
Note that we know consider fixed state and control cost matrices. Observe also that $\mathcal{V}\_\infty(x\_0, U)$ can
be $+\infty$ for some control. For our minimization problem to be well-defined, we'll need to ensure
that $\mathcal{V}^\infty(x\_0, U)$ is bounded above for at least one control sequence. To this end, the main control theoretic
notion we will need  is called _controllability_. The pair $(A, B)$ is said to be controllable if:
$$
\text{rank}\begin{bmatrix} B & AB & \ldots & A^{d\_x-1}B \end{bmatrix}= d\_x \\; . 
$$
Controllability is sufficient to guarantee _for any_ $x\_0$, it exists $T\in\mathbb{N}$ and $U = (u\_0, \ldots, u\_{T-1})$ such that
$x\_T = 0$ -- ensuring that we can indeed find a control sequence such that $\mathcal{V}\_\infty(x\_0, U)<+\infty$. 

{{% toggle_block background-color="#FAD7A0" title="Proof"%}}
Notice that:
$$
x\_{d\_x} = A^{d\_x} x\_0 + \begin{bmatrix}B & AB & \ldots & A^{d\_x-1}B\end{bmatrix}\begin{pmatrix} u\_{d\_x} \\\ \vdots \\\ u\_0 \end{pmatrix}\\; .
$$
By the controllability assumption, the image of $\begin{bmatrix} B & AB & \ldots & A^{d\_x-1}B \end{bmatrix}$ is $\mathbb{R}^{d\_x}$, which
already concludes the proof -- indeed, we are guaranteed existence of $U$ such that $\begin{bmatrix} B & AB & \ldots & A^{d\_x-1}B \end{bmatrix} U = 
-A^{d\_x} x\_0$ .
{{% /toggle_block %}}


Now, it makes sense to look for an optimal control. Inspired from the finite-horizon setting (and the theory
of average cost MDPs) we will seek for a solution to the following Bellman optimality equation:
$$
\mathcal{V}\_\infty^\star(x) = x^\top Q x + \min\_u\Big\\{ u\top R u + \mathcal{V}\_\infty^\star(Ax + Bu)\Big\\}
\quad \text{ for any } x\in\mathbb{R}^{d\_x}\\; .
$$

{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
The main novelty is that we are now looking for some kind of stationary solution to the DP objective.
Why it yields an optimal strategy is formally proven by repeating the argument we had in the finite-horizon case.
{{% /toggle_block %}}

It is easy to check that $\mathcal{V}\_\infty^\star(x) = x^\top P x$ is such a solution (as a matter of fact, it is the only one), where
the matrix $P\succeq 0$ verifies the so-called _Discrete Algebraic Ricatti Equation_ (DARE):
$$
P = Q + A^\top P A - A^\top P B (R + B^\top P B)^{-1}B^\top P A \\; .
$$
It can be shown that if $(A, B)$ is controllable, the DARE has a unique p.s.d solution. The optimal control
strategy is now a stationary one, and it writes:
$$
u\_t = K x\_t  \quad \text{ where } \quad K = -(R + B^\top P B)^{-1}B^\top P A\\; .
$$

{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
By a fixed-point argument, one can show that under the  controllability assumption,
the matrix $\{P\_0\}$ given by the finite-horizon DP approach
converges to the p.s.d solution of the DARE when the horizon $T\to\infty$. 
{{% /toggle_block %}}
 

## Stochastic LQR

We can now have a quick glance at the stochastic case. The dynamics are now upset by some
noisy disturbance, assumed to follow a fixed Gaussian distribution:
$$
x\_{t+1} = A x\_t + B u\_t + \omega\_t, \quad \text{ where } \omega\_t \overset{\text{i.i.d}}{\sim}
\mathcal{N}(0, \Sigma) \\; .
$$

For simplicity, let's go back to the finite-horizon objective. Instead of looking for an open-loop control
$U = (u\_0, \ldots, u\_{T-1})$ we now explicitly look for good strategies -- or _policies_ $\Pi$, made up of a succesion of mapping  $\pi\_t:\mathcal{X} \mapsto \mathcal{U}$.
An optimal policy $\Pi^\star = (\pi^\star\_0, \ldots, \pi^\star\_{T-1}) $ checks:
$$
\mathcal{V}_{\Pi^\star}(x) = \min\_\Pi \mathbb{E}\left\[\sum\_{t=0}^{T-1} \\{ x\_t Q x\_t + u\_t^\top R u\_t \\} + x\_T^\top Q x\_T \middle\vert x\_0; u\_t = \pi\_t(x\_t) \right\] = \mathcal{V}^\star(x) \\;, 
$$
where $x\_{t+1} = A x\_t + B u\_t + \omega\_t$ and the expectation is taken over all the realisation
of the disturbances $\\{\omega\_t\\}\_t$.


{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
We've skipped quite a few steps when we decided to directly look for deterministic, Markov policies -- and when 
we claimed that our minimization problem admitted a solution ($\min$ versus $\inf$). 
All of those are direct consequence of classical MDP theory -- the LQR simply being a MDP with continuous state and action space. 
{{% /toggle_block %}}


A rather surprising (at least it was for me) is that for the LQR, the optimal policy for stochastic setting
is the same as for deterministic one! This property is known as _certainty equivalence_, and as far as I know, is a singularity of the LQR.
To see why it holds, we can repeat the demonstration we had in the deterministic case. Let's recursively define value functions as follows:
$\mathcal{V}_T^\star = x^\top P\_T x + q\_T$ where $q\_T = 0$, and:
$$
    \mathcal{V}^\star\_t(x) := x^\top Q x + \min\_{u}\mathbb{E}\left\[
\mathcal{V}^\star\_{t+1}(Ax + Bu + \omega\_t) \right\] + q\_{t+1} \\; .
$$
We won't go through the all induction process, but assuming that $\mathcal{V}^\star\_{t+1}(x) = x^\top P\_{t+1} x + q\_{t+1}$
we will obtain by using the fact that $\mathbb{E}\[\omega\_t\] = 0$ that:

$$
\begin{aligned}
    \mathcal{V}^\star\_t(x)  &\overset{(i)}{=} x^\top Q x + x^\top A^\top P\_{t+1} A x + \min\_{u\}\left\\{
u^\top B^\top P\_{t+1}Bu + 2u^\top B^\top P\_{t+1}A x\right\\} + \mathbb{E}\left\[\omega^\top P\_{t+1}\omega\right\] + q\_{t+1} \\; ,\\\
&= x^\top\left( Q + A^\top P\_{t+1} A - A^\top P\_{t+1}B(R + B^\top P\_{t+1}B)^{-1}
B^\top P\_{t+1}A\right) x  + \mathbb{E}\left\[\omega^\top P\_{t+1}\omega\right\] + q\_{t+1}\\; .
\end{aligned}
$$
where in $(i)$ the optimal control is given by no other than $u\_t^\star = -(B^\top P\_t B + R\_t)^{-1}B^\top P\_{t} A x$! 
By some simple linear algrebra:
$$
 \mathbb{E}\left\[\omega^\top P\_{t+1}\omega\right\] =  P\_{t+1}\text{Tr}\left(\mathbb{E}\left\[\omega\omega^\top\right\]\right) = P\_{t+1}\text{Tr}\left(\Sigma\right)\\; .
$$
Hence $\mathcal{V}^\star\_t(x)  = x^\top P\_t x + q\_t$ where $P\_t$ is defined recursively as in the deterministic case, and 
$q\_t = q\_{t+1} + P\_{t+1}\text{Tr}\left(\Sigma\right)$. 

In a few words, stochasticity changes the actual value of the value fonctions, but not the optimal strategy, which still writes:
$$
\pi\_t^\star(x) = K\_t x \\; .
$$