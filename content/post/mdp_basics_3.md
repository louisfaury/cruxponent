+++
author = "Louis Faury"
title = "MDP Fundamentals (3/3)"
date = "2022-10-16"
+++

This blog-post is concerned with computational methods for finding optimal policies. We will
cover Value Iteration (VI) and Policy Iteration (PI), which serve as fundamental building blocks for many modern RL methods.
We will also quickly cover the Generalized Policy Iteration approach, on top of which stand all fancy modern
actor-critic methods.

<!--more-->
<br>

## Reminders
Recall that the optimal discounted cost function $v\_\lambda^\star$ is the unique fixed point of $\mathcal{T}\_\lambda^\star$; 
$$
    v\_\lambda^\star = \mathcal{T}\_\lambda^\star(v\_\lambda^\star) \\;\text{  where  }\\; \mathcal{T}\_\lambda^\star(f) =\max\_{d\in\mathcal{D}^\text{MD}} 
\\{ \mathbf{r}\_{d} + \lambda\mathbf{P}\_{d}\cdot f\\} \\; .
$$
Also keep in mind that an optimal policy can be computed by finding a conserving decision-rule $d^\star$:
$$
    d^\star \in \argmax_{d\in\mathcal{D}^\text{MD}} \\{\mathbf{r}\_{d} + \lambda\mathbf{P}\_{d}\cdot v\_\lambda^\star\\} \\; .
$$
and forming the stationary policy $\pi^\star = (d^\star , d^\star, \ldots)$.

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
Remember that we impose a component-wise partial ordering for vectors. The above maximisation program must therefore be carried out component by component.
{{% /toggle_block %}}
<br>

## Value Iteration

The Value Iteration (VI) is a _value-based_ method. It works by first trying to compute $v\_\lambda^\star$ in a recursive fashion. 
Once a decent approximation is computed, and only then, a near optimal policy will be computed.

### Algorithm
A fundamental property of the optimal value function lies in the fact that it is the unique fixed-point of the _contracting_ 
mapping $\mathcal{T}\_\lambda^\star$. We can use that to our advantage, as it means that applying $\mathcal{T}\_\lambda^\star$
to _any_ vector $f\in\mathcal{V}$ gets us a bit closer to $v\_\lambda^\star$. Actually, we just described the main rationale behind Value Iteration!

Indeed, Value Iteration works by repeatedly applying $\mathcal{T}\_\lambda^\star$ to an initial guess $v\_0\in\mathcal{V}$. That's it!
After a while, we will obtain some estimate $v\_t$ of $v\_\lambda^\star$. We can extract a $v\_t$-improving decision-rule:
$$
d\_t \in \argmax_{d\in\mathcal{D}^\text{MD}} \left\\{ \mathbf{r}\_d + \lambda\mathbf{P}\_d \cdot v\_t\right\\} \\; ,
$$
and form the policy $\pi\_t = (d\_t, d\_t, \ldots)\in\mathcal{S}^\text{MD}$ -- hoping that it will not be too far 
from $\pi^\star$! Actually, we can do better than hope: we'll shortly see that by having $t$ large enough we can arbitrarily control
the sub-optimality gap suffered by $\pi\_t$. Let's first recap the Value Iteration algorithm, this time without vectorial notations for clarity:

{{< pseudocode title="Value Iteration" >}} 
$\textbf{init } v_0\in\mathcal{V}, \text{ max. iteration T}\\$
$\textbf{for } t = 0, \ldots, T-1:\\$
$\qquad\textbf{for }  s\in\mathcal{S}:$
$$
    v_{t+1}(s) = \max_{a\in\mathcal{A}} \Big\{ r(s,a) + \lambda\sum_{s'\in\mathcal{S}} \mathbb{P}(s_{t+1}=s'\vert s_t=s, a_t=a) v_{t}(s')\Big\}
$$
$\qquad\textbf{end for}\\$
$\textbf{end for}\\$
$\text{compute for all } s\in\mathcal{S}:$
$$
d_T(s) \in \argmax_{a\in\mathcal{A}} \Big\{ r(s,a) + \lambda\sum_{s'\in\mathcal{S}} \mathbb{P}(s_{t+1}=s'\vert s_t=s, a_t=a) v_{T}(s')\Big\}
$$
$\textbf{return } \pi_T = (d_T, d_T, \ldots)$
{{< /pseudocode >}}

<br>
<br>

We can also decide to write in a more compact way:
$$
\begin{aligned}
    v\_{t+1} &= \mathcal{T}\_\lambda^\star (v\_{t}),\\; \qquad\forall t=0,\ldots, T-1 \\; , \\\
    d\_T &= \argmax_{d\in\mathcal{D}^\text{MD}} \mathcal{T}_\lambda^d(v\_{T}) \\; .
\end{aligned}
$$

### Convergence properties
The first important property of the VI algorithm is that its value estimate $\\{ v\_t \\}\_t$ converge to the optimal
value function:
$$
    v\_t \overset{t\to\infty}{\longrightarrow} v\_\lambda^\star\\; .
$$
That's actually a by-product of Banach's fixed-point theorem, which was used to prove the very
existence of $v\_\lambda^\star$! In that sense, you can consider that VI is nothing more than a glorified **fixed-point algorithm**. 


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof is actually just a copy-paste of the Banach fixed-point theorem's demonstration (without the completeness and Cauchy sequences part). Let's repeat it, just for kicks.
The contraction property of $\mathcal{T}_\lambda^\star$ (w.r.t. the $\ell\_\infty$ norm) is key.  

$$
\begin{aligned}
\\| v\_{t+1} - v\_\lambda^\star \\|\_\infty &= \\| \mathcal{T}\_\lambda^\star(v\_{t+1}) - v\_\lambda^\star \\|\_\infty \\;, &(v\_{t+1}=\mathcal{T}\_\lambda^\star(v\_{t})) \\\
&= \\|\mathcal{T}\_\lambda^\star(v\_{t+1}) - \mathcal{T}\_\lambda^\star(v\_\lambda^\star) \\|\_\infty \\;, &(v\_{\lambda}^\star=\mathcal{T}\_\lambda^\star(v\_{\lambda}^\star)) \\\
&\leq \lambda \\| v\_{t} - v\_\lambda^\star\\|\_\infty  \\;, \\\
&\leq \ldots \\;, \\\
&\leq \lambda^t \\| v\_0 - v\_\lambda^\star \\|\_\infty \\; .
\end{aligned}
$$
Since $\lambda<1$ we therefore have that $\lim\_{t\to\infty} \\| v\_{t+1} - v\_\lambda^\star \\|\_\infty = 0$.
{{% /toggle_block %}}

If $T$ is large enough, then $v\_T$ will be relatively close to $v\_\lambda^\star$, and the
$v\_T$-improving policy should be a decent one. This is formalised below, where we establish
that we can actually make VI output policies that are arbitrarily close to being optimal. The symbol ${\small\blacksquare}$ denotes 
numerical constants, removed to reduce clutter.

{{< boxed title="$\varepsilon$- Sub-Optimal Policies via VI" >}} 

$\qquad\qquad\qquad\qquad\qquad\qquad\qquad\;
\text{Let } \varepsilon>0. \text{ With } T ={\small\blacksquare}\log(\varepsilon) / \log(\lambda) \text{ we have:}$
$$
    v_\lambda^\star -\varepsilon \leq v_\lambda^{\pi_T} \leq v_\lambda^\star \; . 
$$
{{< /boxed >}}
Observe that $T$ grows only with $\log(\varepsilon)$. This is a result of VI's _[linear convergence](https://en.wikipedia.org/wiki/Rate_of_convergence#:~:text=iterative%20methods%20subsection-,Convergence%20definitions,-Order%20estimation)_.

{{< image src="/vi.png" width="500px" align="center">}}
<br>

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Observe that:
$$
\begin{aligned}
\\| v\_\lambda^{\pi_T} - v\_\lambda^\star \\|\_\infty &\leq \\|v\_\lambda^{\pi_T} - v\_{T}\\|\_\infty + 
\\| v\_T - v_\lambda^\star\\|\_\infty\\; .
\end{aligned}
$$
Let us deal with the first term in the above equation's r.h.s;
$$
\begin{aligned}
    \\|v\_\lambda^{\pi\_T} - v\_{T}\\|\_\infty &\leq \\|v_\lambda^{\pi\_T} - \mathcal{T}\_\lambda^\star(v\_{T})\\|\_\infty + 
        \\| \mathcal{T}\_\lambda^\star(v\_{T}) -  v\_{T}\\|\_\infty \\;, \\\ 
    &\leq \|\\mathcal{T}\_\lambda^{d_T}(v\_\lambda^{\pi\_T}) - \mathcal{T}\_\lambda^\star(v\_T)\||\_\infty + 
        \lambda \\| v\_T -  v\_{T-1}\\|\_\infty &(\mathcal{T}\_\lambda^{d\_T}(v\_\lambda^{\pi\_T})=v\_\lambda^{\pi\_T})\\;, \\\
&=\\| \mathcal{T}\_\lambda^{d\_T}(v\_\lambda^{\pi\_T}) - \mathcal{T}\_\lambda^{d\_T}(v\_{T})\\|\_\infty +    \lambda \\| v\_T -  v\_{T-1}\||\_\infty &(\text{c.f. def of } d\_T) \\;, \\\
		&\leq \lambda \\| v\_\lambda^{\pi\_T} - v\_T \\|\_\infty +\lambda \\| v\_T -  v\_{T-1}\\|\_\infty \\; , \\\
\end{aligned}
$$
proving that $|\|v\_\lambda^{\pi_T} - v\_{T}\||\_\infty  \leq \lambda(1-\lambda)^{-1}\\| v\_T -  v\_{T-1}\\|\_\infty$.
As for the second term we have:
$$
\begin{aligned}
\|| v\_{T}- v\_\lambda^\star \||\_\infty &\leq  \sum\_{i=T}^\infty \|| v\_{i+1} - v\_i\||\_\infty &(\text{triangle inequality}) \\;, \\\
		&\leq  \|| v\_{T} - v\_{T-1}\||\_\infty \sum\_{i=1}^\infty \lambda^i\\;, \\\
		&\leq \frac{\lambda}{1-\lambda}\|| v\_{T} - v\_{T-1}\||\_\infty \\; ,
\end{aligned}
$$
By plugging everything together :
$$	
\begin{aligned}
    \|| v\_\lambda^{\pi\_T} - v\_\lambda^\star \\|\_\infty  &\leq \frac{2\lambda}{1-\lambda} \\|v\_{T} - v\_{T-1}\\|\_\infty\\;, \\\
    &=  \frac{2\lambda}{1-\lambda} \||\mathcal{T}\_\lambda^\star(v\_{T-1}) - \mathcal{T}\_\lambda^\star(v\_{T-2})\\|\_\infty \\;, \\\
    &\leq   \frac{2\lambda^2}{1-\lambda} \\|v\_{T-1} - v\_{T-2}\\|\_\infty\\;, \\\
&\leq \ldots \\\
&\leq \frac{2\lambda^T}{1-\lambda}  \\|v\_{1} - v\_{0}\\|\_\infty \\; .
\end{aligned}
$$
Solving for $ \|| v\_\lambda^{\pi\_T} - v\_\lambda^\star \\|\_\infty \leq  \varepsilon$ yields the desired result. 
{{% /toggle_block %}}


## Policy Iteration

While the VI algorithm can be thought of as an application of a general approach for discovering fixed-point, 
the Policy Iteration (PI) algorithm instead leverages the very structure of MDPs. 

### Algorithm 
PI is a policy-based method; instead of refining approximations of the optimal discounted cost $v_\lambda^\star$,
it maintains a sequence of decision-rules $\\{d\_t\\}\_t$ (or equivalently a sequence of stationary policies) approaching the optimal decision-rule $d^\star$. 
The main idea to achieve this is to repeatedly "improve" over the last policy; formally at round $t$ and by denoting $\pi_t=(d_t, d_t, \ldots)$, 
the PI algorithm defines the next iterate as:
$$
    d\_{t+1} \in \argmax\_{d\in\mathcal{D}^\text{MD}}\left\\{ \mathbf{r}\_{d} + \lambda\mathbf{P}\_{d}\cdot v\_\lambda^{\pi\_t}\right\\} \\; .
$$ 
The PI algorithm therefore alternates between two independent mechanisms: _(1)_ the _evaluation_ the current policy,
meaning computing the value of the current policy $\pi\_t = (d\_t, d\_t, \ldots)$ ad _(2)_ the _improvement_ of said policy by computing 
$d\_{t+1}$ as detailed above. To match with our description of VI, we present below a version of the pseudocode where we start by
policy improvement. 

{{< pseudocode title="Policy Iteration" >}} 
$\textbf{init } v_0\in\mathcal{V}, \text{ max. iteration T}\\$
$\textbf{for } t = 0, \ldots, T-1:\\$
$\qquad \text{\color{gray}[Policy Improvement]}\\$
$\qquad\textbf{for }  s\in\mathcal{S}:$
$$
    d_{t+1}(s) \in \argmax_{a\in\mathcal{A}} \Big\{ r(s,a) + \lambda\sum_{s'\in\mathcal{S}} \mathbb{P}(s_{t+1}=s'\vert s_t=s, a_t=a) v_{t}(s')\Big\}
$$
$\qquad\textbf{end for}\\$
$\qquad\text{Form } \pi_{t+1} = (d_{t+1}, d_{t+1}, \ldots).\\$
$\qquad \text{\color{gray}[Policy Evaluation]}\\$
$\qquad \text{Compute } v_{t+1} = v_\lambda^{\pi_{t+1}}, \text{ by solving the system }:\\$
$$
    v_{t+1} = \mathcal{T}_\lambda^{d_{t+1}} (v_{t+1}) \; .
$$
$\textbf{end for}\\$
$\textbf{return } \pi_T = (d_T, d_T, \ldots)$
{{< /pseudocode >}}

<br>
<br>

The policy evaluation step is left here somewhat blurry. We know that $v_\lambda^{\pi_{t+1}}$ is the unique fixed-point of
$\mathcal{T}\_\lambda^{d_{t+1}}$, hence the only requirement we ask is that $v_{t+1}$ is a solution of that fixed-point equation. 
In finite MDPs, we saw that a straight-forward solution is to
write directly:
$$
    v\_{t+1} = (\mathbf{I}-\lambda\mathbf{P}\_{d\_{t+1}})^{-1}\mathbf{r}\_{d\_{t+1}} \\; .
$$

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
This brute-force inversion is expensive when the MDP is large -- indeed, it typically cost $\mathcal{O}(\mathcal{S}^3)$
to invert that matrix. It therefore quickly becomes prohibitively expensive as the number of states gets large. We'll
see shortly how one can relax this operation in order to reduce the computational burden. 
{{% /toggle_block %}}

### Convergence properties
	
A core property of PI lies in its _monotonic improvement._ Indeed, we first make the observation that 
PI produces a sequences of non-decreasing discounted costs. 

{{< boxed title="PI Monotic improvement" >}} 

$\qquad\qquad\qquad\qquad\qquad\qquad\;
\text{Let } (\pi_1, \ldots, \pi_T) \text{ the sequence of policies produced by PI. Then: }$
$$
    v_\lambda^{\pi_1} \leq v_\lambda^{\pi_2} \leq \ldots \leq v_\lambda^{\pi_T}\; .
$$
{{< /boxed >}}

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
By definition of $d\_{t+1}$ we have:
$$
\begin{aligned}
\mathbf{r}\_{d\_{\textnormal t+1}} + \lambda\mathbf{P}\_{d\_{\textnormal t+1}}\cdot v\_\lambda^{\pi\_t} &= \mathcal{T}\_\lambda^\star (v\_t)\\;, \\\
&= \mathcal{T}\_\lambda^\star (v\_\lambda^{\pi\_t}) \\;,\\\
&= \max\_d \mathcal{T}\_\lambda^d (v\_\lambda^{\pi_{t}})\\;,\\\
&\geq \mathcal{T}\_\lambda^{d\_t} (v\_\lambda^{\pi\_{t}})\\;,\\\
&= v\_\lambda^{\pi_{t}}\\; .
\end{aligned}
$$
Hence $\mathbf{r}\_{d\_{\textnormal t+1}} \geq (\mathbf{I} -  \lambda\mathbf{P}\_{d_{\textnormal t+1}})  v_\lambda^{\pi\_{t}}$.
By positivity of $(\mathbf{I} -  \lambda\mathbf{P}\_{d_{\textnormal t+1}})$ we obtain:
$$
\begin{aligned}
    v\_\lambda^{\pi_{t+1}} =  (\mathbf{I} -  \lambda\mathbf{P}\_{d\_{\textnormal t+1}})^{-1}\mathbf{r}\_{d\_{\textnormal t+1}} 
\geq v\_\lambda^{\pi\_t}\\; .
\end{aligned}
$$
Hint: For that last step, recall that since $\mathbf{P}\_{d\_{\textnormal t+1}}$ is a stochastic matrix one has $ (\mathbf{I} -  \lambda\mathbf{P}\_{d\_{\textnormal t+1}})^{-1} = 
\sum\_{k=0}^\infty \lambda^k \mathbf{P}\_{d\_{\textnormal t+1}}^k$.
{{% /toggle_block %}}
	


It turns that in our finite setting this is enough to guarantee the convergence of PI in a _finite number of iterations_. 
Indeed, in the finite case there exists only a finite number of decision-rules (precisely $\vert\mathcal{A}\vert^{\vert\mathcal{S}\vert}$ many ones) 
and the PI algorithm exhaust them in increasing order of their discounted return.
Note that no loop over the policies can arise since the equality case 
$v\_\lambda^{\pi\_{t+1}} \leq J\_\lambda^{\pi\_{t}}$ implies that:
$$
		\mathcal{T}\_\lambda^\star(v\_\lambda^{\pi\_{t}}) = v\_\lambda^{\pi\_{t}}\\; ,
$$
and therefore that $v\_\lambda^{\pi\_{t}}=v\_\lambda^\star$ -- meaning that the PI algorithm will return $d^\star$ at the following iteration. 


{{< image src="/pi.png" width="450px" align="center">}}
<br>

This finite-time convergence of course does not extend to countable MDPs, where the number of policies is (countably) infinite.
It's however fairly straightforward to also establish convergence rates for the PI algorithm; one will find out that, 
as VI, it enjoys linear convergence.

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
Even in countable and continuous settings, Policy Iteration is often considered as faster than Value Iteration.
It is the _a priori_ preferred solution in many cases of practical relevance. This qualitative preference can be made
rigorous; under a few simplifying assumptions one can prove that PI's convergence rate is better than VI's.
{{% /toggle_block %}}


## Generalised Policy Iteration
### Computational Costs
We have so far avoided any discussion around computation cost. It's
fairly easy to establish that running each algorithm for $T$ steps roughly requires the following number of operations:
$$
\begin{aligned}
\text{V.I } \quad \longrightarrow \quad &\mathcal{O}(T\vert\mathcal{S}\vert^2\vert\mathcal{A}\vert) \\;,\\\
\text{P.I } \quad \longrightarrow \quad & \mathcal{O}(T\vert\mathcal{S}\vert^2\vert\mathcal{A}\vert + T \vert \mathcal{S} \vert^3)\\;.
\end{aligned}
$$
As its rather common to have $\vert \mathcal{S} \vert \gg \vert \mathcal{A} \vert$ the computational cost
of PI is often dominated by its policy evaluation step which requires around $\mathcal{O}(\vert \mathcal{S} \vert^3 $) operations.
On the other hand, whenever $\vert \mathcal{A} \vert$ grows large, both algorithms see there efficiency drop as improving
policies or value functions would require a linear scan of the action space. The problem is even tougher for continuous
action spaces (_e.g._ $\mathcal{A} = \mathbb{R}$) since then, at each step and for each state, one has to solve a continuous, 
potentially non-convex optimisation problem. 

### Rationale
The Generalised Policy Iteration (GPI) framework allows us to overcome such computational limitations, while retaining
some of the enjoyable convergence properties of VI and PI. The idea is quite simple. Instead of alternating between
fully complete Policy 
Evaluation and Policy Improvement steps, one can decide to solve each of them only _approximately_ before switching to the other.

For concreteness, let's first focus on the Policy Evaluation step. In the following let $\pi$ some arbitrary 
stationary policy. It's important to realise than solving the system 
$v\_\lambda ^\pi = \mathcal{T}\_\lambda^\pi(v\_\lambda ^\pi)$ can be done several ways -- not only by inverting some large matrix.
This is a fixed-point equation for the mapping $\mathcal{T}\_\lambda^\pi$, which happens to be contracting. 
For the very same reasons that VI's iterates converge to $v\_\lambda^\star$, the sequence:
$$
    v\_{t+1} = \mathcal{T}\_\lambda^\pi(v\_t)
$$
will check $v\_t \overset{t\to\infty}{\longrightarrow} v\_\lambda^\pi$. Such a procedure perfectly fits the GPI framework.
Indeed, instead of bringing this iterative process to full convergence, we could decide to
stop prematurely and pass some $v\_T$ as an approximation of $v\_\lambda^\pi$. If this approximation is reasonable, 
overall the process will still output gracefully improving policies.

Similarly, the Policy Improvement step can be degraded. Instead of finding a $v\_\lambda^{\pi\_t}$-improving policy, it could 
be enough to find $d\_{t+1}$ such that for all $s\in\mathcal{S}$:
$$
\begin{aligned}
r(s,d\_{t+1}(s)) + \lambda\sum\_{s'\in\mathcal{S}} \mathbb{P}(s\_{t+1}=s'\vert s_t=s, a\_t=d\_{t+1}(s)) v\_{t}(s')&\\\
&\geq \\\ 
&r(s,d\_{t}(s)) + \lambda\sum\_{s'\in\mathcal{S}} \mathbb{P}(s\_{t+1}=s'\vert s\_t=s, a\_t=d\_t(s)) v\_{t}(s')
\end{aligned}
$$



### Example
#### Value Iteration
Value Iteration is actually only a special case of GPI. Indeed, if one decides to follow only 1-step of the iterative 
Policy Evaluation procedure we just described, we then have
$$
\begin{aligned}
    v\_{t+1} &= \mathcal{T}\_\lambda^{d\_{t+1}}(v\_t) \\;,\\\
    &=\mathcal{T}\_\lambda^{\star}(v\_t)  \\;,
\end{aligned}
$$
since by definition 
$d\_{t+1}(s) \in \argmax\_{a\in\mathcal{A}} \Big\\{ r(s,a) + \lambda\sum\_{s'\in\mathcal{S}} \mathbb{P}(s\_{t+1}=s'\vert s\_t=s, a\_t=a) v\_{t}(s')\Big\\}$.

#### Actor-Critic Algorithms
Most deep-learning based actor-critic algorithms (_e.g._ PPO, SAC) rely on the GPI framework. Parametric policies
$\pi\_\theta$ are improved via gradient-based optimisation (thanks to some [policy gradient](../policy_gradient) estimates)
while another parametric function $v\_\phi$ is trained to estimate $v\_\lambda^{\pi\_\theta}$ via, _e.g._
Bellman residual minimisation:
$$
\phi \in\argmin \sum\_{s} \left(v\_\phi(s) - r(s, a) - \lambda v\_\phi(s')\right)^2\\;, \qquad a\sim\pi_\theta(s), \\;s'\sim p\_t(\cdot\vert s, a)\\; .
$$

#### Others
Without jumping all the way to modern deep-learning approaches, several mechanisms have been devised to perform each step (evaluation and improvement) as quickly as possible while retaining most
of the PI's performances (_e.g._ Prioritised Value Iteration).
The interested reader is referred to \[[Mausam & Kolobov, 2012](https://link.springer.com/book/10.1007/978-3-031-01559-5#toc)\] for a detailed review.

## Resources
The description of VI and PI is a condensed version of [[Puterman. 94, Chapter 6](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887)]





