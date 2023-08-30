+++
author = "Louis Faury"
title = "MDP Fundamentals (2/3)"
date = "2022-10-04"
+++

In this second blog post of the series, we cover the problem of value prediction and control in finite discounted MDPs.
This will lead us in particular to review the Bellman prediction and control equations, and to a fundamental theorem of MDPs: stationary
policies are enough for optimal control!
<!--more-->



{{< boxed title="" >}}
Before diving in, make sure you've read the previous blog-post on MDPs. Recall that we use vectorial notations,
and adopt the usual partial ordering over $\mathbb{R}^n$:
$$
 x \geq y \Longleftrightarrow x_i \geq y_i \; \text{ for all } i\in\{1,\ldots, n\} \; .
$$
{{< /boxed >}}

 

Our ultimate goal here is to derive some _characterisation_ of an optimal policy for a discounted MDP. Said optimal policy 
would check:
$$
\pi^\star \in \argmax\_{\pi \in \Pi^\text{HR}} v\_\lambda^\pi(s) \text{ for all } s \in\mathcal{S}	 .
$$
We will use the shorthand
$v\_\lambda^\star:=v\_\lambda^{\pi^\star}$ to denote the optimal value function, which therefore checks:
$$
v\_\lambda^\star = \max_{\pi \in \Pi^\text{HR}} v\_\lambda^\pi\\; .
$$


## Warm-up: Policy Evaluation
This first part is concerned with _policy evaluation_: given some stationary policy $\pi\in\mathcal{S}^\text{MD}$,
what recurrence property its discounted value $v\_\lambda^\pi$ satisfies? 
This question can at first seem like a side-quest on our path (after all, we are mostly concerned with optimal policies, 
not just any policy). It will however serve as a gentle introduction for the rest of the blog-post.

Let's dive in with the first technical result, which establishes the **Bellman prediction equations**. 
Let $\pi=(d, d, \ldots)\in\mathcal{S}^\text{MD}$ a stationary policy. 
Then, its expected discounted value $v\_\lambda^\pi$ is the _unique_ bounded function from $\mathcal{S}$ to $\mathbb{R}$ which satisfies for all $s\in\mathcal{S}$;

$$ 
 v\_\lambda^\pi(s) =  r(s, d(s)) + \lambda\mathbb{E}\big[v\_\lambda^\pi(s')\big]  \text{  where } s'\sim \mathcal{P}\_s^{d(s)}(\cdot)\\; .
$$


In our discrete setting we can naturally rewrite this using sums: 
$$
\begin{aligned}
		\\; v\_\lambda^\pi(s) &= r(s, d(s)) + \lambda\sum_{s'\in\mathcal{S}} 
v\_\lambda^\pi(s')\mathbb{P}\left(s\_{t=1}=s'\middle\vert s\_t=s, a\_t=d(s)\right)\\; , 
\end{aligned}
$$
and finally have the very compact vector notation when $\mathcal{S}$ is finite:
$$ 
v\_\lambda^\pi = \mathbf{r}\_{d} + \lambda \mathbf{P}\_{d}v\_\lambda^\pi\\; .
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
We can use the vector notation for a short proof (for the more general countable case the proof is also a simple exercice of conditioning). 
Let $\pi = (d, \ldots) \in\mathcal{S}^\text{MD}$. From the vectorial writing of the value function:
$$
\begin{aligned}
			v\_\lambda^\pi &= \sum_{t=1}^\infty \lambda^{t-1}\mathbf{P}\_{\pi}^{t}\mathbf{r}\_{d}\\\
			& \overset{(i)}{=} \mathbf{r}\_{d} + \sum_{t=2}^\infty \lambda^{t-1}\left(\prod_{i=2}^t \\mathbf{P}\_{d}\right)\mathbf{r}\_{d} \\\
			&=  \mathbf{r}\_{d} + \lambda \\mathbf{P}\_{d} \sum_{t=2}^\infty \lambda^{t-2}\left(\prod_{i=2}^{t-1} \mathbf{P}\_{d}\right)\mathbf{r}\_{d} \\\
			 &=  \mathbf{r}\_{d} + \lambda \\mathbf{P}\_{d} \sum_{t=1}^\infty \lambda^{t-1}\left(\prod_{i=2}^{t} \mathbf{P}\_{d}\right)\mathbf{r}\_{d}
			 =   \mathbf{r}\_{d} + \lambda \mathbf{P}\_{d} v_\lambda^\pi \\; .
\end{aligned}
$$
In $(i)$ we used the fact that $\mathbf{P}\_{\pi}^{t} = \prod_{i=2}^t \\mathbf{P}\_{d}$, which we proved in the previous post.
This establishes that $v_\lambda^\pi$ satisfies the claimed identity. The unicity of the solution is rather direct once on has noticed that the vector notation yields:
$$
(\mathbf{I}\_n - \lambda\mathbf{P}\_{d}) v_\lambda^\pi = \mathbf{r}\_{d}\\; .
$$
Because $\mathbf{P}\_{d}$ is a stochastic matrix $(\mathbf{I}_n - \lambda\mathbf{P}\_{d})$ is invertible (recall that $\lambda<1$ and think about the possible eigen-values of a stochastic matrix). Therefore, the solution
$v\_\lambda^\pi$ of this linear system is unique.
{{% /toggle_block %}}


In the finite case the above proof yields a neat characterisation of the discounted expected cost of a stationary policy:
$$
		v\_\lambda^\pi = (\mathbf{I}\_n - \lambda\mathbf{P}\_{d}) ^{-1}\mathbf{r}\_{d}\\; .
$$
	
There is yet one last way of writing the Bellman prediction equation which will prove useful. For any policy
$\pi=(d, \ldots) \in\mathcal{S}^\text{MR}$, 
let us introduce the policy evaluation operator $\mathcal{T}\_\lambda^\pi$:

$$ 
\begin{aligned}
\mathcal{T}\_\lambda^d : \mathcal{V}&\mapsto \mathcal{V} \\\
			f &\mapsto \mathcal{T}\_\lambda^d(f) \\; \text{ where } \\; \mathcal{T}\_\lambda^d(f)(s):=  r(s, d(s)) + \lambda\mathbb{E}^{d}\_s\big[f(s\_{t+1})\big] \\; .
\end{aligned}
$$


We can therefore neatly write that the discounted expected value of some policy is the unique fixed-point of its associated policy evaluation operator:

{{< boxed title="Bellman prediction equations" >}}
$$
{
	 	v_\lambda^\pi = \mathcal{T}_\lambda^d(v_\lambda^\pi) \; .
}
$$
{{< /boxed >}}


## Optimal Control

We can now move on to the characterisation of the optimal expected discounted cost $v_\lambda^\star$. 
The Bellman optimality equations for the discounted case draw their intuition from the finite horizon case. 
Since we haven't covered it explicitly, we will simply claim their discounted counterpart and proceed with proving 
that they indeed characterise optimal policies.

Formally, we say that $f\in\mathcal{V}$ satisfies the Bellman optimality equations for discounted MDPs if:
	
$$ 
\begin{aligned}
\forall s\in\mathcal{S}, \\; f(s) = \max\_{a\in\mathcal{A}} \left\\{ r(s,a) + \lambda\mathbb{E}\left[f(s\_{t+1})\middle\vert s\_t=s, a\_t=a\right] \right\\}\\; .
\end{aligned}
$$

We can also write the above identity as:
$
f  = \mathcal{T}\_\lambda^\star(f)
$,
where we defined the **Bellman optimality operator**:
$$
\begin{aligned}
\mathcal{T}\_\lambda^\star : \mathcal{V}&\mapsto \mathcal{V} \\\
f &\mapsto \mathcal{T}\_\lambda^\star(f) :=  \max\_{d\in\mathcal{D}^\text{MD}} \mathcal{T}\_\lambda^d(f)\\; .
\end{aligned} 
$$
        
        
It turns out that if $f\in\mathcal{V}$ satisfies this identity, then $f=v_\lambda^\star$!
In others word, the optimal expected discounted cost is the _unique_ solution to Bellman's optimality equations - or 
equivalently the unique fixed point of the operator $\mathcal{T}\_\lambda^\star$.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Let $f\in\mathcal{V}$ such that for all $s\in\mathcal{S}$. The proof goes by establishing the following claims:
- $f \leq \mathcal{T}\_\lambda^\star(f)$ then $f \leq v\_\lambda^\star$,
- $f \geq \mathcal{T}\_\lambda^\star(f)$ then $f \geq v\_\lambda^\star$,
- $f = \mathcal{T}\_\lambda^\star(f)$ then $f = v\_\lambda^\star$,

where we again adopt the classical partial ordering over vectors. Let's first prove the first claim. 
Let $f(s) \geq \mathcal{T}\_\lambda^\star(f)$ for all $s\in\mathcal{S}$. Then for any $d\in\mathcal{D}^\text{MD}$:
$$
\begin{aligned}
		f &\leq \mathbf{r}\_{d} + \lambda\mathbf{P}\_{d}f\\; , \\\
\Longleftrightarrow &(\mathbf{I}\_n-\lambda\mathbf{P}\_{d})f \leq  \mathbf{r}\_{d}\\; .
\end{aligned}
$$
Using the fact that $(\mathbf{I}\_n-\lambda\mathbf{P}\_{d})^{-1} = \sum\_{k=0}^\infty \lambda^k\mathbf{P}\_{d}^k $ which only has positive entries, we obtain that;
$$
	\begin{aligned}
		f &\leq (\mathbf{I}\_n-\lambda\mathbf{P}\_{d})^{-1}( \mathbf{r}\_{d})\\\
		&= v\_\lambda^{\pi} \\; ,
	\end{aligned}
$$
for any $\pi = (d, \ldots) \in\mathcal{S}^\text{MD}$ (we just used the characterisation of a stationary policy).
Therefore: 
$$f \leq \max\_{\pi\in\mathcal{S}^\text{MR}} v\_\lambda^\pi\leq \max\_{\pi\in\Pi^\text{HR}} v\_\lambda^\pi= v\_\lambda^\star\\; . $$

We now prove the second claim. Observe that if $f \leq \mathcal{T}\_\lambda^\star(f)$ then for any $d\_1, d\_2, \ldots \in\mathcal{D}^\text{MR}$:
$$
\begin{aligned}
f &\leq \mathbf{r}\_{d\_1} + \lambda\mathbf{P}\_{d\_1} f \\; \\\
&\leq \mathbf{r}\_{d\_1} + \lambda\mathbf{P}\_{d\_1}\mathbf{r}\_{d\_2} + \lambda^2 \mathbf{P}\_{d\_2} f\\;, \\\
&\leq \ldots \\\
&\leq \sum\_{t=1}^\tau \lambda^{t-1}\mathbf{P}\_{\pi}^t\mathbf{r}\_{d\_t} + \lambda^\tau \mathbf{P}\_{\pi}^\tau\mathbf{r}\_{d\_\tau}\\; .
\end{aligned}
$$
Since $\lambda < 1$ and $\mathbf{P}\_{\pi}^\tau$ is stochastic, we have $\lim_{\tau\to\infty}\lambda^\tau \mathbf{P}\_{\pi}^\tau\mathbf{r}\_{d\_\tau} = 0$ 
and therefore:
$$
f \leq \sum\_{t=1}^\infty \lambda^{t-1}\mathbf{P}\_{\pi}^t\mathbf{r}\_{d\_t} = v\_\lambda^\pi\\; ,
$$
for any $\pi = (d\_1, d\_2 \ldots, ) \in\Pi^\text{MR}$. Hence $f\leq \max\_{\pi\in\Pi^\text{MR}}v\_\lambda^\pi =  \max\_{\pi\in\Pi^\text{HR}}v\_\lambda^\pi$. 
The third claim is easily obtained by combining the first two.
{{% /toggle_block %}}
         
	
We are now left with proving that such a solution exists -- equivalently, that $\mathcal{T}\_\lambda^\star$ 
indeed admits a fixed point. The technical tool to achieve this is the [Banach's fixed point theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem). 
The key claim of this theorem is that if some operator acting on a space _without holes_ (a.k.a complete) is contracting, then it must admit a unique fixed point. 
Turns out, the Bellman optimality operator is contracting. Indeed, for all $f, g\in\mathcal{V}$:
$$
\left\lVert \mathcal{T}\_\lambda^\star(f) -  \mathcal{T}\_\lambda^\star(g) \right\rVert_\infty \leq \lambda \left\lVert f - g\right\rVert_\infty <  \left\lVert f - g\right\rVert_\infty\\\; .
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}

Observe that for any $d\in\mathcal{D}^\text{MD}$ the operator $\mathcal{T}\_\lambda^\pi$ is itself contracting, since for all $f,g\in\mathcal{V}$:
$$
\begin{aligned}
		\\| \mathcal{T}\_\lambda^d(f) - \mathcal{T}\_\lambda^d(g)\\|\_\infty &= \\| \lambda \mathbf{P}\_{d}(f-g)\\|\_\infty\\\; , \\\
		&\leq \lambda \\|  \mathbf{P}\_{d}\\|\_\infty\\|f-g\\|\_\infty\\ ,&(\text{operator norm property})\\\
		&\leq \lambda\\|f-g\\|\_\infty\\; . & (\mathbf{P}\_{d} \text{ stochastic})
\end{aligned}
$$
The fact that $\mathcal{T}\_\lambda^\star = \inf{d\in\mathcal{D}^\text{MD}} \mathcal{T}\_\lambda^d$ finishes the proof; indeed for all $f, g\in\mathcal{V}$ 
let $d\_f = \argmax \mathcal{T}\_\lambda^d f$. Then:
$$
	\begin{aligned}
		 \mathcal{T}\_\lambda^\star(f) - \mathcal{T}\_\lambda^\star(g) &= \mathcal{T}\_\lambda^{d\_f}(f) - \mathcal{T}\_\lambda^\star(g)\\\
		 &\leq \mathcal{T}\_\lambda^{d\_f}(f) -\mathcal{T}\_\lambda^{d\_f}(g)\\\; .
\end{aligned}
$$
This mean that for any $s\in\mathcal{S}$ we have $$\mathcal{T}\_\lambda^\star(f)(s) - \mathcal{T}\_\lambda^\star(g)(s) \leq \mathcal{T}\_\lambda^{d\_f}(f)(s) -\mathcal{T}\_\lambda^{d\_f}(g)(s)$$.
Assuming for now that $\mathcal{T}\_\lambda^\star(f)(s) - \mathcal{T}\_\lambda^\star(g)(s)\geq 0$ yields that:
$$
\begin{aligned}
0 \leq \mathcal{T}\_\lambda^\star(f)(s) - \mathcal{T}\_\lambda^\star(g)(s)\leq \mathcal{T}\_\lambda^{d\_f}(f)(s) -\mathcal{T}\_\lambda^{d\_f}(g)(s) \\;,\\\
\end{aligned}
$$
Therefore:
$$
\begin{aligned}
\mathcal{T}\_\lambda^{d\_f}(f)(s) -\mathcal{T}\_\lambda^{d\_f}(g)(s) &\leq \vert \mathcal{T}\_\lambda^{d\_f}(f)(s) -\mathcal{T}\_\lambda^{d\_f}(g)(s)\vert \\; , \\\
&\leq \lVert  \mathcal{T}\_\lambda^{d\_f}(f) -\mathcal{T}\_\lambda^{d\_f}(g)\rVert \\; ,\\\
&\le \lambda \\| f -g \\| \\; .
\end{aligned}
$$
This yields the desired result when $ \mathcal{T}\_\lambda^\star(f)(s) \geq \mathcal{T}\_\lambda^\star(g)(s)$. 
The completementy case is easily dealt by a reverse argument -- studying $ \mathcal{T}\_\lambda^\star(g) - \mathcal{T}\_\lambda^\star(f)$ instead of  $ \mathcal{T}\_\lambda^\star(f) - \mathcal{T}\_\lambda^\star(g)$. 
{{% /toggle_block %}}


         
This completes this section. We have proven that $\mathcal{T}\_\lambda^\star$ admits a unique fixed point, which is the optimal value function.
The optimal value function $v\_\lambda^\star$ is therefore the _only_ function which satisfies the Bellman's optimality equations.

{{< boxed title="Bellman control equations" >}}
$$
{
	 	v_\lambda^\star = \mathcal{T}_\lambda^\star(v_\lambda^\star) = \max_{d\in\mathcal{D}^\text{MD}} \mathcal{T}_\lambda^d (v_\lambda^\star)\; .
}
$$
{{< /boxed >}}


{{< image src="/mdp_fp.png" width="500px" align="center">}}
<br>

## Optimal Policies
When it comes to optimality we have so far focused on history-dependent randomised policies, then on Markov randomised policies. 
From a computational standpoint it would be desirable to be able to restrict this policy space even further--and focus 
only on stationary policies. Indeed, a stationary policy is made up of a single decision-rule and 
is therefore easy to store and update. 
For this restriction to be principled we are missing one fundamental result which ensures that the optimal discounted cost is attained by a stationary policy. 
We will need  the notion of a _conserving_ decision-rule; we say that $d^\star\in\mathcal{D}^\text{MD}$ is conserving or $v\_\lambda^\star$-improving 
if for all $s\in\mathcal{S}$;
	
$$
\begin{aligned}
		r(s,d^\star(s)) + \mathbb{E}\left[v\_\lambda^\star(s\_{t+1})\middle\vert s\_t=s, a\_t=d^\star(s)\right] := 
\max\_{a\in\mathcal{A}}\left\\{ r(s, a) + \mathbb{E}\left[v\_\lambda^\star(s\_{t+1})\middle\vert s\_t=s, a\_t=a\right]\right\\}\\; .
\end{aligned}	
$$
Equivalently, $d^\star\in\argmax\_{d\in\mathcal{D}^\text{MD}} \mathbf{r}\_d + \lambda\mathbf{P}\_d v\_\lambda^\star$. 
Then the stationary policy $\pi^\star = (d^\star, \ldots)$ is optimal:
$$
v\_\lambda^\star = \max\_{\pi\in\Pi^\text{HR}} v\_\lambda^\pi = v\_\lambda^{\pi^\star}\\; .
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Observe that because $d^\star$ is conserving;
$$
\begin{aligned}
v_\lambda^{\pi^\star} &= \mathbf{r}\_{d^\star} + \lambda \mathbf{P}\_{d^\star}v\_\lambda^{\pi^{\star}} &(v\_\lambda^{\pi^\star}\text{ fixed-point of } \mathcal{T}\_\lambda^{d^\star})\\\
&= \max\_{d\in\mathcal{D}^\text{MD}}   \mathbf{r}\_{d} + \lambda \mathbf{P}\_{d}v\_\lambda^{\pi^\star} &(\text{conserving}) \\\
			&= \mathcal{T}\_\lambda^\star v\_\lambda^{\pi^\star} \\;.
\end{aligned}
$$
By uniqueness of $\mathcal{T}\_\lambda^\star$'s fixed-point we have $v\_\lambda^{\pi^\star} = v\_\lambda^\star$ proving the announced claim.
{{% /toggle_block %}}

Behind this somewhat simple result lies what is sometimes called the fundamental theorem of MDPs, ensuring us
that we can confidently turn away from history-dependent policies and focus on (Markovian) stationary ones.

{{< boxed title="Fundamental theorem of MDPs" >}}

$$
\max_{\pi\in\Pi^\text{HR}} v_\lambda^\pi = \max_{\pi\in\mathcal{S}^\text{MD}} v_\lambda^\pi \; .
$$
{{< /boxed >}}
#### Beyond discrete action spaces	
We have focused so far on discrete MDPs -- with discrete state and action spaces. Most result are portable beyond this setting, 
and extend to continuous action spaces. In this case, the optimal value function becomes the solution of a supremum problem
(since there then exists a continuously infinite number of policies):
$$
v\_\lambda^\star = \sup_{\pi\in\Pi^\text{HR}} v\_\lambda^\pi \\; .
$$
The Bellman optimality operator becomes, for any $f\in\mathcal{V}$:
$$
\mathcal{T}\_\lambda^\star(f) = \sup_{d\in\mathcal{D}^\text{MD}} \\{ \mathbf{r}\_d + \lambda\mathbf{P}\_d \cdot f \\} \\; .
$$
Whenever this supremum is reached (_i.e._ when the reward and transition function are continuous -- or even only upper semi-continuous -- and 
the action space compact) then one can show that again, there exists an optimal _stationary_ policy. 

## Resources
Most of this blog-post is a condensed version of [[Puterman. 94, Chapter 4&5](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887)]
See the previous blog post on MDPs [here](../mdp_basics).




