+++
author = "Louis Faury"
title = "MDP Fundamentals (1/3)"
date = "2022-09-26"
+++

This blog post is the first of a short series on control in discounted, discrete MDPs. 
In particular, we'll cover here the very basics: the definition of a MDP, the different notions
of policies (history-dependent, Markovian, stationary,...), as well as the discounted objective for MDPs.


<!--more-->

## Markov Decision Processes
Markov Decision Processes (MDPs) stand as a fundamental formalisation of sequential decision-making and control problems. 
Throughout the series, we will focus on discrete MDPs – with discrete (i.e finite or countable) state and action spaces. 
Most ideas are portable to continuous MDPs; their proper treatment requiring slightly more subtle measure theoretic argument, we exclude
them for the sake of clarity. 

### Definition
The first ingredients to a fully-observable, stationary MDP are a state space $\mathcal{S}$ and an control/action space $\mathcal{A}$. 
As announced we will focus on the case where $\mathcal{S}$, $\mathcal{A}$ are discrete - that is, finite or countable.
The _stationary_ dynamics between different states is encoded by some transition kernel $\mathcal{P}$, where:
$$
    \mathcal{P}\_s^a(\cdot) = \mathbb{P}(s\_{t+1} = \cdot\vert s\_t=s, a\_t=a) \\; \text{ for any } t\geq 1\\; .
$$
A stationary reward function $r(s, a)$ measures the quality of action $a$ when executed in state $s$. That's basically it;
a MDP $\mathcal{M}$ is a collection of a state-space, an action-space, a probabilistic rule of transition to states from state-action pairs, 
and a measure of success through the reward function.
{{< pseudocode title="MDP" >}}
$$
    \mathcal{M} := (\mathcal{S}, \mathcal{A}, \mathcal{P}, r)
$$
{{< /pseudocode >}}


### Decision-rule and Policies
Solving an MDP informally means that we want to generate trajectories $(s\_1, a\_1, \ldots s\_t, a\_t, \ldots)$
yielding large cumulated rewards (we'll see in a bit how we measure that exactly).

#### Decision-rules
At a given round $t$, an agent trying to solve the MDP is able to pick a _decision-rule_ $d\_t$ 
that maps the history up to round $t$ to the space of distribution over $\mathcal{A}$ - denoted $\Delta_\mathcal{A}$. Formally;
$$
\begin{aligned}
			d\_t : (\mathcal{S}\times\mathcal{A})^{t-1}\times\mathcal{S}&\longmapsto \Delta\_\mathcal{A}\\\
			h\_t = (s\_1, a\_1, \ldots, a\_{t-1}, s\_t) &\longmapsto d\_t(h\_t)
\end{aligned}
$$
The next action that is played is $a\_t \sim d\_t(h\_t)$.
This space of function is called the space of history-dependent, randomised strategies and denoted $\mathcal{D}^{\text{HR}}$. 
It contains other strategies of interest like history-dependent deterministic $\mathcal{D}^{\text{HD}}$, Markovian randomized $\mathcal{D}^{\text{MR}}$ and Markovian deterministic $\mathcal{D}^{\text{MD}}$ decision-rules. 
Recall that a Markovian deterministic policy $d$ takes as an argument only the current state and returns a unique control; 
_i.e._ $d\in\mathcal{D}^\text{MD} \Longrightarrow d:\mathcal{S}\mapsto\mathcal{A}$.
The different classes of policies obey the following relationships;
$$
\boxed{
	\begin{aligned}
		\mathcal{D}^{\text{MD}}\subset \mathcal{D}^{\text{HD}} \subset &\\; \mathcal{D}^{\text{HR}}\\\
		& \\; \cup  \\\
		\mathcal{D}^{\text{MD}}\subset &\mathcal{D}^{\text{MR}}
	\end{aligned}
}
$$

#### Policies
A _policy_ $\pi=(d_1, \ldots, d_T)$ is a sequence of decision rules -- the 
different spaces of policies being denoted $\Pi^{\text{MD}}$, $\Pi^{\text{MR}}$, $\Pi^{\text{HD}}$ and $\Pi^{\text{HR}}$. 
History-dependent, randomised policies are the most general and can represent any decision-making agent.
It is however clear that they might not be the easiest to handle and represent, as they might require infinite memory.
Arguably the most basic policy sub-class is the one of _stationary_ policies, denoted $\mathcal{S}^{\text{MD}}$. 
An agent playing with a stationary policy applies at each time step the same decision-rule:
$$
		\pi\in\mathcal{S}^{\text{MD}} \Longrightarrow \pi = (d, d, \ldots), \text{ where }d\in\mathcal{D}^\text{MD}\\; .
$$
We will see that in our case, this light-weight policy class is actually _enough_ to solve an MDP.
This will make finding an _optimal_ policy (whatever that means for now) reasonable from
a computational standpoint.

#### Important Notations

Given an initial state $s\in\mathcal{S}$, the combination of any policy $\pi\in\Pi^{\text{HR}}$ 
with the transition probability kernels $\mathcal{P}$ induces a probability measure over the sequence of visited states and actions. 
We will use the shorthand notation $\mathbb{P}_s^{\pi}$ to denote this probability measure - and some derived measures, 
_e.g._ the probability measure obtained by marginalising over the states. For instance, 
when $\pi=(d\_1,\ldots,)\in\Pi^{\text{HR}}$ we will note; 
$$
\mathbb{P}\_s^{ \pi}\left(s\_{t+1}=s'\right) = \mathbb{P}\left(s\_t=s'\middle\vert s\_1=s, \\; a\_i \sim d\_i(h\_i),  \\; s\_{i+1}\sim\mathcal{P}\_t(s\_{i}, a\_{i}, \cdot) \text{ for } i\leq t\right)\\; .
$$
We will adopt similar notations when reasoning about expectations; for instance:
$$
\mathbb{E}_s^{\pi}\left[r(s\_t,a\_t)\right] = \mathbb{E}\left[r(s\_t,a\_t)\middle\vert s\_1=s, \\; a\_i \sim d\_i(h\_i),  \\; s\_{i+1}\sim\mathcal{P}\_t(s\_{i}, a\_{i}, \cdot) \text{ for } i\leq t\right] \\;.
$$

### Discounted Objective
We are now ready to precise how we measure the cumulative reward along a trajectory. Here, we will focus on the discounted approach.
Discounting naturally arises in situations where the decision-maker prefer to adopt _myopic_ strategies - it is 
more concerned about the immediate reward of its actions rather than future benefits he might enjoy. Let $\lambda\in[0, 1)$ the _discount_ factor. 
It accounts for the decision-maker's degree of conservatism; the closer to 0, the more preoccupied he is about his immediate reward. 
The $\lambda$-discounted reward of a policy $\pi\in\Pi^{\text{HR}}$ is defined as:

{{< pseudocode title="Discounted objective" >}}
$$ 
    v_\lambda^\pi(s) := \lim_{T\to\infty} \mathbb{E}_s^\pi\left[ \sum_{t=1}^T \lambda^{t-1}r(s_t, a_t)\right], \; s\in\mathcal{S}
$$
{{< /pseudocode >}}


<br>
<br>
{{% toggle_block background-color="#CBE4FE" title="Note" %}}
We focus here on the discounted objective, not because of its practical relevance
but rather because of its practical _usefulness_: it makes up for an easy analysis and approximates well
other criterion - such as the stationary-reward objective, or even some finite-horizon objectives. I won't dive more into
details (for now).
{{% /toggle_block %}}


Our first reflex should be to check that this definition actually makes sense - that is, the limit exists and is bounded. 
It is indeed safe when rewards are bounded and $\lambda<1$, and we can swap limit and expectation:
$$
  v\_\lambda^\pi(s) :=  \mathbb{E}\_s^\pi\left[ \sum\_{t=1}^\infty \lambda^{t-1}r(s\_t, a\_t)\right] \\; .
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The first claim is easily proven by showing that the sequence $\{ A_T = \mathbb{E}^\pi\_s\big[\sum\_{t=1}^T \lambda^{t-1}r(s\_t, a\_t)\big]\}\_T$ is Cauchy - which 
will imply existence of a finite limit by completeness of $\mathbb{R}$. Notice that for all $T, T'\in\mathbb{N}$ with $T'>T$:
$$
\begin{aligned}
	\left\vert A\_{T'} - A\_T\right\vert &= \left\vert \mathbb{E}\Big[\sum\_{t=T+1}^{T'} \lambda^{t-1}r(s\_t, a\_t)\Big]\right\vert\\\
	&\leq  M \sum_\{t=T+1}^{T'} \lambda^{t-1} &(\text{bounded reward})\\\
	&\leq M\lambda^{T}(1-\lambda)^{-1} &(\lambda<1)
\end{aligned}
$$
Therefore for any $\varepsilon>0$ there exists $T$ large enough such that $\left\vert A_{T+n} - A_T\right\vert\leq \varepsilon$ proving that $\{A_T\}$ is indeed Cauchy. 
The second claim is a direct consequence of the dominated convergence theorem, which conditions are easily checked by using the boundedness assumption and the convergence of geometric series.
{{% /toggle_block %}}

Notice how, by marginalising over state and action, we can rewrite the discounted objective as:
$$
v\_\lambda^\pi(s) = \sum\_{s', a'\in\mathcal{S}\times\mathcal{A}} r(s', a') \sum\_{t=1}^\infty \lambda^{t-1} \mathbb{P}\_s^\pi(s\_t=s', a\_t = a') \\; .
$$
It turns out that for any history-dependent policy $\pi\in\Pi^{\text{HR}}$, we can find some Markovian policy $\pi'\in\Pi^\text{MR}$
such that there state-action visit frequencies are the same:
$$
\mathbb{P}\_s^\pi(s\_t=s', a\_t = a') = \mathbb{P}\_s^{\pi'}(s\_t=s', a\_t = a'), \\; \text{ for any } s', a'\in\mathcal{S}\times\mathcal{A}
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The main idea when building $\pi'$ is to make sure that in any state it follows the available actions with the same probabilities as $\pi$; the Markovian nature of the dynamics will do the rest. 
This can be done by averaging over all the possible histories that lead to the given state $s\in\mathcal{S}$. 
Formally at a given round $t$ and triple $s, s', $ we will define $d'{\_t}$ such that; 
$$
\begin{aligned}
		\mathbb{P}^{d'{\_t}}\left(a\middle\vert s\_1=s, s\_t = s'\right) &= \sum\_{h\in(\mathcal{S}\times\mathcal{A})^{t-1}\times\mathcal{S}}
\mathbb{P}\_s^\pi\left(a\_t = a\middle\vert h\_t=h\right)\mathbb{P}\_s^\pi\left(h_t=h\vert s\_t = s'\right)\; ,\\\
		&= \mathbb{P}^\pi\_s\left(a\_t=a\middle\vert s\_t = s'\right)\; .
\end{aligned}
$$
	Note that under this construction, the policy $\pi'=(d'_1, \ldots)$ depends on the initial state $x$. Based on this construction a forward recurrence is enough to prove the result
{{% /toggle_block %}}

In eye of the previous re-writing of $v\_\lambda^\pi$, this will mark our first important result of this series, marking the 
first step towards trimming the very large space of history-dependent policies. Indeed, we just showed that for any $\pi\in\Pi^{\text{HR}}$ and $s\in\mathcal{S}$,
we can construct some $\pi'\in\Pi^\text{MR}$ such that $v\_\lambda^\pi(s) = v\_\lambda^{\pi'}(s).$ Therefore, it is definitely enough to focus only on 
$\Pi^\text{MR}$, which is a much smaller policy class compared to $\Pi^\text{HR}$!

### Vectorial Notations
We now introduce some notations that will allow us some more compact writing.
Let $\mathcal{V}$ be the space of bounded functions mapping the $\mathcal{S}$ to $\mathbb{R}$. 
When $\mathcal{S}$ is _finite_ - _i.e_ $\vert \mathcal{S} \vert = n\in\mathbb{N}$, we can represent such functions 
as $n$-dimensional vectors. Indeed, writing $\mathcal{S} = \{ s^1, \ldots, s^n\}$ and for 
$f\in\mathcal{V}$ we will denote;
        $$
        		\forall i\in\{1, \ldots, n\}, \\; [f]\_i = f(x^i)\in\mathbb{R}\; .
        $$
        
In the finite case, we will identify $\mathcal{V}$ with $\mathbb{R}^n$ and use the so-called vectorial notation 
for finite MDPs to further reduce clutter. We list below such notations before giving out a few identities. 
In the following, $d\in\mathcal{D}^\text{MR}$ is a Markovian randomized decision rule and $\pi\in\Pi^\text{MR}$. 
The three main entities we will work with are the reward vector $\mathbf{r}\_{d}$, 
the instantaneous transition matrix  $\mathbf{P}\_{d}$ and the transition matrix $\mathbf{P}\_{\pi}^{t}$. 
In the following definitions we index the entries of vectors and matrix with $s,s'\in\mathcal{S}$.

{{< pseudocode title="Vector notations" >}}
$$
\begin{aligned}
			[\mathbf{r}_{d}]_s &:= \mathbb{E}_{a\sim d(s)} \left[r(s, a)\right]\; ,\\
			[\mathbf{P}_{d}]_{s,s'} &:= \mathbb{P}_{a\sim d(s)}\left(s_t = s'\vert s_{t-1}=s\right) \\
			[\mathbf{P}_{\pi}^{t}]_{s,s'} &:= \mathbb{P}^\pi_s\left(s_t = s'\right)
\end{aligned}
$$
{{< /pseudocode >}}
        
<br>
<br>

For instance, the entry located at the $s^{\text{th}}$ row  and $s'^{\text{th}}$ column of 
$\mathbf{P}\_{d}$ evaluates the probability of transitioning from $s$ to $s'$ when following the randomised policy $d$. 
Note that $\mathbf{r}\_{d}\in\mathcal{V}$ while $\mathbf{P}\_{d}$ and $\mathbf{P}\_{\pi}^{t}$ are linear operators on $\mathcal{J}$ 
to itself. 

The following identities will be useful in our proofs; the first ties the transition matrix of a policy
$\pi=(d\_1, \ldots)\in\Pi^\text{MR}$ to the instantaneous transition matrices; 
$$
	\mathbf{P}\_{\pi}^{t} = \mathbf{P}\_{\pi}^{t-1} \mathbf{P}\_{d\_t} = \prod_{i=1}^t \mathbf{P}\_{d\_i}
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof is a direct consequence of the law of total probabilities; for all $s, s'\in\mathcal{S}$;
$$
	\begin{aligned}
		[\mathbf{P}\_{\pi}^{t}]\_{ss'} &= \mathbb{P}^\pi\_s\left(s\_t = s'\right)\\\
		&= \sum\_{s''\in\mathcal{S}}   \mathbb{P}^\pi\left(s\_t = s'\middle\vert s\_{t-1}=s''\right) 
\mathbb{P}^\pi\_s\left(s\_{t-1} = s''\right) &(\text{total probabilty})\\\
		&=  \sum\_{s''\in\mathcal{S}}   \mathbb{P}\_{a\sim d\_t(s)}\left(s\_t = s'\vert s\_{t-1}=s''\right)
\mathbb{P}^\pi\_s\left(s\_{t-1} = s''\right) \\\
		&=  \sum\_{s''\in\mathcal{S}}  [\mathbf{P}\_{\pi}^{t}]\_{ss''}[\mathbf{P}\_{d\_t}]\_{s'',s'}\\\
		&= [\mathbf{P}\_{\pi}^{t}\mathbf{P}\_{d\_t}]\_{ss'}\\; .
	\end{aligned}
$$
{{% /toggle_block %}}




The second allows to rewrite the expected discount value of $\pi=(d\_1, \ldots)$ in a fairly compact manner; 
 $$
 	v\_\lambda^\pi  = \sum\_{t=1}^{+\infty} \lambda^{t-1}
\mathbf{P}^{\pi}\_{t} \mathbf{r}\_{d\_{t}} \\; .
$$
where $\mathbf{P}^{\pi}\_{1}=\mathbf{I}\_n$. 
Above, we write the expected discounted value $v_\lambda^\pi\in\mathcal{V}$ in its vectorial form. 

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof is obtained by re-writing the objective as, for any $s\_0\in\mathcal{S}$:
$$
\begin{aligned}
v\_\lambda^\pi(s\_0) &= \mathbb{E}\_s^\pi\left[\sum\_{t=1}^\infty \lambda^{t-1}r(s\_t, a\_t)\right] \\;, \\\
&= \mathbb{E}\_s^\pi\left[\sum\_{t=1}^\infty \sum\_{s\in\mathcal{S}}\sum\_{a\in\mathcal{A}}
\lambda^{t-1}r(s, a) \mathbf{1}(s\_t=s, a\_t =a)\right]  \\; , \\\
&= \sum\_{t=1}^\infty \lambda^{t-1}\sum\_{s\in\mathcal{S}}\sum\_{a\in\mathcal{A}} r(s, a) 
\mathbb{P}\_{s\_0}^\pi(s\_t=s, a\_t=a) \\; ,\\\
&=  \sum\_{t=1}^\infty \lambda^{t-1}\sum\_{s\in\mathcal{S}} \sum\_{a\in\mathcal{A}} r(s, a)\mathbb{P}\_{s\_0}^\pi(s\_t=s)
d\_t(a\vert s)\\;, \\\
&= \sum\_{t=1}^\infty \lambda^{t-1}\sum\_{s\in\mathcal{S}} \mathbb{P}\_{s\_0}^\pi(s\_t=s) [\mathbf{r}\_{d\_t}]\_s\\;, \\\
&=  \sum\_{t=1}^\infty \lambda^{t-1}[\mathbf{P}\_{t}^\pi\cdot\mathbf{r}\_{d\_t}]\_{s\_0}\\; ,
\end{aligned}
$$
where all swap between sum and expectation is justified by the dominated convergence theorem (briefly, it's the geometric decay
of $\lambda^{t-1}$ that always saves the day).
{{% /toggle_block %}}

## Resources
Most of this blog-post is a condensed version of [[Puterman. 94, Chapter 4&5](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887)]





