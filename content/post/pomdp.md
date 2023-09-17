+++
author = "Louis Faury"
title = "Partial Observability and Belief in MDPs"
date = "2023-02-05"
+++

Partially Observable MDPs allow to model sensor noise, data occlusion, .. in 
Markov Decision Processes. Partial observability is unquestionably practically relevant, but at first glance
this richer interaction setting seems to invalidate the fundamental theorem of MDPs, _a.k.a_ the sufficiency of stationary policies. 
The goal of this post is to nuance this last statement: we will detail how POMDPs really are just "large" MDPs and therefore enjoy many of the same features.

<!--more-->

## PODMPs

In a Partially Observable Markov Decision Process, the actual state of the system is not available to the agent: only an _observation_ of the state is.
This observation could be, for instance, some noisy version of the state (imperfect sensor) or a partial view of it (_i.e._ a 
camera observing a robot can only report the robot's position, not its speed). The agent must
therefore base its actions only on said observations, without perfect state knowledge.
Before discussing the main implications
behind this stark difference with fully observed MDPs, let's first clarify what we mean by observations. 
We will stay true to our [MDP series](../mdp_basics)
and stick with finite state and actions spaces for simplicity. Similarly, the observation space will remain finite.

### Observation model 
The basic ingredients of MDPs are still relevant for POMDPs. From our MDP series we will keep our notations for the state space 
$\mathcal{S}$, the action space $\mathcal{A}$, the cost function $r$ and the state transition kernel $p$. To this mix we add a 
finite _observation_ space $\Omega$ ($\vert\Omega\vert < \infty$) as well observation kernel $q(\cdot\vert s)$ which describes the conditional probability of 
witnessing some observation given a state. This probability mass function is sometimes referred to as the _emission likelihood_.
Formally, for all $s, \omega \in\mathcal{S}\times\Omega$:
$$
q(\omega\vert s) = \mathbb{P}\big(\omega\_t = \omega \big\vert s\_t = s\big)\\; .
$$

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
The observation model can be even more general; 
the observation at round $t$ can depend on the state $s\_{t}$ but also on the last action $a\_{t-1}$ that was input to the system:
$$
	q(\omega\vert s, a) = \mathbb{P}\big(\omega\_t=\omega \big\vert s\_t=s, a\_{t-1}=a\big)\\; .
$$
This added generality will however not change the main conclusions drawn from our simplified case.
{{% /toggle_block %}}

The decision-making process unrolls as follows; after the agent inputs $a\_t$,
the state transitions to $s\_{t+1}\sim p(\cdot\vert s\_t, a\_t)$ and outputs $\omega\_{t+1}\sim q(\cdot\vert s\_{t+1})$. 
This extension to MDPs forms the POMDP tuple:
{{< boxed title="POMDP" >}}
$$
\mathcal{M}=(\mathcal{S}, \mathcal{A}, \Omega, p,  q, r)\; .
$$
{{< /boxed >}}

### Policies

In POMDPs the notion of policy slightly changes, as access to the true state is impossible. 
The _history_ of interaction writes $h\_t = (\omega\_1, a\_1, \ldots, a\_{t-1}, \omega\_t)$
and history-dependant decision-rules ingest this modified history:
$$
	d_t : (\Omega\times\mathcal{A})^{t-1}\times\Omega \mapsto \Delta_{\mathcal{A}}\\; .
$$

We can still identify the subclass of Markovian deterministic decision rules $\mathcal{D}^\text{MD}$, 
which directly maps observations to actions:
$
	d :\Omega\mapsto\mathcal{A}\\; .
$

Such definitions are enough to define the general class of history-dependent randomised policies $\Pi^\text{HR}$ 
and the smaller set of stationary policies $\mathcal{S}^\text{MD}$. The main modification from classical MDPs
is that policies are based only on observation, not states. Nonetheless, to make things interesting, 
the evaluation of policies will remain untouched. We can still rank policies $\{\pi \in \Pi^\text{HR}\}$ according to their discounted cost:
$$
	v\_\lambda^\pi(s) = \mathbb{E}\_s^\pi\Big[\sum\_{t=1}^\infty \lambda^{t-1}r(s\_t,a\_t)\Big] \\; , 
$$
where the expectation is now also taken over realisations of the different observations. It is clear that to perform well, 
an agent must try to infer what the current state likely is, and take actions accordingly.

We use the discounted return to illustrate our discussions. Other criteria will yield the same conclusions. 

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
The dependence of the state-value function on some state can be confusing. What's the point of evaluating this
quantity if the state is anyway unknown to us? Actually, we often assume that the _first_ state is known. Equivalently, 
we can hypothesize that it is drawn according to some known initial distribution.
{{% /toggle_block %}}

## Challenges
As we saw in an [earlier blog-post](../mdp_basics_2), under full observability (that is, $\mathcal{S}=\Omega$ and $q(\omega\vert s) = \mathbf{1}(\omega=s)$) 
there exist optimal stationary policies. This proved particularly helpful as solving the MDP boiled down to finding one optimal 
decision-rule $d^\star\in\mathcal{D}^\text{MD}$. Unfortunately, this amazing property is lost when observability is only partial. 
In all generality, one would need to remember the whole history in order to be optimal. Formally we are saying that in POMDPs:
$$
	\sup\_{\pi\in\Pi^\text{HR}}v\_\lambda^\pi(s) > \max\_{\pi\in\mathcal{S}^\text{MD}}v\_\lambda^\pi(s)\\; .
$$


Let's illustrate this fact with a simple example, by displaying a POMDP where any optimal policy must remember history of size 2.
By "chaining" this example we can construct POMPDs where the memory of any optimal policy is arbitrary long. 

{{< image src="/ns.png" width="670px" align="center">}}

All transitions and observations are deterministic. 
The initial state is sampled at random in $\\{s^1, s^2\\}$. From the reward function, one can see that 
an optimal agent will reach the state $s^5$ as quickly as pocible. Any Markov policy that uses only the current observation 
would be suboptimal since the observation is the same in state $s^3$ and $s^4$, yet both states require different actions
to reach $s^5$ with probability 1.
Only be remembering the observation that was received in the previous round can a policy systematically play the optimal action 
(_e.g._, $a^1$ if the sequence of observations is $\{\omega^1, \omega^3\}$). 

This is bad news for us. Indeed, there are many more history-dependent policy than there are stationary ones.
Furthermore, in all generality, storing a history-dependant policy requires infinite memory. This is not looking good
from a computational side!

## Belief MDPs
The previous section bore some pretty bad news, as stationary policies are off the table if we are to preserve optimality. 
In this section we will see that the true story is a bit more nuanced. 
In particular, we will soon see that any finite POMDPs is _equivalent_ to a continuous MDP -- which by nature does admit an optimal stationary policy. 
The state of this MDP is a so-called **belief-state** (or information state): it is a _sufficient summary statistic_ of the history, 
tracking the conditional probability of the POMDP's state.

### The belief-state


To introduce the belief-state, we'll go back to the definition of the discounted reward criterion. 
The main idea is to realise that the objective only depends on the state distribution, conditioned on its history.
Indeed, for any policy $\pi\in\Pi^\text{HR}$:
$$
\begin{aligned}
	v\_\lambda^\pi(s) &= \mathbb{E}\_s^\pi\big[\sum\_{t\geq1}  \lambda^{t-1}r(s\_t, a\_t)\big]\\; , \\\
	&= \mathbb{E}\_s^\pi\big[\sum\_{t=1}^\infty  \lambda^{t-1} \sum\_{h\in(\Omega\times\mathcal{A})^{t-1}}\sum\_{s'\in\mathcal{S}} \sum\_{\omega \in \Omega} r(s', a\_t)\mathbb{1}[h\_{t-1}=h,\omega\_t=\omega, s\_t=s'] \big]\\;, \\\
    &\overset{(i)}{=} \sum\_{h\in(\Omega\times\mathcal{A})^{t-1}}\sum\_{\omega\in\Omega}\left[\sum\_{t\geq 1}  \lambda^{t-1} \sum\_{s'\in\mathcal{S}} r(s', a\_t)\mathbb{P}(s\_t=s'\vert \omega\_t=\omega, h\_{t-1}=h)\right] \mathbb{P}\_s^\pi(h\_{t-1}=h, \omega\_t=\omega)\\;, \\\
    &= \mathbb{E}\_s^\pi\Big[\sum\_{t\geq 1}\lambda^{t-1} \sum\_{s'\in\mathcal{S}} r(s', a\_t) \mathbb{P}(s\_t=s'\vert \omega\_t, h\_{t-1})\Big]
\end{aligned}
$$
In $(i)$ we simply applied Bayes' rule. We can permute every sum thanks to the dominated convergence theorem (the discounted approach is very handy in that sense).

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
By defining proper filtrations and using the tower-rule, this proof is actually a one-liner.
{{% /toggle_block %}}

We now see that the discounted reward directly depends on the distribution over the state, conditional to the history and the
latest observation. This distribution is called the belief state:

{{< boxed title="Belief state" >}}
$\quad\quad\quad\quad\quad\quad \text{For all } s\in\mathcal{S}:$
$$
b_t(s) := \mathbb{P}\big(s_t=s\big\vert h_{t-1}, \omega_t \big)\; .
$$
{{< /boxed >}}

Observe that we can now re-write the discounted cost function as:
$$
	v\_\lambda^\pi(s) = \mathbb{E}\_s^\pi\big[\sum_{t=1}^\infty \lambda^{t-1} \tilde{r}(b\_t, a\_t)\big] \\; ,
$$
where we defined $ \tilde{r}(b, a) = \sum\_{s\in\mathcal{S}}r(s, a)b(s)$. Note that $b\_t(\cdot)$
can be represented by a vector that lives in the $\vert \mathcal{S}\vert$-dimensional simplex; 
we will use the notation $\pmb{b}_t$ to denote this vector. 

### Observability and Markovian property of beliefs
So far, nothing revolutionary: we simply re-wrote the original objective to depend directly on the belief-state.
However, it turns out that the belief-state is _observable_ (more precisely, computable) and has Markovian dynamics!
Therefore, it has the same property as the state in a fully observable MDP (can you see where this is going?). Before moving on,
let us explicit the claim we just made. 
Suppose that we have already computed $\pmb{b}\_t$. (Remember that we assumed the initial
distribution over was known. In other words, we have access to $\pmb{b}\_0$.) After playing action $a\_t=a$ and observing $\omega\_{t+1}=\omega$, we have that
for all $s\in\mathcal{S}$:
$$
\begin{aligned}
b\_{t+1}(s) &=  \frac{1}{\sigma(\pmb{b}\_t, \omega, a)}q(\omega\vert s)\sum\_{s'\in\mathcal{S}} p(s\vert s', a)b\_t(s') \\; ,
\end{aligned}
$$
where $\sigma(\pmb{b}\_t, \omega, a) = \sum\_{s''\in\mathcal{S}}q(\omega\vert s'')\sum\_{s'\in\mathcal{S}} p(s''\vert s', a)b\_t(s')$.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof is an exercise of repeating Bayes' rule. 
$$
\begin{aligned}
b\_{t+1}(s\vert a, \omega &=  \mathbb{P}(s\_{t+1}=s\vert h\_t, \omega\_{t+1}=\omega) \\;, \\\
	&\propto  \mathbb{P}(s\_{t+1}=s, \omega\_{t+1}=\omega\vert h\_t) \\;, \\\
	&= \mathbb{P}(\omega\_{t+1}=\omega\vert s\_{t+1}=s, h\_t)\mathbb{P}(s\_{t+1}=s \vert h\_t) \\;, \\\
	&\overset{(i)}{=} \mathbb{P}(\omega\_{t+1}=\omega\vert s\_{t+1}=s)\mathbb{P}(s\_{t+1}=s \vert h\_t)\\;, \\\
	&\overset{(ii)}{=} \mathbb{P}(\omega\_{t+1}=\omega\vert s\_{t+1}=s)\sum\_{s'\in\mathcal{S}} \mathbb{P}(s\_{t+1}=s \vert s\_t = s', h\_t)\mathbb{P}(s\_t = s'\vert h\_t)\\;, \\\
	&= \mathbb{P}(\omega\_{t+1}=\omega\vert s\_{t+1}=s)\sum\_{s'\in\mathcal{S}} \mathbb{P}(s\_{t+1}=s \vert s\_t = s', a\_t=a)\mathbb{P}(s\_t = s'\vert h\_t)\\;, \\\
	&\overset{(iii)}{=} \mathbb{P}(\omega\_{t+1}=\omega\vert s\_{t+1}=s)\sum\_{s'\in\mathcal{S}} \mathbb{P}(s\_{t+1}=s \vert s\_t = s', a\_t=a)\mathbb{P}(s\_t = s'\vert h\_{t-1}, \omega\_t)\\;, \\\
	&= q(\omega\vert s)\sum\_{s'\in\mathcal{S}} p(s\vert s', a)b\_t(s')\\;. \\\
\end{aligned}
$$
In $(i)$ we used our observation model to write $\mathbb{P}(\omega\_{t+1}=\omega\vert s\_{t+1}=s, h\_t)=\mathbb{P}(\omega\_{t+1}=\omega\vert s\_{t+1}=s)$. In $(ii)$
we used the law of total probability. In $(iii)$ we used the transition model to write $\mathbb{P}(s\_{t+1}=s \vert s\_t = s', h\_t)=\mathbb{P}(s\_{t+1}=s \vert s\_t = s', a\_t=a)$
and $\mathbb{P}(s\_t = s'\vert h\_t) =  \mathbb{P}(s\_t = s'\vert h\_{t-1}, \omega\_t)$.
We can compute the proportionality constant by remembering that $\pmb{b}\_{t+1}$ is a distribution and should normalise to 1. This yields:
$$
b\_{t+1}(s\vert a, \omega) = \frac{1}{\sum\_{s''\in\mathcal{S}}q(\omega\vert s'')\sum\_{s'\in\mathcal{S}} p(s''\vert s', a)b\_t(s')}q(\omega\vert s)\sum\_{s'\in\mathcal{S}} p(s\vert s', a)b\_t(s')\\; .
$$
{{% /toggle_block %}}

As anticipated, this identity reveals two central properties of the belief-state; _(1)_ the belief-state is observable: provided that we know about the initial state (or the initial distribution of state) we can compute it at every round $t$ since we know $a\_{t-1}$ and observe $\omega\_{t}$.
Further, _(2)_ the belief-state has Markovian dynamics; as for the state in a fully observable MDP, the belief state at round $t$ only depends on the previous belief-state, the action that was input and the freshest observation. 

[//]: # (We can indeed now rewrite the belief definition as:)

[//]: # ($$)

[//]: # (\mathbf{b}\_{t+1} = \mathbb{P}&#40;s\_{t+1}=\cdot\vert a\_t, \omega\_{t+1}, \mathbf{b}\_t&#41;\\; .)

[//]: # ($$)


{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
The reader accustomed to state-space filtering (_e.g.,_ Kalman filters) should have an impression of déjà-vu.
Indeed, the belief-state update is nothing more than an optimal filter. It is usually
packaged in two different updates (predict and update), that here are merged into a single one.
{{% /toggle_block %}}


### The belief-MDP

Let's recap: we found a summary statistic (the belief-state) which is observable and follows Markovian dynamics. Furthermore, 
we showed that if we defined an auxiliary reward function $\tilde{r}$, then the belief could replace the state in our objective:
$$
v\_\lambda^\pi(s) = \mathbb{E}\_s^\pi\big[\sum_{t=1}^\infty \lambda^{t-1} \tilde{r}(\pmb{b}\_t, a\_t)\big] \\; .
$$
Packed together, those two observations have some pleasant consequences when it comes to finding optimal policies.
Indeed, this establishes that solving the POMDP is _equivalent_ to solving a fully observable MDP where the state is replaced by the belief-state. 
This equivalent MDP is also known as the **belief-MDP**. Before going any further, let us provide some details on that new MDP.

{{< boxed title="Belief MDP" >}}
$\quad\quad\quad\quad\quad\quad \text{The belief MDP is given by:}$
$$
\widetilde{\mathcal{M}}=(\mathcal{B}, \mathcal{A}, \tilde{p}, \tilde{r})\; .
$$
$\text{where}$
$$
\left\{\begin{aligned}
&\mathcal{B} = \Delta(\mathcal{S}) \text{ is the space of distributions over }\mathcal{S}\,,\\
&\mathcal{A}\text{ is the original action space}\,,\\
&\tilde{p}(\pmb{b}'\vert a, \pmb{b}) = \sum_{\omega\in\Omega} \mathbf{1}\big[\pmb{b}'= b(\cdot\vert \omega, a, \pmb{b})\big]\sigma(\pmb{b}, \omega, a)\,,\\
&\tilde{r}(\pmb{b}, a) = \sum_{s\in\mathcal{S}} r(s, a)b(s)\,.
\end{aligned}\right.
$$
{{< /boxed >}}
{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
We only have left to prove the result claimed for $\tilde{p}$. This is rather direct:
$$
\begin{aligned}
\tilde{p}(\pmb{b}'\vert a, \pmb{b}) &= \mathbb{P}(\pmb{b}\_{t+1}=\pmb{b}' \vert\pmb{b}\_{t}=\pmb{b}, a\_t=a ) \\;, \\\
&= \sum\_{\omega\in\Omega}\mathbb{P}(\pmb{b}\_{t+1}=\pmb{b}' \vert\pmb{b}\_{t}=\pmb{b}, a\_t=a, \omega\_t=\omega )
\mathbb{P}(\omega\_t = \omega \vert \pmb{b}\_{t}=\pmb{b}, a\_t=a)\\\
&=\sum_{\omega\in\Omega} \mathbf{1}\big[\pmb{b}'= b(\cdot\vert \omega, a, \pmb{b})\big]\sigma(\pmb{b}, \omega, a)
\end{aligned}
$$
where $b(\cdot\vert \omega, a, \pmb{b}) = \frac{1}{\sigma(\pmb{b}\_t, \omega, a)}q(\omega\vert s)\sum\_{s'\in\mathcal{S}} p(s\vert s', a)b\_t(s') $ was previously computed to update $b\_{t+1}$ after playing $a\_t$ and observing $\omega\_t$. 
{{% /toggle_block %}}


Because the belief-MDP is a fully observed MDP, the [fundamental theorem of MDPs](../mdp_basics_2) ensures that it can be solved with 
stationary policies. Here, stationary policies are of a particular flavor; they use a unique decision-rule $ d $ that acts directly on the belief space; 
$
	d : \mathcal{B}=\Delta_{ \mathcal{S}} \mapsto \mathcal{A}\\;.
$
Once an optimal stationary policy
$\pi^\star$ has been found in the belief-MDP, it can be exported to the original POMDP $\mathcal{M}$. At every round $t$, after having playing $a\_{t-1}$
and observed $\omega\_t$, we 
simply need to compute the belief (according to the aforementioned update rule) and apply $\pi^\star$. From the point
of view of the POMDP $\mathcal{A}$, the resulting policy is non-stationary. However, it relies on a policy which is
stationary in $\widetilde{\mathcal{M}}$. 

### Computational challenges
Given some POMDP, we can study its equivalent belief-MDP which has the nice property of being fully observed. 
This is rather pleasant, as we are left with finding stationary optimal policies for the belief-MDP. 
We saw some classical control algorithms for MDPs in [classical MDP control algorithms](../mdp_basics_3) that can do just that.

However, let's not forget that the belief-state lives in $\mathcal{B}=\Delta(\mathcal{S})$ which is _continuous_. 
Algorithms like VI and PI won't apply directly to the belief MDP. 
Actually, in all generality, finding an optimal policy in the belief-MDP is intractable because of the state space's continuous nature. 

There are a bunch 
of approximation methods out there; you can check out the awesome monograph of
\[[Cassandra, 1998](https://cs.brown.edu/research/pubs/theses/phd/1998/cassandra.pdf)\] for a (slightly outdated?) overview
on such methods.

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
There is one POMDP which belief MDP can easily be solved: the Linear Quadratic Gaussian (\[[LQG](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic%E2%80%93Gaussian_control)\]) control problem.
The main tools to solve the LQG are Kalman filters on top of LQR controllers. The LQG solution is a bit "magic": you just have to replace the 
belief-state's mean to the fully-observed LQR controller.. and tada! This property has different names; a common one is _certainty equivalence_.
{{% /toggle_block %}}