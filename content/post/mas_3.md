+++
author = "Louis Faury"
title = "Multi-Agent Dynamical Systems (3/3)"
date = "2023-11-03"
+++

We slightly switch gear in this third blog-post of the multi-agent series.
Letting go of game theoretic concepts, we instead discuss some training paradigms for multi-agent reinforcement learning.
We will cover the main limitations behind centralised and independent learning, to land at the 
centralised training with decentralised execution (CTDE), arguably the more established framework to train
autonomous agents in decentralised multiplayer games. 

<!--more-->

{{< infoblock>}}
$\quad$
We will heavily re-use concepts and notations defined in the previous posts of the multi-agent series.
{{< /infoblock >}}


To simplify matter and avoid game-theoretic subtleties, we will restrict our attention
to DecPOMDPs—that is, partially observable stochastic games with a unique, shared reward function:
$$
\mathcal{M} = (n, \mathcal{S}, \mathcal{A}, \Omega, p, q, r)\\;.
$$

{{% toggle_block background-color="#CBE4FE" title="Note" %}}

Most concepts discussed below are portable to general-sum games, although some care and nuances will be needed 
in competitive games, for instance.
{{% /toggle_block %}}


Similar to the POMDP blog post, and in order to reduce clutter, we will assume that the emission probability does not depend on actions
but only on the current state:
$
q(\bm{\omega} \vert s, \bm{a}) = q(\bm{\omega} \vert s) \\; .
$
Finally, we will assume a discounted, infinite-horizon objective--a joint policy $\bm\pi$
will be evaluated according to:
$$
v\_\lambda^{\bm\pi} = \mathbb{E}^{\bm\pi} \left[\sum\_{t\geq 1} \lambda^{t-1} r(s\_t, \bm{a}\_t)\right]\\; .
$$

Because we place ourselves in a reinforcement learning set-up, we now consider
that both the transition kernel $p$, the emission kernel $q$ and the reward function $r$ are unknown.





## Independent Training
Independent training (IT) is a rather naive approach (yet sometimes surprisingly efficient) to solving $\mathcal{M}$. 
It simply consists of training each agent independently of the others, using some pre-defined single-agent training routine.
From a given agent's perspective, this ultimately means fusing the other agents into the environment itself: each of the $n$ agent is facing a 
different POMDP.
We denote $\mathcal{M}^i = (\mathcal{S}, \mathcal{A}, \Omega, p\_i, q\_i, r)$ the POMDP perceived by agent $i$, where:
$$
\begin{aligned}
    p\_i(s'\vert s, a^i) &= \sum\_{a^{-i}\in\mathcal{A}^{-i}} p(s'\vert s, \bm{a})\bm\pi\_{-i}(\bm{a}^{-i}\vert s) \\; , \\\
\quad
q\_i(\omega^i \vert s) &= \sum\_{\bm{\omega}^{-i}\in\Omega^{-i}} q(\bm{\omega}\vert s)\\; ,
\end{aligned}
$$
with $ \bm{a}=(a^i, \bm{a}^{-i})$ and $ \bm{\omega}=(\omega^i, \bm{\omega}^{-i})$. 

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The first claim is an exercice of the law of total probability:
$$
\begin{aligned}
p\_i(s'\vert s, a^i) &= \mathbb{P}(s\_{t+1}=s'\vert s\_t=s, a\_t^i = a^i) \\;, \\\
&= \sum\_{\bm{a}^{-i}\in\mathcal{A}^{-i}} \mathbb{P}(s\_{t+1}=s', \bm{a}\_t^{-i}=\bm{a}^{-i}\vert s\_t=s, a\_t^i = a^i)\\;, \\\
&= \sum\_{\bm{a}^{-i}\in\mathcal{A}^{-i}} \mathbb{P}(s\_{t+1}=s'\vert s\_t=s, a\_t^i = a^i, \bm{a}\_t^{-i}=\bm{a}^{-i})\mathbb{P}(\bm{a}\_t^{-i}=\bm{a}^{-i}\vert s\_t=s, a\_t^i = a^i)\\;, \\\
&= \sum\_{\bm{a}^{-i}\in\mathcal{A}^{-i}} \mathbb{P}(s\_{t+1}=s'\vert s\_t=s, \bm{a}\_t=\bm{a})\mathbb{P}(\bm{a}\_t^{-i}=\bm{a}^{-i}\vert s\_t=s) \text{ where } \bm{a} = (a^i, \bm{a}^{-i})\\;,\\\
&= \sum\_{\bm{a}^{-i}\in\mathcal{A}^{-i}}p(s'\vert s, \bm{a})\bm\pi\_{-i}(\bm{a}^{-i})\\;.\\\
\end{aligned}
$$
Same goes for the claim regarding the emission probability:
$$
\begin{aligned}
q\_i(\omega^i\vert s) &= \mathbb{P}(\omega^i\_t = \omega^i \vert s\_t = s) \\;, \\\
&= \sum\_{\bm\omega^{-i}\in\Omega^{-i}} \mathbb{P}(\omega^i\_t = \omega^i, \bm{\omega}\_t^{-i}=\bm\omega^{-i} \vert s\_t = s) \\;, \\\
&= \sum\_{\bm\omega^{-i}\in\Omega^{-i}} \mathbb{P}(\bm\omega\_t = \bm\omega\vert s\_t = s) \\; &\text{ where } \bm\omega = (\omega^i, \bm\omega^{-i})\\; . \\\
\end{aligned}
$$
{{% /toggle_block %}}

One can notice that $\mathcal{M}^i$ is dependent on $\bm\pi\_{-i}$, the joint policy of 
other agents but $i$. 
 This policy is not static; as other agents learn and change their respective strategies, the dynamic of  $\mathcal{M}^i$
will naturally evolve. 
 Therefore, when trained independently, each agent has to solve a non-stationary POMDP. 
 From a RL perspective, this is quite far from being great news.



Despite its simplicity, IT has been
shown to empirically work relatively well, even in moderately hard scenarii. 
Of course, this approach's main weakness is its blindness to the non-stationarity each agent must face. For this reason, 
training might be quite unstable, if not downright chaotic.
From a practical perspective, the IT framework is one of the simplest to implement. Simply pick your favorite RL algorithm
and train each agent like you would in a typical single-agent RL problem. 

#### Example: Independent DQN
{{< pseudocode title="Independent DQN" >}} 
$\textbf{input } \text{learning rate } \alpha, \text{ polyak averaging coeff. } \gamma, \text{ batch size } m. \\$
$\text{initialise replay buffers } \mathcal{B}_1\leftarrow \emptyset, \ldots, \mathcal{B}_n \leftarrow \emptyset.\\$
$\text{initialise value networks parameters } \phi_1, \ldots, \phi_n, \text{ target networks parameters } \phi^-_1, \ldots, \phi^-_n.\\$
$\textbf{while } \text{not converged:}\\$
$\qquad \textbf{for } i = 1, \ldots, n:\\$
$\qquad \qquad \text{agent } i \text{ observes } \omega^i_t \text{ and places its action } a_t^i \textit{ (e.g. }\varepsilon\text{-greedy}).\\$
$\qquad \textbf{end for}\\$
$\qquad\text{the environment generates a transition}:\\$
$$\bm{a}_t \leftarrow (a_t^1, \ldots, a_t^n),\;  r_t \leftarrow r(\bm{a}_t), \; s_{t+1}\sim p(\cdot\vert s_t, \bm{a}_t), \; \bm\omega_t \sim q(\cdot\vert s_{t+1})\; .$$
$\qquad  \textbf{for } i = 1, \ldots, n:\\$
$\qquad \qquad \text{populate replay buffer }\mathcal{B}_i \leftarrow \mathcal{B}_i \cup \{ \omega_t^i, \,a_t^i, \,r_t, \,\omega_t^{i+1}\}\;.\\$
$\qquad  \qquad \text{sample dataset } \mathcal{D}_i \text{ of } m \text{ transitions from } \mathcal{B}_i \text{ and update:} \\$
$$
\begin{aligned}
\phi_i &\leftarrow \phi_i - \alpha \sum_{\omega^i, a^i, r, \tilde\omega^i\in\mathcal{D}_i} \nabla_{\phi_i}q_{\phi_i}(w^i, a^i)\left[q_{\phi_i}(w^i, a^i) - r_i - \lambda\max_{a\in\mathcal{A}^i}q_{\phi_i^-}(\tilde\omega^i, a) \right]\;,\\
\phi_i^-&\leftarrow \gamma \phi_i^- + (1-\gamma)\phi_i\;.
\end{aligned}
$$
$\qquad  \textbf{end for}\\$
$\textbf{end while}$
{{< /pseudocode >}}

<br> 
<br> 


## Centralised Training
Centralised Training (CT) lies at the other extreme end of the spectrum. It consists of letting go of the
multi-agent nature of $\mathcal{M}$ to adopt a centralised, single-agent solver. 
Formally, CT solves the POMDP $\mathcal{M}$ by directly learning
the joint policy $\bm\pi : \Omega \mapsto \Delta(\mathcal{A})$. 
This solves the non-stationary issues of IT, and allows for seamless coordination between agents.

Of course, CT is often downright unfeasible, except for a few toy examples. The combinatorial nature
of the action space $\mathcal{A} = \mathcal{A}\_1 \times\ldots\times \mathcal{A}\_n$ tears down any hope of efficient learning whenever
there are more than a handful of agents. 
Also, let's not forget that for most realistic multi-agent situations, centralisation simply is not an option.
For instance, networking issues can prevent agents from sending their observation to a common server once
we let go of our training simulator and deploy policies in the "real world".

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
Beyond decentralisation constraints, CT does not make much sense beyond fully cooperative games.
Indeed, when each agent optimises its own reward, one would need to engineer a shared reward to be
able to train the flock of agents as a single one.
{{% /toggle_block %}}

## Centralised Training with Decentralised Execution

Centralised Training with Decentralised Execution (CTDE) attempts to get the best of both worlds. 
The underpinning
idea is quite simple: we will continue to maintain decentralised policies, but allow them to centralise information
during training only.
This will give rise to more stable learning dynamics by mitigating non-stationarity. 
Let's emphasize that when acting, each policy still relies only on its own private observation:
it is at training time that it can broadcast information to a central instance. 
Below, we give some concrete examples of algorithmic approach relying on the CTDE framework.

### Actor-Critic CTDE
The actor-critic architecture easily adapts to the CTDE framework.
In short, the main idea is to leave the "actor" part decentralised while the "critic" part is centralised. 
Concretely, this would typically mean that maintaining decentralised policies:
$
\pi\_i : \Omega\_i \longmapsto \Delta(\mathcal{A}\_i) \\;,
$
while capturing, through the critic, the value of the _joint_ policy based on the joint observations and actions:
$$
\begin{aligned}
    q^{\bm\pi} : \Omega \times \mathcal{A} &\longmapsto \mathbb{R} \\;, \\\
\bm\omega, \bm{a} &\longmapsto q(\bm\omega, \bm a) \approx q\_\lambda^{\bm{\pi}}(s, \bm a) \\; ,
\end{aligned}
$$
where $q\_\lambda^{\bm{\pi}}$ is the state-action joint value function. 
This joint critic serves only at training time to guide gradient-based policy optimisation (see example below). 

{{<infoblock>}}
$\quad$ The main justification for this approach is a generalisation of the <a href="/post/policy_gradient/#policy-gradient-theorem" target="_blank">policy gradient theorem</a> to multi-agent settings.
It states that the gradient of $U(\bm\pi) = v_\lambda^{\bm\pi}$ w.r.t. a marginal policy $\pi_i$ writes:
$$
\nabla_{\pi_i} U(\bm{\pi})  \propto
\mathbb{E}_{s\sim d_\lambda^{\bm{\pi}}(\cdot)}  \mathbb{E}_{a\sim \bm{\pi}(\cdot\vert s)}\left[ q_\lambda^{\bm{\pi}}(s, \bm{a})\nabla_{\pi_i} \log \pi_i(a_i\vert s)\right]
$$
{{</infoblock>}}


{{% toggle_block background-color="#CBE4FE" title="Note" %}}
We consider above Markovian policy for simplicity--in all generality, 
and because of the partial observability, it would make more sense to have history-dependent policies. 

{{% /toggle_block %}}

Of course, there exist countless variants of this approach
(_e.g_ one could decide to learn a state-dependent value function $v^{\bm\pi}$ to use a joint baseline for advantage estimation).
Virtually any single-agent actor-critic can be mapped to its multi-agent variant.

#### Example: Vanilla MA-PG

{{< pseudocode title="Multi-Agent VPG" >}} 
$\textbf{input } \text{learning rate } \alpha, \text{ episode batch size } m. \\$
$\text{initialise value networks parameters } \phi_1, \ldots, \phi_n, \text{initialise policy networks parameters } \theta_1, \ldots, \theta_n.\\$
$\textbf{while } \text{not converged}\\$
$\qquad \text{initialise dataset } \mathcal{D}\leftarrow \emptyset.\\$
$\qquad \textbf{for } t = 1, \ldots, m: \\$  
$\qquad\qquad \textbf{for } i = 1, \ldots, n:\\$
$\qquad\qquad \qquad \text{agent } i \text{ observes } \omega^i_t \text{ and places its action } a_t^i\sim\pi_i(\cdot\vert \omega_t^i).\\$
$\qquad\qquad \textbf{end for}\\$
$\qquad\qquad \text{the environment generates a transition}:\\$
$$\bm{a}_t \leftarrow (a_t^1, \ldots, a_t^n),\;  r_t \leftarrow r(\bm{a}_t), \; s_{t+1}\sim p(\cdot\vert s_t, \bm{a}_t), \; \bm\omega_t \sim q(\cdot\vert s_{t+1})\; .$$
$\qquad\qquad \text{add transition to dataset } \mathcal{D} \leftarrow \mathcal{D}\cup \{\bm\omega_t, \bm a, r_t, \bm\omega_{t+1}\}\\$
$\qquad\qquad \textbf{end for}\\$
$\qquad \textbf{end for}\\$
$\qquad \text{fit the value function:}\\$
$$
\phi \in \argmin \sum_{\bm\omega, r, \bm\omega'\in\mathcal{D}}\left(v_\phi(\bm\omega) - y\right)^2 \text{ where } y = r + \lambda v_\phi(\bm\omega')\; .   
$$
$\qquad \text{gradient-based update step for each agent's policy:}\\$
$\qquad \textbf{for } i = 1, \ldots, n:\\$
    $$\theta_i \leftarrow \theta_i + \alpha \sum_{\bm\omega, \bm a, r\in\mathcal{D}}(r  - v_\phi(\bm\omega)) \nabla_{\theta_i} \log \pi_{\theta_i}(a_i \vert \omega^i)$$
$\qquad \textbf{end for}\\$
$\textbf{end while}\\$
{{< /pseudocode >}}

<br> 
<br> 



### Value-Based CTDE

Value-based method also adapts to the CTDE framework. The leading idea is to train a joint value-function:
$$
    q(\bm\omega, \bm a) \approx q\_\lambda^\star(s, \bm a)
$$ 
and then decompose it into agent-centric value-functions $q\_1(\cdot, \cdot), \ldots, q\_n(\cdot, \cdot)$ where
each $q\_i : \\,\Omega\_i \times \mathcal{A}\_i \longmapsto \mathbb{R}$
acts as the marginal value-function for agent $i$. This is typically done by directly modelling each agent-centric 
value function $q\_i = q\_{\theta_i}$ and assuming that a combination of said values will yield the joint value:
$$
     q(\bm\omega, \bm a) \approx f\Big(q\_1(\omega^1, a^1), \ldots, q\_n(\omega^1, a^1)\Big) \\; .
$$
This joint value-function can be learned during training, while each marginal $q\_i$ can be used for decentralised decision-making
at execution time. 
{{% toggle_block background-color="#CBE4FE" title="Note" %}}
As pointed out in the [QMIX paper](https://arxiv.org/pdf/1803.11485.pdf), each $q\_i$ is rather a _utility_ function than a value-function, 
since it does not estimate an expected return -- only the joint $q(\bm\omega, \bm a)$ does.
{{% /toggle_block %}}

The aggregation function $f$ is either fixed or learned, but it should not be _arbitrary_.
Ideally, we'd need some homogeneity between two sets of maximisers: 
the ones for the joint value function, and the ones for the marginal, decentralised value functions.
This allows the decentralised agents to be consistent with the joint plan that was made offline (during training)
by acting greedily w.r.t their marginal value function.
To stabilize the training, one usually restricts $f$ to be 
monotonously increasing in all its argument: this is to enforce the joint greedy action to match with the collection
of marginal greedy actions. Formally
$$
\argmax_{\bm a \in \mathcal{A} } q(\bm \omega, \bm a)  = \left(\argmax\_{a^1\in\mathcal{A}\_1} q\_1(\omega^1, a^1), \ldots, \argmax\_{a^n\in\mathcal{A}\_1} q\_n(\omega^n, a^n)\right)^\top \\;.
$$
Under this functional constraint, the joint value function is trained like any single agent one -- _e.g._, in a DQN-style. 

{{<infoblock>}}
$\quad$ For instance, the <a href="https://arxiv.org/pdf/1706.05296.pdf" target="_blank">Value Decomposition Network</a> paper demonstrates the efficiency of a somewhat vanilla strategy, 
where $f$ simply is the sum-function:
$$
    q(\bm\omega, \bm a) = \sum_{i=1}^n q_i(\omega^i, a^i) \; .
$$
{{</infoblock>}}




