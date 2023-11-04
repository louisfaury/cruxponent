+++
author = "Louis Faury"
title = "Multi-Agent Dynamical Systems (1/4)"
date = "2023-09-10"
+++

This blog-post is the first of a short serie on Multi-Agent control and RL. The objective here is to cover
the different models of multi-agent interactions, from repeated matrix games to partially observable Markov games. 
Going up the game hierarchy (from less to more general) we will detail what constitutes valid policies and how
they differ from their usual fully and partially observed MDPs cousin.

<!--more-->

We focus below on so-called _normal-form_ games: both players play at the same time. Games that unroll in a turn-based
fashion are called _extensive-form_ games. The difference may seem thin, but deeply impacts the potential solution concepts.
For instance, once can find deterministic optimal policies in some extensive form games since the opponent's move is perfectly known.
In normal-form games, some level of randomness is often required to prevent the opponent from guessing our moves.

<br>
{{< image src="/games.png" width="550px" align="center">}}
<br>

## Agent Objectives
Before diving into a game classification based on agents' observation models, let's discuss the different possibilities
when it comes to the agents' objectives. As in a classical MDP, those will be materialised via rewards functions— 
for concreteness, we will denote them $r^1, \ldots, r^n$ for a game with $n$ agents. In all generality, 
there exists no built-in relationship between said rewards. This is a _general-sum_ game. Such a game
admits two important sub-games:
- In a _zero-sum_ games describe games, agents have opposite objectives and act competitively. Formally, the agents rewards sum to zero at each round of the game:
$$
\sum\_{i=1}^n r\_t^i = 0, \quad \forall t\geq 1\\; .
$$
Perhaps the most common instance of such a game is the two-player zero-sum game, where $r\_t^1 = -r\_t^2$.
- _Cooperative_ games (_a.k.a_ team-games or common-reward games) designate settings where the agents share the same objective and act in a fully cooperative fashion. Formally, 
this means that at any round $t$:
$$
r\_t^1 = \ldots = r\_t^n \\; .
$$

All instances of this game classification
can be coupled with the following one, concerned with observations rather than rewards.

## Agent Observations and Interactions
Another important classification of games concerns the level of information accessible to the agents, as well as 
their actions' impact on their environment. Below, we go up the hierarchy of multi-agents settings, from less to more general.
This hierarchy evolves with the multi-agents' environment dynamic, their knowledge of each other and of the
game's hidden state (if any). 


### Repeated Matrix Games
A _matrix game_ can be represented via its number of player $n\in\mathbb{N}$, a combination of its agent action space: 
$$\mathcal{A}=\mathcal{A}\_1\times\ldots\times\mathcal{A}\_n\\;,$$ 
as well as a collection $\bm{r} = (r^1, \ldots, r^n)$ of reward functions, each mapping
$\mathcal{A}$ to $\mathbb{R}$. There is no notion of state; the agents' actions do not impact the environment in any way.

{{< boxed title="Matrix game" >}} 
$$
\mathcal{M} = (n, \mathcal{A}, \bm{r})\; .
$$
{{< /boxed >}}

A _repeated_ matrix game simply is the repetition of the same matrix game for $T\in\mathbb{N}\cup\\{+\infty\\}$ rounds.
 

{{< pseudocode title="Repeated Matrix Game" >}} 
$\textbf{init } \text{game history } \bm{h}_0 \leftarrow \emptyset\; .\\$
$\textbf{for } t = 1, \ldots, T:\\$
$\qquad \textbf{for } i = 1, \ldots, n:\\$
$\qquad \qquad \text{agent } i \text{ places its action } a_t^i\\$
$\qquad \textbf{end for}\\$
$\qquad \text{the environment receives the joint action } \bm{a}_t = (a_t^1, \ldots, a_t^n) \\$
$\qquad \textbf{for } i = 1, \ldots, n:\\$
$\qquad \qquad \text{agent } i \text{ receives } r_t^i = r^i(\bm{a}_t)\\$
$\qquad \textbf{end for}\\$
$\qquad \text{the game history gets updated } \bm{h}_t \leftarrow \bm{h}_{t-1} \cup \bm{a}_t\\ $
$\textbf{end for}\\$
{{< /pseudocode >}}

<br> 
<br> 

This is a perfect knowledge game; each agent has access to the others' previous actions. 
At each round $t$, each agent can base its decision $a\_t^i$ based on the game's full history $\bm{h}\_{t-1}$.
A valid decision rule $d\_t^{\\,i}$ for agent $i$ therefore maps said history to a distribution over $\mathcal{A}\_i$.
A collection of agent decision-rules form the marginal policy $\pi^i = (d\_1^{\\,i}, d\_2^{\\,i}, \ldots)$. The _joint_ policy is: 
$$\bm{\pi} = (\pi^1, \ldots ,\pi^n)\\; .$$ 

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
When there is only a single agent, a repeated matrix game is nothing more than a bandit game.
{{% /toggle_block %}}

### Stochastic Games
Up the game hierarchy lie stochastic games (_a.k.a_ Markov games) where the agents' actions now influence the environment's state, 
which lives in some set $\mathcal{S}$. The definition extends the classical MDP definitions to handle several agents, by
having the transition kernel now depends on the joint action:
$$
\begin{aligned}
    p(\cdot \vert s, \bm{a}) &= \mathbb{P}(s\_{t+1}=\cdot \vert s\_t=s, \bm{a}\_t = a)\\\
&= \mathbb{P}(s\_{t+1}=\cdot \vert s\_t=s, a\_t^1=a^1, \ldots, a\_t^n = a^n)\\;.
\end{aligned}
$$
The reward functions now map $\mathcal{S}\times\mathcal{A}$ to $\mathbb{R}$. 
{{< boxed title="Stochastic game" >}} 
$$
\mathcal{M} = (n, \mathcal{S}, \mathcal{A}, p, \bm{r})\; .
$$
{{< /boxed >}}


{{< pseudocode title="Stochastic Game" >}} 
$\textbf{init } \text{state } s_1\in\mathcal{S}, \text{ game history } \bm{h}_0 \leftarrow \emptyset\; .\\$
$\textbf{for } t = 1, \ldots, T:\\$
$\qquad \textbf{for } i = 1, \ldots, n:\\$
$\qquad \qquad \text{agent } i \text{ observes the state }s_t \text{ and places its action } a_t^i\\$
$\qquad \textbf{end for}\\$
$\qquad \text{the environment receives the joint action } \bm{a}_t = (a_t^1, \ldots, a_t^n) \\$
$\qquad \textbf{for } i = 1, \ldots, n:\\$
$\qquad \qquad \text{agent } i \text{ receives } r_t^i = r^i(\bm{a}_t)\\$
$\qquad \textbf{end for}\\$
$\qquad \text{the game history gets updated } \bm{h}_t \leftarrow \bm{h}_{t-1} \cup \{ s_t, \, \bm{a}_t\}\\ $
$\qquad \text{the environment state transitions: } \\$
$$
s_{t+1} \sim p(\cdot\vert  s_t, \bm{a}_t)
$$
$\textbf{end for}\\$
{{< /pseudocode >}}

<br> 
<br> 

The agents are still fully aware of the environment and of each other. Namely, they observe the 
joint action and can base their policy on the whole game history, along with the current state.
In other words, at round $t$ a valid decision-rule for some agent will be:
$$
\begin{aligned}
d\_t^{\\, i} : (\mathcal{S}\times\mathcal{A})^{t-1}\times\mathcal{S} &\mapsto \Delta(\mathcal{A})\\; ,\\\
\bm{h}\_{t-1}, s\_t &\mapsto d\_t^{\, i}(\bm{h}\_{t-1}, s\_t)  \\; .
\end{aligned}
$$

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
A MDP is an instance of a stochastic game where $n=1$. In some settings, it can be tempting to try and solve a stochastic 
game as a large MDP. Unfortunately, the combinatorial nature of the action-space often makes this approach intractable. Therefore, 
even in such a fully observed setting, it makes sense to solve the MDP in a _decentralised_ fashion. This means
solving for each policy individually, instead of solving for the joint policy directly.
Since decentralised agents cannot access each other's actions during a given round,
training decentralised policies can come
with an additional _synchronisation_ challenge.
{{% /toggle_block %}}




## Partially Observable Stochastic Games
Partially observable stochastic games (POSGs) are the most general form of normal form games, and extend the notion of POMDPs 
to multi-agent setting. The additional ingredients to go from a stochastic game to a partially
observable stochastic game are _(1)_ a combinatorial observation space $\Omega = \Omega\_1\times\ldots\times\Omega\_n$ and 
_(2)_ an observation kernel $q(\cdot \vert s, \bm{a})$ defined as: 
$$
q(\bm{\omega} \vert s, \bm{a}) = \mathbb{P}\left(\bm{\omega}\_t = \bm{\omega} \vert s\_t = s, \bm{a}\_{t-1}=\bm{a} \right)
,\quad \forall\bm{\omega} = (\omega^1, \ldots, \omega^n)\in\Omega\\; .
$$

{{< boxed title="Partially observable stochastic game" >}} 
$$
\mathcal{M} = (n, \mathcal{S}, \mathcal{A}, \Omega, p, q, \bm{r})\; .
$$
{{< /boxed >}}


{{< pseudocode title="Partially Observable Stochastic Game" >}} 
$\textbf{init } \text{state } s_1\in\mathcal{S}, \text{ agents histories } {h}_0^1 \leftarrow \emptyset, \ldots, {h}_0^n \leftarrow \emptyset\; .\\$
$\textbf{for } t = 1, \ldots, T:\\$
$\qquad \text{the environment generates the joint observation vector } \bm{\omega}_t \sim q(\cdot \vert s_t, \bm{a}_{t-1})\\$
$\qquad \textbf{for } i = 1, \ldots, n:\\$
$\qquad \qquad \text{agent } i \text{ observes } \omega_t^i \text{ and places its action } a_t^i\\$
$\qquad \textbf{end for}\\$
$\qquad \text{the environment receives the joint action } \bm{a}_t = (a_t^1, \ldots, a_t^n) \\$
$\qquad \textbf{for } i = 1, \ldots, n:\\$
$\qquad \qquad \text{agent } i \text{ receives } r_t^i = r^i(\bm{a}_t)\\$
$\qquad\qquad \text{agent } i \text{ history gets updated: } h_t^i \leftarrow h_{t-1}^i \cup \{\omega_t^i, \, a_t^i\}\\$
$\qquad \textbf{end for}\\$
$\qquad \text{the environment state transitions: } \\$
$$
s_{t+1} \sim p(\cdot\vert  s_t, \bm{a}_t)
$$
$\textbf{end for}\\$
{{< /pseudocode >}}

<br> 
<br> 

The particularity of POSGs is that agents can neither observe the state $s\_t$ (similarly to a POMDP),
but neither can they access other agents' observations and past actions. Strategies must therefore be _decentralised_. 
Formally, this means that a decision rule for a given agent only ingests said agent history 
$h\_t^i = (\omega\_1^i, a\_1^i, \ldots, a\_{t-1}^i, \omega\_t^i)$. In other words, a valid decision-rule at round $t$ is:
$$
d\_t^i : (\Omega\_i\times\mathcal{A}\_i)^t \times \Omega\_i \mapsto \Delta(\mathcal{A}\_i)\\; .
$$

### Communications
Despite such a decentralised scenario, agents need not be utterly blind to each others. For instance, the joint observation
space $\Omega$ can explicitly include communication channels, on which agents will exchange information
(this degree of freedom is then included in the joint action space $\mathcal{A}$). 
Agents can also develop implicit communications; information regarding some agent $j$ (e.g. its position) can be available to agent $i$
via its observations $\omega\_i$. As such, agent $j$ can pass implicit signals to agent $i$ (_e.g._, by having its position jitter). 
Given the obligation for decentralised policies, _learning_ such coordination and communication mechanisms is one of the main challenges when it comes to solving POSGs.

### Decentralised POMDPs
Cooperative POSGs are called Decentralised POMDPs. In this setting, one needs to control
a _coalition_ of agents pursuing a common goal $r$, but with limited observations about their environment and each others.

Dec-POMDPs have arguably been one of the main centres of interest in the Multi-Agent RL community. 
Indeed, the cooperative nature of such games
(agents are optimising a common reward) removes the need for more ambiguous game-theoretic objectives. This allows 
one to focus on the development of algorithms that allows emergent coordination and communication mechanisms between agents. 
This cannot be done by direct extension of pure POMDPs (which is a Dec-POMDP with $n=1$). Indeed, the necessity of deploying decentralised policies 
clashes with [typical control methods for POMDPs](../pomdp); for instance, an agent-wise belief can no longer be computed (at least not without some approximations).


## Resources
This blog post's material is inspired from [\[Shoham and  Leyton-Brown\]](http://www.masfoundations.org/mas.pdf)
and the [Multi-Agent RL book](https://www.marl-book.com/).
