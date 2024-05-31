+++
author = "Louis Faury"
title = "Model-based RL  (1/?)"
date = "2024-05-15"
+++

This post is the first of a short series, concerned with model-based RL.
We will start walking this road via the principled trail: describing and analysing $\texttt{UCB-VI}$, a theoretically grounded algorithm to solve unknown finite MDPs.
More precisely, we will see how this model-based approach directly models the unknown MDP and uses optimism for strategic exploration to provably find "good" policies in finite-time.
Concentration inequalities and regret bounds incoming, fun!
<!--more-->

The $\texttt{UCB-VI}$ algorithm was introduced by [[Azar & al, 2017]](https://arxiv.org/pdf/1703.05449) at ICML. 
It comes with improved guarantees compared to its predecessors, such as UCRL [[Jaksch & al, 2010](https://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf)].


{{< infoblock>}}
$\quad$ Check out <a href="../rl_landscape_1" style="text-decoration:none; color:#0074aa;" ">this post</a> for a loose definition of model-based RL. $\texttt{UCB-VI}$ uses value-iteration as a backbone—check out 
<a href="../mdp_basics_3" style="text-decoration:none; color:#0074aa;" ">this post</a> to
refresh your memories about this control algorithm. For MDP notations, refer to  <a href="../mdp_basics" style="text-decoration:none; color:#0074aa;" ">this post</a>.
{{< /infoblock >}}


## Problem definition

Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, r)$ some given MDP.
We will assume than $\mathcal{S}$ and $\mathcal{A}$ are known (required), as well as the reward signal $r$ 
(not required, and after reading this post, you should see this is not a restriction, but rather a simplification for ease of exposition).
Only the transition kernel $\mathcal{P}$ is unknown to us. Wlog, we will also assume that the rewards are absolutely bounded by 1, that is 
$\vert r(s, a) \vert \leq 1$ for any $s, a\in\mathcal{S}\times\mathcal{A}$.

Straying away from [earlier posts](../mdp_basics) where we were concerned with a discounted, infinite-horizon criterion, 
we will here consider solving $\mathcal{M}$ under a finite-time criterion. That is, for some $H\in\mathbb{N}$ and initial state $s\_1\in\mathcal{S}$:
$$
\argmax\_{\pi\in\Pi^\text{MD}} v\_1^\pi(s\_1) := \mathbb{E}\_{s\_1}^\pi \left[\sum\_{h=1}^H r(s\_h, a\_h)\right]\\; .
$$
Recall that $\Pi^\text{MD}$ is the set containing all the sequences $(d\_1, \ldots, d\_H)$ of deterministic, Markovian decision rules.
We will denote $\pi^\star=(d\_1^\star, \ldots, d\_{H}^\star)$ the optimal policy and:
$$
    v\_h^\star(s) := \mathbb{E}\_{s}^{\pi^\star} \left[\sum\_{h^\prime=1}^H r(s\_h^\prime, d\_{h^\prime}(s\_h^\prime))\right] = 
\max\_{a\in\mathcal{A}} \left\\{r(s, a) + \mathbb{E}\_{s^\prime}^a\left[v\_{h+1}^\star(s^\prime)\right]\right\\}\\;.
$$ 
its $h$-tail value function (the second equality being a direct statement of Bellman's optimality principle).

We are interested in algorithms that, in finite-time, are able to find decent policies _without_ the knowledge of $\mathcal{P}$.
One such algorithm can experience the environment in a sequential fashion: at round $k$, it will submit a policy
$\pi\_k$, collect a roll-out, learn useful stuff (hopefully) and refine its policy for the subsequent round.
Our metric for success will be the regret:
$$
\begin{aligned}
\text{Regret}(K) &:= \sum\_{k=1}^K v\_1^\star(s\_1) - v\_1^{\pi\_k}(s\_1)\\;,\\\
&= K v\_1^\star(s\_1) - \sum\_{k=1}^K v\_1^{\pi\_k}(s\_1) \\; .
\end{aligned}
$$
The regret lives in $[0, K]$. 
If we fail to learn anything, we expect $\text{Regret}(K)$  to grow linearly with $K$ as we fail to
get closer the $\pi^\star$. 
On the other hand, if we manage to prove that $\text{Regret}(K) = \mathcal{O}(K^\alpha)$ with $\alpha<1$,
we will know that we are actually learning something -- the regret sublinear growth will be a testimony that the 
sub-optimality gap $v\_1^\star(s\_1) - v\_1^{\pi\_k}(s\_1)$ is getting smaller and smaller. Naturally, the smaller the $\alpha$, the better.

{{< warningblock >}}
$\quad \;\text{The regret is a random variable: it depends on the random realisation of the transitions that will impact}\\
\text{what we learn and how we update the policies }\{\pi_k\}\;.$
{{< /warningblock >}}

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
Despite the above warning block, we are considering a metric that does _partially_ hide the randomness
inherent to the process since the value function already averages trajectories. 
The only random quantities in the regret definition are the policies $\{\pi_k\}$. 
The regret definition we use is actually one called the _pseudo_-regret. 
A "true" regret would involve the actual rewards collected by $\pi^\star$ and $\pi\_k$ on each episodes.
This "true" regret is actually equal to the pseudo-regret up to some martingale difference -- therefore, 
with high-probability, the discrepancy is only $\sqrt{K}$.
{{% /toggle_block %}}

Beyond the growth of the regret w.r.t $K$ we are also interested in the dependency w.r.t to the horizon $H$, the size of the state space $S=\vert \mathcal{S}\vert$ and
 the action space $A=\vert \mathcal{A}\vert$, since it captures how the size of the MDP impacts the performance.
Other dependencies and universal numerical constants will be ignored, and hidden in the symbol $\blacksquare$.
We will also denot $\delta\in(0, 1]$ the algorithm's confidence level.


## $\texttt{UCB-VI}$

### Rationale

Let's first discuss why a naive approach would fail.
It can be tempting to adopt a greedy strategy: at round $k$, build some stimate $\hat{\mathcal{P}}\_k$ of the transition kernel
based on the observed transitions seen so far, run the VI algorithm on $\hat{\mathcal{M}}\_k = (\mathcal{S}, \mathcal{A}, \hat{\mathcal{P}}\_k, r)$
to compute $\pi\_{k+1}$, and go on like that. 
This strategy, of course, fails. Indeed, an unlucky transition at the very beginning of the experiment might
lead you to completely abandon a part of the state space, which is actually quite rewarding. 
To avoid this, we need some level of exploration. This exploration must be reasonable: if we explore all the time, we cannot expect sublinear regret. 
This is the classical exploration / exploitation trade-off. 

A principle way to find the correct balance between exploration and exploitation is the celebrated Optimism in Face of Uncertainty (OFU) principle.
The idea is simple: based on the logged transitions, we will build some confidence interval in which the true transition kernel lies with high-probability.
With some pretty terrible notations, think of something like $\mathcal{P}\in\[\mathcal{P}^-, \mathcal{P}^+\]$, with high probability. 
We will then be looking for an _optimistic_ MDP: one with a plausible transition kernel (_i.e_ one that lies inside the confidence interval) that guarantees the highest possible value:
$$
\mathcal{M}^\text{opt} \in \argmax\_{\[\mathcal{P}^-, \mathcal{P}^+\]} v\_1^\star(s\_1) \\; .
$$
We will then run VI on this optimistic MDP to create $\pi\_{k+1}$. The rationale is simple: if we were right to be optimistic, and we get a lot of rewards. 
If we were wrong, we are learning something useful for the future. 

{{< infoblock>}}
$\quad$ If you are unfamiliar with OFU, you might want to take a look at the bandit literature, a sibling of RL that studies the exploration/exploitation in isolation (without any planning-related complexity).
{{< /infoblock >}}


That's in a nutshell, the driving principle behind $\texttt{UCB-VI}$. 
You will see that the implementation looks a bit different; instead of searching for an optimistic MDP, we will use an exploration bonus when running the value-iteration algorithm.
This exploration bonus is based on the confidence interval's width. 
You can show that both approaches are _strictly_ equivalent. However, the bonus one is much easier to implement in practice. 


{{% toggle_block background-color="#CBE4FE" title="Note" %}}
In finite/linear bandits, this equivalence between confidence-set and bonus-based approached is strikingly easy to establish.
{{% /toggle_block %}}


### Algorithm
The rationale behind $\texttt{UCB-VI}$ is rather straight forward. 
After each roll-out, we will compute estimates $\hat{p}\_k$ of the transition probabilities, along with an exploration bonus $b\_k$ for each state-action pair. 
We will compute the value functions $\\{v\_1^k, q\_1^k, \ldots, q\_{H}^k, v\_{H+1}^k\\}$ by running a modified VI algorithm backward in time. 
This modified VI protocol will involve the estimates $\hat{p}\_k$ to mimic the forward dynamics and the exploration bonus $b\_k$ to promote exploration of promising state-action pairs.
The policy that will collect the roll-out will be greedy w.r.t those value functions (and henceforth deterministic). 
Formally, $\pi\_k = (d\_1^k, \ldots, d\_H^k)$ where:
$$
d\_{h}^{k+1}(s) \in \argmax\_{a\in\mathcal{A}} q\_h^k(s, a) \\; .
$$
with ties broken arbitrarily. 

We are now ready to detail some pseudo-code for $\texttt{UCB-VI}$. Below, we denote $s\_h^k$ (resp. $a\_h^k$) the state (resp. action) encountered at round $h$
during the collection of episode $k$. 

{{< pseudocode title="$\texttt{UCB-VI}$" >}} 
$\textbf{init } \text{history }\mathcal{H} \leftarrow \emptyset, \text{ policy } \pi_1, \text{ confidence level } \delta\;.\\$
$\textbf{for } k = 1, \ldots, K:\\$
$\quad\text{\color{YellowGreen}\# Executing}\\$
$\quad \text{collect a roll-out using } \pi_k\\$
$\quad \textbf{for } h =1 , \ldots, H\\$
$\quad\quad \text{pick } a_h^k \leftarrow d_h^k(s_h^k), \text{ observe } s_{h+1}^k\,\\$
$\quad \textbf{end for}\\$
$\quad \text{ add to history } \mathcal{H}\leftarrow \mathcal{H}\cup \{s_1^k, a_1^k, \ldots a_H^k, s_{H+1}^k\}\\$
$\quad\text{\color{YellowGreen}\# Learning}\\$
$\quad \text{compute visit counts for all } (s, a, s^\prime)\in\mathcal{H}:\\$
$$
\begin{aligned}
n_k(s, a) &\leftarrow \sum_{\mathcal{H}} \mathbf{1}[s_h^k=s, a_h^k=a] \;,\\
n_k(s, a, s^\prime) &\leftarrow \sum_{\mathcal{H}} \mathbf{1}[s_h^k=s, a_h^k=a, s_{h+1}^k=s^\prime] \;.
\end{aligned}
$$
$\quad \text{compute maximum likelihood estimators for all } (s, a, s^\prime)\in\mathcal{H}:\\$
$$
\hat p_k(s^\prime\vert s, a) \leftarrow \frac{n_k(s, a, s^\prime)}{n_k(s, a)}\;.
$$
$\quad \text{compute exploration bonus for all } (s, a)\in\mathcal{H}:\\$
$$
b_k(s, a) \leftarrow c / \sqrt{n_k(s, a)} \text{ with } c= \blacksquare H\log(SAKH/\delta)\; .
$$
$\quad\text{\color{YellowGreen}\# Planning}\\$
$\quad \text{set } v_{H+1}^k(s) \leftarrow 0 \text{ for all } s\in\mathcal{S}\;.\\$
$\quad \textbf{for } h =H , \ldots, 1\\$
$\quad \text{complete a value-iteration step by computing for all} (s, a)\in\mathcal{S}\times\mathcal{A}:\\$
$$
\begin{aligned}
q_h^k(s, a) &\leftarrow 
    \left\{\begin{aligned}
        &\min\left(H, r(s, a) + \sum_{s^\prime\in\mathcal{S}}\hat{p}_k(s^\prime\vert s, a)v_{h+1}^k(s^\prime)\right) &\text{ if }(s, a)\in\mathcal{H}\;,\\
        & H &\text{ otherwise.}
    \end{aligned}\right.\\
v_h^k(s) &\leftarrow \max_{a\in\mathcal{A}} q_h^k(s, a)\; .
\end{aligned}
$$
$\quad \textbf{end for}\\$
$\text{update } \pi_{k+1} \text{ by setting for any } h\in[1, H]\text{ and } s\in\mathcal{S}: \\$
$$
d_h^{k+1}(s) \in\argmax_{a\in\mathcal{A}} q_h^k(s, a)\;. 
$$
$\textbf{end for}\\$
$\textbf{return } \pi_{K+1}\;.$
{{< /pseudocode >}}
<br> 


Observe that the q-values are capped at $H$, their maximal possible value given the bounded reward assumption. This
also holds true for state-action pairs that have not been visited yet—this is a technique known as 
optimistic initialisation, which encourages the learning algorithm to quickly visit all action pairs.  


We are now ready to provide a regret bound for $\texttt{UCB-VI}$.
Because the regret is a random variable, this bound holds only with high-probability. 

{{< boxed title="Regret bound for $\texttt{UCB-VI}$" >}}
$\qquad \qquad \qquad\qquad \qquad \qquad \text{ With probability at least } 1-\delta:$
$$
\text{Regret}(K) \leq \tilde{\blacksquare} \left(H\sqrt{SAKH} + H^3 S^2 A\right)\;,
$$
$\text{where }\tilde{\blacksquare}\text{ hides universal numerical constants and logaritmic dependencies in } S,\, A, \, K \text{ and }H\,.$
{{< /boxed >}}

Let's enjoy for a moment the nice $\sqrt{K}$ regret growth, 
ensuring us of $\texttt{UCB-VI}$'s statistical efficiency. 
We also notice in the regret leading term a sublinear dependency $\sqrt{SA}$ in the size of 
the MDP. Actually, the known regret lower-bound is of the order $\tilde{\Omega}(\sqrt{HSAK})$, which
$\texttt{UCB-VI}$ matches but for the growth with the horizon length $H$.
(A variant of $\texttt{UCB-VI}$ presented in [[Azar & al, 2017]](https://arxiv.org/pdf/1703.05449) matches this lower-bound
by using a refined alternative for the exploration bonus, but we won't explore this version today.)


{{% toggle_block background-color="#CBE4FE" title="Note" %}}
The high-probability bound naturally translates into a similar bound for the expected regret $\mathbb{E}\[\text{Regret}(K)\]$.
Denoting $\mathcal{E}\_\delta$ the "nice" event under which $\text{Regret}(K) \leq \tilde{\blacksquare} \left(H\sqrt{SAKH} + H^3 S^2 A\right)$, we have:
$$
\begin{aligned}
\mathbb{E}\[\text{Regret}(K)\] &= \mathbb{E}\[\text{Regret}(K)\vert \mathcal{E}\_\delta\]\mathbb{P}(\mathcal{E}\_\delta) + \mathbb{E}\left\[\text{Regret}(K)\middle\vert \mathcal{E}\_\delta^c\right\]\mathbb{P}(\mathcal{E}\_\delta^c)\\;, \\\
&\leq  \tilde{\blacksquare} \left(H\sqrt{SAKH} + H^3 S^2 A\right) + KH \delta\\; .
\end{aligned}
$$
Setting $\delta \propto (KH)^{-1}$ settles the issue as we retrieve a similar bound. 
{{% /toggle_block %}}


## Analysis
We now move into the fun part where we get to prove the claimed upper-bound. 
The recipe for bounding the regret is somewhat classical once you are used to this type of algorithm. 
We will first remove the unknown quantity (the ground-truth $v\_h^\star$) by invoking optimism, and then show
that the online error gracefully degrades. 
First, we will need to provide a high-probability bound on the estimation error by resorting to a concentration of measure argument. 

### Concentration inequalities
To give a somewhat self-contained proof, we first provide some remainder on concentration inequalities.
Throughout, we will use the Chernoff-Hoeffding (CH) inequality for i.i.d and martingale difference sequence noise, 
as well as Bernstein inequality.

We state (and prove) a functional version of the Chernoff-Hoeffding bound.
Let $\mathcal{X}$ a finite set, $\nu$ a p.m.f over $\mathcal{X}$ and $f: \mathcal{X}\mapsto\mathbb{R}$ a bounded real-valued function.
Let $x^-, x^+\in\mathbb{R}$ such that $f(x) \in [x^-, x^+]$ for any $x\in\mathcal{X}$, and $F:= x^+-x^-$.
Given $(x\_1, \ldots, x\_n)$ some i.i.d samples from $\nu$, the MLE estimator of the probability mass allocated to some $x\in\mathcal{X}$ writes
$
\hat\nu\_n(x):= \sum\_{x\_i} \mathbf{1}[x\_i=x]/n \text{ for any } x\in\mathcal{X}\\; .
$
Let $\mu := \mathbb{E}\_\nu[f]$ the expected value of $f$ under $\nu$ and $\hat\mu\_n := \sum\_{x\in\mathcal{X}} \hat\nu\_n(x) f(x) $
its estimator. Then for any $\delta\in(0, 1]$:

$$
\tag{1}
\mathbb{P}\left(\vert \mu - \hat\mu_n\vert \leq \frac{2F}{\sqrt{n}} \sqrt{\log(2/\delta)}\right) \geq 1-\delta\; .
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Observe that:
$$
\begin{aligned}
\hat\mu\_n - \mu &= \sum\_{x\in\mathcal{X}} f(x)\left(\frac{1}{n}\sum\_{i=1}^n \mathbf{1}[x\_i=x]\right) - \sum\_{x\in\mathcal{X}} f(x)\nu(x)\\;, \\\
&= \frac{1}{n}\sum\_{i=1}^n \sum\_{x\in\mathcal{X}} \left(\mathbf{1}[x\_i=x] - \nu(x)\right)f(x)\\;, &(\text{re-arranging}) \\\
&=  \frac{1}{n}\sum\_{i=1}^n \eta\_i
\end{aligned}
$$
where $\eta\_i := \sum\_{x\in\mathcal{X}} (\mathbf{1}[x\_i=x] - \nu(x))f(x)$ and $\vert \eta\_i \vert \leq F$ thanks to the boundedness of $f$.
By a straight-forward application of the CH inequality we obtain the claimed result:
$$
\mathbb{P}\left(\vert \hat\mu\_n - \mu \vert \leq 2F \sqrt{\log(2/\delta)/n}\right) \geq 1-\delta\\; .
$$

{{% /toggle_block %}}

We will also need the Azuma-Hoeffding bound, stated below in its "normal" form. 
Let $\\{\eta\_i\\}\_i$ some martingale difference sequence (_i.e_ $\mathbb{E}[\eta\_t \vert \eta\_{<t}]=0$)
such that $\vert \eta\_t\vert < c$ a.s. for any $t\in\mathbb{N}$. Then:
$$
\tag{2}
\mathbb{P}\left(\left\vert \sum\_{i=1}^n \eta\_i \right\vert \leq c\sqrt{2n\log(2/\delta)\}\right)\geq 1-\delta\\; .
$$
{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
You can easily prove this by reproducing the "normal" (read: non-functional) Chernoff-Hoeffding bound
without forgetting that the noise is no longer i.i.d (the tower-rule is your friend).
{{% /toggle_block %}}

We will also need Bernstein inequality—an alternative to CH which
takes variance effects into account. Letting $\\{\eta\_i\\}\_i$ be a collection of i.i.d centered random variable
absolutely bounded by some constant $c$, and denoting $\sigma^2 = \text{Var}(\eta\_i)$ we have for any $\delta\in(0,1]$:
$$
\tag{3}
\mathbb{P}\left(\left\vert \frac{1}{n}\sum\_{i=1}^n \eta\_i \right\vert \leq \sqrt{\frac{\sigma^2}{n}}\sqrt{\log(2/\delta)} + \frac{2c}{3n}\log(2/\delta)\right) \geq 1-\delta\\;. 
$$

### High-probability event
The first step in bounding the regret starts by amputing it of the unknown quantity $v\_1^\star$.
This is done via optimism: it actually turns out that, _often enough_, we are optimistic: $v\_1^k(s\_1) \geq v\_1(s\_1)$ for any $k\in\{1, \ldots, K\}$.
That's useful, as this allows us to write:
$$
\text{Regret}(T) \leq \sum\_{k=1}^K v\_1^k(s\_1) - v\_1^{\pi\_k}(s\_1) \\; ,
$$
which we will in turn see is small. 
Before that, we need to quantify what is "often". We lower-bound the probability of an event that is even larger that what we need for optimism.



{{< boxed title="High-probability event" >}}
$\qquad\qquad\qquad\qquad\qquad\quad\text{Let } \mathcal{E}_\delta = \{ v_h^k(s) \geq v_h^\star(s), \; \forall s\in\mathcal{S}, \; \forall k\leq K, \, \forall h\leq H\}\text{. Then }
\mathbb{P}(\mathcal{E}) \geq 1-\delta\;.
$
{{< /boxed >}}

The proof involves backward induction, applying the functional CH bound at every step, along with some union bounds here and there.
In the following lines, fix $k\leq K$. Observe that $v\_{H+1}^k \equiv  v\_{H+1}^\star\equiv 0$ by construction. 
Now, assume that $v\_{h+1}^k(s) \geq v\_{h+1}^\star(s)$ for some $h\leq H$ and any $s\in\mathcal{S}$. Also, for any $s, a\in\mathcal{S}\times\mathcal{A}$:
$$
\begin{aligned}
q\_{h}^k(s, a) &= r(s, a) + \sum\_{s^\prime}\hat{p}\_k(s^\prime\vert s, a) v\_{h+1}^k(s^\prime) + b\_k(s, a)\\;, &\text{(def)}\\\
&\geq r(s, a) + \sum\_{s^\prime}\hat{p}\_k(s^\prime\vert s, a) v\_{h+1}^\star(s^\prime) + b\_k(s, a)\\;, &\text{(induction hyp.)}\\\
&= r(s, a) + \sum\_{s^\prime}p(s^\prime\vert s, a) v\_{h+1}^\star(s^\prime) + \sum\_{s^\prime}\left(\hat{p}(s^\prime\vert s, a)-p(s^\prime\vert s, a)\right) v\_{h+1}^\star(s^\prime) + b\_k(s, a)\\;, \\\
&= q\_h^\star(s, a) + \sum\_{s^\prime}\left(\hat{p}(s^\prime\vert s, a)-p(s^\prime\vert s, a)\right) v\_{h+1}^\star(s^\prime) + b\_k(s, a)\\;. &\text{(def of $q\_h^\star$)}\\\
\end{aligned}
$$
An application of CH along with a union bound over $\mathcal{S}\times\mathcal{A}$ yields that with probability at least $1-\delta/(KH)$:
$$
\sum\_{s^\prime}\left(\hat{p}(s^\prime\vert s, a)-p(s^\prime\vert s, a)\right) v\_{h+1}^\star(s^\prime) \geq -b\_k(s, a)\quad\text{ for all } s, a\in\mathcal{S}\times\mathcal{A}\\;,
$$
hence concluding the induction, as under this event we have that $q\_{h}^k(s, a) \geq q\_h^\star(s, a)$. 
The same ordering follows for the state-value function since
$$
v\_h^{k}(s) = \max\_{a}q\_h^{k}(s, a) \geq \max\_{a}q\_h^{\star}(s, a) = v\_h^\star(s) \\; .
$$
Only a union bounded over $h\leq H$ and $k\leq K$ is then needed to prove the desired claim. 

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
We make here the statement regarding the application of CH more formal.
Using the fact that $\vert v\_h^\star\vert \leq H$, applying $(1)$ for a fixed state-action pair $(s, a)$
yields that:
$$
\begin{aligned}
1-\delta/(KHSA) &\leq \mathbb{P}\left(\sum\_{s^\prime}\left(\hat{p}(s^\prime\vert s, a)-p(s^\prime\vert s, a)\right) v\_{h+1}^\star(s^\prime) \leq \frac{2H}{\sqrt{n}}\sqrt{\log(2KHSA/\delta)}\right)\\;, \\\
&\leq \mathbb{P}\left(\sum\_{s^\prime}\left(\hat{p}(s^\prime\vert s, a)-p(s^\prime\vert s, a)\right) v\_{h+1}^\star(s^\prime) \leq b\_k(s, a)\right)\\;.
\end{aligned}
$$
An union bound over $\mathcal{S}\times\mathcal{A}$, $h\leq H$ and $k\leq K$ will seal the deal:
$$
\mathbb{P}\left(\sum\_{s^\prime}\left(\hat{p}(s^\prime\vert s, a)-p(s^\prime\vert s, a)\right) v\_{h+1}^\star(s^\prime) \leq b\_k(s, a), \\; \forall s\in\mathcal{S}, \\;\forall a\in\mathcal{A}, \\;\forall k\leq K, \\;\forall h\leq H\right) \geq 1-\delta \\; .
$$
{{% /toggle_block %}}



### Bounding the regret
At this point, we know that with probability at least $1-\delta$ we have:
$$
\begin{aligned}
\text{Regret}(T) &\leq \sum\_{k=1}^K v\_1^k(s\_1) - v\_1^{\pi\_k}(s\_1) \\; , \\\
&= \sum\_{k=1}^K\Delta\_1^k(s\_1)\\;,
\end{aligned}
$$
where $\Delta\_h^k(s):=v\_h^{k}(s) - v\_{h}^{\pi^k}(s)$.
Let's see how we can show that the r.h.s is small enough.
Our strategy is to bound this term in a recursive fashion,  relating $\Delta\_h^k$ to $\Delta\_{h+1}^k$.
$$
\begin{aligned}
\Delta\_h^k(s\_h^k) &= v\_h^{k}(s\_h^k) - v\_{h}^{\pi^k}(s\_{h}^k)\\;,\\\
&= q\_h^k(s\_h^k, a\_h^k) - q\_h^{\pi\_k}(s\_h^k, a\_h^k)\\;, &(a\_h^k= \in \argmax\_a  q\_h^k(s, a)) \\;, \\\
&= \sum\_{s^\prime} \hat p\_k(s^\prime\vert s\_h^k, a\_h^k) v\_{h+1}^k(s^\prime) + b\_k(s\_h^k, a\_h^k)- \sum\_{s^\prime} p(s^\prime\vert s\_h^k, a\_h^k) v\_{h+1}^{\pi\_k}(s^\prime)\\;, &(\text{def}) \\\
&= \sum\_{s^\prime} (\hat p\_k-p)(s^\prime\vert s\_h^k, a\_h^k) v\_{h+1}^k(s^\prime) + b\_k(s\_h^k, a\_h^k)- \sum\_{s^\prime} p(s^\prime\vert s\_h^k, a\_h^k)\Delta\_{h+1}^k(s^\prime)\\; \\\
&= \underbrace{\sum\_{s^\prime} (\hat p\_k-p)(s^\prime\vert s\_h^k, a\_h^k) v\_{h+1}^\star(s^\prime)}\_{\textcircled{1}} + \underbrace{\sum\_{s^\prime} (\hat p\_k-p)(s^\prime\vert s\_h^k, a\_h^k) (v\_{h+1}^k - v\_{h+1}^\star)(s^\prime)}\_{\textcircled{2}} + b\_k(s\_h^k, a\_h^k)- \underbrace{\sum\_{s^\prime} p(s^\prime\vert s\_h^k, a\_h^k)\Delta\_{h+1}^k(s^\prime)}\_{\textcircled{3}}\\;, \\\
\end{aligned}
$$
where the last two lines are simply re-arranging. 

We have actually already bounded $\textcircled{1}$ a few lines ago. Under $\mathcal{E}$:
$$
\textcircled{1} \leq b\_k(s\_h^k, a\_h^k) \\; .
$$


Treating $\textcircled{2}$ is slightly more involved, but the bottom-line is that this term relates to $\textcircled{3}$:
$$
\textcircled{2} \leq \tilde\blacksquare \left(H^{-1} \textcircled{3} + \frac{SH^2}{n\_k(s\_h^k, a\_h^k)}\right)\\;.
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The first part of the proof implies applying Bernstein's inequality $(3)$ to $\hat{p}\_k(s^\prime\vert s\_h^k, a\_h^k) - p(s^\prime\vert s\_h^k, a\_h^k)$.
First, we need to realise that we can write:
$$
\hat{p}\_k(s^\prime\vert s\_h^k, a\_h^k) - p(s^\prime\vert s\_h^k, a\_h^k) = \frac{1}{n\_k(s\_h^k, a\_h^k)}\sum\_{i=1}^{n\_k(s, a)}\eta\_i \\;, 
$$
where $\eta\_i$ is a centered Bernoulli random variable with variance $p(s^\prime\vert s\_h^k, a\_h^k)(1-p(s^\prime\vert s\_h^k, a\_h^k))$. By $(3)$, we 
therefore have that with probability at least $1-\delta$:
$$
\begin{aligned}
\left\vert \hat{p}\_k(s^\prime\vert s\_h^k, a\_h^k) - p(s^\prime\vert s\_h^k, a\_h^k)  \right\vert &\leq \blacksquare \left(\frac{\sqrt{p(s^\prime\vert s\_h^k, a\_h^k)(1-p(s^\prime\vert s\_h^k, a\_h^k))}}{\sqrt{n\_k(s\_h^k, a\_h^k)}}\sqrt{\log(2KH/\delta)} + \frac{\log(2KH/\delta)}{n\_k(s\_h^k, a\_h^k)}\right) \\; ,\\\
&\leq \left(\frac{\sqrt{p(s^\prime\vert s\_h^k, a\_h^k)}}{\sqrt{n\_k(s\_h^k, a\_h^k)}}\sqrt{\log(2KH/\delta)} + \frac{\log(2KH/\delta)}{n\_k(s\_h^k, a\_h^k)}\right) \\; .
\end{aligned}
$$
Note the dependency in $KH$ in the logarithm -- an additional union bound using the fact that $n\_k(s\_h^k, a\_h^k) \leq KH$ a.s. was need to take into account the fact that $n\_k(s, a)$ is a random variable.
Combining this with the fact that $v\_{h+1}^{k}(s^\prime) - v\_{h+1}^\star(s^\prime) \geq 0$  for any $s^\prime\in\mathcal{S}$ under $\mathcal{E}$ we obtain:
$$
\begin{aligned}
\textcircled{2} &= \sum\_{s^\prime} (\hat p\_k-p)(s^\prime\vert s\_h^k, a\_h^k) (v\_{h+1}^k - v\_{h+1}^\star)(s^\prime) \\;, \\\
&\leq \blacksquare\sum\_{s^\prime} \left(\frac{\sqrt{p(s^\prime\vert s\_h^k, a\_h^k)}}{\sqrt{n\_k(s, a)}}\sqrt{\log(2KH/\delta)} + \frac{\log(2KH/\delta)}{n\_k(s\_h^k, a\_h^k)}\right)(v\_{h+1}^k - v\_{h+1}^\star)(s^\prime) \\;, &\\\
&\leq  \blacksquare\sum\_{s^\prime} \left(\frac{\sqrt{p(s^\prime\vert s\_h^k, a\_h^k)}}{\sqrt{n\_k(s\_h^k, a\_h^k)}}\sqrt{\log(2KH/\delta)} + \frac{\log(2KH/\delta)}{n\_k(s\_h^k, a\_h^k)}\right)(v\_{h+1}^k - v\_{h+1}^{\pi\_k})(s^\prime)  \\;, & (v\_{h+1}^\star \geq v\_{h+1}^{\pi\_k})\\\
&\leq \tilde{\blacksquare} \sum\_{s^\prime}  \left(\frac{\sqrt{p(s^\prime\vert s\_h^k, a\_h^k)}}{\sqrt{n\_k(s\_h^k, a\_h^k)}}+ \frac{1}{n\_k(s\_h^k, a\_h^k)}\right) (v\_{h+1}^k(s^\prime) - v\_{h+1}^{\pi\_k}(s^\prime))\\;, \\\
&\leq \tilde{\blacksquare} \sum\_{s^\prime}  \left(p(s^\prime\vert s\_h^k, a\_h^k)/H+ \frac{H}{n\_k(s\_h^k, a\_h^k)}\right) (v\_{h+1}^k(s^\prime) - v\_{h+1}^{\pi\_k}(s^\prime))\\;, &(2ab \leq a^2 + b^2) \\\
&\leq  \frac{\tilde{\blacksquare}}{H} \sum\_{s^\prime} p(s^\prime\vert s\_h^k, a\_h^k) (v\_{h+1}^k(s^\prime) - v\_{h+1}^{\pi\_k}(s^\prime)) + \frac{\tilde\blacksquare SH^2}{n\_k(s\_h^k, a\_h^k)} \\; . & (\vert v\_{h+1}^k(s^\prime) - v\_{h+1}^{\pi\_k}(s^\prime) \vert \leq H)
\end{aligned}
$$
{{% /toggle_block %}}

Putting everything together, we obtain the high probability bound:
$$
\begin{aligned}
\Delta\_h^k(s\_h^k) \leq \tilde\blacksquare \left( b\_k(s\_h^k, a\_h^k)  + (1+1/H) \sum\_{s^\prime} p(s^\prime\vert s\_h^k, a\_h^k)\Delta\_{h+1}^k(s^\prime) + \frac{SH^2}{n\_k(s\_h^k, a\_h^k)}\right)\\;.
\end{aligned}
$$

{{< warningblock >}}
$\quad\, \text{It can seem dangerous to hide constants in } \blacksquare \text{ since they could compound in an exponential fashion while}\\
\text{ unrolling the recurrence. 
Carrying out the computation while minding the constants proves this is safe.}$
{{< /warningblock >}}

We almost have the recursive relationship we were gunning for -- the r.h.s is still missing an 
explicit mention to $\Delta\_h^k(s\_{h+1}^k)$. By adding and removing $(1+H)^{—1}\Delta\_h^k(s\_{h+1}^k)$ and
introducing the terms 
$\xi\_{h}^k := v\_{h+1}^k(s\_{h+1}^k) - \mathbb{E}\_{s\_h^k}^{a\_h^k}[v\_{h+1}^k(s^\prime)]$ and $\zeta\_{h}^k := v\_{h+1}^{\pi\_k}(s\_{h+1}^k) - \mathbb{E}\_{s\_h^k}^{a\_h^k}[v\_{h+1}^{\pi\_k}(s^\prime)]$ 
we finally obtain:
$$
\Delta\_h^k(s\_h^k) \leq \tilde\blacksquare \left( b\_k(s\_h^k, a\_h^k)  + (1+1/H)(\xi\_{h+1}^k + \zeta\_{h+1}^k + \Delta\_{h+1}^k(s\_{h+1}^k)) + \frac{SH^2}{n\_k(s\_h^k, a\_h^k)}\right)\\;.
$$

Observe that $\\{\xi\_h^k\\}\_h$ and $\\{\zeta\_h^k\\}\_h$ are martingale difference sequences -- for instance, $\mathbb{E}[\xi\_h^k \vert s^k\_{h^\prime < h}] = 0$.
Unrolling, and using the fact that $(1+1/x)^x \leq 3$ we eventually obtain that:
$$
\Delta\_1^k(s\_1^k) \leq \tilde\blacksquare \sum\_{h=1}^H b\_k(s\_h^k a\_h^k) + \xi\_h^k + \zeta\_h^k + \Delta\_{h+1}^k(s\_{h+1}^k) +  \frac{SH^2}{n\_k(s\_h^k, a\_h^k)} \\; ,
$$
and therefore:
$$
\tag{4}
\text{Regret}(T) \leq \tilde\blacksquare \sum\_{k=1}^K \sum\_{h=1}^H b\_k(s\_h^k a\_h^k) + \xi\_h^k + \zeta\_h^k +   \frac{SH^2}{n\_k(s\_h^k, a\_h^k)} \\; ,
$$
The last term is a second-order term; it is easy to show that:
$$
\sum\_{k, h} \frac{1}{n\_k(s\_h^k, a\_h^k)} \leq \tilde\blacksquare SAH \\; .
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof follows a standard argument if you're familiar with multi-arm bandits. 
$$
\begin{aligned}
\sum\_{k, h} \frac{1}{n\_k(s\_h^k, a\_h^k)} &= \sum\_{k, h} \sum\_{s, a} \frac{1}{n\_k(s, a)}  \mathbf{1}[s\_h^k=s, a\_h^k=a] \\;, \\\
&= \sum\_{s, a, h}  \sum\_{k} \frac{1}{n\_k(s, a)}  \mathbf{1}[s\_h^k=s, a\_h^k=a]\\;, \\\
&\leq  \sum\_{s, a, h}  \sum\_{k} \frac{1}{k} \\;, \\\
&\leq \blacksquare  \sum\_{s, a, h} \log(K) \\;, \\\
&\leq \blacksquare SAH \log(K)\\; .
\end{aligned}
$$
{{% /toggle_block %}}


By a direct application of the Azuma-Hoefffing inequality $(2)$, and using the fact that 
both $\vert \xi\_k^h \vert \leq H$ and $\vert \zeta\_k^h \vert \leq H$, we have that with high-probability:
$$
\sum\_{h=1}^H \sum\_{k=1}^K \xi\_h^k + \zeta\_h^k \leq \blacksquare H\sqrt{KH} \\; .
$$
We are left with bounding the first term in (4). Using the definition of $b\_k$ we have:
$$
\begin{aligned}
\sum\_{k=1}^K\sum\_{h=1}^H b\_k(s\_h^k, a\_h^k) &= \tilde\blacksquare H \sqrt{HKSA}\\; .
\end{aligned}
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof repeats the one we used to bound $\sum\_{h, k} \frac{1}{n\_k(s\_h^k, a\_h^k)} \leq \tilde\blacksquare HSA$, 
using the fact that $\sum\_{k=1}^K 1\sqrt{k} \leq \sqrt{K}$.
{{% /toggle_block %}}

Combining everything together, we finally obtain the claimed high-probability bound:
$$
\text{Regret}(T) \leq \tilde\blacksquare \left(H\sqrt{HSAK} + H^3S^2 A\right)\\;.
$$

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
Some union bounds here and there are skipped to combine all the right high-probability event together.
{{% /toggle_block %}}


### Outlook
At this point, we are now convinced that $\texttt{UCB-VI}$ is statistically efficient, with a leading regret
term scaling as $\mathcal{O}(H\sqrt{HSAK})$. It gives us a principled tool to reduce epistemic uncertainty, and properly balance the exploration-exploitation via
optimism. 

Of course, $\texttt{UCB-VI}$' practical usefulness doesn't reach much further than small MDPs since only storing $\hat p\_k$ is $\Omega(S^2A)$. 
The main ideas can be extended to structured MDPs, and do support simple (_i.e_ linear) function approximation.
It is natural to wonder whether optimism can be used as a plug-in in the multitude of RL algorithms powered by deep-learning, which painfully miss strategic exploration. 
Unfortunately, the answer so far has been mostly negative, for two reasons: the complexity of building meaningful confidence sets for neural network parameters, and the considerable computational overhead required to search for an optimistic parametrisation of such models. 
Instead, there has been some limited yet convincing success porting ideas from another paradigm: Thompson sampling (Bayesian stuff). That would be for another time, though. 
