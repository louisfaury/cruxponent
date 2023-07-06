+++
author = "Louis Faury"
title = "A Tale of Policy Gradients"
description = "A short theoretical dive in Policy Gradients"
date = "2023-06-21"
+++

This post covers the derivation of several policy gradients expressions.
It browses through the vanilla policy gradient and the policy gradient theorem to finish at deterministic policy gradients.
No attention is given to optimisation aspects (whether and how policy gradients can find locally optimal policies): the focus is the gradient's (semi) formal derivation. 
<!--more-->

## Setting
We'll consider a _finite_, _stationary_ Markov Decision Process (MDP)
$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}\_{s}^a, r)$ under an infinite-horizon, discounted criterion.
Recall the classical definitions for state / state-action value functions:
$$
\begin{aligned}
v\_\lambda^\pi(s) &:= \mathbb{E}\_s^\pi \left[ \sum\_{t=1}^\infty \lambda^{t-1}r(s\_t, a\_t)\right]\\; ,\\\
q_\lambda^\pi(s, a) &:= r(s, a) + \mathbb{E}\_{s'\sim \mathcal{P}\_s^a}\left[v_\lambda^\pi(s')\right] \\; .
\end{aligned}
$$

for any $s, a \in \mathcal{S}, \mathcal{A}$, and $\pi$ some stationary policy. The notation $\mathbb{E}\_s^\pi$ is fairly standard, 
and averages out any stochastic realisation (from the MDP or the policy), starting at state $s_1 = s$ and following $\pi$. 


{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
We consider a finite MDP (i.e. $\vert \mathcal{A}\vert\cdot\vert\mathcal{S}\vert<\infty$) for convenience. This will free us 
from justifying swapping integrals. However, all statements will remain true with countable or continuous state and action spaces.
{{% /toggle_block %}}

Since classical MDP theory \[[Puterman](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887), 6.2.4\] tells us
that it exists an optimal stationary policy, we will stick with this policy class in this post. Also, in contrast with value-based methods,
we will consider stochastic policies (more on that later).
Finally, policies will be _parametrized_ by some parameter $\theta$, living in some Euclidian space (typically the weights
of a neural network).


Before moving forward, we must define a notion of _utility_ for a given policy $\pi_\theta$ -- that is, the criterion that
we will effectively optimize. 
Assuming that the initial state of the MDP always is some state $s^0$, a natural definition of utility is the policy's
value at $s^0$.
$$
U(\theta) := v_\lambda^{\pi_\theta}(s^0)\\; .
$$

{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
This criterion is easily extended to set-up where the initial state is drawn for some initial distribution
over states.
{{% /toggle_block %}}

## Vanilla Policy Gradient
As we shall see shortly, the Vanilla Policy Gradient (VPG) is deeply rooted in Monte-Carlo estimation.
It is therefore adapted to episodic settings: in what follows, we assume there is an absorbing state that every policy eventually reaches
during the first $T$ rounds.
 
The interaction between the environment transition kernel $\mathcal{P}\_s^a$ and the policy $\pi_\theta$ induces a _probability measure_ $\mathcal{T}\_\theta$ over trajectories 
$\tau = (s\_1, a\_1, \ldots, a\_{T-1}, s\_T)$. 
Notice that denoting $R_\\tau:= \sum\_{t=1}^\infty \lambda^{t-1}r(s\_t, a\_t)$ the total return of a trajectory $\tau$, we can rewrite $U(\theta)$ as:
$$
    U(\theta) = \mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}[R\_\tau] \\; .
$$
Our goal is to express $\nabla\_\theta U(\theta)$ as an expectation under $\mathcal{T}\_\theta$; this will allow
us to efficiently compute _stochastic_ gradients by sampling trajectories. It can then be fed to your 
favorite stochastic gradient based optimiser to generate better policies -- according to $U(\theta)$.


### Likelihood Ratio
The likelihood ratio is a neat little trick that allows us to do just that. In short, it asserts that:
$$
\nabla\_\theta U(\theta) = \mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[R\_\tau \nabla\_\theta \mathcal{T}\_\theta(\tau)\right] \\; .
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof main ingredient is to notice that for any function $f$ we have the identity $\nabla\_\theta f(\theta) =   f(\theta)\nabla\_\theta \log f(\theta)$. Indeed:
$$
\begin{aligned}
\nabla\_\theta U(\theta) &= \nabla\_\theta \sum_{\tau} R(\tau) \mathcal{T}\_\theta(\tau)\\; ,\\\
&= \sum_{\tau} R(\tau)\nabla\_\theta \mathcal{T}\_\theta(\tau)\\; ,\\\
&= \sum_{\tau} R(\tau)\mathcal{T}\_\theta(\tau)\nabla\_\theta \log\mathcal{T}\_\theta(\tau)\\; ,\\\
&= \mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[R\_\tau \nabla\_\theta \mathcal{T}\_\theta(\tau)\right] \\; .
\end{aligned}
$$
{{% /toggle_block %}}

Analyzing $\mathcal{T}\_\theta$ shows that this can be further simplified. Indeed:
$$
\begin{aligned}
\mathcal{T}\_\theta(\tau) &= \mathbb{P}\_s^\pi(s\_1, a\_1, \ldots, a\_{T-1}, s\_T)\\; ,\\\
&\overset{(i)}{=} \mathbb{P}\_s^\pi(s\_T\vert s\_1, a\_1, \ldots, a\_{T-1})\mathbb{P}\_s^\pi(a\_{T-1}\vert a\_1, \ldots, s\_{T-1})\mathbb{P}\_s^\pi(s\_1, a\_1, \ldots, s\_{T-1}) \\; ,\\\
&\overset{(ii)}{=} \mathcal{P}\_{s\_{\_{T-1}}}^{a\_{\_{T-1}}}(s\_T) \pi\_\theta(a\_{T-1}\vert s\_{T-1})\mathbb{P}\_s^\pi(s\_1, a\_1, \ldots, s\_{T-1})\\; ,\\\
&\overset{(iii)}{=} \mathbf{1}[s\_1 = s^0]\prod\_{t=1}^{T-1}\mathcal{P}\_{s\_{\_{t}}}^{a\_{\_{t}}}(s\_{t+1})\pi_\theta(a\_t \vert s\_t )\\; ,
\end{aligned}
$$
where $(i)$ is given by Bayes rule, $(ii)$ uses the environment's and policy's Markovian nature and 
$(iii)$ unrolls until $t=1$.
Therefore whenever $s\_1 = s^0$:
$$
\begin{aligned}
\nabla\_\theta \log\mathcal{T}\_\theta(\tau) &= \nabla\_\theta \sum\_{t=1}^{T-1} \left[\log\mathcal{P}\_{s\_{\_{t}}}^{a\_{\_{t}}}(s\_{t+1})+ \log\pi_\theta(a\_t \vert s\_t )\right] \\; ,\\\
&= \sum\_{t=1}^{T-1} \nabla\_\theta\log\pi_\theta(a\_t \vert s\_t ) \\; .
\end{aligned}
$$
Notice how we therefore do not need to explicitely know the transition kernel to compute this gradient. All in all, we are left
with the following expression:
$$
\nabla\_\theta U(\theta) = \mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[R\_\tau\sum\_{t=1}^{T-1} \nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\right]\\; ,
$$
for which unbiased estimators are easily computed. 


### Reducing Variance
As usual with Monte Carlo methods, variance can be a painful thorn in our shoe. Turns out, stochastic estimators
of $\nabla\_\theta U(\theta)$ computed according to the above formula come with substantial variance. Thankfully we have a couple tricks up our sleeves to trim it, 
without having to introduce any bias.

#### Temporal Structure
Our current gradient expression can be further simplified by observing that is contains a
bunch of terms that do not contribute _in expectation_. Indeed note that if $k<t$:
$$
\begin{aligned}
\mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[r(s\_k, a\_k)  \nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\right]\\, .
\end{aligned}
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" %}}
This is a direct consequence of the tower rule -- this is why 
we must have $k<t$. Without using the tower rule, we will have to
explicit conditional expectations:
$$
\begin{aligned}
\mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[r_k  \nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\right] 
&\overset{(i)}{=} \sum_{s, a, s'} r(s, a)\mathbb{P}\_{s^0}^\pi(s\_k = s, a\_k=a, s\_t = s') 
\mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[\nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\middle\vert s\_k = s, a\_k=a, s\_t = s' \right] \\; ,\\\
&\overset{(ii)}{=} \sum_{s, a, s'} r(s, a)\mathbb{P}\_{s^0}^\pi(s\_k = s, a\_k=a, s\_t = s') 
\mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[\nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\middle\vert s\_t = s' \right]\\; , \\\
&= \sum_{s, a, s'} r(s, a)\mathbb{P}\_{s^0}^\pi(s\_k = s, a\_k=a, s\_t = s') 
\mathbb{E}\_{a'\sim\pi\_{\theta}(\cdot\vert s')}\left[\nabla\_\theta\log\pi\_\theta(a'\vert s')\right]\\; . \\\
\end{aligned}
$$
where $(i)$ is the law of total expectation and $(ii)$ use the MDP's temporal structure. Observe that:
$$
\begin{aligned}
\mathbb{E}\_{a'\sim\pi\_{\theta}(\cdot\vert s')}\left[\nabla\_\theta\log\pi\_\theta(a'\vert s')\right] &= 
\sum_{a'\in\mathcal{A}}  \pi\_\theta(a'\vert s') \nabla\_\theta\log\pi\_\theta(a'\vert s')\\; , \\\
&= \sum_{a'\in\mathcal{A}}  \nabla\_\theta\pi\_\theta(a'\vert s') \\; , \\\
&= \nabla\_\theta\sum_{a'\in\mathcal{A}}  \pi\_\theta(a'\vert s') \\; , \\\
&= \nabla\_\theta 1 = 0\\; .
\end{aligned}
$$
which concludes the proof.
{{% /toggle_block %}}
Denoting $G\_\tau^t :=  \sum\_{k=t}^T \lambda^{k-t}r(s\_k, a\_k)$ we obtain a new identity:
$$
\nabla\_\theta U(\theta) = \mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[\sum\_{t=1}^{T-1} \lambda^{t-1}G\_{\tau}^t\nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\right]\\; ,
$$
{{% toggle_block background-color="#FAD7A0" title="Proof" %}}
This is a direct consequence of $R\_\tau's$ expression: 
$$
\begin{aligned}
\nabla\_\theta U(\theta) &= \mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[R\_\tau\sum\_{t=1}^{T-1} \nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\right]\\; , \\\
&=  \mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[\sum\_{t=1}^{T-1} \nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\sum\_{k=1}^{T-1} \lambda^{k-1}r(s\_k, a\_k)\right]\\; ,\\\
&=  \mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[\sum\_{t=1}^{T-1} \nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\sum\_{k=t}^{T-1} \lambda^{k-1}r(s\_k, a\_k)\right]\\; .
\end{aligned}
$$
{{% /toggle_block %}}
This makes a lot of intuitive sense: actions should be reinforced only based on future rewards (what they actually impact),
not past ones.

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
Observe that $\mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[G\_{\tau}^t\middle\vert s\_t, a\_t\right] = q\_{\lambda}^{\pi\_\theta}(s\_t, a\_t)$. 
Therefore by the tower-rule, conditionning on $s\_t$ and $a\_t$, we proved that:
$$
\nabla\_\theta U(\theta) = \mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[\sum\_{t=1}^{T-1} \lambda^{t-1}q\_\lambda^{\pi\_\theta}(s\_t, a\_t)\nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\right]
$$
Anticipating a bit, this provides an alternate proof to the policy gradient's theorem detailed below.
{{% /toggle_block %}}

#### Baselines
Another typical way of reducing variance in Monte-Carlo estimation is to introduce baselines, _a.k.a_ control variates in 
Monte-Carlo literature. The main idea is to add disturbances which do not contribute _in expectation_ to the estimation problem.
When this disturbance is correlated with the random variable which mean we are trying to approximate, this comes with a 
reduced variance of the Monte-Carlo estimator (see \[[Owen](https://artowen.su.domains/mc/), 8.9\]) for a detailed treatment).

For our focus, the main point is first to show that for any deterministic function $b:\mathcal{S}\mapsto \mathbb{R}$,
the following identity holds:
$$
\nabla\_\theta U(\theta)  = \mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[\sum\_{t=1}^{T-1} \lambda^{t-1}\left(G\_{\tau}^t-b(s\_t)\right)\nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\right]\\; .
$$

(The above result is still true when replacing $b(s\_t)$ by $b\_t$, where $b\_t$ is a random variable with appropriate measurability -- it must not 
depend on $a\_t$.)

{{% toggle_block background-color="#FAD7A0" title="Proof" %}}
It is enough to show that for any $t\in[T]$ we have:
$$
\mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[b(s\_t)\nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\right] = 0 \\; .
$$
We only have to condition on $s\_t$:
$$
\begin{aligned}
\mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[b(s\_t)\nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\right] &\overset{(i)}{=} 
\sum\_{s\in\mathcal{S}} b(s)\mathbb{P}\_{s^0}^\pi(s\_t = s)\mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[\nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\middle\vert s\_t=s\right] \\; , \\\
&= \sum\_{s\in\mathcal{S}} b(s)\mathbb{P}\_{s^0}^\pi(s\_t = s)\mathbb{E}\_{a\sim \pi\_\theta(\cdot\vert s)}\left[\nabla\_\theta\log\pi_\theta(a \vert s )\right] \\; , \\\
&= \sum\_{s\in\mathcal{S}} b(s)\mathbb{P}\_{s^0}^\pi(s\_t = s)\sum_{a\in\mathcal{A}} \pi_\theta(a \vert s )\nabla\_\theta\log\pi_\theta(a \vert s ) \\; , \\\
&= \sum\_{s\in\mathcal{S}} b(s)\mathbb{P}\_{s^0}^\pi(s\_t = s)\sum_{a\in\mathcal{A}} \nabla\_\theta\pi_\theta(a \vert s ) \\; , \\\
&= \sum\_{s\in\mathcal{S}} b(s)\mathbb{P}\_{s^0}^\pi(s\_t = s) \nabla\_\theta \sum_{a\in\mathcal{A}}\pi_\theta(a \vert s ) \\; , \\\
&\overset{(ii)}{=} \sum\_{s\in\mathcal{S}} b(s)\mathbb{P}\_{s^0}^\pi(s\_t = s) \nabla\_\theta 1 = 0 \\; .
\end{aligned}
$$
where $(i)$ is the law of total expectation and $(ii)$ uses the fact that $\pi\_\theta(\cdot\vert s)$ is a probability measure.
{{% /toggle_block %}}

It is now reasonable to start wondering about which mapping $b$ could be useful to reduce variance.
Because we want to reinforce (_i.e_ make more likely) rewarding action, it is natural to substract a _baseline_
that represent our current expectation for one action's return. An action associated with a higher return will be reinforced, 
and conversely.
As a result, taking $b(s) = v\_\lambda^{\pi\_\theta}(s)$ is a rather natural choice. Of course, the 
value function is itself unknown, but can also be approximated by samples (see below). The proposed estimator
therefore draws from the identity:
$$
\nabla\_\theta U(\theta) = \mathbb{E}\_{\tau\sim \mathcal{T}\_\theta}\left[\sum\_{t=1}^{T-1} \lambda^{t-1}\left(G\_{\tau}^t-v\_\lambda^{\pi\_\theta}(s\_t)\right)\nabla\_\theta\log\pi_\theta(a\_t \vert s\_t )\right]\\; .
$$


### Implementation example
To make things concrete we can have a look at some pseudo-code using the vanilla policy gradient described above -- an algorithm often
called [Reinforce](). The baseline is chosen to be the state value-function, evaluated in a Monte-Carlo style.
The pseudo-code illustrates a one-step gradient ascent using the VPG estimator, after N roll-outs using $\pi\_\theta$. 
The baseline $b(\cdot)$ is parametrized by $\omega$ living in some Euclidian space. 

{{< pseudocode title="REINFORCE" >}}

fit the baseline:
$$\hat{\omega} \leftarrow \argmin \sum_{n=1}^N \sum_{t=1}^T (b_\omega(s_t^n) - G_n^t)^2$$
compute the stochastic gradient:
$$
\hat{g}_n \leftarrow \frac{1}{N}\sum_{n=1}^N \sum_{t=1}^T \lambda^{t-1}\left(G_n^t - b_\omega(s_t^n)\right)\nabla_\theta \log\pi_\theta(a_t^n\vert s_t^n)
$$
update parameter:
$$
\theta \leftarrow \theta + \eta \cdot \hat{g}_n
$$
{{< /pseudocode >}}


<br>
<br>

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
Observe how in REINFORCE both the value function and the return are estimated from
pure Monte Carlo. 
It would also be reasonable to consider leveraging their Bellman structure -- that is, using bootstraping to speed-up the learning
-- something we would discuss soon.
{{% /toggle_block %}}


## Policy Gradient Theorem
We now detail the Policy Gradient theorem \[[Sutton et al](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)\].
It is a somewhat equivalent result to VPG, however it holds
even in continuing environments. It also holds for the average-reward criterion, under minor adaptations.
(Recall that VPG relied on a Monte Carlo strategy, which made sense only
in episodic settings.) In what follows, let 
$$d\_\lambda^{\pi\_\theta}(s) = (1-\lambda)\sum\_{t=1}^\infty \lambda^{t-1} \mathbb{P}\_{s^0}^\pi
\left(s\_t = s\right)$$
be the _discounted_ state-occupancy measure (it is a valid probability measure over $\mathcal{S}$). The Policy Gradient theorem states that:
$$
\nabla\_\theta U(\theta) = (1-\lambda)^{-1}\sum\_{s\in\mathcal{S}} d\_\lambda^{\pi\_\theta}(s) \sum\_{a\in\mathcal{A}} q\_\lambda^{\pi\_\theta}(s, a) \nabla\_\theta \pi_\theta(a\vert s) \\; .
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" %}}
We will show that for any $T\in\mathbb{N}$ we have:
$$
\nabla\_\theta U(\theta) = \sum\_{s\in\mathcal{S}} 
\left(\sum_{t=1}^T \lambda^{t-1}\mathbb{P}\_{s^0}^{\pi\_\theta}(s\_t=s)\right) q\_\lambda^{\pi\_\theta}(s, a) \nabla\_\theta \pi\_\theta(s, a) + \lambda^T 
\sum\_{s\in\mathcal{S}} \mathbb{P}\_{s^0}^{\pi\_\theta} (s\_{T+1}=s) \nabla\_\theta v\_\lambda^{\pi\_\theta}(s)
\\; .
$$
The claimed result is then trivially justified by letting $T\to\infty$. 
The initialisation at $T=0$ is holds simply by definition of $U(\theta)$.
Assume the results holds at some $T\in\mathbb{N}$. Observe that since
$v\_\lambda^{\pi\_\theta}(s) = \sum\_{a\in\mathcal{A}} \pi\_\theta(a\vert s) q\_\lambda^{\pi\_\theta}(s, a)$ we have:
$$
\begin{aligned}
\nabla\_\theta v\_\lambda^{\pi\_\theta}(s) &= \sum\_{a\in\mathcal{A}} \pi\_\theta(a\vert s) \nabla\_\theta q\_\lambda^{\pi\_\theta}(s, a)+
\sum\_{a\in\mathcal{A}}  q\_\lambda^{\pi\_\theta}(s, a)\nabla\_\theta \pi\_\theta(a\vert s)\\; , \\\
&= \sum\_{a\in\mathcal{A}} \pi\_\theta(a\vert s) \nabla\_\theta \left(r(s, a) + \lambda \sum\_{s'\in\mathcal{S}} \mathcal{P}\_s^a(s')v\_\lambda^{\pi\_\theta}(s')\right)+
\sum\_{a\in\mathcal{A}}  q\_\lambda^{\pi\_\theta}(s, a)\nabla\_\theta \pi\_\theta(a\vert s)\\; , \\\
&= \lambda\sum\_{a\in\mathcal{A}} \pi\_\theta(a\vert s) \left( \sum\_{s'\in\mathcal{S}} \mathcal{P}\_s^a(s')\nabla\_\theta  v\_\lambda^{\pi\_\theta}(s')\right)+
\sum\_{a\in\mathcal{A}}  q\_\lambda^{\pi\_\theta}(s, a)\nabla\_\theta \pi\_\theta(a\vert s)\\; , \\\
&= \lambda\sum\_{s'\in\mathcal{S}}\left(\sum\_{a\in\mathcal{A}} \mathcal{P}\_s^a(s')\pi\_\theta(a\vert s) \right) \nabla\_\theta v\_\lambda^{\pi\_\theta}(s')+
\sum\_{a\in\mathcal{A}}  q\_\lambda^{\pi\_\theta}(s, a)\nabla\_\theta \pi\_\theta(a\vert s)\\; , \\\
&= \lambda\sum\_{s'\in\mathcal{S}}\left(\sum\_{a\in\mathcal{A}} \mathbb{P}(s\_{T+2}=s', a\_{T+1} = a \vert s\_{T+1} = s) \right)\nabla\_\theta v\_\lambda^{\pi\_\theta}(s')+
\sum\_{a\in\mathcal{A}}  q\_\lambda^{\pi\_\theta}(s, a)\nabla\_\theta \pi\_\theta(a\vert s)\\; , \\\
&= \lambda\sum\_{s'\in\mathcal{S}} \mathbb{P}(s\_{T+2}=s'\vert s\_{T+1} = s)\nabla\_\theta v\_\lambda^{\pi\_\theta}(s')+
\sum\_{a\in\mathcal{A}}  q\_\lambda^{\pi\_\theta}(s, a)\nabla\_\theta \pi\_\theta(a\vert s)\\; .
\end{aligned}
$$
Replacing in the identity given by the induction hypothesis at $T$ concludes the induction, and the proof.
{{% /toggle_block %}}

The main point is that $\nabla\_\theta U(\theta)$ does not depend on $\nabla\_\theta q\_\lambda^{\pi\_\theta}$ nor 
$\nabla\_\theta d\_\lambda^{\pi\_\theta}$
-- which would be a serious pain to compute. It allows to write that:
$$
\nabla\_\theta U(\theta) \propto  \mathbb{E}\_{s\sim d\_\lambda^{\pi\_\theta}}
\mathbb{E}\_{a\sim\pi\_\theta(s)}\left[q\_\lambda^{\pi\_\theta}(s, a)\nabla\_\theta\log\pi\_\theta(a\vert s)\right] \\; ,
$$
justifying the computation of associated stochastic gradients. The PG theorem's paper of \[[Sutton et al](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)\]
goes even further, and justifies using function approximation for $q\_\lambda^{\pi\_\theta}$ _without_
biasing the gradient (we won't dive into the details: if you are interested in learning more, the keyword
you are looking for is _compatible_ function approximation.) In practice, modern (read: Deep Learning) implementations
do not really care about bias; neural networks are used to approximate the state-action value function. It can be learned
from full traces (Monte-Carlo style), by using Bellman back-ups, or a clever mix between the two -- the Generalized Advantage Estimation
paper by \[[Schulman & al](https://arxiv.org/pdf/1506.02438.pdf)\] gives a pretty neat summary of the different techniques.


{{% toggle_block background-color="#CBE4FE" title="Note" %}}
The VPG can be re-derived from the PG theorem, by replacing $q\_\lambda^{\pi\_\theta}$ by its expression and using the tower-rule.
{{% /toggle_block %}}


## Deterministic Policy Gradients
We have been so far concerned with optimizing _stochastic_ policies (notice how, due to the $log(\cdot)$ operator, 
the stochastic policy gradient might be undefined for deterministic policies). 
This is not a real issue, as we can expect that the optimized policy might converge to a stochastic one after sufficiently
many gradient updates. Unfortunately, as the policy looses entropy, the associated stochastic policy gradient's variance grows.

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
To justify the above, think about optimizing the objective:
$$
J(\mu) := \mathbb{E}\_{x\sim\mathcal{N}(\mu, \sigma^2)}[f(x)]\\; ,
$$
where $f:\mathbb{R}\mapsto\mathbb{R}$. By the likelihood ratio trick, one has:
$$
\nabla\_\mu J(\mu) := \mathbb{E}\_{x\sim\mathcal{N}(\mu, \sigma^2)}[f(x)\nabla\_\mu\mathcal{N}(x\vert \mu, \sigma^2) ] \\; .
$$
A short computation shows that the variance of a sample-average estimator of the gradient is $\propto 1/\sigma^2$.
{{% /toggle_block %}}

So, what happens if we were to directly optimize for _deterministic_ policies. This is where the Deterministic Policy Gradient (DPG)
of \[[Silver et al.](http://proceedings.mlr.press/v32/silver14.pdf)\] comes into play.
It is mostly useful for _continuous_ action set, _i.e_ $\mathcal{A} = \mathbb{R}^A$.
In what follows, let $\pi\_\theta: \mathcal{S}\mapsto \mathbb{R}^A$ be a _deterministic_ policy. Under
adequate smoothness requirements (of the reward function and the transition probability kernel) 
the DPG theorem claims that:
$$
\nabla\_\theta U(\theta) \propto \mathbb{E}\_{s\sim d\_\lambda^{\pi\_\theta}}
\left[\nabla\_\theta \pi\_\theta(s) \cdot 
\nabla\_a \left.q\_\lambda^{\pi\_\theta}(s, a)\right|_{a=\pi\_\theta(s)}\right] \\; ,
$$
where $\nabla\_\theta \pi\_\theta(s)$ is the Jacobian of the application $\pi\_\theta(s)$ mapping $\theta$ to some
action in $\mathbb{R}^A$ (hence if $\theta$ lives in some $\mathbb{R}^d$, 
$\nabla\_\theta \pi\_\theta(s)\in \mathbb{R}^{d\times A}$). 

{{% toggle_block background-color="#FAD7A0" title="Proof" %}}
The idea is extremely similar to the stochastic policy gradient's theorem; let's just see what the 
first step of the induction looks like. We have that:
$$
\begin{aligned}
\nabla\_\theta U(\theta) &= \nabla\_\theta v\_\lambda^{\pi\_\theta}(s^0) \\; , \\\
&\overset{(i)}{=}  \nabla\_\theta  q\_\lambda^{\pi\_\theta}(s^0, \pi\_\theta(s^0)) \\; , \\\
&=  \nabla\_\theta \left( r(s^0, \pi\_\theta(s^0)) + \lambda \sum\_{s\in\mathcal{S}} \mathcal{P}\_{s^0}^{\pi\_\theta(s^0)}(s)v\_\lambda^{\pi\_\theta}(s)\right) \\; , \\\
&\overset{(ii)}{=} \nabla\_\theta\pi\_\theta(s^0)\cdot \left.\nabla\_a r(s, a)\right\vert_{a=\pi\_\theta(s^0)}+
\lambda \sum\_{s\in\mathcal{S}}\left(\nabla\_\theta\pi\_\theta(s^0)\cdot \nabla_a \left.\mathcal{P}\_{s^0}^{a}(s)\right|\_{a=\pi\_\theta(s^0)}v\_\lambda^{\pi\_\theta}(s) + 
\mathcal{P}\_{s^0}^{\pi\_\theta(s^0)}(s) \nabla\_\theta v\_\lambda^{\pi\_\theta}(s)\right) \\; , \\\
&= \nabla\_\theta\pi\_\theta(s^0)\cdot \nabla_a \left.\left( r(s, a) + \lambda\sum\_{s\in\mathcal{S}}\mathcal{P}\_{s^0}^{a}(s)v\_\lambda^{\pi\_\theta}(s) \right)\right|\_{a=\pi\_\theta(s^0)} + 
\lambda\sum\_{s\in\mathcal{S}}\mathcal{P}\_{s^0}^{\pi\_\theta(s^0)}(s) \nabla\_\theta v\_\lambda^{\pi\_\theta}(s)\\; , \\\
&= \nabla\_\theta\pi\_\theta(s^0)\cdot \nabla_a \left. q\_\lambda^{\pi\_\theta}(s, a)\right\vert\_{a=\pi\_\theta(s^0)} + 
\lambda\sum\_{s\in\mathcal{S}}\mathcal{P}\_{s^0}^{\pi\_\theta(s^0)}(s) \nabla\_\theta v\_\lambda^{\pi\_\theta}(s)\\; ,
\end{aligned}
$$
where we used in $(i)$ the fact that $\pi\_\theta$ is deterministic, and in $(ii)$ 
the chain rule. Unrolling (or using an induction like we did for the PGT) yields the announced result.
{{% /toggle_block %}}

In turns out that the DPG is actually the limiting version of the PGT when the entropy of the stochastic
policies tends to $0$ \[[Silver et al., Theorem 2](http://proceedings.mlr.press/v32/silver14.pdf)\]. As a result, 
the different ideas regarding compatible function approximation remain valid. 

### DPG as approximate Policy Improvement
DPG also has a nice intuitive ties to the generic Policy Iteration (PI) algorithm. Recall the 
policy improvement step of PI at round $t$:
$$
    \pi\_{t+1}(s) \in \argmax\_{a\in\mathcal{A}} q\_\lambda^{\pi\_t} (s, a) \\; .
$$
This requires solving a global maximisation step, quite burdensome
whenever $\mathcal{A}$ is not finite. Instead, we can hope to retain some of the nice
properties of PI by following the gradient of $q\_\lambda$ (w.r.t the action). More precisely, once the policy parametrized, 
we can look for a direction $\delta\theta$ such that:
$$
q\_\lambda^{\pi\_{\_{\theta\_t}}} (s, \pi\_{\theta\_t+\delta\theta}(s)) > q\_\lambda^{\pi\_{\_{\theta_t}}} (s, \pi\_{\theta\_t}(s))
$$
For a small enough $\alpha$, this improvement is guaranteed by the choice:
$$
\begin{aligned}
\delta\theta &= \alpha \nabla\_\theta\left.\left(q\_\lambda^{\pi\_{\_{\theta_t}}} (s, \pi\_{\theta}(s))\right)\right\vert\_{\theta=\theta_t}\\;, \\\
&= \alpha\\, \nabla\_\theta\left.\pi\_{\theta}(s)\right|\_{\theta=\theta\_t}\cdot \nabla\_a \left.q\_\lambda^{\pi\_{\_{\theta\_t}}}(s, a)\right\vert_{a=\pi\_{\theta_t}(s)}\\; .
\end{aligned}
$$
Averaging over the states, we retrieve the DPG identity. 