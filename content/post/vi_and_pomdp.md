+++
author = "Louis Faury"
title = "Variational Inference in POMDPs"
date = "2025-06-20"
+++

The goal of this post is to explore from first principles the learning of belief models in partially observable MDPs. 
We will start with a quick refresher on variational inference 
and then apply it to state estimation in POMDPs. 
Specifically, we will re-derive the update rule used to train Dreamer-like world models.
<!--more-->


## Variational Inference

In this post, we are interested in latent variable models.
We consider couples of random variables $(x, z)$ where $z$ denotes the latent variable
and $x$ the observed one.
Inference refers to the estimation of $p(z\vert x)$.
Often, it does not exist in closed form, and the best one can do is try to approximate it.
Given some family of distribution $\mathcal{P}$, _variational_ inference is about finding the 
best approximation of $p(z\vert x)$ within $\mathcal{P}$:
$$
q^\star \in \argmin_{q\in\mathcal{P}} \text{KL}(q(z\vert x) \\, \\| \\,p(z\vert x)) \\;.
$$


### Maximum-likelihood

{{< infoblock>}}
$\quad$ Our
<a href="../em" style="text-decoration:none; color:#0074aa;" ">post</a>
about the Expectation-Maximisation algorithm can provide a good warm-up here.
{{< /infoblock >}}

We will be interested in variational inference for 
maximum likelihood estimation in latent variable models.
Assuming the relevant distributions are parametrised by some $\theta$,
we want to maximise the observation likelihood $ p\_\theta(x)$.
The marginal $p\_\theta(x)$ could be obtained via integration by the law of total probabilities:
$$
\begin{aligned}
\log p\_\theta(x) &= \log \int_{z} p\_\theta(x\vert z)p\_\theta(z)dz \\;.
\end{aligned}
$$ 
This integral rarely exists in closed form.
We saw in our EM [post](../em) 
how this objective could be optimised nonetheless, given that the posterior $p\_\theta(z\vert x)$
is known.
Sadly, this is again a questionable assumption  – for instance, 
it fails whenever $p\_\theta(x\vert z)$ is represented via a neural-network.
The way forward involves variational inference: introducing a variational
distribution $q\_\phi(z\vert x)$. Indeed, we have:

$$
\tag{1}
\log p\_\theta(x) = \text{KL}(q\_\phi \\, \\| \\, p\_\theta )
    + \underbrace{\mathbb{E}\_{q\_\phi}\left[\log\frac{p\_\theta(x\vert z)p\_\theta(z)}{q\_\phi(z\vert x)}\right]}_{\mathcal{V}(\theta, \phi)} \\;.
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof involves introducing the variational distribution
$q\_\phi(z\vert x)$ arbitrarily at first, and using Bayes rule.
$$
\begin{aligned}
    \log p\_\theta(x) &= \int_z q\_\phi(z\vert x)\log p\_\theta(x) dz  \\;, &  (\log \int_z q\_\phi(z\vert x)dz=1)\\\
        &= \int_z q\_\phi(z\vert x)\log \frac{p\_\theta(x\vert z)p\_\theta(z)}{p\_\theta(z\vert x)} dz\\;, & (\text{Bayes rule}) \\\
        &=\int_z q\_\phi(z\vert x)\log \frac{q\_\phi(z\vert x)p\_\theta(x\vert z)p\_\theta(z)}{q\_\phi(z\vert x)p\_\theta(z\vert x)} dz \\;, \\\
        &=  \text{KL}(q\_\phi \\, \\| \\, p\_\theta ) + \int_z q\_\phi(z\vert x)\log \frac{p\_\theta(x\vert z)p\_\theta(z)}{q\_\phi(z\vert x)} dz \\;.
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

Because the Kullback-Leibler divergence is non-negative we have 
$
\log p\_\theta(x) \geq \mathcal{V}(\theta, \phi)\\;.
$
For this reason, $\mathcal{V}$ is often called the variational lower-bound, or
the evidence lower-bound (some communities call $\log p\_\theta(x)$ the evidence).
It does not require computing a potentially intractable integral, 
and we can derive efficient unbiased estimators for it (see below). 
That's it, really: we
can now focus on this objective, relying on the fact that it lower-bounds the
maximum-likelihood (and hoping it will not be too far off).



{{% toggle_block background-color="#CBE4FE" title="The EM revisited" %}}
There exists a pleasant way to link (1) with the EM algorithm.
Given some current guess $\theta\_t$, we could try to get a tight bound
on (2) by minimising
$\text{KL}(q\_{\phi} \\, \\| \\, p\_{\theta\_t} )$ – and simply set 
$q\_{\phi\_t}(z\vert x) = p\_{\theta\_t}(z\vert x)$.
That's the E-part of the EM. The M-part is about maximising the right hand side $\mathcal{V}(\theta, \phi\_t)$.

This naturally requires $p\_{\theta\_t}(z\vert x)$ to be known in closed-form – which is far from 
trivial. We did see some positive examples in our EM post 
(_e.g._, the Gaussian mixture or hidden Markov models).
{{% /toggle_block %}}

### Estimators
Let us rewrite the variational lower-bound as follows: 
$$
\mathcal{V}(\theta, \phi) = \mathbb{E}\_{q\_\phi}\left[\\, \log p\_\theta(x\vert z)\right] 
    - \text{KL}(q\_\phi(z\vert x) \\,\\|\\, p\_{\theta}(z))
$$

The KL often exists in closed-form – e.g. when $q\_\phi$ and the marginal $p\_{\theta}(z)$ are Gaussians.
We make this assumption from now on. Further, we also remove the dependency $\theta$ from the latter, 
and simply consider this term as a regulariser for $\phi$ – for instance, setting $p\_{\theta}(z) = \mathcal{N}(z\vert 0, 1)$.

The first term can be more annoying. Gradients w.r.t. $\theta$ are uneventful and can easily be approximated via sample average.
Gradients w.r.t $\phi$ are slightly more annoying, as getting unbiased estimators
will typically require the use of the log-derivative trick.
Even though unbiased, they come with high variance.
Instead, it is common to use the _reparametrisation_ trick.
If our choice for $q\_\phi$ allows (_e.g._ a location-scale distribution), 
we will write samples from it as a direct transformation of some random 
variable $\varepsilon$, drawn from a "simple" distribution.
The usual examples come from Gaussian reparametrisation; 
if $q\_\phi(z\vert x) = \mathcal{N}(z\vert \mu\_\phi(x), \sigma^2)$
then one can write
$z = \mu\_\phi(x) + \sigma\varepsilon$ with $\varepsilon \sim\mathcal{N}(0, 1)$.
In general, one writes $z=g\_\phi(x, \varepsilon)$. This allows to write:
$$
\nabla\_{\theta, \phi} \\, \mathcal{V}(\theta, \phi) = \nabla\_{\theta, \phi}\mathbb{E}\_{\varepsilon}[\log p\_\theta(x \vert g\_\phi(x, \varepsilon))]- 
\nabla\_{\theta, \phi} \text{KL}(q\_\phi(z\vert x) \\,\\|\\, p(z))\\; ,
$$

for which sample-average estimators are straight-forward.


{{% toggle_block background-color="#CBE4FE" title="Variational auto-encoders" %}}
There is a direct link with auto-encoders.
The model $q\_\phi(z\vert x) $ relates to an encoder, while
$p\_\theta(x\vert z)$ is the decoder.
The first part of the loss $\mathcal{V}(\theta, \phi)$ is a reconstruction loss in auto-encoder lingo;
the second is a regulariser for the latent distribution.
{{% /toggle_block %}}


## Learning in POMDPs

{{< infoblock>}}
$\quad$ Take a look at the POMDP
<a href="../pomdp" style="text-decoration:none; color:#0074aa;" ">post</a>
for a refresher.
{{< /infoblock >}}


### Setting

We now turn to using variational inference for belief
estimation in a POMDP denoted $\mathcal{M}=(\mathcal{S}, \mathcal{A}, \mathcal{O}, p, q, r)$.
One approach to solve $\mathcal{M}$ is to express the equivalent
belief MDP, where the state is replaced by the belief 
$b\_t = \mathbb{P}(s\_t \vert o\_{1:t}, a\_{1:t-1})$.
Estimating the belief requires both the transition $p$ and
emission kernel $q$ – which are unknown in a RL setting.
Model-based RL approaches attempt to learn predictive models for the belief, 
so that it can be used for explicit planning. 
We will focus here only on the former.


{{< warningblock>}}
$\quad$ Be prepared for overloaded notations when it comes to distributions.
To limit confusion, learned distributions will be subscript by a parameter
(e.g. $p_\theta$) while ground truth are not (e.g. $p$).
{{< /warningblock >}}


### Learning belief models


We wish to learn a probabilistic model $q\_\phi$ for the belief from a trajectory $\\{o\_{1:t}, a\_{1:t-1}\\}$.
To be useful at planning time, we require said model to be causal. Formally, we will use a filtering posterior:
$$
\tag{3}
    q\_\phi(s\_t\vert o\_{1:t}, a\_{1:t-1}) := \prod\_{k=2}^K q\_\phi(s\_k \vert o\_{1:k}, a\_{1:k-1})q\_\phi(s\_1)\\;.
$$

It is unclear how to learn $q\_\phi$ given that states $\\{s\_{i}\\}\_{i=1}^t$ are not observed.
Similarly to the previous section, it will be introduced as a variational distribution
via a maximum-likelihood scheme. 
Indeed, we set out to maximise the likelihood of observing 
$\\{o\_i\\}\_{i=1}^t$, conditioned on $\\{a\_i\\}_{i=1}^{t-1}$, under some model $p\_\theta$.
Following the protocol detailed in the last section, 
one can establish the following variational bound:

$$
\tag{4}
\log p\_\theta(o\_{1:t}\vert a\_{1:t-1}) \geq 
\sum\_{k=1}^t \mathbb{E}\_{q\_\phi}\Big[\log p\_\theta(o\_k \vert s\_k) - \text{KL}(p\_\theta(s\_k\vert s\_{k-1}, a\_{k-1}) \\, \\| \\, q\_\phi(s\_k\vert o\_{1:k}, a\_{1:k-1}))\Big]\\;.
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof follows the same steps as for proving (1). 
Let us first introduce the variational distribution.
$$
\begin{aligned}
\log p\_\theta(o\_{1:t}\vert a\_{1:t-1}) &= \int\_{s\_{1:t}} q\_\phi(s\_{1:t}\vert o\_{1:t}, a\_{1:t-1}) \log p\_\theta(o\_{1:t}\vert a\_{1:t-1})ds\_{1:t} \\;,\\\
&= \int\_{s\_{1:t}} q\_\phi(s\_{1:t}\vert o\_{1:t}, a\_{1:t-1}) \log \frac{p\_\theta(o\_{1:t}\vert a\_{1:t-1})q\_\phi(s\_{1:t}\vert o\_{1:t}, a\_{1:t-1})}{q\_\phi(s\_{1:t}\vert o\_{1:t}, a\_{1:t-1})}ds\_{1:t} \\;.
\end{aligned}
$$
Observe that $p(o\_{1:t}\vert a\_{1:t-1}) = p(o\_{1:t}\vert s\_{1:t}, a\_{1:t-1})p(s\_{1:t}\vert a\_{1:t-1})/p(s\_{1:t}\vert a\_{1:t-1}, o\_{1:t})$ by Bayes' rule. Therefore:
$$
\begin{aligned}
\log p\_\theta(o\_{1:t}\vert a\_{1:t-1}) &= \int\_{s\_{1:t}} q\_\phi(s\_{1:t}\vert o\_{1:t}, a\_{1:t-1}) \log \frac{ p(o\_{1:t}\vert s\_{1:t}, a\_{1:t-1})p(s\_{1:t}\vert a\_{1:t-1}) q\_\phi(s\_{1:t}\vert o\_{1:t}, a\_{1:t-1})}{q\_\phi(s\_{1:t}\vert o\_{1:t}, a\_{1:t-1})p(s\_{1:t}\vert a\_{1:t-1}, o\_{1:t})}ds\_{1:t} \\;,\\\
&= \text{KL}(p\_\theta \\| q\_\phi) + \int\_{s\_{1:t}} q\_\phi(s\_{1:t}\vert o\_{1:t}, a\_{1:t-1}) \log \frac{ p(o\_{1:t}\vert s\_{1:t}, a\_{1:t-1})p(s\_{1:t}\vert a\_{1:t-1})}{q\_\phi(s\_{1:t}\vert o\_{1:t}, a\_{1:t-1})}ds\_{1:t} \\;,\\\
&\geq \mathbb{E}\_{q\_\phi} \left[\log \frac{ p(o\_{1:t}\vert s\_{1:t}, a\_{1:t-1})p(s\_{1:t}\vert a\_{1:t-1})}{q\_\phi(s\_{1:t}\vert o\_{1:t}, a\_{1:t-1})}\right]\\;,
\end{aligned}
$$
since the KL is non-negative. Using the facts that $p\_\theta(s\_{1:t}\vert a\_{1:t-1}) = \prod\_{k=1}^t p\_\theta(s\_k\vert s\_{k-1}, a\_{k-1}$) (starting the
count at $k=1$ just for convenience), that 
$p\_\theta(o\_{1:t}\vert s\_{1:t}, a\_{1:t-1}) = \prod\_{k=1}^t p\_\theta(o\_k\vert s\_k)$ and the filtering posterior (1), we get:
$$
\log p\_\theta(o\_{1:t}\vert a\_{1:t-1}) \geq \sum_\{k=1}^t \log \mathbb{E}\_{q\_\phi}\left[\frac{p\_\theta(o\_k\vert s\_k) p\_\theta(s\_k\vert s\_{k-1}, a\_{k-1})}{q\_\phi(s\_k\vert o\_{1:k}, a\_{1:k-1})} \right]\\;. 
$$


<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}


Let us dissect (4). 
Observe that we maintain two models: one for the transition, one for the emission kernel.
The first term of the r.h.s maximises the likelihood of observation 
$p\_\theta(o\_k\vert s\_k)$.
The second term promotes the consistency of the belief model across transitions measured by the transition model $p\_{\theta}(s\_k\vert s\_{k-1}, a\_{k-1})$.


The variational distribution (the belief model) can be seen as an encoder for the trajectory $\\{o\_{1:t}, a\_{1:t-1}\\}$.
We use it to generate a sequence of beliefs
$\\{s\_{1:t}\\}$ sampled according to (3).
This yields an estimator for (4); by the reparametrisation trick, we also obtain efficient estimator for its gradients.

{{% toggle_block background-color="#CBE4FE" title="Multi-steps consistency" %}}
The derivation of (4) can be modified to ensure consistency after more than one step.
This is useful to promote models which have inherently small compounding errors.
{{% /toggle_block %}}

The belief model is often built using recurrent model, so that
$
    q\_\phi(s\_t\vert o\_{1:t}, a\_{1:t-1}) := \prod\_{k=2}^K q\_\phi(s\_k \vert s\_{k-1}, o\_{k}, a\_{k-1})\\;.
$
Altogether, this is what powers modern model-based approaches like the Dreamer family of models
and algorithms. The lower bound (4) appears already in {{< ref link="planet">}} [2]{{< /ref>}}.
Follow-up papers mostly refine the belief model itself (not the learning algorithm) – by _e.g._,
discretising the belief space, enabling deterministic belief updates, etc.



## References

The variational inference part of this blog post and is condensed version of:

[1] Auto-Encoding Variational Bayes. Kingma and Welling, 2013.

Its application to POMDP is taken from:

<div id="planet"></div>
[2] Learning Latent Dynamics for Planning from Pixels. Hafner et al, 2019.