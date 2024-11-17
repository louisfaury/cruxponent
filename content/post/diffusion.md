+++
author = "Louis Faury"
title = "Diffusion Models from the Ground Up"
date = "2024-11-15"
+++

We all have heard about diffusion models and played with their generative magic.
To me, it felt awkward not properly understanding _how_ they work, from first principles.
After all, they are just another latent variable model, right?
This post dives into the probabilistic foundations that make them tick, 
breaking down the tools needed to re-derive variational inference for diffusion models.

<!--more-->

The term "diffusion model" coins a family of latent variable models containing, _e.g._, 
diffusion probabilistic models {{< ref link="dpm">}} [1]{{< /ref>}}, 
noise-conditioned score functions {{< ref link="ncsf">}} [2]{{< /ref>}}
and denoising diffusion probabilistic model {{< ref link="ddpm">}} [3]{{< /ref>}}.
In this post we will focus on the latter--it builds on top of the first and has close ties with the second.


## Warm-up
There is nothing fundamental tying diffusion models to the normal distribution—however, in practice, people
often use the normal distribution as an elementary building block to a diffusion model (we will soon see how).
We will follow that path here, and first go through some well-known identities related to Gaussian density multiplication and convolution.
You surely have seen those results countless times before; 
we provide the proofs (which are undergraduate course all-time classics) for the sake of completeness.
For the rest of this section, let $p\(x) = \mathcal{N}(x\vert \mu\_1, \sigma\_1)$ and $q(x) = \mathcal{N}(x\vert \mu\_2, \sigma\_2^2)$
two univariate normal densities.


{{< boxed title="Gaussian identities" >}}
$\qquad\qquad\qquad\qquad\quad\text{Gaussian distributions are stable by multiplication and convolution.}\\$
$\text{}\\$
$\text{1. The product of two Gaussian densities is proportional to another Gaussian density:}\\$
$$
\tag{1}
p(x) q(x) \propto \mathcal{N}(x\vert \nu, \tau^2) \text{ where }\nu = (\mu_1\sigma_2^2 + \mu_2\sigma_1^2)/(\sigma_1^2 + \sigma_2^2) \text{ and }
\tau^2 = \sigma_1^2\sigma_2^2/(\sigma_1^2 + \sigma_2^2).
$$
$\text{}\\$
$\text{2. The convolution between the two Gaussian densities is still Gaussian:}\\$
$$
\tag{2}
\int_{t} p(t)q(x-t)dt = \mathcal{N}(x\vert \bar\mu, \bar\sigma^2) \text{ where }\bar\mu = \mu_1 + \mu_2 \text{ and } \bar\sigma^2 = \sigma_1 + \sigma_2.
$$
{{< /boxed >}}


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
The proof of 1. is a classical exercise of the so-called "complete the square" routine. 
$$
\begin{aligned}
p(x)q(x) &\propto \exp\left(-\frac{1}{2\sigma\_1^2}(x-\mu\_1)^2 - \frac{1}{2\sigma\_2^2}(x-\mu\_2)^2\right) \\;,\\\
&= \exp\left(-\frac{1}{2\sigma\_1^2}(x^2-2x\mu\_1) - \frac{1}{2\sigma\_2^2}(x^2-2x\mu\_2)^2\right) \exp\left(-\frac{\mu\_1^2}{2\sigma\_1^2}-\frac{\mu\_2^2}{2\sigma\_2^2}\right) \\;,\\\
&= \exp\left(-\frac{\sigma\_1^2 + \sigma\_2^2}{2\sigma\_1^2\sigma\_2^2}\left[x^2 - 2x\frac{\mu\_1\sigma\_2^2 + \mu\_2\sigma\_1^2}{\sigma\_1^2+\sigma\_2^2}\right]\right) \exp\left(-\frac{\mu\_1^2}{2\sigma\_1^2}-\frac{\mu\_2^2}{2\sigma\_2^2}\right) \\;,\\\
&= \exp\left(-\frac{\sigma\_1^2 + \sigma\_2^2}{2\sigma\_1^2\sigma\_2^2}\left[x - \frac{\mu\_1\sigma\_2^2 + \mu\_2\sigma\_1^2}{\sigma\_1^2+\sigma\_2^2}\right]^2\right) \exp\left(\frac{\sigma\_1^2 + \sigma\_2^2}{2\sigma\_1^2\sigma\_2^2}\frac{(\mu\_1\sigma\_2^2 + \mu\_2\sigma\_1^2)^2}{(\sigma\_1^2+\sigma\_2^2)^2}\right)\exp\left(-\frac{\mu\_1^2}{2\sigma\_1^2}-\frac{\mu\_2^2}{2\sigma\_2^2}\right) \\;, \\\
&= \underbrace{\exp\left(-\frac{\sigma\_1^2 + \sigma\_2^2}{2\sigma\_1^2\sigma\_2^2}\left[x - \frac{\mu\_1\sigma\_2^2 + \mu\_2\sigma\_1^2}{\sigma\_1^2+\sigma\_2^2}\right]^2\right)}_{\mathcal{N}(x\vert \nu, \tau^2)} \exp\left(\frac{(\mu\_1\sigma\_2^2 + \mu\_2\sigma\_1^2)^2}{2\sigma\_1^2\sigma\_2^2(\sigma\_1^2+\sigma\_2^2)}\right)\exp\left(-\frac{\mu\_1^2}{2\sigma\_1^2}-\frac{\mu\_2^2}{2\sigma\_2^2}\right)\\; .
\end{aligned}
$$
The proof of 2. is immediate when realising that the convolution can be mapped to summing two Gaussian random variables.
We will nonetheless carry on with the brute force approach adopted for the previous proof. 
First, observe that:

$$
\begin{aligned}
\int\_{t} p(t)q(t)dt &= \frac{1}{2\pi\sigma\_1\sigma\_2}\exp\left(\frac{(\mu\_1\sigma\_2^2 + \mu\_2\sigma\_1^2)^2}{2\sigma\_1^2\sigma\_2^2(\sigma\_1^2+\sigma\_2^2)}\right)\exp\left(-\frac{\mu\_1^2}{2\sigma\_1^2}-\frac{\mu\_2^2}{2\sigma\_2^2}\right) \int_{x} \mathcal{N}(t\vert \nu, \tau^2)dx \\;, \\\
&\overset{(i)}{=} \frac{1}{\sqrt{2\pi}\sqrt{\sigma\_1^2+ \sigma\_2^2}}\exp\left(\frac{(\mu\_1\sigma\_2^2 + \mu\_2\sigma\_1^2)^2}{2\sigma\_1^2\sigma\_2^2(\sigma\_1^2+\sigma\_2^2)}\right)\exp\left(-\frac{\mu\_1^2}{2\sigma\_1^2}-\frac{\mu\_2^2}{2\sigma\_2^2}\right)  \\;, \\\
&= \frac{1}{\sqrt{2\pi}\sqrt{\sigma\_1^2+ \sigma\_2^2}}\exp\left(\frac{1}{2(\sigma\_1^2 + \sigma\_2^2)}\left[\frac{(\mu\_1\sigma\_2^2 + \mu\_2\sigma\_1^2)^2}{\sigma\_1^2\sigma\_2^2}-\frac{\mu\_1^2}{\sigma\_1^2}(\sigma\_1^2+\sigma\_2^2) - \frac{\mu\_2^2}{\sigma\_2^2}(\sigma\_1^2+\sigma\_2^2)\right]\right)  \\;, \\\
&= \frac{1}{\sqrt{2\pi}\sqrt{\sigma\_1^2+ \sigma\_2^2}}\exp\left(\frac{1}{2(\sigma\_1^2 + \sigma\_2^2)}\left(\mu\_1-\mu\_2\right)^2\right)  \\;.
\end{aligned}
$$
In (1) we used the fact that $\int_{x} \mathcal{N}(x\vert \nu, \tau^2)dx = \sqrt{2\pi}\tau$.
To obtain the result of the convolution, observe that $q(x-t)=\mathcal{N}(t\vert x-\mu, \sigma\_2^2)$. 
Hence, we obtain that for the convolution:
$$
\begin{aligned}
\int\_{t} p(t)q(x-t)dt &= \frac{1}{\sqrt{2\pi}\sqrt{\sigma\_1^2+ \sigma\_2^2}}\exp\left(\frac{1}{2(\sigma\_1^2 + \sigma\_2^2)}\left(\mu\_1-(x-\mu\_2)\right)^2\right)  \\;, \\\
&= \frac{1}{\sqrt{2\pi}\sqrt{\sigma\_1^2+ \sigma\_2^2}}\exp\left(\frac{1}{2(\sigma\_1^2 + \sigma\_2^2)}\left(x-(\mu\_1+\mu\_2)\right)^2\right)  \\;, \\\
&= \mathcal{N}(x\vert \bar\mu, \bar\sigma^2)\\; .
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$ </div>
{{% /toggle_block %}}



## The Model
Let $p$ be some target distribution that we wish to model through a parametric form $p\_\theta$.
For simplicity, we will assume here that $p$ is supported on $\mathbb{R}^d$ for some $d\in\mathbb{N}$.
Let $x\_0 \in\mathbb{R}^d$ a sample from $p$. 
At heart, a diffusion probabilistic model is a latent variable model of the form:
$$
p\_{\theta}(x\_0) = \int\_{x\_{1:T}} p\_\theta(x\_{0:T})dx\_{1:T}\\;,
$$
for some $T\in\mathbb{N}$ , and where we used the short-hand $x\_{1:T} = (x\_1, \ldots, x\_T)$.
A salient feature of the diffusion probabilistic model is that the latent $x\_{1:T}$ live
in the same space $\mathbb{R}^d$ as $x_0$. Further, a diffusion model assumes some temporal dependencies
in the latent space via so-called forward and backward generative processes.

### Forward process
The _forward_ process $x\_0 \rightarrow x\_1 \rightarrow \ldots \rightarrow x\_T$ is defined as a *fixed*, *non-stationary*
Markov Chain which simply adds noise gradually. Formally, for $\\{\beta\_t\\}\_{t\geq 1}$ a sequence of scalars in $(0, 1]$:
$$
q(x\_{1:T}\vert x\_0) = \prod\_{t=1}^T q\_t(x\_t\vert x\_{t-1})\quad \text{ with } \quad
q\_t(x\_t\vert x\_{t-1}) = \mathcal{N}\left(x\_t \vert \sqrt{1-\beta\_t}x\_{t-1}, \beta\_t \mathbf{I}_d\right)\\; .
$$

Readers used to latent variable models can already anticipate that $q$ will
be the variational distribution used for learning $p\_\theta$. 
The approximated posterior $q$ is therefore fixed and has no learnable parameters.

{{< infoblock>}}
$\quad$ In all generality, the parameters $\{\beta_t\}_t$ are learnable, and could be optimised via re-parametrisation.
We ignore this below and do not consider them as degrees of freedom.
{{< /infoblock >}}

### Backward process
The backward process $x\_T \rightarrow x\_{T-1} \rightarrow x\_0$ is 
defined as a Markov Chain with _learnable_ Gaussian dynamics
with mean $\mu\_\theta(x\_t, t)$ and covariance matrix $ \Sigma\_\theta(x\_t, t)$. Concretely, with $p\_\theta(x\_T) = \mathcal{N}(x\_T\vert 0, \mathbf{I}_d)$:

$$
p\_\theta(x\_{0:T}) = p\_\theta(x\_T)\prod\_{t=1}^T p\_\theta(x\_{t-1}\vert x\_t)\quad \text{with} \quad
p\_{\theta}(x\_{t-1}\vert x\_t) = \mathcal{N}(x\_{t-1}\vert \mu\_\theta(x\_t, t), \Sigma\_\theta(x\_t, t))\\; .
$$
One can generate a new sample from $p_\theta$ (hopefully approximating $p$) by sampling $x_T$ from 
a centered noise, simply by walking down the backward process. 


{{< warningblock>}}
$\qquad \text{Both the forward and backward process are non-stationary, and we should write } p_{\theta}(x_{t-1}\vert x_t, t)
\text{ instead of}\\$
$ p_{\theta}(x_{t-1}\vert x_t)\text{ -- we do not in an effort to reduce clutter. However, observe that we keep this nuance for the prior} \\$
$\text{mean and variance where we keep a time dependency, e.g. } \mu_\theta(x_t, t).$
{{< /warningblock >}}

### Useful results

We collect here a few results that will prove useful down the road.
Let $\alpha\_t = \prod\_{s=1}^t (1-\beta\_s)$. The forward process can be summarised as follows, for $t\in\\{1, \ldots, T\\}$:
$$
\tag{3}
q(x\_t \vert x\_0) = \mathcal{N}(x\_t \vert \sqrt{\alpha\_t} x\_0, (1-\alpha\_t)\mathbf{I}\_d)\\; .
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
This can be proven by recurrence. The result is true by definition for $t=1$. 
Assume it holds a $t-1$.
$$
\begin{aligned}
q(x\_t \vert x\_0) &= \int\_{x\_{t-1}} q(x\_t \vert x\_{t-1})q(x\_{t-1} \vert x\_0) dx\_{t-1}\\;, \\\
&= \int\_{x\_{t-1}} \mathcal{N}(x\_t \vert \sqrt{1-\beta\_{t-1}} x\_{t-1}, \beta\_{t-1}\mathbf{I}\_d)q(x\_{t-1} \vert x\_0) dx\_{t-1}\\;, \\\
&= \int\_{x\_{t-1}} \mathcal{N}\left(x\_t \vert \sqrt{1-\beta\_{t-1}} x\_{t-1}, \beta\_{t-1}\mathbf{I}\_d\right)\mathcal{N}\left(x\_{t-1} \vert \sqrt{\alpha\_{t-1}} x\_0, (1-\alpha\_{t-1})\mathbf{I}\_d\right) dx\_{t-1}\\;, &(\text{by hyp.})\\\
&= \int\_{x\_{t-1}} \mathcal{N}\left(x\_t \vert x\_{t-1}, \beta\_{t-1}\mathbf{I}\_d\right)\mathcal{N}\left(x\_{t-1} \vert \sqrt{1-\beta\_{t-1}}\sqrt{\alpha\_{t-1}} x\_0, (1-\beta\_{t-1})(1-\alpha\_{t-1})\mathbf{I}\_d\right) dx\_{t-1}\\;. &(\text{change var.})\\\
\end{aligned}
$$
By applying (2) we get that:
$$
q(x\_t\vert x\_0) = \mathcal{N}(x\_t \vert  \sqrt{1-\beta\_{t-1}}\sqrt{\alpha\_{t-1}}, \left[\beta\_{t-1} + (1-\beta\_{t-1})(1-\alpha\_{t-1})\right]\mathbf{I}\_d )\\; .
$$
Since $\sqrt{1-\beta\_{t-1}}\sqrt{\alpha\_{t-1}} = \alpha\_t$, and given that:
$$
\begin{aligned}
(1-\beta\_{t-1})(1-\alpha\_{t-1}) &= \beta\_{t-1} + 1 - \beta\_{t-1} - (1-\beta\_{t-1})\alpha\_{t-1}\\;, \\\
&= 1-\alpha\_t \\; .
\end{aligned}
$$
this concludes proving the claimed result.<div style="text-align: right"> $\blacksquare$ </div>

{{% /toggle_block %}}

The conditional $q(x\_{t-1}\vert x\_t)$ is intractable, but $q(x\_{t-1}\vert x\_t, x\_0)$ has a closed-form! Indeed,
$q(x\_{t-1}\vert x\_t, x\_0) = \mathcal{N}\left(x\_{t-1}\vert \mu\_{t}, \sigma\_t^2\right)$
where:

$$
\tag{4}
\begin{aligned}
\mu\_t &= \frac{\sqrt{1-\beta\_t}(1-\alpha\_{t-1})}{1-\alpha\_t}x\_t + \frac{\sqrt{\alpha\_{t-1}}\beta\_t}{1-\alpha\_t}x\_0\\;, \\\
\sigma\_t^2 &= \frac{1-\alpha\_{t-1}}{1-\alpha\_{t}}\beta\_t\\; .
\end{aligned}
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
Observe that both $q(x\_t\vert x\_0)$, $q(x\_{t-1}\vert x\_0)$ and $q(x\_{t-1}\vert x\_t)$ are Gaussian -- hence so is $q(x\_{t-1}\vert x\_t, x\_0)$.
Therefore, below, we drop the normalisation constant. 
$$
\begin{aligned}
q(x\_{t-1}\vert x\_t, x\_0)  &\propto q(x\_t \vert x\_{t-1}, x\_0)q(x\_{t-1}\vert x\_0)\\; , &(\text{Bayes rule})\\\
&= \mathcal{N}(x\_t\vert \sqrt{1-\beta\_t}x\_{t-1}, \beta\_t\mathbf{I}\_d) \cdot \mathcal{N}(x\_{t-1}\vert \sqrt{\alpha\_{t-1}}x\_{0}, (1-\alpha\_{t-1})\mathbf{I}\_d) \\;, &(\text{def.})\\\
&\propto \exp(-\frac{1}{2\beta\_t}(x\_t-\sqrt{1-\beta\_t}x\_{t-1})^2)\exp(-\frac{1}{2(1-\alpha\_{t-1})}(x\_{t-1}-\sqrt{\alpha\_{t-1}}x\_0)^2)\\;, \\\
&\propto \exp(-\frac{1-\beta\_t}{2\beta\_t}(x\_{t-1}-x\_t/\sqrt{1-\beta\_t})^2)\exp(-\frac{1}{2(1-\alpha\_{t-1})}(x\_{t-1}-\sqrt{\alpha\_{t-1}}x\_0)^2)\\;, \\\
&= \mathcal{N}(x\_t\vert \mu\_t, \sigma\_t^2) \\;,
\end{aligned}
$$
where, by (1):
$$
\begin{aligned}
\mu\_t &= \frac{(1-\alpha\_{t-1})/\sqrt{1-\beta\_t}}{{\beta\_t/(1-\beta\_t) + 1 - \alpha\_{t-1}}}x\_t + \frac{\sqrt{\alpha\_{t-1}}\beta\_{t}/(1-\beta\_t)}{\beta\_t/(1-\beta\_t) + 1 - \alpha\_{t-1}}x\_0\\;,\\\
&= \frac{(1-\alpha\_{t-1})\sqrt{1-\beta\_t}}{\beta\_t + (1-\beta\_t)(1 - \alpha\_{t-1})}x\_t + \frac{\sqrt{\alpha\_{t-1}}\beta\_{t}}{\beta\_t + (1-\beta\_t)(1 - \alpha\_{t-1})}x\_0\\;,\\\
&= \frac{(1-\alpha\_{t-1})\sqrt{1-\beta\_t}}{1-\alpha\_t}x\_t + \frac{\sqrt{\alpha\_{t-1}}\beta\_{t}}{1-\alpha\_t}x\_0\\;,
\end{aligned}
$$
and
$$
\begin{aligned}
\sigma\_t &= \frac{\beta\_t/(1-\beta\_t)(1-\alpha\_{t-1})}{\beta\_t/(1-\beta\_t) + 1 - \alpha\_{t-1}}\\;,\\\
&= \beta\_t\frac{1-\alpha\_{t-1}}{\beta\_t + (1-\beta\_t)(1 - \alpha\_{t-1})}\\;,\\\
&=\beta\_t(1-\alpha\_{t-1})/(1-\alpha\_t) \\; .
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$ </div>
{{% /toggle_block %}}


## Learning
Ideally, the model would be learned via maximum-likelihood; given samples $\\{x\_0^1, \ldots, x\_0^n\\}$ we want to compute
$$
\theta\_n \in \argmax\_\theta \sum\_{i=1}^n \log p\_\theta(x\_0^i)\\;.
$$
Sadly, the log-likelihood $\log p\_\theta$ is intractable. Instead, we will maximise a variational
lower-bound.

### Variational bound
The variational lower-bound is classically obtained by introducing a variational distribution and using Jensen's inequality.
Let's detail each step along the way:
$$
\begin{aligned}
\log p\_\theta(x\_0) &= \log \int\_{x\_{1:T}} p\_\theta(x\_{0:T})d x\_{1:T} \\;, \\\
&= \log \int\_{x\_{1:T}} \frac{p\_\theta(x\_{0:T})}{q(x\_{1:T}\vert x\_0)}q(x\_{1:T}\vert x\_0)d x\_{1:T}\\;, \\\
&\geq  \int\_{x\_{1:T}} q(x\_{1:T}\vert x\_0)\log\frac{p\_\theta(x\_{0:T})}{q(x\_{1:T}\vert x\_0)}d x\_{1:T}\\;, &(\text{concavity of }\log)\\\
&= \int\_{x\_{1:T}} q(x\_{1:T}\vert x\_0) \log \frac{p\_\theta(x\_T)\prod\_{t=1}^T p\_\theta(x\_{t-1}\vert x\_t)}{\prod\_{t=1}^Tq(x\_t\vert x\_{t-1})}dx\_{1:T}\\;,\\\
&= \mathbb{E}\_{q(\cdot\vert x\_0)}\left[\log p\_\theta(x\_T) + \sum\_{t=1}^T \log\frac{p\_\theta(x\_{t-1}\vert x\_t)}{q(x\_t\vert x\_{t-1})}\right]\\; .
\end{aligned}
$$
At this point, we are now ready to use (4) so to introduce the reverse $q(x\_{t-1}\vert x\_t, x\_0)$.
It would of course be simpler to introduce $q(x\_{t-1}\vert x\_t)$ instead, however we cannot have a closed form for the latter.
Observe that by Bayes rule we have $q(x\_t\vert x\_{t-1}) = q(x\_t\vert x\_{t-1}, x\_0) = q(x\_{t-1}\vert x\_t, x\_0)q(x\_t\vert x\_0) / q(x\_{t-1}\vert x\_0)$.
Hence:
$$
\begin{aligned}
\log p\_\theta(x\_0) &\geq \mathbb{E}\_{q(\cdot\vert x\_0)}\left[\log p\_\theta(x\_T) + \sum\_{t=1}^T \log\frac{p\_\theta(x\_{t-1}\vert x\_t)}{q(x\_{t-1}\vert x\_{t}, x\_0)} + \sum\_{t=1}^T \log\frac{q(x\_{t-1}\vert x\_0)}{q(x\_{t}\vert x\_0)}\right]\\; , \\\
&= \mathbb{E}\_{q(\cdot\vert x\_0)}\left[\sum\_{t=1}^T \log\frac{p\_\theta(x\_{t-1}\vert x\_t)}{q(x\_{t-1}\vert x\_{t}, x\_0)} + \log\frac{p\_\theta(x\_T)}{q(x\_{T}\vert x\_0)}\right]\\; . &(\text{telescopic sum})\\\
&= \mathbb{E}\_{q(\cdot\vert x\_0)}\left[\sum\_{t=1}^T \mathbb{E}\_{x\_{t-1}\sim q(x\_{t-1}\vert x\_{t}, x\_0)}\left[\log\frac{p\_\theta(x\_{t-1}\vert x\_t)}{q(x\_{t-1}\vert x\_{t}, x\_0)}\right]\right\] + \mathbb{E}\_{x\_T\sim q(\cdot\vert x\_0)}\left[\log\frac{p\_\theta(x\_T)}{q(x\_T\vert x\_0)}\right]\\;. &(\text{tower rule})
\end{aligned}
$$
proving the following variational lower-bound:

{{< boxed title="Variational lower-bound" >}}
$\qquad\qquad\qquad\qquad\qquad\qquad \text{For any } x_0 \text{ let:}\\$
$$
\tag{5}
\mathcal{VB}_\theta(x_0) := \mathbb{E}_{q(\cdot\vert x_0)}\left[\sum_{t=1}^T \text{KL}\left(q(x_{t-1}\vert x_{t}, x_0)\,\vert\vert\, p_\theta(x_{t-1}\vert x_t, t)\right) + \text{KL}\left(q(x_T\vert x_0)\,\vert\vert\, p_\theta(x_T)\right)\right]\;.
$$
$\text{Then } \log p_\theta(x_0) \geq \mathcal{VB}_\theta(x_0).$
{{< /boxed >}}

Observe how all terms making up $\mathcal{VB}\_\theta(x\_0)$ involve KL-divergences for Gaussian distributions -- hence can be computed in closed form, 
while the expectation can be approximated via Monte-Carlo sampling.
As a lower-bound to our initial true objective, we now switch our attention to maximising $\mathcal{VB}\_\theta(x\_0)$
as a proxy.
Observe that while maximising $\mathcal{VB}_\theta(x_0)$, we can drop the term 
$\text{KL}\left(q(x\_T\vert x\_0)\\,\vert\vert\\, p\_\theta(x\_T)\right)$, as it does not
capture any learning parameter. We are now ready to state the actual objective we optimise 
to train a diffusion model. Before that, we follow {{< ref link="ncsf">}} [2]{{< /ref>}} and simplify the prior
covariance to $\Sigma\_\theta(x\_t, t) = \tilde\sigma\_t^2\mathbf{I}_d$ where $\tilde\sigma\_t^2$ is a fixed parameter.
Our new learning objective now writes:

$$
\tag{6}
\hat\theta\_n \in \argmax\_\theta \sum\_{i=1}^n \mathbb{E}_{q(\cdot\vert x\_0^i)}\left[\sum\_{t=1}^T \frac{1}{\sigma\_t^2}\\|\\mu\_t^i - \mu\_\theta(x\_t^i, t)\\|\_2^2\right]\\; ,
$$
where $\mu\_t$ and $\sigma\_t$ are defined in (4) -- $\mu\_t^i$ being the notation we used for the $\text{i}^\text{th}$ sample, from $x\_0^i$.


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
This result is a direct consequence of the facts that 1) we ignored the constant term
$\text{KL}\left(q(x\_T\vert x\_0)\\,\vert\vert\\, p\_\theta(x\_T)\right)$ -- recall that we fixed
$p\_\theta=\mathcal{N}(0, \mathbf{I}\_d)$, and 2) the closed-form expression of the Kullback-Leibler
divergence between two Gaussian with diagonal covariance matrices;
$$
\begin{aligned}
\text{KL}\left(q(x_{t-1}\vert x_{t}, x_0)\\,\vert\vert\\, p_\theta(x_{t-1}\vert x_t, t)\right) &= 
\text{KL}\left(\mathcal{N}(\mu\_t, \sigma\_t^2\mathbf{I}\_d) \\,\vert\vert\\, \mathcal{N}(\mu\_\theta(x\_t, t), \tilde\sigma\_t^2\mathbf{I}\_d) \right)\\;, \\\
&= \square \\, + \frac{1}{2\sigma\_t^2}\\| \mu\_t - \mu\_\theta(x\_t, t) \\|\_2^2\\; ,
\end{aligned}
$$
where $\square$ is a constant that does not depend on $\theta$. 
{{% /toggle_block %}}




### Re-parametrisation
Recall the expression for $\mu\_t$ established in (4):
$$
\mu\_t = \frac{\sqrt{1-\beta\_t}(1-\alpha\_{t-1})}{1-\alpha\_t}x\_t + \frac{\sqrt{\alpha\_{t-1}}\beta\_t}{1-\alpha\_t}x\_0\\; .
$$
This is a valid target to regress via our model $\mu\_\theta(x\_t, t)$ for the prior mean.
However, together with (5), we obtain some intuition that we could slightly change _what_ we model in order to ease learning.
Indeed, observe that from (3) one can re-parametrise $x\_t$ as:
$$
\begin{aligned}
x\_t &= \sqrt{\alpha\_t}x\_0 + \sqrt{1-\alpha\_t}\varepsilon\\;,\\\
\tag{7}
\Longleftrightarrow  x\_0 &= \frac{x\_t}{\sqrt{\alpha\_t}} - \frac{\sqrt{1-\alpha\_t}}{\sqrt{\alpha\_t}}\varepsilon \\;,
\end{aligned}
$$
for any $t\geq 0$, with $\varepsilon \sim \mathcal{N}(0, \mathbf{I}\_d)$. Plugging this back in (4) we obtain:
$$
\mu\_t = \frac{1}{\sqrt{1-\beta\_t}}\left(x\_t + \frac{\beta\_t}{\sqrt{1-\alpha\_t}}\varepsilon\right)\\;.
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
This is obtained by direct computation;
$$
\begin{aligned}
\mu\_t &= \frac{\sqrt{1-\beta\_t}(1-\alpha\_{t-1})}{1-\alpha\_t}x\_t + \frac{\sqrt{\alpha\_{t-1}}\beta\_t}{1-\alpha\_t}x\_0\\; ,\\\
&=  \frac{\sqrt{1-\beta\_t}(1-\alpha\_{t-1})}{1-\alpha\_t}x\_t + \frac{\sqrt{\alpha\_{t-1}}\beta\_t}{1-\alpha\_t}\left(\frac{x\_t}{\sqrt{\alpha\_t}} - \frac{\sqrt{1-\alpha\_t}}{\sqrt{\alpha\_t}}\varepsilon\right)\\;, &(\text{by (7)})\\\
&= \frac{\sqrt{1-\beta\_t}(1-\alpha\_{t-1})}{1-\alpha\_t}x\_t + \frac{\beta\_t}{(1-\alpha\_t)\sqrt{1-\beta\_t}}x\_t + \frac{\beta\_t}{\sqrt{1-\beta\_t}\sqrt{1-\alpha\_t}}\varepsilon\\;, \\\
&= \frac{(1-\beta\_t)(1-\alpha\_{t-1}) + \beta\_t}{(1-\alpha\_t)\sqrt{1-\beta\_t}}x\_t + \frac{\beta\_t}{\sqrt{1-\beta\_t}\sqrt{1-\alpha\_t}}\varepsilon\\;, \\\
&= \frac{1}{\sqrt{1-\beta\_t}}\left(x\_t + \frac{\beta\_t}{\sqrt{1-\alpha\_t}}\varepsilon\right)\\;.
\end{aligned}
$$
{{% /toggle_block %}}
Observe how we remove the dependency from $x\_0$ to introduce directly a sampled noise random variable.
This prompts for adding some _structure_ to $\mu\_\theta(x\_t, t)$. Instead of regressing 
directly $\mu\_t$, it now makes sense to write:
$$
\mu\_\theta(x\_t, t) = \frac{1}{\sqrt{1-\beta\_t}}\left(x\_t + \frac{\beta\_t}{\sqrt{1-\alpha\_t}}\varepsilon\_\theta(x\_t, t)\right)\\;,
$$
where $\varepsilon\_\theta(x\_t, t):\mathbb{R}^d\mapsto\mathbb{R}^d$, thanks to (6), can be learned through
$$
\begin{aligned}
\hat\theta\_n \in &\argmax\_\theta \sum\_{i=1}^n \mathbb{E}\_{q(\cdot\vert x\_0^i)}\left[\sum\_{t=1}^T \frac{\beta\_t^2}{(1-\beta\_t)(1-\alpha\_t)}\\|\varepsilon^i - \varepsilon\_\theta(x\_t^i, t)\\|\_2^2\right]\\; ,\\\
\tag{8}
&= \argmax\_\theta \sum\_{i=1}^n \mathbb{E}\_{q(\cdot\vert x\_0^i)}\left[\sum\_{t=1}^T \frac{\beta\_t^2}{(1-\beta\_t)(1-\alpha\_t)}\\|\varepsilon^i - \varepsilon\_\theta(\sqrt{\alpha\_t}x\_0^i + \sqrt{1-\alpha\_t}\varepsilon^i, t)\\|\_2^2\right]\\; .\\\
\end{aligned}
$$
The intuition behind the method now emerges. 
In short, we are training a model to reconstruct the noise $\varepsilon^i$ that was 
used by the forward process to create $x\_t^i$ from $x\_0^i$.
The weights $\frac{\beta\_t^2}{(1-\beta\_t)(1-\alpha\_t)}$ prioritise 
reconstruction for early stages ($t\ll 1$) over later stages $(t\approx T)$.

{{< infoblock>}}
$\quad$  {{< ref link="ncsf">}} [2]{{< /ref>}} reports the weighting, in an image generation context, has little effect on the final performance.
{{< /infoblock >}}

### Wrapping up
We are now ready to wrap-up this deep dive into the probabilistic foundations of diffusion models. 
A training objective can be derived by building a variational lower-bound on the maximum likelihood.
By re-parametrisation, learning a model to inverse some Gaussian noise arises as a valid, efficient design.
A stochastic gradient implementation of the learning protocol is detailed below, for illustration.

{{< pseudocode title="$\texttt{Training}$" >}} 
$\textbf{init } \text{data } \{x_0^i\}_{i=1}^n, \text{ horizon } T, \text{ sequence } \{\beta_{t}\}_{t=1}^T, \text{ parameter } \theta, \text{ learning rate } \gamma\;.\\$
$\textbf{while } \text{not converged}\\$
$\quad\text{sample } i\in\{1\ldots n\},\\$
$\quad\text{sample } t\in\{1, \ldots, T\},\\$
$\quad\text{sample } \varepsilon\sim\mathcal{N}(0, \mathbf{I}_d)\\$
$\quad \theta\leftarrow \theta - \gamma \frac{\beta_t^2}{(1-\beta_t)(1-\alpha_t)}\nabla_\theta\|\varepsilon - \varepsilon_\theta(\sqrt{\alpha_t}x_0^i + \sqrt{1-\alpha_t}\varepsilon, t)\|_2^2\\$
$\textbf{end while}\\$
{{< /pseudocode >}}
<br> 

Below is also detailed the protocol for _sampling_ from $p\_\theta$, by walking down the backward process.
It is directly obtained by the specified parametrisation.


{{< pseudocode title="$\texttt{Sampling}$" >}} 
$\textbf{init } \text{ horizon } T, \text{ sequence } \{\beta_{t}\}_{t=1}^T\;.\\$
$\text{sample } x_T\sim\mathcal{N}(0, \mathbf{I}_d)\\$
$\textbf{for } t=T, \ldots, 1\\$
$$
x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\left(x_t + \frac{\beta_t}{\sqrt{1-\alpha_t}}\varepsilon_\theta(x_t, t)\right)\;.
$$
$\textbf{end for}\\$
$\textbf{return } x_0$
{{< /pseudocode >}}
<br> 

## References

<div id="dpm"></div>
[1] Sohl-Dickstein & al, 2015. Deep Unsupervised Learning using Nonequilibrium Thermodynamics.

<div id="ncsf"></div>
[2] Song & Ermon, 2019. Generative modeling by estimating gradients of the data distribution.

<div id="ddpm"></div>
[3] Ho & al, 2020. Denoising diffusion probabilistic models.
<br>