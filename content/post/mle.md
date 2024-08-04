+++
author = "Louis Faury"
title = "Oldies but goodies: MLE"
date = "2024-07-20"
+++

Maximum-Likelihood Estimation (MLE) is ubiquitous in modern (supervised) machine learning, to the point that we sometimes forget 
it is the very principle behind the criterion we use to train fancy models. 
Justifying the MLE's current popularity from a statistical perspective only would be a stretch; 
nonetheless, it is useful to remember that it enjoys some comfortable (asymptotic) statistical properties.
This post provides proof for two such attributes: consistency
and asymptotic normality.
<!--more-->

Maximum-Likelihood Estimation (MLE) is a generic paradigm for estimating unknown parametric distributions
from a collection of samples.
Under mild regularity conditions, it yields _consistent_ estimators: as the number of samples goes to infinity, we are guaranteed to properly identify the unknown distribution.
This property mirrors the Law of Large Numbers (LLN); another important attribute of MLE echoes the Central Limit Theorem (CLT) and guarantees that
MLE estimators are _asymptotically normal_. 
This result is comforting but barely useful in itself; yet, we will see the variance of said normal random variables
allows us to prove that the MLE is _asymptotically efficient_ as it matches the [Cramér-Rao bound](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound). 


## Setting

{{< warningblock>}}
$\quad$ To keep things simple, we will restrict our short study to real-valued random variables.
The claimed results easily extend to discrete random variables (basically, replacing integrals by sums).
{{< /warningblock >}}

Let's quickly go through the measure-theoretic set-up: let $(\Omega, \mathcal{F}, \mathbb{P})$ a probability space and 
$X : \Omega \mapsto \mathbb{R}$ a $\mathcal{F}/\mathcal{B}(\mathbb{R})$ measurable function (in plain words, our real-valued random variable).
We let $\mathbb{P}_{\theta\_\star}$ the probability law of $X$, assumed absolutely continuous w.r.t the Lebesgue measure $\nu$. 
It is indexed by a fixed but unknown parameter ${\theta\_\star}$; we only assume that $\theta\in\Theta$ where $\Theta$ is a known, compact subset of $\mathbb{R}$.
For any $\theta\in\Theta$ we will denote $p\_\theta := d\mathbb{P}\_{\theta}/d\nu$ the density of the law indexed by $\theta$.

We will assume that we have access to a set of _independent_ realisations $\mathbf{X}\_n := \\{X\_1, \ldots, X\_n \\}  \sim \mathbb{P}\_{\theta\_\star}^n$.
The (normalised) log-likelihood of the dataset $\mathbf{X}\_n$ under an hypothesis $\mathbb{P}\_\theta$ will be denoted $\mathcal{L}\_n(\theta)$:
$$
\begin{aligned}
\mathcal{L}\_n(\theta) :&= \frac{1}{n}\log \frac{d\mathcal{P}\_\theta^n}{d\nu}(\mathbf{X}\_n) \\;, &\\\
&= \frac{1}{n}\log \prod\_{i=1}^n p\_\theta(X\_i) \\;,& (\text{independence}) \\\
&= \frac{1}{n}\sum\_{i=1}^n \log p\_\theta(X\_i) \\;. 
\end{aligned}
$$
The Maximum Likelihood Estimator is defined as the maximiser of the log-likelihood. 
(We will soon impose conditions for this definition to make sense, _i.e._ that an argmax exists and is unique).


$$
\theta\_n := \argmax\_{\theta\in\Theta} \mathcal{L}\_n(\theta) \\; .
$$



{{< infoblock>}}
$\quad$ As a maximiser of some statistic of $\mathbf{X}_n$, the MLE is a special case of an <a href="https://en.wikipedia.org/wiki/Extremum_estimator" style="text-decoration:none; color:#0074aa;" ">extremum estimator</a>.      
{{< /infoblock >}}


## Consistency
The MLE is _consistent_: as our dataset $\mathbf{X}_n$ grows ($n\to\infty$) the estimator $\theta\_n$ converge (in some sense)
to the unknown $\theta\_\star$. Below, we will show that this convergence happens in probability $\theta\_n \overset{\text{p}}{\to} \theta\_\star$:

{{< boxed title="Consistency" >}}
$qquad \qquad \quad\;\text{Under some regularity assumptions (see 1-4 below):}\\$
$$
\forall\varepsilon>0,\; \lim_{n\to\infty}\mathbb{P}(\vert \theta_n - \theta_\star\vert \geq \varepsilon) = 0\; . 
$$
{{< /boxed >}}

This is quite comforting: as we collect more data, we can expect that in the limit we'd recover $\theta\_\star$.
The intuition behind proof is quite straight-forward. It essentially relies on the fact that the expected log-likelihood
$
\mathrm{L}(\theta) := \mathbb{E}\_{p\_{\theta\_\star}}\left[\log p\_\theta(X)\right]
$
is maximised by $\theta\_\star$, and that by the (weak) law of large numbers $\mathcal{L}\_n(\theta) \overset{\text{p}}{\to} \mathrm{L}(\theta)$ for any $\theta$.
Since $\theta\_n$ maximises $\mathcal{L}\_n(\theta)$, it is expected that it converges to $\theta\_\star$.

{{% toggle_block background-color="#CBE4FE" title="Note" default-display="none"%}}
Using the strong law of large numbers naturally yields the stronger almost-sure convergence.
We will focus on convergence in probability, for simplicity.
{{% /toggle_block %}}

We can now work on making this intuition formal. That
$
\mathrm{L}(\theta\_\star) = \max\_{\theta} \mathrm{L}(\theta)
$
can easily be proved using Jensen inequality, without any specific assumptions. 

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
$$
\begin{aligned} \mathrm{L}(\theta) - \mathrm{L}(\theta\_\star) &= \int\_\mathbb{R} \log\frac{p\_{\theta}(x)}{p\_{\theta\_\star}(x)} p\_{\theta\_\star}(x)dx\\;, & \\\
&\leq \log \int\_\mathbb{R} \frac{p\_{\theta}(x)}{p\_{\theta\_\star}(x)} p\_{\theta\_\star}(x)dx \\;, &(\text{concavity})\\\
&= \log \int\_\mathbb{R}p\_{\theta}(x)dx \\;,&\\\
&= \log 1 = 0\\; .
\end{aligned}
$$
{{% /toggle_block %}}

That, for any $\theta$, we have $\mathcal{L}\_n(\theta) \overset{\text{p}}{\to} \mathrm{L}(\theta)$ is a straight-forward of the weak LLN.
However, to guarantee the convergence of the argmax, we will need a little extra help that will come from the _uniform_ law of large number.
We will adopt a set of sufficient (but not necessary) set of regularity assumptions to ensure it holds.
1. Identifiability: $\theta\_1 \neq \theta\_2 \Longleftrightarrow p\_{\theta\_1} \neq p\_{\theta\_2}$,
2. Regularity: $\forall x\in\mathbb{R}, \\; \theta \mapsto p\_\theta(x)$ is continuous,
3. Compactness: $\Theta$ is compact subset of $\mathbb{R}$,
4. Dominance: it exists an integrable $D : \mathbb{R}\mapsto \mathbb{R}$ such that $\log p\_\theta(x) < D(x)$ for any $\theta$ in $\Theta$.

Identifiability is necessary to ensure that convergence of distributions implies convergence in parameter. 
The regularity and compactness assumption typically ensures that $\theta\_n$ is safely defined as the maximiser of $\mathcal{L}\_,$ (_i.e._ a maximiser exists).
The dominance assumption is key to ensure we obtain a uniform law of large number. 
Actually, let's state this result straight away.

{{< boxed title="Uniform (weak) LLN" >}}
$\qquad \qquad \qquad \qquad\qquad \text{Under assumptions 1-4, for any } \varepsilon>0:$
$$
\sup_{\theta\in\Theta} \vert \mathcal{L}_n(\theta) - \mathrm{L}(\theta)\vert \overset{\text{p}}{\to} 0\; .
$$
{{< /boxed >}}

This essentially extends the weak LLN for the convergence to happen uniformy in $\theta$.
The result is immediate if $\Theta$ is finite, thanks to the LLN and a simple union bound. 
When $\Theta$ has infinite cardinality, we fall back to the finite setting thanks to covering argument. 


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
Fix $\varepsilon>0$. Thanks to our regularity and compactness assumption, we can claim uniform contuinity in the sense:
$$
\exists \delta>0 \text{ s.t } \forall (\theta, \theta^\prime), \\; \vert \theta - \theta^\prime \vert \leq \delta \Longrightarrow \sup\_{x} \vert p\_\theta(x) - p\_{\theta^\prime}(x) \vert \leq \varepsilon/4\\; . 
$$
By compactness of $\Theta$, we can build a _finite_ cover of $\Theta$: a collection $\\{\theta\_k\\}\_k$ such that:
$$
\tag{a}
\forall \theta \in \Theta,  \exists \theta\_k \text{ s.t. } \sup\_{x} \vert p\_\theta(x) - p\_{\theta\_k}(x) \vert \leq \varepsilon/4
$$
The rest is a story of decomposition; for any $\theta\in\Theta$:
$$
\begin{aligned}
\vert \mathcal{L}\_n(\theta) - \mathrm{L}(\theta)\vert &=  \left\vert \frac{1}{n}\sum\_{i=1}^n \left(p\_\theta(X\_i) - \mathbb{E}\_{\theta\_\star}[p\_\theta(X)]\right)\right\vert\\;, \\\
&\leq \frac{1}{n}\sum\_{i=1}^n \left\vert p\_\theta(X\_i) - \mathbb{E}\_{\theta\_\star}[p\_\theta(X)]\right\vert \\;, \\\
&\overset{(i)}{\leq} \frac{1}{n}\sum\_{i=1}^n \left\vert p\_\theta(X\_i) - p\_{\theta\_k}(X\_i)\vert + \vert p\_{\theta\_k}(X\_i) - \mathbb{E}\_{\theta\_\star}[p\_{\theta\_k}(X)]\vert + \vert \mathbb{E}\_{\theta\_\star}[p\_{\theta\_k}(X)] - \mathbb{E}\_{\theta\_\star}[p\_\theta(X)]\right\vert \\;, \\\
&\leq \varepsilon/4 + \vert p\_{\theta\_k}(X\_i) - \mathbb{E}\_{\theta\_\star}[p\_{\theta\_k}(X)]\vert + \varepsilon/4\\; .
\end{aligned}
$$
where $(i)$ is a simple triangular inequality and $\theta\_k$ is chosen as in (a). As a result:
$$
\sup\_{\theta\in\Theta}\vert \mathcal{L}\_n(\theta) - \mathrm{L}(\theta)\vert \leq \varepsilon/2 + \max\_{\theta\_1, \ldots, \theta\_k}  \vert p\_{\theta\_k}(X\_i) - \mathbb{E}\_{\theta\_\star}[p\_{\theta\_k}(X)]\vert\\; .
$$
Note that for any $\theta\_k$ we have $p\_{\theta\_k}(X\_i) \overset{\text{p}}{\to} \mathbb{E}\_{\theta\_\star}[p\_{\theta\_k}(X)]$ by the weak LLN. 
Since there is only a finite number of them, we get our result via a simple union bound. Formally:
$$
\mathbb{P}\left(\sup\_{\theta\in\Theta}\vert \mathcal{L}\_n(\theta) - \mathrm{L}(\theta)\vert \geq \varepsilon\right) \leq \mathbb{P}\left(\max\_{\theta\_1, \ldots, \theta\_k}  \vert p\_{\theta\_k}(X\_i) - \mathbb{E}\_{\theta\_\star}[p\_{\theta\_k}(X)]\vert \geq \varepsilon/2\right) \overset{n\to\infty}{\longrightarrow} 0\\; .
$$
{{% /toggle_block %}}

We now have the tools to show convergence of the argmax -- that is, $\theta\_n$ for $\mathcal{L}\_n$ towards $\theta\_\star$ for $\mathrm{L}$.
Indeed, for any $\varepsilon>0$, using the uniform convergence easily yields that:
$$
\tag{1}
\lim\_{n\to\infty} \mathbb{P}\left(\vert \mathcal{L}\_n(\theta\_n) - \mathrm{L}(\theta\_\star)\vert \geq \varepsilon\right) = 0 \\; .
$$
In others words, we showed that $\mathcal{L}\_n(\theta\_n) \overset{\text{p}}{\to} \mathrm{L}(\theta)$.
{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
$$
\begin{aligned}
\mathbb{P}\left(\sup\_{\theta\in\Theta} \vert \mathcal{L}\_n(\theta) - \mathrm{L}(\theta)\vert \leq \varepsilon\right) &= \mathbb{P}\left( \forall\theta\in\Theta,\\, \vert \mathcal{L}\_n(\theta) - \mathrm{L}(\theta)\vert \leq \varepsilon\right)\\;, \\\
&\leq \mathbb{P}\left( \forall\theta\in\Theta, \\, \mathcal{L}\_n(\theta) \leq \mathrm{L}(\theta) + \varepsilon\right)\\;, &(\mathbb{P}(A\cap B) \leq \mathbb{P}(A))\\\
&\leq  \mathbb{P}\left( \forall\theta\in\Theta, \\, \mathcal{L}\_n(\theta) \leq \mathrm{L}(\theta\_\star) + \varepsilon\right)\\;, &(\theta\_\star = \argmax\_{\theta}\mathrm{L}(\theta))\\\
&= \mathbb{P}\left(\mathcal{L}\_n(\theta\_n) \leq \mathrm{L}(\theta\_\star) + \varepsilon\right)\\;. &(\theta\_n = \argmax\_{\theta}\mathcal{L}\_n(\theta))
\end{aligned}
$$
Since $\lim\_{n\to\infty} \mathbb{P}\left(\sup\_{\theta\in\Theta} \vert \mathcal{L}\_n(\theta) - \mathrm{L}(\theta)\vert \leq \varepsilon\right) =1$ we get that $\lim\_{n\to\infty}\mathbb{P}\left(\mathcal{L}\_n(\theta\_n) \leq \mathrm{L}(\theta\_\star) + \varepsilon\right) = 1$.
Using a similar argument yields $\lim\_{n\to\infty}\mathbb{P}\left(\mathrm{L}(\theta\_\star) \leq  \mathcal{L}\_n(\theta\_n) + \varepsilon\right) = 1$. Putting those together proves our claim. 
{{% /toggle_block %}}
Finally, observe that by the triangular inequality we have:

$$
\vert \mathrm{L}(\theta\_n) - \mathrm{L}(\theta\_\star) \vert \leq  \vert \mathrm{L}(\theta\_n) - \mathcal{L}\_n(\theta\_n) \vert + \vert \mathcal{L}\_n(\theta\_n) - \mathrm{L}(\theta\_\star) \vert\\; ,
$$
where $\vert \mathcal{L}\_n(\theta\_n) - \mathrm{L}(\theta\_\star) \vert \overset{\text{p}}{\to} 0$ thanks to (1) and 
$\vert \mathrm{L}(\theta\_n) - \mathcal{L}\_n(\theta\_n) \vert \overset{\text{p}}{\to}0$ by uniform convergence. 
Hence 
$
\mathrm{L}(\theta\_n) \overset{\text{p}}{\to} \mathrm{L}(\theta\_\star)
$.
Thanks to identifiability assumptions 1., we proved that $\theta\_n$ converges in probability to $\theta\_\star$. 

## Asymptotic Normality
So far, under some regularity assumptions, we showed that $\theta\_n \overset{\text{p}}{\to} \theta_\star$ -- a property mimimic the (weak) LLN.
It is natural to wonder "how fast" this convergence happens, at least asymptotically. 
In other words, we are asking whether a Central Limit Theorem (CLT) holds by the MLE. Good news, it does!
Below, we denote:
$$
\mathcal{I}(\theta\_\star) := \mathbb{E}\_{\theta_\star}\left[\left(\left.\frac{d}{d\theta}\log p\_\theta(X)\right\vert_{\theta\_\star}\right)^2\right]\\;,
$$
a quantity known as the [Fisher information](https://en.wikipedia.org/wiki/Fisher_information).
{{< boxed title="Asymptotic Normality" >}}
$\qquad \qquad \qquad\qquad \qquad \;\, \text{Under some regularity assumptions (see below):}\\$
$$
\sqrt{n}(\theta_n-\theta_\star) \overset{\text{d}}{\longrightarrow} \mathcal{N}(0, \mathcal{I}(\theta_\star)^{-1})\; .
$$
{{< /boxed >}}

In a CLT-like fashion, $\sqrt{n}(\theta\_n-\theta\_\star)$ is asymptotically normally distributed.
The variance of said normal distribution is of particular interest. Indeed, the Cramer-Rao bound states that any 
unbiased estimator $\theta^\prime$ of $\theta_\star$ checks
$
\mathbb{V}\text{ar}(\theta^\prime) \geq \mathcal{I}(\theta_\star)^{-1}
$. We just claimed that asymptotically, $\theta\_n$ matches this lower-bound: it is therefore the unbiased estimator with smallest variance. 
Such an estimator is sometimes called _asymptotically efficient_.

To prove the asymptotical normality of the MLE, we will assume that:
1. The assumptions 1-4 hold, 
2. For any $x$, the function $\theta\mapsto p\_\theta(x)$ is twice continuously differentiable. 

Before jumping into the proofs, it is useful to recall some important properties.
First, the expectation of the score function is always 0:
$$
\tag{2}
\mathbb{E}\_{\theta\_\star}\left[\left.\frac{d}{d\theta}\log p\_{\theta}(X)\right\vert\_{\theta\_\star}\right] = 0 \\;. 
$$
Further, under 2., the following identity holds:
$$
\tag{3}
\mathcal{I}(\theta\_\star) = \mathbb{E}\_\theta\left[\left.\frac{d^2}{d\theta}\log p\_{\theta}(X)\right\vert\_{\theta\_\star}\right]\\; .
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
Let us start by proving (2). The proof involves a straight-forward computation. For any $\theta$:
$$
\begin{aligned}
\mathbb{E}\_{\theta}\left[\left.\frac{d}{d\theta}\log p\_{\theta}(X)\right\vert\_{\theta}\right] &=  \mathbb{E}\_{\theta}\left[\frac{1}{p\_{\theta}(X)}\left.\frac{d}{d\theta} p\_{\theta}(X)\right\vert\_{\theta}\right]\\;,\\\
&= \int\_{\mathbb{R}} \left.\frac{d}{d\theta} p\_{\theta}(x)\right\vert\_{\theta}dx\\;, \\\
&=  \left.\frac{d}{d\theta}\left(\int\_{\mathbb{R}} p\_{\theta}(x)dx\right)\right\vert\_{\theta}\\;, \\\
&= \frac{d}{d\theta} 1 = 0\\; .
\end{aligned}
$$
We now prove (3). Bootstrapping on the computation above, for any $\theta$:
$$
\begin{aligned}
\mathbb{E}\_\theta\left[\left.\frac{d^2}{d\theta}\log p\_{\theta}(X)\right\vert\_{\theta}\right] &= \mathbb{E}\_{\theta}\left[\left(\frac{1}{p\_{\theta}(X)}\left.\frac{d}{d\theta} p\_{\theta}(X)\right\vert\_{\theta}\right)^2\right] + \mathbb{E}\_{\theta}\left[\frac{1}{p\_{\theta}(X)}\left.\frac{d^2}{d\theta^2} p\_{\theta}(X)\right\vert\_{\theta}\right]\\;.\\\
&\overset{(i)}{=} \mathbb{E}\_{\theta}\left[\left(\frac{1}{p\_{\theta}(X)}\left.\frac{d}{d\theta} p\_{\theta}(X)\right\vert\_{\theta}\right)^2\right] \\;, \\\ 
&= \mathbb{E}\_{\theta}\left[\left(\left.\frac{d}{d\theta} \log p\_\theta(X) \right\vert\_{\theta}\right)^2\right]\\; ,
\end{aligned}
$$
where in $(i)$ we resorted to the same argument we used while proving (2) in order to show $\mathbb{E}\_{\theta}\left[\frac{1}{p\_{\theta}(X)}\left.\frac{d^2}{d\theta^2} p\_{\theta}(X)\right\vert\_{\theta}\right]=0$.

{{% /toggle_block %}}


$\theta\_n$ being the argmax of $\mathcal{L}\_n(\theta)$, we have $\left.\frac{d}{d\theta} \mathcal{L}\_n(\theta)\right\vert\_{\theta\_n}=0$ (assuming it lies within $\Theta$).
By an exact Taylor expansion, and adopting more convenient notations:
$$
\begin{aligned}
0 &= \mathcal{L}\_n^\prime(\theta\_\star) +  \mathcal{L}\_n^{\prime\prime}(\tilde\theta\_n)(\theta\_n - \theta\_\star)\\; , \\\
&\Longrightarrow \theta\_n - \theta\_\star = -\mathcal{L}\_n^\prime(\theta\_\star) / \mathcal{L}\_n^{\prime\prime}(\tilde\theta\_n) \\; ,
\end{aligned}
$$
where $\tilde\theta\_n\in[\theta\_n, \theta\_\star]$. 
Observe that:
$$
\begin{aligned}
\mathcal{L}\_n^\prime(\theta\_\star) &= \mathcal{L}\_n^\prime(\theta\_\star) - \mathrm{L}(\theta\_\star)\\;,   &(\mathrm{L}(\theta\_\star) = 0) \\\
&= \frac{1}{n}\sum\_{i=1}^n \frac{d}{d\theta}\log p\_\theta(X\_i)\vert\_{\theta\_\star} - \mathbb{E}\_{\theta\_\star}\left[\frac{d}{d\theta}\log p\_\theta(X)\right] \\;. \\\ 
\end{aligned}
$$
The CLT applies, guaranteeing that $\mathcal{L}\_n^\prime(\theta\_\star) \overset{\text{d}}{\to} \mathcal{N}(0, \mathcal{I}(\theta\_\star))$.


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
The CLT establishes that:
$$
\mathcal{L}\_n^\prime(\theta\_\star) \overset{\text{d}}{\to} \mathcal{N}\left(0, \mathbb{V}\text{ar}\left(\frac{d}{d\theta}\log p\_\theta(X)\vert\_{\theta\_\star}\right)\right)
$$
We only have left to calculate $\frac{d}{d\theta}\log p\_\theta(X\_i)\vert\_{\theta\_\star}$.
$$
\begin{aligned}
\mathbb{V}\text{ar}\left(\frac{d}{d\theta}\log p\_\theta(X)\vert\_{\theta\_\star}\right) &= \mathbb{E}\_{\theta\_\star}\left[\left(\left.\frac{d}{d\theta}\log p\_\theta(X)\right\vert\_{\theta\_\star}\right)^2\right] -  \left(\mathbb{E}\_{\theta\_\star}\left[\left.\frac{d}{d\theta}\log p\_\theta(X)\right\vert\_{\theta\_\star}\right]\right)^2\\;, \\\
&\overset{(i)}{=} \mathbb{E}\_{\theta\_\star}\left[\left(\left.\frac{d}{d\theta}\log p\_\theta(X)\right\vert\_{\theta\_\star}\right)^2\right] = \mathcal{I}(\theta\_\star)\\;.
\end{aligned}
$$
where we used (2) in $(i)$.
{{% /toggle_block %}}
Finally, by consistency of $\theta\_n$, we have the $\tilde\theta\_n \overset{\text{p}}{\to} \theta\_\star$. 
Using another uniform convergence argument (similar to the one introduced to demonstrate consistency) one can easily show that:
$$
\mathcal{L}\_n^{\prime\prime}(\tilde\theta\_n) \overset{\text{p}}{\to} \mathbb{E}\_\theta\left[\left.\frac{d^2}{d\theta}\log p\_{\theta}(X)\right\vert\_{\theta\_\star}\right] = \mathcal{I}(\theta\_\star)\\;,
$$
thanks to (3).
Putting our findings together: we showed that $\theta\_n - \theta\_\star = -\mathcal{L}\_n^\prime(\theta\_\star) / \mathcal{L}\_n^{\prime\prime}(\tilde\theta\_n)$, where
$\mathcal{L}\_n^\prime(\theta\_\star) \overset{\text{d}}{\to} \mathcal{N}(0, \mathcal{I}(\theta\_\star))$ and 
$\mathcal{L}\_n^{\prime\prime}(\tilde\theta\_n) \overset{\text{p}}{\to} \mathcal{I}(\theta\_\star)$; a simple application of [Stutsky's lemma](https://en.wikipedia.org/wiki/Slutsky%27s_theorem) yields the announced claim:
$$
\theta\_n - \theta\_\star \overset{\text{d}}{\to} \mathcal{N}(0, \mathcal{I}(\theta\_\star)^{-1})\\; .
$$