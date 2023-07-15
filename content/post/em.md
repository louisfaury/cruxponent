+++
author = "Louis Faury"
title = "Expectation Maximization"
date = "2023-07-15"
+++

This post covers the Expectation Maximization (EM) algorithm, a popular heuristic
to (approximately) compute maximum likelihood estimates when dealing with unobserved /
latent data. We will motivate and derive the EM's inner mechanisms before making them explicit
on two classical examples (Gaussian Mixture parameter estimation and Hidden Markov Model identification).

<!--more-->


## Deriving the EM
### Setting and notations
We will introduce the EM through a toy example. Consider the following
data generating process:
$$
    Z \sim p\_\nu \quad \text{ and } X \sim p\_\mu(z) \\; .
$$
Both $Z$ and $X$ are both assumed to be real-valued. The variables $\nu$ and $\mu$ which respectively parametrize
the marginal distribution of $Z$ and the conditional distribution of $X$, are assumed to live in some Euclidian space.
We only observe $X$ -- the variable $Z$ is hidden or _latent_. Our goal is to estimate $\theta = (\nu, \mu)$ jointly. 
The Maximum Likelihood Estimator (MLE) is defined as:
$$
\begin{aligned}
\hat{\theta}:&\in \argmax_{\theta} p(X\vert \theta)\\;, & \\\
&= \argmax_{\theta} \int_{z} p(X, z\vert \theta)dz\\;, & (\text{law of total probability})\\\ 
&= \argmax_{\theta} \int_{z} p\_\mu(X\vert z)p\_\nu(z)dz\\;.
\end{aligned}
$$
As we'll see in our examples, this objective is typically _non-convex_ -- in contrast to similar settings
where there is no hidden variables. The Expectation Maximisation is a heuristic algorithm designed to optimize such a non-convex landscape.
When it applies, it is preferred over gradient-based approaches -- even though most modern libraries would combined the two. 

Before moving on, we will introduce the negative log-likelihood:
$$
\boxed{
\mathcal{L}(\theta) = -  \log \int_{z} p\_\mu(X\vert z)p\_\nu(z)dz \\;, }
$$
so that $\hat{\theta} \in \argmin_\theta  \mathcal{L}(\theta)$.

### Upper-bounding the negative log-likelihood
Much like a gradient-based algorithm, the EM will produce a sequence of estimate $\\{\theta\_t\\}\_{t\geq 1}$. At 
every iteration $t$, it will construct based on $\theta\_t$ an _upper-bound_ on the true loss $\mathcal{L}(\theta)$. The minimizer of this
upper-bound will be $\theta\_{t+1}$. The goal of this section is to derive said upper-bound. 


Let us define:
$$
{
\ell(\theta\vert\theta\_t) = \mathcal{L}(\theta\_t) -\int_{z} \log\left(\frac{p(X, z \vert \theta)}{p(X, z \vert \theta\_t)}\right)p(z \vert X, \theta\_t) dz \\; .
}
$$
Then, $\ell(\theta\vert\theta\_t)$ is a tangent upper-bound on $\mathcal{L}(\theta)$, meaning:
$$
\mathcal{L}(\theta) \leq \ell(\theta\vert\theta\_t) \text{ and } \mathcal{L}(\theta\_t) = \ell(\theta\_t\vert\theta\_t)\\; .
$$



{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
In the following we will use interchangeably $p\_\nu(z)$ with $p(z\vert \nu)$, and $p\_\mu(x\vert z)$ with $p(x\vert z, \mu)$.
Fix $\theta\_t$ and observe that:
$$
\begin{aligned}
\mathcal{L}(\theta) &= - \log \int_{z} p(X, z\vert \theta)dz \\;, \\\
&= - \log \int_{z} p(X, z\vert \theta)\frac{p(X, z \vert \theta\_t)}{p(X, z \vert \theta\_t)}dz \\;, \\\
&= - \log \int_{z} p(X\vert \mu, z)p(z\vert \nu)\frac{p(X \vert \theta\_t) }{p(X, z \vert \theta\_t)}p(z \vert X, \theta) dz \\;, \\\
&\leq -\int_{z} \log\left(p(X\vert \mu, z)p(z\vert \nu)\frac{p(X \vert \theta\_t)}{p(X, z \vert \theta\_t)}\right)p(z \vert X, \theta\_t) dz\\;, &(\text{Jensen's inequality}) \\\
&= -\int_{z} \log\left(p(X \vert \theta\_t)\right)p(z \vert X, \theta\_t) dz -\int_{z} \log\left(\frac{p(X, z \vert \theta)}{p(X, z \vert \theta\_t)}\right)p(z \vert X, \theta\_t) dz \\;,  \qquad & (\log(ab) = \log a + \log b) 
\\\
&= -  \log p(X\vert \theta\_t)-\int_{z} \log\left(\frac{p(X, z \vert \theta)}{p(X, z \vert \theta\_t)}\right)p(z \vert X, \theta\_t) dz \\;,  &\left(\int_{z} p(z \vert X, \theta\_t) dz = 1\right)  \\\
&= \mathcal{L}(\theta\_t) -\int_{z} \log\left(\frac{p(X, z \vert \theta)}{p(X, z \vert \theta\_t)}\right)p(z \vert X, \theta\_t) dz \\;.  &(\mathcal{L}(\theta) = -\log p(X\vert \theta))
\end{aligned}
$$
The equality at $\theta\_t$ is trivial from the definition of $\ell(\theta\vert \theta\_t)$.
{{% /toggle_block %}}


{{< image src="/em_curve.png" width="300px" align="right" >}}
The EM next iterate will be $\theta\_{t+1} = \text{arg min}\_{\theta}\\; \ell(\theta\vert\theta\_{t+1})$. Observe that this ensures that:
$\mathcal{L}(\theta\_{t+1})\leq \mathcal{L}(\theta\_{t})$ -- in other words, we have _guaranteed improvement_. Indeed:
$$
\begin{aligned}
    \mathcal{L}(\theta\_{t+1}) &\leq \ell(\theta\_{t+1} \vert \theta\_t) \\;, \\\
     &\overset{(i)}{\leq}  \ell(\theta\_{t} \vert \theta\_t) \\;, \\\
    &= \mathcal{L}(\theta\_t)\\;,
\end{aligned}
$$
where $(i)$ uses that $ \ell(\theta\_{t+1} \vert \theta\_t) = \min\_\theta \ell(\theta\vert\theta\_t)$.

We can now write down a very naked first description the EM:
{{< pseudocode title="EM" >}}
Input: $\theta_0$
<br>
For every $t\geq 1$:
$$
    \theta_{t+1} \in \argmin_\theta \ell(\theta\vert \theta_t) \; .
$$
{{< /pseudocode >}}

<br>

This will guarantee improvement at every iteration. Furthermore, it is rather easy to show that the
sequence $\\{\theta\_t\\}\_t$ converges to a _stationary_ point of $\mathcal{L}$. 
The name of this heuristic is still rather obscure; what do we say that this is Expectation Maximization? 


### Expectation Maximisation
Let's dive deeper into the minimization of $ \ell(\theta\vert \theta_t)$. Start by removing constant terms w.r.t
$\theta$, we obtain:
$$
\begin{aligned}
    \argmin\_\theta \ell(\theta\vert \theta\_t) &= \argmin\_\theta\mathcal{L}(\theta\_t) -\int_{z} \log\left(\frac{p(X, z \vert \theta)}{p(X, z \vert \theta\_t)}\right)p(z \vert X, \theta\_t) dz \\;, \\\
&= \argmin\_\theta  -\int_{z} \log\left(p(X, z \vert \theta)\right)p(z \vert X, \theta\_t) dz \\;,\\\
&= \argmax\_\theta  \mathbb{E}\left[\log p(X, z \vert \theta) \middle\vert X, \theta\_t \right]\\;.
\end{aligned}
$$

The quantity $\mathbb{E}\left[\log p(X, z \vert \theta) \middle\vert X, \theta\_t \right]$ is usually called the
_complete-data likelihood_, in opposition to the incomplete-data likelihood $\mathcal{L}(\theta)$. 
The first step in completing the EM step is therefore to compute this expectation,  or equivalenty to compute the
conditional $p(z \vert X, \theta\_t)$. This is called the **Expectation** step. The second step consist in
maximizing this expection w.r.t $\theta$; this is the **Maximization** step. 

{{< pseudocode title="EM" >}}
Input: $\theta_0$
<br>
For every $t\geq 1$:
<br>
[Expectation] compute $p(z \vert X, \theta_t)$ and:
$$
 \mathbb{E}\left[\log p(X, z \vert \theta) \middle\vert X, \theta_t \right] = \int_{z} \log\left(p(X, z \vert \theta)\right)p(z \vert X, \theta\_t) dz \; .
$$
[Maximization]
$$
\theta_{t+1} \in \argmax_\theta \mathbb{E}\left[\log p(X, z \vert \theta) \middle\vert X, \theta_t \right] \; .
$$
{{< /pseudocode >}}
How to conduct each step might still be quite obscure. In the rest of this post, we will make it explicit for
two distinct examples.

## Fitting a Gaussian Mixture Model

## Learning a finite Hidden Markov Model