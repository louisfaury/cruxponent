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
\hat{\theta}&\in \argmax_{\theta} p(X\vert \theta)\\;, & \\\
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
&= \argmax\_\theta  \left\\{\mathcal{Q}(\theta\vert \theta\_t):= \mathbb{E}\left[\log p(X, z \vert \theta) \middle\vert X, \theta\_t \right]\right\\}\\;.
\end{aligned}
$$

The quantity $p(x, z \vert \theta)$ is usually called the
_complete-data likelihood_, in opposition to the incomplete-data likelihood $p(x\vert \theta)$.
Its conditional expectation with respect to the observations and the model estimates $\theta\_t$ is denoted 
$\mathcal{Q}(\theta\vert \theta\_t):=\mathbb{E}\left[\log p(X, z \vert \theta) \middle\vert X, \theta\_t \right]$,
and refered to as the _auxiliary log-likelihood_.

The first step in completing the EM step is to materialise the auxiliary log-likelihood, which requires computing the
conditional $p(z \vert X, \theta\_t)$. This is called the **Expectation** step. The second step consist in
maximizing the auxiliary log-likelihood w.r.t $\theta$; this is the **Maximization** step. 

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
\theta_{t+1} \in \argmax_\theta \left\{\mathcal{Q}(\theta\vert \theta_t)= \mathbb{E}\left[\log p(X, z \vert \theta) \middle\vert X, \theta_t \right] \right\}\; .
$$
{{< /pseudocode >}}
How to conduct each step might still be quite obscure. In the rest of this post, we will make it explicit for
two distinct examples.

## Fitting a Gaussian Mixture Model
 

{{< image src="/gmm_samples.png" width="300px" align="right" >}}


Consider the data-generating process of a Gaussian Mixture Process (GMM).
Formally, let $\alpha\_1, \ldots, \alpha\_K \in \mathbb{R}^d$
and $\Sigma\_1, \ldots, \Sigma\_K \succeq 0$ be, respectively, the means and covariance matrices of $K$ Gaussian random variables.
A new sample $X$ from the GMM is drawn by first sampling its hidden variable $Z\in\\{1, \ldots, K\\}$, representing which Gaussian component will
generate $X$. We then generate $X\sim \mathcal{N}(\alpha\_Z, \Sigma\_Z)$. Observe that the latent variable $p\_\nu$ being discrete, 
it parameterizable by a vector living in the $K$-dimensional simplex:
$$
\nu \in \Delta(K) := \\{ \omega \in \mathbb{R}^K, \\; \omega\_i \geq 0 \\, \forall i \in\\{1, \ldots, K\\} \\,, \\; \sum\_{i=1}^K \omega\_i = 1  \\} \\; ,
$$
so that $p\_\nu(z) = \nu\_z$ for any $z\in \\{ 1, \ldots, K\\}$.

Let $X\_1, \ldots, X\_n$ be $n$ i.i.d samples from the GMM. Recall that we wish to jointly estimate jointly:
$$
\theta := ( \nu\_1, \ldots, \nu\_K, \alpha\_1, \ldots, \alpha\_K, \Sigma\_1, \ldots, \Sigma\_K) \\; . 
$$
The log-likelihood of the model:
$$
\mathcal{L}(\theta)  = -\sum_{i=1}^n \log\left(\sum_{z=1}^K \nu\_z \mathcal{N}(x\_i \vert \alpha\_z, \Sigma\_z) \right) \\; ,
$$
is easily shown to be a *non-convex* function of $\theta$. We therefore turn to the EM procedure, of which we now make each step explicit. 
In what follows, let $\theta\_t$ be the EM current iteration's estimator. 

### Expectation Step
We must compute, for each sample, the conditional distribution $p(z\_i\vert x\_i, \theta\_t)$. 
To reduce clutter we will use the short-hand $\pi_k^i := p(z\_i = k\vert x\_i, \theta\_t)$. Observe that:
$$
\begin{aligned}
    \pi_k^i &= p(z\_i=k\vert x\_i, \theta\_t) \\; ,\\\
&\propto p(z\_i=k\vert \theta\_t)p(x\_i\vert z\_i=k, \theta\_t) &(\text{Bayes rule}) \\; , \\\
&\propto \nu_k^t \cdot \mathcal{N}(x\_i\vert \alpha\_k, \Sigma\_k\)
\end{aligned}
$$
where $\nu\_k^t$ is our current mixture weight estimator for the $k$' component (included in $\theta\_t$).
Normalizing the distribution yields the following expression, concluding the Expectation step: 
$$
\boxed{
\pi_k^i = \frac{ \nu_k^t \cdot \mathcal{N}(x\_i\vert \alpha\_k, \Sigma\_k\)}
{\sum_{\ell=1}^K  \nu_{\ell}^t \cdot \mathcal{N}(x\_i\vert \alpha\_{\ell}, \Sigma\_{\ell}\)}\\; .
}
$$

### Maximisation step
Let us now write down the complete-data likelihood:
$$
\begin{aligned}
    \mathcal{Q}(\theta\vert\theta\_t) &= \sum\_{i=1}^n \left(\sum_{k=1}^K p(z\_i = k\vert x\_i , \theta\_t)\log p(x\_i, z\_i=k\vert \theta)\right) \\;, \\\
    &= \sum_{i=1}^n \left(\sum\_{k=1}^K \pi_k^i \log \left(p(x\_i, z\_i=k\vert \theta)\right)\right) \\;, &(\text{using shorthand}) \\\
    &= \sum_{i=1}^n \left(\sum\_{k=1}^K \pi_k^i \log p(x\_i\vert z\_i=k, \theta) + \pi_k^i \log p(z\_i=k\vert \theta\_t) \right) \\;, &(\text{Bayes rule}) \\\
    &= \sum_{i=1}^n \sum\_{k=1}^K \pi_k^i \log \mathcal{N}(x\_i\vert \alpha_k, \Sigma_k) + \sum_{i=1}^n \sum_{k=1}^K \pi_k^i  \log \nu_k \\;. &(\text{re-arranging}) \\\
\end{aligned}
$$
This is a *concave* function of $\theta$ -- hence its maximisation via gradient-based approach is indeed principled.
Our luck doesn't stop here in this case;  we actually have a closed-form for $\theta\_{t+1} \in \argmax \mathcal{Q}(\theta\vert \theta\_t)$:
$$
\boxed{
\begin{aligned}
\nu_k^{t+1} &= \frac{\sum\_{i=1}^n \pi\_k^i}{\sum\_{i=1}^n \sum\_{\ell=1}^K \pi\_\ell^i} \\;, \\\
\alpha\_k^{t+1} &= \frac{1}{\sum\_{i=1}^n \pi\_k^i} \sum\_{i=1}^n \pi\_k^i x\_i\\;, \\\
\Sigma\_k^{t+1} &= \frac{1}{\sum\_{i=1}^n \pi\_k^i} \sum\_{i=1}^n  \pi\_k^i (x\_i - \mu\_i^{t+1})(x\_i - \mu\_i^{t+1}) ^\top \\;.
\end{aligned}
}
$$
for all $k\in\\{1, \ldots, K\\}$. 

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
We will present the proof only for the first component $\nu_k^{t+1}$ -- the rest is of similar nature. We have:
$$
\nu\_k^{t+1} \in \argmax\_{\nu\in\Delta(K)} \left \\{ \sum\_{i=1}^n\sum\_{k=1}^K \pi\_k^i \log\nu\_k \right\\}\\; .
$$
Re-writing the constraints, and by the [KKT conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions) 
we are looking for $\nu\_k^{t+1}=\nu^\star$ such that $(\nu^\star, \lambda^{\star})$ checks:
$$
\sum\_{i=1}^n\sum\_{k=1}^K \pi\_k^i \log\nu^\star\_k + \lambda^\star\left(\sum\_{k=1}^n \nu^\star\_k - 1\right) = \max\_{\nu\geq 0} \min\_{\lambda}  \sum\_{i=1}^n\sum\_{k=1}^K \pi\_k^i \log\nu\_k + \lambda\left(\sum\_{k=1}^n \nu\_k - 1\right)
$$
By strong duality, this is equivalent to solving:
$$
\min\_{\lambda}   -\lambda +  \max\_{\nu\geq 0} \sum\_{i=1}^n\sum\_{k=1}^K \pi\_k^i \log\nu\_k + \lambda \nu\_k/n
$$
It is easy to check that $\nu\_k^\star$ must check, for every $k\in\\{1, \ldots, K\\}$:
$$
 \nu\_k^{t+1} = -\frac{\sum\_{i=1}^n \pi\_k^i}{n \lambda^\star} \\; .
$$
Replacing we obtain that:
$$
\lambda^\star \in \argmin\_{\lambda\leq 0}\left\\{ -\lambda +  \sum\_{i=1}^n\sum\_{k=1}^K \pi\_k\log(-1/\lambda) \right\\} \\;,
$$
yielding $\lambda^\star = -\frac{1}{n}  \sum\_{i=1}^n\sum\_{k=1}^K \pi\_k^i$. Substituting in $\nu\_k^\star$'s expression
yields the announced claim.
{{% /toggle_block %}}


## Learning a finite Hidden Markov Model

In this section we consider the problem of learning the parameters of a finite HMM.
Let $\mathcal{Z} = \\{1, \ldots, n\\}$ the state space and $\mathcal{X} = \\{1, \ldots, m\\}$ the observation space.
Recall the HMM dynamics; for $t\geq 1$:
$$
\begin{aligned}
x\_t &\sim p\_\nu(\cdot\vert z\_t)\\; ,\\\
z\_{t+1} &\sim p\_\mu(\cdot\vert z\_t) \\;, \\\
\end{aligned}
$$
where $p\_\nu$ describes the state's dynamics and $p\_\mu$ the observation kernel. For simplicity, we assume that
we have access to only one trajectory of observations $(x\_1, \ldots, x\_n)\in\mathcal{X}^n$ to learn from.
Our goal is to learn 
the HMM parameters $\theta = (\mu, \nu)$ jointly. Gradient descent on the maximum likelihood objective:
$$
\mathcal{L}(\theta) = p\_\theta(x\_1, \ldots, x\_n)\\;,
$$
is a valid approach, as gradients can be computed in a _filtering_ fashion -- see
\[[Krishnamurthy §4.2](https://www.cambridge.org/core/books/partially-observed-markov-decision-processes/505ADAE28B3F22D1594F837DEAFF1E0C)\].
However, the EM often is the method of choice
for bootstrapping this optimisation process. Below, we detail both the Expecation and Maximisation step
after $t$ iterations. 

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
The application of the EM procedure to estimate the parameters of a finite HMM with discrete observation space is known under
the name of [Baum-Welch](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) algorithm.
{{% /toggle_block %}}

### Expectation
We will need to compute the conditional distribution $p(z\_k \vert x\_{1:n}, \theta\_t)$ for every $k=1, \ldots, t$. This 
is a typical exercise of fixed-interval smoothing. We provide below the recursive implementation of an optimal fixed-interval
smoother. For details and derivation, we refer the interested reader to
\[[Krishnamurthy §3.3.5](https://www.cambridge.org/core/books/partially-observed-markov-decision-processes/505ADAE28B3F22D1594F837DEAFF1E0C)\].
Let us denote $\pi_{k\vert n}(z) = p(z\_k=z \vert x\_{1:n}, \theta\_t)$. We easily obtain that:
$$
\pi_{k\vert n}(z) = \frac{\pi\_k(z)\beta\_{k\vert n}(z)}{\sum_{z'\in\mathcal{Z}}\pi\_k(z')\beta\_{k\vert n}(z')}\\;.
$$
Above,  $\pi\_k(z) = p(z\_k=z\vert x\_{1:k}, \theta\_t)$ is computed is a forward filter:
$$
\pi\_k(z) = \eta^{-1} p\_{\nu\_t}(x\_k\vert z\_k=z) \sum_{z'\in\mathcal{Z}} p\_{\mu\_t}(z\_k=z \vert z\_{k-1}=z')\pi\_{k-1}(z')
$$
where $\eta$ is a normalisation variable, omitted here to reduce clutter.  Further, 
$\beta\_{k\vert n}(z) = p(x\_{k+1:n}\vert z\_k, \theta\_t)$ is computed via _backward_ recursion:
$$
\beta\_{k\vert n}(z) = \sum\_{z'\in\mathcal{Z}} p\_{\nu\_t}(x\_{k+1}\vert z\_{k+1}=z') 
p\_{\mu\_t}(z\_{k+1}=z'\vert z\_k=z)\beta\_{k+1\vert n}(z')
$$
initialised at $\beta\_{n\vert n}(z)=1$. 
Another smoothed estimate we'll need 
$\pi_{k\vert n}(z, z') := p(z\_k=z, z\_{k-1}=z'\vert x\_{1:n}, \theta\_t)$ is computed similarly. 

With those computations out of the way, we can now materialise the auxiliary log-likelihood:
$$
\mathcal{Q}(\theta\vert \theta\_t) = \sum\_{k=1}^n \sum\_{z\in\mathcal{Z}}\pi\_{k\vert n}(z)\log p\_\nu(x\_{k}\vert z\_{k}=z) + 
\sum\_{k=1}^n \sum\_{z, z'\in\mathcal{Z}} \pi\_{k\vert n}(z, z')\log p\_\mu(z\_{k}=z\vert z\_{k-1}=z')\\;.
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Observe that by applying Bayes rules twice, and using the Markovian structure of the HMM:
$$
\begin{aligned}
p(z\_{1:n}, x\_{1:n} \vert \theta) &= p\_\mu(z\_{n}\vert z\_{n-1}) p\_\nu(x\_n \vert z\_n) p(z\_{1:n-1}, x\_{1:n-1} \vert \theta) \\; ,\\\
&= \prod\_{k=1}^n p\_\mu(z\_{k}\vert z\_{k-1}) p\_\nu(x\_k \vert z\_k) \\;,
\end{aligned}
$$
where we omitted the marginal $p(z\_0)$ as it is assumed known and not relevant for optimisation.
Therefore, the auxiliary log-likelihood will write:
$$
\begin{aligned}
\mathcal{Q}(\theta\vert \theta\_t) &= \mathbb{E}\left[\log p(x\_{1:n}, z\_{1:n} \vert \theta) \middle\vert x\_{1:n}, \theta\_t \right] \\;,\\\
&=  \sum\_{k=1}^n \sum\_{z\in\mathcal{Z}}p\_\nu(z\_k=z \vert x\_{1:n})\log p\_\nu(x\_{k}\vert z\_k=z) + 
\sum\_{k=1}^n \sum\_{z, z'\in\mathcal{Z}} p(z\_k=z, z\_{k-1}=z'\vert x\_{1:n}, \theta\_t)\log p\_\mu(z\_{k}=z\vert z\_{k-1}=z')\\;, \\\
&= \sum\_{k=1}^n \sum\_{z\in\mathcal{Z}}\pi\_{k\vert n}(z)\log p\_\nu(x\_{k}\vert z\_{k}=z) + 
\sum\_{k=1}^n \sum\_{z, z'\in\mathcal{Z}} \pi\_{k\vert n}(z, z')\log p\_\mu(z\_{k}=z\vert z\_{k-1}=z')\\;.
\end{aligned}
$$
{{% /toggle_block %}}


### Maximisation
We are now ready to maximise $\mathcal{Q}(\theta\vert \theta\_t)$. Thanks to the finite nature of the HMM, the 
parametrisation $\nu$ and $\mu$ of the observation likelihood and transitions kernel, respectively, can simply be _stochastic_ matrices: 
$$
\begin{aligned}
\mu \in \big\\{ \mathbf{P} \in \mathbb{R}^{n\times n},  \mathbf{P}\_{ij}\geq 0 \\; \text{ and } \sum\_{j=1}^n  \mathbf{P}\_{ij} = 1 \big\\} \\;, \\\
\nu \in \big\\{ \mathbf{P} \in \mathbb{R}^{n\times m},  \mathbf{P}\_{ij}\geq 0 \\; \text{ and } \sum\_{j=1}^n  \mathbf{P}\_{ij} = 1 \big\\} \\;.
\end{aligned}
$$
For instance, for every $z, z'\in\mathcal{Z}$ and $x\in\mathcal{X}$:
$$
\begin{aligned}
\mu\_{zz'} &= p(z\_{t+1}=z' \vert z\_{t+1}=z) \\; , \\\
\nu\_{zx} &= p(x\_t = x \vert z\_{t}=z) \\; .
\end{aligned}
$$
Rewriting the auxiliary log-likelihood under this convention yields:
$$
\mathcal{Q}(\theta\vert \theta\_t) =  \sum\_{k=1}^n \sum\_{z\in\mathcal{Z}}\pi\_{k\vert n}(z)\log \nu\_{zx\_k}+ 
\sum\_{k=1}^n \sum\_{z, z'\in\mathcal{Z}} \pi\_{k\vert n}(z, z')\log \mu\_{zz'}\\;.
$$
Both parameters $\mu$ and $\nu$ can therefore be updated independently. Solving each program under the stochastic 
matrix constraint will yield, for every $z, z'\in\mathcal{Z}$ and $x\in\mathcal{X}$:
$$
\begin{aligned}
[\mu\_{t+1}]\_{zz'} &= \frac{\sum\_{k=1}^n \pi\_{k\vert n}(z, z')}{\sum\_{k=1}^n \pi\_{k\vert n}(z)}\\; , \\\
[\nu\_{t+1}]\_{xz} &= \frac{\sum\_{k=1}^n \pi\_{k\vert n}(z) \mathbf{1}[x\_k=x]}{\sum\_{k=1}^n \pi\_{k\vert n}(z)}\\; .
\end{aligned}
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof simply relies on satisfying the KKT first-order conditions for each constrained program. It is quite close
to the proof we conducted to update the GMM's mixture parameter with the EM, and hence is left to the reader as an exercice.
{{% /toggle_block %}}

## Resources
The HMM part of this blog-post is highly inspired from \[[Krishnamurthy §4](https://www.cambridge.org/core/books/partially-observed-markov-decision-processes/505ADAE28B3F22D1594F837DEAFF1E0C)\].