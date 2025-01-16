+++
author = "Louis Faury"
title = "Oldie but goodies: Optimal State Estimation"
date = "2024-12-26"
+++

This post is interested in state estimation in HMMs: filtering, prediction and smoothing.
We will introduce state estimation as the solution of an optimisation problem,
and prove the celebrated recursive updates for each inference use-case.
A special attention will be given to HMM filters (and how they easily generalise to the celebrated Kalman filters).

<!--more-->

{{< infoblock>}}
$\quad$ The reader interested about filtering in POMDPs can directly jump to in <a href="../pomdp" style="text-decoration:none; color:#0074aa;" ">this post</a>.
{{< /infoblock >}}

## Hidden Markov Models

<br>
{{< image src="/hmm.png" width="460px" align="center" caption="Fig1. The graphical model for the first steps of an HMM.">}}
<br>

We study Hidden Markov Models (HMMs): partially observable processes formalised by a tuple
$(\mathcal{X}, \mathcal{Y}, p, q)$. 
$\mathcal{X}$ is the state-space and 
$\mathcal{Y}$ the observation space.
The distribution $p$ materialises the stochastic process for the state $x\_t\in\mathcal{X}$,
which be observed via observations $y\_t\in\mathcal{Y}$.
$$
\left\\{\begin{aligned}
x\_{t+1} &\sim p(\cdot \vert x\_t) \\;, \\\
y\_{t+1} &\sim q(\cdot \vert x\_{t+1}) \\;.
\end{aligned}\right.
$$

HMMs come with strong Markovian structure (see Fig. 1).
In particular, we have the following identities:
$$
\begin{aligned}
    p(x\_t \\, \vert \\, x\_{1}, \ldots x\_{t-1}) &= p(x\_t \\, \vert \\, x\_{t-1}) \\;,\\\
    p(y\_t \\, \vert \\, x\_{1}, \ldots x\_{t-1}) &= p(y\_t \\,\vert \\, x\_t)\\; .
\end{aligned}
$$

Given the finite nature of the problem,
it can be useful to adopt vectorial notations. 
Let $n=\vert \mathcal{X}\vert$ and $m=\vert \mathcal{Y}\vert$.
Below, bold notations refer to matrices and vectors.
For instance, if $\pi$ is a distribution over $\mathcal{X}$ we denote
$\boldsymbol{\pi} := (\pi\_{x\_1}, \ldots, \pi\_{x\_n})^\top\in\mathbb{R}^{n}$.
Similarly, we will use the matrix notation $\mathbf{P}\in\mathbb{R}^{n\times n}$ and 
$\mathbf{Q}\in\mathbb{R}^{n\times m}$ for:
$$
\begin{aligned}
[\mathbf{P}]\_{xx^\prime} = p(x^\prime\vert x)\\;,\\\
[\mathbf{Q}]\_{xy} = q(y\vert x)\\;.
\end{aligned}
$$

{{< infoblock>}}
$\quad$ We study HMMs here for the sake of simplicity.
Treating processes living in continuous spaces simply requires swapping sums with integrals.
{{< /infoblock >}}

## Optimality
We are concerned with _state estimation_: building estimate $\hat{x}\_{t\vert n}$
of the state $x_t$, given a set of 
contiguous observations $\\{y\_{1:n}\\}$. 
The actual value of $n$ allows us to separate several uses cases:
1. $t=n$ is *filtering*: producing an estimate of the current state given all observations collected so far,
2. $t>n$ is predicting: estimating the future value of the state given all observations collected so far,
3. $t<n$ is smoothing: estimating the current value of the state given all observations (future and past).

We are interested in estimators minimising a squared error criterion.
In the rest of this section, we will make this explicit for filtering.
It naturally extends to the other set-up.
In optimal filtering, we compute:
$$
\tag{1}
\hat{x}\_{t\vert t} \in \argmin\_{x} \mathbb{E}\left[(x-x\_t)^2 \\, \middle\vert\\, y\_{1:t}\right]\\; .
$$

{{% toggle_block background-color="#f2b1ac" title="Measure theoretic nitpick" default-display="none"%}}
(1) is a comprehensible but quite imprecise (at least) definition, from a measure theoretic view-point.
We should define the filtration $\mathcal{F}\_t=\sigma(y\_1, \ldots, y\_t)$ and the space 
$\mathcal{Z}_t$ of $\mathcal{F}\_t$-measurable functions. Then:
$$
\hat{x}\_{t\vert t} \in \argmin\_{z\in\mathcal{Z}\_t} \mathbb{E}\left[(x-x\_t)^2 \\, \middle\vert\\, \mathcal{F}\_t\right]\\; .
$$
We will not bother ourselves too much with such formalism in the rest of the post.
{{% /toggle_block %}} 

Such an estimator has the good taste of admitting a closed-form:
$$
\begin{aligned}
\tag{2}
\hat{x}\_{t\vert t} &= \mathbb{E}\left[x\_t \\,\middle\vert\\, y\_{1:t}\right]\\;.\\\
&= \sum\_{x\in\mathcal{X}} x \cdot p(x\_t=x\vert y\_{1:t})\\; .
\end{aligned}
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof is immediate;
$$
\begin{aligned}
\mathbb{E}\left[(x-x\_t)^2 \\, \middle\vert\\, y\_{1:t}\right] &= x^2 - 2x \mathbb{E}\left[x\_t \\,\middle\vert\\, y\_{1:t}\right] + \square
\end{aligned}
$$
where $\square$ is a constant. The minimum of this quadratic is attained at $\mathbb{E}\left[x\_t \\,\middle\vert\\, y\_{1:t}\right]$.

<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

Thanks to the smoothing property of expectation, it is also unbiased.
Further, it is quite clear that to compute this estimator, 
one should first compute the conditional $p(\cdot\vert y\_{1:t})$.


#### Bregman-divergences
The $\ell_2$-norm is sometimes a clumsy way to measure distance between states.
For instance, one usually prefers distributional distances, e.g. the Kullback-Leibler 
divergence if the state lives in a simplex.
The good news is that the estimator from (2) is also the solution of the following program,
where the discrepancy to $x\_t$ is measured by _any_ Bregman divergence!
Formally, for any differentiable and convex function $f:\mathcal{X}\mapsto\mathbb{R}$, denote $D\_f : \mathcal{X}\times\mathcal{X}\mapsto\mathbb{R}$
the associated Bregman divergence:
$$
D\_f(x\\, \| \\, x^\prime) := f(x) - f(x^\prime) - f^\prime(x^\prime)(x-x^\prime)\\;.
$$
Then, we have that $\hat{x}\_{t\vert t}$ is also the solution to:
$$
\hat{x}\_{t\vert t} \in \argmin\_{x} \mathbb{E}\left[D\_f(x\_t\\, \| \\, x) \middle\vert y\_{1:t}\right]\\;.
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}

Let $z\_t$ be any other estimator based on $y\_{1:t}$. Then:
$$
\begin{aligned}
\mathbb{E}\left[D\_f(x\_t\\, \| \\, z\_t)  - D\_f(x\_t\\, \| \\, \hat{x}\_{t\vert t}) \middle\vert y\_{1:t}\right] 
&= \mathbb{E}\left[f(\hat{x}\_{t\vert t}) - f(z\_t) + f^\prime(\hat{x}\_{t\vert t})(x\_t - \hat{x}\_{t\vert t}) - f^\prime(z\_t)(z\_t - \hat{x}\_{t\vert t})
\middle\vert y\_{1:t}\right] \\;, \\\
&= f(\hat{x}\_{t\vert t}) - f(z\_t) + f^\prime(\hat{x}\_{t\vert t})(\mathbb{E}\left[x\_t\middle\vert y\_{1:t}\right] - \hat{x}\_{t\vert t}) - f^\prime(z\_t)(\mathbb{E}\left[x\_t\middle\vert y\_{1:t}\right] - z\_t)\\;,\\\
&= f(\hat{x}\_{t\vert t}) - f(z\_t) + f^\prime(\hat{x}\_{t\vert t})(\hat{x}\_{t\vert t} - \hat{x}\_{t\vert t}) - f^\prime(z\_t)(\hat{x}\_{t\vert t}- z\_t)\\;,\\\
&= D\_f(\hat{x}\_{t\vert t} \\, \| \\, z\_t)\\;,\\\
&\geq 0\\;,
\end{aligned}
$$
where the last line uses the convexity of $f$, and concludes the proof.
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}


## Filtering
We saw in the previous section that the filtering problem boiled down to computing the conditional 
$$
\pi\_{t}(\cdot) := p(x\_t=\cdot\vert y\_{1:t})\\;.
$$
Sounds tedious. But we are in luck: it actually follows a nice recursive structure. 

{{< boxed title="Recursive updates" >}}
$\qquad \qquad \qquad\qquad\; \text{ For any }x\in\mathcal{X}$:
$$
\tag{3}
\pi_{t}(x) \propto q(y_t\vert x)\sum_{x^\prime\in \mathcal{X}} p(x\vert x^\prime)\pi_{t-1}(x^\prime)\; . 
$$
{{< /boxed >}}




{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
For any $x\in\mathcal{X}$:
$$
\begin{aligned}
p(x\_t=x\vert y\_{1:t}) &\propto p(x\_t =x, y\_t \vert y\_{1:t-1})\\;, &(\text{Bayes rule})\\\
&= p(y\_t \vert y\_{1:t-1}, x\_t =x)p(x\_t =x\vert y\_{1:t-1})\\;, \\\
&= p(y\_t \vert x\_t =x)p(x\_t =x\vert y\_{1:t-1})\\;, \\\
&= p(y\_t \vert x\_t =x) \sum\_{x^\prime\in\mathcal{X}} p(x\_t =x, x\_{t-1}=x^\prime\vert y\_{1:t-1})\\;,&(\text{total probability})\\\
&= p(y\_t \vert x\_t =x) \sum\_{x^\prime\in\mathcal{X}} p(x\_t =x\vert x\_{t-1}=x^\prime, y\_{1:t-1})p(x\_{t-1}=x^\prime\vert y\_{1:t-1})\\;,\\\
&= p(y\_t \vert x\_t =x) \sum\_{x^\prime\in\mathcal{X}} p(x\_t =x\vert x\_{t-1}=x^\prime)p(x\_{t-1}=x^\prime\vert y\_{1:t-1})\\;,\\\
&= q(y\_t \vert x)\sum\_{x^\prime\in\mathcal{X}}p(x\vert x^\prime)\pi\_{t-1}(x^\prime)\\;.
\end{aligned}
$$
Hence, we have:
$$
\pi\_t(x) = \eta(y\_{t})q(y\_t\vert x)\sum\_{x^\prime\in \mathcal{X}} p(x\vert x^\prime)\pi\_{t-1}(x^\prime)
$$
where $\eta(y\_{t})$ is a normalisation constant:
$$
1/\eta(y\_{t}) = \sum\_{x\in\mathcal{X}}q(y\_t\vert x)\sum\_{x^\prime\in \mathcal{X}} p(x\vert x^\prime)\pi\_{t-1}(x^\prime)\\;.
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

(3) can be written in vectorial form:
$$
\boldsymbol{\pi}\_t \propto \text{diag}(\mathbf{Q}\_{y\_t})\mathbf{P}^\top\boldsymbol{\pi}\_{t-1}\\; ,
$$
where $\mathbf{Q}\_{y\_t}$ is the $y\_t$ line of $\mathbf{Q}$. Concretely, 
this means that one can compute $\pi\_t$ recursively, 
by incorporating the observations $\\{y\_{1:t}\\}\_t$ one at a time.
The update rule (3) is sometimes decomposed into two steps:
1. _Prediction_ computes $\pi\_{t\vert t-1}$, the state distribution after the HMM steps before the observation is emitted:
$$
\pi\_{t\vert t-1}(x) = \sum\_{x^\prime\in \mathcal{X}} p(x\vert x^\prime)\pi\_{t-1}(x^\prime)\\; .
$$
2. _Measurement_ incorporates the knowledge of $y\_t$:
$$
\tag{4}
\pi\_t(x) \propto p(y\_t\vert x) \pi\_{t\vert t-1}(x)\\; .
$$

#### Beyond HMMs
The ideas presented above are easily generalised to _continuous_ states and observations spaces.
Representing the different probability measures by their densities, one can write:
$$
\pi\_t(x) \propto p(y\_t\vert x\_t) \int\_{\mathbb{R}} p(x\vert x^\prime)\pi\_{t-1}(x^\prime) dx^\prime\\; .
$$
The usual difficulty arises when this update step does not admit a closed form (which is basically almost always), and the resulting
$\pi\_t$ does not have to belong to a parametric distribution (e.g. Gaussian).
The well-known exception to this rule arise when $\pi\_{t-1}$ is a normal distribution, and both the 
transition and emission kernels are also normal. Then (4) admits an explicit form and $\pi\_t$ is also Gaussian.
This is the setting of the celebrated Kalman filter—which is often presented via the prediction and measurement 
framework we discussed above.


## Prediction
This will be a short section: the main steps for completing predictions
were already covered in the previous section. 
For filtering, the prediction step computes:
$$
\pi\_{t+1\vert t}(x) = \sum\_{x^\prime\in \mathcal{X}} p(x\vert x^\prime)\pi\_{t}(x^\prime)\\; .
$$
This can be iterated a few times to compute $\pi\_{t+k\vert t}$.

## Smoothing
Smoothing actually covers several concrete use cases (fixed-point, fixed-lag, fixed-interval).
Below, we are interested in the fixed-interval setting.
Let $n$ be fixed; we wish to estimate $\pi\_{t\vert n}$ for every $t=1, \ldots, n$ given $y\_{1:n}$.
Concretely, we wish to compute all the conditionals $p(x\_t\vert y\_{1:n})$. Observe that:
$$
\begin{aligned}
    p(x\_t\vert y\_{1:n}) &\propto p(x\_t, y\_{t+1:n} \vert y\_{1:t})\\;, &(\text{Bayes rule}) \\\
    &= p(y\_{t+1:n}\vert x\_t, y\_{1:t}) p(x\_t\vert y\_{1:t})\\;,\\\
    &= p(y\_{t+1:n}\vert x\_t)p(x\_t\vert y\_{1:t})\\;,
\end{aligned}
$$
which, by denoting $\gamma\_{t\vert n}(x) = p(y\_{t+1:n}\vert x\_t=x)$ we will write:

{{< boxed title="Smoothing posterior" >}}
$\qquad \qquad \qquad\qquad\quad\; \text{ For any }x\in\mathcal{X}$:
$$
\tag{5}
\pi_{t\vert n}(x) = \gamma_{t\vert n}(x) \pi_{t}(x)\; .
$$
{{< /boxed >}}


Another good news: $\gamma\_{t\vert n}$ also checks a recursive update rule.For any $x\in\mathcal{X}$:
$$
\gamma\_{t\vert n}(x) = \sum\_{x^\prime\in\mathcal{X}} q(y\_{t+1}\vert x^\prime)p(x^\prime\vert x)\gamma\_{t+1\vert n}(x^\prime)\\;.
$$
It can therefore be efficiently computed via backward recursion in time, starting from $\gamma\_{n\vert n}\equiv 1$.
In vectorial notations, it writes:
$$
\boldsymbol{\gamma}\_{t\vert n} = \mathbf{P}\text{diag}({\mathbf{B}}\_{y\_{t+1}})\boldsymbol{\gamma}\_{t+1\vert n}\\;.
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Observe that:
$$
\begin{aligned}
    \gamma\_{t\vert n}(x) &= p(y\_{t+1:n}\vert x\_t = x) \\;, \\\
    &= \sum\_{x^\prime\in\mathcal{X}}  p(y\_{t+1:n}, x\_{t+1}=x^\prime \vert x\_t = x) \\;, &(\text{total probability})\\\
    &= \sum\_{x^\prime\in\mathcal{X}}  p(y\_{t+1:n}\vert x\_{t+1}=x^\prime, x\_t = x) p(x^\prime\vert x)\\;, \\\
    &= \sum\_{x^\prime\in\mathcal{X}}  p(y\_{t+1:n}\vert x\_{t+1}=x^\prime) p(x^\prime\vert x)\\;, \\\
    &= \sum\_{x^\prime\in\mathcal{X}}  p(y\_{t+2:n}\vert y\_{t+1}, x\_{t+1}=x^\prime) q(y\_{t+1}\vert x\_{t+1})p(x^\prime\vert x)\\;, \\\
    &= \sum\_{x^\prime\in\mathcal{X}}  p(y\_{t+2:n}\vert x\_{t+1}=x^\prime) q(y\_{t+1}\vert x\_{t+1})p(x^\prime\vert x)\\;, \\\
\end{aligned}
$$
which concludes the proof.
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

Going back to smoothing, we now understand it can be achieved by a forward / backward algorithm. 
A first forward loop returns the filtering estimates $\\{\pi\_t\\}\_t$, while a backward loop
sets $\\{\gamma\_t\\}\_t$. The smoothing posterior is then simply obtained by multiplying both–see (5).

<br>
{{< image src="/fb.png" width="480px" align="center" caption="Fig2. Illustration the forward / backward loops for fixed-interval smoothing.">}}
<br>