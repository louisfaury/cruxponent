+++
author = "Louis Faury"
title = "Information Theory Cheat-Sheet"
date = "2025-10-13"
+++
Entropy, divergence, mutual information, etc. are central concepts in statistical machine learning.
This post ties them together in a short collection of elementary information theoretic results.

<!--more-->
Below, we consider random variables $\mathrm{X}, \mathrm{Y}$ that take values in some discrete sets 
$\mathcal{X}$ and $\mathcal{Y}$. 
We denote, respectively, $p\_{\tiny\mathrm{X}}\in\Delta\_\mathcal{X}$ and $p\_{\tiny\mathrm{Y}}\in\Delta\_\mathcal{Y}$ their respective probability
mass function. Their joint pmf is denoted $p\_{\tiny\mathrm{XY}}$.

{{< infoblock>}}
$\quad$ Discrete support is for simplicity: one can replace sums with integrals in the
continuous case.
{{< /infoblock >}}



## Entropy

The information entropy of $\mathrm{X}$ measures the amount of unpredictability of this random variable.
The surprise of the event $\\{X=x\\}$ can be measured via $-\log p\_{\tiny\mathrm{X}}(x)$: the lower this event's probability, the
higher the surprise when it actually happens. The entropy is the expected value of this score:
$$
\tag{1}
\mathcal{H}(\mathrm{X}) := \mathbb{E}\_{p\_{\tiny\mathrm{X}}}\left[-\log p\_{\tiny\mathrm{X}}(X)\right\] = - \sum\_{x\in\mathcal{X}} p\_{\tiny\mathrm{X}}(x)\log p\_{\tiny\mathrm{X}}(x)\\;.
$$
The entropy is maximum when $p$ is uniform over $\mathcal{X}$, and in the discrete case, it is also positive.
The entropy is measured in _nats_ when using the natural logarithm (or in bits, when the basis of the logarithm is 2).

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
We drop subcripts to reduce clutter. Let's start with positivity, which is a consequence of Jensen's inequality.
$$
\tag{a}
\begin{aligned}
    \mathcal{H}(\mathrm{X}) &= - \sum\_{x\in\mathcal{X}} p(x)\log p(x)\\;, \\\
&\geq - \log \sum\_{x\in\mathcal{X}} p(x)^2\\;, &(\log \text{concavity}) \\\
&\geq - \log \sum\_{x\in\mathcal{X}} p(x)\\;, &(p(x)\leq 1) \\\
&\geq - \log 1 = 0 \\\
\end{aligned}
$$
Let's move on to prove that the entropy's maximum value is reached for uniform distribution.
We are interested in the following program, where we denote $p\_x=p(x)$ for brevity:
$$
\max\_{p\in\Delta\_\mathcal{X}} -\sum\_{x}p\_x \log p\_x \\;,
$$

whose Lagrangian is, for any $\lambda\in\mathbb{R}$ and $p\in\Delta\_\mathcal{X}$:
$$
\mathcal{L}(\lambda, p) := -\sum\_{x} p\_x \log p\_x + \lambda \big(\sum\_{x}p\_x - 1\big)\\;.
$$
(The positivity constraint is ignored here; the solution we will arrive to is naturally primally feasible.)
Observe that $\mathcal{H}(\mathrm{X})$ is a concave function of $p\_{\tiny\mathrm{X}}$ and 
$\Delta\_\mathcal{X}$ is a convex set; by the KKT conditions, the solution $p^\star$ to our original program (a) checks:
$$
\frac{\partial\mathcal{L}}{\partial\mathcal{p}}(\lambda^\star, p^\star) = 0 \text{ and } \frac{\partial\mathcal{L}}{\partial\lambda}(\lambda^\star, p^\star) = 0\\;.
$$
Solving this system easily yields that $p\_x = 1/\vert\mathcal{X}\vert$ for all $x\in\mathcal{X}$, which proves the claim.

<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

{{% toggle_block background-color="#CBE4FE" title="On positivity" default-display="none"%}}
The fact that the entropy is positive is only true for discrete random variables. The proof above 'breaks' when it uses $p(x)\leq 1$, which no longer holds true when $p$
is a density.
{{% /toggle_block %}}

## Conditional entropy
One can define the _joint_ entropy $\mathcal{H}(\mathrm{X},\mathrm{Y})$ of $\mathrm{X}$ and $\mathrm{Y}$ by 
measuring the entropy of the joint $p\_{\tiny\mathrm{X}\mathrm{Y}}$.
The _conditional_ entropy of $\mathrm{X}$ wrt $\mathrm{Y}$ is the expected value of
$\mathrm{X}$'s entropy conditionally on the realisation of $\mathrm{Y}$:
$$
\tag{2}
\mathcal{H}(\mathrm{X}\vert \mathrm{Y}) := -\sum\_{y} p\_{\tiny\mathrm{Y}}(y) \sum\_{x} p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x \vert y)\log p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x \vert y)\\;.
$$
There is a chain rule for the entropy, in the sense that $\mathcal{H}(\mathrm{X},\mathrm{Y}) = \mathcal{H}(\mathrm{X}\vert\mathrm{Y}) + \mathcal{H}(\mathrm{Y})$.
When $\mathrm{X}$ and $\mathrm{Y}$ are independent, the joint entropy is therefore the sum of the marginal entropies, or
$\mathcal{H}(\mathrm{X},\mathrm{Y}) = \mathcal{H}(\mathrm{X}) + \mathcal{H}(\mathrm{Y})$.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Let's prove the chain rule:
$$
\begin{aligned}
\mathcal{H}(\mathrm{X},\mathrm{Y}) &= \sum\_{x, y} p\_{\tiny\mathrm{X}\mathrm{Y}}(x, y)\log p\_{\tiny\mathrm{X}\mathrm{Y}}(x, y)\\;,\\\
&= -\sum\_{x, y} p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y)p\_{\tiny\mathrm{Y}}(y)\log\big(p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y)p\_{\tiny\mathrm{Y}}(y)\big)\\;, &(\text{Bayes rule})\\\
&=  -\sum\_{x, y} p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y)p\_{\tiny\mathrm{Y}}(y)\log p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y) - \sum\_{x, y} p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y)p\_{\tiny\mathrm{Y}}(y)\log p\_{\tiny\mathrm{Y}}(y)\\;,\\\
&=  -\sum\_{x, y} p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y)p\_{\tiny\mathrm{Y}}(y)\log p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y) - \sum\_{ y} p\_{\tiny\mathrm{Y}}(y)\log p\_{\tiny\mathrm{Y}}(y)\sum\_{x}p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y)\\;,\\\
&=  -\sum\_{x, y} p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y)p\_{\tiny\mathrm{Y}}(y)\log p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y) - \sum\_{ y} p\_{\tiny\mathrm{Y}}(y)\log p\_{\tiny\mathrm{Y}}(y)\\;,\\\
&= \mathcal{H}(\mathrm{X}\vert \mathrm{Y}) + \mathcal{H}(\mathrm{Y})\\;.
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

## Cross entropy
When $\mathcal{X}=\mathcal{Y}$, one can define the _cross_ entropy of $\mathrm{Y}$ relative to
$\mathrm{X}$ as
$
\mathcal{C}(\mathrm{X}, \mathrm{Y}) := -\sum\_{x} p\_{\tiny\mathrm{X}}(x)\log p\_{\tiny\mathrm{Y}}(x) \\;.
$
It is a standard result known as Gibb's inequality that it is lower-bounded by the entropy of $\mathrm{X}$:
$$
\mathcal{C}(\mathrm{X}, \mathrm{Y}) \geq \mathcal{H}(\mathrm{X}) \\; .\tag{3}
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
$$
\begin{aligned}
\mathcal{C}(\mathrm{X}, \mathrm{Y}) - \mathcal{H}(\mathrm{X}) &= -\sum\_{x} p\_{\tiny\mathrm{X}}(x)\log \frac{p\_{\tiny\mathrm{Y}}(x)}{p\_{\tiny\mathrm{X}}(x)}\\;, \\\
&\geq \sum\_{x} p\_{\tiny\mathrm{X}}(x)\big(1 - \frac{p\_{\tiny\mathrm{Y}}(x)}{p\_{\tiny\mathrm{X}}(x)}\big) \\;, &(\log(x)\leq x-1 \text{ for } x>0)\\\
&= \sum\_{x} p\_{\tiny\mathrm{X}}(x) - \sum\_{x} p\_{\tiny\mathrm{Y}}(x) = 1-1 = 0\\;.
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

## Relative entropy
In this section we assume that $\mathcal{X}=\mathcal{Y}$.
The relative entropy of $\mathrm{X}$ wrt $\mathrm{Y}$,
also known as the Kullback-Leibler divergence of $\mathrm{X}$ from $\mathrm{Y}$, is defined as the difference between the 
cross-entropy and the entropy:
$$
\text{KL}(p\_{\tiny\mathrm{X}} \\|\\, p\_{\tiny\mathrm{Y}}) := \mathcal{C}(\mathrm{X}, \mathrm{Y}) - \mathcal{H}(\mathrm{X})\\;.
$$
A direct consequence of (3) is that the relative entropy is positive. It reaches zero if and only if 
$p\_{\tiny \mathrm{X}} =p\_{\tiny \mathrm{Y}}$. 
It is only a divergence; it is not symmetric and does not satisfy the triangle inequality.
It can, however, be lower-bounded by the total variation metric (that's called Pinsker inequality, not proven here):
$$
\text{TV}(\mathrm{X}, \mathrm{Y}) \leq \sqrt{\text{KL}(\mathrm{X} \\| \mathrm{Y})/2}\\;.
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Finding counter-examples for the triangle inequality is left as an exercise.
To prove that the relative entropy is zero iff the two distributions are equal, it is
enough to reverse the proof of (3), and notice that the inequality 
$\log(x) \leq x -1$ is an equality iff $x=1$.
{{% /toggle_block %}}
{{% toggle_block background-color="#CBE4FE" title="Tighter bounds" default-display="none"%}}
Since the total variation is bounded by 1, Pinsker's inequality is hopelessly loose when
$\text{KL}(\mathrm{X} \\| \mathrm{Y})>2$.
In this case, one should resort to [more powerful](https://en.wikipedia.org/wiki/Bretagnolle%E2%80%93Huber_inequality) bounds.
{{% /toggle_block %}}


## Mutual information
The mutual information $\mathcal{I}$ between $\mathrm{X}$ and $\mathrm{Y}$ is defined by the 
difference between the entropy and conditional entropy, in an effort to measure the 
mutual dependencies between the random variables:
$$
\mathcal{I}(\mathrm{X}, \mathrm{Y}) := \mathcal{H}(\mathrm{X}) - \mathcal{H}(\mathrm{X}\vert \mathrm{Y})\\;.
$$
The definition is of course symmetric (the position of each variable can be swapped).
A useful identity to see how this measures the information about one variable contained in the other 
involves the relative entropy between the joint distribution and the product of marginals. Slightly abusing notations:
$$
\tag{4}
\mathcal{I}(\mathrm{X}, \mathrm{Y}) = \text{KL}(p\_{\tiny\mathrm{XY}}\\|\\,p\_{\tiny\mathrm{X}}\cdot p\_{\tiny\mathrm{Y}})\\;.
$$
Observe that this proves the mutual information is non-negative.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
$$
\begin{aligned}
\mathcal{I}(\mathrm{X}, \mathrm{Y}) &= \mathcal{H}(\mathrm{X}) - \mathcal{H}(\mathrm{X}\vert \mathrm{Y}) \\;, \\\
&= -\sum\_{x} p\_{\tiny\mathrm{X}}(x)\log p\_{\tiny\mathrm{X}}(x) + \sum\_{y}p\_{\tiny\mathrm{Y}}(y) \sum\_{x} p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y) \log p\_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y)\\;\\\
&= -\sum\_{x, ,y} p\_{\tiny\mathrm{X}, \mathrm{Y}}(x, y)\log p\_{\tiny\mathrm{X}}(x) + \sum\_{y}p\_{\tiny\mathrm{Y}}(y) \sum\_{x} p\_{\tiny\mathrm{X}\vert y}(x\vert y) \log p \_{\tiny\mathrm{X}\vert y}(x\vert y)\\;&(\text{total probability})\\\
&= -\sum\_{x, ,y} p\_{\tiny\mathrm{X}, \mathrm{Y}}(x, y)\log p\_{\tiny\mathrm{X}}(x) + \sum\_{x, y} p\_{\tiny\mathrm{X}, \mathrm{Y}}(x, y) \log p \_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y)\\;,\\\
&= \sum\_{x, ,y} p\_{\tiny\mathrm{X}, \mathrm{Y}}(x, y)\log\frac{p \_{\tiny\mathrm{X}\vert \mathrm{Y}}(x\vert y)}{p\_{\tiny\mathrm{X}}(x)}\\;,\\\
&= \sum\_{x, ,y} p\_{\tiny\mathrm{X}, \mathrm{Y}}(x, y)\log\frac{p \_{\tiny\mathrm{X}, \mathrm{Y}}(x, y)}{p\_{\tiny\mathrm{X}}(x)p\_{\tiny\mathrm{Y}}(y)}= \text{KL}(p\_{\tiny\mathrm{XY}}\\|\\,p\_{\tiny\mathrm{X}}\cdot p\_{\tiny\mathrm{Y}})\\;.
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}


## Conditional Mutual Information

The goal of this section is to prove the _data processing inequality_, a result 
that embodies that no information can be gained via processing. 
Before that, we need to introduce the notion of _conditional_ mutual information.
Let $\mathrm{Z}$ be some random variable in $\mathcal{Z}$. 
The conditional mutual information between $\mathrm{X}$ and $\mathrm{Y}$ with respect to $\mathrm{Z}$
is defined as the expected mutual information between $\mathrm{X}$ and $\mathrm{Y}$ given 
realisations of $\mathrm{Z}$. 
Concretely:
$$
\mathcal{I}(\mathrm{X}, \mathrm{Y} \vert \mathrm{Z}) := \mathcal{H}(\mathrm{X}\vert \mathrm{Z}) - \mathcal{H}(\mathrm{X}\vert \mathrm{Y}, \mathrm{Z})\\;.
$$

We can give an equivalent but more intuitive definition that follows (4) via 
$
\mathcal{I}(\mathrm{X}, \mathrm{Y} \vert \mathrm{Z}) = \text{KL}(p\_{\tiny\mathrm{XY\vert Z}}\\|\\,p\_{\tiny\mathrm{X\vert Z}}\cdot p\_{\tiny\mathrm{Y\vert Z}})\\;.
$
The conditional mutual information is involved in an identity called the chain rule for mutual information:
$$
\tag{5}
\mathcal{I}(\mathrm{X}, (\mathrm{Y}, \mathrm{Z})) = \mathcal{I}(\mathrm{X}, \mathrm{Z}) + \mathcal{I}(\mathrm{X},  \mathrm{Y}\vert \mathrm{Z})\\;. 
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
$$
\begin{aligned}
\mathcal{I}(\mathrm{X}, (\mathrm{Y}, \mathrm{Z})) &= \mathcal{H}(\mathrm{X}) -  \mathcal{H}(\mathrm{X}\vert \mathrm{Y}, \mathrm{Z})\\;,\\\
&= \mathcal{I}(\mathrm{X}, \mathrm{Z}) + \mathcal{H}(\mathrm{X}\vert \mathrm{Z})-  \mathcal{H}(\mathrm{X}\vert \mathrm{Y}, \mathrm{Z})\\;, \\\
&=  \mathcal{I}(\mathrm{X}, \mathrm{Z}) + \mathcal{I}(\mathrm{X},  \mathrm{Y}\vert \mathrm{Z})\\;.
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

We are now ready to state and prove the data-processing inequality.


{{< boxed title="Data-processing inequality" >}}
$\qquad\qquad\qquad\qquad\qquad\qquad\; \text{Let } \mathrm{X}, \mathrm{Y} \text{ and } \mathrm{Z} \text{ form the Markov Chain } \mathrm{X}\rightarrow \mathrm{Y}\rightarrow \mathrm{Z}. \text{ Then:}$
$$
\mathcal{I}(\mathrm{X}, \mathrm{Y}) \geq \mathcal{I}(\mathrm{X}, \mathrm{Z})\;.
$$
{{< /boxed >}}

The fact that $ \mathrm{X}\rightarrow \mathrm{Y}\rightarrow \mathrm{Z}$ means that $\mathrm{Z}$ is conditionally independent of $\mathrm{X}$ given $\mathrm{Y}$.
This captures settings where one would like to add some post-processing power over some signal $\mathrm{Y}$
encoded from $\mathrm{X}$. The data-processing inequality establishes that no additional signal can be created out of thin air;
in other words, post-processing can only decrease the mutual information with $\mathrm{X}$.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
This is proven using the chain-rule for mutual information. Indeed, from (5) we have that:
$$
\begin{aligned}
\mathcal{I}(\mathrm{X}, (\mathrm{Y}, \mathrm{Z})) &= \mathcal{I}(\mathrm{X}, \mathrm{Z}) + \mathcal{I}(\mathrm{X},  \mathrm{Y}\vert \mathrm{Z})\\;,\\\
&= \mathcal{I}(\mathrm{X}, \mathrm{Y}) + \mathcal{I}(\mathrm{X},  \mathrm{Z}\vert \mathrm{Y})\\;.
\end{aligned}
$$
As a result we have:
$$
\begin{aligned}
    \mathcal{I}(\mathrm{X}, \mathrm{Z}) &= \mathcal{I}(\mathrm{X}, \mathrm{Y}) + \mathcal{I}(\mathrm{X},  \mathrm{Z}\vert \mathrm{Y})- \mathcal{I}(\mathrm{X},  \mathrm{Y}\vert \mathrm{Z})\\;,\\\
&= \mathcal{I}(\mathrm{X}, \mathrm{Y}) - \mathcal{I}(\mathrm{X},  \mathrm{Y}\vert \mathrm{Z})\\;, \\\
&\leq \mathcal{I}(\mathrm{X}, \mathrm{Y})\\;,
\end{aligned}
$$
where we sequentially used the conditional structure of the problem to claim that $\mathcal{I}(\mathrm{X},  \mathrm{Z}\vert \mathrm{Y})=0$, and the
non-negativity of mutual information.

<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}
