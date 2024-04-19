+++
author = "Louis Faury"
title = "Oldies but goodies: Ergodic Theorem"
date = "2024-02-11"
+++

The goal of this blog post is to provide a self-contained proof of the so-called fundamental theorem
of Markov chains.
It states that provided some _basic_ properties (spoiler: ergodicity) any Markov chain
random walk converges to the same limiting distribution—irrespective of the starting state. 
A hidden goal is to introduce some important tools
for the analysis of the average-reward criterion in MDPs.
<!--more-->

{{< infoblock>}}
$\quad$ In this post we abundantly mix functional and vectorial notations. 
Given some finite set $\mathcal{X}$, any real-valued function $f$ on $\mathcal{X}$ will be associated
a vector $\mathbf{f}\in\mathbb{R}^{\vert \mathcal{X}\vert}$ where $\mathbf{f}_x = f(x)$ for $x\in\mathcal{X}$.
{{< /infoblock >}}


## Warm-up <a name="warmup"></a>
We will now go through some useful definitions, tools and identities we will need on our journey. 
First, we should once again properly state said journey's goal, which is to prove that:
$$
\textit{any ergodic Markov chain converges to its unique stationary distribution, no matter where it started.}
$$
The actual meaning of this statement will hopefully become clearer and clearer through this post.

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
Despite being a tremondously interesting topic, we will restrict our attention to the asymptotic behaviour, 
and will completely leave aside the rate of convergence to said behaviour.
{{% /toggle_block %}}

### Notations
Let $\mathcal{X} = \\{ 1, \ldots, n\\}$ with $n<\infty$. 
We are interested in random walks $\\{ X\_t \\}_t$ over $\mathcal{X}$ that satisfy the strong Markov property.
Formally, this requires that for any $(x\_1, \ldots, x\_{t+1})\in\mathcal{X}^{t+1}$:
$$
\begin{aligned}
\mathbb{P}(X\_{t+1}=x\_{t+1} \vert X\_{1:t} = x\_{1:t}) &= \mathbb{P}(X\_{t+1}=x\_{t+1} \vert X\_{t} = x\_{t})\\;, \\\
&=: p\_t(x\_{t+1} \vert x\_t) \\; .
\end{aligned}
$$
Such a process is _memory-less_: future values, conditionally on the whole history,
 depend only on the last value. 
We also restrict our attention to _time-homogeneous_ processes, such that $p\_t = p$ is independent of the time $t$. 
The function $p: \mathcal{X} \mapsto \Delta\_\mathcal{X}$ is called a transition kernel:
$$
p(\cdot\vert x) = \mathbb{P}(X\_{t+1}=\cdot \vert X\_t=x)\\;.
$$
It is often represented via a matrix $\mathbf{P}\in\mathbb{R}^{n\times n}$ where given $x, y\in\mathcal{X}$, one defines
$
\mathbf{P}\_{xy} := p(y\vert x).
$
Together with some initial distribution $\nu\in\Delta\_\mathcal{X}$,
it fully defines a Markov chain over $\mathcal{X}$. 

{{< boxed title="Markov chain" >}}
$$
\mathcal{M} =(\mathcal{X}, \mathbf{P}, \nu) \; .
$$
{{< /boxed >}}

Overloading previous notations, we will denote $p\_t(\cdot\vert x)$ the probability mass function occupied by $X\_t$
conditionally on $X\_0 = x$. Concretely, for $x, y\in\mathcal{X}$ we let
$
p\_t(y\vert x) = \mathbb{P}(X\_t = y \vert X\_0 =x)
$.
This p.m.f will be given a row vector notation $\mathbf{p}\_t(\cdot\vert x)$.
For instance, we will write:
$$
p\_t(y\vert x) = \mathbf{p}\_t(\cdot\vert x) \mathbf{y}\\;,
$$
where $\mathbf{y}$, in this context, is a column vector one-hot representation of $y$.

### Stepping the chain
A useful result regards iterating the matrix $\mathbf{P}$. In particular, for any $x\in\mathcal{X}$ one has:
$$
\mathbf{p}\_t(\cdot\vert x) = \mathbf{p}\_{t-1}(\cdot\vert x) \mathbf{P} \\; .
$$
(Recall that we are using line vector -- the above expression does make sense.) Iterating we obtain:
$$
\tag{1}
\mathbf{p}^t(\cdot\vert x) = \mathbf{x}^\top \mathbf{P}^t\\; .
$$
The main message is that by multiplying some distribution vector by $\mathbf{P}$ on the right, we are able
to step the Markov chain by one unit of time. 
{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
This result is basically restating the law of total probability. Indeed, for any $y\in\mathcal{X}$:
$$
\begin{aligned}
p\_t(y\vert x) &= \mathbb{P}(X\_t = y \vert X\_0 =x)  \\;, \\\
&= \sum\_{x^{\prime\prime}}\mathbb{P}(X\_t = y \vert X\_{t-1}=x^{\prime\prime})\mathbb{P}(X\_{t-1}=x^{\prime\prime} \vert X\_0 =x)\\;, \\\
&= \sum\_{x^{\prime\prime}}p(y \vert x^{\prime\prime})p\_{t-1}^x(x^{\prime\prime})\\;, \\\
&= \sum\_{x^{\prime\prime}}\mathbf{P}\_{x^{\prime\prime}, y}[\mathbf{p}\_{t-1}(\cdot\vert x)]\_{x^{\prime\prime}}\\;, \\\
&= [p\_{t-1}^x\mathbf{P}]\_{y}
\end{aligned}
$$
{{% /toggle_block %}}

### State classification
The states $x\in\mathcal{X}$ of a Markov chain can be classified into different categories, based on how
often they are visited by a random walk. 
For any $x\in\mathcal{X}$ let $\tau_x$ be the stopping time measure the first return to $x$:
$$
\tau\_x := \min\\{{t\geq 1}, \\; X\_t = x \text{ given } X\_0 =x \\} \\; ,
$$
with the convention that $\min\emptyset = +\infty$.
States such that $\mathbb{P}(\tau\_x < +\infty)=1$ are called _recurrent_ states.
Others, which check $\mathbb{P}(\tau\_x < +\infty)<1$ are called _transient_.
Briefly, transient states will, at one point, stop being visited by any random walk on $\mathcal{M}$ -- while
recurrent states will always be visited.

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
For countable Markov chains, the notion of recurrence has to be slightly nuanced for they can
exist states such that $\mathbb{P}(\tau\_x < +\infty)=1$ but with $\mathbb{E}\left[\tau\_x\right] = +\infty$.
{{% /toggle_block %}}


### Irreducibility
We say that some state $y$ is reachable from $x$, denoted $x\rightarrow y$ if it exists $t\in\mathbb{N}$ such that
$p\_t(y\vert x) > 0$. 
Further, $x$ and $y$ are said to _communicate_, denoted $x\leftrightarrow y$, if $x\rightarrow y$ and 
$y\rightarrow x$.
A subset $\mathcal{X}^\prime\subseteq \mathcal{X}$ is said to be _irreducible_ if all of its 
members communicates only with each others. 
All the states inside an irreducible set $\mathcal{X}^\prime$ share the same recurrence property:
they are either all recurrent, or all transient. 

<br>
{{< image src="/states.png" width="420px" align="center">}}
<br>

A Markov chain is irreducible if all states $x$ are recurrent and within the same irreducible class.
It is straightforward to understand why irreducibility is a necessary condition for the ergodic theorem. 
In a Markov chain with several irreducible classes, the limiting distribution will be necessarily dependent
on the starting state (it will be a function of which irreducible class the random walk was started from).

### Aperiodicity
Another important state property for studying ergodicity is _periodicity_. 
The period $T\_x$ of a state $x$ is defined as the greatest common denominator of the length of all 
paths joining $x$ to itself with positive probability:
$$
T\_x = \text{gcd}\\{t\geq 1 \text{ s.t } p\_t(x\vert x)>0\\}\\; .
$$
It is fairly easy to show that within an irreducible class, all the states have the same periodicity.
If $T$ is the period of an irreducible Markov chain, then $T=1$ leads us to call this chain _aperiodic_. 

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Let $x, y\in\mathcal{X}$ be in the same irreducible class. 
Therefore, it exists $t\_{x\rightarrow y}$ and $t\_{y\rightarrow x}$ such that 
$p\_{t\_{x\rightarrow y}}(y\vert x) > 0$ and $p\_{t\_{y\rightarrow x}}(x\vert y) > 0$.
Further let $t\_y$ so that $p\_{t\_y}(y\vert y)>0$ and observe that:
$$
\begin{aligned}
p\_{t\_{x\rightarrow y} + t\_y + t\_{y\rightarrow x}}(x\vert x) &= \mathbb{P}\left(X\_{t\_{x\rightarrow y} + t\_y + t\_{y\rightarrow x}} = x \vert X\_0 = x\right)\\;, \\\
&\geq \mathbb{P}\left(X\_{t\_{x\rightarrow y} + t\_y + t\_{y\rightarrow x}} = x \vert X\_{t\_{x\rightarrow y}} = y\right)\mathbb{P}(X\_{t\_{x\rightarrow y}} = y \vert X\_0=x)\\;,  &(\text{total probability})\\\
&= \mathbb{P}\left(X\_{t\_y + t\_{y\rightarrow x}} = x \vert X\_{0} = y\right)p_{t\_{x\rightarrow y}}(y\vert x)\\;,  &(\text{Markov property})\\\
&\geq \ldots &(\text{repeating})\\\
&\geq p\_{t\_{y\rightarrow x}}(x\vert y) p\_{t\_y}(y\vert y)p_{t\_{x\rightarrow y}}(y\vert x)\\;, \\\
&>0 \\; .
\end{aligned}
$$
Therefore, denoting $\mathcal{T}\_x := \\{ t\geq 1, p\_t(x\vert x)>0\\}$ we have that 
$t\_y\in\mathcal{T}\_y \Rightarrow t\_y + t\_{x\rightarrow y} + t\_{y\rightarrow x} \in\mathcal{T}\_x$.
As, a result, $T\_x = \text{gcd}(\mathcal{T}\_x) \leq \text{gcd}(\mathcal{T}\_y) = T\_y$. A symmetric argument yields
the contraposition and the announced result. 
{{% /toggle_block %}}

Again, it is fairly intuitive that periodicity is a necessary requirement for something like the 
ergodic theorem to emerge. Below is illustrated a basic periodic Markov chain, with period $T=2$.
It is clear that $p\_t(\cdot\vert x\_1) = (1, 0)$ for any $t\in 2\mathbb{N}$, and  $p\_t(\cdot\vert x\_1) = (0, 1)$ otherwise.
Therefore, $p\_t(\cdot\vert x\_1)$ does not admit a limit, and the ergodic theorem falls short. 

<br>
{{< image src="/period.png" width="500px" align="center">}}
<br>

### Ergodicity
Irreducibility and aperiodicity are two necessary conditions for a result like the ergodic theorem to emerge.
It turns out that together, they are also sufficient. A Markov chain checking both properties is called _ergodic_.
An important property of ergodic chains concerns their transition matrix.
If $\mathcal{M} = (\nu, \mathbf{P})$ is _ergodic_, then:
$$
\tag{2}
\exists t^\star \text{ s.t. } \mathbf{P}^t > 0 \text{ for all } t\geq t^\star\\,.
$$

By $\mathbf{P}^t > 0$, we mean that for all $x, y\in\mathcal{X}$ we have $[\mathbf{P}^t]\_{xy}> 0$, _i.e._ $p\_{t}(y\vert x) > 0$.
This means an ergodic Markov chain starting from any point, after waiting long enough, 
has a positive probability of being in _any_ state.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
We will first prove that it exists $t^\prime$ such that for all $t\geq t'$ and $x\in\mathcal{X}$ we have
$[\mathbf{P}\_t]\_{xx}>0$ -- that is, $p\_t(x \vert x) > 0$. Since the chain is aperiodic we know that for every
$x\in\mathcal{X}$ we have that $\text{gdc}(\mathcal{T}\_x) = 1$ where $\mathcal{T}\_x := \\{ t\geq 1, p\_t(x\vert x)> 0\\}$.
Note that $\mathcal{T}\_x$ is closed under addition; if $t\_1, t\_2 \in \mathcal{T}\_x$ then $t\_1 + t\_2 \in  \mathcal{T}\_x$:
$$
\begin{aligned}
p\_{t\_1+t\_2}(x \vert x) &= \mathbb{P}(X\_{t\_1+t\_2}=x \vert X\_0= x)\\;, \\\
&\geq   \mathbb{P}(X\_{t\_1+t\_2}=x \vert X\_{t\_1}= x) \mathbb{P}(X\_{t\_1=x} \vert X\_0= x)\\;, \\\
&=  \mathbb{P}(X\_{t\_2}=x, \vert X\_{0}= x) \mathbb{P}(X\_{t\_1}=x \vert X\_0= x)\\;, \\\
&= p\_{t\_2}(x\vert x)p\_{t\_1}(x\vert x)\\;,\\\
&>0 \\; .& (t\_1, \\, t\_2 \in\mathcal{T}\_{x})
\end{aligned}
$$
It is a known result from number theory that therefore it exists $t^\star\_x$ such that 
$\\{ t\geq t^\star\_x\\} \subseteq \mathcal{T}\_x$. 
Letting $t^\star = \max\_{x\in\mathcal{X}} t^\star\_x$ we obtain the announced result.
(Observe that this result is a consequence of aperiodicity only!). 
Recurrence is enough to finish the proof. 
Indeed, by definition, we know for any $x, y\in\mathcal{X}$ it exists $t\_{x\rightarrow y}$ such that
$p\_{t\_{x\rightarrow y}}(y\vert x)> 0$.
For any $t\geq t^\star\_x + t\_{x\rightarrow y }$ we have that:
$$
\begin{aligned}
p\_{t}(y\vert x) &= \mathbb{P}(X\_t=y\vert X\_0 =x)\\;, \\\
&\geq \mathbb{P}(X\_{t}=y\vert X\_{t-t\_{x\rightarrow y}})\mathbb{P}(X\_{t-t\_{x\rightarrow y}}\vert X\_0 =x)\\;, \\\
&= p\_{t\_{x\rightarrow y}}(y\vert x)\mathbb{P}(X\_{t-t\_{x\rightarrow y}}\vert X\_0 =x)\\;, \\\
&>0\\;.
\end{aligned}
$$
since $t-t\_{x\rightarrow y} \geq t^\star\_x$.
Letting $t^\star = \max\_{x, y\in\mathcal{X}} \\{ t^\star\_x + t\_{x\rightarrow y}\\}$ yields the announced result.


{{% /toggle_block %}}

### Total-variation distance

Proving the ergodic theorem will involve showing that some notion of distance between probability distributions over $\mathcal{X}$ decreases over time.
The distance we will be interested in is the total-variation distance; for any p.m.f $\mu$ and $\nu$ over $\mathcal{X}$:
$$
\begin{aligned}
\lVert \mu - \nu \rVert_{\tiny \text{TV}} :&= \sup\_{\mathcal{Y}\subseteq \mathcal{X}} \mu(\mathcal{Y}) - \nu(\mathcal{Y})\\;, \\\
&= \frac{1}{2}\sum\_{x\in\mathcal{X}} \vert \mu(x) - \nu(x) \vert\\; .\tag{3}
\end{aligned}
$$


## Ergodic Theorem
We will now  list several equivalent formulations of the ergodic theorem. They all boil down to the same property: that 
the distribution followed by a Markov chain process converges to a limiting distribution, that is independant of the initial state.
This is an important result; whenever we are interested in any long-term behaviour, we can safely ignore the initial conditions and focus on said limiting distribution.


{{< boxed title="Ergodic Theorem (1/3)" >}}
$\qquad\quad\qquad\qquad\qquad\qquad\text{It exists }\pi \in \Delta(\mathcal{X})\text{ such that  for any } x\in\mathcal{X}$:
$$
\tag{5}
\lim_{t\to\infty} \lVert p_t(\cdot \vert x) - \pi \rVert_{\text{\tiny TV}}  = 0
$$
{{< /boxed >}}

The distribution $\pi$ is called a stationary distribution.
The reason for this name is that any process starting under $\pi$ finds itself in the _same_ distribution after one step of the Markov chain.
One can write this as:
$$
\tag{6}
\boldsymbol{\pi} \mathbf{P} = \boldsymbol{\pi} \\; .
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Observe a reformulation of (5) writes that for any $x\in\mathcal{X}$ we have
$
\lim\_{t\to\infty} \mathbf{x}^\top \mathbf{P}^t = \boldsymbol{\pi}\\; .
$
Therefore:
$$
\begin{aligned}
 \boldsymbol{\pi} &= \sum\_{x\in\mathcal{X}} \pi(x) \boldsymbol{\pi}\\;, &(\boldsymbol{\pi}\in\Delta(\mathcal{X})) \\\
&= \lim\_{t\to\infty} \sum\_{x\in\mathcal{X}} \pi(x) \mathbf{x}^\top \mathbf{P}^t \\;, \\\
&= \sum\_{x\in\mathcal{X}} \pi(x)\mathbf{x}^\top \lim\_{t\to\infty}\mathbf{P}^t \\;, \\\
&= \boldsymbol{\pi} \lim\_{t\to\infty}\mathbf{P}^t \\; .
\end{aligned}
$$
This establishes the claimed result since:
$$
\boldsymbol{\pi} \mathbf{P} = \boldsymbol{\pi} (\lim\_{t\to\infty}\mathbf{P}^{t})\mathbf{P} = \boldsymbol{\pi} \lim\_{t\to\infty}\mathbf{P}^{t+1} =  \boldsymbol{\pi} \lim\_{t\to\infty}\mathbf{P}^{t} = \boldsymbol{\pi}\\; .
$$
{{% /toggle_block %}}

As defined in (6) the stationary distribution is a left-eigenvector of $\mathbf{P}$ with eigenvalue 1.
We will see shortly that this uniquely defines $\boldsymbol{\pi}$. Using vectorial notations, we can rewrite the ergodic theorem 
as follows.

{{< boxed title="Ergodic Theorem (2/3)" >}}
$\qquad\quad\qquad\qquad\qquad\qquad\text{It exists }\pi \in \Delta(\mathcal{X})\text{ such that:}$
$$
\lim_{t\to\infty} \mathbf{P}^t = \begin{pmatrix}\boldsymbol{\pi} \\\ \boldsymbol{\pi} \\\ \vdots \\\ \boldsymbol{\pi}  \end{pmatrix}
$$
{{< /boxed >}}

Sometimes, the ergodic theorem writes much like the law of large numbers.
The next statement embodies the fact that, with Markov chains, _the temporal average is the spatial average_.

{{< boxed title="Ergodic Theorem (3/3)" >}}
$\qquad\quad\qquad\qquad\qquad\qquad\text{Let $\mathcal{M}=(\mathcal{X},\mathbf{P})$ a Markov chain.
Then, if } \pi \text{ is its stationary distribution, we have}\\\text{ that for any } f:\mathcal{X}\mapsto \mathbb{R}:$
$$
\lim_{t\to\infty} \frac{1}{t}\sum_{k=1}^t f(X_k) = \sum_{x\in\mathcal{X}} \pi(x)f(x) = \mathbf{E}_{X\sim\pi}[f(X)]\; .
$$
{{< /boxed >}}



## A proof based on coupling
Before diving in va first proof for the ergodic theorem, we need an important result.
It will indeed prove convenient to establish that the stationary distribution is properly defined via $\boldsymbol{\pi} = \boldsymbol{\pi}\boldsymbol{P}$.
We can then focus on the convergence of $p\_t(\cdot\vert x)$, already having a properly defined candidate for the limit.
To do so, we will rely on the following fundamental result:
$$
\textit{Every ergodic Markov chain admits a unique stationary distribution }\pi \\; .
$$

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
The above statement is a slight overkill since it only takes irreducibility for a unique stationary distribution to exist.
However, we already discussed that without aperiodicity, we simply cannot hope for said stationary distribution to be a limiting distribution
(since $p\_t(\cdot\vert x)$ simply does not converge!).
{{% /toggle_block %}}



{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
We will prove here only the existence -- unicity is demonstrated using similar tools. 
It is enough to prove the existence of a non-zero vector $\boldsymbol{\nu}$ with positive entries such that $\boldsymbol{\nu}\mathbf{P}= \boldsymbol{\nu}$ (we will obtain
our desired result up to renormalisation of $\boldsymbol{\nu}$). 
For some $x\in\mathcal{X}$ let $\boldsymbol{\nu}$ be the expected number of visits to each state in $\mathcal{X}$, between two consecutive visits of $x$:
$$
\begin{aligned}
\nu\_y &:= \mathbb{E}\left[ \sum\_{t\leq \tau\_x} 1[X\_t = y] \middle\vert X\_0 = x\right]\\; ,\\\
&= \mathbb{E}\left[ \sum\_{t\leq \infty} 1[X\_t = y, t \leq \tau\_x] \middle\vert X\_0 = x\right]\\;, \\\
&= \sum\_{t\leq \infty} \mathbb{P}\left(X\_t=y, t\leq \tau\_x \middle\vert X\_0=x\right)\\; .
\end{aligned}
$$
with $\tau\_x := \min\\{{t\geq 1}, \\; X\_t = x \text{ given } X\_0 =x \\}$ (we have $\tau\_x <\infty$ a.s.
by the chain's irreducibility.) Observe that for any $z\in\mathcal{X}$:
$$
\begin{aligned}
\sum\_{y} \nu\_y \mathbf{P}\_{yz} &= \sum\_y \sum\_{t\leq \infty} \mathbb{P}\left(X\_t=y, t\leq \tau\_x \middle\vert X\_0=x\right)\mathbf{P}\_{yz}\\;, \\\
&=\sum\_{t=1}^\infty \sum\_y \mathbb{P}\left(X\_t=y, t\leq \tau\_x \middle\vert X\_0=x\right)\mathbb{P}(X\_{t+1} = z \vert X\_{t}=y)\\;, \\\
&= \sum\_{t=1}^\infty \mathbb{P}\left(X\_{t+1}=z, t\leq \tau\_x \middle\vert X\_0=x\right)\\;, \\\
&=  \mathbb{E}\left[ \sum\_{t\leq \tau\_x} 1[X\_{t+1} = z] \middle\vert X\_0 = x\right] \\;, \\\
&=  \mathbb{E}\left[ \sum\_{t=2}^{\tau\_x} 1[X\_{t} = z] \middle\vert X\_0 = x\right] + \mathbb{E}\left[1[X\_{\tau\_x + 1} = z] \middle\vert X\_0 = x\right] \\;, \\\
&=  \mathbb{E}\left[ \sum\_{t=2}^{\tau\_x} 1[X\_{t} = z] \middle\vert X\_0 = x\right] + \mathbb{E}\left[1[X\_{\tau\_x + 1} = z] \middle\vert X\_{\tau\_x} = x\right] \\;, \\\
& \overset{(i)}{=}  \mathbb{E}\left[ \sum\_{t=2}^{\tau\_x} 1[X\_{t} = z] \middle\vert X\_0 = x\right] + \mathbb{E}\left[1[X\_{1} = z] \middle\vert X\_{0} = x\right] \\;, \\\
&=  \mathbb{E}\left[ \sum\_{t\leq \tau\_x} 1[X\_{t} = z] \middle\vert X\_0 = x\right] \\;, \\\
&= \nu\_{z}\\; .
\end{aligned}
$$
In $\text{(i)}$, we simply used the strong Markov property. 
This proved the claimed result that $\boldsymbol{\nu}\mathbf{P}= \boldsymbol{\nu}$.
{{% /toggle_block %}}


### Reminders on coupling
We will use a line of proof that uses _coupling_. 
Given two distributions $\mu$ and $\nu$ over $\mathcal{X}$, we say that $\omega$ is a coupling
of $(\mu, \nu)$ if for all $x, y\in\mathcal{X}$:
$$
\begin{aligned}
\sum\_{y} \omega(x, y) &= \mu(x)\\;, \\\
\sum\_{x} \omega(x, y) &= \nu(y) \\;. \\\
\end{aligned}
$$
In other words, $\mu$ and $\nu$ are $\omega$'s marginal. 

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
A trivial coupling is the independent coupling $\omega(x, y) = \mu(x)\nu(y)$ -- but it is seldom useful. 
{{% /toggle_block %}}

The fundamental result we will use is known as the _coupling inequality_.
Let $\omega$ a coupling of $(\mu, \nu)$, and $(X, Y) \sim \omega$. Then:
$$
\tag{7}
\lVert \mu - \nu \rVert_{\text{TV}} \leq \mathbb{P}(X \neq Y) \\; .
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Observe that for any $\mathcal{Y}\subseteq \mathcal{X}$ we have:
$$
\begin{aligned}
\mu(\mathcal{Y}) - \nu(\mathcal{Y}) &= \mathbb{P}(X\in\mathcal{Y}) - \mathbb{P}(Y\in\mathcal{Y})\\;,  \\\
&= \mathbb{P}(X\in\mathcal{Y}, X=Y) + \mathbb{P}(X\in\mathcal{Y}, X\neq Y) - \mathbb{P}(Y\in\mathcal{Y}, X=Y) - \mathbb{P}(Y\in\mathcal{Y}, X\neq Y)\\;, \\\
&= \mathbb{P}(X\in\mathcal{Y}, X\neq Y) - \mathbb{P}(Y\in\mathcal{Y}, X\neq Y)\\;, \\\
&= \mathbb{P}(X\neq Y)\left(\mathbb{P}(X\in\mathcal{Y}\vert X\neq Y) - \mathbb{P}(Y\in\mathcal{Y}\vert X\neq Y)\right)\\; .
\end{aligned}
$$
As a result, $\sup\_{\mathcal{Y}} \mu(\mathcal{Y}) - \nu(\mathcal{Y}) \leq \mathbb{P}(X\neq Y)$ which is the claimed result.

{{% /toggle_block %}}

### The meet-and-stick coupling
We will now build a coupling that will lead us to the ergodic theorem via the coupling inequality.
Consider some process $X\_t$ following a random walk over our Markov chain, with some arbitrary starting point $x\in\mathcal{X}$.
On the other hand, let $Y\_t$ evolve according to the same random walk, yet initialised via $Y\_0 \sim \pi$ -- so that
for any $t\geq 1$ we have that $Y\_t\sim\pi$, since $\pi$ is stationary.
Let $N\_x$ the first time $X\_t$ and $Y\_t$ meet:
$$
N\_x := \inf\\{t\geq 1, \\; X\_t = Y\_t\\}\\; .
$$
After this "meeting" round, we will "stick" both processes: $X\_t = Y\_t$ for all $t\geq N\_x$.
Observe that, thanks to ergodicity, both _must_ meet after a finite time.
In other words $\mathbb{P}(N\_x = \infty) = 0$.
(Indeed, we know that after some time $t^\star$ we have $p\_{t^\star}(y\vert x)>0$ for any $y\in\mathcal{Y}$).
By the coupling inequality (7) we have that:
$$
\begin{aligned}
\left\lVert p\_t(\cdot\vert x) - \pi\right\rVert_{\text{TV}} &\leq \mathbb{P}(X\_t \neq Y\_t) \\;, \\\
&= \mathbb{P}(t\leq N\_x)\\; .
\end{aligned}
$$
Therefore, $\lim\_{t\to\infty} \left\lVert p\_t(\cdot\vert x) - \pi\right\rVert_{\text{TV}} = 0$, and there we have our proof!


## A proof based on the Perron-Frobenius theorem
We now go through another proof that completely ignores the probabilistic rooting of Markov chains.
Instead, we will use _only_ linear algebra. (What makes the beauty of Markov chain in my eyes
is the versatility of tools we can use to study them.)

The main result we will use is the Perron-Frobenius theorem. Recall that we use the following notation:
for $\mathbf{A} = (a\_{ij})$, $\mathbf{A}>0$ (or $\mathbf{A}$ is _positive_) means that $a\_{ij}>0$ for all $i, j$.

{{< boxed title="Perron Frobenius Theorem" >}}
$\qquad\qquad\qquad\qquad\qquad\qquad\;\; \text{Let }\mathbf{A}\in\mathbb{R}^{n\times n} \text{ be a positive square matrix, and }  \text{Sp}(\mathbf{A}) \text{ its eigenvalues. Then:}\\$
$$\begin{aligned}
1.& \exists\lambda_1 \in \text{Sp}(\mathbf{A}) \text{ s.t } \lambda_1\in\mathbb{R} \text{ and } \lambda_1 \geq \vert \lambda_j\vert \text{ for all }j\neq 1\;, \\
2.& \exists \mathbf{v} \in \mathbb{R}^n \text{ s.t. } \mathbf{A}\mathbf{v}  = \lambda_1 \mathbf{v}  \text{ and } \mathbf{v} >0\; .
\end{aligned}
$$
{{< /boxed >}}

Bluntly, the Perron-Frobenis identifies, for positive matrices, a real-valued _leading_ eigenvalue $\lambda\_1$ dominating all others (other eigenvalues can be complex). 
Its associated eigenvector can be chosen so that all of its entries are strictly positive.

At first sight, it is unclear how this helps us. 
The transition matrix $\mathbf{P}$ is not positive (only non-negative, $\mathbf{P}\geq 0$). 
For the sake of simplicity, let us cheat: we already now (see above) that $\mathbf{P}^t$ is
a convergent sequence. 
All of its subsequences therefore converge—in particular, so does the sequence $\\{(\mathbf{P}^{t^\star})^k \\}\_k$.
It is enough to study the convergence of this last sequence, which depends only $\mathbf{P}^{t^\star}$.
And, as we saw in (2), $\mathbf{P}^{t^\star}$ is positive!

{{< warningblock>}}
$\quad$ This is an hand wavy argument that takes the problem by its solution.
The point is to somewhat justify that we can assume without loss of generality
that $\mathbf{P}>0$–our working assumption from now on.
{{< /warningblock >}}

It is fairly straightforward to prove that for positive stochastic matrix $\mathbf{P}$ the leading eigenvalue is $\lambda\_1=1$.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Let $\lambda\in\text{Sp}(\mathbf{P})$. We will first prove that $\vert \lambda \vert \leq 1$. 
Since it exists $u\in\mathbb{R}^n\setminus\\{0\\}$ such that $\mathbf{P}u = \lambda u$:
$$
\begin{aligned}
\vert \lambda \vert &= \lVert \mathbf{P}u \rVert_\infty / \lVert u \rVert_\infty \\;, \\\
&\leq  \|\|\| \mathbf{P} \|\|\|\_\infty\\;, \\\
&= \sup\_{x\in\mathcal{X}}\sum\_{y\in\mathcal{Y}} \mathbf{P}\_{xy} \\;, \\\
&= 1 \\;,
\end{aligned}
$$
where $\|\|\| \mathbf{P} \|\|\|\_\infty$ is the operator-norm for the $\ell\_\infty$-norm.
Further, observe that if $e = (1, \ldots, 1)$ then $\mathbf{P}e = e$.
In other words, we have proven that $1 = \max \text{Sp}(\mathbf{P})$. 
By the Perron-Frobenius theorem, this means that:
$$
\text{Sp}(\mathbf{P}) =  ( 1, \lambda\_2, \ldots)\\;, 
$$
with $\vert \lambda\_j\vert  < 1$ for any $j\neq 1$. 
{{% /toggle_block %}}

Since $\mathbf{P}$ and $\mathbf{P}^\top$ share the same eigenvalues, and because the Perron-Frobenius 
theorem also applies to $\mathbf{P}^\top$, we conclude that there is a line vector $\mathbf{v}$ with positive entries such that
$\mathbf{v}\mathbf{P} = \mathbf{v}$. By rescaling $\mathbf{v}$, we obtain that there is a unique $\boldsymbol{\pi}$, with positive entries,
such that $\lVert \boldsymbol{\pi}\rVert\_1 = 1$ and $\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}$.
We have just proven the existence and uniqueness of the stationary distribution!

Let's now go one step further and prove the ergodic theorem itself. 
Because we are using linear algebra tools, you can already guess that we are after the formulation [(2/3)](/post/mc/#:~:text=Ergodic%20Theorem%20(2/3)) of the ergodic theorem.
The proof relies on the [Jordan decomposition](https://en.wikipedia.org/wiki/Jordan_normal_form) of a square matrix.
Thanks to the Perron-Frobenius theorem, we know that the Jordan decomposition of $\mathbf{P}$ writes:
$$
\mathbf{P} = \mathbf{Q}\begin{pmatrix} 1  & 0 & \ldots &\ldots\\\ 0 & J\_2 & 0 & \ldots\\\0 & 0 & \ddots & 0\\\ \ldots & 0 & 0 & J\_n \end{pmatrix} \mathbf{Q}^{-1}, \quad \text{with } J\_i = \begin{pmatrix} \lambda\_i & 1 & 0 &\ldots & 0 \\\ 0 & \lambda\_i & 1 &\ddots & \vdots \\\ \vdots & \ddots &\ddots & \ddots & 0 \\\ 0 & \ddots & \ddots &\ddots & 1 \\\ 0 &\ldots & \ldots & 0 & \lambda\_i\end{pmatrix}\\;,
$$
and $\mathbf{Q}$ some non-singular matrix. From this one can show that:
$$
\mathbf{P}^\infty = \lim\_{n\to\infty}\mathbf{P}^n = \mathbf{Q}\begin{pmatrix} 1  & 0 & \ldots &\ldots\\\ 0 & 0 & 0 & \ldots\\\0 & 0 & \ddots & 0\\\ 0 & 0 & 0 & 0 \end{pmatrix}\mathbf{Q}^{-1}\\;,
$$
by using the fact that all $J\_i$ are nilpotent, since $\lambda\_i<1$. In others words, $\mathbf{P}^\infty=\lim\_{n\to\infty}\mathbf{P}^n$ exists
and is a rank one matrix. Because $\mathbf{P}^\infty$ must be a stochastic matrix ($\mathbf{P}^n$ is stochastic for any $n\in\mathbb{N}$),
we conclude that all its ranks sum to $1$. 
Under this constraint, the only possibility for $\mathbf{P}^\infty$ to be of rank 1 is that all of its rows are equal.
We conclude that there exists a line vector $\mathbf{q}$ such that:
$$
\mathbf{P}^\infty = \begin{pmatrix} \mathbf{q} \\\ \vdots \\\ \mathbf{q}\end{pmatrix}\\;. 
$$


Further, one can show that this line vector checks $\mathbf{q} \mathbf{P} = \mathbf{q}$ -- the stationary distribution characterisation.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Notice that:
$$
\begin{aligned}
\begin{pmatrix} \mathbf{q} \\\ \vdots \\\ \mathbf{q}\end{pmatrix} \mathbf{P}&= \mathbf{P}^\infty\mathbf{P}\\;, \\\
&= (\lim\_{n\to\infty} \mathbf{P}^n)\mathbf{P} \\;, \\\
&= (\lim\_{n\to\infty} \mathbf{P}^{n+1}) \\;, \\\
&= \mathbf{P}^\infty\\;, \\\
&=\begin{pmatrix} \mathbf{q} \\\ \vdots \\\ \mathbf{q}\end{pmatrix}\\; .
\end{aligned}
$$
Identifying row by row, we get that $\mathbf{q} \mathbf{P} = \mathbf{q}$. 
{{% /toggle_block %}}

By unicity of the stationary distribution, we can safely conclude that $\mathbf{q} = \boldsymbol{\pi}$.
We can now rest, as we just proved the formulation [(2/3)](/post/mc/#:~:text=Ergodic%20Theorem%20(2/3)) of the ergodic theorem:
$$
\mathbf{P}^\infty = \begin{pmatrix} \boldsymbol{\pi}\\\ \vdots \\\ \boldsymbol{\pi}\end{pmatrix}\\; .
$$



### Resources
[[1]](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf) is a nicely written book on Markov Chains, that goes well beyond "just" stationary distributions.

[[2]](https://arxiv.org/pdf/2204.00784.pdf) is a collection of various proofs for the ergodic theorem. 

[[3]](https://mpaldridge.github.io/math2750/S11-long-term-chains.html) is a pretty neat course on Markov Chains.

If you're interested in convergence rates and mixing times, I really enjoy [this](https://www.ceremade.dauphine.fr/~salez/mix.pdf).