+++
author = "Louis Faury"
title = "From VI to DQN"
date = "2024-02-10"
+++


The goal of this blog post is walk the path that saw the Value Iteration algorithm
mature into a variety of modern value-based reinforcement learning methods empowered by deep learning.
In particular, we will first see how VI gave birth to the Q-learning algorithm.
We will then see how this tabular algorithm was gradually modified to handle function approximation
and give rise to the watershed DQN algorithm. 

<!--more-->

{{< infoblock>}}
$\quad$ Take a look at the MDP fundamentals series if you need a warm-up.
{{< /infoblock >}}

Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, p, r)$ be the MDP of interest.
We will assume for simplicity that $\mathcal{M}$ is finite: $\vert \mathcal{S} \vert \times \vert \mathcal{A}\vert < \infty$.
We will denote $\mathbb{R}^{\mathcal{S}}$ (resp. $\mathbb{R}^{\mathcal{S}\times\mathcal{A}}$)
the space of function mapping $\mathcal{S}$ (resp. $\mathcal{S}\times\mathcal{A}$) to $\mathbb{R}$.
Finally, to match with our initial exposition of the VI algorithm, we will consider the _discounted_ criterion to evaluate policies.


## The qVI algorithm

See [here](/post/mdp_basics_3/#:~:text=Note-,Value%20Iteration,-The%20Value%20Iteration)
for some refresher on the Value-Iteration (VI) algorithm and its convergence properties. 
For reasons that will soon become clear, we will here rely on an alternative version of VI, denoted qVI,
which tracks the optimal state-action value function $q\_\lambda^\star$ instead of its state-only counterpart $v\_\lambda^\star$.

Formally, let $\mathcal{T}\_\lambda^\star : \mathbb{R}^{\mathcal{S}\times\mathcal{A}} \mapsto \mathbb{R}^{\mathcal{S}\times\mathcal{A}}$
such that for any $q\in \mathbb{R}^{\mathcal{S}\times\mathcal{A}}$:
$$
\tag{1}
\mathcal{T}\_\lambda^\star(q)(s, a) = r(s, a) + \lambda \sum\_{s^\prime\in\mathcal{S}} p(s^\prime\vert s, a) \max_{a^\prime\in\mathcal{A}} q(s^\prime, a^\prime) \qquad  \forall s, a \in \mathcal{S}\times\mathcal{A}\\; .
$$
The update rule for qVI is defined
by $
q\_{t+1} = \mathcal{T}\_\lambda^\star(q\_t)\\; .
$
We detail below the resulting pseudocode.

{{< pseudocode title="qVI" >}} 
$\textbf{init } q_0\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}, \text{ max. iteration T}\\$
$\textbf{for } t = 0, \ldots, T-1:\\$
$\qquad\textbf{for }  s\in\mathcal{S}, \, a\in\mathcal{A}:$
$$
    q_{t+1}(s, a) = r(s,a) + \lambda\sum_{s^\prime\in\mathcal{S}} p(s^\prime\vert s, a) \max_{a^\prime\in\mathcal{A}} q_{t}(s^\prime, a^\prime)
$$
$\qquad\textbf{end for}\\$
$\textbf{end for}\\$
$\textbf{return } q_T$
{{< /pseudocode >}}

It is easy to show, by the same fixed-point argument that we used for the state value-function, that <br>
the sequence of iterates produced by qVI converges to the optimal $q\_\lambda^\star$:
$$
q_{t}(s, a) \underset{t\to\infty}{\longrightarrow} q_\lambda^\star(s, a)\\; .
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof mirrors the one we derived for establishing the convergence of VI.
We will only prove that $q\_\lambda^\star$ is the fixed-point of a contracting operator--the rest of the proof will follow 
from the demonstration we completed for VI and is left as an exercise.
By the very definition of $q\_\lambda^\star$ we have:
$$
q\_\lambda^\star(s, a) = r(s, a) + \sum\_{s^\prime\in\mathcal{S}} p(s^\prime\vert s, a) v\_\lambda^\star(s^\prime), \qquad \forall s, a \in\mathcal{S}\times\mathcal{A}\\; .
$$
Also recall that $v\_\lambda^\star$ is the unique solution of the fixed-point equation
$$
\begin{aligned}
v\_\lambda^\star(s) &= \max\_{a\in\mathcal{A}} r(s, a) + \lambda \sum\_{s^\prime\in\mathcal{S}} p(s^\prime\vert s, a) v\_\lambda^\star(s^\prime) \\;, \\\
&= \max\_{a\in\mathcal{A}} q\_\lambda^\star(s, a)\\;,
\end{aligned}
$$
according to the above identity. Replacing, we obtain that $q\_\lambda^\star$ is a solution to the fixed-point:
$$
\begin{aligned}
q\_\lambda^\star(s, a) &= r(s, a) + \sum\_{s^\prime\in\mathcal{S}} p(s^\prime\vert s, a)\max\_{a^\prime\in\mathcal{A}} q\_\lambda^\star(s^\prime, a^\prime)\\;, \\\
&= \mathcal{T}\_\lambda^\star(q\_\lambda^\star)(s, a) \\; .
\end{aligned}
$$
Following our proof for VI, it is fairly easy to show that $\mathcal{T}\_\lambda^\star$ is a _contracting_ operator;
the convergence of qVI then directly flows from this fact.
{{% /toggle_block %}}

The qVI is a control algorithm: it relies on an explicit knowledge of both
the reward signal $r$ and the transition kernel $p$. 
Naturally, this is a fairly restrictive assumption that fails to hold in many settings.

## Stochastic approximation and Q-learning
We will now lift the assumption that $r$ and $p$ are known. 
Instead, we will suppose that we can only access them via _sampling_. Formally, we consider
the following sequential interaction setting.
At every round, we submit an action $a\_t$ and observe both the reward $r(s\_t, a\_t)$
and the next state $s\_{t+1} \sim p(\cdot\vert s\_t, a\_t)$. We then submit $a\_{t+1}$,
observe $r(s\_{t+1}, a\_{t+1})$, .. and so on.

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
The described sequential interaction setting hits close to the definition of a _simulator_.
Another interaction paradigm studied in the RL literature is the _generative model_: 
instead of having to query $r$ and $p$ at the current state $s\_t$, one can ask for the value of
$r(s, a)$ and a realisation from $p(\cdot\vert s, a)$ for _any_ $s, a\in\mathcal{S}\times\mathcal{A}$. 
{{% /toggle_block %}}

Let us define the operator $\hat{\mathcal{T}}\_\lambda: \mathbb{R}^{\mathcal{S}\times\mathcal{A}} \mapsto \mathbb{R}^{\mathcal{S}\times\mathcal{A}}$
such that for any $q\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}$:
$$
\hat{\mathcal{T}}\_\lambda(q)(s\_t, a\_t) = r(s\_t, a\_t) + \lambda \max\_{a^\prime\in\mathcal{A}} q(s\_{t+1}, a^\prime)\\; .
$$
Observe that:
$$
\begin{aligned}
\mathbb{E}\left[\hat{\mathcal{T}}\_\lambda(q)(s\_t, a\_t)\vert s\_t, a\_t\right] &= \mathbb{E}\left[r(s\_t, a\_t) + \lambda \max\_{a^\prime\in\mathcal{A}} q(s\_{t+1}, a^\prime)\right]\\;, \\\
&= r(s\_t, a\_t) + \lambda \sum\_{s^\prime\in\mathcal{S}} p(s^\prime\vert s\_t, a\_t) \max\_{a^\prime\in\mathcal{A}} q(s^\prime, a^\prime) \\;, \\\
&= \mathcal{T}\_\lambda^\star(q)(s\_t, a\_t)\\; .
\end{aligned}
$$
That is, $\mathbb{E}[\hat{\mathcal{T}}\_\lambda(q)]  = \mathcal{T}\_\lambda^\star(q)$.
We now have access to a _noisy_ version of the Bellman operator $\mathcal{T}\_\lambda^\star$.
To understand how should qVI be modified to fit this new setting, we must talk about 
stochastic approximation algorithms.

### Stochastic Approximation
The [stochastic approximation problem](https://en.wikipedia.org/wiki/Stochastic_approximation)
refers to the computation of a root $x^\star$ for some operator $F$, 
when $F$ can only be queried up to some noise:
$
\hat{F}(x) := F(x) + \varepsilon$
and
$\mathbb{E}\left[\varepsilon\right] = 0 \\; .
$
The Robbins-Monro algorithm is an iterative algorithm which builds a sequence $\\{x\_t\\}\_t$ as follows:
$$
x\_{t+1} = x\_t - \alpha\_t \hat{F}(x\_t)\\;,
$$
where $\\{\alpha\_t\\}$ follows the so-called _tapering_ conditions:
$\sum\_{t\geq 0} \alpha\_t = +\infty$ and $\sum\_{t\geq 0} \alpha\_t^2 < +\infty$. 
Then, under appropriate conditions over $F$ and the noise's distribution, we have the almost-sure convergence $x\_t \overset{\text{a.s}}{\to} x^\star$.
In our case, we are looking for the unique fixed-point of $\mathcal{T}\_\lambda^\star$, or 
equivalently, for the unique root of the operator $\mathcal{B}\_\lambda^\star := \mathcal{T}\_\lambda^\star - \text{Id}$.
The Robbins-Monro algorithm applied to $\hat{\mathcal{B}}\_\lambda := \hat{\mathcal{T}}\_\lambda - \text{Id}$ suggests maintaining the sequence:
$$
\tag{2}
q\_{t+1} = q\_t - \alpha\_t(\hat{\mathcal{T}}\_\lambda - \text{Id}) q\_t  \\; .
$$
For a given state-action couple $(s, a)$ this yields:
$$
\begin{aligned}
q\_{t+1}(s, a) &= q\_t(s, a) - \alpha\_t \big[r(s, a) + \lambda \max\_{a^\prime\in\mathcal{A}} q\_t(s^\prime, a^\prime) -  q\_t(s, a) \big]\\; ,  &(s^\prime\sim p(\cdot\vert s, a))\\\
&= q\_t(s, a) + \alpha\_t \delta\_t(s, a)
\end{aligned}
$$
where $\delta\_t(s, a):=r(s, a) + \lambda \max\_{a^\prime\in\mathcal{A}} q\_t(s^\prime, a^\prime) -  q\_t(s, a)$ is often refered to as the temporal-difference
error. Equation $\text{(2)}$ is, in essence, the Q-learning algorithm, which is therefore
not much more than "just" the stochastic approximation of the qVI algorithm (up to some details.)


### Q-Learning

We provide below the pseudocode for Q-learning, before moving to its convergence properties.

{{< pseudocode title="Q-learning" >}} 
$\textbf{init } s_0 \in\mathcal{S}, q_0\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}, \text{ max. iteration T}, 
\text{ tapering step-sizes } \{\alpha_t\}_t, \text{ behavioral decision-rule } d_\beta\\$
$\textbf{for } t = 0, \ldots, T-1:\\$
$\qquad \text{sample } a_t \sim d_\beta(s_t), \text{ observe } r_t \text{ and } s_{t+1}, \\$
$\qquad\textbf{for }  s\in\mathcal{S}, \, a\in\mathcal{A}:$
$$
\begin{aligned}
    q_{t+1}(s, a) &= q_t(s, a) + \alpha_t \left(r_t + \lambda \max_{a^\prime\in\mathcal{A}} q_t(s_{t+1}, a^\prime)-q_t(s_t, a_t)\right) &\text{ if } s=s_t, \; a =a_t, \\
    q_{t+1}(s, a) &= q_t(s, a) &\text{ otherwise.} 
\end{aligned}
$$
$\qquad\textbf{end for}\\$
$\textbf{end for}\\$
$\textbf{return } q_T$
{{< /pseudocode >}}


How we visit the MDP is dictated by a so-called behavioral policy $\pi\_\beta= (d\_\beta, d\_\beta, \ldots)$, 
which is <br>
independent of the optimal $\pi^\star$. For this reason, Q-learning is an _off-policy_ algorithm.
There are only a few constraints imposed on this data-collection policy (see below).


{{% toggle_block background-color="#CBE4FE" title="Note" %}}
The behavioral policy $\pi\_\beta$ is not necessarily stationary. We followed this design here for simplicity.
{{% /toggle_block %}}

Notice how under the Q-learning algorithm the update at round $t$ writes:
$$
\begin{aligned}
q\_{t+1}(s, a) &= q\_{t}(s, a) + \alpha\_t\hat{\mathcal{B}}(q\_{t})(s, a) \mathbf{1}\left[s\_t=s, a\_t=a\right]\\; ,\\\
&= q\_{t}(s, a) +\alpha\_t( \underbrace{\hat{\mathcal{B}}-\mathcal{B}^\star)(q\_{t})(s, a)}_{\mathbb{E}[\cdot] = 0} \mathbf{1}\left[s\_t=s, a\_t=a\right] + \alpha\_t\mathcal{B}^\star (q\_{t})(s, a) \mathbf{1}\left[s\_t=s, a\_t=a\right] \\; ,\\
\end{aligned}
$$
where $\mathbf{1}[\cdot]$ is the indicator function. Therefore:
$$
\begin{aligned}
\mathbb{E}[q\_{t+1}(s, a)] &= q\_{t}(s, a) + \alpha\_t \mathcal{B}\_\lambda^\star q\_t(s, a) \mathbf{1}\left[s\_t=s, a\_t=a\right]\\; ,\\\
&= q\_{t}(s, a) + \alpha\_t \mathcal{B}\_\lambda^\star q\_t(s, a) \text{ only if } s=s\_t, \\; a = a\_t.
\end{aligned}
$$
Unfortunately, this does not perfectly match with the description of the Robbins-Monro algorithm, which required our
surrogate to be an unbiased estimate of the noiseless function we were trying to find the root of. Indeed, 
$\mathbb{E}[q\_{t+1}(s, a)] \neq q\_{t}(s, a) + \alpha\_t \mathcal{B}\_\lambda^\star q\_t(s, a)$ for any couple
$(s, a) \neq (s\_t, a\_t)$. 
Fortunately, this unbiased assumption can be lessened, asking only for the bias to be "small" enough
for it to vanish over time. We state below the proper theorem for the sake of completeness:


{{< boxed title="Jaakkola and al. 1993" >}}
$\qquad\qquad\qquad\qquad \qquad \text{Let }\{\Delta_t\}_t\text{ an }
\mathbb{R}^d\text{-value stochastic process, recursively defined as:}$
	$$
		\Delta_{t+1}(i) = \Delta_t(i) + \alpha_t(i)(\eta_t(i)-\Delta_t(i))\text{ for } i\in\{1, \ldots, d\}\;,
	$$
$\text{where \textbf{1)} for any } i \text{ the step-sizes }\{\alpha_t(i)\}_t\text{ are tapering, \textbf{2)}
the noise }\{\eta_t\}_t\text{ checks for any }t\in\mathbb{N}:\\$
$$
\begin{aligned}
\big\|\mathbb{E}\big[\eta_t \big\vert \Delta_{1:t-1} \big] \big\|_\infty&\leq \gamma \|\Delta_t\|_\infty \text{ with }  \gamma\in[0, 1)\; , \\
			\mathbb{E}\big[\|\eta_t\|_\infty^2 \big \vert \Delta_{1:t-1} \big] &\leq c(1+\|\Delta_t\|_\infty)^2 \; .
\end{aligned}
$$
$\text{Then the sequence }\{\Delta_t\}_t \text{ converges almost surely to 0.}$
{{< /boxed >}}
This theorem provides an extension to the classical Robbins-Monro algorithm when there is a small
(contracting) bias. By setting $\Delta\_t = q\_t - q\_\lambda^\star$ and applying said theorem, we
obtain that the sequence of values maintained by the Q-learning algorithm converges almost surely to $q\_\lambda^\star$.
This is true, however, up to one last assumption: we need the behavioral policy $\pi\_\beta$
to explore the MDP "enough". Concretely, this means visiting every pair $(s, a) \in \mathcal{S}\times\mathcal{A}$
infinitely often such that all q-values can converge. Then, we have:
$$
\forall s, a \in \mathcal{S}\times\mathcal{A}, \quad q\_t(s, a) \overset{\text{a.s}}{\longrightarrow} q\_\lambda^\star(s, a) \\; .
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Let's briefly discuss how to apply the aforementioned theorem to prove the almost-sure convergence of
the Q-learning algorithm. (We will take one or two shortcuts for the sake of conciseness -- see below for references to rigorous demonstrations.)
Let $\Delta\_t(s, a) := q\_\lambda^\star(s, a) - q\_t(s, a)$, for any pair $(s, a)\in\mathcal{S}\times\mathcal{A}$.
We will show that defined as such, $\Delta\_t$ checks the contracting bias property (the bounded variance is left as an exercise).
Clearly, for $(s, a) \neq (s_t, a_t)$ we have $\Delta\_{t+1}(s, a) = \Delta\_t(s, a)$. 
We can therefore safely set $\eta\_t(s, a) = 0$ and define $\alpha_t(s, a):=0$.
Furthermore, with $s^\prime\sim p(\cdot\vert s\_t, a\_t)$, observe that:
$$
\begin{aligned}
\Delta\_{t+1}(s\_t, a\_t) &= q\_\lambda^\star(s\_t, a\_t) - q\_{t+1}(s\_t, a\_t)\\;, & \\\
&= q\_\lambda^\star(s\_t, a\_t) - q\_t(s\_t, a\_t) - \alpha\_t\big(r(s\_t, a\_t) + \lambda \max\_{a^\prime}q\_{t}(s^\prime, a^\prime) - q_t(s\_t, a\_t)\big)  \\;, \\\
&\overset{(i)}{=} \Delta\_{t}(s\_t, a\_t) - \alpha\_t \left\\{q\_\lambda^\star(s\_t, a\_t) - q\_t(s\_t, a\_t) - \lambda\mathbb{E}\_{s^\prime\sim p(\cdot\vert s\_t, a\_t)}[\max_{a^\prime} q\_\lambda^\star(s^\prime, a^\prime)] + \lambda\max\_{a^\prime}q\_{t}(s^\prime, a^\prime)\right\\}\\;, \\\
&= \Delta\_t (s\_t, a\_t) + \alpha\_t \left\\{\eta\_t(s\_t, a\_t) - \Delta\_t(s\_t, a\_t)\right\\}\\;,
\end{aligned}
$$
where $\alpha\_t(s\_t, a\_t) = \alpha\_t$ and 
$$\eta\_t(s\_t, a\_t) := \lambda \left(\mathbb{E}\_{s^\prime\sim p(\cdot\vert s\_t, a\_t)}[\max\_{a^\prime} q\_\lambda^\star(s^\prime, a^\prime)] -  \max\_{a^\prime}q\_{t}(s^\prime, a^\prime)\right)\\; .$$ 
We used in $\text{(i)}$ the fact that $q\_\lambda^\star = \mathcal{T}\_\lambda^\star(q\_\lambda^\star)$. Overall, we have that:
$$
\begin{aligned}
\|| \mathbb{E}\left[\eta\_t \middle\vert \eta\_{1:t-1}\right] \||_\infty &= \lambda \vert \mathbb{E}\_{s^\prime\sim p(\cdot\vert s\_t, a\_t)}[\max\_{a^\prime} q\_\lambda^\star(s^\prime, a^\prime)] - \max\_{a^\prime} q_t(s^\prime, a^\prime)\vert \\;, \\\
&\leq \lambda \max\_{s^\prime} \vert \max\_{a^\prime} q\_\lambda^\star(s^\prime, a^\prime) - \max\_{a^\prime} q\_t(s^\prime, a^\prime)\vert \\;.
\end{aligned}
$$
Denote $a\_1 \in \argmax\_{a^\prime}q\_\lambda^\star(s^\prime, a^\prime)$ and $a\_2 \in \argmax\_{a^\prime}q\_t(s^\prime, a^\prime)$. Assume that 
$q\_\lambda^\star(s^\prime, a\_1) \geq q\_t(s^\prime, a\_2)$; then:
$$
\begin{aligned}
0 \leq q\_\lambda^\star(s^\prime, a\_1) - q\_t(s^\prime, a\_2) &\leq q\_\lambda^\star(s^\prime, a\_1) -  q\_t(s^\prime, a\_2)\\;,\\\
&\leq \max\_{a^\prime}\\{q\_\lambda^\star(s^\prime, a^\prime) - q\_t(s^\prime, a^\prime)\\} \\;, \\\
& \leq  \max\_{a^\prime}\\{\vert q\_\lambda^\star(s^\prime, a^\prime) - q\_t(s^\prime, a^\prime)\vert \\}
\end{aligned}
$$
Overall, we have $\vert q\_\lambda^\star(s^\prime, a\_1) - q\_t(s^\prime, a\_2) \vert \leq \max\_{a^\prime}\\{\vert q\_\lambda^\star(s^\prime, a^\prime) - q\_t(s^\prime, a^\prime)\vert \\}$.
Repeating a similar argument when $q\_\lambda^\star(s, a\_1) \leq q\_t(s^\prime, a\_2)$ yields that $\max\_{a^\prime} q\_\lambda^\star(s^\prime, a^\prime) - \max\_{a^\prime} q\_t(s^\prime, a^\prime) \leq \max\_{a^\prime} \vert \vert q\_\lambda^\star(s^\prime, a^\prime) - q\_t(s^\prime, a^\prime) \vert$.
Overall, we proved that:
$$
\begin{aligned}
\|| \mathbb{E}\left[\eta\_t \middle\vert \eta\_{1:t-1}\right] \||\_\infty &\leq \lambda \max\_{s^\prime}\max\_{a^\prime} \vert q\_\lambda^\star(s^\prime, a^\prime)-q\_t(s^\prime, a^\prime)\vert \\;, \\\
&= \lambda \lVert q\_\lambda^\star - q\_t \rVert\_\infty \\; \\\
&< \lVert \Delta \rVert\_\infty \\; .
\end{aligned}
$$
A similar proof can be conducted for $\eta$^\primes second-order moment (left as an exercise), finishing the proof.
{{% /toggle_block %}}


{{% toggle_block background-color="#CBE4FE" title="Note - Bibliographical references" default-display="none"%}}
The interested reader can refer to [\[Watkins & Dayan, 1992\]](https://link.springer.com/article/10.1007/BF00992698)
for the original proof of Q-learning's almost-sure convergence. Another line of proof, closer to the stochastic approximation
method covered herein-before is conducted by [\[Tsitsikls, 1994\]](http://web.mit.edu/jnt/www/Papers/J052-94-jnt-q.pdf).
Finally, [\[Borkar & Meyn, 2000\]](https://epubs.siam.org/doi/abs/10.1137/S0363012997331639?journalCode=sjcodc) prove the same
result via Lyapunov stability of ordinary differential equations. 
{{% /toggle_block %}}




## Function approximation

Stochastic approximation and the Q-learning algorithm release us from having the precise knowledge 
of the transition kernel $p$ -- it now suffices to be able to query it for new samples.
Practical implementation of Q-learning is, however, still limited by its **1)** memory footprint and **2)** generalisation abilities.

Regarding **1)**, the main culprits are the q-values themselves: there are $\vert \mathcal{S}\vert \times \vert \mathcal{A} \vert$ of them, and
many relevant environments have the bad idea of coming with vast state spaces (_i.e._ $\vert \mathcal{S} \vert \gg 1$)
-- if not infinite ($\mathcal{S} = \mathbb{R}^d$).
When it comes to **2)**, observe that the tabular representation of the q-function does not allow for permeability between states, since we have
no way of saying that two states are "close". 
Equipping the state space with some structure would allow transferring knowledge acquired in one state to similar ones,
effectively reducing Q-learning's sample complexity. 

The typical way forward is to let go of the tabular representation of the q-values to adopt a parametric one.
For concreteness, assume the existence of a feature map
$
	\phi : \\, \mathcal{S}\times\mathcal{A} \mapsto \mathbb{R}^d \\
$ where $d \ll \vert \mathcal{S} \vert$. We will search for $q\_\lambda^\star$ (or at least a good approximation of it)
within the class of linear functions:
$$
\mathcal{F} := \\{ (s, a) \mapsto \theta^\top\phi(s, a), \\, \theta \in \Theta\\}
$$
where $\Theta$ is a compact subset of $\mathbb{R}^d$. Below we'll denote $q\_\theta$ the function that assigns
$\theta^\top\phi(s, a)$ to the couple $(s, a)\in\mathcal{S}\times\mathcal{A}$. 
Now comes a design choice: what is a "good" $\theta^\star \in \Theta$ to represent $q\_\lambda^\star$?
Explicitly enforcing a small representation error $\lVert q\_{\theta^\star} - q\_\lambda^\star \rVert_\infty$
is of course unfeasible, as this would require the very knowledge of $q\_\lambda^\star$. An implicit way to achieve
this kind of control is by minimizing the so-called Bellman residuals:
$$
\tag{2}
	\theta^\star \in \argmin\_\theta \lVert q\_\theta - \mathcal{T}\_\lambda^\star (q\_\theta)\rVert\_\infty \\; .
$$
Observe that if $q\_\lambda^\star \in \mathcal{F}$, then $q\_{\theta^\star} = q\_\lambda^\star$ -- the Bellman residual of $q\_\lambda^\star$ is 0. Conversely, we can have a bound on our approximation error, which
naturally involves the best-in-class approximation error of $\mathcal{F}$:
$$
\lVert q\_{\theta^\star} - q\_\lambda^\star \rVert\_\infty \leq \frac{1+\lambda}{1-\lambda} \min_{q\_\theta \in \mathcal{F}} \lVert q\_\theta - q\_\lambda^\star \rVert\_\infty \\; .
$$


{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Observe that:
$$
\begin{aligned}
\lVert q\_{\theta^\star} -q\_\lambda^\star\rVert\_\infty &\leq \lVert \mathcal{T}\_\lambda^\star(q\_\lambda^\star) - \mathcal{T}\_\lambda^\star(q\_{\theta^\star})\rVert\_\infty + \lVert q\_{\theta^\star} - \mathcal{T}\_\lambda^\star(q\_{\theta^\star})\rVert\_\infty\\;, &(\mathcal{T}\_\lambda^\star(q\_\lambda^\star)=q\_\lambda^\star) \\\
&\leq \lambda \lVert q\_{\theta^\star} -q\_\lambda^\star\rVert\_\infty + \lVert q\_{\theta^\star} - \mathcal{T}\_\lambda^\star(q\_{\theta^\star})\rVert\_\infty\\;. &(\mathcal{T}\_\lambda^\star\\; \text{contracts})
\end{aligned}
$$
Hence:
$$
\begin{aligned}
\lVert q\_{\theta^\star} -q\_\lambda^\star\rVert\_\infty &\leq (1-\lambda)^{-1} \lVert q\_{\theta^\star} - \mathcal{T}\_\lambda^\star(q\_{\theta^\star})\rVert\_\infty\\;, \\\
&= (1-\lambda)^{-1} \min\_\theta \lVert q\_{\theta} - \mathcal{T}\_\lambda^\star(q\_{\theta})\rVert\_\infty\\;, \\\
&\leq (1-\lambda)^{-1} \min\_\theta  \\{ \lVert q\_{\theta} - q\_\lambda^\star\rVert\_\infty + \lVert  q\_\lambda^\star - \mathcal{T}\_\lambda^\star(q\_{\theta})\rVert\_\infty \\}\\;, \\\
&\leq (1+\lambda)(1-\lambda)^{-1} \min\_\theta\lVert q\_{\theta} - q\_\lambda^\star\rVert\_\infty\\; .
\end{aligned}
$$

{{% /toggle_block %}}

The objective $\text{(2)}$ is therefore a promising candidate for finding meaningful approximations of $q\_\lambda^\star$ supported by a low-dimensional parameter. 
However, it is not continuously differentiable and does not easily undergo gradient-based optimisation. 
Instead, we will pursue an $\ell_2$ objective, more amenable to numerical optimisation techniques:
$$
\theta\_* \in \argmin\_{\theta} \\, \lVert q\_\theta - \mathcal{T}\_\lambda^\star(q\_\theta) \rVert\_2 \\; .
$$

Of course, at this point, if we judged that $\vert \mathcal{S}\vert$ was large enough to motivate function approximation,
surely we cannot plan on using the Bellman operator $\mathcal{T}\_\lambda^\star$ -- it costs roughly $\mathcal{O}(\vert\mathcal{S}\vert^2\cdot\vert\mathcal{A}\vert)$ operations
to compute. It's time for stochastic approximation to kick in again. Materialising the full $\ell\_2$-norm 
is also questionable: instead, we have to settle for minimising the Bellman residuals over a much smaller set of 
state and actions. Concretely, for a given set $(s\_1, a\_1, s^\prime\_1, \ldots, s\_n, a\_n, s^\prime\_n)$, we compute $\theta\_\star \in \argmin\_\theta J(\theta)$ where:
$$
\begin{aligned}
J(\theta) &:=\sum\_{s, a\in\mathcal{D}} (q\_\theta(s, a) - \hat{\mathcal{T}}\_\lambda (q\_\theta)(s, a))^2 \\; , \\\
&= \sum\_{s, a\in\mathcal{D}} (q\_\theta(s, a) - r(s, a) - \lambda \max\_{a^\prime\in\mathcal{A}} q\_\theta(s^\prime, a^\prime) )^2\\\;. & (s^\prime\sim p(\cdot\vert s, a))
\end{aligned}
$$


### Fitted Q-iterations
This optimisation program $\text{(4)}$ is still somewhat funny looking -- namely because of the term $\max\_{a^\prime\in\mathcal{A}} q\_\theta(s^\prime, a^\prime)$.
One way to resolve this is to use a reference parameter $\theta\_\text{ref}$ to produce said targets. 
Concretely, we would be interested in computing $\theta\in\argmin\_\theta J(\theta, \theta\_\text{ref})$ where:
$$
J(\theta, \theta\_\text{ref}) =\sum\_{(s, a, s^\prime)\in\mathcal{D}} (q\_\theta(s, a) - r(s, a) - \lambda \max\_{a^\prime\in\mathcal{A}} q\_{\theta\_\text{ref}}(s^\prime, a^\prime) )^2
$$
Of course, to provide reasonable candidate for our fixed-point problem, one must have $\theta\_\text{ref} \approx \theta$.
The fitted Q-iteration algorithm (introduced by [\[Ernst, 2005\]](https://jmlr.org/papers/volume6/ernst05a/ernst05a.pdf) with tree-based function approximation)
leverages this rationale, by iteratively applying:
$$
\tag{3}
\begin{aligned}
	\theta\_{t+1} &= \mathcal{T}\_\lambda^\text{FQ}(\theta, \theta\_t) \\;, \\\
				  &= \argmin\_\theta J(\theta, \theta\_t) \\; .
\end{aligned}
$$
For our linearly parametrised example $q\_\theta(\cdot) = \theta^\top\phi(\cdot)$, this boils down
to solving a $d$-dimensional linear system. For completeness, we give below some pseudocode for the
resulting linearly fitted q-iterations. The action selection process is typically $\varepsilon$-greedy:
$$
	a \sim \pi\_\theta^\varepsilon(\cdot\vert s) \text{ where } \pi\_\theta^\varepsilon(a\vert s) = \left\\{\begin{aligned} 1 - \varepsilon &\text{ if } a\in\argmax\_{a^\prime} q\_\theta(s, a^\prime)\\;,\\\ \frac{\varepsilon}{\vert \mathcal{A}\vert - 1} &\text{ otherwise.}\end{aligned}\right.
$$


{{< pseudocode title="Linear fitted Q-iterations" >}} 
$\textbf{init } \theta_1\in\mathbb{R}^{d}, \text{ max. iteration T}, \text{batch size }n, \, \mathcal{D} \leftarrow \emptyset\\$
$\textbf{for } t = 1, \ldots, T-1:\\$
$\qquad \text{Start from } s \in \mathcal{S}\\$
$\qquad \textbf{for } n \text{ steps}\\$
$\qquad \qquad \text{pick } a\sim\pi_{\theta_t}^\varepsilon(\cdot\vert s),\; \text{observe }s^\prime \sim p(\cdot\vert s, a)  \\$
$ \qquad \qquad \mathcal{D} \leftarrow \mathcal{D} \cup (s, a, s^\prime), \; s \leftarrow s^\prime\\$
$\qquad \textbf{end for}\\$
$ \qquad \text{Form targets } \ell(s, a, s^\prime) \leftarrow r(s, a) + \lambda\max_{a^\prime}q_{\theta_t}(s^\prime, a^\prime) \text{ for } (s, a, s^\prime)\in\mathcal{D}\\$
$ \qquad \text{Fit next parameter:} \\$
$$
\theta_{t+1} \leftarrow \big(\sum_{(s, a, s^\prime)} \phi(s, a)\phi(s, a)^\top\big)^{-1}\sum_{s, a, s^\prime, \ell} \phi(s, a)\ell(s, a, s^\prime) \; .
$$
$\textbf{end for}\\$
$\textbf{return } \theta_T$
{{< /pseudocode >}}

<br>
<br>


### The DQN breakthrough

The Q-fitted iterations could, in theory, support deep neural networks function approximation.
(Actually, people have made it work in this setting, but mostly for toy environments).
The main appeal is, of course, to search within much richer function classes without having to undertake
painful feature engineering when, say, the state space $\mathcal{S}$ is the visual rendering of some Atari game.
One can list a few obstacles standing in the road for neural-networks powered Q-learning, partially solved by the original [DQN paper](https://www.nature.com/articles/nature14236?).

The first one is **1)** computational: solving Eq. $\text{(3)}$ at each round is challenging to say the least -- because
there is no longer a closed form, we are talking about solving a non-convex program at each round. 
Instead, one can settle for a one-step stochastic gradient descent update.
When considering a unique sampled experience $(s, a, r, s^\prime)$ this would look like $\theta\_{t+1} = \theta\_t - \alpha \Delta\theta\_t$ where:
$$
\tag{4}
\begin{aligned}
	\Delta\theta\_{t} &\propto \nabla\_{\theta}\left(q\_\theta(s, a) - r - \lambda\max\_{a^\prime}q\_{\theta\_t}(s^\prime, a^\prime)\right)^2\Big\vert\_{\theta=\theta\_t}\\;, \\\
  				  &\propto \left(q\_{\theta\_t}(s, a) - r - \lambda\max\_{a^\prime}q\_{\theta\_t}(s^\prime, a^\prime)\right) \nabla\_{\theta} q\_\theta(s, a)\Big\vert\_{\theta=\theta\_t} \\; .
\end{aligned}
$$
Now, this update should be computed on a minibatch of experience instead of a single one.
Drawing this minibatch from the freshest bunch of experience is delicate, as those are highly correlated by nature (temporal correlation). 
This could lead to dangerous spurious updates. 
The DQN paper solves this by sampling experience uniformly at random within $\mathcal{D}$, which will now be re-baptised the _replay buffer_. 

We just touched to the second main challenge, which is tied to **2)** stability. 
Because of their representational capacity, training neural networks for finding fixed-points is tough: as we just saw, this involves having the neural network provides its own regression targets. 
Since said neural network keeps on getting updated, this induces tracking some non-stationary targets.
Therefore, bluntly applying $\text{(4)}$ leads to oscillations, divergences, and overall disappointing performance.
The DQN authors fixed this by introducing the concept of _target networks_. Briefly, they froze the underlying networks for a few iterations, enforcing stationarity in the targets and henceforth reducing oscillations.
Concretely, this means maintaining another set of parameter $\theta\_t^{-}$ that only get periodically updated throughout the training:
$$
\begin{aligned}
	\theta\_{t+1}^{-} &=  \theta\_{t+1} \text{ if } t\equiv 0 \pmod{\tau} \\;, \\\
&= \theta\_{t}^{-} \text{ otherwise},
\end{aligned}
$$
which will provide the q-targets (notice the $\color{orange}\text{difference}$ with Eq. (4)):
$$
\tag{5}
	\Delta\theta\_{t}   = \left(q\_{\theta\_t}(s, a) - r - \lambda\max\_{a^\prime}q\_{\color{orange}\boldsymbol{\theta^{-}\_t}}(s^\prime, a^\prime)\right) \nabla\_{\theta} q\_\theta(s, a)\Big\vert\_{\theta=\theta\_t} \\; .
$$

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
Most recent implementation of target networks prefer a smoother update rule, and have $\theta\_t^{-}$
follow a slow exponential moving-average of $\theta\_t$:
$$
\theta^{-}\_{t+1} = (1-\kappa) \theta\_t^{-} + \kappa \theta\_{t+1} \\;, 
$$
with $\kappa \ll 1$. 
{{% /toggle_block %}}

That's it--starting from first principles (qVI) we can now write down some pseudo-code for DQN.
Below, the dimension $d$ refers to the degrees of freedom (number of weights and biases) of some neural-network architecture.
Similarly to above, we collect data using the $\varepsilon$-greedy strategy $\pi\_\theta^\varepsilon$.

{{< pseudocode title="DQN" >}} 
$\textbf{input } \theta_1\in\mathbb{R}^{d}, \; \text{ max. iteration T}, \text{mini-batch size }n, \text{ learning-rate } \alpha, \text{ update period } \tau\\$
$\textbf{init }\theta_1^{-}\leftarrow \theta_1, \,\text{ initial state } s_1\in\mathcal{S}, \,  \mathcal{D} \leftarrow \emptyset\\$
$\textbf{for } t = 1, \ldots, T-1:\\$
$\qquad \text{Play } a_t \sim \pi_{\theta_t}^\varepsilon(s_t), \text{ observe } r_t \text{ and } s_{t+1}.\\$
$\qquad \text{Update the replay buffer } \mathcal{D} \leftarrow \mathcal{D} \cup (s_t, a_t, r_t, s_{t+1})\\$
$\qquad \text{Sample mini-batch of experience } \mathcal{D}_n \subset \mathcal{D}\\$
$\qquad \text{Update network: }\\$
$$
\theta_{t+1} \leftarrow \theta_t - \frac{\alpha}{n} \sum_{(s, a, r, s^\prime)\in\mathcal{D}_n} \big[q_{\theta_t}(s, a) - r - \lambda \max_{a^\prime}q_{\theta_t^-}(s^\prime, a^\prime)\big]^2 \nabla_\theta q_\theta(s, a)\big\vert_{\theta=\theta_t} 
$$
$\qquad \textbf{if } t \equiv 0 \pmod{\tau}\\$
$\qquad \qquad \text{ Update target network } \theta^{-}_{t+1} \leftarrow \theta_{t+1}\\$
$\qquad \textbf{else }\\$
$\qquad \qquad \text{Freeze target network } \theta_{t+1}^- \leftarrow \theta_t^- \\$
$\qquad \textbf{fi}\\$
$\textbf{end for}\\$
$\textbf{return } \theta_T$
{{< /pseudocode >}}

<br>
<br>

### Refinements

That is not the end of the story. Since the original DQN paper, 
myriad of improvements got thrown into the mix. The [Rainbow](), [Agent57]() and [BBF]() papers
all, in their time, combined the most relevant improvements to yield ever better versions of DQN (at least, from the Atari benchmark perspective.)
The goal here is not to cover all of them tricks; 
however, we will quickly brush over some that will (maybe) one day be topics of blog-post of their own 🤞. 

##### Double Q-learning
This is about fighting the over-estimation bias inherent to Q-learning. 
Because of random realisations, or unlucky function approximation, it is _possible_ that our learning algorithm over-estimates, at one point, some Q-value.
Say, for some $s^+, a^+\in\mathcal{S} \times \mathcal{A}$ that
$
q\_t(s^+, a^+) > q\_\lambda^\star(s^+, a^+)\\; .
$
If $a^+ \in \argmax\_a q\_t(s^+, a)$ this will directly back-propagate our whole set of q-values since the Q-learning update at some state-action pair $(s, a)$ precessing $s^+$ writes:
$$
q\_{t+1}(s, a) = (1-\alpha)q\_t(s, a) + \alpha\left(r(s, a)+\lambda q\_t(s^+, a^+)\right)\\; .
$$
To avoid propagating over-estimated values, the original [double Q-learning](https://proceedings.neurips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf) paper suggests introducing two seperates
estimators $q\_t^1$ and $q\_t^2$, using distincts set of samples.
When updating $q\_t^1$, the target value is obtained by using $q\_t^1$ on top of an action picked by $q\_t^2$ (and vice-versa).
Concretely:
$$
q^1\_{t+1}(s, a) = (1-\alpha)q^1\_t(s, a) + \alpha\left(r(s, a)+\lambda q^1\_t(s^+, \argmax\_a q^2\_t)\right)\\; .
$$
Via this decoupling between selection and predection, and since $q^1$ and $q^2$ were trained on different samples, we can hope to lessen over-estimation.

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
It is important to insist that over-estimation is not a problem in-itself (because of its conceptual ties with _optimism_, we could even argue the opposite). 
However, reducing it has been shown to come with convincing empirical improvements. 
{{% /toggle_block %}}

Maintaining two sets of neural networks can be (was?) judged too cumbersome -- in DQN-like approach, the online network replaces $q\_2^t$ to select the maximising action.
The update rule (5) becomes:
$$
\tag{5}
	\Delta\theta\_{t}  \left(q\_{\theta\_t}(s, a) - r - \lambda q\_{\theta^{-}\_t}(s^\prime, \argmax_{a^\prime} q\_{\theta\_t}(s^\prime, a^\prime))\right) \nabla\_{\theta} q\_\theta(s, a)\Big\vert\_{\theta=\theta\_t} \\; .
$$

##### Prioritized replay buffers
Prioritized experience replay has now become a central component of modern value-based method.s 
The idea takes its roots in the algorithmic concept of Asynchronous Value Iteration--more precisely the Prioritized Value Iteration algorithm.
Without going into too many details (that would be for another post), the main idea is to replace the uniform sampling in $\mathcal{D}$
by some sampling rule that enforces a uniform _error_ through $\mathcal{D}$.
Concretely, the [prioritised experience replay](https://arxiv.org/pdf/1511.05952.pdf) paper encourages over-sampling transitions $(s, a, r, s')$ which
come with a large online Bellman residual error $\Delta$, where:
$$
\Delta := \left\vert q\_{\theta\_t}(s, a) - r(s, a) - \lambda \max\_{a^\prime} q\_{\theta\_t}(s^\prime, a^\prime)\right\vert \\; .
$$


##### Multi-step learning
This of course touches to the never-ending bias vs. variance trade-off. 
Recall the fixed-point equation solved by $q\_\lambda^\star$; for any $s, a\in\mathcal{S}\times\mathcal{A}$:
$$
\begin{aligned}
q\_\lambda^\star(s, a) &= \mathcal{T}\_\lambda^\star(q\_\lambda^\star)(s, a) \\;, \\\
&= r(s, a) + \lambda\mathbb{E}\_{s^\prime}\left[\max\_{a^\prime} q\_\lambda^\star(s', a')\right] \\; .
\end{aligned}
$$
Iterating twice, we obtain:
$$
\tag{6}
\begin{aligned}
q\_\lambda^\star(s, a) &= (\mathcal{T}\_\lambda^\star)^2(q\_\lambda^\star)(s, a) \\;, \\\
&= r(s, a) + \lambda\mathbb{E}\_{s^\prime}\left[\max\_{a^\prime} \left\\{r(s', a') + \lambda\mathbb{E}_{s^{\prime\prime}}\left[\max\_{a^{\prime\prime}}q\_\lambda^\star(s^{\prime\prime}, a^{\prime\prime})\right]\right\\}\right] \\; ,
\end{aligned}
$$
and so on. Now, unlike many TD($\lambda$)-like methods, Q-learning is off-policy -- we don't (can't) collect
data according to $\pi^\star$ (that would make things a bit too easy). 
Computing an n-step target based on (6), thanks to data collected by an $\varepsilon$-greedy policy is tricky. 
Instead of delving into Watkin's and Peng's Q($\lambda$) methods, the descendants of DQN adopt a pragmatic (but somewhat unprincipled)
approach. For $n\in\mathcal{N}$ and a chunk of trajectory $(s\_1, a\_1, \ldots, a\_{n}, s\_{n+1})$ they compute the following target:
$$
\sum\_{i\leq n} \lambda^{i-1} r(s\_i, a\_i) + \lambda^n \max\_{a} q\_t(s\_{n+1}, a) \\; ,
$$
given us the update rule:
$$
\tag{5}
	\Delta\theta\_{t}  \left(q\_{\theta\_t}(s, a) - \sum\_{i\leq n} \lambda^{i-1} r(s\_i, a\_i) - \lambda^n \max\_{a} q\_t(s\_{n+1}, a)\right) \nabla\_{\theta} q\_\theta(s, a)\Big\vert\_{\theta=\theta\_t} \\; .
$$
Small values of $n$ (_e.g._ $n=5$) have been shown to empirically outperform the 1-step targets.
Most recent approaches have $n$ vary throughout the learning; using large values (high variance but low bias) at first, 
then smaller once the q-values estimates are more reliable (low variance and reasonable bias). 

{{% toggle_block background-color="#CBE4FE" title="Note" %}}
Proper (read, with guarantees) multi-step off-policy evaluation will be the topic of another blog-post.
In the meantime, [this](http://incompleteideas.net/book/ebook/node78.html) is some good material to refresh some memories about $Q(\lambda)$.
{{% /toggle_block %}}
