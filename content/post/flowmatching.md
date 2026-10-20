+++
author = "Louis Faury"
title = "Flow Matching from the Ground Up"
date = "2026-09-26"
+++

Simulation-based generative models are all the rage.
In particular, flow matching now dominates the image and video generation landscape.
Because it tolerates few simulation steps well, it has also been widely adopted
in applications sensitive to inference time (mostly, robotics).
Like we did for denoising diffusion models, this post studies flow matching from first principles.

<!--more-->


{{< infoblock>}}
$\quad$ This post overlaps with another, focused on  
ODE <a href="../ode" style="text-decoration:none; color:#0074aa;">solvers</a>. 
The latter is a good complementary read.
{{< /infoblock >}}

{{< infoblock>}}
$\quad$ Also relevant is the post on 
<a href="../diffusion" style="text-decoration:none; color:#0074aa;">denoising diffusion probabilistic models</a>.
{{< /infoblock >}}


## Preliminaries

This section introduces the necessary tools to understand and describe flow matching.
In particular, we will focus on ordinary differential equations (ODEs).
In the next section, we will see that colloquially, flow matching is about learning an ODE's vector field,
and letting a numerical solver generate samples by simulating some particle's dynamics. 
But first, we will need some foundations.

### Cauchy problems

Let $f:\mathbb{R}^d\times[0,1]\mapsto\mathbb{R}^d$ 
describe a family of vector fields indexed by the time variable $t\in[0,1]$ 
(these bounds are for convenience; any bounded interval of $\mathbb{R}$ would do).
Let $y:[0,1]\mapsto\mathbb{R}^d$ describe the motion of a 'particle' which
follows the instantaneous vector field $f(\cdot, t)$. 
This motion is described by a non-autonomous ODE;
when paired with an initial position $y\_0$ in $\mathbb{R}^d$, it forms a _Cauchy problem_:
$$
\tag{1}
\frac{dy}{dt}(t) = f\big(y(t), t\big)\\; \text{ for any } t\in[0,1] \text{ and } y(0)=y\_0\\;.
$$

The existence and uniqueness of a solution $y$ to a given Cauchy problem is not guaranteed, and essentially depends
on the smoothness of the vector fields. 
This is captured by the Cauchy-Lipschitz theorem, which states some sufficient conditions for a Cauchy problem
to be well-posed. 
We state a (weak) version of said theorem:

{{< boxed title="Cauchy-Lipschitz" >}}
$\qquad\qquad\qquad\qquad  \text{Let } f \text{ be continuous over } \mathbb{R}^d\times[0,1] \text{ and } f(\cdot, t) \text{ } L\text{-Lipschitz for all } t\in[0,1]. \\ \qquad\qquad\qquad\qquad \text{Then
(1) has a unique solution.}
$
{{< /boxed >}}

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
We have already provided a proof for the autonomous case in [our ODE solver post](../ode), 
based on a fixed-point argument. 
This argument extends directly to the non-autonomous case, since all vector fields 
$\\{f(\cdot, t)\\}\_t$ share the same Lipschitz constant $L$.
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

### Flow

From now on, the vector fields $f$ respect the conditions of the above theorem.
In such conditions, each Cauchy problem admits a unique solution. 
It is quite convenient to have a shorthand for the state of the solution of any given Cauchy problem
at some given time $t$.
That's the point of the _flow_ $\psi: \mathbb{R}^d \times [0,1]\mapsto\mathbb{R}^d$ where, concretely, 
$\psi(y\_0, t)=y(t)$ with $y$ the solution of the ODE with initial condition $y(0)=y\_0$.
In particular, we have that for any $y\_0\in\mathbb{R}^d$ and $t\in[0,1]$:
$$
\frac{\partial \psi}{\partial t}(y\_0, t) = f(\psi(y\_0, t), t) \quad\text{ and }\quad \psi(y\_0, 0) = y\_0\\;.
$$

A consequence of the Cauchy-Lipschitz theorem is that $\psi(\cdot, t)$ is an invertible map
from $\mathbb{R}^d$ to itself. 
In practical terms, this means that trajectories starting from different initial positions do not collide; two 
particles that started from different positions cannot sit at the same place at the same time.
When $f$ is continuously differentiable, one can further prove that the flow is a _diffeomorphism_: a continuously differentiable, 
invertible map from $\mathbb{R}^d$ to itself, whose inverse is also continuously differentiable (we will not prove this result here). 

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
The proof studies a Cauchy problem with a different boundary condition. With $\tau\in[0,1]$ and $\bar{y}\in\mathbb{R}^d$:
$$
\tag{$\star$}
\frac{dy}{dt}(t) = f\big(y(t), t\big)\\; \text{ for any } t\in[0,\tau] \text{ and } y(\tau)=\bar{y}\\;.
$$
Clearly, the Cauchy-Lipschitz theorem holds for (1) iff it holds for ($\star$).

The flow's injectivity is proven as follows. Fix $\tau$ and assume that there exist $y\_0, y\_0^\prime$
such that $\psi(y\_0, \tau) = \psi(y\_0^\prime, \tau)=\bar{y}$. Both $\psi(y\_0, \cdot)$ and $\psi(y\_0^\prime, \cdot)$ are solutions
of ($\star$); by uniqueness of the solution, they are therefore equal on $[0,\tau]$ and in particular at $t=0$. 
This proves that $y\_0 = y\_0^\prime$, and the injectivity of $\psi$.

Surjectivity follows a similar argument; since a solution for ($\star$) exists for any $\bar{y}$, we can find a predecessor for any point of $\mathbb{R}^d$ under ($\star$).
One can easily check that this provides a starting condition $y\_0$ for (1) that will check $\psi(y\_0, \tau)=\bar{y}$. This is true
for any $\bar{y}$ and proves that the flow is a surjective map. 

<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

{{< flow_animation caption="Fig1. The flow of a two-dimensional (autonomous) ODE." >}}


### Continuity equation

The flow $\psi$ describes the evolution of each particle under the vector fields $f(\cdot, t)$. 
Below, we will be interested in how densities change under said fields.
More precisely, given some probability measure $p$, we will want to know the push-forward 
$p\_t := [\psi(\cdot, t)]\_\sharp p$.
For convenience, we will overload probability measures with their densities
and denote $p(\cdot, t)$ the density of $p\_t$.
(Observe that $p\_t$ does have a density, since $\psi(\cdot, t)$ is a diffeomorphism.)
It is characterised by a differential equation known as the _continuity equation_.


{{< boxed title="Continuity equation" >}}
$\qquad\qquad\qquad\qquad\quad \; \text{ For any }t\in[0,1] , \, p(\cdot, t) \text{ is the unique solution to the partial differential equation}:$
$$
\tag{2}
\frac{\partial p}{\partial t}(y, t) + \sum_{i=1}^d \frac{\partial (p f_i)}{\partial y_i}(y, t) = 0,\quad  \forall y\in\mathbb{R}^d\;.
$$
{{< /boxed >}}

The continuity equation is essentially a statement of mass preservation; the variation of mass in an (infinitesimal) volume
is explained by how much density comes in and out of said volume. 

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
There are elegant proofs of this result; this is not one of them—we will rely on brute-force computations.
Fix $t\in[0,1]$. 
The density $p(\cdot, t)$ of the push-forward of $p$ under $\psi(\cdot, t)$ is given by the change-of-variable formula; 
equivalently $p$ is the push-forward of $p(\cdot, t)$ under $\psi(\cdot, t)^{-1}$. For any $y\in\mathbb{R}^d$ this writes:
$$
p(y) = \vert \det J\psi(y, t)\vert p(\psi(y, t), t) \\;,
$$
with $J\psi$ being the flow's Jacobian matrix. 
The left-hand side does not depend on time; differentiating wrt. time therefore yields (with $\det J\psi > 0$, by continuity from $\det J\psi(y, 0) = 1$):
$$
\tag{$\star$}
 \frac{\partial \det J\psi}{\partial t}(y, t)\\,p(\psi(y, t), t) + \vert \det J\psi(y, t)\vert\left(\frac{\partial \psi}{\partial t}(y, t)^\top\frac{\partial p}{\partial y}(\psi(y, t), t) + \frac{\partial p}{\partial t}(\psi(y, t), t)\right) = 0\\;.
$$ 
Observe that by definition we have $\partial\_t \psi(y, t) = f(\psi(y, t), t)$. Dividing by $\vert \det J\psi(y, t)\vert$ (never zero since $\psi(\cdot, t)$ is a diffeomorphism) yields:
$$
 \underbrace{\vert \det J\psi(y, t)\vert^{-1} \frac{\partial \det J\psi}{\partial t}(y, t)\\,p(\psi(y, t), t)}\_{(\text{i})} + f(\psi(y, t), t)^\top\frac{\partial p}{\partial y}(\psi(y, t), t) + \frac{\partial p}{\partial t}(\psi(y, t), t) = 0\\;.
$$
Let's focus on (i) for now. Recall that for any invertible matrix $A(t)$ we have $\partial\_t \det(A(t))= \det(A(t))\text{Tr}(A^{-1}(t)\partial\_t A(t))$. 
Applying it to the first term of the above, we get:
$$
(\text{i}) = \text{Tr}\left(J\psi(y, t)^{-1}\frac{\partial J\psi}{\partial t}(y, t)\right)p(\psi(y, t), t)\\;.
$$
We can simplify this one further; indeed, for any $i, j$:
$$
\begin{aligned}
\frac{\partial }{\partial t} \frac{\partial \psi\_i}{\partial y\_j} &= \frac{\partial}{\partial y\_j}\frac{\partial  \psi\_i}{\partial t}\\;, &(\text{smoothness}) \\\
&=  \frac{\partial}{\partial y\_j}f\_i(\psi(y, t), t)\\;, &(\text{def of flow}) \\\
&= \sum\_{k=1}^d \frac{\partial f\_i}{\partial y\_k}(\psi(y, t), t) \frac{\partial \psi\_k}{\partial y\_j}(y, t)\\;.
\end{aligned}
$$
This yields that $\frac{\partial J\psi}{\partial t}(y, t) = Jf(\psi(y, t), t) \cdot J\psi(y, t)$ and therefore that:
$$
\begin{aligned}
(\text{i}) &= \text{Tr}\left(J\psi(y, t)^{-1}Jf(\psi(y, t), t) J\psi(y, t)\right)p(\psi(y, t), t)\\;,\\\
&= \text{Tr}\left(J\psi(y, t)J\psi(y, t)^{-1}Jf(\psi(y, t), t)\right)p(\psi(y, t), t)\\;, &(\text{Tr}(AB)=\text{Tr}(BA))\\\
&= \text{Tr}\left(Jf(\psi(y, t), t)\right)p(\psi(y, t), t)\\;.
\end{aligned}
$$
Putting everything back together, and denoting $z=\psi(y, t)$ as a shorthand, we get:
$$
\text{Tr}\left(Jf(z, t)\right)p(z, t) + f(z, t)^\top\frac{\partial p}{\partial y}(z, t) + \frac{\partial p}{\partial t}(z, t) = 0\\;,
$$
which is exactly the continuity equation at $z$ (expand the divergence of $pf$); since $\psi(\cdot, t)$ is surjective, it holds for all $z\in\mathbb{R}^d$. We proved that $p(\cdot, t)$ is a solution to the continuity equation.
To prove uniqueness, we walk back to ($\star$). Observe that ($\star$) characterises $p(\cdot, t)$; 
if it holds, then $\vert \det J\psi(y, t)\vert p(\psi(y, t), t)$
is constant wrt. $t$—in particular, it equals for all $t$ its value at $t=0$:
$$
\begin{aligned}
\vert \det J\psi(y, t)\vert p(\psi(y, t), t) &= \vert \det J\psi(y, 0)\vert p(\psi(y, 0), 0) \\;, \\\
&= \vert \det I\_d\vert p(y, 0) \\;, \\\
&= p(y)\\;,
\end{aligned}
$$
since $\psi(y, 0)=y$ for any $y\in\mathbb{R}^d$.
This is the change-of-variable formula; hence $p(\cdot, t)$ is the push-forward of $p$ by $\psi(\cdot, t)$, which concludes the proof.

<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}


## Flow matching
Let $p\_\star$ some distribution over $\mathbb{R}^d$ that we which to model—at least, create some $\hat{p}$ that 
resembles $p\_\star$ and that we can freely sample from. 
Similarly to denoising diffusion models, flow matching {{< refs link="fmgm,fsf" label="1,2" >}}
attempts this by "transforming noise to data".
Given some initial distribution $p\_0$, it transforms its samples by simulating their motion under a non-autonomous ordinary differential
equation (ODE). It is the vector fields of said ODE that are learned from data, so that the resulting $\hat{p}$
provides a good approximation of $p\_\star$.

{{< fm_pitch_animation caption="Fig2. Transporting $p_0$ to $p_\star$." >}}

{{% toggle_block background-color="#CBE4FE" title="Solving the ODE" default-display="none"%}}
Like almost always, we will not be able to solve the ODE in closed-form. 
Sampling through this generative process will require the use of ODE solvers; 
often, the simplest ones (_e.g._, the explicit Euler method) will be enough
for the applications we pursue here.
{{% /toggle_block %}}

From now on, we write the time index after a bar: $p(\cdot\vert t)$ denotes the density at time $t$
(it plays the role of $p(\cdot, t)$ in the previous section) and $f(\cdot\vert t)$ the vector field at time $t$.
This leaves room for extra conditioning variables, as in $p(\cdot\vert z, t)$.

### Conditional paths
We first look at the simpler problem 
of turning $p\_0$ into $\delta\_z$, the point mass located at $z\in\mathbb{R}^d$. 
How we interpolate between the two distributions is called the _conditional probability path_ $p(\cdot\vert z, t)$;
it is a sequence of distributions that only has to satisfy the boundary conditions that 
$p(\cdot\vert z, 0)=p\_0(\cdot)$ and $p(\cdot\vert z, 1)=\delta\_z(\cdot)$.
The conditional probability path is _declarative_—in the sense that it is a design choice left
to the experimenter. 

{{% toggle_block background-color="#CBE4FE" title="About point masses" default-display="none"%}}
Strictly speaking no flow can reach a point-mass since $\psi(\cdot, 1)$ is a bijection.
A rigorous treatment targets $\mathcal{N}(z, \sigma\_{\min}^2\mathbf{I}\_d)$ instead, for some small $\sigma\_{\min}>0$, 
and assumes that $p\_\star$ has bounded support.
In practice, though, using $\sigma\_{\min}=0$ works fine since neither training nor sampling evaluates the vector fields at $t=1$.
{{% /toggle_block %}}


To be relevant, one still needs to find a family of _conditional_ vector fields $\\{f(\cdot\vert z, t)\\}\_t$ that generates it—
meaning that the flow of the related ODE checks
$
[\psi(\cdot, t)]\_\sharp p\_0 = p(\cdot\vert z, t)\\;.
$
Moving forward, we will make the assumption that the conditional vector fields supporting the chosen interpolating path are known.
By the continuity equation, this means that for any $y\in\mathbb{R}^d$:
$$
\tag{3}
\frac{\partial p}{\partial t}(y\vert z, t) + \sum\_{i=1}^d \frac{\partial (pf\_i)}{\partial y\_i}(y\vert z, t) = 0\\;.
$$


### Marginal paths
Now that we can turn $p\_0$ into a point mass, we can go back to transforming it to $p\_\star$. 
Choosing a conditional interpolating path $p(\cdot\vert z, t)$ is enough, in the sense that the _marginal_ path:
$$
p(\cdot\vert t) := \int\_z p\_\star(z)p(\cdot\vert z, t)dz
$$
turns $p\_0$ into $p\_\star$. 
Indeed, it checks $p(y\vert 0)=p\_0(y)$ and $p(y\vert 1)=p\_\star(y)$.
The vector fields that support this transformation, called the _marginal_ vector fields, 
can be derived from the conditional ones. 
Indeed, we have $[\psi(\cdot, t)]\_\sharp p\_0 = p(\cdot\vert t)$ for the ODE whose
vector fields write, for any $y\in\mathbb{R}^d$:
$$
\tag{4}
f\_\star(y\vert t) = \int\_{z} f(y \vert z, t) \frac{p\_\star(z)p(y\vert z, t)}{p(y\vert t)}dz \\;.
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
This is a proof by verification; it is enough to check that for the vector fields (4) the continuity 
equation for $p(\cdot\vert t)$ is satisfied. (Indeed, the continuity equation _characterises_ the push-forward.)

Unrolling the computation, for any $y\in\mathbb{R}^d$:
$$
\begin{aligned}
\frac{\partial p}{\partial t}(y\vert t) &= \frac{\partial}{\partial t}\int\_{z}p\_\star(z)p(y\vert z, t)dz \\;, \\\
&= \int\_{z}p\_\star(z) \frac{\partial}{\partial t}p(y\vert z, t)dz\\;, &(\text{dominated convergence}) \\\
&= -\sum\_{i=1}^d \int\_{z} p\_\star(z)\frac{\partial (pf\_i)}{\partial y\_i}(y\vert z, t)dz\\;, &(\text{by (3)})\\\
&= -\sum\_{i=1}^d \frac{\partial}{\partial y\_i} \int\_{z}p\_\star(z)p(y\vert z, t)f\_i(y\vert z, t)dz \\;, \\\\
&= -\sum\_{i=1}^d \frac{\partial}{\partial y\_i}\Big[ p(y\vert t)\int\_{z}p\_\star(z)\frac{p(y\vert z, t)f\_i(y\vert z, t)}{p(y\vert t)}dz\Big ]\\;,\\\
&= -\sum\_{i=1}^d \frac{\partial (pf\_\star)}{\partial y\_i}(y\vert t)\\;.
\end{aligned}
$$
The continuity equation is verified, which concludes the proof.
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}


### Learning objective

There is one issue with (4): it requires computing an integral which involves the unknown $p\_\star$. 
Instead, we will approximate the vector fields $f\_\star$ as the solution of a regression problem.
Define, for $t\in[0,1]$:
$$
\tag{5}
\hat{f}(\cdot\vert t) := \argmin\_{\tilde f\in\mathcal{F}} \mathbb{E}\_{z\sim p\_\star}\mathbb{E}\_{y\sim p(\cdot\vert z, t)}\Big[ \big\\| \tilde f(y\vert t) - f(y\vert z, t)\big\\|^2\Big] \\;,
$$
where $\mathcal{F}$ is some hypothesis class. Observe that we are regressing directly against the (assumed known) conditional vector fields.
This is not an issue; one can show that if $f\_\star\in\mathcal{F}$, then $\hat{f}=f\_\star$. 
As (5) can easily be approximated by sampling, finding $\hat{f}$ essentially boils down to some form of stochastic gradient descent. 
We illustrate this plainly in the following section.

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
It is enough to prove that:
$$
\argmin\_{\tilde f\in\mathcal{F}} \mathbb{E}\_{z\sim p\_\star}\mathbb{E}\_{y\sim p(\cdot\vert z, t)}\Big[ \big\\| \tilde f(y\vert t) - f(y\vert z, t)\big\\|^2\Big]  = \argmin\_{\tilde f\in\mathcal{F}}\mathbb{E}\_{y\sim p(\cdot\vert t)}\Big[ \big\\| \tilde f(y\vert t) - f\_\star(y\vert t)\big\\|^2\Big]\\;. 
$$
Let's expand the left-hand side; the term inside the expectation decomposes as:
$$
\begin{aligned}
   \\| \tilde f(y\vert t) - f(y\vert z, t)\\|^2
        &=\\| \tilde f(y\vert t) - f\_\star(y\vert t)\\|^2 + \\|f\_\star(y\vert t)- f(y\vert z, t)\\|^2 + 2(\tilde f(y\vert t) - f\_\star(y\vert t))^\top(f\_\star(y\vert t)- f(y\vert z, t))\\;.\\\
\end{aligned}
$$
Since $\tilde f(y\vert t) - f\_\star(y\vert t)$ does not depend on $z$, the cross term writes:
$$
\int\_y \big(\tilde f(y\vert t) - f\_\star(y\vert t)\big)^\top \underbrace{\Big[f\_\star(y\vert t)p(y\vert t) - \int\_z f(y\vert z, t)p\_\star(z)p(y\vert z, t)dz\Big]}\_{=0 \text{ by (4)}}dy = 0\\;,
$$
where we used $\int\_z p\_\star(z)p(y\vert z, t)dz = p(y\vert t)$. For the same reason, the first term integrates against $p(y\vert t)$. Hence:
$$
\mathbb{E}\_{z\sim p\_\star}\mathbb{E}\_{y\sim p(\cdot\vert z, t)}\Big[ \big\\| \tilde f(y\vert t) - f(y\vert z, t)\big\\|^2\Big] = \mathbb{E}\_{y\sim p(\cdot\vert t)}\Big[ \big\\| \tilde f(y\vert t) - f\_\star(y\vert t)\big\\|^2\Big] + \blacksquare\\;, 
$$
where $\blacksquare$ is a constant wrt. $\tilde{f}$—which concludes the proof. 
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}


{{% toggle_block background-color="#CBE4FE" title="About conditional distributions" default-display="none"%}}
The methods described here naturally extend to modelling conditional distributions $p\_\star(\cdot\vert c)$—typically,
with $c$ some label describing some class to generate a sample from. 
The extension works well on paper; in practice, it needs some augmentation like 
classifier-free guidance {{< ref link="cfg">}} [4]{{< /ref>}} to produce satisfying results.
{{% /toggle_block %}}

## Gaussian paths

Let's take a step back to describe the workflow we have proposed so far to transform $p\_0$ to $p\_\star$:
$$
\text{choose }p(\cdot\vert z, t) \underset{(3)}{\longrightarrow} \text{ find } f(\cdot \vert z, t)\underset{(4)}{\longrightarrow} \text{ compute }f\_\star(\cdot\vert t)
$$
We have so far assumed that the second step was done for us—this is far from trivial as we do not have a _constructive_ tool for it,
only a way to _verify_ a candidate (the continuity equation). In this section, we choose a particular conditional interpolating path which
satisfies this assumption. It also births the algorithm that most actually refer to when talking about flow matching.

Let $p\_0 = \mathcal{N}(0,\mathbf{I}\_d)$ and $\mu, \sigma$ be two real-valued functions, such that  $\mu(0)=\sigma(1)=0$ and $\mu(1)=\sigma(0)=1$. Define:
$$
    \tag{6}
    p(y\vert z, t) := \mathcal{N}\big(y\vert \mu(t)z, \sigma(t)^2\mathbf{I}\_d\big)\\;.
$$

{{% toggle_block background-color="#CBE4FE" title="About denoising diffusion models" default-display="none"%}}
The 'typical' flow-matching schedule writes $\mu(t)=t$ and $\sigma(t)=1-t$; it yields so-called _straight_ interpolating paths.
Also of interest are _variance preserving_ paths, where
$\mu(t) = \sqrt{\alpha(t)}$
and $\sigma(t) = \sqrt{1-\alpha(t)}$ for some well-chosen schedule $\alpha$.
In this case, rewriting (6) accordingly:
$$
p(y\vert z, t) = \mathcal{N}(y\vert \sqrt{\alpha(t)}z, (1-\alpha(t))\mathbf{I}\_d)\\;,
$$
is reminiscent of denoising diffusion models—see eq. (3) in [this post](../diffusion). 
The relationship between flow matching and denoising diffusion models goes deeper; see 
{{< ref link="fmgm">}} [1]{{< /ref>}}{{< ref link="fmddpm">}} [3]{{< /ref>}}.
{{% /toggle_block %}}

The conditional path of (6) is generated by the conditional vector fields:

$$
\tag{7}
f(y\vert z, t) := \dot\mu(t) z + \big(y-\mu(t)z\big)\frac{\dot\sigma(t)}{\sigma(t)} \\;.
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
As we said, this is about verification; let's show that (6) and (7) check the continuity equation.
Observe that 
$$
\begin{aligned}
\frac{\partial p}{\partial t}(y\vert z, t) &= p(y\vert z, t)\frac{\partial\log p}{\partial t}(y\vert z, t)  \\;, \\\\
 &= p(y\vert z, t)\frac{\partial}{\partial t}\left(-\frac{d}{2}\log(2\pi\sigma(t)^2) - \\| y - \mu(t)z\\|^2/(2\sigma(t)^2)\right) \\;,  \\\
&= p(y\vert z, t)\left(-d\frac{\dot\sigma(t)}{\sigma(t)}+ \frac{\dot\mu(t)z^\top(y-\mu(t)z)}{\sigma(t)^2}+\frac{\dot\sigma(t)\\| y - \mu(t)z\\|^2}{\sigma(t)^3}\right)\\;.
\end{aligned}
$$
On the other hand:
$$
\begin{aligned}
    \frac{\partial (pf\_i)}{\partial y\_i}(y\vert z, t) &= f\_i(y\vert z, t)\frac{\partial p}{\partial y\_i}(y\vert z, t) + p(y\vert z, t)\frac{\partial f\_i}{\partial y\_i}(y\vert z, t)\\;,\\\
            &= f\_i(y\vert z, t)p(y\vert z, t)\frac{\partial \log p}{\partial y\_i}(y\vert z, t) + p(y\vert z, t)\frac{\dot\sigma(t)}{\sigma(t)}\\;, \\\
            &= -p(y\vert z, t)\big( \dot\mu(t) z\_i + \dot\sigma(t)(y\_i-\mu(t)z\_i)/\sigma(t)\big)(y\_i - \mu(t)z\_i)/\sigma(t)^2 + p(y\vert z, t)\frac{\dot\sigma(t)}{\sigma(t)}\\;, \\\
            &= p(y\vert z, t)\left(-\frac{\dot\mu(t) z\_i(y\_i - \mu(t)z\_i)}{\sigma(t)^2} - \frac{\dot\sigma(t)(y\_i - \mu(t)z\_i)^2}{\sigma(t)^3} + \frac{\dot\sigma(t)}{\sigma(t)}\right)\\;.
\end{aligned}
$$
Clearly, the continuity equation is verified.
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

{{% toggle_block background-color="#CBE4FE" title="About straight paths" default-display="none"%}}
With the 'straight' schedule $\mu(t)=t$ and $\sigma(t)=1-t$, the vector fields of (7) become
$$f(y\vert z, t) = (z-y)/(1-t)\\;.$$ 
Along the conditional flow started at $y\_0$, we have $y(t) = tz + (1-t)y\_0$ and $f(y(t)\vert z, t) = z - y\_0$: 
each conditional trajectory is a straight line, travelled at constant speed.
That is often understood to back the intuition behind 'flow matching requires fewer integration steps'.


This is however misleading. 'Straightness' does not carry over to the marginal trajectories, which will 
mechanically bend if $p\_\star$ has finite support (which, through the lense of the training process, is true).
To get truly straight marginal trajectories, additional work is needed {{< ref link="fsf">}} [2]{{< /ref>}}.
{{% /toggle_block %}}

That is enough to give 'the' flow matching algorithm. 
Below, we provide some pseudocode that illustrates this; the regression searches over a parametric hypothesis class 
$\mathcal{F}\_\Theta=\\{f\_\theta : \mathbb{R}^d\times[0,1]\mapsto\mathbb{R}^d, \theta\in\Theta\\}$.
Compared to (5), the regression loss also includes sampling over the time interval. Observe that the training algorithm
is simulation-free; we do not have to simulate the ODE. 

<style>
  /* Pin the pseudocode label inside its panel (the theme hangs it below). */
  div.gray-block > div.title-frame {
    bottom: 0;
  }
</style>

{{< pseudocode title="$\texttt{Training}$" >}} 
$\textbf{init } \text{data } \{z^i\}_{i=1}^n, \text{ functions } \mu, \sigma, \text{ parameter } \theta, \text{ learning rate } \gamma\;.\\$
$\textbf{while } \text{not converged}\\$
$\quad\text{sample } i\in\{1\ldots n\},\\$
$\quad\text{sample } t\sim\mathcal{U}([0, 1]),\\$
$\quad\text{sample } \varepsilon\sim\mathcal{N}(0, \mathbf{I}_d)\\$
$\quad\text{interpolate } y = \mu(t)z^i + \sigma(t)\varepsilon,\\$
$\quad\text{compute target } u = \dot\mu(t)z^i + \dot\sigma(t)\varepsilon,\\$
$\quad\text{update } \theta\leftarrow \theta - \gamma \nabla_\theta\|f_\theta(y\vert t) - u\|_2^2\\$
$\textbf{end while}\\$
{{< /pseudocode >}}


As described earlier, drawing samples from our learned model $\hat{p}$ boils down to 'solving' the ODE 
induced by the vector fields $f\_{\hat\theta}(\cdot\vert t)$. 
We illustrate below how this can be approximated, with a simple Euler scheme. 

{{< pseudocode title="$\texttt{Sampling}$" >}} 
$\textbf{init } \text{number of steps } N, \text{ parameter } \hat\theta\;.\\$
$\text{sample } y_0\sim\mathcal{N}(0, \mathbf{I}_d)\\$
$\textbf{for } k=0, \ldots, N-1\\$
$$
y_{k+1} = y_k + \frac{1}{N}f_{\hat\theta}\left(y_k\vert k/N\right)\;.
$$
$\textbf{end for}\\$
$\textbf{return } y_N$
{{< /pseudocode >}}


## Computing likelihood

We finish this post by showing how one can
derive the likelihood of any sample $y\_1$ under the flow matching model. 
This makes it an 'explicit' generative model (opposed to implicit, which can only produce samples).
Denote $g(t) = \log p(y(t)\vert t)$ the log-probability density following a particle—that is, with $y(t)$ satisfying (1). 
Thanks to the continuity equation, one can show that $g(t)$ also checks a differential equation; for $t\in[0,1]$:
$$
    \tag{8}
    \frac{dg}{dt}(t) = - \sum\_{i=1}^d \frac{\partial f\_i}{\partial y\_i}(y(t)\vert t)\\;.
$$

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none"%}}
Observe that:
$$
\begin{aligned}
    p(y(t)\vert t)\frac{dg}{dt}(t)    &= \frac{d}{dt}p(y(t)\vert t) \\;, \\\
                        &= \frac{\partial p}{\partial t} (y(t)\vert t) + \sum\_{i=1}^d  \frac{\partial p}{\partial y\_i}(y(t)\vert t) \frac{d y\_i}{dt}(t) \\;,\\\
                        &=  \frac{\partial p}{\partial t} (y(t)\vert t) + \sum\_{i=1}^d  \frac{\partial p}{\partial y\_i}(y(t)\vert t)f\_i(y(t)\vert t) \\;.
\end{aligned}
$$
since $y(t)$ solves (1). From there, using the continuity equation at $y(t)$ yields the announced result:
$$
\begin{aligned}
    p(y(t)\vert t)\frac{dg}{dt}(t) &=  -\sum\_{i=1}^d \left(\frac{\partial p}{\partial y\_i}(y(t)\vert t)f\_i(y(t)\vert t) + p(y(t)\vert t)\frac{\partial f\_i}{\partial y\_i}(y(t)\vert t)\right) + \sum\_{i=1}^d  \frac{\partial p}{\partial y\_i}(y(t)\vert t)f\_i(y(t)\vert t) \\;, \\\
                        &= -p(y(t)\vert t)\sum\_{i=1}^d \frac{\partial f\_i}{\partial y\_i}(y(t)\vert t)\\;.
\end{aligned}
$$
Dividing both sides by $p(y(t)\vert t)>0$ yields the announced result.
<div style="text-align: right"> $\blacksquare$&nbsp; </div>
{{% /toggle_block %}}

This gives us a direct route to computing $p(y\_1\vert 1)$ for any $y\_1\in\mathbb{R}^d$. 
We start by solving the original ODE from (1) backward in time, starting from $y(1)=y\_1$. 
This gives us the trajectory $y(\cdot)$ that checks the final condition, and we
denote $y\_0 = y(0)$ the associated initial condition.
We can then solve (8) forward, since $g(0)=\log p(y\_0\vert 0)$ is known. This yields $g(1)=\log p(y(1)\vert 1)$
and the desired result.

{{< warningblock>}}
$\quad$ Above is a $\textit{very}$ inefficient (but somewhat intuitive) protocol. See Section C of {{< ref link="fmgm">}} [1]{{< /ref>}} for the real deal.
{{< /warningblock >}}


## References

<div id="fmgm"></div>
[1] Flow matching for generative modelling. Lipman et al. 2023.

<div id="fsf"></div>
[2] Flow straight and fast. Liu et al. ICLR 2023

<div id="fmddpm"></div>
[3] Diffusion meets flow matching: two sides of the same coin. Gao et al. 2024

<div id="cfg"></div>
[4] Classifier-free diffusion guidance. Ho & Salimans, 2022.

<div id=""></div>

The structure (and content) for this post was very largely influenced by [this course](https://diffusion.csail.mit.edu/2026/index.html).