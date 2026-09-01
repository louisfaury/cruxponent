+++
author = "Louis Faury"
title = "Oldies but goodies: ODE solvers"
date = "2026-08-22"
+++

Some recent work on physical simulation sent me back to solving ordinary differential equations—and how I had little intuition for how
non-trivial numerical solvers are born. 
After covering some basic properties, this post digs into the quadrature viewpoint
for synthesising 'any' solvers from the Runge-Kutta family.
We illustrate the method by rederiving the celebrated RK4 solver from first principles.

<!--more-->


## Setting

### Motivation
Let $\mathcal{I}$ be some interval of $\mathbb{R}$ 
and $f:\mathbb{R}^d\times\mathcal{I}\mapsto\mathbb{R}^d$ describe a family of vector fields
indexed by some variable $t\in\mathcal{I}$.
Let $y:\mathbb{R}\mapsto\mathbb{R}^d$ describe the motion of a 'particle' which, at each instant $t\in\mathcal{I}$,
follows the vector field $f(\cdot, t)$. 
It is characterised by a non-autonomous ordinary differential equation (ODE); for any $t\in\mathcal{I}$:
$$
\frac{dy}{dt}(t) = f\big(y(t), t\big)\\;.
$$
When given an initial condition, finding such $y$ is known as a _Cauchy problem_.
Only in some notorious cases (mostly, linear) we know of a closed-form solution for this system.
In other cases, we can only be satisfied with numerical approximations—the topic of this post.
But first, it will be in good taste to understand the conditions required for a Cauchy problem to be well-posed
(_i.e._ it admits a solution, and only one).

### Assumptions
We will work under a simplified setting. Namely, we will assume that the ODE is autonomous: the vector field is constant
wrt. the time variable. 
We will also take $\mathcal{I}=[0, 1]$.
(Those assumptions are essentially to limit clutter;
all results presented below all easily extend to the non-autonomous case
and to arbitrary compact $\mathcal{I}$.)
Therefore, throughout this post we are interested in the solution of the Cauchy problem:

<div style="background-color: #d8fbc9; width: 100%; padding: 10px 0;">

$$
\tag{$\star$}
\frac{dy}{dt}(t) = f(y(t)) \\; \text{for } t\in[0,1]\\;\text{ and } y(0) = y\_0\in\mathbb{R}^d \\; .
$$

</div>


### The Cauchy-Lipschitz theorem
The Cauchy-Lipschitz theorem guarantees that under some rather mild smoothness assumption, 
a unique solution exists.
Here, we will focus on the 'strong' version which requires _global_ Lipschitz continuity—some weaker versions exists, with slightly degraded claims.


{{< boxed title="Cauchy-Lipschitz" >}}
$\qquad\qquad\qquad\qquad\; \text{If } f \text{ is Lipschitz continuous then}$
$\text{Cauchy problem }(\star)\text{ has a unique solution.}$
{{< /boxed >}}

"It exists and is unique" is a solid tell that some fixed-point argument is at work here. 
That's the path taken in the proof below, which studies the following operator before invoking the Banach fixed-point theorem.
$$
\tag{1}
\mathcal{T} : y(t) \mapsto y_0 + \int_{0}^t f(y(\tau))d\tau\\;.
$$ 

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
Below denote $\mathcal{F}$ the space of functions mapping $\mathbb{R}$ to $\mathbb{R}^d$.
The fundamental theorem of calculus ensures us that $y\in\mathcal{F}$ is a solution to $(\star)$
iff it is a fixed-point of $\mathcal{T}:\mathcal{F}\mapsto\mathcal{F}$ defined in (1). 
Therefore, if we prove that $\mathcal{T}$ contracts wrt to some norm $\\| \cdot \\|$ and that 
$(\mathcal{F}, \\| \cdot \\|)$ is complete the Banach fixed-point theorem serves us the result.
Not any norm will do though, and we will have to use a 'custom' one. Let $\lambda>0$ and write for $y\in\mathcal{F}$:
$$
\|| y\||\_\lambda := \sup\_{t} \big(e^{-\lambda t} \\| y(t)\\| \big)\\;,
$$
It is straightforward to show that $\|| \cdot\||\_\lambda$ is equivalent to the supremum norm, for any $\lambda>0$.
Since the latter makes $\mathcal{F}$ complete, so does the former. 

We only have left to show that $\mathcal{T}$ contracts under  $\|| \cdot\||\_\lambda$, for some (well-chosen) $\lambda$.
By assumption, it exists $L$ such that for every $x,y$ we have $\\| f(x) - f(y)\\| \leq L\\|x-y\\|$. 
Then for any $y_1, y_2\in\mathcal{F}$:
$$
\begin{aligned}
\\| \mathcal{T}(y_1) -  \mathcal{T}(y_2)\\| \_\lambda &= \sup\_{t\in[0,1]} e^{-\lambda t} \\| \mathcal{T}(y_1)(t) -  \mathcal{T}(y_2)(t)\\| \\;, \\\
&= \sup\_{t\in[0,1]} e^{-\lambda t} \Big\\| \int\_{0}^t \big(f(y\_1(\tau)) - f(y\_2(\tau))\big)d\tau\Big\\| \\;, \\\
&\leq L \sup\_{t\in[0,1]} e^{-\lambda t} \int\_{0}^t \\| y\_1(\tau) - y\_2(\tau)\\| d\tau \\;, &(\text{Lipschitz})\\\
&= L \sup\_{t\in[0,1]}  \int\_{0}^t e^{-(t-\tau)}e^{-\tau}\\| y\_1(\tau) - y\_2(\tau)\\| d\tau \\;, \\\
&\leq L \\|y_1-y_2\\|\_{\lambda} \sup\_{t\in[0,1]}\int\_0^t e^{-(t-\tau)}d\tau\\;, &( \smallint fg \leq \sup g\smallint f)\\\
&\leq e^{-\lambda}L \\|y_1-y_2\\|\_{\lambda}\\;.
\end{aligned}
$$
Choosing any $\lambda^\star$ large enough therefore yields that $\mathcal{T}$ contracts under $\\|\cdot\\|\_{\lambda^\star}$. 
It now becomes obvious why we need this custom norm—we need to fight the Lipschitz constant $L$, which has no reason to be $<1$.

The Banach fixed-point theorem justifies $\mathcal{T}$'s fixed-point exists and is unique, which concludes the proof.
<div style="text-align: right"> $\blacksquare$ </div>
{{% /toggle_block %}}

Below, assume this assumption holds, so it makes sense to try and approximate the solution of $(\star)$.


## Numerical solvers
Let $y$ be the unique solution of $(\star)$. 
A numerical solver builds an approximation of $y$ piecemeal, over
a regular, discrete mesh $t\_1, \ldots, t\_N\in[0,1]$ where $t\_{i+1} = t\_i + h$ and $h>0$. 
It produces some $y\_1, \ldots y\_N\in\mathbb{R}^d$ such that $y\_i \approx  y(t\_i)$ for all $i$.
Here, we focus on _explicit single-step_ solvers, for which the update writes:
$$
\tag{2}
y\_{i+1} = y_i + h\phi(y\_i, h)\\;.
$$


The goal of this section is to motivate some basic properties we ask of such numerical solvers.
Two quantities are especially relevant to our study; the scheme's _global_ error $E_i := y_i - y(t\_i)$ and its truncation error:
$$
T\_i(h) := \frac{y(t\_{i+1}) - y(t\_i)}{h} - \phi(y(t\_i), h)\\;.
$$

While the global error measures the total divergence between approximated and true solution after $i$ steps,
the truncation error essentially captures the one-step error. Clearly, we want _both_ to be small.

{{% toggle_block background-color="#CBE4FE" title="About implicit solvers (1/2)" default-display="none"%}}
We explicitly target explicit solvers instead of _implicit_ ones. 
Implicit one-step solvers have an update rule that looks like:
$$
y\_{i+1} = y\_i + h \phi(y\_i, y\_{i+1}, h)\\;.
$$
Observe that $y\_{i+1}$ sits on both sides of the equal sign; to actually compute it one has to solve the resulting equation (which
often requires a numerical solver of its own). One example is the implicit Euler, which writes $y\_{i+1} = y\_i + h f(y\_{i+1})$.
Implicit solvers require more computation, but are a much better fit for _stiff_ equations—a concept we discuss later in this post.
{{% /toggle_block %}}


### Stability

The scheme detailed in (2) is called _stable_ if it exists $C$ and $H$ such that for all $h<H$ and $x,y\in\mathbb{R}^d$:

$$
\\| \phi(x, h) - \phi(y, h) \\| \leq C \\|x-y\\|\\;.
$$

Stability allows to relate the scheme's global (accumulated) error to the truncation error.
Indeed, if stable:
$$
\tag{3}
\\| E\_i \\| \leq \blacksquare \sup\_{j\leq i} \\| T\_j(h) \\|\\; ,
$$

with $\blacksquare$ some (bad) constant. Given our assumption that $f$ is Lipschitz continuous, stability is hardly a tough condition to meet.
Below we assume (2) to be stable so we focus our analysis only on the truncation error. 

{{% toggle_block background-color="#FAD7A0" title="Proof" default-display="none" %}}
Observe that by some re-arrangement:
$$
\begin{aligned}
    E\_{i+1} &= y(t\_{i+1}) - y\_{i+1} \\;, \\\
&= y(t\_{i+1}) + y(t\_i) - y(t\_i) - y\_{i+1}\\;, \\\
&= y(t\_{i+1}) + y(t\_i) - y(t\_i) - y\_{i} - h\phi(y\_i, h)\\;,\\\
&=  y(t\_{i+1}) + y(t\_i) - y(t\_i) - y\_i - h\phi(y\_i, h) + h\phi(y(t\_i), h) - h\phi(y(t\_i), h) \\;, \\\
&= E\_i + hT\_i(h) + h\phi(y(t\_i), h) -  h\phi(y\_i, h)\\;.
\end{aligned}
$$
Therefore, by our stability assumption:
$$
\begin{aligned}
\\| E\_{i+1} \\| &\leq (1+hC)\\|E\_i\\| + h \\| T\_i(h)\\| \\;,\\\
&\leq h \\| T\_i(h)\\| + h(1+hC)\\| T\_{i-1}(h)\\| + (1+hC)^2\\|E\_{i-1}\\|\\;,\\\
&\leq h \sum\_{j=0}^i (1+hC)^j \\| T\_j(h)\\| + (1+hC)^{i}\\|E\_{0}\\|\\;, &(\text{unrolling})\\\
&= h \sum\_{j=0}^i (1+hC)^j \\| T\_j(h)\\|\\;, &(\\|E\_{0}\\|=0) \\\
&\leq h \sup\_{j\leq i}\\| T\_j(h)\\| \sum\_{j=0}^i (1+hC)^j \\;, \\\
&\leq h \sup\_{j\leq i}\\| T\_j(h)\\| \frac{(1+hC)^{i+1}-1}{hC}\\;, \\\
&\leq \frac{e^{(i+1)hC}-1}{C}\sup\_{j\leq i}\\| T\_j(h)\\|\\;. &(\log(1+x)\leq x)
\end{aligned}
$$
<div style="text-align: right"> $\blacksquare$ </div>
{{% /toggle_block %}}

{{% toggle_block background-color="#CBE4FE" title="About the constant" default-display="none"%}}
From the proof we see that $\blacksquare \propto \exp(ihC)$—which is a pretty bad constant: the error's upper-bound
grows exponentially fast with time. This is reminiscent of the Grönwall theorem, and something we will blissfully 
ignore here to focus on the dependency wrt. the step-size $h$.
{{% /toggle_block %}}


### Consistency and order
A scheme is called _consistent_ if $\lim\_{h\to 0} \phi(x, h) = f(x)$ for any $x\in\mathbb{R}^d$. 
Thanks to (3) we get that a consistent scheme is also a convergent one, as its global error goes to 0 as $h$ does.
If consistency captures that the truncation error goes to 0, it doesn't say how fast—and that would be the role of the
scheme's order. The solver (2) is said to be of order $p\in\mathbb{N}$ if $T\_i(h) = O(h^p)$ for all $i\in\\{1,\ldots, N\\}$.
A higher order is desirable, but inevitably comes with extra computations. 

### Examples

Perhaps the simplest of all solvers is known as the explicit Euler, which writes
$
    y\_{i+1} = y\_i + hf(y\_i)\\;.
$
It is trivially stable and consistent, and one easily shows that it is of order 1. 
Its sibling, the improved Euler:
$$
    y\_{i+1} = y\_i + \frac{h}{2}\big(f(y\_i)+ f(y\_i + hf(y\_i))\big)\\;,
$$
is also stable, consistent, and it enjoys a better $O(h^2)$ rate
(the proof being a simple Taylor expansion).
Both those solvers are member of the Runge-Kutta family, which the rest of this post studies.
Especially, we will be interested in the rationale and mechanisms used to _generate_ them.

## Synthesising solvers

The fundamental theorem of calculus essentially translates differential equations into integral ones;
$$
\tag{4}
    y\_{i+1} = y\_i + \int\_{t\_i}^{t\_{i+1}}f(y(\tau))d\tau\\;.
$$
Under this perspective, creating a new solver essentially boils down to making some integral approximation choices. 
For instance, the explicit Euler essentially considers $f$ to be the constant $f(y(t\_i))$.
This essentially opens up an entire inventory of methods tapping into the many existing quadrature rules.
For instance, using the trapezoidal rule yields the (implicit) _trapezoidal method_:
$$
\tag{5}
y\_{i+1}= y\_i + \frac{h}{2}(f(y\_{i}) + f(y\_{i+1}))\\;.
$$
Since $y\_{i+1}$ stands on both sides of the equal sign, (5) is an implicit solver. 
We fall back to an explicit one by making a further approximation on the right-hand side, using the explicit Euler quadrature $y\_{i+1} = y\_i + hf(y\_i)$—and recovering the improved Euler method.
The following section essentially shows how to generalise and mechanise this approach to higher orders.



### Nested quadratures
Let's rewrite (4) with a simple change of variable:
$$
    y\_{i+1} = y\_i + h\int_{z=0}^1 f(y(t\_i + hz))dz\\;.
$$
A quadrature rule for approximating the above integral picks $M\in\mathbb{N}$ anchor points $\{z\_1, \ldots, z\_M\}\in[0,1]$
along with some weights $b\_1, \ldots, b\_M\in\mathbb{R}$ to form a weighted sum:
$$
\tag{6}
y\_{i+1} \approx y\_i + h\sum\_{j=1}^M b\_j f(\tilde{y}\_{ij}) \text{ where } \tilde{y}\_{ij} := y(t\_i + hz\_j)\\;.
$$
Observe that each $\tilde{y}\_{ij}$ is itself unknown. 
They can too be approximated via quadrature rules—an inner integral approximation, 
which will re-use the anchors points we just defined but aggregate them with different coefficients.
Concretely, for $j\in\\{1, \ldots, M\\}$ we introduce $\\{a\_{j1}, \ldots, a\_{j(j-1)}\\}\in\mathbb{R}$ and write:
$$
\tag{7}
\tilde{y}_{ij} = y\_i + h \int\_{0}^{z\_{j}} f(y(\tau))d\tau \approx y\_i + h\sum\_{k< j} a\_{jk} f(\tilde y\_{ik})\\;.
$$

{{% toggle_block background-color="#CBE4FE" title="About implicit solvers (2/2)" default-display="none"%}}
Observe that $a\_{jk}=0$ for $k\geq j$ since we require the solver to be explicit. In all generality 
we could write:
$$
\tilde{y}_{ij} \approx y\_i + h\sum\_{k\leq M} a\_{jk} f(\tilde y\_{ik})\\;,
$$
which altogether yields implicit relationships between the $\\{\tilde y\_{i,j}\\}\_j$.
If $a\_{jk}=0$ for $k>j$, the scheme is called diagonally implicit—see the section
about Butcher tables to understand why.
{{% /toggle_block %}}


Combining (6) and (7) yields the entire update rule. 
At this point we introduced a bunch of free parameters; the anchors $\\{z\_j\\}\_j$, their outer weights $\\{b\_j\\}\_j$
and inner weights $\\{a\_{jk}\\}\_{jk}$.
Some constraints can reduce the number of degrees of freedom; for instance, it is typical to
require for our quadrature rules to integrate exactly any constant function.
This yields the following constraints:
$$
\tag{8}
\sum\_{j=1}^M b\_j = 1 \text{ and } \sum\_{k<j} a\_{jk} = z\_j \text{ for all } j\in\\{1,\ldots, M\\}.
$$

{{% toggle_block background-color="#CBE4FE" title="Remaining degrees of freedom" default-display="none"%}}
Even after fixing the anchors $\\{z\_j\\}\_j$ and outer weights $\\{b\_j\\}\_j$ (by _e.g._ looking up some quadrature rule)
one can notice that we still have remaining many degrees of freedom as soon as $M\geq 3$. 
As we will see when synthesising RK4 ($M=4$), a good way to resolve them is via Taylor expansion matching.
{{% /toggle_block %}}


### Butcher tables
A solver in those likes can be compactly described by a so-called Butcher table:
$$
\begin{array}{c|cccc}
z\_1 & 0 & 0 & \cdots & 0 \\\
z\_2 & a\_{21} & 0 & \cdots & 0 \\\
\vdots & \vdots & \ddots & \ddots & \vdots \\\
z\_M & a\_{M1} & a\_{M2} & \cdots & 0 \\\
\hline
 & b\_1 & b\_2 & \cdots & b\_M
\end{array}\\;.
$$
The diagonal and upper-diagonal blocks are filled by 0s, as per our requirement for the solver to be explicit.
The constraints (8) impose that each row of the inner block sums to its anchor,
and that the bottom row sums to one.
For instance, the explicit Euler and the improved Euler are respectively described by:
$$
\begin{array}{c|c}
0 & 0 \\\
\hline
 & 1
\end{array}
\qquad\text{and}\qquad
\begin{array}{c|cc}
0 & 0 & 0 \\\
1 & 1 & 0 \\\
\hline
 & 1/2 & 1/2
\end{array}\\;.
$$

## Building RK4

{{< warningblock>}}
$\quad$ This section is computation heavy, but details a useful Taylor expansion matching mechanism.
{{< /warningblock >}}

This section uses the quadrature approach to derive the Runge-Kutta 4th-order solver from the ground-up. 
The starting point is Simpson's quadrature rule, which writes:
$$
\int\_0^1 f(x)dx \approx f(0)/6 + 2f(1/2)/3 + f(1)/6\\;.
$$
To reach a higher order, the RK4 method duplicates the midpoint. Concretely, we will be using:
<div style="display: flex; justify-content: center;">

| $i$   | 1   | 2   | 3   | 4   |
|-------|-----|-----|-----|-----|
| $z\_i$ | $0$ | $1/2$ | $1/2$ | $1$ |
| $b\_i$ | $1/6$ | $1/3$ | $1/3$ | $1/6$ |

</div>

We can, for completeness, spell out the relationships defined in (7) along with the constraints from (8):
$$
\left\\{
\begin{aligned}
\tilde y\_{i1} &= y\_i\\;,\\\
\tilde y\_{i2} &= y\_i + ha\_{21}f(\tilde y\_{i1})\\;,  &\text{ s.t } a\_{21}=1/2\\\
\tilde y\_{i3} &= y\_i + ha\_{31}f(\tilde y\_{i1}) + ha\_{32}f(\tilde y\_{i2})\\;, &\text{ s.t } a\_{31}+a\_{32}=1/2 \\\
\tilde y\_{i4} &= y\_i + ha\_{41}f(\tilde y\_{i1}) + ha\_{42}f(\tilde y\_{i2}) + ha\_{43}f(\tilde y\_{i3})\\;, &\text{ s.t } a\_{41}+a\_{42}+a\_{43}=1\\;.\\\
\end{aligned}
\right.
$$

It stands out that we still have $3$ unresolved degrees of freedom. 
To settle them, we resort to Taylor expansion matching: we choose the remaining degrees of freedom
so the scheme can be of the highest possible order.
Concretely, we will try to set the $\\{a\_{jk}\\}\_{jk}$ so that they respect the aforementioned constraints and allow
to write $T\_i(h) = O(h^p)$ with the highest possible value of $p$. 
The computation is rather tedious as it involves somewhat long Taylor expansions; we omit it here to directly reveal the result:

$$
\begin{aligned}
\frac{y(t\_{i+1}) - y_{i+1}}{h} = & h^2 \left[ \frac{1}{12} \left( 2 - 2a_{32} - a_{42} - a_{43} \right) \right] f'^2 f \\\
& + h^3 \left[ \frac{1}{48} \left( 8 - 6a_{32} - 5(a_{42} + a_{43}) \right) \right] f'f'' f^2 \\\
& + h^3 \left[ \frac{1}{24} \left( 1 - 2 a_{43} a_{32} \right) \right] f'^3 f \\\
& + O(h^4)\\;,
\end{aligned}
$$
where $f$ and its derivatives are all evaluated at $y\_i$. 
Now we can retrieve have our remaining constraints, that ensure that all $O(h^2)$ and $O(h^3)$ term are zeroes.
That's three of them, one for each unresolved degree of freedom. 
Combining them with previous ones, we get the system:
$$
\left\\{
\begin{aligned}
&a\_{21} = 1/2\\;, \\\
&a\_{31} + a\_{32} = 1/2\\;, \\\
&a\_{41} + a\_{42} + a\_{43} = 1\\;, \\\
&2a\_{32} + a\_{42} + a\_{43} = 2\\;, \\\
&6a\_{32} + 5(a\_{42} + a\_{43}) = 8\\;, \\\
&2a\_{43}\\,a\_{32} = 1\\;.
\end{aligned}
\right.
$$
This system admits a unique solution:
$a\_{21} = 1/2$, $a\_{31} = 0$, $a\_{32} = 1/2$, $a\_{41} = a\_{42} = 0$ and $a\_{43} = 1$.
From this we recover the Butcher table of the celebrated RK4 solver:
<div style="display: flex; justify-content: center;">

$$
\begin{array}{c|cccc}
0 & 0 &  &  &  \\\
1/2 & 1/2 & 0 &  &  \\\
1/2 & 0 & 1/2 & 0 &  \\\
1 & 0 & 0 & 1 & 0 \\\
\hline
 & 1/6 & 1/3 & 1/3 & 1/6
\end{array}\\;,
$$

</div>

or in a form that stands close to the actual implementation:

<div style="background-color: #d8fbc9; width: 100%; padding: 10px 0;">

$$
\begin{aligned}
k\_1 &= f(y\_i)\\;, \\\
k\_2 &= f(y\_i + \tfrac{h}{2}\\,k\_1)\\;, \\\
k\_3 &= f(y\_i + \tfrac{h}{2}\\,k\_2)\\;, \\\
k\_4 &= f(y\_i + h\\,k\_3)\\;, \\\
y\_{i+1} &= y\_i + \tfrac{h}{6}\big(k\_1 + 2k\_2 + 2k\_3 + k\_4\big)\\;.
\end{aligned}
$$

</div>

As a by-product, we proved that $T\_i(h)=O(h^4)$ for any $i\in\\{1,\ldots, M\\}$. In other words, RK4 is of order 4.


## Implicit schemes

This post so far focused essentially on explicit single-step solvers.
While powerful, many (most) advanced applications require more sophistication. There is a _very_ large zoology out there;
symplectic solvers, multistep solvers,
adaptive-step schemes, etc.
Covering them is out of scope here; instead, we 
hereinafter briefly introduce implicit solvers. 
To understand their relevance, we first need to discuss _stiffness_.

### Stiffness
Consider the one-dimensional Cauchy problem:
$$
\frac{dy}{dt}(t) = - \beta y(t) \text{ and } y(0)=1\\;, \tag{9}
$$
with $\beta\gg 1$. Its unique solution is the quickly decaying $y(t) = \exp(-\beta t)$. It is fairly easy to show that the sequence of 
estimates produced by the explicit Euler solver is, for any $i\in\mathbb{N}$:
$$
y\_i = (1 - h\beta)^{i}\\;.
$$
For any $h>2/\beta$ this sequence quickly diverges, even though the explicit Euler scheme is stable and consistent—and hence convergent 
(observe that this does not contradict earlier statements). Essentially, the requirement for the scheme to behave 
is that the discretisation step
is small—at least compared to $\beta$.

This becomes an issue for multidimensional systems, where some quickly decaying components impose an absurdly small global step-size for approximating other components. For example, the system described by:
$$
\frac{dy}{dt}(t) = \begin{pmatrix} -\beta & 0 \\\ 0 & -\beta^{-1}\end{pmatrix}y(t)\\;,
$$
is _stiff_. Solving it (naively) via an explicit numerical solver requires $h\lessapprox 1/\beta$ because of its first
component; but this is risible (as in, unnecessarily small) step-size to approximate the second component which evolves slowly.
Explicit solvers ultimately all suffer from this defect (formally, one says that they are not A-stable).

### An implicit scheme
Single-step implicit solvers write as:
$$
y\_{i+1} = y\_i + h\phi(y\_i, y\_{i+1}, h)\\;.
$$
In general, they require their own solver (_e.g._, a root finder) to materialise $y\_{i+1}$; hence, they 
will typically ask for more computations. In return, they generally pair much better with stiff equations. 
There is not (to my knowledge) any _general_ demonstration of this claim (perhaps because there is no universally agreed upon
definition of stiffness) and this is mostly folk knowledge.
It is educative to look at the behaviour of the implicit Euler scheme 
$y\_{i+1} = y\_i + hf(y\_{i+1})$ on the example from (9); one can easily show that then:
$$
y\_{i} = (1+h\beta)^{-i}\\;,
$$
which provides reasonable a estimate regardless of the value of $h/\beta$ (it is A-stable).
Implicit solvers also have a Butcher table representation, with a non-zero entries on and/or above the diagonal.
For instance, the implicit Euler being a diagonally implicit scheme has the following representation:
$$
\begin{array}{c|c}
1 & 1 \\\
\hline
 & 1
\end{array}\\;.
$$