---
layout: post
title: "Flow Matching and Diffusion Models 1"
date: 2026-06-21
category: "General"
tags: [flow matching, diffusion models, machine learning]
author: "Haozhe Yang"
excerpt: "This is the first post in a series of notes on MIT's Introduction to Flow Matching and Diffusion Models 2026. In this post, we will cover the basics of data modalities and how generation can be modeled as sampling from a data distribution."
---



This series of posts is my self-learning notes for MIT's Introduction to Flow Matching and Diffusion Models 2026, available [here](https://diffusion.csail.mit.edu/2026/index.html).

# 1. Introduction

## Data Modalities

- Images: $z \in \mathbb{R}^{H \times W \times C}$
- Video: $z \in \mathbb{R}^{T \times H \times W \times C}$
- Molecular structures: $z \in \mathbb{R}^{N \times 3}$, where $N$ is the number of atoms

> Key Idea 1: Identify the objects being generated as vectors $z \in \mathbb{R}^d$ (potentially after flattening).

## Generation as sampling

Data distribution: $p_{data}$

Probability density: $p_{data}: \mathbb{R}^d \to \mathbb{R}_{\ge 0}$

> Key Idea 2: Model generation as sampling from a data distribution $z \sim p_{data}$

In generative modeling, we usually have access to a finite number of examples sampled independently from $p_{data}$, which together serve as a proxy for the true data distribution.

> Key Idea 3 (dataset): finite number of samples $z_1, z_2, \ldots, z_n$ drawn i.i.d. from $p_{data}$

> Key Idea 4 (conditioned generation): sampling $z \sim p_{data}(\cdot \mid c)$, where $c$ is some conditioning variable (e.g., class label, text description, etc.)

# 2. Flow and Diffusion Models

This section describes how a generative model can be built as the simulation of a suitably constructed differential equation. For example, flow matching and diffusion models can be viewed as simulating the solution of an ODE and an SDE, respectively.

## Flow Models

A solution of an ODE is defined by a trajectory

$$X: [0,1] \to \mathbb{R}^d, \quad t \mapsto X_t$$

that maps from time $t$ to some location in space $\mathbb{R}^d$.

Every ODE is defined by a vector field $u$:

$$u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d, \quad (x,t) \mapsto u_t(x)$$

Meaning: for every time $t$ and location $x$ we get a vector $u_t(x)$ that describes the direction and speed of motion at that point in space and time.

We formalize such a trajectory as being the solution to the equation

$$
\begin{equation}
\frac{dX_t}{dt} = u_t(X_t), \quad X_0 = x_0 \tag{1}
\end{equation}
$$

which are the ODE and the initial condition, respectively.

The solution to equation (1) is a function called the flow:

$$
\begin{equation}
\psi: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d, \quad (x_0, t) \mapsto \psi_t(x_0), \qquad \frac{d}{dt} \psi_t(x_0) = u_t(\psi_t(x_0)), \quad \psi_0(x_0) = x_0 \tag{2}
\end{equation}
$$

For a given initial condition $X_0 = x_0$, a trajectory of the ODE is given by $X_t = \psi_t(x_0)$, which is the solution to equation (2). **Vector fields define ODEs whose solutions are flows.**

> Theorem 3 (flow existence and uniqueness): If $u : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ is continuously differentiable with a bounded derivative, then the ODE in (2) has a unique solution given by a flow $\psi_t$. In this case, $\psi_t$ is a diffeomorphism for every $t$, meaning that it is invertible and both $\psi_t$ and $\psi_t^{-1}$ are continuously differentiable.

Some side notes on the theorem:
1. The theorem connects closely to the Existence and Uniqueness Theorem for ODEs (Picard–Lindelöf theorem). The physical intuition is that the vector field is smooth enough that it does not create any "tears" in the space, allowing a unique trajectory to be traced out from any starting point.

2. For existence and uniqueness, we can relax the continuously-differentiable-with-bounded-derivative condition to a Lipschitz condition (the original form of the Picard–Lindelöf theorem). However, $\psi_t$ is then only guaranteed to be a homeomorphism: it is continuous and invertible with a continuous inverse, but not necessarily differentiable. A reason for requiring $C^1$ regularity is that we want the flow to be smooth enough for the change-of-variables formula to hold.

3. Recall uniform continuity: a function $f: X \to Y$ is uniformly continuous if for every $\epsilon > 0$ there exists a $\delta > 0$ such that for all $x_1, x_2 \in X$, if $d_X(x_1, x_2) < \delta$, then $d_Y(f(x_1), f(x_2)) < \epsilon$. A bounded derivative implies global Lipschitz continuity, which in turn implies uniform continuity, and rules out finite-time blow-up.

4. Diffeomorphism: I haven't gotten this far in topology and geometry yet. Some potential topics to explore: Multivariable Calculus / Measure Theory, Differential Geometry, Dynamical Systems.

> Example 4 (Linear Vector Fields): Consider a simple vector field $u_t(x) = -\theta x$ for some $\theta > 0$. Then the function $\psi_t(x_0) = e^{-\theta t} x_0$ is a solution to the ODE in (2).

**Simulating an ODE.** In general it is not possible to compute the flow $\psi_t$ in closed form if the vector field $u$ is not simple, so we use numerical methods.

The most intuitive method is the Euler method. Initialize $X_0 = x_0$ and update via

$$
\begin{equation}
X_{t + h} = X_t + h\, u_t(X_t) \quad \text{for } t = 0, h, 2h, \ldots, 1 - h \tag{3}
\end{equation}
$$

A more sophisticated method, Heun's method (a second-order Runge–Kutta method), is given by

$$
\begin{align}
X^{'}_{t + h} &= X_t + h\, u_t(X_t) &\text{(Euler guess step)} \tag{4}\\
X_{t + h} &= X_t + \frac{h}{2} \left(u_t(X_t) + u_{t + h}(X^{'}_{t + h})\right) &\text{(Heun's update)} \tag{5}
\end{align}
$$

*Some more advanced methods like RK4 are not covered in these notes. Maybe I will have a blog about numerical methods in the future.*

Finally, we introduce the concept of a flow model. We can construct a generative model via an ODE by making the vector field a neural network $u_t^{\theta}$.

An ODE is itself deterministic, but we can introduce randomness by sampling the initial condition $X_0 \sim p_{init}$ from a simple distribution such as a Gaussian $\mathcal{N}(0, I)$.

Goal: make the endpoint $X_1$ follow the data distribution $p_{data}$.

$$
X_1 \sim p_{data} \quad \text{where } X_1 = \psi_1^{\theta}(X_0) \text{ and } X_0 \sim p_{init}
$$

Algorithm to sample from a flow model:

Require: vector field $u_t^{\theta}$, number of steps $n$

1. Set $t = 0$
2. Set step size $h = 1/n$
3. Sample $X_0 \sim p_{init}$
4. For $i = 0, 1, \ldots, n-1$ do
    - $X_{t + h} = X_t + h\, u_t^{\theta}(X_t)$
    - $t \leftarrow t + h$
5. end for
6. Return $X_1$

## Diffusion Models

Stochastic Differential Equations (SDEs) extend the deterministic trajectories of ODEs with stochastic trajectories. A stochastic trajectory is a stochastic process $(X_t)_{0 \leq t \leq 1}$.

$X_t$ is a random variable for every $t \in [0,1]$, and each draw yields a trajectory $X: [0,1] \to \mathbb{R}^d, \ t \mapsto X_t$.

SDEs are constructed via a Brownian motion. A Brownian motion $W = (W_t)_{0 \leq t \leq 1}$ is a stochastic process such that $W_0 = 0$, the trajectories $t \mapsto W_t$ are continuous, and the following two conditions hold:

1. Normal increments: for $s < t$, $W_t - W_s \sim \mathcal{N}(0, (t - s)I_d)$, i.e. increments are Gaussian with variance increasing linearly in time.

2. Independent increments: for any $0 \leq t_0 < t_1 < \ldots < t_n = 1$, the increments $W_{t_1} - W_{t_0}, W_{t_2} - W_{t_1}, \ldots, W_{t_n} - W_{t_{n-1}}$ are independent random variables.

Brownian motion can be simulated as follows. Take step size $h > 0$ and set $W_0 = 0$:

$$
\begin{equation}
W_{t + h} = W_t + \sqrt{h}\, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d) \quad \text{for } t = 0, h, 2h, \ldots, 1 - h \tag{6}
\end{equation}
$$

There will be a blog about the derivation of Brownian motion in the future. (继续挖坑)

**From ODEs to SDEs.** The idea of an SDE is to extend the deterministic dynamics of an ODE with a stochastic term. But we first need an equivalent formulation of ODEs that does not use derivatives.

The expression with derivatives is

$$
\frac{dX_t}{dt} = u_t(X_t) \quad \Longleftrightarrow \quad \frac{1}{h}(X_{t + h} - X_t) = u_t(X_t) + R_t(h)
$$

where $R_t(h)$ is a remainder term that goes to 0 as $h \to 0$.

The above expression is equivalent to

$$
X_{t + h} = X_t + h\, u_t(X_t) + h R_t(h)
$$

We may now make it stochastic by adding a Brownian motion term:

$$
\begin{equation}
X_{t + h} = X_t + \underbrace{h\, u_t(X_t)}_{\text{deterministic part}} + \underbrace{\sigma_t \cdot (W_{t + h} - W_t)}_{\text{stochastic part}} + \underbrace{h R_t(h)}_{\text{remainder}} \tag{7}
\end{equation}
$$

where $\sigma_t$ is the diffusion coefficient and $R_t(h)$ is a stochastic error term such that $\mathbb{E}[\lVert R_t(h)\rVert^2] = o(h)$.

The above is commonly written in differential form as

$$
\begin{equation}
dX_t = u_t(X_t)\, dt + \sigma_t \cdot dW_t, \quad X_0 = x_0 \tag{8}
\end{equation}
$$

However, keep in mind that the $dX_t$ notation above is informal notation for equation (7). Since the value of $X_t$ is not fully determined by $X_0 \sim p_{init}$, SDEs do not have a flow map $\psi_t$; the evolution of trajectories is stochastic. Still, in the same way as for ODEs, we have:

> Theorem 5 (SDE existence and uniqueness): If $u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ is continuously differentiable with bounded derivatives and $\sigma_t: [0,1] \to \mathbb{R}^{d \times d}$ is continuous, then the SDE in (8) has a solution given by a unique stochastic process $(X_t)_{0 \leq t \leq 1}$ satisfying equation (7).

Clarification: $\sigma_t$ is a **matrix-valued function** that describes the diffusion coefficient of the SDE. It determines how the stochastic term $dW_t$ affects the evolution of the process $X_t$.

Note that every ODE is also an SDE with $\sigma_t = 0$.

### Some Rigorous Treatment of SDEs

**Probability space and filtration.**
In classical calculus, the variable is not stochastic. In stochastic calculus, we need to define a probability space for the underlying randomness.

A probability space is a triple $(\Omega, \mathcal{F}, \mathbb{P})$ where $\Omega$ is the sample space, $\mathcal{F}$ is a sigma-algebra of events (which contains the measurable events), and $\mathbb{P}$ is a probability measure that assigns probabilities in $[0,1]$ to events in $\mathcal{F}$.

A filtration is a family of sigma-algebras $(\mathcal{F}_t)_{t \geq 0}$ such that $\mathcal{F}_s \subseteq \mathcal{F}_t$ for all $s \leq t$. It represents the information available up to time $t$.

**Rigorous definition of Brownian motion.**

1. Initial condition: $W_0 = 0$ almost surely.
2. Independent increments: (similar to above)
3. Stationary Gaussian increments: (similar to above)
4. Continuous paths: for almost all $\omega \in \Omega$, the function $t \mapsto W_t(\omega)$ is continuous.

The fundamental problem here: although Brownian motion is continuous everywhere, its paths have infinite total variation. If we compute the length of a Brownian path, we get infinite length, so the Riemann–Stieltjes integral $\int H_t\, dW_t$ cannot be defined pathwise. That is why we need the Itô integral.

**Itô integral.**

The construction of the Itô integral is similar to that of the Lebesgue integral. We first define the integral for simple step processes and then extend it to more general processes by taking limits.

Consider a simple process $H_t$ that is piecewise constant on intervals $[t_i, t_{i+1})$ and adapted, meaning that $H_t$ depends only on the information in $\mathcal{F}_t$. Take a time partition $0 = t_0 < t_1 < \ldots < t_n = T$:

$$
\begin{equation}
H_t(\omega) = \sum_{i=0}^{n-1} e_{i}(\omega)\, \mathbf{1}_{[t_i, t_{i+1})}(t) \tag{9}
\end{equation}
$$

where $e_i$ is $\mathcal{F}_{t_i}$-measurable.

$$
\begin{equation}
I(H) := \int_0^T H_t\, dW_t = \sum_{i=0}^{n-1} e_i (W_{t_{i+1}} - W_{t_i}) \tag{10}
\end{equation}
$$

The magic happens with the Itô isometry. It can be shown that simple integrals satisfy

$$
\begin{equation}
\mathbb{E}\left[\left(I(H)\right)^2\right] = \mathbb{E}\left[\int_0^T H_t^2\, dt\right] \tag{11}
\end{equation}
$$

**Extension via the $L^2$ limit.** For any square-integrable adapted process $X_t$, we can find simple processes $\{H^{(n)}\}_{n=1}^{\infty}$ approximating $X_t$ in the sense that $\lim_{n \to \infty} \mathbb{E}\left[\int_0^T (H^{(n)}_t - X_t)^2\, dt\right] = 0$.

By the Itô isometry, this guarantees that the sequence of simple integrals $\{I(H^{(n)})\}_{n=1}^{\infty}$ is Cauchy in $L^2(\Omega)$. By completeness of the Hilbert space $L^2(\Omega)$, we define the Itô integral of $X_t$ as the limit of the simple integrals:

$$
\begin{equation}
\int_0^T X_t\, dW_t := \lim_{n \to \infty} \int_0^T H^{(n)}_t\, dW_t \tag{12}
\end{equation}
$$

**Going back to SDEs.**

With the Itô integral, we can now define the SDE in (8) rigorously as

$$
\begin{equation}
X_t = X_0 + \int_0^t u(X_s, s)\, ds + \int_0^t \sigma(X_s, s)\, dW_s \tag{13}
\end{equation}
$$

- The left-hand side is an adapted stochastic process.
- The first integral term on the RHS is a Lebesgue integral defined per sample path $\omega$.
- The second integral term on the RHS is an Itô integral.

Remark (from Gemini): The conditions "continuously differentiable with bounded derivatives" guarantee that, when the integral equation is treated as an operator, it satisfies the Banach fixed-point theorem, and hence existence and uniqueness of the solution can be proven.

[这边要继续挖个坑，实分析没法写的太详细，一方面我半路出家没有实分析的基础，另一方面这个系列的笔记要变成实分析笔记了。]

> Example 6 (Ornstein–Uhlenbeck Process): consider a constant diffusion coefficient $\sigma_t = \sigma \geq 0$ and a constant linear drift $u_t(x) = -\theta x$ for some $\theta > 0$. Then the SDE in (8) becomes $$\begin{equation}dX_t = -\theta X_t\, dt + \sigma\, dW_t \tag{14}\end{equation}$$ A solution $(X_t)_{0 \leq t \leq 1}$ is known as an Ornstein–Uhlenbeck (OU) process. It is a mean-reverting process that tends to drift towards its long-term mean (0 in this case) over time, with the speed of reversion determined by $\theta$. The stochastic term $\sigma\, dW_t$ introduces random fluctuations around this mean. This process converges towards a Gaussian distribution $\mathcal{N}(0, \sigma^2/(2\theta))$.

**Simulating an SDE with the Euler–Maruyama method.**

Initialize $X_0 = x_0$ and update via

$$
\begin{equation}
X_{t + h} = X_t + h\, u_t(X_t) + \sqrt{h}\, \sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d) \tag{15}
\end{equation}
$$

where $h = \frac{1}{n}$ is a step-size hyperparameter for $n \in \mathbb{N}$. In other words, we can simulate an SDE by adding a Gaussian noise term scaled by $\sqrt{h}\, \sigma_t$ to the Euler method for ODEs.

We can now finally construct a generative model via an SDE in the same way as for ODEs. The goal is still to convert a simple distribution $p_{init}$ into a complex data distribution $p_{data}$.

Algorithm to sample from a diffusion model:

Require: neural network vector field $u_t^{\theta}$, diffusion coefficient $\sigma_t$, number of steps $n$

1. Set $t = 0$
2. Set step size $h = 1/n$
3. Sample $X_0 \sim p_{init}$
4. For $i = 0, 1, \ldots, n-1$ do
    - $X_{t + h} = X_t + h\, u_t^{\theta}(X_t) + \sqrt{h}\, \sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d)$
    - $t \leftarrow t + h$
5. end for
6. Return $X_1$

---

**A quick summary for this section on SDE generative models:**

A diffusion model consists of a neural-network-parametrized vector field $u_t^{\theta}$ and a diffusion coefficient $\sigma_t$.

$$
\begin{aligned}
\text{NN:} \quad & u_t^{\theta}: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d, \quad (x,t) \mapsto u_t^{\theta}(x) \\
\text{Diffusion:} \quad & \sigma_t: [0,1] \to \mathbb{R}_{\geq 0}, \quad t \mapsto \sigma_t
\end{aligned}
$$

To obtain samples from the SDE:

$$
\begin{aligned}
X_0 &\sim p_{init} \\
dX_t &= u_t^{\theta}(X_t)\, dt + \sigma_t\, dW_t \\
X_1 &\sim p_{data}
\end{aligned}
$$

A diffusion model with $\sigma_t = 0$ reduces to a flow model. A flow model is a special case of a diffusion model.
