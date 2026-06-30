---
layout: post
title: "Flow Matching and Diffusion Models 2"
date: 2026-06-28
category: "Machine Learning"
tags: [flow matching, diffusion models, generative models]
author: "Haozhe Yang"
image: /assets/images/posts/flow-matching-2.svg
license: cc-by-nc-sa
acknowledgement: "These notes are based on MIT Course 6.S184: *Generative AI with Stochastic Differential Equations* ([course site](https://diffusion.csail.mit.edu/))."
excerpt: "This is the second post in a series of notes on MIT's Introduction to Flow Matching and Diffusion Models 2026. In this post, we discuss how to train vector fields."
---

# Training flow matching models

We discuss the flow matching algorithm to train $$u_t^{\theta}$$. We have a neural network $$u_t^{\theta}$$ and obtain samples from the generative model by simulating the ODE

$$
\begin{equation}
X_0 \sim p_{\text{init}}, \quad dX_t = u_t^{\theta}(X_t)\, dt
\end{equation}
$$

and using the endpoint $$X_1$$ at $$t=1$$ as a sample: $$X_1 \sim p_{\text{data}}$$.

## Conditional and Marginal Probability Path

A probability path specifies a gradual interpolation between the initial noise $$p_{\text{init}}$$ and the data distribution $$p_{\text{data}}$$. In between, we have some freedom to choose what happens.

For a data point $$z \in \mathbb{R}^d$$, we denote by $$\delta_z$$ the Dirac delta distribution. Sampling from $$\delta_z$$ always returns $$z$$.

> Side note 1: Dirac distribution. In probability theory and measure theory, the Dirac distribution, or more precisely the Dirac measure, is a rigorously defined probability measure. Suppose $$(\Omega, \mathcal{F})$$ is a measurable space. For a given point $$z \in \Omega$$, the Dirac measure centered at $$z$$ is defined as the measure $$\delta_z : \mathcal{F} \to [0,1]$$ such that for any measurable set $$A \in \mathcal{F}$$, we have
>
> $$
> \delta_z(A) = \begin{cases}
> 1 & \text{if } z \in A, \\
> 0 & \text{if } z \notin A.
> \end{cases}
> $$
> The intuition is that it describes a deterministic random variable with 100% probability of being equal to $$z$$.


> Side note 2: Dirac Delta "Function". In engineering and physics, we often refer to the Dirac delta "function" $$\delta(z)$$, and define it as follows: a. when $$z \neq 0$$, $$\delta(z) = 0$$; b. when $$z = 0$$, $$\delta(z) = \infty$$, and $$\int_{-\infty}^{\infty} \delta(z)\, dz = 1$$. The conflict with Lebesgue measure theory is that if a function is $$0$$ except at a single point, then its integral is $$0$$. This motivates the theory of generalized functions. In functional analysis, the Dirac delta is not a function but a continuous linear functional on the space of test functions $$\phi(z)$$, denoted as the pairing $$\langle \delta_z, \phi \rangle = \phi(z)$$.

A conditional probability path is a set of distributions $$p_t(x \mid z)$$ over $$\mathbb{R}^d$$ such that:

$$
\begin{equation}
p_0(x \mid z) = p_{\text{init}}(x), \quad p_1(x \mid z) = \delta_z(x), \quad \text{for all } z \in \mathbb{R}^d.
\end{equation}
$$

Every conditional probability path $$p_t(x \mid z)$$ induces a marginal probability path $$p_t(x)$$ defined as the distribution that we obtain by first sampling a data point $$z \sim p_{\text{data}}$$ from the data distribution and then sampling from $$p_t(\cdot \mid z)$$:

$$
\begin{align}
z \sim p_{\text{data}}, \quad x \sim p_t(\cdot \mid z) \implies x \sim p_t(x) \quad \quad &\text{sampling from marginal path} \\
p_t(x) = \int p_t(x \mid z) p_{\text{data}}(z)\, dz \quad \quad &\text{density of marginal path}
\end{align}
$$

About the $$\implies$$ relation in (3), the idea is that we first sample $$z$$ and then sample $$x$$ from $$p_t(\cdot \mid z)$$. Imagine we get a tuple $$(x, z)$$ instead of discarding $$z$$. The probability of getting such a tuple is just $$p_t(x \mid z) p_{\text{data}}(z)$$. Then we can get the probability of $$x$$ by marginalizing out $$z$$. So overall, the effect of the first two steps is just sampling $$x$$ from the marginal distribution $$p_t(x)$$.

Note that we know how to sample from the marginal probability path $$p_t(x)$$, but we do not know how to evaluate its density, as the integral in (4) is intractable (we cannot evaluate the integral over all data points $$z$$).

By the conditions on $$p_t(\cdot \mid z)$$ in (2),

$$p_0 = \int p_0(x \mid z) p_{\text{data}}(z)\, dz = \int p_{\text{init}}(x) p_{\text{data}}(z)\, dz = p_{\text{init}}(x)$$

and

$$p_1 = \int p_1(x \mid z) p_{\text{data}}(z)\, dz = \int \delta_z(x) p_{\text{data}}(z)\, dz = p_{\text{data}}(x)$$

Therefore the marginal probability path $$p_t(x)$$ indeed interpolates between the initial noise $$p_{\text{init}}$$ and the data distribution $$p_{\text{data}}$$.

The following example shows how to derive a Gaussian probability path.

> Example 1: Gaussian Conditional Probability Path.
>
> This probability path is used by most state-of-the-art models. Let $$\alpha_t$$, $$\beta_t$$ be noise schedulers: two continuously differentiable and monotonic functions with $$\alpha_0 = \beta_1 = 0$$ and $$\alpha_1 = \beta_0 = 1$$. Define the conditional probability path as follows:
>
> $$
> \begin{equation}
> p_t(\cdot \mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I) \quad \quad \text{(Gaussian conditional path)}
> \end{equation}
> $$
>
> Check the conditions we imposed on $$\alpha_t$$ and $$\beta_t$$. $$p_0(\cdot \mid z) = \mathcal{N}(\alpha_0 z, \beta_0^2 I) = \mathcal{N}(0, I)$$ and $$p_1(\cdot \mid z) = \mathcal{N}(\alpha_1 z, \beta_1^2 I) = \mathcal{N}(z, 0) = \delta_z$$. So the conditions are satisfied. We used the fact that $$\mathcal{N}(z, 0) = \delta_z$$.
>
> We can express sampling from the marginal path $$p_t(x)$$ as:
>
> $$
> \begin{equation}
> z \sim p_{\text{data}}, \ \epsilon \sim p_{\text{init}} = \mathcal{N}(0, I) \implies x = \alpha_t z + \beta_t \epsilon \sim p_t(x)
> \end{equation}
> $$
>
> Intuitively, the above procedure adds more noise for lower $$t$$, until time $$t = 0$$ at which point there is only noise.

The algebraic transformation (6) is known as the reparameterization trick. It separates the deterministic parameters (mean $$\alpha_t z$$ and variance $$\beta_t^2$$) from the pure randomness ($$\epsilon$$).

```python
# PyTorch code to sample from the marginal path p_t(x)
z = next(data_loader)
epsilon = torch.randn_like(z)
x = alpha_t * z + beta_t * epsilon
```

## Conditional and Marginal Vector Fields

A probability path $$(p_t)_{0 \leq t \leq 1}$$ specifies what distribution $$X_t \sim p_t$$ the points along a trajectory should follow. Up to this point, it is still unclear how to find a vector field such that the trajectories $$X_t$$ follow the path.

For every data point $$z$$, we can define a **conditional vector field**. This can be any vector field such that the corresponding ODE yields the conditional probability path $$p_t(\cdot \mid z)$$:

$$
\begin{equation}
X_0 \sim p_{\text{init}}, \space \space \frac{dX_t}{dt} = u_t^{\text{target}}(X_t \mid z) \implies X_t \sim p_t(\cdot \mid z) \space \space t \in [0, 1]
\end{equation}
$$

The conditional vector field may seem useless at first sight because all endpoints of the ODE $$X_1$$ will collapse to the known data point $$z$$. But it is useful because it allows us to define a **marginal vector field** that yields the marginal probability path, which generates samples from the data distribution $$p_{\text{data}}$$:

> Theorem 1 (Marginalization trick):
>
> Let $$u_t^{\text{target}}(x \mid z)$$ be a conditional vector field. Then the marginal vector field $$u_t^{\text{target}}(x)$$ is given by:
>
> $$
> \begin{equation}
> u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x \mid z) \frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)}\, dz
> \end{equation}
> $$
>
> follows the marginal probability path, i.e.
>
> $$
> \begin{equation}
> X_0 \sim p_{\text{init}}, \space \space \frac{dX_t}{dt} = u_t^{\text{target}}(X_t) \implies X_t \sim p_t \space \space t \in [0, 1]
> \end{equation}
> $$
>
> In particular, $$X_1 \sim p_{\text{data}}$$ for this ODE, so we can say "$$u_t^{\text{target}}$$ converts noise $$p_{\text{init}}$$ to data $$p_{\text{data}}$$".

Viewing (8), the term $$\frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)}$$ is actually the posterior $$p_t(z \mid x)$$. The marginal vector field is the average of conditional vector fields weighted by the posterior probability of $$z$$ given $$x$$, i.e. $$\mathbb{E}_{p_t(z \mid x)}[u_t^{\text{target}}(x \mid z)]$$. The intuition is that we need to update our belief about $$z$$ given $$x$$. A more rigorous derivation is given later in this post (via the continuity equation).

> Example 2: Target ODE for Gaussian probability path.
> The same setting as Example 1. Let $$p_t(\cdot \mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I)$$ for noise schedulers $$\alpha_t$$, $$\beta_t$$. Let $$\dot{\alpha_t} = \partial_t \alpha_t$$ and $$\dot{\beta_t} = \partial_t \beta_t$$ be the respective time derivatives of the noise schedulers. We want to show that the **conditional Gaussian vector field** given by:
>
> $$
> \begin{equation}
> u_t^{\text{target}}(x \mid z) = \left(\dot{\alpha_t} - \frac{\dot{\beta_t}}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta_t}}{\beta_t}x
> \end{equation}
> $$
>
> is a valid conditional vector field that fits in Theorem 1: its ODE trajectories $$X_t$$ satisfy $$X_t \sim p_t(\cdot \mid z)$$ if $$X_0 \sim \mathcal{N}(0, I)$$.
>
> Proof: construct a conditional flow model $$\psi_t^{\text{target}}(x \mid z)$$ first by defining
>
> $$
> \begin{equation}
> \psi_t^{\text{target}}(x \mid z) = \alpha_t z + \beta_t x
> \end{equation}
> $$
>
> If $$X_t$$ is the ODE trajectory of $$\psi_t^{\text{target}}(\cdot \mid z)$$ with $$X_0 \sim \mathcal{N}(0, I)$$, then by definition
>
> $$
> \begin{equation}
> X_t = \psi_t^{\text{target}}(X_0 \mid z) = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z, \beta_t^2 I) = p_t(\cdot \mid z)
> \end{equation}
> $$
>
> It follows that the trajectories are distributed like the conditional probability path (7). It remains to extract the vector field $$u_t^{\text{target}}(x \mid z)$$ from the flow model $$\psi_t^{\text{target}}(x \mid z)$$. From the last post, a flow should follow
>
> $$
> \begin{align}
> &\frac{d}{dt}\psi_t^{\text{target}}(x \mid z) = u_t^{\text{target}}(\psi_t^{\text{target}}(x \mid z) \mid z) \quad \text{for all } x, z \in \mathbb{R}^d \\
> &\iff \dot{\alpha_t} z + \dot{\beta_t} x = u_t^{\text{target}}(\alpha_t z + \beta_t x \mid z) \quad \text{for all } x, z \in \mathbb{R}^d \\
> &\iff \dot{\alpha_t} z + \dot{\beta_t} \left(\frac{x - \alpha_t z}{\beta_t}\right) = u_t^{\text{target}}(x \mid z) \quad \text{for all } x, z \in \mathbb{R}^d \\
> &\iff \left(\dot{\alpha_t} - \frac{\dot{\beta_t}}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta_t}}{\beta_t}x = u_t^{\text{target}}(x \mid z) \quad \text{for all } x, z \in \mathbb{R}^d
> \end{align}
> $$
>
> In (15), the reparametrization trick $$x \rightarrow \frac{x - \alpha_t z}{\beta_t}$$ is used. In fact, if we look at their physical interpretation, $$x$$ is essentially $$X_0$$ and $$\frac{x - \alpha_t z}{\beta_t}$$ is essentially the current position of the fluid particle at time $$t$$. (16) is exactly the conditional vector field we wanted to show.

Up to this point I have become completely lost about the concepts in the above derivation 🤣. It is a good time to review the important concepts.

1. The probability path $$p_t$$ (the statistical view): the probability density function of the fluid at time $$t$$.
2. The flow $$\psi_t(x_0)$$ (the Lagrangian view): a deterministic trajectory function. It takes an initial point $$x_0$$ and returns the position of the fluid particle at time $$t$$.
3. The vector field $$u_t(x)$$ (the Eulerian view): the velocity of the fluid at a specific spatial location $$x$$ and time $$t$$.

In the proof step (11), the introduction of $$\psi$$ is easier to think of if we start with the goal: $$p_t(\cdot \mid z) \sim \mathcal{N}(\alpha_t z, \beta_t^2 I)$$. To build this Gaussian conditional probability path, we only need to mix $$z$$ with a Gaussian noise $$X_0 \sim \mathcal{N}(0, I)$$. The flow model $$\psi_t^{\text{target}}(x \mid z) = \alpha_t z + \beta_t x$$ is a simple linear transformation that achieves this.

Another question that arises in my mind is why ODE (13) uses the conditional vector field $$u_t^{\text{target}}(\psi_t^{\text{target}}(x \mid z) \mid z)$$ instead of $$u_t^{\text{target}}(x \mid z)$$. A good explanation is that when we track the conditional flow, we use the conditional vector field.

In the Flow Matching discussion, there are two parallel worlds of differential equations:

| Aspect | The Conditional World | The Marginal World |
| :--- | :--- | :--- |
| *Physical Setting* | The destination $$z$$ is **known** and fixed. | The destination is **unknown**. Particles evolve blindly. |
| *Probability Path* | $$p_t(x \mid z)$$ <br>*(Simple, e.g., Gaussian transition)* | $$p_t(x) = \int p_t(x \mid z) p_{\text{data}}(z)\, dz$$ <br>*(Complex mixture distribution)* |
| *Flow (Trajectory)* | $$\psi_t(x_0 \mid z)$$ <br>*(Deterministic path strictly bound to $$z$$)* | $$\psi_t(x_0)$$ <br>*(Path governed by the collective dataset)* |
| *Vector Field* | $$u_t(x \mid z)$$ <br>*(Trivially computable, closed-form)* | $$u_t(x) = \mathbb{E}_{z \sim p_t(\cdot \mid x)}[u_t(x \mid z)]$$ <br>*(Intractable without neural approximation)* |
| *Governing ODE* | $$\frac{d}{dt}\psi_t(x_0 \mid z) = u_t(\psi_t(x_0 \mid z) \mid z)$$ | $$\frac{d}{dt}\psi_t(x_0) = u_t(\psi_t(x_0))$$ |
| *Role in Framework* | Used strictly during **Training** as the simple regression target for the neural network. | Used during **Inference (Sampling)** to generate novel, diverse data points. |

---

Next, we will rigorously discuss Theorem 1 using the continuity equation from mathematics and physics. First, define the divergence operator $$\text{div}$$ as:

$$
\begin{equation}
\text{div}(v_t)(x) = \sum_{i=1}^d \frac{\partial v_t^i}{\partial x_i}(x)
\end{equation}
$$

where $$v_t^i$$ is the $$i$$-th component of vector field $$v_t$$. It is also denoted $$\nabla \cdot v_t$$.

> Theorem 2 (Continuity Equation):
> Consider a flow model with vector field $$u_t^{\text{target}}$$ and $$X_0 \sim p_{\text{init}} = p_0$$. Then $$X_t \sim p_t$$ for all $$t \in [0, 1]$$ if and only if the following continuity equation is satisfied:
>
> $$
> \begin{equation}
> \partial_t p_t(x) = - \text{div}(p_t(x) u_t^{\text{target}}(x)) \quad \text{for all } x \in \mathbb{R}^d, t \in [0, 1]
> \end{equation}
> $$

Intuitively, the LHS of (18) is the rate of change of probability density at a point $$x$$. This should equal the net inflow. For the RHS of (18), at $$x$$ the field flows along the vector field $$u_t^{\text{target}}(x)$$. Weighting by the density $$p_t(x)$$ and taking the negative divergence, we get the net inflow at $$x$$. Since probability mass is conserved, the rate of change of probability density at $$x$$ should equal the net inflow at $$x$$.

Proof of Theorem 1 with Theorem 2:

We have to show that the marginal vector field $$u_t^{\text{target}}(x)$$, as defined in (8), satisfies the continuity equation (18). The proof is by direct calculation.

$$
\begin{align}
\partial_t p_t(x) &= \partial_t \int p_t(x \mid z) p_{\text{data}}(z)\, dz \\
&= \int \partial_t p_t(x \mid z) p_{\text{data}}(z)\, dz \\
&= -\int \text{div}(p_t(\cdot \mid z) u_t^{\text{target}}(\cdot \mid z))(x) p_{\text{data}}(z)\, dz \\
&= - \text{div} \left(\int p_t(x \mid z) u_t^{\text{target}}(x \mid z) p_{\text{data}}(z)\, dz\right) \\
&= - \text{div} \left(p_t(x) \int u_t^{\text{target}}(x \mid z) \frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)}\, dz\right)(x) \\
&= - \text{div} (p_t(x) u_t^{\text{target}}(x))(x)
\end{align}
$$

where the step (20) $$\implies$$ (21) uses the continuity equation (Theorem 2) for the conditional vector field $$u_t^{\text{target}}(x \mid z)$$, because we already know that the conditional vector field generates the conditional probability path.

The steps of the proof essentially show that, given the conditional vector field $$u_t^{\text{target}}(x \mid z)$$ and the marginal vector field defined in (8), the continuity equation for the marginal vector field $$u_t^{\text{target}}(x)$$ is satisfied. By Theorem 2, we conclude that the marginal vector field $$u_t^{\text{target}}(x)$$ indeed follows the marginal probability path $$p_t(x)$$. So we are done with the proof of Theorem 1.

## Learning the Marginal Vector Field

Finally, we are ready for the training algorithm. Goal: train the neural network $$u_t^{\theta}$$ to approximate the marginal vector field $$u_t^{\text{target}}$$. If this holds, the endpoint $$X_1 \sim p_{\text{data}}$$ has the desired distribution by Theorem 1.

An intuitive way of obtaining $$u_t^{\theta} \approx u_t^{\text{target}}$$ is to minimize the expected squared difference between the two vector fields:

$$ \begin{align}
\mathcal{L}_{\text{FM}}(\theta) &= \mathbb{E}_{t \sim \text{U}[0, 1],x \sim p_t} \left[\|u_t^{\theta}(x) - u_t^{\text{target}}(x)\|_2^2\right] \\
&= \mathbb{E}_{t \sim \text{U}[0, 1],z \sim p_{\text{data}},x \sim p_t(\cdot \mid z)} \left[\|u_t^{\theta}(x) - u_t^{\text{target}}(x)\|_2^2\right] 
\end{align}
$$

where $$p_t(x) = \int p_t(x \mid z) p_{\text{data}}(z)\, dz$$ is the marginal probability path. We used the sampling procedure given in (3). The loss essentially says:
1. Sample a time $$t$$ uniformly from $$[0, 1]$$.
2. Sample a data point $$z$$ from the data distribution $$p_{\text{data}}$$.
3. Sample a point $$x$$ from the conditional probability path $$p_t(\cdot \mid z)$$. This is done by adding some noise to $$z$$ according to the noise schedule.
4. Compute MSE between the neural network $$u_t^{\theta}(x)$$ and the marginal vector field $$u_t^{\text{target}}(x)$$. 

Unfortunately, we cannot compute $$u_t^{\text{target}}(x)$$ directly because it requires computing the intractable integral, even though we know the formula of $$u_t^{\text{target}}(x)$$ from Theorem 1. Instead, we leverage the fact that the conditional vector field $$u_t^{\text{target}}$$ is tractable. Define conditional flow matching loss as:

$$
\begin{equation}
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1], z \sim p_{\text{data}}, x \sim p_t(\cdot \mid z)} \left[\|u_t^{\theta}(x) - u_t^{\text{target}}(x \mid z)\|_2^2\right]
\end{equation}
$$

As we have an analytical formula for $$u_t^{\text{target}}(x \mid z)$$, we can compute the conditional flow matching loss $$\mathcal{L}_{\text{CFM}}(\theta)$$ and use it to train the neural network $$u_t^{\theta}$$. The following theorem shows that minimizing the conditional flow matching loss is equivalent to minimizing the flow matching loss.

> Theorem 3 (Conditional Flow Matching):
> The marginal flow matching loss equals the conditional flow matching loss up to a constant. That is,
>
> $$\mathcal{L}_{\text{FM}}(\theta) = \mathcal{L}_{\text{CFM}}(\theta) + C$$
>
> where $$C$$ is a constant independent of $$\theta$$. Therefore, the gradient with respect to $$\theta$$ is the same for both losses:
> 
> $$\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)$$
> 
> Hence, minimizing $$\mathcal{L}_{\text{CFM}}(\theta)$$ with, e.g., stochastic gradient descent is equivalent to minimizing $$\mathcal{L}_{\text{FM}}(\theta)$$ in the same fashion. In particular, for the minimizer $$\theta^*$$ of $$\mathcal{L}_{\text{CFM}}(\theta)$$, we have $$u_t^{\theta^*} \approx u_t^{\text{target}}$$, i.e. the neural network approximates the marginal vector field (assuming infinitely expressive parametrization).

Proof. For brevity, write $$\mathbb{E}_{p_t}$$ for the expectation over $$t \sim U[0,1],\, x \sim p_t$$, and $$\mathbb{E}_{p_t(\cdot \mid z)}$$ for the expectation over $$t \sim U[0,1],\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)$$. Expanding the MSE,

$$
\begin{align}
\mathcal{L}_{\text{FM}}(\theta) &= \mathbb{E}_{p_t} \left[\|u_t^{\theta}(x) - u_t^{\text{target}}(x)\|_2^2\right] \\
&= \mathbb{E}_{p_t} \left[\|u_t^{\theta}(x)\|_2^2 - 2 u_t^{\theta}(x)^\top u_t^{\text{target}}(x) + \|u_t^{\text{target}}(x)\|_2^2\right] \\
&= \mathbb{E}_{p_t} \left[\|u_t^{\theta}(x)\|_2^2\right] - 2 \mathbb{E}_{p_t} \left[u_t^{\theta}(x)^\top u_t^{\text{target}}(x)\right] + \underbrace{\mathbb{E}_{p_t} \left[\|u_t^{\text{target}}(x)\|_2^2\right]}_{=:C_1} \\
&= \mathbb{E}_{p_t(\cdot \mid z)} \left[\|u_t^{\theta}(x)\|_2^2\right] - 2 \mathbb{E}_{p_t} \left[u_t^{\theta}(x)^\top u_t^{\text{target}}(x)\right] + C_1
\end{align}
$$

A constant has been factored out. Let's examine the second term:

$$ \begin{align}
\mathbb{E}_{p_t} \left[u_t^{\theta}(x)^\top u_t^{\text{target}}(x)\right] &= \int_0^1 \int_{\mathbb{R}^d} p_t(x) u_t^{\theta}(x)^\top u_t^{\text{target}}(x)\, dx\, dt \\
&= \int_0^1 \int_{\mathbb{R}^d} p_t(x) u_t^{\theta}(x)^\top \left[ \int u_t^{\text{target}}(x \mid z) \frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)}\, dz \right] dx\, dt \\
&= \int_0^1 \int_{\mathbb{R}^d} \int u_t^{\theta}(x)^\top u_t^{\text{target}}(x \mid z) p_t(x \mid z) p_{\text{data}}(z)\, dz\, dx\, dt \\
&= \mathbb{E}_{p_t(\cdot \mid z)} \left[u_t^{\theta}(x)^\top u_t^{\text{target}}(x \mid z)\right]
\end{align}
$$

The core change in this step is that we start from an expectation over the marginal distribution $$p_t(x)$$ and convert it into an expectation over the conditional distribution $$p_t(x \mid z)$$. Defining $$C_2 := \mathbb{E}_{p_t(\cdot \mid z)}\left[\|u_t^{\text{target}}(x \mid z)\|_2^2\right]$$ and putting this back into $$\mathcal{L}_{\text{FM}}$$:

$$\begin{align}
\mathcal{L}_{\text{FM}}(\theta) &= \mathbb{E}_{p_t(\cdot \mid z)} \left[\|u_t^{\theta}(x)\|_2^2\right] - 2 \mathbb{E}_{p_t(\cdot \mid z)} \left[u_t^{\theta}(x)^\top u_t^{\text{target}}(x \mid z)\right] + C_1 \\
&= \mathbb{E}_{p_t(\cdot \mid z)} \left[\|u_t^{\theta}(x)\|_2^2 - 2 u_t^{\theta}(x)^\top u_t^{\text{target}}(x \mid z) + \|u_t^{\text{target}}(x \mid z)\|_2^2\right] + C_1 - C_2 \\
&= \mathbb{E}_{p_t(\cdot \mid z)} \left[\|u_t^{\theta}(x) - u_t^{\text{target}}(x \mid z)\|_2^2\right] + C_1 - C_2 \\
&= \mathcal{L}_{\text{CFM}}(\theta) + \underbrace{C_1 - C_2}_{=:C}
\end{align}
$$

This finishes the proof.

Therefore, flow matching training consists of minimizing the conditional flow matching loss. Several striking features of this training algorithm are:
1. The training is **simulation-free**. No need to actually simulate any ODEs.
2. An easy regression target $$u_t^{\text{target}}(x \mid z)$$.

After training $$u_t^{\theta}$$, we can simulate the flow model with algorithm mentioned in the last post to generate samples from the data distribution $$p_{\text{data}}$$.

Algorithm to train flow matching model (for Gaussian CondOT path $$p_t(x \mid z) = \mathcal{N}(tz, (1-t)^2 I)$$)

Require: A dataset of samples $$z \sim p_{\text{data}}$$, neural network $$u_t^{\theta}$$
1. **for** each mini-batch of data points **do**
2.   sample a data example $$z$$ from the dataset.
3.   sample a time $$t \sim U[0, 1]$$.
4.   sample a noise $$\epsilon \sim \mathcal{N}(0, I)$$.
5.   compute $$x =  tz + (1-t)\epsilon$$. (General case: $$x \sim p_t(\cdot \mid z)$$)
6.   compute loss $$\mathcal{L}_{\text{CFM}}(\theta) = \|u_t^{\theta}(x) - (z - \epsilon) \|_2^2$$. (General case:$$= \|u_t^{\theta}(x) - u_t^{\text{target}}(x \mid z)\|_2^2$$)
7.   update $$\theta \leftarrow \text{gradUpdate}(\mathcal{L}_{\text{CFM}}(\theta))$$.
8. **end for**

> **CondOT** means Conditional Optimal Transport. It is a special case of the Gaussian conditional probability path with schedulers $$\alpha_t = t$$ and $$\beta_t = 1-t$$. The trajectory is a straight line, compared to the curved paths in classical DDPM. Classical methods are heavily governed by Brownian motion, so they must take small steps. CondOT allows numerical ODE solvers (like Euler or Runge-Kutta) to take massive step sizes, enabling high-quality generation in very few steps.

For loss computation, we are computing the MSE between the neural network $$u_t^{\theta}(x)$$ and the conditional vector field $$u_t^{\text{target}}(x \mid z)$$. For the Gaussian conditional probability path, we have a closed-form formula as shown in Example 2. With schedulers $$\alpha_t = t$$ and $$\beta_t = 1-t$$, the derivatives are $$\dot{\alpha_t} = 1$$ and $$\dot{\beta_t} = -1$$, so $$\frac{\dot{\beta_t}}{\beta_t} = \frac{-1}{1-t}$$. Hence, $$u_t^{\text{target}}(x \mid z) = \left(1 - \frac{-1}{1-t}\, t\right)z + \frac{-1}{1-t}\big(tz + (1-t)\epsilon\big) = \frac{1}{1-t}z - \frac{t}{1-t}z - \epsilon = \frac{1-t}{1-t}z - \epsilon = z - \epsilon, $$ which matches the $$z - \epsilon$$ used in the algorithm above. (The coefficient of $$z$$ is $$1 - \frac{\dot{\beta_t}}{\beta_t}\alpha_t = 1 - \frac{-1}{1-t}\cdot t = \frac{1}{1-t}$$; forgetting the factor $$\alpha_t = t$$ here is what produces the erroneous $$2z - \epsilon$$.)

> Example 3: Flow Matching for Gaussian CondOT Path.
> 
> Consider the general case instead of special situation in the algorithm. Let $$p_t(\cdot \mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I)$$ be a Gaussian conditional probability path with noise schedulers $$\alpha_t$$ and $$\beta_t$$. Let $$\dot{\alpha_t} = \partial_t \alpha_t$$ and $$\dot{\beta_t} = \partial_t \beta_t$$ be the respective time derivatives of the noise schedulers. We may sample from the conditional path via
> 
> $$ \begin{equation}
> \epsilon \sim \mathcal{N}(0, I) \implies x = \alpha_t z + \beta_t \epsilon \sim \mathcal{N}(\alpha_t z, \beta_t^2 I) = p_t(\cdot \mid z)
> \end{equation}$$
>
> As we derived in Example 2, the conditional vector field $$u_t^{\text{target}}(x \mid z)$$ is given by:
>
> $$
> u_t^{\text{target}}(x \mid z) = \left(\dot{\alpha_t} - \frac{\dot{\beta_t}}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta_t}}{\beta_t}x
> $$
>
> Hence, the conditional flow matching loss is given by:
>
> $$ \begin{align}
> \mathcal{L}_{\text{CFM}}(\theta) &= \mathbb{E}_{z \sim p_{\text{data}}, t \sim U[0, 1], x \sim \mathcal{N}(\alpha_t z, \beta_t^2 I)} \left[ \|u_t^{\theta}(x) - (\dot{\alpha_t} - \frac{\dot{\beta_t}}{\beta_t}\alpha_t)z - \frac{\dot{\beta_t}}{\beta_t}x\|_2^2 \right] \\
> &= E_{z \sim p_{\text{data}}, t \sim U[0, 1], \epsilon \sim \mathcal{N}(0, I)} \left[ \|u_t^{\theta}(\alpha_t z + \beta_t \epsilon) - (\dot{\alpha_t} - \frac{\dot{\beta_t}}{\beta_t}\alpha_t)z - \frac{\dot{\beta_t}}{\beta_t}(\alpha_t z + \beta_t \epsilon)\|_2^2 \right] \\
> &= E_{z \sim p_{\text{data}}, t \sim U[0, 1], \epsilon \sim \mathcal{N}(0, I)} \left[ \|u_t^{\theta}(\alpha_t z + \beta_t \epsilon) - (\dot{\alpha_t}z + \dot{\beta_t}\epsilon)\|_2^2 \right]
> \end{align} $$
>
> The second line substitutes $$x = \alpha_t z + \beta_t \epsilon$$; the third line uses algebraic simplification. The final loss is a simple MSE between the neural network $$u_t^{\theta}$$ and the linear target $$\dot{\alpha_t}z + \dot{\beta_t}\epsilon$$.
>
> In special case with $$\alpha_t = t$$ and $$\beta_t = 1-t$$, we have $$\dot{\alpha_t} = 1$$ and $$\dot{\beta_t} = -1$$. Hence, the conditional flow matching loss is given by:
>
> $$ \mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{z \sim p_{\text{data}}, t \sim U[0, 1], \epsilon \sim \mathcal{N}(0, I)} \left[ \|u_t^{\theta}(tz + (1-t)\epsilon) - (z - \epsilon)\|_2^2 \right] $$
> 
> Many famous SOTA models have been trained using this simple yet effective procedure, e.g. Stable Diffusion 3, Meta's Movie Gen Video.