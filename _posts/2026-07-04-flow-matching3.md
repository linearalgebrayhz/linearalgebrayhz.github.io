---
layout: post
title: "Flow Matching and Diffusion Models 3"
date: 2026-07-04
category: "Machine Learning"
tags: [score matching, diffusion models, generative models]
author: "Haozhe Yang"
image: /assets/images/posts/flow-matching-3.svg
license: cc-by-nc-sa
acknowledgement: "These notes are based on MIT Course 6.S184: *Generative AI with Stochastic Differential Equations* ([course site](https://diffusion.csail.mit.edu/))."
excerpt: "In this post, we discuss the score function and score matching."
---

In previous posts, we have discussed how to train a flow model with flow matching. In this post, we discuss **diffusion models** and how to train them with **score matching**. With this section we should also be able to see some connections between flow matching and score matching.

## Conditional and Marginal Score Functions

Let $$q(x)$$ be an arbitrary data distribution. Then the score function of $$q(x)$$ is defined as: $$\nabla \log q(x)$$ i.e. the gradient of the log-likelihood of $$q$$ with respect to $$x$$. Intuitively, the score is the direction of steepest ascent with respect to log-likelihood. 

Return to the setting of conditional probability path $$p_t(x \mid z)$$ and marginal probability path $$p_t(x)$$ from the previous post. Then we can equivalently define the conditional score function as $$\nabla_x \log p_t(x \mid z)$$ and the marginal score function as $$\nabla_x \log p_t(x)$$. Similarly, we can have marginal score expressed as:

$$\begin{equation}\nabla \log p_t(x) = \int \nabla_x \log p_t(x \mid z) \frac{p_t(x\mid z) p_{\text{data}}(z)}{p_t(x)} dz \end{equation}$$

Again, the marginal score is the expectation of the conditional score with respect to the posterior distribution $$p_t(z \mid x)$$. Similar to what we have done in the previous post, $$ \nabla_x \log p_t(x) = \mathbb{E}_{z \sim p_t(z \mid x)} [\nabla_x \log p_t(x \mid z)] $$. Physical intuition: when given a sample $$x$$ with noise, we do not know which $$z$$ it corresponds to. The posterior distribution $$p_t(z \mid x)$$ serves as a confidence measure of how likely $$x$$ corresponds to each $$z$$. The marginal score is the expectation of the conditional score with respect to this confidence measure.

The derivation is also straightforward. We have:

$$\begin{align}
\nabla \log p_t(x) = \frac{\nabla p_t(x)}{p_t(x)} &= \frac{\nabla \int p_t(x\mid z) p_{\text{data}}(z) dz}{p_t(x)} \\ &= \frac{\int \nabla p_t(x\mid z) p_{\text{data}}(z) dz}{p_t(x)} \\ &= \int \nabla \log p_t(x \mid z) \frac{p_t(x\mid z) p_{\text{data}}(z)}{p_t(x)} dz
\end{align}$$

> Example 1 (Score function for Gaussian Probability Paths.)
> 
> For the Gaussian path $$p_t(x \mid z) = \mathcal{N}(x; \alpha_t z, \beta_t^2 I)$$, we can use the form of the Gaussian probability density $$p_t(x \mid z) = \frac{1}{(2\pi)^{d/2}\beta_t^d} \exp\left(-\frac{1}{2} (x - \alpha_t z)^T (\beta_t^2 I)^{-1} (x - \alpha_t z)\right)$$ to get
> 
> $$\begin{align}
> \nabla_x \log p_t(x \mid z) &= \nabla_x (-\frac{1}{2}(x - \alpha_t z)^T (\beta_t^2 I)^{-1} (x - \alpha_t z)) + \nabla_x \log \frac{1}{(2\pi)^{d/2}\beta_t^d} \\ &= -(\beta_t^2 I)^{-1} (x - \alpha_t z) + 0 \\ &= -\frac{1}{\beta_t^2} (x - \alpha_t z)
> \end{align}$$
> 

> side note: derivative of quadratic form: $$\nabla x^T A x = (A + A^T) x$$, for vector $$x$$ and matrix $$A$$. If $$A$$ is symmetric, then $$\nabla x^T A x = 2Ax$$. In our case, $$A = (\beta_t^2 I)^{-1}$$ is symmetric. (actually the inverse here is easy to get).
> 
> Quick derivation: for  $$f: \mathbb{R}^n \to \mathbb{R}$$, $$f(x) = x^T A x$$, the gradient with respect to vector $$x \in \mathbb{R}^n$$ should be a vector in $$\mathbb{R}^n$$. Let $$x$$ be $$x = [x_1, x_2, \ldots, x_n]^T$$. Matrix $$A = [a_{ij}]_{n \times n}$$. $$f(x) = x^T A x = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j$$. For any components $$x_k$$, we have $$\frac{\partial f}{\partial x_k} = \sum_{j=1}^n a_{kj} x_j + \sum_{i=1}^n a_{ik} x_i = [Ax]_k + [A^T x]_k$$.
> 
> Other common vector-variable derivatives: $$\nabla_x (b^T x) = b$$, $$\nabla_x (x^T b) = b$$ for vector $$b \in \mathbb{R}^n$$, which is not a function of $$x$$ (These are easy to verify by writing out the components).
> 
> Taking derivative of vector-valued function with respect to vector yields jacobian matrix. For example, $$f: \mathbb{R}^n \to \mathbb{R}^m$$, $$f(x) = Ax$$, then $$\nabla_x f(x) = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} = A$$. (For A not being a function of $$x$$)

Note that the score function for Gaussian probability paths is linear in $$x$$ and $$z$$, as it is for conditional vector field $$u_t(x \mid z)$$. Therefore, it is possible to convert between them. 

> Proposition 1 (Conversion Formula for Gaussian Probability Paths.)
>
> For the Gaussian probability path $$p_t(x \mid z) = \mathcal{N}(x; \alpha_t z, \beta_t^2 I)$$, the conditional (resp. marginal) vector field and the conditional (resp. marginal) score function are related by the following identities:
>
> $$\begin{align}
> u_t^{\text{target}}(x \mid z) &= a_t \nabla \log p_t(x \mid z) + b_t x, \, a_t = (\beta_t^2 \frac{\dot{\alpha_t}}{\alpha_t} - \dot{\beta_t}\beta_t), \, b_t = \frac{\dot{\alpha_t}}    {\alpha_t} \\
> u_t^{\text{target}}(x) &= a_t \nabla \log p_t(x) + b_t x.
> \end{align}$$

*Proof.* For the conditional vector field and conditional score, we can derive:

$$u_t^{\text{target}}(x \mid z) = (\dot{\alpha_t} - \frac{\dot{\beta_t}}{\beta_t} \alpha_t) z + \frac{\dot{\beta_t}}{\beta_t} x = (\beta_t^2\frac{\dot{\alpha_t}}{\alpha_t} - \dot{\beta_t}\beta_t)(\frac{\alpha_t z - x}{\beta_t^2}) + \frac{\dot{\alpha_t}}{\alpha_t}x = (\beta_t^2\frac{\dot{\alpha_t}}{\alpha_t} - \dot{\beta_t}\beta_t) \nabla \log p_t(x \mid z) + \frac{\dot{\alpha_t}}{\alpha_t} x$$

Then, for the marginal vector field and marginal score, we can derive:

$$u_t^{\text{target}}(x) = \mathbb{E}_{z \sim p_t(z \mid x)} [u_t^{\text{target}}(x \mid z)] = \mathbb{E}_{z \sim p_t(z \mid x)} [a_t \nabla \log p_t(x \mid z) + b_t x] = a_t \mathbb{E}_{z \sim p_t(z \mid x)} [\nabla \log p_t(x \mid z)] + b_t x = a_t \nabla \log p_t(x) + b_t x$$

which is straightforward from linearity of expectation. This finishes the proof.

Proposition 1 shows that the conditional and marginal vector fields can be expressed as linear combinations of the conditional and marginal score functions, respectively. Once we learned $$u_t^{\text{target}}$$, we've also learned the score function $$\nabla \log p_t(x)$$, and vice versa. Therefore, many diffusion models use a neural network to learn the score function instead of the vector field.

> Remark: Reparametrization of Score.
>
> The reparametrization formula for Gaussian probability paths in (8) is possible because both sides are linear functions in $$x$$ and $$z$$. Expressing the marginalized score function (or vector field) in the expectation form over the posterior distribution $$p_t(z \mid x)$$ helps the derivation. Furthermore, doing so helps with numerical/training stability. One common choice is the posterior mean, often referred to as the **denoiser**. Formally, we define the conditional and marginal denoisers as:
>
> $$\begin{align}
> D_t(x \mid z) &= z \\ D_t(x) &= \int z \, \frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)} dz \\ &= \frac{1}{\dot{\alpha_t}\beta_t - \dot{\beta_t}\alpha_t}(\beta_t u_t^{\text{target}}(x) - \dot{\beta_t} x)
> \end{align}$$
>
> The denoiser has a very intuitive meaning: it is the expected value of clean data z given noisy data x. 

The lecture notes are very clear here about why we introduce the denoiser. Previously, we dealt with the vector field $$u_t(x)$$ and score function $$\nabla \log p_t(x)$$. They point out where to go next when we are at a noisy point $$x$$. The numerical instability arises as $$t \to 1$$ (very close to clean data, where $$\beta_t \to 0$$). The denoiser is very straightforward. Instead of asking where to go next, it directly asks what is the final clean data $$z$$. $$D_t(x)$$ is exactly the posterior mean of $$z$$ when $$x$$ is observed. No matter what $$t$$ is, the range of a clean data $$z$$ is fixed and bounded. It is much more stable for neural networks to learn some bounded target.

Detailed derivation in the above remark: we first try to find the relationship between the denoiser and the score. Adding noise to clean data $$z$$ is Gaussian Probability path $$p_t(x \mid z) = \mathcal{N}(x; \alpha_t z, \beta_t^2 I)$$. For Gaussian distribution, the conditional score can be expressed analytically (see Example 1): $$\frac{\alpha_t z - x}{\beta_t^2} $$. Then, marginal score is the conditional score averaged over the posterior distribution $$p_t(z \mid x)$$: $$ \nabla_x \log p_t(x) = \mathbb{E}_{z \sim p_t(z \mid x)} [\nabla_x \log p_t(x \mid z)] $$.

Plugging in the conditional score, we have: $$ \nabla_x \log p_t(x) = \mathbb{E}_{z \sim p_t(z \mid x)} \left[ \frac{\alpha_t z - x}{\beta_t^2} \right] = \frac{\alpha_t \mathbb{E}_{z \sim p_t(z \mid x)}[z] - x}{\beta_t^2} = \frac{\alpha_t D_t(x) - x}{\beta_t^2}$$. Rearranging a little bit, we reach the conversion formula between the denoiser and the score function (Tweedie's formula):

$$ \begin{equation}
D_t(x) = \frac{x}{\alpha_t} + \frac{\beta_t^2}{\alpha_t} \nabla_x \log p_t(x)
\end{equation} $$

Next, we plug in vector field $$u_t$$. We already know that $$ u_t^{\text{target}}(x) = a_t \nabla \log p_t(x) + b_t x $$. Score: $$ \nabla \log p_t(x) = \frac{u_t^{\text{target}}(x) - b_t x}{a_t} $$. Plugging in the score function into Tweedie's formula (13), we have:

$$ \begin{equation}
D_t(x) = \frac{x}{\alpha_t} + \frac{\beta_t^2}{\alpha_t} \left( \frac{u_t^{\text{target}}(x) - b_t x}{a_t} \right) 
\end{equation} $$

$$a_t$$, $$b_t$$ are defined in Proposition 1. Rearranging the terms, we have:

$$ \frac{u_t - b_t x}{a_t} = \frac{\alpha_t}{\beta_t (\beta_t \dot{\alpha}_t - \dot{\beta}_t \alpha_t)} \left( u_t - \frac{\dot{\alpha}_t}{\alpha_t} x \right) $$

$$ \frac{\beta_t^2}{\alpha_t} \times \frac{\alpha_t}{\beta_t (\beta_t \dot{\alpha}_t - \dot{\beta}_t \alpha_t)} \left( u_t - \frac{\dot{\alpha}_t}{\alpha_t} x \right) = \frac{\beta_t}{\dot{\alpha}_t \beta_t - \dot{\beta}_t \alpha_t} \left( u_t - \frac{\dot{\alpha}_t}{\alpha_t} x \right) $$

So $$D_t(x)$$ can be expressed as:

$$ D_t(x) = \frac{1}{\dot{\alpha}_t \beta_t - \dot{\beta}_t \alpha_t} \left[ \frac{\dot{\alpha}_t \beta_t - \dot{\beta}_t \alpha_t}{\alpha_t} x + \beta_t u_t - \frac{\beta_t \dot{\alpha}_t}{\alpha_t} x \right] $$

Finally,

$$ D_t(x) = \frac{1}{\dot{\alpha}_t \beta_t - \dot{\beta}_t \alpha_t} (\beta_t u_t^{\text{target}}(x) - \dot{\beta}_t x) $$

People often call such models denoising diffusion models as learning the denoiser is equivalent to learning the score function, which is equivalent to learning the vector field.

## Sampling with SDEs

We have discussed how to construct a trajectory $$X_t$$ of an ODE that follows a desired probability path $$p_t(x)$$ via a marginal vector field $$u_t(x)$$. But this approach is constrained to flow models. What about diffusion models? We can extend the result to SDEs using score functions. 

> Theorem 1 (SDE Extension)
>
> Define the conditional and marginal vector fields as $$u_t^{\text{target}}(x \mid z)$$ and $$u_t^{\text{target}}(x)$$ as before. Then, for any diffusion coefficient $$\sigma_t \geq 0$$, we may construct an SDE by adding **stochastic dynamics** to the original ODE as follows:
> 
$$
\begin{equation}
\begin{aligned}
X_0 \sim p_{\text{init}}, \quad dX_t &= \underbrace{u_t^{\text{target}}(X_t) dt}_{\text{ODE}} + \underbrace{\frac{\sigma_t^2}{2}\nabla \log p_t(X_t)dt + \sigma_t dW_t}_{\text{stochastic dynamics}} \\ &= \left[u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2}\nabla \log p_t(X_t)\right] dt + \sigma_t dW_t
\end{aligned}
\end{equation}$$
>
$$ \begin{equation}
\Rightarrow X_t \sim p_t \quad \forall t \in [0, 1]
\end{equation}
$$
>
> In particular, $$X_1 \sim p_{\text{data}}$$ for this SDE. We note that the stochastic dynamics are closely related to the **Langevin dynamics**, and can be thought of as injecting noise while preserving the marginal distribution $$p_t$$. We will discuss Langevin dynamics in more detail later.

Note that the above result is striking in that we can choose any diffusion coefficient $$\sigma_t$$ even after having trained the networks. In theory, Theorem 1 holds for any choice of $$\sigma_t$$. However, in practice, we suffer from both training error (the neural network does not perfectly approximate the marginal vector field and score) and simulation error (e.g. for $$\sigma_t \gg 0$$ we would need to take prohibitively small step sizes in Algorithm of Sampling from a Diffusion Model).

For Gaussian probability paths, we get the score function for free by having learned the marginal vector field.

> Example 2 (Gaussian SDE Extension trick)
>
> By proposition 1, for Gaussian probability paths, we can express the SDE from Theorem 1 purly using score functions:
>
$$
\begin{align}
&X_0 \sim p_{\text{init}}, \quad dX_t = \left[(a_t + \frac{\sigma_t^2}{2})\nabla \log p_t(X_t) + b_t X_t\right] dt + \sigma_t dW_t \\ \Rightarrow &X_t \sim p_t \quad \forall t \in [0, 1]
\end{align}
$$
> 

In the remainder of this section, we prove the above theorem via the **Fokker-Planck equation**, which extends the continuity equation from ODEs to SDEs. First, define the Laplacian operator $$\Delta$$ as:

$$\begin{equation}
\Delta w_t(x) = \sum_{i=1}^d \frac{\partial^2 w_t(x)}{\partial x_i^2} = \text{div}(\nabla w_t)(x) 
\end{equation}
$$

for scalar field $$w_t: \mathbb{R}^d \to \mathbb{R}$$.

> Theorem 2 (Fokker-Planck Equation)
>
> Let $$p_t$$ be a probability path and let us consider the SDE
>
$$
X_0 \sim p_{\text{init}}, \quad dX_t = u_t(X_t) dt + \sigma_t dW_t.
$$
>
> Then $$X_t$$ has distribution $$p_t$$ for all $$t \in [0, 1]$$ if and only if the **Fokker-Planck equation** is satisfied:
>
$$
\begin{equation}
\partial_t p_t(x) = -\text{div}(p_t(x) u_t(x)) + \frac{\sigma_t^2}{2} \Delta p_t(x) \quad \forall t \in [0, 1], x \in \mathbb{R}^d
\end{equation}
$$

Compare with continuity equation from last post 

$$\partial_t p_t(x) = - \text{div}(p_t(x) u_t^{\text{target}}(x)) \quad \text{for all } x \in \mathbb{R}^d, t \in [0, 1]$$

The Laplacian term $$\frac{\sigma_t^2}{2} \Delta p_t(x)$$ is the new term that arises from the stochastic dynamics. We can draw some connection to the heat diffusion equation.

Proof of Theorem 1. By Theorem 2, we need to show that the SDE in Theorem 1 satisfies the Fokker-Planck equation. We have:

$$ \begin{aligned}
\partial_t p_t(x) &= -\text{div}(p_t(x) u_t^{\text{target}}(x)) \\ &= -\text{div}(p_t(x) u_t^{\text{target}}(x)) - \frac{\sigma_t^2}{2}\Delta p_t(x) + \frac{\sigma_t^2}{2}\Delta p_t(x) \\ &= -\text{div}(p_t(x) u_t^{\text{target}}(x)) - \underbrace{\text{div}(\frac{\sigma_t^2}{2} \nabla p_t(x))}_{\text{recall laplacian (19)}} + \frac{\sigma_t^2}{2}\Delta p_t(x) \\ &= -\text{div}(p_t(x) u_t^{\text{target}}(x)) - \text{div}(p_t\left[\frac{\sigma_t^2}{2} \nabla \log p_t\right])(x) + \frac{\sigma_t^2}{2}\Delta p_t(x) \\ &= -\text{div}\left(p_t \left[u_t^{\text{target}} + \frac{\sigma_t^2}{2} \nabla \log p_t\right]\right)(x) + \frac{\sigma_t^2}{2}\Delta p_t(x)
\end{aligned}
$$

The above derivation shows that the SDE in Theorem 1 satisfies the Fokker-Planck equation, which implies that $$X_t \sim p_t$$ for all $$t \in [0, 1]$$. This completes the proof of Theorem 1.

> Remark on Langevin Dynamics. 
> The above construction has a famous special case when the probability path is constant, i.e. $$p_t(x) = p$$ for a fixed distribution $$p$$. In this case, we set $$u_t^{\text{target}}(x) = 0$$ and the SDE reduces to:
> 
$$
\begin{equation}
dX_t = \frac{\sigma_t^2}{2} \nabla \log p_t(X_t) dt + \sigma_t dW_t
\end{equation}
$$
>
> which is commonly known as Langevin dynamics. $$p_t$$ is constant implies that $$\partial_t p_t(x) = 0$$. It follows immediately from Theorem 1 that these dynamics satisfy the Fokker-Planck equation for the static path $$p_t = p$$. Therefore, we may conclude that $$p$$ is a stationary distribution of the Langevin dynamics. 
>
$$\begin{equation}
X_0 \sim p \implies X_t \sim p \quad \forall t \geq 0
\end{equation}
$$
>
> As with many Markov processes, these dynamics converge to the stationary distribution. That is, if we instead take $$X_0 \sim p' \neq p$$, so that $$X_t \sim p_t'$$, then under mild conditions $$p_t' \rightarrow p$$. This fact makes Langevin dynamics extremely useful, and it accordingly serves as the basis for e.g., molecular dynamics simulations, and many other Markov chain Monte Carlo (MCMC) methods across Bayesian statistics and the natural sciences. In particular, the Ornstein-Uhlenbeck processes are recovered as the special case of the Langevin dynamics when $$p$$ is a Gaussian, and serve as the basis for initial formulations of diffusion models.

*If I have time I will make some posts about MCMC and bayesian statistics.(LOL)*

## Score Matching

It remains to show how we can learn the marginal score function $$\nabla \log p_t(x)$$. For Gaussian probability path, we can simply transform using proposition 1. For the general case, we can also learn marginal score functions directly. To approximate the marginal score $$\nabla \log p_t(x)$$, we use a neural network (called the score network) $$s^{\theta}_t : \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d$$. In the same way as we define flow matching loss, we define the **score matching loss** as:

$$
\begin{aligned}
\mathcal{L}_{\text{SM}}(\theta) &= \mathbb{E}_{t \sim \text{Uniform}[0, 1], z \sim p_{\text{data}}, x \sim p_t(\cdot \mid z)} \left[ \| s^{\theta}_t(x) - \nabla \log p_t(x) \|_2^2 \right] \\
\mathcal{L}_{\text{CSM}}(\theta) &= \mathbb{E}_{t \sim \text{Uniform}[0, 1], z \sim p_{\text{data}}, x \sim p_t(\cdot \mid z)} \left[ \| s^{\theta}_t(x) - \nabla \log p_t(x \mid z) \|_2^2 \right]
\end{aligned}
$$

As before, we would want to minimize the score matching loss but we do not know the score function $$\nabla \log p_t(x)$$. However, we can use the same trick as in flow matching to rewrite the loss in conditional form.

> Theorem 3
> The score matching loss equals the conditional score matching loss up to a constant:
>
$$
\mathcal{L}_{\text{SM}}(\theta) = \mathcal{L}_{\text{CSM}}(\theta) + \text{constant}
$$
>
Therefore, their gradients:
>
$$
\nabla_{\theta} \mathcal{L}_{\text{SM}}(\theta) = \nabla_{\theta} \mathcal{L}_{\text{CSM}}(\theta)
$$
>
> In particular, for the minimizer $$\theta^*$$, it will hold that $$s_t^{\theta^*} = \nabla \log p_t$$

The proof is the same as that in flow matching by replacing $$u_t^{\text{target}}$$ with $$\nabla \log p_t$$. We shall not repeat it here. 

The following example instantiates the denoising score matching loss for Gaussian probability paths.

> Example 3 (Denoising Score Matching Loss for Gaussian Probability Paths)
>
> $$p_t(x \mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I)$$. As we derived in example 1, the conditional score function is 
>
$$\begin{equation}
\nabla \log p_t(x \mid z) = - \frac{1}{\beta_t^2}(x - \alpha_t z)
\end{equation}
$$
>
> plugging this into the denoising score matching loss, we have:
>
$$\begin{aligned}
\mathcal{L}_{\text{CSM}}(\theta) &= \mathbb{E}_{t \sim \text{Uniform}[0, 1], z \sim p_{\text{data}}, x \sim p_t(\cdot \mid z)} \left[ \| s^{\theta}_t(x) - \nabla \log p_t(x \mid z) \|_2^2 \right] \\
&= \mathbb{E}_{t \sim \text{Uniform}[0, 1], z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I)} \left[ \| s^{\theta}_t(x) + \frac{1}{\beta_t^2}(x - \alpha_t z) \|_2^2 \right] \\
&= \mathbb{E}_{t \sim \text{Uniform}[0, 1], z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I)} \left[ \| s^{\theta}_t(\alpha_t z + \beta_t \epsilon) + \frac{\epsilon}{\beta_t} \|_2^2 \right] \\
&= \mathbb{E}_{t \sim \text{Uniform}[0, 1], z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I)} \left[ \frac{1}{\beta_t^2} \| \beta_t s^{\theta}_t(\alpha_t z + \beta_t \epsilon) + \epsilon \|_2^2 \right]
\end{aligned}
$$
>
> recall the important reparameterization trick: $$x = \alpha_t z + \beta_t \epsilon$$, where $$\epsilon \sim \mathcal{N}(0, I)$$. Note that the network essentially learns to predict the noise that was used to corrupt the clean data $$z$$. We can realize that the above loss is numerically unstable when $$\beta_t \to 0$$ (i.e. denoising score matching only works if you add a sufficient amount of noise). In DDPM, it was proposed to drop the constant $$\frac{1}{\beta_t^2}$$ in the loss and reparameterize $$s_t^{\theta}$$ into a noise predictor $$\epsilon_t^{\theta}: \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d$$ via:
>
$$ \begin{equation}
\begin{aligned}
-\beta_t s_t^{\theta}(x) &= \epsilon_t^{\theta}(x) \\ \implies \mathcal{L}_{\text{DDPM}}(\theta) &= \mathbb{E}_{t \sim \text{Uniform}[0, 1], z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I)} \left[ \| \epsilon_t^{\theta}(\alpha_t z + \beta_t \epsilon) - \epsilon \|_2^2 \right]
\end{aligned}
\end{equation}
$$
> 
> As before, the network learns to predict the noise that was used to corrupt the clean data $$z$$. 

Finally, we can summarize the training procedure for Gaussian probability paths. The network is trained to predict the noise $$\epsilon$$ added to a clean sample; at inference we plug the learned score (equivalently, the denoiser or noise predictor) into the SDE of Theorem 1.

```pseudocode
\begin{algorithm}
\caption{Denoising Score Matching training (Gaussian probability path)}
\begin{algorithmic}
\REQUIRE dataset of samples $z \sim p_{\text{data}}$; noise schedule $\alpha_t, \beta_t$; noise predictor $\epsilon_t^\theta$ or score network $s_t^\theta$; 
\STATE initialize network parameters $\theta$
\REPEAT
    \STATE sample $z \sim p_{\text{data}}$, $\ t \sim \mathrm{Uniform}[0,1]$, $\ \epsilon \sim \mathcal{N}(0, I)$
    \STATE $x_t \gets \alpha_t z + \beta_t \epsilon$ \COMMENT{noisy sample from $p_t(\cdot \mid z)$}
    \STATE $\mathcal{L}(\theta) \gets \lVert \epsilon_t^\theta(x_t) - \epsilon \rVert_2^2$ or alternatively $\mathcal{L}(\theta) \gets \lVert s_t^\theta(x_t) + \frac{\epsilon}{\beta_t} \rVert_2^2$
    \STATE $\theta \gets \mathrm{GradientUpdate}\!\left(\theta, \nabla_\theta \mathcal{L}(\theta)\right)$
\UNTIL{converged}
\end{algorithmic}
\end{algorithm}
```