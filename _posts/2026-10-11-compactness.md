---
layout: post
title: "Compactness in General Topological Spaces"
date: 2026-10-31
category: "Mathematics"
tags: [topology]
author: "Haozhe Yang"
image:
license: cc-by-nc-sa
acknowledgement: "These notes draw on James R. Munkres, *Topology* (2nd ed.), and HKU MATH3401 *Analysis I*."
excerpt: "In this post, we discuss compactness, sequential compactness and limit point compactness in general topological spaces."
---

In general topological spaces, **compactness, sequential compactness and limit point compactness** are three closely related but non-equivalent concepts. In metric space, these three concepts are equivalent, but in general topological spaces, they can differ. In this post, we will discuss these three concepts and their relationships.

## 1. Basic Definitions

Let $$(X, \mathcal{T})$$ be a topological space. 

### 1.1 Compactness

The space $$X$$ is said to be **compact** if every open cover of $$X$$ 

$$X \subseteq \bigcup_{\alpha \in I}{U_{\alpha}} \quad U_{\alpha} \in \mathcal{T}$$

has a finite subcover, i.e., there exist finitely many indices $$\alpha_1, \alpha_2, \ldots, \alpha_n \in I$$ such that

$$X \subseteq \bigcup_{i=1}^{n}{U_{\alpha_i}}.$$

> Every open cover admits a finite subcover.

- The index set $$I$$ can be uncountable.
- Does not require the space to be Hausdorff or satisfy $$T_1$$ axiom.
- In general topological spaces, compactness is not characterized by "closed and bounded" (boundedness is not even defined without a metric, and a compact subset need not be closed unless the space is Hausdorff).

### 1.2 Sequential Compactness

If for any sequence $$(x_n)_{n \in \mathbb{N}}$$ in $$X$$, there exists a subsequence $$(x_{n_k})_{k \in \mathbb{N}}$$ that converges to some point $$x \in X$$, then $$X$$ is said to be **sequentially compact**.

The convergence of sequence is defined in terms of the topology:

$$x_n \to x \iff \forall U \in \mathcal{T} \text{ with } x \in U, \ \exists N \in \mathbb{N} \text{ such that } x_n \in U \ \forall n \geq N.$$

> A side note: In general topological spaces, the limit of a sequence may not be unique. The main reason is that there may not be enough open sets to separate points by disjoint neighbourhoods. Consider the trivial (indiscrete) topology on the set $$X = \{a, b\}$$, i.e. $$\mathcal{T} = \{\emptyset, X\}$$. Hence the only open neighbourhood of $$a$$ is $$X$$, and the only open neighbourhood of $$b$$ is also $$X$$. Consider the sequence $$x_n = a$$ for all $$n \in \mathbb{N}$$. Then we have $$x_n \to a$$ and $$x_n \to b$$, which means the limit of the sequence is not unique. The uniqueness of limit is guaranteed by the $$T_2$$ axiom (Hausdorff space).

### 1.3 Limit Point Compactness

Recall the definition of limit point: A point $$x \in X$$ is a **limit point** of a subset $$A \subseteq X$$ if every open neighbourhood of $$x$$ contains at least one point of $$A$$ different from $$x$$ itself. Formally,

$$\forall U \in \mathcal{T}, x \in U \implies U \cap (A \setminus \{x\}) \neq \emptyset.$$

We say $$X$$ is limit point compact if

$$\forall A \subseteq X, A \text{ is infinite } \implies A \text{ has a limit point in } X.$$

> Limit point does not have to be in the set $$A$$. e.g. Consider the set $$A = \{1/n : n \in \mathbb{N}\}$$ in $$\mathbb{R}$$ with standard topology. The point $$0$$ is a limit point of $$A$$, but $$0 \notin A$$.

## 2. Relationships in general topological spaces

Two general conclusions can be drawn from the definitions above:

$$\begin{aligned} \text{Compactness} &\implies \text{Limit Point Compactness} \\ \text{Sequential Compactness} &\implies \text{Limit Point Compactness}\end{aligned}$$

**Proposition 1.** Compactness implies limit point compactness.

**Proof.** Suppose $$X$$ is compact and assume to the contrary that there exists an infinite subset $$A \subseteq X$$ that has no limit point in $$X$$. 

Hence for any point $$x \in X$$, there exists an open neighbourhood $$U_x$$ of $$x$$ such that 

$$U_x \cap (A \setminus \{x\}) = \emptyset$$

Therefore,

$$U_x \cap A \subseteq \{x\}.$$

Since $$\{U_x: x \in X\}$$ forms an open cover of $$X$$, by compactness, there exists $$x_1, x_2, \ldots, x_n \in X$$ such that

$$X \subseteq \bigcup_{i=1}^{n}{U_{x_i}}.$$

Therefore, we have 

$$A = \bigcup_{i=1}^{n}{(U_{x_i} \cap A)} \subseteq \bigcup_{i=1}^{n}{\{x_i\}}.$$

which is a finite set, contradicting the assumption that $$A$$ is infinite. Hence $$X$$ is limit point compact.

**Proposition 2.** Sequential compactness implies limit point compactness.

**Proof.** Suppose $$X$$ is sequentially compact. Take an arbitrary infinite subset $$A \subseteq X$$. Take distinct points $$x_1, x_2, \ldots$$ in $$A$$ to form a sequence $$(x_n)_{n \in \mathbb{N}}$$. By sequential compactness, there exists a subsequence $$(x_{n_k})_{k \in \mathbb{N}}$$ that converges to some point $$x \in X$$.

Take open neighbourhood $$U$$ of $$x$$. By the definition of convergence, there exists $$K \in \mathbb{N}$$ such that

$$x_{n_k} \in U, \forall k \geq K.$$

Since $$x_{n_k}$$ are distinct points in $$A$$, at most one of them can be equal to $$x$$. Hence there exists some $$k \geq K$$ such that 

$$x_{n_k} \in U \cap (A \setminus \{x\})$$

So $$x$$ is a limit point of $$A$$. The choice of $$A$$ is arbitrary, hence $$X$$ is limit point compact.

## 3. Counterexamples for implications between Compactness and sequential compactness

**Counterexample 1. Sequentially Compact $$ \not\Rightarrow $$ Compact**

Consider the first uncountable ordinal $$\omega_1$$. Let

$$X = [0, \omega_1)$$

be equipped with the order topology. Here $$\omega_1$$ is the set of all countable ordinals. Hence every element of $$X$$ is a countable ordinal, but $$X$$ itself is uncountable. 

*Claim 1. $$X$$ is sequentially compact.*

Take any sequence $$(x_n) \subseteq X$$. Every sequence in a totally ordered set has a monotone subsequence, and since ordinals are well-ordered there is no infinite strictly decreasing sequence, so $$(x_n)$$ has a non-decreasing subsequence $$(x_{n_k})$$. Let

$$x = \sup_{k \in \mathbb{N}}{x_{n_k}}$$

The supremum of countably many countable ordinals is a countable ordinal (a countable union of countable sets is countable), hence $$x \lt \omega_1$$. Therefore, $$x \in X$$.

A non-decreasing sequence converges to its supremum in the order topology, so $$x_{n_k} \to x$$. Hence, we have found a convergent subsequence of the original sequence. Since the choice of sequence is arbitrary, $$X$$ is sequentially compact.

*Claim 2. $$X$$ is not compact.*

Consider the open cover

$$U = \{[0, \alpha) : \alpha \in X\}$$

Every set in $$U$$ is an open set in the order topology and 

$$X = \bigcup_{\alpha \in X}{[0, \alpha)}$$

However, for any finite subfamily of $$U$$

$$[0, \alpha_1), [0, \alpha_2), \ldots, [0, \alpha_n)$$

$$\bigcup_{i=1}^{n}{[0, \alpha_i)} = [0, \max_i\{\alpha_i\}) \neq X$$

Hence $$X$$ is not compact.

**Counterexample 2. Compact $$ \not\Rightarrow $$ Sequentially Compact**

Let $$I = \mathcal{P}(\mathbb{N})$$ and $$X = \{0, 1\}^I$$ be the product space of $$\{0, 1\}$$ indexed by $$I$$.

$$\{0, 1\}$$ is equipped with the discrete topology and $$X$$ is equipped with the product topology.

*Claim 1. $$X$$ is compact.*

Since finite discrete space $$\{0, 1\}$$ is compact, by Tychonoff's theorem, the product space $$X$$ is compact.

*Claim 2. $$X$$ is not sequentially compact.*

Construct a sequence $$(x_n) \subseteq X$$ and define:

$$x_n(A) = \begin{cases} 1, & n \in A \\ 0, & n \notin A \end{cases} \quad A \subseteq \mathbb{N}$$

Assume that there exists a subsequence $$(x_{n_k})$$ that converges to some $$x \in X$$. 

Product topology convergence implies convergence in each coordinate. Hence, for each $$A \subseteq \mathbb{N}$$, we have

$$x_{n_k}(A) \to x(A)$$

Take $$A = \{n_2, n_4, n_6, \ldots\}$$. Then we have

$$x_{n_k}(A) = \begin{cases} 1, & k \text{ is even} \\ 0, & k \text{ is odd} \end{cases}$$

This is an alternating sequence in the discrete space $$\{0, 1\}$$, which does not converge. Hence, the assumption that there exists a convergent subsequence is false. Therefore, X is not sequentially compact.

---

These two counterexamples show that compactness and sequential compactness are incomparable in general topological spaces. 

## 4. Three concepts in metric spaces

**Theorem.** In any metric space $$(X, d)$$, compactness, sequential compactness and limit point compactness are equivalent.