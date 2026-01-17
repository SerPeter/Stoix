from typing import Optional, Sequence, Tuple, Union

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from flax import linen as nn
from flax.linen.initializers import Initializer, lecun_normal, orthogonal
from tensorflow_probability.substrates.jax.distributions import (
    Categorical,
    Deterministic,
    Independent,
    MultivariateNormalDiag,
    Normal,
    TransformedDistribution,
)

from stoix.networks.distributions import (
    AffineTanhTransformedDistribution,
    ClippedBeta,
    DiscreteValuedTfpDistribution,
    MultiDiscreteActionDistribution,
)

tfb = tfp.bijectors


class CategoricalHead(nn.Module):
    action_dim: Union[int, Sequence[int]]
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> Categorical:
        logits = nn.Dense(np.prod(self.action_dim), kernel_init=self.kernel_init)(embedding)

        if not isinstance(self.action_dim, int):
            logits = logits.reshape(self.action_dim)

        return Categorical(logits=logits)


class NormalAffineTanhDistributionHead(nn.Module):

    action_dim: int
    minimum: float
    maximum: float
    min_scale: float = 1e-3
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> Independent:

        loc = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)
        scale = (
            jax.nn.softplus(nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding))
            + self.min_scale
        )
        distribution = Normal(loc=loc, scale=scale)

        return Independent(
            AffineTanhTransformedDistribution(distribution, self.minimum, self.maximum),
            reinterpreted_batch_ndims=1,
        )


class BetaDistributionHead(nn.Module):

    action_dim: int
    minimum: float
    maximum: float
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> Independent:

        # Use alpha and beta >= 1 according to [Chou et. al, 2017]
        alpha = (
            jax.nn.softplus(nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)) + 1
        )
        beta = (
            jax.nn.softplus(nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)) + 1
        )
        # Calculate scale and shift for the affine transformation to achieve the range
        # [minimum, maximum].
        scale = self.maximum - self.minimum
        shift = self.minimum
        affine_bijector = tfb.Chain([tfb.Shift(shift), tfb.Scale(scale)])

        transformed_distribution = TransformedDistribution(
            ClippedBeta(alpha, beta), bijector=affine_bijector
        )

        return Independent(
            transformed_distribution,
            reinterpreted_batch_ndims=1,
        )


class MultivariateNormalDiagHead(nn.Module):

    action_dim: int
    init_scale: float = 0.3
    min_scale: float = 1e-3
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> distrax.DistributionLike:
        loc = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)
        scale = jax.nn.softplus(nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding))
        scale *= self.init_scale / jax.nn.softplus(0.0)
        scale += self.min_scale
        return MultivariateNormalDiag(loc=loc, scale_diag=scale)


class DeterministicHead(nn.Module):
    action_dim: int
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:

        x = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)

        return Deterministic(x)


class ScalarCriticHead(nn.Module):
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:
        return nn.Dense(1, kernel_init=self.kernel_init)(embedding).squeeze(axis=-1)


class CategoricalCriticHead(nn.Module):

    num_atoms: int = 601
    vmax: Optional[float] = None
    vmin: Optional[float] = None
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> distrax.DistributionLike:
        vmax = self.vmax if self.vmax is not None else 0.5 * (self.num_atoms - 1)
        vmin = self.vmin if self.vmin is not None else -1.0 * vmax

        output = DiscreteValuedTfpHead(
            vmin=vmin,
            vmax=vmax,
            logits_shape=(),
            num_atoms=self.num_atoms,
            kernel_init=self.kernel_init,
        )(embedding)

        return output


class DiscreteValuedTfpHead(nn.Module):
    """Represents a parameterized discrete valued distribution.

    The returned distribution is essentially a `tfd.Categorical` that knows its
    support and thus can compute the mean value.
    If vmin and vmax have shape S, this will store the category values as a
    Tensor of shape (S*, num_atoms).

    Args:
        vmin: Minimum of the value range
        vmax: Maximum of the value range
        num_atoms: The atom values associated with each bin.
        logits_shape: The shape of the logits, excluding batch and num_atoms
        dimensions.
        kernel_init: The initializer for the dense layer.
    """

    vmin: float
    vmax: float
    num_atoms: int
    logits_shape: Optional[Sequence[int]] = None
    kernel_init: Initializer = lecun_normal()

    def setup(self) -> None:
        self._values = np.linspace(self.vmin, self.vmax, num=self.num_atoms, axis=-1)
        if not self.logits_shape:
            logits_shape: Sequence[int] = ()
        else:
            logits_shape = self.logits_shape
        self._logits_shape = (
            *logits_shape,
            self.num_atoms,
        )
        self._logits_size = np.prod(self._logits_shape)
        self._net = nn.Dense(self._logits_size, kernel_init=self.kernel_init)

    def __call__(self, inputs: chex.Array) -> distrax.DistributionLike:
        logits = self._net(inputs)
        logits = logits.reshape(logits.shape[:-1] + self._logits_shape)
        return DiscreteValuedTfpDistribution(values=self._values, logits=logits)


class DiscreteQNetworkHead(nn.Module):
    action_dim: int
    epsilon: float = 0.1
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(
        self, embedding: chex.Array, epsilon: Optional[float] = None
    ) -> distrax.EpsilonGreedy:

        q_values = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)

        if epsilon is None:
            epsilon = self.epsilon

        return distrax.EpsilonGreedy(preferences=q_values, epsilon=epsilon)


class PolicyValueHead(nn.Module):
    action_head: nn.Module
    critic_head: nn.Module

    @nn.compact
    def __call__(
        self, embedding: chex.Array
    ) -> Tuple[distrax.DistributionLike, Union[chex.Array, distrax.DistributionLike]]:

        action_distribution = self.action_head(embedding)
        value = self.critic_head(embedding)

        return action_distribution, value


class DistributionalDiscreteQNetwork(nn.Module):
    action_dim: int
    epsilon: float
    num_atoms: int
    vmin: float
    vmax: float
    kernel_init: Initializer = lecun_normal()

    @nn.compact
    def __call__(
        self, embedding: chex.Array
    ) -> Tuple[distrax.EpsilonGreedy, chex.Array, chex.Array]:
        atoms = jnp.linspace(self.vmin, self.vmax, self.num_atoms)
        q_logits = nn.Dense(self.action_dim * self.num_atoms, kernel_init=self.kernel_init)(
            embedding
        )
        q_logits = jnp.reshape(q_logits, (-1, self.action_dim, self.num_atoms))
        q_dist = jax.nn.softmax(q_logits)
        q_values = jnp.sum(q_dist * atoms, axis=2)
        q_values = jax.lax.stop_gradient(q_values)
        atoms = jnp.broadcast_to(atoms, (q_values.shape[0], self.num_atoms))
        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon), q_logits, atoms


class DistributionalContinuousQNetwork(nn.Module):
    num_atoms: int
    vmin: float
    vmax: float
    kernel_init: Initializer = lecun_normal()

    @nn.compact
    def __call__(
        self, embedding: chex.Array
    ) -> Tuple[distrax.EpsilonGreedy, chex.Array, chex.Array]:
        atoms = jnp.linspace(self.vmin, self.vmax, self.num_atoms)
        q_logits = nn.Dense(self.num_atoms, kernel_init=self.kernel_init)(embedding)
        q_dist = jax.nn.softmax(q_logits)
        q_value = jnp.sum(q_dist * atoms, axis=-1)
        atoms = jnp.broadcast_to(atoms, (*q_value.shape, self.num_atoms))
        return q_value, q_logits, atoms


class QuantileDiscreteQNetwork(nn.Module):
    action_dim: int
    epsilon: float
    num_quantiles: int
    kernel_init: Initializer = lecun_normal()

    @nn.compact
    def __call__(self, embedding: chex.Array) -> Tuple[distrax.EpsilonGreedy, chex.Array]:
        q_logits = nn.Dense(self.action_dim * self.num_quantiles, kernel_init=self.kernel_init)(
            embedding
        )
        q_dist = jnp.reshape(q_logits, (-1, self.action_dim, self.num_quantiles))
        q_values = jnp.mean(q_dist, axis=-1)
        q_values = jax.lax.stop_gradient(q_values)
        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon), q_dist


class ImplicitQuantileDiscreteQNetwork(nn.Module):
    """Implicit Quantile Network (IQN) head for discrete actions.

    Unlike QR-DQN which uses fixed quantiles, IQN samples tau values at runtime
    and embeds them using cosine features. This allows learning an implicit
    representation of the full return distribution.

    Reference: Dabney et al. (2018) "Implicit Quantile Networks for Distributional RL"
    https://arxiv.org/abs/1806.06923
    """

    action_dim: int
    epsilon: float
    num_tau_samples: int = 64
    num_cosines: int = 64
    embedding_dim: int = 64
    kernel_init: Initializer = lecun_normal()

    @nn.compact
    def __call__(
        self, embedding: chex.Array, tau: Optional[chex.Array] = None, key: Optional[chex.PRNGKey] = None
    ) -> Tuple[distrax.EpsilonGreedy, chex.Array, chex.Array]:
        """Forward pass.

        Args:
            embedding: (batch, embed_dim) observation embedding from torso
            tau: (batch, num_tau) optional pre-sampled tau values in [0, 1]
            key: PRNG key for sampling tau if not provided

        Returns:
            policy: EpsilonGreedy distribution over actions
            q_quantiles: (batch, num_tau, action_dim) quantile Q-values
            tau: (batch, num_tau) the tau values used
        """
        batch_size = embedding.shape[0]

        # Sample tau if not provided
        if tau is None:
            if key is None:
                raise ValueError("Either tau or key must be provided")
            tau = jax.random.uniform(key, (batch_size, self.num_tau_samples))

        num_tau = tau.shape[1]

        # Cosine embedding of tau: cos(pi * i * tau) for i in [0, num_cosines)
        # tau: (batch, num_tau) -> (batch, num_tau, 1)
        tau_expanded = tau[:, :, None]
        # i_pi: (1, 1, num_cosines)
        i_pi = jnp.pi * jnp.arange(self.num_cosines)[None, None, :]
        # cosine features: (batch, num_tau, num_cosines)
        cosine_features = jnp.cos(tau_expanded * i_pi)

        # Embed cosine features: (batch, num_tau, embedding_dim)
        tau_embedding = nn.Dense(self.embedding_dim, kernel_init=self.kernel_init)(cosine_features)
        tau_embedding = jax.nn.relu(tau_embedding)

        # Combine observation embedding with tau embedding
        # embedding: (batch, embed_dim) -> (batch, 1, embed_dim) -> (batch, num_tau, embed_dim)
        obs_embedding = jnp.expand_dims(embedding, axis=1)
        obs_embedding = jnp.broadcast_to(obs_embedding, (batch_size, num_tau, embedding.shape[-1]))

        # Element-wise product (Hadamard product as in IQN paper)
        # Note: We project obs_embedding to match tau_embedding dim first
        obs_projected = nn.Dense(self.embedding_dim, kernel_init=self.kernel_init)(obs_embedding)
        combined = obs_projected * tau_embedding

        # Output Q-values for each action at each quantile
        # combined: (batch, num_tau, embedding_dim) -> (batch, num_tau, action_dim)
        q_quantiles = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(combined)

        # Mean Q-values across quantiles for action selection
        q_values = jnp.mean(q_quantiles, axis=1)  # (batch, action_dim)
        q_values = jax.lax.stop_gradient(q_values)

        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon), q_quantiles, tau


class LinearHead(nn.Module):
    output_dim: int
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:

        return nn.Dense(self.output_dim, kernel_init=self.kernel_init)(embedding)


class QuantileContinuousQNetwork(nn.Module):
    """Distributional Q-network head for continuous actions using quantile regression.

    Instead of outputting a single Q-value, outputs N quantile values
    representing the return distribution Z(s, a). This enables risk-sensitive
    policies via CVaR, VaR, etc.

    Used in DSAC (Distributional Soft Actor-Critic).

    Reference: Ma et al. (2020) "DSAC: Distributional Soft Actor Critic"
    https://arxiv.org/abs/2004.14547
    """

    num_quantiles: int = 32
    kernel_init: Initializer = lecun_normal()

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:
        """Forward pass.

        Args:
            embedding: (batch, embed_dim) observation-action embedding from torso

        Returns:
            quantiles: (batch, num_quantiles) return distribution quantiles
        """
        return nn.Dense(self.num_quantiles, kernel_init=self.kernel_init)(embedding)


class MultiDiscreteHead(nn.Module):
    """A head for multi-discrete action spaces, where each
        discrete subspace can have a different number of dimensions.

    Arguments:
        action_dim: Total number of actions across all discrete subspaces.
        number_of_dims_per_distribution: A list where each element
            represents the number of dimensions for each discrete subspace.
        kernel_init: Initializer for the dense layer.
    """

    action_dim: int
    number_of_dims_per_distribution: list[int]
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> MultiDiscreteActionDistribution:
        assert sum(self.number_of_dims_per_distribution) == self.action_dim, (
            f"Sum of number_of_dims_per_distribution {sum(self.number_of_dims_per_distribution)} "
            f"must equal action_dim {self.action_dim}."
        )
        logits = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)

        return MultiDiscreteActionDistribution(
            flat_logits=logits, number_of_dims_per_distribution=self.number_of_dims_per_distribution
        )
