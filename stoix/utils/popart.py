"""PopArt (Preserving Outputs Precisely while Adaptively Rescaling Targets).

PopArt is a value normalization technique that:
1. Tracks running mean and variance of value targets using Welford's algorithm
2. Normalizes value targets for loss computation
3. Rescales critic output layer weights/biases when statistics change to preserve
   the network's learned value function

This enables stable training across different reward scales and regime changes.

Reference: van Hasselt et al. (2016) "Learning values across many orders of magnitude"
https://arxiv.org/abs/1602.07714

Usage:
    # Initialize state
    popart_state = PopArtState.create()

    # In training loop:
    # 1. Normalize targets before critic loss
    normalized_targets = normalize_target(targets, popart_state)

    # 2. Compute critic loss with normalized targets
    critic_loss = compute_loss(critic_output, normalized_targets)

    # 3. Update stats and rescale output layer
    new_popart_state = update_popart_stats(popart_state, targets)
    critic_params = rescale_output_layer(critic_params, popart_state, new_popart_state)

    # 4. Denormalize values for action selection / GAE
    real_values = denormalize_value(normalized_values, popart_state)
"""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict


@struct.dataclass
class PopArtState:
    """State for PopArt value normalization.

    Tracks running mean and variance of value targets using Welford's
    online algorithm for numerical stability.

    Attributes:
        mean: Running mean of value targets (scalar or per-output).
        var: Running variance of value targets.
        count: Number of samples seen (for Welford update).
        beta: Update rate for exponential moving average (0 = Welford, >0 = EMA blend).
    """

    mean: chex.Array
    var: chex.Array
    count: chex.Array
    beta: float = struct.field(pytree_node=False, default=0.0)

    @classmethod
    def create(cls, shape: Tuple[int, ...] = (), beta: float = 0.0) -> "PopArtState":
        """Create initial PopArt state.

        Args:
            shape: Shape of mean/var arrays. Use () for scalar, (n,) for per-output.
            beta: Update rate for EMA blending. 0 = pure Welford, >0 = EMA blend.
                  Typical values: 0.0001 to 0.001 for faster adaptation.

        Returns:
            Initialized PopArt state with zero mean, unit variance, zero count.
        """
        return cls(
            mean=jnp.zeros(shape, dtype=jnp.float32),
            var=jnp.ones(shape, dtype=jnp.float32),
            count=jnp.zeros((), dtype=jnp.float32),
            beta=beta,
        )

    @property
    def std(self) -> chex.Array:
        """Get standard deviation (with numerical stability)."""
        return jnp.sqrt(self.var + 1e-8)


def update_popart_stats(state: PopArtState, targets: chex.Array) -> PopArtState:
    """Update PopArt statistics using Welford's online algorithm.

    Uses a combination of Welford's algorithm for numerical stability
    and optional EMA blending (controlled by state.beta) for faster
    adaptation to regime changes.

    Args:
        state: Current PopArt state with mean, var, count.
        targets: New value targets to incorporate, shape (batch_size,) or (batch_size, n).

    Returns:
        Updated PopArt state with new mean, var, count.
    """
    # Flatten targets if needed (for batched updates)
    targets_flat = targets.reshape(-1)
    batch_size = targets_flat.shape[0]

    # Welford's online algorithm for batch update
    # See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    new_count = state.count + batch_size
    batch_mean = jnp.mean(targets_flat)
    batch_var = jnp.var(targets_flat)

    # Delta between batch mean and running mean
    delta = batch_mean - state.mean

    # Combined mean using parallel Welford formula
    new_mean = state.mean + delta * batch_size / jnp.maximum(new_count, 1.0)

    # Combined variance using parallel Welford formula
    # M2 = sum of squared differences from mean
    m2_old = state.var * state.count
    m2_batch = batch_var * batch_size
    m2_combined = m2_old + m2_batch + delta**2 * state.count * batch_size / jnp.maximum(new_count, 1.0)
    new_var = m2_combined / jnp.maximum(new_count, 1.0)

    # Optional EMA blending for faster adaptation
    # When beta > 0, blend Welford estimates with EMA
    def apply_ema_blending(
        new_mean: chex.Array, new_var: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        # EMA update: stat = (1 - beta) * stat + beta * batch_stat
        ema_mean = (1 - state.beta) * state.mean + state.beta * batch_mean
        ema_var = (1 - state.beta) * state.var + state.beta * (batch_var + delta**2)

        # Blend Welford and EMA based on count (more EMA early, more Welford later)
        # This provides fast initial adaptation while maintaining stability
        blend_factor = jnp.minimum(state.count / 1000.0, 1.0)  # Transition over 1000 samples
        blended_mean = blend_factor * new_mean + (1 - blend_factor) * ema_mean
        blended_var = blend_factor * new_var + (1 - blend_factor) * ema_var
        return blended_mean, blended_var

    new_mean, new_var = jax.lax.cond(
        state.beta > 0,
        lambda: apply_ema_blending(new_mean, new_var),
        lambda: (new_mean, new_var),
    )

    # Ensure variance doesn't collapse to zero
    new_var = jnp.maximum(new_var, 1e-6)

    return state.replace(mean=new_mean, var=new_var, count=new_count)


def rescale_output_layer(
    params: FrozenDict,
    old_state: PopArtState,
    new_state: PopArtState,
    layer_path: Tuple[str, ...] = ("params", "critic_head", "Dense_0"),
) -> FrozenDict:
    """Rescale critic output layer weights to preserve network outputs.

    When PopArt statistics change, the network's learned value structure
    would be lost without rescaling. This function adjusts the output
    layer weights and biases to maintain the same effective output.

    The transformation preserves: denormalize(normalized_output)
    by applying:
        W' = W * (std_old / std_new)
        b' = (b * std_old + mean_old - mean_new) / std_new
           = (b - (mean_new - mean_old) / std_old) * (std_old / std_new)

    This function is JAX JIT-compatible by using jax.tree_util operations.

    Args:
        params: Network parameters (FrozenDict or dict).
        old_state: PopArt state before update.
        new_state: PopArt state after update.
        layer_path: Path to output layer in params dict.
                   Default ("params", "critic_head", "Dense_0") works for Stoix networks.

    Returns:
        Updated parameters with rescaled output layer.
    """
    old_std = old_state.std
    new_std = new_state.std
    old_mean = old_state.mean
    new_mean = new_state.mean

    # Scale factor for weights
    scale = old_std / new_std
    bias_shift = (old_mean - new_mean) / new_std

    def rescale_fn(path, leaf):
        """Rescale function applied via tree_map_with_path."""
        # Convert path to tuple of strings for comparison
        path_strs = tuple(str(p.key) if hasattr(p, "key") else str(p) for p in path)

        # Check if this leaf is in the target output layer
        if len(path_strs) >= len(layer_path) + 1:
            prefix = path_strs[: len(layer_path)]
            param_name = path_strs[len(layer_path)]

            if prefix == layer_path:
                if param_name == "kernel":
                    # Rescale kernel: W' = W * scale
                    return leaf * scale
                if param_name == "bias":
                    # Rescale bias: b' = b * scale + (mean_old - mean_new) / std_new
                    return leaf * scale + bias_shift

        # Return unchanged for non-target params
        return leaf

    # Use tree_map_with_path to apply rescaling only to target layer
    return jax.tree_util.tree_map_with_path(rescale_fn, params)


def normalize_target(target: chex.Array, state: PopArtState) -> chex.Array:
    """Normalize value targets for loss computation.

    Args:
        target: Raw value targets, any shape.
        state: PopArt state with mean and std.

    Returns:
        Normalized targets: (target - mean) / std
    """
    return (target - state.mean) / state.std


def denormalize_value(value: chex.Array, state: PopArtState) -> chex.Array:
    """Denormalize network outputs to real value scale.

    The network outputs normalized values; this converts them back
    to the original scale for action selection and GAE computation.

    Args:
        value: Normalized network output, any shape.
        state: PopArt state with mean and std.

    Returns:
        Denormalized values: value * std + mean
    """
    return value * state.std + state.mean


def find_critic_output_layer(params: FrozenDict) -> Optional[Tuple[str, ...]]:
    """Attempt to find the critic output layer path in params.

    Searches for common patterns in network parameter structures.
    This is a heuristic for networks that don't use the default structure.

    Args:
        params: Network parameters.

    Returns:
        Tuple path to output layer, or None if not found.
    """
    # Common paths for different network architectures
    common_paths = [
        ("params", "critic_head", "Dense_0"),  # Stoix standard
        ("params", "Dense_0"),  # Simple networks
        ("params", "value_head", "Dense_0"),  # Custom with named heads
        ("params", "output_layer"),  # Alternative naming
        ("params", "head", "Dense_0"),  # Head-based architecture
    ]

    params_dict = params.unfreeze() if hasattr(params, "unfreeze") else params

    for path in common_paths:
        try:
            layer = params_dict
            for key in path:
                layer = layer[key]
            if "kernel" in layer:
                return path
        except (KeyError, TypeError):
            continue

    return None
