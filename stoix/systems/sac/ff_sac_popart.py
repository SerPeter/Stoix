"""SAC with PopArt value normalization.

PopArt (Preserving Outputs Precisely while Adaptively Rescaling Targets) enables
stable training across different reward scales by:
1. Normalizing Q-value targets for critic loss computation
2. Tracking running statistics of TD targets
3. Rescaling Q-network output layer weights to preserve learned value function

For SAC, we apply PopArt to the twin Q-networks, normalizing the TD targets.

Reference: van Hasselt et al. (2016) "Learning values across many orders of magnitude"
https://arxiv.org/abs/1602.07714
"""

import copy
import time
from typing import Any, Callable, Tuple

import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from stoa import Environment, TimeStep, WrapperState, get_final_step_metrics

from stoix.base_types import (
    ActorApply,
    AnakinExperimentOutput,
    ContinuousQApply,
    LearnerFn,
    OffPolicyLearnerState,
    OnlineAndTarget,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import CompositeNetwork
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import MultiNetwork
from stoix.systems.q_learning.dqn_types import Transition
from stoix.systems.sac.sac_types import SACOptStates, SACParams
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.popart import (
    PopArtState,
    denormalize_value,
    normalize_target,
    rescale_output_layer,
    update_popart_stats,
)
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate


def get_warmup_fn(
    env: Environment,
    params: SACParams,
    actor_apply_fn: ActorApply,
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:
    def warmup(
        env_states: WrapperState,
        timesteps: TimeStep,
        buffer_states: BufferState,
        keys: chex.PRNGKey,
    ) -> Tuple[WrapperState, TimeStep, BufferState, chex.PRNGKey]:
        def _env_step(
            carry: Tuple[WrapperState, TimeStep, chex.PRNGKey], _: Any
        ) -> Tuple[Tuple[WrapperState, TimeStep, chex.PRNGKey], Transition]:
            """Step the environment."""

            env_state, last_timestep, key = carry
            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = actor_apply_fn(params.actor_params, last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]

            transition = Transition(
                last_timestep.observation, action, timestep.reward, done, next_obs, info
            )

            return (env_state, timestep, key), transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        (env_states, timesteps, keys), traj_batch = jax.lax.scan(
            _env_step, (env_states, timesteps, keys), None, config.system.warmup_steps
        )

        # Add the trajectory to the buffer.
        buffer_states = buffer_add_fn(buffer_states, traj_batch)

        return env_states, timesteps, keys, buffer_states

    batched_warmup_step: Callable = jax.vmap(
        warmup, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0), axis_name="batch"
    )

    return batched_warmup_step


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, ContinuousQApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn, optax.TransformUpdateFn],
    buffer_fns: Tuple[Callable, Callable],
    config: DictConfig,
) -> LearnerFn[OffPolicyLearnerState]:
    """Get the learner function with PopArt value normalization."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, q_apply_fn = apply_fns
    actor_update_fn, q_update_fn, alpha_update_fn = update_fns
    buffer_add_fn, buffer_sample_fn = buffer_fns

    def _update_step(
        learner_state: OffPolicyLearnerState, _: Any
    ) -> Tuple[OffPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: OffPolicyLearnerState, _: Any
        ) -> Tuple[OffPolicyLearnerState, Transition]:
            """Step the environment."""
            params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = actor_apply_fn(params.actor_params, last_timestep.observation)

            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]

            transition = Transition(
                last_timestep.observation, action, timestep.reward, done, next_obs, info
            )

            learner_state = OffPolicyLearnerState(
                params, opt_states, buffer_state, key, env_state, timestep
            )
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        # Add the trajectory to the buffer.
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        # Get PopArt state
        popart_state = getattr(learner_state, "popart_state", None)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _alpha_loss_fn(
                log_alpha: chex.Array,
                actor_params: FrozenDict,
                transitions: Transition,
                key: chex.PRNGKey,
            ) -> jnp.ndarray:
                """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
                actor_policy = actor_apply_fn(actor_params, transitions.obs)
                action = actor_policy.sample(seed=key)
                log_prob = actor_policy.log_prob(action)
                alpha = jnp.exp(log_alpha)
                alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - config.system.target_entropy)

                loss_info = {
                    "alpha_loss": jnp.mean(alpha_loss),
                    "alpha": jnp.mean(alpha),
                }
                return jnp.mean(alpha_loss), loss_info

            def _q_loss_fn(
                q_params: FrozenDict,
                actor_params: FrozenDict,
                target_q_params: FrozenDict,
                alpha: jnp.ndarray,
                transitions: Transition,
                key: chex.PRNGKey,
                popart_state: PopArtState,
            ) -> jnp.ndarray:
                """Q-loss with PopArt normalization."""
                # Get current Q-values (normalized if PopArt enabled)
                q_old_action = q_apply_fn(q_params, transitions.obs, transitions.action)

                # Compute next state value
                next_actor_policy = actor_apply_fn(actor_params, transitions.next_obs)
                next_action = next_actor_policy.sample(seed=key)
                next_log_prob = next_actor_policy.log_prob(next_action)
                next_q = q_apply_fn(target_q_params, transitions.next_obs, next_action)

                # Denormalize next Q for target computation
                if popart_state is not None:
                    next_q_denorm = denormalize_value(next_q, popart_state)
                else:
                    next_q_denorm = next_q

                next_v = jnp.min(next_q_denorm, axis=-1) - alpha * next_log_prob

                # Compute TD target in real scale
                target_q = jax.lax.stop_gradient(
                    transitions.reward + (1.0 - transitions.done) * config.system.gamma * next_v
                )

                # Normalize target for loss computation
                if popart_state is not None:
                    target_q_norm = normalize_target(target_q, popart_state)
                else:
                    target_q_norm = target_q

                q_error = q_old_action - jnp.expand_dims(target_q_norm, -1)
                q_loss = 0.5 * jnp.mean(jnp.square(q_error))

                loss_info = {
                    "q_loss": jnp.mean(q_loss),
                    "q_error": jnp.mean(jnp.abs(q_error)),
                    "q1_pred": jnp.mean(next_q[..., 0]),
                    "q2_pred": jnp.mean(next_q[..., 1]),
                    "target_q": jnp.mean(target_q),
                }
                return q_loss, (loss_info, target_q)

            def _actor_loss_fn(
                actor_params: FrozenDict,
                q_params: FrozenDict,
                alpha: chex.Array,
                transitions: Transition,
                key: chex.PRNGKey,
                popart_state: PopArtState,
            ) -> chex.Array:
                """Actor loss with denormalized Q-values."""
                actor_policy = actor_apply_fn(actor_params, transitions.obs)
                action = actor_policy.sample(seed=key)
                log_prob = actor_policy.log_prob(action)
                q_action = q_apply_fn(q_params, transitions.obs, action)

                # Denormalize Q-values for actor loss
                if popart_state is not None:
                    q_action = denormalize_value(q_action, popart_state)

                min_q = jnp.min(q_action, axis=-1)
                actor_loss = alpha * log_prob - min_q

                loss_info = {
                    "actor_loss": jnp.mean(actor_loss),
                    "entropy": jnp.mean(-log_prob),
                }
                return jnp.mean(actor_loss), loss_info

            params, opt_states, buffer_state, key, popart_state = update_state

            key, sample_key, actor_key, q_key, alpha_key = jax.random.split(key, num=5)

            # SAMPLE TRANSITIONS
            transition_sample = buffer_sample_fn(buffer_state, sample_key)
            transitions: Transition = transition_sample.experience
            alpha = jnp.exp(params.log_alpha)

            # CALCULATE Q LOSS (and get TD targets for PopArt update)
            q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
            q_grads, (q_loss_info, target_q) = q_grad_fn(
                params.q_params.online,
                params.actor_params,
                params.q_params.target,
                alpha,
                transitions,
                q_key,
                popart_state,
            )

            # UPDATE POPART STATISTICS with TD targets
            new_popart_state = popart_state
            if popart_state is not None:
                new_popart_state = update_popart_stats(popart_state, target_q)

                # Rescale Q-network output layers
                new_online_q_params = rescale_output_layer(
                    params.q_params.online,
                    popart_state,
                    new_popart_state,
                    layer_path=("params", "networks_0", "layers_2", "Dense_0"),
                )
                new_online_q_params = rescale_output_layer(
                    new_online_q_params,
                    popart_state,
                    new_popart_state,
                    layer_path=("params", "networks_1", "layers_2", "Dense_0"),
                )

                new_target_q_params = rescale_output_layer(
                    params.q_params.target,
                    popart_state,
                    new_popart_state,
                    layer_path=("params", "networks_0", "layers_2", "Dense_0"),
                )
                new_target_q_params = rescale_output_layer(
                    new_target_q_params,
                    popart_state,
                    new_popart_state,
                    layer_path=("params", "networks_1", "layers_2", "Dense_0"),
                )

                params = SACParams(
                    params.actor_params,
                    OnlineAndTarget(new_online_q_params, new_target_q_params),
                    params.log_alpha,
                )

                # Add PopArt stats to loss info
                q_loss_info["popart_mean"] = new_popart_state.mean
                q_loss_info["popart_std"] = new_popart_state.std

            # CALCULATE ACTOR LOSS (using updated popart state)
            actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
            actor_grads, actor_loss_info = actor_grad_fn(
                params.actor_params, params.q_params.online, alpha, transitions, actor_key, new_popart_state
            )

            # Compute the parallel mean (pmean) over the batch.
            actor_grads, actor_loss_info = jax.lax.pmean(
                (actor_grads, actor_loss_info), axis_name="batch"
            )
            actor_grads, actor_loss_info = jax.lax.pmean(
                (actor_grads, actor_loss_info), axis_name="device"
            )

            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="batch")
            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")

            if config.system.autotune:
                alpha_grad_fn = jax.grad(_alpha_loss_fn, has_aux=True)
                alpha_grads, alpha_loss_info = alpha_grad_fn(
                    params.log_alpha, params.actor_params, transitions, alpha_key
                )
                alpha_grads, alpha_loss_info = jax.lax.pmean(
                    (alpha_grads, alpha_loss_info), axis_name="batch"
                )
                alpha_grads, alpha_loss_info = jax.lax.pmean(
                    (alpha_grads, alpha_loss_info), axis_name="device"
                )
                log_alpha_updates, alpha_new_opt_state = alpha_update_fn(
                    alpha_grads, opt_states.alpha_opt_state
                )
                log_alpha_new_params = optax.apply_updates(params.log_alpha, log_alpha_updates)
            else:
                log_alpha_new_params = params.log_alpha
                alpha_new_opt_state = opt_states.alpha_opt_state
                alpha_loss_info = {"alpha_loss": 0.0, "alpha": alpha}

            # UPDATE ACTOR PARAMS AND OPTIMISER STATE
            actor_updates, actor_new_opt_state = actor_update_fn(
                actor_grads, opt_states.actor_opt_state
            )
            actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

            # UPDATE Q PARAMS AND OPTIMISER STATE
            q_updates, q_new_opt_state = q_update_fn(q_grads, opt_states.q_opt_state)
            q_new_online_params = optax.apply_updates(params.q_params.online, q_updates)
            # Target network polyak update.
            new_target_q_params = optax.incremental_update(
                q_new_online_params, params.q_params.target, config.system.tau
            )
            q_new_params = OnlineAndTarget(q_new_online_params, new_target_q_params)

            # PACK NEW PARAMS AND OPTIMISER STATE
            new_params = SACParams(actor_new_params, q_new_params, log_alpha_new_params)
            new_opt_state = SACOptStates(actor_new_opt_state, q_new_opt_state, alpha_new_opt_state)

            # PACK LOSS INFO
            loss_info = {
                **actor_loss_info,
                **q_loss_info,
                **alpha_loss_info,
            }
            return (new_params, new_opt_state, buffer_state, key, new_popart_state), loss_info

        update_state = (params, opt_states, buffer_state, key, popart_state)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, buffer_state, key, popart_state = update_state

        # Reconstruct learner state with PopArt state
        if popart_state is not None:
            learner_state = OffPolicyLearnerState(
                params, opt_states, buffer_state, key, env_state, last_timestep
            )
            learner_state = learner_state._replace(popart_state=popart_state)
        else:
            learner_state = OffPolicyLearnerState(
                params, opt_states, buffer_state, key, env_state, last_timestep
            )

        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OffPolicyLearnerState,
    ) -> AnakinExperimentOutput[OffPolicyLearnerState]:
        """Learner function with PopArt normalization."""

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[OffPolicyLearnerState], Actor, OffPolicyLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of actions or action dimension from the environment.
    action_dim = int(env.action_space().shape[-1])
    config.system.action_dim = action_dim
    config.system.action_minimum = float(env.action_space().minimum)
    config.system.action_maximum = float(env.action_space().maximum)

    # PRNG keys.
    key, actor_net_key, q_net_key = keys

    # Define actor_network, q_network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        minimum=config.system.action_minimum,
        maximum=config.system.action_maximum,
    )
    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)

    def create_q_network(cfg: DictConfig) -> CompositeNetwork:
        q_network_input = hydra.utils.instantiate(cfg.network.q_network.input_layer)
        q_network_torso = hydra.utils.instantiate(cfg.network.q_network.pre_torso)
        q_network_head = hydra.utils.instantiate(cfg.network.q_network.critic_head)
        return CompositeNetwork([q_network_input, q_network_torso, q_network_head])

    double_q_network = MultiNetwork([create_q_network(config), create_q_network(config)])

    actor_lr = make_learning_rate(config.system.actor_lr, config, config.system.epochs)
    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    q_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(q_lr, eps=1e-5),
    )

    # Initialise observation
    init_x = env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
    init_a = jnp.zeros((1, action_dim))

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise q params and optimiser state.
    online_q_params = double_q_network.init(q_net_key, init_x, init_a)
    target_q_params = online_q_params
    q_opt_state = q_optim.init(online_q_params)

    # Automatic entropy tuning
    target_entropy = -config.system.target_entropy_scale * action_dim
    if config.system.autotune:
        log_alpha = jnp.zeros_like(target_entropy)
    else:
        log_alpha = jnp.log(config.system.init_alpha)

    config.system.target_entropy = target_entropy

    alpha_lr = make_learning_rate(config.system.alpha_lr, config, config.system.epochs)
    alpha_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(alpha_lr, eps=1e-5),
    )
    alpha_opt_state = alpha_optim.init(log_alpha)

    params = SACParams(actor_params, OnlineAndTarget(online_q_params, target_q_params), log_alpha)
    opt_states = SACOptStates(actor_opt_state, q_opt_state, alpha_opt_state)

    actor_network_apply_fn = actor_network.apply
    q_network_apply_fn = double_q_network.apply

    # Pack apply and update functions.
    apply_fns = (actor_network_apply_fn, q_network_apply_fn)
    update_fns = (actor_optim.update, q_optim.update, alpha_optim.update)

    # Create replay buffer
    dummy_transition = Transition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((action_dim), dtype=float),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False},
    )

    assert config.system.total_buffer_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total buffer size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    assert config.system.total_batch_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total batch size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    config.system.buffer_size = config.system.total_buffer_size // (
        n_devices * config.arch.update_batch_size
    )
    config.system.batch_size = config.system.total_batch_size // (
        n_devices * config.arch.update_batch_size
    )
    buffer_fn = fbx.make_item_buffer(
        max_length=config.system.buffer_size,
        min_length=config.system.batch_size,
        sample_batch_size=config.system.batch_size,
        add_batches=True,
        add_sequences=True,
    )
    buffer_fns = (buffer_fn.add, buffer_fn.sample)
    buffer_states = buffer_fn.init(dummy_transition)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, buffer_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    warmup = get_warmup_fn(env, params, actor_network_apply_fn, buffer_fn.add, config)
    warmup = jax.pmap(warmup, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))
    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(warmup_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    warmup_keys = reshape_keys(jnp.stack(warmup_keys))

    replicate_learner = (params, opt_states, buffer_states)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, buffer_states = replicate_learner
    # Warmup the buffer.
    env_states, timesteps, keys, buffer_states = warmup(
        env_states, timesteps, buffer_states, warmup_keys
    )
    init_learner_state = OffPolicyLearnerState(
        params, opt_states, buffer_states, step_keys, env_states, timesteps
    )

    # Initialize PopArt state if enabled
    if config.system.use_popart:
        print(
            f"{Fore.YELLOW}{Style.BRIGHT}Initializing PopArt value normalization "
            f"(beta={config.system.popart_beta})...{Style.RESET_ALL}"
        )
        popart_state = PopArtState.create(beta=config.system.popart_beta)
        # Broadcast and replicate like other state
        popart_state = jax.tree_util.tree_map(broadcast, popart_state)
        popart_state = flax.jax_utils.replicate(popart_state, devices=jax.devices())

        # Create extended learner state with PopArt
        class OffPolicyLearnerStateWithPopArt(OffPolicyLearnerState):
            popart_state: PopArtState = None

            def _replace(self, **kwargs):
                base_kwargs = {k: v for k, v in kwargs.items() if k in OffPolicyLearnerState._fields}
                popart_kwargs = {k: v for k, v in kwargs.items() if k == "popart_state"}

                if base_kwargs:
                    result = super()._replace(**base_kwargs)
                else:
                    result = self

                if popart_kwargs:
                    return OffPolicyLearnerStateWithPopArt(
                        params=result.params,
                        opt_states=result.opt_states,
                        buffer_state=result.buffer_state,
                        key=result.key,
                        env_state=result.env_state,
                        timestep=result.timestep,
                        popart_state=popart_kwargs["popart_state"],
                    )
                return result

        init_learner_state = OffPolicyLearnerStateWithPopArt(
            params=init_learner_state.params,
            opt_states=init_learner_state.opt_states,
            buffer_state=init_learner_state.buffer_state,
            key=init_learner_state.key,
            env_state=init_learner_state.env_state,
            timestep=init_learner_state.timestep,
            popart_state=popart_state,
        )

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Create the environments for train and eval.
    env, eval_env = environments.make(config=config)

    # PRNG keys.
    key, key_e, actor_net_key, q_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, q_net_key), config
    )

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_distribution_act_fn(config, actor_network.apply),
        params=learner_state.params.actor_params,
        config=config,
    )

    # Calculate number of updates per evaluation.
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = StoixLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(f"{Fore.YELLOW}{Style.BRIGHT}JAX Global Devices {jax.devices()}{Style.RESET_ALL}")
    if config.system.use_popart:
        print(f"{Fore.CYAN}{Style.BRIGHT}PopArt enabled for value normalization{Style.RESET_ALL}")

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = unreplicate_batch_dim(learner_state.params.actor_params)
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        opt_steps_per_eval = config.arch.num_updates_per_eval * (config.system.epochs)
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.params.actor_params
        )
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()
    eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    return eval_performance


@hydra.main(
    config_path="../../configs/default/anakin",
    config_name="default_ff_sac_popart.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    t0 = time.time()
    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}SAC-PopArt experiment completed in "
        f"{time.time() - t0:.2f} seconds.{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
