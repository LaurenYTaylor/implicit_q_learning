"""Implementations of algorithms for continuous control."""
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from actor import update as awr_update_actor
from common import Batch, InfoDict, Model, PRNGKey
from critic import update_q, update_v
from functools import partial


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@jax.jit
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    new_value, value_info = update_v(target_critic, value, batch, expectile)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor(key, actor, target_critic,
                                             new_value, batch, temperature)

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }


class Learner(object):
    def __init__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 seed: int = 0,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 distribution: str = "continuous",
                 policy_dim: int = None,
                 **kwargs):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """
        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]

        if actions[0][0].dtype == np.int64:
            self.activation = lambda x: int(x>=0.5)
        else:
            self.activation = lambda x: x

        if distribution == "continuous":
            actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)
        elif distribution == 'discrete':
            actor_def = policy.DiscretePolicy(hidden_dims,
                                              policy_dim,
                                              dropout_rate=dropout_rate)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        if distribution == "discrete":
            critic_def = value_net.DiscDoubleCritic(hidden_dims, policy_dim)
        else:
            critic_def = value_net.DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = value_net.ValueCritic(hidden_dims)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng
        if not isinstance(policy, policy.DiscretePolicy):
            actions = jnp.nan_to_num(actions)
            actions = np.asarray(actions)
            actions = np.clip(self.activation(actions), -1, 1)

        return actions

    def sample_deterministic_actions(self,
                       observations: np.ndarray) -> jnp.ndarray:

        if isinstance(policy, policy.DiscretePolicy):
            dist = self.actor.apply_fn.apply({'params': self.actor.params}, observations)
            possible_actions = list(range(self.actor.apply_fn.action_dim))
            lps = dist.log_prob(possible_actions)
            actions = jnp.argmax(lps)
            rng, key = jax.random.split(self.rng)
            return rng, actions
        else:
            rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                                 self.actor.params, observations,
                                                 temperature=0)
            actions = jnp.nan_to_num(actions)
            actions = np.asarray(actions)
            actions = np.clip(self.activation(actions), -1, 1)

        self.rng = rng
        return actions

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng, self.actor, self.critic, self.value, self.target_critic,
            batch, self.discount, self.tau, self.expectile, self.temperature)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info
