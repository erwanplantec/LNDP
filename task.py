import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jaxtyping import PyTree
from typing import Callable, Optional
from utils.task import *
from utils.foraging import GridEpisodicTask

class MultiTask:

	def __init__(self, tasks):
		self.tasks = tasks

	def __call__(self, params, key, data=None):
		f_total = 0
		data = {}
		for i, task in enumerate(self.tasks):
			f, d = task(params, key, data)
			f_total += f
			data[f"tsk_{i}"] = f
			for k, v in d.items():
				data[f"tsk_{i}_{k}"] = v

		return f_total, data

class MultiEpisodeGymnaxTask(GymnaxTask):
	"""
	"""
	#-------------------------------------------------------------------
	n_episodes: int
	fitness_transform: Optional[Callable]
	l1_penalty: float
	episode_intervention: Callable
	dev_after_episode: bool
	#-------------------------------------------------------------------

	def __init__(
		self, 
		statics: PyTree,
		env: str,
		n_episodes: int=8,
		env_params: Optional[PyTree] = None,
		fitness_transform: Optional[Callable]=None,
		data_fn: Callable=lambda d: d,
		l1_penalty: float=0.,
		episode_intervention: Callable=lambda pi, e, k: pi,
		dev_after_episode: bool=False):
		
		super().__init__(statics, env, env_params, data_fn)

		self.n_episodes = n_episodes
		self.fitness_transform = fitness_transform
		self.l1_penalty = l1_penalty
		self.episode_intervention = episode_intervention
		self.dev_after_episode = dev_after_episode

	#-------------------------------------------------------------------

	def __call__(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None):

		policy_state=None
		full_return = 0.
		returns = []
		densities = []
		for episode in range(self.n_episodes):
			key, key_, key__ = jr.split(key, 3)
			policy_state, episode_return, data = self._rollout(params, key_, policy_state)
			d = policy_state.G.A.sum() / (policy_state.G.A.shape[0]**2)
			l1 = d * self.l1_penalty
			densities.append(d)
			full_return = (full_return + (episode_return-l1) / self.n_episodes)
			returns.append(episode_return)
			policy_state = self.episode_intervention(policy_state, episode, key__)

		if self.fitness_transform is not None:
			fitness = self.fitness_transform(full_return, data) #type:ignore
		else:
			fitness = full_return
		data = {f"ep_{i}":r for i, r in enumerate(returns)}
		data["density"] = sum(densities) / self.n_episodes
		return fitness, data
	#-------------------------------------------------------------------

	def _rollout(self, params: Params, key: jax.Array, policy_state: Optional[PolicyState]=None)->Tuple[PolicyState,Float,PyTree]:
		"""
		code adapted from: https://github.com/RobertTLange/gymnax/blob/main/gymnax/experimental/rollout.py
		"""
		
		model = eqx.combine(params, self.statics)
		key_reset, key_episode, key_model, key_dev = jr.split(key, 4)
		obs, state = self.env.reset(key_reset, self.env_params)

		def policy_step(state_input, tmp):
			"""lax.scan compatible step transition in jax env."""
			policy_state, obs, state, rng, last_reward, cum_reward, valid_mask = state_input
			rng, rng_step, rng_net = jax.random.split(rng, 3)
			action, policy_state = model(obs, policy_state._replace(r=last_reward), rng_net)
			next_obs, next_state, reward, done, _ = self.env.step(
				rng_step, state, action, self.env_params
			)
			new_cum_reward = cum_reward + reward * valid_mask
			new_valid_mask = valid_mask * (1 - done)
			carry = [
				policy_state._replace(d=1-valid_mask),
				next_obs,
				next_state,
				rng,
				reward*valid_mask,
				new_cum_reward,
				new_valid_mask,
			]
			y = [policy_state, obs, action, reward*valid_mask, next_obs, done]
			return carry, y
		
		if policy_state is None:
			policy_state = model.initialize(key_model)
		# Scan over episode step loop
		carry_out, scan_out = jax.lax.scan(
			policy_step,
			[
				policy_state,
				obs,
				state,
				key_episode,
				jnp.array([0.0]),
				jnp.array([0.0]),
				jnp.array([1.0]),
			],
			(),
			self.env.default_params.max_steps_in_episode,
		)
		# Return the sum of rewards accumulated by agent in episode rollout
		policy_states, obs, action, reward, _, _ = scan_out
		policy_state, *_ = carry_out
		cum_return = carry_out[-2][0]
		data = {"policy_states": policy_states, "obs": obs, 
				"action": action, "rewards": reward}
		data = self.data_fn(data)

		if self.dev_after_episode:
			policy_state = model.dev(policy_state._replace(r=jnp.zeros((1,))), key_dev)#type:ignore

		return policy_state, cum_return, data

class MultiepisodeBraxTask(BraxTask):
	
	"""
	"""
	#-------------------------------------------------------------------
	n_episodes: int
	fitness_transform: Optional[Callable]
	#-------------------------------------------------------------------

	def __init__(
		self, 
		statics: PyTree,
		env: str,
		n_episodes: int=8,
		backend: str="positional",
		fitness_transform: Optional[Callable]=None,
		data_fn: Callable=lambda d: d):
		
		super().__init__(statics, env, 500, backend, data_fn)

		self.n_episodes = n_episodes
		self.fitness_transform = fitness_transform

	#-------------------------------------------------------------------

	def __call__(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None):

		policy_state=None
		full_return = 0.
		returns = []
		
		for _ in range(self.n_episodes):
			key, key_ = jr.split(key)
			policy_state, episode_return, data = self._rollout(params, key_, policy_state)
			full_return = full_return + episode_return / self.n_episodes
			returns.append(episode_return)

		if self.fitness_transform is not None:
			fitness = self.fitness_transform(full_return, data) #type:ignore
		else:
			fitness = full_return
		return fitness, {f"ep_{i}":r for i, r in enumerate(returns)}

	#-------------------------------------------------------------------

	def _rollout(self, params: Params, key: jax.Array, policy_state: Optional[PolicyState]=None):

		policy = eqx.combine(params, self.statics)
		key, init_env_key, init_policy_key, rollout_key = jr.split(key, 4)

		policy_state = policy.initialize(init_policy_key) if policy_state is None else policy_state
		env_state = self.initialize(init_env_key)
		init_state = State(env_state=env_state, policy_state=policy_state)

		def env_step(carry, x):
			state, key = carry
			key, _key = jr.split(key)
			action, policy_state = policy(state.env_state.obs, state.policy_state._replace(r=state.env_state.reward[None], d=state.env_state.done[None]), _key)
			env_state = self.env.step(state.env_state, action)
			new_state = State(env_state=env_state, policy_state=policy_state)
			
			return [new_state, key], state

		[state, _], states = jax.lax.scan(env_step, [init_state, rollout_key], None, self.max_steps)
		data = {"policy_states": states.policy_state, "obs": states.env_state.obs}
		data = self.data_fn(data)
		data["reward"] = states.env_state.reward
		return state.policy_state, states.env_state.reward.sum(), data


def make(config, statics):
	data_fn = lambda d: d
	if config.env_name=="Grid":
		return GridEpisodicTask(statics, p_switch=config.p_switch, env_size=config.env_size, dense_reward=bool(config.dense_reward))
	elif config.env_name[0].isupper():
		return MultiEpisodeGymnaxTask(statics, n_episodes=config.n_episodes, env=config.env_name, 
									  data_fn=data_fn, l1_penalty=config.l1_penalty, 
									  dev_after_episode=bool(config.dev_after_episode))
	else:
		return MultiepisodeBraxTask(statics, env=config.env_name, n_episodes=config.n_episodes)
