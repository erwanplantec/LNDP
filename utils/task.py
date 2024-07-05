from ast import Call
from typing import Callable, NamedTuple, Optional, Tuple, TypeAlias, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import gymnax

from brax import envs
from brax.envs import Env
from jaxtyping import Float, PyTree

Params: TypeAlias = PyTree
TaskParams: TypeAlias = PyTree
EnvState: TypeAlias = PyTree
Action: TypeAlias = jax.Array
PolicyState: TypeAlias = PyTree
BraxEnv: TypeAlias = Env
GymEnv: TypeAlias = gymnax.environments.environment.Environment

class State(NamedTuple):
	env_state: EnvState
	policy_state: PolicyState


#=======================================================================
#=======================================================================
#=======================================================================

class BraxTask(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	env: BraxEnv
	statics: PyTree[...]
	max_steps: int
	data_fn: Callable[[PyTree], dict]
	#-------------------------------------------------------------------

	def __init__(
		self, 
		statics: PyTree[...],
		env: Union[str, BraxEnv],
		max_steps: int,
		backend: str="positional",
		data_fn: Callable=lambda x: x, 
		env_kwargs: dict={}):
		
		if isinstance(env, str):
			self.env = envs.get_environment(env, backend=backend, **env_kwargs)
		else:
			self.env = env

		self.statics = statics
		self.max_steps = max_steps
		self.data_fn = data_fn

	#-------------------------------------------------------------------

	def __call__(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None)->Tuple[Float, PyTree]:

		_, _, data = self.rollout(params, key)
		return jnp.sum(data["reward"]), data

	#-------------------------------------------------------------------

	def rollout(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None)->Tuple[State, State, dict]:
		
		key, init_env_key, init_policy_key, rollout_key = jr.split(key, 4)
		policy = eqx.combine(params, self.statics)
		
		policy_state = policy.initialize(init_policy_key)
		env_state = self.initialize(init_env_key)
		init_state = State(env_state=env_state, policy_state=policy_state)

		def env_step(carry, x):
			state, key = carry
			key, _key = jr.split(key)
			action, policy_state = policy(state.env_state.obs, state.policy_state, _key)
			env_state = self.env.step(state.env_state, action)
			new_state = State(env_state=env_state, policy_state=policy_state)
			
			return [new_state, key], state

		[state, _], states = jax.lax.scan(env_step, [init_state, rollout_key], None, self.max_steps)
		data = {"policy_states": states.policy_state, "obs": states.env_state.obs}
		data = self.data_fn(data)
		data["reward"] = states.env_state.reward
		return state, states, data

	#-------------------------------------------------------------------

	def step(self, *args, **kwargs):
		return self.env.step(*args, **kwargs)

	def reset(self, *args, **kwargs):
		return self.env.reset(*args, **kwargs)

	#-------------------------------------------------------------------

	def initialize(self, key:jax.Array)->EnvState:
		
		return self.env.reset(key)

	#-------------------------------------------------------------------


#=======================================================================
#=======================================================================
#=======================================================================


class GymnaxTask(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	statics: PyTree
	env: GymEnv
	env_params: PyTree
	data_fn: Callable
	#-------------------------------------------------------------------

	def __init__(
		self, 
		statics: PyTree,
		env: str,
		env_params: Optional[PyTree] = None,
		data_fn: Callable=lambda d: d):

		self.statics = statics
		self.env, default_env_params = gymnax.make(env) #type: ignore
		self.env_params = env_params if env_params is not None else default_env_params 
		self.data_fn = data_fn

	#-------------------------------------------------------------------

	def __call__(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None)->Tuple[Float, PyTree]:

		return self.rollout(params, key, task_params)

	#-------------------------------------------------------------------

	def rollout(self, params: Params, key: jax.Array, task_params: Optional[TaskParams]=None)->Tuple[Float, PyTree]:
		"""
		code adapted from: https://github.com/RobertTLange/gymnax/blob/main/gymnax/experimental/rollout.py
		"""
		
		model = eqx.combine(params, self.statics)
		key_reset, key_episode, key_model = jr.split(key, 3)
		obs, state = self.env.reset(key_reset, self.env_params)

		def policy_step(state_input, tmp):
			"""lax.scan compatible step transition in jax env."""
			policy_state, obs, state, rng, cum_reward, valid_mask = state_input
			rng, rng_step, rng_net = jax.random.split(rng, 3)
			
			action, policy_state = model(obs, policy_state, rng_net)
			next_obs, next_state, reward, done, _ = self.env.step(
				rng_step, state, action, self.env_params
			)
			new_cum_reward = cum_reward + reward * valid_mask
			new_valid_mask = valid_mask * (1 - done)
			carry = [
				policy_state,
				next_obs,
				next_state,
				rng,
				new_cum_reward,
				new_valid_mask,
			]
			y = [policy_state, obs, action, reward, next_obs, done]
			return carry, y
			
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
				jnp.array([1.0]),
			],
			(),
			self.env.default_params.max_steps_in_episode,
		)
		# Return the sum of rewards accumulated by agent in episode rollout
		policy_state, obs, action, reward, _, _ = scan_out
		cum_return = carry_out[-2][0]
		data = {"policy_states": policy_state, "obs": obs, 
				"action": action, "rewards": reward}
		data = self.data_fn(data)
		return cum_return, data

	#-------------------------------------------------------------------


#=======================================================================
#=======================================================================
#=======================================================================

class RandomDiscretePolicy(eqx.Module):
	"""
	"""
	#-------------------------------------------------------------------
	n_actions: int
	#-------------------------------------------------------------------
	def __init__(self, n_actions: int):
		self.n_actions = n_actions
	#-------------------------------------------------------------------
	def __call__(self, env_state: EnvState, policy_state: PolicyState, key: jax.Array):
		return jr.randint(key, (), 0, self.n_actions), None
	#-------------------------------------------------------------------
	def initialize(self, *args, **kwargs):
		return None

class RandomContinuousPolicy(eqx.Module):
	"""
	"""
	#-------------------------------------------------------------------
	action_dims: int
	#-------------------------------------------------------------------
	def __init__(self, action_dims: int):
		self.action_dims = action_dims
	#-------------------------------------------------------------------
	def __call__(self, env_state: EnvState, policy_state: PolicyState, key: jax.Array):
		return jr.normal(key, (self.action_dims,)), None
	#-------------------------------------------------------------------
	def initialize(self, *args, **kwargs):
		return None

class StatefulPolicyWrapper(eqx.Module):
	"""
	Wrapper adding a policy state to the signature call of a stateless policy
	"""
	#-------------------------------------------------------------------
	policy: Union[PyTree[...], Callable[[EnvState, jax.Array], Action]]
	#-------------------------------------------------------------------
	def __init__(self, policy: Union[PyTree[...], Callable[[EnvState, jax.Array], Action]]):
		self.policy = policy
	#-------------------------------------------------------------------
	def __call__(self, env_state, policy_state, key):
		action = self.policy(env_state, key)
		return action, None
	#-------------------------------------------------------------------
	def initialize(self, *args, **kwargs):
		return None


ENV_SPACES = {
	"CartPole-v1": (4, 2, "discrete"),
	"Acrobot-v1": (6, 3, "discrete"),
	"MountainCar-v0": (2, 3, "discrete"),
	"halfcheetah": (17, 6, "continuous"),
	"ant": (27, 8, "continuous"),
	"walker2d": (17, 6, "continuous"),
	"inverted_pendulum": (4, 1, "continuous"),
	'inverted_double_pendulum': (8, 1, "continuous"),
	"hopper": (11, 3, "continuous"),
	"Pendulum-v1": (3, 1, "continuous"),
	"PointRobot-misc": (6, 2, "continuous"),
	"MetaMaze-misc": (15, 4, "discrete"),
	"Reacher-misc": (8, 2, "continuous")
}


