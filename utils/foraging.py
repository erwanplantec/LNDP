import jax
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple
from jaxtyping import Bool, Float, Int
import equinox as eqx


class EnvState(NamedTuple):
	obs: Float
	reward: Float
	done: Bool
	pos: Int
	goal: Int
	steps: Int


action_effects = jnp.array([1, -1, 0])

def manhattan_distance(a, b):
	return jnp.sum(jnp.absolute(a-b))


class GridChemotaxis:

	#-------------------------------------------------------------------
	
	def __init__(self, p_switch:float=0., n_types: int=2, env_size: int=5, max_steps=10, dense_reward=False) -> None:
		
		self.n_types = n_types
		self.env_size = env_size
		self.max_steps = max_steps
		self.p_switch = p_switch
		self.dense_reward = dense_reward

	#-------------------------------------------------------------------

	def step(self, state: EnvState, action: Int, key: jax.Array)->EnvState:

		goals_pos = jnp.array([0, self.env_size-1], dtype=int)
		goal_pos = goals_pos[state.goal]

		dp = action_effects[action]
		np = jnp.clip(state.pos + dp, 0, self.env_size-1)
		dist_to_goal = jnp.abs(np-goal_pos)
		close_to_goal = dist_to_goal == 0
		
		if not self.dense_reward:
			r = (close_to_goal.astype(float)*10) 
		else:
			r = - dist_to_goal
		
		done = close_to_goal | (state.steps==self.max_steps)
		kp, kres = jr.split(key)
		switch = jr.uniform(kp)<self.p_switch
		new_goal = jax.lax.cond(
			switch,
			lambda g: (g+1) % 2,
			lambda g: g,
			state.goal
		)
		new_state = jax.lax.cond(
			done,
			lambda k: self.reset(kres)._replace(reward=r, goal=new_goal) ,
			lambda K: state._replace(pos=np,
						   			 obs=self._get_obs(np, state.goal),
						   			 reward=r,
						   			 steps=state.steps+1),
			key)
		
		return new_state

	#-------------------------------------------------------------------

	def reset(self, key: jax.Array)->EnvState:
		kpos, ktpos, kt = jr.split(key, 3)
		
		pos = jnp.array(self.env_size//2, dtype=int)
		goal = self._sample_goal(kt)

		return EnvState(pos=pos, obs=self._get_obs(pos, goal), reward=jnp.zeros(()),
						done=jnp.zeros(()).astype(bool), steps=jnp.zeros((), dtype=int),
						goal=goal)

	#-------------------------------------------------------------------

	def _sample_goal(self, key):

		return jr.randint(key, (), minval=0, maxval=2)

	#-------------------------------------------------------------------

	def _get_obs(self, pos, target_pos):
		return pos

	#-------------------------------------------------------------------

class GridTask:

	#-------------------------------------------------------------------

	def __init__(self, statics, n_steps=100, **kwargs):
		
		self.statics = statics
		self.n_steps = n_steps
		self.env = GridChemotaxis(**kwargs)

	#-------------------------------------------------------------------

	def __call__(self, params, key, *args, **kwargs):

		raise NotImplementedError

	#-------------------------------------------------------------------

class GridEpisodicTask(GridTask):

	#-------------------------------------------------------------------

	def __call__(self, params, key, *args, **kwargs):

		key, kpinit, keinit = jr.split(key,3)

		pi = eqx.combine(params, self.statics)
		pi_state = pi.initialize(kpinit)

		env_state = self.env.reset(keinit)
		rews = jnp.zeros(())
		def step(c, _):
			pi_state, env_state, key = c
			key, k = jr.split(key)
			pi_state = pi_state._replace(r=env_state.reward[None])
			action, pi_state = pi(env_state.obs, pi_state, k)
			env_state = self.env.step(env_state, action, k)
			return [pi_state, env_state, key], [env_state.reward, env_state]

		[pi_state, env_state, _], [rews, env_states] = jax.lax.scan(step, [pi_state, env_state, key], None, self.n_steps) 
		return rews.sum(), dict()


class PiState(NamedTuple):
	r: jax.Array=jnp.zeros((1,))

class Pi(eqx.Module):
	"""
	"""
	#-------------------------------------------------------------------
	# Parameters:

	# Statics:
	
	#-------------------------------------------------------------------

	def __call__(self, obs, state, key, *args, **kwargs):
		p = obs
		action = jnp.argmax((p*action_effects))
		return action, state

	#-------------------------------------------------------------------

	def initialize(self, key):
		return PiState()

if __name__ == '__main__':
	env = GridChemotaxis()
	s = env.reset(jr.key(1))
	key = jr.key(2)
	ret = 0
	for i in range(100):
		key, k = jr.split(key)
		s = env.step(s, jnp.array(1,dtype=int), k)
		ret += s.reward
	print(ret)

