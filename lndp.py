from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple, TypeAlias
from jaxtyping import Bool, Float, Array, PyTree
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
from numpy import s_

from utils.model import *
#from src.gnn.generators import meta_reservoir, reservoir
#from src.tasks.rl import ENV_SPACES

double_vmap = lambda f: jax.vmap(jax.vmap(f))
normalize = lambda x: x/(jnp.linalg.norm(x, axis=-1, keepdims=True)+1e-8)

class PolicyState(NamedTuple):
	G: Graph
	a: jax.Array
	w: jax.Array
	r: Float
	d: Float
	t: Float
	a_seq: jax.Array


#================================================================================================================================================
#================================================================================================================================================
#================================================================================================================================================


class LNDP(eqx.Module):

	#-------------------------------------------------------------------
	# Params :
	gnn: Callable
	Wpre: Callable
	node_fn: Callable
	edge_fn: Callable
	# Statics:
	action_dims: int
	obs_dims: int
	n_nodes: int
	dev_steps: int
	rnn_iters: int
	node_features: int
	edge_features: int
	p_hh: float
	s_hh: float
	p_ih: float
	s_ih: float
	p_ho: float
	s_ho: float
	mask_A: Callable
	discrete_action: bool
	env_is_pendulum: bool
	use_bias: bool
	is_recurrent: bool
	gnn_iters: int
	stochastic_decisions: bool
	pruning: bool
	synaptogenesis: bool
	ablate_gt: bool=False
	block_lt_updates: bool=False
	# Optional Params
	prune_fn: Optional[Callable]=None
	adde_fn: Optional[Callable]=None
	sa_fn: Optional[OrnsteinUhlenbeckProcess]=None
	#-------------------------------------------------------------------

	def __init__(self, 
				 action_dims: int, 
				 obs_dims: int, 
				 n_nodes: int,
				 edge_features: int=4,
				 dev_steps: int=0,
				 rnn_iters: int=5,
				 node_features: int=8,
				 p_hh: float=.2,
				 s_hh: float=.05,
				 p_ih: float=.2,
				 s_ih: float=.05,
				 p_ho: float=.2,
				 s_ho: float=.05,
				 discrete_action: bool=True,
				 env_is_pendulum: bool=False,
				 use_bias: bool=False,
				 is_recurrent: bool=False,
				 gnn_iters: int=1,
				 stochastic_decisions: bool=False,
				 pruning: bool=True,
				 synaptogenesis: bool=True,
				 ablate_gt: bool=False,
				 block_lt_updates: bool=False,
				 *, 
				 key: jax.Array):

		"""
		Nodes are also GRUs
		"""
		# ---

		self.action_dims=action_dims
		self.obs_dims = obs_dims
		self.n_nodes = n_nodes
		self.dev_steps = dev_steps
		self.rnn_iters = rnn_iters
		self.node_features = node_features
		self.edge_features = edge_features
		self.p_hh = p_hh
		self.s_hh = s_hh
		self.p_ih = p_ih
		self.s_ih = s_ih
		self.p_ho = p_ho
		self.s_ho = s_ho
		self.discrete_action = discrete_action
		self.env_is_pendulum = env_is_pendulum
		self.use_bias = use_bias
		self.is_recurrent = is_recurrent
		self.gnn_iters = gnn_iters
		self.stochastic_decisions = stochastic_decisions
		self.pruning = pruning
		self.synaptogenesis = synaptogenesis
		self.ablate_gt = ablate_gt
		self.block_lt_updates = block_lt_updates

		# ---

		key, key_gnn, key_Wpre = jr.split(key,3)
		self.gnn = GraphTransformer(in_features=node_features,
									out_features=node_features,
									qk_features=4,
									value_features=8,
									n_heads=3,
									use_edge_features=True,
									in_edge_features=edge_features+3,
									key=key_gnn)
		self.Wpre = nn.Linear(5+rnn_iters+node_features+1, node_features,  key=key_Wpre)

		# ---

		key, key_node = jr.split(key)
		self.node_fn = nn.GRUCell(node_features, node_features, key=key_node)

		# ---

		key, key_edge = jr.split(key)
		self.edge_fn = nn.GRUCell(2*node_features + 2*(rnn_iters+1) + 1,
								  edge_features,
								  key=key_edge)

		# ---

		if synaptogenesis:
			key, key_adde = jr.split(key)
			self.adde_fn = nn.MLP(2*node_features, 1, 16, 1, key=key_adde, final_activation=jnn.sigmoid)
		else: 
			self.adde_fn = None

		# ---

		if pruning:
			key, key_prune = jr.split(key)
			self.prune_fn = nn.MLP(edge_features, 1, 16, 1, key=key_prune, final_activation=jnn.sigmoid)
		else:
			self.prune_fn = None

		# ---

		if dev_steps:
			key, key_sa = jr.split(key)
			self.sa_fn = OrnsteinUhlenbeckProcess(obs_dims, key_sa)
		else: 
			self.sa_fn = None

		# ---

		self.mask_A =  partial(reservoir, 
							   N=n_nodes, 
							   in_dims=obs_dims, 
							   out_dims=action_dims, 
							   p_hh=1., 
							   p_ih=1., 
							   p_ho=1.,
							   key=jr.key(1))

	#-------------------------------------------------------------------

	def __call__(self, obs: jax.Array, state: PolicyState, key: jax.Array)->Tuple[jax.Array, PolicyState]:
		"""
		"""
		#state = self.update_state(state, key)
		if not self.block_lt_updates:
			state = jax.lax.cond(
				state.d.sum(),
				lambda s, k: s,
				lambda s, k: self.update_state(s, k),
				state, key)
		# ---
		a, a_seq = self.forward_rnn(obs, state)
		state = state._replace(a_seq=a_seq)
		# --- 
		k = 2. if self.env_is_pendulum else 1.
		action = jnp.argmax(a[-self.action_dims:]) 	\
				 if self.discrete_action 		   	\
				 else a[-self.action_dims:]*k
		if not self.discrete_action:
			action = jnp.where(jnp.isnan(action), 0, action)
		if self.is_recurrent:
			state = state._replace(a=a)
		return action, state

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array)->PolicyState:
		"""
		"""
		key_A, key_e, key_h, key_dev = jr.split(key, 4)
		A = meta_reservoir(N=self.n_nodes,
				   		   in_dims=self.obs_dims,
				   		   out_dims=self.action_dims,	
				   		   mu_hh=self.p_hh, 
				   		   s_hh=self.s_hh,
				   		   mu_ih=self.p_ih, 
				   		   s_ih=self.s_ih,
				   		   mu_ho=self.p_ho,
				   		   s_ho=self.s_ho,
				   		   clip=False,
				   		   key=key_A)
		h = jr.uniform(key_h, (self.n_nodes, self.node_features), minval=-1., maxval=1.)
		e = jr.uniform(key_e, (self.n_nodes, self.n_nodes, self.edge_features), minval=-1., maxval=1.) * A[...,None]
		G = Graph(A=A, e=e, h=h)
		a = jnp.zeros((self.n_nodes,))
		w = e[...,0]
		state =  PolicyState(a=a, w=w, G=G, r=jnp.zeros((1,)), d=jnp.zeros((1,)), 
							 t=jnp.zeros(()), a_seq=jnp.zeros((self.n_nodes, self.rnn_iters+1)))

		# --- Dvpt phase ---
		if self.dev_steps:
			state = self.dev(state, key_dev)

		return state
	#-------------------------------------------------------------------

	def dev(self, state: PolicyState, key: jax.Array):

		assert self.sa_fn is not None

		def dev_step(i, s):
			state, sa, key = s
			key, key_sa, key_up = jr.split(key, 3)
			sa = self.sa_fn(sa, key_sa) #type:ignore
			state = self.update_state(state, key_up, is_dev=jnp.ones(()))
			_, a_seq = self.forward_rnn(sa, state)
			state = state._replace(a_seq=a_seq)
			return [state,sa,key]

		key_init, key_dev = jr.split(key)
		sa = self.sa_fn.initialize(key_init)
		state, *_ = jax.lax.fori_loop(0, self.dev_steps, dev_step, [state,sa,key_dev])
		state = state._replace(a=jnp.zeros_like(state.a))
		return state

	#-------------------------------------------------------------------

	def update_state(self, state: PolicyState, key: jax.Array, is_dev: Float=jnp.zeros(()))->PolicyState:
		"""
		Update the state of the network:
			- Update nodes based on their activation sequences and gnn perception
			- Update edges based on node states
		"""
		G = state.G
		N = G.h.shape[0]

		# --- Update node states ---

		h = G.h
		a = state.a_seq
		x = jax.vmap(self.Wpre)(jnp.concatenate([self.get_node_features(G), a, h], axis=-1))
		x = jax.lax.fori_loop(0, self.gnn_iters, lambda _, G: self.gnn(G), G._replace(h=x, e=self.get_edge_features(G))).h
		h = jax.vmap(self.node_fn)(x, h)

		# --- Update edge states ---

		e = G.e
		hh = jnp.concatenate(
			[jnp.repeat(h[None,:], N, axis=0),
			 jnp.repeat(h[:,None], N, axis=1),
			 jnp.repeat(a[None,:], N, axis=0),
			 jnp.repeat(a[:, None], N, axis=1),
			 jnp.ones((self.n_nodes, self.n_nodes, 1))*state.r],
			axis=-1)
		e = double_vmap(self.edge_fn)(hh, e) * state.G.A[...,None]
		w = e[...,0]

		# --- Prune edges ---
		if self.pruning:
			key, key_prune = jr.split(key)
			p_pruned = double_vmap(self.prune_fn)(e)[...,0]
			if self.stochastic_decisions:
				pruned = (jr.uniform(key_prune, p_pruned.shape)<(p_pruned*.5)).astype(float) * self.mask_A()
			else:
				pruned = (p_pruned>.6).astype(float) * self.mask_A()
		else:
			pruned = jnp.zeros((self.n_nodes,self.n_nodes))

		# --- Add edges ---
		if self.synaptogenesis:
			key, key_add = jr.split(key)
			hh = jnp.concatenate(
				[jnp.repeat(h[None,:], N, axis=0),
				 jnp.repeat(h[:,None], N, axis=1)],
				axis=-1)
			p_add = double_vmap(self.adde_fn)(hh)[...,0]
			if self.stochastic_decisions:
				add = (jr.uniform(key_add, p_add.shape)<(p_add*.5)).astype(float)* self.mask_A()
			else:
				add = (p_add > .6).astype(float) * self.mask_A()
		else:
			add = jnp.zeros((self.n_nodes,self.n_nodes))
		
		# Modify adjacency matrix
		A = jnp.where(add, 1., G.A)
		A = jnp.where(pruned, 0., A)

		return eqx.tree_at(lambda s: [s.G.h, s.G.e, s.G.A, s.w, s.t], 
						   state, 
						   [h, e, A, w, state.t+1.])

	#-------------------------------------------------------------------

	def get_edge_features(self, graph: Graph)->jax.Array:

		return jnp.concatenate(
			[
				graph.e,
				graph.A[...,None],
				graph.A.T[...,None],
				jnp.identity(graph.N)[...,None]
			], axis=-1
		)

	#-------------------------------------------------------------------

	def get_node_features(self, graph: Graph)->jax.Array:
		"""
		6 features: 
			- in degree
			- out degree
			- degree
			- layer (1-hot encoding of {inp, hidden, ou}) 	
		"""
		i_d = graph.A.sum(1)[...,None] / 10.
		o_d = graph.A.sum(0)[...,None] / 10.
		d = i_d+o_d / 20.
		typ = jnp.zeros((self.n_nodes, 2)).at[:self.obs_dims, 0].set(1.).at[-self.action_dims:,1].set(1.)

		return jnp.concatenate([i_d, o_d, d, typ], axis=-1)

	#-------------------------------------------------------------------

	def forward_rnn(self, 
					obs: Float[Array, "O"], 
					state: PolicyState
					)->Tuple[Float[Array, "N"], Float[Array, "t+1 N"]]:
		"""
		Run RNN forward and return the final and sequence of activations for each node
		N: number of nodes
		t: number of iterations of rnn
		"""
		if self.use_bias:
			b = state.G.h[:, 0]
		else:
			b = jnp.zeros_like(state.a)
		rnn = lambda a, _: (jnn.tanh(a.at[:self.obs_dims].set(obs) @ state.w + b), a.at[:self.obs_dims].set(obs))
		a, a_seq = jax.lax.scan(rnn, state.a, None, self.rnn_iters)
		return a, jnp.concatenate([a_seq, a[None,:]], axis=0).T



#================================================================================================================================================
#================================================================================================================================================
#================================================================================================================================================

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



def make(cfg, key):

	if cfg.env_name=="Grid":
		obs_dims, action_dims, action_type = 1, 3, "discrete"
	else:
		obs_dims, action_dims, action_type = ENV_SPACES.get(cfg.env_name, None)

	return LNDP(action_dims=action_dims,
				  obs_dims=obs_dims,
				  n_nodes=cfg.n_nodes,
				  edge_features=cfg.edge_features,
				  dev_steps=cfg.dev_steps,
				  rnn_iters=cfg.rnn_iters,
				  node_features=cfg.node_features,
				  p_hh=cfg.p_hh,
				  s_hh=cfg.s_hh,
				  p_ih=cfg.p_ih,
				  s_ih=cfg.s_ih,
				  p_ho=cfg.p_ho,
				  s_ho=cfg.s_ho,
				  discrete_action=action_type in ["d", "discrete"],
				  env_is_pendulum=cfg.env_name=="Pendulum-v1",
				  use_bias=bool(cfg.use_bias),
				  is_recurrent=bool(cfg.is_recurrent),
				  gnn_iters=cfg.gnn_iters,
				  stochastic_decisions=bool(cfg.stochastic_decisions),
				  pruning=bool(cfg.pruning),
				  synaptogenesis=bool(cfg.synaptogenesis),
				  block_lt_updates=bool(cfg.block_lt_updates),
				  key=key)







