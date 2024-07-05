from typing import NamedTuple, Optional, Callable, Union
from jaxtyping import Float
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn

class Graph(NamedTuple):
	A: jax.Array
	h: jax.Array
	e: jax.Array

	@property
	def N(self):
		return self.h.shape[0]


def simple_rnn(h, w):
	h = jnn.tanh(jnp.dot(h,w))
	return h

scaled_dot_product = lambda q, k: jnp.dot(q,k.T) / jnp.sqrt(k.shape[-1])

normalize = lambda x: x / (jnp.linalg.norm(x, axis=-1, keepdims=True)+1e-8)


class GraphTransformer(eqx.Module):
	
	"""
	paper: https://arxiv.org/pdf/2012.09699v2.pdf
	"""
	#-------------------------------------------------------------------
	# params :
	Q: nn.Linear   # Query function
	K: nn.Linear   # Key
	V: nn.Linear   # Value
	O: nn.Linear   # Output V*heads->do
	E: Optional[nn.Linear]  # Edge attention
	# statics : 
	n_heads: int
	use_edge_features: bool
	qk_features: int
	value_features: int
	#-------------------------------------------------------------------

	def __init__(self, 
				 in_features: int, 
				 out_features: int, 
				 qk_features: int, 
				 value_features: int, 
				 n_heads: int, 
				 *, 
				 use_edge_features: bool=False,
				 in_edge_features: int=1,
				 use_bias: bool=False,
				 key: jax.Array):
		"""
		"""
		key_Q, key_K, key_V, key_O, key_E = jr.split(key, 5)
		self.n_heads = n_heads
		self.use_edge_features = use_edge_features
		self.qk_features = qk_features
		self.value_features = value_features
		
		self.Q = nn.Linear(in_features, qk_features*n_heads, key=key_Q, use_bias=use_bias)
		self.K = nn.Linear(in_features, qk_features*n_heads, key=key_K, use_bias=use_bias)
		self.V = nn.Linear(in_features, value_features*n_heads, key=key_V, use_bias=use_bias)
		self.O = nn.Linear(value_features*n_heads, out_features, key=key_O)
		if use_edge_features:
			self.E = nn.Linear(in_edge_features, n_heads, key=key_E)
		else:
			self.E = None

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph)->Graph:
		"""return features aggregated through attention"""
		h = graph.h
		N = h.shape[0]
		q, k, v = jax.vmap(self.Q)(h), jax.vmap(self.K)(h), jax.vmap(self.V)(h)
		# Compute attention scores (before softmax) (N x N x H)
		scores = jax.vmap(scaled_dot_product, in_axes=-1, out_axes=-1)(q.reshape((N, self.qk_features, -1)), 
																	   k.reshape((N, self.qk_features, -1)))
		if self.use_edge_features:
			assert self.E is not None
			# use edge features to compute attention scores
			we = jax.vmap(jax.vmap(self.E))(graph.e)
			scores = scores * we

		w = jnn.softmax(scores, axis=1) # (N x N x H)
		x = jax.vmap(jnp.dot, in_axes=-1, out_axes=-1)(
			w.transpose((1,0,2)), v.reshape(N, self.value_features, -1)
		) # (N x dv X H)
		x = x.reshape((N, -1)) # (N x dv*H) (concatenate the heads)

		h = jax.vmap(self.O)(x)

		return eqx.tree_at(lambda G: G.h, graph, h)


def erdos_renyi(key: jax.Array, N: int, p: float, self_loops: bool=False):
	"""random adjacemcy matrix"""
	A = (jr.uniform(key, (N,N)) < p).astype(float)
	if not self_loops:
		A = jnp.where(jnp.identity(N), 0., A)
	return A

def reservoir(key: jax.Array, N: int, in_dims: int, out_dims: int, p_hh: float=1., p_ih: float=.3, p_ho: float=.5):
	key_ih, key_hh, key_ho = jr.split(key, 3)
	A = jnp.zeros((N, N))
	I = jnp.arange(in_dims)
	O = jnp.arange(out_dims) + (N-out_dims)
	H = jnp.arange(N-out_dims-in_dims) + in_dims

	A = A.at[jnp.ix_(I, H)].set((jr.uniform(key_ih, (in_dims, len(H)))<p_ih).astype(float))
	A = A.at[jnp.ix_(H, H)].set(erdos_renyi(key_hh, N=len(H), p=p_hh))
	A = A.at[jnp.ix_(H, O)].set((jr.uniform(key_ho, (len(H), out_dims))<p_ho).astype(float))

	return A

def truncated_normal(key: jax.Array, mu, sigma, lower, upper):
	return jr.truncated_normal(key, -(mu-lower)/sigma, (upper-mu)/sigma) * sigma+mu

def meta_reservoir(key, N, in_dims, out_dims, mu_hh, mu_ih, mu_ho, s_hh, s_ih, s_ho, clip=True):
	"""Darws a reservoit type net where connection probas are also drawn from a distribution"""
	khh, kih, kho, knet = jr.split(key, 4)

	m, l, u, s = mu_hh, .0, 1., s_hh
	phh = jr.truncated_normal(khh, -(m-l)/s, (u-m)/s) * s + m

	pih = jr.normal(kih) * s_ih + mu_ih
	pho = jr.normal(kho) * s_ho + mu_ho

	if clip:
		pih = jnp.clip(jr.normal(kih) * s_ih + mu_ih, .05, 1.)
		pho = jnp.clip(jr.normal(kho) * s_ho + mu_ho, .05, 1.)

	return reservoir(knet, N, in_dims, out_dims, phh, pih, pho) #type:ignore



class OrnsteinUhlenbeckProcess(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	# Parameters:
	mu: jax.Array
	alpha: jax.Array
	sigma: jax.Array
	dt: jax.Array
	# Statics:
	d: int
	#-------------------------------------------------------------------

	def __init__(self, d: int, key: jax.Array):
		
		self.d = d

		k_mu, k_sigma, k_alpha, k_dt = jr.split(key, 4)

		self.mu = jr.normal(k_mu, (d,))
		self.sigma = jr.uniform(k_sigma, (d,d), minval=-1., maxval=1.)
		self.alpha = jr.uniform(k_alpha, (d,), minval=0.01, maxval=1.)
		self.dt = jr.uniform(k_dt, (d,), minval=0.01, maxval=0.5)

	#-------------------------------------------------------------------

	def __call__(self, x: jax.Array, key: jax.Array)->jax.Array:
		
		sigma = jnp.clip(self.sigma, -1., 1.)
		alpha = jnp.clip(self.alpha, 0., jnp.inf)
		mu = self.mu
		dt = jnp.clip(self.dt, 0.01, 1.)
		W = jr.multivariate_normal(key, mean=jnp.zeros((self.d,)), cov=sigma, method="svd")
		x = x + alpha*(mu-x) * dt + W
		return x

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array)->jax.Array:
		sigma = jnp.clip(self.sigma, -1., 1.)
		W = jr.multivariate_normal(key, mean=jnp.zeros((self.d,)), cov=sigma, method="svd")
		return W