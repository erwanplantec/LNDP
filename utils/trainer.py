import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jax.experimental.shard_map import shard_map as shmap
import jax.experimental.host_callback as hcb
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import evosax as ex
import equinox as eqx
from typing import Any, Callable, Dict, Optional, Union, Tuple, TypeAlias
from jaxtyping import PyTree
import os
import wandb
from utils.exputils import *

Data: TypeAlias = PyTree[...]
TaskParams: TypeAlias = PyTree[...]
TrainState: TypeAlias = PyTree[...]

class Logger:

	#-------------------------------------------------------------------

	def __init__(
		self, 
		wandb_log: bool,
		metrics_fn: Callable[[TrainState, Data], Tuple[Data, Data, int]],
		ckpt_file: Optional[str]=None, 
		ckpt_freq: int=100,
		verbose: bool=False):
		
		if ckpt_file is not None and "/" in ckpt_file:
			if not os.path.isdir(ckpt_file[:ckpt_file.rindex("/")]):
				os.makedirs(ckpt_file[:ckpt_file.rindex("/")])
		self.wandb_log = wandb_log
		self.metrics_fn = metrics_fn
		self.ckpt_file = ckpt_file
		self.ckpt_freq = ckpt_freq
		self.epoch = [0]
		self.verbose = verbose

	#-------------------------------------------------------------------

	def log(self, state: TrainState, data: Data):
		
		log_data, ckpt_data, epoch = self.metrics_fn(state, data)
		if self.wandb_log:
			self._log(log_data)
		self.save_chkpt(ckpt_data, epoch)
		return log_data

	#-------------------------------------------------------------------

	def _log(self, data: dict):
		hcb.id_tap(
			lambda d, *_: wandb.log(d), data
		)

	#-------------------------------------------------------------------

	def save_chkpt(self, data: dict, epoch: int):

		def save(data):
			assert self.ckpt_file is not None
			file = f"{self.ckpt_file}.eqx"
			if self.verbose:
				print("saving data at: ", file)
			eqx.tree_serialise_leaves(file, data)

		def tap_save(data):
			hcb.id_tap(lambda d, *_: save(d), data)
			return None

		if self.ckpt_file is not None:
			jax.lax.cond(
				(jnp.mod(epoch, self.ckpt_freq))==0,
				lambda data : tap_save(data),
				lambda data : None,
				data
			)

	#-------------------------------------------------------------------

	def wandb_init(self, project: str, config: dict, **kwargs):
		if self.wandb_log:
			wandb.init(project=project, config=config, **kwargs)

	#-------------------------------------------------------------------

	def wandb_finish(self, *args, **kwargs):
		if self.wandb_log:
			wandb.finish(*args, **kwargs)



class BaseTrainer(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	train_steps: int
	logger: Optional[Logger]
	progress_bar: Optional[bool]
	#-------------------------------------------------------------------

	def __init__(self, 
				 train_steps: int, 
				 logger: Optional[Logger]=None,
				 progress_bar: Optional[bool]=False):
		
		self.train_steps = train_steps
		self.progress_bar = progress_bar
		self.logger = logger

	#-------------------------------------------------------------------

	def __call__(self, key: jax.Array):

		return self.init_and_train(key)

	#-------------------------------------------------------------------

	def train(self, state: TrainState, key: jax.Array, data: Optional[Data]=None)->Tuple[TrainState, Data]:

		def _step(c, x):
			s, k = c
			k, k_ = jr.split(k)
			s, data = self.train_step(s, k_)
			
			if self.logger is not None:
				self.logger.log(s, data)

			return [s, k], {"states": s, "metrics": data}

		if self.progress_bar:
			_step = progress_bar_scan(self.train_steps)(_step) #type: ignore

		[state, key], data = jax.lax.scan(_step, [state, key], jnp.arange(self.train_steps))

		return state, data

	#-------------------------------------------------------------------

	def train_(self, state: TrainState, key: jax.Array, data: Optional[Data]=None)->TrainState:

		def _step(i, c):
			s, k = c
			k, k_ = jr.split(k)
			s, data = self.train_step(s, k_)
			if self.logger is not None:
				self.logger.log(s, data)
			return [s, k]

		if self.progress_bar:
			_step = progress_bar_fori(self.train_steps)(_step) #type: ignore

		[state, key] = jax.lax.fori_loop(0, self.train_steps, _step, [state, key])
		return state

	#-------------------------------------------------------------------

	def log(self, data):
		hcb.id_tap(
			lambda d, *_: wandb.log(d), data
		)

	#-------------------------------------------------------------------

	def init_and_train(self, key: jax.Array, data: Optional[Data]=None)->Tuple[TrainState, Data]:
		init_key, train_key = jr.split(key)
		state = self.initialize(init_key)
		return self.train(state, train_key, data)

	#-------------------------------------------------------------------

	def init_and_train_(self, key: jax.Array, data: Optional[Data]=None)->TrainState:
		init_key, train_key = jr.split(key)
		state = self.initialize(init_key)
		return self.train_(state, train_key, data)

	#-------------------------------------------------------------------


Params = PyTree[...]
Task = Callable


def default_metrics(state, data):
	y = {}
	y["best"] = state.best_fitness
	y["gen_best"] = data["fitness"].min()
	y["gen_mean"] = data["fitness"].mean()
	y["gen_worse"] = data["fitness"].max()
	y["var"] = data["fitness"].var()
	return y


class EvosaxTrainer(BaseTrainer):
	
	"""
	"""
	#-------------------------------------------------------------------
	strategy: ex.Strategy
	es_params: ex.EvoParams
	params_shaper: ex.ParameterReshaper
	task: Task
	fitness_shaper: ex.FitnessShaper
	n_devices: int
	multi_device_mode: str
	#-------------------------------------------------------------------

	def __init__(
		self, 
		train_steps: int,
		strategy: Union[ex.Strategy, str],
		task: Callable,
		params_shaper: ex.ParameterReshaper,
		popsize: Optional[int]=None,
		fitness_shaper: Optional[ex.FitnessShaper]=None,
		es_kws: Optional[Dict[str, Any]]={},
		es_params: Optional[ex.EvoParams]=None,
		eval_reps: int=1,
		logger: Optional[Logger]=None,
	    progress_bar: Optional[bool]=True,
	    n_devices: int=1,
	    multi_device_mode: str="shmap"):

		super().__init__(train_steps=train_steps, 
						 logger=logger, 
						 progress_bar=progress_bar)
		
		if isinstance(strategy, str):
			assert popsize is not None
			self.strategy = self.create_strategy(strategy, popsize, params_shaper.total_params, **es_kws) # type: ignore
		else:
			self.strategy = strategy

		if es_params is None:
			self.es_params = self.strategy.default_params
		else:
			self.es_params = es_params

		self.params_shaper = params_shaper

		if eval_reps > 1:
			def _eval_fn(p: Params, k: jax.Array, tp: Optional[PyTree]=None):
				"""
				"""
				fit, info = jax.vmap(task, in_axes=(None,0,None))(p, jr.split(k,eval_reps), tp)
				return jnp.mean(fit), info
			self.task = _eval_fn
		else :
			self.task = task

		if fitness_shaper is None:
			self.fitness_shaper = ex.FitnessShaper()
		else:
			self.fitness_shaper = fitness_shaper

		self.n_devices = n_devices
		self.multi_device_mode = multi_device_mode

	#-------------------------------------------------------------------

	def eval(self, *args, **kwargs):
		
		if self.n_devices == 1:
			return self._eval(*args, **kwargs)
		if self.multi_device_mode=="shmap":
			return self._eval_shmap(*args, **kwargs)
		elif self.multi_device_mode == "pmap":
			return self._eval_pmap(*args, **kwargs)
		else:
			raise ValueError(f"multi_device_mode {self.multi_device_mode} is not a valid mode")

	#-------------------------------------------------------------------

	def _eval(self, x: jax.Array, key: jax.Array, task_params: PyTree)->Tuple[jax.Array, PyTree]:
		
		params = self.params_shaper.reshape(x)
		_eval = jax.vmap(self.task, in_axes=(0, 0, None))
		return _eval(params, jr.split(key, x.shape[0]), task_params)

	#-------------------------------------------------------------------

	def _eval_shmap(self, x: jax.Array, key: jax.Array, task_params: PyTree)->Tuple[jax.Array, PyTree]:
		
		devices = mesh_utils.create_device_mesh((self.n_devices,))
		device_mesh = Mesh(devices, axis_names=("p"))

		_eval = lambda x, k: self.task(self.params_shaper.reshape_single(x), k)
		batch_eval = jax.vmap(_eval, in_axes=(0,None))
		sheval = shmap(batch_eval, 
					   mesh=device_mesh, 
					   in_specs=(P("p",), P()),
					   out_specs=(P("p"), P("p")),
					   check_rep=False)

		return sheval(x, key)

	#-------------------------------------------------------------------

	def _eval_pmap(self, x: jax.Array, key: jax.Array, data: PyTree)->Tuple[jax.Array, PyTree]:
		
		_eval = lambda x, k: self.task(self.params_shaper.reshape_single(x), k)
		batch_eval = jax.vmap(_eval, in_axes=(0,None))
		pop_batch = x.shape[0] // self.n_devices
		x = x.reshape((self.n_devices, pop_batch, -1))
		pmapeval = jax.pmap(batch_eval, in_axes=(0,None)) #type: ignore
		f, eval_data = pmapeval(x, key)
		return f.reshape((-1,)), eval_data

	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jax.Array, data: Optional[TaskParams]=None) -> Tuple[TrainState, Data]:
		
		ask_key, eval_key = jr.split(key, 2)
		x, state = self.strategy.ask(ask_key, state, self.es_params)
		fitness, eval_data = self.eval(x, eval_key, data)
		f = self.fitness_shaper.apply(x, fitness)
		state = self.strategy.tell(x, f, state, self.es_params)
		state = self._update_evo_state(state, x, fitness)
		return state, {"fitness": fitness, "data": eval_data} #TODO def best as >=

	#-------------------------------------------------------------------

	def _update_evo_state(self, state: TrainState, x: jax.Array, f: jax.Array)->TrainState:
		is_best = f.min() <= state.best_fitness
		gen_best = x[jnp.argmin(f)]
		best_member = jax.lax.select(is_best, gen_best, state.best_member)
		state = state.replace(best_member=best_member) #type:ignore
		return state

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array, **kwargs) -> TrainState:
		
		state = self.strategy.initialize(key, self.es_params)
		state = state.replace(**kwargs)
		return state

	#-------------------------------------------------------------------

	def create_strategy(self, name: str, popsize: int, num_dims: int, **kwargs)->ex.Strategy:
		
		ES = getattr(ex, name)
		es = ES(popsize=popsize, num_dims=num_dims, **kwargs)
		return es

	#-------------------------------------------------------------------

	def load_ckpt(self, ckpt_path: str)->Params:
		params = eqx.tree_deserialise_leaves(
			ckpt_path, jnp.zeros((self.params_shaper.total_params,))
		)
		return params

	#-------------------------------------------------------------------

	def train_from_model_ckpt(self, ckpt_path: str, key: jax.Array)->Tuple[TrainState, Data]: #type:ignore
		
		key_init, key_train = jr.split(key)
		params = self.load_ckpt(ckpt_path)
		state = self.initialize(key_init, mean=self.params_shaper.flatten_single(params))
		return self.train(state, key_train)

	#-------------------------------------------------------------------

	def train_from_model_ckpt_(self, ckpt_path: str, key: jax.Array)->TrainState:#type:ignore
		
		key_init, key_train = jr.split(key)
		params = self.load_ckpt(ckpt_path)
		state = self.initialize(key_init, mean=self.params_shaper.flatten_single(params))
		return self.train_(state, key_train)