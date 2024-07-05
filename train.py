from utils.exputils import load_config
from lndp import make as model_factory
from task import make as task_factory, MultiTask
from trainer import make as trainer_factory

from typing import NamedTuple
import equinox as eqx
import jax.random as jr

PROJECT = "lndp"

class Config(NamedTuple):
	project: str=PROJECT #wandb project name
	seed: int=1
	n_seeds: int=1
	# --- trainer ---
	strategy: str="CMA_ES"
	popsize: int=256
	generations: int=10_000
	ckpt_file: str=""
	log: int=0 # 1 for logging to wandb
	eval_reps: int=1 # number of evaluations to average over
	# --- task ---
	env_name: str="CartPole-v1"
	n_episodes: int=3 # number of enviornment episodes
	l1_penalty: float=0.
	dev_after_episode: int=0
	env_size: int=5
	p_switch: float=0.
	dense_reward: int=0
	# --- model ---
	n_nodes: int=32 # max nb of nodes in the network
	node_features: int=8
	edge_features: int=4
	pruning: int=1
	synaptogenesis: int=1
	rnn_iters: int=3 #number of propagation steps
	dev_steps: int=0 #number of developmental steps
	p_hh: float=0.1 #initial connection probabilities (avergae/variance)
	s_hh: float=0.0001
	p_ih: float=0.1
	s_ih: float=0.0001
	p_ho: float=0.1
	s_ho: float=0.0001
	use_bias: int=0
	is_recurrent: int=0 # activity is resetted between env steps if is_recurrent=0
	gnn_iters: int=1 # number of GNN forward pass
	stochastic_decisions: int=0 # if synaptogenesis or pruning is probabilistic
	block_lt_updates: int=0 # if 1 will block any change during the agent lifetime


if __name__ == '__main__':
	
	cfg = load_config(Config)
	key_model, key_train = jr.split(jr.key(cfg.seed))

	if "," not in cfg.env_name:
		mdl = model_factory(cfg, key_model)
		params, statics = eqx.partition(mdl, eqx.is_array)
		task = task_factory(cfg, statics)
	else:
		env_names=cfg.env_name.split(",")
		statics = []
		tsks = []
		for env_name in env_names:
			_cfg = cfg._replace(env_name=env_name)
			mdl = model_factory(_cfg, key_model)
			params, statics = eqx.partition(mdl, eqx.is_array)
			task = task_factory(_cfg, statics)
			tsks.append(task)
		task = MultiTask(tsks)


	trainer = trainer_factory(cfg, task, params) #type:ignore

	for seed in range(cfg.n_seeds):
		key_train, _ktrain = jr.split(key_train)
		if cfg.log:
			trainer.logger.init(cfg.project, cfg._replace(seed=cfg.seed+seed)._asdict()) #type:ignore
		trainer.init_and_train_(_ktrain)
		if cfg.log:
			trainer.logger.finish() #type:ignore
