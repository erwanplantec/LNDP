import evosax as ex
import jax.numpy as jnp
from utils.trainer import *
import jax
import equinox as eqx


def make(config, task, params_like)->EvosaxTrainer:

	def metrics_fn(state, data):
		y = {}
		for k, v in data["data"].items():
			y[f"{k}_mean"] = v.mean()
			y[f"{k}_max"] = v.max()
		y["best"] = - state.best_fitness
		y["gen_best"] = data["fitness"].max()
		y["gen_mean"] = data["fitness"].mean()
		y["gen_worse"] = data["fitness"].min()
		y["var"] = jnp.var(data["fitness"])
		return y, state.mean, state.gen_counter

	params_shaper = ex.ParameterReshaper(params_like)

	fitness_shaper = ex.FitnessShaper(maximize=True)

	if config.ckpt_file:
		ckpt_file = f"{config.ckpt_file}_{config.env_name}_{config.seed}"
	else:
		ckpt_file = f"./ckpts/{config.env_name}_{config.seed}"
	# save config
	eqx.tree_serialise_leaves(f"{ckpt_file}_config.eqx", config)
	logger = Logger(bool(config.log), metrics_fn=metrics_fn, ckpt_file=ckpt_file)

	trainer = EvosaxTrainer(train_steps = config.generations,
							task = task,
							params_shaper=params_shaper,
							strategy=config.strategy,
							popsize=config.popsize,
							fitness_shaper=fitness_shaper,
							eval_reps=config.eval_reps,
							logger=logger,
							n_devices=len(jax.devices()))

	return trainer


	