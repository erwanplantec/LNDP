import argparse

def load_config(factory):
	default_config = factory()
	parser = argparse.ArgumentParser()
	bools = []
	for k, v in default_config._asdict().items():
		dtype = int if isinstance(v, bool) else type(v)
		dv = int(v) if isinstance(v, bool) else v
		if isinstance(v, bool): bools.append(k)
		parser.add_argument(f"--{k}", type=dtype, default=dv)
	config = vars(parser.parse_args())
	for k in bools:
		config[k] = bool(config[k])
	config = factory(**config)
	return config

import jax
import jax.experimental.host_callback as hcb
from tqdm import tqdm

def progress_bar_scan(num_samples, message=None):
    "Progress bar for a JAX scan"
    if message is None:
        message=""
    tqdm_bars = {}

    print_rate = 5
    remainder = num_samples % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(num_samples))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: hcb.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )
        _ = jax.lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples-remainder),
            lambda _: hcb.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )
        _ = jax.lax.cond(
            # update tqdm by `remainder`
            iter_num == num_samples-remainder,
            lambda _: hcb.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return jax.lax.cond(
            iter_num == num_samples-1,
            lambda _: hcb.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """
        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x   
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan



def progress_bar_fori(num_samples, message=None):
    "Progress bar for a JAX scan"
    if message is None:
        message=""
    tqdm_bars = {}

    print_rate = 5
    remainder = num_samples % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(num_samples))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: hcb.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )
        _ = jax.lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples-remainder),
            lambda _: hcb.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )
        _ = jax.lax.cond(
            # update tqdm by `remainder`
            iter_num == num_samples-remainder,
            lambda _: hcb.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return jax.lax.cond(
            iter_num == num_samples-1,
            lambda _: hcb.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """
        def wrapper_progress_bar(x, carry):
            iter_num = x   
            _update_progress_bar(iter_num)
            result = func(x, carry)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan