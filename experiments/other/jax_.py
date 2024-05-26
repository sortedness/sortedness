import jax.numpy as jnp
from jax import grad

def jax_function(x):
    return jnp.sin(x) + jnp.cos(x) * x**3

def display_derivative_code(derivative_func):
    import inspect
    source_lines, _ = inspect.getsourcelines(derivative_func)
    print("".join(source_lines))

derivative = grad(jax_function)

display_derivative_code(derivative)
