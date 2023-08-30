from desc.compute.utils import get_transforms, get_profiles, get_params, dot, cross
from desc.compute import compute as compute_fun
from desc.backend import jnp
from desc.grid import Grid
import desc.io
import desc.examples
from functools import partial
from jax import jit
from jax.experimental.ode import odeint as jax_odeint
import matplotlib.pyplot as plt
import numpy as np

eq = desc.examples.get("DSHAPE")
eq._iota = eq.get_profile("iota")
eq._current = None

def rhs(w, t, a):
    
    #initial conditions
    psi, theta, zeta, vpar = w
    
    mu = a[0]

    #obtaining data from DESC   
    keys = ["B", "|B|", "grad(|B|)", "grad(psi)", "e^theta", "e^zeta", "G"] # etc etc, whatever terms you need
    grid = Grid(jnp.array([psi, theta, zeta]).T, jitable=True, sort=False)
    transforms = get_transforms(keys, eq, grid, jitable=True)
    profiles = get_profiles(keys, eq, grid, jitable=True)
    params = get_params(keys, eq)
    data = compute_fun(eq, keys, params, transforms, profiles)
    
    
    psidot = a[1]*(1/data["|B|"]**3)*(mu*data["|B|"] + vpar**2)*jnp.sum(jnp.cross(data["B"], data["grad(|B|)"], axis=-1) * data["grad(psi)"]) # etc etc
    
    
    thetadot = a[2]*vpar/data["|B|"] * jnp.sum(data["B"] * data["e^theta"]) + (1/data["|B|"]**3)*(mu*data["|B|"] + vpar**2)*jnp.sum(jnp.cross(data["B"], data["grad(|B|)"], axis=-1) * data["e^theta"])
    
    
    zetadot = a[3]*vpar*(data["|B|"]/data["G"]) # (vpar/data["|B|"]) * dot(data["B"], data["e^zeta"]) 
    
    b = data["B"]/data["|B|"]

    part1 = (b + (1/(vpar*data["|B|"]**3)) * (mu*data["|B|"] + vpar**2) * jnp.cross(data["B"], data["grad(|B|)"], axis=-1))
    part2 = data["grad(|B|)"]
    vpardot = -mu*dot(part1,part2)
    #vpardot = -mu*jnp.sum(((data["B"]/data["|B|"])+ (1/vpar*data["|B|"]**3)*(mu*data["|B|"] + vpar**2)*jnp.cross(data["B"], data["grad(|B|)"], axis=-1)) * data["grad(|B|)"])
    
    return jnp.array([psidot, thetadot, zetadot, vpardot]) #, zetadot, vpardot])


initial_conditions = [0.2, 1, 1, 10]
a_initial = [1, 1, 1, 1]

rhs(initial_conditions, 1, a_initial)



nt_per_time_unit = 100
tmin = 0
tmax = 2
nt = int(nt_per_time_unit * (tmax - tmin))
a = [1, 1, 1]

def solve_with_jax(a=None):
    initial_conditions_jax = jnp.array(initial_conditions, dtype=jnp.float64)
    if a is not None:
        a_jax = a
    else:
        a_jax = jnp.array(a_initial, dtype=jnp.float64)
    t_jax = jnp.linspace(tmin, tmax, nt)
    system_jit = jit(rhs)
    #solution_jax = jax_odeint(system_jit, initial_conditions_jax, t_jax, a_jax)
    solution_jax = jax_odeint(partial(system_jit, a=a_jax), initial_conditions_jax, t_jax)
    return solution_jax

sol = solve_with_jax()

print(sol)

plt.plot(np.sqrt(sol[:, 0]) * np.cos(sol[:, 1]), np.sqrt(sol[:, 0]) * np.sin(sol[:, 1]))
plt.show()