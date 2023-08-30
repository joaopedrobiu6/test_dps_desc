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
from desc.equilibrium import Equilibrium
from desc.plotting import plot_surfaces, plot_3d
from scipy.integrate import solve_ivp, odeint

eq = desc.io.load("/home/joaobiu/DESC/desc/examples/test_run.h5")
eq._iota = eq.get_profile("iota")
eq._current = None 

#plot_3d(eq, "|B|")


def rhs(w, t, a):
    
    #initial conditions
    psi, theta, zeta, vpar = w
    
    E = a[0]

    #obtaining data from DESC   
    keys = ["B", "|B|", "grad(|B|)", "grad(psi)", "e^theta", "e^zeta", "G"] # etc etc, whatever terms you need
    grid = Grid(np.array([psi, theta, zeta]).T, jitable=False, sort=False)
    transforms = get_transforms(keys, eq, grid, jitable=False)
    profiles = get_profiles(keys, eq, grid, jitable=False)
    params = get_params(keys, eq)
    data = compute_fun(eq, keys, params, transforms, profiles)
    
    mu = E/(data["|B|"]) - (vpar**2)/(4*data["|B|"])
    
    psidot = a[1]*(1/data["|B|"]**3)*(mu*data["|B|"] + vpar**2)*np.sum(np.cross(data["B"], data["grad(|B|)"], axis=-1) * data["grad(psi)"]) # etc etc
    
    
    thetadot = a[2]*vpar/data["|B|"] * np.sum(data["B"] * data["e^theta"]) + a[1]*(1/data["|B|"]**3)*(mu*data["|B|"] + vpar**2)*np.sum(np.cross(data["B"], data["grad(|B|)"], axis=-1) * data["e^theta"])
    
    
    zetadot = a[2]*(vpar/data["|B|"]) * np.sum(data["B"] * data["e^zeta"]) 
    
    b = data["B"]/data["|B|"]

    teste1 = (b + (1/(vpar*data["|B|"]**3)) * (mu*data["|B|"] + vpar**2) * np.cross(data["B"], data["grad(|B|)"], axis=-1))
    teste2 = data["grad(|B|)"]
    vpardot = -mu*np.sum(teste1 * teste2)
    #vpardot = -mu*jnp.sum(((data["B"]/data["|B|"])+ (1/vpar*data["|B|"]**3)*(mu*data["|B|"] + vpar**2)*jnp.cross(data["B"], data["grad(|B|)"], axis=-1)) * data["grad(|B|)"])
    
    ret = np.asarray[psidot, thetadot, zetadot, vpardot]
    ret_reshape = ret.reshape(4,)

    return ret_reshape


e_charge = 1.6e-19
E = 3.52e4
v_parallel = 0.8*jnp.sqrt(E/8)

a_initial = np.asarray([E, 2, 1])
initial_conditions = np.asarray([0.2, 0.2, 0.1, v_parallel])



nt_per_time_unit = 10
tmin = 0
tmax = 10
nt = int(nt_per_time_unit * (tmax - tmin))

def solve_with_scipy(a=None):
    t = np.linspace(tmin, tmax, nt)
    if a is not None:
        a = a
    else:
        a = a_initial
    # sol = solve_ivp(lambda t, w: system(w, t, a), [tmin, tmax], initial_conditions, t_eval=t, method='RK45')
    # return sol.y.T
    sol = odeint(lambda w, t: rhs(w, t, a), initial_conditions, t)
    return sol

sol = solve_with_scipy(a_initial)

print(sol)

plt.plot(np.sqrt(sol[1:, 0]) * np.cos(sol[1:, 1]), np.sqrt(sol[1:, 0]) * np.sin(sol[1:, 1]))
plt.xlabel(r'sqrt($\psi$)*cos($\theta$)')
plt.ylabel(r'sqrt($\psi$)*sin($\theta$)')
plt.title('ODE Solutions')
plt.show()