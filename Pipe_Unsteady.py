import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Material Properties
rho = 1e3
mu = 1e-2

# Dimensions of Wind Tunnel
D = 0.25
L = 1

# Reynolds Number
Re = 2000

# Compute inlet velocity
u_in = (Re * mu) / (rho * D)

print(f"u inlet = {u_in}")

t0 = 0.
t_final = 25

print(f"Simulation up until t = {t_final} s")

# Create Wind Tunnel geometry
geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[L, D])
timedomain = dde.geometry.TimeDomain(t0, t_final)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define Boundaries

# Top and Bottom Walls
def boundary_wall(X, on_boundary):
    on_wall = np.logical_and(np.logical_or(np.isclose(X[1], 0, rtol = 1e-5, atol = 1e-8), np.isclose(X[1], D, rtol = 1e-5, atol = 1e-8)), on_boundary)
    return on_wall

# Inlet
def boundary_inlet(X, on_boundary):
    on_inlet = np.logical_and(np.isclose(X[0], 0, rtol = 1e-5, atol = 1e-8), on_boundary)
    return on_inlet

# Outlet
def boundary_outlet(X, on_boundary):
    on_outlet = np.logical_and(np.isclose(X[0], L, rtol = 1e-5, atol = 1e-8), on_boundary)
    return on_outlet

## Define Boundary Condition Functions##

def u_walls(x):
  return 0.

def v_walls(x):
  return 0.

def u_inlet(x):
  return u_in

def v_inlet(x):
  return 0.

def v_outlet(x):
  return 0.

def p_outlet(x):
  return 0.

def u_init(x):
  return 0.

def v_init(x):
  return 0.

def p_init(x):
  return 0.


# No-slip Top and Bottom Walls
bc_wall_u = dde.DirichletBC(
    geomtime, u_walls, boundary_wall, component=0
)
bc_wall_v = dde.DirichletBC(
    geomtime, v_walls, boundary_wall, component=1
)

# Velocity Inlet (u-component = u_in, v-component = 0)
bc_inlet_u = dde.DirichletBC(
    geomtime, u_inlet, boundary_inlet, component=0
)
bc_inlet_v = dde.DirichletBC(
    geomtime, v_inlet, boundary_inlet, component=1
)

# Pressure Outlet
bc_outlet_p = dde.DirichletBC(
    geomtime, p_outlet, boundary_outlet, component=2
)
bc_outlet_v = dde.DirichletBC(
    geomtime, v_outlet, boundary_outlet, component=1
)



## Define the PDE ##
def pde(X, Y):

    # Compute derivatives using Auto-Differentiation
    du_x = dde.grad.jacobian(Y, X, i=0, j=0)
    du_y = dde.grad.jacobian(Y, X, i=0, j=1)
    dv_x = dde.grad.jacobian(Y, X, i=1, j=0)
    dv_y = dde.grad.jacobian(Y, X, i=1, j=1)
    dp_x = dde.grad.jacobian(Y, X, i=2, j=0)
    dp_y = dde.grad.jacobian(Y, X, i=2, j=1)

    du_xx = dde.grad.hessian(Y, X, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(Y, X, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(Y, X, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(Y, X, component=1, i=1, j=1)

    du_t = dde.grad.jacobian(Y, X, i=0, j=2)
    dv_t = dde.grad.jacobian(Y, X, i=1, j=2)

    # x-momentum Navier Stokes equation
    pde_u = (du_t + Y[:, 0:1] * du_x + Y[:, 1:2] * du_y + dp_x - (1 / Re) * (du_xx + du_yy))

    # y-momentum Navier Stokes equation
    pde_v = (dv_t + Y[:, 0:1] * dv_x + Y[:, 1:2] * dv_y + dp_y - (1 / Re) * (dv_xx + dv_yy))

    # Continuity equation
    pde_cont = du_x + dv_y

    return [pde_u, pde_v, pde_cont]

ic_u = dde.icbc.IC(geomtime, u_init, lambda _, on_initial: on_initial, component=0)
ic_v = dde.icbc.IC(geomtime, v_init, lambda _, on_initial: on_initial, component=1)
ic_p = dde.icbc.IC(geomtime, p_init, lambda _, on_initial: on_initial, component=2)


## Define Training Session ##



# Create Data to compute the Losses, Set Domain Points, Boundary Points and Number of Test Points
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p, bc_outlet_v, ic_u, ic_v, ic_p],
    num_domain=25_000,
    num_boundary=3_500,
    num_initial=3_500,
    num_test=5_000
)

# Define the architecture of the PINN
net = dde.maps.FNN([3] + [80] * 6 + [3], "tanh", "Glorot uniform")

# Create the model of the PINN
model = dde.Model(data, net)

# Compile with ADAM
model.compile("adam", lr=1e-3)

# Set epochs for ADAM optimization
losshistory, train_state = model.train(epochs=10_000, display_every=500)

# Optimize with L-BFGS
dde.optimizers.config.set_LBFGS_options(maxiter=3_000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# Define time instances for visualization
step = 0.5
time_instances = np.arange(t0, 11, step)

# Create Prediction Points
samples = geom.random_points(500_000)

# Create function to visualize predictions at specific time instance
def visualize_predictions(model, samples, time_instance):
    # Evaluate the model at the given time instance
    T = np.ones((samples.shape[0], 1)) * time_instance
    XYT = np.hstack((samples, T))
    result = model.predict(XYT)

    # # Plot the predicted u-component of velocity
    # plt.figure(figsize=(20, 4))
    # u_scatter = plt.scatter(samples[:, 0], samples[:, 1], c=result[:, 0], cmap="jet", s=2)
    # plt.colorbar(u_scatter, label='u')
    # plt.clim([min(result[:, 0]), max(result[:, 0])])
    # plt.xlim((0, L))
    # plt.ylim((0, D))
    # plt.tight_layout()
    # plt.title(f"u component of the velocity at t = {time_instance}, Re = {Re}")

    # # Plot the predicted v-component of velocity
    # plt.figure(figsize=(20, 4))
    # v_scatter = plt.scatter(samples[:, 0], samples[:, 1], c=result[:, 1], cmap="jet", s=2)
    # plt.colorbar(v_scatter, label='v')
    # plt.clim([min(result[:, 1]), max(result[:, 1])])
    # plt.xlim((0, L))
    # plt.ylim((0, D))
    # plt.tight_layout()
    # plt.title(f"v component of the velocity at t = {time_instance}, Re = {Re}")

    # # Plot the predicted pressure
    # plt.figure(figsize=(20, 4))
    # p_scatter = plt.scatter(samples[:, 0], samples[:, 1], c=result[:, 2], cmap="jet", s=2)
    # plt.colorbar(p_scatter, label='Pressure')
    # plt.clim([min(result[:, 2]), max(result[:, 2])])
    # plt.xlim((0, L))
    # plt.ylim((0, D))
    # plt.tight_layout()
    # plt.title(f"Pressure at t = {time_instance}, Re = {Re}")

    # Calculate and plot velocity magnitude
    velocity_magnitude = np.sqrt(result[:, 0] ** 2 + result[:, 1] ** 2)
    plt.figure(figsize=(20, 4))
    vel_mag_scatter = plt.scatter(samples[:, 0], samples[:, 1], c=velocity_magnitude, cmap="jet", s=2)
    plt.colorbar(vel_mag_scatter, label='Velocity Magnitude')
    plt.clim([min(velocity_magnitude), max(velocity_magnitude)])
    plt.xlim((0, L))
    plt.ylim((0, D))
    plt.tight_layout()
    plt.title(f"Velocity Magnitude at t = {time_instance}, Re = {Re}")

    plt.show()

# Visualize predictions at specified time instances
for t in time_instances:
    visualize_predictions(model, samples, t)

