import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt


# Material Properties
rho = 1e1
mu = 1e-1

# Dimensions of Lid Driven Cavity
D = 1
L = 1

# Reynolds Number
Re = 100

# Compute inlet velocity
u_in = (Re * mu) / (rho * L)

print(f"u inlet = {u_in}")

# Create Lid Driven Cavity geometry
geom = dde.geometry.Rectangle(xmin = [0, 0], xmax = [L, D])


## Define Boundaries ##

# Top Wall
def top_wall(X, on_boundary):
    on_top_wall = np.logical_and(np.isclose(X[1], D, rtol = 1e-5, atol = 1e-8), on_boundary)
    return on_top_wall

# Bottom Wall
def bottom_wall(X, on_boundary):
    on_bottom_wall = np.logical_and(np.isclose(X[1], 0, rtol = 1e-5, atol = 1e-8), on_boundary)
    return on_bottom_wall

# Left Wall
def left_wall(X, on_boundary):
    on_inlet = np.logical_and(np.isclose(X[0], 0, rtol = 1e-5, atol = 1e-8), on_boundary)
    return on_inlet

# Right Wall
def right_wall(X, on_boundary):
    on_outlet = np.logical_and(np.isclose(X[0], L, rtol = 1e-5, atol = 1e-8), on_boundary)
    return on_outlet


## Define Boundary Conditions ##

# No-slip Bottom, Left and Right Walls
bc_bottom_wall_u = dde.DirichletBC(geom, lambda X: 0., bottom_wall, component = 0)
bc_bottom_wall_v = dde.DirichletBC(geom, lambda X: 0., bottom_wall, component = 1)

bc_left_wall_u = dde.DirichletBC(geom, lambda X: 0, left_wall, component = 0)
bc_left_wall_v = dde.DirichletBC(geom, lambda X: 0, left_wall, component = 1)

bc_right_wall_u = dde.DirichletBC(geom, lambda X: 0., right_wall, component = 0)
bc_right_wall_v = dde.DirichletBC(geom, lambda X: 0., right_wall, component = 1)

# Set Velocity for the Top Wall
bc_top_wall_u = dde.DirichletBC(geom, lambda X: u_in, top_wall, component = 0)
bc_top_wall_v = dde.DirichletBC(geom, lambda X: 0., top_wall, component = 1)


## Define the PDE ##
def pde(X, Y):

    # Compute derivatives using Auto-Differentiation
    du_x = dde.grad.jacobian(Y, X, i = 0, j = 0)
    du_y = dde.grad.jacobian(Y, X, i = 0, j = 1)
    dv_x = dde.grad.jacobian(Y, X, i = 1, j = 0)
    dv_y = dde.grad.jacobian(Y, X, i = 1, j = 1)
    dp_x = dde.grad.jacobian(Y, X, i = 2, j = 0)
    dp_y = dde.grad.jacobian(Y, X, i = 2, j = 1)

    du_xx = dde.grad.hessian(Y, X, component = 0, i = 0, j = 0)
    du_yy = dde.grad.hessian(Y, X, component = 0, i = 1, j = 1)
    dv_xx = dde.grad.hessian(Y, X, component = 1, i = 0, j = 0)
    dv_yy = dde.grad.hessian(Y, X, component = 1, i = 1, j = 1)

    # x-momentum Navier Stokes equation
    pde_u = (Y[:, 0:1] * du_x + Y[:, 1:2] * du_y + dp_x - (1 / Re) * (du_xx + du_yy))
    
    # y-momentum Navier Stokes equation
    pde_v = (Y[:, 0:1] * dv_x + Y[:, 1:2] * dv_y + dp_y - (1 / Re) * (dv_xx + dv_yy))
    
    # Continuity equation
    pde_cont = du_x + dv_y
    return [pde_u, pde_v, pde_cont]


## Define Training Session ##

# Create Data to compute the Losses, Set Domain Points, Boundary Points and Number of Test Points
data = dde.data.PDE(geom,
                    pde,
                    [bc_bottom_wall_u, bc_bottom_wall_v, bc_top_wall_u, bc_top_wall_v, bc_left_wall_u, bc_left_wall_v, bc_right_wall_u, bc_right_wall_v],
                    num_domain = 20_000,
                    num_boundary = 2_500,
                    num_test = 4_000) 


# Visualize Training Data Points - Samples
plt.figure(figsize = (10, 8))
plt.scatter(data.train_x_all[:, 0], data.train_x_all[:, 1], s = 0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Training Data Points - Samples")
plt.show()


# Define the architecture of the PINN
net = dde.maps.FNN([2] + [64] * 5 + [3], "tanh", "Glorot uniform")

# Create the model of the PINN
model = dde.Model(data, net)

# Compile with ADAM
model.compile("adam", lr = 1e-3)

# Set epochs for ADAM optimization
losshistory, train_state = model.train(epochs = 10_000, display_every=100)

# Optimize with L-BFGS    
dde.optimizers.config.set_LBFGS_options(maxiter = 3_000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave = False, isplot = False)


## Prediction ##

# Create Prediction Points
samples = geom.random_points(500_000)

# Model Prediction at the Prediction Points
result = model.predict(samples)

# Visualize the predicted u-component of the velocity
plt.figure(figsize = (10, 10))
u_scatter = plt.scatter(samples[:, 0],
            samples[:, 1],
            c = result[:, 0],
            cmap = "jet",
            s = 2)
plt.colorbar(u_scatter, label = "u")
plt.clim([min(result[:, 0]), max(result[:, 0])])
plt.xlim((0, L))
plt.ylim((0, D))
plt.tight_layout()
plt.title(f"u component of the velocity, Re = {Re}")

# Visualize the predicted v-component of the velocity
plt.figure(figsize = (10, 10))
v_scatter = plt.scatter(samples[:, 0],
            samples[:, 1],
            c = result[:, 1],
            cmap = "jet",
            s = 2)
plt.colorbar(v_scatter, label = 'v')
plt.clim([min(result[:, 1]), max(result[:, 1])])
plt.xlim((0, L))
plt.ylim((0, D))
plt.tight_layout()
plt.title(f"v component of the velocity, Re = {Re}")

# Visualize the predicted Pressure
plt.figure(figsize = (10, 10))
p_scatter = plt.scatter(samples[:, 0],
            samples[:, 1],
            c = result[:, 2],
            cmap = "jet",
            s = 2)
plt.colorbar(p_scatter, label = 'Pressure')
plt.clim([min(result[:, 2]), max(result[:, 2])])
plt.xlim((0, L))
plt.ylim((0, D))
plt.tight_layout()
plt.title(f"Pressure, Re = {Re}")

# Visualize the predicted Velocity Magnitude
velocity_magnitude =np.sqrt(result[:, 0]**2 + result[:, 1]**2)
plt.figure(figsize = (10, 10))
vel_mag_scatter = plt.scatter(samples[:, 0],
            samples[:, 1],
            c = velocity_magnitude,
            cmap = "jet",
            s = 2)
plt.colorbar(vel_mag_scatter, label = 'velocity magnitude')
plt.clim([min(velocity_magnitude), max(velocity_magnitude)])
plt.xlim((0, L))
plt.ylim((0, D))
plt.tight_layout()
plt.title(f"Velocity Magnitude, Re = {Re}")


# Generate grid for streamlines
x_min, x_max = 0, L
y_min, y_max = 0, D
x_stream, y_stream = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Interpolate the velocity field on the streamlines grid
result_stream = model.predict(np.hstack([x_stream.reshape(-1, 1), y_stream.reshape(-1, 1)]))

# Reshape the results to match the grid shape
u_stream = result_stream[:, 0].reshape(x_stream.shape)
v_stream = result_stream[:, 1].reshape(y_stream.shape)

# Plot streamlines on top of the velocity field
plt.figure(figsize=(10, 10))
vel_mag_scatter = plt.scatter(samples[:, 0], samples[:, 1], c=velocity_magnitude, cmap="jet", s=2)
plt.colorbar(vel_mag_scatter, label = 'velocity magnitude')
plt.clim([min(velocity_magnitude), max(velocity_magnitude)])
plt.xlim((0, L))
plt.ylim((0, D))
plt.title(f"Velocity Magnitude with Streamlines, Re = {Re}")
plt.streamplot(x_stream, y_stream, u_stream, v_stream, density=5, color='black', linewidth=.5, arrowsize=0.5, arrowstyle='-')
plt.show()
