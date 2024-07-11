import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

rho = 1
mu = 0.01
u_in = 1
D = 10
L = 30
cylinder_radius = 0.5  # Radius of the cylinder

# Reynolds Number
Re = (rho * u_in * 2 * cylinder_radius) / mu

print(f"Re = {Re}")

# Define the wind tunnel geometry
wind_tunnel = dde.geometry.Rectangle(xmin=[-L/4, -D/2], xmax=[L/2, D/2])

# Define the cylinder geometry
cylinder = dde.geometry.Disk(center=(0, 0), radius=cylinder_radius)

# Subtract the cylinder from the wind tunnel
geom = dde.geometry.CSGDifference(wind_tunnel, cylinder)

def boundary_wall(X, on_boundary):
    on_wall = np.logical_and(np.logical_or(np.isclose(X[1], -D/2, rtol=1e-5, atol=1e-8),
                                           np.isclose(X[1], D/2, rtol=1e-5, atol=1e-8)), on_boundary)
    return on_wall

def boundary_inlet(X, on_boundary):
    on_inlet = np.logical_and(np.isclose(X[0], -L/4, rtol=1e-5, atol=1e-8), on_boundary)
    return on_inlet

def boundary_outlet(X, on_boundary):
    on_outlet = np.logical_and(np.isclose(X[0], L/2, rtol=1e-5, atol=1e-8), on_boundary)
    return on_outlet

def boundary_cylinder(X, on_boundary):
    on_cylinder = np.linalg.norm(X - np.array([0, 0]), axis=0) <= cylinder_radius
    return np.logical_and(on_cylinder, on_boundary)

bc_wall_u = dde.DirichletBC(geom, lambda X: u_in, boundary_wall, component=0)
bc_wall_v = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component=1)

bc_inlet_u = dde.DirichletBC(geom, lambda X: u_in, boundary_inlet, component=0)
bc_inlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_inlet, component=1)

bc_outlet_p = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component=2)

bc_cylinder_u = dde.DirichletBC(geom, lambda X: 0., boundary_cylinder, component=0)
bc_cylinder_v = dde.DirichletBC(geom, lambda X: 0., boundary_cylinder, component=1)

def pde(X, Y):
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

    pde_u = (Y[:, 0:1] * du_x + Y[:, 1:2] * du_y + dp_x - (1 / Re) * (du_xx + du_yy))
    pde_v = (Y[:, 0:1] * dv_x + Y[:, 1:2] * dv_y + dp_y - (1 / Re) * (dv_xx + dv_yy))
    pde_cont = du_x + dv_y
    
    return [pde_u, pde_v, pde_cont]

# Replace the creation of PDE data with this code
num_domain = 50_000
num_boundary = 10_000
num_test = 10_000

data = dde.data.PDE(geom, pde, [bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p, bc_cylinder_u, bc_cylinder_v],
                    num_domain=num_domain, num_boundary=num_boundary, num_test=num_test)

# Visualization of the training points
plt.figure(figsize=(10, 4))
plt.scatter(data.train_x_all[:, 0], data.train_x_all[:, 1], s=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

net = dde.maps.FNN([2] + [65] * 10 + [3], "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

losshistory, train_state = model.train(epochs=10_000)

dde.optimizers.config.set_LBFGS_options(maxiter = 3_000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave = False, isplot = False)

samples = geom.random_points(500_000)
result = model.predict(samples)

# Visualize the predicted u-component of the velocity
plt.figure(figsize = (10, 4))
u_scatter = plt.scatter(samples[:, 0],
            samples[:, 1],
            c = result[:, 0],
            cmap = "jet",
            s = 2)
plt.colorbar(u_scatter, label = "u")
plt.clim([min(result[:, 0]), max(result[:, 0])])
plt.xlim((-L/4, L/2))
plt.ylim((-D/2, D/2))
plt.tight_layout()
plt.title(f"u component of the velocity, Re = {Re}")

# Visualize the predicted v-component of the velocity
plt.figure(figsize = (10, 4))
v_scatter = plt.scatter(samples[:, 0],
            samples[:, 1],
            c = result[:, 1],
            cmap = "jet",
            s = 2)
plt.colorbar(v_scatter, label = 'v')
plt.clim([min(result[:, 1]), max(result[:, 1])])
plt.xlim((-L/4, L/2))
plt.ylim((-D/2, D/2))
plt.tight_layout()
plt.title(f"v component of the velocity, Re = {Re}")

# Visualize the predicted Pressure
plt.figure(figsize = (10, 4))
p_scatter = plt.scatter(samples[:, 0],
            samples[:, 1],
            c = result[:, 2],
            cmap = "jet",
            s = 2)
plt.colorbar(p_scatter, label = 'Pressure')
plt.clim([min(result[:, 2]), max(result[:, 2])])
plt.xlim((-L/4, L/2))
plt.ylim((-D/2, D/2))
plt.tight_layout()
plt.title(f"Pressure, Re = {Re}")

# Visualize the predicted Velocity Magnitude
velocity_magnitude =np.sqrt(result[:, 0]**2 + result[:, 1]**2)
plt.figure(figsize = (10, 4))
vel_mag_scatter = plt.scatter(samples[:, 0],
            samples[:, 1],
            c = velocity_magnitude,
            cmap = "jet",
            s = 2)
plt.colorbar(vel_mag_scatter, label = 'velocity magnitude')
plt.clim([min(velocity_magnitude), max(velocity_magnitude)])
plt.xlim((-L/4, L/2))
plt.ylim((-D/2, D/2))
plt.tight_layout()
plt.title(f"Velocity Magnitude, Re = {Re}")


# Generate grid for streamlines
x_min, x_max = -L/4, L/2
y_min, y_max = -D/2, D/2
x_stream, y_stream = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Interpolate the velocity field on the streamlines grid
result_stream = model.predict(np.hstack([x_stream.reshape(-1, 1), y_stream.reshape(-1, 1)]))

# Reshape the results to match the grid shape
u_stream = result_stream[:, 0].reshape(x_stream.shape)
v_stream = result_stream[:, 1].reshape(y_stream.shape)

# Set velocity components inside the cylinder region to zero
cylinder_mask = (x_stream - 0.0)**2 + (y_stream - 0.0)**2 <= cylinder_radius**2
u_stream[cylinder_mask] = 0
v_stream[cylinder_mask] = 0

# Plot streamlines on top of the velocity field
plt.figure(figsize=(10, 4))
vel_mag_scatter = plt.scatter(samples[:, 0], samples[:, 1], c=velocity_magnitude, cmap="jet", s=2)
plt.colorbar(vel_mag_scatter, label='velocity magnitude')
plt.clim([min(velocity_magnitude), max(velocity_magnitude)])
plt.xlim((-L/4, L/2))
plt.ylim((-D/2, D/2))
plt.title(f"Velocity Magnitude with Streamlines, Re = {Re}")
plt.streamplot(x_stream, y_stream, u_stream, v_stream, density=5, color='black', linewidth=.25, arrowsize=0.5, arrowstyle='-')
plt.show()