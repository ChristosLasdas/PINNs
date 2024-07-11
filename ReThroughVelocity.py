import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

D = 0.25
L = 1
rho = 1e3
mu = 1e-2

# Define the range of inlet velocities
inlet_velocities = [0.5, 0.008, 0.001, 0.1, 0.05]

for velocity in inlet_velocities:
  print(f'Re = {rho*velocity*D/mu}')

# Increase the number of points on each xy plane
num_points_per_plane = 1500

# Generate training data on xy planes for each inlet velocity
train_data_interior = []
train_data_boundary = []

for u_in in inlet_velocities:
    # Generate random points in the xy-plane for interior
    points_xy_plane_interior = np.random.rand(num_points_per_plane, 2)
    points_xy_plane_interior[:, 0] = points_xy_plane_interior[:, 0] * L
    points_xy_plane_interior[:, 1] = points_xy_plane_interior[:, 1] * D
    points_u_in = np.full((num_points_per_plane, 1), u_in)
    train_data_interior.append(np.hstack((points_xy_plane_interior, points_u_in)))

    # Generate points on the boundaries
    points_boundary_top_wall = np.array([[x, D, u_in] for x in np.linspace(0, L, num_points_per_plane)])
    points_boundary_bottom_wall = np.array([[x, 0, u_in] for x in np.linspace(0, L, num_points_per_plane)])
    points_boundary_inlet = np.array([[0, y, u_in] for y in np.linspace(0, D, num_points_per_plane)])
    points_boundary_outlet = np.array([[L, y, u_in] for y in np.linspace(0, D, num_points_per_plane)])

    train_data_boundary.extend([points_boundary_top_wall, points_boundary_bottom_wall, points_boundary_inlet, points_boundary_outlet])

train_data_interior = np.vstack(train_data_interior)
train_data_boundary = np.vstack(train_data_boundary)

# Use train_data_interior for interior points and train_data_boundary for boundary points
geom = dde.geometry.PointCloud(points=train_data_interior, boundary_points=train_data_boundary)

def boundary_wall(X, on_boundary):
    on_wall = np.logical_and(np.logical_or(np.isclose(X[1], 0, rtol=1e-5, atol=1e-8),
                                           np.isclose(X[1], D, rtol=1e-5, atol=1e-8)), on_boundary)
    return on_wall

def boundary_inlet(X, on_boundary):
    on_inlet = np.logical_and(np.isclose(X[0], 0, rtol=1e-5, atol=1e-8), on_boundary)
    return on_inlet

def boundary_outlet(X, on_boundary):
    on_outlet = np.logical_and(np.isclose(X[0], L, rtol=1e-5, atol=1e-8), on_boundary)
    return on_outlet

bc_wall_u = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component=0)
bc_wall_v = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component=1)

bc_inlet_u = dde.DirichletBC(geom, lambda X: X[:, 2:3], boundary_inlet, component=0)
bc_inlet_v = dde.DirichletBC(geom, lambda X: 0, boundary_inlet, component=1)

bc_outlet_p = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component=2)
bc_outlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component=1)

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

    pde_u = Y[:, 0:1] * du_x + Y[:, 1:2] * du_y + dp_x - (mu / (rho*X[:, 2:3]*D)) * (du_xx + du_yy)
    pde_v = Y[:, 0:1] * dv_x + Y[:, 1:2] * dv_y + dp_y - (mu / (rho*X[:, 2:3]*D)) * (dv_xx + dv_yy)
    pde_cont = du_x + dv_y

    return [pde_u, pde_v, pde_cont]

# Number of domain points for prediction
num_domain_predict = len(inlet_velocities) * num_points_per_plane

data = dde.data.PDE(geom,
                    pde,
                    [bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p, bc_outlet_v],
                    num_domain=num_domain_predict,
                    num_boundary=num_points_per_plane,
                    num_test=2500)

net = dde.maps.FNN([3] + [80] * 8 + [3], "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

losshistory, train_state = model.train(epochs=15000)

dde.optimizers.config.set_LBFGS_options(maxiter=3000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# Generate prediction samples on xy plane for the desired inlet velocities
inlet_velocities_to_predict = [0.008, .5]
for v in inlet_velocities_to_predict:
  print(f'Re = {rho*velocity*D/mu}')
num_predict_points = 500000
predict_samples_interior = np.random.rand(num_predict_points, 3)
predict_samples_interior[:, 0] = predict_samples_interior[:, 0] * L
predict_samples_interior[:, 1] = predict_samples_interior[:, 1] * D
predict_samples_interior[:, 2] = np.random.choice(inlet_velocities_to_predict, num_predict_points)  # Set inlet velocity

# Predict at the specified points
predict_result_interior = model.predict(predict_samples_interior)

# Compute velocity magnitude
velocity_magnitude = np.sqrt(np.sum(predict_result_interior[:, :2]**2, axis=1))

# Plot the predicted velocity magnitude on the XY plane for the desired inlet velocities
for u_in in inlet_velocities_to_predict:
    plt.figure(figsize=(20, 4))
    plt.scatter(predict_samples_interior[predict_samples_interior[:, 2] == u_in][:, 0],
                predict_samples_interior[predict_samples_interior[:, 2] == u_in][:, 1],
                c=velocity_magnitude[predict_samples_interior[:, 2] == u_in],
                cmap="jet", s=2)
    plt.colorbar(label='Velocity Magnitude')
    plt.xlim((0, L))
    plt.ylim((0, D))
    plt.title(f'Predicted Velocity Magnitude on XY Plane for Inlet Velocity {u_in}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.tight_layout()
    plt.show()
