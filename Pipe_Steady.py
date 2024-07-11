import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Material Properties
rho = 1e3
mu = 1e-1

# Dimensions of Wind Tunnel
D = 0.25
L = 1

# Reynolds Number
Re = 5000

# Compute inlet velocity
u_in = (Re * mu) / (rho * D)

print(f"u inlet = {u_in}")


# Create Wind Tunnel geometry
geom = dde.geometry.Rectangle(xmin = [0, 0], xmax = [L, D])


## Define Boundaries ##

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


## Define Boundary Conditions ##

# No-slip Top and Bottom Walls
bc_wall_u = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component = 0)
bc_wall_v = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component = 1)

# Velocity Inlet (u-component = u_in, v-component = 0)
bc_inlet_u = dde.DirichletBC(geom, lambda X: u_in, boundary_inlet, component = 0)
bc_inlet_v = dde.DirichletBC(geom, lambda X: 0, boundary_inlet, component = 1)

# Pressure Outlet
bc_outlet_p = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component = 2)
bc_outlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component = 1)


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
                    [bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p, bc_outlet_v],
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
plt.figure(figsize = (20, 4))
u_scatter = plt.scatter(samples[:, 0],
            samples[:, 1],
            c = result[:, 0],
            cmap = "jet",
            s = 2)
plt.colorbar(u_scatter, label = 'u')
plt.clim([min(result[:, 0]), max(result[:, 0])])
plt.xlim((0, L))
plt.ylim((0, D))
plt.tight_layout()
plt.title(f"u component of the velocity, Re = {Re}")

# Visualize the predicted v-component of the velocity
plt.figure(figsize = (20, 4))
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
plt.figure(figsize = (20, 4))
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
plt.figure(figsize = (20, 4))
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

plt.show()

## Prediction ##

# Plotting the velocity profile at a specific x-coordinate
x_value = 0.95
num_points = 200
y_values = np.linspace(0, D, num_points)
x_values = np.full_like(y_values, x_value)
line_points = np.column_stack((x_values, y_values))
velocities = model.predict(line_points)
u_profile = velocities[:, 0]
v_profile = velocities[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(u_profile, y_values, label='u velocity')
plt.xlabel('Velocity')
plt.ylabel('Distance (y)')
plt.title(f'Velocity Profile at x = {x_value}')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(v_profile, y_values, label='v velocity')
plt.xlabel('Velocity')
plt.ylabel('Distance (y)')
plt.title(f'Velocity Profile at x = {x_value}')
plt.legend()
plt.grid(True)

# Comparing the velocity profile with analytical solution (Poiseuille flow)
if Re < 4000:
    G_poiseuille = 8 * max(velocity_magnitude) * mu / D ** 2
    def poiseuille_velocity_profile(y):
        return G_poiseuille / (2 * mu) * y * (D - y)
    y_values_analytical = np.linspace(0, D, num_points)
    u_profile_analytical = poiseuille_velocity_profile(y_values_analytical)
    plt.figure(figsize=(8, 6))
    plt.plot(u_profile, y_values, label="Model Prediction")
    plt.plot(u_profile_analytical, y_values_analytical, "--", label="Poiseuille Flow (Analytical)")
    plt.xlabel("Velocity")
    plt.ylabel("Distance (y)")
    plt.title(f"Comparison of Velocity Profiles at x = {x_value}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"********** Poiseuille Velocity u Profile Values")
    print(u_profile_analytical)

    l2_difference_u = dde.metrics.l2_relative_error(u_profile_analytical, u_profile)
    l2_difference_v = dde.metrics.l2_relative_error(np.zeros_like(velocities[:, 2]), velocities[:, 1])
    residual = np.mean(np.absolute(samples))
    print(f"********** Mean residual: {residual} **********")
    print(f"********** L2 relative error in u: {l2_difference_u} **********")
    print(f"********** L2 relative error in v: {l2_difference_v} **********")


else:
    R = D / 2
    delta_P = 8 * max(velocity_magnitude) * mu * L / D**2
    print(f"Pressure Drop = {delta_P}")
    prandtl_von_karman_velocity_profile = max(velocity_magnitude) * (1 - ((y_values-R) / R)**2)**(1/7)
    plt.figure(figsize=(8, 6))
    plt.plot(u_profile, y_values, label="Model Prediction")
    plt.plot(prandtl_von_karman_velocity_profile, y_values, "--", label="Prandtl-von Kármán")
    plt.xlabel("Velocity")
    plt.ylabel("Distance (y)")
    plt.title(f"Comparison of Velocity Profiles at x = {x_value}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"********** Prandtl - von Kármán Velocity u Profile Values")
    print(prandtl_von_karman_velocity_profile)

    l2_difference_u = dde.metrics.l2_relative_error(prandtl_von_karman_velocity_profile, u_profile)
    l2_difference_v = dde.metrics.l2_relative_error(np.zeros_like(velocities[:, 2]), velocities[:, 1])
    residual = np.mean(np.absolute(samples))

    print(f"********** Mean residual: {residual} **********")
    print(f"********** L2 relative error in u: {l2_difference_u} **********")
    print(f"********** L2 relative error in v: {l2_difference_v} **********")


# Specify the desired y-coordinate for the centerline
y_centerline = 0.125

# Generate points along the centerline at the specified y-value
num_points_centerline = 200  # Number of points along the line
x_values_centerline = np.linspace(0, L, num_points_centerline)

# Generate points along the centerline at the specified y-value and inlet velocity
centerline_points = np.column_stack((x_values_centerline, np.full_like(x_values_centerline, y_centerline)))

# Predict pressure at the centerline points
pressure = model.predict(centerline_points)[:, 2]

# Plot the pressure distribution along the centerline
plt.figure(figsize=(8, 6))
plt.plot(x_values_centerline, pressure, label=f"Inlet Velocity = {u_in}")
plt.xlabel("Distance (x)")
plt.ylabel("Pressure")
plt.title(f"Pressure Distribution along Centerline at y = {y_centerline}")
plt.legend()
plt.grid(True)
plt.show()

print(f"Max Value of Velocity Magnitude = {max(velocity_magnitude)}")

print("********** Velocity u Values **********")
for vel in u_profile:
  print(vel)

print(f"********** Velocity v Values **********")
for v in v_profile:
  print(v)

print("********** Pressure Values **********")
for p in pressure:
  print(p)





