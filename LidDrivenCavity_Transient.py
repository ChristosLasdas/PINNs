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

t0 = 0.
t_final = 25

print(f"Simulation up until t = {t_final} s")

# Create Wind Tunnel geometry
geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[L, D])
timedomain = dde.geometry.TimeDomain(t0, t_final)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

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

## Define Boundary Condition Functions##

def u_bottom_wall(x):
  return 0.

def v_bottom_wall(x):
  return 0.

def u_left_wall(x):
  return 0.

def v_left_wall(x):
  return 0. 

def u_right_wall(x):
  return 0.

def v_right_wall(x):
  return 0. 

def u_top_wall(x):
  return u_in

def v_top_wall(x):
  return 0.

def u_init(x):
  return 0.

def v_init(x):
  return 0.

def p_init(x):
  return 0.


# No-slip Top and Bottom Walls
bc_top_wall_u = dde.DirichletBC(
    geomtime, u_top_wall, top_wall, component=0
)
bc_top_wall_v = dde.DirichletBC(
    geomtime, v_top_wall, top_wall, component=1
)

bc_bottom_wall_u = dde.DirichletBC(
    geomtime, u_bottom_wall, bottom_wall, component=0
)
bc_bottom_wall_v = dde.DirichletBC(
    geomtime, v_bottom_wall, bottom_wall, component=1
)

bc_right_wall_u = dde.DirichletBC(
    geomtime, u_right_wall, right_wall, component=0
)
bc_right_wall_v = dde.DirichletBC(
    geomtime, v_right_wall, right_wall, component=1
)

bc_left_wall_u = dde.DirichletBC(
    geomtime, u_left_wall, left_wall, component=0
)
bc_left_wall_v = dde.DirichletBC(
    geomtime, v_left_wall, left_wall, component=1
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
    [bc_top_wall_u, bc_top_wall_v,
     bc_bottom_wall_u, bc_bottom_wall_v,
     bc_left_wall_u, bc_left_wall_v,
     bc_right_wall_u, bc_right_wall_v,
     ic_u, ic_v, ic_p],
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
time_instances = np.arange(t0, tf, step)

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
    plt.figure(figsize=(10, 10))
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

# Set the value of x where you want to plot the velocity profile
x_value = L / 2  # Assuming L is the length of the cavity

# Generate points along a vertical line at the specified x-value
num_points = 100  # Number of points along the line
y_values = np.linspace(0, D, num_points)
x_values = np.full_like(y_values, x_value)
line_points = np.column_stack((x_values, y_values))

# Define a constant time value for all points
time_value = 2.5*t_final  # Example: Use the mid-point of the simulation time

# Append the time component to the line points
line_points_with_time = np.column_stack((line_points, np.full_like(y_values, time_value)))

# Use the trained model to predict velocity components at the updated points
velocities = model.predict(line_points_with_time)

# Extract u component of velocity
u_profile = velocities[:, 0]

# Define theoretical data
theoretical_data = [
    [1.0, 1.0],
    [0.84123, 0.9766],
    [0.78871, 0.9688],
    [0.73722, 0.9609],
    [0.68717, 0.9531],
    [0.23151, 0.8516],
    [0.00332, 0.7344],
    [-0.13641, 0.6172],
    [-0.20581, 0.5],
    [-0.2109, 0.4531],
    [-0.15662, 0.2813],
    [-0.1015, 0.1719],
    [-0.06434, 0.1016],
    [-0.04775, 0.0703],
    [-0.04192, 0.0625],
    [-0.03717, 0.0547],
    [0.0, 0.0]
]

# Convert theoretical data to numpy array for easy manipulation
theoretical_data = np.array(theoretical_data)

# Extract theoretical u velocity profile
u_theoretical = theoretical_data[:, 0]
y_theoretical = theoretical_data[:, 1]

# Plot the velocity profiles
plt.figure(figsize=(8, 6))

# Plot predicted u velocity profile
plt.plot(u_profile, y_values, label='Predicted', color='blue')

# Plot theoretical u velocity profile
plt.plot(u_theoretical, y_theoretical, label='Theoretical', linestyle='--', color='red')

plt.xlabel('Velocity (u)')
plt.ylabel('Distance (y)')
plt.title(f'u Velocity Profile at x = {x_value} and t = {time_value}')
plt.legend()
plt.grid(True)
plt.show()

# Define the theoretical value of u at y=0.5
u_theoretical_y05 = -0.20581

# Calculate the relative error at y=0.5 for each time instance
relative_errors_y05 = []
time_instances = np.arange(t0, 3*t_final+1, 1)
for t in time_instances:
    # Evaluate the model at the given time instance
    T = np.ones((1, 1)) * t
    XYT = np.array([[x_value, 0.5, t]])
    result = model.predict(XYT)

    # Extract the predicted u velocity at y=0.5
    u_predicted_y05 = result[0, 0]

    # Calculate the relative error
    relative_error = abs(u_predicted_y05 - u_theoretical_y05) / abs(u_theoretical_y05)
    relative_errors_y05.append(relative_error)

# Plot the relative error over time
plt.figure(figsize=(8, 6))
plt.plot(time_instances, relative_errors_y05, marker='o', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Relative Error at y=0.5')
plt.title('Relative Error of Predicted u Velocity at y=0.5')
plt.grid(True)
plt.show()


def visualize_predictions_with_abs_error(model, samples, time_instances, theoretical_data):
    # Define the point at y=0.00332
    y_target = 0.5
    u_theoretical_target = -0.20581

    # Initialize lists to store absolute errors
    absolute_errors = []
    time_instances = np.arange(t0, t_final+30, 2)
    for t in time_instances:
        # Evaluate the model at the given time instance
        T = np.ones((samples.shape[0], 1)) * t
        XYT = np.hstack((samples, T))
        result = model.predict(XYT)

        # Find the index of the closest y value to the target y
        idx_closest_y = np.argmin(np.abs(samples[:, 1] - y_target))

        # Get the predicted u value at the closest y point
        u_predicted = result[idx_closest_y, 0]

        # Compute absolute error
        absolute_error = np.abs(u_predicted - u_theoretical_target)
        absolute_errors.append(absolute_error)

    # Plot the absolute errors over time
    plt.figure(figsize=(8, 6))
    plt.plot(time_instances, absolute_errors, marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error of Predicted u at y=0.00332 over Time')
    plt.grid(True)
    plt.show()

# Visualize absolute errors over time
visualize_predictions_with_abs_error(model, samples, time_instances, theoretical_data)

def visualize_predictions_with_error(model, samples, time_instances, theoretical_data):
    # Define the point at y=0.00332
    y_target = 0.5
    u_theoretical_target = -0.20581

    # Initialize lists to store errors
    errors = []

    for t in time_instances:
        # Evaluate the model at the given time instance
        T = np.ones((samples.shape[0], 1)) * t
        XYT = np.hstack((samples, T))
        result = model.predict(XYT)

        # Find the index of the closest y value to the target y
        idx_closest_y = np.argmin(np.abs(samples[:, 1] - y_target))

        # Get the predicted u value at the closest y point
        u_predicted = result[idx_closest_y, 0]

        # Compute error
        error = u_predicted - u_theoretical_target
        errors.append(error)

    # Plot the errors over time
    plt.figure(figsize=(8, 6))
    plt.plot(time_instances, errors, marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title('Error of Predicted u at y=0.00332 over Time')
    plt.grid(True)
    plt.show()

# Visualize errors over time
visualize_predictions_with_error(model, samples, time_instances, theoretical_data)
