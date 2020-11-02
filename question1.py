import numpy as np
from scipy.integrate import solve_ivp # solve IVPs (initial value problems)
# IVP = diff. equation [z' = f(z)] + initial condition [z(0) = z subscript 0]
import matplotlib.pyplot as plt # used to plot graphs in Python

v = 5
L = 2.3
u = 2 * np.pi / 180  # require calculation to convert from degrees to radians

# x: x-coordinate of the car,     v: velocity
# y: y-coordinate of the car,     θ: orientation of the car
# u: steering wheel angle,        L: axle-to-axle length


def system_dynamics(t, z):
    # expects two arguments, i.e. f(t, z) where t is time
    """
    This function defines the dynamical equations of a
    car, as follows:

    z' = f(z)

    where z = [x, y, θ] and f is given by:

    f(z) = [ v * cos(θ)    ]
           [ v * sin(θ)    ]
           [ v * tan(u) / L]

    :param z: z = [x, y, θ]
    :return: f(z)
    """
    theta = z[2]
    return [v * np.cos(theta),  # dynamical model from equation 2.1 on assignment
            v * np.sin(theta),
            v * np.tan(u)/L]


num_points = 100
t_final = 2
z_initial_condition = [0, 30, 5 * np.pi / 8]
solution = solve_ivp(system_dynamics,
                     [0, t_final],
                     z_initial_condition,
                     t_eval=np.linspace(0, t_final, num_points))

x_solution = solution.y[0]  # stores x values within a single array
y_solution = solution.y[1]
theta_solution = solution.y[2]
t_points = solution.t

plt.plot(t_points, y_solution)
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Lateral position, y (cm)")
# initially as seen however for the other graphs, i.e. x(t) and theta(t), this was changed accordingly
plt.show()
