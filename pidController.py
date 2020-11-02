import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Car:

    def __init__(self, length=2.3, velocity=5, x=0, y=0, theta=0):
        self.length = length
        self.velocity = velocity
        self.x = x
        self.y = y
        self.theta = theta

    def move(self, steering_angle_rad, dt):
        """
        This function computes and updates the new position and
        orientation of the car if we apply the given steering action
        for a time "dt"
        :param steering_angle_rad:
        :param dt:
        :return:
        """
        def system_dynamics(t, z):
            theta = z[2]
            return [self.velocity * np.cos(theta),
                    self.velocity * np.sin(theta),
                    self.velocity * np.tan(steering_angle_rad) / self.length]

        # next we need to solve this IVP:
        z_initial = [self.x, self.y, self.theta]
        solution = solve_ivp(system_dynamics, [0, dt], z_initial)

        self.x = solution.y[0][-1]
        self.y = solution.y[1][-1]
        self.theta = solution.y[2][-1]


class PIDController:

    def __init__(self, kp=0.1, ki=0, kd=0, ts=0.01):
        self.kp = kp
        self.ki = ki * ts
        self.kd = kd / ts
        self.ts = ts
        self.error_previous = None
        self.sum_errors = 0  # int_0^0 e(τ) dτ = 0

    def control(self, y, y_set_point=0):
        # u = kp * e + Kd * (e - e_previous)  <-- PD-Controller
        error = y_set_point - y
        u = self.kp * error   # proportional control action

        if self.error_previous is not None:
            u += self.kd * (error - self.error_previous)

        u += self.ki * self.sum_errors
        self.error_previous = error
        self.sum_errors += error
        return u


sampling_time = 0.025
dwayne = Car(y=0.5)
controller = PIDController(kp=0.2, ki=0.05, kd=0.25, ts=sampling_time)

num_points = 2000
u_disturbance = 3 * np.pi / 180  # 3 degrees (in radians)
y_black_box = np.array([dwayne.y])
x_black_box = np.array([dwayne.x])
steering = None
u_black_box = np.array(steering)
for t in range(num_points):
    steering = controller.control(y=dwayne.y)
    dwayne.move(steering + u_disturbance, sampling_time)
    # murphy.y <---- not cached / not stored (need for y_black_box)
    y_black_box = np.append(y_black_box, dwayne.y)
    x_black_box = np.append(x_black_box, dwayne.x)
    u_black_box = np.append(u_black_box, steering)

print(u_black_box)
t_span = sampling_time * np.arange(num_points + 1)
plt.plot(x_black_box, y_black_box)
plt.grid()
plt.xlabel('Lateral position, x (m)')
plt.ylabel('Lateral position, y (m)')
plt.show()
