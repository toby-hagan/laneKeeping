import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
# this wass used to plot multiple results on one graph


class Car:  # creates a Car class so that any car can be created easily using constructor

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

    def control(self, y, y_set_point=0):
        """
        This is the proportional controller that causes the vehicle to move towards y=0
        :param y:
        :param y_set_point:
        :return:
        """
        error = y_set_point - y
        u = self.kp * error   # proportional control action
        return u


sampling_time = 0.025  # sampling rate = 40Hz so 1/40
dwayne = Car(theta=0+1 * np.pi / 180)
controller = PIDController(kp=0.1, ts=sampling_time)

num_points = 2000
y1_black_box = np.array([dwayne.y])
x_black_box = np.array([dwayne.x])

for t in range(num_points):
    steering = controller.control(y=dwayne.y)
    dwayne.move(steering, sampling_time)
    # murphy.y <---- not cached / not stored (need for y_black_box)
    y1_black_box = np.append(y1_black_box, dwayne.y)
    x_black_box = np.append(x_black_box, dwayne.x)

dwayne = Car(theta=0+1 * np.pi / 180, y=0, x=0)  # resets the car position and values for consistency
controller = PIDController(kp=0.3, ts=sampling_time)  # updates the value of Kp
y2_black_box = np.array([dwayne.y])

for t in range(num_points):
    steering = controller.control(y=dwayne.y)
    dwayne.move(steering, sampling_time)
    y2_black_box = np.append(y2_black_box, dwayne.y)

dwayne = Car(theta=0+1 * np.pi / 180, y=0, x=0)
controller = PIDController(kp=0.5, ts=sampling_time)
y3_black_box = np.array([dwayne.y])

for t in range(num_points):
    steering = controller.control(y=dwayne.y)
    dwayne.move(steering, sampling_time)
    y3_black_box = np.append(y3_black_box, dwayne.y)
# for loops ran three times for three different results
t_span = sampling_time * np.arange(num_points + 1)


df = pd.DataFrame({'x': x_black_box, 'y1': y1_black_box, 'y2': y2_black_box, 'y3': y3_black_box })
# stores the results in variables for the graph plot

plt.plot('x', 'y1', data=df, marker='', color='blue', linewidth=2, label='Kp = 0.1')
plt.plot('x', 'y2', data=df, marker='', color='red', linewidth=2, label='Kp = 0.5')
plt.plot('x', 'y3', data=df, marker='', color='grey', linewidth=2, label='Kp = 1')
plt.legend()
plt.grid()
plt.xlabel('Lateral position, x (m)')
plt.ylabel('Lateral position, y (m)')
plt.show()
