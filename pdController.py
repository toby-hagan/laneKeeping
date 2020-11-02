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


class PIDController:  # really a PD controller

    def __init__(self, kp=0.1, ki=0, kd=0, ts=0.01):
        self.kp = kp
        self.ki = ki * ts
        self.kd = kd / ts
        self.ts = ts
        self.error_previous = 1 * np.pi / 180

    def control(self, y, y_set_point=0):
        error = y_set_point - y
        u = self.kp * error   # proportional control action

        if self.error_previous is not None:
            u += self.kd * (error - self.error_previous)

        self.error_previous = error
        return u


u_disturbance = np.pi / 180  # the steering wheel error outlined in question
sampling_time = 0.025
dwayne = Car(y=0.5)
controller = PIDController(kp=0.2, kd=0.1, ts=sampling_time)

num_points = 2000
y_black_box = np.array([dwayne.y])
x_black_box = np.array([dwayne.x])
for t in range(num_points):
    steering = controller.control(y=dwayne.y)
    dwayne.move(steering+ u_disturbance, sampling_time)
    y_black_box = np.append(y_black_box, dwayne.y)
    x_black_box = np.append(x_black_box, dwayne.x)

plt.plot(x_black_box, y_black_box, label='Kd=0.1')

dwayne = Car(y=0.5)
controller = PIDController(kp=0.2, kd=0.2, ts=sampling_time)
y_black_box = np.array([dwayne.y])
x_black_box = np.array([dwayne.x])
for t in range(num_points):
    steering = controller.control(y=dwayne.y)
    dwayne.move(steering+ u_disturbance, sampling_time)
    y_black_box = np.append(y_black_box, dwayne.y)
    x_black_box = np.append(x_black_box, dwayne.x)

plt.plot(x_black_box, y_black_box, label='Kd=0.2')

dwayne = Car(y=0.5)
controller = PIDController(kp=0.2, kd=0.3, ts=sampling_time)
y_black_box = np.array([dwayne.y])
x_black_box = np.array([dwayne.x])  # again resetting all values to ensure consistency
for t in range(num_points):
    steering = controller.control(y=dwayne.y)
    dwayne.move(steering+ u_disturbance, sampling_time)
    y_black_box = np.append(y_black_box, dwayne.y)
    x_black_box = np.append(x_black_box, dwayne.x)

plt.plot(x_black_box, y_black_box, label='Kd=0.3')  # much tidier than previous code and achieves the same

t_span = sampling_time * np.arange(num_points + 1)
plt.grid()
plt.xlabel('Lateral position, x (m)')
plt.ylabel('Lateral position, y (m)')
plt.legend()
plt.show()
