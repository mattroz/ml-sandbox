# Get this code from https://habrahabr.ru/post/307312/
# for the sake of understanding batch gradient descent visually

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


STEP_COUNT = 100
STEP_SIZE = 0.1
X = np.array([i for i in np.linspace(-1, 1, 1000)])
Y = np.array([i for i in np.linspace(-1, 1, 1000)])


def func(X, Y):
    return np.sin(5*X)*np.cos(5*Y)/5


def dx(x, y):
    return np.sin(5*x)*np.sin(5*y)


def dy(x,y):
    return np.sin(5*x)*np.sin(5*y)


skip_first = True
def draw_gradient_points(num, point, line):
    global previous_x, previous_y, skip_first, ax
    if skip_first:
        skip_first = False
        return point
    current_x = previous_x - STEP_SIZE * dx(previous_x, previous_y)
    current_y = previous_y - STEP_SIZE * dy(previous_x, previous_y)
    print("Step:", num, "CurX:", current_x, "CurY", current_y, "Fun:", func(current_x, current_y))
    point.set_data([current_x], [current_y])
    # Blah-blah
    new_x = list(line.get_xdata()) + [previous_x, current_x]
    new_y = list(line.get_ydata()) + [previous_y, current_y]
    line.set_xdata(new_x)
    line.set_ydata(new_y)

    previous_x = current_x
    previous_y = current_y
    return point


previous_x, previous_y = .6, .3
fig, ax = plt.subplots()
p = ax.get_position()
ax.set_position([p.x0, p.y0, p.width, p.height])
ax.set_xlabel("X", fontsize=18)
ax.set_ylabel("Y", fontsize=18)

X, Y = np.meshgrid(X, Y)
plt.contour(X, Y, func(X, Y))
point, = plt.plot([.6], [.3], 'bo')
line, = plt.plot([], color='black')


gradient_anim = anim.FuncAnimation(fig, draw_gradient_points, frames=STEP_COUNT,
                                   fargs=(point, line),
                                   interval=350)

# Need ffmpeg to be installed
gradient_anim.save("contour_plot.gif", writer="imagemagick")
