from typing import List, Tuple
from matplotlib import pyplot as plt
from matplotlib import animation as anim
from scipy.interpolate import barycentric_interpolate
import numpy as np
import math

def runge_function(x: float) -> float:
    return 1 / (1 + 25 * x * x)


fig, ax = plt.subplots()
fig.set_tight_layout(True)
ax.grid()
ax.set(ylim=(-2, 2))

x = np.linspace(-1, 1, 200)

# Persistent
ax.plot(x, runge_function(x), 'k')

n = 2
samples = np.linspace(-1, 1, n)
interpolated = barycentric_interpolate(samples, runge_function(samples), x)

# To be updated
points, = ax.plot(samples, runge_function(samples), 'or')
fit, = ax.plot(x, interpolated, ':b')
text = ax.text(-0.5, 1.51, f'{n} uniform nodes')

print(points)
print(fit)

def update(n: int):
    label = f'{n} uniform nodes'
    print(label)

    samples = np.linspace(-1, 1, n)
    interpolated = barycentric_interpolate(samples, runge_function(samples), x)

    points.set_data(samples, runge_function(samples))
    fit.set_ydata(interpolated)
    text.set_text(label)
    return points, fit, text

if __name__ == '__main__':
    animation = anim.FuncAnimation(fig, update, frames=np.arange(2, 15), interval=750)
    animation.save('uniform_nodes.gif', dpi=80, writer='imagemagick')
