from typing import List, Tuple
from matplotlib import pyplot as plt
from matplotlib import animation as anim
from scipy.interpolate import barycentric_interpolate
import numpy as np
import math

def chebyshev_nodes(
        n: int, 
        interval: Tuple[float, float]=(-1, 1)
        ) -> List[float]:
    a0 = (1 / 2) * (interval[0] + interval[1])
    a1 = (1 / 2) * (interval[1] - interval[0])
    f = lambda k: math.cos(math.pi * (2 * k - 1) / (2 * n))
    
    return np.array([a0 + a1 * f(k) for k in range(1, n + 1)])

def runge_function(x: float) -> float:
    return 1 / (1 + 25 * x * x)

fig, ax = plt.subplots()
fig.set_tight_layout(True)
ax.grid()

x = np.linspace(-1, 1, 200)

# Persistent
ax.plot(x, runge_function(x), 'k')

n = 2
samples = chebyshev_nodes(n)
interpolated = barycentric_interpolate(samples, runge_function(samples), x)

# To be updated
points, = ax.plot(samples, runge_function(samples), 'or')
fit, = ax.plot(x, interpolated, ':b')
text = ax.text(-1.08, 0.81, f'{n} uniform nodes')

print(points)
print(fit)

def update(n: int):
    label = f'{n} Chebyshev distributed nodes'
    print(label)

    samples = chebyshev_nodes(n)
    interpolated = barycentric_interpolate(samples, runge_function(samples), x)

    points.set_data(samples, runge_function(samples))
    fit.set_ydata(interpolated)
    text.set_text(label)
    return points, fit, text

if __name__ == '__main__':
    animation = anim.FuncAnimation(fig, update, frames=np.arange(2, 31),
            interval=400)
    animation.save('chebyshev_nodes.gif', dpi=80, writer='imagemagick')
