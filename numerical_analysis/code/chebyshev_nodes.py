import math
from typing import List, Tuple

from matplotlib import pyplot as plt
from scipy.interpolate import barycentric_interpolate
import numpy as np

def chebyshev_nodes(
        n: int, 
        interval: Tuple[float, float]=(-1, 1)
        ) -> List[float]:
    a0 = (1 / 2) * (interval[0] + interval[1])
    a1 = (1 / 2) * (interval[1] - interval[0])
    f = lambda k: math.cos(math.pi * (2 * k - 1) / (2 * n))
    
    return [a0 + a1 * f(k) for k in range(1, n + 1)]

def runge_function(x: float) -> float:
    return 1 / (1 + 25 * x * x)


# Number of datapoints (nodes) to sample from
node_count = 40

spaced_x = np.array(chebyshev_nodes(node_count))
uniform_x = np.linspace(-1, 1, node_count)

interp_cheb_x = np.linspace(min(spaced_x), max(spaced_x), 150)
interp_uniform_x = np.linspace(min(uniform_x), max(uniform_x), 150)
actual_x = np.linspace(-1, 1, 150)

chebyshev_y = barycentric_interpolate(spaced_x, runge_function(spaced_x), interp_cheb_x)
uniform_y = barycentric_interpolate(uniform_x, runge_function(uniform_x), interp_uniform_x)
actual_y = runge_function(actual_x)

# Plotting code
fig, (ax1, ax2) = plt.subplots(nrows=2)

ax1.plot(actual_x, actual_y, 'k')
ax1.plot(interp_cheb_x, chebyshev_y, '--r')
ax2.plot(actual_x, actual_y, 'k')
ax2.plot(interp_uniform_x, uniform_y, '--b')
ax2.set(ylim=(-1, 1.2))

ax1.grid()
ax2.grid()

plt.show()
