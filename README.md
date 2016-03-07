# pyshoal

## An implementation of [Particle Swarm Optimisation][pso] in Python.

Typical usage:

```python
import numpy as np
from pyshoal import PSO

def rastrigin(a, b):
    """Objective function."""
    return 10 * 2 + \
           (a**2 - (10 * np.cos(2 * np.pi * a))) + \
           (b**2 - (10 * np.cos(2 * np.pi * b)))

my_pso = PSO(obj_func=rastrigin,
             box_bounds=((-500, 500), (-500, 500)),
             n_parts=144,
             topo='gbest',
             weights=[0.9, 0.4, 2.1, 2.1],
             extra_args=None,
             minimise=True)

results = my_pso.opt(max_itr=100,
                     tol_thres=(0.01, 0.01),
                     tol_win=5,
                     plot=True,
                     save_plots=False)

(swarm_best, swarm_best_perf, n_itrs) = results
```

Please see the docstrings in `pso.py` for more info on pyshoal usage.

Note that pyshoal originally allowed for parallel execution of the objective function at each PSO step.  This is not presently supported.

[pso]: https://en.wikipedia.org/wiki/Particle_swarm_optimization
