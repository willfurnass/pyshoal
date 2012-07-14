===========
PyShoal
===========

PyShoal is an implementation of <Particle Swarm Optimisation> in Python.

Typical usage::

    #!/usr/bin/env python
    from pso import PSO

    def rastrigin(a,b):
        return 0- (10 * 2 + \
               (a**2 - (10 * np.cos(2 * np.pi * a))) + \
               (b**2 - (10 * np.cos(2 * np.pi * b))))

    obj_func = rastrigin
    o = PSO(obj_func = obj_func, 
            init_var_ranges = ((-500,500),(-500,500)), 
            n_parts = 144, 
            topo="gbest", 
            weights=[0.9, 0.4, 1.0, 2.5])
    res = o.opt(max_itr = 100, 
                tol_thres = (0.01,0.01), 
                tol_win = 5, 
                plot = True, 
                save_plots=False)

