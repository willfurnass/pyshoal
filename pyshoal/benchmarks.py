import numpy as np
import pso

"""
Benchmark functions taken from:

Yao, X., and Liu, Y. (1996). Fast evolutionary programming. In Proceedings of the Fifth 
Annual Conference on Evolutionary Programming (Vol. 19). Cambridge, MA: The MIT Press.

"""

class Benchmark(object):
    def __init__(self, name, f, ndim, lower_bound, upper_bound, optimal_params, optimal_perf, params_tol_dp, perf_tol_dp = 4):
        self.name, self.f, self.ndim, self.lower_bound, self.upper_bound,        self.optimal_params, self.optimal_perf,       self.params_tol_dp, self.perf_tol_dp = \
             name,      f,      ndim,      lower_bound,      upper_bound, np.asfarray(optimal_params),     optimal_perf, np.asarray(params_tol_dp),     perf_tol_dp
        min_calc_val = self.f(*self.optimal_params)
        if not np.abs(min_calc_val - self.optimal_perf) < (10 ** - float(perf_tol_dp)):
            raise Exception("Error when checking benchmark calcs for func {}: {} != perf of {}".format(self.name, min_calc_val, self.optimal_perf))
            
    def opt(self, n_parts = 25, topo = 'gbest', weights = (0.9, 0.4, 2.1, 2.1), max_itr = 250, tol_win = 5):   
        # Bounds as single col of multiple rows with each of latter being min + max
        box_bounds = np.tile((self.lower_bound, self.upper_bound), (self.ndim, 1))
    
        shoal = pso.PSO(self.f, 
                            init_var_ranges = box_bounds, 
                            n_parts = n_parts, 
                            topo = topo, 
                            weights = weights, 
                            opt_args = None, 
                            box_bounds = box_bounds,
                            minimise = True)
        
        swarm_best, swarm_best_perf, final_itr = shoal.opt(
                max_itr = max_itr,
                tol_thres = 10 ** - np.asfarray(self.params_tol_dp),
                tol_win = tol_win,
                plot = False)
        return BenchmarkResult(self, swarm_best, swarm_best_perf, final_itr)
    
class BenchmarkResult(object):
    def __init__(self, benchmark, swarm_best, swarm_best_perf, final_itr):
        self.benchmark, self.swarm_best, self.swarm_best_perf, self.final_itr = \
        benchmark, swarm_best, swarm_best_perf, final_itr
        
    def check(self):
        """Are the optimisation results within the required tolerances for all params?"""
        #param_tols = 10 ** - (np.asfarray(self.benchmark.params_tol_dp))
        #return np.all(np.abs(self.swarm_best - self.benchmark.optimal_params) < param_tols)
        perf_tol = 10 ** - self.benchmark.perf_tol_dp
        return np.abs(self.swarm_best_perf - self.benchmark.optimal_perf) < perf_tol
        
    def __str__(self):
        if self.check():
            return "Found global opt for func {} of {} ~= {} at {} in {} itrs".format(
                self.benchmark.name, self.swarm_best_perf, self.benchmark.optimal_perf, 
                self.swarm_best, self.final_itr)
        else:
            return "Global optimum NOT FOUND for func {}: expected {} at {}, found {} at {} after {} itrs".format(
                self.benchmark.name, 
                self.benchmark.optimal_perf, self.benchmark.optimal_params, 
                self.swarm_best_perf, self.swarm_best, self.final_itr)



# Specific benchmark functions and Benchmark objects

def f_01(*x):
    return np.sum(np.asarray(x) ** 2)
bmark_01a = Benchmark(name = 'f_01a', f = f_01, ndim = 30, lower_bound = -5.12, upper_bound = 5.12, 
                     optimal_params = np.ones(30) * 1e-4, optimal_perf = 3e-7, 
                     params_tol_dp = np.ones(30) * 4, perf_tol_dp = 4)

def f_16(x_0, x_1):
    return (4*(x_0**2)) - (2.1*(x_0**4)) + ((1/3.0)*(x_0**6)) + (x_0*x_1) - (4*(x_1**2)) + (4*(x_1**4))

bmark_16 = Benchmark(name = 'f_16', f = f_16, ndim = 2, lower_bound = -5.0, upper_bound = 5.0, 
                     optimal_params = np.array([-0.09, 0.71]), optimal_perf = -1.0316, 
                     params_tol_dp = (2, 2), perf_tol_dp = 4)

def f_17(x_0, x_1):
    return (x_1 - (5.1 / (4 * (np.pi**2))*(x_0 ** 2)) + ((5./np.pi) * x_0) - 6  )**2 + (10 * (1-(1/(8*np.pi))) * np.cos(x_0)) + 10
    
bmark_17 = Benchmark(name = 'f_17', f = f_17, ndim = 2, lower_bound = -5.0, upper_bound = 5.0, 
                     optimal_params = np.array([9.42, 2.47]), optimal_perf = 0.398, 
                     params_tol_dp = (4, 4), perf_tol_dp = 3)

def f_18(x_0, x_1):
    a = x_0 + x_1 + 1
    b = 19 - (14*x_0) + (3*(x_0**2)) - (14*x_1) + (6*x_0*x_1) + (3*(x_1**2))
    c = (2*x_0) - (3*x_1)
    d = 18 - (32*x_0) + (12*(x_0**2)) + (48*x_1) - (36*x_0*x_1) + (27*(x_1**2))
    return (1 + (a**2)*b) * (30 + (c**2)*d)
#bmark_18 = Benchmark('f_18', f_18, 2, -2.0, 2.0, np.array([1.49e-5, 1]), 3.0, (5, 2), 4)

a = np.array([
    [4, 4, 4, 4],
    [1, 1, 1, 1],
    [8, 8, 8, 8],
    [6, 6, 6, 6],
    [3, 7, 3, 7],
    [2, 9, 2, 9],
    [5, 5, 3, 3], 
    [8, 1, 8, 1],
    [6 ,2, 6, 2],
    [7, 3.6, 7, 3.6]
    ])
c = np.array([
    0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5
    ])

def f_21(x_0, x_1, x_2, x_3):
    """Low-dimensional function with only a few local minima."""
    x = np.array([x_0, x_1, x_2, x_3])
    return - np.sum([(np.dot((x - a[i]).T, (x - a[i])) + c[i]) ** -1 for i in xrange(4)])
bmark_21 = Benchmark(name = 'f_21', f = f_21, ndim = 4, lower_bound = 0.0, upper_bound = 10.0, 
                     optimal_params = np.ones(4) * 1e-5 + 4, optimal_perf = -10.10417, 
                     params_tol_dp = (5,5,5,5), perf_tol_dp = 1)
