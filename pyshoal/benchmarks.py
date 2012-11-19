import numpy as np
import logging
import pyshoal

class Benchmark(object):
    def __init__(self, name, f, ndim, lower_bound, upper_bound, min_at, min_val, input_tol, output_tol = 1e-4):
        self.name, self.f, self.ndim, self.lower_bound, self.upper_bound, self.min_at, self.min_val, self.input_tol, self.output_tol = \
             name,      f,      ndim,      lower_bound,      upper_bound,      min_at,      min_val,      input_tol,      output_tol
        min_calc_val = self.f(*self.min_at)
        if not np.abs(min_calc_val - self.min_val) < output_tol:
            raise Exception("Error when checking benchmark calcs for func {}: {} != {}".format(self.name, min_calc_val, self.min_val))
            
    def opt(self, n_parts = 16, topo = 'von_neumann', weights = (0.6, 0.4, 1.5, 2.5), max_itr = 250, tol_win = 5):   
        param_bounds = [(self.lower_bound, self.upper_bound) for d in xrange(self.ndim)]
        #w_inertia_start, w_inertia_end, w_nostalgia, w_societal = 0.6, 0.4, 1.5, 2.5
    
        shoal = pyshoal.PSO(self.f, 
                            init_var_ranges = param_bounds, 
                            n_parts = n_parts, 
                            topo = topo, 
                            weights = weights, 
                            opt_args = None, 
                            bounds = param_bounds,
                            minimise = True)
        
        swarm_best, swarm_best_perf, final_itr = shoal.opt(
                max_itr = max_itr,
                tol_thres = self.input_tol,
                tol_win = tol_win,
                plot = False)
        return BenchmarkResult(self, swarm_best, swarm_best_perf, final_itr)
    
class BenchmarkResult(object):
    def __init__(self, benchmark, swarm_best, swarm_best_perf, final_itr):
        self.benchmark, self.swarm_best, self.swarm_best_perf, self.final_itr = \
        benchmark, swarm_best, swarm_best_perf, final_itr
        
    def check(self):
        if not np.abs(self.swarm_best_perf - self.benchmark.min_val) < self.benchmark.output_tol:
            logging.info("Global optimum not found for func {}: {} != {}".format(
                self.benchmark.name, self.swarm_best, self.benchmark.min_val)) 
            return False
        else:
            return True
        
    def __str__(self):
        if self.check():
            return "Found global optimum for func {} of {} ~== {} at {} in {} iterations".format(
                self.benchmark.name, self.swarm_best_perf, self.benchmark.min_val, 
                self.swarm_best, self.final_itr)
        else:
            return "Global optimum not found for func {}: {} (at {}) != {}".format(
                self.benchmark.name, self.swarm_best_perf, self.swarm_best, self.benchmark.min_val)
                
def f_16(x_0, x_1):
    return (4*(x_0**2)) - (2.1*(x_0**4)) + ((1/3.0)*(x_0**6)) + (x_0*x_1) - (4*(x_1**2)) + (4*(x_1**4))

bmark_16 = Benchmark(name = 'f_16', f = f_16, ndim = 2, lower_bound = -5.0, upper_bound = 5.0, 
                     min_at = np.array([-0.09, 0.71]), min_val = -1.0316, 
                     input_tol = (1e-4, 1e-4), output_tol = 1e-4)
res_16 = bmark_16.opt()
print res_16

def f_17(x_0, x_1):
    return (x_1 - (5.1 / (4 * (np.pi**2))*(x_0 ** 2)) + ((5./np.pi) * x_0) - 6  )**2 + (10 * (1-(1/(8*np.pi))) * np.cos(x_0)) + 10
    
bmark_17 = Benchmark(name = 'f_17', f = f_17, ndim = 2, lower_bound = -5.0, upper_bound = 5.0, 
                     min_at = np.array([9.42, 2.47]), min_val = 0.398, 
                     input_tol = (1e-4, 1e-4), output_tol = 1e-3)
res_17 = bmark_17.opt()
print res_17

def f_18(x_0, x_1):
    a = x_0 + x_1 + 1
    b = 19 - (14*x_0) + (3*(x_0**2)) - (14*x_1) + (6*x_0*x_1) + (3*(x_1**2))
    c = (2*x_0) - (3*x_1)
    d = 18 - (32*x_0) + (12*(x_0**2)) + (48*x_1) - (36*x_0*x_1) + (27*(x_1**2))
    return (1 + (a**2)*b) * (30 + (c**2)*d)
bmark_18 = Benchmark('f_18', f_18, 2, -2.0, 2.0, np.array([1.49e-5, 1]), 3.0, (1e-4, 1e-4))
res_18 = bmark_18.opt()
print res_18
