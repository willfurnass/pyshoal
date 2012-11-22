import numpy as np
import numpy.random as np_rand

from time import sleep
import matplotlib.pyplot as plt
import sys

import pdb

import logging
#logger.basicConfig(format='%(levelname)s: %(message)s', level=logger.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

# Allow quick setting of module-level logging
def set_log_level(level):
    logger.setLevel(level)
    logger.handlers[0].setLevel(level)

class PSO(object):
    """
    Particle Swarm Optimisation implementation

    References:
    Eberhart, R.C., Shi, Y., 2001. Particle swarm optimization: developments, 
    applications and resources, in: Proceedings of the 2001 Congress on 
    Evolutionary Computation. Presented at the Congress on Evolutionary 
    Computation, pp. 81-86.

    Kennedy, J., Mendes, R., 2002. Population structure and particle swarm 
    performance. IEEE, pp. 1671-1676.

    Xu, S., Rahmat-Samii, Y., 2007. Boundary conditions in particle swarm 
    optimization revisited. IEEE Transactions on Antennas and Propagation, 
    55, 760-765.

    """
    def __init__(self, obj_func, init_var_ranges, n_parts = 5, topo="gbest", weights = [0.9, 0.4, 1.5, 2.5], opt_args = None, bounds = None, minimise = True):
        """Initialise the positions and velocities of the particles, the particle memories and the swarm-level params.

        Keyword args:
        obj_func -- Objective function
        init_var_ranges -- tuple of (lower_bound, upper_bound) tuples (or 
                           equivalent ndarray), the length of the former 
                           being the number of dimensions e.g. ((0,10),(0,30))
                           for 2 dimensions
        n_parts -- number of particles
        topo -- 'gbest', 'ring' or 'von_neumann'
        weights -- 4-vector of weights for inertial (initial and final), 
                   nostalgic and societal velocity components
        opt_args -- dictionary of keyword arguments to be passed to the 
                    objective function; these arguments do not correspond to 
                    variables that are to be optimised
        bounds -- tuple of (lower_bound, upper_bound) tuples (or equivalent 
                  ndarray), the length of the former being the number of 
                  dimensions e.g. ((0,10),(0,30)) for 2 dims.  
                  Restricted damping is used (Xu and Rahmat-Samii, 2007) when 
                  particles go out of bounds.
        minimise -- whether to find global minima or maxima for the objective function

        Notes on swarm and neighbourhood sizes (Eberhart and Shi, 2001)
        Swarm size of 20-50 most common.
        Neighbourhood size of ~15% swarm size used in many applications.

        Notes on topologies:
        Two neighbourhood topologies have been implemented (see Kennedy and 
        Mendes, 2002).  These are both social rather than geographic 
        topologies and differ in terms of degree of connectivity (number of 
        neighbours per particle) but not in terms of clustering (number of 
        common neighbours of two particles).  The supported topologies are:
        ring -- exactly two neighbours / particle
        von_neumann -- exactly four neighbours / particle (swarm size must 
                       be square number)
        gbest -- the neighbourhood of each particle is the entire swarm 

        Notes on weights (Eberhart and Shi, 2001, Xu and Rahmat-Samii, 2007):
        inertia weight   -- decrese linearly from 0.9 to 0.4 over a run
        nostalgia weight -- 2.05
        societal weight -- 2.05
        NB sum of nostalgia and societal weights should be >4 if using
        Clerc's constriction factor.
        """

        self.minimise = minimise

        if opt_args:
            # Create partial function through baking arguments into the objective function
            # that are not parameters to be optimised
            obj_partial_func = lambda *args : obj_func(*args+tuple(opt_args))

            # Vectorize the objective function if it has not been done so already
            self.obj_func_np = np.vectorize(obj_partial_func)
        else:
            self.obj_func_np = np.vectorize(obj_func)

        # Initialise velocity weights
        (self.w_inertia_start, self.w_inertia_end, self.w_nostalgia, self.w_societal) = weights
        
        # Set ranges used to bound initial particle positions
        # NB: should really parse init_var_ranges to ensure that it is valid
        self.init_pos_min, self.init_pos_max = np.asfarray(init_var_ranges).T

        # Set number of particles
        self._n_parts = n_parts

        # Set number of dimensions
        self._n_dims = len(init_var_ranges)

        # Initialise particle positions
        # Each row is the position vector of a particular particle
        self.pos = np_rand.uniform(self.init_pos_min, 
                                   self.init_pos_max, 
                                   (self._n_parts, self._n_dims))

        # Previous best position for all particles is starting position
        self.pbest = self.pos.copy()

        # Initialise velocity matrix
        # Each row is the velocity vector of a particular particle
        self.vel = np_rand.uniform(- (self.init_pos_max - self.init_pos_min), 
                                   self.init_pos_max - self.init_pos_min, 
                                   (self._n_parts, self._n_dims))

        # Determine the problem space boundaries...
        self.lower_bounds, self.upper_bounds = np.asfarray(bounds).T

        # then find the performance per particle (matrix of n_parts rows and n_dims cols)
        self.perf = self.obj_func_np(*self.pos.T) 

        # The personal best position per particle is initially the starting position
        self.pbest_perf = self.perf.copy()

        # Initialise swarm best position (array of length n_dims)
        # and the swarm best performance  (scalar)
        if self.minimise:
            self.swarm_best = self.pos[self.perf.argmin()]
            self.swarm_best_perf = self.perf.min()
        else:
            self.swarm_best = self.pos[self.perf.argmax()]
            self.swarm_best_perf = self.perf.max()
        
        if logger.level < logging.INFO:
            logger.debug("INIT POS:  %s", self.pos)
            logger.debug("INIT PERF: %s", self.perf)
            logger.debug("INIT VEL:  %s", self.vel)

        # Determine particle neighbours if a swarm topology has been chosen
        self.topo = topo
        if self.topo in ("von_neumann", "ring"):
            self._cache_neighbourhoods()
        elif self.topo != "gbest":
            raise Exception("Topology '{}' not recognised/supported".format(self.topo))

    def _cache_neighbourhoods(self):
        """Determines the indices of the neighbours per particle and stores them in the neighbourhoods attribute.

        Currently the Von Neumann lattice and Ring topologies are supported (see Kennedy and Mendes, 2002).

        """
        n = self._n_parts

        if self.topo == "von_neumann":
            # Check that p_idx is square
            n_sqrt = int(np.sqrt(n))
            if not n_sqrt**2 == n:
                raise Exception("Number of particles needs to be perfect square if using Von Neumann neighbourhood topologies")
            self.neighbourhoods = np.zeros((n, 5), dtype=int)
            for p in xrange(n):
                self.neighbourhoods[p] = [
                    p,                                      # particle itself
                    (p - n_sqrt) % n,                         # particle above
                    (p + n_sqrt) % n,                         # particle below
                    ((p / n_sqrt) * n_sqrt) + ((p + 1) % n_sqrt), # particle to right
                    ((p / n_sqrt) * n_sqrt) + ((p - 1) % n_sqrt)  # particle to left
                    ]
        elif self.topo == "ring":
            self.neighbourhoods = np.zeros((n, 3), dtype=int)
            for p in xrange(n):
                self.neighbourhoods[p] = [
                    (p - 1) % n, # particle to left
                    p,           # particle itself
                    (p + 1) % n  # particle to right
                    ]

    def _velocity_updates(self, itr, max_itr, use_constr_factor = True):
        """Update particle velocities.

        Keyword arguments:
        itr     -- current timestep
        max_itr -- maximum number of timesteps
        use_constr_factor -- whether Clerc's constriction factor should be applied

        New velocities determined using
         - the supplied velocity factor weights
         - random variables to ensure the process is not deterministic
         - the current velocity per particle
         - the best performing position in each particle's personal history
         - the best performing current position in each particle's neighbourhood
        
        Max velocities clamped to length of problem space boundaries and
        Clerc's constriction factor applied as per Eberhart and Shi (2001) 

        Spits out the following if logging level is INFO
         - best neighbours per particle
         - velocity components per particle
         - velocities per particle

        NB should only be called from the VCDM class's _tstep method.
        """
        w_inertia = self.w_inertia_start + \
                        (self.w_inertia_end - self.w_inertia_start) * (itr / float(max_itr))

        inertia_vel_comp   = w_inertia * self.vel
        nostalgia_vel_comp = self.w_nostalgia * np_rand.rand() * (self.pbest - self.pos)
        societal_vel_comp  = self.w_societal * np_rand.rand() * (self.best_neigh - self.pos)

        self.vel = inertia_vel_comp + nostalgia_vel_comp + societal_vel_comp

        # Constriction factor
        if use_constr_factor:
            phi = self.w_nostalgia + self.w_societal
            if not phi > 4:
                raise Exception("Cannot apply constriction factor as sum of societal and nostalgic weights <= 4") 
            K = 2. / np.abs(2. - phi - np.sqrt((phi**2) - (4.*phi)))
            self.vel *= K

        # Velocity clamping
        self.vel.clip(self.lower_bounds - self.upper_bounds,
                      self.upper_bounds - self.lower_bounds,
                      out = self.vel)

        if logger.level < logging.INFO:
            logger.debug("BEST NEIGHBOURS:   {}".format(self.best_neigh))
            logger.debug("INERTIA V COMP:    {}".format(inertial_vel_comp))
            logger.debug("NOSTALGIA V COMP:  {}".format(nostalgia_vel_comp)) 
            logger.debug("SOCIETAL V COMP:   {}".format(societal_vel_comp))
            logger.debug("VELOCITIES:        {}".format(self.vel))
            
    def _bounds_checking(self):
        """Apply restrictive damping if position updates have caused particles 
        to leave problem space boundaries.

        Restrictive damping explained in:
        Xu, S., Rahmat-Samii, Y. (2007). Boundary conditions in particle swarm 
        optimization revisited. IEEE Transactions on Antennas and Propagation, 
        vol 55, pp 760-765.

        """
        too_low = self.pos < self.lower_bounds
        too_high = self.pos > self.upper_bounds

        # Ensure all particles within bounds of problem space
        self.pos.clip(self.lower_bounds, self.upper_bounds, out = self.pos)

        old_vel = self.vel.copy()
        self.vel[too_low | too_high] *= -1. * np_rand.random((self.vel[too_low | too_high]).shape)

        if np.array_equal(old_vel, self.vel) and np.any(too_low | too_high):
            raise Exception("Velocity updates in boundary checking code not worked")

        # What if reflected particles now comes out the other side of the problem space? TODO 
        too_low = self.pos < self.lower_bounds
        too_high = self.pos > self.upper_bounds
        if np.any(too_low) or np.any(too_high):
            raise Exception("Need multiple-pass bounds checking")

    def _tstep(self, itr, max_itr):
        """Optimisation timestep function

        Keyword arguments:
        itr -- current timestep
        max_itr -- maximum number of timesteps

        """
        if self.topo in ("von_neumann", "ring"):
            # For each particle:
            #   find the 'relative' index of the best performing neighbour 
            #   e.g. 0, 1 or 2 for particles in ring topologies
            if self.minimise:
                best_neigh_idx = self.neighbourhoods[np.arange(self._n_parts), 
                                                 self.perf[self.neighbourhoods].argmin(axis=1)]
            else:
                best_neigh_idx = self.neighbourhoods[np.arange(self._n_parts), 
                                                 self.perf[self.neighbourhoods].argmax(axis=1)]
            #   then generate a vector of the _positions_ of the best 
            #   performing particles in each neighbourhood 
            #   (the length of this vector will be equal to the number of particles)
            self.best_neigh = self.pos[best_neigh_idx]

        else: # self.topo == "gbest"
            # For the 'global best' approach the best performing neighbour 
            # is the best performing particle in the entire swarm
            self.best_neigh = self.swarm_best

        # Update the velocity and position of each particle
        self._velocity_updates(itr, max_itr)
        self.pos += self.vel

        # Check that all particles within problem boundaries; 
        # if not move to edges of prob space and 
        # flip signs of and dampen velocity components that took particles 
        # over boundaries
        self._bounds_checking()

        # Cache the current performance per particle
        self.perf = self.obj_func_np(*self.pos.T)

        # Update each particle's personal best position if an improvement has been made this timestep
        if self.minimise:
            improvement_made_idx = self.perf < self.pbest_perf
        else:
            improvement_made_idx = self.perf > self.pbest_perf
        self.pbest[improvement_made_idx] = self.pos[improvement_made_idx]
        self.pbest_perf[improvement_made_idx] = self.perf[improvement_made_idx]

        # Update swarm best position with current best particle position
        if self.minimise:
            self.swarm_best = self.pos[self.perf.argmin()]
            self.swarm_best_perf = self.perf.min()
        else:
            self.swarm_best = self.pos[self.perf.argmax()]
            self.swarm_best_perf = self.perf.max()

        if logger.level < logging.INFO:
            logger.debug("NEW POS:           {}".format(self.pos))
            logger.debug("NEW PERF:          {}".format(self.perf))
            logger.debug("SWARM BEST POS:    {}".format(self.swarm_best))
            logger.debug("SWARM BEST PERF:   {}".format(self.swarm_best_perf))

        return self.swarm_best, self.swarm_best_perf
    
    def plot_swarm(self, itr, xlims, ylims, contours_delta = None, sleep_dur = 0.1, save_plots = False):
        """Plot the swarm in 2D along with performance vs iterations.

        Keyword arguments:
        itr -- current interation (integer)
        xlims -- tuple of min and max x bounds to use for plotting the swarm
        ylims -- tuple of min and max y bounds to use for plotting the swarm
        contours_delta -- delta used for producing contours of objective function.  
                          Set to 'None' to disable contour plotting.
        sleep_dur -- time to sleep for between iterations / plots (seconds)
        save_plots -- save plots as a series of png images if True

        """
        if itr == 0:
            # Initialise plots and cache contour data
            if self._n_dims != 2 or len(xlims) != 2 or len(ylims) != 2:
                raise Exception("Cannot plot swarm if the number of dimensions is not 2.")

            self._plt_fig = plt.figure(figsize=(10,5))
            self._plt_swarm_ax = self._plt_fig.add_subplot(1,2,1)
            self._plt_perf_ax = self._plt_fig.add_subplot(1,2,2)
            if contours_delta:
                x = np.arange(xlims[0], xlims[1], contours_delta)
                y = np.arange(ylims[0], ylims[1], contours_delta)
                self._contour_X, self._contour_Y = np.meshgrid(x, y)
                self._contour_Z = self.obj_func_np(self._contour_X, self._contour_Y)
        else:
            self._plt_swarm_ax.clear()
            self._plt_perf_ax.clear()

        self._plt_swarm_ax.set_xlim(xlims)
        self._plt_swarm_ax.set_ylim(ylims)
    
        # Plot contours
        if contours_delta:
            cplot = self._plt_swarm_ax.contour(self._contour_X, self._contour_Y, self._contour_Z)
            self._plt_swarm_ax.clabel(cplot, inline=1, fontsize=10)

        # Plot swarm
        self._plt_swarm_ax.scatter(self.pos.T[0], self.pos.T[1])
        self._plt_perf_ax.plot(self.swarm_best_perf_hist)
        self._plt_perf_ax.set_xlim(xmax=itr)

        # Add title
        title = "Parts: {}   Dims: {}\nTopo: {}   ".format(self._n_parts, self._n_dims, self.topo)
        self._plt_swarm_ax.set_title(title)
        title = "Best pos: {}\nBest perf: {:5.5f}".format(self.swarm_best, self.swarm_best_perf)
        self._plt_perf_ax.set_title(title)
        #plt.subplots_adjust(top=0.80)
        self._plt_swarm_ax.set_xlabel("Var 1")
        self._plt_swarm_ax.set_ylabel("Var 2")
        self._plt_perf_ax.set_xlabel("Iter (current: {})".format(itr))

        plt.draw()
        if save_plots:
            self._plt_fig.savefig("%03d.png" % itr)
        sleep(sleep_dur)

    def opt(self, max_itr=100, tol_thres=None, tol_win=5, plot=False, save_plots=False):
        """Attempt to find the global optimum objective function.

        Keyword args:
        max_itr -- maximum number of iterations
        tol_thres -- convergence tolerance vector (optional); length must be equal to 
                     the number of dimensions.  Can be ndarray, tuple or list.
        tol_win -- number of timesteps for which the swarm best position must be 
                   less than convergence tolerances for the funtion to then return a result

        Returns:
         -- swarm best position
         -- performance at that position
         -- iterations taken to converge

        """
        if plot:
            plt.ion()

        self.swarm_best_hist = np.zeros((max_itr, self._n_dims))
        self.swarm_best_perf_hist = np.zeros((max_itr,))

        self.tol_thres = np.asfarray(tol_thres)
        if self.tol_thres.shape != (self._n_dims,):
            raise Exception("The length of the tolerance vector must be equal to the number of dimensions")

        itr = 0
        while itr < max_itr:
            # Run timestep code
            self.swarm_best_hist[itr], self.swarm_best_perf_hist[itr] = self._tstep(itr, max_itr)
            
            if plot:
                self.plot_swarm(itr, xlims = (-50,50), ylims = (-50,50), contours_delta = 1., sleep_dur = 0.001, save_plots = save_plots )

            # If the convergence tolerance has been reached then return
            if self.tol_thres is not None and itr > tol_win:
                win = self.swarm_best_hist[itr-tol_win+1:itr+1] 
                if np.all(win.max(axis=0) - win.min(axis=0) < self.tol_thres):
                    logger.debug("PSO converged in {} iterations.".format(itr))
                    break
            itr += 1
            if itr >= max_itr:
                logger.info("PSO failed to converge within {} iterations.".format(max_itr))

        if plot:
            plt.ioff()
            plt.show()
        return self.swarm_best, self.swarm_best_perf, itr
# END class PSO



def obj_func_1(x,y,z):
    return 3*(x-y)**4 + 4*(x-z)**2 / (1 + x**2 + y**2 + z**2) + np.cosh(x - 1) + np.tanh(y + z)

def obj_func_quadratic(x):
    return 3 - ((12 + x)**2)

def obj_func_rastrigin(a,b):
    return 0- (10 * 2 + \
    (a**2 - (10 * np.cos(2 * np.pi * a))) + \
    (b**2 - (10 * np.cos(2 * np.pi * b))))

def obj_func_sinc(a,b):
    return np.sinc(a) + np.sinc(b)

def obj_func_parabola(a,b):
    return 0 - a**2 - b**2



if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2: 
        print "Need objective function name as argument."
        sys.exit(-1)
    if sys.argv[1] == 'quadratic':
        obj_func = obj_func_quadratic
    elif sys.argv[1] == 'rastrigin':
        obj_func = obj_func_rastrigin
    elif sys.argv[1] == 'sinc':
        obj_func = obj_func_sinc
    elif sys.argv[1] == 'parabola':
        obj_func = obj_func_parabola
    else:
        raise Exception("objective function {} not supported.".format(sys.argv[1]))

    #o = PSO(obj_func = obj_func, init_var_ranges = ((-500,500),(-500,500)), n_parts = 144, topo="gbest", weights=[0.9, 0.4, 1.0, 2.5], minimise=False)
    o = PSO(obj_func = obj_func, init_var_ranges = ((-50,50),(-50,50)), n_parts = 64, topo="von_neumann", weights=[0.9, 0.4, 2.1, 2.1], bounds=((-5,50),(-5,50)), minimise=False)
    res = o.opt(max_itr = 100, tol_thres = (0.01,0.01), tol_win=5, plot=True, save_plots=False)
    logger.info("\nBest position: {}\nBest perf: {}\nNum iter: {}\n".format(*res))
