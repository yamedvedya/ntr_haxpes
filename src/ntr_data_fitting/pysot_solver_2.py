from pySOT.optimization_problems import OptimizationProblem
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.surrogate import SurrogateUnitBox, RBFInterpolant, LinearTail
from pySOT.strategy import SRBFStrategy
from poap.mpiserve import MPIController, MPISimpleWorker
from poap.controller import ThreadController, BasicWorkerThread

import numpy as np
import time
from optparse import OptionParser
from src.ntr_data_fitting.subfunctions import get_shifts
from src.ntr_data_fitting.ntr_fitter import NTR_fitter

try:
    from mpi4py import MPI
except Exception as err:
    print("ERROR: You need mpi4py to use the POAP MPI controller.")
    exit()


class HAXPESOptimizationProblem(OptimizationProblem):

    def __init__(self, parent):
        self.parent = parent
        self.dim = 2 * self.parent.potential_model['num_depth_dof'] + self.parent.potential_model['only_voltage_dof']
        self.ub, self.lb = self._evaluate_borders()
        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)
        self.results = []

    def _evaluate_borders(self):
        """
        Evaluates borders for defined dataset
        :return: tuple (upper_border, lower_border)
        """
        max_voltage = self.parent.settings['VOLT_MAX'] - 0.01
        min_voltage = -max_voltage
        max_depth = self.parent.data_set_for_fitting['fit_depth_points'][-1]
        min_depth = self.parent.data_set_for_fitting['fit_depth_points'][0]
        n_intermediate_points = int((self.dim - 2) / 2)
        ub = np.empty(self.dim)
        lb = np.empty(self.dim)
        ub[:(n_intermediate_points + 2)] = max_voltage
        ub[(n_intermediate_points + 2):] = max_depth
        lb[:(n_intermediate_points + 2)] = min_voltage
        lb[(n_intermediate_points + 2):] = min_depth
        return ub, lb

    def _extract_sets(self, x):
        """
        Format x into volt_set and depth_set

        :param x: numpy array [V_left, V1, ..., Vn, V_right, x1, ..., xn]
        :return: tuple of numpy arrays (volt_set, depth_set)
        """
        n_intermediate_points = int((self.dim - 2) / 2)
        volt_set = x[0:(n_intermediate_points + 2)]
        left_border = self.parent.data_set_for_fitting['fit_depth_points'][0] + 1e-10
        right_border = self.parent.data_set_for_fitting['fit_depth_points'][-1] - 1e-10
        borders = self.parent.start_values[1]
        depth_set = np.concatenate(([left_border], x[(n_intermediate_points + 2):], [right_border]))

        return volt_set, depth_set

    def eval(self, x):
        """
        Returns the value of the objective function at given point (x)
        :param x: numpy array [V_left, V1, ..., Vn, V_right, x1, ..., xn]
        :return: float
        """
        super().__check_input__(x)
        volt_set, depth_set = self._extract_sets(x)
        shifts, _ = get_shifts(self.parent.data_set_for_fitting, depth_set, volt_set)
        if shifts is None:
            obj_value = np.array(100)  # костыль!
        else:
            obj_value = np.sum((shifts - self.parent.data_set_for_fitting['spectroscopic_data'][:, 2]) ** 2)
        return obj_value


def main_worker(objfunction):
    MPISimpleWorker(objfunction).run()


def main_master(opt_problem, n_workers, max_evals):

    exp_design = SymmetricLatinHypercube(dim=opt_problem.dim, num_pts=2 * (opt_problem.dim + n_workers))
    srgt_model = SurrogateUnitBox(RBFInterpolant(dim=opt_problem.dim, tail=LinearTail(dim=opt_problem.dim)),
                                  lb=opt_problem.lb, ub=opt_problem.ub)

    strategy = SRBFStrategy(max_evals=max_evals, opt_prob=opt_problem, exp_design=exp_design,
                            surrogate=srgt_model, asynchronous=True, batch_size=n_workers)

    controller = MPIController(strategy)

    start = time.time()
    result = controller.run()
    stop = time.time()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf)))
    print(stop - start)


def mpi_execution(fitter, max_evals):
    """
    Run in multiple threads. Master thread uses main_master() function, others use main_worker() function
    Requires launch with mpiexec! Does not work in single thread!
    :param fitter: NTR_Fitter object with loaded .set
    :param max_evals: max number of evaluations
    """
    opt_problem = HAXPESOptimizationProblem(fitter)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if rank == 0:
        main_master(opt_problem, nprocs, max_evals)
    else:
        main_worker(opt_problem.eval)


def single_thread_execution(fitter, max_evals):
    """
    Run in a single thread
    :param fitter: NTR_Fitter object with loaded .set
    :param max_evals: max number of evaluations
    """
    opt_problem = HAXPESOptimizationProblem(fitter)
    exp_design = SymmetricLatinHypercube(dim=opt_problem.dim, num_pts=2 * (opt_problem.dim + 1))
    srgt_model = SurrogateUnitBox(RBFInterpolant(dim=opt_problem.dim, tail=LinearTail(dim=opt_problem.dim)),
                                  lb=opt_problem.lb, ub=opt_problem.ub)

    controller = ThreadController()
    controller.strategy = SRBFStrategy(max_evals=max_evals, opt_prob=opt_problem, exp_design=exp_design,
                                       surrogate=srgt_model, asynchronous=True)

    worker = BasicWorkerThread(controller, opt_problem.eval)
    controller.launch_worker(worker)
    start = time.time()
    result = controller.run()
    stop = time.time()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf)))
    print(stop - start)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--data_set", dest="data_set")
    (options, _) = parser.parse_args()

    if options.data_set:
        pass
    else:
        raise RuntimeError('add path to .set file')

    fitter = NTR_fitter()
    start_values = fitter.load_fit_set(options.data_set)
    max_evals = 100
    mpi_execution(fitter, max_evals)
    # single_thread_execution(fitter, max_evals)


