from src.ntr_data_fitting.subfunctions import *
import ctypes
import json
from scipy import stats
import multiprocessing as mp
from mpi4py import MPI
import time
import pickle
import os

class Gradient_Mesh_Solver():

    _keys_to_save = ('_fit_points', '_depthset', '_voltset', '_mse', 'v_graphs_history',
                    'd_graphs_history', 'potential_graphs_history', 'shifts_graphs_history',
                     'solution_history', 't_val', 'cycle', 'best_ksi')

    _key_to_set = ('_potential_model', '_settings', '_data_set_for_fitting', '_start_values',
                  '_directory', '_sample_name')

    # ----------------------------------------------------------------------
    def __init__(self):

        self._directory = ''
        self._sample_name = ''

        self.best_ksi = [np.inf]
        self._fill_counter = 0
        self.t_val = np.Inf
        
        self._depthset = None
        self._voltset = None
        self._mse = None

        self.v_graphs_history = []
        self.d_graphs_history = []
        self.potential_graphs_history = []
        self.shifts_graphs_history = []

        self.solution_history = []

        self._potential_model = {}
        self._settings = {}
        self._data_set_for_fitting = None
        self._start_values = None

        self.fit_in_progress = False

        self.cycle = -1

    # ----------------------------------------------------------------------
    def reset_fit(self, potential_model, settings, data_set_for_fitting, start_values,
                  directory, fit_name):

        self._directory = directory
        self._sample_name = fit_name

        self._potential_model = potential_model
        self._settings = settings
        self._data_set_for_fitting = data_set_for_fitting
        self._start_values = start_values

        self.prepare_variables()

    # ----------------------------------------------------------------------
    def prepare_variables(self):
        self.cycle = -1
        
        self.best_ksi = [np.inf]
        self.solution_history = []
        
        self.v_graphs_history = [[] for _ in range(self._potential_model['num_depth_dof'] +
                                                   self._potential_model['only_voltage_dof'])]
        self.d_graphs_history = [[] for _ in range(self._potential_model['num_depth_dof'])]
        self.potential_graphs_history = []
        self.shifts_graphs_history = []


    # ----------------------------------------------------------------------
    def _generate_start_set(self):

        v_set = np.zeros((self._potential_model['num_depth_dof'] + self._potential_model['only_voltage_dof'],
                          self._settings['V_MESH']))

        d_set = np.zeros((self._potential_model['num_depth_dof'], self._settings['D_MESH']))

        return v_set, d_set

    # ----------------------------------------------------------------------
    def _fill_start_set(self, v_set, d_set):

        v_set = np.zeros((self._potential_model['num_depth_dof'] + self._potential_model['only_voltage_dof'],
                          self._settings['V_MESH']))

        v_half_steps = int(np.floor(self._settings['V_MESH'] / 2))

        for point in range(self._potential_model['num_depth_dof'] +
                           self._potential_model['only_voltage_dof']):
            v_set[point, :] = self._start_values[point][1] + np.linspace(-self._settings['V_STEP'] * v_half_steps,
                                                                self._settings['V_STEP'] * v_half_steps,
                                                                self._settings['V_MESH'])

        d_half_steps = int(np.floor(self._settings['D_MESH'] / 2))

        d_min = self._data_set_for_fitting['fit_depth_points'][0] + self._settings['D_STEP'] * 1e-9
        d_max = self._data_set_for_fitting['fit_depth_points'][-1] - self._settings['D_STEP'] * 1e-9

        for point in range(self._potential_model['num_depth_dof']):
            d_set[point, :] = self._start_values[point + 1][0] + np.linspace(-self._settings['D_STEP'] * d_half_steps,
                                                                    self._settings['D_STEP'] * d_half_steps,
                                                                    self._settings['D_MESH']) * 1e-9

            if d_set[point, 0] < d_min:
                d_set[point, :] += d_min - d_set[point, 0]

            if d_set[point, -1] > d_max:
                d_set[point, :] -= d_set[point, -1] - d_max

        return v_set, d_set

    # ----------------------------------------------------------------------
    def _get_fit_set(self, v_set, d_set):
        num_dof = self._potential_model['num_depth_dof'] + self._potential_model['only_voltage_dof']

        self._fit_points = 1

        self._fit_points *= v_set.shape[1] ** v_set.shape[0]
        self._fit_points *= d_set.shape[1] ** d_set.shape[0]

        self.t_val = stats.t.ppf(1 - self._settings['T'], self._data_set_for_fitting['spectroscopic_data'].shape[0])
        self._depthset = np.zeros((num_dof, self._fit_points))
        self._voltset = np.zeros((num_dof, self._fit_points))
        self._mse = np.zeros(self._fit_points)
        self.statistics = {}

        self._fill_counter = 0
        self._fill_voltages([], v_set, d_set)

    # ----------------------------------------------------------------------
    def _fill_voltages(self, selected_v, last_v_set, d_set):
        if len(last_v_set) > 1:
            for v in last_v_set[0, :]:
                self._fill_voltages(np.append(selected_v, v), last_v_set[1:, :], d_set)
        else:
            for v in last_v_set[0, :]:
                self._fill_depth(np.append(selected_v, v), [], d_set)

    # ----------------------------------------------------------------------
    def _fill_depth(self, selected_v, selected_d, last_d_set):
        if len(last_d_set) > 1:
            for d in last_d_set[0, :]:
                self._fill_depth(selected_v, np.append(selected_d, d), last_d_set[1:, :])
        elif len(last_d_set) == 1:
            for d in last_d_set[0, :]:
                self._depthset[:, self._fill_counter] = \
                    np.append(self._data_set_for_fitting['fit_depth_points'][0],
                              np.append(np.append(selected_d, d), self._data_set_for_fitting['fit_depth_points'][-1]))

                self._voltset[:, self._fill_counter] = selected_v
                self._fill_counter += 1

        else:
            self._depthset[:, self._fill_counter] = [self._data_set_for_fitting['fit_depth_points'][0],
                                                                      self._data_set_for_fitting['fit_depth_points'][-1]]
            self._voltset[:, self._fill_counter] = selected_v
            self._fill_counter += 1

    # ----------------------------------------------------------------------
    def _analyse_results(self, v_set, d_set):

        solution_found = True
        self.best_ksi.append(np.min(self._mse))

        best_ksi_ind = np.argmin(self._mse)
        depth_set = self._depthset[:, best_ksi_ind]
        volt_set = self._voltset[:, best_ksi_ind]

        self.solution_history.append(np.vstack((depth_set, volt_set)))

        self.potential_graphs_history.append(calculatePotential(depth_set, volt_set, self._data_set_for_fitting['fit_depth_points'],
                                                                        self._potential_model['code']))

        last_best_shifts, _ = get_shifts(self._data_set_for_fitting, depth_set, volt_set)

        self.shifts_graphs_history.append(last_best_shifts)

        for point in range(self._potential_model['num_depth_dof'] + self._potential_model['only_voltage_dof']):
            new_set, statistics, parameter_solution_found = \
                self._analyse_variable_statistic(v_set[point, :], self._voltset[point, :],
                                                 'volt_point', [-self._settings['VOLT_MAX'],
                                                                self._settings['VOLT_MAX']])

            full_statistics = np.vstack((v_set[point, :], statistics))
            self.v_graphs_history[point].append(full_statistics)
            v_set[point, :] = new_set
            solution_found *= parameter_solution_found

        for point in range(self._potential_model['num_depth_dof']):
            new_set, statistics, parameter_solution_found = \
                self._analyse_variable_statistic(d_set[point, :], self._depthset[point + 1, :],
                                                 'depth_point', [self._data_set_for_fitting['fit_depth_points'][0],
                                                 self._data_set_for_fitting['fit_depth_points'][-1]])

            full_statistics = np.vstack((d_set[point, :] - self._data_set_for_fitting['fit_depth_points'][0], statistics))
            self.d_graphs_history[point].append(full_statistics)
            d_set[point, :] = new_set

            solution_found *= parameter_solution_found

        return solution_found, v_set, d_set

     # ----------------------------------------------------------------------
    def _analyse_variable_statistic(self, variable_set, data_cut, mode, limits):

        pre_set = np.zeros(3)
        ksiset = np.zeros((len(variable_set), 2))

        if mode == 'volt_point':
            mesh = self._settings['V_MESH']
            step = self._settings['V_STEP']
        else:
            mesh = self._settings['D_MESH']
            step = self._settings['D_STEP']*1e-9

        limits[0] += step
        limits[1] -= step

        for v_point in range(len(variable_set)):
            all_inds = np.nonzero(data_cut == variable_set[v_point])
            ksiset[v_point, 0] = variable_set[v_point]
            ksiset[v_point, 1] = np.min(self._mse[all_inds])

        t_statistics = np.sqrt(ksiset[:, 1] - self.best_ksi[-1]) / np.sqrt(
            self.best_ksi[-1] / self._data_set_for_fitting['spectroscopic_data'].shape[0])

        if self.best_ksi[-1]*(1 + self._settings['KSI_TOLLERANCE']) < self.best_ksi[-2]:
            half_steps = int(np.floor(mesh / 2))
            pre_set = ksiset[np.argmin(ksiset[:, 1]), 0] + np.linspace(-step * half_steps, step * half_steps, mesh)
            calculated_set = [i for i in pre_set if limits[0] < i < limits[1]]
            solution_found = False
        else:
            solution_found = True
            region_of_interest = np.nonzero(t_statistics < self.t_val)
            lowend = region_of_interest[0][0]
            upend = region_of_interest[0][-1]

            if lowend == 0:
                pre_set[0] = variable_set[0] - step
                if pre_set[0] < limits[0]:
                    pre_set[0] = limits[0]
                else:
                    solution_found *= False
            elif lowend == mesh - 1:
                pre_set[0] = variable_set[-1]
                solution_found *= False
            else:
                pre_set[0] = np.interp(self.t_val, t_statistics[lowend - 1:lowend + 1],
                                       ksiset[lowend - 1:lowend + 1, 0], period=np.inf)

            pre_set[1] = ksiset[np.argmin(ksiset[:, 1]), 0]

            if upend == 0:
                pre_set[2] = variable_set[0]
                solution_found *= False
            elif upend == mesh - 1:
                pre_set[2] = variable_set[-1] + step
                if pre_set[2] > limits[1]:
                    pre_set[2] = limits[1]
                else:
                    solution_found *= False
            else:
                pre_set[2] = np.interp(self.t_val, t_statistics[upend:upend + 2],
                                       ksiset[upend:upend + 2, 0], period=np.inf)

            calculated_set = np.append(np.linspace(pre_set[0], pre_set[1], int(np.ceil(mesh / 2))),
                                       np.linspace(pre_set[1], pre_set[2], int(np.ceil(mesh / 2)))[1:])

        return np.array(calculated_set), np.array(t_statistics), solution_found

    # ----------------------------------------------------------------------
    def _mse_calculator(self, data_set_for_fitting, volt_set, depth_set, tasks_queue,
                        result_array, num_local_tasks, start_task_index):

        while True:
            local_job_range = tasks_queue.get()
            print('Mse calculator tasks {}'.format(local_job_range))
            if local_job_range == None:
                break

            mse_list = np.reshape(np.frombuffer(result_array), num_local_tasks)

            for ind in range(local_job_range[0], local_job_range[1]):
                shifts, _ = get_shifts(data_set_for_fitting, depth_set[:, ind], volt_set[:, ind])
                if shifts is not None:
                    shifts -= data_set_for_fitting['spectroscopic_data'][:, 2]
                    mse_list[ind - start_task_index] = np.inner(shifts, shifts)
                else:
                    mse_list[ind - start_task_index] = 1e6

            tasks_queue.task_done()

        tasks_queue.task_done()

    # ----------------------------------------------------------------------
    def _node_worker(self, local_task_list):

        print('Node worker got tasks {}'.format(local_task_list))

        if self._settings['MULTIPROCESSING']:
            node_cpu = mp.cpu_count()
        else:
            node_cpu = 1
        node_task_sets = self._settings['N_SUB_JOBS'] * node_cpu
        num_local_tasks = local_task_list[1] - local_task_list[0]

        mse_list = mp.RawArray(ctypes.c_double, num_local_tasks)

        stop_points = np.ceil(np.linspace(local_task_list[0], local_task_list[1], node_task_sets + 1))
        jobs_list = []

        for i in range(node_task_sets):
            jobs_list.append((int(stop_points[i]), int(min(stop_points[i + 1], local_task_list[1]))))

        jobs_queue = mp.JoinableQueue()
        for job in jobs_list:
            jobs_queue.put(job)
        for _ in range(node_cpu):
            jobs_queue.put(None)

        workers = []
        for i in range(node_cpu):
            worker = mp.Process(target=self._mse_calculator,
                                args=(self._data_set_for_fitting, self._voltset, self._depthset,
                                      jobs_queue, mse_list, num_local_tasks, local_task_list[0]))
            workers.append(worker)
            worker.start()
        print('All workers started')
        jobs_queue.join()

        return np.reshape(np.frombuffer(mse_list), num_local_tasks)

    # ----------------------------------------------------------------------
    def abort(self):
        self.fit_in_progress = False

    # ----------------------------------------------------------------------
    def _save_data(self):

        if not "results" in os.listdir(self._directory):
            os.mkdir(os.path.join(self._directory, "results"))

        file_name = '{}.res'.format(self._sample_name)
        full_path = os.path.join(os.path.join(self._directory, "results"), file_name)

        data_set = {}
        for key in self._keys_to_save:
            data_set[key] = getattr(self, key)

        with open(full_path, 'wb') as f:
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    # ----------------------------------------------------------------------
    def load_data(self, file_name):

        with open(file_name, 'rb') as fr:
            loaded_data = pickle.load(fr)
            for key in self._keys_to_save:
                setattr(self, key, loaded_data[key])

    # ----------------------------------------------------------------------
    def load_fit_set(self, file_name):

        with open(file_name, 'rb') as fr:
            loaded_data = pickle.load(fr)
            for key in self._key_to_set:
                setattr(self, key, loaded_data[key])

        self.prepare_variables()

    # ----------------------------------------------------------------------
    def dump_fit_set(self, file_name):

        data_set = {}
        for key in self._key_to_set:
            data_set[key] = getattr(self, key)

        with open(file_name, 'wb') as f:
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    # ----------------------------------------------------------------------
    def do_fit(self, cycles=np.Inf):

        self.fit_in_progress = True
        result_found = False

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        v_set, d_set = self._generate_start_set()

        if rank == 0:
            v_set, d_set = self._fill_start_set(v_set, d_set)

        while self.fit_in_progress and not result_found and self.cycle<cycles:
            start_time = time.time()

            v_set = np.array(json.loads(comm.bcast([json.dumps(v_set.tolist())], root=0)[0]))
            d_set = np.array(json.loads(comm.bcast([json.dumps(d_set.tolist())], root=0)[0]))
            self._get_fit_set(v_set, d_set)

            print('Fit set preparation time: {}'.format(time.time() - start_time))

            start_time = time.time()

            print('Total tasks {}'.format(self._fit_points))
            num_cores = comm.gather(mp.cpu_count(), root=0)

            if rank == 0:
                print("-- Got the following cpus set: {}".format(num_cores))
                total_cpus = int(np.sum(num_cores))
                task_list_per_cpu = np.ceil(np.linspace(0, self._fit_points, total_cpus + 1))
                start_points = np.append(np.array([0]), np.cumsum(num_cores))
                jobs_list = []

                for i in range(size):
                    jobs_list.append(json.dumps((int(task_list_per_cpu[start_points[i]]),
                                                 int(task_list_per_cpu[start_points[i + 1]]))))
            else:
                jobs_list = None

            recvbuf = comm.scatter(jobs_list, root=0)
            local_result = self._node_worker(json.loads(recvbuf))
            final_result = comm.gather(local_result, root=0)

            if rank == 0:
                self._mse = np.concatenate(final_result)
                error_cycle_time = time.time() - start_time
                try:
                    time_pre_point = np.round(error_cycle_time / self._fit_points, 6)
                except:
                    time_pre_point = 0

                print('Error calculation time: {}, time per point: {}'.format(error_cycle_time, time_pre_point))
                result_found, v_set, d_set = self._analyse_results(v_set, d_set)
            else:
                v_set, d_set = self._generate_start_set()
                result_found = False

            result_found = json.loads(comm.bcast([json.dumps(result_found)], root=0)[0])

            if rank == 0:
                cycle = self.cycle +  1
                self._save_data()
                print('Cycle {} completed'.format(self.cycle))
            else:
                cycle = None

            self.cycle = json.loads(comm.bcast([json.dumps(cycle)], root=0)[0])