from src.ntr_data_fitting.subfunctions import *
from src.ntr_data_fitting.potential_models import calculatePotential
import matplotlib.pyplot as plt
import ctypes
import json
from scipy import stats
import multiprocessing as mp
from mpi4py import MPI
import look_and_feel as lookandfeel
import time

class Gradient_Mesh_Solver():

    settings = {}

    # ----------------------------------------------------------------------
    def __init__(self, parent):

        self.parent = parent

        self.local_data_set = {}

        self.best_ksi = [np.inf]
        self._fill_counter = 0

        self.solution_history = []

        self.v_graphs_history = []
        self.d_graphs_history = []
        self.potential_graphs_history = []
        self.shifts_graphs_history = []

        self.v_graphs_stack = []
        self.d_graphs_stack = []
        self.potential_graph = None
        self.shifts_graph = None

        self.fig = None
        self.axes = None
        self.plots = []

        self.v_set = []
        self.d_set = []

    # ----------------------------------------------------------------------
    def reset_fit(self):

        self.best_ksi = [np.inf]

        self.solution_history = []

        self.v_graphs_history = [[] for _ in range(self.parent.potential_model['num_depth_dof'] +
                                                   self.parent.potential_model['only_voltage_dof'])]
        self.d_graphs_history = [[] for _ in range(self.parent.potential_model['num_depth_dof'])]
        self.potential_graphs_history = []
        self.shifts_graphs_history = []

        self.v_graphs_stack = []
        self.d_graphs_stack = []
        self.potential_graph = None
        self.shifts_graph = None

        if self.parent.STAND_ALONE_MODE and self.parent.DO_PLOT:
            total_num_plots = self.parent.potential_model['only_voltage_dof'] + \
                              2 * self.parent.potential_model['num_depth_dof'] + 2
            self.num_colums = total_num_plots // 2

            self.fig, self.axes = plt.subplots(nrows=2, ncols=self.num_colums)
            self.plots = [[] for _ in range(total_num_plots)]
            plt.ion()

    # ----------------------------------------------------------------------
    def _get_fit_set(self):
        num_dof = self.parent.potential_model['num_depth_dof'] + self.parent.potential_model['only_voltage_dof']

        fit_points = 1

        for v_points in self.v_set:
            fit_points *= len(v_points)

        for d_points in self.d_set:
            fit_points *= len(d_points)

        self.local_data_set = {'fit_points': fit_points,
                    'depthset': np.zeros((num_dof, fit_points)),
                    'voltset': np.zeros((num_dof, fit_points)),
                    'mse': np.zeros(fit_points),
                    'last_best_shifts': np.zeros_like(self.parent.data_set_for_fitting['spectroscopic_data'][:, 2]),
                    'last_intensity': np.zeros_like(self.parent.data_set_for_fitting['spectroscopic_data'][:, 1]),
                    'last_best_potential': np.zeros_like(num_dof),
                    'statistics': {},
                    't_val': stats.t.ppf(1 - self.parent.settings['T'], self.parent.data_set_for_fitting['spectroscopic_data'].shape[0])}

        self._fill_counter = 0
        self._fill_voltages([], self.v_set, self.d_set)

    # ----------------------------------------------------------------------
    def set_external_graphs(self, graphs_layout):

        for ind in range(self.parent.potential_model['num_depth_dof'] + self.parent.potential_model['only_voltage_dof']):
            plot_axes = graphs_layout.addPlot(title="V_{}".format(ind), row=0, col=ind)
            plot_axes.setLabel('bottom', 'BE, eV')
            plot_axes.setLabel('left', 't statistics')
            plot_axes.setYRange(0, 3)
            self.v_graphs_stack.append([plot_axes.plot(range(self.parent.settings['V_MESH']),
                                                                        np.ones(self.parent.settings['V_MESH']),
                                                                        **lookandfeel.MAX_T_STYLE),
                                        plot_axes.plot(range(self.parent.settings['V_MESH']),
                                                                        np.zeros(self.parent.settings['V_MESH']),
                                                                        **lookandfeel.T_STAT_STYLE)])

        ind = 0
        for ind in range(self.parent.potential_model['num_depth_dof']):
            plot_axes = graphs_layout.addPlot(title="D_{}".format(ind), row=1, col=ind)
            plot_axes.setLabel('bottom', 'Depth, nm')
            plot_axes.setLabel('left', 't statistics')
            plot_axes.setYRange(0, 3)
            self.d_graphs_stack.append([plot_axes.plot(range(self.parent.settings['D_MESH']),
                                                                        np.ones(self.parent.settings['D_MESH']),
                                                                        **lookandfeel.MAX_T_STYLE),
                                        plot_axes.plot(range(self.parent.settings['D_MESH']),
                                                                        np.zeros(self.parent.settings['D_MESH']),
                                                                        **lookandfeel.T_STAT_STYLE)])

        if self.parent.potential_model['num_depth_dof']:
            start_ind = ind + 1
        else:
            start_ind = 0

        shifts_plot = graphs_layout.addPlot(title="Shifts", row=1, col=start_ind + 1)
        shifts_plot.plot(self.parent.data_set_for_fitting['spectroscopic_data'][:, 0],
                         self.parent.be_shift + self.parent.data_set_for_fitting['spectroscopic_data'][:, 2],
                         **lookandfeel.CURRENT_SOURCE_SHIFT)

        self.shifts_graph = shifts_plot.plot(self.parent.data_set_for_fitting['spectroscopic_data'][:, 0],
                                             self.parent.data_set_for_fitting['spectroscopic_data'][:, 2],
                                             **lookandfeel.CURRENT_SIM_SHIFT)

        potential_plot = graphs_layout.addPlot(title="Potential", row=1, col=start_ind)
        self.potential_graph = potential_plot.plot((self.parent.data_set_for_fitting['fit_depth_points']
                                                    - self.parent.data_set_for_fitting['fit_depth_points'][0])*1e9,
                                                    np.zeros_like(self.parent.data_set_for_fitting['fit_depth_points']),
                                                    **lookandfeel.CURRENT_POTENTIAL_STYLE)

    # ----------------------------------------------------------------------
    def prepare_stand_alone_plots(self):

        for ind in range(self.parent.potential_model['num_depth_dof'] + self.parent.potential_model['only_voltage_dof']):
            self.axes[0, ind].set_title('V point {}'.format(ind + 1))
            self.axes[0, ind].set_xlabel('BE, eV')
            self.axes[0, ind].set_ylabel('t statistics')
            self.axes[0, ind].set_ylim([0, 3])
            self.v_graphs_stack.append([self.axes[0, ind],
                                        self.axes[0, ind].plot(range(self.parent.settings['V_MESH']),
                                                               np.ones(self.parent.settings['V_MESH']), 'g--')[0],
                                        self.axes[0, ind].plot(range(self.parent.settings['V_MESH']),
                                                               np.zeros(self.parent.settings['V_MESH']), '*-')[0]])

        for ind in range(self.parent.potential_model['num_depth_dof']):
            self.axes[1, ind].set_title('D point {}'.format(ind + 1))
            self.axes[1, ind].set_xlabel('Depth, nm')
            self.axes[1, ind].set_ylabel('t statistics')
            self.axes[1, ind].set_ylim([0, 3])
            self.d_graphs_stack.append([self.axes[1, ind],
                                        self.axes[1, ind].plot(range(self.parent.settings['D_MESH']),
                                                               np.ones(self.parent.settings['D_MESH']), 'g--')[0],
                                        self.axes[1, ind].plot(range(self.parent.settings['D_MESH']),
                                                               np.zeros(self.parent.settings['D_MESH']), '*-')[0]])

        self.axes[1, self.num_colums - 1].plot(self.parent.data_set_for_fitting['spectroscopic_data'][:, 0],
                                               self.parent.be_shift + self.parent.data_set_for_fitting['spectroscopic_data'][:, 2], 'x')
        self.shifts_graph = [self.axes[1, self.num_colums - 1],
                             self.axes[1, self.num_colums - 1].plot(self.parent.data_set_for_fitting['spectroscopic_data'][:, 0],
                                                                    self.parent.data_set_for_fitting['spectroscopic_data'][:, 2], '-')[0]]
        self.potential_graph = [self.axes[1, self.num_colums - 2],
                                self.axes[1, self.num_colums - 2].plot((self.parent.data_set_for_fitting['fit_depth_points']
                                                                        - self.parent.data_set_for_fitting['fit_depth_points'][0])*1e9,
                                                                        np.zeros_like(self.parent.data_set_for_fitting['fit_depth_points']))[0]]

        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(1)
        plt.show()

    # ----------------------------------------------------------------------
    def _plot_result(self):
        best_ksi_ind = np.argmin(self.local_data_set['mse'])
        depth_set = self.local_data_set['depthset'][:, best_ksi_ind]
        volt_set = self.local_data_set['voltset'][:, best_ksi_ind]

        self.solution_history.append(np.vstack((depth_set, volt_set)))

        self.local_data_set['last_best_potential'] = calculatePotential(depth_set, volt_set,
                                                                        self.parent.data_set_for_fitting['fit_depth_points'],
                                                                        self.parent.potential_model['code'])

        self.potential_graphs_history.append(self.local_data_set['last_best_potential'])

        self.local_data_set['last_best_shifts'], self.local_data_set['last_intensity'] = \
            get_shifts(self.parent.data_set_for_fitting, depth_set, volt_set)

        self.shifts_graphs_history.append(self.local_data_set['last_best_shifts'])

        if self.parent.DO_PLOT:
            if self.parent.STAND_ALONE_MODE:
                self.potential_graph[1].set_ydata(self.local_data_set['last_best_potential'])
                self.potential_graph[0].relim()
                self.potential_graph[0].autoscale_view()

                self.shifts_graph[1].set_ydata(self.local_data_set['last_best_shifts'])
                self.shifts_graph[0].relim()
                self.shifts_graph[0].autoscale_view()
            else:
                self.potential_graph.setData((self.parent.data_set_for_fitting['fit_depth_points'] -
                                              self.parent.data_set_for_fitting['fit_depth_points'][0])*1e9,
                                             self.local_data_set['last_best_potential'])
                self.shifts_graph.setData(self.parent.data_set_for_fitting['spectroscopic_data'][:, 0],
                                          self.local_data_set['last_best_shifts'])

    # ----------------------------------------------------------------------
    def _generate_start_set(self, start_values):

        self.v_set = [np.zeros(self.parent.settings['V_MESH'])
                      for _ in range(self.parent.potential_model['num_depth_dof'] +
                      self.parent.potential_model['only_voltage_dof'])]

        v_half_steps = int(np.floor(self.parent.settings['V_MESH'] / 2))

        for point in range(self.parent.potential_model['num_depth_dof'] +
                           self.parent.potential_model['only_voltage_dof']):

            self.v_set[point] = start_values[point][1] + np.linspace(-self.parent.settings['V_STEP'] * v_half_steps,
                                                                     self.parent.settings['V_STEP'] * v_half_steps,
                                                                     self.parent.settings['V_MESH'])

        self.d_set = [np.zeros(self.parent.settings['D_MESH'])
                      for _ in range(self.parent.potential_model['num_depth_dof'])]

        d_half_steps = int(np.floor(self.parent.settings['D_MESH'] / 2))

        d_min = self.parent.data_set_for_fitting['fit_depth_points'][0] + self.parent.settings['D_STEP']*1e-9
        d_max = self.parent.data_set_for_fitting['fit_depth_points'][-1] - self.parent.settings['D_STEP']*1e-9

        for point in range(self.parent.potential_model['num_depth_dof']):
            self.d_set[point] = start_values[point + 1][0] + np.linspace(-self.parent.settings['D_STEP'] * d_half_steps,
                                                                     self.parent.settings['D_STEP'] * d_half_steps,
                                                                     self.parent.settings['D_MESH']) * 1e-9

            if self.d_set[point][0] < d_min:
                self.d_set[point] += d_min - self.d_set[point][0]

            if self.d_set[point][-1] > d_max:
                self.d_set[point] -= self.d_set[point][-1] - d_max

    # ----------------------------------------------------------------------
    def _fill_voltages(self, selected_v, last_v_set, d_set):
        if len(last_v_set) > 1:
            for v in last_v_set[0]:
                self._fill_voltages(np.append(selected_v, v), last_v_set[1:], d_set)
        else:
            for v in last_v_set[0]:
                self._fill_depth(np.append(selected_v, v), [], d_set)

    # ----------------------------------------------------------------------
    def _fill_depth(self, selected_v, selected_d, last_d_set):
        if len(last_d_set) > 1:
            for d in last_d_set[0]:
                self._fill_depth(selected_v, np.append(selected_d, d), last_d_set[1:])
        elif len(last_d_set) == 1:
            for d in last_d_set[0]:
                self.local_data_set['depthset'][:, self._fill_counter] = \
                    np.append(self.parent.data_set_for_fitting['fit_depth_points'][0],
                              np.append(np.append(selected_d, d), self.parent.data_set_for_fitting['fit_depth_points'][-1]))

                self.local_data_set['voltset'][:, self._fill_counter] = selected_v
                self._fill_counter += 1

        else:
            self.local_data_set['depthset'][:, self._fill_counter] = [self.parent.data_set_for_fitting['fit_depth_points'][0],
                                                                      self.parent.data_set_for_fitting['fit_depth_points'][-1]]
            self.local_data_set['voltset'][:, self._fill_counter] = selected_v
            self._fill_counter += 1

    # ----------------------------------------------------------------------
    def _analyse_results(self):
        solution_found = True
        self.best_ksi.append(np.min(self.local_data_set['mse']))

        self.local_data_set['statistics'] = {"V_points":
                                                 [[] for _ in range(self.parent.potential_model['num_depth_dof'] +
                                                                    self.parent.potential_model['only_voltage_dof'])],
                                            "D_points":
                                                [[] for _ in range(self.parent.potential_model['num_depth_dof'])]}

        for point in range(self.parent.potential_model['num_depth_dof'] +
                                                                    self.parent.potential_model['only_voltage_dof']):
            new_set, statistics, parameter_solution_found = \
                self._analyse_variable_statistic(self.v_set[point], self.local_data_set['voltset'][point, :],
                                                 'volt_point', [-self.parent.settings['VOLT_MAX'],
                                                                self.parent.settings['VOLT_MAX']])

            if self.parent.DO_PLOT:
                self.plot_statistics(self.v_graphs_stack[point], self.v_set[point], statistics)
            full_statistics = np.vstack((self.v_set[point], statistics))
            self.local_data_set['statistics']["V_points"][point] = full_statistics
            self.v_graphs_history[point].append(full_statistics)
            self.v_set[point] = new_set
            solution_found *= parameter_solution_found

        for point in range(self.parent.potential_model['num_depth_dof']):
            new_set, statistics, parameter_solution_found = \
                self._analyse_variable_statistic(self.d_set[point], self.local_data_set['depthset'][point + 1, :],
                                                 'depth_point', [self.parent.data_set_for_fitting['fit_depth_points'][0],
                                                 self.parent.data_set_for_fitting['fit_depth_points'][-1]])

            if self.parent.DO_PLOT:
                self.plot_statistics(self.d_graphs_stack[point], self.d_set[point] -
                                     self.parent.data_set_for_fitting['fit_depth_points'][0], statistics)

            full_statistics = np.vstack((self.d_set[point] - self.parent.data_set_for_fitting['fit_depth_points'][0], statistics))
            self.local_data_set['statistics']["D_points"][point] = full_statistics
            self.d_graphs_history[point].append(full_statistics)
            self.d_set[point] = new_set

            solution_found *= parameter_solution_found

        return solution_found

    # ----------------------------------------------------------------------
    def plot_statistics(self, graphs, var_set, statistics):

        if self.parent.STAND_ALONE_MODE:
            if self.parent.DO_PLOT:
                graphs[1].set_xdata(var_set)
                graphs[1].set_ydata(np.ones_like(var_set)*self.local_data_set['t_val'])
                graphs[2].set_xdata(var_set)
                graphs[2].set_ydata(statistics)
                graphs[0].relim()
                graphs[0].autoscale_view()
        else:
            graphs[0].setData(var_set, np.ones_like(var_set)*self.local_data_set['t_val'])
            graphs[1].setData(var_set, statistics)

    # ----------------------------------------------------------------------
    def _analyse_variable_statistic(self, variable_set, data_cut, mode, limits):

        pre_set = np.zeros(3)
        ksiset = np.zeros((len(variable_set), 2))

        if mode == 'volt_point':
            mesh = self.parent.settings['V_MESH']
            step = self.parent.settings['V_STEP']
        else:
            mesh = self.parent.settings['D_MESH']
            step = self.parent.settings['D_STEP']*1e-9

        limits[0] += step
        limits[1] -= step

        for v_point in range(len(variable_set)):
            all_inds = np.nonzero(data_cut == variable_set[v_point])
            ksiset[v_point, 0] = variable_set[v_point]
            ksiset[v_point, 1] = np.min(self.local_data_set['mse'][all_inds])

        t_statistics = np.sqrt(ksiset[:, 1] - self.best_ksi[-1]) / np.sqrt(
            self.best_ksi[-1] / self.parent.data_set_for_fitting['spectroscopic_data'].shape[0])

        if self.best_ksi[-1]*(1 + self.parent.settings['KSI_TOLLERANCE']) < self.best_ksi[-2]:
            half_steps = int(np.floor(mesh / 2))
            pre_set = ksiset[np.argmin(ksiset[:, 1]), 0] + np.linspace(-step * half_steps, step * half_steps, mesh)
            calculated_set = [i for i in pre_set if limits[0] < i < limits[1]]
            solution_found = False
        else:
            solution_found = True
            region_of_interest = np.nonzero(t_statistics < self.local_data_set['t_val'])
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
                pre_set[0] = np.interp(self.local_data_set['t_val'], t_statistics[lowend - 1:lowend + 1],
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
                pre_set[2] = np.interp(self.local_data_set['t_val'], t_statistics[upend:upend + 2],
                                       ksiset[upend:upend + 2, 0], period=np.inf)

            calculated_set = np.append(np.linspace(pre_set[0], pre_set[1], int(np.ceil(mesh / 2))),
                                       np.linspace(pre_set[1], pre_set[2], int(np.ceil(mesh / 2)))[1:])

        return np.array(calculated_set), np.array(t_statistics), solution_found

    # ----------------------------------------------------------------------
    def _mse_calculator(self, data_set_for_fitting, local_data_set, tasks_queue,
                        result_array, num_local_tasks, start_task_index):

        while True:
            local_job_range = tasks_queue.get()
            # print('Mse calculator tasks {}'.format(local_job_range))
            if local_job_range == None:
                break

            mse_list = np.reshape(np.frombuffer(result_array), num_local_tasks)

            for ind in range(local_job_range[0], local_job_range[1]):
                depth_set = local_data_set['depthset'][:, ind]
                volt_set = local_data_set['voltset'][:, ind]
                shifts, _ = get_shifts(data_set_for_fitting, depth_set, volt_set)
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

        node_cpu = mp.cpu_count()
        node_task_sets = self.parent.settings['N_SUB_JOBS'] * node_cpu
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
                                args=(self.parent.data_set_for_fitting, self.local_data_set,
                                      jobs_queue, mse_list, num_local_tasks, local_task_list[0]))
            workers.append(worker)
            worker.start()
        print('All workers started')
        jobs_queue.join()

        return np.reshape(np.frombuffer(mse_list), num_local_tasks)

    # ----------------------------------------------------------------------
    def get_data_for_save(self):

        return self.local_data_set
    # ----------------------------------------------------------------------
    def do_fit(self, start_values):

        worker_start_time = time.time()
        self._generate_start_set(start_values)
        self.cycle = 0
        result_found = False

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        while self.parent.fit_in_progress and not result_found:
            start_time = time.time()
            self._get_fit_set()

            print('Fit set preparation time: {}'.format(time.time() - start_time))

            start_time = time.time()

            if self.parent.settings['MULTIPROCESSING']:

                print('Total tasks {}'.format(self.local_data_set['fit_points']))
                num_cores = comm.gather(mp.cpu_count(), root=0)

                if rank == 0:
                    print("-- Got the following cpus set: {}".format(num_cores))
                    total_cpus = int(np.sum(num_cores))
                    task_list_per_cpu = np.ceil(np.linspace(0, self.local_data_set['fit_points'], total_cpus + 1))
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
                    self.local_data_set['mse'] = np.concatenate(final_result)
            else:
                for ind in range(self.local_data_set['fit_points']):
                    depth_set = self.local_data_set['depthset'][:, ind]
                    volt_set = self.local_data_set['voltset'][:, ind]
                    shifts, _ = get_shifts(self.parent.data_set_for_fitting, depth_set, volt_set)
                    if shifts is not None:
                        shifts -= self.parent.data_set_for_fitting['spectroscopic_data'][:, 2]
                        self.local_data_set['mse'][ind] = np.inner(shifts, shifts)
                    else:
                        self.local_data_set['mse'][ind] = 1e6

            if rank == 0:
                error_cycle_time = time.time() - start_time
                try:
                    time_pre_point = np.round(error_cycle_time / self.local_data_set['fit_points'], 6)
                except:
                    time_pre_point = 0

                print('Error calculation time: {}, time per point: {}'.format(error_cycle_time, time_pre_point))

                start_time = time.time()
                self._plot_result()
                result_found = self._analyse_results()
                self.parent.save_fit_res()
                print('Plot and save time: {}'.format(time.time() - start_time))

                self.cycle += 1
                print('Cycle {} completed'.format(self.cycle))

                if self.parent.STAND_ALONE_MODE:
                    if self.parent.DO_PLOT:
                        plt.draw()
                        plt.gcf().canvas.flush_events()
                else:
                    self.parent.gui.update_potential_fit_cycles(self.cycle, self.best_ksi[self.cycle], self.solution_history[self.cycle - 1])

        self.cycle -= 1
        plt.ioff()
        plt.show()

    # ----------------------------------------------------------------------
    def show_results(self, idx):

        ind = 0
        if self.cycle > 0:
            ind = int(min(self.cycle, idx))

        if len(self.shifts_graphs_history) > 0:
            if self.parent.STAND_ALONE_MODE:
                if self.parent.DO_PLOT:
                    self.potential_graph[1].set_ydata(self.potential_graphs_history[ind])
                    self.potential_graph[0].relim()
                    self.potential_graph[0].autoscale_view()

                    self.shifts_graph[1].set_ydata(self.shifts_graphs_history[ind])
                    self.shifts_graph[0].relim()
                    self.shifts_graph[0].autoscale_view()

                    for point in range(self.parent.potential_model['num_depth_dof'] +
                                                                    self.parent.potential_model['only_voltage_dof']):
                        self.v_graphs_stack[point][1].set_xdata(self.v_graphs_history[point][ind][0, :])
                        self.v_graphs_stack[point][2].set_xdata(self.v_graphs_history[point][ind][0, :])
                        self.v_graphs_stack[point][2].set_ydata(self.v_graphs_history[point][ind][1, :])
                        self.v_graphs_stack[point][0].relim()
                        self.v_graphs_stack[point][0].autoscale_view()

                    for point in range(self.parent.potential_model['num_depth_dof']):
                        self.d_graphs_stack[point][1].set_xdata(self.d_graphs_history[point][ind][0, :])
                        self.d_graphs_stack[point][2].set_xdata(self.d_graphs_history[point][ind][0, :])
                        self.d_graphs_stack[point][2].set_ydata(self.d_graphs_history[point][ind][1, :])
                        self.d_graphs_stack[point][0].relim()
                        self.d_graphs_stack[point][0].autoscale_view()
            else:
                self.potential_graph.setData((self.parent.data_set_for_fitting['fit_depth_points']
                                              - self.parent.data_set_for_fitting['fit_depth_points'][0])*1e9,
                                             self.potential_graphs_history[ind])
                self.shifts_graph.setData(self.parent.data_set_for_fitting['spectroscopic_data'][:, 0],
                                          self.parent.be_shift + self.shifts_graphs_history[ind])

                for point in range(self.parent.potential_model['num_depth_dof'] +
                                                                    self.parent.potential_model['only_voltage_dof']):
                    self.v_graphs_stack[point][0].setData(self.v_graphs_history[point][ind][0, :],
                                                          np.ones_like(self.v_graphs_history[point][ind][0, :])
                                                          * self.local_data_set['t_val'])
                    self.v_graphs_stack[point][1].setData(self.v_graphs_history[point][ind][0, :],
                                                          self.v_graphs_history[point][ind][1, :])

                for point in range(self.parent.potential_model['num_depth_dof']):
                    self.d_graphs_stack[point][0].setData(self.d_graphs_history[point][ind][0, :],
                                                          np.ones_like(self.d_graphs_history[point][ind][0, :])
                                                          * self.local_data_set['t_val'])
                    self.d_graphs_stack[point][1].setData(self.d_graphs_history[point][ind][0, :],
                                                          self.d_graphs_history[point][ind][1, :])

                return self.best_ksi[ind+1], self.solution_history[ind]

    # ----------------------------------------------------------------------
    def load_fit_res(self, loaded_data):

        self.cycle += 1
        self.local_data_set['t_val'] = loaded_data['t_val']
        self.best_ksi.append(np.min(loaded_data['mse']))
        self.potential_graphs_history.append(loaded_data['last_best_potential'])
        self.shifts_graphs_history.append(loaded_data['last_best_shifts'])
        min_ind = np.argmin(loaded_data['mse'])
        self.solution_history.append(
            np.vstack((loaded_data['depthset'][:, min_ind], loaded_data['voltset'][:, min_ind])))

        for point in range(self.parent.potential_model['num_depth_dof'] +
                                                                    self.parent.potential_model['only_voltage_dof']):
            self.v_graphs_history[point].append(loaded_data['statistics']["V_points"][point])

        for point in range(self.parent.potential_model['num_depth_dof']):
            self.d_graphs_history[point].append(loaded_data['statistics']["D_points"][point])
