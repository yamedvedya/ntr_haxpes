#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser

from subfunctions import *
from potential_models import calculatePotential
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
import pickle
import os
import ctypes
import multiprocessing as mp
import look_and_feel as lookandfeel

class NTR_fitter():

    STAND_ALONE_MODE = True
    settings = {}
    METHODS = ('mesh_gradient', 'gradient')

    # ----------------------------------------------------------------------
    def __init__(self):

        self.best_ksi = [np.inf]
        self._fill_counter = 0
        self.settings = {}

        self.main_data_set = {}
        self._sample_name = None
        self._directory = None
        self.structure = None
        self.angle_shift = None
        self.be_shift = None
        self.t_val = None
        self.sw = None
        self.num_depth_points = None

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
    def set_basic_settings(self, settings):

        for key, value in settings.items():
            self.settings[key] = value

        self.form_basic_data()
    # ----------------------------------------------------------------------
    def form_basic_data(self):

        self.main_data_set['FIELD_MAX'] = self.settings['FIELD_MAX']
        self.main_data_set['SUB_LAYERS'] = self.settings['SUB_LAYERS']
        self.main_data_set['BE_STEP'] = self.settings['BE_STEP']

        self.main_data_set['ref_spectra'] = syn_spectra(-(self.settings['VOLT_MAX'] + self.settings['SIM_SPECTRA_WIDTH']),
                                                        self.settings['VOLT_MAX'] + self.settings['SIM_SPECTRA_WIDTH'],
                                                        self.settings['BE_STEP'], self.settings['G'], self.settings['L'])
        self.main_data_set['ref_spectra_points'] = int((self.settings['VOLT_MAX'] +
                                                        self.settings['SIM_SPECTRA_WIDTH'])/self.settings['BE_STEP'])

        self.main_data_set['sum_spectra_point'] = int(self.settings['SIM_SPECTRA_WIDTH']/self.settings['BE_STEP'])
        self.main_data_set['sum_spectra_energy'] = np.linspace(-self.settings['SIM_SPECTRA_WIDTH'],
                                                               self.settings['SIM_SPECTRA_WIDTH'],
                                                                2*self.main_data_set['sum_spectra_point'] + 1)

    # ----------------------------------------------------------------------
    def set_new_data_file(self, data_file, set):

        self._sample_name = os.path.splitext(os.path.basename(data_file))[0]
        self._directory = os.path.dirname(data_file)
        self.main_data_set['data'], self.structure, self.angle_shift = get_data_set(data_file, set)
        self.main_data_set['fit_depth_points'] = np.linspace(self.structure[0], self.structure[0] + self.structure[1],
                                                             self.settings['SUB_LAYERS'])

        self.main_data_set['data'][:, 0] += self.angle_shift

        self.be_shift = np.mean(self.main_data_set['data'][:, 2])
        self.main_data_set['data'][:, 2] -= self.be_shift
        self.t_val = stats.t.ppf(1 - self.settings['T'], self.main_data_set['data'].shape[0])

    # ----------------------------------------------------------------------
    def set_new_sw_file(self, sw_file):

        self.sw = get_sw(sw_file)

    # ----------------------------------------------------------------------
    def set_model(self, model, n_points):
        self.main_data_set['model'] = model
        if self.main_data_set['model'] == 'sqrt':
            self.num_depth_points = 4
        else:
            self.num_depth_points = int(n_points) + 2

    # ----------------------------------------------------------------------
    def prepare_graps(self):

        self.v_graphs_history = [[] for _ in range(self.num_depth_points)]
        self.d_graphs_history = [[] for _ in range(self.num_depth_points-2)]
        self.potential_graphs_history = []
        self.shifts_graphs_history = []

        self.v_graphs_stack = []
        self.d_graphs_stack = []
        self.potential_graph = None
        self.shifts_graph = None

        if self.STAND_ALONE_MODE:
            self.fig, self.axes = plt.subplots(nrows=2, ncols=self.num_depth_points)
            self.plots = [[] for _ in range(self.num_depth_points * 2)]
            plt.ion()
    # ----------------------------------------------------------------------
    def _generate_start_set(self):

        self.v_set = [np.zeros(self.settings['V_MESH']) for _ in range(self.num_depth_points)]

        v_half_steps = int(np.floor(self.settings['V_MESH']/2))

        for point in range(self.num_depth_points):
            self.v_set[point] = np.linspace(-self.settings['V_STEP']*v_half_steps, self.settings['V_STEP']*v_half_steps,
                                            self.settings['V_MESH'])

        self.d_set = [np.ones(self.settings['D_MESH'])*self.structure[0] for _ in range(self.num_depth_points - 2)]
        start_points = np.linspace(self.structure[0], self.structure[0] + self.structure[1], self.num_depth_points-1)

        for point in range(self.num_depth_points-2):
            self.d_set[point] = np.linspace(start_points[0+point], start_points[1+point],
                                            self.settings['D_MESH'] + 2)[1:-1]

    # ----------------------------------------------------------------------
    def fill_voltages(self, selected_v, last_v_set, d_set):
        if len(last_v_set) > 1:
            for v in last_v_set[0]:
                self.fill_voltages(np.append(selected_v, v), last_v_set[1:], d_set)
        else:
            for v in last_v_set[0]:
                self.fill_depth(np.append(selected_v, v), [], d_set)

    # ----------------------------------------------------------------------
    def fill_depth(self, selected_v, selected_d, last_d_set):
        if len(last_d_set) > 1:
            for d in last_d_set[0]:
                self.fill_depth(selected_v, np.append(selected_d, d), last_d_set[1:])
        elif len(last_d_set) == 1:
            for d in last_d_set[0]:
                self.res['depthset'][:, self._fill_counter] = np.append(self.structure[0],
                                                                        np.append(np.append(selected_d, d),
                                                                        self.structure[0] + self.structure[1]))
                self.res['voltset'][:, self._fill_counter] = selected_v
                self._fill_counter += 1
        else:
            self.res['depthset'][:, self._fill_counter] = [self.structure[0], self.structure[0] + self.structure[1]]
            self.res['voltset'][:, self._fill_counter] = selected_v
            self._fill_counter += 1

    # ----------------------------------------------------------------------
    def _get_fit_set(self):

        self.main_data_set['fit_points'] = 1

        for v_points in self.v_set:
            self.main_data_set['fit_points'] *= len(v_points)

        for d_points in self.d_set:
            self.main_data_set['fit_points'] *= len(d_points)

        self.res = {'depthset': np.zeros((self.num_depth_points, self.main_data_set['fit_points'])),
                    'voltset': np.zeros((self.num_depth_points, self.main_data_set['fit_points'])),
                    'mse': np.zeros(self.main_data_set['fit_points']),
                    'last_best_shifts': np.zeros_like(self.main_data_set['data'][:, 2]),
                    'last_intensity': np.zeros_like(self.main_data_set['data'][:, 1]),
                    'statistics': {}}

        self._fill_counter = 0
        self.fill_voltages([], self.v_set, self.d_set)

    # ----------------------------------------------------------------------
    def analyse_results(self):

        self.best_ksi.append(np.min(self.res['mse']))

        self.res['statistics'] = {"V_points": [[] for _ in range(self.num_depth_points)],
                                  "D_points": [[] for _ in range(self.num_depth_points - 2)]}

        for point in range(self.num_depth_points):
            new_set, statistics = self.analyse_statistic(self.v_set[point],
                                                                      self.res['voltset'][point, :],
                                                                      self.v_graphs_stack[point], 'volt_point')

            full_statistics = np.vstack((self.v_set[point], statistics))
            self.res['statistics']["V_points"][point] = full_statistics
            self.v_graphs_history[point].append(full_statistics)
            self.v_set[point] = new_set

        for point in range(self.num_depth_points - 2):
            new_set, statistics = self.analyse_statistic(self.d_set[point],
                                                                      self.res['depthset'][point + 1, :],
                                                                      self.d_graphs_stack[point], 'depth_point')
            full_statistics = np.vstack((self.d_set[point], statistics))
            self.res['statistics']["D_points"][point] = full_statistics
            self.d_graphs_history[point].append(full_statistics)
            self.d_set[point] = new_set

            if self.d_set[point][0] < self.structure[0] + self.settings['D_STEP']:
                self.d_set[point] += self.structure[0] - self.d_set[point][0] + self.settings['D_STEP']

            if point > 0:
                if self.d_set[point][0] < self.d_set[point - 1][-1] + self.settings['D_STEP']:
                    self.d_set[point] += self.d_set[point - 1][-1] - self.d_set[point][0] + self.settings['D_STEP']

            if self.d_set[point][-1] > self.structure[0] + self.structure[1] - self.settings['D_STEP']:
                self.d_set[point] -= self.d_set[point][-1] - self.structure[0] - \
                                     self.structure[1] - self.settings['D_STEP']

    # ----------------------------------------------------------------------
    def analyse_statistic(self, variable_set, data_cut, graphs, mode):

        pre_set = np.zeros(3)
        ksiset = np.zeros((len(variable_set), 2))

        if mode == 'volt_point':
            mesh = self.settings['V_MESH']
            step = self.settings['V_STEP']
        else:
            mesh = self.settings['D_MESH']
            step = self.settings['D_STEP']

        for v_point in range(len(variable_set)):
            all_inds = np.nonzero(data_cut == variable_set[v_point])
            ksiset[v_point, 0] = variable_set[v_point]
            ksiset[v_point, 1] = np.min(self.res['mse'][all_inds])

        t_statistics = np.sqrt(ksiset[:, 1] - self.best_ksi[-1]) / np.sqrt(self.best_ksi[-1] / self.main_data_set['data'].shape[0])

        if self.STAND_ALONE_MODE:
            graphs[1].set_xdata(variable_set)
            graphs[2].set_xdata(variable_set)
            graphs[2].set_ydata(t_statistics)
            graphs[0].relim()
            graphs[0].autoscale_view()
        else:
            graphs[0].setData(variable_set, np.ones_like(variable_set) * self.t_val)
            graphs[1].setData(variable_set, t_statistics)

        if self.best_ksi[-1] + self.settings['KSI_TOLLERANCE'] < self.best_ksi[-2]:
            half_steps = int(np.floor(mesh / 2))
            calculated_set = ksiset[np.argmin(ksiset[:, 1]), 0] + np.linspace(-step*half_steps, step*half_steps, mesh)

        else:
            region_of_interest = np.nonzero(t_statistics < self.t_val)
            lowend = region_of_interest[0][0]
            upend = region_of_interest[0][-1]

            if lowend == 0:
                pre_set[0] = variable_set[0] - step
            elif lowend == mesh - 1:
                pre_set[0] = variable_set[-1]
            else:
                pre_set[0] = np.interp(self.t_val, t_statistics[lowend - 1:lowend + 1],
                                              ksiset[lowend - 1:lowend + 1,0], period=np.inf)

            pre_set[1] = np.mean(ksiset[np.nonzero(ksiset[:, 1] == np.min(ksiset[:, 1])), 0])

            if upend == 0:
                pre_set[2] = variable_set[0]
            elif upend == mesh - 1:
                pre_set[2] = variable_set[-1] + step
            else:
                pre_set[2] = np.interp(self.t_val, t_statistics[upend:upend + 2],
                                              ksiset[upend:upend + 2, 0], period=np.inf)

            calculated_set = np.append(np.linspace(pre_set[0], pre_set[1], np.ceil(mesh/2)),
                                       np.linspace(pre_set[1], pre_set[2], np.ceil(mesh/2))[1:])

        return calculated_set, t_statistics

    # ----------------------------------------------------------------------
    def set_external_graphs(self, plot_axes):

        for ind in range(self.num_depth_points):
            plot_axes['v_points'][ind].setLabel('bottom', 'BE, eV')
            plot_axes['v_points'][ind].setLabel('left', 't statistics')
            plot_axes['v_points'][ind].setYRange(0, self.t_val * 1.5)
            self.v_graphs_stack.append([plot_axes['v_points'][ind].plot(range(self.settings['V_MESH']),
                                                                        np.ones(self.settings['V_MESH'])
                                                                        * self.t_val, **lookandfeel.MAX_T_STYLE),
                                        plot_axes['v_points'][ind].plot(range(self.settings['V_MESH']),
                                                                        np.zeros(self.settings['V_MESH']),
                                                                        **lookandfeel.T_STAT_STYLE)])

        for ind in range(self.num_depth_points - 2):
            plot_axes['d_points'][ind].setLabel('bottom', 'Depth, nm')
            plot_axes['d_points'][ind].setLabel('left', 't statistics')
            plot_axes['d_points'][ind].setYRange(0, self.t_val * 1.5)
            self.d_graphs_stack.append([plot_axes['d_points'][ind].plot(range(self.settings['D_MESH']),
                                                                        np.ones(self.settings['D_MESH']) * self.t_val,
                                                                        **lookandfeel.MAX_T_STYLE),
                                        plot_axes['d_points'][ind].plot(range(self.settings['D_MESH']),
                                                                        np.zeros(self.settings['D_MESH']),
                                                                        **lookandfeel.T_STAT_STYLE)])

        plot_axes['shifts'].plot(self.main_data_set['data'][:, 0], self.main_data_set['data'][:, 2],
                                 **lookandfeel.CURRENT_SOURCE_SHIFT)
        self.shifts_graph = plot_axes['shifts'].plot(self.main_data_set['data'][:, 0], self.main_data_set['data'][:, 2],
                                                     **lookandfeel.CURRENT_SIM_SHIFT)
        self.potential_graph = plot_axes['potential'].plot(self.main_data_set['fit_depth_points'],
                                                           np.zeros_like(self.main_data_set['fit_depth_points']), **lookandfeel.CURRENT_POTENTIAL_STYLE)

    # ----------------------------------------------------------------------
    def prepare_plots(self):

        for ind in range(self.num_depth_points):
            self.axes[0, ind].set_title('V point {}'.format(ind + 1))
            self.axes[0, ind].set_xlabel('BE, eV')
            self.axes[0, ind].set_ylabel('t statistics')
            self.axes[0, ind].set_ylim([0, self.t_val * 1.5])
            self.v_graphs_stack.append([self.axes[0, ind],
                                        self.axes[0, ind].plot(range(self.settings['V_MESH']),
                                                               np.ones(self.settings['V_MESH']) * self.t_val, 'g--')[0],
                                        self.axes[0, ind].plot(range(self.settings['V_MESH']),
                                                               np.zeros(self.settings['V_MESH']), '*-')[0]])


        for ind in range(self.num_depth_points - 2):
            self.axes[1, ind].set_title('D point {}'.format(ind + 1))
            self.axes[1, ind].set_xlabel('Depth, nm')
            self.axes[1, ind].set_ylabel('t statistics')
            self.axes[1, ind].set_ylim([0, self.t_val * 1.5])
            self.d_graphs_stack.append([self.axes[1, ind],
                                        self.axes[1, ind].plot(range(self.settings['D_MESH']),
                                                               np.ones(self.settings['D_MESH']) * self.t_val, 'g--')[0],
                                        self.axes[1, ind].plot(range(self.settings['D_MESH']),
                                                               np.zeros(self.settings['D_MESH']), '*-')[0]])

        self.axes[1, self.num_depth_points - 1].plot(self.main_data_set['data'][:, 0], self.main_data_set['data'][:, 2], 'x')
        self.shifts_graph = [self.axes[1, self.num_depth_points - 1],
                             self.axes[1, self.num_depth_points - 1].plot(self.main_data_set['data'][:, 0],
                                                                          self.main_data_set['data'][:, 2], '-')[0]]
        self.potential_graph = [self.axes[1, self.num_depth_points - 2],
                                self.axes[1, self.num_depth_points - 2].plot(self.main_data_set['fit_depth_points'],
                                                                            np.zeros_like(self.main_data_set['fit_depth_points']))[0]]

        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        plt.draw()
        plt.gcf().canvas.flush_events()
        # time.sleep(5)
        plt.show()

    # ----------------------------------------------------------------------
    def save_data(self, cycle):
        with open(os.path.join(self._directory, self._sample_name +
                                                   '{}_{}.res'.format(self._sample_name, cycle)), 'wb') as f:
            pickle.dump(self.res, f, pickle.HIGHEST_PROTOCOL)

    # ----------------------------------------------------------------------
    def plot_result(self):
        best_ksi_ind = np.argmin(self.res['mse'])
        volts_values = calculatePotential(self.res['depthset'][:, best_ksi_ind], self.res['voltset'][:, best_ksi_ind],
                                          self.main_data_set['fit_depth_points'], self.main_data_set['model'])

        self.potential_graphs_history.append(volts_values)

        self.last_best_shifts, _ = get_shifts(self.main_data_set, best_ksi_ind)
        self.shifts_graphs_history.append(self.last_best_shifts)

        if self.STAND_ALONE_MODE:
            self.potential_graph[1].set_ydata(volts_values)
            self.potential_graph[0].relim()
            self.potential_graph[0].autoscale_view()

            self.shifts_graph[1].set_ydata(self.last_best_shifts)
            self.shifts_graph[0].relim()
            self.shifts_graph[0].autoscale_view()
        else:
            self.potential_graph.setData(self.main_data_set['fit_depth_points'], volts_values)
            self.shifts_graph.setData(self.main_data_set['data'][:, 0], self.last_best_shifts)

    # ----------------------------------------------------------------------
    def sim_profile_shifts(self, d_set, v_set, pot_plot, source_dat_plot, sim_data_plot):

        d_set += self.structure[0]
        diff = np.diff(d_set)
        for ind in np.where(diff == 0):
            d_set[ind] += 1e-10

        volts_values = calculatePotential(d_set, v_set, self.main_data_set['fit_depth_points'], self.main_data_set['model'])

        self.main_data_set['depthset'] = np.vstack(d_set)
        self.main_data_set['voltset'] = np.vstack(v_set)

        self.main_data_set['fit_spectra_set'] = generate_fit_set(self.main_data_set['ref_spectra'],
                                                                 self.main_data_set['fit_depth_points'],
                                                                 self.main_data_set['data'][:, 0], self.sw,
                                                                 self.settings['LAMBDA'])

        shifts, _ = get_shifts(self.main_data_set, 0)

        pot_plot.setData(self.main_data_set['fit_depth_points'], volts_values)
        source_dat_plot.setData(self.main_data_set['data'][:, 0], self.main_data_set['data'][:, 2])
        if shifts is None:
            sim_data_plot.setData(self.main_data_set['data'][:, 0], np.zeros_like(self.main_data_set['data'][:, 0]))
        else:
            sim_data_plot.setData(self.main_data_set['data'][:, 0], shifts)

    # ----------------------------------------------------------------------
    def prepare_fit_set(self, file_name):

        self.main_data_set['fit_spectra_set'] = generate_fit_set(self.main_data_set['ref_spectra'],
                                                                 self.main_data_set['fit_depth_points'],
                                                                 self.main_data_set['data'][:, 0],
                                                                 self.sw, self.settings['LAMBDA'])

        data_to_save = {'sw': self.sw, 'main_data_set' : self.main_data_set, 'num_depth_points': self.main_data_set,
                        'settings': self.settings, 'structure': self.structure, 'angle_shift': self.angle_shift,
                        'be_shift': self.be_shift, 't_val': self.t_val}

        with open(file_name, 'wb') as f:
            pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)

    # ----------------------------------------------------------------------
    def load_fit_set(self, file_name):
        self._sample_name = None
        self._directory = None


    # ----------------------------------------------------------------------
    def do_intensity_fit(self, cycles=np.inf):

        if cycles < np.inf:
            up_lim = cycles
        else:
            up_lim = 50
        if self.STAND_ALONE_MODE:
            self.sidx = Slider(plt.axes([0.1, 0.02, 0.8, 0.03]), 'Cycle#', 0, up_lim, valinit=up_lim, valstep=1)
            self.sidx.on_changed(self.show_results)
            self.prepare_plots()

        self.main_data_set['fit_spectra_set'] = generate_fit_set(self.main_data_set['ref_spectra'],
                                                                 self.main_data_set['fit_depth_points'],
                                                                 self.main_data_set['data'][:, 0],
                                                                 self.sw, self.settings['LAMBDA'])

        self._generate_start_set()
        self.cycle = 0

        while self.cycle < cycles:
            start_time = time.time()
            self._get_fit_set()

            self.main_data_set['depthset'] = self.res['depthset']
            self.main_data_set['voltset'] = self.res['voltset']

            print('Fit set preparation time: {}'.format(time.time() - start_time))

            start_time = time.time()

            if self.settings['MULTIPROCESSING']:

                n_cpu = mp.cpu_count()
                n_tasks = self.settings['N_SUB_JOBS'] * n_cpu

                mse_list = mp.RawArray(ctypes.c_double, self.main_data_set['fit_points'])

                stop_points = np.ceil(np.linspace(0, self.main_data_set['fit_points'], n_tasks + 1))
                jobs_list = []

                for i in range(n_tasks):
                    jobs_list.append((int(stop_points[i]), int(min(stop_points[i+1], self.main_data_set['fit_points']))))

                jobs_queue = mp.JoinableQueue()
                for job in jobs_list:
                    jobs_queue.put(job)
                for _ in range(n_cpu):
                    jobs_queue.put(None)

                workers = []
                for i in range(n_cpu):
                    worker = mp.Process(target=mse_calculator,
                                        args=(self.main_data_set, jobs_queue, mse_list))
                    workers.append(worker)
                    worker.start()

                jobs_queue.join()

                self.res['mse'] = np.reshape(np.frombuffer(mse_list), self.main_data_set['fit_points'])
            else:
                for ind in range(self.main_data_set['fit_points']):
                    shifts, _ = get_shifts(self.main_data_set, ind)
                    if shifts is not None:
                        shifts -= self.main_data_set['data'][:, 2]
                        self.res['mse'][ind] = np.inner(shifts, shifts)
                    else:
                        self.res['mse'][ind] = 1e6

            error_cycle_time = time.time() - start_time
            print('Error calculation time: {}, time per point: {}'.format(error_cycle_time,
                                                   np.round(error_cycle_time/self.main_data_set['fit_points'], 6)))

            start_time = time.time()
            self.plot_result()
            self.analyse_results()
            self.save_data(self.cycle)
            print('Plot and save time: {}'.format(time.time() - start_time))

            if self.STAND_ALONE_MODE:
                plt.draw()
                plt.gcf().canvas.flush_events()
            # time.sleep(5)

            print('Cycle {} from {} completed'.format(self.cycle + 1, cycles))
            self.cycle += 1

        self.cycle -= 1
        plt.ioff()
        plt.show()


    # ----------------------------------------------------------------------
    def show_results(self, idx):

        ind = 0
        if self.cycle > 0:
            ind = int(min(self.cycle, idx))

        if len(self.shifts_graphs_history) > 0:
            self.potential_graph[1].set_ydata(self.potential_graphs_history[ind])
            self.potential_graph[0].relim()
            self.potential_graph[0].autoscale_view()

            self.shifts_graph[1].set_ydata(self.shifts_graphs_history[ind])
            self.shifts_graph[0].relim()
            self.shifts_graph[0].autoscale_view()

            for point in range(self.num_depth_points):
                self.v_graphs_stack[point][1].set_xdata(self.v_graphs_history[point][ind][0, :])
                self.v_graphs_stack[point][2].set_xdata(self.v_graphs_history[point][ind][0, :])
                self.v_graphs_stack[point][2].set_ydata(self.v_graphs_history[point][ind][1, :])
                self.v_graphs_stack[point][0].relim()
                self.v_graphs_stack[point][0].autoscale_view()

            for point in range(self.num_depth_points-2):
                self.d_graphs_stack[point][1].set_xdata(self.d_graphs_history[point][ind][0, :])
                self.d_graphs_stack[point][2].set_xdata(self.d_graphs_history[point][ind][0, :])
                self.d_graphs_stack[point][2].set_ydata(self.d_graphs_history[point][ind][1, :])
                self.d_graphs_stack[point][0].relim()
                self.d_graphs_stack[point][0].autoscale_view()

    # ----------------------------------------------------------------------
    def do_potential_fit(self):

        pass

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--s", "--data_set", dest="data_set")
    (options, _) = parser.parse_args()
