import matplotlib.pyplot as plt
import numpy as np
import look_and_feel as lookandfeel
import time

from src.ntr_data_fitting.gradient_mesh import Gradient_Mesh_Solver

from src.general.propagating_thread import ExcThread
from queue import Queue

class Visualisation_For_Gradient_Mesh():

    settings = {}

    # ----------------------------------------------------------------------
    def __init__(self, parent):

        self.parent = parent
        self.solver = Gradient_Mesh_Solver()

        self.v_graphs_stack = []
        self.d_graphs_stack = []
        self.potential_graph = None
        self.shifts_graph = None

        self.fig = None
        self.axes = None
        self.plots = []

        self._error_queue = Queue()
        self._potential_worker_state = 'idle'

        self.last_cycle = 0

        self._potential_worker= None

    # ----------------------------------------------------------------------
    def reset_fit(self):

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
    def _plot_result(self, cycle):

        if self.parent.DO_PLOT:
            if self.parent.STAND_ALONE_MODE:
                self.potential_graph[1].set_ydata(self.solver.potential_graphs_history[cycle])
                self.potential_graph[0].relim()
                self.potential_graph[0].autoscale_view()

                self.shifts_graph[1].set_ydata(self.solver.shifts_graphs_history[cycle])
                self.shifts_graph[0].relim()
                self.shifts_graph[0].autoscale_view()
            else:
                self.potential_graph.setData((self.parent.data_set_for_fitting['fit_depth_points'] -
                                              self.parent.data_set_for_fitting['fit_depth_points'][0])*1e9,
                                             self.solver.potential_graphs_history[cycle])
                self.shifts_graph.setData(self.parent.data_set_for_fitting['spectroscopic_data'][:, 0],
                                          self.solver.shifts_graphs_history[cycle])

            for point in range(self.parent.potential_model['num_depth_dof'] +
                             self.parent.potential_model['only_voltage_dof']):
                self.plot_statistics(self.v_graphs_stack[point], self.solver.v_graphs_history[point][cycle])

            for point in range(self.parent.potential_model['num_depth_dof']):
                self.plot_statistics(self.d_graphs_stack[point], self.solver.d_graphs_history[point][cycle])

        if self.parent.STAND_ALONE_MODE:
            if self.parent.DO_PLOT:
                plt.draw()
                plt.gcf().canvas.flush_events()
        else:
            self.parent.gui.update_potential_fit_cycles(cycle, self.solver.best_ksi[cycle + 1],
                                                        self.solver.solution_history[cycle])

    # ----------------------------------------------------------------------
    def plot_statistics(self, graphs, statistics):

        if self.parent.STAND_ALONE_MODE:
            if self.parent.DO_PLOT:
                graphs[1].set_xdata(statistics[0])
                graphs[1].set_ydata(np.ones_like(statistics[0])*self.solver.t_val)
                graphs[2].set_xdata(statistics[0])
                graphs[2].set_ydata(statistics[1])
                graphs[0].relim()
                graphs[0].autoscale_view()
        else:
            graphs[0].setData(statistics[0], np.ones_like(statistics[0])*self.solver.t_val)
            graphs[1].setData(statistics[0], statistics[1])

    # ----------------------------------------------------------------------
    def get_data_for_save(self):

        return self.solver.get_data_for_save()

    # ----------------------------------------------------------------------
    def abort(self):
        self.solver.fit_in_progress = False


    # ----------------------------------------------------------------------
    def dump_fit_set(self, file_name, start_values, fit_name):

        self.solver.reset_fit(self.parent.potential_model, self.parent.settings,
                              self.parent.data_set_for_fitting, start_values,
                              self.parent.directory, fit_name)

        self.solver.dump_fit_set(file_name)

    # ----------------------------------------------------------------------
    def do_fit(self, start_values, fit_name):

        self.solver.reset_fit(self.parent.potential_model, self.parent.settings,
                              self.parent.data_set_for_fitting, start_values,
                              self.parent.directory, fit_name)

        self.last_cycle = self.solver.cycle
        self._potential_worker = ExcThread(self._fitter_potential_fit_worker, 'fitter_worker', self._error_queue)
        self._potential_worker_state = 'run'
        self._potential_worker.start()

        while self._potential_worker_state != 'idle':
            if self.solver.cycle > self.last_cycle:
                self._plot_result(self.solver.cycle)
                self.last_cycle = self.solver.cycle
            time.sleep(1)

        if self.parent.STAND_ALONE_MODE and self.parent.DO_PLOT:
            plt.ioff()
            plt.show()

    # ----------------------------------------------------------------------
    def _fitter_potential_fit_worker(self):

        try:
            self.solver.do_fit()
        except Exception as err:
            self._potential_worker_state = 'idle'
            self._error_queue.put(err)
        self._potential_worker_state = 'idle'

    # ----------------------------------------------------------------------
    def show_results(self, idx):

        ind = 0
        if self.solver.cycle > 0:
            ind = int(min(self.solver.cycle, idx))

        if len(self.solver.shifts_graphs_history) > 0:
            if self.parent.STAND_ALONE_MODE:
                if self.parent.DO_PLOT:
                    self.potential_graph[1].set_ydata(self.solver.potential_graphs_history[ind])
                    self.potential_graph[0].relim()
                    self.potential_graph[0].autoscale_view()

                    self.shifts_graph[1].set_ydata(self.solver.shifts_graphs_history[ind])
                    self.shifts_graph[0].relim()
                    self.shifts_graph[0].autoscale_view()

                    for point in range(self.parent.potential_model['num_depth_dof'] +
                                                                    self.parent.potential_model['only_voltage_dof']):
                        self.v_graphs_stack[point][1].set_xdata(self.solver.v_graphs_history[point][ind][0])
                        self.v_graphs_stack[point][2].set_xdata(self.solver.v_graphs_history[point][ind][0])
                        self.v_graphs_stack[point][2].set_ydata(self.solver.v_graphs_history[point][ind][1])
                        self.v_graphs_stack[point][0].relim()
                        self.v_graphs_stack[point][0].autoscale_view()

                    for point in range(self.parent.potential_model['num_depth_dof']):
                        self.d_graphs_stack[point][1].set_xdata(self.solver.d_graphs_history[point][ind][0])
                        self.d_graphs_stack[point][2].set_xdata(self.solver.d_graphs_history[point][ind][0])
                        self.d_graphs_stack[point][2].set_ydata(self.solver.d_graphs_history[point][ind][1])
                        self.d_graphs_stack[point][0].relim()
                        self.d_graphs_stack[point][0].autoscale_view()
            else:
                self.potential_graph.setData((self.parent.data_set_for_fitting['fit_depth_points']
                                              - self.parent.data_set_for_fitting['fit_depth_points'][0])*1e9,
                                             self.solver.potential_graphs_history[ind])
                self.shifts_graph.setData(self.parent.data_set_for_fitting['spectroscopic_data'][:, 0],
                                          self.parent.be_shift + self.solver.shifts_graphs_history[ind])

                for point in range(self.parent.potential_model['num_depth_dof'] +
                                                                    self.parent.potential_model['only_voltage_dof']):
                    self.v_graphs_stack[point][0].setData(self.solver.v_graphs_history[point][ind][0, :],
                                                          np.ones_like(self.solver.v_graphs_history[point][ind][0, :])
                                                          * self.solver.t_val)
                    self.v_graphs_stack[point][1].setData(self.solver.v_graphs_history[point][ind][0, :],
                                                          self.solver.v_graphs_history[point][ind][1, :])

                for point in range(self.parent.potential_model['num_depth_dof']):
                    self.d_graphs_stack[point][0].setData(self.solver.d_graphs_history[point][ind][0, :],
                                                          np.ones_like(self.solver.d_graphs_history[point][ind][0, :])
                                                          * self.solver.t_val)
                    self.d_graphs_stack[point][1].setData(self.solver.d_graphs_history[point][ind][0, :],
                                                          self.solver.d_graphs_history[point][ind][1, :])

                return self.solver.best_ksi[ind + 1], self.solver.solution_history[ind]

    # ----------------------------------------------------------------------
    def load_fit_res(self, file_name):
        self.solver.load_data(file_name)
        self.last_cycle = self.solver.cycle
