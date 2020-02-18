from subfunctions import *
from potential_models import calculatePotential
import matplotlib.pyplot as plt
import look_and_feel as lookandfeel

class Base_For_External_Solver():

    settings = {}

    POSSIBLE_TO_DISPLAY_INTERMEDIATE_STEPS = False

    # ----------------------------------------------------------------------
    def __init__(self, parent):

        self.parent = parent

        self.cycle = 0
        self.solution_history = []

        self.v_history = []
        self.d_history = []
        self.potential_graphs_history = []
        self.shifts_graphs_history = []

        self.v_graphs_stack = []
        self.d_graphs_stack = []
        self.potential_graph = None
        self.shifts_graph = None

        self.fig = None
        self.axes = None
        self.plots = []

    # ----------------------------------------------------------------------
    def reset_fit(self):

        self.cycle = 0

        self.v_history = [[] for _ in range(self.parent.num_depth_points)]
        self.d_history = [[] for _ in range(self.parent.num_depth_points - 2)]
        self.potential_graphs_history = []
        self.shifts_graphs_history = []

        self.v_graphs_stack = []
        self.d_graphs_stack = []
        self.potential_graph = None
        self.shifts_graph = None

        if self.parent.STAND_ALONE_MODE and self.parent.DO_PLOT:
            if self.parent.settings['MONITOR_FIT'] and self.POSSIBLE_TO_DISPLAY_INTERMEDIATE_STEPS:
                self.fig, self.axes = plt.subplots(nrows=2, ncols=self.parent.num_depth_points)
                self.plots = [[] for _ in range(self.parent.num_depth_points * 2)]
            else:
                self.fig, self.axes = plt.subplots(nrows=2, ncols=0)
                self.plots = [[], []]
            plt.ion()

    # ----------------------------------------------------------------------
    def set_external_graphs(self, graphs_layout):

        if self.parent.settings['MONITOR_FIT'] and self.POSSIBLE_TO_DISPLAY_INTERMEDIATE_STEPS:
            for ind in range(self.parent.num_depth_points):
                plot_axes = graphs_layout.addPlot(title="V_{}".format(ind), row=0, col=ind)
                plot_axes.setLabel('bottom', 'Cycle, N')
                plot_axes.setLabel('left', 'BE, eV')
                self.v_graphs_stack.append(plot_axes.plot([0], [0], **lookandfeel.T_STAT_STYLE))

            ind = 0
            for ind in range(self.parent.num_depth_points - 2):
                plot_axes = graphs_layout.addPlot(title="D_{}".format(ind), row=1, col=ind)
                plot_axes.setLabel('bottom', 'Cycle, N')
                plot_axes.setLabel('left', 'Depth, nm')
                self.d_graphs_stack.append(plot_axes.plot([0], [0], **lookandfeel.T_STAT_STYLE))

            if self.parent.num_depth_points - 2:
                start_ind = ind + 1
            else:
                start_ind = 0

            shifts_plot = graphs_layout.addPlot(title="Shifts", row=1, col=start_ind + 1)
            potential_plot = graphs_layout.addPlot(title="Potential", row=1, col=start_ind)
        else:
            shifts_plot = graphs_layout.addPlot(title="Shifts", row=0, col=0)
            potential_plot = graphs_layout.addPlot(title="Potential", row=1, col=0)

        shifts_plot.plot(self.parent.main_data_set['data'][:, 0], self.parent.main_data_set['data'][:, 2],
                         **lookandfeel.CURRENT_SOURCE_SHIFT)

        self.shifts_graph = shifts_plot.plot(self.parent.main_data_set['data'][:, 0],
                                             self.parent.main_data_set['data'][:, 2],
                                             **lookandfeel.CURRENT_SIM_SHIFT)

        self.potential_graph = potential_plot.plot(
            (self.parent.main_data_set['fit_depth_points'] - self.parent.structure[0]) * 1e9,
            np.zeros_like(self.parent.main_data_set['fit_depth_points']),
            **lookandfeel.CURRENT_POTENTIAL_STYLE)

    # ----------------------------------------------------------------------
    def prepare_stand_alone_plots(self):

        if self.parent.settings['MONITOR_FIT'] and self.POSSIBLE_TO_DISPLAY_INTERMEDIATE_STEPS:
            for ind in range(self.parent.num_depth_points):
                self.axes[0, ind].set_title('V point {}'.format(ind + 1))
                self.axes[0, ind].set_xlabel('Cycle, N')
                self.axes[0, ind].set_ylabel('BE, eV')
                self.v_graphs_stack.append([self.axes[0, ind],
                                            self.axes[0, ind].plot(range(self.parent.settings['V_MESH']),
                                                                   np.ones(self.parent.settings['V_MESH']), 'g--')[0]])

            for ind in range(self.parent.num_depth_points - 2):
                self.axes[1, ind].set_title('D point {}'.format(ind + 1))
                self.axes[1, ind].set_xlabel('Cycle, N')
                self.axes[1, ind].set_ylabel('Depth, nm')
                self.axes[1, ind].set_ylim([0, 3])
                self.d_graphs_stack.append([self.axes[1, ind],
                                            self.axes[1, ind].plot(range(self.parent.settings['D_MESH']),
                                                                   np.ones(self.parent.settings['D_MESH']), 'g--')[0]])

            self.axes[1, self.parent.num_depth_points - 1].plot(self.parent.main_data_set['data'][:, 0],
                                                                self.parent.main_data_set['data'][:, 2], 'x')
            self.shifts_graph = [self.axes[1, self.parent.num_depth_points - 1],
                                 self.axes[1, self.parent.num_depth_points - 1].plot(
                                     self.parent.main_data_set['data'][:, 0],
                                     self.parent.main_data_set['data'][:, 2], '-')[0]]
            self.potential_graph = [self.axes[1, self.parent.num_depth_points - 2],
                                    self.axes[1, self.parent.num_depth_points - 2].plot(
                                        self.parent.main_data_set['fit_depth_points'],
                                        np.zeros_like(self.parent.main_data_set['fit_depth_points']))[0]]

        else:
            self.axes[0, 0].plot(self.parent.main_data_set['data'][:, 0], self.parent.main_data_set['data'][:, 2], 'x')
            self.shifts_graph = [self.axes[0, 0],
                                 self.axes[0, 0].plot(self.parent.main_data_set['data'][:, 0],
                                                      self.parent.main_data_set['data'][:, 2], '-')[0]]

            self.potential_graph = [self.axes[1, 0], self.axes[1, 0].plot(self.parent.main_data_set['fit_depth_points'],
                                                                          np.zeros_like(self.parent.main_data_set[
                                                                                            'fit_depth_points']))[0]]

        plt.draw()
        plt.gcf().canvas.flush_events()
        plt.show()

    # ----------------------------------------------------------------------
    def _display_step(self, res):

        self.cycle += 1

        volt_set, depth_set = self._extract_sets(res)

        for ind in range(self.parent.num_depth_points):
            self.v_history[ind].append(volt_set[ind])

        for ind in range(self.parent.num_depth_points - 2):
            self.d_history[ind].append(volt_set[ind + 1])

        self.solution_history.append(np.vstack((depth_set, volt_set)))

        last_best_potential = calculatePotential(depth_set, volt_set, self.parent.main_data_set['fit_depth_points'],
                                                 self.parent.main_data_set['model'])

        self.potential_graphs_history.append(last_best_potential)

        last_best_shifts, last_intensity = get_shifts(self.parent.main_data_set, depth_set, volt_set)
        self.shifts_graphs_history.append(last_best_shifts)

        cycles = np.arange(self.cycle)

        if self.parent.STAND_ALONE_MODE:
            if self.parent.DO_PLOT:
                if self.POSSIBLE_TO_DISPLAY_INTERMEDIATE_STEPS:
                    for ind in range(self.parent.num_depth_points):
                        self.v_graphs_stack[ind][1].set_xdata(cycles)
                        self.v_graphs_stack[ind][1].set_ydata(self.v_history[ind])
                        self.v_graphs_stack[ind][0].relim()
                        self.v_graphs_stack[ind][0].autoscale_view()

                    for ind in range(self.parent.num_depth_points - 2):
                        self.d_graphs_stack[ind][1].set_xdata(cycles)
                        self.d_graphs_stack[ind][1].setData(self.d_history[ind])
                        self.d_graphs_stack[ind][0].relim()
                        self.d_graphs_stack[ind][0].autoscale_view()

                self.potential_graph[1].set_ydata(last_best_potential)
                self.potential_graph[0].relim()
                self.potential_graph[0].autoscale_view()

                self.shifts_graph[1].set_ydata(last_best_shifts)
                self.shifts_graph[0].relim()
                self.shifts_graph[0].autoscale_view()
        else:
            if self.POSSIBLE_TO_DISPLAY_INTERMEDIATE_STEPS:
                for ind in range(self.parent.num_depth_points):
                    self.v_graphs_stack[ind].setData(cycles, self.v_history[ind])
    
                for ind in range(self.parent.num_depth_points - 2):
                    self.d_graphs_stack[ind].setData(cycles, self.d_history[ind])

            self.potential_graph.setData(self.parent.main_data_set['fit_depth_points'],
                                         last_best_potential)

            self.shifts_graph.setData(self.parent.main_data_set['data'][:, 0], last_best_shifts)

        if self.parent.STAND_ALONE_MODE:
            if self.parent.DO_PLOT:
                plt.draw()
                plt.gcf().canvas.flush_events()
        else:
            self.parent.gui.update_cycles(self.cycle, 0,
                                          self.solution_history[self.cycle - 1])

     # ----------------------------------------------------------------------
    def _errFunc(self, params):
        """ calculate total residual for fits to several data sets held
        in a 2-D array, and modeled by model function"""

        volt_set, depth_set = self._extract_sets(params)
        shifts, _ = get_shifts(self.parent.main_data_set, depth_set, volt_set)

        return shifts - self.parent.main_data_set['data'][:, 2]

    # ----------------------------------------------------------------------
    def get_data_for_save(self):

        return {'cycle': self.cycle,
                'v_history': self.v_history,
                'd_history': self.d_history,
                'solution_history': self.solution_history,
                'potential_graphs_history': self.potential_graphs_history,
                'shifts_graphs_history': self.shifts_graphs_history}

    # ----------------------------------------------------------------------
    def show_results(self, idx):

        ind = 0
        if self.cycle > 0:
            ind = int(min(self.cycle, idx))

        cycles = np.arange(self.cycle)

        if len(self.shifts_graphs_history) > 0:
            if self.parent.STAND_ALONE_MODE:
                if self.parent.DO_PLOT:
                    if self.POSSIBLE_TO_DISPLAY_INTERMEDIATE_STEPS:
                        for ind in range(self.parent.num_depth_points):
                            self.v_graphs_stack[ind][1].set_xdata(cycles)
                            self.v_graphs_stack[ind][1].set_ydata(self.v_history[ind])
                            self.v_graphs_stack[ind][0].relim()
                            self.v_graphs_stack[ind][0].autoscale_view()

                        for ind in range(self.parent.num_depth_points - 2):
                            self.d_graphs_stack[ind][1].set_xdata(cycles)
                            self.d_graphs_stack[ind][1].setData(self.d_history[ind])
                            self.d_graphs_stack[ind][0].relim()
                            self.d_graphs_stack[ind][0].autoscale_view()

                    self.potential_graph[1].set_ydata(self.potential_graphs_history[ind])
                    self.potential_graph[0].relim()
                    self.potential_graph[0].autoscale_view()

                    self.shifts_graph[1].set_ydata(self.shifts_graphs_history[ind])
                    self.shifts_graph[0].relim()
                    self.shifts_graph[0].autoscale_view()

            else:
                if self.POSSIBLE_TO_DISPLAY_INTERMEDIATE_STEPS:
                    for ind in range(self.parent.num_depth_points):
                        self.v_graphs_stack[ind].setData(cycles, self.v_history[ind])

                    for ind in range(self.parent.num_depth_points - 2):
                        self.d_graphs_stack[ind].setData(cycles, self.d_history[ind])

                self.potential_graph.setData(self.parent.main_data_set['fit_depth_points'],
                                             self.potential_graphs_history[ind])

                self.shifts_graph.setData(self.parent.main_data_set['data'][:, 0],
                                          self.shifts_graphs_history[ind])

                return 0, self.solution_history[ind]

    # ----------------------------------------------------------------------
    def load_fit_res(self, loaded_data):

        for key in loaded_data.keys():
            setattr(self, key, loaded_data[key])