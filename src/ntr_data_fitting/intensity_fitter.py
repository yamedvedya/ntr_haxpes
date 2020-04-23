from src.ntr_data_fitting.subfunctions import *
import numpy.linalg as ln
import numpy as np

import pickle
import os
import copy

class Intensity_Solver():

    sw_history = []
    len_sw_history = 0
    sw_history_file_name = ''

    _current_structure = {}

    _var_list = None

    fit_in_progress = False

    _C2 = 0.8

    # ----------------------------------------------------------------------
    def __init__(self, parent, directory, sample_name):

        self.parent = parent
        self.set_new_history_file(directory, sample_name)

    # ----------------------------------------------------------------------
    def add_sw_to_history(self, structure, sw):

        data = {'structure': structure, 'sw': sw}
        self.sw_history.append(data)
        self.len_sw_history = len(self.sw_history)

        if self.sw_history_file_name:
            with open(self.sw_history_file_name, 'ab') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    # ----------------------------------------------------------------------
    def look_for_sw_in_history(self, structure):

        for sw_set_ind, sw_set in enumerate(self.sw_history):
            layer_match = True
            for ind, layer in enumerate(structure):
                for key, value in layer.items():
                    if value is not None:
                        if key in ['x1', 'x2', 'x3']:
                            layer_match *= np.isclose(sw_set['structure'][ind][key], value,
                                                      atol=self.parent.settings['COMPSTEP']/2)
                        elif key in ['thick', 'sigma', 'w0', 'x0']:
                            layer_match *= np.isclose(sw_set['structure'][ind][key], value,
                                                      atol=self.parent.settings['{}STEP'.format(key.upper())]/2)
                        else:
                            layer_match *= sw_set['structure'][ind][key] == value
            if layer_match:
                self.parent.request_sw_from_history(sw_set_ind)
                return sw_set_ind

        self.parent.request_sw_from_server()
    # ----------------------------------------------------------------------
    def get_sw_from_history(self, ind):

        return self.sw_history[ind]['structure'], self.sw_history[ind]['sw']

    # ----------------------------------------------------------------------
    def _reoder_sw_history(self, structure):

        ind = self.look_for_sw_in_history(structure)
        self.sw_history.append(self.sw_history[ind])
        del self.sw_history[ind]
        self.parent.request_sw_from_history(-1)

    # ----------------------------------------------------------------------
    def set_new_history_file(self, directory, sample_name):

        self.sw_history = []

        if directory:
            file_name = '{}.sw'.format(sample_name)
            self.sw_history_file_name = os.path.join(directory, file_name)
            if file_name in os.listdir(directory):
                with open(self.sw_history_file_name, 'rb') as fr:
                    try:
                        while True:
                            self.sw_history.append(pickle.load(fr))
                    except EOFError:
                        pass

        self.len_sw_history = len(self.sw_history)

    # ----------------------------------------------------------------------
    def _balance_composition(self, composition, layer_to_change, nsteps):

        composition = list(filter(None, composition))
        new_value = composition[layer_to_change] + self.parent.settings['COMPSTEP']*nsteps

        del composition[layer_to_change]

        rest = np.round(1 - new_value, 2)
        diff = np.round(rest - np.sum(composition), 2)
        if composition[-1] + diff >= 0.01:
            composition[-1] += diff
        else:
            composition[-1] = 0.01
            diff -= (composition[-1] - 0.01)
            if len(composition) > 1:
                composition[0] += diff
            else:
                new_value -= 0.01

        composition.insert(layer_to_change, new_value)

        return composition

    # ----------------------------------------------------------------------
    def _rss_calculator(self):

        return np.sum(np.square(self.parent.data_set_for_fitting['spectroscopic_data'][:, 1] -
                                 np.interp(self.parent.data_set_for_fitting['spectroscopic_data'][:, 0],
                                           self.parent.sw['angles'],
                                           self.parent.sim_profile_intensity())))

    # ----------------------------------------------------------------------
    def _add_delta_to_structure(self, var, nsteps):

        if var[1] in ['COMP1', 'COMP2']:
            comp = [self.parent.structure[var[0]]['x{}'.format(ind)] for ind in range(1, 4)]
            comp = self._balance_composition(comp, int(var[1][-1:]) - 1, nsteps)
            for ind, value in enumerate(comp):
                self.parent.structure[var[0]]['x{}'.format(ind+1)] = \
                    np.round(value, get_precision(self.parent.structure[var[0]]['x{}'.format(ind+1)]))

        elif var[1] in ['SIGMA', 'X0', 'THICK', 'W0']:
            self.parent.structure[var[0]][var[1].lower()] = \
                np.round(self.parent.structure[var[0]][var[1].lower()] +
                         self.parent.settings['{}STEP'.format(var[1])]*nsteps,
                         get_precision(self.parent.structure[var[0]][var[1].lower()]))
        else:
            raise RuntimeError('Wrong variable type')

    # ----------------------------------------------------------------------
    def _rss_in_point(self, values):
        for ind, var in enumerate(self._var_list):
            self.parent.structure[var[0]][var[1].lower()] = \
                np.round(values[ind], get_precision(self.parent.structure[var[0]][var[1].lower()]))

        self.look_for_sw_in_history(self.parent.structure)
        return self._rss_calculator()

    # ----------------------------------------------------------------------
    def _diff_calculator(self):

        rss_in_point = self._rss_calculator()

        diffs = np.zeros(len(self._var_list))
        for ind, var in enumerate(self._var_list):
            self.parent.structure = copy.deepcopy(self._current_structure)
            self._add_delta_to_structure(var, 1)
            self.look_for_sw_in_history(self.parent.structure)
            diffs[ind] = (self._rss_calculator() - rss_in_point)/rss_in_point

        return diffs, rss_in_point

    # ----------------------------------------------------------------------
    def _line_search(self, directions, diffs, rss_in_point):

        point_found = False

        while not point_found:
            self.parent.structure = copy.deepcopy(self._current_structure)
            for ind, var in enumerate(self._var_list):
                self._add_delta_to_structure(var, directions[ind])

            self.look_for_sw_in_history(self.parent.structure)
            new_rss = self._rss_calculator()
            expected_rss = rss_in_point - self._C2 * ln.norm(diffs * rss_in_point * np.sign(directions) *
                                                             (np.abs(directions)-1))

            if new_rss > expected_rss:
                point_found = True
                for ind in range(len(directions)):
                    directions[ind] = np.sign(directions[ind]) * (np.abs(directions[ind]) // 2)
                    point_found *= directions[ind] == 0
            else:
                point_found = True

        return directions

    # ----------------------------------------------------------------------
    def _check_oscillations(self):
        oscillating_point = True
        for ind in range(len(self._var_list)):
            oscillating_point *= self._directions[ind][-1] == 0

        return oscillating_point
    # ----------------------------------------------------------------------
    def _check_limits(self, directions):
        for ind, var in enumerate(self._var_list):
            if var[1] in ['COMP1', 'COMP2']:
                step = self.parent.settings['COMPSTEP']
                value = self.parent.structure[var[0]]['x{}'.format(var[1][-1:])]
            else:
                step = self.parent.settings['{}STEP'.format(var[1])]
                value = self.parent.structure[var[0]][var[1].lower()]

            if directions[ind] > 0:
                if value + step * directions[ind] > var[3]:
                    directions[ind] = (var[3] - value) // step
            else:
                if value + step * directions[ind] < var[2]:
                    directions[ind] = - ((value - var[2]) // step)

            if not directions[ind]:
                self._var_at_lim[ind] = True

        return directions

    # ----------------------------------------------------------------------
    def abort(self):
        self.fit_in_progress = False

    # ----------------------------------------------------------------------
    def bfgs_decend(self, var_list):

        self.fit_in_progress = True
        self._var_list = var_list
        self._directions = [[] for _ in range(len(var_list))]
        self._var_at_lim = [False for _ in range(len(var_list))]
        self._rss_history = []
        msg = 'Start Intensity Fit. Fit params:'
        for ind, var in enumerate(self._var_list):
            msg += ' l{}_{}, start value: {}, min: {}, max: {}'.format(var[0], var[1].lower(),
                                                                      (self.parent.structure[var[0]][var[1].lower()]),
                                                                      var[2], var[3])
        self._current_structure = copy.deepcopy(self.parent.structure)
        self._cycle = 0
        solution_found = False
        self.look_for_sw_in_history(self.parent.structure)
        diffs, rss_in_point = self._diff_calculator()
        self._rss_history.append(rss_in_point)
        msg += ' Start RSS: {:0.3e}'.format(rss_in_point)
        self.parent.gui.add_message_to_fit_history(msg)

        while not solution_found and self.fit_in_progress:
            directions = np.zeros(len(var_list))
            for ind, var in enumerate(self._var_list):
                directions[ind] = -np.sign(diffs[ind])*(max(1, 0.1 // np.abs(diffs[ind])))

            self.parent.structure = copy.deepcopy(self._current_structure)
            directions = self._check_limits(directions)
            final_direction = self._line_search(directions, diffs, rss_in_point)

            self.parent.structure = copy.deepcopy(self._current_structure)
            for ind, var in enumerate(self._var_list):
                self._add_delta_to_structure(var, final_direction[ind])
                self._directions[ind].append(final_direction[ind])

            self.look_for_sw_in_history(self.parent.structure)
            new_rss = self._rss_calculator()
            self._rss_history.append(rss_in_point)
            self._current_structure = copy.deepcopy(self.parent.structure)

            msg = 'Moving to point:'
            for ind, var in enumerate(self._var_list):
                msg += ' l{}_{}, value: {}'.format(var[0], var[1].lower(), (self.parent.structure[var[0]][var[1].lower()]))
            msg += ' New RSS: {:0.3e}'.format(new_rss)
            self.parent.gui.add_message_to_fit_history(msg)

            if self._cycle:
                oscillating_point = self._check_oscillations()
            else:
                oscillating_point = False

            if (1+self.parent.settings['RSSTOL']) * new_rss > rss_in_point or oscillating_point:
                solution_found = True
            else:
                diffs, rss_in_point = self._diff_calculator()

            self._cycle += 1

        self._reoder_sw_history(self.parent.structure)
        self.parent.gui.add_message_to_fit_history('Solution found')
        self.parent.fit_in_progress = False