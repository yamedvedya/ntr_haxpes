#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser
from gradient_mesh import Gradient_Mesh_Solver
from lmfit_solver import LMFit_Solver
from pysot_solver import PySOT_Solver
from subfunctions import *
from potential_models import calculatePotential
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from distutils.util import strtobool
import pickle
import os

class NTR_fitter():

    STAND_ALONE_MODE = True
    DO_PLOT = True
    settings = {}
    METHODS = ('mesh_gradient', 'lm_fit', 'pysot')

    # ----------------------------------------------------------------------
    def __init__(self, gui=None):

        self.settings = {}

        self.solver = None

        self.main_data_set = {}
        self._sample_name = None
        self._directory = None
        self.structure = None
        self.angle_shift = None
        self.be_shift = None
        self.t_val = None
        self.sw = None
        self.num_depth_points = None

        self.fit_in_progress = False

        self.gui = gui

    # ----------------------------------------------------------------------
    def set_basic_settings(self, settings):

        for key, value in settings.items():
            self.settings[key] = value

        self.form_basic_data()
        self._set_solver()

    # ----------------------------------------------------------------------
    def _set_solver(self):

        if self.settings['FIT_SOLVER'] == 'mesh_gradient':
            self.solver = Gradient_Mesh_Solver(self)
        elif self.settings['FIT_SOLVER'] == 'lm_fit':
            self.solver = LMFit_Solver(self)
        elif self.settings['FIT_SOLVER'] == 'pysot':
            self.solver = PySOT_Solver(self)

    # ----------------------------------------------------------------------
    def load_fit_set(self, file_name):

        self._directory = os.path.split(file_name)[0]

        with open(file_name, "rb") as input_file:
            loaded_data = pickle.load(input_file)

        for key in loaded_data.keys():
            setattr(self, key, loaded_data[key])

        self._set_solver()
        self.solver.reset_fit()

        return loaded_data['start_values']

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
    def sim_profile_shifts(self, d_set, v_set, pot_plot, source_dat_plot, sim_data_plot):

        d_set += self.structure[0]
        diff = np.diff(d_set)
        for ind in np.where(diff == 0):
            d_set[ind] += 1e-10

        volts_values = calculatePotential(d_set, v_set, self.main_data_set['fit_depth_points'], self.main_data_set['model'])

        self.main_data_set['fit_spectra_set'] = generate_fit_set(self.main_data_set['ref_spectra'],
                                                                 self.main_data_set['fit_depth_points'],
                                                                 self.main_data_set['data'][:, 0], self.sw,
                                                                 self.settings['LAMBDA'])

        shifts, _ = get_shifts(self.main_data_set, d_set, v_set)

        pot_plot.setData((self.main_data_set['fit_depth_points'] - self.structure[0])*1e9, volts_values)
        source_dat_plot.setData(self.main_data_set['data'][:, 0], self.main_data_set['data'][:, 2])
        if shifts is None:
            sim_data_plot.setData(self.main_data_set['data'][:, 0], np.zeros_like(self.main_data_set['data'][:, 0]))
        else:
            sim_data_plot.setData(self.main_data_set['data'][:, 0], shifts)

    # ----------------------------------------------------------------------
    def dump_fit_set(self, file_name, fit_type=None, generate_data_set=True, start_values=None):

        if generate_data_set:
            self.main_data_set['fit_spectra_set'] = generate_fit_set(self.main_data_set['ref_spectra'],
                                                                     self.main_data_set['fit_depth_points'],
                                                                     self.main_data_set['data'][:, 0],
                                                                     self.sw, self.settings['LAMBDA'])

        with open(file_name, 'wb') as f:
            pickle.dump({'_sample_name': self._sample_name, 'fit_type': fit_type, 'sw': self.sw,
                         'main_data_set': self.main_data_set, 'num_depth_points': self.num_depth_points,
                         'settings': self.settings, 'structure': self.structure, 'angle_shift': self.angle_shift,
                         'be_shift': self.be_shift, 't_val': self.t_val,
                         'start_values': start_values}, f, pickle.HIGHEST_PROTOCOL)

    # ----------------------------------------------------------------------
    def save_fit_res(self):

        if not "results" in os.listdir(self._directory):
            os.mkdir(os.path.join(self._directory, "results"))

        file_name = '{}.res'.format(self._fit_name)
        full_path = os.path.join(os.path.join(self._directory, "results"), file_name)

        if not file_name in os.listdir(os.path.join(self._directory, "results")):
            self.dump_fit_set(full_path, fit_type='pot')

        with open(full_path, 'ab') as f:
            pickle.dump(self.solver.get_data_for_save(), f, pickle.HIGHEST_PROTOCOL)

    # ----------------------------------------------------------------------
    def load_fit_res(self, file_name):

        self._directory = os.path.split(os.path.split(file_name)[0])[0]
        self.solver.cycle = 0
        with open(file_name, 'rb') as fr:
            try:
                while True:
                    loaded_data = pickle.load(fr)
                    if not self.solver.cycle:
                        for key in loaded_data.keys():
                            setattr(self, key, loaded_data[key])
                        self.solver.reset_fit()
                    else:
                        self.solver.load_fit_res(loaded_data)
                    self.solver.cycle += 1
            except EOFError:
                pass
        self.solver.cycle -= 2
        self.gui.update_cycles(self.solver.cycle, self.solver.best_ksi[self.solver.cycle],
                               self.solver.solution_history[self.solver.cycle - 1])

        return 'pot', self.main_data_set['model'], self.num_depth_points-2

    # ----------------------------------------------------------------------
    def do_intensity_fit(self, start_values):

        self.fit_in_progress = True

        if self.STAND_ALONE_MODE and self.DO_PLOT:
            self.sidx = Slider(plt.axes([0.1, 0.02, 0.8, 0.03]), 'Cycle#', 0, 1, valinit=1, valstep=1)
            self.sidx.on_changed(self.solver.show_results)
            self.solver.prepare_stand_alone_plots()

        self.main_data_set['fit_spectra_set'] = generate_fit_set(self.main_data_set['ref_spectra'],
                                                                 self.main_data_set['fit_depth_points'],
                                                                 self.main_data_set['data'][:, 0],
                                                                 self.sw, self.settings['LAMBDA'])

        self._fit_name = self._sample_name + "_" + datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

        self.solver.do_fit(start_values)

    # ----------------------------------------------------------------------
    def stop_fit(self):

        self.fit_in_progress = False

    # ----------------------------------------------------------------------
    def do_potential_fit(self):

        pass

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-s", "--data_set", dest="data_set")
    parser.add_option("-p", "--plot", dest="do_plot", default=False)
    (options, _) = parser.parse_args()
    if options.data_set:
        fitter = NTR_fitter()
        if options.do_plot:
            fitter.DO_PLOT = strtobool(options.do_plot)
        else:
            fitter.DO_PLOT = False
        start_values = fitter.load_fit_set(options.data_set)
        fitter.do_intensity_fit(start_values)
