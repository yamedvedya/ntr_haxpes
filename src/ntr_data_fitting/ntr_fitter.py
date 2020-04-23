#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.ntr_data_fitting.visualisation_for_gradient_mesh import Visualisation_For_Gradient_Mesh
from src.ntr_data_fitting.lmfit_solver import LMFit_Potential_Solver
from src.ntr_data_fitting.pysot_solver import PySOT_Potential_Solver
from src.ntr_data_fitting.intensity_fitter import Intensity_Solver
from src.ntr_data_fitting.subfunctions import *
from src.ntr_data_fitting.stepanov_server import get_sw_from_server
from src.ntr_data_fitting.potential_models import calculatePotential
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
import os

class NTR_fitter():

    STAND_ALONE_MODE = True
    DO_PLOT = True
    METHODS = ('mesh_gradient', 'lm_fit', 'pysot')

    settings = {}

    potential_solver = None
    intensity_solver = None

    data_set_for_fitting = {}
    _original_spectroscopy_data = []

    _sample_name = None
    directory = None

    structure = None
    potential_model = None
    functional_layer = None
    angle_shift = None
    be_shift = None
    intensity_correction = False

    t_val = None
    sw = None

    fit_in_progress = False
    sw_synchronized = False

    potential_plot = None
    source_shifts_plot = None
    sim_shifts_plot = None
    source_intensity_plot = None
    sim_intensity_plot = None

    _last_sim_potential = {}

    # ----------------------------------------------------------------------
    def __init__(self, gui=None):

        self.gui = gui

    # ----------------------------------------------------------------------
    def set_default_plots(self, potential_plot, source_shifts_plot, sim_shifts_plot, source_intensity_plot, sim_intensity_plot):

        self.potential_plot = potential_plot
        self.source_shifts_plot = source_shifts_plot
        self.sim_shifts_plot = sim_shifts_plot
        self.source_intensity_plot = source_intensity_plot
        self.sim_intensity_plot = sim_intensity_plot

    # ----------------------------------------------------------------------
    def set_basic_settings(self, settings):

        for key, value in settings.items():
            self.settings[key] = value

        self.form_basic_data()
        self._set_solver()

    # ----------------------------------------------------------------------
    def _set_solver(self):

        self.intensity_solver = Intensity_Solver(self, self.directory, self._sample_name)

        # if self.settings['FIT_SOLVER'] == 'mesh_gradient':
        self.potential_solver = Visualisation_For_Gradient_Mesh(self)
        # elif self.settings['FIT_SOLVER'] == 'lm_fit':
        #     self.potential_solver = LMFit_Potential_Solver(self)
        # elif self.settings['FIT_SOLVER'] == 'pysot':
        #     self.potential_solver = PySOT_Potential_Solver(self)

    # ----------------------------------------------------------------------
    def form_basic_data(self):

        self.data_set_for_fitting['FIELD_MAX'] = self.settings['FIELD_MAX']
        self.data_set_for_fitting['SUB_LAYERS'] = self.settings['SUB_LAYERS']
        self.data_set_for_fitting['BE_STEP'] = self.settings['BE_STEP']

        ref_spectra_width = 1.1*(self.settings['VOLT_MAX'] + self.settings['SIM_SPECTRA_WIDTH'])

        self.data_set_for_fitting['ref_spectra'] = syn_spectra(-ref_spectra_width, ref_spectra_width,
                                                        self.settings['BE_STEP'], self.settings['G'], self.settings['L'])
        self.data_set_for_fitting['ref_spectra_points'] = int(ref_spectra_width/self.settings['BE_STEP'])

        self.data_set_for_fitting['sum_spectra_point'] = int(self.settings['SIM_SPECTRA_WIDTH']/self.settings['BE_STEP'])
        self.data_set_for_fitting['sum_spectra_energy'] = np.linspace(-self.settings['SIM_SPECTRA_WIDTH'],
                                                               self.settings['SIM_SPECTRA_WIDTH'],
                                                                2*self.data_set_for_fitting['sum_spectra_point'] + 1)

    # ----------------------------------------------------------------------
    def _update_source_plots(self):

        self.source_shifts_plot.setData(self.data_set_for_fitting['spectroscopic_data'][:, 0],
                                self.be_shift + self.data_set_for_fitting['spectroscopic_data'][:, 2])

        self.source_intensity_plot.setData(self.data_set_for_fitting['spectroscopic_data'][:, 0],
                                self.data_set_for_fitting['spectroscopic_data'][:, 1])
    # ----------------------------------------------------------------------
    def set_spectroscopy_data(self, data, file_name=None, sample_name=None, directory=None):

        if file_name is not None:
            self._sample_name = os.path.splitext(os.path.basename(file_name))[0]
            self.directory = os.path.dirname(file_name)
        else:
            self._sample_name = sample_name
            self.directory = directory
        self.intensity_solver.set_new_history_file(self.directory, self._sample_name)
        self._original_spectroscopy_data = data
        self.data_set_for_fitting['spectroscopic_data'] = self._original_spectroscopy_data.copy()

        self.be_shift = np.mean(self.data_set_for_fitting['spectroscopic_data'][:, 2])
        self.data_set_for_fitting['spectroscopic_data'][:, 2] -= self.be_shift

        self._update_source_plots()
        self.gui.update_sw_srb(self.intensity_solver.len_sw_history)

    # ----------------------------------------------------------------------
    def set_new_data_file(self, data_file, set):

        self._sample_name = os.path.splitext(os.path.basename(data_file))[0]
        self.directory = os.path.dirname(data_file)
        self.intensity_solver.set_new_history_file(self.directory, self._sample_name)

        self._original_spectroscopy_data, self.structure, self.angle_shift = get_data_set(data_file, set)

        self.data_set_for_fitting['spectroscopic_data'] = self._original_spectroscopy_data.copy()
        self.data_set_for_fitting['spectroscopic_data'][:, 0] += self.angle_shift
        self.data_set_for_fitting['spectroscopic_data'][:, 1] /= np.sum(self.data_set_for_fitting['spectroscopic_data'][:, 1])

        self.be_shift = np.mean(self.data_set_for_fitting['spectroscopic_data'][:, 2])
        self.data_set_for_fitting['spectroscopic_data'][:, 2] -= self.be_shift

        self.set_new_functional_layer()

    # ----------------------------------------------------------------------
    def set_new_functional_layer(self):

        thicknesses = [layer['thick'] for layer in self.structure]

        self.data_set_for_fitting['fit_depth_points'] = np.linspace(np.sum(thicknesses[0:self.functional_layer]),
                                                                    np.sum(thicknesses[0:self.functional_layer+1]),
                                                                    self.settings['SUB_LAYERS'])*1e-10

    # ----------------------------------------------------------------------
    def request_sw_from_server(self):

        self.sw = get_sw_from_server(self.settings, self.structure, self.directory)
        self.intensity_solver.add_sw_to_history(self.structure, self.sw)
        if not self.STAND_ALONE_MODE:
            self.gui.update_sw_srb(self.intensity_solver.len_sw_history)

    # ----------------------------------------------------------------------
    def request_sw_from_history(self, index):

        self.structure, self.sw = self.intensity_solver.get_sw_from_history(index)
        self.sim_profile_intensity()
        self.sw_synchronized = True

    # ----------------------------------------------------------------------
    def set_sw_form_file(self, sw_file):

        self.sw = get_sw(sw_file)
        self.intensity_solver.add_sw_to_history(self.structure, self.sw)

     # ----------------------------------------------------------------------
    def sim_profile_shifts(self, d_set, v_set):

        self._last_sim_potential = {'d_set': d_set, 'v_set': v_set}

        d_set += self.data_set_for_fitting['fit_depth_points'][0]

        diff = np.diff(d_set)
        for ind in np.where(diff == 0):
            d_set[ind] += 1e-10

        volts_values = calculatePotential(d_set, v_set, self.data_set_for_fitting['fit_depth_points'],
                                          self.potential_model['code'])

        self._prepare_data_set_for_fit()
        shifts, _ = get_shifts(self.data_set_for_fitting, d_set, v_set)

        self.potential_plot.setData((self.data_set_for_fitting['fit_depth_points'] -
                          self.data_set_for_fitting['fit_depth_points'][0])*1e9, volts_values)
        if shifts is None:
            self.sim_shifts_plot.setData(self.data_set_for_fitting['spectroscopic_data'][:, 0],
                                  self.be_shift + np.zeros_like(self.data_set_for_fitting['spectroscopic_data'][:, 0]))
        else:
            self.sim_shifts_plot.setData(self.data_set_for_fitting['spectroscopic_data'][:, 0], self.be_shift + shifts)

    # ----------------------------------------------------------------------
    def sim_profile_intensity(self):

        intensities, self.angle_shift = get_intensity_simple(self._original_spectroscopy_data[:, 0],
                                                             self.data_set_for_fitting['spectroscopic_data'][:, 1],
                                                             self.data_set_for_fitting['fit_depth_points'], self.sw,
                                                             self.settings['LAMBDA'] * 1e-9)

        self.data_set_for_fitting['spectroscopic_data'][:, 0] = self._original_spectroscopy_data[:, 0].copy() + \
                                                                self.angle_shift

        self.sim_intensity_plot.setData(self.sw['angles'], intensities)
        self._update_source_plots()

        return intensities

    # ----------------------------------------------------------------------
    def dump_session(self, f):

        pickle.dump({'_sample_name': self._sample_name, 'sw': self.sw,
                     '_original_spectroscopy_data': self._original_spectroscopy_data,
                     'settings': self.settings, 'data_set_for_fitting': self.data_set_for_fitting,
                     'structure': self.structure, 'potential_model': self.potential_model,
                     'angle_shift': self.angle_shift, 'be_shift': self.be_shift, 't_val': self.t_val,
                     'functional_layer': self.functional_layer, '_last_sim_potential': self._last_sim_potential,
                     }, f, pickle.HIGHEST_PROTOCOL)

    # ----------------------------------------------------------------------
    def dump_fit_set(self, file_name, start_values):

        for pair in start_values:
            pair[0] += self.data_set_for_fitting['fit_depth_points'][0]

        self._prepare_data_set_for_fit()
        self.potential_solver.dump_fit_set(file_name, start_values,
                                           self._sample_name + "_" + datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))

    # ----------------------------------------------------------------------
    def restore_session(self, fr, directory):

        self.directory = directory

        loaded_data = pickle.load(fr)
        for key in loaded_data.keys():
            setattr(self, key, loaded_data[key])

        if self._original_spectroscopy_data != []:
            self.be_shift = np.mean(self._original_spectroscopy_data[:, 2])
        if 'spectroscopic_data' in self.data_set_for_fitting.keys():
            self._update_source_plots()

        self.intensity_solver.set_new_history_file(self.directory, self._sample_name)

        if not self.STAND_ALONE_MODE:
            self.gui.update_sw_srb(self.intensity_solver.len_sw_history)

        if self.sw:
            self.sim_profile_intensity()

    # ----------------------------------------------------------------------
    def manual_angle_correction(self, angle):

        self.angle_shift = angle
        self.data_set_for_fitting['spectroscopic_data'][:, 0] = self._original_spectroscopy_data[:, 0].copy() + self.angle_shift
        self._update_source_plots()

    # ----------------------------------------------------------------------
    def correct_intensity(self):

        functional_layer_thickness = self.structure[self.functional_layer]['thick']*1e-10

        if self.intensity_correction:
            self.data_set_for_fitting['spectroscopic_data'][:, 1] = correct_intensity(self._original_spectroscopy_data,
                                                                                      functional_layer_thickness).copy()
        else:
            self.data_set_for_fitting['spectroscopic_data'][:, 1] = self._original_spectroscopy_data[:, 1].copy()

        self._update_source_plots()

        if self.sw:
            self.sim_profile_intensity()

    # ----------------------------------------------------------------------
    def _prepare_data_set_for_fit(self):
        self.data_set_for_fitting['fit_spectra_set'] = generate_fit_set(self.data_set_for_fitting['ref_spectra'],
                                                                        self.data_set_for_fitting['fit_depth_points'],
                                                                        self.data_set_for_fitting['spectroscopic_data'][:, 0],
                                                                        self.sw, self.settings['LAMBDA']*1e-9)
        self.data_set_for_fitting['model_code'] = self.potential_model['code']

    # ----------------------------------------------------------------------
    def do_potential_fit(self, start_values):

        self.fit_in_progress = True

        for pair in start_values:
            pair[0] += self.data_set_for_fitting['fit_depth_points'][0]

        if self.STAND_ALONE_MODE and self.DO_PLOT:
            self.sidx = Slider(plt.axes([0.1, 0.02, 0.8, 0.03]), 'Cycle#', 0, 1, valinit=1, valstep=1)
            self.sidx.on_changed(self.potential_solver.show_results)
            self.potential_solver.prepare_stand_alone_plots()

        self._prepare_data_set_for_fit()

        try:
            self.potential_solver.do_fit(start_values, self._sample_name + "_" + datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
        except Exception as err:
            self.fit_in_progress = False
            raise RuntimeError(err)
        self.fit_in_progress = False

    # ----------------------------------------------------------------------
    def do_intensity_fit(self, fit_variables):

        self.fit_in_progress = True
        self._fit_name = self._sample_name + "_" + datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        try:
            self.intensity_solver.bfgs_decend(fit_variables)
        except Exception as err:
            self.fit_in_progress = False
            raise RuntimeError(err)
        self.fit_in_progress = False

    # ----------------------------------------------------------------------
    def stop_fit(self):

        self.potential_solver.abort()
        self.intensity_solver.abort()
        self.fit_in_progress = False