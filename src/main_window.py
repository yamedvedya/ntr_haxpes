import numpy as np
import random
import os
import pyqtgraph as pg
from scipy.io import loadmat

from src.general.propagating_thread import ExcThread
from src.widgets.main_window_ui import Ui_MainWindow
from src.general.settings_dialog import Settings_Window
from src.general.import_dialog import Import_Dialog
from src.ntr_data_fitting.potential_models import get_model_list
from src.ntr_data_fitting.ntr_fitter import NTR_fitter

from src.spectra_fit.spectra_fitter import Spectra_fitter

from src.general.potential_widgets import *
from src.general.component_widgets import *
from src.general.layer_widget import LayerWidget

from src.general.auxiliary_functions import *
import look_and_feel as lookandfeel
import default_settings as default_settings

from queue import Queue
from queue import Empty as emptyQueue

# ----------------------------------------------------------------------
class NTR_Window(QtWidgets.QMainWindow):

    settings = {}
    _potential_model_widgets = []
    _layers_widgets = []
    _backgrounds_widgets = []
    _components_widgets = []
    _spectra_worker_state = 'idle'
    _intensity_worker_state = 'idle'
    _potential_worker_state = 'idle'

    # ----------------------------------------------------------------------
    def __init__(self, options):
        """
        """
        super(NTR_Window, self).__init__()

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        self.error_queue = Queue()

        # self._ui.tab_mode.setEnabled(False)

        self._set_default_settings()
        self.settings_window = Settings_Window()

        self._components_grid = QtWidgets.QGridLayout(self._ui.s_wc_models)
        self._pot_model_grid = QtWidgets.QGridLayout(self._ui.p_wc_deg_freedom)
        self._layers_grid = QtWidgets.QGridLayout(self._ui.i_wc_layers)

        self._fill_potential_fit_combos()

        self.ntr_fitter = NTR_fitter(self)
        self.ntr_fitter.set_basic_settings(self.settings)
        self.ntr_fitter.STAND_ALONE_MODE = False

        self.spectra_fitter = Spectra_fitter()

        self.settings_window.fill_combos(self.ntr_fitter.METHODS)

        self._working_dir = os.getcwd()
        self._connect_actions()
        self._make_default_intensity_fit_graphics()
        self._make_default_potential_fit_graphics()
        self._make_default_spectra_fit_graphics()
        self._set_default_plots_to_fitter()

        self._order_layers(0, '')

        self._refresh_status_timer = QtCore.QTimer(self)
        self._refresh_status_timer.timeout.connect(self._refresh_status)
        self._refresh_status_timer.start(500)

# ----------------------------------------------------------------------
#       General code
# ----------------------------------------------------------------------

    def _block_signals(self, flag):

        self._ui.but_settings_potential.blockSignals(flag)
        self._ui.but_settings_intensity.blockSignals(flag)

        self._ui.cb_experimental_set.blockSignals(flag)
        self._ui.b_select_file.blockSignals(flag)

        self._ui.p_cb_cmb_model.blockSignals(flag)
        self._ui.p_sb_deg_freedom.blockSignals(flag)

        self._ui.p_but_pot_fit.blockSignals(flag)
        self._ui.p_but_prepare_fit_set.blockSignals(flag)
        self._ui.p_chk_correction.blockSignals(flag)
        self._ui.p_sb_max_potential.blockSignals(flag)
        self._ui.p_srb_cycle.blockSignals(flag)

        self._ui.i_but_simulate.blockSignals(flag)
        self._ui.i_but_start_fit.blockSignals(flag)
        self._ui.i_dsp_shift.blockSignals(flag)
        self._ui.i_but_revert_structure.blockSignals(flag)
        self._ui.i_scr_history.blockSignals(flag)

        self._ui.but_save_session.blockSignals(flag)

    # ----------------------------------------------------------------------
    def _connect_actions(self):

        self._ui.but_settings_potential.clicked.connect(lambda: self._show_settings('potential'))
        self._ui.but_settings_intensity.clicked.connect(lambda: self._show_settings('intensity'))

        self._ui.s_cmd_add_back.clicked.connect(lambda: self._add_component('bkg'))
        self._ui.s_cmd_add_line.clicked.connect(lambda: self._add_component('line'))
        self._ui.s_cmd_add_doublet.clicked.connect(lambda: self._add_component('dbl'))
        self._ui.s_scr_spectra.valueChanged.connect(lambda value: self._show_spectra(value))
        self._ui.s_cmd_fit_one.clicked.connect(self._fit_current_spectra)
        self._ui.s_cmd_fit_all.clicked.connect(self._fit_all_spectra)
        self._ui.s_bg_fit_spectra.buttonClicked.connect(lambda selection: self._range_fitting(selection))
        self._ui.s_sp_fit_from.valueChanged.connect(lambda value, x='from': self._check_range(x, value))
        self._ui.s_sp_fit_to.valueChanged.connect(lambda value, x='from': self._check_range(x, value))

        self._ui.s_dsp_cut_from.valueChanged.connect(lambda value, x='from': self._check_cut(x, value))
        self._ui.s_dsp_cut_to.valueChanged.connect(lambda value, x='to': self._check_cut(x, value))
        self._ui.s_bg_cut_range.buttonClicked.connect(lambda selection: self._cut_range(selection))
        self._ui.s_pb_range_to_all.clicked.connect(self._cut_range_to_all)

        self._ui.s_pb_set_data.clicked.connect(self._set_data_to_fitter)

        self._ui.cb_experimental_set.currentIndexChanged.connect(self._new_set_selected)

        self.settings_window.settings_changed.connect(self._settings_changed)
        self._ui.b_select_file.clicked.connect(self._load_file_clicked)

        self._ui.p_cb_cmb_model.currentIndexChanged.connect(lambda: self._potential_model_selected(True))
        self._ui.p_sb_deg_freedom.valueChanged.connect(lambda: self._change_degrees(True))

        self._ui.p_but_pot_fit.clicked.connect(self._start_stop_pot_fit)
        self._ui.p_but_prepare_fit_set.clicked.connect(self._prepare_potential_fit_set)
        self._ui.p_chk_correction.stateChanged.connect(lambda state: self._correct_intensity(state))
        self._ui.p_sb_max_potential.valueChanged.connect(self._change_potential_fit_v_range)
        self._ui.p_srb_cycle.valueChanged.connect(self._display_potential_fit_cycle)

        self._ui.i_but_simulate.clicked.connect(self._sim_intensity)
        self._ui.i_but_start_fit.clicked.connect(self._i_start_fit)
        self._ui.i_but_stop_fit.clicked.connect(self._i_stop_fit)
        self._ui.i_dsp_shift.valueChanged.connect(lambda value: self.ntr_fitter.manual_angle_correction(value))
        self._ui.i_but_revert_structure.clicked.connect(self._restore_layers)
        self._ui.i_scr_history.valueChanged.connect(lambda value: self._recall_sw(value))

        self._ui.but_save_session.clicked.connect(self._save_session)

    # ----------------------------------------------------------------------
    def _save_session(self):
        new_file = QtWidgets.QFileDialog.getSaveFileName(self, "Create file", self._working_dir, '.ntr')
        if new_file[0]:
            if '.ntr' in new_file[0]:
                file_name = new_file[0]
            else:
                file_name = "".join(new_file)

            with open(file_name, 'wb') as f:
                self.spectra_fitter.dump_session(f)
                self.ntr_fitter.dump_session(f)

    # ----------------------------------------------------------------------
    def _set_default_settings(self):

        all_settings = [arg for arg in dir(default_settings) if not arg.startswith('_')]
        for setting in all_settings:
            self.settings[setting] = getattr(default_settings, setting)

    # ----------------------------------------------------------------------
    def _show_settings(self, mode):

        self.settings_window.set_options(self.settings)
        self.settings_window.show(mode)

    # ----------------------------------------------------------------------
    def _settings_changed(self, settings, mode_changed):

        for key, value in settings.items():
            self.settings[key] = value

        self.ntr_fitter.set_basic_settings(self.settings)
        if 'intensity' in mode_changed:
            self.ntr_fitter.sw_synchronized = False
        elif 'potential' in mode_changed:
            self._potential_model_edited()

    # ----------------------------------------------------------------------
    def _load_file_clicked(self):

        self._block_potential_fit_signals(True)
        new_file = QtWidgets.QFileDialog.getOpenFileNames(self, "Open file", self._working_dir,
                'HDF5 spectra (*.h5);; Fitter session data set (*.ntr);; Text data set (*.txt);; Fit results (*.res)')

        if new_file[0]:

            self._working_dir = os.path.split(new_file[0][0])[0]

            if '.res' in new_file[1]:
                self._working_file = new_file[0][0]
                self._ui.l_folder.setText(new_file[0][0])
                self._ui.tab_mode.setEnabled(True)
                self._ui.cb_experimental_set.setEnabled(False)
                self.ntr_fitter.potential_solver.reset_fit()
                fit_type = self.ntr_fitter.load_fit_res(new_file[0][0])
                self._prepare_potential_fit_fit_graphs()
                if fit_type == 'pot':
                    self._restore_model(False)
                    if hasattr(self.ntr_fitter.potential_solver, "best_ksi"):
                        ksi = self.ntr_fitter.potential_solver.best_ksi[self.ntr_fitter.potential_solver.cycle]
                    else:
                        ksi = 0
                    self.update_potential_fit_cycles(self.ntr_fitter.potential_solver.cycle, ksi,
                                                         self.ntr_fitter.potential_solver.solution_history[
                                                             self.ntr_fitter.potential_solver.cycle - 1])

                    self._display_potential_fit_cycle()
                else:
                    raise RuntimeError('Not implemented')

            elif '.h5' in new_file[1]:
                self._working_file = os.path.split(self._working_dir)[-1]
                self._ui.l_folder.setText(self._working_dir + '/' + self._working_file)
                num_spectra = self.spectra_fitter.load_data(new_file[0])
                if num_spectra:
                    self._ui.s_scr_spectra.setMaximum(num_spectra - 1)
                    self._ui.s_sp_fit_to.setMaximum(num_spectra - 1)
                    self._ui.s_sp_fit_to.setValue(num_spectra - 1)
                    self._ui.s_sp_fit_from.setMaximum(num_spectra - 1)
                    self._ui.s_scr_spectra.setEnabled(True)
                    self._show_spectra(0)

            elif '.txt' in new_file[1]:
                self._working_file = new_file[0][0]
                self._ui.l_folder.setText(new_file[0][0])
                with open(new_file[0][0]) as f:
                    _file_lines = f.readlines()

                dlg = Import_Dialog(_file_lines, new_file[0][0])
                dlg.load_data.connect(self._load_data)
                dlg.show()

            elif '.ntr' in new_file[1]:
                self._working_file = new_file[0][0]
                self._ui.l_folder.setText(new_file[0][0])
                self._ui.tab_mode.setEnabled(True)
                with open(new_file[0][0], 'rb') as fr:
                    self.spectra_fitter.restore_session(fr)
                    self.ntr_fitter.restore_session(fr, os.path.split(new_file[0][0])[0])
                self._restore_components()
                self._reset_functional_peak()
                self._restore_layers()
                self._restore_model(True)
            else:
                raise RuntimeError('Not implemented')

        self._block_potential_fit_signals(False)

    # ----------------------------------------------------------------------
    def _load_data(self, data, file_name):
        self._ui.tab_mode.setEnabled(True)
        self.ntr_fitter.set_spectroscopy_data(data, file_name= file_name)
        self._potential_model_selected(True)

    # ----------------------------------------------------------------------
    def _new_set_selected(self):

        self._working_set = self._ui.cb_experimental_set.currentText()
        self.ntr_fitter.set_new_data_file(self._working_file, self._working_set)
        self._potential_model_edited()
    # ----------------------------------------------------------------------
    def _get_data_sets(self):

        data = loadmat(self._working_file)
        self._ui.cb_experimental_set.clear()
        for set in data['data_sets'][0]:
            self._ui.cb_experimental_set.addItem(set[0])
        self._working_set = self._ui.cb_experimental_set.currentText()

    # ----------------------------------------------------------------------
    def _refresh_status(self):

        self._ui.tab_potential.setEnabled(self.ntr_fitter.sw_synchronized)

        if self._intensity_worker_state == 'sim':
            self._ui.i_but_simulate.setText('In progress...')
            self._ui.i_but_simulate.setEnabled(False)
            self._ui.i_but_start_fit.setEnabled(False)
        elif self._intensity_worker_state == 'fit':
            self._ui.i_but_simulate.setEnabled(False)
            self._ui.i_but_start_fit.setEnabled(False)
            self._ui.i_but_stop_fit.setEnabled(True)
        else:
            self._ui.i_but_simulate.setText('Simulate')
            self._ui.i_but_simulate.setEnabled(True)
            self._ui.i_but_start_fit.setEnabled(True)
            self._ui.i_but_stop_fit.setEnabled(False)

        if self._spectra_worker_state != 'idle':
            self._ui.s_cmd_fit_one.setEnabled(False)
            self._ui.s_cmd_fit_all.setEnabled(False)
            self._ui.s_cmd_fit_all.setText('Fitting spectra {}...'.format(self.spectra_fitter.current_fit_num))
        else:
            self._ui.s_cmd_fit_one.setEnabled(True)
            self._ui.s_cmd_fit_all.setEnabled(True)
            self._ui.s_cmd_fit_all.setText('Fit series')

        if self.ntr_fitter.angle_shift:
            self._ui.i_dsp_shift.setValue(self.ntr_fitter.angle_shift)
        else:
            self._ui.i_dsp_shift.setValue(0)

        try:
            error = self.error_queue.get(block=False)
        except emptyQueue:
            return
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            try:
                trace = error[2]
                trbck_msg = ''
                while trace.tb_next:
                    trace = trace.tb_next
                    trbck_msg = trace.tb_frame
                msg.setInformativeText(error[1].args[0] + str(trbck_msg))
            except:
                msg.setInformativeText(str(error))
            msg.setWindowTitle("Error")
            msg.exec_()

    # ----------------------------------------------------------------------
    def _update_layout(self, container, widgets):

        layout = container.layout()
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item:
                w = layout.itemAt(i).widget()
                if w:
                    layout.removeWidget(w)
                    w.setVisible(False)

        QtWidgets.QWidget().setLayout(container.layout())
        layout = QtWidgets.QVBoxLayout(container)

        for widget in widgets:
            widget.setVisible(True)
            layout.addWidget(widget, alignment=QtCore.Qt.AlignTop)
        layout.addStretch()

    # ----------------------------------------------------------------------
    def _set_default_plots_to_fitter(self):
        self.ntr_fitter.set_default_plots(self.g_ps_pot_plot, self.g_ps_shift_source_plot, self.g_ps_shift_sim_plot,
                                          self.g_i_source_plot, self.g_i_sim_plot)

        self.spectra_fitter.set_default_plots(self.spectra_experiment_plot, self.spectra_sum_plot,
                                              self.spectra_bcg_plot, self.spectra_plots)

# ----------------------------------------------------------------------
#       Spectra fit code
# ----------------------------------------------------------------------

    def _set_data_to_fitter(self):
        functional_peak_name = None
        for _, widget in self._components_widgets:
            if widget.is_functional():
                functional_peak_name = widget.get_name()

        if functional_peak_name is not None:
            self.ntr_fitter.set_spectroscopy_data(self.spectra_fitter.collect_data_for_fitter(functional_peak_name),
                                                  sample_name=os.path.splitext(os.path.basename(self._working_file))[0],
                                                  directory=self._working_dir)

    # ----------------------------------------------------------------------
    def _cut_range_to_all(self):
        if self._ui.s_rb_range_all.isChecked():
            mode = 'auto'
        else:
            mode = 'manual'

        to =self._ui.s_dsp_cut_to.value()
        start = self._ui.s_dsp_cut_from.value()

        for data in self.spectra_fitter.data:
            data['range'] = mode
            data['limits'] = [to, start]
    # ----------------------------------------------------------------------
    def _cut_range(self, selection):
        spectra_num = self._ui.s_scr_spectra.value()
        if selection.objectName() == 's_rb_range_all':
            self._set_range_uis(False)
            self.spectra_fitter.data[spectra_num]['range'] = 'auto'
        else:
            self._set_range_uis(True)
            self.spectra_fitter.data[spectra_num]['range'] = 'manual'

    # ----------------------------------------------------------------------
    def _set_range_uis(self, status):
        self._ui.s_dsp_cut_from.setEnabled(status)
        self._ui.s_dsp_cut_to.setEnabled(status)
        self.range_line_from.setMovable(status)
        self.range_line_to.setMovable(status)
    # ----------------------------------------------------------------------
    def _check_cut(self, type, value):
        spectra_num = self._ui.s_scr_spectra.value()
        if type == 'from':
            self.range_line_from.setValue(value)
            self.spectra_fitter.data[spectra_num]['limits'][1] = value

            to = self._ui.s_dsp_cut_to.value()
            if to > value:
                self._ui.s_dsp_cut_to.setValue(value)
                self.range_line_to.setValue(value)
                self.spectra_fitter.data[spectra_num]['limits'][0] = value

        elif type == 'to':
            self.range_line_to.setValue(value)
            self.spectra_fitter.data[spectra_num]['limits'][0] = value

            start = self._ui.s_dsp_cut_from.value()
            if start < value:
                self._ui.s_dsp_cut_from.setValue(value)
                self.range_line_from.setValue(value)
                self.spectra_fitter.data[spectra_num]['limits'][1] = value

    # ----------------------------------------------------------------------
    def _restore_components(self):
        num_spectra = self.spectra_fitter.ndata
        if num_spectra:
            self._ui.s_scr_spectra.setMaximum(num_spectra - 1)
            self._ui.s_sp_fit_to.setMaximum(num_spectra - 1)
            self._ui.s_sp_fit_to.setValue(num_spectra - 1)
            self._ui.s_sp_fit_from.setMaximum(num_spectra - 1)
            self._ui.s_scr_spectra.setEnabled(True)
            self._show_spectra(0)

    # ----------------------------------------------------------------------
    def _show_params(self, index):
        need_update = False

        if self.spectra_fitter.bg_params[index]:
            num_bkg_components = len(self.spectra_fitter.bg_params[index].keys())
        else:
            num_bkg_components = 0
        num_bkg_widgets = len(self._backgrounds_widgets)

        if num_bkg_components > num_bkg_widgets:
            for _ in range(num_bkg_widgets, num_bkg_components):
                widget_id = random.random()
                widget = Background(self, widget_id)
                widget.delete_component.connect(self._delete_component)
                widget.widget_edited.connect(self._sim_spectra)
                self._backgrounds_widgets.append((widget_id, widget))
            need_update = True
        elif num_bkg_components < num_bkg_widgets:
            self._layers_widgets[num_bkg_components - num_bkg_widgets:] = []
            need_update = True

        if num_bkg_components:
            counter = 0
            for type, params in self.spectra_fitter.bg_params[index].items():
                self._backgrounds_widgets[counter][1].set_values(type, params)
                counter += 1

        num_peak_components = len(self.spectra_fitter.peaks_info[index])
        num_peak_widgets = len(self._components_widgets)

        if num_peak_components > num_peak_widgets:
            for counter in range(num_peak_widgets, num_peak_components):
                widget_id = random.random()
                if self.spectra_fitter.peaks_info[index][counter]['peakType'] == 'voigth':
                    widget = Single_line(self, widget_id)
                else:
                    widget = Doublet(self, widget_id)
                widget.delete_component.connect(self._delete_component)
                widget.widget_edited.connect(self._sim_spectra)
                widget.set_functional_signal.connect(self._set_new_functional_peak)
                self._components_widgets.append((widget_id, widget))
            need_update = True
        elif num_peak_components < num_peak_widgets:
            self._layers_widgets[num_peak_components - num_peak_widgets:] = []
            need_update = True

        for index, info in enumerate(self.spectra_fitter.peaks_info[index]):
            id, type = self._components_widgets[index][1].get_widget_type()
            if type != info['peakType']:
                if info['peakType'] == 'voigth':
                    widget = Single_line(self, id)
                else:
                    widget = Doublet(self, id)
                self._components_widgets[index] = (id, widget)
                need_update = True
            self._components_widgets[index][1].set_values(info['name'], info['params'])

        if need_update:
            self._show_components()

    # ----------------------------------------------------------------------
    def _range_fitting(self, selection):
        part_fit = selection.objectName() == 's_rb_part_fit'
        self._ui.s_sp_fit_from.setEnabled(part_fit)
        self._ui.s_sp_fit_to.setEnabled(part_fit)

    # ----------------------------------------------------------------------
    def _check_range(self, type, value):
        if type == 'from':
            to = self._ui.s_sp_fit_to.value()
            if to < value:
                self._ui.s_sp_fit_to.setValue(value)
        elif type == 'to':
            start = self._ui.s_sp_fit_from.value()
            if start > value:
                self._ui.s_sp_fit_from.setValue(value)

    # ----------------------------------------------------------------------
    def _make_default_spectra_fit_graphics(self):

        # potential simulation tab
        self.spectra_layout = QtWidgets.QGridLayout(self._ui.spectra_plots)
        self.spectra_layout.setObjectName("spectra_plots_layout")

        self.spectra_widget = pg.PlotWidget(self._ui.spectra_plots)
        self.spectra_widget.setObjectName("spectra_plots")
        self.spectra_layout.addWidget(self.spectra_widget, 0, 0, 1, 1)

        self.spectra_graphs_layout = pg.GraphicsLayout(border=(10, 10, 10))

        self.spectra_widget.setBackground('w')
        self.spectra_widget.setStyleSheet("")
        self.spectra_widget.setCentralItem(self.spectra_graphs_layout)

        self.range_line_from = pg.InfiniteLine(angle=90, movable=False)
        self.range_line_from.sigPositionChanged.connect(lambda: self._ui.s_dsp_cut_from.setValue(self.range_line_from.value()))
        self.range_line_to = pg.InfiniteLine(angle=90, movable=False)
        self.range_line_to.sigPositionChanged.connect(lambda: self._ui.s_dsp_cut_to.setValue(self.range_line_to.value()))

        self.spectra_plot = self.spectra_graphs_layout.addPlot()
        self.spectra_plot.getViewBox().invertX(True)
        self.spectra_plot.addItem(self.range_line_from, ignoreBounds=True)
        self.spectra_plot.addItem(self.range_line_to, ignoreBounds=True)
        self.spectra_experiment_plot = self.spectra_plot.plot([], name='Experiment', **lookandfeel.EXPERIMENT_SPECTRA)
        self.spectra_sum_plot = self.spectra_plot.plot([], name='Summary', **lookandfeel.SUM_SPECTRA)

        self.spectra_bcg_plot = self.spectra_plot.plot([], name='Background', **lookandfeel.BACKGROUND)

        self.spectra_plots = []
        pen_counter = 0
        for ind in range(self.settings['MAX_NUM_COMPONENTS']):
            color = lookandfeel.PLOT_COLORS[pen_counter % len(lookandfeel.PLOT_COLORS)]

            self.spectra_plots.append(self.spectra_plot.plot([], name='Component_{}'.format(ind),
                                                               pen = pg.mkPen(color=color, **lookandfeel.COMPONENT)))
            pen_counter += 1

    # ----------------------------------------------------------------------
    def _show_spectra(self, numer):
        self._ui.s_lb_spectra_num.setText('Spectra {} from {}'.format(numer, self.spectra_fitter.ndata-1))
        self.spectra_fitter.plot_spectra(numer)
        if self.spectra_fitter.data[numer]['range'] == 'auto':
            self._ui.s_rb_range_all.setChecked(True)
            self._set_range_uis(False)
        else:
            self._ui.s_rb_range_cut.setChecked(True)
            self._set_range_uis(True)

        self._ui.s_dsp_cut_from.setValue(self.spectra_fitter.data[numer]['limits'][1])
        self.range_line_from.setValue(self.spectra_fitter.data[numer]['limits'][1])
        self._ui.s_dsp_cut_to.setValue(self.spectra_fitter.data[numer]['limits'][0])
        self.range_line_to.setValue(self.spectra_fitter.data[numer]['limits'][0])

        self._show_params(numer)

    # ----------------------------------------------------------------------
    def _add_component(self, type):
        widget_id = random.random()
        widget = None

        if type == 'bkg':
            widget = Background(self, widget_id)
            self._backgrounds_widgets.append((widget_id, widget))
        elif type == 'line':
            widget = Single_line(self, widget_id)
            self._components_widgets.append((widget_id, widget))
            widget.set_functional_signal.connect(self._set_new_functional_peak)
        elif type == 'dbl':
            widget = Doublet(self, widget_id)
            self._components_widgets.append((widget_id, widget))
            widget.set_functional_signal.connect(self._set_new_functional_peak)

        widget.delete_component.connect(self._delete_component)
        widget.widget_edited.connect(self._sim_spectra)

        self._show_components()

    # ----------------------------------------------------------------------
    def _set_new_functional_peak(self, id_to_set):
        for _, widget in self._components_widgets:
            widget.set_functional(False)
        for id, widget in self._components_widgets:
            if id_to_set == id:
                widget.set_functional(True)
                self.spectra_fitter.functional_peak = widget.get_name()

    # ----------------------------------------------------------------------
    def _reset_functional_peak(self):
        for _, widget in self._components_widgets:
            if widget.get_name() == self.spectra_fitter.functional_peak:
                widget.set_functional(True)

    # ----------------------------------------------------------------------
    def _show_components(self):
        widget_list = []
        for _, widget in self._backgrounds_widgets + self._components_widgets:
            widget_list.append(widget)

        self._update_layout(self._ui.s_wc_models, widget_list)

    # ----------------------------------------------------------------------
    def _delete_component(self, idx):
        for ind, data in enumerate(self._backgrounds_widgets):
            if data[0] == idx:
                del self._backgrounds_widgets[ind]
                self._show_components()
                return
        for ind, data in enumerate(self._components_widgets):
            if data[0] == idx:
                del self._components_widgets[ind]
                self._show_components()
                return

    # ----------------------------------------------------------------------
    def _collect_components_model(self):
        bg_params = {}
        for record in self._backgrounds_widgets:
            type, params = record[1].get_values()
            bg_params[type] = params

        peaks_info = []
        for record in self._components_widgets:
            peaks_info.append(record[1].get_values())

        return bg_params, peaks_info

    # ----------------------------------------------------------------------
    def _sim_spectra(self):
        index = self._ui.s_scr_spectra.value()
        bg_params, peaks_info = self._collect_components_model()
        self.spectra_fitter.set_peak_params([index], bg_params, peaks_info)
        self.spectra_fitter.plot_spectra(index)

    # ----------------------------------------------------------------------
    def _fit_current_spectra(self):
        self._spectra_worker = ExcThread(self._spectra_fit_worker, 'spectra_worker', self.error_queue,
                                         [self._ui.s_scr_spectra.value()])
        self._spectra_worker.start()

    # ----------------------------------------------------------------------
    def _fit_all_spectra(self):

        if self._ui.s_rb_part_fit.isChecked():
            indexes = range(self._ui.s_sp_fit_from.value(), self._ui.s_sp_fit_to.value() + 1)
        else:
            indexes = range(self._ui.s_sp_fit_to.maximum() + 1)

        self._spectra_worker = ExcThread(self._spectra_fit_worker, 'spectra_worker', self.error_queue, indexes)
        self._spectra_worker.start()

    # ----------------------------------------------------------------------
    def _spectra_fit_worker(self, indexes):
        self._spectra_worker_state = 'working'
        bg_params, peaks_info = self._collect_components_model()
        self.spectra_fitter.set_peak_params(indexes, bg_params, peaks_info)
        try:
            self.spectra_fitter.fit(indexes)
        except Exception as err:
            self.error_queue.put(err)
        self._show_params(self._ui.s_scr_spectra.value())
        self.spectra_fitter.plot_spectra(self._ui.s_scr_spectra.value())
        self._spectra_worker_state = 'idle'

# ----------------------------------------------------------------------
#       Intensity fit code
# ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def add_message_to_fit_history(self, msg):

        self._ui.p_l_fit_history.addItem(msg)

    # ----------------------------------------------------------------------
    def update_sw_srb(self, num_sws):

        if num_sws > 0:
            self._ui.i_scr_history.setEnabled(True)
            self._ui.i_scr_history.setMaximum(num_sws-1)
            self._ui.i_scr_history.setValue(num_sws-1)
        else:
            self._ui.i_scr_history.setEnabled(False)

    # ----------------------------------------------------------------------
    def _make_default_intensity_fit_graphics(self):

        # potential simulation tab
        self.sim_int_layout = QtWidgets.QGridLayout(self._ui.i_plots)
        self.sim_int_layout.setObjectName("int_sim_layout")

        self.sim_int_widget = pg.PlotWidget(self._ui.i_plots)
        self.sim_int_widget.setObjectName("int_sim_graphs")
        self.sim_int_layout.addWidget(self.sim_int_widget, 0, 0, 1, 1)

        self.sim_int_graphs_layout = pg.GraphicsLayout(border=(10, 10, 10))

        self.sim_int_widget.setBackground('w')
        self.sim_int_widget.setStyleSheet("")
        self.sim_int_widget.setCentralItem(self.sim_int_graphs_layout)

        self.g_int_item = self.sim_int_graphs_layout.addPlot()
        self.g_i_source_plot = self.g_int_item.plot([], name='Data shifts', **lookandfeel.CURRENT_SOURCE_SHIFT)
        self.g_i_sim_plot = self.g_int_item.plot([], name='Sim shifts', **lookandfeel.CURRENT_SIM_SHIFT)

    # ----------------------------------------------------------------------
    def _correct_intensity(self, state):

        self.ntr_fitter.intensity_correction = state
        self.ntr_fitter.correct_intensity()

    # ----------------------------------------------------------------------
    def _recall_sw(self, ind):

        self.ntr_fitter.request_sw_from_history(ind)
        self._restore_layers()

    # ----------------------------------------------------------------------
    def _restore_layers(self):

        num_existing_widgets = len(self._layers_widgets)
        num_layers_in_structure = len(self.ntr_fitter.structure)
        if num_layers_in_structure > num_existing_widgets:
            for ind in range(num_layers_in_structure-num_existing_widgets):
                widget = self._get_layer_widget(num_existing_widgets + ind)
                self._layers_widgets.append(widget)
            self._update_layout(self._ui.i_wc_layers, self._layers_widgets)
        elif num_layers_in_structure < num_existing_widgets:
            self._layers_widgets[num_layers_in_structure-num_existing_widgets:] = []
            self._update_layout(self._ui.i_wc_layers, self._layers_widgets)

        for ind, layer in enumerate(self.ntr_fitter.structure):
            self._layers_widgets[ind].set_values(layer)

        self._change_functional_layer(self.ntr_fitter.functional_layer)
        if self.ntr_fitter.sw:
            self.ntr_fitter.sw_synchronized = True

    # ----------------------------------------------------------------------
    def _collect_layers_info(self):

        structure = []
        functional = []
        for ind, widget in enumerate(self._layers_widgets):
            structure.append(widget.get_values())
            if widget.is_functional():
                functional = ind

        return structure, functional

    # ----------------------------------------------------------------------
    def _update_structure(self):

        self.ntr_fitter.structure, self.ntr_fitter.functional_layer = self._collect_layers_info()
        self.ntr_fitter.set_new_functional_layer()

    # ----------------------------------------------------------------------
    def _i_start_fit(self):

        self._intensity_worker = ExcThread(self._intensity_fit_worker, 'intensity_worker', self.error_queue)
        self._intensity_worker.start()

    # ----------------------------------------------------------------------
    def _i_stop_fit(self):
        self.ntr_fitter.stop_fit()

    # ----------------------------------------------------------------------
    def _intensity_fit_worker(self):

        self._ui.p_l_fit_history.clear()
        self._update_structure()
        param_list = []
        for ind, widget in enumerate(self._layers_widgets):
            param_list += widget.get_fittable_parameters()
        if param_list:
            try:
                self._intensity_worker_state = 'fit'
                self.ntr_fitter.do_intensity_fit (param_list)
            except Exception as err:
                self.error_queue.put(err)
            finally:
                self._intensity_worker_state = 'idle'
            if self._ui.i_scr_history.value() != self._ui.i_scr_history.maximum():
                self._ui.i_scr_history.setValue(self._ui.i_scr_history.maximum())
            else:
                self._recall_sw(-1)

        else:
            self.error_queue.put('No fit parameters found!')

    # ----------------------------------------------------------------------
    def _sim_intensity(self):

        self._intensity_worker = ExcThread(self._intensity_sim_worker, 'intensity_worker', self.error_queue)
        self._intensity_worker.start()

    # ----------------------------------------------------------------------
    def _intensity_sim_worker(self):

        self._intensity_worker_state = 'sim'
        try:
            if not self.ntr_fitter.sw_synchronized:
                self._update_structure()
                self.ntr_fitter.request_sw_from_server()
            self.ntr_fitter.sim_profile_intensity()
            self._update_potential_widgets()
            self.ntr_fitter.sw_synchronized = True
        except Exception as err:
            self.error_queue.put(err)
        finally:
            self._intensity_worker_state = 'idle'

    # ----------------------------------------------------------------------
    def _change_functional_layer(self, layer):

        for widget in self._layers_widgets:
            widget.set_functional(False)
        self._layers_widgets[layer].set_functional(True)
        self._update_structure()
        if self.ntr_fitter and self.ntr_fitter.sw_synchronized:
            self.ntr_fitter.sim_profile_intensity()

    # ----------------------------------------------------------------------
    def _order_layers(self, layer, command):

        if command == 'add_above':
            ind = max(layer-1, 0)
            self._layers_widgets.insert(ind, self._get_layer_widget(ind))
        elif command == 'add_below':
            self._layers_widgets.insert(layer+1, self._get_layer_widget(layer+1))
        elif command == 'move_up':
            if layer > 0:
                self._layers_widgets[layer-1], self._layers_widgets[layer] = \
                    self._layers_widgets[layer], self._layers_widgets[layer-1]
        elif command == 'move_down':
            if layer < len(self._layers_widgets) - 1:
                self._layers_widgets[layer], self._layers_widgets[layer+1] = \
                    self._layers_widgets[layer+1], self._layers_widgets[layer]
        elif command == 'delete':
            del self._layers_widgets[layer]

        for ind in range(len(self._layers_widgets)):
            self._layers_widgets[ind].update_layer_num(ind)

        if not self._layers_widgets:
            self._layers_widgets.insert(0, self._get_layer_widget(0))
            self._layers_widgets[0].set_functional(True)

        self._update_layout(self._ui.i_wc_layers, self._layers_widgets)
        self._update_structure()

    # ----------------------------------------------------------------------
    def _get_layer_widget(self, layer_num):
         widget = LayerWidget(self, layer_num)
         # widget.widget_edited.connect(self._sim_intensity)
         widget.add_delete_move_layer.connect(self._order_layers)
         widget.layer_set_as_functional.connect(self._change_functional_layer)
         widget.widget_edited.connect(self._layer_changed)
         return widget

    # ----------------------------------------------------------------------
    def _layer_changed(self):
        self.ntr_fitter.sw_synchronized = False

# ----------------------------------------------------------------------
#       Potential fit code
# ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def _restore_model(self, reset_model):
        self._block_potential_fit_signals(True)
        self._potential_model_widgets = []
        self._ui.p_sb_max_potential.setValue(self.settings['VOLT_MAX'])

        if self.ntr_fitter.potential_model:
            for model in self.list_of_models:
                if model['code'] == self.ntr_fitter.potential_model['code']:
                    if refresh_combo_box(self._ui.p_cb_cmb_model, model['name']):
                        self._ui.p_sb_deg_freedom.setValue(self.ntr_fitter.potential_model['num_depth_dof']
                                                           - self.ntr_fitter.potential_model['only_voltage_dof'])
                        self._potential_model_selected(False)

        if reset_model:
            try:
                counter = 1
                for widget in self._potential_model_widgets:
                    if isinstance(widget, TopBottomPotential):
                        widget.set_values((self.ntr_fitter._last_sim_potential['v_set'][0],
                                           self.ntr_fitter._last_sim_potential['v_set'][-1]))
                    elif isinstance(widget, BreakingPoint):
                        widget.set_values(((self.ntr_fitter._last_sim_potential['d_set'][counter] -
                                           self.ntr_fitter._last_sim_potential['d_set'][0])*1e10,
                                           self.ntr_fitter._last_sim_potential['v_set'][counter]))
                        counter = counter + 1
            except:
                pass

            self._potential_model_edited()
        self._block_potential_fit_signals(False)
    # ----------------------------------------------------------------------
    def _block_potential_fit_signals(self, flag):

        self._ui.b_select_file.blockSignals(flag)
        self._ui.p_cb_cmb_model.blockSignals(flag)
        self._ui.p_sb_deg_freedom.blockSignals(flag)
        self._ui.cb_experimental_set.blockSignals(flag)
        self._ui.p_srb_cycle.blockSignals(flag)

    # ----------------------------------------------------------------------
    def _make_default_potential_fit_graphics(self):

        # potential simulation tab
        self.sim_pot_layout = QtWidgets.QGridLayout(self._ui.p_sim_res)
        self.sim_pot_layout.setObjectName("pot_sim_layout")

        self.sim_pot_widget = pg.PlotWidget(self._ui.p_sim_res)
        self.sim_pot_widget.setObjectName("pot_sim_graphs")
        self.sim_pot_layout.addWidget(self.sim_pot_widget, 0, 0, 1, 1)

        self.sim_pot_graphs_layout = pg.GraphicsLayout(border=(10, 10, 10))

        self.sim_pot_widget.setBackground('w')
        self.sim_pot_widget.setStyleSheet("")
        self.sim_pot_widget.setCentralItem(self.sim_pot_graphs_layout)

        self.g_ps_pot_item = self.sim_pot_graphs_layout.addPlot(title="Potential", row=1, col=1)
        self.g_ps_pot_item.setYRange(-self.settings['VOLT_MAX'], self.settings['VOLT_MAX'])
        self.g_ps_pot_plot = self.g_ps_pot_item.plot([], name='Potential', **lookandfeel.CURRENT_POTENTIAL_STYLE)

        self.g_ps_shift_item = self.sim_pot_graphs_layout.addPlot(title="Shifts", row=2, col=1)
        self.g_ps_shift_source_plot = self.g_ps_shift_item.plot([], name='Data shifts', **lookandfeel.CURRENT_SOURCE_SHIFT)
        self.g_ps_shift_sim_plot = self.g_ps_shift_item.plot([], name='Sim shifts', **lookandfeel.CURRENT_SIM_SHIFT)

        # potential fit tab
        self.fit_pot_layout = QtWidgets.QGridLayout(self._ui.p_fit_res)
        self.fit_pot_layout.setObjectName("pot_fit_layout")

        self.fit_pot_widget = pg.PlotWidget(self._ui.p_fit_res)
        self.fit_pot_widget.setObjectName("pot_fit_graphs")
        self.fit_pot_layout.addWidget(self.fit_pot_widget, 0, 0, 1, 1)

        self.fit_pot_graphs_layout = pg.GraphicsLayout(border=(10, 10, 10))

        self.fit_pot_widget.setBackground('w')
        self.fit_pot_widget.setStyleSheet("")
        self.fit_pot_widget.setCentralItem(self.fit_pot_graphs_layout)

    # ----------------------------------------------------------------------
    def _update_potential_widgets(self):
        for widget in self._potential_model_widgets:
            if hasattr(widget, 'change_layer_thickness'):
                widget.change_layer_thickness(self.ntr_fitter.structure[self.ntr_fitter.functional_layer]['thick'])

    # ----------------------------------------------------------------------
    def _change_potential_fit_v_range(self):

        self.settings['VOLT_MAX'] = self._ui.p_sb_max_potential.value()
        self.ntr_fitter.set_basic_settings(self.settings)

    # ----------------------------------------------------------------------
    def _fill_potential_fit_combos(self):

        self.list_of_models = get_model_list()
        for model in self.list_of_models:
            self._ui.p_cb_cmb_model.addItem(model['name'])

    # ----------------------------------------------------------------------
    def _change_degrees(self, do_refresh):

        self.ntr_fitter.potential_model['num_depth_dof'] = int(self._ui.p_sb_deg_freedom.value())
        if self.ntr_fitter.potential_model['num_depth_dof'] + 1 > len(self._potential_model_widgets):
            for point in range(self.ntr_fitter.potential_model['num_depth_dof'] - len(self._potential_model_widgets) + 1):
                for widget in self.ntr_fitter.potential_model['additional_widgets']:
                    self._potential_model_widgets.append(self._get_potential_fit_widget(widget, point + len(self._potential_model_widgets)))
        else:
            for _ in range(len(self._potential_model_widgets) - self.ntr_fitter.potential_model['num_depth_dof'] - 1):
                for _ in self.ntr_fitter.potential_model['additional_widgets']:
                    del self._potential_model_widgets[-1]

        self._update_layout(self._ui.p_wc_deg_freedom, self._potential_model_widgets)

        start_depths = np.linspace(0, self.ntr_fitter.structure[self.ntr_fitter.functional_layer]['thick'],
                                   self.ntr_fitter.potential_model['num_depth_dof'] +
                                   self.ntr_fitter.potential_model['only_voltage_dof'])

        counter = 1
        for widget in self._potential_model_widgets:
            if isinstance(widget, BreakingPoint):
                raw_ans = widget.getValues()
                widget.set_values((start_depths[counter], raw_ans[0][1]))
                counter = counter + 1

        if do_refresh:
            self._potential_model_edited()
    # ----------------------------------------------------------------------
    def _potential_model_selected(self, do_refresh):

        selected_model = str(self._ui.p_cb_cmb_model.currentText())
        for model in self.list_of_models:
            if model['name'] == selected_model:
                self._block_potential_fit_signals(True)
                self._potential_model_widgets = []
                self.ntr_fitter.potential_model = model
                self._ui.p_sb_deg_freedom.setValue(self.ntr_fitter.potential_model['num_depth_dof'])
                self._ui.p_sb_deg_freedom.setEnabled(not self.ntr_fitter.potential_model['fixed_depth_dof'])
                self._potential_model_widgets.append(self._get_potential_fit_widget(model['default_widget']))
                self._change_degrees(do_refresh)
                self._block_potential_fit_signals(False)

    # ----------------------------------------------------------------------
    def _get_potential_fit_widget(self, name, point_num=None):

        if name == 'top_bottom_potential':
             widget = TopBottomPotential(self, self.ntr_fitter.structure[self.ntr_fitter.functional_layer]['thick'], self.settings['VOLT_MAX'])
             widget.widget_edited.connect(self._potential_model_edited)
             return widget

        elif name == 'breaking_point':
            widget = BreakingPoint(self, point_num, self.ntr_fitter.structure[self.ntr_fitter.functional_layer]['thick'] / 2,
                                   self.ntr_fitter.structure[self.ntr_fitter.functional_layer]['thick'],
                                   self.settings['VOLT_MAX'])
            widget.widget_edited.connect(self._potential_model_edited)
            return widget

        raise RuntimeError('Cannot find widget for model: {}'.format(name))

    # ----------------------------------------------------------------------
    def _get_potential_fit_variable_values(self):

        new_set = []
        for widget in self._potential_model_widgets:
            raw_ans = widget.getValues()
            for pair in raw_ans:
                new_set.append(pair)

        new_set = np.vstack(new_set)
        return new_set[new_set.argsort(axis=0)[:, 0]]

    # ----------------------------------------------------------------------
    def _potential_model_edited(self):

        new_set = self._get_potential_fit_variable_values()

        if self.ntr_fitter.sw:
            self.ntr_fitter.sim_profile_shifts(new_set[:, 0], new_set[:, 1])

    # ----------------------------------------------------------------------
    def _prepare_potential_fit_fit_graphs(self):

        self.fit_pot_graphs_layout.clear()
        self.ntr_fitter.potential_solver.set_external_graphs(self.fit_pot_graphs_layout)

    # ----------------------------------------------------------------------
    def _start_stop_pot_fit(self):

        if self._potential_worker_state == 'idle':
            self._ui.p_but_pot_fit.setText('Stop')
            self._prepare_potential_fit_fit_graphs()
            self.ntr_fitter.potential_solver.reset_fit()

            self._potential_worker = ExcThread(self._fitter_potential_fit_worker, 'fitter_worker', self.error_queue)
            self._potential_worker.start()
        else:
            self._ui.p_but_pot_fit.setText('Start')
            self.ntr_fitter.stop_fit()

    # ----------------------------------------------------------------------
    def _fitter_potential_fit_worker(self):

        self._potential_worker_state = 'run'
        self.ntr_fitter.do_potential_fit(self._get_potential_fit_variable_values())
        self._potential_worker_state = 'idle'

    # ----------------------------------------------------------------------
    def update_potential_fit_cycles(self, num_cycles, ksi, best_solution):

        self._block_potential_fit_signals(True)
        self._ui.p_lb_fit_cycle.setText('Cycle: {}. Ksi: {:.2e}'.format(num_cycles, ksi))
        self._ui.p_srb_cycle.setMaximum(num_cycles)
        self._ui.p_srb_cycle.setValue(num_cycles)
        self._ui.p_lb_fit_status.setText('Completed cycles: {}. Best ksi : {:.2e}'.format(num_cycles, ksi))

        counter = 1
        for widget in self._potential_model_widgets:
            if isinstance(widget, TopBottomPotential):
                widget.set_values((best_solution[1][0], best_solution[1][-1]))
            elif isinstance(widget, BreakingPoint):
                widget.set_values(((best_solution[0][counter] - best_solution[0][0])*1e10, best_solution[1][counter]))
                counter = counter + 1

        self._block_potential_fit_signals(False)

    # ----------------------------------------------------------------------
    def _display_potential_fit_cycle(self):
        try:
            cycle = int(self._ui.p_srb_cycle.value())
            ksi, solution = self.ntr_fitter.potential_solver.show_results(cycle)
            self._ui.p_lb_fit_cycle.setText('Cycle: {}. Ksi: {:.2e}'.format(cycle, ksi))
            counter = 1
            for widget in self._potential_model_widgets:
                if isinstance(widget, TopBottomPotential):
                    widget.set_values((solution[1][0], solution[1][-1]))
                elif isinstance(widget, BreakingPoint):
                    widget.set_values(((solution[0][counter] - solution[0][0])*1e10* solution[1][counter]))
                    counter = counter + 1
        except:
            pass

    # ----------------------------------------------------------------------
    def _prepare_potential_fit_set(self):
        new_file = QtWidgets.QFileDialog.getSaveFileName(self, "Create file", self._working_dir, '.set')
        if new_file:
            if '.set' in new_file[0]:
                file_name = new_file[0]
            else:
                file_name = "".join(new_file)
            self.ntr_fitter.dump_fit_set(file_name, generate_data_set=True, start_values=self._get_potential_fit_variable_values())