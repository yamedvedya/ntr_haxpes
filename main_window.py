import numpy as np
from propagatingThread import ExcThread
import pyqtgraph as pg
from widgets.main_window_ui import Ui_MainWindow
from settings_dialog import Settings_Window
from potential_models import get_model_list
from fitter import NTR_fitter
from scipy.io import loadmat
from potential_widgets import *
import look_and_feel as lookandfeel
import default_settings as default_settings
import os

from queue import Queue
from queue import Empty as emptyQueue

# ----------------------------------------------------------------------
class NTR_Window(QtWidgets.QMainWindow):

    settings = {}
    _model_widgets = []

    # ----------------------------------------------------------------------
    def __init__(self, options):
        """
        """
        super(NTR_Window, self).__init__()

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        self._local_error_queue = Queue()

        self._ui.tab_intensity.setEnabled(False)
        self._ui.tab_potential.setEnabled(False)

        self._set_default_settings()
        self.settings_window = Settings_Window()

        self._pot_model_grid = QtWidgets.QGridLayout(self._ui.p_wc_deg_freedom)

        self._fill_combos()
        self.fitter = NTR_fitter(self)
        self.fitter.set_basic_settings(self.settings)
        self.fitter.STAND_ALONE_MODE = False
        self.settings_window.fill_methods(self.fitter.METHODS)

        self._working_dir = os.getcwd()
        self._parse_options(options)
        self._connect_actions()
        self._make_default_graphics()

        self._refresh_status_timer = QtCore.QTimer(self)
        self._refresh_status_timer.timeout.connect(self._refresh_status)
        self._refresh_status_timer.start(500)

    # ----------------------------------------------------------------------
    def _connect_actions(self):

        self._ui.b_select_file.clicked.connect(self._load_file_clicked)
        self._ui.p_cb_cmb_model.currentIndexChanged.connect(self._potential_model_selected)
        self._ui.p_sb_deg_freedom.valueChanged.connect(self._change_degrees)

        self._ui.p_but_start_pot_fit.clicked.connect(self._start_pot_fit)
        self._ui.p_but_stop_pot_fit.clicked.connect(self._stop_pot_fit)
        self._ui.p_but_prepare_fit_set.clicked.connect(self._prepare_fit_set)

        self._ui.but_settings_potential.clicked.connect(self._show_settings)
        self._ui.cb_experimental_set.currentIndexChanged.connect(self._new_set_selected)

        self.settings_window.settings_changed.connect(self._settings_changed)
        self._ui.p_sb_max_potential.valueChanged.connect(self._change_v_range)

        self._ui.p_srb_cycle.valueChanged.connect(self._display_cycle)

    # ----------------------------------------------------------------------
    def _block_signals(self, flag):

        self._ui.b_select_file.blockSignals(flag)
        self._ui.p_cb_cmb_model.blockSignals(flag)
        self._ui.p_sb_deg_freedom.blockSignals(flag)
        self._ui.cb_experimental_set.blockSignals(flag)
        self._ui.p_srb_cycle.blockSignals(flag)

    # ----------------------------------------------------------------------
    def _refresh_status(self):

        try:
            error = self._local_error_queue.get(block=False)
        except emptyQueue:
            return
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            trace = error[2]
            while trace.tb_next:
                trace = trace.tb_next
                trbck_msg = trace.tb_frame
            msg.setInformativeText(error[1].args[0] + str(trbck_msg))
            msg.setWindowTitle("Error")
            msg.exec_()

    # ----------------------------------------------------------------------
    def _set_default_settings(self):

        all_settings = [arg for arg in dir(default_settings) if not arg.startswith('_')]
        for setting in all_settings:
            self.settings[setting] = getattr(default_settings, setting)

    # ----------------------------------------------------------------------
    def _parse_options(self, options):

        if hasattr(options, 'data_file'):
            if options.data_file and options.set:
                self._ui.l_folder.setText(options.data_file)
                self._working_file = options.data_file
                self._get_data_sets()
                if hasattr(options, 'set'):
                    if self.refreshComboBox(self._ui.cb_experimental_set, options.set):
                        self._working_set = options.set
                        self.fitter.set_new_data_file(options.data_file, options.set)
                        self._ui.tab_intensity.setEnabled(True)

                    if hasattr(options, 'sw_file'):
                        if options.sw_file:
                            self.fitter.set_new_sw_file(options.sw_file)
                            self._ui.tab_potential.setEnabled(True)

                            if hasattr(options, 'model') and hasattr(options, 'n_points'):
                                if self._set_initial_model(options.model, options.n_points):
                                    self.fitter.set_model(self._current_model['code'], self._current_model['num_deg_freedom'])


    # ----------------------------------------------------------------------
    def _make_default_graphics(self):

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
    def _show_settings(self):

        self.settings_window.set_options(self.settings)
        self.settings_window.show()

    # ----------------------------------------------------------------------
    def _settings_changed(self, settings):

        for key, value in settings.items():
            self.settings[key] = value

        self.fitter.set_basic_settings(self.settings)
        self._potential_model_edited()

    # ----------------------------------------------------------------------
    def _change_v_range(self):

        self.settings['VOLT_MAX'] = self._ui.p_sb_max_potential.value()
        self.fitter.set_basic_settings(self.settings)

    # ----------------------------------------------------------------------
    def _fill_combos(self):

        self.list_of_models = get_model_list()
        for model in self.list_of_models:
            self._ui.p_cb_cmb_model.addItem(model['name'])

    # ----------------------------------------------------------------------
    def _load_file_clicked(self):

        self._block_signals(True)
        new_file = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", self._working_dir, 'Data set (*.mat);; Fit results (*.res);; Fit set (*.fset)')
        if new_file[0]:

            self._working_dir = os.path.split(new_file[0])[0]
            self._working_file = new_file[0]
            self._ui.l_folder.setText(new_file[0])

            if '.res' in new_file[1]:
                self._ui.cb_experimental_set.setEnabled(False)
                fit_type, pot_model, num_depth_point = self.fitter.load_fit_res(new_file[0])
                if fit_type == 'pot':
                    self._set_initial_model(pot_model, num_depth_point)
                    self._potential_model_selected()
                    self.fit_pot_graphs_layout.clear()
                    self.fitter.solver.set_external_graphs(self.fit_pot_graphs_layout)
                    self._display_cycle()
                    self._ui.tab_intensity.setEnabled(True)
                    self._ui.tab_potential.setEnabled(True)
                else:
                    raise RuntimeError('Not implemented')

            elif '.mat' in new_file[1]:
                self._ui.cb_experimental_set.setEnabled(True)
                self._get_data_sets()
                self.fitter.set_new_data_file(self._working_file, self._working_set)
                self._ui.tab_intensity.setEnabled(True)

                file_list = [f for f in os.listdir(self._working_dir) if os.path.isfile(os.path.join(self._working_dir, f))
                                                       and os.path.splitext(f)[-1] == ".sw"]

                for file in file_list:
                    if os.path.splitext(file)[0] == os.path.splitext(os.path.split(new_file[0])[1])[0]:
                        self.fitter.set_new_sw_file(os.path.join(self._working_dir, file))
                        self._potential_model_selected()
                        self._ui.tab_potential.setEnabled(True)
                pass
            else:
                raise RuntimeError('Not implemented')

        self._block_signals(False)
    # ----------------------------------------------------------------------
    def _new_set_selected(self):

        self._working_set = self._ui.cb_experimental_set.currentText()
        self.fitter.set_new_data_file(self._working_file, self._working_set)
        self._potential_model_edited()
    # ----------------------------------------------------------------------
    def _get_data_sets(self):

        data = loadmat(self._working_file)
        self._ui.cb_experimental_set.clear()
        for set in data['data_sets'][0]:
            self._ui.cb_experimental_set.addItem(set[0])
        self._working_set = self._ui.cb_experimental_set.currentText()

    # ----------------------------------------------------------------------
    def _set_initial_model(self, set_model, points=None):

        for model in self.list_of_models:
            if model['code'] == set_model:
                if self.refreshComboBox(self._ui.p_cb_cmb_model, model['name']):
                    self._potential_model_selected()

                if points:
                    self._ui.p_sb_deg_freedom.setValue(int(points))
                    self._change_degrees()

                return True
        return False

    # ----------------------------------------------------------------------
    def _change_degrees(self):

        self._current_model['num_deg_freedom'] = int(self._ui.p_sb_deg_freedom.value())
        if self._current_model['num_deg_freedom'] + 1 > len(self._model_widgets):
            for point in range(self._current_model['num_deg_freedom'] - len(self._model_widgets) + 1):
                for widget in self._current_model['additional_widgets']:
                    self._model_widgets.append(self._get_widget(widget, point + len(self._model_widgets)))
        else:
            for _ in range(len(self._model_widgets) - self._current_model['num_deg_freedom'] - 1):
                for _ in self._current_model['additional_widgets']:
                    del self._model_widgets[-1]

        self._update_layouts(self._ui.p_wc_deg_freedom, self._model_widgets)

        self._potential_model_edited()
    # ----------------------------------------------------------------------
    def _potential_model_selected(self):

        self._model_widgets = []
        selected_model = str(self._ui.p_cb_cmb_model.currentText())
        for model in self.list_of_models:
            if model['name'] == selected_model:
                self._current_model = model
                self._current_model['num_deg_freedom'] = 0
                self._ui.p_sb_deg_freedom.setValue(0)
                self._ui.p_sb_deg_freedom.setEnabled(not self._current_model['fixed_degrees_of_freedom'])
                self._model_widgets.append(self._get_widget(model['default_widget']))

        self._update_layouts(self._ui.p_wc_deg_freedom, self._model_widgets)
        self._potential_model_edited()

    # ----------------------------------------------------------------------
    def _get_widget(self, name, point_num=None):

        if name == 'top_bottom_potential':
             widget = TopBottomPotential(self, self.fitter.structure[1], self.settings['VOLT_MAX'])
             widget.widget_edited.connect(self._potential_model_edited)
             return widget

        elif name == 'breaking_point':
            widget = BreakingPoint(self, point_num, self.fitter.structure[1]/2, self.fitter.structure[1], self.settings['VOLT_MAX'])
            widget.widget_edited.connect(self._potential_model_edited)
            return widget

        raise RuntimeError('Cannot find widget for model: {}'.format(name))

    # ----------------------------------------------------------------------
    def _update_layouts(self, container, widgets):

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
    def _get_variable_values(self):
        new_set = []
        for widget in self._model_widgets:
            raw_ans = widget.getValues()
            for pair in raw_ans:
                new_set.append(pair)

        new_set = np.vstack(new_set)
        return new_set[new_set.argsort(axis=0)[:, 0]]

    # ----------------------------------------------------------------------
    def _potential_model_edited(self):

        new_set = self._get_variable_values()

        self.fitter.set_model(self._current_model['code'], self._current_model['num_deg_freedom'])
        self.fitter.sim_profile_shifts(new_set[:, 0], new_set[:, 1], self.g_ps_pot_plot,
                                       self.g_ps_shift_source_plot, self.g_ps_shift_sim_plot)

    # ----------------------------------------------------------------------
    def _prepare_fit_graphs(self):

        self.fit_pot_graphs_layout.clear()

    # ----------------------------------------------------------------------
    def _start_pot_fit(self):

        self._ui.p_but_start_pot_fit.setEnabled(False)
        self._ui.p_but_stop_pot_fit.setEnabled(True)
        self.fit_pot_graphs_layout.clear()
        self.fitter.solver.reset_fit()
        self.fitter.solver.set_external_graphs(self.fit_pot_graphs_layout)
        self._worker = ExcThread(self._fitter_worker, 'fitter_worker', self._local_error_queue)
        self._worker.start()

    # ----------------------------------------------------------------------
    def _fitter_worker(self):

        self.fitter.do_intensity_fit(self._get_variable_values())

    # ----------------------------------------------------------------------
    def _stop_pot_fit(self):

        self._ui.p_but_start_pot_fit.setEnabled(True)
        self._ui.p_but_stop_pot_fit.setEnabled(False)

        self.fitter.stop_fit()

    # ----------------------------------------------------------------------
    def update_cycles(self, num_cycles, ksi, best_solution):

        self._block_signals(True)
        self._ui.p_lb_fit_cycle.setText('Cycle: {}. Ksi: {:.2e}'.format(num_cycles, ksi))
        self._ui.p_srb_cycle.setMaximum(num_cycles)
        self._ui.p_srb_cycle.setValue(num_cycles)
        self._ui.p_lb_fit_status.setText('Completed cycles: {}. Best ksi : {:.2e}'.format(num_cycles, ksi))

        counter = 1
        for widget in self._model_widgets:
            if isinstance(widget, TopBottomPotential):
                widget.set_values((best_solution[1][0], best_solution[1][-1]))
            elif isinstance(widget, BreakingPoint):
                widget.set_values((best_solution[0][counter] - best_solution[0][0], best_solution[1][counter]))
                counter = counter + 1

        self._block_signals(False)

    # ----------------------------------------------------------------------
    def _display_cycle(self):
        try:
            cycle = int(self._ui.p_srb_cycle.value())
            ksi, solution = self.fitter.solver.show_results(cycle)
            self._ui.p_lb_fit_cycle.setText('Cycle: {}. Ksi: {:.2e}'.format(cycle, ksi))
            counter = 1
            for widget in self._model_widgets:
                if isinstance(widget, TopBottomPotential):
                    widget.set_values((solution[1][0], solution[1][-1]))
                elif isinstance(widget, BreakingPoint):
                    widget.set_values((solution[0][counter] - solution[0][0], solution[1][counter]))
                    counter = counter + 1

        except:
            pass

    # ----------------------------------------------------------------------
    def _prepare_fit_set(self):
        new_file = QtWidgets.QFileDialog.getSaveFileName(self, "Create file", self._working_dir, '.set')
        if new_file:
            if '.set' in new_file[0]:
                file_name = new_file[0]
            else:
                file_name = "".join(new_file)
            self.fitter.dump_fit_set(file_name, generate_data_set=True, start_values=self._get_variable_values())

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------  ------
    def refreshComboBox(self, comboBox, text):
        """Auxiliary function refreshing combo box with a given text.
        """
        idx = comboBox.findText(text)
        if idx != -1:
            comboBox.setCurrentIndex(idx)
            return True
        else:
            comboBox.setCurrentIndex(0)
            return False