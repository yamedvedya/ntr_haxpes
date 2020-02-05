from PyQt5 import QtWidgets, QtCore
import numpy as np
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

# ----------------------------------------------------------------------
class NTR_Window(QtWidgets.QMainWindow):

    settings = {}

    # ----------------------------------------------------------------------
    def __init__(self, options):
        """
        """
        super(NTR_Window, self).__init__()

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        self._ui.tab_intensity.setEnabled(False)
        self._ui.tab_potential.setEnabled(False)

        self._set_default_settings()
        self.settings_window = Settings_Window()

        self._pot_model_grid = QtWidgets.QGridLayout(self._ui.p_wc_deg_freedom)

        self._fill_combos()
        self.fitter = NTR_fitter()
        self.fitter.set_basic_settings(self.settings)
        self.fitter.STAND_ALONE_MODE = False
        self.settings_window.fill_methods(self.fitter.METHODS)

        self._working_dir = os.getcwd()
        self._parse_options(options)
        self._connect_actions()
        self._make_default_graphics()

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
    def _connect_actions(self):

        self._ui.b_select_folder.clicked.connect(self._folder_change_clicked)
        self._ui.p_cb_cmb_model.currentIndexChanged.connect(self._potential_model_selected)
        self._ui.p_sb_deg_freedom.valueChanged.connect(self._change_degrees)
        self._ui.p_but_start_pot_fit.clicked.connect(self._start_pot_fit)
        self._ui.p_but_prepare_fit_set.clicked.connect(self._prepare_fit_set)

        self._ui.but_settings_potential.clicked.connect(self._show_settings)
        self._ui.cb_experimental_set.currentIndexChanged.connect(self._new_set_selected)

        self.settings_window.settings_changed.connect(self._settings_changed)
        self._ui.p_sb_max_potential.valueChanged.connect(self._change_v_range)

    # ----------------------------------------------------------------------
    def _block_signals(self, flag):
        self._ui.b_select_folder.blockSignals(flag)
        self._ui.p_cb_cmb_model.blockSignals(flag)
        self._ui.p_sb_deg_freedom.blockSignals(flag)
        self._ui.cb_experimental_set.blockSignals(flag)

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
    def _prepare_fit_set(self):
        new_file = QtWidgets.QFileDialog.getSaveFileName(self, "Select Directory", self._working_dir, '.set')
        if new_file:
            self.fitter.prepare_fit_set("".join(new_file))
    # ----------------------------------------------------------------------
    def _folder_change_clicked(self):
        self._block_signals(True)
        new_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", self._working_dir))
        if new_folder:
            file_list = [f for f in os.listdir(new_folder) if os.path.isfile(os.path.join(new_folder, f))
                                                   and os.path.splitext(f)[-1] == ".mat"]
            if len(file_list) > 1:
                selectItems = []
                msg = ''
                for item in file_list:
                    selectItems.append('{}'.format(item))
                    msg += item + '\n'
                item, ok = QtWidgets.QInputDialog.getItem(self, "Select desired file",
                                                      "Several data files found:\n\n" + msg + "\nselect desired:",
                                                      selectItems, 0, False)
                ind = selectItems.index(item)
                if ok:
                    file_name = os.path.splitext(file_list[ind])[0]
                else:
                    return
            else:
                file_name = os.path.splitext(file_list[0])[0]

            self._working_dir = new_folder
            self._working_file = os.path.join(new_folder, file_name + '.mat')
            self._ui.l_folder.setText(self._working_file)
            self._get_data_sets()
            self.fitter.set_new_data_file(self._working_file, self._working_set)
            self._ui.tab_intensity.setEnabled(True)

            file_list = [f for f in os.listdir(new_folder) if os.path.isfile(os.path.join(new_folder, f))
                                                   and os.path.splitext(f)[-1] == ".sw"]

            for file in file_list:
                if os.path.splitext(file)[0] == file_name:
                    self.fitter.set_new_sw_file(os.path.join(new_folder, file))
                    self._potential_model_selected()
                    self._ui.tab_potential.setEnabled(True)

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
        self.fitter.set_model(self._current_model['code'], self._current_model['num_deg_freedom'])
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
    def _potential_model_edited(self):
        new_set = []
        for widget in self._model_widgets:
            raw_ans = widget.getValues()
            for pair in raw_ans:
                new_set.append(pair)

        new_set = np.vstack(new_set)
        new_set = new_set[new_set.argsort(axis=0)[:, 0]]

        self.fitter.sim_profile_shifts(new_set[:, 0], new_set[:, 1], self.g_ps_pot_plot,
                                       self.g_ps_shift_source_plot, self.g_ps_shift_sim_plot)

    # ----------------------------------------------------------------------
    def _prepare_fit_graphs(self):

        self.fit_graphs_items = {'potential': None, 'd_points': [], 'v_points': [], 'shifts': None}

        for ind in range(self._current_model['default_degree_of_freedom'] + self._current_model['num_deg_freedom']):
            self.fit_graphs_items['v_points'].append(self.fit_pot_graphs_layout.addPlot(title="V_{}".format(ind),
                                                                                        row=0, col=ind))
        ind = 0
        for ind in range(self._current_model['num_deg_freedom']):
            self.fit_graphs_items['d_points'].append(self.fit_pot_graphs_layout.addPlot(title="D_{}".format(ind),
                                                                                        row=1, col=ind))
        self.fit_graphs_items['potential'] = self.fit_pot_graphs_layout.addPlot(title="Potential", row=1, col=ind + 1)
        self.fit_graphs_items['shifts'] = self.fit_pot_graphs_layout.addPlot(title="Shifts", row=1, col=ind + 2)

    # ----------------------------------------------------------------------
    def _start_pot_fit(self):

        self._prepare_fit_graphs()
        self.fitter.prepare_graps()
        self.fitter.set_external_graphs(self.fit_graphs_items)
        self.fitter.do_intensity_fit(1)

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
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