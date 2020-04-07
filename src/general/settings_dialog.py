from PyQt5 import QtWidgets, QtCore
from src.widgets.settings_ui import Ui_Settings
import default_settings as default_settings
from multiprocessing import cpu_count
from src.ntr_data_fitting.subfunctions import get_precision
from src.general.auxiliary_functions import *
from src.ntr_data_fitting.compounds import COMPAUNDS
import numpy as np

# ----------------------------------------------------------------------
class Settings_Window(QtWidgets.QMainWindow):

    settings_changed = QtCore.pyqtSignal(dict, list)

    settings_objects = (
        ('METHOD_SPECTRA_FIT', 'cb_spectra_fit_method', 'cb', 'spectra'),
        ('OPTIONS', 'le_spectra_fit_parameters', 'le_txt', 'spectra'),
        ('MONITOR_SPECTRA_FIT', 'chk_spectra_fit_monitor', 'chk', 'spectra'),

        ('G', 'sb_ref_gauss', 'sb_dbl', 'potential'),
        ('L', 'sb_ref_lorentz', 'sb_dbl', 'potential'),
        ('BE_STEP', 'sb_ref_be_step', 'sb_dbl', 'potential'),
        ('SIM_SPECTRA_WIDTH', 'sb_ref_width', 'sb_dbl', 'potential'),

        ('T', 'sp_t_value', 'sb_dbl', 'potential'),
        ('KSI_TOLLERANCE', 'sb_fit_tolerance', 'sb_dbl', 'potential'),
        ('FIT_SOLVER', 'cb_solver', 'cb', 'potential'),
        ('V_MESH', 'sp_v_mesh', 'sb_int', 'potential'),
        ('D_MESH', 'sp_d_mesh', 'sb_int', 'potential'),
        ('V_STEP', 'sp_v_step', 'sb_dbl', 'potential'),
        ('D_STEP', 'sp_d_step', 'sb_dbl', 'potential'),

        ('FIELD_MAX', 'dsp_max_e', 'sb_dbl', 'potential'),
        ('SUB_LAYERS', 'sp_sub_layers', 'sb_int', 'potential'),
        ('LAMBDA', 'dsp_lambda', 'sb_dbl', 'potential'),

        ('MULTIPROCESSING', 'chk_use_mp', 'chk', 'potential'),
        ('N_SUB_JOBS', 'sb_jobs_cpu', 'sb_int', 'potential'),
        ('USE_ALL_CORES', 'chk_mp_all_cores', 'chk', 'potential'),
        ('NUM_CORES', 'sp_num_cores', 'sb_int', 'potential'),

        ('DISPLAY_EACH_X_STEP', 'sb_lm_fit_monitor_step', 'sb_int', 'potential'),
        ('MONITOR_FIT', 'chk_lm_fit_monitor', 'chk', 'potential'),
        ('METHOD', 'cb_lm_fit_method', 'cb', 'potential'),

        ('X_WAY', 'cb_x_way', 'cb_text', 'intensity'),
        ('WAVE', 'dsp_xray_value', 'sb_dbl', 'intensity'),
        ('IPOL', 'cb_xray_pol', 'cb_text', 'intensity'),
        ('LINE', 'cb_xray_line', 'cb', 'intensity'),

        ('SUBWAY', 'bg_sub_code', 'bg', 'intensity'),
        ('CODE', 'cb_subsrate_database', 'cb', 'intensity'),
        ('DF1DF2', 'cb_subsrate_database_f', 'cb_text', 'intensity'),
        ('CHEM', 'le_sub_formula', 'le_txt', 'intensity'),
        ('RHO', 'dsp_sub_rho', 'sb_dbl', 'intensity'),
        ('X0', 'le_sub_susep', 'le_txt', 'intensity'),
        ('W0', 'dsp_sub_w0', 'sb_dbl', 'intensity'),
        ('SIGMA', 'dsp_sub_sigma', 'sb_dbl', 'intensity'),
        ('TR', 'dsp_sub_tr', 'sb_dbl', 'intensity'),

        ('SCANMIN', 'dsp_sw_ang_from', 'sb_dbl', 'intensity'),
        ('SCANMAX', 'dsp_sw_ang_to', 'sb_dbl', 'intensity'),
        ('NSCAN', 'sp_sw_ang_points', 'sb_int', 'intensity'),

        ('SWMIN', 'dsp_sw_depth_from', 'sb_dbl', 'intensity'),
        ('SWMAX', 'dsp_sw_depth_to', 'sb_dbl', 'intensity'),
        ('SWPTS', 'sp_sw_depth_points', 'sb_int', 'intensity'),

        ('RSSTOL', 'dsp_rss_tol', 'sb_dbl', 'inten_fit'),
        ('THICKSTEP', 'dsp_thick_step', 'sb_dbl', 'inten_fit'),
        ('COMPSTEP', 'dsp_comp_step', 'sb_dbl', 'inten_fit'),
        ('SIGMASTEP', 'dsp_sigma_step', 'sb_dbl', 'inten_fit'),
        ('X0STEP', 'dsp_x0_step', 'sb_dbl', 'inten_fit'),
        ('W0STEP', 'dsp_w0_step', 'sb_dbl', 'inten_fit'))

    ui_codes = {'cb_x_way': ((1, 'Wavelength (A)'),
                             (2, 'Energy (eV)'),
                             (3, 'X-Ray line')),
                'cb_xray_pol': ((1, 'Sigma'),
                                (2, 'Pi')),
                'bg_sub_code': ((1, 'rb_sub_code'),
                                (2, 'rb_sub_formula'),
                                (3, 'rb_sub_suscep')),
                'cb_subsrate_database_f': ( (-1, 'Automatic DB'),
                                            (0, 'X0h data (5-25 KeV; 0.5-2.5 A)'),
                                            (1, 'Henke data (0.01-30 KeV; 0.4-1200 A)'),
                                            (2, 'Brennan data (0.03-700 KeV; 0.02-400 A)'))
                }

    _available_methods = []
    _last_settings = None

    # ----------------------------------------------------------------------
    def __init__(self):
        """
        """
        super(Settings_Window, self).__init__()

        self._ui = Ui_Settings()
        self._ui.setupUi(self)

        self._ui.sp_num_cores.setMaximum(cpu_count())

        self._connet_actions()

    # ----------------------------------------------------------------------
    def show(self, mode):

        self._ui.stw_mode.setCurrentIndex(mode)
        super(Settings_Window, self).show()

    # ----------------------------------------------------------------------
    def _connet_actions(self):

        self._ui.dialog_buttons.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply)
        self._ui.dialog_buttons.button(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(lambda: self.hide())
        self._ui.dialog_buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(self._reset_defaults)

        # ----------------------------------------------------------------------
        #   intensity fit
        # ----------------------------------------------------------------------

        self._ui.cb_x_way.currentIndexChanged.connect(self._select_xrays)
        self._ui.bg_sub_code.buttonClicked.connect(self._select_substrate)

        # ----------------------------------------------------------------------
        #   potential fit
        # ----------------------------------------------------------------------

        self._ui.chk_use_mp.stateChanged.connect(self._enable_mp)
        self._ui.chk_mp_all_cores.stateChanged.connect(
            lambda: self._ui.sp_num_cores.setEnabled(not self._ui.chk_mp_all_cores.isChecked()))

        self._ui.cb_solver.currentIndexChanged.connect(self._select_method)

        self._ui.chk_lm_fit_monitor.stateChanged.connect(
            lambda: self._ui.sb_lm_fit_monitor_step.setEnabled(not self._ui.chk_lm_fit_monitor.isChecked()))


    # ----------------------------------------------------------------------
    def _reset_defaults(self):
        settings = {}
        all_settings_keys = [arg for arg in dir(default_settings) if not arg.startswith('_')]
        for key in all_settings_keys:
            settings[key] = getattr(default_settings, key)
        self.set_options(settings)

    # ----------------------------------------------------------------------
    def set_options(self, options):

        self._last_settings = options
        for opt_key, value in options.items():
            for setting_key, ui_name, ui_type, _ in self.settings_objects:
                if opt_key == setting_key:
                    if ui_type == 'chk':
                        getattr(self._ui, ui_name).setChecked(value)
                    elif ui_type == 'sb_dbl':
                        getattr(self._ui, ui_name).setValue(float(value))
                        getattr(self._ui, ui_name).setDecimals(get_precision(value))
                    elif ui_type == 'sb_int':
                        getattr(self._ui, ui_name).setValue(int(value))
                    elif ui_type == 'le_exp':
                        getattr(self._ui, ui_name).setText('{:0.2e}'.format(value))
                    elif ui_type == 'le_int':
                        getattr(self._ui, ui_name).setText('{}'.format(value))
                    elif ui_type == 'cb':
                        refresh_combo_box(getattr(self._ui, ui_name), value)
                    elif ui_type == 'cb_text':
                        for key, opt in self.ui_codes[ui_name]:
                            if key == value:
                                refresh_combo_box(getattr(self._ui, ui_name), opt)
                    elif ui_type == 'bg':
                        for key, opt in self.ui_codes[ui_name]:
                            if key == value:
                                getattr(self._ui, opt).setChecked(True)
                    elif ui_type == 'le_txt':
                        getattr(self._ui, ui_name).setText(value)

        self._select_method()

    # ----------------------------------------------------------------------
    def apply(self):
        settings = {}
        mode_changed = []
        for setting_key, ui_name, ui_type, mode in self.settings_objects:
            if ui_type == 'chk':
                settings[setting_key] = getattr(self._ui, ui_name).isChecked()
            elif ui_type =='sb_dbl':
                settings[setting_key] = np.round(float(getattr(self._ui, ui_name).value()),
                                                 get_precision(self._last_settings[setting_key]))
            elif ui_type == 'sb_int':
                settings[setting_key] = int(getattr(self._ui, ui_name).value())
            elif ui_type == 'le_exp':
                settings[setting_key] = np.round(float(getattr(self._ui, ui_name).text()),
                                                 get_precision(self._last_settings[setting_key]))
            elif ui_type == 'le_int':
                settings[setting_key] = int(getattr(self._ui, ui_name).text())
            elif ui_type == 'cb':
                settings[setting_key] = getattr(self._ui, ui_name).currentText()
            elif ui_type == 'cb_text':
                for key, opt in self.ui_codes[ui_name]:
                    value = getattr(self._ui, ui_name).currentText()
                    if opt == value:
                        settings[setting_key] = key
            elif ui_type == 'bg':
                value = self._ui.bg_sub_code.checkedButton().objectName()
                for key, opt in self.ui_codes[ui_name]:
                    if opt == value:
                        settings[setting_key] = key
            elif ui_type == 'le_txt':
                settings[setting_key] = getattr(self._ui, ui_name).text()

            try:
                if isinstance(settings[setting_key], float):
                    if np.round(settings[setting_key], getattr(self._ui, ui_name).decimals()) != \
                            np.round(self._last_settings[setting_key], getattr(self._ui, ui_name).decimals()):
                        mode_changed.append(mode)
                elif settings[setting_key] != self._last_settings[setting_key]:
                    mode_changed.append(mode)
            except:
                pass

        self.settings_changed.emit(settings, list(set(mode_changed)))
        self.hide()

    # ----------------------------------------------------------------------
    def fill_combos(self, methods):

        for method in methods:
            self._ui.cb_solver.addItem(str(method))

        self._available_methods = methods
        for compound in COMPAUNDS:
            self._ui.cb_subsrate_database.addItem(compound)

    # ----------------------------------------------------------------------
    #   intensity fit
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def _select_xrays(self):
        direct_input = self._ui.cb_solver.currentIndex() < 2
        self._ui.dsp_xray_value.setEnabled(direct_input)
        self._ui.cb_xray_line.setEnabled(not direct_input)

    # ----------------------------------------------------------------------
    def _select_substrate(self):
        self._ui.cb_subsrate_database.setEnabled(self._ui.rb_sub_code.isChecked())
        self._ui.cb_subsrate_database_f.setEnabled(self._ui.rb_sub_code.isChecked())

        self._ui.le_sub_formula.setEnabled(self._ui.rb_sub_formula.isChecked())
        self._ui.dsp_sub_rho.setEnabled(self._ui.rb_sub_formula.isChecked())

        self._ui.le_sub_susep.setEnabled(self._ui.rb_sub_suscep.isChecked())

    # ----------------------------------------------------------------------
    #   potential fit
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def _enable_mp(self):
        state = self._ui.chk_use_mp.isChecked()
        self._ui.sb_jobs_cpu.setEnabled(state)
        self._ui.chk_mp_all_cores.setEnabled(state)
        self._ui.sp_num_cores.setEnabled(state and not self._ui.chk_mp_all_cores.isChecked())

    # ----------------------------------------------------------------------
    def _select_method(self):

        for method in self._available_methods:
            getattr(self._ui, 'f_{}'.format(method)).setVisible(False)

        selected_method = str(self._ui.cb_solver.currentText())
        getattr(self._ui, 'f_{}'.format(selected_method)).setVisible(True)