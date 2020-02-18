from PyQt5 import QtWidgets, QtCore
from widgets.settings_ui import Ui_Settings
import default_settings as default_settings
from multiprocessing import cpu_count
# ----------------------------------------------------------------------
class Settings_Window(QtWidgets.QMainWindow):

    settings_changed = QtCore.pyqtSignal(dict)

    settings_objects = (
        ('G', 'sb_ref_gauss', 'sb_dbl'),
        ('L', 'sb_ref_lorentz', 'sb_dbl'),
        ('BE_STEP', 'sb_ref_be_step', 'sb_dbl'),
        ('SIM_SPECTRA_WIDTH', 'sb_ref_width', 'sb_dbl'),

        ('T', 'sp_t_value', 'sb_dbl'),
        ('KSI_TOLLERANCE', 'sb_fit_tolerance', 'sb_dbl'),
        ('FIT_SOLVER', 'cb_solver', 'cb'),
        ('V_MESH', 'sp_v_mesh', 'sb_int'),
        ('D_MESH', 'sp_d_mesh', 'sb_int'),
        ('V_STEP', 'sp_v_step', 'sb_dbl'),
        ('D_STEP', 'sp_d_step', 'sb_dbl'),

        ('FIELD_MAX', 'le_max_e', 'le_exp'),
        ('SUB_LAYERS', 'sp_sub_layers', 'sb_int'),
        ('LAMBDA', 'le_lambda', 'le_exp'),

        ('MULTIPROCESSING', 'chk_use_mp', 'chk'),
        ('N_SUB_JOBS', 'sb_jobs_cpu', 'sb_int'),
        ('USE_ALL_CORES', 'chk_mp_all_cores', 'chk'),
        ('NUM_CORES', 'sp_num_cores', 'sb_int'),

        ('DISPLAY_EACH_X_STEP', 'sb_lm_fit_monitor_step', 'sb_int'),
        ('MONITOR_FIT', 'chk_lm_fit_monitor', 'chk'),
        ('METHOD', 'cb_lm_fit_method', 'cb'))

    _available_methods = []

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
    def _connet_actions(self):

        self._ui.dialog_buttons.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply)
        self._ui.dialog_buttons.button(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(lambda: self.hide())
        self._ui.dialog_buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(self._reset_defaults)

        self._ui.chk_use_mp.stateChanged.connect(self._enable_mp)
        self._ui.chk_mp_all_cores.stateChanged.connect(
            lambda: self._ui.sp_num_cores.setEnabled(not self._ui.chk_mp_all_cores.isChecked()))

        self._ui.cb_solver.currentIndexChanged.connect(self._select_method)

        self._ui.chk_lm_fit_monitor.stateChanged.connect(
            lambda: self._ui.sb_lm_fit_monitor_step.setEnabled(not self._ui.chk_lm_fit_monitor.isChecked()))

    # ----------------------------------------------------------------------
    def _enable_mp(self):
        state = self._ui.chk_use_mp.isChecked()
        self._ui.sb_jobs_cpu.setEnabled(state)
        self._ui.chk_mp_all_cores.setEnabled(state)
        self._ui.sp_num_cores.setEnabled(state and not self._ui.chk_mp_all_cores.isChecked())

    # ----------------------------------------------------------------------
    def set_options(self, options):

        for opt_key, value in options.items():
            for setting_key, ui_name, ui_type in self.settings_objects:
                if opt_key == setting_key:
                    if ui_type == 'chk':
                        getattr(self._ui, ui_name).setChecked(value)
                    elif ui_type == 'sb_dbl':
                        getattr(self._ui, ui_name).setValue(float(value))
                    elif ui_type == 'sb_int':
                        getattr(self._ui, ui_name).setValue(int(value))
                    elif ui_type == 'le_exp':
                        getattr(self._ui, ui_name).setText('{:0.2e}'.format(value))
                    elif ui_type == 'cb':
                        self.refreshComboBox(getattr(self._ui, ui_name), value)

        self._select_method()

    # ----------------------------------------------------------------------
    def apply(self):
        settings = {}
        for setting_key, ui_name, ui_type in self.settings_objects:
            if ui_type == 'chk':
                settings[setting_key] = getattr(self._ui, ui_name).isChecked()
            elif ui_type =='sb_dbl':
                settings[setting_key] = float(getattr(self._ui, ui_name).value())
            elif ui_type == 'sb_int':
                settings[setting_key] = int(getattr(self._ui, ui_name).value())
            elif ui_type == 'le_exp':
                settings[setting_key] = float(getattr(self._ui, ui_name).text())
            elif ui_type == 'cb':
                settings[setting_key] = getattr(self._ui, ui_name).currentText()

        self.settings_changed.emit(settings)
        self.hide()

    # ----------------------------------------------------------------------
    def fill_methods(self, methods):

        for method in methods:
            self._ui.cb_solver.addItem(str(method))

        self._available_methods = methods

    # ----------------------------------------------------------------------
    def _select_method(self):

        for method in self._available_methods:
            getattr(self._ui, 'f_{}'.format(method)).setVisible(False)

        selected_method = str(self._ui.cb_solver.currentText())
        getattr(self._ui, 'f_{}'.format(selected_method)).setVisible(True)

    # ----------------------------------------------------------------------
    def _reset_defaults(self):
        settings = {}
        all_settings_keys = [arg for arg in dir(default_settings) if not arg.startswith('_')]
        for key in all_settings_keys:
            settings[key] = getattr(default_settings, key)
        self.set_options(settings)


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