from PyQt5 import QtWidgets, QtCore
from src.widgets.background_ui import Ui_background
from src.widgets.single_line_ui import Ui_single_line
from src.widgets.doublet_line_ui import Ui_doublet

from src.spectra_fit.spectra_models import bck_models
from src.general.auxiliary_functions import *

from look_and_feel import WARNING_STYLE, WIDGET_TOLERANCE

import numpy as np

# ----------------------------------------------------------------------
class Background(QtWidgets.QWidget):

    widget_edited = QtCore.pyqtSignal()
    delete_component = QtCore.pyqtSignal(float)

    _param_list = (('type', 'cb_back_type', 'cb'),
                   ('value', 'dsp_value', 'sb_dbl'),
                   ('special_value', 'cmb_special_value', 'cb'),
                   ('fitable', 'chk_fit', 'chk'),
                   ('max', 'dsb_max', 'sb_dbl'),
                   ('min', 'dsb_min', 'sb_dbl'))

    # ----------------------------------------------------------------------
    def __init__(self, parent, id):
        super(Background, self).__init__()

        self._id = id
        self._parent = parent

        self._ui = Ui_background()
        self._ui.setupUi(self)

        self._fill_combo()

        self._connect_actions()

        self.STD_STYLE = self._ui.dsp_value.styleSheet()

    # ----------------------------------------------------------------------
    def _fill_combo(self):
        for model in bck_models:
            self._ui.cb_back_type.addItem(str(model))

    # ----------------------------------------------------------------------
    def _connect_actions(self):

        self._ui.pb_delete.clicked.connect(lambda: self.delete_component.emit(self._id))
        self._ui.cb_back_type.currentIndexChanged.connect(lambda: self._new_type())
        self._ui.chk_special_value.stateChanged.connect(lambda value: self._special_value(value))
        self._ui.chk_fit.stateChanged.connect(lambda value: self._make_fitable(value))

        self._ui.dsp_value.editingFinished.connect(self._check_value)

        self._ui.dsb_max.editingFinished.connect(self._check_limits)
        self._ui.dsb_min.editingFinished.connect(self._check_limits)

    # ----------------------------------------------------------------------
    def _new_type(self):

        self._block_signals(True)
        self._ui.dsp_value.setValue(0)
        self._ui.chk_special_value.setChecked(False)
        self._ui.cmb_special_value.setCurrentIndex(0)
        self._ui.cmb_special_value.setEnabled(False)
        if self._ui.cb_back_type.currentText() != 'constant':
            self._ui.chk_special_value.setEnabled(False)
        else:
            self._ui.chk_special_value.setEnabled(True)
        self._ui.chk_fit.setChecked(False)
        self._ui.dsb_min.setValue(0)
        self._ui.dsb_max.setValue(0)
        self._ui.dsb_min.setEnabled(False)
        self._ui.dsb_max.setEnabled(False)
        self._block_signals(False)

    # ----------------------------------------------------------------------
    def _check_value(self):
        v_min = self._ui.dsb_min.value()
        v_max = self._ui.dsb_max.value()
        value = self._ui.dsp_value.value()

        if np.isclose(value, v_min, rtol=WIDGET_TOLERANCE):
            style, msg = WARNING_STYLE, "Value at min"
        elif np.isclose(value, v_max, rtol=WIDGET_TOLERANCE):
            style, msg = WARNING_STYLE, "Value at max"
        else:
            style, msg = self.STD_STYLE, ""

        colorizeWidget(self._ui.dsp_value, style, "")

        self.widget_edited.emit()

    # ----------------------------------------------------------------------
    def _check_limits(self):
        v_min = self._ui.dsb_min.value()
        v_max = self._ui.dsb_max.value()
        value = self._ui.dsp_value.value()

        self._ui.dsp_value.setValue(min(max(value, v_min), v_max))
        self._check_value()

    # ----------------------------------------------------------------------
    def _special_value(self, state):
        self._ui.cmb_special_value.setEnabled(state)
        self._ui.dsp_value.setEnabled(not state)
        self.widget_edited.emit()

    # ----------------------------------------------------------------------
    def _make_fitable(self, state):

        self._ui.dsb_min.setEnabled(state)
        self._ui.dsb_max.setEnabled(state)
        self.widget_edited.emit()

    # ----------------------------------------------------------------------
    def _block_signals(self, flag):

        self._ui.chk_special_value.blockSignals(flag)
        for _, uis, _ in self._param_list:
            getattr(self._ui, uis).blockSignals(flag)

    # ----------------------------------------------------------------------
    def get_values(self):
        #'constant': {'value': 'min', 'fitable': False, 'min': 30000, 'max': 35000}
        value = None
        if self._ui.chk_special_value.isChecked():
            selection = str(self._ui.cmb_special_value.currentText())
            if selection == 'min Intensity':
                value = 'min'
            elif selection == 'max Intensity':
                value = 'max'
            elif selection == 'Intensity(minBE)':
                value = 'first'
            elif selection == 'Intensity(maxBE)':
                value = 'last'
        else:
            value = self._ui.dsp_value.value()
        return str(self._ui.cb_back_type.currentText()), {'value': value, 'fitable': self._ui.chk_fit.isChecked(),
                                                           'min': self._ui.dsb_min.value(),
                                                           'max': self._ui.dsb_max.value()}

    # ----------------------------------------------------------------------
    def set_values(self, type, values):

        self._block_signals(True)
        refresh_combo_box(self._ui.cb_back_type, type)

        if values['value'] in ['min', 'max', 'first', 'last']:
            special_value = True
            if values['value'] == 'min':
                refresh_combo_box(self._ui.cb_back_type, 'min Intensity')
            elif values['value'] == 'max':
                refresh_combo_box(self._ui.cb_back_type, 'max Intensity')
            elif values['value'] == 'first':
                refresh_combo_box(self._ui.cb_back_type, 'Intensity(minBE)')
            elif values['value'] == 'last':
                refresh_combo_box(self._ui.cb_back_type, 'Intensity(maxBE)')
        else:
            style, msg = self.STD_STYLE, ""
            if values['fitable']:
                if np.isclose(values['value'], values['min'], rtol=WIDGET_TOLERANCE):
                    style, msg = WARNING_STYLE, "Value at min"
                elif np.isclose(values['value'], values['max'], rtol=WIDGET_TOLERANCE):
                    style, msg = WARNING_STYLE, "Value at max"
            colorizeWidget(self._ui.dsp_value, style, msg)
            special_value = False
            self._ui.dsp_value.setValue(values['value'])
        self._ui.chk_special_value.setChecked(special_value)
        self._ui.cmb_special_value.setEnabled(special_value)
        self._ui.dsp_value.setEnabled(not special_value)

        self._ui.chk_fit.setChecked(values['fitable'])
        self._ui.dsb_min.setValue(values['min'])
        self._ui.dsb_min.setEnabled(values['fitable'])
        self._ui.dsb_max.setValue(values['max'])
        self._ui.dsb_max.setEnabled(values['fitable'])

        self._block_signals(False)

# ----------------------------------------------------------------------
class Line_Widget(QtWidgets.QWidget):

    widget_edited = QtCore.pyqtSignal()
    delete_component = QtCore.pyqtSignal(float)
    name_changed = QtCore.pyqtSignal(float, str)
    set_functional_signal = QtCore.pyqtSignal(float)

    # ----------------------------------------------------------------------
    def __init__(self, parent, id):

        super(Line_Widget, self).__init__()

        self._id = id
        self._parent = parent

    # ----------------------------------------------------------------------
    def _finish_init(self):

        self._connect_actions()
        self.STD_STYLE = getattr(self._ui, 'dsp_{}'.format(self._param_list[0][1])).styleSheet()

    # ----------------------------------------------------------------------
    def _block_signals(self, state):

        self._ui.le_name.blockSignals(state)
        self._ui.chk_functional_peak.blockSignals(state)

        for _, ui, _ in self._param_list:
            getattr(self._ui, 'dsp_{}'.format(ui)).blockSignals(state)
            getattr(self._ui, 'chk_{}_dep'.format(ui)).blockSignals(state)
            getattr(self._ui, 'cmb_{}_dep'.format(ui)).blockSignals(state)
            getattr(self._ui, 'chk_{}_fit'.format(ui)).blockSignals(state)
            getattr(self._ui, 'rb_{}_abolut'.format(ui)).blockSignals(state)
            getattr(self._ui, 'rb_{}_relative'.format(ui)).blockSignals(state)
            getattr(self._ui, 'dsb_{}_min'.format(ui)).blockSignals(state)
            getattr(self._ui, 'dsb_{}_max'.format(ui)).blockSignals(state)

    # ----------------------------------------------------------------------
    def _reset_widget(self):

        for _, ui, _ in self._param_list:
            getattr(self._ui, 'dsp_{}'.format(ui)).setValue(0)
            getattr(self._ui, 'chk_{}_dep'.format(ui)).setChecked(False)
            getattr(self._ui, 'cmb_{}_dep'.format(ui)).setEnabled(False)
            getattr(self._ui, 'cmb_{}_dep'.format(ui)).setCurrentIndex(0)
            getattr(self._ui, 'chk_{}_fit'.format(ui)).setChecked(False)
            getattr(self._ui, 'rb_{}_abolut'.format(ui)).setChecked(True)
            getattr(self._ui, 'dsb_{}_min'.format(ui)).setValue(0)
            getattr(self._ui, 'dsb_{}_max'.format(ui)).setValue(0)
            self._toggle_fittable(False, ui)

    # ----------------------------------------------------------------------
    def _connect_actions(self):

        self._ui.pb_delete.clicked.connect(lambda: self.delete_component.emit(self._id))
        self._ui.le_name.editingFinished.connect(lambda x=str(self._ui.le_name.text()):
                                                 self.name_changed.emit(self._id, x))
        self._ui.chk_functional_peak.stateChanged.connect(lambda: self.set_functional_signal.emit(self._id))
        for _, ui, _ in self._param_list:
            getattr(self._ui, 'dsp_{}'.format(ui)).editingFinished.connect(lambda x=ui: self._check_value(x))

            getattr(self._ui, 'chk_{}_dep'.format(ui)).stateChanged.connect(lambda value, x=ui:
                                                                            self._toggle_dependence(value, x))
            getattr(self._ui, 'cmb_{}_dep'.format(ui)).currentIndexChanged.connect(lambda value, x=ui:
                                                                                   self._recalculate_value(value, x))

            getattr(self._ui, 'chk_{}_fit'.format(ui)).stateChanged.connect(lambda state, x=ui:
                                                                            self._toggle_fittable(bool(state), x))
            getattr(self._ui, 'dsb_{}_min'.format(ui)).editingFinished.connect(lambda x=ui: self._check_limits(x))
            getattr(self._ui, 'dsb_{}_max'.format(ui)).editingFinished.connect(lambda x=ui: self._check_limits(x))
            getattr(self._ui, 'bg_{}'.format(ui)).buttonClicked.connect(lambda type, x=ui: self._recalculate_limits(type, x))

    # ----------------------------------------------------------------------
    def _toggle_dependence(self, state, uis):

        getattr(self._ui, 'cmb_{}_dep'.format(uis)).setEnabled(state)
        self.widget_edited.emit()

    # ----------------------------------------------------------------------
    def _toggle_fittable(self, state, uis):

        getattr(self._ui, 'rb_{}_abolut'.format(uis)).setEnabled(state)
        getattr(self._ui, 'rb_{}_relative'.format(uis)).setEnabled(state)
        getattr(self._ui, 'dsb_{}_min'.format(uis)).setEnabled(state)
        getattr(self._ui, 'dsb_{}_max'.format(uis)).setEnabled(state)

    # ----------------------------------------------------------------------
    def _check_value(self, ui):
        v_min = getattr(self._ui, 'dsb_{}_min'.format(ui)).value()
        v_max = getattr(self._ui, 'dsb_{}_max'.format(ui)).value()
        value = getattr(self._ui, 'dsp_{}'.format(ui)).value()

        if np.isclose(value, v_min, rtol=WIDGET_TOLERANCE):
            style, msg = WARNING_STYLE, "Value at min"
        elif np.isclose(value, v_max, rtol=WIDGET_TOLERANCE):
            style, msg = WARNING_STYLE, "Value at max"
        else:
            style, msg = self.STD_STYLE, ""

        colorizeWidget(getattr(self._ui, 'dsp_{}'.format(ui)), style, "")

        self.widget_edited.emit()

    # ----------------------------------------------------------------------
    def _check_limits(self, ui):

        v_min = getattr(self._ui, 'dsb_{}_min'.format(ui)).value()
        v_max = getattr(self._ui, 'dsb_{}_max'.format(ui)).value()
        value = getattr(self._ui, 'dsp_{}'.format(ui)).value()

        if getattr(self._ui, 'rb_{}_abolut'.format(ui)).isChecked():
            getattr(self._ui, 'dsp_{}'.format(ui)).setValue(min(max(value, v_min), v_max))

        self._check_value(ui)

    # ----------------------------------------------------------------------
    def _recalculate_limits(self, type, ui):

        current_max = getattr(self._ui, 'dsb_{}_max'.format(ui)).value()
        current_min = getattr(self._ui, 'dsb_{}_min'.format(ui)).value()
        current_value = getattr(self._ui, 'dsp_{}'.format(ui)).value()

        self._block_signals(True)
        if type.objectName() == 'rb_{}_relative'.format(ui):
            getattr(self._ui, 'dsb_{}_max'.format(ui)).setValue(current_max/current_value)
            getattr(self._ui, 'dsb_{}_min'.format(ui)).setValue(current_min/current_value)
        else:
            getattr(self._ui, 'dsb_{}_max'.format(ui)).setValue(current_max*current_value)
            getattr(self._ui, 'dsb_{}_min'.format(ui)).setValue(current_min/current_value)
        self._block_signals(False)

    # ----------------------------------------------------------------------
    def _recalculate_value(self, mode, ui):
        pass

    # ----------------------------------------------------------------------
    def get_values(self):
        #{'name': 'Hf4f', 'peakType': 'voigthDoublet', 'params':
        #    {'areaMain': {'value': 400, 'fitable': True, 'model': 'Flex',
        #                  'limModel': 'absolute', 'min': 10, 'max': 40000},

        params = {}
        list_of_relative_params = {}

        for name, ui, dependence_type in self._param_list:
            value = getattr(self._ui, 'dsp_{}'.format(ui)).value()
            params[name] = {'value': value}

            if getattr(self._ui, 'chk_{}_fit'.format(ui)).isChecked():
                params[name]['fitable'] = True
                min_v = getattr(self._ui, 'dsb_{}_min'.format(ui)).value()
                max_v = getattr(self._ui, 'dsb_{}_max'.format(ui)).value()

                if getattr(self._ui, 'rb_{}_relative'.format(ui)).isChecked():
                    params[name]['limModel'] = 'relative'
                    list_of_relative_params[name] = (value - min_v, value + max_v)
                else:
                    params[name]['limModel'] = 'absolute'

                params[name]['min'] = min_v
                params[name]['max'] = max_v


            else:
                params[name]['fitable'] = False

            if getattr(self._ui, 'chk_{}_dep'.format(ui)).isChecked():
                params[name]['model'] = 'Dependent'
                params[name]['baseValue'] = getattr(self._ui, 'cmb_{}_dep'.format(ui)).currentText()
                params[name]['linkType'] = dependence_type
            else:
                params[name]['model'] = 'Flex'

        return {'name': str(self._ui.le_name.text()), 'peakType': self._type,
                'params': params}, list_of_relative_params

    # ----------------------------------------------------------------------
    def get_widget_type(self):

        return self._id, self._type

    # ----------------------------------------------------------------------
    def set_values(self, name, values):
        self._block_signals(True)
        self._ui.le_name.setText(name)
        for key, item in values.items():
            for name, ui, _ in self._param_list:
                if name == key:
                    getattr(self._ui, 'dsp_{}'.format(ui)).setValue(item['value'])
                    style, msg = self.STD_STYLE, ""

                    if item['fitable']:
                        getattr(self._ui, 'chk_{}_fit'.format(ui)).setChecked(True)
                        self._toggle_fittable(True, ui)
                        if item['limModel'] == 'relative':
                            getattr(self._ui, 'rb_{}_relative'.format(ui)).setChecked(True)
                            if np.isclose(1, item['min'], rtol=WIDGET_TOLERANCE):
                                style, msg = WARNING_STYLE, "Value at min"
                            elif np.isclose(1, item['max'], rtol=WIDGET_TOLERANCE):
                                style, msg = WARNING_STYLE, "Value at max"
                        else:
                            getattr(self._ui, 'rb_{}_abolut'.format(ui)).setChecked(True)
                            if np.isclose(item['value'], item['min'], rtol=WIDGET_TOLERANCE):
                                style, msg = WARNING_STYLE, "Value at min"
                            elif np.isclose(item['value'], item['max'], rtol=WIDGET_TOLERANCE):
                                style, msg = WARNING_STYLE, "Value at max"
                        getattr(self._ui, 'dsb_{}_min'.format(ui)).setValue(item['min'])
                        getattr(self._ui, 'dsb_{}_max'.format(ui)).setValue(item['max'])

                    else:
                        self._toggle_fittable(False, ui)

                    if item['model'] == 'Dependent':
                        getattr(self._ui, 'chk_{}_dep'.format(ui)).setChecked(True)
                        getattr(self._ui, 'cmb_{}_dep'.format(ui)).setEnabled(True)
                        refresh_combo_box(getattr(self._ui, 'cmb_{}_dep'.format(ui)), item['baseValue'])
                    else:
                        getattr(self._ui, 'chk_{}_dep'.format(ui)).setChecked(False)
                        getattr(self._ui, 'cmb_{}_dep'.format(ui)).setEnabled(False)

                    colorizeWidget(getattr(self._ui, 'dsp_{}'.format(ui)), style, "")

        self._block_signals(False)

    # ----------------------------------------------------------------------
    def set_functional(self, status):
        self.blockSignals(True)
        self._ui.chk_functional_peak.setChecked(status)
        self.blockSignals(False)

    # ----------------------------------------------------------------------
    def is_functional(self):
        return self._ui.chk_functional_peak.isChecked()

    # ----------------------------------------------------------------------
    def get_name(self):
        return self._ui.le_name.text()

# ----------------------------------------------------------------------
class Single_line(Line_Widget):
    _param_list = (('area', 'area', 'multiplication'),
                   ('center', 'pos', 'additive'),
                   ('gauss', 'gaus', 'multiplication'),
                   ('lorenz', 'lor', 'multiplication'))

    _type = 'voigth'

    # ----------------------------------------------------------------------
    def __init__(self, parent, id):
        super(Single_line, self).__init__(parent, id)

        self._ui = Ui_single_line()
        self._ui.setupUi(self)

        self._finish_init()

# ----------------------------------------------------------------------
class Doublet(Line_Widget):
    _param_list = (('areaMain', 'area_main', 'multiplication'),
                   ('areaRation', 'area_said', 'multiplication'),
                   ('centerMain', 'pos_main', 'additive'),
                   ('separation', 'pos_said', 'additive'),
                   ('gaussMain', 'gauss_main', 'multiplication'),
                   ('gaussSecond', 'gauss_said', 'multiplication'),
                   ('lorenzMain', 'lor_main', 'multiplication'),
                   ('lorenzSecond', 'lor_said', 'multiplication'))

    _type = 'voigth_doublet'

    # ----------------------------------------------------------------------
    def __init__(self, parent, id):
        super(Doublet, self).__init__(parent, id)

        self._ui = Ui_doublet()
        self._ui.setupUi(self)

        self._finish_init()