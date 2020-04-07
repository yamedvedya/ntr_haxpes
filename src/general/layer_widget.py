import os
import sys
import numpy as np
from src.ntr_data_fitting.subfunctions import get_precision
from src.general.auxiliary_functions import *

from PyQt5 import QtWidgets, QtCore
from src.widgets.layer_ui import Ui_layer
from src.ntr_data_fitting.compounds import COMPAUNDS

# ----------------------------------------------------------------------
class LayerWidget(QtWidgets.QWidget):

    _param_list = (('THICK', 'chk_fit_thick', 'sb_thichness'),
                   ('SIGMA', 'chk_fit_roug', 'sb_roughness'),
                   ('X0', 'chk_fit_xrs', 'dsb_x0'),
                   ('W0', 'chk_fit_dwlf', 'dsb_w0'))

    widget_edited = QtCore.pyqtSignal()
    layer_set_as_functional = QtCore.pyqtSignal(int)
    add_delete_move_layer = QtCore.pyqtSignal(int, str)

    # ----------------------------------------------------------------------
    def __init__(self, parent, layer_num):
        super(LayerWidget, self).__init__(parent)

        self._ui = Ui_layer()
        self._ui.setupUi(self)

        self._materials = [[None, None], [None, None], [None, None]]
        self.layer_num = layer_num
        self._ui.l_layer_num.setText('Layer {}'.format(layer_num))

        self._fill_combos()
        self._default_view()
        self._connect_actions()

    # ----------------------------------------------------------------------
    def _default_view(self):

        for ind in range(3):
            refresh_combo_box(getattr(self._ui, 'cb_material_{}'.format(ind)), 'None')

        self._ui.l_fraction_0.setVisible(False)
        self._ui.l_fraction_1.setVisible(False)
        self._ui.l_fraction_2.setVisible(False)

        self._ui.dsb_material_0.setVisible(False)
        self._ui.dsb_material_1.setVisible(False)
        self._ui.dsb_material_2.setVisible(False)

        self._ui.cb_material_1.setVisible(False)
        self._ui.cb_material_2.setVisible(False)

        self._ui.lin_1.setVisible(False)
        self._ui.lin_2.setVisible(False)

        self._ui.l_material_1.setVisible(False)
        self._ui.l_material_2.setVisible(False)

        self._ui.dsb_material_0.setEnabled(False)
        self._ui.dsb_material_1.setEnabled(False)
        self._ui.dsb_material_2.setEnabled(False)

        self._ui.chk_fit_thick.setChecked(False)
        self._ui.sb_thichness_min.setEnabled(False)
        self._ui.sb_thichness_max.setEnabled(False)

        self._ui.chk_fit_roug.setChecked(False)
        self._ui.sb_roughness_min.setEnabled(False)
        self._ui.sb_roughness_max.setEnabled(False)

        self._ui.chk_fit_xrs.setChecked(False)
        self._ui.dsb_x0_min.setEnabled(False)
        self._ui.dsb_x0_max.setEnabled(False)

        self._ui.chk_fit_dwlf.setChecked(False)
        self._ui.dsb_w0_min.setEnabled(False)
        self._ui.dsb_w0_max.setEnabled(False)

        self._ui.chk_fit_comp.setChecked(False)

    # ----------------------------------------------------------------------
    def _display_material(self):

        self._default_view()

        new_lim = 1 - 0.01 * (len([i for i, val in enumerate(self._materials) if val[0]]) - 1)
        for ind in range(3):
            getattr(self._ui, 'dsb_material_{}'.format(ind)).setMaximum(new_lim)

        for ind in range(3):
            if self._materials[ind][0]:
                refresh_combo_box(getattr(self._ui, 'cb_material_{}'.format(ind)), self._materials[ind][0])
                getattr(self._ui, 'dsb_material_{}'.format(ind)).setValue(self._materials[ind][1])

                getattr(self._ui, 'l_fraction_{}'.format(ind)).setVisible(True)
                getattr(self._ui, 'dsb_material_{}'.format(ind)).setVisible(True)

                if ind < 2:
                    getattr(self._ui, 'l_material_{}'.format(ind + 1)).setVisible(True)
                    getattr(self._ui, 'cb_material_{}'.format(ind + 1)).setVisible(True)
                    getattr(self._ui, 'lin_{}'.format(ind + 1)).setVisible(True)
                if ind > 0:
                    getattr(self._ui, 'dsb_material_{}'.format(ind - 1)).setEnabled(True)
    # ----------------------------------------------------------------------
    def _fill_combos(self):

        for ind in range(3):
            getattr(self._ui, 'cb_material_{}'.format(ind)).addItem('None')

        for compound in COMPAUNDS:
            for ind in range(3):
                getattr(self._ui, 'cb_material_{}'.format(ind)).addItem(compound)

    # ----------------------------------------------------------------------
    def _change_x0(self):
        self._ui.dsb_x0.setEnabled(self._ui.rb_specify_x0.isChecked())
        self._emit_edited_command()

    # ----------------------------------------------------------------------
    def _change_w0(self):
        self._ui.dsb_w0.setEnabled(self._ui.rb_specify_w0.isChecked())
        self._emit_edited_command()

    # ----------------------------------------------------------------------
    def _switch_limits(self, state, variable):
        getattr(self._ui, '{}_min'.format(variable)).setEnabled(state)
        getattr(self._ui, '{}_max'.format(variable)).setEnabled(state)

    # ----------------------------------------------------------------------
    def _connect_actions(self):

        self._ui.chk_fit_thick.stateChanged.connect(lambda value: self._switch_limits(value, 'sb_thichness'))
        self._ui.chk_fit_roug.stateChanged.connect(lambda value: self._switch_limits(value, 'sb_roughness'))
        self._ui.chk_fit_xrs.stateChanged.connect(lambda value: self._switch_limits(value, 'dsb_x0'))
        self._ui.chk_fit_dwlf.stateChanged.connect(lambda value: self._switch_limits(value, 'dsb_w0'))

        self._ui.sb_roughness.valueChanged.connect(self._emit_edited_command)
        self._ui.sb_thichness.valueChanged.connect(self._emit_edited_command)
        self._ui.dsb_x0.valueChanged.connect(self._emit_edited_command)
        self._ui.dsb_w0.valueChanged.connect(self._emit_edited_command)

        self._ui.rb_specify_w0.toggled.connect(lambda value: self._ui.dsb_w0.setEnabled(value))
        self._ui.rb_specify_x0.toggled.connect(lambda value: self._ui.dsb_x0.setEnabled(value))

        for ind in range(2):
            getattr(self._ui, 'dsb_material_{}'.format(ind)).valueChanged.connect(
                lambda ind, x=ind: self._layer_composition_change(x))

        self._ui.chk_functional_layer.stateChanged.connect(
                lambda value, x=self.layer_num: self.layer_set_as_functional.emit(x))

        for ind in range(3):
            getattr(self._ui, 'cb_material_{}'.format(ind)).currentIndexChanged.connect(
                lambda ind, x=ind: self._layer_activate(x))

        self._ui.pb_add_above.clicked.connect(lambda: self._emit_move_command('add_above'))
        self._ui.pb_move_up.clicked.connect(lambda: self._emit_move_command('move_up'))
        self._ui.pb_delete.clicked.connect(lambda: self._emit_move_command('delete'))
        self._ui.pb_move_down.clicked.connect(lambda: self._emit_move_command('move_down'))
        self._ui.pb_add_below.clicked.connect(lambda: self._emit_move_command('add_below'))

        self._ui.rb_specify_x0.toggled.connect(self._change_x0)
        self._ui.rb_specify_w0.toggled.connect(self._change_w0)

    # ----------------------------------------------------------------------
    def _emit_move_command(self, command):
        self.add_delete_move_layer.emit(self.layer_num, command)

    # ----------------------------------------------------------------------
    def _emit_set_as_functional(self):
        self.layer_set_as_functional.emit(self.layer_num)

    # ----------------------------------------------------------------------
    def _emit_edited_command(self):
        self.widget_edited.emit()

    # ----------------------------------------------------------------------
    def _layer_composition_change(self, material_num):

        self.blockSignals(True)
        new_value = getattr(self._ui, 'dsb_material_{}'.format(material_num)).value()
        rest_layers = [i for i, val in enumerate(self._materials) if val[0]]

        if rest_layers:
            if material_num in rest_layers:
                rest_layers.remove(material_num)
            else:
                material_num = rest_layers[-1]
                del rest_layers[-1]

            values = np.array([])
            if rest_layers:
                for ind in rest_layers:
                    values = np.append(values, self._materials[ind][1])

                rest = np.round(1 - new_value, 2)
                diff = np.round(rest - values.sum(), 2)
                if values[-1] + diff >= 0.01:
                    self._materials[rest_layers[-1]][1] = values[-1] + diff
                    self._materials[material_num][1] = new_value
                else:
                    self._materials[rest_layers[-1]][1] = 0.01
                    diff -= (values[-1] - 0.01)
                    if len(rest_layers) > 1:
                        self._materials[rest_layers[0]][1] = values[0] + diff
                        self._materials[material_num][1] = new_value
                    else:
                        self._materials[material_num][1] = new_value - 0.01
            else:
                self._materials[material_num][1] = 1

        for ind in range(3):
            getattr(self._ui, 'dsb_material_{}'.format(ind)).setValue(self._materials[ind][1]
                                                                      if self._materials[ind][1] else 0)

        self._emit_edited_command()
        self.blockSignals(False)

    # ----------------------------------------------------------------------
    def _layer_activate(self, layer_num):

        text = getattr(self._ui, 'cb_material_{}'.format(layer_num)).currentText()
        self._materials[layer_num] = [text,  getattr(self._ui, 'dsb_material_{}'.format(layer_num)).value()] \
            if text != 'None' else [None, None]

        self._materials.sort(key=lambda x: (x[0] is None))

        self.blockSignals(True)
        self._display_material()
        self._layer_composition_change(layer_num)
        self.blockSignals(False)

    # ----------------------------------------------------------------------
    def blockSignals(self, flag):
        """
        Args:
            flag (bool)
        """
        for ind in range(3):
            getattr(self._ui, 'cb_material_{}'.format(ind)).blockSignals(flag)
            getattr(self._ui, 'dsb_material_{}'.format(ind)).blockSignals(flag)

        self._ui.rb_specify_x0.blockSignals(flag)
        self._ui.rb_specify_w0.blockSignals(flag)

        self._ui.chk_functional_layer.blockSignals(flag)

    # ----------------------------------------------------------------------
    def get_values(self):

        layer_data = {'thick': int(self._ui.sb_thichness.value()),
                      'sigma': int(self._ui.sb_roughness.value()),
                      'w0': np.round(self._ui.dsb_w0.value(), self._ui.dsb_w0.decimals()) if self._ui.rb_specify_w0.isChecked() else None,
                      'x0': np.round(self._ui.dsb_x0.value(), self._ui.dsb_x0.decimals()) if self._ui.rb_specify_x0.isChecked() else None}

        for ind in range(3):
            layer_data['comp{}'.format(ind+1)] = self._materials[ind][0]
            layer_data['x{}'.format(ind + 1)] = \
                np.round(self._materials[ind][1], getattr(self._ui, 'dsb_material_{}'.format(ind)).decimals()) if self._materials[ind][1] is not None else None

        return layer_data

    # ----------------------------------------------------------------------
    def get_fittable_parameters(self):

        params = []
        for name, ui, lims in self._param_list:
            if getattr(self._ui, ui).isChecked():
                params.append((self.layer_num, name, getattr(self._ui, '{}_min'.format(lims)).value(),
                               getattr(self._ui, '{}_max'.format(lims)).value()))

        if self._ui.chk_fit_comp.isChecked():
            if self._materials[1][0] is not None:
                params.append((self.layer_num, 'COMP1', 0.01, self._ui.dsb_material_0.maximum()))
            if self._materials[2][0] is not None:
                params.append((self.layer_num, 'COMP2', 0.01, self._ui.dsb_material_1.maximum()))

        return params

    # ----------------------------------------------------------------------
    def set_values(self, layer_data):
        self.blockSignals(True)

        self._ui.sb_thichness.setValue(layer_data['thick'])
        self._ui.sb_roughness.setValue(layer_data['sigma'])

        if layer_data['x0'] is not None:
            decitimals = get_precision(layer_data['x0'])
            self._ui.rb_specify_x0.setChecked(True)
            self._ui.dsb_x0.setEnabled(True)
            self._ui.dsb_x0.setValue(layer_data['x0'])
            self._ui.dsb_x0.setDecimals(decitimals)
            self._ui.dsb_x0_min.setDecimals(decitimals)
            self._ui.dsb_x0_max.setDecimals(decitimals)
        else:
            self._ui.rb_auto_x0.setChecked(True)
            self._ui.dsb_x0.setEnabled(False)

        if layer_data['w0'] is not None:
            decitimals = get_precision(layer_data['w0'])
            self._ui.rb_specify_w0.setChecked(True)
            self._ui.dsb_w0.setEnabled(True)
            self._ui.dsb_w0.setValue(layer_data['w0'])
            self._ui.dsb_w0.setDecimals(decitimals)
            self._ui.dsb_w0_min.setDecimals(decitimals)
            self._ui.dsb_w0_max.setDecimals(decitimals)
        else:
            self._ui.rb_auto_w0.setChecked(True)
            self._ui.dsb_w0.setEnabled(False)

        for ind in range(3):
            self._materials[ind][0] = layer_data['comp{}'.format(ind+1)]
            self._materials[ind][1] = layer_data['x{}'.format(ind + 1)]

        self._display_material()
        self.blockSignals(False)

    # ----------------------------------------------------------------------
    def update_layer_num(self, layer_num):
        self.layer_num = layer_num
        self._ui.l_layer_num.setText('Layer {}'.format(layer_num))

    # ----------------------------------------------------------------------
    def set_functional(self, status):
        self.blockSignals(True)
        self._ui.chk_functional_layer.setChecked(status)
        self.blockSignals(False)

    # ----------------------------------------------------------------------
    def is_functional(self):
        return self._ui.chk_functional_layer.isChecked()