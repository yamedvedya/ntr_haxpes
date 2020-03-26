from PyQt5 import QtWidgets, QtCore
from src.widgets.breaking_point_ui import Ui_breaking_point
from src.widgets.top_botom_potential_ui import Ui_top_bottom_potential


# ----------------------------------------------------------------------
class TopBottomPotential(QtWidgets.QWidget):

    widget_edited = QtCore.pyqtSignal()

    # ----------------------------------------------------------------------
    def __init__(self, parent, max_depth, max_voltage):

        super(TopBottomPotential, self).__init__(parent)

        self._ui = Ui_top_bottom_potential()
        self._ui.setupUi(self)

        self.connect_actions()
        self.max_depth = max_depth*1e-10
        self.max_voltage = max_voltage - 0.01
        self._ui.sb_top_potential.maximum = max_voltage - 0.01
        self._ui.sb_top_potential.minimum = -max_voltage + 0.01
        self._ui.sb_bottom_potential.maximum = max_voltage - 0.01
        self._ui.sb_bottom_potential.minimum = -max_voltage + 0.01

    # ----------------------------------------------------------------------
    def connect_actions(self):
        self._ui.sb_top_potential.valueChanged.connect(lambda: self.sb_potential_changed('top'))
        self._ui.sb_bottom_potential.valueChanged.connect(lambda: self.sb_potential_changed('top'))

        self._ui.sl_top_potential.valueChanged.connect(lambda: self.sl_potential_changed('top'))
        self._ui.sl_bottom_potential.valueChanged.connect(lambda: self.sl_potential_changed('bottom'))

    # ----------------------------------------------------------------------
    def blockSignals(self, flag):
        """
        Args:
            flag (bool)
        """
        self._ui.sb_top_potential.blockSignals(flag)
        self._ui.sb_bottom_potential.blockSignals(flag)

        self._ui.sl_top_potential.blockSignals(flag)
        self._ui.sl_bottom_potential.blockSignals(flag)
        
    # ----------------------------------------------------------------------
    def getValues(self):
        return [[0, float(self._ui.sb_top_potential.value())], [self.max_depth, float(self._ui.sb_bottom_potential.value())]]

    # ----------------------------------------------------------------------
    def sl_potential_changed(self, mode):
        self.blockSignals(True)
        getattr(self._ui, 'sb_{}_potential'.format(mode)).setValue(
            self.max_voltage*(float(getattr(self._ui, 'sl_{}_potential'.format(mode)).value()/50)-1))
        self.widget_edited.emit()
        self.blockSignals(False)

    # ----------------------------------------------------------------------
    def sb_potential_changed(self, value):
        self.blockSignals(True)
        getattr(self._ui, 'sl_{}_potential'.format(value)).setValue(
            int(50*(1 + float(getattr(self._ui, 'sb_{}_potential'.format(value)).value())/self.max_voltage)))
        self.widget_edited.emit()
        self.blockSignals(False)

    # ----------------------------------------------------------------------
    def set_values(self, values):

        self.blockSignals(True)

        self._ui.sb_top_potential.setValue(values[0])
        self._ui.sl_top_potential.setValue(int(50*(1 + values[0]/self.max_voltage)))

        self._ui.sb_bottom_potential.setValue(values[1])
        self._ui.sl_bottom_potential.setValue(int(50*(1 + values[1]/self.max_voltage)))

        self.blockSignals(False)

    # ----------------------------------------------------------------------
    def change_layer_thickness(self, new_value):
        self.max_depth = new_value*1e-10
# ----------------------------------------------------------------------
class BreakingPoint(QtWidgets.QWidget):

    widget_edited = QtCore.pyqtSignal()

    # ----------------------------------------------------------------------
    def __init__(self, parent, point_num, point_pos, max_depth, max_voltage):
        super(BreakingPoint, self).__init__(parent)

        self._ui = Ui_breaking_point()
        self._ui.setupUi(self)
        self._ui.l_point_name.setText('Point {}'.format(point_num))

        self.max_position = (max_depth-1) * 0.1
        self.max_voltage = max_voltage - 0.01
        self._ui.sb_point_position.setValue(point_pos/10)
        self._ui.sl_point_position.setValue(1e2*point_pos/max_depth)

        self._ui.sb_point_position.maximum = (max_depth - 1) * 0.1
        self._ui.sb_point_potential.maximum = max_voltage - 0.01
        self._ui.sb_point_potential.minimum = -max_voltage + 0.01
        self.connect_actions()

    # ----------------------------------------------------------------------
    def blockSignals(self, flag):
        """
        Args:
            flag (bool)
        """
        self._ui.sb_point_position.blockSignals(flag)
        self._ui.sb_point_potential.blockSignals(flag)

        self._ui.sl_point_position.blockSignals(flag)
        self._ui.sl_point_potential.blockSignals(flag)

    # ----------------------------------------------------------------------
    def connect_actions(self):
        self._ui.sb_point_position.valueChanged.connect(lambda: self.sb_changed('position'))
        self._ui.sb_point_potential.valueChanged.connect(lambda: self.sb_changed('potential'))

        self._ui.sl_point_position.valueChanged.connect(lambda: self.sl_changed('position'))
        self._ui.sl_point_potential.valueChanged.connect(lambda: self.sl_changed('potential'))
        
    # ----------------------------------------------------------------------
    def getValues(self):
        return [[float(self._ui.sb_point_position.value()*1e-9), float(self._ui.sb_point_potential.value())]]

    # ----------------------------------------------------------------------
    def sl_changed(self, mode):
        self.blockSignals(True)
        if mode == 'position':
            self._ui.sb_point_position.setValue(max([0.01, self.max_position*(float(self._ui.sl_point_position.value()/100))]))
        else:
            self._ui.sb_point_potential.setValue(self.max_voltage * (float(self._ui.sl_point_potential.value() / 50) - 1))
        self.widget_edited.emit()
        self.blockSignals(False)

    # ----------------------------------------------------------------------
    def sb_changed(self, mode):
        self.blockSignals(True)
        if mode == 'position':
            self._ui.sl_point_position.setValue(int(100 * float(self._ui.sb_point_position.value()) /
                                                           self.max_position))
        else:
            self._ui.sl_point_potential.setValue(int(50 * (1 + float(self._ui.sb_point_potential.value()) /
                                                           self.max_voltage)))
        self.widget_edited.emit()
        self.blockSignals(False)

    # ----------------------------------------------------------------------
    def set_values(self, values):

        self.blockSignals(True)

        self._ui.sb_point_position.setValue(values[0]/10)
        self._ui.sl_point_position.setValue(int(10 * values[0] / self.max_position))

        self._ui.sb_point_potential.setValue(values[1])
        self._ui.sl_point_potential.setValue(int(50 * (1 + values[1] / self.max_voltage)))

        self.blockSignals(False)

    # ----------------------------------------------------------------------
    def change_layer_thickness(self, new_value):
        self.max_depth = new_value*1e-10 - 0.01