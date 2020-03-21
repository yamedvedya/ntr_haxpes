from PyQt5 import QtWidgets, QtCore
from src.widgets.import_dialog_ui import Ui_ImportDialog
from src.general.auxiliary_functions import *
import numpy as np

# ----------------------------------------------------------------------
class Import_Dialog(QtWidgets.QMainWindow):

    load_data = QtCore.pyqtSignal(np.ndarray, str)

    _options = {'Skip': None, 'Angle': 0, 'Intensity': 1, 'BE': 2, 'Int. error': 3, 'BE error': 4}

    _data = []
    _rows = None
    _file_name = None

    # ----------------------------------------------------------------------
    def __init__(self, data, file_name):
        """
        """
        super(Import_Dialog, self).__init__()

        self._ui = Ui_ImportDialog()
        self._ui.setupUi(self)

        self._fill_combos()
        self._connect_actions()
        self._display_data(data)
        self._file_name = file_name

    # ----------------------------------------------------------------------
    def _fill_combos(self):
        for ind in range(5):
            for key, _ in self._options.items():
                getattr(self._ui, 'cb_colum_{}'.format(ind)).addItem(key)

    # ----------------------------------------------------------------------
    def _connect_actions(self):

        self._ui.but_cancel.clicked.connect(lambda: self.close())
        self._ui.but_ok.clicked.connect(self._ok)

    # ----------------------------------------------------------------------
    def _display_data(self, data):
        self._rows = len(data)
        self._data = np.zeros((self._rows, 5))

        parsed_line = []
        for line in range(self._rows):
            parsed_line = list(map(float, data[line].replace(',', '.').split()))
            self._data[line, 0:len(parsed_line)] = parsed_line

        num_data_colums = len(parsed_line)

        self._ui.tab_data.setRowCount(self._rows)

        keys = [key for key, _ in self._options.items()][1:]
        for colum in range(num_data_colums):
            refresh_combo_box(getattr(self._ui, 'cb_colum_{}'.format(colum)), keys[colum])

        for colum in range(num_data_colums, 5):
            getattr(self._ui, 'cb_colum_{}'.format(colum)).setEnabled(False)

        for row in range(self._rows):
            for colum in range(5):
                self._ui.tab_data.setItem(row, colum, QtWidgets.QTableWidgetItem("{}".format(self._data[row, colum])
                                                                                 if colum < num_data_colums else ''))

    # ----------------------------------------------------------------------
    def _ok(self):

        all_values = [getattr(self._ui, 'cb_colum_{}'.format(ind)).currentText() for ind in range(5)]
        selected_values = [val for val in all_values if val != 'Skip']

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText("Error")
        msg.setWindowTitle("Error")
        if selected_values:
            if len(selected_values) == len(set(selected_values)):
                out_data = np.zeros((self._rows, 5))
                for colum in range(5):
                    if self._options[all_values[colum]] is not None:
                        out_data[:, self._options[all_values[colum]]] = self._data[:, colum]

                self.load_data.emit(out_data, self._file_name)
                self.close()
            else:
                msg.setInformativeText('Non unique colums selected')
                msg.exec_()
        else:
            msg.setInformativeText('Select at least one colum')
            msg.exec_()