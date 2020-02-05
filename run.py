#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from optparse import OptionParser

from main_window import NTR_Window
from PyQt5 import QtWidgets


# ----------------------------------------------------------------------
def main():
    """Main event loop...
    """
    parser = OptionParser()
    # parser.add_option("-d", "--data_file", dest="data_file",
    #                   default="D:/DESY_Cloud/SW_data/s130_data.mat")
    # parser.add_option("-s", "--set", dest="set",
    #                   default="c3")
    # parser.add_option("-w", "--sw_file", dest="sw_file",
    #                   default="D:/DESY_Cloud/SW_data/sw_s130.txt")
    # parser.add_option("-n", "--n_points", dest="n_points",
    #                   default="1")
    # parser.add_option("-m", "--model", dest="model",
    #                   default="lin")

    (options, _) = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    mainWindow = NTR_Window(options)
    mainWindow.show()

    sys.exit(app.exec_())

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

