#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from optparse import OptionParser

from src.main_window import NTR_Window
from PyQt5 import QtWidgets


# ----------------------------------------------------------------------
def main():
    """Main event loop...
    """
    parser = OptionParser()
    (options, _) = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    mainWindow = NTR_Window(options)
    mainWindow.show()

    sys.exit(app.exec_())

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

