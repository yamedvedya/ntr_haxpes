from PyQt5 import QtCore
import pyqtgraph as pg

WARNING_STYLE = "background-color: #FFC4C4"

WIDGET_TOLERANCE = 1e-4

CURRENT_POTENTIAL_STYLE = {'pen': pg.mkPen(color=(255, 85, 0), width=2, style=QtCore.Qt.SolidLine)}

MAX_T_STYLE = {'pen': pg.mkPen(color=(85, 255, 0), width=2, style=QtCore.Qt.DashLine)}

T_STAT_STYLE = {'pen': pg.mkPen(color=(0, 85, 255), width=2, style=QtCore.Qt.SolidLine),
                "symbol": 'x',
                "symbolSize": 2,
                "symbolPen": pg.mkPen(color=(0, 85, 255), width=2, style=QtCore.Qt.SolidLine),
                "symbolBrush": None}

CURRENT_SOURCE_SHIFT = {'pen': None,
                        "symbol": 'x',
                        "symbolSize": 2,
                        "symbolPen": pg.mkPen(color=(255, 85, 0), width=2, style=QtCore.Qt.SolidLine),
                        "symbolBrush": None}

CURRENT_SIM_SHIFT = {'pen': pg.mkPen(color=(255, 85, 0), width=1, style=QtCore.Qt.SolidLine)}

EXPERIMENT_SPECTRA = {'pen': pg.mkPen(color=(0, 0, 0), width=3, style=QtCore.Qt.SolidLine)}

SUM_SPECTRA = {'pen': pg.mkPen(color=(255, 85, 0), width=3, style=QtCore.Qt.SolidLine)}

COMPONENT = {'width': 2, 'style': QtCore.Qt.SolidLine}

BACKGROUND = {'pen': pg.mkPen(color=(255, 85, 0), width=2, style=QtCore.Qt.DashLine)}

PLOT_COLORS = [(85, 170, 127),
               (0, 170, 0),
               (85, 170, 255),
               (0, 0, 127),
               (82, 190, 128),
               (229, 152, 102),
               (85, 0, 255),
               (116, 176, 255),
               (0, 85, 127),
               (85, 85, 127),
               (187, 143, 206),
               (244, 208, 63),
               (70, 70, 70),
               (170, 85, 255),
               (255, 85, 0),
               (170, 0, 0),
               (255, 170, 0),
               (170, 0, 0),
               (0, 85, 0)]