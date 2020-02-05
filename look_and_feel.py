from PyQt5 import QtCore
import pyqtgraph as pg

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