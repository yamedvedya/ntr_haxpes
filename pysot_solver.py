import matplotlib.pyplot as plt
import numpy as np

from base_for_external_solver import Base_For_External_Solver

class PySOT_Solver(Base_For_External_Solver):

    # ----------------------------------------------------------------------
    def __init__(self, parent):

        super(PySOT_Solver, self).__init__(parent)

    # ----------------------------------------------------------------------
    def reset_fit(self):
        super(PySOT_Solver, self).reset_fit()

    # ----------------------------------------------------------------------
    def _fit_monitor(self, params, iter, resid, *args, **kws):

        if self.parent.settings['MONITOR_FIT']:

            if not iter % self.parent.settings['DISPLAY_EACH_X_STEP']:
                self._display_step(params)

    # ----------------------------------------------------------------------
    def _make_params(self, start_values):

        pass

    # ----------------------------------------------------------------------
    def _extract_sets(self, params):

        volt_set = [0 + params['point_{}_voltage'.format(ind)] for ind in range(self.parent.num_depth_points)]
        depth_set = np.hstack((self.parent.structure[0], [0 + params['point_{}_position'.format(ind)] for ind in range(self.parent.num_depth_points-2)]))
        depth_set = np.hstack((depth_set, self.parent.structure[0] + self.parent.structure[1]))

        return volt_set, depth_set

    # ----------------------------------------------------------------------
    def do_fit(self, start_values):

        self._make_params(start_values)

        while self.parent.fit_in_progress:
            self.result = minimize(self._errFunc, self.parameters, iter_cb=self._fit_monitor)

            self._display_step(self.result.params)
            self.parent.save_fit_res()

            if self.parent.STAND_ALONE_MODE and self.parent.DO_PLOT:
                plt.draw()
                plt.gcf().canvas.flush_events()

        plt.ioff()
        plt.show()