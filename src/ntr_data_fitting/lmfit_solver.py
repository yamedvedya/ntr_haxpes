import matplotlib.pyplot as plt
import numpy as np

from lmfit import Parameters, minimize
from src.ntr_data_fitting.base_for_external_potential_solver import Base_For_External_Potential_Solver

class LMFit_Potential_Solver(Base_For_External_Potential_Solver):

    # ----------------------------------------------------------------------
    def __init__(self, parent):

        super(LMFit_Potential_Solver, self).__init__(parent)

        self.parameters = Parameters()
        self.POSSIBLE_TO_DISPLAY_INTERMEDIATE_STEPS = True

    # ----------------------------------------------------------------------
    def reset_fit(self):
        super(LMFit_Potential_Solver, self).reset_fit()
        self.parameters = Parameters()

    # ----------------------------------------------------------------------
    def _fit_monitor(self, params, iter, resid, *args, **kws):

    # не обязательно, просто lm_fit умеет на каждой иттерации вызывать внешнюю функцию, чтобы ты контролировал процесс фита

        if self.parent.settings['MONITOR_FIT']:

            if not iter % self.parent.settings['DISPLAY_EACH_X_STEP']:
                self._display_step(params)

    # ----------------------------------------------------------------------
    def _make_params(self, start_values):

    # собирает начальные параметры для lm_fit. Дума, что для любого другого фитера нужна похожая

        for ind in range(self.parent.potential_model['num_depth_dof'] +
                                                                    self.parent.potential_model['only_voltage_dof']):
            self.parameters.add('point_{}_voltage'.format(ind), value=start_values[ind][1],
                                min=-self.parent.settings['VOLT_MAX'] + self.parent.settings['V_STEP'],
                                max=self.parent.settings['VOLT_MAX'] - self.parent.settings['V_STEP'])

        for ind in range(self.parent.potential_model['num_depth_dof']):
            self.parameters.add('point_{}_position'.format(ind), value=start_values[ind+1][0],
                                min=self.parent.data_set_for_fitting['fit_depth_points'][0] + self.parent.settings['D_STEP']*1e-9,
                                max=self.parent.data_set_for_fitting['fit_depth_points'][-1] - self.parent.settings['D_STEP']*1e-9)

    # ----------------------------------------------------------------------
    def _extract_sets(self, params):

    # обязательная функция!!! Вызывается в _errFunc
    # принимает параметры от фиттера и превращает их в volt_set и depth_set,
    # которые использутся для рассчёта спектров

        volt_set = [0 + params['point_{}_voltage'.format(ind)] for ind in range(self.parent.potential_model['num_depth_dof'] +
                                                                    self.parent.potential_model['only_voltage_dof'])]
        depth_set = np.hstack((self.parent.data_set_for_fitting['fit_depth_points'][0],
                               [0 + params['point_{}_position'.format(ind)] for ind in
                                range(self.parent.potential_model['num_depth_dof'])]))

        depth_set = np.hstack((depth_set, self.parent.data_set_for_fitting['fit_depth_points'][-1]))

        return volt_set, depth_set

    # ----------------------------------------------------------------------
    def do_fit(self, start_values):

    # обязательная функция!!! собственно вызывает фиттер

        self._make_params(start_values)

        while self.parent.fit_in_progress:

            # фактически тебе надо поменять вот эти две строчки:
            self.result = minimize(self._errFunc, self.parameters, method='leastsq', iter_cb=self._fit_monitor)
            self._display_step(self.result.params)


            self.parent.save_fit_res()

            if self.parent.STAND_ALONE_MODE and self.parent.DO_PLOT:
                plt.draw()
                plt.gcf().canvas.flush_events()

        plt.ioff()
        plt.show()