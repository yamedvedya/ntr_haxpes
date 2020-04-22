import numpy as np
from scipy.special import wofz

bck_models = ('constant', 'linear', 'square', 'shirley')
line_models = ('voigth', 'voigth_doublet', 'doniach_sunjic', 'doublet_doniach_sunjic')

class models():

    doniach_sunjic_coeffs = [[3.25335746e+02, -2.57572066e+01, -2.23619810e-01,  7.54041108e-01, -1.16215539e-01,  6.07640602e-03],
                             [9.24088147e+02, -4.69743112e+01,  7.05091173e-02, -1.24358460e-01, 5.97691848e-02, -4.79809704e-03],
                             [8.16058435e+03,  3.97239095e+01, -1.65502859e-01, -5.27393306e-01, 7.51854789e-02, -3.35246447e-03],
                             [-1.83782188e+04, -2.33281472e+02,  1.67126031e+00, -1.40024326e-01, 5.93066372e-03,  6.23108666e-04],
                             [3.79082150e+04,  3.22712317e+02, -2.25628469e+00, -6.25437750e-02, -2.42131344e-02,  1.99688026e-03],
                             [-2.89248614e+04, -5.58933030e+01,  9.00885735e-01,  1.00236307e-01, -4.61647604e-04, -5.45420412e-04]]

    def __init__(self):
        pass

    # ----------------------------------------------------------------------
    def getModel(self, model, energy, intensity, params):
        return getattr(self, model)(energy, intensity, params)

    # ----------------------------------------------------------------------
    def constant(self, energy, intensity, params):
        # calculates constant background for simulated spectrum

        return np.ones_like(energy) * float(params['value'])

    # ----------------------------------------------------------------------
    def linear(self, energy, intensity, params):
        # calculates linear background for simulated spectrum

        bg = np.zeros_like(energy)

        if energy[0] < energy[-1]:
            need_to_reversed = False
        else:
            need_to_reversed = True
            energy = energy[::-1]

        bg = bg + (energy - energy[0])*params['value']

        if need_to_reversed:
            return bg[::-1]
        else:
            return bg

    # ----------------------------------------------------------------------
    def square(self, energy, intensity, params):
        # calculates square background for simulated spectrum

        bg = np.zeros_like(energy)

        if energy[0] < energy[-1]:
            need_to_reversed = False
        else:
            need_to_reversed = True
            energy = energy[::-1]

        bg = bg + np.square((energy - energy[0]))*params['value']

        if need_to_reversed:
            return bg[::-1]
        else:
            return bg

    # ----------------------------------------------------------------------
    def shirley(self, energy, intensity, params):
        # calculates shirley background for simulated spectrum

        bg = np.zeros_like(energy)

        if energy[0] < energy[-1]:
            need_to_reversed = False
        else:
            need_to_reversed = True
            intensity = intensity[::-1]

        bg = bg + (np.cumsum(intensity) - intensity) * params['value']

        if need_to_reversed:
            return bg[::-1]
        else:
            return bg

    # ----------------------------------------------------------------------
    def voigth_function(self, x, gaus, lorenz):

        gauss = gaus / np.sqrt(2 * np.log(2))
        return np.real(wofz((x + 1j * lorenz) / gauss / np.sqrt(2))) / gauss / np.sqrt(2 * np.pi)

    # ----------------------------------------------------------------------
    def voigth(self, energy, intensity, params):
        """
        Return the Voigt line shape

        """
        return params['area']['value'] * self.voigth_function(np.array(energy) - params['center']['value'],
                                                              params['gauss']['value'], params['lorenz']['value'])

    # ----------------------------------------------------------------------
    def voigth_doublet(self, energy, intensity, params):
        """
        Return the double Voigt line shape at x

        """
        component1 = params['areaMain']['value'] * self.voigth_function(np.array(energy) - params['centerMain']['value'],
                                                              params['gaussMain']['value'], params['lorenzMain']['value'])

        component2 = params['areaMain']['value']*params['areaRation']['value'] * \
                     self.voigth_function(np.array(energy) - params['centerMain']['value'] - params['separation']['value'],
                                          params['gaussMain']['value']*params['gaussSecond']['value'],
                                          params['lorenzMain']['value']*params['lorenzSecond']['value'])

        return component1 + component2

    # ----------------------------------------------------------------------
    def doniach_sunjic_function(self, x, fwhm, asymmetry):

        numerator = np.cos(np.pi * asymmetry / 2 + (1 - asymmetry) * np.arctan(-x/fwhm))
        denumerator = ((fwhm**2+x**2)**((1-asymmetry)/2))

        return numerator/denumerator

    # ----------------------------------------------------------------------
    def doniach_sunjic(self, energy, intensity, params):
        """
        Return the Doniach Sunjic line shape at x
        """
        return params['area']['value'] * self.doniach_sunjic_function(np.array(energy) - params['center']['value'],
                                                                        params['fwhm']['value'],
                                                                        params['asymmetry']['value'])

    # ----------------------------------------------------------------------
    def doublet_doniach_sunjic(self, energy, intensity, params):
        """
        Return the Doniach Sunjic line shape at x
        """
        def _get_intensity_scale(fwhm, asymmetry):
            coefs = np.zeros(len(self.doniach_sunjic_coeffs))
            for set_ind, coef_set in enumerate(self.doniach_sunjic_coeffs):
                for ind, coef in enumerate(coef_set):
                    coefs[set_ind] += coef * fwhm ** ind

            scale_factor = 0
            for coef_ind, coef in enumerate(coefs):
                scale_factor += coef * asymmetry ** coef_ind
            return scale_factor

        fwhmSecond = params['fwhmMain']['value']*params['fwhmSecond']['value']
        asymmetrySecond = params['asymmetryMain']['value']*params['asymmetrySecond']['value']

        factor_main = _get_intensity_scale(params['fwhmMain']['value'], params['asymmetryMain']['value'])
        factor_said = _get_intensity_scale(fwhmSecond, asymmetrySecond)

        intensity_said = params['areaMain']['value']*params['areaRation']['value']*factor_main/factor_said
        main_peak = params['areaMain']['value'] * self.doniach_sunjic_function(np.array(energy) - params['centerMain']['value'],
                                                                 params['fwhmMain']['value'],
                                                                 params['asymmetryMain']['value'])

        said_peak = intensity_said * self.doniach_sunjic_function(np.array(energy) - params['centerMain']['value'] -
                                                                  params['separation']['value'], fwhmSecond, asymmetrySecond)

        return main_peak + said_peak
