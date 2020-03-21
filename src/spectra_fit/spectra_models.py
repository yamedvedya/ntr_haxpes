import numpy as np
from scipy.special import wofz

bck_models = ('constant', 'linear', 'square', 'shirley')
line_models = ('voigth', 'voigth_doublet')

class models():

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
    def voigth(self, energy, intensity, params):
        """
        Return the Voigt line shape at x with Lorentzian component HWHM lorenz
        and Gaussian component HWHM gauss.

        """
        x = np.array(energy) - params['center']['value']
        gauss = params['gauss']['value'] / np.sqrt(2 * np.log(2))
        return params['area']['value'] * np.real(wofz((x + 1j * params['lorenz']['value']) / gauss / np.sqrt(2))) / gauss \
                        / np.sqrt(2 * np.pi)


    # ----------------------------------------------------------------------
    def voigth_doublet(self, energy, intensity, params):
        """
        Return the double Voigt line shape at x with Lorentzian component HWHM lorenz
        and Gaussian component HWHM gauss.

        """
        x = np.array(energy) - params['centerMain']['value']
        gauss = params['gaussMain']['value'] / np.sqrt(2 * np.log(2))
        component1 = params['areaMain']['value'] * np.real(wofz((x + 1j * params['lorenzMain']['value']) / gauss / np.sqrt(2))) / gauss \
                        / np.sqrt(2 * np.pi)

        x = np.array(energy) - params['centerMain']['value'] - params['separation']['value']
        gauss = params['gaussMain']['value']*params['gaussSecond']['value']/ np.sqrt(2 * np.log(2))
        component2 = params['areaMain']['value']*params['areaRation']['value']* np.real(wofz((x + 1j * params['lorenzMain']['value']*params['lorenzSecond']['value']) / gauss / np.sqrt(2))) / gauss \
                        / np.sqrt(2 * np.pi)

        return component1 + component2