import numpy as np
from scipy.special import wofz
from scipy.io import loadmat
from scipy.optimize import leastsq
from src.ntr_data_fitting.potential_models import calculatePotential

ABSORPTION_LENGTH = 5e-6


# ----------------------------------------------------------------------
def get_shifts(data_set, depth_set, volt_set):

    if np.where(np.abs(np.diff(volt_set) / np.diff(depth_set)) > data_set['FIELD_MAX']*1e9)[0].size == 0:
        volts_values = calculatePotential(depth_set, volt_set, data_set['fit_depth_points'], data_set['model_code'])
        peak_fit_result = np.zeros_like(data_set['spectroscopic_data'][:, 0])
        intensities = np.zeros_like(data_set['spectroscopic_data'][:, 0])

        for angle in range(len(data_set['spectroscopic_data'][:, 0])):
            simulated_spectra, intensities[angle] = simulate_spectra(data_set['fit_spectra_set'][angle, :, :],
                                                                     data_set['sum_spectra_energy'],
                                                                     volts_values, data_set['SUB_LAYERS'],
                                                                     data_set['sum_spectra_point'],
                                                                     data_set['ref_spectra_points'],
                                                                     data_set['BE_STEP'])

            peak_fit_result[angle] = fit_spectra(data_set['ref_spectra'], data_set['sum_spectra_energy'],
                                                 simulated_spectra)

        return peak_fit_result, intensities
    else:
        return None, None


# ----------------------------------------------------------------------
def get_intensity_simple(exp_angles, exp_intensities, depths_set, sw, lam):

    def comare_spectra(params):
        sim_sp = np.interp(exp_angles + params[0], sw['angles'],
                           intensities)*params[1]
        return exp_intensities - sim_sp

    intensities = np.zeros_like(sw['angles'])
    atten_coefs = np.exp(-depths_set / lam)

    for angle in range(len(sw['angles'])):
        intensities[angle] = np.sum(np.interp(depths_set, sw['depth'], sw['sw'][angle, :]) * atten_coefs)

    result = leastsq(comare_spectra, np.array([sw['angles'][np.argmax(intensities)] -
                                                            exp_angles[np.argmax(exp_intensities)],
                                               np.max(exp_intensities)/np.max(intensities)]))

    return intensities*result[0][1], result[0][0]


# ----------------------------------------------------------------------
def get_sw(sw_file):
    with open(sw_file) as f:
        _file_lines = f.readlines()

    return parse_sw(_file_lines)


# ----------------------------------------------------------------------
def parse_sw(lines):

    sw = {}

    num_depths, num_brags = list(map(int, lines[1].split()))
    start_depth, end_depth = list(map(float, lines[2].split()))
    start_angle, end_angle = list(map(float, lines[3].split()))

    sw['depth'] = np.linspace(start_depth, end_depth, num_depths) * 1e-10
    sw['angles'] = np.linspace(start_angle, end_angle, num_brags)

    sw['sw'] = np.zeros((num_brags, num_depths))
    counter = 0
    for line in lines[5:]:
        sw['sw'][counter, :] = list(map(float, line.split()))
        counter += 1

    return sw


# ----------------------------------------------------------------------
def get_data_set(data_file, sample):
    mat = loadmat(data_file)
    return np.array(mat[sample][:, 0:3]), mat['thicknesses'][0]*1e-10, mat['shift{}'.format(sample)][0][0]


# ----------------------------------------------------------------------
def correct_intensity(data, thichness):

    return data[:, 1]/(1 - np.exp(-(thichness)/(np.sin(np.deg2rad(data[:, 0]))*ABSORPTION_LENGTH)))


# ----------------------------------------------------------------------
def syn_spectra(min_e, max_e, step, G, L):

    energy_scale = np.linspace(min_e, max_e, int((max_e - min_e)/step)+1)
    spectra = np.zeros((len(energy_scale), 2))
    spectra[:, 0] = energy_scale

    G /= np.sqrt(2 * np.log(2))
    spectra[:, 1] = np.real(wofz((energy_scale + 1j * L) / G / np.sqrt(2))) / G / np.sqrt(2 * np.pi)

    spectra[:, 1] /= np.max(spectra[:, 1])

    return spectra


# ----------------------------------------------------------------------
def generate_fit_set(source_spectra, depths_set, angle_set, sw, lam):

    num_depth_points = len(depths_set)
    num_energy_points = source_spectra.shape[0]
    num_angle_points = len(angle_set)

    fit_elements = np.zeros((num_angle_points, num_depth_points, num_energy_points))
    atten_coefs = np.exp(-depths_set/lam)

    angles = np.zeros_like(angle_set, dtype=int)
    for ind in range(len(angles)):
        angles[ind] = int(np.argmin(np.abs(angle_set[ind] - sw['angles'])))

    for angle in range(num_angle_points):
        sw_cut = np.interp(depths_set, sw['depth'], sw['sw'][angles[angle], :])*atten_coefs
        for depth in range(num_depth_points):
            fit_elements[angle, depth, :] = source_spectra[:, 1]*sw_cut[depth]

    return fit_elements


# ----------------------------------------------------------------------
def simulate_spectra(fit_spectra_set, sum_spectra_energy, volts_values,
                     fit_points, fit_spectra_points, source_spectra_points, be_step):

    default_ind = source_spectra_points - fit_spectra_points

    sum_spectra_intensity = np.zeros_like(sum_spectra_energy)
    max_ind = fit_spectra_set.shape[1]
    for ind in range(fit_points):
        start_ind = int(default_ind - volts_values[ind]/be_step)
        end_ind = start_ind + 2 * fit_spectra_points + 1
        if start_ind > 0 and end_ind < max_ind:
            sum_spectra_intensity += fit_spectra_set[ind, start_ind:start_ind + 2 * fit_spectra_points + 1]
        else:
            raise RuntimeError('Too big voltage')

    intensity = np.amax(sum_spectra_intensity)
    sum_spectra_intensity /= intensity

    return sum_spectra_intensity, intensity


# ----------------------------------------------------------------------
def fit_spectra(source_spectra, simulated_spectra_energy, simulated_spectra_intensity):

    def comare_spectra(params):
        sim_sp = np.interp(simulated_spectra_energy, source_spectra[:, 0]*params[0] + params[1],
                           source_spectra[:, 1])
        return simulated_spectra_intensity - sim_sp

    result = leastsq(comare_spectra, np.array([1, simulated_spectra_energy[np.argmax(simulated_spectra_intensity)]]))

    return result[0][1]


# ----------------------------------------------------------------------
def get_precision(f):
    str_f = str(f)
    if '.' not in str_f:
        return 0

    return len(str_f[str_f.index('.') + 1:])

# ----------------------------------------------------------------------
if __name__ == "__main__":
    pass