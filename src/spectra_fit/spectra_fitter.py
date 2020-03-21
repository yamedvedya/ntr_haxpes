import numpy as np
import copy
import h5py
from lmfit import Parameters, report_fit, minimize
import pickle
from src.spectra_fit.spectra_models import models

class Spectra_fitter():
    
    def __init__(self):
        self.data = []
        self.ndata  = 0
        self.current_fit_num = 0

        self.spectra_experiment_plot = None
        self.spectra_sum_plot = None
        self.spectra_bcg_plot = None
        self.spectra_plots = None

        self.bg_params = []
        self.peaks_info = []

        self.fit_params = Parameters()

        self.resid = []
        self.models = models()
        
        self._fit_params = None
        self.baseValues = {}
        self.functional_peak = ''

    # ----------------------------------------------------------------------
    def load_data(self, file_list):

        for file_name in file_list:
            file = h5py.File(file_name, "r")
            for key in file.keys():
                try:
                    intensity = np.array(file["{}/data/intensity".format(key)])
                    shape = intensity.shape
                    intensity = np.mean(intensity.reshape(shape[1], shape[2]), axis=1)

                    energy = float(file['{}'.format(key)]['metainfo']['mask']['excitationEnergy'][()]) - \
                             np.array(file["{}/data/energy".format(key)])
                    self.data.append({'angle': file['{}'.format(key)]['instrument']['HAXPES']['polar']['position'][()],
                                      'energy': energy,
                                      'intensity': intensity,
                                      'range': 'auto',
                                      'limits': [np.min(energy), np.max(energy)],
                                      'lim_index': [0, -1]})
                except:
                    print('Cannot read set {} in file {}'.format(key, file_name))

        self.ndata = len(self.data)

        self.bg_params = [[] for _ in range(self.ndata)]
        self.peaks_info = [[] for _ in range(self.ndata)]

        return self.ndata

    # ----------------------------------------------------------------------
    def set_default_plots(self, spectra_experiment_plot, spectra_sum_plot, spectra_bcg_plot, spectra_plots):
        
        self.spectra_experiment_plot = spectra_experiment_plot
        self.spectra_sum_plot = spectra_sum_plot
        self.spectra_bcg_plot = spectra_bcg_plot
        self.spectra_plots = spectra_plots
        
    # ----------------------------------------------------------------------
    def plot_spectra(self, spectra_ind):

        self.spectra_experiment_plot.setData(self.data[spectra_ind]['energy'], self.data[spectra_ind]['intensity'])
        
        #plots the component i onto ax using params


        if self.bg_params[spectra_ind] or self.peaks_info[spectra_ind]:
            sum_line, bg = self.sim_spectra(spectra_ind, [])

            self.spectra_bcg_plot.setData(self.data[spectra_ind]['energy'], bg)
            self.spectra_sum_plot.setData(self.data[spectra_ind]['energy'], sum_line)

            counter = 0
            for peak in self.peaks_info[spectra_ind]:
                line = bg + self.models.getModel(peak['peakType'], self.data[spectra_ind]['energy'],
                                                 self.data[spectra_ind]['intensity'], peak['params'])

                self.spectra_plots[counter].setData(self.data[spectra_ind]['energy'], line)
                counter += 1
    
    # ----------------------------------------------------------------------
    def _get_param_value(self, peak_name, param_data, param_name, spectra_ind, fit_params):

        if param_data['model'] in ['Common_Dependent', 'Dependent', 'Dependent_on_fixed']:
            tokens = param_data['baseValue'].split('/')
            if len(tokens) == 1:
                tokens.append(param_name)
        else:
            tokens = []

        if param_data['fitable']:
            if param_data['model'] == 'Flex':
                return 0 + fit_params['{}_{}_{}'.format(peak_name, param_name, spectra_ind)]

            elif param_data['model'] == 'Dependent':
                if param_data['linkType'] == 'additive':
                    return fit_params['{}_{}_{}'.format(tokens[0], tokens[1], spectra_ind)] +\
                           fit_params['{}_{}_{}'.format(peak_name, param_name, spectra_ind)]
                else:
                    return fit_params['{}_{}_{}'.format(tokens[0], tokens[1], spectra_ind)] *\
                           fit_params['{}_{}_{}'.format(peak_name, param_name, spectra_ind)]

            elif param_data['model'] == 'Dependent_on_fixed':
                if param_data['linkType'] == 'additive':
                    return self.baseValues['{}_{}_{}'.format(tokens[0], tokens[1], spectra_ind)] +\
                           fit_params['{}_{}_{}'.format(peak_name, param_name, spectra_ind)]
                else:
                    return self.baseValues['{}_{}_{}'.format(tokens[0], tokens[1], spectra_ind)] *\
                           fit_params['{}_{}_{}'.format(peak_name, param_name, spectra_ind)]
            else:
                raise RuntimeError('Unknown model')
        else:
            if param_data['model'] == 'Dependent':
                if param_data['linkType'] == 'additive':
                    return fit_params['{}_{}_{}'.format(tokens[0], tokens[1], spectra_ind)] +\
                           param_data['value']
                else:
                    return fit_params['{}_{}_{}'.format(tokens[0], tokens[1], spectra_ind)] *\
                           param_data['value']

            else:
                return param_data['value']

    # ----------------------------------------------------------------------
    def _get_param_error(self, peak_name, param_data, param_name, spectra_ind, fit_params):
        try:
            if param_data['fitable']:
                if param_data['model'] == 'Flex':
                    return 0 + fit_params[spectra_ind]['{}_{}_{}'.format(peak_name, param_name, spectra_ind)].stderr

                elif param_data['model'] == 'Dependent':
                       return fit_params[spectra_ind]['{}_{}_{}'.format(param_data['baseValue'], param_name, spectra_ind)].stderr + \
                              fit_params[spectra_ind]['{}_{}_{}'.format(peak_name, param_name, spectra_ind)].stderr
                else:
                    raise RuntimeError('Unknown model')
            else:
                if param_data['model'] == 'Dependent':
                    return fit_params[spectra_ind]['{}_{}_{}'.format(param_data['baseValue'], param_name, spectra_ind)].stderr * \
                           param_data['value']
                else:
                    return 0
        except:
            return 0
    # ----------------------------------------------------------------------
    def _get_bck_values(self, spectra_ind, fit_params):

        local_bg = copy.deepcopy(self.bg_params[spectra_ind])

        for type, params in local_bg.items():
            if fit_params:
                if params['fitable']:
                    params['value'] = fit_params['bg_{}_{}'.format(type, spectra_ind)]
            if type == 'constant':
                if params['value'] == 'first':
                    if self.data[spectra_ind]['energy'][0] < self.data[spectra_ind]['energy'][-1]:
                        params['value'] = self.data[spectra_ind]['intensity'][0]
                    else:
                        params['value'] = self.data[spectra_ind]['intensity'][-1]
                elif params['value'] == 'min':
                    params['value'] = np.min(self.data[spectra_ind]['intensity'])

        return local_bg

    # ----------------------------------------------------------------------
    def sim_spectra(self, spectra_ind, fit_params):

        # defines the model for the fit (doniachs, voigts, shirley bg, linear bg

        line = np.zeros_like(self.data[spectra_ind]['intensity'])
        bg = np.zeros_like(self.data[spectra_ind]['intensity'])

        local_bg = self._get_bck_values(spectra_ind, fit_params)

        for peak in copy.deepcopy(self.peaks_info[spectra_ind]):
            if fit_params:
                for param_name, param_data in peak['params'].items():
                    param_data['value'] = self._get_param_value(peak['name'], param_data,
                                                                param_name, spectra_ind, fit_params)

            line += self.models.getModel(peak['peakType'], self.data[spectra_ind]['energy'],
                                         self.data[spectra_ind]['intensity'], peak['params'])

        for type, params in local_bg.items():
            bg += self.models.getModel(type, self.data[spectra_ind]['energy'], line, params)

        line += bg

        return line, bg

    # ----------------------------------------------------------------------
    def err_func(self, fit_params):
        """ calculate total residual for fits to several data sets held
        in a 2-D array, and modeled by model function"""

        self.resid = []

        ind_start =self.data[self.current_fit_num]['lim_index'][0]
        ind_end = self.data[self.current_fit_num]['lim_index'][1]

        spectra, _ = self.sim_spectra(self.current_fit_num, fit_params)
        self.resid.append(self.data[self.current_fit_num]['intensity'][ind_start:ind_end] - spectra[ind_start:ind_end])

        return [item for innerlist in self.resid for item in innerlist]

    # ----------------------------------------------------------------------
    def get_start_value_back(self, index, type):
        if type == 'first':
            if self.data[index]['energy'][0] > self.data[index]['energy'][1]:
                return self.data[index]['intensity'][-1]
            else:
                return self.data[index]['intensity'][0]
        elif type == 'min':
            return np.min(self.data[index]['intensity'])

    # ----------------------------------------------------------------------
    def set_peak_params(self, indexes, bg_params, peaks_info):
        for index in indexes:
            self.bg_params[index] = copy.deepcopy(bg_params)
            self.peaks_info[index] = copy.deepcopy(peaks_info)

    # ----------------------------------------------------------------------
    def make_params(self, index):
        # uses the starting values for the first peak to construct
        # free variable voigts and doniachs for peak 0
        # and dependent centers, sigmas and gammas for all other peaks
        # amplitude is only free parameter

        self.fit_params = Parameters()
        for type, params in self.bg_params[index].items():
            if params['fitable']:
                if type == 'constant':
                    if params['value'] in ['first', 'last', 'min', 'max']:
                        value = self.get_start_value_back(index, params['value'])
                    else:
                        value = float(params['value'])

                    if params['min'] in ['first', 'last', 'min', 'max']:
                        min = self.get_start_value_back(index, params['value'])
                    else:
                        min = float(params['min'])

                    if params['max'] in ['first', 'last', 'min', 'max']:
                        max = self.get_start_value_back(index, params['value'])
                    else:
                        max = float(params['max'])

                else:
                    value = float(params['value'])
                    min = float(params['min'])
                    max = float(params['max'])

                self.fit_params.add('bg_{}_{}'.format(type, index), value=value,
                                   min=min, max=max)

        for peak in self.peaks_info[index]:
            for param_name, param_data in peak['params'].items():
                if param_data['fitable']:
                    if param_data['model'] in ['Flex', 'Dependent']:
                        if param_data['limModel'] == 'absolute':
                            self.fit_params.add('{}_{}_{}'.format(peak['name'], param_name, index), value=param_data['value'],
                                               min=param_data['min'], max=param_data['max'])
                        else:
                            self.fit_params.add('{}_{}_{}'.format(peak['name'], param_name, index), value=param_data['value'],
                                               min = param_data['value'] - param_data['min'],
                                               max = param_data['value'] + param_data['max'])

                        if param_data['model'] == 'Dependent':
                            tokens = param_data['baseValue'].split('/')
                            if len(tokens) == 1:
                                tokens.append(param_name)
                            for subPeak in self.peaks_info:
                                if subPeak['name'] == tokens[0]:
                                    if not subPeak['params'][tokens[1]]['fitable']:
                                        self.baseValues['{}_{}_{}'.format(peak['name'], param_name, index)] = \
                                            subPeak['params'][tokens[1]]['value']
                                        param_data['model'] = 'Dependent_on_fixed'

                elif param_data['model'] in ['Dependent']:
                    tokens = param_data['baseValue'].split('/')
                    if len(tokens) == 1:
                        tokens.append(param_name)
                    for subPeak in self.peaks_info:
                        if subPeak['name'] == tokens[0]:
                            if not subPeak['params'][tokens[1]]['fitable']:
                                if param_data['linkType'] == 'additive':
                                    param_data['value'] = subPeak['params'][tokens[1]]['value'] + param_data['value']
                                elif param_data['linkType'] == 'multiplication':
                                    param_data['value'] = subPeak['params'][tokens[1]]['value'] * param_data['value']
                                else:
                                    raise RuntimeError('Unknown link type')

                                param_data['model'] = 'fixed'

    # ----------------------------------------------------------------------
    def fit(self, indexes):
        # calls minimize from lmfit using the objective function and the parameters

        for index in indexes:
            self.make_params(index)
            self.current_fit_num = index
            if self.data[index]['range'] == 'auto':
                self.data[index]['lim_index'] = [0, -1]
            else:
                self.data[index]['lim_index'] = [np.where(self.data[index]['energy'] < self.data[index]['limits'][1])[0][0],
                                                 np.where(self.data[index]['energy'] > self.data[index]['limits'][0])[0][-1]]
            result = minimize(self.err_func, self.fit_params)
            self._update_peak_params(index, result.params)

    # ----------------------------------------------------------------------
    def _update_peak_params(self, index_to_refresh, params):
        for key, param in params.items():
            peak_name, parameter, index = key.split('_')
            if int(index) == index_to_refresh:
                if peak_name == 'bg':
                    self.bg_params[index_to_refresh][parameter]['value'] = param.value
                else:
                    for peak in self.peaks_info[index_to_refresh]:
                        if peak['name'] == peak_name:
                            peak['params'][parameter]['value'] = param.value

    # ----------------------------------------------------------------------
    def collect_data_for_fitter(self, peak_name):
        data = np.zeros((len(self.peaks_info), 5))
        for ind, set in enumerate(self.peaks_info):
            if set:
                for peak in set:
                    if peak['name'] == peak_name:
                        if peak['peakType'] == 'voigth':
                            params = ('area', 'center')
                        else:
                            params = ('areaMain', 'centerMain')
                        data[ind, 0:3] = [self.data[ind]['angle'], peak['params'][params[0]]['value'],
                                          peak['params'][params[1]]['value']]
        return data
    # ----------------------------------------------------------------------
    def dump_session(self, f):

        pickle.dump({'data': self.data, 'ndata': self.ndata,
                     'bg_params': self.bg_params, 'peaks_info': self.peaks_info,
                     'functional_peak': self.functional_peak},
                    f, pickle.HIGHEST_PROTOCOL)

    # ----------------------------------------------------------------------
    def restore_session(self, fr):
        loaded_data = pickle.load(fr)
        for key in loaded_data.keys():
            setattr(self, key, loaded_data[key])

        pass