import numpy as np

# ----------------------------------------------------------------------
def get_model_list():

    model_1 = {'name': 'Piecewise linear',
               'code': 'lin',
               'default_widget': 'top_bottom_potential',
               'only_voltage_dof': 2,
               'fixed_depth_dof': False,
               'num_depth_dof': 0,
               'additional_widgets': ['breaking_point']}

    model_2 = {'name': 'Double square',
               'code': 'dbl_sqrt',
               'default_widget': 'top_bottom_potential',
               'only_voltage_dof': 2,
               'fixed_depth_dof': True,
               'num_depth_dof': 2,
               'additional_widgets': ['breaking_point']}

    return [model_1, model_2]
# ----------------------------------------------------------------------
def calculatePotential(depth_set, volt_set, fit_depth_points, model):

    if model == 'lin':
        return np.interp(fit_depth_points, depth_set, volt_set)

    elif model == 'dbl_sqrt':
        volts = np.zeros_like(fit_depth_points)

        ind_top = np.argmin(np.abs(fit_depth_points-depth_set[1]))
        squares = np.square(fit_depth_points[0:ind_top] - depth_set[1])
        volts[0:ind_top] = volt_set[1] + squares/np.amax(squares) * (volt_set[0]-volt_set[1])

        ind_bottom = np.argmin(np.abs(fit_depth_points-depth_set[2]))
        squares = np.square(fit_depth_points[ind_bottom:] - depth_set[2])
        volts[ind_bottom:] = volt_set[2] + squares/np.amax(squares) * (volt_set[3]-volt_set[2])

        volts[ind_top:ind_bottom] = np.interp(fit_depth_points[ind_top:ind_bottom], depth_set, volt_set)

        return volts
    else:
        raise RuntimeError('Unknown potential model')