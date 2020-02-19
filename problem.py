import numpy as np
import time
from optparse import OptionParser
from subfunctions import get_shifts
from fitter import NTR_fitter


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--data_set", dest="data_set")
    (options, _) = parser.parse_args()

    if options.data_set:
        pass
    else:
        raise RuntimeError('add path to .set file')

    fitter = NTR_fitter()
    start_values = fitter.load_fit_set(options.data_set)

    volt_sets = np.array([[-0.22950525, -0.00392041],
                      [-0.17509115, -0.00451116],
                      [-0.2013305 , -0.0053407],
                      [-0.30514265, -0.0064793]])

    depth_sets = np.array([1.03e-08, 0.00e+00])

    results = []
    for volt_set in volt_sets:
        ans, _ = get_shifts(fitter.main_data_set, depth_sets, volt_set)
        results.append(ans)

    print(results[0] == results[1])