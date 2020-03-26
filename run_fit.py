from distutils.util import strtobool
from optparse import OptionParser
from src.ntr_data_fitting.ntr_fitter import NTR_fitter

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-s", "--data_set", dest="data_set")
    parser.add_option("-p", "--plot", dest="do_plot", default=False)
    (options, _) = parser.parse_args()
    if options.data_set:
        fitter = NTR_fitter()
        if options.do_plot:
            fitter.DO_PLOT = strtobool(options.do_plot)
        else:
            fitter.DO_PLOT = False
        start_values = fitter.load_fit_set(options.data_set)
        fitter.do_potential_fit(start_values)