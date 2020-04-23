from optparse import OptionParser
from src.ntr_data_fitting.gradient_mesh import Gradient_Mesh_Solver

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-s", "--data_set", dest="data_set")
    parser.add_option("-p", "--plot", dest="do_plot", default=False)
    (options, _) = parser.parse_args()
    if options.data_set:
        fitter = Gradient_Mesh_Solver()
        fitter.load_fit_set(options.data_set)
        fitter.do_fit(3)