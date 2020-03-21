# ----------------------------------------------------------------------
# DEFAULT GENERAL SETTINGS
# ----------------------------------------------------------------------

SUB_LAYERS = 101
LAMBDA = 9.6

MULTIPROCESSING = False
N_SUB_JOBS = 2
USE_ALL_CORES = True
NUM_CORES = 1

# ----------------------------------------------------------------------
# DEFAULT INTENSITY FIT SETTINGS
# ----------------------------------------------------------------------

MAX_NUM_COMPONENTS = 10

# ----------------------------------------------------------------------
# DEFAULT INTENSITY FIT SETTINGS
# ----------------------------------------------------------------------

X_WAY = 2
WAVE = 4.60
LINE = ''
IPOL = 2

SUBWAY = 1
CODE = 'Si'
W0 = 1.00
X0 = '(0., 0.)'
CHEM = ''
RHO = 0
SIGMA = 4
TR = 0
DF1DF2 = -1

SCANMIN = 0.20
SCANMAX = 1.20
UNIS = 0
NSCAN = 201

SWFLAG = 1
SWREF = 0
SWMIN = 0
SWMAX = 250
SWPTS = 251

THICKSTEP = 1
COMPSTEP = 0.05
SIGMASTEP = 1
X0STEP = 0.05
W0STEP = 0.05

RSSTOL = 0.01
SEARCHSTEPS = 5

# ----------------------------------------------------------------------
# DEFAULT POTENTIAL FIT SETTINGS
# ----------------------------------------------------------------------

G = 0.40
L = 0.30
BE_STEP = 0.01
SIM_SPECTRA_WIDTH = 2.00
VOLT_MAX = 1.00

T = 0.05
KSI_TOLLERANCE = 0.010
FIT_SOLVER = 'mesh_gradient'
MONITOR_FIT = True
DISPLAY_EACH_X_STEP = 1
V_MESH = 5
D_MESH = 5
V_STEP = 0.01
D_STEP = 0.50

FIELD_MAX = 1

METHOD = ''