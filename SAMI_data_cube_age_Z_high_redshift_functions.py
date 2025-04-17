import numpy as np
from astropy.io import fits
from ppxf.ppxf_util import log_rebin
from pathlib import Path
from urllib import request
import matplotlib.pyplot as plt
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from SAMI_data_cube_quality_cut_functions import data_cube_clean_percentage, data_cube_clean_snr
#--------------------------------------------------------------------------------------------
