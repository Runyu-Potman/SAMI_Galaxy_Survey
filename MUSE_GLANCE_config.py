
import os
import configparser
from ast import literal_eval

# --------------------------------------

def create_config(gal_list):
  
  CONFIG_TEMPLATE = """# ==========================
# GLANCE configuration file
# ==========================

[io]
GAL_INFO = {gal_list}

[paths]
SSP_DIR = 
STT_DIR = 
BIN_DIR = 
BAYES_DIR = 

[general]
CORES = 0
NEB_CONT = 1
N_MCS = (100,100)
SPEC_LIM = 8740

[fado]
TEMPS_FD = ['BC03']
LIBS_FD = ['Z4.LT.BC']

[starlight]
TEMPS_SL = ['BC03']
LIBS_SL = ['Z4.LT.BC']
SL_CONF = 'LTG.conf'

[ry]
RY_SSC = ['FADO', 'Starlight']
RY_TCS = [0.031,0.102,0.302,1.02,3.01,5.01,7.01,9.01]

[ppxf]
WR_PPXF = 'F'
TEMPS_PPXF = ['IUS']
LINE_STP = 'sep_tied'
NMOM = 4
GST_STP = 'GST'

[dyn_ppxf]
CHI2_TYPE = 'kinmapchi2'
TODO_PDYN = ['MGE', 'kin_clean', 'config']
HST_FILT = 'F814W'
MGE_M = 0
SINF = 0

[bayes]
TEMP_BAYES = 'IUS'
REGUL = 'SP'
WR_BAYES = ['red']

[orbit_decomp]
OCUT = None
V_SIGMA = 'fit'
ERR_DSBP = True

[sinfoni]
TEMPS_SINF = ['GNIRS']
NMOM_SINF = 4 """

  INSTRUCTIONS = r"""
 ==========================
 GLANCE configuration guide
 ==========================

 1. A file named 'glance_config.ini' was created in the current directory.

 2. Open it and fill the parameters for the modules you intend to run (note: provide fulls path and add a slash at the end).
    
 - Quick summary of key options:

 [io]
   GAL_INFO     : name of the file with the ID's of your galaxy sample (one per line)

 [paths]
   SSP_DIR      : where SSP libraries & libfiles are located (BC03, CB19, ...)
   STT_DIR      : where your stellar templates are located (IUS, MILES, ...)
   BIN_DIR      : base directory where external binaries (FADO, RY, ...) are located
   BAYES_DIR    : full path to BayesLOSVD dynamical code

 [general]
   CORES        : number of cores (0 = all available)
   NEB_CONT     : nebular continuum subtraction before pPXF / Starlight (1/0)
   N_MCS        : (x,y) where x/y are nº of Monte Carlo realizations for pPXF (x) and FADO / Starlight (y)
   SPEC_LIM     : upper wavelength limit (units of Angstrom)

 [fado]
   TEMPS_FD    : ['BC03'], ['CB19'] or ['BC03','CB19'] 
   LIBS_FD     : name/names of the SSP library to be adopted (per temp in TEMPS_FD; ex. Z4.LT.BC)

 [starlight]
   TEMPS_SL    : ['BC03'], ['CB19'], ['MILES'] or any combination
   LIBS_SL     : name/names of the SSP library to be adopted (per temp in TEMPS_SL; ex. Z4.LT.BC)
   SL_CONF     : Starlight's config file: we provide 'LTG.conf' & 'ELG.conf', optimal for LTGs and ELGs, respectively (however one can use others)

 [ry]
   RY_SSC      : ['FADO'], ['Starlight'], or ['FADO', 'Starlight']
   RY_TCS      : List with age cuts for RY (default: [0.031,0.102,0.302,1.02,3.01,5.01,7.01,9.01] Gyr)

 [ppxf]
   WR_PPXF      : spectral range to extract kinematics 'B' (blue; 4820-6000; only for IUS), 'R' (red; 7700-8720; only for IUS), 'F' (full spectral range), or any combination (ex: 'FBR')
   TEMPS_PPXF   : ['IUS'], ['MILES'] or ['IUS', 'MILES']
   LINE_STP     : 'sep_tied', 'sep' or 'join' (see pPXF documentation)
   NMOM         : LOSVD moments (recommended: 4 if Dynamite is used)
   GST_STP      : 'ST' (extract stellar kin) or 'GST' (extract stellar+gas kin simultaneously)

 [dyn_ppxf]
   CHI2_TYPE    : 'kinmapchi2' or 'kinchi2' (see Dynamite's documentation)
   TODO_PDYN    : list of comma-separated strings indicating what preparatory files youll create: ['MGE', 'kin_Noclean'/'kin_clean', 'config']
   HST_FILT     : 'F814W' or 'F606W'
   MGE_M        : use FADO's derived MLR in MGE & dynamics (1/0)
   SINF         : 1 if SINFONI data are to be considered in this run; else 0

 [bayes]
   TEMP_BAYES  : 'IUS' or 'MILES_SSP'
   REGUL       : 1 of: 'SP', 'RW', 'AR1', 'AR2', 'AR3', 'Bsplines3', 'Bsplines4', 'GHfree' (see BayesLOSVD documentation)
   WR_BAYES    : 'all', 'red', or 'blue' (note that 'all' might take a long time to complete)

 [orbit_decomp]
   OCUT        : None (default: [0.8, 0.25, -0.25]) or three comma-separated cuts
   V_SIGMA     : 'fit' or 'moments' ('fit' is the recommended)
   ERR_DSBP    : True = compute SBP uncertainties, False = skip

 [sinfoni]
   TEMPS_SINF  : ['GNIRS'], ['NIFS'] or ['GNIRS', 'NIFS']
   NMOM_SINF   : number of SINFONI moments (recommended: 4 if Dynamite is used)

 3. Save 'glance_config.ini' and you are ready to run GLANCE ;)
 
# ----------------------------------------------------------------------------------------"""
    
  print('')
  if os.path.exists('glance_config.ini'):
    df = 'y'
    ans = input(' ⇒ A configuration file exists in this folder. Would you like to keep it (Y) or create an empty new one (n)? ').strip().lower() or df
    if ans == 'y':
      pass
    else:
      with open('glance_config.ini', 'w') as f:
        f.write(CONFIG_TEMPLATE.format(gal_list=gal_list))
  else:
    print(' ➞ An empty file was created, fill the desired parameters according to the following instructions:')
    
    with open('glance_config.ini', 'w') as f:
      f.write(CONFIG_TEMPLATE.format(gal_list=gal_list))
                    
  print(INSTRUCTIONS)

# --------------------------------------

def parse_value(value):
  value = value.strip()

  if value.startswith("'") and value.endswith("'"):
    return value[1:-1]

  try:
    return literal_eval(value)
  except Exception:
    pass

  try:
    return int(value)
  except ValueError:
    pass

  return value

# --------------------------------------

def read_config():
  
  config = configparser.ConfigParser()
  config.read('glance_config.ini')
  params = {}

  for section in config.sections():
    params[section] = {}
    for key, value in config[section].items():
      params[section][key] = parse_value(value)

  GAL_INFO   = params['io']['gal_info']

  SSP_DIR    = params['paths']['ssp_dir']
  STT_DIR    = params['paths']['stt_dir']
  BIN_DIR    = params['paths']['bin_dir']
  BAYES_DIR  = params['paths']['bayes_dir']

  CORES      = params['general']['cores']
  NEB_CONT   = params['general']['neb_cont']
  N_MCS      = params['general']['n_mcs']
  SPEC_LIM   = params['general']['spec_lim']

  TEMPS_FD   = params['fado']['temps_fd']
  LIBS_FD    = params['fado']['libs_fd']

  TEMPS_SL   = params['starlight']['temps_sl']
  LIBS_SL    = params['starlight']['libs_sl']
  SL_CONF    = params['starlight']['sl_conf']
  
  RY_SSC     = params['ry']['ry_ssc']
  RY_TCS     = params['ry']['ry_tcs']

  WR_PPXF    = params['ppxf']['wr_ppxf']
  TEMPS_PPXF = params['ppxf']['temps_ppxf']
  LINE_STP   = params['ppxf']['line_stp']
  NMOM       = params['ppxf']['nmom']
  GST_STP    = params['ppxf']['gst_stp']

  TEMP_BAYES = params['bayes']['temp_bayes']
  REGUL      = params['bayes']['regul']
  WR_BAYES   = params['bayes']['wr_bayes']

  CHI2_TYPE  = params['dyn_ppxf']['chi2_type']
  TODO_PDYN  = params['dyn_ppxf']['todo_pdyn']
  HST_FILT   = params['dyn_ppxf']['hst_filt']
  MGE_M      = params['dyn_ppxf']['mge_m']
  SINF       = params['dyn_ppxf']['sinf']

  OCUT       = params['orbit_decomp']['ocut']
  V_SIGMA    = params['orbit_decomp']['v_sigma']
  ERR_DSBP   = params['orbit_decomp']['err_dsbp']

  TEMPS_SINF = params['sinfoni']['temps_sinf']
  NMOM_SINF  = params['sinfoni']['nmom_sinf']

  return GAL_INFO, SSP_DIR, STT_DIR, BIN_DIR, BAYES_DIR, CORES, NEB_CONT, N_MCS, SPEC_LIM, WR_PPXF, TEMPS_PPXF, LINE_STP, NMOM, GST_STP, CHI2_TYPE, TODO_PDYN, HST_FILT, MGE_M, SINF, TEMP_BAYES, REGUL, WR_BAYES, OCUT, V_SIGMA, ERR_DSBP, TEMPS_FD, LIBS_FD, TEMPS_SL, LIBS_SL, SL_CONF, RY_SSC, RY_TCS, TEMPS_SINF, NMOM_SINF
