
import re
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import units as u
from ppxf.ppxf import robust_sigma

from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy
from scipy.interpolate import CubicSpline

from dust_extinction.parameter_averages import G23

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Plotting
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import cmastro
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
fsize = 15; fontsize = 14 ; ticksize = 14 ; labsize = 16
font = {'family' : 'Times New Roman'}
axes = {'titlesize' : labsize,
        'labelsize' : labsize}
plt.rc('font', **font)
plt.rc('axes', **axes)
plt.rc('xtick', labelsize=ticksize)
plt.rc('ytick', labelsize=ticksize)
plt.rc('legend', fontsize=15)
plt.rcParams['mathtext.fontset'] = 'stix'
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image

# Multiprocessing
import ray
import logging

# Read from config
from GLANCE_config import read_config

# NOTE: Define here the FADO version to use
FADO_version = 'int-FADOV1B' # 'FADO.v1B_SL_output_1'

# --------------------------

def truncate_cmap(cmap,minval=0.2,maxval=1.0,n=256):
  cmap = LinearSegmentedColormap.from_list(f'trunc({cmap.name},{minval:.2f},{maxval:.2f})', cmap(np.linspace(minval, maxval, n)))
  return cmap

cmap = truncate_cmap(plt.get_cmap('cma:emph'), minval=0.3, maxval=1.0)
perc1 = 5 ; perc2 = 95

# --------------------------
tie_balmer=True ; limit_doublets=True
HOMEpath  = os.path.expanduser('~')
C = 299792.458  # speed of light in km/s
Uage = 13.7
Zs = 0.0196
# -----------------------------------------------------

# Correction for the Balmer Decrement (λ must be supplied in μm)
# https://sandrasavaglio.info/wp-content/uploads/2018/05/sfr_script.pdf
def factE(wvl,flx,HaHb):

  Rv = 3.1
  ext = G23(Rv=Rv) # Reddening law
  Halpha = 6564.614/10000 ; Hbeta  = 4862.721/10000
  Av_Ha = ext(Halpha*u.micrometer)*Rv
  Av_Hb = ext(Hbeta*u.micrometer)*Rv
  Avi = 2.5/(Av_Ha-Av_Hb)*np.log10(2.86/(HaHb))
  
  Av = ext(wvl*u.micrometer)*Rv
  return flx*10**(0.4*Av*Avi)

# -----------------------------------------------------

# Masking of emission lines & sky problematic areas 
# Ha, Hb, [OIII]4959,5007, [NII]6548,6583, [SII]6717,6730 & [OI]6300 shall not be masked
def flagging(wvl):
  
  flags = [
    (wvl >= 3718) & (wvl <= 3735), # *[OII] 3726   
    (wvl >= 3720) & (wvl <= 3737), # *[OII] 3728.8   
    (wvl >= 3796) & (wvl <= 3802), # H10 3797.9   
    (wvl >= 3833) & (wvl <= 3841), # H90 3835.4   
    (wvl >= 3862) & (wvl <= 3871), # *[NeIII] 3869.1   
    (wvl >= 3882) & (wvl <= 3891), # H8HeI 3889   
    (wvl >= 3961) & (wvl <= 3973), # [NeIII] 3967.8   
    (wvl >= 3961) & (wvl <= 3974), # Hepsilon 3970.1   
    (wvl >= 4024) & (wvl <= 4030), # HeI 4026.2   
    (wvl >= 4067) & (wvl <= 4070), # [SII] 4068.6   
    (wvl >= 4095) & (wvl <= 4107), # *Hdelta 4101.7   
    (wvl >= 4332) & (wvl <= 4346), # *Hgamma 4340.5   
    (wvl >= 4469) & (wvl <= 4475), # HeI 4471.5   
    (wvl >= 5009) & (wvl <= 5022), # HeI 5015.7   
    (wvl >= 5156) & (wvl <= 5164), # [FeVII] 5158.4   
    (wvl >= 5193) & (wvl <= 5202), # [NI] 5199.1   
    (wvl >= 5267) & (wvl <= 5274), # [FeIII] 5270.4   
    (wvl >= 5868) & (wvl <= 5883), # HeI 5875.6   
    (wvl >= 6303) & (wvl <= 6315), # [SIII] 6312.1   
    (wvl >= 6355) & (wvl <= 6372), # [OI] 6363.8   
    (wvl >= 6667) & (wvl <= 6681), # HeI 6678.2   
    (wvl >= 7061) & (wvl <= 7072), # HeI 7065.2   
    (wvl >= 7125) & (wvl <= 7142), # [ArIII] 7135.8   
    (wvl >= 7315) & (wvl <= 7329), # [OII] 7319.5   
    (wvl >= 7320) & (wvl <= 7335), # [OII] 7330.2   
    (wvl >= 6845) & (wvl <= 6945), # BC03-bug    
    (wvl >= 7165) & (wvl <= 7210), # BC03-bug?    
    (wvl >= 7550) & (wvl <= 7725), # BC03-bug    
    (wvl >= 5880) & (wvl <= 5906), # NaD 5890 & 5896 
    (wvl >= 9059) & (wvl <= 9079), # [SIII] 9069 
    # Telluric features prone to bad subtraction 
    # (https://academic.oup.com/mnras/article/479/2/2443/5026634#sec3-4)
    (wvl >= 6832) & (wvl <= 6920),
    (wvl >= 7556) & (wvl <= 7661),
    (wvl >= 7335) & (wvl <= 7748),
    (wvl >= 8406) & (wvl <= 8416),
    (wvl >= 8736) & (wvl <= 8747),
    (wvl >= 8782) & (wvl <= 8794),
    (wvl >= 8842) & (wvl <= 8853),
    (wvl >= 8874) & (wvl <= 8890)]

  return flags

# -----------------------------------------------------
  
# create the input fits files for FADO
def input_ctx(lam_gal,spectra,err,norm_fact,normFF,pathDIR,bN): 
  
  # λ of gal in steps of 1 Å
  wvl = np.arange(int(lam_gal[0]),int(lam_gal[-1]),1)
  
  flags = flagging(wvl)
  msk_b = np.where(np.any(flags, axis=0), 99, 0)

  for j in range(len(spectra)):
    
    msk = msk_b.copy()
     
    # clean error
    error = np.copy(err[j])
    m = (error >= 0) & (error <= 1000)
    avr = np.mean(error[m])
    error[~m] = avr

    # mask NaNs/Infs
    mS = ~np.logical_or(np.isnan(spectra[j]), np.isinf(spectra[j]))
    mE = ~np.logical_or(np.isnan(error), np.isinf(error))
    if len(lam_gal[mS]) < 0.1*len(lam_gal): continue
    if len(lam_gal[mE]) < 0.1*len(lam_gal): continue
  
    spectrum = spectra[j]*norm_fact[j]

    # interpolate signal to 1 by 1 Å
    cs = CubicSpline(lam_gal[mS], spectrum[mS])
    specI = cs(wvl)
    specI = np.array(specI, dtype=np.float32)
    specI = specI*normFF # to be in units of 1e-17 for Starlight

    # interpolate error to 1 by 1 Å
    cs = CubicSpline(lam_gal[mE], error[mE])
    noisI = cs(wvl)
    noisI = np.array(noisI, dtype=np.float32)
    noisI = noisI*normFF # to be in units of 1e-17
    
    # mask if spectrum or error is negative
    c = (noisI < 0) | (specI < 0) ; msk[c] = 99

    # create input txt files
    data = np.column_stack((wvl, specI, noisI, msk))
    np.savetxt('%s/IN/Bin%04d.cxt'%(pathDIR,bN[j]), data, fmt='%-15.5f')

# -----------------------------------------------------

@ray.remote
def input_files_MC(i,binInfo,lam_gal,BIN_L,BF_MODEL,FNORM_L,normFF,err,pathDIR):
  
  bN   = BIN_L[i] ; f = binInfo.index(bN) # index of the error
    
  # [wvl,obs,err,neb,bst]
  # Truncate the best-fitting model to the observed lambda range
  wvl  = BF_MODEL[i][0]
  iI   = np.argmin(abs(wvl-lam_gal[0]))
  iF   = np.argmin(abs(wvl-lam_gal[-1]))
  
  wvl  = wvl[iI:iF]
  obs  = BF_MODEL[i][1][iI:iF]
  neb  = BF_MODEL[i][3][iI:iF]
  best = BF_MODEL[i][4][iI:iF]
  norm = FNORM_L[i]
  
  # clean error
  wvl  = np.asarray(wvl,  dtype=np.float64)
  obs  = np.asarray(obs,  dtype=np.float64)
  neb  = np.asarray(neb,  dtype=np.float64)
  best = np.asarray(best, dtype=np.float64)
  
  error = np.copy(err[i])
  m = (error >= 0) & (error <= 1000)
  avr = np.mean(error[m])
  error[~m] = avr

  # mask NaNs/Infs
  mE = ~np.logical_or(np.isnan(error), np.isinf(error))
  
  # interpolate error to 1 by 1 Å
  cs = CubicSpline(lam_gal[mE], error[mE])
  noisI = cs(wvl)
  noisI = np.array(noisI, dtype=np.float32)
  noisI = noisI*normFF # to be in units of 1e-17
  flags = flagging(wvl)
  msk = np.where(np.any(flags, axis=0), 99, 0)
  
  # clip outliers
  mask = clip_outliers(obs, best)
  for j in range(N_MC):
    y = obs + np.random.normal(0, 1, len(best)) * np.std(obs[mask] - best[mask])
    specI = y*norm
    c = (noisI < 0) | (specI < 0) ; msk[c] = 99
    data = np.column_stack((wvl, specI, noisI, msk))
    np.savetxt('%s/MC/IN/Bin%04d_MC_%03d.cxt'%(pathDIR,bN,j), data, fmt='%-15.5f')

# -----------------------------------------------------

@ray.remote
def read_FD_output_REM(outF,mc):
    try:
      
      BIN,normF,CHI2,ageM,ageL,ZM,ZL,massLnow,massLevr,SFHs_Mever,SFHs_Mnow,T_ELEC,D_ELEC,SL_EXT,NB_EXT,ST_VEL,ST_SIG,NB_VEL,NB_SIG,TAU,Hb_F,Hb_EW,OIII_4959_F,OIII_4959_EW,OIII_5007_F,OIII_5007_EW,NII_6548_F,NII_6548_EW,Ha_F,Ha_EW,Ha_VEL,Ha_SIG,NII_6584_F,NII_6584_EW,SII_6717_F,SII_6717_EW,SII_6731_F,SII_6731_EW,spec = ray.get(read_FD_output.remote(outF,mc))
    
      # Return extracted data
      if mc:
        return {
            'AGE_M': ageM.nominal_value,
            'AGE_L': ageL.nominal_value,
            'MET_M': ZM.nominal_value,
            'MET_L': ZL.nominal_value,
            'LMASS_P': massLnow.nominal_value,
            'LMASS_E': massLevr.nominal_value,
            'N2Ha': np.log10(NII_6584_F.nominal_value / Ha_F.nominal_value),
            'O3Hb': np.log10(OIII_5007_F.nominal_value / Hb_F.nominal_value),
            'SFHsMe': SFHs_Mever,
            'SFHsMn': SFHs_Mnow}
      
      else:
        return {
          'BIN': BIN,
          'CHI2': CHI2,
          'AGE_M': ageM,
          'AGE_L': ageL,
          'MET_M': ZM,
          'MET_L': ZL,
          'T_ELEC': T_ELEC,
          'D_ELEC': D_ELEC,
          'LMASS_P': massLnow,
          'LMASS_E': massLevr,
          'AV': SL_EXT,
          'NEBV': NB_EXT,
          'ST_VEL': ST_VEL,
          'ST_SIG': ST_SIG,
          'NB_VEL': NB_VEL,
          'NB_SIG': NB_SIG,
          'TAU': TAU,
          # Extinction corrected emission line fluxes
          'HaF': Ha_F,
          'HaEW': Ha_EW,
          'Ha_VEL': Ha_VEL,
          'Ha_SIG': Ha_SIG,
          'HbF': Hb_F,
          'HbEW': Hb_EW,
          'OIIIaF': OIII_4959_F,
          'OIIIaEW': OIII_4959_EW,
          'OIIIbF': OIII_5007_F,
          'OIIIbEW': OIII_5007_EW,
          'NIIaF': NII_6548_F,
          'NIIaEW': NII_6548_EW,
          'NIIbF': NII_6584_F,
          'NIIbEW': NII_6584_EW,
          'SIIaF': SII_6717_F,
          'SIIaEW': SII_6717_EW,
          'SIIbF': SII_6731_F,
          'SIIbEW': SII_6731_EW,
          'BF_MODEL': spec,
          'FNORM': normF,
          'SFHsMe': SFHs_Mever,
          'SFHsMn': SFHs_Mnow}
  
    except:
      pass

#----------------------------------------------------

@ray.remote
def read_FD_output(pathFile,mc):

  # --------------------------
  # Open output files
  # --------------------------

  fado = fits.open('%s.FADO.fits'%pathFile)
  oneD = fado[0]
  el   = fado[1]
  de   = fado[2]
  st   = fado[3]
    
  # --------------------------
  Av = st.header['GEXTINCT'] # intrinsic extinction from FADO
  # --------------------------    
  
  # open output files
  wvl = np.arange(oneD.header['NGLAMBDA']) + oneD.header['CRVAL1']
  obs = oneD.data[0]
  err = oneD.data[1]
  stl = oneD.data[7]
  neb = oneD.data[8]
  bst = stl + neb
  
  # --------------------------
  # extinction % https://iopscience.iop.org/article/10.1088/0004-637X/793/2/127/meta
  normF = el.header['GALSNORM'] 
  HaHb = el.data[0][845]/el.data[0][488]
  # --------------------------
  if mc:
    match = re.search(r'/OUT/Bin(\d+)\_MC_(\d+)\.cxt', pathFile)
  else:
    match = re.search(r'/OUT/Bin(\d+)\.cxt', pathFile)
    
  BIN   = int(match.group(1))
  CHI2  = oneD.header['CHI2_RED']
  
  ageM = ufloat(st.header['BST_MAGE']/1e9, abs(st.header['DEV_MAGE']/1e9))  
  ageL = ufloat(st.header['BST_LAGE']/1e9, abs(st.header['DEV_LAGE']/1e9))
  
  ZM = ufloat(st.header['BST_MMET'], abs(st.header['DEV_MMET']))
  ZL = ufloat(st.header['BST_LMET'], abs(st.header['DEV_LMET']))
  
  massLnow = ufloat(st.header['LOGMCBST'], abs(st.header['LOGMCDEV']))
  massLevr = ufloat(st.header['LOGMEBST'], abs(st.header['LOGMEDEV']))

  T_ELEC = st.header['TELECTRO'] ; D_ELEC = st.header['DELECTRO']
  
  SL_EXT = ufloat(st.header['GEXTINCT'], abs(st.header['GEXTBDEV']))  
  NB_EXT = ufloat(st.header['GNEBULAR'], abs(st.header['GNEBBDEV']))
  
  ST_VEL = ufloat(st.header['V0SYSGAL'], abs(st.header['V0SYSDEV']))  
  ST_SIG = ufloat(st.header['VDSYSGAL'], abs(st.header['VDSYSDEV']))
  
  NB_VEL = ufloat(st.header['V0NEBGAL'], abs(st.header['V0NEBDEV']))
  NB_SIG = ufloat(st.header['VDNEBGAL'], abs(st.header['VDNEBDEV']))
  
  TAU = ufloat(st.header['TAU__BST'], abs(st.header['TAU__DEV']))
  
  try:
    flx = factE(el.data[0][483]/10000,el.data[0][488]*normF,HaHb)
    Hb_F = ufloat(flx, abs(el.data[1][488])*normF)
    if unumpy.nominal_values(Hb_F) < 0: Hb_F = ufloat(np.nan,np.nan)
    Hb_EW = ufloat(el.data[0][489], abs(el.data[1][489]))
    if unumpy.nominal_values(Hb_EW) < 0: Hb_EW = ufloat(np.nan,np.nan)
  except:
    Hb_F = ufloat(np.nan,np.nan)
    Hb_EW = ufloat(np.nan,np.nan)
    
  try:
    flx = factE(el.data[0][525]/10000,el.data[0][530]*normF,HaHb)
    OIII_4959_F = ufloat(flx, abs(el.data[1][530])*normF)
    if unumpy.nominal_values(OIII_4959_F) < 0: OIII_4959_F = ufloat(np.nan,np.nan)
    OIII_4959_EW = ufloat(el.data[0][531], abs(el.data[1][531]))
    if unumpy.nominal_values(OIII_4959_EW) < 0: OIII_4959_EW = ufloat(np.nan,np.nan)
  except:
    OIII_4959_F = ufloat(np.nan,np.nan)
    OIII_4959_EW = ufloat(np.nan,np.nan)
  
  try:
    flx = factE(el.data[0][567]/10000,el.data[0][572]*normF,HaHb)
    OIII_5007_F = ufloat(flx, abs(el.data[1][572])*normF)
    if unumpy.nominal_values(OIII_5007_F) < 0: OIII_5007_F = ufloat(np.nan,np.nan)
    OIII_5007_EW = ufloat(el.data[0][573], abs(el.data[1][573]))
    if unumpy.nominal_values(OIII_5007_EW) < 0: OIII_5007_EW = ufloat(np.nan,np.nan)
  except:
    OIII_5007_F = ufloat(np.nan,np.nan)
    OIII_5007_EW = ufloat(np.nan,np.nan)
  
  try:
    flx = factE(el.data[0][819]/10000,el.data[0][824]*normF,HaHb)
    NII_6548_F = ufloat(flx, abs(el.data[1][824])*normF)
    if unumpy.nominal_values(NII_6548_F) < 0: NII_6548_F = ufloat(np.nan,np.nan)
    NII_6548_EW = ufloat(el.data[0][825], abs(el.data[1][825]))
    if unumpy.nominal_values(NII_6548_EW) < 0: NII_6548_EW = ufloat(np.nan,np.nan)
  except:
    NII_6548_F = ufloat(np.nan,np.nan)
    NII_6548_EW = ufloat(np.nan,np.nan)
  
  try:
    flx = factE(el.data[0][840]/10000,el.data[0][845]*normF,HaHb)
    Ha_F = ufloat(flx, abs(el.data[1][845])*normF)
    if unumpy.nominal_values(Ha_F) < 0: Ha_F = ufloat(np.nan,np.nan)
    Ha_EW = ufloat(el.data[0][846], abs(el.data[1][846]))
    if unumpy.nominal_values(Ha_EW) < 0: Ha_EW = ufloat(np.nan,np.nan)
  except:
    Ha_F = ufloat(np.nan,np.nan)
    Ha_EW = ufloat(np.nan,np.nan)
  
  Ha_VEL = ufloat(el.data[0][843], abs(el.data[1][843]))
  if unumpy.nominal_values(Ha_VEL) < 0: Ha_VEL = ufloat(np.nan,np.nan)
  Ha_SIG = ufloat(el.data[0][842], abs(el.data[1][842]))
  if unumpy.nominal_values(Ha_SIG) < 0: Ha_SIG = ufloat(np.nan,np.nan)
  
  try:
    flx = factE(el.data[0][861]/10000,el.data[0][866]*normF,HaHb)
    NII_6584_F = ufloat(flx, abs(el.data[1][866])*normF)
    if unumpy.nominal_values(NII_6584_F) < 0: NII_6584_F = ufloat(np.nan,np.nan)
    NII_6584_EW = ufloat(el.data[0][867], abs(el.data[1][867]))
    if unumpy.nominal_values(NII_6584_EW) < 0: NII_6584_EW = ufloat(np.nan,np.nan)
  except:
    NII_6584_F = ufloat(np.nan,np.nan)
    NII_6584_EW = ufloat(np.nan,np.nan)

  try:
    flx = factE(el.data[0][903]/10000,el.data[0][908]*normF,HaHb)
    SII_6717_F = ufloat(flx, abs(el.data[1][908])*normF)
    if unumpy.nominal_values(SII_6717_F) < 0: SII_6717_F = ufloat(np.nan,np.nan)
    SII_6717_EW = ufloat(el.data[0][909], abs(el.data[1][909]))
    if unumpy.nominal_values(SII_6717_EW) < 0: SII_6717_EW = ufloat(np.nan,np.nan)
  except:
    SII_6717_F = ufloat(np.nan,np.nan)
    SII_6717_EW = ufloat(np.nan,np.nan)
  
  try:
    flx = factE(el.data[0][924]/10000,el.data[0][929]*normF,HaHb)
    SII_6731_F = ufloat(flx, abs(el.data[1][929])*normF)
    if unumpy.nominal_values(SII_6731_F) < 0: SII_6731_F = ufloat(np.nan,np.nan)
    SII_6731_EW = ufloat(el.data[0][930], abs(el.data[1][930]))
    if unumpy.nominal_values(SII_6731_EW) < 0: SII_6731_EW = ufloat(np.nan,np.nan)
  except:
    SII_6731_F = ufloat(np.nan,np.nan)
    SII_6731_EW = ufloat(np.nan,np.nan)
  
  # SFHs ------------------------------------------
  num_base   = de.header['NUM_BASE']
  SFHs_Mnow  = de.data[4,0:num_base]/sum(de.data[4,0:num_base])*10**(massLnow.n)
  SFHs_Mever = de.data[8,0:num_base]/sum(de.data[8,0:num_base])*10**(massLevr.n)

  fado.close()
  
  # ------------------------------------------------------------------
  
  return BIN,normF,CHI2,ageM,ageL,ZM,ZL,massLnow,massLevr,SFHs_Mever,SFHs_Mnow,T_ELEC,D_ELEC,SL_EXT,NB_EXT,ST_VEL,ST_SIG,NB_VEL,NB_SIG,TAU,Hb_F,Hb_EW,OIII_4959_F,OIII_4959_EW,OIII_5007_F,OIII_5007_EW,NII_6548_F,NII_6548_EW,Ha_F,Ha_EW,Ha_VEL,Ha_SIG,NII_6584_F,NII_6584_EW,SII_6717_F,SII_6717_EW,SII_6731_F,SII_6731_EW,[wvl,obs,err,neb,bst]
  
#----------------------------------------------------

def clip_outliers(spectrum,bestfit,threshold=3):
  
  resid = bestfit - spectrum
  err = robust_sigma(resid, zero=1)
  goodpixels = np.abs(resid) < threshold * err
  
  return goodpixels

#--------------------------------
  
# Kewley+01
def Kew01(x): return (0.61/(x - 0.47)) + 1.19
# Schawinski+07
def Sch07(x): return 1.05*x + 0.45 
# Kauffmann+03
def Kau03(x): return 0.61/(x - 0.05) + 1.3

def scientific(x, pos):
  if np.isnan(x) or x == 0: 
    return ''
  exponent = int(np.floor(np.log10(abs(x))))
  return r'${0:.0f} . 10^{{{1}}}$'.format(x / 10**exponent, exponent)

# ----------------------------------------------------------------

@ray.remote
def workerFADO(pathDIR,dist,temp,lib,bN,mF,res):
  
  for j in range(N_MC):
    inFile = 'Bin%04d_MC_%03d.cxt'%(bN,j)
    ray.get(run_FADO.remote(pathDIR,dist,temp,lib,inFile,mF,True,res))
    
# -----------------------------------------------------

@ray.remote
def run_FADO(pathDIR,dist,temp,lib,inFile,mF,mc,res):
  
  if mc:
    pathIN  = pathDIR + '/MC/IN/' ; pathOUT = pathDIR + '/MC/OUT/'
  else:
    pathIN  = pathDIR + '/IN/' ; pathOUT = pathDIR + '/OUT/'
  
  os.system('%s%s -i %s%s -b %s%s/%s -u 1.e-17 -z -500.0 500.0 20.0 -s %i %i 1 -o %s -e CCMR -d %f -r %f'%(BIN_DIR,FADO_version,pathIN,inFile,SSP_DIR,temp,lib,mF,SPEC_LIM,pathOUT,dist,res))
  
  fL = ['%s.FADO_1D.fits'%inFile, '%s.FADO_EL.fits'%inFile, '%s.FADO_DE.fits'%inFile, '%s.FADO_ST.fits'%inFile]
  cF = fits.HDUList()
  try:
    # concatenate FADO's output files into 1 fits file
    for f in fL:
      with fits.open(pathOUT + f) as h:
        cF.append(fits.PrimaryHDU(data=h[0].data, header=h[0].header))
      h.close()
      os.system('rm %s%s'%(pathOUT,f))
    cF.writeto('%s%s.FADO.fits'%(pathOUT,inFile), overwrite=True)
  except Exception as e:
    print(e)
    pass # Some FADO fits files are problematic
 
  if mc:
    os.system('rm %s%s.FADO.ascii %s%s.FADO.coL.eps 2>&1'%(pathOUT,inFile,pathOUT,inFile))
  else:
    os.system('mv %s%s.FADO.ascii %s%s.FADO.BN 2>&1'%(pathOUT,inFile,pathOUT,inFile))

# -----------------------------------------------------

def fmt_no_trailing_zeros(x, pos):
  s = f'{x:.2f}'
  s = s.rstrip('0').rstrip('.')
  return s

def imshow_with_cbar(ax, img, title, vmin=None, vmax=None, perc=(5, 95), cmap=None, extent=None, Z=None, cbar_lab=None):
  
  if Z == 1: img = img / Zs

  if vmin is None:
    vmin = np.nanpercentile(img, perc[0])
  if vmax is None:
    vmax = np.nanpercentile(img, perc[1])

  ax.set_title(title, pad=10)
  im = ax.imshow(img, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

  div = make_axes_locatable(ax)
  cax = div.append_axes('right', size='4%', pad=0.05)
  cb = ax.figure.colorbar(im, cax=cax)

  if cbar_lab:
    cb.set_label(cbar_lab, rotation=-90, labelpad=15)

  lo, hi = im.get_clim()
  if np.isfinite(lo) and np.isfinite(hi) and hi != 0 and (abs(hi) < 1e-2 or abs(hi - lo) < 1e-6):
    cb.formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    cb.formatter.set_powerlimits((0, 0))
  else:
    cb.formatter = FuncFormatter(fmt_no_trailing_zeros)

  cb.update_ticks()
  cb.ax.tick_params(labelsize=12)

  return im
# -----------------------------------------------------

def mean_std(A): return np.nanmedian(A), np.nanstd(A)

def spec_synth_FADO(Galaxy,dist,pathDIR,temp,lib,xL,yL):
  
  global nbins,SSP_DIR,BIN_DIR,N_MC,SPEC_LIM
  from run_GLANCE import get_contours
  
  try:
    GAL_INFO, SSP_DIR, STT_DIR, BIN_DIR, BAYES_DIR, CORES, NEB_CONT, N_MCS, SPEC_LIM, WR_PPXF, TEMPS_PPXF, LINE_STP, NMOM, GST_STP, CHI2_TYPE, TODO_PDYN, HST_FILT, MGE_M, SINF, TEMP_BAYES, REGUL, WR_BAYES, OCUT, V_SIGMA, ERR_DSBP, TEMPS_FD, LIBS_FD, TEMPS_SL, LIBS_SL, SL_CONF, RY_SSC, RY_TCS, TEMPS_SINF, NMOM_SINF = read_config()
  except Exception as e:
    print(f' × Problem in read_config(): {e} ×')
    print(' × Check your config file ×')
    sys.exit(1)
  
  if CORES == 0: CORES = os.cpu_count()
  _, N_MC = N_MCS
  mc = True if N_MC > 0 else False
  imgCont,levels = get_contours(Galaxy)

  # -------------------------------------
  # Open info in bins
  binInfo, spectra, err, norm_fact = ([],[],[],[])
  content = pd.read_pickle('%s_Bins.pkl'%Galaxy)
  for line in content[:-1]:
    binInfo.append(line[0])
    norm_fact.append(line[1])  # median flux of logspec (normalizing factor)
    spectra.append(line[2])    # log rebin (!! normalized !!) spectra
    err.append(line[3])        # log rebin noise
  lam_gal,_ = content[-1]      # log rebin wvl
  lam_gal = np.exp(lam_gal)
  
  df = pd.read_csv('%s_VorBin.csv'%Galaxy)
  df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
  # -------------------------------------

  # Find the center of the galaxy
  maps = fits.open('%s_maps.fits'%Galaxy) ; maps = maps[0].data
  STRcont = maps[1]
  MSK = fits.open(Galaxy + '_mask.fits')
  MSK = MSK[0].data ; MSK = np.array(MSK) ; MSK = np.logical_not(MSK).astype(int)
  ctr = np.unravel_index(np.nanargmax(STRcont*MSK), np.shape(STRcont*MSK))
  yC, xC = ctr
  
  # Sanity check
  #plt.imshow(np.log10(STRcont)*MSK, origin='lower')
  #plt.plot(xC, yC, 'rx', ms=8, mew=2) ; plt.show()
  
  # -------------------------------------

  # Lower λ limit, normalization factor (to have flux in units of 1e-17), spectral resolution
  mF = 4700 ; normFF = 0.001 ; res = 2.35
  # Be sure the fitting range doesnt exceed the observed wvl
  if mF < lam_gal[0]  : mF = int(lam_gal[0])  + 10
  if SPEC_LIM > lam_gal[-1] : SPEC_LIM = int(lam_gal[-1]) - 10
    
  nbins = len(binInfo)
  R_kpc = 0.2*(np.pi/180/3600)*(dist*1000) # convert MUSE's pixel scale in kpc (1" = pi/180/3600)
  
  # create ctx files 
  in_files = glob.glob('%s/IN/*'%pathDIR)
  if len(in_files) < nbins: input_ctx(lam_gal,spectra,err,norm_fact,normFF,pathDIR,binInfo)

  # -----------------------------
  # Spectral Synthesis
  # -----------------------------
  
  if not glob.glob('FADO.config'): os.system('cp %sFADO.config .'%BIN_DIR)
  with open('FADO.config', 'r+') as f:
    content = f.read()
    content = content.replace('0             #    [IsFlagOn]    #   Integer #         ---- #    0020   #','1             #    [IsFlagOn]    #   Integer #         ---- #    0020   #')
    f.seek(0)
    f.write(content)
    f.truncate()
    
  print('*                               Running FADO in ||                                *')
  print('-----------------------------------------------------------------------------------')
  print(' • Spectral Library : %s (%s)'%(temp,lib))
  print(' • Fitting range    : %.2f to %.2f Å'%(mF,SPEC_LIM))
  #print('-----------------------------------------------------------------------------------')
  
  # Initialise ray
  if os.path.exists('FADO.config.lock'): os.system('rm FADO.config.lock')
  if nbins % CORES != 0: CORES = min(nbins, CORES)
  # NOTE: For debuging, remove logging_level & log_to_driver
  ray.init(num_cpus=CORES,logging_level=logging.ERROR,log_to_driver=False)
  
  tasks = []
  for bN in binInfo:
    inFile = 'Bin%04d.cxt'%bN
    
    if 'SL' in FADO_version:
      isSol = os.path.exists(pathDIR + '/OUT/' + inFile + '.FADO.BN') and os.path.getsize(pathDIR + '/OUT/' + inFile + '.FADO.BN') > 0
    else:  
      isSol = os.path.exists(pathDIR + '/OUT/' + inFile + '.FADO.fits') and os.path.getsize(pathDIR + '/OUT/' + inFile + '.FADO.fits') > 0

    if isSol == False:
      p = run_FADO.remote(pathDIR,dist,temp,lib,inFile,mF,False,res)
      tasks.append(p)
    
  if len(tasks) == 0: 
    print(' ✓ All bins are already processed!')
    print('-----------------------------------------------------------------------------------')
    
    # Read evr from file
    try:
      with open('%s_FADO.pkl'%Galaxy, 'rb') as f: evr = pickle.load(f)
    except:
      with open(pathDIR + '/%s_FADO_%s.pkl'%(Galaxy,temp), 'rb') as f: evr = pickle.load(f)

    keys = ['BIN', 'FNORM', 'CHI2', 'AGE_M', 'AGE_L', 'MET_M', 'MET_L', 'LMASS_P', 'LMASS_E', 'SFHsMe', 'SFHsMn', 'T_ELEC', 'D_ELEC', 'AV', 'NEBV', 'ST_VEL', 'ST_SIG', 'NB_VEL', 'NB_SIG', 'TAU', 'HbF', 'HbEW', 'OIIIaF', 'OIIIaEW', 'OIIIbF', 'OIIIbEW', 'NIIaF', 'NIIaEW', 'HaF', 'HaEW', 'Ha_VEL', 'Ha_SIG', 'NIIbF', 'NIIbEW', 'SIIaF', 'SIIaEW', 'SIIbF', 'SIIbEW', 'BF_MODEL']

    evr_dict = {key: np.array([row[i] for row in evr], dtype=object) for i, key in enumerate(keys)}

    BIN_L, FNORM_L, CHI2_L, AGE_M, AGE_L, MET_M, MET_L, LMASS_P, LMASS_E, SFHsMe, SFHsMn, T_ELEC, D_ELEC, AV_L, NEBV_L, ST_VEL, ST_SIG, NB_VEL, NB_SIG, TAU, HbF_L, HbEW_L, OIIIaF_L, OIIIaEW_L, OIIIbF_L, OIIIbEW_L, NIIaF_L, NIIaEW_L, HaF_L, HaEW_L, Ha_VEL, Ha_SIG, NIIbF_L, NIIbEW_L, SIIaF_L, SIIaEW_L, SIIbF_L, SIIbEW_L, BF_MODEL = evr_dict.values()
    
  else:
    
    ray.get(tasks) # Run FADO in parallel

    # --------------------------------------------
    # read output from FADO in parallel
    # --------------------------------------------

    evr = []
    BIN_L, AGE_M, AGE_L, MET_M, MET_L, LMASS_P, LMASS_E, AV_L, NEBV_L = ([],[],[],[],[],[],[],[],[])
    HbF_L, HbEW_L, OIIIaF_L, OIIIaEW_L, OIIIbF_L, OIIIbEW_L, NIIaF_L, NIIaEW_L, HaF_L, HaEW_L, NIIbF_L, NIIbEW_L, SIIaF_L, SIIaEW_L, SIIbF_L, SIIbEW_L, BF_MODEL, FNORM_L = ([],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])
    
    inFiles  = np.sort(glob.glob(pathDIR + '/OUT/*.FADO.fits'))
    outFiles = [s.replace('.FADO.fits', '') for s in inFiles]
    tasks = [read_FD_output_REM.remote(outF,False) for outF in outFiles]
        
    results = ray.get(tasks)
    for result in results:
      if result:
        
        BIN_L.append(result['BIN'])
        AGE_M.append(result['AGE_M'])     ; AGE_L.append(result['AGE_L'])
        MET_M.append(result['MET_M'])     ; MET_L.append(result['MET_L'])
        LMASS_P.append(result['LMASS_P']) ; LMASS_E.append(result['LMASS_E'])
        # Extinctions
        AV_L.append(result['AV']) ; NEBV_L.append(result['NEBV'])
        # Extinction corrected emission line fluxes
        HaF_L.append(result['HaF'])       ; HaEW_L.append(result['HaEW'])
        HbF_L.append(result['HbF'])       ; HbEW_L.append(result['HbEW'])
        OIIIaF_L.append(result['OIIIaF']) ; OIIIaEW_L.append(result['OIIIaEW'])
        OIIIbF_L.append(result['OIIIbF']) ; OIIIbEW_L.append(result['OIIIbEW'])
        NIIaF_L.append(result['NIIaF'])   ; NIIaEW_L.append(result['NIIaEW'])
        NIIbF_L.append(result['NIIbF'])   ; NIIbEW_L.append(result['NIIbEW'])
        SIIaF_L.append(result['SIIaF'])   ; SIIaEW_L.append(result['SIIaEW'])
        SIIbF_L.append(result['SIIbF'])   ; SIIbEW_L.append(result['SIIbEW'])
        BF_MODEL.append(result['BF_MODEL']) ; FNORM_L.append(result['FNORM'])
        # NOTE: spec -> [wvl,obs,err,neb,bst] -> the error from FADO is 0 everywhere, dont use it after in the MC
        
        evr.append([result['BIN'],result['FNORM'],result['CHI2'],result['AGE_M'],result['AGE_L'],result['MET_M'],result['MET_L'],result['LMASS_P'],result['LMASS_E'],result['SFHsMe'],result['SFHsMn'],result['T_ELEC'],result['D_ELEC'],result['AV'],result['NEBV'],result['ST_VEL'],result['ST_SIG'],result['NB_VEL'],result['NB_SIG'],result['TAU'],result['HbF'],result['HbEW'],result['OIIIaF'],result['OIIIaEW'],result['OIIIbF'],result['OIIIbEW'],result['NIIaF'],result['NIIaEW'],result['HaF'],result['HaEW'],result['Ha_VEL'],result['Ha_SIG'],result['NIIbF'],result['NIIbEW'],result['SIIaF'],result['SIIaEW'],result['SIIbF'],result['SIIbEW'],result['BF_MODEL']])
        
    with open('%s_FADO.pkl'%Galaxy,'wb') as f: pickle.dump(evr, f)

  # --------------------------------------------
    
  # Create final maps :
  header= fits.Header()
  header['LAYER00'] = 'tM (Gyr)' ; header['LAYER01'] = 'tL (Gyr)'
  header['LAYER02'] = 'zM'       ; header['LAYER03'] = 'ZL'
  header['LAYER04'] = 'M*now (log)' 
  header['LAYER05'] = 'M*ever (log)'
  header['LAYER06'] = 'Stellar Surf Density'
  header['LAYER07'] = 'HbF'      ; header['LAYER08'] = 'HbEW'
  header['LAYER09'] = 'OIIIaF'   ; header['LAYER10'] = 'OIIIaEW'
  header['LAYER11'] = 'OIIIbF'   ; header['LAYER12'] = 'OIIIbEW'
  header['LAYER13'] = 'NIIaF'    ; header['LAYER14'] = 'NIIaEW'
  header['LAYER15'] = 'HaF'      ; header['LAYER16'] = 'HaEW'
  header['LAYER17'] = 'NIIbF'    ; header['LAYER18'] = 'NIIbEW'
  header['LAYER19'] = 'SIIaF'    ; header['LAYER20'] = 'SIIaEW'
  header['LAYER21'] = 'SIIbF'    ; header['LAYER22'] = 'SIIbEW'
  header['LAYER23'] = 'SText'    ; header['LAYER24'] = 'NBext'
  header['LAYER25'] = 'Kennicutt relation (Ha luminosity)'
  header['COMMENT'] = 'Fitting range: %i - %i'%(mF,SPEC_LIM)
  header['COMMENT'] = 'Stellar library: %s'%temp
  header['COMMENT'] = 'ST lib file: %s'%lib
  header['COMMENT'] = 'Flux vals - use 10**vals * 1e-17 to convert back to units of flux'
  header['COMMENT'] = 'Ages in Gyr'
  header['COMMENT'] = 'Masses in log'

  R, N2Ha, O3Hb = ([],[],[])
  imgs = np.full((26,yL,xL), np.nan)
    
  for j in range(len(BIN_L)):
    bn = BIN_L[j]
    tM = AGE_M[j]   ; tL = AGE_L[j]
    ZM = MET_M[j]   ; ZL = MET_L[j]
    Mn = LMASS_P[j] ; Me = LMASS_E[j]
    HbF = HbF_L[j] ; HbEW = HbEW_L[j]
    OIIIaF = OIIIaF_L[j] ; OIIIaEW = OIIIaEW_L[j]
    OIIIbF = OIIIbF_L[j] ; OIIIbEW = OIIIbEW_L[j]
    NIIaF = NIIaF_L[j] ; NIIaEW = NIIaEW_L[j]
    HaF = HaF_L[j] ; HaEW = HaEW_L[j]
    NIIbF = NIIbF_L[j] ; NIIbEW = NIIbEW_L[j]
    SIIaF = SIIaF_L[j] ; SIIaEW = SIIaEW_L[j]
    SIIbF = SIIbF_L[j] ; SIIbEW = SIIbEW_L[j]
    SText = AV_L[j]    ; NEBext = NEBV_L[j]
    try:
      N2Ha.append(np.log10(NIIbF.nominal_value/HaF.nominal_value))
    except:
      N2Ha.append(np.nan)
    try:  
      O3Hb.append(np.log10(OIIIbF.nominal_value/HbF.nominal_value))
    except:
      O3Hb.append(np.nan)
    
    f = df.index[df['BinVor'] == bn].tolist()
    SSD = np.log10(10**(Mn.nominal_value)/len(f)/(R_kpc**2))
    rad = []
    for i in f:
      xI = df['x'][i] ; yI = df['y'][i]
      imgs[0,yI,xI]  = tM.nominal_value     ; imgs[1,yI,xI]  = tL.nominal_value
      imgs[2,yI,xI]  = ZM.nominal_value     ; imgs[3,yI,xI]  = ZL.nominal_value
      imgs[4,yI,xI]  = Mn.nominal_value - np.log10(len(f))
      imgs[5,yI,xI]  = Me.nominal_value - np.log10(len(f))
      imgs[6,yI,xI]  = SSD
      imgs[7,yI,xI]  = np.log10(HbF.nominal_value/len(f))    ; imgs[8,yI,xI]  = HbEW.nominal_value
      imgs[9,yI,xI]  = np.log10(OIIIaF.nominal_value/len(f)) ; imgs[10,yI,xI] = OIIIaEW.nominal_value
      imgs[11,yI,xI] = np.log10(OIIIbF.nominal_value/len(f)) ; imgs[12,yI,xI] = OIIIbEW.nominal_value
      imgs[13,yI,xI] = np.log10(NIIaF.nominal_value/len(f))  ; imgs[14,yI,xI] = NIIaEW.nominal_value
      imgs[15,yI,xI] = np.log10(HaF.nominal_value/len(f))    ; imgs[16,yI,xI] = HaEW.nominal_value
      imgs[17,yI,xI] = np.log10(NIIbF.nominal_value/len(f))  ; imgs[18,yI,xI] = NIIbEW.nominal_value
      imgs[19,yI,xI] = np.log10(SIIaF.nominal_value/len(f))  ; imgs[20,yI,xI] = SIIaEW.nominal_value
      imgs[21,yI,xI] = np.log10(SIIbF.nominal_value/len(f))  ; imgs[22,yI,xI] = SIIbEW.nominal_value
      imgs[23,yI,xI] = SText.nominal_value         ; imgs[24,yI,xI] = NEBext.nominal_value
      # Kennicutt, Tamblyn and Congdon (1994)
      imgs[25,yI,xI] = 7.9e-42 * 4*np.pi*(dist*3.086e24)**2 * HaF.nominal_value/len(f) * 1e-17
      rad.append(np.sqrt((xI - xC)**2 + (yI - yC)**2))

    R.append(np.mean(rad))
  
  # --------------------------------
  imgsN = np.transpose(imgs, (0, 2, 1))
  MSK = fits.open(Galaxy + '_mask.fits')
  MSK = MSK[0].data ; MSK = np.array(MSK) ; MSK = np.logical_not(MSK).astype(int)

  imgs = []
  for im in imgsN: 
    im = im*MSK
    im[im == 0] = np.nan
    imgs.append(im)
    
  # Fluxes in units of 1e-17
  hdu = fits.PrimaryHDU(data=imgs, header=header)
  hdu.writeto('%s_FADO_maps.fits'%Galaxy, overwrite=True)
  
  ny, nx = imgs[0].shape
  extent = [(-nx/2)*psO, (nx/2)*psO, (-ny/2)*psO, (ny/2)*psO]

  titles = ['Mass-weighted age (Gyr)', 'Lum-weighted age (Gyr)', r'Mass-weighted Z (Z$_\odot$)', r'Lum-weighted Z (Z$_\odot$)', r'Present-day M$\star$ (log M$_\odot$)', r'Ever-formed M$\star$ (log M$_\odot$)', r'H$\beta$ flux', r'EW(H$\beta$)', r'[OIII]a flux', r'EW([OIII]a)', r'[OIII]b flux', r'EW([OIII]b)']
  idxs = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        
  fig, ax = plt.subplots(6, 2, figsize=(7, 19))
  
  for ax_i, i, title in zip(ax.ravel(), idxs, titles):
    Z = 1 if '(Z$_\\odot$)' in title else None
    imshow_with_cbar(ax_i, imgs[i], title, perc=(perc1, perc2), cmap=cmap, extent=extent, Z=Z)
    ax_i.contour(imgCont, levels=levels, colors='#5b5f61', linewidths=1, origin='lower', extent=extent)

  [ax.set(xlabel='″', ylabel='″') for ax in ax.ravel()]
  
  if R_bulge != None:
    cO_arc = ((cO[0] - nx/2) * psO, (cO[1] - ny/2) * psO)
    for ax in fig.axes: ax.add_patch(Circle(cO_arc, R_bulge, color='#899499', fill=False, linewidth=1))
    
  plt.subplots_adjust(bottom=0.03, left=0.03, right=1, top=0.97, hspace=0.40, wspace=0.02)
  plt.savefig('%s_FADO_maps_01.png'%Galaxy) ; plt.close()

  titles = [r'H$\alpha$ flux', r'EW(H$\alpha$)', r'[NII]a flux', r'EW([NII]a)', r'[NII]b flux', r'EW([NII]b)', r'[SII]a flux', r'EW([SII]a)', r'[SII]b flux', r'EW([SII]b)', 'Nebular Extinction', 'Stellar Extinction']
  idxs = [15, 16, 13, 14, 17, 18, 19, 20, 21, 22, 24, 23]
  
  fig, ax = plt.subplots(6, 2, figsize=(7, 19))
  
  for ax_i, i, title in zip(ax.ravel(), idxs, titles):
    Z = 1 if '(Z$_\\odot$)' in title else None
    imshow_with_cbar(ax_i, imgs[i], title, perc=(perc1, perc2), cmap=cmap, extent=extent, Z=Z)
    ax_i.contour(imgCont, levels=levels, colors='#5b5f61', linewidths=1, origin='lower', extent=extent)
    
  [ax.set(xlabel='″', ylabel='″') for ax in ax.ravel()]
  
  if R_bulge != None:
    cO_arc = ((cO[0] - nx/2) * psO, (cO[1] - ny/2) * psO)
    for ax in fig.axes: ax.add_patch(Circle(cO_arc, R_bulge, color='#899499', fill=False, linewidth=1))
  
  plt.subplots_adjust(bottom=0.03, left=0.03, right=1, top=0.97, hspace=0.40, wspace=0.02)
  plt.savefig('%s_FADO_maps_02.png'%Galaxy) ; plt.close()
  imgsF = np.copy(imgs)
  
  # Merge the two pngs ----------------------------------------------------------
  img1 = Image.open('%s_FADO_maps_01.png'%Galaxy) ; img2 = Image.open('%s_FADO_maps_02.png'%Galaxy)
  w1, h1 = img1.size ; w2, h2 = img2.size

  nw = w1 + w2
  nh = max(h1, h2)
  ni = Image.new('RGB', (nw, nh), (255, 255, 255))

  ni.paste(img1, (0, 0)) ; ni.paste(img2, (w1, 0))
  ni.save('%s_FADO_maps.png'%Galaxy)
  
  os.remove('%s_FADO_maps_01.png'%Galaxy) ; os.remove('%s_FADO_maps_02.png'%Galaxy)
  
  # delete LOG files
  logF = glob.glob(pathDIR + '/OUT/*.LOG') ; [os.remove(f) for f in logF if os.path.exists(f)]

  # ----------------------------------------
  # Estimating errors with Monte Carlo 
  # ----------------------------------------

  if mc:
    
    print('*               Running FADO in || for error statistics with MC sim.              *')
    if len(TEMPS_FD) > 1 and TEMPS_FD[0] == temp:
      print('-----------------------------------------------------------------------------------')

    # Create input files for MC
    # ----------------------------------
    nbins = len(BIN_L) ; tasks = []
    if nbins > 5500 and N_MC > 50:
      print('-----------------------------------------------------------------------------------')
      print('* × ATTENTION ×   The number of bins is too high, decreasing N_MC to 50           *')
      N_MC = 50
      
    for j in range(len(BIN_L)):
      in_files = glob.glob('%s/MC/IN/Bin%04d*MC*'%(pathDIR,j))
      if len(in_files) < N_MC: 
        p = input_files_MC.remote(j,binInfo,lam_gal,BIN_L,BF_MODEL,FNORM_L,normFF,err,pathDIR)
        tasks.append(p)
    ray.get(tasks)
    
    # Run the MC simulations
    # ----------------------------------
    tasks = []
    for bN in BIN_L:
      out_files = glob.glob('%s/MC/OUT/Bin%04d_MC*FADO.fits'%(pathDIR,bN))
      if len(out_files) < N_MC:
        p = workerFADO.remote(pathDIR,dist,temp,lib,bN,mF,res)
        tasks.append(p)
    ray.get(tasks)
    
    # ----------------------------------------

    # open previously obtained results
    try:
      with open('%s_FADO.pkl'%Galaxy, 'rb') as f: content = pickle.load(f)
    except:
      with open(pathDIR + '/%s_FADO_%s.pkl'%(Galaxy,temp), 'rb') as f: content = pickle.load(f)
      
    imgs = np.full((12,yL,xL), np.nan)

    # delete LOG files for MC
    logF  = glob.glob(pathDIR + '/MC/OUT/*.LOG') ; [os.remove(f) for f in logF if os.path.exists(f)]

    # save results in pkl file
    pklsv = []
    N2HaE, O3HbE = ([],[])
    for j in range(len(content)):
      
      bN = content[j][0]
      
      resFiles  = glob.glob(pathDIR + '/MC/OUT/Bin%04d_MC*.FADO.fits'%bN)
      outFiles = [s.replace('.FADO.fits', '') for s in resFiles]
      
      # Read FD output MC in parallel
      AGE_M,AGE_L,MET_M,MET_L,LMASS_P,LMASS_E,N2Ha_L,O3Hb_L,SFHsMe,SFHsMn = ([],[],[],[],[],[],[],[],[],[])
      tasks = [read_FD_output_REM.remote(outF,True) for outF in outFiles]
      
      results = ray.get(tasks)
      for result in results:
        if result:
          AGE_M.append(result['AGE_M'])
          AGE_L.append(result['AGE_L'])
          MET_M.append(result['MET_M'])
          MET_L.append(result['MET_L'])
          LMASS_P.append(result['LMASS_P'])
          LMASS_E.append(result['LMASS_E'])
          N2Ha_L.append(result['N2Ha'])
          O3Hb_L.append(result['O3Hb'])
          SFHsMe.append(result['SFHsMe'])
          SFHsMn.append(result['SFHsMn'])
      
      AGE_M_err   = [np.nanmedian(AGE_M)  , np.nanstd(AGE_M)]
      AGE_L_err   = [np.nanmedian(AGE_L)  , np.nanstd(AGE_L)]
      MET_M_err   = [np.nanmedian(MET_M)  , np.nanstd(MET_M)]
      MET_L_err   = [np.nanmedian(MET_L)  , np.nanstd(MET_L)]
      LMASS_P_err = [np.nanmedian(LMASS_P), np.nanstd(LMASS_P)]
      LMASS_E_err = [np.nanmedian(LMASS_E), np.nanstd(LMASS_E)]
      # SFHs
      SFHsMe_err  = [np.nanmedian(SFHsMe, axis=0), np.nanstd(SFHsMe, axis=0)]
      SFHsMn_err  = [np.nanmedian(SFHsMn, axis=0), np.nanstd(SFHsMn, axis=0)]      
      # BPT errors
      N2HaE.append(np.nanstd(N2Ha_L))
      O3HbE.append(np.nanstd(O3Hb_L))
  
      pklsv.append([bN,AGE_M_err,AGE_L_err,MET_M_err,MET_L_err,LMASS_P_err,LMASS_E_err,SFHsMe_err,SFHsMn_err,np.count_nonzero(~np.isnan(AGE_M_err))])
      
      # Add values to the images
      f = df.index[df['BinVor'] == bN].tolist()
      for i in f:
        xI = df['x'][i] ; yI = df['y'][i]
        imgs[0,yI,xI]  = AGE_M_err[0] ; imgs[1,yI,xI]  = AGE_M_err[1]
        imgs[2,yI,xI]  = AGE_L_err[0] ; imgs[3,yI,xI]  = AGE_L_err[1]
        imgs[4,yI,xI]  = MET_M_err[0] ; imgs[5,yI,xI]  = MET_M_err[1]
        imgs[6,yI,xI]  = MET_L_err[0] ; imgs[7,yI,xI]  = MET_L_err[1]
        # NOTE: The obtained mass comes from the sum of all the spaxels in a bin.
        # For us to be able to sum the image and get the correct total mass
        # one must divide the mass by the number of bins, to get the average
        # This is not the same for the error, since the std gives us how much
        # the values vary, independently of how bright they are
        imgs[8,yI,xI]  = LMASS_P_err[0] - np.log10(len(f))
        imgs[9,yI,xI]  = LMASS_P_err[1]
        imgs[10,yI,xI] = LMASS_E_err[0] - np.log10(len(f))
        imgs[11,yI,xI] = LMASS_E_err[1]

    # --------------------------------------------
    
    with open('%s_FADO_MC.pkl'%Galaxy,'wb') as f: pickle.dump(pklsv, f)

    # Create final maps : 
    header= fits.Header()
    header['LAYER00'] = '< tM >'     ; header['LAYER01'] = 'sigma tM'
    header['LAYER02'] = '< tL >'     ; header['LAYER03'] = 'sigma tL'
    header['LAYER04'] = '< ZM >'     ; header['LAYER05'] = 'sigma ZM'
    header['LAYER06'] = '< ZL >'     ; header['LAYER07'] = 'sigma ZL'
    header['LAYER08'] = '< M*now >'  ; header['LAYER09'] = 'sigma M*now'
    header['LAYER10'] = '< M*ever >' ; header['LAYER11'] = 'sigma M*ever'
    header['COMMENT'] = 'Fitting range: %i - %i'%(mF,SPEC_LIM)
    header['COMMENT'] = 'Sterllar library: %s'%temp
    header['COMMENT'] = 'ST lib file: %s'%lib  
    header['COMMENT'] = 'Number of MC realisations: %i'%N_MC
    
    m = (imgs[0] == 0)
    for im in imgs: im[m] = np.nan
    
    imgs = np.transpose(imgs, (0, 2, 1))
    hdu = fits.PrimaryHDU(data=imgs, header=header)
    hdu.writeto('%s_FADO_maps_MC.fits'%Galaxy, overwrite=True)
    
    # --------------------------------
    ny, nx = imgs[0].shape
    extent = [(-nx/2)*psO, (nx/2)*psO, (-ny/2)*psO, (ny/2)*psO]
    
    titles = [r'Mass-weighted age (Gyr; $\tilde{x}$ & $\sigma$)', r'Lum-weighted age (Gyr; $\tilde{x}$ & $\sigma$)', r'Mass-weighted Z (Z$_\odot$; $\tilde{x}$ & $\sigma$)', r'Lum-weighted Z (Z$_\odot$; $\tilde{x}$ & $\sigma$)', r'Present-day M$\star$ (log M$_\odot$; $\tilde{x}$ & $\sigma$)', r'Ever-formed M$\star$ (log M$_\odot$; $\tilde{x}$ & $\sigma$)']
    idxs = [0,  1, 2,  3, 4,  5, 6,  7, 8,  9, 10, 11]
    xx   = 0.52
    yy   = [0.985, 0.820, 0.655, 0.49, 0.325, 0.16]
    
    fig, ax = plt.subplots(6, 2, figsize=(7, 19))
        
    for p, (ax_i, i) in enumerate(zip(ax.ravel(), idxs)):
      l = p // 2 
      is_median = (p % 2 == 0)
      Z = 1 if 'Z$_\\odot$' in titles[l] else None
      imshow_with_cbar(ax_i, imgs[i], None, perc=(perc1, perc2), cmap=cmap, extent=extent, Z=Z)
      if i % 2 == 0: fig.text(xx, yy[l], titles[l], ha='center', fontsize=12)
      ax_i.contour(imgCont, levels=levels, colors='#5b5f61', linewidths=1, origin='lower', extent=extent)
     
    
    [ax.set(xlabel='″', ylabel='″') for ax in ax.ravel()]
    if R_bulge != None:
      cO_arc = ((cO[0] - nx/2) * psO, (cO[1] - ny/2) * psO)
      for ax in fig.axes: ax.add_patch(Circle(cO_arc, R_bulge, color='#899499', fill=False, linewidth=1))
    
    plt.subplots_adjust(bottom=0.03, left=0.03, right=1, top=0.97, hspace=0.40, wspace=0.02)
    plt.savefig('%s_FADO_maps_MC.png'%Galaxy) ; plt.close()

  # Shutdown ray
  ray.shutdown()
    
  # ----------------------------------------
  # BPT & SFR
  # ----------------------------------------
  if mc:
    data = {
      'N2Ha': ('N2Ha values (log([NII]/Hα))', N2Ha),
      'O3Hb': ('O3Hb values (log([OIII]/Hβ))', O3Hb),
      'N2HaE': ('Uncertainty in N2Ha', N2HaE),
      'O3HbE': ('Uncertainty in O3Hb', O3HbE),
      'R': ('distance from the center (pix)', R)}
  else:
    data = {
      'N2Ha': ('N2Ha values (log([NII]/Hα))', N2Ha),
      'O3Hb': ('O3Hb values (log([OIII]/Hβ))', O3Hb),
      'R': ('distance from the center (pix)', R)}

  with open('%s_FADO_BPT.pkl'%Galaxy, 'wb') as f: pickle.dump(data, f)
  
  '''
  # NOTE: Read FADO's pkl file contents:
  with open('%s_BPT_FADO_%s.pkl'%(Galaxy,temp), 'rb') as f: bpt = pickle.load(f)
  N2Ha = bpt['N2Ha'][1]
  O3Hb = bpt['O3Hb'][1]
  N2HaE = bpt['N2HaE'][1]
  O3HbE = bpt['O3HbE'][1]
  R = bpt['R'][1]
  for key, (desc, array) in bpt.items(): print(f'{key}: {desc}')
  '''
  
  cmapR = 'plasma_r' ; size = 40
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False, sharey=False, gridspec_kw={'width_ratios': [1, 1.3]})
  plt.subplots_adjust(wspace=0.3)
  
  x = np.linspace(-1.5,0.3)
  ax1.plot(x, Kew01(x), 'k--')
  x = np.linspace(-0.180,1.5)
  ax1.plot(x, Sch07(x),'k.')
  x = np.linspace(-1.5,0.)
  ax1.plot(x, Kau03(x),'k')

  if mc: ax1.errorbar(N2Ha[::-1], O3Hb[::-1], xerr=N2HaE[::-1], yerr=O3HbE[::-1], fmt='none', color='grey', capsize=3, alpha=0.2, zorder=1, markersize=5)
  ax1.scatter(N2Ha[::-1], O3Hb[::-1], c=R[::-1], s=size, cmap=cmapR, edgecolor='k', zorder=2)
    
  ax1.set_xlabel('log([NII]λ6584/Hα)')
  ax1.set_ylabel('log([OIII]λ5007/Hβ)')
  ax1.set_xlim([-1.22,0.5])
  ax1.set_ylim([-1.5,1.5])
  ax1.set_title('BPT', fontsize=15) 
  
  pos = ax1.get_position()
  ax1.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])

  m = cm.ScalarMappable(cmap=cmapR)
  m.set_array(R)
  cax0 = fig.add_axes([0.43, 0.12, 0.02, 0.74])
  cbar0 = plt.colorbar(m, cax=cax0, cmap=cmapR)
  cbar0.set_label('Distance from the center (pix)', rotation=-90, labelpad=15)
  
  img = imgsF[25]
  ny, nx = img.shape
  extent = [(-nx/2)*psO, (nx/2)*psO, (-ny/2)*psO, (ny/2)*psO]
  imshow_with_cbar(ax2, img, 'Kennicutt Relation', perc=(perc1, perc2), cmap='Blues_r', extent=extent, cbar_lab=r'SFR (M$\odot$/yr)')
  ax2.contour(imgCont, levels=levels, colors='#5b5f61', linewidths=1, origin='lower', extent=extent)

  ax2.set(xlabel='″', ylabel='″')
  if R_bulge != None:
    cO_arc = ((cO[0] - nx/2) * psO, (cO[1] - ny/2) * psO)
    ax2.add_patch(Circle(cO_arc, R_bulge, color='#899499', fill=False, linewidth=1))
  
  plt.savefig('%s_FADO_BPT.png'%Galaxy) ; plt.close()
