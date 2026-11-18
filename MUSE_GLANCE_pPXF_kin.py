
# Utilities
import os
import sys
import glob
import math
import time
import shutil
import pickle
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from os import path
from astropy.io import fits
from scipy import ndimage, signal
from scipy.ndimage import rotate
from extinction import calzetti00
import astropy.units as u

# Suppress warnings
import warnings
import contextlib
warnings.filterwarnings('ignore')

# Plotting
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import cmasher as cmr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
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
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# pPXF
import ppxf.ppxf_util as util
from ppxf.ppxf import ppxf, robust_sigma

# Multiprocessing
import ray
import logging
from printStatus import printStatus

# global kinematic PA
from pafit import fit_kinematic_pa

# spline interpolate
from scipy.interpolate import interp1d

# Read from config
from GLANCE_config import read_config

# Suppress output
from GLANCE_OrbitDecomp import suppress_output

# --------------------------

# Avoid for GLANCE to be stuck if pPXF get stuck on 1 bin
import signal as pysignal

PPXF_TIMEOUT = 60 * 60  # 1 hour
class PPXFTimeout(BaseException): pass
def _ppxf_timeout_handler(signum, frame): raise PPXFTimeout()

# --------------------------

cmap = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.95)
perc1 = 5 ; perc2 = 95 # percentiles for imaging

# -----------------------------------------------
class Object(object): pass
# -----------------------------------------------
HOMEpath = os.path.expanduser('~')
C = 299792.458  # speed of light in km/s
#-----------------------------------------------------------------------------
# NOTE: Additive polynomials can additional correct for imperfect sky subtraction or scattered light. The multiplicative polynomials can correct for inaccuracies in the spectral calibration, and make the fit insensitive to reddening by dust, without the need to adopt a specific reddening curve (https://arxiv.org/pdf/1607.08538.pdf) ; IMPORTANT: Ideally one would like not to use any polynomial in the fit as the continuum shape contains important information on the population (...) Multiplicative polynomials are not recommended for kinematic extraction, where additive polynomials are always recommended (ppxf_example_population_gas_sdss.py)
#-----------------------------------------------------------------------------
# ATTENTION: This code does NOT provide gas velocity maps (yet). However, all required info is stored in the final pkl file.
# Also be sure that all the templates & data are in air/vacuum units of wavelength
#-----------------------------------------------------------------------------

def check_spec(lam,flux,neg_frac_max=0.3,jump_sigma=2.5):
  
  f = flux[np.isfinite(flux)]
  if f.size == 0: return False

  frac_neg = np.mean(f < 0)
  if frac_neg > neg_frac_max: return False

  med = np.median(f) ; mad = np.median(np.abs(f - med))
  sig = 1.4826 * mad if mad > 0 else np.std(f)
  
  # check Paschen jump (if its over-subtracted)
  WL0 = 8200 ; dWL = 50
  w = (lam > WL0 - dWL) & (lam < WL0 + dWL) & np.isfinite(flux)

  if np.sum(w) > 2:
    dy = np.diff(flux[w])
    if np.max(np.abs(dy)) > jump_sigma * sig: return False
  
  return True

def neb_cont(Galaxy,binInfo,lam_gal,specBIN,neb_file):
  
  # --------------------------------------------------------------------------
  # From FADO's results get the stellar extinction & compute the nebular cont
  # --------------------------------------------------------------------------
  with open(neb_file, 'rb') as f: content = pickle.load(f)
  
  # https://www-astro.physics.ox.ac.uk/~cappellari/software/ppxf_manual.pdf -> one can add the values of reddening & gas_reddening to the ppxf fit
  BIN,ST_EXT,NB_EXT,Ha_F,Hb_F = ([],[],[],[],[])
  for j in range(len(content)):
    BIN.append(content[j][0])
    ST_EXT.append(content[j][13]) # AV (mag)
    NB_EXT.append(content[j][14]) # AV (mag)
    Ha_F.append(content[j][28])
    Hb_F.append(content[j][20])
  
  # To convert from AV to E(B-V) - as ppxf needs:
  # AV = R*E(B-V) ; R = 3.1 for the MW
  redS,redN,Ha,Hb = ([],[],[],[])
  for b in binInfo:
    try:
      i = BIN.index(b)
      redS.append(ST_EXT[i]/3.1)
      redN.append(NB_EXT[i]/3.1)
      Ha.append(Ha_F[i])
      Hb.append(Hb_F[i])
    except:
      # The bin was not processed with FADO for some reason
      pass
      
  # ------------------------------
  # Compute the nebular continuum
  # ------------------------------
  
  # Cannonical nebular continuum
  neb_contF = fits.open('%sFnebcon.fits'%BIN_DIR)
  s = neb_contF[0].header['CRVAL1']
  i = neb_contF[0].header['CDELT1']
  l = neb_contF[0].header['NAXIS1']
  wNeb = s + i * np.arange(l)
  fNeb = neb_contF[0].data
  neb_contF.close()
  
  # 1) estimation of the V-band extinction in the nebular component using the Balmer decrement
  HaF = [x.n for x in Ha] ; HbF = [x.n for x in Hb]
  AVneb = 7.96*np.log((np.array(HaF)/np.array(HbF))/2.86)
      
  # 2) compute the dimming of the nebular component for the previously derived AVneb and the Calzetti 2000 extinction law
  specBINC,extS,extN = ([],[],[])
  for j,b in enumerate(binInfo):
    try:
      i = binInfo.index(b)
      AV = AVneb[i]
      ext = calzetti00(np.array(wNeb), AV, 3.1)
      NebEx = 10**(0.4*AV*ext)
      
      # 3) apply AVneb to the pre-computed UV-through-IR SED of the nebular continuum and scale it (the extincted nebular continuum SED Fnebcon0 needs to be scaled to the observed (therefore also extincted) Hα flux - The level of the nebular continuum at 6563 Å (:= f_NebCon@Hα) is f_NebCon@Hα = fHα*2.03e-4)
      Fnebcon0 = HaF[i]*2.03e-4*(fNeb/NebEx)
    
      # 4) spline interpolate to subtract it from the observed spectra
      si = interp1d(wNeb, Fnebcon0, kind='cubic')
    
      if not np.isnan(AVneb[i]):
        # Subtract it from the observed spectra
        spec = specBIN[j] - si(lam_gal)
        # check if the spectra is ok
        isgood = check_spec(lam_gal,spec)
        if isgood:
          specBINC.append(spec)
        else:
          specBINC.append(specBIN[j])
      else:
        specBINC.append(specBIN[j])
      
      extS.append(redS[i]) ; extN.append(redN[i])        
    except:
      # Not processed with FADO
      specBINC.append(specBIN[j])
      extS.append(0) ; extN.append(0)
  
    if np.any(np.isnan(specBINC[-1])): specBINC[-1] = specBIN[j]

  return specBINC,extS,extN

#----------------------------------------------------

# Returns the functional resolution of MUSE as a function of λ
def inst_resolution(museF):
  R_lam = np.loadtxt(museF, skiprows = 1)
  z = np.polyfit(R_lam[:,0]*10.,R_lam[:,0]*10./R_lam[:,1],3)
  p = np.poly1d(z)
  return p

def mean_std(A):
  A = A[A > 0]
  return np.nanmean(A), np.nanstd(A)

#----------------------------------------------------

# Fit the stellar kinematics
def fit_stars(nN,templates,spectrum,noise,velscale,start,goodpixels0,lam,lam_temp,redS,degree,plot,quiet):

  kwargs = {}
  if redS is not None: kwargs['reddening'] = redS
    
  m = 10 ; n = 0
  while abs(1 - m) > 0.001 and n < nN:
    goodpixels = np.copy(goodpixels0)
    if nN == 2: # MC simulations -> bias = 0
      pp = ppxf(templates, spectrum, noise, velscale, start, moments=NMOM, degree=degree, mdegree=-1, lam=lam, lam_temp=lam_temp, goodpixels=goodpixels, quiet=quiet, plot=plot, bias=0, **kwargs)
    else:
      pp = ppxf(templates, spectrum, noise, velscale, start, moments=NMOM, degree=degree, mdegree=-1, lam=lam, lam_temp=lam_temp, goodpixels=goodpixels, quiet=quiet, plot=plot, **kwargs)      

    goodpixels0 = clip_outliers(spectrum, pp.bestfit, goodpixels)
    fsim = len(np.intersect1d(goodpixels, goodpixels0)) / len(goodpixels0)
    fnew = len(goodpixels0)/len(goodpixels)
    m = (fsim + fnew)/2
    n = n + 1

  try:
    optimal_template = templates @ pp.weights
  except:
    optimal_template = 0
  
  return pp, optimal_template

#----------------------------------------------------

# Fit 2 stellar kinematic components - not being used for now but someone can use it if they want
def fit_stars_2comp(nN,templates,spectrum,noise,velscale,start,goodpixels0,lam,lam_temp,redS,degree,plot,quiet):

  stars_2_temp = np.column_stack([templates, templates])
  start_2_temp = [start[0], start[0]]

  kwargs = {}
  if redS is not None: kwargs['reddening'] = redS
    
  m = 10 ; n = 0
  while abs(1 - m) > 0.001 and n < nN:
    goodpixels = np.copy(goodpixels0)
    if nN == 2: # MC simulations -> bias = 0
      pp = ppxf(stars_2_temp, spectrum, noise, velscale, start_2_temp, moments=NMOM, degree=degree, mdegree=-1, lam=lam, lam_temp=lam_temp, goodpixels=goodpixels, quiet=quiet, plot=plot, bias=0, **kwargs)
    else:
      pp = ppxf(stars_2_temp, spectrum, noise, velscale, start_2_temp, moments=NMOM, degree=degree, mdegree=-1, lam=lam, lam_temp=lam_temp, goodpixels=goodpixels, quiet=quiet, plot=plot, **kwargs)      
 
    goodpixels0 = clip_outliers(spectrum, pp.bestfit, goodpixels)
    fsim = len(np.intersect1d(goodpixels, goodpixels0)) / len(goodpixels0)
    fnew = len(goodpixels0)/len(goodpixels)
    m = (fsim + fnew)/2
    n = n + 1

  try:
    optimal_template = templates @ pp.weights
  except:
    optimal_template = 0
  
  return pp, optimal_template

#----------------------------------------------------

# Fit stellar + gas kinematics
def fit_stars_gas(nN,templates,spectrum,noise,velscale,start,lam,lam_temp,moments,component,gas_component,gas_names,redS,redN,constr_kinem,tied,degree,goodpixels0,plot,quiet):

  kwargs = {}
  if redS is not None: kwargs['reddening'] = redS
  if redN is not None: kwargs['gas_reddening'] = redN
    
  m = 10 ; n = 0
  while abs(1 - m) > 0.001 and n < nN:
    goodpixels = np.copy(goodpixels0)
    pp = ppxf(templates, spectrum, noise, velscale, start, goodpixels=goodpixels, moments=moments, degree=degree, mdegree=-1, lam=lam, lam_temp=lam_temp, component=component, gas_component=gas_component, gas_names=gas_names, constr_kinem=constr_kinem, tied=tied, quiet=quiet, plot=plot, **kwargs)
    
    goodpixels0 = clip_outliers(spectrum, pp.bestfit, goodpixels)
    fsim = len(np.intersect1d(goodpixels, goodpixels0)) / len(goodpixels0)
    fnew = len(goodpixels0)/len(goodpixels)
    m = (fsim + fnew)/2
    n = n + 1

  try:
    optimal_template = templates[:, ~gas_component] @ pp.weights[~gas_component]
  except Exception as e:
    print(e)
    optimal_template = 0
  
  return pp, optimal_template
  
#----------------------------------------------------

# Iteratively clip the outliers
def clip_outliers(spectrum,bestfit,goodpixels):

  # Repeat the fit after clipping bins deviants more than 3σ in relative error until the bad bins don't change any more.
  while True:
    scale = spectrum[goodpixels] @ bestfit[goodpixels]/np.sum(bestfit[goodpixels]**2)
    resid = scale*bestfit[goodpixels] - spectrum[goodpixels]
    err = robust_sigma(resid, zero=1)
    ok_old = goodpixels
    goodpixels = np.flatnonzero(np.abs(bestfit - spectrum) < 3*err)
    if np.array_equal(goodpixels, ok_old):
        break
          
  return goodpixels
  
#------------------------------

def build_library(files,velscale,FWHM_temp,FWHM_func,norm_range=None):

  # NOTE: Based on the function ppxf.miles_util (at ~/.local/lib/python3.8/site-packages/ppxf) for stellar templates instead of SSPs following the example at python3.8/site-packages/ppxf/examples/ppxf_example_kinematics_sdss.py

  hdu = fits.open(files[0])
  ssp = hdu[0].data
  if len(ssp) == 1:
    ssp = hdu[0].data[0]
  
  h2 = hdu[0].header
  lam = h2['CRVAL1'] + np.arange(h2['NAXIS1'])*h2['CDELT1']
  lam_range_temp = lam[[0, -1]]
  ssp_new, ln_lam_temp = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[:2]
  
  if FWHM_func:
    FWHM_gal = FWHM_func(lam) # MUSE's resolution function in Å
  else:
    FWHM_gal = 6 # assume a constant spectral resolution of 6 Å (valid for both SINFONI & CALIFA)
    
  lam_temp = np.exp(ln_lam_temp)
  if norm_range:
      band = (norm_range[0] <= lam_temp) & (lam_temp <= norm_range[1])

  temp = np.empty((ssp_new.size, len(files)))
  
  if FWHM_gal is not None:
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_temp**2)
    sigma = FWHM_dif/2.355/h2['CDELT1'] # σ difference in pixels (a σ of 2.355 contains ~97% of all values)
  
  flux = []
  for f in range(len(files)):
    hdu = fits.open(files[f])
    ssp = hdu[0].data
    if len(ssp) == 1:
      ssp = hdu[0].data[0]

    if FWHM_gal is not None:
      if np.isscalar(FWHM_gal):
        if sigma > 0.1: # skip convolution for nearly zero σ
          ssp = ndimage.gaussian_filter1d(ssp, sigma)
        else:
          ssp = util.gaussian_filter1d(ssp, sigma) # convolve with σ
    ssp_new = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[0]
    if norm_range is not None:
      ssp_new /= np.median(ssp_new[band]) # normalize every spectrum        
      flux.append(np.median(ssp_new[band]))
    temp[:,f] = ssp_new
    
  if norm_range is None:
    flux = np.median(temp[temp > 0])
    temp /= flux # normalize by a scalar
  
  templates = Object()
  templates.templates = temp
  templates.ln_lam_temp = ln_lam_temp
  templates.lam_temp = lam_temp
  templates.flux = flux
  
  return templates

# -------------------------------------------------------------------------------

@ray.remote
def workerPPXF(c,mc,stars_templates,specBIN,errBIN,norm_fact,velscale,startST,goodpixels0,lam_gal,st_lib,stars_gas_templates,start,moments,deg,component,gas_component,gas_names,Galaxy,pathDIR,redS,redN,jstart,jfin,binInfo,GST_STP,NMOM,N_MC):

  run_ppxf.__globals__['GST_STP'] = GST_STP
  run_ppxf.__globals__['NMOM'] = NMOM
  run_ppxf.__globals__['N_MC'] = N_MC
  fit_stars.__globals__['NMOM'] = NMOM
  fit_stars_2comp.__globals__['NMOM'] = NMOM
  
  ppSol = []
  if jfin > len(binInfo): jfin = len(binInfo)
  for j in range(jstart,jfin):
    bN = binInfo[j]
    spectrum = np.array(specBIN[j])
    if spectrum.any():
      noise = errBIN[j].copy()
      noise[noise < 0] = 0
      normF = norm_fact[j]
      try:
        extS = redS[j].nominal_value ; extN  = redN[j].nominal_value
      except Exception as e:
        print(e)
        extS = None ; extN = None
        
      old_handler = pysignal.signal(pysignal.SIGALRM, _ppxf_timeout_handler)
      try:
        
        pysignal.alarm(PPXF_TIMEOUT)
        sol = run_ppxf(bN,mc,stars_templates,spectrum,noise,normF,velscale,startST,goodpixels0,lam_gal,st_lib,stars_gas_templates,start,moments,deg,component,gas_component,gas_names,Galaxy,pathDIR,extS,extN)
        pysignal.alarm(0)

      except PPXFTimeout:
        
        pysignal.alarm(0)
        print(f' × pPXF timeout: bin {bN} exceeded {PPXF_TIMEOUT/60:.1f} min')

        plt.close('all')
        sol = [bN, np.nan]

      finally:
        pysignal.alarm(0)
        pysignal.signal(pysignal.SIGALRM, old_handler)

      ppSol.append(sol)
    
  with open('%s/ppxf_res_%03d.pkl'%(pathDIR,c),'wb') as f:
    pickle.dump(ppSol, f)
    
# -------------------------------------------------------------------------------

def run_ppxf(j,mc,stars_templates,spectrum,noise,normF,velscale,startST,goodpixels0,lam_gal,st_lib,stars_gas_templates,start,moments,deg,component,gas_component,gas_names,Galaxy,pathDIR,extS,extN):
  
  global GST_STP
  
  startT = time.time()
  m = np.where(spectrum == 0)[0]
  gp = np.setdiff1d(goodpixels0, m)
  
  # NOTE: Purely stellar fit for SINFONI:
  if '_SINF' in pathDIR: GST_STP = 'ST'
  
  if GST_STP == 'GST':
    fig = plt.figure(figsize=(12, 8))
  else:
    fig = plt.figure(figsize=(12, 5))
    
  plotMC = False
  plt.suptitle(f'Voronoi bin {j}')  
  try:
    solST, solG = ([],[])
    if 'ST' in GST_STP:

      # ------------------------------- PURELY STELLAR FIT ---------------------------------
      if GST_STP == 'GST':
        plt.subplot(211)
      else:
        plt.subplot(111)
        
      pp, bestfit_template = fit_stars(10,stars_templates,spectrum,noise,velscale,startST,gp,lam_gal,st_lib.lam_temp,extS,degree=deg,plot=True,quiet=True)

      if 'Red' in pathDIR:
        plt.title('Purely stellar - Red spectral range - V & σ %.2f, %.2f km/s'%(pp.sol[0],pp.sol[1]))
      elif 'Blue' in pathDIR:
        plt.title('Purely stellar - Blue spectral range - V & σ %.2f, %.2f km/s'%(pp.sol[0],pp.sol[1]))
      else:
        plt.title('Purely stellar - V & σ %.2f, %.2f km/s'%(pp.sol[0],pp.sol[1]))
            
      pp.plot()
      
      # Error estimation  ----------------------------
      if mc:
        errs = np.full((N_MC, len(pp.sol)), np.nan)
        
        # save the 3 first realizations in a figure
        if plotMC: figMC = plt.figure(figsize=(12, 14)) ; pN = 311
        for i in range(N_MC):
          y = spectrum + np.random.normal(0, 1, len(pp.bestfit)) * np.std(pp.galaxy[pp.goodpixels] - pp.bestfit[pp.goodpixels])
          sim, _ = fit_stars(2,bestfit_template,y,noise,pp.velscale,pp.sol,pp.goodpixels,lam_gal,st_lib.lam_temp,extS,degree=pp.degree,plot=plotMC,quiet=True)
          errs[i] = sim.sol

          if plotMC and pN < 314:
            plt.subplot(pN)
            plt.title('V & σ %.2f, %.2f km/s'%(sim.sol[0],sim.sol[1]))
            sim.plot()
            pN = pN + 1

        mc_err = np.std(errs, axis=0)
        mc_avr = np.median(errs, axis=0)
        pp.mc_err = mc_err ; pp.mc_avr = mc_avr
        
        if plotMC:
          plt.subplots_adjust(hspace=0.4)
          plt.savefig('%s/fits/%s_Bin_%04d_MC.png'%(pathDIR,Galaxy,j))
          plt.close()
        
      else:
        pp.mc_err = pp.error ; pp.mc_avr = pp.sol
      # ----------------------------------------------

      solST = [j,pp.lam,pp.galaxy,pp.noise,pp.goodpixels,pp.bestfit,pp.chi2,pp.sol,pp.mc_err,pp.mc_avr,pp.degree,normF]
      
    # ----------------------------------------------------------------------------

    if 'G' in GST_STP:
      
      # -------------------------------- STELLAR & GAS -------------------------------------
      if GST_STP == 'GST':
        plt.xticks([]) ; plt.xlabel('')
        plt.subplot(212)
      else:
        plt.subplot(111)

      goodpixels0 = np.arange(0,len(spectrum),1)
      m = np.where(spectrum == 0)[0]
      gp = np.setdiff1d(goodpixels0, m)

      pp, bestfit_template = fit_stars_gas(10, stars_gas_templates, spectrum, noise, velscale, start, lam_gal, st_lib.lam_temp, moments, component, gas_component, gas_names, extS, extN, degree=8, constr_kinem=None, tied=None, goodpixels0=gp, plot=True, quiet=True)
      
      plt.figure(fig.number)
      plt.title('Stars + gas - V & σ %.2f, %.2f km/s'%(pp.sol[0][0],pp.sol[0][1]))
      pp.plot()
      
      # Error estimation  ----------------------------
      if mc:
        errs = np.full((N_MC, len(pp.sol[0])), np.nan)
        
        # save the 3 first realizations in a figure
        if plotMC: figMC = plt.figure(figsize=(12, 14)) ; pN = 311
        for i in range(N_MC):
          y = spectrum + np.random.normal(0, 1, len(pp.bestfit)) * np.std(pp.galaxy[pp.goodpixels] - pp.bestfit[pp.goodpixels])
          sim, _ = fit_stars(2,bestfit_template,y,noise,pp.velscale,pp.sol[0],pp.goodpixels,lam_gal,st_lib.lam_temp,extS,degree=pp.degree,plot=plotMC,quiet=True)
          errs[i] = sim.sol
          
          if plotMC and pN < 314:
            plt.subplot(pN)
            plt.title('V & σ %.2f, %.2f km/s'%(sim.sol[0],sim.sol[1]))
            sim.plot()
            pN = pN + 1
            
        mc_err = np.std(errs, axis=0)
        mc_avr = np.median(errs, axis=0)
        pp.mc_err = mc_err ; pp.mc_avr = mc_avr
        
        if plotMC:
          plt.subplots_adjust(hspace=0.4)
          plt.savefig('%s/fits/%s_Bin_%04d_MC_stgas.png'%(pathDIR,Galaxy,j))
          plt.close()
        
      else:
        pp.mc_err = pp.error[0] ; pp.mc_avr = pp.sol[0]
      # ----------------------------------------------
      
      solG = [j,pp.lam,pp.galaxy,pp.noise,pp.goodpixels,pp.bestfit,pp.chi2,pp.sol,pp.mc_err,pp.mc_avr,pp.degree,pp.gas_names,pp.gas_flux,pp.gas_flux_error,normF]
    
    plt.figure(fig.number)
    plt.savefig('%s/fits/%s_Bin_%04d.png'%(pathDIR,Galaxy,j))
    t = (time.time()-startT)/60
    plt.close(fig)
    
    return([solST,solG,t])
    
  except Exception as e:
    print(e)
    return([np.nan, np.nan])

# --------------------------

def fmt(x, pos):
  s = f'{x:.2f}'
  s = s.rstrip('0').rstrip('.')
  return s

def imshow_with_cbar(ax, img, title, perc=(5, 95), cmap=None, extent=None, sym=False):
  
  vmin = np.nanpercentile(img, perc[0])
  vmax = np.nanpercentile(img, perc[1])
    
  if sym:
    vmax_abs = min(abs(vmin), abs(vmax))
    vmin = -vmax_abs
    vmax =  vmax_abs  
  
  ax.set_title(title, pad=10)
  im = ax.imshow(img, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

  div = make_axes_locatable(ax)
  cax = div.append_axes('right', size='4%', pad=0.05)
  cb = ax.figure.colorbar(im, cax=cax)
  
  cb.formatter = FuncFormatter(fmt)
  cb.update_ticks()
  cb.ax.tick_params(labelsize=12)

  return im

# -------------------------------------------------------------------------------

def kin_pPXF(Galaxy,df,xL,yL,pathDIR,FWHM_temp,temp,limsFIT):

  from run_GLANCE import get_contours
  global nbins,GST_STP,NMOM,BIN_DIR,N_MC

  try:
    GAL_INFO, SSP_DIR, STT_DIR, BIN_DIR, BAYES_DIR, CORES, NEB_CONT, N_MCS, SPEC_LIM, WR_PPXF, TEMPS_PPXF, LINE_STP, NMOM, GST_STP, CHI2_TYPE, TODO_PDYN, HST_FILT, MGE_M, SINF, TEMP_BAYES, REGUL, WR_BAYES, OCUT, V_SIGMA, ERR_DSBP, TEMPS_FD, LIBS_FD, TEMPS_SL, LIBS_SL, SL_CONF, RY_SSC, RY_TCS, TEMPS_SINF, NMOM_SINF = read_config()
  except Exception as e:
    print(f' × Problem in read_config(): {e} ×')
    print(' × Check your config file ×')
    sys.exit(1)
  
  if CORES == 0: CORES = os.cpu_count() // 2

  N_MC, _ = N_MCS
  mc = True if N_MC > 0 else False
  imgCont,levels = get_contours(Galaxy)

  # --------------------------------

  museF = '%sMUSE_spectral_resolution.txt'%BIN_DIR
  
  # NOTE: Blue/Red pPXF fits are purely stellar
  if limsFIT is not None:
    print('× ATTENTION × Blue/Red fits are purely stellar') 
    print('-----------------------------------------------------------------------------------')
    GST_STP = 'ST'
  
  # --------------------------------
  # Find center of the galaxy
  maps = fits.open('%s_maps.fits'%Galaxy) ; maps = maps[0].data
  STRcont = maps[1]

  MSK = fits.open(Galaxy + '_mask.fits')
  MSK = MSK[0].data ; MSK = np.array(MSK) ; MSK = np.logical_not(MSK).astype(int)
  ctr = np.unravel_index(np.nanargmax(STRcont*MSK), np.shape(STRcont*MSK))
  yC, xC = ctr
  
  # --------------------------------
  # Start preparing the spectra
  # --------------------------------
  
  with open('%s_Bins.pkl'%Galaxy, 'rb') as f: content = pickle.load(f)
  logLam,velscale = content[-1]
  lam_gal = np.exp(logLam)

  # Find the widest spectral range the both spec and tmpl encompass  
  files = glob.glob(STT_DIR + temp + '/*.fits')
  hdu = fits.open(files[0]) ; h2 = hdu[0].header
  lam_temp = h2['CRVAL1'] + np.arange(h2['NAXIS1'])*h2['CDELT1']

  # Truncate templates and obs spec to the same λ range
  if limsFIT is None:
    m = lam_temp[0] ; M = lam_temp[-1]
  else:
    m = limsFIT[0]  ; M = limsFIT[1]
    
  if lam_gal[0] > m  : m = lam_gal[0]
  if lam_gal[-1] < M : M = lam_gal[-1]
  if (m - lam_temp[0]) < 100  : m = m + 100 - (m - lam_temp[0])
  if (lam_temp[-1] - M) < 100 : M = M - 100 + (lam_temp[-1] - M)
  if m < 4687: m = 4690 # to avoid the line HeII4687
  
  # Truncate the spectra to the desired λ range
  v = min(abs(logLam - np.log(m))) ; n1 = list(abs(logLam - np.log(m))).index(v) ; n1 = n1 + 1
  v = min(abs(logLam - np.log(M))) ; n2 = list(abs(logLam - np.log(M))).index(v) ; n2 = n2 - 1
  logLam  = logLam[n1:n2]
  lam_gal = lam_gal[n1:n2]
  
  nbins = len(content)-1
  binInfo, specBIN, errBIN, norm_fact = ([],[],[],[])
  for i in range(nbins):
    binInfo.append(content[i][0])
    norm_fact.append(content[i][1])
    specBIN.append(content[i][2][n1:n2])
    errBIN.append(content[i][3][n1:n2])

  # ----------------------------------------------------
  # Correct for the nebular cont & get reddening values
  # ----------------------------------------------------
  
  if NEB_CONT:
    neb_file = '%s_analysis/FADO/%s/%s_FADO_%s.pkl'%(Galaxy,TEMPS_FD[0],Galaxy,TEMPS_FD[0])
    if path.isfile(neb_file) == True:
      specBINC,redS,redN = neb_cont(Galaxy,binInfo,lam_gal,specBIN,neb_file)
    else:
      specBINC = np.copy(specBIN)
      redS = None ; redN = None
  else:
    specBINC = np.copy(specBIN)
    redS = None ; redN = None
  
  # -----------------------------
  # Setup the stellar templates
  # -----------------------------
  
  # Convolve the stellar templates to the same resolution as MUSE data 
  FWHM_func = inst_resolution(museF)
    
  st_lib = build_library(files, velscale, FWHM_temp, FWHM_func, norm_range=None)
  stars_templates, ln_lam_temp = st_lib.templates, st_lib.ln_lam_temp

  vel0 = C*np.log(1 + 0) # initial estimate of the galaxy velocity in km/s. eq. (8) of Cappellari (2017)
  startST = [vel0, 200.] # (km/s), starting guess for [V,σ]
  
  if limsFIT is None:
    lam_range_temp = np.exp(ln_lam_temp[[0, -1]])
  else:
    lam_range_temp = [lam_gal[0],lam_gal[-1]]

  goodpixels0 = util.determine_goodpixels(logLam, lam_range_temp, 0)
  
  # -----------------------------
  # Setup the gas templates
  # -----------------------------

  # Construct a set of Gaussian emission line templates
  lam_range_gal = [lam_gal[0],lam_gal[-1]]
  # The FWHM of the templates will be a function of wavelength
  with suppress_output(): gas_templates, gas_names, line_wave = util.emission_lines(st_lib.ln_lam_temp, lam_range_gal, FWHM_func)
  
  # Joined kinematics for the emission lines -------------------------------------------------------------
  if LINE_STP == 'join':
    ngas_comp = 2 # 1 narrow + 1 broad
    gas_templates = np.tile(gas_templates, ngas_comp)
    gas_names = np.asarray([a + f'_({p+1})' for p in range(ngas_comp) for a in gas_names])
    line_wave = np.tile(line_wave, ngas_comp)
    # Combining the stellar and gaseous templates into a single array
    stars_gas_templates = np.column_stack([stars_templates, gas_templates])
    
    # Assigning component=0 to the stellar template and one component per emission line x2 (narrow + broad)
    component = [0]*np.shape(stars_templates)[1] + [1]*int(len(gas_names)/ngas_comp) + [2]*int(len(gas_names)/ngas_comp) 

    # Set the moments (only 2 for gas components)
    moments = [NMOM, 2, 2]
    
    if NMOM == 6:
      start = [[-200, 200.0, 0, 0, 0, 0], [100, 100.0], [100, 100.0]]
    else:
      start = [[-200, 200.0, 0, 0], [100, 100.0], [100, 100.0]]
    
  # Separated kinematics for the emission lines ----------------------------------------------------------
  elif LINE_STP == 'sep':
    ngas_comp = 1 # 1 narrow
    gas_templates = np.tile(gas_templates, ngas_comp)
    line_wave = np.tile(line_wave, ngas_comp)
    # Combining the stellar and gaseous templates into a single array
    stars_gas_templates = np.column_stack([stars_templates, gas_templates])
    
    # Assigning component=0 to the stellar template and one component per emission line (narrow)
    component = [0]*np.shape(stars_templates)[1]
    # Set the moments (6 for stars and 2 for gas)
    moments = [NMOM] # for such good SNR go for 6 moments
    
    # Start values for each component
    if NMOM == 6:
      start = [[-200, 200.0, 0, 0, 0, 0]] ; start_gas = [100, 100.0]
    else:
      start = [[-200, 200.0, 0, 0]] ; start_gas = [100, 100.0] 
      
    for i in range(len(gas_names)):
      component = component + [i+1]
      moments.append(2)
      start.append(start_gas)
  # Separated kinematics for the emission lines with TIED ------------------------------------------------
  elif LINE_STP == 'sep_tied':
    # Combining the stellar and gaseous templates into a single array
    stars_gas_templates = np.column_stack([stars_templates, gas_templates])
    component = [0]*np.shape(stars_templates)[1] # stellar template
    # component 1: 'Hbeta' 'Halpha' # component 2: '[SII]6716' '[SII]6731'
    # component 3: 'HeI5876'        # component 4: '[OIII]5007_d'
    # component 5: '[OI]6300_d'     # component 6: '[NII]6583_d'
    component += [1]*2              # The Balmer lines share the same LOSVD and are assigned the same kinematic component=1
    component += [2, 2, 3, 4, 5, 6] # Doublets with fixed 1/3 ratio are in the same kinematic component (4, 5, 6)
    component = np.array(component)

    # Set the moments (NMOM for stars and 2 for gas)
    moments = [NMOM] + [2]*6
    
    # [[V0, sig0, h3, h4, h5, h6], #  stellar kinematics
    #  [V1, sig1],  # Balmer lines #  [V2, sig2],  # '[SII]6716' '[SII]6731'
    #  [V5, sig5],  # 'HeI5876'    #  [V6, sig6],  # '[OIII]5007_d'
    #  [V7, sig7],  # '[OI]6300_d' #  [V8, sig8]]  # '[NII]6583_d'
    if NMOM == 6:
      start = [[-200, 200.0, 0, 0, 0, 0]] ; start_gas = [100, 100.0]
    else:
      start = [[-200, 200.0, 0, 0]] ; start_gas = [100, 100.0]      
    
    for i in range(len(moments)-1):
      start.append(start_gas)
  # ---------------------------------------
  
  # The gas components are all components larger than zero
  gas_component = np.array(component) > 0

  # ------------------------------------------
  # Fit kinematics on Voronoi binned data
  # ------------------------------------------

  print('*                                 Running pPXF in ||                              *')
  print('-----------------------------------------------------------------------------------')
  print(' • Spectral Library : %s'%temp)
  print(' • Fitting range    : %.2f to %.2f Å'%(m,M))
  print('-----------------------------------------------------------------------------------')
  
  CORES = min(CORES, 100, nbins)
  if nbins == 0:
    print(' × ATTENTION × No spectra to process!!')
    return

  # remove possible corrupted pkl files 
  pkls = glob.glob('%s/ppxf_res_*.pkl'%pathDIR)
  for pkl in pkls:
    try:
      with open(pkl, 'rb') as f: pickle.load(f)
    except Exception:
      os.remove(pkl)

  doneBins = []
  pkls = glob.glob('%s/ppxf_res_*.pkl'%pathDIR)
  lpkl = max([int(os.path.basename(p).split('_')[-1].split('.')[0]) for p in pkls], default=0)
  for pkl in pkls: 
    with open(pkl, 'rb') as f:
      ppSol = pickle.load(f)
    for sol in ppSol: 
      if len(sol) == 3: doneBins.append(sol[0][0])
      if len(sol) == 2: doneBins.append(sol[0])  

  remIdx = [i for i,b in enumerate(binInfo) if b not in doneBins]
  if 0 < len(remIdx) < len(binInfo):
    print(' → Out of %i there are still %i bins to run'%(len(binInfo),len(remIdx)))
  if len(remIdx) == 0:
    print(' × No remaining bins to run')
  else:
    binInfoR = [binInfo[i] for i in remIdx]
    specBINCR = [specBINC[i] for i in remIdx]
    errBINR = [errBIN[i] for i in remIdx]
    norm_factR = [norm_fact[i] for i in remIdx]

    if redS is not None:
      redSR = [redS[i] for i in remIdx]
      redNR = [redN[i] for i in remIdx]
    else:
      redSR = None
      redNR = None

    nb = len(binInfoR) ; nbins = nb
    CORES = min(CORES, 100, nb)    
    ray.init(num_cpus=int(CORES),logging_level=logging.ERROR,log_to_driver=False)
    deg = 6

    st_lib_small = Object()
    st_lib_small.lam_temp = st_lib.lam_temp

    stars_templates_ref = ray.put(stars_templates)
    if stars_gas_templates is not None:
      stars_gas_templates_ref = ray.put(stars_gas_templates)
    else:
      stars_gas_templates_ref = None

    st_lib_ref = ray.put(st_lib_small)
    goodpixels0_ref = ray.put(goodpixels0)
    lam_gal_ref = ray.put(lam_gal)
    chunk = 5 ; jstart = 0 ; processes = []
    
    for c in range(math.ceil(nb/chunk)):
      jfin = min(jstart + chunk, nb)

      if jstart < nb:
        spec_chunk = specBINCR[jstart:jfin]
        err_chunk = errBINR[jstart:jfin]
        norm_chunk = norm_factR[jstart:jfin]
        bin_chunk  = binInfoR[jstart:jfin]

        if redSR is not None:
          redS_chunk = redSR[jstart:jfin]
          redN_chunk = redNR[jstart:jfin]
        else:
          redS_chunk = None
          redN_chunk = None

        p = workerPPXF.remote(c + len(pkls) + 1, mc, stars_templates_ref, spec_chunk, err_chunk, norm_chunk, velscale, startST, goodpixels0_ref, lam_gal_ref, st_lib_ref, stars_gas_templates_ref, start, moments, deg, component, gas_component, gas_names, Galaxy, pathDIR, redS_chunk, redN_chunk, 0, len(bin_chunk), bin_chunk, GST_STP, NMOM, N_MC)
        processes.append(p)

      jstart = jfin

    ray.get(processes)
    ray.shutdown()
    
  # -------------------------------------------------------------------------------

  # NOTE: Output results - layer 0 - Vel ; layer 1 - Sig ; layer 2 - h3  ; layer 3 - h4 ; layer 4 - h5  ; layer 5 - h6
  KINSTimgs = np.full((NMOM,xL,yL), np.nan)
  KINGimgs  = np.full((NMOM,xL,yL), np.nan)
  # ---------------------------------------
  
  # Open pkl files and merge them together
  ppxf_res = []
  pkls = sorted(glob.glob('%s/ppxf_res_*.pkl'%pathDIR))
  for pkl in pkls: 
    with open(pkl, 'rb') as f:
      content = pickle.load(f)
      for l in range(len(content)):
        ppxf_res.append(content[l])
    
  # ---------------------------------------
  # Save Results
  if GST_STP == 'GST':
    
    # Save the pickle files
    ppxf_resST = [] ; ppxf_resG = []
    for i in range(len(ppxf_res)):
      ppxf_resST.append(ppxf_res[i][0])
      ppxf_resG.append(ppxf_res[i][1])

    with open('%s_pPXF_ST_%s.pkl'%(Galaxy,temp),'wb') as f:
      pickle.dump(ppxf_resST, f)
    with open('%s_pPXF_G_%s.pkl'%(Galaxy,temp),'wb') as f:
      pickle.dump(ppxf_resG, f)

    # ---------------------------------------
    X, Y , VG, eVG, VST, eVST = ([],[],[],[],[],[])
    # Add values to the images
    for j in range(len(ppxf_res)):
      try:
        bn = ppxf_resST[j][0]
        solST = ppxf_resST[j][7]  ; errST = ppxf_resST[j][8]
        solG = ppxf_resG[j][7][0] ; errG  = ppxf_resG[j][8]
        idx = df.index[df['BinVor'] == bn].tolist()
        for i in idx:
          xI = df['x'][i] ; yI = df['y'][i]
          KINSTimgs[0,xI,yI] = solST[0] ; KINSTimgs[1,xI,yI] = solST[1]
          KINSTimgs[2,xI,yI] = solST[2] ; KINSTimgs[3,xI,yI] = solST[3]
          if NMOM == 6: KINSTimgs[4,xI,yI] = solST[4] ; KINSTimgs[5,xI,yI] = solST[5]
          
          KINGimgs[0,xI,yI] = solG[0] ; KINGimgs[1,xI,yI] = solG[1]
          KINGimgs[2,xI,yI] = solG[2] ; KINGimgs[3,xI,yI] = solG[3]
          if NMOM == 6: KINGimgs[4,xI,yI] = solG[4] ; KINGimgs[5,xI,yI] = solG[5]
          
          # for kin_pa
          X.append(xI) ; Y.append(yI)
          VG.append(solG[0])   ; eVG.append(errG[0])
          VST.append(solST[0]) ; eVST.append(errST[0])

      except: pass

  if GST_STP == 'ST':
    
    # Save the pickle files
    ppxf_resST = []
    for i in range(len(ppxf_res)):
      ppxf_resST.append(ppxf_res[i][0])

    with open('%s_pPXF_ST_%s.pkl'%(Galaxy,temp),'wb') as f:
      pickle.dump(ppxf_resST, f)

    # ---------------------------------------
    X, Y , VST, eVST = ([],[],[],[])
    # Add values to the images
    for j in range(len(ppxf_res)):
      try:
        bn = ppxf_resST[j][0]
        solST = ppxf_resST[j][7]  ; errST = ppxf_resST[j][8]
        idx = df.index[df['BinVor'] == bn].tolist()
        for i in idx:
          xI = df['x'][i] ; yI = df['y'][i]
          KINSTimgs[0,xI,yI] = solST[0] ; KINSTimgs[1,xI,yI] = solST[1]
          KINSTimgs[2,xI,yI] = solST[2] ; KINSTimgs[3,xI,yI] = solST[3]
          if NMOM == 6: KINSTimgs[4,xI,yI] = solST[4] ; KINSTimgs[5,xI,yI] = solST[5]
                    
          # for kin_pa
          X.append(xI) ; Y.append(yI)
          VST.append(solST[0]) ; eVST.append(errST[0])

      except: pass

  if GST_STP == 'G':
    
    # Save the pickle files
    ppxf_resG = []
    for i in range(len(ppxf_res)):
      ppxf_resG.append(ppxf_res[i][1])

    with open('%s_pPXF_G_%s.pkl'%(Galaxy,temp),'wb') as f:
      pickle.dump(ppxf_resG, f)

    # ---------------------------------------
    X, Y , VG, eVG = ([],[],[],[])
    # Add values to the images
    for j in range(len(ppxf_res)):
      try:
        bn = ppxf_resG[j][0]
        solG = ppxf_resG[j][7][0] ; errG  = ppxf_resG[j][8]
        idx = df.index[df['BinVor'] == bn].tolist()
        for i in idx:
          xI = df['x'][i] ; yI = df['y'][i]
          
          KINGimgs[0,xI,yI] = solG[0] ; KINGimgs[1,xI,yI] = solG[1]
          KINGimgs[2,xI,yI] = solG[2] ; KINGimgs[3,xI,yI] = solG[3]
          if NMOM == 6: KINGimgs[4,xI,yI] = solG[4] ; KINGimgs[5,xI,yI] = solG[5]
          
          # for kin_pa
          X.append(xI) ; Y.append(yI)
          VG.append(solG[0])   ; eVG.append(errG[0])

      except: pass

    # ---------------------------------------
    '''
    # NOTE: Create an image for the Ha vel
    f = pd.read_pickle('%s_pPXF_G_IUS.pkl'%Galaxy)
    maps = fits.open('%s_maps.fits'%Galaxy)
    xL, yL = np.shape(maps[0].data[0])
    df = pd.read_csv('%s_VorBin.csv'%Galaxy)
    VHa = np.zeros((xL,yL))
    for j in range(len(f)):
      try:
        bn = f[j][0]
        solG = f[j][7][1]
        idx = df.index[df['BinVor'] == bn].tolist()
        for i in idx:
          xI = df['x'][i] ; yI = df['y'][i]
          VHa[xI,yI] = solG[0]
      except:
        pass
        
    hdu = fits.PrimaryHDU(data=VHa) 
    hdu.writeto('%s_Ha_vel_pPXF.fits'%Galaxy, overwrite=True)
    
    # For the rest of the lines (using sep_tied)
    f[j][7][1] = Balmer gas component: Hbeta + Halpha
    f[j][7][2] = [SII]6716 + [SII]6731
    f[j][7][3] = HeI5876
    f[j][7][4] = [OIII]5007_d
    f[j][7][5] = [OI]6300_d
    f[j][7][6] = [NII]6583_d
    '''
    # ---------------------------------------------------------------------------------------------
      
  # Estimate the correction on the systemic velocity
  print('*               Fit and subtract the global kinematic position angle              *')
  print('-----------------------------------------------------------------------------------')

  # --------------------------------
  # Stellar + gas
  # --------------------------------

  if 'G' in GST_STP:

    print(' ➞ Stars + Gas (vel data obtained fitting emission lines simultaneously)')
    print('-----------------------------------------------------------------------------------')
  
    try:
      x = np.array(X) ; y = np.array(Y) ; z = np.array(VG) ; eV = np.array(eVG)
      nanM = np.isnan(x) | np.isnan(y) | np.isnan(z) ; nanM = ~nanM # Remove nans
      plt.close('all') ; plt.figure()      
      with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        angBest,angErr,vSyst = fit_kinematic_pa.fit_kinematic_pa(x[nanM]-xC, y[nanM]-yC, z[nanM], dvel=eV[nanM], nodots=True)
      plt.savefig('%s_pa_G.png'%Galaxy) ; plt.close()
      print(' • Kin PA %.2f +/- %.2f (3σ error) '%(angBest,angErr))
      print(' • Systemic velocity of %.2f km/s'%vSyst)
    except Exception as e:
      print(f' × Couldnt fit PA × : {e}')
      vSyst   = np.nanmedian(VG)
      angBest = 0 ; angErr = 0
    z_cor = vSyst/C
    print('-----------------------------------------------------------------------------------')
  
    m1 = KINGimgs[0] ; m2 = KINGimgs[1]
    m3 = KINGimgs[2] ; m4 = KINGimgs[3]
    
    if NMOM == 6: 
      m5 = KINGimgs[4] ; m6 = KINGimgs[5]
      saveInfo = np.array([m1 - vSyst,m2,m3 - np.nanmedian(m3),m4,m5 - np.nanmedian(m5),m6])
    else:
      saveInfo = np.array([m1 - vSyst,m2,m3 - np.nanmedian(m3),m4])

    hdu = fits.PrimaryHDU(data=saveInfo) 
    hdu.header['pa']    = angBest ; hdu.header['pa_err'] = angErr
    hdu.header['z_cor'] = z_cor   ; hdu.header['vsyst']  = vSyst
    hdu.writeto('%s_pPXF_G_%s.fits'%(Galaxy,temp), overwrite=True)
    
    # Rotate according to PA
    saveInfo = np.array([rotate(img, angle=-angBest, reshape=False, order=0, mode='constant', cval=np.nan) for img in saveInfo])
    imgContR_g = rotate(imgCont, angle=-angBest, reshape=False, order=0, mode='constant', cval=np.nan)

    # --------------------------------

    if NMOM == 4: 
      fig, ax = plt.subplots(2, 2, figsize=(7.5, 6))
      idxs = [0, 1, 2, 3]
      titles = ['Rotation Velocity', 'Velocity Dispersion', 'h3', 'h4']

    if NMOM == 6: 
      fig, ax = plt.subplots(3, 2, figsize=(7.5, 9))
      idxs = [0, 1, 2, 3, 4, 5]
      titles = ['Rotation Velocity', 'Velocity Dispersion', 'h3', 'h4', 'h5', 'h6']

    img0 = saveInfo[0] ; img1 = saveInfo[1]
    v_sig_G = np.abs(img0)/img1 # V over sigma
    
    ny, nx = img0.shape
    extent = [(-xC)*psO, (nx-1-xC)*psO, (-yC)*psO, (ny-1-yC)*psO]

    fig.suptitle('%s : Stellar + gas'%Galaxy, fontsize=16)
    
    for ax_i, i, title in zip(ax.ravel(), idxs, titles):
      sym = (i == 0)
      imshow_with_cbar(ax_i, saveInfo[i], title, perc=(perc1, perc2), cmap=cmap, extent=extent, sym=sym)
      ax_i.contour(imgContR_g, levels=levels, colors='#5b5f61', linewidths=1, origin='lower', extent=extent)

    [ax.set(xlabel='″', ylabel='″') for ax in ax.ravel()]
    
    if R_bulge != None: 
      cO_arc = ((cO[0] - nx/2) * psO, (cO[1] - ny/2) * psO)
      for ax in fig.axes: ax.add_patch(Circle(cO_arc, R_bulge, color='#899499', fill=False, linewidth=1))
    
    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, hspace=0.35, wspace=0.15)
    plt.savefig('%s_pPXF_G_%s.png'%(Galaxy,temp)) ; plt.close()
  
  # --------------------------------
  # Purely stellar
  # --------------------------------

  if 'ST' in GST_STP:
    
    try:
      x = np.array(X) ; y = np.array(Y) ; z = np.array(VST) ; eV = np.array(eVST)
      nanM = np.isnan(x) | np.isnan(y) | np.isnan(z) ; nanM = ~nanM # Remove nans
      plt.close('all') ; plt.figure()
      with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        angBest,angErr,vSyst = fit_kinematic_pa.fit_kinematic_pa(x[nanM]-xC, y[nanM]-yC, z[nanM], dvel=eV[nanM], nodots=True)
      plt.savefig('%s_pa_ST.png'%Galaxy) ; plt.close()
      print(' ➞ Purely stellar (vel data obtained masking emission lines)')
      print('-----------------------------------------------------------------------------------')
      print(' • Kin PA %.2f +/- %.2f (3σ error) '%(angBest,angErr))
      print(' • Systemic velocity of %.2f km/s'%vSyst)
    except Exception as e:
      print(f' × Couldnt fit PA × : {e}')
      vSyst   = np.nanmedian(VST)
      angBest = 0 ; angErr = 0
    z_cor = vSyst/C
        
    m1 = KINSTimgs[0] ; m2 = KINSTimgs[1]
    m3 = KINSTimgs[2] ; m4 = KINSTimgs[3]
          
    if NMOM == 6: 
      m5 = KINSTimgs[4] ; m6 = KINSTimgs[5]
      saveInfo = np.array([m1 - vSyst,m2,m3 - np.nanmedian(m3),m4,m5 - np.nanmedian(m5),m6], dtype=float)
    else:
      saveInfo = np.array([m1 - vSyst,m2,m3 - np.nanmedian(m3),m4], dtype=float)
      
    saveInfo[saveInfo == 1000] = np.nan
    hdu = fits.PrimaryHDU(data=saveInfo)
    hdu.header['pa']    = angBest ; hdu.header['pa_err'] = angErr
    hdu.header['z_cor'] = z_cor   ; hdu.header['vsyst']  = vSyst
    hdu.writeto('%s_pPXF_ST_%s.fits'%(Galaxy,temp), overwrite=True)
    
    saveInfo = np.array([rotate(img, angle=-angBest, reshape=False, order=0, mode='constant', cval=np.nan) for img in saveInfo])
    imgContR_st = rotate(imgCont, angle=-angBest, reshape=False, order=0, mode='constant', cval=np.nan)

    # --------------------------------
    if 'Red' in pathDIR:
      string = '%s : Purely stellar - Red spectral range'%Galaxy
    elif 'Blue' in pathDIR:
      string = '%s : Purely stellar - Blue spectral range'%Galaxy
    else:
      string = '%s : Purely stellar'%Galaxy

    if NMOM == 4: 
      fig, ax = plt.subplots(2, 2, figsize=(9, 7))
      idxs = list(np.arange(0,4))
      titles = ['Rotation Velocity', 'Velocity Dispersion', 'h3', 'h4']

    if NMOM == 6: 
      fig, ax = plt.subplots(3, 2, figsize=(8.5, 10.5))
      idxs = list(np.arange(0,6))
      titles = ['Rotation Velocity', 'Velocity Dispersion', 'h3', 'h4', 'h5', 'h6']

    img0 = saveInfo[0] ; img1 = saveInfo[1]
    v_sig_ST = np.abs(img0)/img1 # V over sigma
    ny, nx = img0.shape
    extent = [(-xC)*psO, (nx-1-xC)*psO, (-yC)*psO, (ny-1-yC)*psO]

    fig.suptitle(string, fontsize=16)
    for ax_i, i, title in zip(ax.ravel(), idxs, titles):
      sym = (i == 0)
      imshow_with_cbar(ax_i, saveInfo[i], title, perc=(perc1, perc2), cmap=cmap, extent=extent, sym=sym)
      ax_i.contour(imgContR_st, levels=levels, colors='#5b5f61', linewidths=1, origin='lower', extent=extent)
    
    [ax.set(xlabel='″', ylabel='″') for ax in ax.ravel()]
  
    if R_bulge != None:
      cO_arc = ((cO[0] - nx/2) * psO, (cO[1] - ny/2) * psO)
      for ax in fig.axes: ax.add_patch(Circle(cO_arc, R_bulge, color='#899499', fill=False, linewidth=1))
    
    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, hspace=0.35, wspace=0.15)
    plt.savefig('%s_pPXF_ST_%s.png'%(Galaxy,temp)) ; plt.close()

  # --------------------------------    
  # V over sigma plots
  # --------------------------------
 
  if GST_STP == 'GST':

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4))

    im1 = ax1.imshow(v_sig_G, cmap='RdBu', origin='lower', vmin=np.nanpercentile(v_sig_G, perc1), vmax=np.nanpercentile(v_sig_G, perc2), extent=extent)
    ax1.contour(imgContR_g, levels=levels, colors='#5b5f61', linewidths=1, origin='lower', extent=extent)
    ax1.set_title('Stars + gas')

    im2 = ax2.imshow(v_sig_ST, cmap='RdBu', origin='lower', vmin=np.nanpercentile(v_sig_ST, perc1), vmax=np.nanpercentile(v_sig_ST, perc2), extent=extent)
    ax2.contour(imgContR_st, levels=levels, colors='#5b5f61', linewidths=1, origin='lower', extent=extent)
    ax2.set_title('Purely stellar')

    cbar1 = fig.colorbar(im1, ax=ax1) ; cbar2 = fig.colorbar(im2, ax=ax2)
    cbar1.set_label(r'V/$\sigma$')     ; cbar2.set_label(r'V/$\sigma$')

    [ax.set(xlabel='″', ylabel='″') for ax in [ax1, ax2]]
    
    if R_bulge != None:
      cO_arc = ((cO[0] - nx/2) * psO, (cO[1] - ny/2) * psO)  
      for ax in fig.axes: ax.add_patch(Circle(cO_arc, R_bulge, color='#899499', fill=False, linewidth=1))
    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9)
    plt.savefig('%s_pPXF_VrotSig_%s.png'%(Galaxy,temp)) ; plt.close()
  
  elif GST_STP == 'G':

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    im = ax.imshow(v_sig_G, cmap='RdBu', origin='lower', vmin=np.nanpercentile(v_sig_G, perc1), vmax=np.nanpercentile(v_sig_G, perc2), extent=extent)
    ax.contour(imgContR_g, levels=levels, colors='#5b5f61', linewidths=1, origin='lower', extent=extent)
    ax.set_title('Stars + gas')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'V/$\sigma$')

    ax.set(xlabel='″', ylabel='″')
    
    if R_bulge != None:
      cO_arc = ((cO[0] - nx/2) * psO, (cO[1] - ny/2) * psO)
      ax.add_patch(Circle(cO_arc, R_bulge, color='#899499', fill=False, linewidth=1))

    plt.subplots_adjust(bottom=0.12, left=0.12, right=0.88, top=0.88)
    plt.savefig('%s_pPXF_VrotSig_G_%s.png'%(Galaxy,temp))
    plt.close()

  elif GST_STP == 'ST':

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    im = ax.imshow(v_sig_ST, cmap='RdBu', origin='lower', vmin=np.nanpercentile(v_sig_ST, perc1), vmax=np.nanpercentile(v_sig_ST, perc2), extent=extent)
    ax.contour(imgContR_st, levels=levels, colors='#5b5f61', linewidths=1, origin='lower', extent=extent)
    ax.set_title('Purely stellar')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'V/$\sigma$')

    ax.set(xlabel='″', ylabel='″')
    
    if R_bulge != None:
      cO_arc = ((cO[0] - nx/2) * psO, (cO[1] - ny/2) * psO)
      ax.add_patch(Circle(cO_arc, R_bulge, color='#899499', fill=False, linewidth=1))

    plt.subplots_adjust(bottom=0.12, left=0.12, right=0.88, top=0.88)
    plt.savefig('%s_pPXF_VrotSig_ST_%s.png'%(Galaxy,temp))
    plt.close()    

  if len(WR_PPXF) > 1: 
    print('***********************************************************************************')
    
# ----------------------------------------
# Running pPXF using SINFONI
# ----------------------------------------

def kin_pPXF_SINF(Galaxy,df,wvl_dZ,dataCdZ,pathDIR,temp):
  
  global nbins,GST_STP,NMOM,BIN_DIR,N_MC

  try:
    GAL_INFO, SSP_DIR, STT_DIR, BIN_DIR, BAYES_DIR, CORES, NEB_CONT, N_MCS, SPEC_LIM, WR_PPXF, TEMPS_PPXF, LINE_STP, NMOM, GST_STP, CHI2_TYPE, TODO_PDYN, HST_FILT, MGE_M, SINF, TEMP_BAYES, REGUL, WR_BAYES, OCUT, V_SIGMA, ERR_DSBP, TEMPS_FD, LIBS_FD, TEMPS_SL, LIBS_SL, SL_CONF, RY_SSC, RY_TCS, TEMPS_SINF, NMOM_SINF = read_config()
  except Exception as e:
    print(f' × Problem in read_config(): {e} ×')
    print(' × Check your config file ×')
    sys.exit(1)
    
  NMOM = NMOM_SINF
  N_MC, _ = N_MCS
  
  if CORES == 0: CORES = os.cpu_count() // 2
  mc = True if N_MC > 0 else False
    
  if 'XSL' in temp:
    # XSL spectral res -> https://arxiv.org/pdf/2110.10188.pdf (R~8000 at 2.2? -> 2.75 Å)
    FWHM_temp = 2.75
  if 'GNIRS' in temp:
    # GNIRS spectral res -> https://iopscience.iop.org/article/10.1088/0067-0049/185/1/186 (R~6000 at 2.335 μm using the 111 mm grating)
    FWHM_temp = 3.89
    m = 2.23 ; M = 2.40
  if 'NIFS' in temp:
    # NIFS spectral res -> https://iopscience.iop.org/article/10.1088/0067-0049/185/1/186 (R~5290 at 2.2 μm)
    FWHM_temp = 4.16
    m = 2.189 ; M = 2.406

  # --------------------------------
  # Find center of the galaxy
  maps = fits.open('%s_maps_SINF.fits'%Galaxy) ; maps = maps[0].data
  STRcont = maps[1]

  ny, nx = STRcont.shape
  x0 = (nx - 1)/2. ; y0 = (ny - 1)/2.
  r = ny/2.
  yy, xx = np.indices(STRcont.shape)
  ma = (xx - x0)**2 + (yy - y0)**2 <= r**2

  tmp = STRcont.copy() ; tmp[~ma] = np.nan

  ctr = np.unravel_index(np.nanargmax(tmp), tmp.shape)
  yC, xC = ctr

  # --------------------------------
  # Start preparing the spectra
  # --------------------------------
  
  v = min(abs(wvl_dZ - m)) ; n1 = list(abs(wvl_dZ - m)).index(v) ; n1 = n1 + 1
  v = min(abs(wvl_dZ - M)) ; n2 = list(abs(wvl_dZ - M)).index(v) ; n2 = n2 - 1
  lam_gal = wvl_dZ[n1:n2]*1e4
  
  spectra = [] ; error = [] ; x = [] ; y = []
  for i in range(len(df['x'].values)):
    if df['BinVor'][i] != -999:
      xI = df['x'][i] ; yI = df['y'][i]
      f = dataCdZ[xI,yI][n1:n2]
      # Poisson error
      e = np.where(f >= 1, np.sqrt(abs(f)), np.sqrt(abs(f + 1)))
      x.append(xI) ; y.append(yI)
      spectra.append(f)
      error.append(e)
  
  velscale = C*np.diff(np.log(lam_gal[-2:]))
  lamRange = [m*1e4, M*1e4]
  LogSpec, logLam, velscale = util.log_rebin(lamRange, spectra[0], velscale=velscale)

  # ---------------------------------------
  
  binInfo = df['BinVor'].values ; binInfo = binInfo[binInfo != -999]
  nbins = int(np.nanmax(binInfo)) + 1
  CORES = min(CORES, 100, nbins)
  
  # Create an array with all bins
  specBIN, errBIN, BIN_L, norm_fact = ([],[],[],[])
  print('*                             Constructing the bins                               *')
  for j in range(nbins):
    try:
      w = binInfo == j
      if np.sum(w.astype(int)) > 0:
        
        spectrum = np.sum(np.array(spectra).T[:,w], 1)
        LogSpec, logLam, velscale = util.log_rebin(lamRange, spectrum, velscale=velscale)
        spectrum = LogSpec/np.median(LogSpec) # normalize spectrum to be at the same level of the stellar templates
        noise    = np.sqrt(np.sum(np.array(error).T[:,w]**2, 1))
        LogNoise, _, _ = util.log_rebin(lamRange, noise, velscale=velscale)
        noise = np.copy(LogNoise)
        
        noise[noise < 0] = 0
        if np.isnan(noise).any() == True:
          np.nan_to_num(noise, copy=False, nan=np.nanmax(noise)*10)
        
        if np.sum(spectrum)/len(spectrum) > 0.1:
          specBIN.append(spectrum)
          errBIN.append(noise)
          norm_fact.append(np.median(LogSpec))
          BIN_L.append(j)
    except: pass
  
  # ----------------------------
  # Setup the stellar templates 
  # ----------------------------
  
  nbins = len(BIN_L) ; nb = nbins
  files = glob.glob(os.path.join(str(STT_DIR), str(temp), '*.fits'))
  st_lib = build_library(files, velscale, FWHM_temp, None, norm_range=None)
  stars_templates, ln_lam_temp = st_lib.templates, st_lib.ln_lam_temp

  startST = [0, 200.] # (km/s), starting guess for [V,σ]

  lam_range_temp = np.exp(ln_lam_temp[[0, -1]])
  goodpixels0 = util.determine_goodpixels(logLam, lam_range_temp, 0)
  lam_gal = np.exp(logLam)
  
  component = [0]*np.shape(stars_templates)[1] # stellar template
  # Set the moments (4 for stars)
  moments = [NMOM_SINF]
  # [[V0, sig0, h3, h4] - stellar kinematics
  if NMOM_SINF == 4: start = [[-200, 200.0, 0, 0]]
  if NMOM_SINF == 6: start = [[-200, 200.0, 0, 0, 0, 0]]
  deg = 6
  # ------------------------------------------      

  # Fit kinematics on Voronoi binned data
  print('-----------------------------------------------------------------------------------')
  print('*                               Running pPXF in ||                                *')
  print('-----------------------------------------------------------------------------------')
  print(' • Spectral Library : %s'%temp)
  print(' • Fitting range    : %.2f to %.2f µm'%(m,M))
  print('-----------------------------------------------------------------------------------')
  
  # NOTE: For debuging, remove logging_level & log_to_driver
  if nbins % CORES != 0: CORES = min(nbins, CORES)
  ray.init(num_cpus=int(CORES),logging_level=logging.ERROR,log_to_driver=False)
  if nbins % 2 != 0: nb = nbins + 1
  jstart = 0 ; processes = []
  
  for c in range(CORES):
    jfin = jstart + round(nb/CORES)
    p = workerPPXF.remote(c,mc,stars_templates,specBIN,errBIN,norm_fact,velscale[0],startST,goodpixels0,lam_gal,st_lib,None,start,moments,deg,component,None,None,Galaxy,pathDIR+temp,None,None,jstart,jfin,BIN_L,GST_STP,NMOM,N_MC)
    processes.append(p)
    jstart = jfin
  ray.get(processes)
  ray.shutdown()

  # -------------------------------------------------------------------------------

  # create empty images : layer 0 - Vel ; layer 1 - Sig ; layer 2 - h3  ; layer 3 - h4
  xL, yL, _ = np.shape(dataCdZ)
  KINSTimgs = np.full((NMOM_SINF,xL,yL), np.nan)
  # ---------------------------------------
  
  # Open pkl files and merge them together
  ppxf_res = []
  for c in range(CORES):
    with open('%s%s/ppxf_res_%03d.pkl'%(pathDIR,temp,c), 'rb') as f:
      content = pickle.load(f)
      for l in range(len(content)):
        ppxf_res.append(content[l])
    
  # ---------------------------------------

  # Save the pickle files
  ppxf_res1F = []
  for i in range(len(ppxf_res)):
    ppxf_res1F.append(ppxf_res[i][0])
          
  with open('%s_pPXF_SINF_%s.pkl'%(Galaxy,temp),'wb') as f:
    pickle.dump(ppxf_res1F, f)
  
  X, Y , V, eV = ([],[],[],[])
  for j in range(len(ppxf_res1F)):
    bn = ppxf_res1F[j][0]
    solST = ppxf_res1F[j][7]
    idx = df.index[df['BinVor'] == bn].tolist()
    err = ppxf_res1F[j][8]
    for i in idx:
      xI = df['x'][i] ; yI = df['y'][i]
      KINSTimgs[0,xI,yI] = solST[0] ; KINSTimgs[1,xI,yI] = solST[1]
      KINSTimgs[2,xI,yI] = solST[2] ; KINSTimgs[3,xI,yI] = solST[3]
      if NMOM_SINF == 6: KINSTimgs[4,xI,yI] = solST[4] ; KINSTimgs[5,xI,yI] = solST[5]
      # for kin_pa
      X.append(df['x'][i]) ; Y.append(df['y'][i])
      V.append(solST[0]) ; eV.append(err[0])          
    
  # ---------------------------------------------------------------------------------------------      

  # Estimate the correction on the systemic velocity
  print('*               Fit and subtract the global kinematic position angle              *')
  print('-----------------------------------------------------------------------------------')
  print(' ➞ Purely stellar')
  print('-----------------------------------------------------------------------------------')
  try:
    x = np.array(X) ; y = np.array(Y) ; z = np.array(V) 
    z = z - np.nanmedian(z) ; eV = np.array(eV)
    nanM = np.isnan(x) | np.isnan(y) | np.isnan(z) ; nanM = ~nanM
    plt.close('all') ; plt.figure()
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
      angBest,angErr,vSyst = fit_kinematic_pa.fit_kinematic_pa(x[nanM]-xC, y[nanM]-yC, z[nanM], dvel=eV[nanM], nodots=True)
    plt.savefig('%s_SINF_pa.png'%Galaxy) ; plt.close()
    print(' • Kin PA %.2f +/- %.2f (3σ error) '%(angBest,angErr))
    print(' • Systemic velocity of %.2f km/s'%vSyst)
  except Exception as e:
    print(f' × Couldnt fit PA × : {e}')
    vSyst   = np.nanmedian(z)
    angBest = 0 ; angErr = 0
  z_cor = vSyst/C
  if len(TEMPS_SINF) > 1 and temp == TEMPS_SINF[0]: 
    print('-----------------------------------------------------------------------------------')
  
  # ---------------------------------------------------------------------------------------------

  KINSTimgs[KINSTimgs == 0] = np.nan
  
  m1 = KINSTimgs[0] ; m2 = KINSTimgs[1]
  m3 = KINSTimgs[2] ; m4 = KINSTimgs[3]
    
  if NMOM_SINF == 6:
    m5 = KINSTimgs[4] ; m6 = KINSTimgs[5]
    saveInfo = np.array([m1 - vSyst,m2,m3 - np.nanmedian(m3),m4,m5 - np.nanmedian(m5),m6])
  else:
    saveInfo = np.array([m1 - vSyst,m2,m3 - np.nanmedian(m3),m4])
    
  hdu = fits.PrimaryHDU(data=saveInfo)
  hdu.header['pa']    = angBest ; hdu.header['pa_err'] = angErr
  hdu.header['z_cor'] = z_cor   ; hdu.header['vsyst']  = vSyst
  hdu.writeto('%s_pPXF_SINF_%s.fits'%(Galaxy,temp), overwrite=True)
  saveInfo = np.array([rotate(img, angle=-angBest, reshape=False, order=0, mode='constant', cval=np.nan) for img in saveInfo])
  
  img0 = saveInfo[0]
  ny, nx = img0.shape
  extent = [(-xC)*psS, (nx-1-xC)*psS, (-yC)*psS, (ny-1-yC)*psS]
  
  string = 'Purely stellar'
  # --------------------------------
  
  if NMOM == 4: 
    fig, ax = plt.subplots(2, 2, figsize=(7.5, 6))
    idxs = [0, 1, 2, 3]
    titles = ['Rotation Velocity', 'Velocity Dispersion', 'h3', 'h4']

  if NMOM == 6: 
    fig, ax = plt.subplots(3, 2, figsize=(7.5, 9))
    idxs = [0, 1, 2, 3, 4, 5]
    titles = ['Rotation Velocity', 'Velocity Dispersion', 'h3', 'h4', 'h5', 'h6']

  fig.suptitle('%s : %s'%(Galaxy,string), fontsize=16)
  for ax_i, i, title in zip(ax.ravel(), idxs, titles):
    sym = (i == 0)
    imshow_with_cbar(ax_i, saveInfo[i], title, perc=(perc1, perc2), cmap=cmap, extent=extent, sym=sym)
  
  [ax.set(xlabel='″', ylabel='″') for ax in ax.ravel()]
  
  if R_bulge != None:
    cO_arc = ((cO[0] - nx/2) * psO, (cO[1] - ny/2) * psO)    
    for ax in fig.axes: ax.add_patch(Circle(cO_arc, R_bulge, color='#899499', fill=False, linewidth=1))
  
  plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, hspace=0.35, wspace=0.15)
  plt.savefig('%s_pPXF_SINF_%s.png'%(Galaxy,temp)) ; plt.close()
  
  # --------------------------------

  # save output in respective folder
  os.system('mv %s_pPXF_SINF_%s.pkl  %s/'%(Galaxy,temp,pathDIR+temp))
  os.system('mv %s_pPXF_SINF_%s.fits %s/'%(Galaxy,temp,pathDIR+temp))
  os.system('mv %s_pPXF_SINF_%s.png  %s/'%(Galaxy,temp,pathDIR+temp))
  os.system('mv %s_SINF_pa.png  %s/'%(Galaxy,pathDIR+temp))
  os.system('rm %s/ppxf_res_*.pkl'%(pathDIR+temp))
