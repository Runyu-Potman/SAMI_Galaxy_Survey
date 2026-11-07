
# Utilities
import os
import sys
import pandas as pd
import numpy as np
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')
from random import random

# Extinction CCM89 function:
from astroquery.ipac.irsa.irsa_dust import IrsaDust
from dust_extinction.parameter_averages import CCM89
ext = CCM89(Rv=3.1)

# Astropy:
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval
zscale = ZScaleInterval()

# Plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import EllipseSelector
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
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

# Astroquery
from astroquery.ipac.ned import Ned

# Voronoi binning
from vorbin.voronoi_2d_binning import voronoi_2d_binning

# Read from config
from GLANCE_config import create_config

# ------------------------------------------------------------------------------------
# This script must run 1 time before running the pipeline
# This script is iteractive!
# It needs a file with the names of the galaxies
# It assumes your data-cubes are named {galaxy}_DC.fits
# It will update GAL_INFO with name,z,Av,ra,dec,minSNR,tSNR - *best to have a backup GAL_INFO file*
# To be run like:
# $ python run_1st_time.py GAL_INFO # where GAL_INFO should be the name of your file listing the galaxies you want to process
# ------------------------------------------------------------------------------------

# Determine the optimal minimum SNR:
def minSNR_func(minSNR,fig,ax):
  arr = np.copy(imgSNR) ; arr[arr < minSNR] = np.nan  
  ax.clear()
  ax.imshow(arr, origin='lower')
  fig.canvas.draw_idle()        
  print(' ➞ With the chosen min SNR (%s) this is the SNR map.'%minSNR)
  df = 'n'
  ans = input(' ⇒ Do you want to change the min SNR? (y/N) ').strip().lower()  or df
  if ans == 'y':
    minSNR = int(input(' ⇒ Please enter a new SNR: ')) 
    
  return minSNR,ans

# ---------------------

# Get coordinates, redshift and extinction from NED:
def getInfo(Galaxy):

  ned = Ned.query_object(Galaxy)
  ra = ned['RA'][0] ; dec = ned['DEC'][0]
  coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
  z = ned['Redshift'].value.data[0]
  
  ext = IrsaDust.get_extinction_table(coords)
  flt = ext['Filter_name'] == 'CTIO V' 
  Av = ext[flt]['A_SandF'].value[0]
  
  return ra,dec,z,Av

# ---------------------

# Choose a rectangular region in plot:
def onselect_rect(start, end):
  global mask, sel

  ny, nx = data.shape
  x, y = np.meshgrid(np.arange(nx), np.arange(ny))
  pix = np.vstack((x.ravel(), y.ravel())).T

  verts = np.array([
    [start.xdata, start.ydata],
    [start.xdata, end.ydata],
    [end.xdata,   end.ydata],
    [end.xdata,   start.ydata]])

  p = Path(verts)
  ind = p.contains_points(pix)
  mask = np.zeros(data.shape, dtype=int) + 1
  mask.flat[ind] = 0

  sel = True

# Choose an elliptical region in plot:
def onselect_ell(start, end):
  
  global mask,sel

  xc = (start.xdata + end.xdata) / 2
  yc = (start.ydata + end.ydata) / 2
  rx = abs(start.xdata - end.xdata) / 2
  ry = abs(start.ydata - end.ydata) / 2
  y, x = np.ogrid[:data.shape[0], :data.shape[1]]
  selected = (((x - xc) / rx) ** 2 + ((y - yc) / ry) ** 2) <= 1
  
  mask = np.zeros_like(data)
  mask[selected] = 1
  
  sel = True

# ---------------------

# Prepare MUSE data;
def prepMUSE(Galaxy,imgSNR,MSK,minSNR):
  
  xL,yL = np.shape(imgSNR)
  xLST = [] ; yLST = [] ; SignalLST = [] ; NoiseLST = [] 
  for xI in range(xL):
    for yI in range(yL):
      
      if imgSNR[xI,yI] >= minSNR and MSK[xI,yI] == 0:
        xLST.append(xI) ; yLST.append(yI)  
        SignalLST.append(sigDC[xI,yI])
        NoiseLST.append(noiDC[xI,yI])
 
  df = pd.DataFrame()
  df['x'] = pd.Series(xLST)
  df['y'] = pd.Series(yLST)
  df['Signal']  = pd.Series(SignalLST)
  df['Noise']   = pd.Series(NoiseLST)
  df.to_csv('%s_VorBin.csv'%Galaxy)
  
  return df

# ---------------------

# run the Voronoi binning routine:
def runVoronoi(Maps,df,imgSNR,minSNR,tSNR):

  print('... Creating Voronoi maps for a target SNR of %s ...'%tSNR)

  xP = [] ; yP = [] ; bm = [] ; b = 0 ; n = 0
  for Map in Maps:
    n = n + 1
    xV = [] ; yV = [] ; sig = [] ; noise = []
    for x in range(Map.shape[0]):
      for y in range(Map.shape[1]):
        if (Map[x,y] != 0) & (imgSNR[x,y] > minSNR):
          try:
            idx = df.index[(df['x']==x) & (df['y']==y)].tolist()[0]
            xV.append(x) ; yV.append(y)
            sig.append(df['Signal'][idx])
            noise.append(df['Noise'][idx])
          except:
            pass
          
    xV = np.array(xV) ; yV = np.array(yV) ; sig = np.array(sig) ; noise = np.array(noise)
    try:
      
      binNum, _, _, _, _, _, _, _ = voronoi_2d_binning(xV,yV,sig,noise,tSNR,plot=False,quiet=True,pixelsize=1)
      bm.append(binNum + b)
      b = np.max(np.hstack(bm)) + 1
      xP.append(xV) ; yP.append(yV) 

    except Exception as e: 
      print(e)

  xMap = np.hstack(xP)
  yMap = np.hstack(yP)
  BinMap = np.hstack(bm)
  
  xL,yL = np.shape(Maps[0])
  imgBinMap  = np.zeros((xL,yL))
  for x,y,z in zip(xMap,yMap,BinMap):
    imgBinMap[x,y] = z
  
  nb = np.max(BinMap)
  return nb,imgBinMap,BinMap,xMap,yMap

# --------------------------------------

def zscale_image(img):
  
  vmin, vmax = zscale.get_limits(img)
  if vmin == 0: 
    m = img == 0
    vmin, vmax = zscale.get_limits(img[~m])
  img = np.nan_to_num(img, nan=vmin)
  img[img <= 0] = vmin
  
  imgL = np.log10(img)
  vmin, vmax = np.nanpercentile(imgL, [2, 99.9])
  imgL = np.nan_to_num(imgL, nan=vmin)
  
  return imgL, vmin, vmax

# --------------------------------------

# SINFONI 1st-time 
def run_1st_time_SINF(Galaxy):

  global data,imgSNR

  SINF_3D = Galaxy + '_SINF_DC.fits'
  print('')
  print(' ➞ Preparing %s (SINFONI):'%Galaxy)
  print(' • SINF 3D: %s'%SINF_3D)

  # Metadata from NED/IRSA
  RA, DEC, z, Av = getInfo(Galaxy)
  try:
    sinf_3D = fits.open(SINF_3D)
  except:
    print(' × ABORTING × %s cannot be found!!!'%SINF_3D)
    return

  # Data / error
  data = sinf_3D[0].data
  data = np.swapaxes(data,0,2) ; data = np.swapaxes(data,0,1)
  try:
    snE = 1
    err = sinf_3D[1].data
    err = np.swapaxes(err, 0, 2) ; data = np.swapaxes(data,0,1)
  except:
    snE = None
    print(' × No error DC found, taking the std of the signal!')

  head = sinf_3D[0].header
  zL  = head['NAXIS3']
  sW  = head['CRVAL3']
  rP  = head['CRPIX3']
  st  = head['CD3_3']
  wvl = (np.arange(zL) - (rP-1))*st + sW  # microns

  # De-redshift wavelengths
  wvl_dZ = wvl / (1 + z)

  # Vacuum -> air (Donald Morton 2000)
  s = 10**4 / (wvl * u.um.to(u.AA))
  n = 1 + 8.34254e-5 + 2.406147e-2/(130 - s**2) + 1.5998e-4/(38.9 - s**2)
  v = (wvl * u.um.to(u.AA)) / n
  wvl = v * u.AA.to(u.um)

  s = 10**4 / (wvl_dZ * u.um.to(u.AA))
  n = 1 + 8.34254e-5 + 2.406147e-2/(130 - s**2) + 1.5998e-4/(38.9 - s**2)
  v = (wvl_dZ * u.um.to(u.AA)) / n
  wvl_dZ = v * u.AA.to(u.um)

  # Remove negatives
  data[data < 0] = np.nan

  # Galactic extinction correction
  dataC = data / ext.extinguish(wvl * u.micron, Av=Av)
  if snE:
    errC = err / ext.extinguish(wvl * u.micron, Av=Av)

  # De-redshift flux
  dataCdZ = dataC * (1 + z)
  if snE:
    errCdZ = errC * (1 + z)

  # SNR window (rest-frame)
  n1 = int(np.argmin(abs(wvl_dZ - 2.189)))
  n2 = int(np.argmin(abs(wvl_dZ - 2.421)))

  sigDC = np.nanmean(dataCdZ[:, :, n1:n2+1], axis=2)
  noiDC = np.nanstd(dataCdZ[:, :, n1:n2+1], axis=2)

  noiDC[noiDC == 0] = 0.01
  imgSNR = sigDC / noiDC
  imgSNR[imgSNR > 1615] = np.nan
  imgSNR[imgSNR == 0] = np.nan
  
  ans1 = 'y' ; minSNR = 1
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.imshow(imgSNR, vmin=np.percentile(imgSNR,5), vmax=np.percentile(imgSNR,95), origin='lower')
  plt.show(block=False) ; plt.pause(1)
  while ans1 == 'y': minSNR, ans1 = minSNR_func(minSNR, fig, ax)
  plt.close()
  
  # Pseudo continuum 
  STRcont = np.nanmean(dataCdZ[:, :, n1:n2+1], axis=2)
  STRcont = np.array(STRcont)
  STRcont[STRcont <= 0] = np.nan

  xL, yL = np.shape(imgSNR)
  xLST = [] ; yLST = [] ; SignalLST = [] ; NoiseLST = []
  for xI in range(xL):
    for yI in range(yL):
      if imgSNR[xI, yI] >= minSNR: #and MSK[xI, yI] == 0:
        xLST.append(xI) ; yLST.append(yI)
        SignalLST.append(sigDC[xI, yI])
        NoiseLST.append(noiDC[xI, yI])

  tab = pd.DataFrame()
  tab['x'] = pd.Series(xLST)
  tab['y'] = pd.Series(yLST)
  tab['Signal'] = pd.Series(SignalLST)
  tab['Noise'] = pd.Series(NoiseLST)
  tab.to_csv('%s_VorBin_SINF.csv'%Galaxy)

  # Run Voronoi
  Map = np.zeros(STRcont.shape) + 1
  Maps = [Map]

  df = 'y'
  ans2 = input(' ⇒ What is the target SNR? ')
  print('')
  tSNR = int(ans2)

  ans = 'y'
  while ans == 'y': 
    
    nb,imgBinMap,BinMap,xMap,yMap = runVoronoi(Maps,tab,imgSNR,minSNR,tSNR)

    # Save Voronoi list
    xVor = [] ; yVor = [] ; BinVor = []
    for i in range(len(xMap)):
      if imgBinMap[xMap[i], yMap[i]] != -999:
        xVor.append(xMap[i])
        yVor.append(yMap[i])
        BinVor.append(BinMap[i])

    np.savetxt('%s_Voronoi_SINF_SN%.2i.dat'%(Galaxy,tSNR), np.column_stack([xVor,yVor,BinVor]), fmt=b'%8i %8i %8i')

    # Add BinNum to table
    for i in range(len(xVor)):
      idx = tab.index[(tab['x']==xVor[i]) & (tab['y']==yVor[i])].tolist()[0]
      try:
        if imgBinMap[xVor[i], yVor[i]] != -999:
          tab.at[idx, 'BinVor'] = BinVor[i]
      except:
        pass

    tab['BinVor'] = tab['BinVor'].fillna(-999).astype(int)
    tab.to_csv('%s_VorBin_SINF.csv'%Galaxy)

    # Plot quicklook
    img, vmin, vmax = zscale_image(STRcont)
    img[img == 0] = np.nan

    fig, ax = plt.subplots(1, 3, figsize=(9, 3.5))
    fig.suptitle('%s (SINFONI)'%Galaxy)
    ax[0].set_title('Signal to Noise')
    ax[0].imshow(imgSNR, origin='lower')
    ax[1].set_title('Pseudo continuum')
    ax[1].imshow(img, vmin=vmin, vmax=vmax, origin='lower')
    ax[2].set_title(r'Segmentation map (SNR %i)'%tSNR)
    colors = [(1,1,1)] + [(random(),random(),random()) for i in range(500)]
    new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=500)
    ax[2].imshow(imgBinMap, cmap=new_map, origin='lower')
    [a.set_axis_off() for a in ax.ravel()]
    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9)
    plt.tight_layout()
    plt.show(block=False) ; plt.pause(1)
    plt.savefig('%s_maps_SINF.png'%Galaxy)
    plt.pause(10)
    plt.close()

    print(' ➞ With the chosen target SNR (%s) this is the VorBin map, with %i bins.'%(tSNR, max(tab['BinVor'])))
    ans = input(' ⇒ Do you want to change the target SNR? (y/N) ').strip().lower() or 'n'
    if ans == 'y':
      tSNR = int(input(' ⇒ Input new target SNR: '))
      print('')

  saveInfo = np.array([imgSNR.T,STRcont.T,imgBinMap.T])
  imgs = np.transpose(saveInfo, (0, 2, 1))
  hdu = fits.PrimaryHDU(data=imgs)
  hdu.writeto('%s_maps_SINF.fits'%Galaxy, overwrite=True)

  if os.path.exists('%s_analysis'%Galaxy):
    os.system('cp %s_VorBin_SINF.csv %s_analysis/'%(Galaxy, Galaxy))
    os.system('cp %s_maps_SINF.fits %s_analysis/'%(Galaxy, Galaxy))
    os.system('cp %s_maps_SINF.png %s_analysis/'%(Galaxy, Galaxy))

  return minSNR,tSNR

# --------------------------------------
# --------------------------------------

print(' ')
print(' ★  ••••••••••••••••• ★')
print(' ★  WELCOME TO GLANCE ★')
print(' ★  ••••••••••••••••• ★')

try:
  GAL_INFO = sys.argv[1]
except:
  print(' ')
  GAL_INFO = input(' ⇒ Please insert the name of your file with the galaxies IDs: ')
  
create_config(GAL_INFO)

# --------------------------------------------------------------

galL = open(GAL_INFO, 'r')
lines = galL.readlines()

global data

# --------------------------------------------------------------

for l in range(len(lines)):
  
  line = lines[l]  
  nc = line.count(',')
  
  if nc < 6: 
  
    Galaxy = line.rstrip('\n')
    print('')
    print(' ➞ Preparing %s:'%Galaxy)
    if not os.path.exists('%s_analysis'%Galaxy): os.makedirs('%s_analysis'%Galaxy)
    
    MSKSN = None
    MUSE_3D = Galaxy + '_DC.fits'

    try:
      with fits.open(MUSE_3D) as hdul:
        head = hdul[1].header
        dist = head['distance']
        z = head['redshift']
        morph = head['morph']
        Av = head['Av']
        RA = head['ra'] ; DEC = head['dec']
    except: 
      RA,DEC,z,Av = getInfo(Galaxy)

    # --------------------------------------------------------------
    # Open and prepare the data-cube
    print(' • MUSE 3D: %s'%MUSE_3D)
    print(' • Redshift: %s'%z)
    print(' • Av: %s'%Av)
    print(' • Ra: %s'%RA)
    print(' • Dec: %s'%DEC)
    print('')
    
    try:
      muse_3D = fits.open(MUSE_3D)
    except:
      print(' × ABORTING × %s_DC.fits cannot be found!!!'%Galaxy)
      sys.exit()
      
    data = muse_3D[1].data
    data = np.swapaxes(data,0,2) ; data = np.swapaxes(data,0,1)
    try:
      snE = 1
      err  = muse_3D[2].data
      err  = np.swapaxes(err,0,2) ; err  = np.swapaxes(err,0,1)
    except:
      snE = None
      print(' × No error DC found, taking the std of the signal!')
    
    head = muse_3D[1].header
    xL  = head['NAXIS1'] # number of x pix
    yL  = head['NAXIS2'] # number of y pix
    zL  = head['NAXIS3'] # number of z pix
    sW  = head['CRVAL3'] # starting wavelength
    rP  = head['CRPIX3'] # ref pix lambda
    st  = head['CD3_3']  # wavelength step
    psM = abs(head['CD1_1'])*3600  # pixel size (arcsec/pix)
    wvl = (np.arange(zL) - (rP-1))*st + sW 
    
    # ------------------------
    # De-redshift the wavelength
    wvl_dZ = wvl / (1 + z)
    
    # Remove possible negative values:
    data[data < 0] = np.nan
    
    # Correct data-cube for galactic extinction
    dataC = data/ext.extinguish(wvl*u.AA, Av=Av)
    if snE : errC  = err/ext.extinguish(wvl*u.AA, Av=Av)

    # De-redshift the flux
    dataCdZ = dataC * (1 + z) # units of 10**-20 erg cm-2 s-1 AA-1
    if snE : errCdZ = errC * (1 + z)
    
    # check desired minSNR
    ans1 = 'y'
    sigDC  = np.nanmean(dataCdZ, axis=2)
    # the units of the error data-cube are (10**(-20)*erg/s/cm**2/Angstrom)**2 so one must take the sqrt
    if snE : 
      noiDC  = np.nanmean(np.sqrt(errCdZ), axis=2)
    else:
      # mask emission lines and other features like the calcium triplet
      emL = [4862.721, 4960.295, 5008.239, 5909.75, 6511, 6549.86, 6564.614, 6585.27, 6718.29, 6732.68, 6897.25, 7183.5, 7599.75, 8241, 8498, 8542, 8662]
      mask = np.ones_like(wvl_dZ, dtype=bool)

      for v in emL: mask &= ~((wvl_dZ >= v - 20) & (wvl_dZ <= v + 20))
      dt = dataCdZ[:, :, mask]
      
      noiDC = np.nanstd(dt, axis=2)
      noiDC[noiDC == 0] = 0.01
      
    imgSNR = sigDC/noiDC
    
    minSNR = 1
    fig = plt.figure() ; ax = fig.add_subplot()
    ax.imshow(imgSNR, origin='lower') 
    plt.show(block=False) ; plt.pause(1)
    
    while ans1 == 'y':
      minSNR,ans1 = minSNR_func(minSNR,fig,ax) # minimum SNR for the spaxel to be processed
    
    df = 'y'
    mask = np.zeros_like(imgSNR) ; mask = mask.astype(int)
    MSK  = np.zeros_like(imgSNR) ; MSK = MSK.astype(int)
    
    ans0 = input(' ⇒ Do you want to continue with this galaxy? (Y/n) ').strip().lower() or df
    if ans0 == 'y':

      # -----------------------------
      # Create (pseudo) stellar continuum Image from 6390 to 6490
      t = min(abs(wvl_dZ - 6540)) ; n1 = list(abs(wvl_dZ - 6540)).index(t)
      t = min(abs(wvl_dZ - 6580)) ; n2 = list(abs(wvl_dZ - 6580)).index(t)

      i = 0
      for j in range(n1, n2, 1):
        f = dataCdZ[:,:,j]
        i = i + f 

      STRcont = i/len(range(n1, n2, 1)) # average
      STRcont = np.array(STRcont)
      STRcont[STRcont < 0] = np.nan
      
      # -----------------------------
      
      df = 'n'
      ans1 = input(' ⇒ Do you want to select part of your galaxy? (y/N) ').strip().lower() or df
      data = np.copy(imgSNR) ; data[data < minSNR] = np.nan
      if ans1 == 'y':
        print(' ➞ You can now select the region in the SNR map')
        sel = False
        lasso = RectangleSelector(ax, onselect_rect)
        while not sel: plt.pause(0.1)

        STRcont = np.logical_not(mask).astype(int)*STRcont
        STRcont[STRcont == 0] = np.nan
        imgSNR = np.logical_not(mask).astype(int)*imgSNR
        imgSNR[imgSNR == 0] = np.nan 
        
        ax.clear()
        ax.imshow(imgSNR, origin='lower')
        fig.canvas.draw_idle()
        MSK = MSK + mask

      # -----------------------------
      # Mask MW stars, if existing
      data = np.copy(imgSNR)
      data[data < minSNR] = np.nan
      data[data == 0] = np.nan
      STARS = False
      ans1 = input(' ⇒ Do you want to mask MW stars? (y/N) ').strip().lower() or df
      if ans1 == 'y':
        STARS = True
        ans2 = input(' ⇒ How many stars you want to mask? ')
        print(' ➞ You can now select them in the SNR map')
        for i in range(0,int(ans2)):
          sel = False
          lasso = EllipseSelector(ax, onselect_ell)
          while not sel: plt.pause(0.1)
          MSK = MSK + mask.astype(int)
          data = data*np.logical_not(MSK).astype(int)
          data[data == 0] = np.nan
          ax.clear()
          ax.imshow(data, origin='lower')
          fig.canvas.draw_idle()        
          
      elif ans1 == 'n':
        plt.close()
      
      plt.pause(5) ; plt.close()
      hdu = fits.PrimaryHDU(data=MSK)
      if STARS: hdu.header['STARS'] = 1
      hdu.writeto('%s_mask.fits'%Galaxy, overwrite=True)

      imgSNR = np.logical_not(MSK).astype(int)*imgSNR
      imgSNR[imgSNR == 0] = np.nan
      STRcont = np.logical_not(MSK).astype(int)*STRcont
      STRcont[STRcont == 0] = np.nan
        
      # run Voronoi
      # -----------------------------
      
      tab = prepMUSE(Galaxy,imgSNR,MSK,minSNR)
      Map = np.zeros(STRcont.shape) + 1 ; Maps = [] ; Maps.append(Map)

      ans = 'y'
      ans2 = input(' ⇒ What is the target SNR? ') ; print('')
      tSNR = int(ans2)
      while ans == 'y':
        
        nb,imgBinMap,BinMap,xMap,yMap = runVoronoi(Maps,tab,imgSNR,minSNR,tSNR)

        # Saving the results
        #---------------------------
        xVor = [] ; yVor = [] ; BinVor = []
        for i in range(len(xMap)):
          if imgBinMap[xMap[i],yMap[i]] != -999:
            xVor.append(xMap[i])
            yVor.append(yMap[i])
            BinVor.append(BinMap[i])

        np.savetxt('%s_Voronoi_SN%.2i.dat'%(Galaxy,tSNR), np.column_stack([xVor, yVor, BinVor]), fmt=b'%8i %8i %8i')
        #---------------------------

        # Add BinNum info to the table
        for i in range(len(xVor)):
          idx = tab.index[(tab['x']==xVor[i]) & (tab['y']==yVor[i])].tolist()[0]
          try:
            if imgBinMap[xMap[i],yMap[i]] != -999:
              tab.at[idx,'BinVor'] = BinVor[i]
          except:
            pass

        tab['BinVor'] = tab['BinVor'].fillna(-999).astype(int)
        tab.to_csv('%s_VorBin.csv'%Galaxy)

        img,vmin,vmax = zscale_image(STRcont)
        img = img*np.logical_not(MSK).astype(int)
        img[img == 0] = np.nan
        
        fig, ax = plt.subplots(1, 3, figsize=(7.5, 3))
        fig.suptitle('%s'%Galaxy)
        ax[0].set_title('Signal to Noise')
        ax[0].imshow(imgSNR, origin='lower')
        ax[1].set_title('Stellar Continnum')
        ax[1].imshow(img, vmin=vmin, vmax=vmax, origin='lower')
        ax[2].set_title(r'Segmentation map (SNR %i)'%tSNR)
        colors = [(1,1,1)] + [(random(),random(),random()) for i in range(500)]
        new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=500)
        ax[2].imshow(imgBinMap, cmap=new_map, origin='lower')

        [ax.set_axis_off() for ax in ax.ravel()]
        plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9)
        plt.tight_layout()
        plt.show(block=False) ; plt.pause(1)
        plt.savefig('%s_maps.png'%Galaxy)
        plt.pause(10) ; plt.close()
        
        print(' ➞ With the chosen target SNR (%s) this is the VorBin map, with %i number of bins.'%(tSNR,max(tab['BinVor'])))
        if max(tab['BinVor']) > 400: print(' • Note that too many bins might take too long to process....')
        ans = input(' ⇒ Do you want to change the target SNR? (y/N) ').strip().lower() or df
        if ans == 'y':
          tSNR = int(input(' ⇒ Input new target SNR: ')) ; print('')
        plt.close()

      imgBinMap[imgBinMap == 0] = np.nan
      saveInfo = np.array([imgSNR.T,STRcont.T,imgBinMap.T])
      imgs = np.transpose(saveInfo, (0, 2, 1))
      hdu = fits.PrimaryHDU(data=imgs) ; hdu.writeto('%s_maps.fits'%Galaxy, overwrite=True)

      os.system('cp %s_VorBin.csv %s_analysis/'%(Galaxy,Galaxy))
      os.system('cp %s_maps.fits %s_analysis/'%(Galaxy,Galaxy))
      os.system('cp %s_maps.png %s_analysis/'%(Galaxy,Galaxy))

      # Optional: also run the SINFONI 1st-time prep
      SINF = input(' ⇒ Do you want to process SINFONI data too? (y/N) ').strip().lower() or df
      if SINF == 'y':
        minSNR_SINF,tSNR_SINF = run_1st_time_SINF(Galaxy)
        
        SMBH = input(' ⇒ Insert SMBH mass (log) if you intent to perform dyn modelling (leave it empty if you want to estimate it): ')
        if SMBH:
          lines[l] = '%s,%f,%f,%f,%f,%i,%i,%i,%i,%.2f\n'%(Galaxy,z,Av,RA,DEC,minSNR,tSNR,minSNR_SINF,tSNR_SINF,float(SMBH))
        else:
          lines[l] = '%s,%f,%f,%f,%f,%i,%i,%i,%i\n'%(Galaxy,z,Av,RA,DEC,minSNR,tSNR,minSNR_SINF,tSNR_SINF)
      else:        
        SMBH = input(' ⇒ Insert SMBH mass (log) if you intent to perform dyn modelling (leave it empty if you want to estimate it): ')
        if SMBH:
          lines[l] = '%s,%f,%f,%f,%f,%i,%i,%.2f\n'%(Galaxy,z,Av,RA,DEC,minSNR,tSNR,float(SMBH))
        else:
          lines[l] = '%s,%f,%f,%f,%f,%i,%i\n'%(Galaxy,z,Av,RA,DEC,minSNR,tSNR)

    else:
      plt.close('all')
  else:
    Galaxy = line.rstrip('\n').split(',')[0]
    print(' ')
    print(' ➞ %s seems to be already processed. Skipping...'%Galaxy)

# --------------------------------------------------------------

# Update the file:
lines = [line for line in lines if ',' in line]
with open(GAL_INFO, 'w') as f: f.writelines(lines)
