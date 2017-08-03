'''
Natalia@Falguiere - 26/Jan/2017
'''

# --------------------------------------------------------------------
from os import path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.table

from pycasso import fitsQ3DataCube, EmLinesDataCube
from pycasso2.reddening import CCM
import natastro.plotutils as nplot
import natastro.utils as utils
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Debug?
debug = False
saveTable = True

# Version?
# version = 'px1'
version = 'q050'

# Galaxies to run
#gal_ids = ['K0010', 'K0023', 'K0025', 'K0028', 'K0073']
#gal_ids = ['K0010']

list_file = '/Users/natalia/data/CALIFA/Tables/dig_gals_sim.txt'
aux = astropy.table.Table.read(list_file, format = 'ascii.fast_no_header')
gal_ids = aux['col1']
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Read QH for base
fileQH = '/Users/natalia/data/SSPs/K6BaseDir/BASE.gsd6e.square.QH'

t = astropy.table.Table.read(fileQH, format = 'ascii.fixed_width_two_line')
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Define directories and file names

if version == 'q050':
    fits_dir = '/Users/natalia/data/califa/IAA/legacy/q050/%s/Bgsd6e/'
    emlines_subdir = 'EML'
    pycasso_subdir = 'superfits'
    emlines_suffix = '.EML.MC100'
    pycasso_suffix = ''
    file_name = '%s_synthesis_eBR_v20_q050.d15a512.ps03.k1.mE.CCM.Bgsd6e%s.fits'

if version == 'px1':
    fits_dir = '/Users/natalia/data/califa/IAA/legacy/q057/%s/px1Bgstf6e/'
    emlines_subdir = 'EML'
    pycasso_subdir = 'superfits'
    emlines_suffix = '.EML.MC100'
    pycasso_suffix = ''
    file_name = '%s_synthesis_eBR_px1_q057.d22a512.ps03.k1.mE.CCM.Bgstf6e%s.fits'
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Calculate

# Start a table to keep the results
outtable = astropy.table.Table({'CALIFAID': gal_ids})
for outvar in ['LHa_Exp__G', 'LHa_Exp_POP5__G',
               'El_O_6563__HII', 'El_O_6563__DIG', 'El_O_6563__G',
               'El_I_6563__HII', 'El_I_6563__DIG', 'El_I_6563__G', 
               'El_IC_6563__HII', 'El_IC_6563__G', 
               'HaHbC__HII', 'HaHbC__G',
               'tauT_6563__HII', 'tauC_6563__HII', 'tauC_6563__G']:
    outtable[outvar] = np.zeros(len(gal_ids))


# Calculate galaxy by galaxy
for ig, gal_id in enumerate(gal_ids):

    # Define file names
    pycasso_file = path.join(fits_dir % pycasso_subdir, file_name % (gal_id, pycasso_suffix))
    emlines_file = path.join(fits_dir % emlines_subdir, file_name % (gal_id, emlines_suffix))
    
    # Read the pycasso and the emission line cubes
    c = fitsQ3DataCube(pycasso_file)
    c.loadEmLinesDataCube(emlines_file)

    
    # Get the number of ionizing photons for the base
    shape__tZ = c.popmu_ini__tZyx.shape[:2]
    log_QH_base__tZ = t['log_QH'].reshape(shape__tZ)
    
    # This is just to check that the reshaping is doing the proper thing
    t__tZ = t['age_base'].reshape(shape__tZ)
    Z__tZ = t['Z_base'].reshape(shape__tZ)
    check_t = np.all(np.isclose(c.ageBase, 10**t__tZ[..., 0], rtol=1.e-5, atol=1e-5))
    check_Z = np.all(np.isclose(c.metBase, Z__tZ[0, ...], rtol=1.e-5, atol=1e-5))
    if (not check_t) | (not check_Z):
        sys.exit('Please check your square base.')

    # Calc the total percentage of popini (~100%) in each zone 
    norm_popini__z = c.popmu_ini.sum(axis = (0,1))
    popmu_ini_frac__tZz = c.popmu_ini/norm_popini__z
    
    # Calc the number of ionizing photons per stellar mass for different stellar populations
    age_base_low   = [0     ,  1.01e7 ,  1.016e8,  1.279e9,  9.99e7]
    age_base_upp   = [1.01e7,  1.016e8,  1.279e9,  1.00e20, 1.00e20]

    log_qH__z = utils.safe_log10( np.sum(popmu_ini_frac__tZz * 10**log_QH_base__tZ[..., np.newaxis], axis = (0, 1)) )
    
    flag_POP5__t = (c.ageBase >= age_base_low[-1]) & (c.ageBase < age_base_upp[-1])
    log_qH_POP5__z = utils.safe_log10( np.sum(popmu_ini_frac__tZz[flag_POP5__t] * 10**log_QH_base__tZ[flag_POP5__t, :, np.newaxis], axis = (0, 1)) )
    
    # Calc Q by multiplying q by the mass
    log_QH__z = log_qH__z + np.log10(c.Mini__z)
    log_QH_POP5__z = log_qH_POP5__z + np.log10(c.Mini__z)


    # Transform in Ha
    clight  = 2.99792458  # * 10**18 angstrom/s
    hplanck = 6.6260755   # * 10**(-27) erg s
    lsun    = 3.826       # * 10**33 erg/s
    _k_q    = np.log10(lsun / (clight * hplanck)) + 33 + 27 - 18
    _k0     = 1. / (2.226 * 6562.80)

    LHa_Exp__z     = _k0 * 10**(log_QH__z - _k_q)
    log_LHa_Exp__z = np.log10(LHa_Exp__z)
    
    LHa_Exp_POP5__z     = _k0 * 10**(log_QH_POP5__z - _k_q)
    log_LHa_POP5_Exp__z = np.log10(LHa_Exp_POP5__z)
    
    # Transform into xy
    log_QH__yx = utils.safe_log10( c.zoneToYX( 10**log_QH__z / c.zoneArea_pix, extensive = False) )
    log_QH_POP5__yx = utils.safe_log10( c.zoneToYX( 10**log_QH_POP5__z / c.zoneArea_pix, extensive = False) )

    log_LHa_Exp__yx = utils.safe_log10( c.zoneToYX( LHa_Exp__z / c.zoneArea_pix, extensive = False) )
    log_LHa_Exp_POP5__yx = utils.safe_log10( c.zoneToYX( LHa_Exp_POP5__z / c.zoneArea_pix, extensive = False) )
    

    # Now calculate the Ha expected for the entire galaxy
    LHa_Exp__G      = np.sum(LHa_Exp__z)
    LHa_Exp_POP5__G = np.sum(LHa_Exp_POP5__z)


    
    # Read Halpha and Hbeta
    El_F_6563__z  = c.EL.flux[c.EL.lines.index('6563')]
    El_F_6563__yx = c.zoneToYX( El_F_6563__z / c.zoneArea_pix, extensive = False )
    
    El_F_4861__z  = c.EL.flux[c.EL.lines.index('4861')]
    El_F_4861__yx = c.zoneToYX( El_F_4861__z / c.zoneArea_pix, extensive = False )

    log_4PId2  = np.log10(4 * np.pi * c.distance_Mpc**2) + 2 * np.log10(1e6 * 3.086e18)
    El_L_6563__z  = El_F_6563__z * 10**log_4PId2 / 3.826e33
    El_L_4861__z  = El_F_4861__z * 10**log_4PId2 / 3.826e33
    
    El_EW_6563__z  = c.EL.EW[c.EL.lines.index('6563')]
    El_EW_6563__yx = c.zoneToYX( El_EW_6563__z, extensive = False )

    
    # Separate HII regions
    EWHa_lim = 6
    flag_HII__z = (El_EW_6563__z > EWHa_lim)
    flag_DIG__z = (El_EW_6563__z <= EWHa_lim) & (El_EW_6563__z > 0)

    flag_HII__yx = c.zoneToYX( flag_HII__z, extensive = False )
    flag_DIG__yx = c.zoneToYX( flag_DIG__z, extensive = False )
    
    # Calculate the intrinsic Halpha flux for individual zones
    D_HII = 2.86
    HaHb__z = El_F_6563__z / El_F_4861__z
    tau_6563__z = np.power(CCM(4861) - CCM(6563), -1) * utils.safe_ln( HaHb__z / D_HII )
    tau_6563__yx = c.zoneToYX( tau_6563__z, extensive = False )

    # Do a very basic sanity-control check in tau (see zone 25 in galaxy K0010)
    tau_6563__z[tau_6563__z > 10.] = 0.
    El_I_6563__z = np.where(tau_6563__z <= 0, El_L_6563__z, El_L_6563__z * np.exp(tau_6563__z))
    El_I_6563__yx = c.zoneToYX( El_I_6563__z / c.zoneArea_pix, extensive = False )

    
    # Now calculate intrinsic Halpha for the HII regions, zone by zone
    El_I_6563__HII = np.sum(El_I_6563__z[flag_HII__z])
    
    # And calculate the intrinsic Halpha from the `HII' spectrum (C = coadded)
    El_O_6563__HII = np.sum(El_L_6563__z[flag_HII__z])
    El_O_4861__HII = np.sum(El_L_4861__z[flag_HII__z])
    HaHbC__HII = El_O_6563__HII / El_O_4861__HII
    tauC_6563__HII = np.power(CCM(4861) - CCM(6563), -1) * utils.safe_ln( HaHbC__HII / D_HII )
    El_IC_6563__HII = El_O_6563__HII * np.exp(tauC_6563__HII)


    # Now calculate intrinsic Halpha for the DIG regions, zone by zone
    El_I_6563__DIG = np.sum(El_I_6563__z[flag_DIG__z])
    
    # And calculate the intrinsic Halpha from the `DIG' spectrum (C = coadded)
    D_DIG = 2.86
    El_O_6563__DIG = np.sum(El_L_6563__z[flag_DIG__z])
    El_O_4861__DIG = np.sum(El_L_4861__z[flag_DIG__z])
    HaHbC__DIG = El_O_6563__DIG / El_O_4861__DIG
    tauC_6563__DIG = np.power(CCM(4861) - CCM(6563), -1) * utils.safe_ln( HaHbC__DIG / D_DIG )
    El_IC_6563__DIG = El_O_6563__DIG * np.exp(tauC_6563__DIG)

    
    # We can also calculate the `effective' tau
    tauT_6563__HII = - np.log(El_O_6563__HII / El_I_6563__HII)

    # And we repeat the intrinsic Halpha for the whole global spectrum
    El_O_6563__G = np.sum(El_L_6563__z)
    El_O_4861__G = np.sum(El_L_4861__z)
    HaHbC__G = El_O_6563__G / El_O_4861__G
    tauC_6563__G = np.power(CCM(4861) - CCM(6563), -1) * np.log( HaHbC__G / D_HII )
    El_IC_6563__G = El_O_6563__G * np.exp(tauC_6563__G)

    # Now calculate intrinsic Halpha for the all regions, zone by zone
    El_I_6563__G = np.sum(El_I_6563__z)

    
    # Save info to the astropy table
    #print El_I_6563__HII, El_IC_6563__HII, El_IC_6563__HII / El_I_6563__HII  
    #print HaHbC__HII, tauT_6563__HII, tauC_6563__HII, tauC_6563__HII / tauT_6563__HII
    outtable['LHa_Exp__G'][ig]      = LHa_Exp__G     
    outtable['LHa_Exp_POP5__G'][ig] = LHa_Exp_POP5__G
    outtable['El_O_6563__HII'][ig]  = El_O_6563__HII 
    outtable['El_O_6563__DIG'][ig]  = El_O_6563__DIG 
    outtable['El_O_6563__G'][ig]    = El_O_6563__G   
    outtable['El_I_6563__HII'][ig]  = El_I_6563__HII
    outtable['El_I_6563__DIG'][ig]  = El_I_6563__DIG
    outtable['El_I_6563__G'][ig]    = El_I_6563__G
    outtable['El_IC_6563__HII'][ig] = El_IC_6563__HII
    outtable['El_IC_6563__G'][ig]   = El_IC_6563__G
    outtable['HaHbC__HII'][ig]      = HaHbC__HII
    outtable['HaHbC__G'][ig]        = HaHbC__G
    outtable['tauT_6563__HII'][ig]  = tauT_6563__HII
    outtable['tauC_6563__HII'][ig]  = tauC_6563__HII
    outtable['tauC_6563__G'][ig]    = tauC_6563__G

    
# Save table to a file
if saveTable:
    outtable.write('calc_Ha_tau.txt', format = 'ascii.fixed_width_two_line')
# --------------------------------------------------------------------




#=========================================================================
# ===> Set up plots for debugging

columnwidth = 240.
textwidth = 504.
screenwidth = 1024.
#psetup = nplot.plotSetupMinion
psetup = nplot.plotSetup

if debug:

    psetup(fig_width_pt=0.8*screenwidth, aspect=.6, lw=1., fontsize=15)
    fig = plt.figure(1)
    plt.clf()
    
    
    gs = gridspec.GridSpec(4, 4)
    
    #cmap = sns.set_hls_values('k', l=1.)
    #colours = [sns.set_hls_values('k', l=l) for l in np.linspace(1, 0.0, 12)]
    #cmap = sns.blend_palette(colours, as_cmap=True)
    
    
    # Ha surface brightness
    ax = plt.subplot(gs[0, 0])
    plt.title(r'$\Sigma_\mathrm{H\alpha}$')
    im = plt.pcolormesh(El_F_6563__yx, vmax = 1e-15) #, vmin = -17, vmax = -14, cmap = cmap, zorder = 0)
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    
    # Hb surface brightness
    ax = plt.subplot(gs[0, 1])
    plt.title(r'$\Sigma_\mathrm{H\beta}$ [erg s$^{-1}$ \AA$^{-1}$ cm$^{2}$ px$^{-1}$]')
    im = plt.pcolormesh(El_F_4861__yx) #, vmin = -17, vmax = -14, cmap = cmap, zorder = 0)
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    
    # EW Ha
    ax = plt.subplot(gs[0, 2])
    plt.title(r'$\mathrm{EW}_\mathrm{H\alpha}$ [\AA]')
    im = plt.pcolormesh(El_EW_6563__yx) #, vmin = -17, vmax = -14, cmap = cmap, zorder = 0)
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    
    # HII regions
    ax = plt.subplot(gs[0, 3])
    plt.title(r'$\mathrm{EW}_\mathrm{H\alpha}$ [\AA]')
    im = plt.pcolormesh(flag_HII__yx) #, vmin = -17, vmax = -14, cmap = cmap, zorder = 0)
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    
    # Intrinsic surface brightness
    ax = plt.subplot(gs[1, 0])
    plt.title(r'$\log \Sigma_\mathrm{H\alpha}$ dr')
    im = plt.pcolormesh(utils.safe_log10(El_I_6563__yx))
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    
    # tau
    ax = plt.subplot(gs[1, 1])
    plt.title(r'$\tau_\mathrm{H\alpha}$')
    im = plt.pcolormesh(tau_6563__yx, vmin = 0, vmax = 5)
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    
    # Ha/Hb
    ax = plt.subplot(gs[1, 2])
    plt.title(r'$\mathrm{H\alpha}/\mathrm{H\beta}$')
    im = plt.pcolormesh(El_F_6563__yx / El_F_4861__yx, vmax = 10)
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])


    # Q(H)s
    ax = plt.subplot(gs[2, 0])
    plt.pcolormesh(log_QH__yx, vmin=49, vmax=54)
    plt.colorbar()
    plt.title(r'log Q(H) all')
        
    ax = plt.subplot(gs[2, 1])
    plt.pcolormesh(log_QH_POP5__yx, vmin=49, vmax=54)
    plt.colorbar()
    plt.title(r'log Q(H) old')
    
    ax = plt.subplot(gs[2, 2])
    plt.pcolormesh(log_LHa_Exp__yx, vmin=4, vmax=8)
    plt.colorbar()
    plt.title(r'log LHa exp all')
        
    ax = plt.subplot(gs[2, 3])
    plt.pcolormesh(log_LHa_Exp_POP5__yx, vmin=4, vmax=8)
    plt.colorbar()
    plt.title(r'log LHa exp old')

    fig.tight_layout()
#=========================================================================

