import os
import sys
import time
import numpy as np
import matplotlib as mpl
from pytu.lines import Lines
from matplotlib import pyplot as plt
from pytu.functions import debug_var
from scipy.interpolate import interp1d
from CALIFAUtils.scripts import ma_mask_xyz
from CALIFAUtils.plots import DrawHLRCircle
from pystarlight.util.constants import L_sun
from CALIFAUtils.plots import plot_gal_img_ax
from CALIFAUtils.scripts import read_one_cube
from mpl_toolkits.axes_grid1 import make_axes_locatable

debug = False  # DEBUG MODE OFF
debug = True   # DEBUG MODE ON

mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
dflt_kw_scatter = dict(marker='o', s=1, edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, debug=True, gs_prc=True, poly1d=True)
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal')
img_dir = '%s/CALIFA/images/' % os.environ['HOME']


def plot_OH(ax, distance_HLR__r, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range):
    x = np.ma.ravel(distance_HLR__yx)
    y = np.ma.ravel(OH__yx)
    ax.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, **dflt_kw_scatter)
    ax.set_xlabel(r'R [HLR]')
    ax.set_ylabel(r'$12\ +\ \log$ (O/H) - %s [Z${}_\odot$]' % OH_label)
    ax.grid()
    ax.set_xlim(distance_range)
    ax.set_ylim(OH_range)
    debug_var(debug, OH_label=OH_label, MAX_OH__yx=OH__yx.max(), MIN_OH__yx=OH__yx.min())
    return ax


def plot_cumulative_OH(ax, R_bin_center__r, OH_cumsum__r, OH_cumsum_HII__r, OH_label, OH_range, distance_range):
    ax.plot(R_bin_center__r, OH_cumsum__r, '-', label='total', marker='o', lw=2, c='k')
    ax.plot(R_bin_center__r, OH_cumsum_HII__r, '--', label='HII', marker='o', lw=2, c='b')
    ax.set_xlabel(r'R [HLR]')
    ax.set_ylabel(r'$12\ +\ \log$ (O/H) - %s [Z${}_\odot$]' % OH_label)
    ax.legend(loc='best', frameon=False, fontsize=9)
    ax.grid()
    ax.set_title('cumulative profile')
    ax.set_xlim(distance_range)
    ax.set_ylim(OH_range)
    return ax


def cmap_discrete(colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)], n_bins=3, cmap_name='myMap'):
    cm = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cm


def logO23(fOII, fHb, f5007, tau_V=None):
    fOIII = f5007 * 1.34
    fiOII = fOII
    fiOIII = fOIII
    fiHb = fHb
    if tau_V is not None:
        fiOII = fOII * np.ma.exp(tau_V)
        fiOIII = fOIII * np.ma.exp(tau_V)
        fiHb = fHb * np.ma.exp(tau_V)
    return np.ma.log10((fiOII + fiOIII) / fiHb)


def logN2O2(fNII, fOII, tau_V=None):
    fiNII = fNII
    fiOII = fOII
    if tau_V is not None:
        fiNII = fNII * np.ma.exp(tau_V)
        fiOII = fOII * np.ma.exp(tau_V)
    return np.ma.log10(fiNII/fiOII)


def OH_O23(logO23_ratio=None, logN2O2_ratio=None):
    x_upperBranch = np.linspace(8.04, 9.3, 1000) - 8.69
    x_lowerBranch = np.linspace(7, 8.04, 1000) - 8.69
    p = [-0.2524, -0.6154, -0.9401, -0.7149, 0.7462]
    logO23_upperBranch = np.polyval(p, x_upperBranch)
    logO23_lowerBranch = np.polyval(p, x_lowerBranch)
    interp_upperBranch = interp1d(logO23_upperBranch, x_upperBranch, kind='linear', bounds_error=False)
    interp_lowerBranch = interp1d(logO23_lowerBranch, x_lowerBranch, kind='linear', bounds_error=False)
    return np.where(np.greater(logN2O2_ratio, -1), interp_upperBranch(logO23_ratio), interp_lowerBranch(logO23_ratio)) + 8.69


def OH_N2O2(logN2O2_ratio=None):
    R = logN2O2_ratio
    p = [0.167977, 1.26602, 1.54020]
    return np.log10(np.polyval(p, R)) + 8.93
    # log(O/H) + 12 = log [1.54020 + 1.26602 R + 0.167977 R2] + 8.93, R=log [Nii]/[Oii]
    # x = np.linspace(7, 9.5, 1000) - 8.93
    # p = [0.167977, 1.26602, 1.54020]
    # interp = interp1d(np.log10(np.polyval(p, np.linspace(-2, 1, 1000))), x, kind='linear', bounds_error=False)
    # return interp(logN2O2_ratio) + 8.93


if __name__ == '__main__':
    t_init_prog = time.clock()

    L = Lines()

    rbinini = 0.
    rbinfin = 3.
    rbinstep = 0.2
    R_bin__r = np.arange(rbinini, rbinfin + rbinstep, rbinstep)
    R_bin_center__r = (R_bin__r[:-1] + R_bin__r[1:]) / 2.0
    N_R_bins = len(R_bin_center__r)

    califaID = sys.argv[1]

    K = read_one_cube(califaID, EL=True, GP=True, elliptical=True, config=-2)

    print '# califaID:', califaID, ' N_zones:', K.N_zone, ' lines:', K.EL.lines

    lines = K.EL.lines
    print lines

    EWHa__yx = K.zoneToYX(K.EL.EW[K.EL.lines.index('6563')], extensive=False)
    distance_HLR__yx = K.pixelDistance__yx/K.pixelsPerHLR
    distance_HLR__r = R_bin_center__r

    # fluxes, SBs and cumulative SB
    f_obs__lz = {}
    f_obs__lyx = {}
    SB_obs__lyx = {}
    SB_obs_2__lyx = {}
    SB_obs_sum__lr = {}
    SB_obs_npts__lr = {}
    SB_obs_cumsum__lr = {}
    for i, l in enumerate(lines):
        f_obs__z = K.EL.flux[i]
        mask = np.bitwise_or(~np.isfinite(f_obs__z), np.less(f_obs__z, 1e-40))
        f_obs__lz[l] = np.ma.masked_array(f_obs__z, mask=mask, copy=True)
        # SB per arcsec^2
        f_obs__lyx[l] = K.zoneToYX(f_obs__lz[l]/K.zoneArea_pix, extensive=False)
        # SB per kpc^2
        L_obs__z = K.EL._F_to_L(f_obs__lz[l])/L_sun
        SB_obs__lyx[l] = K.zoneToYX(L_obs__z/(K.zoneArea_pc2 * 1e-6), extensive=False)
        # sum of SB(line) per bin
        SB_obs_sum__lr[l], SB_obs_npts__lr[l] = K.radialProfile(SB_obs__lyx[l], R_bin__r, rad_scale=K.HLR_pix, mode='sum', return_npts=True)
        # cumulative sum of SB(line) per bin
        SB_obs_cumsum__lr[l] = SB_obs_sum__lr[l].filled(0.).cumsum()

    # DIG COMP HII Decomposition by SB6563
    # "SB6563 > 1e39 erg/s/kpc^2 select reliable H ii region dominated spaxels"
    HII_Zhang_threshold = 1e39/L_sun
    DIG_Zhang_threshold = 10**38.5/L_sun
    sel_Zhang_DIG__yx = (SB_obs__lyx['6563'] < DIG_Zhang_threshold).filled(False)
    sel_Zhang_COMP__yx = np.bitwise_and((SB_obs__lyx['6563'] >= DIG_Zhang_threshold).filled(False), (SB_obs__lyx['6563'] < HII_Zhang_threshold).filled(False))
    sel_Zhang_HII__yx = (SB_obs__lyx['6563'] >= HII_Zhang_threshold).filled(False)
    sel_Zhang_DIG_label = 'DIG regions (SB(Ha) < %d)' % DIG_Zhang_threshold
    sel_Zhang_COMP_label = 'COMPOSITE regions (%d <= SB(Ha) < %d)' % (DIG_Zhang_threshold, HII_Zhang_threshold)
    sel_Zhang_HII_label = 'HII regions (SB(Ha) >= %d)' % DIG_Zhang_threshold
    map_Zhang__yx = np.ma.masked_all((K.N_y, K.N_x))
    map_Zhang__yx[sel_Zhang_DIG__yx] = 1  # DIG
    map_Zhang__yx[sel_Zhang_COMP__yx] = 2  # COMP
    map_Zhang__yx[sel_Zhang_HII__yx] = 3  # HII

    # DIG COMP HII Decomposition by EWHa
    DIG_EWHa_threshold = 10
    HII_EWHa_threshold = 20
    sel_EWHa_DIG__yx = (EWHa__yx < DIG_EWHa_threshold).filled(False)
    sel_EWHa_COMP__yx = np.bitwise_and((EWHa__yx >= DIG_EWHa_threshold).filled(False), (EWHa__yx < HII_EWHa_threshold).filled(False))
    sel_EWHa_HII__yx = (EWHa__yx >= HII_EWHa_threshold).filled(False)
    sel_EWHa_DIG_label = 'DIG regions (EW < %d)' % DIG_EWHa_threshold
    sel_EWHa_COMP_label = 'COMPOSITE regions (%d <= EW < %d)' % (DIG_EWHa_threshold, HII_EWHa_threshold)
    sel_EWHa_HII_label = 'HII regions (EW >= %d)' % DIG_EWHa_threshold
    map_EWHa__yx = np.ma.masked_all((K.N_y, K.N_x))
    map_EWHa__yx[sel_EWHa_DIG__yx] = 1  # DIG
    map_EWHa__yx[sel_EWHa_COMP__yx] = 2  # COMP
    map_EWHa__yx[sel_EWHa_HII__yx] = 3  # HII

    sel_DIG__yx = sel_EWHa_DIG__yx
    sel_COMP__yx = sel_EWHa_COMP__yx
    sel_HII__yx = sel_EWHa_HII__yx
    sel_DIG_label = sel_EWHa_DIG_label
    sel_COMP_label = sel_EWHa_COMP_label
    sel_HII_label = sel_EWHa_HII_label
    map__yx = map_EWHa__yx

    SB_obs_HII__lyx = {}
    SB_obs_sum_HII__lr = {}
    SB_obs_cumsum_HII__lr = {}
    SB_obs_npts_HII__lr = {}
    SB_obs_COMP__lyx = {}
    SB_obs_sum_COMP__lr = {}
    SB_obs_cumsum_COMP__lr = {}
    SB_obs_npts_COMP__lr = {}
    SB_obs_DIG__lyx = {}
    SB_obs_sum_DIG__lr = {}
    SB_obs_cumsum_DIG__lr = {}
    SB_obs_npts_DIG__lr = {}
    for k, v in SB_obs__lyx.iteritems():
        SB_obs_HII__lyx[k] = np.ma.masked_array(v, mask=~sel_HII__yx, copy=True)
        SB_obs_sum_HII__lr[k], SB_obs_npts_HII__lr[k] = K.radialProfile(SB_obs_HII__lyx[k], R_bin__r, rad_scale=K.HLR_pix, mode='sum', return_npts=True)
        SB_obs_cumsum_HII__lr[k] = SB_obs_sum_HII__lr[k].filled(0.).cumsum()
        SB_obs_COMP__lyx[k] = np.ma.masked_array(v, mask=~sel_COMP__yx, copy=True)
        SB_obs_sum_COMP__lr[k], SB_obs_npts_COMP__lr[k] = K.radialProfile(SB_obs_COMP__lyx[k], R_bin__r, rad_scale=K.HLR_pix, mode='sum', return_npts=True)
        SB_obs_cumsum_COMP__lr[k] = SB_obs_sum_COMP__lr[k].filled(0.).cumsum()
        SB_obs_DIG__lyx[k] = np.ma.masked_array(v, mask=~sel_DIG__yx, copy=True)
        SB_obs_sum_DIG__lr[k], SB_obs_npts_DIG__lr[k] = K.radialProfile(SB_obs_DIG__lyx[k], R_bin__r, rad_scale=K.HLR_pix, mode='sum', return_npts=True)
        SB_obs_cumsum_HII__lr[k] = SB_obs_sum_HII__lr[k].filled(0.).cumsum()

    HaHb__yx = SB_obs__lyx['6563']/SB_obs__lyx['4861']
    tau_V_neb__yx = np.log(HaHb__yx / 2.86) / (K.EL._qCCM['4861'] - K.EL._qCCM['6563'])
    HaHb_cumsum__r = SB_obs_sum__lr['6563'].filled(0.).cumsum()/SB_obs_sum__lr['4861'].filled(0.).cumsum()
    tau_V_neb_cumsum__r = np.log(HaHb_cumsum__r / 2.86) / (K.EL._qCCM['4861'] - K.EL._qCCM['6563'])
    HaHb_sum_HII__r = SB_obs_sum_HII__lr['6563'].filled(0.).cumsum()/SB_obs_sum_HII__lr['4861'].filled(0.).cumsum()
    tau_V_neb_cumsum_HII__r = np.log(HaHb_sum_HII__r / 2.86) / (K.EL._qCCM['4861'] - K.EL._qCCM['6563'])

    tau_V_neb__yx = np.where(np.less(tau_V_neb__yx.filled(-1), 0), 0, tau_V_neb__yx)
    tau_V_neb_cumsum__r = np.where(np.less(tau_V_neb_cumsum__r, 0), 0, tau_V_neb_cumsum__r)
    tau_V_neb_cumsum_HII__r = np.where(np.less(tau_V_neb_cumsum_HII__r, 0), 0, tau_V_neb_cumsum_HII__r)

    ####################################
    # O/H - Relative Oxygen abundances #
    ####################################
    #############
    # O3N2 PP04 #
    #############
    O3Hb__yx = SB_obs__lyx['5007']/SB_obs__lyx['4861']
    N2Ha__yx = SB_obs__lyx['6583']/SB_obs__lyx['6563']
    OH_O3N2__yx = 8.73 - 0.32 * np.ma.log10(O3Hb__yx/N2Ha__yx)
    O3Hb_cumsum__r = SB_obs_cumsum__lr['5007']/SB_obs_cumsum__lr['4861']
    N2Ha_cumsum__r = SB_obs_cumsum__lr['6583']/SB_obs_cumsum__lr['6563']
    OH_O3N2_cumsum__r = 8.73 - 0.32 * np.ma.log10(O3Hb_cumsum__r/N2Ha_cumsum__r)
    O3Hb_cumsum_HII__r = SB_obs_cumsum_HII__lr['5007']/SB_obs_cumsum_HII__lr['4861']
    N2Ha_cumsum_HII__r = SB_obs_cumsum_HII__lr['6583']/SB_obs_cumsum_HII__lr['6563']
    OH_O3N2_cumsum_HII__r = 8.73 - 0.32 * np.ma.log10(O3Hb_cumsum_HII__r/N2Ha_cumsum_HII__r)
    ###########
    # N2 PP04 #
    ###########
    OH_N2Ha__yx = 8.90 + 0.57 * np.ma.log10(N2Ha__yx)
    OH_N2Ha_cumsum__r = 8.90 + 0.57 * np.ma.log10(N2Ha_cumsum__r)
    OH_N2Ha_cumsum_HII__r = 8.90 + 0.57 * np.ma.log10(N2Ha_cumsum_HII__r)
    #################################
    # O23 Maiolino, R. et al (2008) #
    #################################
    mask = np.zeros((K.N_y, K.N_x), dtype='bool')
    mask = np.bitwise_or(mask, SB_obs__lyx['3727'].mask)
    mask = np.bitwise_or(mask, SB_obs__lyx['4861'].mask)
    mask = np.bitwise_or(mask, SB_obs__lyx['5007'].mask)
    mask = np.bitwise_or(mask, SB_obs__lyx['6583'].mask)
    logO23__yx = logO23(fOII=SB_obs__lyx['3727'], fHb=SB_obs__lyx['4861'], f5007=SB_obs__lyx['5007'], tau_V=tau_V_neb__yx)
    logN2O2__yx = logN2O2(fNII=SB_obs__lyx['6583'], fOII=SB_obs__lyx['3727'], tau_V=tau_V_neb__yx)
    mask = np.bitwise_or(mask, logO23__yx.mask)
    mask = np.bitwise_or(mask, logN2O2__yx.mask)
    mlogO23__yx, mlogN2O2__yx = ma_mask_xyz(logO23__yx, logN2O2__yx, mask=mask)
    OH_O23__yx = np.ma.masked_all((K.N_y, K.N_x))
    OH_O23__yx[~mask] = OH_O23(logO23_ratio=mlogO23__yx.compressed(), logN2O2_ratio=mlogN2O2__yx.compressed())
    OH_O23__yx[~np.isfinite(OH_O23__yx)] = np.ma.masked
    # HII cumulative O/H
    mask = np.zeros((N_R_bins), dtype='bool')
    logO23_cumsum_HII__r = logO23(fOII=SB_obs_cumsum_HII__lr['3727'], fHb=SB_obs_cumsum_HII__lr['4861'], f5007=SB_obs_cumsum_HII__lr['5007'], tau_V=tau_V_neb_cumsum_HII__r)
    logN2O2_cumsum_HII__r = logN2O2(fNII=SB_obs_cumsum_HII__lr['6583'], fOII=SB_obs_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
    mask = np.bitwise_or(mask, logO23_cumsum_HII__r.mask)
    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
    mlogO23_cumsum_HII__r, mlogN2O2_cumsum_HII__r = ma_mask_xyz(logO23_cumsum_HII__r, logN2O2_cumsum_HII__r, mask=mask)
    OH_O23_cumsum_HII__r = np.ma.masked_all((N_R_bins))
    OH_O23_cumsum_HII__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum_HII__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
    OH_O23_cumsum_HII__r[~np.isfinite(OH_O23_cumsum_HII__r)] = np.ma.masked
    # total cumulative O/H
    mask = np.zeros((N_R_bins), dtype='bool')
    logO23_cumsum__r = logO23(fOII=SB_obs_cumsum__lr['3727'], fHb=SB_obs_cumsum__lr['4861'], f5007=SB_obs_cumsum__lr['5007'], tau_V=tau_V_neb_cumsum__r)
    logN2O2_cumsum__r = logN2O2(fNII=SB_obs_cumsum__lr['6583'], fOII=SB_obs_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
    mask = np.bitwise_or(mask, logO23_cumsum__r.mask)
    mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
    mlogO23_cumsum__r, mlogN2O2_cumsum__r = ma_mask_xyz(logO23_cumsum__r, logN2O2_cumsum__r, mask=mask)
    OH_O23_cumsum__r = np.ma.masked_all((N_R_bins))
    OH_O23_cumsum__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
    OH_O23_cumsum__r[~np.isfinite(OH_O23_cumsum__r)] = np.ma.masked
    #############################
    # N2O2 Dopita et al. (2013) #
    #############################
    mask = np.zeros((K.N_y, K.N_x), dtype='bool')
    mask = np.bitwise_or(mask, SB_obs__lyx['3727'].mask)
    mask = np.bitwise_or(mask, SB_obs__lyx['6583'].mask)
    logN2O2__yx = logN2O2(fNII=SB_obs__lyx['6583'], fOII=SB_obs__lyx['3727'], tau_V=tau_V_neb__yx)
    mask = np.bitwise_or(mask, logN2O2__yx.mask)
    mlogN2O2__yx = np.ma.masked_array(logN2O2__yx, mask=mask)
    OH_N2O2__yx = np.ma.masked_all((K.N_y, K.N_x))
    OH_N2O2__yx[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2__yx.compressed())
    OH_N2O2__yx[~np.isfinite(OH_N2O2__yx)] = np.ma.masked
    # HII cumulative O/H
    mask = np.zeros((N_R_bins), dtype='bool')
    logN2O2_cumsum_HII__r = logN2O2(fNII=SB_obs_cumsum_HII__lr['6583'], fOII=SB_obs_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
    mlogN2O2_cumsum_HII__r = np.ma.masked_array(logN2O2_cumsum_HII__r, mask=mask)
    OH_N2O2_cumsum_HII__r = np.ma.masked_all((N_R_bins))
    OH_N2O2_cumsum_HII__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
    # total cumulative O/H
    mask = np.zeros((N_R_bins), dtype='bool')
    logN2O2_cumsum__r = logN2O2(fNII=SB_obs_cumsum__lr['6583'], fOII=SB_obs_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
    mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
    mlogN2O2_cumsum__r = np.ma.masked_array(logN2O2_cumsum__r, mask=mask)
    OH_N2O2_cumsum__r = np.ma.masked_all((N_R_bins))
    OH_N2O2_cumsum__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
    #####################

    #####################
    # PLOT
    #####################
    N_cols = 4
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=300, figsize=(15, 10))
    ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = axArr
    distance_range = [0, 3]
    OH_range = [8, 9.5]
    logEWHa_range = [0, 2.5]
    logSBHa_range = [3.5, 7]
    # AXIS 1
    img_file = '%s%s.jpg' % (img_dir, califaID)
    plot_gal_img_ax(ax1, img_file, califaID, 0.02, 0.98, 16, K, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    # AXIS 2
    # cmap = plt.cm.get_cmap('jet_r', 3)
    cmap = cmap_discrete()
    im = ax2.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax2)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
    cb.set_ticklabels(['DIG', 'COMP', 'HII'])
    DrawHLRCircle(ax2, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    # AXIS 3
    x = np.ma.ravel(distance_HLR__yx)
    y = np.ma.ravel(np.ma.log10(EWHa__yx))
    ax3.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, **dflt_kw_scatter)
    ax3.set_xlim(distance_range)
    ax3.set_ylim(logEWHa_range)
    ax3.set_xlabel(r'R [HLR]')
    ax3.set_ylabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    ax3.grid()
    # AXIS 4
    x = np.ma.ravel(distance_HLR__yx)
    y = np.ma.ravel(np.ma.log10(SB_obs__lyx['6563']))
    ax4.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, **dflt_kw_scatter)
    ax4.set_xlim(distance_range)
    ax4.set_ylim(logSBHa_range)
    ax4.set_xlabel(r'R [HLR]')
    ax4.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
    ax4.grid()
    # AXIS 5
    OH__yx = OH_O3N2__yx
    OH_label = 'O3N2'
    ax = ax5
    plot_OH(ax, distance_HLR__r, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range)
    # AXIS 9
    OH_cumsum__r = OH_O3N2_cumsum__r
    OH_cumsum_HII__r = OH_O3N2_cumsum_HII__r
    ax = ax9
    plot_cumulative_OH(ax, R_bin_center__r, OH_cumsum__r, OH_cumsum_HII__r, OH_label, OH_range, distance_range)
    # AXIS 6
    OH__yx = OH_N2Ha__yx
    OH_label = 'N2'
    ax = ax6
    plot_OH(ax, distance_HLR__r, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range)
    # AXIS 10
    OH_cumsum__r = OH_N2Ha_cumsum__r
    OH_cumsum_HII__r = OH_N2Ha_cumsum_HII__r
    ax = ax10
    plot_cumulative_OH(ax, R_bin_center__r, OH_cumsum__r, OH_cumsum_HII__r, OH_label, OH_range, distance_range)
    # AXIS 7
    OH__yx = OH_O23__yx
    OH_label = 'O23'
    ax = ax7
    plot_OH(ax, distance_HLR__r, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range)
    # AXIS 11
    OH_cumsum__r = OH_O23_cumsum__r
    OH_cumsum_HII__r = OH_O23_cumsum_HII__r
    ax = ax11
    plot_cumulative_OH(ax, R_bin_center__r, OH_cumsum__r, OH_cumsum_HII__r, OH_label, OH_range, distance_range)
    # AXIS 8
    OH__yx = OH_N2O2__yx
    OH_label = 'N2O2'
    ax = ax8
    plot_OH(ax, distance_HLR__r, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range)
    # AXIS 12
    OH_cumsum__r = OH_N2O2_cumsum__r
    OH_cumsum_HII__r = OH_N2O2_cumsum_HII__r
    ax = ax12
    plot_cumulative_OH(ax, R_bin_center__r, OH_cumsum__r, OH_cumsum_HII__r, OH_label, OH_range, distance_range)
    f.tight_layout()
    f.savefig('%s-dig-lineratios-pixels.png' % califaID)
