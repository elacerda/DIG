import os
import sys
import time
import numpy as np
import matplotlib as mpl
from scipy import stats as st
from pytu.objects import runstats
from matplotlib import pyplot as plt
from pytu.functions import debug_var
from pytu.functions import ma_mask_xyz
from CALIFAUtils.scripts import calc_xY
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import NullFormatter
from CALIFAUtils.plots import DrawHLRCircle
from CALIFAUtils.plots import plot_gal_img_ax
from CALIFAUtils.scripts import read_one_cube
from pytu.plots import plot_text_ax, density_contour
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
dflt_kw_scatter = dict(cmap='viridis_r', marker='o', s=5, edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, debug=True, gs_prc=True, poly1d=True)
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal', cmap='viridis_r')
img_dir = '%s/CALIFA/images/' % os.environ['HOME']


if __name__ == '__main__':
    t_init_prog = time.clock()

    rbinini = 0.
    rbinfin = 3.
    rbinstep = 0.2
    R_bin__r = np.arange(rbinini, rbinfin + rbinstep, rbinstep)
    R_bin_center__r = (R_bin__r[:-1] + R_bin__r[1:]) / 2.0
    N_R_bins = len(R_bin_center__r)

    califaID = sys.argv[1]

    K = read_one_cube(califaID, EL=True, GP=True)

    print '# califaID:', califaID, ' N_zones:', K.N_zone, ' lines:', K.EL.lines

    pa, ba = K.getEllipseParams()
    K.setGeometry(pa, ba)

    lines = K.EL.lines
    f_obs__lz = {}
    for i, l in enumerate(lines):
        mask = np.bitwise_or(~np.isfinite(K.EL.flux[i]), np.less(K.EL.flux[i], 1e-40))
        f_obs__lz[l] = np.ma.masked_array(K.EL.flux[i], mask=mask)
        print l, f_obs__lz[l].max(), f_obs__lz[l].min(), f_obs__lz[l].mean()

    x_Y__z, integrated_x_Y = calc_xY(K, 3.2e7)
    EWHa__z = K.EL.EW[lines.index('6563')]
    Ha__z = f_obs__lz['6563'].filled(0.0)

    DIG_EWHa_threshold = 3
    sel_EWHa_DIG__z = (EWHa__z < DIG_EWHa_threshold).filled(False)
    sel_EWHa_HII__z = (EWHa__z >= DIG_EWHa_threshold).filled(False)
    sel_EWHa_DIG_label = 'DIG regions (EW < %d)' % DIG_EWHa_threshold
    sel_EWHa_HII_label = 'HII regions (EW >= %d)' % DIG_EWHa_threshold
    DIG_x_Y_threshold = 0.1
    sel_x_Y_DIG__z = (x_Y__z < DIG_x_Y_threshold)
    sel_x_Y_HII__z = (x_Y__z >= DIG_x_Y_threshold)
    sel_x_Y_DIG_label = 'DIG regions ($x_Y$ < %.1f)' % DIG_x_Y_threshold
    sel_x_Y_HII_label = 'HII regions ($x_Y$ >= %.1f)' % DIG_x_Y_threshold

    EWHa = EWHa__z.filled(0.0)
    HaN2 = f_obs__lz['6563'].filled(0.0) / f_obs__lz['6583'].filled(0.0)
    HaN2 = np.ma.masked_array(HaN2, mask=~(np.isfinite(HaN2)))
    S2S2 = f_obs__lz['6731'].filled(0.0) / f_obs__lz['6717'].filled(0.0)
    S2S2 = np.ma.masked_array(S2S2, mask=~(np.isfinite(S2S2)))
    tau_V_neb__z = np.where((K.EL.tau_V_neb__z < 0).filled(True), 0, K.EL.tau_V_neb__z)
    u_EWHa = (EWHa - EWHa.mean()) / EWHa.std()
    print u_EWHa.max(), u_EWHa.min()
    u_x_Y = (x_Y__z - x_Y__z.mean()) / x_Y__z.std()
    print u_x_Y.max(), u_x_Y.min()
    u_HaN2 = (HaN2 - HaN2.mean()) / HaN2.std()
    print u_HaN2.max(), u_HaN2.min()
    u_tauVneb = (tau_V_neb__z - tau_V_neb__z.mean()) / tau_V_neb__z.std()
    print u_tauVneb.max(), u_tauVneb.min()
    u_S2S2 = (S2S2 - S2S2.mean()) / S2S2.std()
    print u_S2S2.max(), u_S2S2.min()

    DIG_etasum_threshold = 0
    # eta = x_Y__z * HaN2 * K.EL.EW[K.EL.lines.index('6563')] * np.where((K.EL.tau_V_neb__z < 0).filled(True), 0, K.EL.tau_V_neb__z)
    etasum = u_x_Y + u_EWHa + u_HaN2 + u_tauVneb + u_S2S2
    sel_etasum_DIG__z = (etasum < DIG_etasum_threshold).filled(False)
    sel_etasum_HII__z = (etasum >= DIG_etasum_threshold).filled(False)
    sel_etasum_DIG_label = 'DIG regions ($\eta$ < %.1f)' % DIG_etasum_threshold
    sel_etasum_HII_label = 'HII regions ($\eta$ >= %.1f)' % DIG_etasum_threshold

    sel_DIG__z = sel_etasum_DIG__z
    sel_HII__z = sel_etasum_HII__z
    sel_DIG_label = sel_etasum_DIG_label
    sel_HII_label = sel_etasum_HII_label

    f_obs_HII__lyx = {}
    f_obs_DIG__lyx = {}
    for k, v in f_obs__lz.iteritems():
        f_obs_HII__lyx[k] = K.zoneToYX(np.ma.masked_array(v, mask=~sel_HII__z), surface_density=False)
        f_obs_DIG__lyx[k] = K.zoneToYX(np.ma.masked_array(v, mask=~sel_DIG__z), surface_density=False)
    f_obs_HII__lr = {}
    f_obs_HII_npts__lr = {}
    for k, v in f_obs_HII__lyx.iteritems():
        f_obs_HII__lr[k], f_obs_HII_npts__lr[k] = K.radialProfile(v, R_bin__r, rad_scale=K.HLR_pix, mode='sum', return_npts=True)
    f_obs_DIG__lr = {}
    f_obs_DIG_npts__lr = {}
    for k, v in f_obs_DIG__lyx.iteritems():
        f_obs_DIG__lr[k], f_obs_DIG_npts__lr[k] = K.radialProfile(v, R_bin__r, rad_scale=K.HLR_pix, mode='sum', return_npts=True)

    Ha__yx = K.zoneToYX(K.EL.flux[lines.index('6563')])  # , extensive=True, surface_density=True)
    from scipy.signal import medfilt2d
    Ha_medianSmooth__yx = medfilt2d(Ha__yx.filled(0.0), kernel_size=[7, 7])
    import scipy.ndimage.filters as spfilters
    Ha_medianSmooth2__yx = spfilters.median_filter(Ha__yx.filled(0.0), size=[7, 7])
    Ha_gaussSmooth__yx = spfilters.gaussian_filter(Ha__yx.filled(0.0), sigma=2)
    import scipy.ndimage.fourier as spfourier
    Ha_gaussFourierSmooth__yx = spfourier.fourier_gaussian(Ha__yx.filled(0.0), sigma=2)

    N_cols = 3
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=100, figsize=(5 * N_cols, 5 * N_rows))
    cmap = plt.cm.get_cmap('rainbow', 6)
    ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = axArr
    ax3.set_axis_off()
    # AXIS 1
    img_file = '%s%s.jpg' % (img_dir, califaID)
    plot_gal_img_ax(ax1, img_file, califaID, 0.02, 0.98, 16, K, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    # AXIS 2
    im = ax2.imshow(np.ma.masked_array(Ha_medianSmooth__yx/Ha__yx, mask=(Ha_gaussSmooth__yx < 1.2e-21)), **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax2)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'F${}_{H\alpha}^{obs}$ [erg/s/cm${}^2/\AA$]')
    DrawHLRCircle(ax2, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax2.set_title('observed Ha flux')
    # AXIS 4
    im = ax4.imshow(Ha_medianSmooth__yx, **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax4)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'F${}_{H\alpha}^{obs}$')
    DrawHLRCircle(ax4, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax4.set_title('median filter')
    # AXIS 5
    im = ax5.imshow(Ha_gaussSmooth__yx, **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax5)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'F${}_{H\alpha}^{obs}$')
    DrawHLRCircle(ax5, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax5.set_title('Gaussian filter')
    # AXIS 6
    im = ax6.imshow(Ha_gaussFourierSmooth__yx, **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax6)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'F${}_{H\alpha}^{obs}$')
    DrawHLRCircle(ax6, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax6.set_title('Fourier Gaussian filter')
    # AXIS 7
    im = ax7.imshow(Ha_medianSmooth__yx/Ha__yx, **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax7)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'F${}_{H\alpha}^{obs}$')
    DrawHLRCircle(ax7, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax7.set_title('median filter')
    # AXIS 8
    im = ax8.imshow(Ha_gaussSmooth__yx/Ha__yx, **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax8)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'F${}_{H\alpha}^{obs}$')
    DrawHLRCircle(ax8, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax8.set_title('Gaussian filter')
    # AXIS 9
    im = ax9.imshow(Ha_gaussFourierSmooth__yx/Ha__yx, **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax9)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'F${}_{H\alpha}^{obs}$')
    DrawHLRCircle(ax9, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax9.set_title('Fourier Gaussian filter')
    f.tight_layout()
    f.savefig('%s-HIIthreshold.png' % califaID)
