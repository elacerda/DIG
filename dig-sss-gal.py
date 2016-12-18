import os
import sys
import time
import numpy as np
import matplotlib as mpl
from pytu.plots import plotBPT
from pytu.objects import runstats
from pytu.plots import plot_text_ax
from pytu.lines import Lines
from matplotlib import pyplot as plt
from pytu.plots import density_contour
from pytu.functions import ma_mask_xyz
from CALIFAUtils.scripts import calc_xY
from CALIFAUtils.plots import DrawHLRCircle
from CALIFAUtils.plots import plot_gal_img_ax
from CALIFAUtils.scripts import read_one_cube
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
img_dir = '%s/califa/images/' % os.environ['HOME']


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
    EWHa__z = K.EL.EW[K.EL.lines.index('6563')]

    O3Hb__z = f_obs__lz['5007']/f_obs__lz['4861']
    N2Ha__z = f_obs__lz['6583']/f_obs__lz['6563']
    x, y = np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z)
    sel_S06_HII__z = L.belowlinebpt('S06', x, y)
    sel_S06_DIG__z = ~sel_S06_HII__z
    sel_S06_DIG_label = 'DIG regions (above S06)'
    sel_S06_HII_label = 'HII regions (below S06)'

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
    etasum = u_x_Y + u_EWHa + u_HaN2 + u_tauVneb + u_S2S2
    sel_etasum_DIG__z = (etasum < DIG_etasum_threshold).filled(False)
    sel_etasum_HII__z = (etasum >= DIG_etasum_threshold).filled(False)
    sel_etasum_DIG_label = 'DIG regions ($\eta$ < %.1f)' % DIG_etasum_threshold
    sel_etasum_HII_label = 'HII regions ($\eta$ >= %.1f)' % DIG_etasum_threshold

    sel_DIG__z = sel_S06_DIG__z
    sel_HII__z = sel_S06_HII__z
    sel_DIG_label = sel_S06_DIG_label
    sel_HII_label = sel_S06_HII_label

    # califaID_threshold = {
    #     'K0009': -20.5,
    #     'K0100': -19.6,
    #     'K0154': -20.5,
    #     'K0155': -20.8,
    #     'K0476': -20.4,
    #     'K0548': -20.5,
    #     'K0577': -21.3,
    #     'K0610': -21.6,
    #     'K0680': -19.6,
    #     'K0740': -21,
    #     'K0758': -20.4,
    #     'K0791': -20.8,
    #     'K0822': -21.6,
    #     'K0876': -21,
    #     'K0904': -20.4,
    #     'K0909': -20.4,
    #     'K0915': -20,
    #     'K0939': -21,
    # }

    f_obs_HII__lyx = {}
    f_obs_DIG__lyx = {}
    for k, v in f_obs__lz.iteritems():
        f_obs_HII__lyx[k] = K.zoneToYX(np.ma.masked_array(v/K.zoneArea_pix, mask=~sel_HII__z), extensive=False, surface_density=False)
        f_obs_DIG__lyx[k] = K.zoneToYX(np.ma.masked_array(v/K.zoneArea_pix, mask=~sel_DIG__z), extensive=False, surface_density=False)
    f_obs_HII__lr = {}
    f_obs_HII_npts__lr = {}
    for k, v in f_obs_HII__lyx.iteritems():
        f_obs_HII__lr[k], f_obs_HII_npts__lr[k] = K.radialProfile(v, R_bin__r, rad_scale=K.HLR_pix, mode='sum', return_npts=True)
    f_obs_DIG__lr = {}
    f_obs_DIG_npts__lr = {}
    for k, v in f_obs_DIG__lyx.iteritems():
        f_obs_DIG__lr[k], f_obs_DIG_npts__lr[k] = K.radialProfile(v, R_bin__r, rad_scale=K.HLR_pix, mode='sum', return_npts=True)

    # sys.exit(1)

    N_cols = 3
    N_rows = 2
    f, axArr = plt.subplots(N_rows, N_cols, dpi=100, figsize=(15, 10))
    cmap = plt.cm.get_cmap('jet_r', 6)
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axArr
    # AXIS 1
    img_file = '%s%s.jpg' % (img_dir, califaID)
    plot_gal_img_ax(ax1, img_file, califaID, 0.02, 0.98, 16, K, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    # AXIS 2
    range = [-0.5, 3]
    im = ax2.imshow(np.ma.log10(K.zoneToYX(np.ma.masked_array(EWHa__z, mask=~sel_DIG__z), extensive=False)), vmin=range[0], vmax=range[1], **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax2)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'$\log$ W${}_{H\alpha}$')
    DrawHLRCircle(ax2, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax2.set_title(sel_DIG_label)
    # AXIS 3
    im = ax3.imshow(np.ma.log10(K.zoneToYX(np.ma.masked_array(EWHa__z, mask=~sel_HII__z), extensive=False)), vmin=range[0], vmax=range[1], **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax3)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'$\log$ W${}_{H\alpha}$')
    DrawHLRCircle(ax3, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax3.set_title(sel_HII_label)
    # AXIS 4 & 5
    desired_lines_to_plot = set(['3727', '4861', '5007', '6563', '6583'])
    avaible_lines_to_plot = set(K.EL.lines)
    lines_to_plot = sorted(avaible_lines_to_plot.intersection(desired_lines_to_plot))
    for i, l in enumerate(lines_to_plot):
        j = 6 - i
        x = f_obs_DIG__lr[l].filled(0.0)
        y = f_obs_DIG__lr[l].filled(0.0) + f_obs_HII__lr[l].filled(0.0)
        ax4.plot(R_bin_center__r, np.ma.masked_array(x / y, mask=~np.isfinite(x / y)), c=cmap(j), marker='o', lw=1.5)
        ax5.plot(R_bin_center__r, np.ma.masked_array(x.cumsum() / y.cumsum(), mask=~np.isfinite(x.cumsum() / y.cumsum())), c=cmap(j), label=l, marker='o', lw=1.5)
    if '6717' and '6731' in lines:
        f_obs_S2_DIG__r = f_obs_DIG__lr['6717'].filled(0.0) + f_obs_DIG__lr['6731'].filled(0.0)
        f_obs_S2_HII__r = f_obs_HII__lr['6717'].filled(0.0) + f_obs_HII__lr['6731'].filled(0.0)
        x = f_obs_S2_DIG__r
        y = f_obs_S2_DIG__r + f_obs_S2_HII__r
        ax4.plot(R_bin_center__r, np.ma.masked_array(x / y, mask=~np.isfinite(x / y)), c=cmap(0), marker='o', lw=1.5)
        ax5.plot(R_bin_center__r, np.ma.masked_array(x.cumsum() / y.cumsum(), mask=~np.isfinite(x.cumsum() / y.cumsum())), c=cmap(0), label='6717+6731', marker='o', lw=1.5)
    ax5.legend(loc='best', frameon=False, fontsize=9)
    ax4.set_xlabel('R [HLR]')
    ax4.set_ylabel(r'$\sum_{z \in R_i}$ SB${}_z^{DIG}/\sum_{z \in R_i}$ SB${}_z^{DIG\ +\ HII}$')
    ax4.set_xlim(0, 3)
    ax5.set_xlabel('R [HLR]')
    ax5.set_ylabel(r'$\sum_{z \to R_i}$ SB${}_z^{DIG}/\sum_{z \to R_i}$ SB${}_z^{DIG\ +\ HII}$')
    ax5.set_xlim(0, 3)
    # AXIS 6
    c = {'5007': 'b', '6583': 'r', '6717': 'g', '6563': 'k'}
    dividend_lines_quocient = ['5007', '6583', '6717', '6563']
    divisor_lines_quocient = ['4861', '6563', '6731', '4861']
    for l1, l2 in zip(dividend_lines_quocient, divisor_lines_quocient):
        if l1 and l2 in lines:
            ax6.plot(R_bin_center__r, f_obs_DIG__lr[l1]/f_obs_DIG__lr[l2], '%c.' % c[l1], label='%s/%s DIG' % (l1, l2))
            ax6.plot(R_bin_center__r, f_obs_HII__lr[l1]/f_obs_HII__lr[l2], '%c-' % c[l1], label='%s/%s HII' % (l1, l2))
    ax6.legend(loc='best', frameon=False, fontsize=9)
    ax6.set_xlabel('R [HLR]')
    ax6.set_ylabel(r'$\sum_{z \in R_i}$ SB${}_z^{\lambda_1}/$ SB${}_z^{\lambda_2}$')
    ymin, ymax = ax6.get_ylim()
    if ymax > 6:
        ymax = 6
    ax6.set_ylim(ymin, ymax)
    ax6.set_xlim(0, 3)
    ax4.grid()
    ax5.grid()
    ax6.grid()
    f.tight_layout()
    f.savefig('%s-DIG.png' % califaID)

    desired_lines_to_plot = set(['3727', '4861', '5007', '6563', '6583', '6717', '6731'])
    avaible_lines_to_plot = set(K.EL.lines)
    lines_to_plot = sorted(avaible_lines_to_plot.intersection(desired_lines_to_plot))
    for l in lines_to_plot:
        N_cols = 2
        N_rows = 2
        f, axArr = plt.subplots(N_rows, N_cols, dpi=100, figsize=(10, 10))
        ((ax1, ax2), (ax3, ax4)) = axArr
        # AXIS 1
        img_file = '%s%s.jpg' % (img_dir, califaID)
        plot_gal_img_ax(ax1, img_file, califaID, 0.02, 0.98, 16, K, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # AXIS 2
        SB_obs_line__yx = K.zoneToYX(f_obs__lz[l]/K.zoneArea_pix, extensive=False, surface_density=False)
        im = ax2.imshow(np.ma.log10(SB_obs_line__yx), **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log$ SB${}_{obs}$ [erg/s/cm${}^2/$arcsec${}^2$]')
        DrawHLRCircle(ax2, K, color='k', bins=[0.5, 1, 1.5, 2, 2.5, 3])
        ax2.set_title('Observed SB')
        # AXIS 3
        SB_obs_line_DIG__yx = K.zoneToYX(np.ma.masked_array(f_obs__lz[l]/K.zoneArea_pix, mask=~sel_DIG__z), extensive=False, surface_density=False)
        im = ax3.imshow(np.ma.log10(SB_obs_line_DIG__yx), **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax3)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log$ SB${}_{obs}$ [erg/s/cm${}^2/$arcsec${}^2$]')
        DrawHLRCircle(ax3, K, color='k', bins=[0.5, 1, 1.5, 2, 2.5, 3])
        ax3.set_title(sel_DIG_label)
        # AXIS 4
        SB_obs_line_HII__yx = K.zoneToYX(np.ma.masked_array(f_obs__lz[l]/K.zoneArea_pix, mask=~sel_HII__z), extensive=False, surface_density=False)
        im = ax4.imshow(np.ma.log10(SB_obs_line_HII__yx), **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax4)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log$ SB${}_{obs}$ [erg/s/cm${}^2/$arcsec${}^2$]')
        DrawHLRCircle(ax4, K, color='k', bins=[0.5, 1, 1.5, 2, 2.5, 3])
        ax4.set_title(sel_HII_label)
        f.suptitle('%s' % l, fontsize=20)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('%s-%s.png' % (califaID, l))
