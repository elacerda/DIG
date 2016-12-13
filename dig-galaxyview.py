import os
import sys
import time
import numpy as np
import matplotlib as mpl
from pytu.plots import plotBPT
from pytu.lines import Lines
from matplotlib import pyplot as plt
from pytu.plots import plot_histo_ax
import matplotlib.gridspec as gridspec
from CALIFAUtils.plots import DrawHLRCircle
from CALIFAUtils.plots import plot_gal_img_ax
from CALIFAUtils.scripts import read_one_cube
from mpl_toolkits.axes_grid1 import make_axes_locatable


def ne_S2(S2S2):
    from scipy.interpolate import interp1d
    # Read S2 emissivities
    _f6716 = np.loadtxt('%s/dev/astro/dig/emis_S2/emis_S2_6716A.txt' % os.environ['HOME'])
    _f6731 = np.loadtxt('%s/dev/astro/dig/emis_S2/emis_S2_6731A.txt' % os.environ['HOME'])

    # Select T = 10.000 K
    _ff = _f6716[:, 0] == 10000
    ne = _f6716[0]
    e6716 = _f6716[_ff][0]

    _ff = _f6731[:, 0] == 10000
    e6731 = _f6731[_ff][0]
    # plt.figure(2)
    # plt.clf()
    # plt.plot(np.log10(ne), e6731/e6716)
    # Calc densities from S2
    interp = interp1d(e6731/e6716, np.log10(ne), kind='linear', bounds_error=False)
    return interp(S2S2)


mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
dflt_kw_scatter = dict(cmap='viridis_r', marker='o', s=5, edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, debug=True, gs_prc=True, poly1d=True)
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal', cmap='viridis_r')
img_dir = '%s/CALIFA/images/' % os.environ['HOME']


# from pytu.plots import plotBPT
# from CALIFAUtils.scripts import read_one_cube
# def O3N2(EL=None):
#     if EL is None:
#         return None
#     O3Hb = EL.O3_obs__z/EL.Hb_obs__z
#     N2Ha = EL.N2_obs__z/EL.Ha_obs__z
#     return O3Hb, N2Ha
# K = read_one_cube('K0073', EL=True, GP=True)
# O3Hb, N2Ha = O3N2(K.EL)
# x, y = np.ma.log10(N2Ha), np.ma.log10(O3Hb)

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

    print ba, K.masterListData['ba']

    if (eval(K.masterListData['ba']) < 0.6):
        print 'edge-on'
        sys.exit(1)

    lines = K.EL.lines
    f_obs__lz = {}
    for i, l in enumerate(lines):
        mask = np.bitwise_or(~np.isfinite(K.EL.flux[i]), np.less(K.EL.flux[i], 1e-40))
        f_obs__lz[l] = np.ma.masked_array(K.EL.flux[i], mask=mask)
        print l, f_obs__lz[l].max(), f_obs__lz[l].min(), f_obs__lz[l].mean()

    f_obs_S2__z = f_obs__lz['6717'] + f_obs__lz['6731']
    N2Ha__z = f_obs__lz['6583']/f_obs__lz['6563']
    O3Hb__z = f_obs__lz['5007']/f_obs__lz['4861']

    O3Hb__yx = K.zoneToYX(O3Hb__z, extensive=False, surface_density=False)
    N2Ha__yx = K.zoneToYX(N2Ha__z, extensive=False, surface_density=False)
    x, y = np.ma.log10(N2Ha__yx), np.ma.log10(O3Hb__yx)
    sel_below_S06 = L.belowlinebpt('S06', x, y)
    sel_below_K03 = L.belowlinebpt('K03', x, y)
    sel_below_K01 = L.belowlinebpt('K01', x, y)
    sel_between_S06K03 = np.bitwise_and(sel_below_K03, ~sel_below_S06)
    sel_between_K03K01 = np.bitwise_and(~sel_below_K03, sel_below_K01)
    sel_above_K01 = ~sel_below_K01

    sel_6563__z = np.bitwise_and(np.isfinite(f_obs__lz['6563']), np.greater(f_obs__lz['6563'], 1e-40))
    SB6563__z = np.ma.masked_array(f_obs__lz['6563']/K.zoneArea_pix, mask=~sel_6563__z)
    SB6563__yx = K.zoneToYX(SB6563__z, extensive=False)

    sel_4861__z = np.bitwise_and(np.isfinite(f_obs__lz['4861']), np.greater(f_obs__lz['4861'], 1e-40))
    SB4861__z = np.ma.masked_array(f_obs__lz['4861']/K.zoneArea_pc2, mask=~sel_4861__z)
    SB4861__yx = K.zoneToYX(SB4861__z, extensive=False)

    sel_S2__z = np.bitwise_and(np.isfinite(f_obs_S2__z), np.greater(f_obs_S2__z, 1e-40))
    SBS2__z = np.ma.masked_array(f_obs_S2__z/K.zoneArea_pix, mask=~sel_S2__z)
    SBS2__yx = K.zoneToYX(SBS2__z, extensive=False)

    HaHb__z = f_obs__lz['6563']/f_obs__lz['4861']
    HaHb__yx = K.zoneToYX(HaHb__z, extensive=False, surface_density=False)
    EWHa__z = K.EL.EW[lines.index('6563')]
    EWHa__yx = K.zoneToYX(EWHa__z, extensive=False, surface_density=False)

    sel_S2S2__z = np.bitwise_and(np.isfinite(f_obs__lz['6717']), np.greater(f_obs__lz['6717'], 1e-40))
    sel_S2S2__z = np.bitwise_and(sel_S2S2__z, np.isfinite(f_obs__lz['6731']))
    sel_S2S2__z = np.bitwise_and(sel_S2S2__z, np.greater(f_obs__lz['6731'], 1e-40))
    S2S2__z = np.ma.masked_array(f_obs__lz['6731']/f_obs__lz['6717'], mask=~sel_S2S2__z)
    S2S2__yx = K.zoneToYX(S2S2__z, extensive=False, surface_density=False)
    aux = ne_S2(S2S2__z.data)
    logne__z = np.ma.masked_array(aux, mask=np.bitwise_or(S2S2__z.mask, ~np.isfinite(aux)))
    aux = ne_S2(S2S2__yx.data)
    logne__yx = np.ma.masked_array(aux, mask=np.bitwise_or(S2S2__yx.mask, ~np.isfinite(aux)))

    # FIGURE
    G = gridspec.GridSpec(4, 4)
    f = plt.figure(dpi=200, figsize=(20, 20))
    ax_img = plt.subplot(G[0:2, 0:2])
    ax_BPTSBS2 = plt.subplot(G[0, 2])
    ax_BPTSBHa = plt.subplot(G[0, 3])
    ax_BPTEWHa = plt.subplot(G[1, 2])
    ax_mapBPT = plt.subplot(G[1, 3])
    ax_mapHaHb = plt.subplot(G[2, 0])
    ax_histHaHb = plt.subplot(G[3, 0])
    ax_mapS2S2 = plt.subplot(G[2, 1])
    ax_histS2S2 = plt.subplot(G[3, 1])
    ax_mapSBHa = plt.subplot(G[2, 2])
    ax_histSBHa = plt.subplot(G[3, 2])
    ax_mapEWHa = plt.subplot(G[2, 3])
    ax_histEWHa = plt.subplot(G[3, 3])

    img_file = '%s%s.jpg' % (img_dir, califaID)
    plot_gal_img_ax(ax_img, img_file, califaID, 0.02, 0.98, 20, K, bins=[0.5, 1, 1.5, 2, 2.5, 3])

    plotBPT(ax_BPTSBS2, np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z), z=logne__z, vmin=1, vmax=2.5, cb_label=r'electron density', cmap='viridis_r')
    plotBPT(ax_BPTSBHa, np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z), z=np.ma.log10(SB6563__z), cb_label=r'$\log$ SB $H\alpha$', cmap='viridis_r')
    plotBPT(ax_BPTEWHa, np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z), z=np.ma.log10(EWHa__z), cb_label=r'$\log$ W${}_H\alpha$', cmap='viridis_r')

    # creating map of BPT position
    mapBPT__yx = np.ma.masked_array(np.zeros((K.N_y, K.N_x)))
    mapBPT__yx[sel_below_S06] = 4
    mapBPT__yx[sel_between_S06K03] = 3
    mapBPT__yx[sel_between_K03K01] = 2
    mapBPT__yx[sel_above_K01] = 1
    mapBPT__yx[~K.qMask] = np.ma.masked
    im = ax_mapBPT.imshow(mapBPT__yx,  origin='lower', interpolation='nearest', aspect='equal', cmap=plt.cm.get_cmap('jet_r', 4))
    the_divider = make_axes_locatable(ax_mapBPT)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_ticks(3/8. * np.asarray([1,3,5,7]) + 1.)
    cb.set_ticklabels(['> K01', 'K03-K01', 'S06-K03', '< S06'])
    DrawHLRCircle(ax_mapBPT, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax_mapBPT.set_title(r'Position in BPT')

    x = np.ma.log10(SB6563__yx)
    im = ax_mapSBHa.imshow(x, **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax_mapSBHa)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'$\log\ \Sigma_{H\alpha}^{obs}$')
    DrawHLRCircle(ax_mapSBHa, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax_mapSBHa.set_title(r'SB $H\alpha$')
    plot_histo_ax(ax_histSBHa, x.compressed(), y_v_space=0.06, c='k', first=True, kwargs_histo=dict(color='b', normed=False))

    range = [1, 2.5]
    x = logne__yx
    im = ax_mapS2S2.imshow(x, vmin=range[0], vmax=range[1], **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax_mapS2S2)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'$\log\ n_e\ [cm^{-3}]$')
    # cb.set_label(r'$6731/6717$')
    DrawHLRCircle(ax_mapS2S2, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax_mapS2S2.set_title(r'electron density')
    plot_histo_ax(ax_histS2S2, x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=range))

    range = [0, 6]
    x = HaHb__yx
    im = ax_mapHaHb.imshow(x, vmin=range[0], vmax=range[1], **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax_mapHaHb)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'$6563/4861$')
    DrawHLRCircle(ax_mapHaHb, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax_mapHaHb.set_title(r'$6563/4861$')
    plot_histo_ax(ax_histHaHb, x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=range))

    range = [-0.5, 3]
    x = np.ma.log10(EWHa__yx)
    im = ax_mapEWHa.imshow(x, vmin=range[0], vmax=range[1], **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax_mapEWHa)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'$\log$ W${}_{H\alpha}$')
    DrawHLRCircle(ax_mapEWHa, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax_mapEWHa.set_title(r'$\log$ W${}_{H\alpha}$')
    plot_histo_ax(ax_histEWHa, x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=range))

    G.tight_layout(f)
    f.savefig('%s-DIGHII.png' % califaID)

    # desired_lines_to_plot = set(['6563'])
    # avaible_lines_to_plot = set(K.EL.lines)
    # lines_to_plot = sorted(avaible_lines_to_plot.intersection(desired_lines_to_plot))
    # for l in lines_to_plot:
    #     N_cols = 2
    #     N_rows = 1
    #     f, axArr = plt.subplots(N_rows, N_cols, dpi=100, figsize=(10, 5))
    #     (ax1, ax2) = axArr
    #     # AXIS 1
    #     img_file = '%s%s.jpg' % (img_dir, califaID)
    #     plot_gal_img_ax(ax1, img_file, califaID, 0.02, 0.98, 16, K, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    #     # AXIS 2
    #     im = ax2.imshow(np.ma.log10(K.zoneToYX(f_obs__lz[l])), **dflt_kw_imshow)
    #     the_divider = make_axes_locatable(ax2)
    #     color_axis = the_divider.append_axes('right', size='5%', pad=0)
    #     cb = plt.colorbar(im, cax=color_axis)
    #     cb.set_label(r'$\log$ F${}_{obs}$ [erg/s/cm${}^2/\AA$]')
    #     DrawHLRCircle(ax2, K, color='k', bins=[0.5, 1, 1.5, 2, 2.5, 3])
    #     ax2.set_title('Observed flux')
    #     # AXIS 3
    #     # im = ax3.imshow(np.ma.log10(f_obs_DIG__lyx[l]), **dflt_kw_imshow)
    #     # the_divider = make_axes_locatable(ax3)
    #     # color_axis = the_divider.append_axes('right', size='5%', pad=0)
    #     # cb = plt.colorbar(im, cax=color_axis)
    #     # cb.set_label(r'$\log$ F${}_{obs}$ [erg/s/cm${}^2/\AA$]')
    #     # DrawHLRCircle(ax3, K, color='k', bins=[0.5, 1, 1.5, 2, 2.5, 3])
    #     # ax3.set_title(sel_DIG_label)
    #     # # AXIS 4
    #     # im = ax4.imshow(np.ma.log10(f_obs_HII__lyx[l]), **dflt_kw_imshow)
    #     # the_divider = make_axes_locatable(ax4)
    #     # color_axis = the_divider.append_axes('right', size='5%', pad=0)
    #     # cb = plt.colorbar(im, cax=color_axis)
    #     # cb.set_label(r'$\log$ F${}_{obs}$ [erg/s/cm${}^2/\AA$]')
    #     # DrawHLRCircle(ax4, K, color='k', bins=[0.5, 1, 1.5, 2, 2.5, 3])
    #     # ax4.set_title(sel_HII_label)
    #     f.suptitle('%s' % l, fontsize=20)
    #     f.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     f.savefig('%s-%s.png' % (califaID, l))
