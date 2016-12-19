import sys
import numpy as np
import matplotlib as mpl
from pytu.objects import runstats
from matplotlib import pyplot as plt
from pytu.functions import ma_mask_xyz
from pystarlight.util.constants import L_sun
from CALIFAUtils.scripts import get_NEDName_by_CALIFAID
from CALIFAUtils.objects import stack_gals, CALIFAPaths
from mpl_toolkits.axes_grid1 import make_axes_locatable
from CALIFAUtils.plots import DrawHLRCircleInSDSSImage, DrawHLRCircle
from pytu.plots import cmap_discrete, plot_histo_ax, plot_text_ax, plot_scatter_histo


mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'

# config variables
lineratios_range = {
    '6563/4861': [0, 1],
    '6583/6563': [-0.8, 0],
    '6300/6563': [-2, 0],
    '6717+6731/6563': [-0.8, 0.2],
}
logSBHa_range = [4, 7]
logWHa_range = [0, 2.5]
DtauV_range = [-2, 3]
x_Y_range = [0, 0.6]
# age to calc xY
tY = 32e6
minSNR = 0
config = -2
EL = True
elliptical = True
DIG_WHa_threshold = 10
HII_WHa_threshold = 20
HII_Zhang_threshold = 1e39/L_sun
DIG_Zhang_threshold = 10**38.5/L_sun
lines = ['3727', '4363', '4861', '4959', '5007', '6300', '6563', '6583', '6717', '6731']
rbinini = 0.
rbinfin = 3.
rbinstep = 0.2
R_bin__r = np.arange(rbinini, rbinfin + rbinstep, rbinstep)
R_bin_center__r = (R_bin__r[:-1] + R_bin__r[1:]) / 2.0
N_R_bins = len(R_bin_center__r)
kw_cube = dict(EL=EL, config=config, elliptical=elliptical)
dflt_kw_scatter = dict(marker='o', edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, debug=True)  # , tendency=True)
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal')
P = CALIFAPaths()


def main(argv=None):
    sample_filename = sys.argv[1]
    ALL = stack_gals().load(sample_filename)

    try:
        # read gals file
        with open(sys.argv[2]) as f:
            gals = [line.strip() for line in f.xreadlines() if line.strip()[0] != '#']
    except IndexError:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)].tolist()

    # # sample histograms of Ha/Hb ... tauNeb - tauStar...
    # histograms_HaHb_Dt(ALL, gals)

    # Dtau's x xY
    # Dt_xY_profile_sample(ALL, gals)

    # # SDSS stamp + EWHa + SBHa + HaHb maps + DIG/COMP/HII map
    # maps_colorsWHa(ALL, gals)
    # maps_colorsZhang(ALL, gals)

    # # Ha/Hb x R  N2Ha x R O1Ha x R S2Ha x R
    # maps_lineratios_colorsWHa(ALL, gals)
    # maps_lineratios_colorsZhang(ALL, gals)

    # # sample histograms of WHa x SBHa
    WHa_SBHa_sample_histograms(ALL, gals)
    # # sample histograms of tauHII(R) - tauDIG(R)?
    # # tauNeb - tauStar


def WHa_SBHa_sample_histograms(ALL, gals=None):
    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')
    sel_gals__gyx = np.zeros((ALL.califaID__yx.shape), dtype='bool')

    for g in gals:
        where_gals__gz = np.where(ALL.califaID__z == g)
        where_gals__gyx = np.where(ALL.califaID__yx == g)
        if not where_gals__gz:
            continue
        sel_gals__gz[where_gals__gz] = True
        sel_gals__gyx[where_gals__gyx] = True

    if (sel_gals__gz).any():
        W6563__gyx = ALL.W6563__yx[sel_gals__gyx]

        SB6563__gyx = ALL.SB6563__yx[sel_gals__gyx]

        # WHa DIG-COMP-HII decomposition
        sel_WHa_DIG__gyx = (W6563__gyx < DIG_WHa_threshold).filled(False)
        sel_WHa_COMP__gyx = np.bitwise_and((W6563__gyx >= DIG_WHa_threshold).filled(False), (W6563__gyx < HII_WHa_threshold).filled(False))
        sel_WHa_HII__gyx = (W6563__gyx >= HII_WHa_threshold).filled(False)

        # SBHa-Zhang DIG-COMP-HII decomposition
        sel_Zhang_DIG__gyx = (SB6563__gyx < DIG_Zhang_threshold).filled(False)
        sel_Zhang_COMP__gyx = np.bitwise_and((SB6563__gyx >= DIG_Zhang_threshold).filled(False), (SB6563__gyx < HII_Zhang_threshold).filled(False))
        sel_Zhang_HII__gyx = (SB6563__gyx >= HII_Zhang_threshold).filled(False)

        x = np.ma.log10(W6563__gyx)
        y = np.ma.log10(SB6563__gyx)
        f = plt.figure(figsize=(8, 8))
        x_ds = [x[sel_WHa_DIG__gyx].compressed(), x[sel_WHa_COMP__gyx].compressed(), x[sel_WHa_HII__gyx].compressed()]
        y_ds = [y[sel_WHa_DIG__gyx].compressed(), y[sel_WHa_COMP__gyx].compressed(), y[sel_WHa_HII__gyx].compressed()]
        axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logWHa_range, logSBHa_range, 30, 30,
                                             figure=f, c=['r', 'g', 'b'], scatter=False, s=1,
                                             ylabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]',
                                             xlabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logWHa_range, logSBHa_range, 30, 30,
                                             axScatter=axS, axHistx=axH1, axHisty=axH2, c=['r', 'g', 'b'], histo=False, s=1,
                                             ylabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]',
                                             xlabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        xm, ym = ma_mask_xyz(x, y)
        rs = runstats(xm.compressed(), ym.compressed(), gs_prc=True, **dflt_kw_runstats)
        for i in xrange(len(rs.xPrcS)):
            axS.plot(rs.xPrcS[i], rs.yPrcS[i], 'k--', lw=2)
        if sel_WHa_DIG__gyx.astype('int').sum() > 0:
            xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_DIG__gyx)
            rs = runstats(xm.compressed(), ym.compressed(), gs_prc=True, **dflt_kw_runstats)
            axS.plot(rs.xS, rs.yS, 'r', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
        if sel_WHa_COMP__gyx.astype('int').sum() > 0:
            xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_COMP__gyx)
            rs = runstats(xm.compressed(), ym.compressed(), gs_prc=True, **dflt_kw_runstats)
            axS.plot(rs.xS, rs.yS, 'g', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
        if sel_WHa_HII__gyx.astype('int').sum() > 0:
            xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_HII__gyx)
            rs = runstats(xm.compressed(), ym.compressed(), gs_prc=True, **dflt_kw_runstats)
            axS.plot(rs.xS, rs.yS, 'b', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
        axS.grid()
        f.savefig('dig-sample-WHa_SBHa-classifWHa.png')
        plt.close(f)

        x = np.ma.log10(W6563__gyx)
        y = np.ma.log10(SB6563__gyx)
        f = plt.figure(figsize=(8, 8))
        x_ds = [x[sel_Zhang_DIG__gyx].compressed(), x[sel_Zhang_COMP__gyx].compressed(), x[sel_Zhang_HII__gyx].compressed()]
        y_ds = [y[sel_Zhang_DIG__gyx].compressed(), y[sel_Zhang_COMP__gyx].compressed(), y[sel_Zhang_HII__gyx].compressed()]
        axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logWHa_range, logSBHa_range, 30, 30,
                                             figure=f, c=['r', 'g', 'b'], scatter=False, s=1,
                                             ylabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]',
                                             xlabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logWHa_range, logSBHa_range, 30, 30,
                                             axScatter=axS, axHistx=axH1, axHisty=axH2, c=['r', 'g', 'b'], histo=False, s=1,
                                             ylabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]',
                                             xlabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        xm, ym = ma_mask_xyz(x, y)
        rs = runstats(ym.compressed(), xm.compressed(), gs_prc=True, **dflt_kw_runstats)
        for i in xrange(len(rs.xPrcS)):
            axS.plot(rs.yPrcS[i], rs.xPrcS[i], 'k--', lw=2)
        if sel_Zhang_DIG__gyx.astype('int').sum() > 0:
            xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_DIG__gyx)
            rs = runstats(ym.compressed(), xm.compressed(), gs_prc=True, **dflt_kw_runstats)
            axS.plot(rs.yS, rs.xS, 'r', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
        if sel_Zhang_COMP__gyx.astype('int').sum() > 0:
            xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_COMP__gyx)
            rs = runstats(ym.compressed(), xm.compressed(), gs_prc=True, **dflt_kw_runstats)
            axS.plot(rs.yS, rs.xS, 'g', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
        if sel_Zhang_HII__gyx.astype('int').sum() > 0:
            xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_HII__gyx)
            rs = runstats(ym.compressed(), xm.compressed(), gs_prc=True, **dflt_kw_runstats)
            axS.plot(rs.yS, rs.xS, 'b', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
        axS.grid()
        f.savefig('dig-sample-WHa_SBHa-classifZhang.png')
        plt.close(f)


def maps_lineratios_colorsWHa(ALL, gals=None):
    # WHa DIG-COMP-HII decomposition
    sel_WHa_DIG__yx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__yx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__yx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        pixelDistance__yx = ALL.get_gal_prop(califaID, ALL.pixelDistance__yx).reshape(N_y, N_x)
        pixelDistance_HLR__yx = pixelDistance__yx / HLR_pix
        SB__yx = {'%s' % L: ALL.get_gal_prop(califaID, 'SB%s__yx' % L).reshape(N_y, N_x) for L in lines}
        sel_DIG__yx = ALL.get_gal_prop(califaID, sel_WHa_DIG__yx).reshape(N_y, N_x)
        sel_COMP__yx = ALL.get_gal_prop(califaID, sel_WHa_COMP__yx).reshape(N_y, N_x)
        sel_HII__yx = ALL.get_gal_prop(califaID, sel_WHa_HII__yx).reshape(N_y, N_x)
        map__yx = np.ma.masked_all((N_y, N_x))
        map__yx[sel_DIG__yx] = 1
        map__yx[sel_COMP__yx] = 2
        map__yx[sel_HII__yx] = 3

        distance_range = [0, 3]
        N_cols = 4
        N_rows = 3
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 4))
        cmap = cmap_discrete()
        ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = axArr
        f.suptitle(r'%s - %s: %d pixels (%d zones) - classif. W${}_{H\alpha}$' % (califaID, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))
        # AXIS 1, 5, 9
        l_to_plot = ['6563', '4861']
        axs = [ax1, ax5, ax9]
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]]))
        axs[0].scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        axs[0].set_xlim(distance_range)
        axs[0].set_ylim(lineratios_range['/'.join(l_to_plot)])
        axs[0].set_xlabel(r'R [HLR]')
        axs[0].set_ylabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        axs[0].grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        x = np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]])
        im = axs[1].imshow(x, vmin=lineratios_range['/'.join(l_to_plot)][0], vmax=lineratios_range['/'.join(l_to_plot)][1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(axs[1])
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        DrawHLRCircle(axs[1], a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(axs[2], x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=lineratios_range['/'.join(l_to_plot)]))
        axs[2].set_xlabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        # AXIS 2, 6, 10
        l_to_plot = ['6583', '6563']
        axs = [ax2, ax6, ax10]
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]]))
        axs[0].scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        axs[0].set_xlim(distance_range)
        axs[0].set_ylim(lineratios_range['/'.join(l_to_plot)])
        axs[0].set_xlabel(r'R [HLR]')
        axs[0].set_ylabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        axs[0].grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        x = np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]])
        im = axs[1].imshow(x, vmin=lineratios_range['/'.join(l_to_plot)][0], vmax=lineratios_range['/'.join(l_to_plot)][1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(axs[1])
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        DrawHLRCircle(axs[1], a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(axs[2], x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=lineratios_range['/'.join(l_to_plot)]))
        axs[2].set_xlabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        # AXIS 3, 7, 11
        l_to_plot = ['6300', '6563']
        axs = [ax3, ax7, ax11]
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]]))
        axs[0].scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        axs[0].set_xlim(distance_range)
        axs[0].set_ylim(lineratios_range['/'.join(l_to_plot)])
        axs[0].set_xlabel(r'R [HLR]')
        axs[0].set_ylabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        axs[0].grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        x = np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]])
        im = axs[1].imshow(x, vmin=lineratios_range['/'.join(l_to_plot)][0], vmax=lineratios_range['/'.join(l_to_plot)][1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(axs[1])
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        DrawHLRCircle(axs[1], a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(axs[2], x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=lineratios_range['/'.join(l_to_plot)]))
        axs[2].set_xlabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        # AXIS 4, 8, 12
        l_to_plot = ['6717+6731', '6563']
        axs = [ax4, ax8, ax12]
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10((SB__yx['6717']+SB__yx['6731'])/SB__yx[l_to_plot[1]]))
        axs[0].scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        axs[0].set_xlim(distance_range)
        axs[0].set_ylim(lineratios_range['/'.join(l_to_plot)])
        axs[0].set_xlabel(r'R [HLR]')
        axs[0].set_ylabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        axs[0].grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        x = np.ma.log10((SB__yx['6717']+SB__yx['6731'])/SB__yx[l_to_plot[1]])
        im = axs[1].imshow(x, vmin=lineratios_range['/'.join(l_to_plot)][0], vmax=lineratios_range['/'.join(l_to_plot)][1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(axs[1])
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        DrawHLRCircle(axs[1], a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(axs[2], x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=lineratios_range['/'.join(l_to_plot)]))
        axs[2].set_xlabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))

        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('%s-maps_lineratios_colorsWHa.png' % califaID)
        plt.close(f)


def maps_lineratios_colorsZhang(ALL, gals=None):
    # Zhang DIG-COMP-HII decomposition
    sel_Zhang_DIG__yx = (ALL.SB6563__yx < DIG_Zhang_threshold).filled(False)
    sel_Zhang_COMP__yx = np.bitwise_and((ALL.SB6563__yx >= DIG_Zhang_threshold).filled(False), (ALL.SB6563__yx < HII_Zhang_threshold).filled(False))
    sel_Zhang_HII__yx = (ALL.SB6563__yx >= HII_Zhang_threshold).filled(False)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        pixelDistance__yx = ALL.get_gal_prop(califaID, ALL.pixelDistance__yx).reshape(N_y, N_x)
        pixelDistance_HLR__yx = pixelDistance__yx / HLR_pix
        SB__yx = {'%s' % L: ALL.get_gal_prop(califaID, 'SB%s__yx' % L).reshape(N_y, N_x) for L in lines}
        sel_DIG__yx = ALL.get_gal_prop(califaID, sel_Zhang_DIG__yx).reshape(N_y, N_x)
        sel_COMP__yx = ALL.get_gal_prop(califaID, sel_Zhang_COMP__yx).reshape(N_y, N_x)
        sel_HII__yx = ALL.get_gal_prop(califaID, sel_Zhang_HII__yx).reshape(N_y, N_x)
        map__yx = np.ma.masked_all((N_y, N_x))
        map__yx[sel_DIG__yx] = 1
        map__yx[sel_COMP__yx] = 2
        map__yx[sel_HII__yx] = 3

        distance_range = [0, 3]
        N_cols = 4
        N_rows = 3
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 4))
        cmap = cmap_discrete()
        ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = axArr
        f.suptitle(r'%s - %s: %d pixels (%d zones) - classif. $\Sigma_{H\alpha}$' % (califaID, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))
        # AXIS 1, 5, 9
        l_to_plot = ['6563', '4861']
        axs = [ax1, ax5, ax9]
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]]))
        axs[0].scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        axs[0].set_xlim(distance_range)
        axs[0].set_ylim(lineratios_range['/'.join(l_to_plot)])
        axs[0].set_xlabel(r'R [HLR]')
        axs[0].set_ylabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        axs[0].grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        x = np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]])
        im = axs[1].imshow(x, vmin=lineratios_range['/'.join(l_to_plot)][0], vmax=lineratios_range['/'.join(l_to_plot)][1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(axs[1])
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        DrawHLRCircle(axs[1], a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(axs[2], x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=lineratios_range['/'.join(l_to_plot)]))
        axs[2].set_xlabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        # AXIS 2, 6, 10
        l_to_plot = ['6583', '6563']
        axs = [ax2, ax6, ax10]
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]]))
        axs[0].scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        axs[0].set_xlim(distance_range)
        axs[0].set_ylim(lineratios_range['/'.join(l_to_plot)])
        axs[0].set_xlabel(r'R [HLR]')
        axs[0].set_ylabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        axs[0].grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        x = np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]])
        im = axs[1].imshow(x, vmin=lineratios_range['/'.join(l_to_plot)][0], vmax=lineratios_range['/'.join(l_to_plot)][1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(axs[1])
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        DrawHLRCircle(axs[1], a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(axs[2], x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=lineratios_range['/'.join(l_to_plot)]))
        axs[2].set_xlabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        # AXIS 3, 7, 11
        l_to_plot = ['6300', '6563']
        axs = [ax3, ax7, ax11]
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]]))
        axs[0].scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        axs[0].set_xlim(distance_range)
        axs[0].set_ylim(lineratios_range['/'.join(l_to_plot)])
        axs[0].set_xlabel(r'R [HLR]')
        axs[0].set_ylabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        axs[0].grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        x = np.ma.log10(SB__yx[l_to_plot[0]]/SB__yx[l_to_plot[1]])
        im = axs[1].imshow(x, vmin=lineratios_range['/'.join(l_to_plot)][0], vmax=lineratios_range['/'.join(l_to_plot)][1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(axs[1])
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        DrawHLRCircle(axs[1], a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(axs[2], x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=lineratios_range['/'.join(l_to_plot)]))
        axs[2].set_xlabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        # AXIS 4, 8, 12
        l_to_plot = ['6717+6731', '6563']
        axs = [ax4, ax8, ax12]
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10((SB__yx['6717']+SB__yx['6731'])/SB__yx[l_to_plot[1]]))
        axs[0].scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        axs[0].set_xlim(distance_range)
        axs[0].set_ylim(lineratios_range['/'.join(l_to_plot)])
        axs[0].set_xlabel(r'R [HLR]')
        axs[0].set_ylabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        axs[0].grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            axs[0].plot(rs.xS, rs.yS, 'k--', lw=2)
            axs[0].plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        x = np.ma.log10((SB__yx['6717']+SB__yx['6731'])/SB__yx[l_to_plot[1]])
        im = axs[1].imshow(x, vmin=lineratios_range['/'.join(l_to_plot)][0], vmax=lineratios_range['/'.join(l_to_plot)][1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(axs[1])
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))
        DrawHLRCircle(axs[1], a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(axs[2], x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=lineratios_range['/'.join(l_to_plot)]))
        axs[2].set_xlabel(r'$\log\ %s/%s$' % (l_to_plot[0], l_to_plot[1]))

        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('%s-maps_lineratios_colorsZhang.png' % califaID)
        plt.close(f)


def maps_colorsWHa(ALL, gals=None):
    # WHa DIG-COMP-HII decomposition
    sel_WHa_DIG__yx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__yx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__yx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        pixelDistance__yx = ALL.get_gal_prop(califaID, ALL.pixelDistance__yx).reshape(N_y, N_x)
        pixelDistance_HLR__yx = pixelDistance__yx / HLR_pix
        SB4861__yx = ALL.get_gal_prop(califaID, ALL.SB4861__yx).reshape(N_y, N_x)
        SB6563__yx = ALL.get_gal_prop(califaID, ALL.SB6563__yx).reshape(N_y, N_x)
        W6563__yx = ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x)
        sel_DIG__yx = ALL.get_gal_prop(califaID, sel_WHa_DIG__yx).reshape(N_y, N_x)
        sel_COMP__yx = ALL.get_gal_prop(califaID, sel_WHa_COMP__yx).reshape(N_y, N_x)
        sel_HII__yx = ALL.get_gal_prop(califaID, sel_WHa_HII__yx).reshape(N_y, N_x)
        map__yx = np.ma.masked_all((N_y, N_x))
        map__yx[sel_DIG__yx] = 1
        map__yx[sel_COMP__yx] = 2
        map__yx[sel_HII__yx] = 3

        distance_range = [0, 3]
        N_cols = 3
        N_rows = 4
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 4))
        cmap = cmap_discrete()
        ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = axArr
        f.suptitle(r'%s - %s: %d pixels (%d zones) - classif. W${}_{H\alpha}$' % (califaID, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))
        # AXIS 1
        galimg = plt.imread(P.get_image_file(califaID))[::-1, :, :]
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.imshow(galimg, origin='lower', aspect='equal')
        DrawHLRCircleInSDSSImage(ax1, HLR_pix, pa, ba, lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        txt = '%s' % califaID
        plot_text_ax(ax1, txt, 0.02, 0.98, 16, 'top', 'left', color='w')
        # AXIS 2
        im = ax2.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(['DIG', 'COMP', 'HII'])
        DrawHLRCircle(ax2, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # AXIS 3
        ax3.set_axis_off()
        # AXIS 4
        x = np.ravel(pixelDistance_HLR__yx)
        y = np.ravel(np.ma.log10(W6563__yx))
        ax4.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax4.set_xlim(distance_range)
        ax4.set_ylim(logWHa_range)
        ax4.set_xlabel(r'R [HLR]')
        ax4.set_ylabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        ax4.grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax4.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax4.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax4.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax4.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax4.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax4.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        # AXIS 5
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB6563__yx))
        ax5.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax5.set_xlim(distance_range)
        ax5.set_ylim(logSBHa_range)
        ax5.set_xlabel(r'R [HLR]')
        ax5.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax5.grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax5.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax5.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax5.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax5.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax5.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax5.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        # AXIS 6
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB6563__yx/SB4861__yx))
        ax6.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax6.set_xlim(distance_range)
        ax6.set_ylim(lineratios_range['6563/4861'])
        ax6.set_xlabel(r'R [HLR]')
        ax6.set_ylabel(r'$\log\ 6563/4861$')
        ax6.grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax6.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax6.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax6.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax6.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax6.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax6.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)

        x = np.ma.log10(W6563__yx)
        im = ax7.imshow(x, vmin=logWHa_range[0], vmax=logWHa_range[1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax7)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        DrawHLRCircle(ax7, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(ax10, x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=logWHa_range))
        ax10.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')

        x = np.ma.log10(SB6563__yx)
        im = ax8.imshow(x, vmin=logSBHa_range[0], vmax=logSBHa_range[1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax8)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        DrawHLRCircle(ax8, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(ax11, x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=logSBHa_range))
        ax11.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')

        x = np.ma.log10(SB6563__yx/SB4861__yx)
        im = ax9.imshow(x, vmin=lineratios_range['6563/4861'][0], vmax=lineratios_range['6563/4861'][1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax9)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ 6563/4861$')
        DrawHLRCircle(ax9, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(ax12, x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=lineratios_range['6563/4861']))
        ax12.set_xlabel(r'$\log\ 6563/4861$')

        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('%s-maps_colorsWHa.png' % califaID)
        plt.close(f)


def maps_colorsZhang(ALL, gals=None):
    # SBHa-Zhang DIG-COMP-HII decomposition
    sel_Zhang_DIG__yx = (ALL.SB6563__yx < DIG_Zhang_threshold).filled(False)
    sel_Zhang_COMP__yx = np.bitwise_and((ALL.SB6563__yx >= DIG_Zhang_threshold).filled(False), (ALL.SB6563__yx < HII_Zhang_threshold).filled(False))
    sel_Zhang_HII__yx = (ALL.SB6563__yx >= HII_Zhang_threshold).filled(False)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        pixelDistance__yx = ALL.get_gal_prop(califaID, ALL.pixelDistance__yx).reshape(N_y, N_x)
        pixelDistance_HLR__yx = pixelDistance__yx / HLR_pix
        SB4861__yx = ALL.get_gal_prop(califaID, ALL.SB4861__yx).reshape(N_y, N_x)
        SB6563__yx = ALL.get_gal_prop(califaID, ALL.SB6563__yx).reshape(N_y, N_x)
        W6563__yx = ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x)
        sel_DIG__yx = ALL.get_gal_prop(califaID, sel_Zhang_DIG__yx).reshape(N_y, N_x)
        sel_COMP__yx = ALL.get_gal_prop(califaID, sel_Zhang_COMP__yx).reshape(N_y, N_x)
        sel_HII__yx = ALL.get_gal_prop(califaID, sel_Zhang_HII__yx).reshape(N_y, N_x)
        map__yx = np.ma.masked_all((N_y, N_x))
        map__yx[sel_DIG__yx] = 1
        map__yx[sel_COMP__yx] = 2
        map__yx[sel_HII__yx] = 3

        distance_range = [0, 3]
        N_cols = 3
        N_rows = 4
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 4))
        cmap = cmap_discrete()
        ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = axArr
        f.suptitle(r'%s - %s: %d pixels (%d zones) - classif. $\Sigma_{H\alpha}$' % (califaID, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))
        # AXIS 1
        galimg = plt.imread(P.get_image_file(califaID))[::-1, :, :]
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.imshow(galimg, origin='lower', aspect='equal')
        DrawHLRCircleInSDSSImage(ax1, HLR_pix, pa, ba, lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        txt = '%s' % califaID
        plot_text_ax(ax1, txt, 0.02, 0.98, 16, 'top', 'left', color='w')
        # AXIS 2
        im = ax2.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(['DIG', 'COMP', 'HII'])
        DrawHLRCircle(ax2, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # AXIS 3
        ax3.set_axis_off()
        # AXIS 4
        x = np.ravel(pixelDistance_HLR__yx)
        y = np.ravel(np.ma.log10(W6563__yx))
        ax4.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax4.set_xlim(distance_range)
        ax4.set_ylim(logWHa_range)
        ax4.set_xlabel(r'R [HLR]')
        ax4.set_ylabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        ax4.grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax4.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax4.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax4.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax4.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax4.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax4.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        # AXIS 5
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB6563__yx))
        ax5.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax5.set_xlim(distance_range)
        ax5.set_ylim(logSBHa_range)
        ax5.set_xlabel(r'R [HLR]')
        ax5.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax5.grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax5.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax5.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax5.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax5.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax5.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax5.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        # AXIS 6
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB6563__yx/SB4861__yx))
        ax6.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax6.set_xlim(distance_range)
        ax6.set_ylim(lineratios_range['6563/4861'])
        ax6.set_xlabel(r'R [HLR]')
        ax6.set_ylabel(r'$\log\ 6563/4861$')
        ax6.grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax6.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax6.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax6.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax6.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax6.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax6.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)

        x = np.ma.log10(W6563__yx)
        im = ax7.imshow(x, vmin=logWHa_range[0], vmax=logWHa_range[1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax7)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        DrawHLRCircle(ax7, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(ax10, x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=logWHa_range))
        ax10.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')

        x = np.ma.log10(SB6563__yx)
        im = ax8.imshow(x, vmin=logSBHa_range[0], vmax=logSBHa_range[1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax8)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        DrawHLRCircle(ax8, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(ax11, x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=logSBHa_range))
        ax11.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')

        x = np.ma.log10(SB6563__yx/SB4861__yx)
        im = ax9.imshow(x, vmin=lineratios_range['6563/4861'][0], vmax=lineratios_range['6563/4861'][1], cmap='viridis', **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax9)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ 6563/4861$')
        DrawHLRCircle(ax9, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        plot_histo_ax(ax12, x.compressed(), y_v_space=0.06, first=True, c='k', kwargs_histo=dict(color='b', normed=False, range=lineratios_range['6563/4861']))
        ax12.set_xlabel(r'$\log\ 6563/4861$')

        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('%s-maps_colorsZhang.png' % califaID)
        plt.close(f)


def histograms_HaHb_Dt(ALL, gals=None):
    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')
    sel_gals__gyx = np.zeros((ALL.califaID__yx.shape), dtype='bool')

    for g in gals:
        where_gals__gz = np.where(ALL.califaID__z == g)
        where_gals__gyx = np.where(ALL.califaID__yx == g)
        if not where_gals__gz:
            continue
        sel_gals__gz[where_gals__gz] = True
        sel_gals__gyx[where_gals__gyx] = True

    if (sel_gals__gz).any():
        W6563__gz = ALL.W6563__z[sel_gals__gz]
        W6563__gyx = ALL.W6563__yx[sel_gals__gyx]

        SB4861__gz = ALL.SB4861__z[sel_gals__gz]
        SB4861__gyx = ALL.SB4861__yx[sel_gals__gyx]
        SB6563__gz = ALL.SB6563__z[sel_gals__gz]
        SB6563__gyx = ALL.SB6563__yx[sel_gals__gyx]

        tau_V_neb__gz = ALL.tau_V_neb__z[sel_gals__gz]
        tau_V_neb__gyx = ALL.tau_V_neb__yx[sel_gals__gyx]
        tau_V__gz = ALL.tau_V__z[sel_gals__gz]
        tau_V__gyx = ALL.tau_V__yx[sel_gals__gyx]

        # WHa DIG-COMP-HII decomposition
        sel_WHa_DIG__gz = (W6563__gz < DIG_WHa_threshold).filled(False)
        sel_WHa_COMP__gz = np.bitwise_and((W6563__gz >= DIG_WHa_threshold).filled(False), (W6563__gz < HII_WHa_threshold).filled(False))
        sel_WHa_HII__gz = (W6563__gz >= HII_WHa_threshold).filled(False)
        sel_WHa_DIG__gyx = (W6563__gyx < DIG_WHa_threshold).filled(False)
        sel_WHa_COMP__gyx = np.bitwise_and((W6563__gyx >= DIG_WHa_threshold).filled(False), (W6563__gyx < HII_WHa_threshold).filled(False))
        sel_WHa_HII__gyx = (W6563__gyx >= HII_WHa_threshold).filled(False)

        # SBHa-Zhang DIG-COMP-HII decomposition
        sel_Zhang_DIG__gz = (SB6563__gz < DIG_Zhang_threshold).filled(False)
        sel_Zhang_COMP__gz = np.bitwise_and((SB6563__gz >= DIG_Zhang_threshold).filled(False), (SB6563__gz < HII_Zhang_threshold).filled(False))
        sel_Zhang_HII__gz = (SB6563__gz >= HII_Zhang_threshold).filled(False)
        sel_Zhang_DIG__gyx = (SB6563__gyx < DIG_Zhang_threshold).filled(False)
        sel_Zhang_COMP__gyx = np.bitwise_and((SB6563__gyx >= DIG_Zhang_threshold).filled(False), (SB6563__gyx < HII_Zhang_threshold).filled(False))
        sel_Zhang_HII__gyx = (SB6563__gyx >= HII_Zhang_threshold).filled(False)

        N_cols = 2
        N_rows = 2
        f, axArr = plt.subplots(N_rows, N_cols, dpi=100, figsize=(15, 10))
        ((ax1, ax2), (ax3, ax4)) = axArr
        x = np.ma.log10(SB6563__gz/SB4861__gz)
        range = lineratios_range['6563/4861']
        xDs = [x[sel_WHa_DIG__gz].compressed(),  x[sel_WHa_COMP__gz].compressed(),  x[sel_WHa_HII__gz].compressed()]
        ax1.set_title('zones')
        plot_histo_ax(ax1, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        plot_histo_ax(ax1, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
        ax1.set_xlabel(r'$\log\ H\alpha/H\beta$')
        x = tau_V_neb__gz - tau_V__gz
        xDs = [x[sel_WHa_DIG__gz].compressed(),  x[sel_WHa_COMP__gz].compressed(),  x[sel_WHa_HII__gz].compressed()]
        range = DtauV_range
        ax2.set_title('zones')
        plot_histo_ax(ax2, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        plot_histo_ax(ax2, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
        ax2.set_xlabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        x = np.ma.log10(SB6563__gyx/SB4861__gyx)
        xDs = [x[sel_WHa_DIG__gyx].compressed(),  x[sel_WHa_COMP__gyx].compressed(),  x[sel_WHa_HII__gyx].compressed()]
        range = lineratios_range['6563/4861']
        ax3.set_title('pixels')
        plot_histo_ax(ax3, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        plot_histo_ax(ax3, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
        ax3.set_xlabel(r'$\log\ H\alpha/H\beta$')
        x = tau_V_neb__gyx - tau_V__gyx
        xDs = [x[sel_WHa_DIG__gyx].compressed(),  x[sel_WHa_COMP__gyx].compressed(),  x[sel_WHa_HII__gyx].compressed()]
        range = DtauV_range
        ax4.set_title('pixels')
        plot_histo_ax(ax4, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        plot_histo_ax(ax4, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
        ax4.set_xlabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        f.tight_layout()
        f.savefig('dig-sample-histo-logHaHb_Dt_classifWHa.png')

        N_cols = 2
        N_rows = 2
        f, axArr = plt.subplots(N_rows, N_cols, dpi=100, figsize=(15, 10))
        ((ax1, ax2), (ax3, ax4)) = axArr
        x = np.ma.log10(SB6563__gz/SB4861__gz)
        range = lineratios_range['6563/4861']
        xDs = [x[sel_Zhang_DIG__gz].compressed(),  x[sel_Zhang_COMP__gz].compressed(),  x[sel_Zhang_HII__gz].compressed()]
        ax1.set_title('zones')
        plot_histo_ax(ax1, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        plot_histo_ax(ax1, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
        ax1.set_xlabel(r'$\log\ H\alpha/H\beta$')
        x = tau_V_neb__gz - tau_V__gz
        xDs = [x[sel_Zhang_DIG__gz].compressed(),  x[sel_Zhang_COMP__gz].compressed(),  x[sel_Zhang_HII__gz].compressed()]
        range = DtauV_range
        ax2.set_title('zones')
        plot_histo_ax(ax2, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        plot_histo_ax(ax2, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
        ax2.set_xlabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        x = np.ma.log10(SB6563__gyx/SB4861__gyx)
        xDs = [x[sel_Zhang_DIG__gyx].compressed(),  x[sel_Zhang_COMP__gyx].compressed(),  x[sel_Zhang_HII__gyx].compressed()]
        range = lineratios_range['6563/4861']
        ax3.set_title('pixels')
        plot_histo_ax(ax3, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        plot_histo_ax(ax3, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
        ax3.set_xlabel(r'$\log\ H\alpha/H\beta$')
        x = tau_V_neb__gyx - tau_V__gyx
        xDs = [x[sel_Zhang_DIG__gyx].compressed(),  x[sel_Zhang_COMP__gyx].compressed(),  x[sel_Zhang_HII__gyx].compressed()]
        range = DtauV_range
        ax4.set_title('pixels')
        plot_histo_ax(ax4, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        plot_histo_ax(ax4, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
        ax4.set_xlabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        f.tight_layout()
        f.savefig('dig-sample-histo-logHaHb_Dt_classifZhang.png')


def Dt_xY_profile_sample(ALL, gals=None):
    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')
    sel_gals__gyx = np.zeros((ALL.califaID__yx.shape), dtype='bool')

    for g in gals:
        where_gals__gz = np.where(ALL.califaID__z == g)
        where_gals__gyx = np.where(ALL.califaID__yx == g)
        if not where_gals__gz:
            continue
        sel_gals__gz[where_gals__gz] = True
        sel_gals__gyx[where_gals__gyx] = True

    if (sel_gals__gz).any():
        W6563__gz = ALL.W6563__z[sel_gals__gz]
        W6563__gyx = ALL.W6563__yx[sel_gals__gyx]

        SB6563__gz = ALL.SB6563__z[sel_gals__gz]
        SB6563__gyx = ALL.SB6563__yx[sel_gals__gyx]

        x_Y__gz = ALL.x_Y__z[sel_gals__gz]
        x_Y__gyx = ALL.x_Y__yx[sel_gals__gyx]
        tau_V_neb__gz = ALL.tau_V_neb__z[sel_gals__gz]
        tau_V_neb__gyx = ALL.tau_V_neb__yx[sel_gals__gyx]
        tau_V__gz = ALL.tau_V__z[sel_gals__gz]
        tau_V__gyx = ALL.tau_V__yx[sel_gals__gyx]

        # WHa DIG-COMP-HII decomposition
        sel_WHa_DIG__gz = (W6563__gz < DIG_WHa_threshold).filled(False)
        sel_WHa_COMP__gz = np.bitwise_and((W6563__gz >= DIG_WHa_threshold).filled(False), (W6563__gz < HII_WHa_threshold).filled(False))
        sel_WHa_HII__gz = (W6563__gz >= HII_WHa_threshold).filled(False)
        sel_WHa_DIG__gyx = (W6563__gyx < DIG_WHa_threshold).filled(False)
        sel_WHa_COMP__gyx = np.bitwise_and((W6563__gyx >= DIG_WHa_threshold).filled(False), (W6563__gyx < HII_WHa_threshold).filled(False))
        sel_WHa_HII__gyx = (W6563__gyx >= HII_WHa_threshold).filled(False)

        # SBHa-Zhang DIG-COMP-HII decomposition
        sel_Zhang_DIG__gz = (SB6563__gz < DIG_Zhang_threshold).filled(False)
        sel_Zhang_COMP__gz = np.bitwise_and((SB6563__gz >= DIG_Zhang_threshold).filled(False), (SB6563__gz < HII_Zhang_threshold).filled(False))
        sel_Zhang_HII__gz = (SB6563__gz >= HII_Zhang_threshold).filled(False)
        sel_Zhang_DIG__gyx = (SB6563__gyx < DIG_Zhang_threshold).filled(False)
        sel_Zhang_COMP__gyx = np.bitwise_and((SB6563__gyx >= DIG_Zhang_threshold).filled(False), (SB6563__gyx < HII_Zhang_threshold).filled(False))
        sel_Zhang_HII__gyx = (SB6563__gyx >= HII_Zhang_threshold).filled(False)

        N_cols = 2
        N_rows = 1
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        cmap = cmap_discrete()
        ax1, ax2 = axArr
        # AXIS 1
        x = x_Y__gz
        y = tau_V_neb__gz - tau_V__gz
        classif = np.ma.masked_all(SB6563__gz.shape)
        classif[sel_WHa_DIG__gz] = 1
        classif[sel_WHa_COMP__gz] = 2
        classif[sel_WHa_HII__gz] = 3
        xbin = np.linspace(0, 0.6, 30)
        sc = ax1.scatter(x, y, c=classif, cmap=cmap, s=1, **dflt_kw_scatter)
        the_divider = make_axes_locatable(ax1)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(sc, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(['DIG', 'COMP', 'HII'])
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_DIG__gz)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax1.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax1.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_COMP__gz)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax1.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax1.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_HII__gz)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax1.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax1.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        ax1.set_xlim(x_Y_range)
        ax1.set_ylim(DtauV_range)
        ax1.set_xlabel(r'x${}_Y$ [frac.]')
        ax1.set_ylabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        ax1.set_title('zones')
        ax1.grid()
        # AXIS 2
        x = x_Y__gyx
        y = tau_V_neb__gyx - tau_V__gyx
        classif = np.ma.masked_all(SB6563__gyx.shape)
        classif[sel_WHa_DIG__gyx] = 1
        classif[sel_WHa_COMP__gyx] = 2
        classif[sel_WHa_HII__gyx] = 3
        xbin = np.linspace(0, 0.6, 30)
        sc = ax2.scatter(x, y, c=classif, cmap=cmap, s=1, **dflt_kw_scatter)
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(sc, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(['DIG', 'COMP', 'HII'])
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_DIG__gyx)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax2.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax2.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_COMP__gyx)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax2.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax2.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_HII__gyx)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax2.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax2.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        ax2.set_xlim(x_Y_range)
        ax2.set_ylim(DtauV_range)
        ax2.set_xlabel(r'x${}_Y$ [frac.]')
        ax2.set_ylabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        ax2.grid()
        ax2.set_title('pixels')
        f.tight_layout()
        f.savefig('dig-sample-xY_Dt-classifWHa.png')

        N_cols = 2
        N_rows = 1
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        cmap = cmap_discrete()
        ax1, ax2 = axArr
        # AXIS 1
        x = x_Y__gz
        y = tau_V_neb__gz - tau_V__gz
        classif = np.ma.masked_all(SB6563__gz.shape)
        classif[sel_Zhang_DIG__gz] = 1
        classif[sel_Zhang_COMP__gz] = 2
        classif[sel_Zhang_HII__gz] = 3
        xbin = np.linspace(0, 0.6, 30)
        sc = ax1.scatter(x, y, c=classif, cmap=cmap, s=1, **dflt_kw_scatter)
        the_divider = make_axes_locatable(ax1)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(sc, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(['DIG', 'COMP', 'HII'])
        xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_DIG__gz)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax1.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax1.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_COMP__gz)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax1.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax1.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_HII__gz)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax1.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax1.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        ax1.set_xlim(x_Y_range)
        ax1.set_ylim(DtauV_range)
        ax1.set_xlabel(r'x${}_Y$ [frac.]')
        ax1.set_ylabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        ax1.set_title('zones')
        ax1.grid()
        # AXIS 2
        x = x_Y__gyx
        y = tau_V_neb__gyx - tau_V__gyx
        classif = np.ma.masked_all(SB6563__gyx.shape)
        classif[sel_Zhang_DIG__gyx] = 1
        classif[sel_Zhang_COMP__gyx] = 2
        classif[sel_Zhang_HII__gyx] = 3
        xbin = np.linspace(0, 0.6, 30)
        sc = ax2.scatter(x, y, c=classif, cmap=cmap, s=1, **dflt_kw_scatter)
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(sc, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(['DIG', 'COMP', 'HII'])
        xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_DIG__gyx)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax2.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax2.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_COMP__gyx)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax2.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax2.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_HII__gyx)
        rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
        ax2.plot(rs.xS, rs.yS, 'k--', lw=3)
        ax2.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        ax2.set_xlim(x_Y_range)
        ax2.set_ylim(DtauV_range)
        ax2.set_xlabel(r'x${}_Y$ [frac.]')
        ax2.set_ylabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        ax2.grid()
        ax2.set_title('pixels')
        f.tight_layout()
        f.savefig('dig-sample-xY_Dt-classifZhang.png')


if __name__ == '__main__':
    main(sys.argv)
