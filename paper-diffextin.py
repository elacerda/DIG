import sys
import numpy as np
import matplotlib as mpl
from pytu.lines import Lines
from matplotlib import pyplot as plt
from pycasso.util import radialProfile
from pystarlight.util.constants import L_sun
from matplotlib.ticker import AutoMinorLocator
from pytu.functions import ma_mask_xyz, debug_var
from pystarlight.util.redenninglaws import calc_redlaw
from CALIFAUtils.scripts import get_NEDName_by_CALIFAID
from CALIFAUtils.objects import stack_gals, CALIFAPaths
from mpl_toolkits.axes_grid1 import make_axes_locatable
from CALIFAUtils.plots import DrawHLRCircleInSDSSImage, DrawHLRCircle
from pytu.plots import cmap_discrete, plot_text_ax, density_contour, plot_scatter_histo, plot_histo_ax, stats_med12sigma, add_subplot_axes, plot_text_ax

mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
colors_DIG_COMP_HII = ['tomato', 'lightgreen', 'royalblue']
colors_lines_DIG_COMP_HII = ['darkred', 'olive', 'mediumblue']
classif_labels = ['DIG', 'COMP', 'SF']
cmap_R = plt.cm.copper_r
minorLocator = AutoMinorLocator(5)

debug = False
# debug = True
# CCM reddening law
q = calc_redlaw([4861, 6563], R_V=3.1, redlaw='CCM')
# config variables
lineratios_range = {
    '6563/4861': [0.15, 0.75],
    '6583/6563': [-0.7, 0.1],
    '6300/6563': [-2, 0],
    '6717+6731/6563': [-0.8, 0.2],
}
distance_range = [0, 3]
tauVneb_range = [0, 5]
tauVneb_neg_range = [-1, 5]
logSBHa_range = [4, 7]
logWHa_range = [0, 2.5]
DtauV_range = [-3, 3]
DeltatauV_range = [-1, 3]
DtauVnorm_range = [-1, 4]
x_Y_range = [0, 0.6]
OH_range = [8, 9.5]
# age to calc xY
tY = 100e6
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
dflt_kw_runstats = dict(smooth=True, sigma=1.2, frac=0.07, debug=True)  # , tendency=True)
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

    # if debug:
    #     gals = ['K0073']
    # sel3gals = ['K0010', 'K0813', 'K0187']
    sel3gals = ['K0010', 'K0813', 'K0836']
    """
    Fig 1.
        Example fig (maps, images, spectral fit and a zoom on EML on resid. spectra)
    """
    # fig1(ALL, gals)

    """
    Fig 2.
        Scatter and histogram of SBHa vs WHa.
        Histograms should be colored by WHa classif.
        Scatter could be colored by zone distance from center. (grayscale?)
        -- WHa_SBHa_zones_sample_histograms() from diffextin-experiences.py
    """
    fig2(ALL, gals)

    """
    Fig 3.
        Maps: SDSS stamp, SBHa, WHa, galaxy map colored by WHa classif.
        TODO: choose 3 example galaxies (Sa, Sb and Sc??)
    """
    # fig3(ALL, gals)  # gals=['K0010', 'K0187', 'K0813', 'K0388'])
    fig3_3gals(ALL, gals=sel3gals)

    """
    Fig 4.
        BPT diagrams: [OIII]/Hb vs (panel a:[NII]/Ha, panel b:[SII]/Ha, panel c:[OI]/Ha)
        Should be those same example galaxies from Fig. 3.
    """
    #  fig4(ALL, gals)  # gals=['K0010', 'K0187', 'K0813', 'K0388'])
    fig4_3gals(ALL, gals=sel3gals)

    """
    Fig 5.
        panel a: [OIII]/Hb vs [NII]/Ha for entire sample
        panel b: [OIII]/Hb vs [NII]/Ha 2d histogram painting 2d cell choosing
            the color by the bootstrap classif. stats.
    """
    # fig5(ALL, gals)
    fig5_2panels(ALL, gals)

    """
    Fig 6.
        Ha/Hb (or tau_V_neb) vs R
        Should be those same example galaxies from Fig. 3.
    """
    # fig6(ALL, gals)  #  gals=['K0010', 'K0187', 'K0813', 'K0388'])
    fig6_3gals(ALL, gals=sel3gals)

    """
    Fig 7.
        panel a: Histogram of Ha/Hb (or tau_V_neb)
        panel b: Histogram of D_tau_classif (tau_HII - tau_DIG)
        panel c: Histogram of D_tau_classif (tau_HII - tau_DIG)/integrated_tau_V_neb
        All sample.
    """
    fig7(ALL, gals)

    """
    Fig 8.
        Histogram of D_tau (tau_V_neb - tau_V)
        All sample.
        -- histograms_HaHb_Dt() from diffextin-experiences.py
    """
    fig8(ALL, gals)

    """
    Fig 9.
        D_tau (tau_V_neb - tau_V) vs x_Y
        All sample.
        -- Dt_xY_profile_sample() from diffextin-experiences.py
    """
    # fig9(ALL, gals)


def plotBPT(ax, N2Ha, O3Hb, z=None, cmap='viridis', mask=None, labels=True, N=False, cb_label=r'R [HLR]', vmax=None, vmin=None, dcontour=True, s=10):
    if mask is None:
        mask = np.zeros_like(O3Hb, dtype=np.bool_)
    extent = [-1.5, 1, -1.5, 1.5]
    if z is None:
        bins = [30, 30]
        xm, ym = ma_mask_xyz(N2Ha, O3Hb, mask=mask)
        if dcontour:
            density_contour(xm.compressed(), ym.compressed(), bins[0], bins[1], ax, range=[extent[0:2], extent[2:4]], colors=['b', 'y', 'r'])
        sc = ax.scatter(xm, ym, marker='o', c='0.5', s=s, edgecolor='none', alpha=0.4)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
    else:
        xm, ym, z = ma_mask_xyz(N2Ha, O3Hb, z, mask=mask)
        sc = ax.scatter(xm, ym, c=z, cmap=cmap, vmin=vmin, vmax=vmax, marker='o', s=s, edgecolor='none')
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
        ax.set_aspect('equal', 'box')
        the_divider = make_axes_locatable(ax)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(sc, cax=color_axis)
        # cb = plt.colorbar(sc, ax=ax, ticks=[0, .5, 1, 1.5, 2, 2.5, 3], pad=0)
        cb.set_label(cb_label)
    if labels:
        ax.set_xlabel(r'$\log\ [NII]/H\alpha$')
        ax.set_ylabel(r'$\log\ [OIII]/H\beta$')
    L = Lines()
    if not N:
        N = xm.count()
    c = ''
    if (xm.compressed() < extent[0]).any():
        c += 'x-'
    if (xm.compressed() > extent[1]).any():
        c += 'x+'
    if (ym.compressed() < extent[2]).any():
        c += 'y-'
    if (ym.compressed() > extent[3]).any():
        c += 'y+'
    plot_text_ax(ax, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
    plot_text_ax(ax, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
    ax.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
    ax.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
    ax.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
    plot_text_ax(ax, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=38)  # 44.62)
    ax.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
    L.fixCF10('S06')
    return ax


def fig2(ALL, gals=None):
    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')
    sel_gals__gyx = np.zeros((ALL.califaID__yx.shape), dtype='bool')

    N_gals = 0
    for g in gals:
        where_gals__gz = np.where(ALL.califaID__z == g)
        where_gals__gyx = np.where(ALL.califaID__yx == g)
        if not where_gals__gz:
            continue
        sel_gals__gz[where_gals__gz] = True
        sel_gals__gyx[where_gals__gyx] = True
        N_gals += 1

    if (sel_gals__gz).any():
        W6563__gz = ALL.W6563__z[sel_gals__gz]
        SB6563__gz = ALL.SB6563__z[sel_gals__gz]
        dist__gz = ALL.zoneDistance_HLR[sel_gals__gz]

        # WHa DIG-COMP-HII decomposition
        sel_WHa_DIG__gz = (W6563__gz < DIG_WHa_threshold).filled(False)
        sel_WHa_COMP__gz = np.bitwise_and((W6563__gz >= DIG_WHa_threshold).filled(False), (W6563__gz < HII_WHa_threshold).filled(False))
        sel_WHa_HII__gz = (W6563__gz >= HII_WHa_threshold).filled(False)

        x = np.ma.log10(W6563__gz)
        y = np.ma.log10(SB6563__gz)
        f = plt.figure(figsize=(8, 8))
        x_ds = [x[sel_WHa_DIG__gz].compressed(), x[sel_WHa_COMP__gz].compressed(), x[sel_WHa_HII__gz].compressed()]
        y_ds = [y[sel_WHa_DIG__gz].compressed(), y[sel_WHa_COMP__gz].compressed(), y[sel_WHa_HII__gz].compressed()]
        axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logWHa_range, logSBHa_range, 50, 50,
                                             figure=f, c=colors_DIG_COMP_HII, scatter=False, s=1,
                                             ylabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]',
                                             xlabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]', histtype='step')

        scater_kwargs = dict(c=dist__gz, s=3, vmax=3, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
        sc = axS.scatter(x, y, **scater_kwargs)
        cbaxes = f.add_axes([0.6, 0.17, 0.12, 0.02])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[0, 1, 2, 3], orientation='horizontal')
        cb.set_label(r'R [HLR]', fontsize=14)
        axS.axhline(y=np.log10(DIG_Zhang_threshold), color='k', linestyle='-.', lw=2)
        axS.axhline(y=np.log10(HII_Zhang_threshold), color='k', linestyle='-.', lw=2)
        axS.set_xlim(logWHa_range)
        axS.set_ylim(logSBHa_range)
        axS.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        axS.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        xbins = np.linspace(0.1, 2, 20)
        yMean, prc, bin_center, npts = stats_med12sigma(x, y, xbins)
        yMedian = prc[2]
        y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
        axS.plot(bin_center, yMedian, 'k-', lw=2)
        for y_prc in y_12sigma:
            axS.plot(bin_center, y_prc, 'k--', lw=2)
        axS.grid()
        f.savefig('fig2.png')
        plt.close(f)


def fig3(ALL, gals=None):
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
        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        SB__lyx = {'%s' % L: ALL.get_gal_prop(califaID, 'SB%s__yx' % L).reshape(N_y, N_x) for L in lines}
        W6563__yx = ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x)

        N_cols = 4
        N_rows = 1
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        cmap = cmap_discrete(colors=colors_DIG_COMP_HII)
        ax1, ax2, ax3, ax4 = axArr
        f.suptitle(r'%s - %s (%s): %d pixels (%d zones)' % (califaID, mto, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))
        # AXIS 1
        galimg = plt.imread(P.get_image_file(califaID))[::-1, :, :]
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.imshow(galimg, origin='lower', aspect='equal')
        DrawHLRCircleInSDSSImage(ax1, HLR_pix, pa, ba, lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        txt = '%s' % califaID
        plot_text_ax(ax1, txt, 0.02, 0.98, 16, 'top', 'left', color='w')
        ax1.set_title('SDSS stamp')
        # AXIS 2
        x = np.ma.log10(W6563__yx)
        im = ax2.imshow(x, vmin=logWHa_range[0], vmax=logWHa_range[1], cmap=cmap_R, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        DrawHLRCircle(ax2, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        ax2.set_title(r'$\log$ W${}_{H\alpha}$ map')
        # AXIS 3
        x = np.ma.log10(SB__lyx['6563'])
        im = ax3.imshow(x, vmin=logSBHa_range[0], vmax=logSBHa_range[1], cmap=cmap_R, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax3)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_label(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        DrawHLRCircle(ax3, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        ax3.set_title(r'$\log\ \Sigma_{H\alpha}$ map')
        # AXIS 4
        map__yx = create_segmented_map(ALL, califaID, sel_WHa_DIG__yx, sel_WHa_COMP__yx, sel_WHa_HII__yx)
        im = ax4.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax4)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(classif_labels)
        DrawHLRCircle(ax4, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        ax4.set_title(r'classif. map')
        # FINAL
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('fig3-%s.png' % califaID)
        plt.close(f)


def fig3_3gals(ALL, gals=None):
    # WHa DIG-COMP-HII decomposition
    sel_WHa_DIG__yx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__yx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__yx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)

    N_cols = 4
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
    cmap = cmap_discrete(colors=colors_DIG_COMP_HII)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    row = 0
    for califaID in gals:
        (ax1, ax2, ax3, ax4) = axArr[row]
        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        SB__lyx = {'%s' % L: ALL.get_gal_prop(califaID, 'SB%s__yx' % L).reshape(N_y, N_x) for L in lines}
        W6563__yx = ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x)
        # AXIS 1
        galimg = plt.imread(P.get_image_file(califaID))[::-1, :, :]
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.imshow(galimg, origin='lower', aspect='equal')
        DrawHLRCircleInSDSSImage(ax1, HLR_pix, pa, ba, lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.02, 0.98, 16, 'top', 'left', color='w')
        ax1.set_ylabel('%s' % mto, fontsize=24)
        # AXIS 2
        x = np.ma.log10(W6563__yx)
        im = ax2.imshow(x, vmin=logWHa_range[0], vmax=logWHa_range[1], cmap=cmap_R, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        # cb.set_label(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        DrawHLRCircle(ax2, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # AXIS 3
        x = np.ma.log10(SB__lyx['6563'])
        im = ax3.imshow(x, vmin=logSBHa_range[0], vmax=logSBHa_range[1], cmap=cmap_R, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax3)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        # cb.set_label(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        DrawHLRCircle(ax3, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # AXIS 4
        map__yx = create_segmented_map(ALL, califaID, sel_WHa_DIG__yx, sel_WHa_COMP__yx, sel_WHa_HII__yx)
        im = ax4.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax4)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(classif_labels)
        DrawHLRCircle(ax4, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        if row == 0:
            ax1.set_title('SDSS stamp', fontsize=18)
            # ax2.set_title(r'$\log$ W${}_{H\alpha}$ map')
            ax2.set_title(r'$\log$ W${}_{H\alpha}$ [$\AA$]', fontsize=18)
            # ax3.set_title(r'$\log\ \Sigma_{H\alpha}$ map')
            ax3.set_title(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]', fontsize=18)
            ax4.set_title(r'classif. map', fontsize=18)
        row += 1
    # FINAL
    f.tight_layout()
    f.savefig('fig3.png')
    plt.close(f)


def fig4(ALL, gals=None):
    # WHa DIG-COMP-HII decomposition
    sel_WHa_DIG__z = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__z = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__z = (ALL.W6563__z >= HII_WHa_threshold).filled(False)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        L = Lines()
        L.addLine('K01S2', L.linebpt, (1.30, 0.72, -0.32), np.linspace(-10.0, 0.32, L.xn + 1)[:-1])
        L.addLine('K01OI', L.linebpt, (1.33, 0.73, 0.59), np.linspace(-10.0, -0.59, L.xn + 1)[:-1])
        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        f__lz = {'%s' % L: ALL.get_gal_prop(califaID, 'f%s__z' % L) for L in lines}
        O3Hb__z = np.ma.log10(f__lz['5007']/f__lz['4861'])
        N2Ha__z = np.ma.log10(f__lz['6583']/f__lz['6563'])
        S2Ha__z = np.ma.log10((f__lz['6717'] + f__lz['6731'])/f__lz['6563'])
        O1Ha__z = np.ma.log10(f__lz['6300']/f__lz['6563'])

        N_cols = 3
        N_rows = 1
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        cmap = cmap_discrete(colors=colors_DIG_COMP_HII)
        ax1, ax2, ax3 = axArr
        f.suptitle(r'%s - %s (%s): %d pixels (%d zones)' % (califaID, mto, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))

        # AXIS 1
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_HII__z)
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(N2Ha__z, O3Hb__z)
        sc = ax1.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        ax1.set_xlim(extent[0:2])
        ax1.set_ylim(extent[2:4])
        ax1.set_xlabel(r'$\log\ [NII]/H\alpha$')
        ax1.set_ylabel(r'$\log\ [OIII]/H\beta$')
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax1, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax1, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
        ax1.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax1.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        ax1.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        plot_text_ax(ax1, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=35)  # 44.62)
        ax1.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
        L.fixCF10('S06')

        # AXIS 2
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_HII__z)
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(S2Ha__z, O3Hb__z)
        sc = ax2.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        ax2.set_xlim(extent[0:2])
        ax2.set_ylim(extent[2:4])
        ax2.set_xlabel(r'$\log\ [SII]/H\alpha$')
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax2, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax2, 'K01', 0.75, 0.02, 20, 'bottom', 'right', 'k')
        ax2.plot(L.x['K01S2'], L.y['K01S2'], 'k-', label='K01')

        # AXIS 3
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_HII__z)
        extent = [-2.5, 0, -1.5, 1.5]
        xm, ym = ma_mask_xyz(O1Ha__z, O3Hb__z)
        sc = ax3.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        ax3.set_xlim(extent[0:2])
        ax3.set_ylim(extent[2:4])
        cbaxes = add_subplot_axes(ax3, [0.8, 0.91, 0.5, 0.04])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        cb.ax.set_xticklabels(classif_labels, fontsize=9)
        ax3.set_xlabel(r'$\log\ [OI]/H\alpha$')
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax3, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax3, 'K01', 0.80, 0.02, 20, 'bottom', 'right', 'k')
        ax3.plot(L.x['K01OI'], L.y['K01OI'], 'k-', label='K01')

        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)

        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        ax2.xaxis.set_minor_locator(minorLocator)
        ax2.yaxis.set_minor_locator(minorLocator)
        ax3.xaxis.set_minor_locator(minorLocator)
        ax3.yaxis.set_minor_locator(minorLocator)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('fig4-%s.png' % califaID)
        plt.close(f)


def fig4_3gals(ALL, gals=None):
    # WHa DIG-COMP-HII decomposition
    sel_WHa_DIG__z = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__z = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__z = (ALL.W6563__z >= HII_WHa_threshold).filled(False)

    N_cols = 3
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
    cmap = cmap_discrete(colors=colors_DIG_COMP_HII)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    row = 0
    for califaID in gals:
        ax1, ax2, ax3 = axArr[row]

        L = Lines()
        L.addLine('K01S2', L.linebpt, (1.30, 0.72, -0.32), np.linspace(-10.0, 0.32, L.xn + 1)[:-1])
        L.addLine('K01OI', L.linebpt, (1.33, 0.73, 0.59), np.linspace(-10.0, -0.59, L.xn + 1)[:-1])
        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        f__lz = {'%s' % L: ALL.get_gal_prop(califaID, 'f%s__z' % L) for L in lines}
        O3Hb__z = np.ma.log10(f__lz['5007']/f__lz['4861'])
        N2Ha__z = np.ma.log10(f__lz['6583']/f__lz['6563'])
        S2Ha__z = np.ma.log10((f__lz['6717'] + f__lz['6731'])/f__lz['6563'])
        O1Ha__z = np.ma.log10(f__lz['6300']/f__lz['6563'])

        # AXIS 1
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_HII__z)
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(N2Ha__z, O3Hb__z)
        sc = ax1.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        ax1.set_xlim(extent[0:2])
        ax1.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax1, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax1, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
        ax1.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax1.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        ax1.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        plot_text_ax(ax1, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=35)  # 44.62)
        ax1.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
        L.fixCF10('S06')
        # AXIS 2
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_HII__z)
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(S2Ha__z, O3Hb__z)
        sc = ax2.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        ax2.set_xlim(extent[0:2])
        ax2.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax2, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax2, 'K01', 0.60, 0.02, 20, 'bottom', 'right', 'k')
        ax2.plot(L.x['K01S2'], L.y['K01S2'], 'k-', label='K01')
        # AXIS 3
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_HII__z)
        extent = [-2.5, 0, -1.5, 1.5]
        xm, ym = ma_mask_xyz(O1Ha__z, O3Hb__z)
        sc = ax3.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        ax3.set_xlim(extent[0:2])
        ax3.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax3, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax3, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
        plot_text_ax(ax3, 'K01', 0.65, 0.02, 20, 'bottom', 'right', 'k')
        ax3.plot(L.x['K01OI'], L.y['K01OI'], 'k-', label='K01')

        if row < 2:
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax3.get_xticklabels(), visible=False)
        if row == 1:
            ax1.set_ylabel(r'$\log\ [OIII]/H\beta$')
        if row == 2:
            ax1.set_xlabel(r'$\log\ [NII]/H\alpha$')
            ax2.set_xlabel(r'$\log\ [SII]/H\alpha$')
            ax3.set_xlabel(r'$\log\ [OI]/H\alpha$')
            cbaxes = add_subplot_axes(ax3, [0.7, 0.99, 0.6, 0.07])
            cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
            cb.ax.set_xticklabels(classif_labels, fontsize=12)

        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        ax2.xaxis.set_minor_locator(minorLocator)
        ax2.yaxis.set_minor_locator(minorLocator)
        ax3.xaxis.set_minor_locator(minorLocator)
        ax3.yaxis.set_minor_locator(minorLocator)
        row += 1
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.tight_layout(w_pad=0.05, h_pad=0)
    f.savefig('fig4.png')
    plt.close(f)


def fig5(ALL, gals=None):
    sel_WHa_DIG__gz = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__gz = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__gz = (ALL.W6563__z >= HII_WHa_threshold).filled(False)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')

    N_gals = 0
    for g in gals:
        where_gals__gz = np.where(ALL.califaID__z == g)
        if not where_gals__gz:
            continue
        sel_gals__gz[where_gals__gz] = True
        N_gals += 1

    if (sel_gals__gz).any():
        W6563__gz = ALL.W6563__z[sel_gals__gz]

        O3Hb__gz = np.ma.log10(ALL.f5007__z/ALL.f4861__z)
        N2Ha__gz = np.ma.log10(ALL.f6583__z/ALL.f6563__z)
        S2Ha__gz = np.ma.log10((ALL.f6717__z+ALL.f6731__z)/ALL.f6563__z)
        OIHa__gz = np.ma.log10(ALL.f6300__z/ALL.f6563__z)
        L = Lines()
        L.addLine('K01S2', L.linebpt, (1.30, 0.72, -0.32), np.linspace(-10.0, 0.32, L.xn + 1)[:-1])
        L.addLine('K01OI', L.linebpt, (1.33, 0.73, 0.59), np.linspace(-10.0, -0.59, L.xn + 1)[:-1])

        classif = np.ma.masked_all(W6563__gz.shape)
        classif[sel_WHa_DIG__gz] = 1
        classif[sel_WHa_COMP__gz] = 2
        classif[sel_WHa_HII__gz] = 3

        N_cols = 3
        N_rows = 1
        # f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        f = plt.figure(dpi=200, figsize=(6, 5))
        # ax1, ax2, ax3 = axArr
        ax1 = f.gca()
        cmap = cmap_discrete(colors_DIG_COMP_HII)
        # AXIS 1
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(N2Ha__gz, O3Hb__gz)
        # sc = ax1.scatter(xm, ym, c=np.ma.log10(W6563__gz), vmin=np.log10(DIG_WHa_threshold), vmax=np.log10(HII_WHa_threshold), cmap='rainbow_r', s=1, marker='o', edgecolor='none')
        sc = ax1.scatter(xm, ym, c=classif, cmap=cmap, s=1, marker='o', edgecolor='none')
        ax1.set_xlim(extent[0:2])
        ax1.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax1, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax1, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        # plot_text_ax(ax1, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K03', 0.55, 0.02, 20, 'bottom', 'left', 'k')
        # plot_text_ax(ax1, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
        ax1.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax1.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        # ax1.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        # plot_text_ax(ax1, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=35)  # 44.62)
        # ax1.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
        # L.fixCF10('S06')
        # # AXIS 2
        # extent = [-1.5, 1, -1.5, 1.5]
        # xm, ym = ma_mask_xyz(S2Ha__gz, O3Hb__gz)
        # # sc = ax2.scatter(xm, ym, c=np.ma.log10(W6563__gz), vmin=np.log10(DIG_WHa_threshold), vmax=np.log10(HII_WHa_threshold), cmap='rainbow_r', s=1, marker='o', edgecolor='none')
        # sc = ax2.scatter(xm, ym, c=classif, cmap=cmap, s=1, marker='o', edgecolor='none')
        # ax2.set_xlim(extent[0:2])
        # ax2.set_ylim(extent[2:4])
        # N = xm.count()
        # c = ''
        # if (xm.compressed() < extent[0]).any():
        #     c += 'x-'
        # if (xm.compressed() > extent[1]).any():
        #     c += 'x+'
        # if (ym.compressed() < extent[2]).any():
        #     c += 'y-'
        # if (ym.compressed() > extent[3]).any():
        #     c += 'y+'
        # plot_text_ax(ax2, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        # plot_text_ax(ax2, 'K01', 0.75, 0.02, 20, 'bottom', 'right', 'k')
        # ax2.plot(L.x['K01S2'], L.y['K01S2'], 'k-', label='K01')
        # # AXIS 3
        # extent = [-2.5, 0, -1.5, 1.5]
        # xm, ym = ma_mask_xyz(OIHa__gz, O3Hb__gz)
        # # sc = ax3.scatter(xm, ym, c=np.ma.log10(W6563__gz), vmin=np.log10(DIG_WHa_threshold), vmax=np.log10(HII_WHa_threshold), cmap='rainbow_r', s=1, marker='o', edgecolor='none')
        # sc = ax3.scatter(xm, ym, c=classif, cmap=cmap, s=1, marker='o', edgecolor='none')
        # ax3.set_xlim(extent[0:2])
        # ax3.set_ylim(extent[2:4])
        # N = xm.count()
        # c = ''
        # if (xm.compressed() < extent[0]).any():
        #     c += 'x-'
        # if (xm.compressed() > extent[1]).any():
        #     c += 'x+'
        # if (ym.compressed() < extent[2]).any():
        #     c += 'y-'
        # if (ym.compressed() > extent[3]).any():
        #     c += 'y+'
        # plot_text_ax(ax3, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        # plot_text_ax(ax3, 'K01', 0.80, 0.02, 20, 'bottom', 'right', 'k')
        # ax3.plot(L.x['K01OI'], L.y['K01OI'], 'k-', label='K01')
        ax1.set_ylabel(r'$\log\ [OIII]/H\beta$')
        ax1.set_xlabel(r'$\log\ [NII]/H\alpha$')
        # ax2.set_xlabel(r'$\log\ [SII]/H\alpha$')
        # ax3.set_xlabel(r'$\log\ [OI]/H\alpha$')
        # cbaxes = add_subplot_axes(ax1, [0.75, 0.95, 0.5, 0.05])
        cbaxes = add_subplot_axes(ax1, [0.51, 0.95, 0.5, 0.05])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        cb.ax.xaxis.labelpad = -29
        cb.ax.set_xticklabels(classif_labels, fontsize=12)
        # cb.ax.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]', fontsize=11)
        # plt.setp(ax2.get_yticklabels(), visible=False)
        # plt.setp(ax3.get_yticklabels(), visible=False)
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        # ax2.xaxis.set_minor_locator(minorLocator)
        # ax2.yaxis.set_minor_locator(minorLocator)
        # ax3.xaxis.set_minor_locator(minorLocator)
        # ax3.yaxis.set_minor_locator(minorLocator)

        f.tight_layout()
        f.savefig('fig5.png')
        plt.close(f)


def fig5_2panels(ALL, gals=None):
    sel_WHa_DIG__gz = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__gz = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__gz = (ALL.W6563__z >= HII_WHa_threshold).filled(False)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')

    N_gals = 0
    for g in gals:
        where_gals__gz = np.where(ALL.califaID__z == g)
        if not where_gals__gz:
            continue
        sel_gals__gz[where_gals__gz] = True
        N_gals += 1

    if (sel_gals__gz).any():
        W6563__gz = ALL.W6563__z[sel_gals__gz]

        O3Hb__gz = np.ma.log10(ALL.f5007__z/ALL.f4861__z)
        N2Ha__gz = np.ma.log10(ALL.f6583__z/ALL.f6563__z)
        S2Ha__gz = np.ma.log10((ALL.f6717__z+ALL.f6731__z)/ALL.f6563__z)
        OIHa__gz = np.ma.log10(ALL.f6300__z/ALL.f6563__z)
        L = Lines()
        L.addLine('K01S2', L.linebpt, (1.30, 0.72, -0.32), np.linspace(-10.0, 0.32, L.xn + 1)[:-1])
        L.addLine('K01OI', L.linebpt, (1.33, 0.73, 0.59), np.linspace(-10.0, -0.59, L.xn + 1)[:-1])

        classif = np.ma.masked_all(W6563__gz.shape)
        classif[sel_WHa_DIG__gz] = 1
        classif[sel_WHa_COMP__gz] = 2
        classif[sel_WHa_HII__gz] = 3

        N_cols = 1
        N_rows = 2
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 4))
        ax2, ax1 = axArr
        cmap = cmap_discrete(colors_DIG_COMP_HII)
        # AXIS 1
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(N2Ha__gz, O3Hb__gz)
        sc = ax1.scatter(xm, ym, c=classif, cmap=cmap, s=1, marker='o', edgecolor='none')
        ax1.set_xlim(extent[0:2])
        ax1.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        ax1.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax1.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        plot_text_ax(ax1, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax1, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K03', 0.55, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'a)', 0.02, 0.02, 16, 'bottom', 'left', 'k')
        cbaxes = add_subplot_axes(ax1, [0.54, 0.99, 0.46, 0.06])
        # cbaxes = add_subplot_axes(ax1, [0.51, 1.06, 0.5, 0.06])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        cb.ax.set_xticklabels(classif_labels, fontsize=12)
        # AXIS 2
        xm, ym = ma_mask_xyz(N2Ha__gz, O3Hb__gz)
        # sc = ax2.scatter(xm, ym, c=np.ma.log10(W6563__gz), vmin=np.log10(DIG_WHa_threshold), vmax=np.log10(HII_WHa_threshold), cmap='rainbow_r', s=1, marker='o', edgecolor='none')
        sc = ax2.scatter(xm, ym, c=np.ma.log10(W6563__gz), vmin=logWHa_range[0], vmax=logWHa_range[1], cmap='rainbow_r', s=1, marker='o', edgecolor='none')
        ax2.set_xlim(extent[0:2])
        ax2.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        ax2.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax2.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        plot_text_ax(ax2, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax2, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax2, 'K03', 0.55, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax2, 'b)', 0.02, 0.02, 16, 'bottom', 'left', 'k')
        # cbaxes = add_subplot_axes(ax2, [0.51, 0.99, 0.49, 0.06])
        cbaxes = add_subplot_axes(ax2, [0.51, 1.06, 0.47, 0.06])
        # cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.0, 1.1, 1.2, 1.3], orientation='horizontal')
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[0, 1.25, 2.5], orientation='horizontal')
        cb.ax.xaxis.labelpad = -28
        cb.ax.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]', fontsize=11)

        ax2.set_ylabel(r'$\log\ [OIII]/H\beta$')
        # ax.set_xlabel(r'$\log\ [NII]/H\alpha$')
        ax1.set_ylabel(r'$\log\ [OIII]/H\beta$')
        ax1.set_xlabel(r'$\log\ [NII]/H\alpha$')
        # ax1.set_ylabel(r'$\log\ [OIII]/H\beta$')
        # # ax.set_xlabel(r'$\log\ [NII]/H\alpha$')
        # ax2.set_ylabel(r'$\log\ [OIII]/H\beta$')
        # ax2.set_xlabel(r'$\log\ [NII]/H\alpha$')
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        ax2.xaxis.set_minor_locator(minorLocator)
        ax2.yaxis.set_minor_locator(minorLocator)
        f.tight_layout()
        f.savefig('fig5_2panels.png')
        plt.close(f)


def fig6(ALL, gals=None):
    # WHa DIG-COMP-HII decomposition
    sel_WHa_DIG__yx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__yx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__yx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)
    sel_WHa_DIG__z = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__z = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__z = (ALL.W6563__z >= HII_WHa_threshold).filled(False)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        distance_HLR__z = ALL.get_gal_prop(califaID, ALL.zoneDistance_HLR)
        f__lz = {'%s' % L: ALL.get_gal_prop(califaID, 'f%s__z' % L) for L in lines}

        f = plt.figure(dpi=200, figsize=(6, 5))
        cmap = cmap_discrete(colors=colors_DIG_COMP_HII)
        f.suptitle(r'%s - %s (%s): %d pixels (%d zones)' % (califaID, mto, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))

        ax1 = f.gca()
        x = distance_HLR__z
        y = np.ma.log10(f__lz['6563']/f__lz['4861'])
        x_range = distance_range
        y_range = lineratios_range['6563/4861']
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_HII__z)
        sc = ax1.scatter(x, y, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        cbaxes = add_subplot_axes(ax1, [0.59, 0.91, 0.35, 0.04])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        cb.ax.set_xticklabels(classif_labels, fontsize=9)
        ax1.set_ylabel(r'$\log\ H\alpha/H\beta$')
        ax1.set_xlabel('R [HLR]')
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)

        sel_DIG__z, sel_COMP__z, sel_HII__z = get_selections_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_HII__z)

        min_npts = 4
        xm, ym = ma_mask_xyz(x, y, mask=~sel_DIG__z)
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel = npts_DIG > min_npts
            ax1.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[0])
            ax1.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_COMP__z)
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel = npts_COMP > min_npts
            ax1.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[1])
            ax1.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_HII__z)
        yMean_HII, yPrc_HII, bin_center_HII, npts_HII = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_HII.any():
            sel = npts_HII > min_npts
            ax1.plot(bin_center_HII[sel], yPrc_HII[2][sel], 'k--', lw=2, c=colors_lines_DIG_COMP_HII[2])
            ax1.plot(bin_center_HII[sel], yPrc_HII[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[2], markersize=10)

        ax2 = ax1.twinx()
        mn, mx = ax1.get_ylim()
        ax2.set_ylim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))
        ax2.set_ylabel(r'$\tau_V$')

        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('fig6-%s.png' % califaID)
        plt.close(f)


def fig6_3gals(ALL, gals=None):
    # WHa DIG-COMP-HII decomposition
    sel_WHa_DIG__z = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__z = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__z = (ALL.W6563__z >= HII_WHa_threshold).filled(False)

    N_cols = 1
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 6, N_rows * 4))
    cmap = cmap_discrete(colors=colors_DIG_COMP_HII)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    row = 0
    for califaID in gals:
        ax1 = axArr[row]

        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        distance_HLR__z = ALL.get_gal_prop(califaID, ALL.zoneDistance_HLR)
        f__lz = {'%s' % L: ALL.get_gal_prop(califaID, 'f%s__z' % L) for L in lines}

        x = distance_HLR__z
        y = (1./(q[0] - q[1])) * np.ma.log(f__lz['6563']/f__lz['4861']/2.86)
        x_range = distance_range
        y_range = lineratios_range['6563/4861']
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_HII__z)
        sc = ax1.scatter(x, y, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')

        sel_DIG__z, sel_COMP__z, sel_HII__z = get_selections_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_HII__z)

        min_npts = 4
        xm, ym = ma_mask_xyz(x, y, mask=~sel_DIG__z)
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel = npts_DIG > min_npts
            ax1.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[0])
            ax1.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_COMP__z)
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel = npts_COMP > min_npts
            ax1.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[1])
            ax1.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_HII__z)
        yMean_HII, yPrc_HII, bin_center_HII, npts_HII = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_HII.any():
            sel = npts_HII > min_npts
            ax1.plot(bin_center_HII[sel], yPrc_HII[2][sel], 'k--', lw=2, c=colors_lines_DIG_COMP_HII[2])
            ax1.plot(bin_center_HII[sel], yPrc_HII[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[2], markersize=10)

        ax2 = ax1.twinx()
        mn, mx = ax1.get_ylim()
        unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
        ax2.set_ylim(unit_converter(mn), unit_converter(mx))
        # ax2.set_ylim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))

        if row < 2:
            plt.setp(ax1.get_xticklabels(), visible=False)
        if row == 2:
            ax1.set_xlabel('R [HLR]')
            cbaxes = add_subplot_axes(ax1, [0.55, 0.99, 0.4, 0.07])
            cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
            cb.ax.set_xticklabels(classif_labels, fontsize=9)
        if row == 1:
            ax1.set_ylabel(r'$\tau_V$')
            ax2.set_ylabel(r'$\log\ H\alpha/H\beta$')
        row += 1

    f.tight_layout(h_pad=0)
    f.savefig('fig6.png')
    plt.close(f)


def fig7(ALL, gals=None):
    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')

    # WHa DIG-COMP-HII decomposition
    sel_WHa_DIG__yx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__yx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__yx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)

    N_gals = 0
    for g in gals:
        where_gals__gz = np.where(ALL.califaID__z == g)
        if not where_gals__gz:
            continue
        sel_gals__gz[where_gals__gz] = True
        N_gals += 1

    if (sel_gals__gz).any():
        tau_V_neb_DIG = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_DIG_npts = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_HII = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_HII_npts = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_GAL = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_sumGAL = np.ma.masked_all((N_gals, N_R_bins))
        delta_tau = np.ma.masked_all((N_gals, N_R_bins))
        delta_tau_norm_GAL = np.ma.masked_all((N_gals, N_R_bins))
        delta_tau_norm_sumGAL = np.ma.masked_all((N_gals, N_R_bins))

        for i_g, califaID in enumerate(gals):
            if califaID not in ALL.califaID__z:
                print 'Fig_1: %s not in sample pickle' % califaID
                continue
            HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
            pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
            ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
            x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
            y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
            N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
            N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
            f__yx = {'%s' % L: ALL.get_gal_prop(califaID, 'f%s__yx' % L).reshape(N_y, N_x) for L in lines}
            sel_DIG__yx, sel_COMP__yx, sel_HII__yx = get_selections(ALL, califaID, sel_WHa_DIG__yx, sel_WHa_COMP__yx, sel_WHa_HII__yx)
            mean_f4861_DIG__r = radialProfile(f__yx['4861'], R_bin__r, x0, y0, pa, ba, HLR_pix, sel_DIG__yx, 'mean')
            mean_f6563_DIG__r = radialProfile(f__yx['6563'], R_bin__r, x0, y0, pa, ba, HLR_pix, sel_DIG__yx, 'mean')
            mean_f4861_HII__r = radialProfile(f__yx['4861'], R_bin__r, x0, y0, pa, ba, HLR_pix, sel_HII__yx, 'mean')
            mean_f6563_HII__r = radialProfile(f__yx['6563'], R_bin__r, x0, y0, pa, ba, HLR_pix, sel_HII__yx, 'mean')
            tau_V_neb_DIG[i_g] = np.ma.log(mean_f6563_DIG__r / mean_f4861_DIG__r / 2.86) / (q[0] - q[1])
            tau_V_neb_DIG_npts[i_g] = (~np.bitwise_or(~sel_DIG__yx, np.bitwise_or(f__yx['4861'].mask, f__yx['6563'].mask))).astype('int').sum()
            tau_V_neb_HII[i_g] = np.ma.log(mean_f6563_HII__r / mean_f4861_HII__r / 2.86) / (q[0] - q[1])
            tau_V_neb_HII_npts[i_g] = (~np.bitwise_or(~sel_HII__yx, np.bitwise_or(f__yx['4861'].mask, f__yx['6563'].mask))).astype('int').sum()
            tau_V_neb_GAL[i_g] = ALL.get_gal_prop_unique(califaID, ALL.integrated_tau_V_neb)
            xm, ym = ma_mask_xyz(f__yx['6563'], f__yx['4861'])
            tau_V_neb_sumGAL[i_g] = np.ma.log(xm.sum() / ym.sum() / 2.86) / (q[0] - q[1])
            delta_tau[i_g] = tau_V_neb_HII[i_g] - tau_V_neb_DIG[i_g]
            delta_tau_norm_GAL[i_g] = (tau_V_neb_HII[i_g] - tau_V_neb_DIG[i_g])/tau_V_neb_GAL[i_g]
            delta_tau_norm_sumGAL[i_g] = (tau_V_neb_HII[i_g] - tau_V_neb_DIG[i_g])/tau_V_neb_sumGAL[i_g]

        W6563__gz = ALL.W6563__z[sel_gals__gz]
        SB4861__gz = ALL.SB4861__z[sel_gals__gz]
        SB6563__gz = ALL.SB6563__z[sel_gals__gz]

        # WHa DIG-COMP-HII decomposition
        sel_WHa_DIG__gz = (W6563__gz < DIG_WHa_threshold).filled(False)
        sel_WHa_COMP__gz = np.bitwise_and((W6563__gz >= DIG_WHa_threshold).filled(False), (W6563__gz < HII_WHa_threshold).filled(False))
        sel_WHa_HII__gz = (W6563__gz >= HII_WHa_threshold).filled(False)

        N_cols = 1
        N_rows = 3
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        ax1, ax2, ax3 = axArr

        # AXIS 1
        x = (1./(q[0] - q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
        range = [-2, 2]
        xDs = [x[sel_WHa_DIG__gz].compressed(),  x[sel_WHa_COMP__gz].compressed(),  x[sel_WHa_HII__gz].compressed()]
        # ax1.set_title('zones')
        plot_histo_ax(ax1, x.compressed(), histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        plot_histo_ax(ax1, xDs, y_v_space=0.06, y_h_space=0.11, first=False, c=colors_lines_DIG_COMP_HII, kwargs_histo=dict(histtype='step', color=colors_DIG_COMP_HII, normed=False, range=range))
        ax1.set_xlabel(r'$\tau_V$')
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1_top = ax1.twiny()
        mn, mx = ax1.get_xlim()
        unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
        ax1_top.set_xlim(unit_converter(mn), unit_converter(mx))
        #ax1_top.set_xlim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))
        ax1_top.set_xlabel(r'$\log\ H\alpha/H\beta$')
        plot_text_ax(ax1, 'a)', 0.02, 0.98, 16, 'top', 'left', 'k')

        # AXIS 2
        x = delta_tau
        range = DtauVnorm_range
        plot_histo_ax(ax2, x.compressed(), y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
        ax2.set_xlabel(r'$\Delta \tau\ =\ \tau_V^{HII}\ -\ \tau_V^{DIG}}$')
        ax2.xaxis.set_minor_locator(minorLocator)
        plot_text_ax(ax2, 'b)', 0.02, 0.98, 16, 'top', 'left', 'k')

        # AXIS 3
        x = delta_tau_norm_GAL
        range = DtauVnorm_range
        plot_histo_ax(ax3, x.compressed(), y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
        ax3.set_xlabel(r'$\Delta \tau\ =\ (\tau_V^{HII}\ -\ \tau_V^{DIG}) / \tau_V^{GAL}$')
        ax3.xaxis.set_minor_locator(minorLocator)
        plot_text_ax(ax3, 'c)', 0.02, 0.98, 16, 'top', 'left', 'k')

        f.tight_layout(h_pad=0.05)
        f.savefig('fig7.png')


def fig8(ALL, gals=None):
    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')

    N_gals = 0
    for g in gals:
        where_gals__gz = np.where(ALL.califaID__z == g)
        if not where_gals__gz:
            continue
        sel_gals__gz[where_gals__gz] = True
        N_gals += 1

    if (sel_gals__gz).any():
        W6563__gz = ALL.W6563__z[sel_gals__gz]

        SB4861__gz = ALL.SB4861__z[sel_gals__gz]
        SB6563__gz = ALL.SB6563__z[sel_gals__gz]

        tau_V_neb__gz = ALL.tau_V_neb__z[sel_gals__gz]
        tau_V__gz = ALL.tau_V__z[sel_gals__gz]

        # WHa DIG-COMP-HII decomposition
        sel_WHa_DIG__gz = (W6563__gz < DIG_WHa_threshold).filled(False)
        sel_WHa_COMP__gz = np.bitwise_and((W6563__gz >= DIG_WHa_threshold).filled(False), (W6563__gz < HII_WHa_threshold).filled(False))
        sel_WHa_HII__gz = (W6563__gz >= HII_WHa_threshold).filled(False)

        f = plt.figure(dpi=200, figsize=(6, 5))
        ax1 = f.gca()
        x = tau_V_neb__gz - tau_V__gz
        xDs = [x[sel_WHa_DIG__gz].compressed(),  x[sel_WHa_COMP__gz].compressed(),  x[sel_WHa_HII__gz].compressed()]
        range = DtauV_range
        # ax1.set_title('zones')
        plot_histo_ax(ax1, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        plot_histo_ax(ax1, xDs, y_v_space=0.06, y_h_space=0.11, first=False, c=colors_lines_DIG_COMP_HII, kwargs_histo=dict(histtype='step', color=colors_DIG_COMP_HII, normed=False, range=range))
        ax1.set_xlabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        ax1.xaxis.set_minor_locator(minorLocator)

        f.tight_layout()
        f.savefig('fig8.png')


def fig9(ALL, gals=None):
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

        x_Y__gz = ALL.x_Y__z[sel_gals__gz]
        tau_V_neb__gz = ALL.tau_V_neb__z[sel_gals__gz]
        tau_V__gz = ALL.tau_V__z[sel_gals__gz]

        # WHa DIG-COMP-HII decomposition
        sel_WHa_DIG__gz = (W6563__gz < DIG_WHa_threshold).filled(False)
        sel_WHa_COMP__gz = np.bitwise_and((W6563__gz >= DIG_WHa_threshold).filled(False), (W6563__gz < HII_WHa_threshold).filled(False))
        sel_WHa_HII__gz = (W6563__gz >= HII_WHa_threshold).filled(False)

        N_cols = 2
        N_rows = 1
        f = plt.figure(dpi=200, figsize=(8,6))

        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.85
        left_h = left + width + 0.02
        rect_scatter = [left, bottom, width, height]
        rect_histy = [left_h, bottom, 0.2, height]
        ax1 = f.add_axes(rect_scatter)
        ax2 = f.add_axes(rect_histy)
        cmap = cmap_discrete(colors_DIG_COMP_HII)
        # AXIS 1
        x = x_Y__gz
        y = tau_V_neb__gz - tau_V__gz
        classif = np.ma.masked_all(W6563__gz.shape)
        classif[sel_WHa_DIG__gz] = 1
        classif[sel_WHa_COMP__gz] = 2
        classif[sel_WHa_HII__gz] = 3
        sc = ax1.scatter(x, y, c=classif, cmap=cmap, s=1, **dflt_kw_scatter)
        cbaxes = add_subplot_axes(ax1, [0.59, 0.95, 0.4, 0.04])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        cb.ax.set_xticklabels(classif_labels, fontsize=9)
        ax1.set_xlim(x_Y_range)
        ax1.set_ylim(DtauV_range)
        ax1.set_xlabel(r'x${}_Y$ [frac.]')
        ax1.set_ylabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        ax1.grid()
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_DIG__gz)
        xbin = np.linspace(0, 0.6, 30)
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), xbin)
        mask = (npts_DIG > 30)
        ax1.plot(bin_center_DIG[mask], yPrc_DIG[2][mask], '--', lw=2, c=colors_lines_DIG_COMP_HII[0])
        ax1.plot(bin_center_DIG[mask], yPrc_DIG[2][mask], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_COMP__gz)
        xbin = np.linspace(0, 0.6, 30)
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), xbin)
        mask = (npts_COMP > 30)
        ax1.plot(bin_center_COMP[mask], yPrc_COMP[2][mask], '--', lw=2, c=colors_lines_DIG_COMP_HII[1])
        ax1.plot(bin_center_COMP[mask], yPrc_COMP[2][mask], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_HII__gz)
        xbin = np.linspace(0, 0.6, 30)
        yMean_HII, yPrc_HII, bin_center_HII, npts_HII = stats_med12sigma(xm.compressed(), ym.compressed(), xbin)
        mask = (npts_HII > 30)
        ax1.plot(bin_center_HII[mask], yPrc_HII[2][mask], '--', lw=2, c=colors_lines_DIG_COMP_HII[2])
        ax1.plot(bin_center_HII[mask], yPrc_HII[2][mask], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[2], markersize=10)
        # AXIS 2
        yDs = [y[sel_WHa_DIG__gz].compressed(), y[sel_WHa_COMP__gz].compressed(), y[sel_WHa_HII__gz].compressed()]
        ax2.hist(yDs, bins=30, range=DtauV_range, orientation='horizontal', color=colors_DIG_COMP_HII, histtype='step')
        from matplotlib.ticker import NullFormatter
        nullfmt = NullFormatter()  # no labels
        ax2.yaxis.set_major_formatter(nullfmt)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=270)
        f.subplots_adjust(wspace=0.05)
        f.savefig('fig9.png')


if __name__ == '__main__':
    main(sys.argv)
