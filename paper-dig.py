import sys
import numpy as np
import matplotlib as mpl
from pytu.lines import Lines
from pytu.objects import runstats
from matplotlib import pyplot as plt
from pycasso.util import radialProfile
from scipy.interpolate import interp1d
from pystarlight.util.constants import L_sun
from matplotlib.ticker import AutoMinorLocator
from pytu.functions import ma_mask_xyz, debug_var
from pystarlight.util.redenninglaws import calc_redlaw
from CALIFAUtils.scripts import get_NEDName_by_CALIFAID
from CALIFAUtils.objects import stack_gals, CALIFAPaths
from mpl_toolkits.axes_grid1 import make_axes_locatable
from CALIFAUtils.plots import DrawHLRCircleInSDSSImage, DrawHLRCircle
from pytu.plots import cmap_discrete, plot_text_ax, density_contour, stats_med12sigma, add_subplot_axes


# mpl.rcParams['font.size'] = 20
# mpl.rcParams['axes.labelsize'] = 16
# mpl.rcParams['axes.titlesize'] = 16
# mpl.rcParams['xtick.labelsize'] = 14
# mpl.rcParams['ytick.labelsize'] = 14
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = 'Times New Roman'
colors_DIG_COMP_HII = ['tomato', 'lightgreen', 'royalblue']
colors_lines_DIG_COMP_HII = ['darkred', 'olive', 'mediumblue']
classif_labels = ['DIG', 'COMP', 'HII']
cmap_R = plt.cm.copper_r
minorLocator = AutoMinorLocator(5)

debug = False
# CCM reddening law
q = calc_redlaw([4861, 6563], R_V=3.1, redlaw='CCM')
qHbHa = q[0] - q[1]
# config variables
lineratios_range = {
    '6563/4861': [0, 1],
    '6583/6563': [-0.7, 0.1],
    '6300/6563': [-2, 0],
    '6717+6731/6563': [-0.8, 0.2],
}
distance_range = [0, 3]
tauVneb_range = [0, 5]
tauVneb_neg_range = [-1, 5]
logSBHa_range = [4, 7]
logWHa_range = [0, 2.5]
DtauV_range = [-2, 3]
x_Y_range = [0, 0.6]
OH_range = [8, 9.5]
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
kw_cube = dict(EL=EL, config=config, elliptical=elliptical, debug=True)
dflt_kw_scatter = dict(marker='o', edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, frac=0.07, debug=True)  # , tendency=True)
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal')


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

    """
    Fig 1.
        painel a: imagem sdss
        painel a: mapa colorido pelas 4 zonas BPT: S06, K03, H01
        painel b: BPT colorido pelas zonas WHa
        painel c: BPT colorido pelas zonas sigma_Ha
        painel a: mapa colorido pelas zonas WHa
        painel b: log WHa vs R colorido  pelas zonas WHa
        painel c: log sigma_Ha colorido pelas zonas WHa
        painel a: mapa colorido pelas zonas sigma_Ha
        painel b: log WHa vs R colorido pelas zonas sigma_Ha
        painel c: log sigma_Ha colorido pelas zonas sigma_Ha
    """
    # fig1(ALL, gals)

    """
    Fig 2.
        painel a: O/H (o3n2) vs R[HLR} pixel a pixel colorido por zonas WHa
        painel b: O/H (o3n2) vs R[HLR} integrated (so regioes HII e total)
        painel c: O/H (n2)   vs R[HLR} pixel a pixel colorido por zonas WHa
        painel d: O/H (n2)   vs R[HLR} integrated (so regioes HII e total)
        painel e: O/H (o23)  vs R[HLR} pixel a pixel colorido por zonas WHa
        painel f: O/H (o23)  vs R[HLR} integrated (so regioes HII e total)
        painel g: O/H (n2o2) vs R[HLR} pixel a pixel colorido por zonas WHa
        painel h: O/H (n2o2) vs R[HLR} integrated (so regioes HII e total)
    """
    # fig2(ALL, gals)

    """ TODO """
    # fig3(ALL, gals)

    """
    Fig 4.
        O/H (O3N2)
        Uma figura resumindo todas as galaxias de califa com emission lines para
        analisar o efeito do dig e o efeito da abertura no sdss
        3 bins de CI (concentration index) e 3 bins de b/a (9 paineis)
        Em cada painel, plotar, para cada galaxia, o valor de log O/H(O3N2)
        integrated em funcao de R[pixel]/R_50[pixel].
        Se poderia ate colorir cada curva em funcao, por exemplo, da massa total
        da galaxia, ou entao em funcao da posicao do oiii/hb vs nii/ha integrado
        no bpt.
    """
    fig4_profile(ALL, gals)
    # fig4_cumulative_profile(ALL, gals)
    """
    Fig 5.
        Igual a Fig 4 mas usando O/H (N2)
    """
    fig5_profile_new(ALL, gals)
    # fig5_profile(ALL, gals)
    # fig5_cumulative_profile(ALL, gals)
    """
    Fig 6.
        Igual a Fig 4 mas usando O/H (O23)
    """
    fig6_profile_new(ALL, gals)
    # fig6_profile(ALL, gals)
    # fig6_cumulative_profile(ALL, gals)
    """
    Fig 7.
        Igual a Fig 4 mas usando O/H (N2O2)
    """
    fig7_profile_new(ALL, gals)
    # fig7_profile(ALL, gals)
    # fig7_cumulative_profile(ALL, gals)
    # figs4567(ALL, gals)
    # fig4567_new(ALL, gals)


def cumulative_profiles(ALL, califaID, selDIG, selCOMP, selHII, negative_tauV=False):
    HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
    pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
    ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
    N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
    N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
    x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
    y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
    SB__lyx = {'%s' % L: ALL.get_gal_prop(califaID, 'SB%s__yx' % L).reshape(N_y, N_x) for L in lines}
    sel_DIG__yx, sel_COMP__yx, sel_HII__yx = get_selections(ALL, califaID, selDIG, selCOMP, selHII)

    SB_sum__lr = {}
    SB_cumsum__lr = {}
    SB_npts__lr = {}
    SB_DIG__lyx = {}
    SB_sum_DIG__lr = {}
    SB_cumsum_DIG__lr = {}
    SB_npts_DIG__lr = {}
    SB_COMP__lyx = {}
    SB_sum_COMP__lr = {}
    SB_cumsum_COMP__lr = {}
    SB_npts_COMP__lr = {}
    SB_HII__lyx = {}
    SB_sum_HII__lr = {}
    SB_cumsum_HII__lr = {}
    SB_npts_HII__lr = {}
    for k, v in SB__lyx.iteritems():
        SB_sum__lr[k], SB_npts__lr[k] = radialProfile(v, R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum', True)
        SB_cumsum__lr[k] = SB_sum__lr[k].filled(0.).cumsum()
        SB_HII__lyx[k] = np.ma.masked_array(v, mask=~sel_HII__yx, copy=True)
        SB_sum_HII__lr[k], SB_npts_HII__lr[k] = radialProfile(SB_HII__lyx[k], R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum', True)
        SB_cumsum_HII__lr[k] = SB_sum_HII__lr[k].filled(0.).cumsum()
        SB_COMP__lyx[k] = np.ma.masked_array(v, mask=~sel_COMP__yx, copy=True)
        SB_sum_COMP__lr[k], SB_npts_COMP__lr[k] = radialProfile(SB_COMP__lyx[k], R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum', True)
        SB_cumsum_COMP__lr[k] = SB_sum_COMP__lr[k].filled(0.).cumsum()
        SB_DIG__lyx[k] = np.ma.masked_array(v, mask=~sel_DIG__yx, copy=True)
        SB_sum_DIG__lr[k], SB_npts_DIG__lr[k] = radialProfile(SB_DIG__lyx[k], R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum', True)
        SB_cumsum_DIG__lr[k] = SB_sum_DIG__lr[k].filled(0.).cumsum()

    HaHb__yx = SB__lyx['6563']/SB__lyx['4861']
    tau_V_neb__yx = np.log(HaHb__yx / 2.86) / qHbHa
    HaHb_cumsum__r = SB_sum__lr['6563'].filled(0.).cumsum()/SB_sum__lr['4861'].filled(0.).cumsum()
    tau_V_neb_cumsum__r = np.log(HaHb_cumsum__r / 2.86) / qHbHa

    HaHb_DIG__yx = SB_DIG__lyx['6563']/SB_DIG__lyx['4861']
    HaHb_cumsum_DIG__r = SB_sum_DIG__lr['6563'].filled(0.).cumsum()/SB_sum_DIG__lr['4861'].filled(0.).cumsum()
    tau_V_neb_cumsum_DIG__r = np.log(HaHb_cumsum_DIG__r / 2.86) / qHbHa
    HaHb_COMP__yx = SB_COMP__lyx['6563']/SB_COMP__lyx['4861']
    HaHb_cumsum_COMP__r = SB_sum_COMP__lr['6563'].filled(0.).cumsum()/SB_sum_COMP__lr['4861'].filled(0.).cumsum()
    tau_V_neb_cumsum_COMP__r = np.log(HaHb_cumsum_COMP__r / 2.86) / qHbHa
    HaHb_HII__yx = SB_HII__lyx['6563']/SB_HII__lyx['4861']
    HaHb_cumsum_HII__r = SB_sum_HII__lr['6563'].filled(0.).cumsum()/SB_sum_HII__lr['4861'].filled(0.).cumsum()
    tau_V_neb_cumsum_HII__r = np.log(HaHb_cumsum_HII__r / 2.86) / qHbHa

    HaHb_classif = dict(
        total=dict(HaHb__yx=HaHb__yx, tau_V_neb__yx=tau_V_neb__yx, HaHb_cumsum__r=HaHb_cumsum__r, tau_V_neb_cumsum__r=tau_V_neb_cumsum__r),
        DIG=dict(HaHb__yx=HaHb_DIG__yx, HaHb_cumsum__r=HaHb_cumsum_DIG__r, tau_V_neb_cumsum__r=tau_V_neb_cumsum_DIG__r),
        COMP=dict(HaHb__yx=HaHb_COMP__yx, HaHb_cumsum__r=HaHb_cumsum_COMP__r, tau_V_neb_cumsum__r=tau_V_neb_cumsum_COMP__r),
        HII=dict(HaHb__yx=HaHb_HII__yx, HaHb_cumsum__r=HaHb_cumsum_HII__r, tau_V_neb_cumsum__r=tau_V_neb_cumsum_HII__r),
    )

    cumsum_classif = dict(
        total__lr=dict(v=SB_cumsum__lr, vsum=SB_sum__lr, npts=SB_npts__lr, img=SB__lyx),
        DIG__lr=dict(v=SB_cumsum_DIG__lr, vsum=SB_sum_DIG__lr, npts=SB_npts_DIG__lr, img=SB_DIG__lyx),
        COMP__lr=dict(v=SB_cumsum_COMP__lr, vsum=SB_sum_COMP__lr, npts=SB_npts_COMP__lr, img=SB_COMP__lyx),
        HII__lr=dict(v=SB_cumsum_HII__lr, vsum=SB_sum_HII__lr, npts=SB_npts_HII__lr, img=SB_HII__lyx),
    )

    return SB__lyx, HaHb_classif, cumsum_classif,


def create_segmented_map(ALL, califaID, selDIG, selCOMP, selHII):
    N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
    N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
    sel_DIG__yx, sel_COMP__yx, sel_HII__yx = get_selections(ALL, califaID, selDIG, selCOMP, selHII)
    map__yx = np.ma.masked_all((N_y, N_x))
    map__yx[sel_DIG__yx] = 1
    map__yx[sel_COMP__yx] = 2
    map__yx[sel_HII__yx] = 3
    return map__yx


def create_segmented_map_zones(ALL, califaID, selDIG, selCOMP, selHII):
    N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
    sel_DIG__z, sel_COMP__z, sel_HII__z = get_selections_zones(ALL, califaID, selDIG, selCOMP, selHII)
    map__z = np.ma.masked_all((N_zone))
    map__z[sel_DIG__z] = 1
    map__z[sel_COMP__z] = 2
    map__z[sel_HII__z] = 3
    return map__z


def get_selections(ALL, califaID, selDIG, selCOMP, selHII):
    N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
    N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
    sel_DIG__yx = ALL.get_gal_prop(califaID, selDIG).reshape(N_y, N_x)
    sel_COMP__yx = ALL.get_gal_prop(califaID, selCOMP).reshape(N_y, N_x)
    sel_HII__yx = ALL.get_gal_prop(califaID, selHII).reshape(N_y, N_x)
    return sel_DIG__yx, sel_COMP__yx, sel_HII__yx


def get_selections_zones(ALL, califaID, selDIG, selCOMP, selHII):
    sel_DIG__z = ALL.get_gal_prop(califaID, selDIG)
    sel_COMP__z = ALL.get_gal_prop(califaID, selCOMP)
    sel_HII__z = ALL.get_gal_prop(califaID, selHII)
    return sel_DIG__z, sel_COMP__z, sel_HII__z


def plot_OH(ax, distance_HLR__yx, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range):
    x = np.ma.ravel(distance_HLR__yx)
    y = np.ma.ravel(OH__yx)
    ax.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=3, **dflt_kw_scatter)
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
    ax.legend(loc='best', frameon=False, fontsize=12)
    ax.grid()
    ax.set_title('cumulative profile')
    ax.set_xlim(distance_range)
    ax.set_ylim(OH_range)
    return ax


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


def plotBPT(ax, N2Ha, O3Hb, z=None, cmap='viridis', mask=None, labels=True, N=False, cb_label=r'R [HLR]', vmax=None, vmin=None, dcontour=True):
    if mask is None:
        mask = np.zeros_like(O3Hb, dtype=np.bool_)
    extent = [-1.5, 1, -1.5, 1.5]
    if z is None:
        bins = [30, 30]
        xm, ym = ma_mask_xyz(N2Ha, O3Hb, mask=mask)
        if dcontour:
            density_contour(xm.compressed(), ym.compressed(), bins[0], bins[1], ax, range=[extent[0:2], extent[2:4]], colors=['b', 'y', 'r'])
        sc = ax.scatter(xm, ym, marker='o', c='0.5', s=10, edgecolor='none', alpha=0.4)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
    else:
        xm, ym, z = ma_mask_xyz(N2Ha, O3Hb, z, mask=mask)
        sc = ax.scatter(xm, ym, c=z, cmap=cmap, vmin=vmin, vmax=vmax, marker='o', s=10, edgecolor='none')
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
    return ax


def fig1(ALL, gals=None):
    """
    Fig 1.
        painel a: imagem sdss
        painel a: mapa colorido pelas 4 zonas BPT: S06, K03, H01
        painel b: BPT colorido pelas zonas WHa
        painel c: BPT colorido pelas zonas sigma_Ha
        painel a: mapa colorido pelas zonas WHa
        painel b: log WHa vs R colorido  pelas zonas WHa
        painel c: log sigma_Ha colorido pelas zonas WHa
        painel a: mapa colorido pelas zonas sigma_Ha
        painel b: log WHa vs R colorido pelas zonas sigma_Ha
        painel c: log sigma_Ha colorido pelas zonas sigma_Ha
    """
    P = CALIFAPaths()

    # WHa DIG-COMP-HII decomposition
    sel_WHa_DIG__yx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__yx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__yx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)
    sel_WHa_DIG__z = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__z = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__z = (ALL.W6563__z >= HII_WHa_threshold).filled(False)

    # Zhang DIG-COMP-HII decomposition
    sel_Zhang_DIG__yx = (ALL.SB6563__yx < DIG_Zhang_threshold).filled(False)
    sel_Zhang_COMP__yx = np.bitwise_and((ALL.SB6563__yx >= DIG_Zhang_threshold).filled(False), (ALL.SB6563__yx < HII_Zhang_threshold).filled(False))
    sel_Zhang_HII__yx = (ALL.SB6563__yx >= HII_Zhang_threshold).filled(False)
    sel_Zhang_DIG__z = (ALL.SB6563__z < DIG_Zhang_threshold).filled(False)
    sel_Zhang_COMP__z = np.bitwise_and((ALL.SB6563__z >= DIG_Zhang_threshold).filled(False), (ALL.SB6563__z < HII_Zhang_threshold).filled(False))
    sel_Zhang_HII__z = (ALL.SB6563__z >= HII_Zhang_threshold).filled(False)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        L = Lines()
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
        zoneDistance_HLR__z = ALL.zoneDistance_HLR
        SB__lyx = {'%s' % L: ALL.get_gal_prop(califaID, 'SB%s__yx' % L).reshape(N_y, N_x) for L in lines}
        SB__lz = {'%s' % L: ALL.get_gal_prop(califaID, 'SB%s__z' % L) for L in lines}
        f__lz = {'%s' % L: ALL.get_gal_prop(califaID, 'f%s__z' % L) for L in lines}
        W6563__yx = ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x)
        W6563__z = ALL.get_gal_prop(califaID, ALL.W6563__z)
        O3Hb__yx = SB__lyx['5007']/SB__lyx['4861']
        N2Ha__yx = SB__lyx['6583']/SB__lyx['6563']
        O3Hb__z = f__lz['5007']/f__lz['4861']
        N2Ha__z = f__lz['6583']/f__lz['6563']

        N_cols = 3
        N_rows = 4
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        cmap = cmap_discrete(colors_DIG_COMP_HII)
        ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = axArr
        f.suptitle(r'%s - %s: %d pixels (%d zones)' % (califaID, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))

        # AXIS 1, 2, 3
        # 1
        galimg = plt.imread(P.get_image_file(califaID))[::-1, :, :]
        # plt.setp(ax2.get_xticklabels(), visible=False)
        # plt.setp(ax2.get_yticklabels(), visible=False)
        ax1.imshow(galimg, origin='lower', aspect='equal')
        DrawHLRCircleInSDSSImage(ax1, HLR_pix, pa, ba, lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        txt = '%s' % califaID
        plot_text_ax(ax1, txt, 0.02, 0.98, 16, 'top', 'left', color='w')

        # 2
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_HII__z)
        extent = [-1.5, 1, -1.5, 1.5]
        x, y = np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z)
        xm, ym = ma_mask_xyz(x, y)
        sc = ax2.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=5, edgecolor='none')
        ax2.set_xlim(extent[0:2])
        ax2.set_ylim(extent[2:4])
        ax2.set_xlabel(r'$\log\ [NII]/H\alpha$')
        ax2.set_ylabel(r'$\log\ [OIII]/H\beta$')
        ax2.set_title(r'classif. W${}_{H\alpha}$')
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
        plot_text_ax(ax2, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax2, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax2, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
        ax2.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax2.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        ax2.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        plot_text_ax(ax2, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=35)  # 44.62)
        ax2.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
        L.fixCF10('S06')
        cbaxes = add_subplot_axes(ax2, [0.28, 1.06, 0.38, 0.05])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        cb.ax.xaxis.labelpad = -29
        cb.ax.set_xticklabels(classif_labels, fontsize=9)
        # 3
        map__z = create_segmented_map_zones(ALL, califaID, sel_Zhang_DIG__z, sel_Zhang_COMP__z, sel_Zhang_HII__z)
        extent = [-1.5, 1, -1.5, 1.5]
        x, y = np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z)
        xm, ym = ma_mask_xyz(x, y)
        sc = ax3.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=5, edgecolor='none')
        ax3.set_xlim(extent[0:2])
        ax3.set_ylim(extent[2:4])
        ax3.set_xlabel(r'$\log\ [NII]/H\alpha$')
        ax3.set_ylabel(r'$\log\ [OIII]/H\beta$')
        ax3.set_title(r'classif. $ \Sigma_{H\alpha}$')
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
        plot_text_ax(ax3, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax3, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax3, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
        ax3.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax3.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        ax3.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        plot_text_ax(ax3, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=35)  # 44.62)
        ax3.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
        L.fixCF10('S06')
        cbaxes = add_subplot_axes(ax3, [0.53, 1.06, 0.38, 0.05])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        cb.ax.xaxis.labelpad = -29
        cb.ax.set_xticklabels(classif_labels, fontsize=9)

        # AXIS 4, 5, 6
        x, y = np.ma.log10(N2Ha__yx), np.ma.log10(O3Hb__yx)
        sel_below_S06 = L.belowlinebpt('S06', x, y)
        sel_below_K03 = L.belowlinebpt('K03', x, y)
        sel_below_K01 = L.belowlinebpt('K01', x, y)
        sel_between_S06K03 = np.bitwise_and(sel_below_K03, ~sel_below_S06)
        sel_between_K03K01 = np.bitwise_and(~sel_below_K03, sel_below_K01)
        sel_above_K01 = ~sel_below_K01
        # creating map of BPT position
        mapBPT__yx = np.ma.masked_all((N_y, N_x))
        mapBPT__yx[sel_below_S06] = 4
        mapBPT__yx[sel_between_S06K03] = 3
        mapBPT__yx[sel_between_K03K01] = 2
        mapBPT__yx[sel_above_K01] = 1
        mapBPT__yx[sel_above_K01.mask] = np.ma.masked
        # 4
        im = ax4.imshow(mapBPT__yx, origin='lower', interpolation='nearest', aspect='equal', cmap=plt.cm.get_cmap('jet_r', 4))
        the_divider = make_axes_locatable(ax4)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        cb.set_ticks(3/8. * np.asarray([1, 3, 5, 7]) + 1.)
        cb.set_ticklabels(['> K01', 'K03-K01', 'S06-K03', '< S06'])
        DrawHLRCircle(ax4, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        ax4.set_title(r'Position in BPT')
        sel_DIG__z = ALL.get_gal_prop(califaID, sel_WHa_DIG__z)
        sel_COMP__z = ALL.get_gal_prop(califaID, sel_WHa_COMP__z)
        sel_HII__z = ALL.get_gal_prop(califaID, sel_WHa_HII__z)
        # 5
        plotBPT(ax5, np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z), z=np.ma.log10(W6563__z), vmin=logWHa_range[0], vmax=logWHa_range[1], cb_label=r'W${}_{H\alpha}\ [\AA]$', cmap='viridis_r')
        # 6
        plotBPT(ax6, np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z), z=np.ma.log10(SB__lz['6563']), vmin=logSBHa_range[0], vmax=logSBHa_range[1], cb_label=r'$\Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]', cmap='viridis_r')

        # AXIS 7, 8 e 9
        map__yx = create_segmented_map(ALL, califaID, sel_WHa_DIG__yx, sel_WHa_COMP__yx, sel_WHa_HII__yx)
        # 7
        im = ax7.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax7)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(classif_labels)
        ax7.set_title(r'classif. W${}_{H\alpha}$')
        DrawHLRCircle(ax7, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # 8
        x = np.ravel(pixelDistance_HLR__yx)
        y = np.ravel(np.ma.log10(W6563__yx))
        ax8.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax8.set_xlim(distance_range)
        ax8.set_ylim(logWHa_range)
        ax8.set_xlabel(r'R [HLR]')
        ax8.set_ylabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        ax8.grid()
        sel_DIG__yx, sel_COMP__yx, sel_HII__yx = get_selections(ALL, califaID, sel_WHa_DIG__yx, sel_WHa_COMP__yx, sel_WHa_HII__yx)
        min_npts = 10
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel = npts_DIG > min_npts
            ax8.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[0])
            ax8.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel = npts_COMP > min_npts
            ax8.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[1])
            ax8.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
        yMean_HII, yPrc_HII, bin_center_HII, npts_HII = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_HII.any():
            sel = npts_HII > min_npts
            ax8.plot(bin_center_HII[sel], yPrc_HII[2][sel], 'k--', lw=2, c=colors_lines_DIG_COMP_HII[2])
            ax8.plot(bin_center_HII[sel], yPrc_HII[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[2], markersize=10)
        # 9
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__lyx['6563']))
        ax9.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax9.set_xlim(distance_range)
        ax9.set_ylim(logSBHa_range)
        ax9.set_xlabel(r'R [HLR]')
        ax9.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax9.grid()
        sel_DIG__yx, sel_COMP__yx, sel_HII__yx = get_selections(ALL, califaID, sel_WHa_DIG__yx, sel_WHa_COMP__yx, sel_WHa_HII__yx)
        min_npts = 10
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel = npts_DIG > min_npts
            ax9.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[0])
            ax9.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel = npts_COMP > min_npts
            ax9.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[1])
            ax9.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
        yMean_HII, yPrc_HII, bin_center_HII, npts_HII = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_HII.any():
            sel = npts_HII > min_npts
            ax9.plot(bin_center_HII[sel], yPrc_HII[2][sel], 'k--', lw=2, c=colors_lines_DIG_COMP_HII[2])
            ax9.plot(bin_center_HII[sel], yPrc_HII[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[2], markersize=10)

        # AXIS 10, 11 e 12
        map__yx = create_segmented_map(ALL, califaID, sel_Zhang_DIG__yx, sel_Zhang_COMP__yx, sel_Zhang_HII__yx)
        # 10
        im = ax10.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax10)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(classif_labels)
        ax10.set_title(r'classif. $ \Sigma_{H\alpha}$')
        DrawHLRCircle(ax10, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # 11
        x = np.ravel(pixelDistance_HLR__yx)
        y = np.ravel(np.ma.log10(W6563__yx))
        ax11.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax11.set_xlim(distance_range)
        ax11.set_ylim(logWHa_range)
        ax11.set_xlabel(r'R [HLR]')
        ax11.set_ylabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        ax11.grid()
        sel_DIG__yx, sel_COMP__yx, sel_HII__yx = get_selections(ALL, califaID, sel_WHa_DIG__yx, sel_WHa_COMP__yx, sel_WHa_HII__yx)
        min_npts = 10
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel = npts_DIG > min_npts
            ax11.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[0])
            ax11.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel = npts_COMP > min_npts
            ax11.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[1])
            ax11.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
        yMean_HII, yPrc_HII, bin_center_HII, npts_HII = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_HII.any():
            sel = npts_HII > min_npts
            ax11.plot(bin_center_HII[sel], yPrc_HII[2][sel], 'k--', lw=2, c=colors_lines_DIG_COMP_HII[2])
            ax11.plot(bin_center_HII[sel], yPrc_HII[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[2], markersize=10)
        # 12
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__lyx['6563']))
        ax12.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax12.set_xlim(distance_range)
        ax12.set_ylim(logSBHa_range)
        ax12.set_xlabel(r'R [HLR]')
        ax12.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax12.grid()
        sel_DIG__yx, sel_COMP__yx, sel_HII__yx = get_selections(ALL, califaID, sel_WHa_DIG__yx, sel_WHa_COMP__yx, sel_WHa_HII__yx)
        min_npts = 10
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel = npts_DIG > min_npts
            ax12.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[0])
            ax12.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel = npts_COMP > min_npts
            ax12.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_HII[1])
            ax12.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
        yMean_HII, yPrc_HII, bin_center_HII, npts_HII = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_HII.any():
            sel = npts_HII > min_npts
            ax12.plot(bin_center_HII[sel], yPrc_HII[2][sel], 'k--', lw=2, c=colors_lines_DIG_COMP_HII[2])
            ax12.plot(bin_center_HII[sel], yPrc_HII[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[2], markersize=10)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('%s-fig1.png' % califaID)
        plt.close(f)


def fig2(ALL, gals=None):
    """
    Fig 2.
        painel a: O/H (o3n2) vs R[HLR} pixel a pixel colorido por zonas WHa
        painel b: O/H (o3n2) vs R[HLR} integrated (so regioes HII e total)
        painel c: O/H (n2)   vs R[HLR} pixel a pixel colorido por zonas WHa
        painel d: O/H (n2)   vs R[HLR} integrated (so regioes HII e total)
        painel e: O/H (o23)  vs R[HLR} pixel a pixel colorido por zonas WHa
        painel f: O/H (o23)  vs R[HLR} integrated (so regioes HII e total)
        painel g: O/H (n2o2) vs R[HLR} pixel a pixel colorido por zonas WHa
        painel h: O/H (n2o2) vs R[HLR} integrated (so regioes HII e total)
    """
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
        SB__lyx = {'%s' % L: ALL.get_gal_prop(califaID, 'SB%s__yx' % L).reshape(N_y, N_x) for L in lines}
        SB_sum__lr = {'%s' % L: radialProfile(SB__lyx[L], R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum') for L in lines}
        SB_cumsum__lr = {'%s' % L: SB_sum__lr[L].filled(0.).cumsum() for L in lines}

        sel_DIG__yx = ALL.get_gal_prop(califaID, sel_WHa_DIG__yx).reshape(N_y, N_x)
        sel_COMP__yx = ALL.get_gal_prop(califaID, sel_WHa_COMP__yx).reshape(N_y, N_x)
        sel_HII__yx = ALL.get_gal_prop(califaID, sel_WHa_HII__yx).reshape(N_y, N_x)
        map__yx = np.ma.masked_all((N_y, N_x))
        map__yx[sel_DIG__yx] = 1
        map__yx[sel_COMP__yx] = 2
        map__yx[sel_HII__yx] = 3

        SB_HII__lyx = {}
        SB_sum_HII__lr = {}
        SB_cumsum_HII__lr = {}
        SB_npts_HII__lr = {}
        SB_COMP__lyx = {}
        SB_sum_COMP__lr = {}
        SB_cumsum_COMP__lr = {}
        SB_npts_COMP__lr = {}
        SB_DIG__lyx = {}
        SB_sum_DIG__lr = {}
        SB_cumsum_DIG__lr = {}
        SB_npts_DIG__lr = {}
        for k, v in SB__lyx.iteritems():
            SB_HII__lyx[k] = np.ma.masked_array(v, mask=~sel_HII__yx, copy=True)
            SB_sum_HII__lr[k], SB_npts_HII__lr[k] = radialProfile(SB_HII__lyx[k], R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum', True)
            SB_cumsum_HII__lr[k] = SB_sum_HII__lr[k].filled(0.).cumsum()
            SB_COMP__lyx[k] = np.ma.masked_array(v, mask=~sel_COMP__yx, copy=True)
            SB_sum_COMP__lr[k], SB_npts_COMP__lr[k] = radialProfile(SB_COMP__lyx[k], R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum', True)
            SB_cumsum_COMP__lr[k] = SB_sum_COMP__lr[k].filled(0.).cumsum()
            SB_DIG__lyx[k] = np.ma.masked_array(v, mask=~sel_DIG__yx, copy=True)
            SB_sum_DIG__lr[k], SB_npts_DIG__lr[k] = radialProfile(SB_DIG__lyx[k], R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum', True)
            SB_cumsum_DIG__lr[k] = SB_sum_DIG__lr[k].filled(0.).cumsum()

        HaHb__yx = SB__lyx['6563']/SB__lyx['4861']
        tau_V_neb__yx = np.log(HaHb__yx / 2.86) / qHbHa
        HaHb_cumsum__r = SB_sum__lr['6563'].filled(0.).cumsum()/SB_sum__lr['4861'].filled(0.).cumsum()
        tau_V_neb_cumsum__r = np.log(HaHb_cumsum__r / 2.86) / qHbHa
        HaHb_sum_HII__r = SB_sum_HII__lr['6563'].filled(0.).cumsum()/SB_sum_HII__lr['4861'].filled(0.).cumsum()
        tau_V_neb_cumsum_HII__r = np.log(HaHb_sum_HII__r / 2.86) / qHbHa

        tau_V_neb__yx = np.where(np.less(tau_V_neb__yx.filled(-1), 0), 0, tau_V_neb__yx)
        tau_V_neb_cumsum__r = np.where(np.less(tau_V_neb_cumsum__r, 0), 0, tau_V_neb_cumsum__r)
        tau_V_neb_cumsum_HII__r = np.where(np.less(tau_V_neb_cumsum_HII__r, 0), 0, tau_V_neb_cumsum_HII__r)

        ####################################
        # O/H - Relative Oxygen abundances #
        ####################################
        #############
        # O3N2 PP04 #
        #############
        O3Hb__yx = SB__lyx['5007']/SB__lyx['4861']
        N2Ha__yx = SB__lyx['6583']/SB__lyx['6563']
        OH_O3N2__yx = 8.73 - 0.32 * np.ma.log10(O3Hb__yx/N2Ha__yx)
        O3Hb_cumsum__r = SB_cumsum__lr['5007']/SB_cumsum__lr['4861']
        N2Ha_cumsum__r = SB_cumsum__lr['6583']/SB_cumsum__lr['6563']
        OH_O3N2_cumsum__r = 8.73 - 0.32 * np.ma.log10(O3Hb_cumsum__r/N2Ha_cumsum__r)
        O3Hb_cumsum_HII__r = SB_cumsum_HII__lr['5007']/SB_cumsum_HII__lr['4861']
        N2Ha_cumsum_HII__r = SB_cumsum_HII__lr['6583']/SB_cumsum_HII__lr['6563']
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
        mask = np.zeros((N_y, N_x), dtype='bool')
        mask = np.bitwise_or(mask, SB__lyx['3727'].mask)
        mask = np.bitwise_or(mask, SB__lyx['4861'].mask)
        mask = np.bitwise_or(mask, SB__lyx['5007'].mask)
        mask = np.bitwise_or(mask, SB__lyx['6583'].mask)
        logO23__yx = logO23(fOII=SB__lyx['3727'], fHb=SB__lyx['4861'], f5007=SB__lyx['5007'], tau_V=tau_V_neb__yx)
        logN2O2__yx = logN2O2(fNII=SB__lyx['6583'], fOII=SB__lyx['3727'], tau_V=tau_V_neb__yx)
        mask = np.bitwise_or(mask, logO23__yx.mask)
        mask = np.bitwise_or(mask, logN2O2__yx.mask)
        mlogO23__yx, mlogN2O2__yx = ma_mask_xyz(logO23__yx, logN2O2__yx, mask=mask)
        OH_O23__yx = np.ma.masked_all((N_y, N_x))
        OH_O23__yx[~mask] = OH_O23(logO23_ratio=mlogO23__yx.compressed(), logN2O2_ratio=mlogN2O2__yx.compressed())
        OH_O23__yx[~np.isfinite(OH_O23__yx)] = np.ma.masked
        # HII cumulative O/H
        mask = np.zeros((N_R_bins), dtype='bool')
        logO23_cumsum_HII__r = logO23(fOII=SB_cumsum_HII__lr['3727'], fHb=SB_cumsum_HII__lr['4861'], f5007=SB_cumsum_HII__lr['5007'], tau_V=tau_V_neb_cumsum_HII__r)
        logN2O2_cumsum_HII__r = logN2O2(fNII=SB_cumsum_HII__lr['6583'], fOII=SB_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
        mask = np.bitwise_or(mask, logO23_cumsum_HII__r.mask)
        mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
        mlogO23_cumsum_HII__r, mlogN2O2_cumsum_HII__r = ma_mask_xyz(logO23_cumsum_HII__r, logN2O2_cumsum_HII__r, mask=mask)
        OH_O23_cumsum_HII__r = np.ma.masked_all((N_R_bins))
        OH_O23_cumsum_HII__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum_HII__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
        OH_O23_cumsum_HII__r[~np.isfinite(OH_O23_cumsum_HII__r)] = np.ma.masked
        # total cumulative O/H
        mask = np.zeros((N_R_bins), dtype='bool')
        logO23_cumsum__r = logO23(fOII=SB_cumsum__lr['3727'], fHb=SB_cumsum__lr['4861'], f5007=SB_cumsum__lr['5007'], tau_V=tau_V_neb_cumsum__r)
        logN2O2_cumsum__r = logN2O2(fNII=SB_cumsum__lr['6583'], fOII=SB_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
        mask = np.bitwise_or(mask, logO23_cumsum__r.mask)
        mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
        mlogO23_cumsum__r, mlogN2O2_cumsum__r = ma_mask_xyz(logO23_cumsum__r, logN2O2_cumsum__r, mask=mask)
        OH_O23_cumsum__r = np.ma.masked_all((N_R_bins))
        OH_O23_cumsum__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
        OH_O23_cumsum__r[~np.isfinite(OH_O23_cumsum__r)] = np.ma.masked
        #############################
        # N2O2 Dopita et al. (2013) #
        #############################
        mask = np.zeros((N_y, N_x), dtype='bool')
        mask = np.bitwise_or(mask, SB__lyx['3727'].mask)
        mask = np.bitwise_or(mask, SB__lyx['6583'].mask)
        logN2O2__yx = logN2O2(fNII=SB__lyx['6583'], fOII=SB__lyx['3727'], tau_V=tau_V_neb__yx)
        mask = np.bitwise_or(mask, logN2O2__yx.mask)
        mlogN2O2__yx = np.ma.masked_array(logN2O2__yx, mask=mask)
        OH_N2O2__yx = np.ma.masked_all((N_y, N_x))
        OH_N2O2__yx[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2__yx.compressed())
        OH_N2O2__yx[~np.isfinite(OH_N2O2__yx)] = np.ma.masked
        # HII cumulative O/H
        mask = np.zeros((N_R_bins), dtype='bool')
        logN2O2_cumsum_HII__r = logN2O2(fNII=SB_cumsum_HII__lr['6583'], fOII=SB_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
        mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
        mlogN2O2_cumsum_HII__r = np.ma.masked_array(logN2O2_cumsum_HII__r, mask=mask)
        OH_N2O2_cumsum_HII__r = np.ma.masked_all((N_R_bins))
        OH_N2O2_cumsum_HII__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
        # total cumulative O/H
        mask = np.zeros((N_R_bins), dtype='bool')
        logN2O2_cumsum__r = logN2O2(fNII=SB_cumsum__lr['6583'], fOII=SB_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
        mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
        mlogN2O2_cumsum__r = np.ma.masked_array(logN2O2_cumsum__r, mask=mask)
        OH_N2O2_cumsum__r = np.ma.masked_all((N_R_bins))
        OH_N2O2_cumsum__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
        #####################

        #####################
        # PLOT
        #####################
        N_cols = 4
        N_rows = 2
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        cmap = cmap_discrete()
        (axes_row1, axes_row2) = axArr
        f.suptitle(r'%s - %s: %d pixels (%d zones) - classif. W${}_{H\alpha}$' % (califaID, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))
        plot_dict = dict(
            O3N2=dict(OH__yx=OH_O3N2__yx, OH_cumsum__r=OH_O3N2_cumsum__r, OH_cumsum_HII__r=OH_O3N2_cumsum_HII__r),
            N2=dict(OH__yx=OH_N2Ha__yx, OH_cumsum__r=OH_N2Ha_cumsum__r, OH_cumsum_HII__r=OH_N2Ha_cumsum_HII__r),
            O23=dict(OH__yx=OH_O23__yx, OH_cumsum__r=OH_O23_cumsum__r, OH_cumsum_HII__r=OH_O23_cumsum_HII__r),
            N2O2=dict(OH__yx=OH_N2O2__yx, OH_cumsum__r=OH_N2O2_cumsum__r, OH_cumsum_HII__r=OH_N2O2_cumsum_HII__r),
        )
        for i, k in enumerate(['O3N2', 'N2', 'O23', 'N2O2']):
            ax1, ax2 = zip(axes_row1, axes_row2)[i]
            OH__yx = plot_dict[k]['OH__yx']
            OH_label = k
            plot_OH(ax1, pixelDistance_HLR__yx, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range)
            plot_text_ax(ax1, '%c)' % chr(97+i*2), 0.02, 0.98, 16, 'top', 'left', 'k')
            OH_cumsum__r = plot_dict[k]['OH_cumsum__r']
            OH_cumsum_HII__r = plot_dict[k]['OH_cumsum_HII__r']
            plot_cumulative_OH(ax2, R_bin_center__r, OH_cumsum__r, OH_cumsum_HII__r, OH_label, OH_range, distance_range)
            plot_text_ax(ax2, '%c)' % chr(97+(i*2+1)), 0.02, 0.98, 16, 'top', 'left', 'k')
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('%s-fig2.png' % califaID)
        plt.close(f)


def figs4567(ALL, gals=None):
    """
    Fig 4.
        O/H (O3N2)
        Uma figura resumindo todas as galaxias de califa com emission lines para
        analisar o efeito do dig e o efeito da abertura no sdss
        3 bins de CI (concentration index) e 3 bins de b/a (9 paineis)
        Em cada painel, plotar, para cada galaxia, o valor de log O/H(O3N2)
        integrated em funcao de R[pixel]/R_50[pixel].
        Se poderia ate colorir cada curva em funcao, por exemplo, da massa total
        da galaxia, ou entao em funcao da posicao do oiii/hb vs nii/ha integrado
        no bpt.

    Fig 5.
        Igual a Fig 4 mas usando O/H (N2)

    Fig 6.
        Igual a Fig 4 mas usando O/H (O23)

    Fig 7.
        Igual a Fig 4 mas usando O/H (N2O2)
    """
    N_zones = len(ALL.califaID__z)
    # p33CI, p66CI = np.percentile(ALL.CI, [33, 66])
    p33CI, p66CI = 2.6, 3
    p33ba, p66ba = np.percentile(ALL.ba, [33, 66])
    # p33ba, p66ba = 0.34, 0.67
    import itertools
    # CI = np.hstack(list(itertools.chain(list(itertools.repeat(CI, ALL.N_x[i] * ALL.N_y[i])) for i, CI in enumerate(ALL.CI))))
    # ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_x[i] * ALL.N_y[i])) for i, ba in enumerate(ALL.ba))))
    CI = np.hstack(list(itertools.chain(list(itertools.repeat(CI, ALL.N_zone[i])) for i, CI in enumerate(ALL.CI))))
    ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_zone[i])) for i, ba in enumerate(ALL.ba))))
    Mtot = np.hstack(list(itertools.chain(list(itertools.repeat(Mtot, ALL.N_zone[i])) for i, Mtot in enumerate(ALL.Mtot))))
    CI_ba_sel = [
        [np.where(np.bitwise_and(CI < p33CI, ba < p33ba)),
         np.where(np.bitwise_and(CI < p33CI, np.bitwise_and(ba >= p33ba, ba < p66ba))),
         np.where(np.bitwise_and(CI < p33CI, ba >= p66ba))],
        [np.where(np.bitwise_and(np.bitwise_and(CI >= p33CI, CI < p66CI), ba < p33ba)),
         np.where(np.bitwise_and(np.bitwise_and(CI >= p33CI, CI < p66CI), np.bitwise_and(ba >= p33ba, ba < p66ba))),
         np.where(np.bitwise_and(np.bitwise_and(CI >= p33CI, CI < p66CI), ba >= p66ba))],
        [np.where(np.bitwise_and(CI >= p66CI, ba < p33ba)),
         np.where(np.bitwise_and(CI >= p66CI, np.bitwise_and(ba >= p33ba, ba < p66ba))),
         np.where(np.bitwise_and(CI >= p66CI, ba >= p66ba))]
    ]
    labels = [
        [r'CI < %.2f - ba < %.2f' % (p33CI, p33ba), r'CI < %.2f - %.2f $\leq$ ba < %.2f' % (p33CI, p33ba, p66ba), r'CI < %.2f - ba $\geq$ %.2f' % (p33CI, p66ba)],
        [r'%.2f $\leq$ CI < %.2f - ba < %.2f' % (p33CI, p66CI, p33ba), r'%.2f $\leq$ CI < %.2f - %.2f $\leq$ ba < %.2f' % (p33CI, p66CI, p33ba, p66ba), r'%.2f $\leq$ CI < %.2f - ba $\geq$ %.2f' % (p33CI, p66CI, p66ba)],
        [r'CI $\geq$ %.2f - ba < %.2f' % (p66CI, p33ba), r'CI $\geq$ %.2f - %.2f $\leq$ ba < %.2f' % (p66CI, p33ba, p66ba), r'CI $\geq$ %.2f - ba $\geq$ %.2f' % (p66CI, p66ba)],
    ]
    # fig 4
    ALL_O3Hb__z = ALL.SB5007__z/ALL.SB4861__z
    ALL_N2Ha__z = ALL.SB6583__z/ALL.SB6563__z
    ALL_OH_O3N2__z = 8.73 - 0.32 * np.ma.log10(ALL_O3Hb__z/ALL_N2Ha__z)
    print 'FIG 4'
    figs4567_plots(ALL, ALL_OH_O3N2__z, 'O3N2', CI_ba_sel, labels, Mtot)
    # fig 5
    ALL_OH_N2Ha__z = 8.90 + 0.57 * np.ma.log10(ALL_N2Ha__z)
    print 'FIG 5'
    figs4567_plots(ALL, ALL_OH_N2Ha__z, 'N2', CI_ba_sel, labels, Mtot)
    # fig 6
    ALL_tau_V_neb__z = np.where(np.less(ALL.tau_V_neb__z.filled(-1), 0), 0, ALL.tau_V_neb__z)
    ALL_logO23__z = logO23(fOII=ALL.f3727__z, fHb=ALL.f4861__z, f5007=ALL.f5007__z, tau_V=ALL_tau_V_neb__z)
    ALL_logN2O2__z = logN2O2(fNII=ALL.f6583__z, fOII=ALL.f3727__z, tau_V=ALL_tau_V_neb__z)
    mask = np.zeros((N_zones), dtype='bool')
    mask = np.bitwise_or(mask, ALL.f3727__z.mask)
    mask = np.bitwise_or(mask, ALL.f4861__z.mask)
    mask = np.bitwise_or(mask, ALL.f5007__z.mask)
    mask = np.bitwise_or(mask, ALL.f6583__z.mask)
    mask = np.bitwise_or(mask, ALL_logO23__z.mask)
    mask = np.bitwise_or(mask, ALL_logN2O2__z.mask)
    ALL_mlogO23__z, ALL_mlogN2O2__z = ma_mask_xyz(ALL_logO23__z, ALL_logN2O2__z, mask=mask)
    ALL_OH_O23__z = np.ma.masked_all((N_zones))
    ALL_OH_O23__z[~mask] = OH_O23(logO23_ratio=ALL_mlogO23__z.compressed(), logN2O2_ratio=ALL_mlogN2O2__z.compressed())
    ALL_OH_O23__z[~np.isfinite(ALL_OH_O23__z)] = np.ma.masked
    print 'FIG 6'
    figs4567_plots(ALL, ALL_OH_O23__z, 'O23', CI_ba_sel, labels, Mtot)
    # fig 7
    mask = np.zeros((N_zones), dtype='bool')
    mask = np.bitwise_or(mask, ALL.f3727__z.mask)
    mask = np.bitwise_or(mask, ALL.f6583__z.mask)
    mask = np.bitwise_or(mask, ALL_logN2O2__z.mask)
    ALL_mlogN2O2__z = np.ma.masked_array(ALL_logN2O2__z, mask=mask)
    ALL_OH_N2O2__z = np.ma.masked_all((N_zones))
    ALL_OH_N2O2__z[~mask] = OH_N2O2(logN2O2_ratio=ALL_mlogN2O2__z.compressed())
    ALL_OH_N2O2__z[~np.isfinite(ALL_OH_N2O2__z)] = np.ma.masked
    print 'FIG 7'
    figs4567_plots(ALL, ALL_OH_N2O2__z, 'N2O2', CI_ba_sel, labels, Mtot)


def figs4567_plots(ALL, y, y_label, CI_ba_sel, labels, c):
    N_gals = len(ALL.ba)
    N_zones = len(ALL.califaID__z)
    N_rows = 3
    N_cols = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
    f.suptitle(r'%d galaxies (%d zones) - %s' % (N_gals, N_zones, y_label))
    for i in range(N_rows):
        for j in range(N_cols):
            ax = axArr[i][j]
            sel = CI_ba_sel[i][j]
            print i, j, y[sel]
            label = labels[i][j]
            ax.set_xlim(distance_range)
            ax.set_ylim(OH_range)
            sc = ax.scatter(ALL.zoneDistance_HLR[sel], y[sel], c=np.log10(c[sel]), cmap='viridis_r', s=3, vmax=11.4, vmin=9, **dflt_kw_scatter)
            # ax.set_aspect('equal', 'box')
            the_divider = make_axes_locatable(ax)
            color_axis = the_divider.append_axes('right', size='5%', pad=0)
            cb = plt.colorbar(sc, cax=color_axis)  # ax=ax, ticks=[0, .5, 1, 1.5, 2, 2.5, 3], pad=0)
            cb.set_label(r'$\log$ M${}_{tot}^{gal}$ [M${}_\odot$]')
            xm, ym = ma_mask_xyz(ALL.zoneDistance_HLR[sel], y[sel])
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            for k in xrange(len(rs.xPrc)):
                ax.plot(rs.xPrc[k], rs.yPrc[k], 'k--', lw=3)
            ax.plot(rs.xMedian, rs.yMedian, 'k-', lw=3)
            ax.set_title(label)
            # plot_text_ax(ax, '%s' % label, 0.02, 0.98, 18, 'top', 'left', 'k')
            plot_text_ax(ax, '%d' % xm.count(), 0.02, 0.02, 18, 'bottom', 'left', 'k')
            if i < N_rows - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            if j:
                plt.setp(ax.get_yticklabels(), visible=False)
    axArr[2][1].set_xlabel(r'R [HLR]')
    axArr[1][0].set_ylabel(r'$12\ +\ \log$ (O/H) - (%s) [Z${}_\odot$]' % y_label)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.subplots_adjust(hspace=0.2, wspace=0.3)
    f.savefig('%s_R.png' % y_label)
    plt.close(f)



def fig4_profile(ALL, gals=None):
    sel_WHa_DIG__gz = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__gz = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__gz = (ALL.W6563__z >= HII_WHa_threshold).filled(False)
    sel_WHa_DIG__gyx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__gyx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__gyx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)

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
        OH_name = 'O3N2'
        p33ba, p66ba = np.percentile(ALL.ba, [33, 66])

        sel_gals_ba = [
            np.less(ALL.ba, p33ba),
            np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba)),
            np.greater_equal(ALL.ba, p66ba)
        ]
        ba_labels = [
            r'b/a < %.2f' % p33ba,
            r'%.2f <= b/a < %.2f' % (p33ba, p66ba),
            r'b/a >= %.2f' % p66ba,
        ]

        versions = dict(
            Mtot=dict(var=np.ma.log10(ALL.Mtot), label=r'$\log$ M${}_\star}$'),
            CI_9050=dict(var=ALL.CI_9050, label=r'CI${}_{50}^{90}$'),
        )
        for k_ver, v_ver in versions.iteritems():
            page = 1
            sort_var = v_ver['var']
            for i_sel, selection in enumerate(sel_gals_ba):
                print k_ver, page
                sel_gals = np.asarray(gals)[selection]
                N_gals_sel = len(sel_gals)
                iS = np.argsort(sort_var[selection])
                N_rows, N_cols = 10, 12
                row, col = 0, 0
                # f = plt.figure(dpi=300, figsize=(N_cols * 2, N_rows * 2))
                # from mpl_toolkits.axes_grid1 import Grid
                # grid = Grid(f, rect=111, nrows_ncols=(N_rows, N_cols), axes_pad=0.0, label_mode='all',)
                f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
                f.suptitle('OH (%s) profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_name, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
                # for i in xrange(N_rows):
                #     for j in xrange(N_cols):
                #         # axArr[i, j].axis('off')
                ax_i = 0
                N_axes = N_cols * N_rows
                for g in sel_gals[iS]:
                    ax = np.ravel(axArr)[ax_i]
                    # ax = grid[i]
                    ax = axArr[row][col]
                    # ax.axis('off')

                    HLR_pix = ALL.get_gal_prop_unique(g, ALL.HLR_pix)
                    N_x = ALL.get_gal_prop_unique(g, ALL.N_x)
                    N_y = ALL.get_gal_prop_unique(g, ALL.N_y)
                    pixelDistance__yx = ALL.get_gal_prop(g, ALL.pixelDistance__yx).reshape(N_y, N_x)
                    pixelDistance_HLR__yx = pixelDistance__yx / HLR_pix
                    SB__lyx, HaHb_classif, cumsum_classif = cumulative_profiles(ALL, g, sel_WHa_DIG__gyx, sel_WHa_COMP__gyx, sel_WHa_HII__gyx)
                    SB_cumsum__lr = cumsum_classif['total__lr']['v']
                    SB_cumsum_HII__lr = cumsum_classif['HII__lr']['v']
                    tau_V_neb__yx = HaHb_classif['total']['tau_V_neb__yx']
                    tau_V_neb_cumsum__r = HaHb_classif['total']['tau_V_neb_cumsum__r']
                    tau_V_neb_cumsum_HII__r = HaHb_classif['HII']['tau_V_neb_cumsum__r']
                    ####################################
                    # O/H - Relative Oxygen abundances #
                    ####################################
                    #############
                    # O3N2 PP04 #
                    #############
                    O3Hb__yx = SB__lyx['5007']/SB__lyx['4861']
                    N2Ha__yx = SB__lyx['6583']/SB__lyx['6563']
                    OH_O3N2__yx = 8.73 - 0.32 * np.ma.log10(O3Hb__yx/N2Ha__yx)
                    O3Hb_cumsum__r = SB_cumsum__lr['5007']/SB_cumsum__lr['4861']
                    N2Ha_cumsum__r = SB_cumsum__lr['6583']/SB_cumsum__lr['6563']
                    OH_O3N2_cumsum__r = 8.73 - 0.32 * np.ma.log10(O3Hb_cumsum__r/N2Ha_cumsum__r)
                    O3Hb_cumsum_HII__r = SB_cumsum_HII__lr['5007']/SB_cumsum_HII__lr['4861']
                    N2Ha_cumsum_HII__r = SB_cumsum_HII__lr['6583']/SB_cumsum_HII__lr['6563']
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
                    mask = np.zeros((N_y, N_x), dtype='bool')
                    mask = np.bitwise_or(mask, SB__lyx['3727'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['4861'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['5007'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['6583'].mask)
                    logO23__yx = logO23(fOII=SB__lyx['3727'], fHb=SB__lyx['4861'], f5007=SB__lyx['5007'], tau_V=tau_V_neb__yx)
                    logN2O2__yx = logN2O2(fNII=SB__lyx['6583'], fOII=SB__lyx['3727'], tau_V=tau_V_neb__yx)
                    mask = np.bitwise_or(mask, logO23__yx.mask)
                    mask = np.bitwise_or(mask, logN2O2__yx.mask)
                    mlogO23__yx, mlogN2O2__yx = ma_mask_xyz(logO23__yx, logN2O2__yx, mask=mask)
                    OH_O23__yx = np.ma.masked_all((N_y, N_x))
                    OH_O23__yx[~mask] = OH_O23(logO23_ratio=mlogO23__yx.compressed(), logN2O2_ratio=mlogN2O2__yx.compressed())
                    OH_O23__yx[~np.isfinite(OH_O23__yx)] = np.ma.masked
                    # HII cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logO23_cumsum_HII__r = logO23(fOII=SB_cumsum_HII__lr['3727'], fHb=SB_cumsum_HII__lr['4861'], f5007=SB_cumsum_HII__lr['5007'], tau_V=tau_V_neb_cumsum_HII__r)
                    logN2O2_cumsum_HII__r = logN2O2(fNII=SB_cumsum_HII__lr['6583'], fOII=SB_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
                    mask = np.bitwise_or(mask, logO23_cumsum_HII__r.mask)
                    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
                    mlogO23_cumsum_HII__r, mlogN2O2_cumsum_HII__r = ma_mask_xyz(logO23_cumsum_HII__r, logN2O2_cumsum_HII__r, mask=mask)
                    OH_O23_cumsum_HII__r = np.ma.masked_all((N_R_bins))
                    OH_O23_cumsum_HII__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum_HII__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
                    OH_O23_cumsum_HII__r[~np.isfinite(OH_O23_cumsum_HII__r)] = np.ma.masked
                    # total cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logO23_cumsum__r = logO23(fOII=SB_cumsum__lr['3727'], fHb=SB_cumsum__lr['4861'], f5007=SB_cumsum__lr['5007'], tau_V=tau_V_neb_cumsum__r)
                    logN2O2_cumsum__r = logN2O2(fNII=SB_cumsum__lr['6583'], fOII=SB_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
                    mask = np.bitwise_or(mask, logO23_cumsum__r.mask)
                    mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
                    mlogO23_cumsum__r, mlogN2O2_cumsum__r = ma_mask_xyz(logO23_cumsum__r, logN2O2_cumsum__r, mask=mask)
                    OH_O23_cumsum__r = np.ma.masked_all((N_R_bins))
                    OH_O23_cumsum__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
                    OH_O23_cumsum__r[~np.isfinite(OH_O23_cumsum__r)] = np.ma.masked
                    #############################
                    # N2O2 Dopita et al. (2013) #
                    #############################
                    mask = np.zeros((N_y, N_x), dtype='bool')
                    mask = np.bitwise_or(mask, SB__lyx['3727'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['6583'].mask)
                    logN2O2__yx = logN2O2(fNII=SB__lyx['6583'], fOII=SB__lyx['3727'], tau_V=tau_V_neb__yx)
                    mask = np.bitwise_or(mask, logN2O2__yx.mask)
                    mlogN2O2__yx = np.ma.masked_array(logN2O2__yx, mask=mask)
                    OH_N2O2__yx = np.ma.masked_all((N_y, N_x))
                    OH_N2O2__yx[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2__yx.compressed())
                    OH_N2O2__yx[~np.isfinite(OH_N2O2__yx)] = np.ma.masked
                    # HII cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logN2O2_cumsum_HII__r = logN2O2(fNII=SB_cumsum_HII__lr['6583'], fOII=SB_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
                    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
                    mlogN2O2_cumsum_HII__r = np.ma.masked_array(logN2O2_cumsum_HII__r, mask=mask)
                    OH_N2O2_cumsum_HII__r = np.ma.masked_all((N_R_bins))
                    OH_N2O2_cumsum_HII__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
                    # total cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logN2O2_cumsum__r = logN2O2(fNII=SB_cumsum__lr['6583'], fOII=SB_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
                    mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
                    mlogN2O2_cumsum__r = np.ma.masked_array(logN2O2_cumsum__r, mask=mask)
                    OH_N2O2_cumsum__r = np.ma.masked_all((N_R_bins))
                    OH_N2O2_cumsum__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
                    #####################
                    y = OH_O3N2__yx
                    y_cumsum = OH_O3N2_cumsum__r
                    y_cumsum_HII = OH_O3N2_cumsum_HII__r
                    sel_DIG__yx, sel_COMP__yx, sel_HII__yx = get_selections(ALL, g, sel_WHa_DIG__gyx, sel_WHa_COMP__gyx, sel_WHa_HII__gyx)

                    ax.scatter(pixelDistance_HLR__yx, y, s=5, color='silver', **dflt_kw_scatter)
                    ax.scatter(pixelDistance_HLR__yx[sel_HII__yx], y[sel_HII__yx], s=5, color='blue', **dflt_kw_scatter)
                    xm, ym = ma_mask_xyz(pixelDistance_HLR__yx, y, mask=None)
                    ax.plot(R_bin_center__r, y_cumsum, linewidth=2, linestyle='-', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.plot(R_bin_center__r, y_cumsum_HII, linewidth=2, linestyle='-', marker='*', markeredgewidth=0, markeredgecolor=colors_lines_DIG_COMP_HII[2], c='cyan', markersize=5)
                    min_npts = 10
                    yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
                    txt = '%s' % g
                    plot_text_ax(ax, txt, 0.02, 0.98, 16, 'top', 'left', color='k')
                    txt = r'%s: $%.2f$' % (v_ver['label'], ALL.get_gal_prop_unique(g, sort_var))
                    plot_text_ax(ax, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
                    if npts.any():
                        sel = npts > min_npts
                        ax.plot(bin_center, yPrc[2], '-', lw=1, c='gray')
                        ax.plot(bin_center[sel], yPrc[2][sel], '-', lw=1, c='k')
                        # ax.plot(bin_center[sel], yPrc[2][sel], linest-yle='', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.yaxis.set_ticks([8, 8.25, 8.5, 8.75, 9, 9.25])
                    ax.xaxis.set_ticks([0, 1, 2])
                    ax.set_xlim(distance_range)
                    ax.set_ylim(OH_range)
                    ax.grid('on')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    if g == sel_gals[iS][-1]:
                        axArr[row][0].xaxis.set_visible(True)
                        plt.setp(axArr[row][0].get_xticklabels(), visible=True)
                        axArr[row][0].set_xlabel(r'R [HLR]')
                        for i_r in xrange(0, row+1):
                            axArr[i_r][0].yaxis.set_visible(True)
                            plt.setp(axArr[i_r][0].get_yticklabels(), visible=True)
                    col += 1
                    if col == N_cols:
                        col = 0
                        row += 1
                    ax_i += 1
                if ax_i < N_axes:
                    for i in xrange(ax_i, N_axes):
                        np.ravel(axArr)[i].set_axis_off()
                f.subplots_adjust(left=0.05, bottom=0.05, wspace=0, hspace=0, right=0.95, top=0.95)
                f.savefig('%s_%s_p%d.png' % (OH_name, k_ver, page), orientation='landscape', dpi=200)
                plt.close(f)
                page += 1


def fig5_profile_new(ALL, gals=None):
    sel_WHa_DIG__gz = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__gz = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__gz = (ALL.W6563__z >= HII_WHa_threshold).filled(False)
    sel_WHa_DIG__gyx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__gyx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__gyx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)

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
        OH_name = 'N2'
        p33ba, p66ba = np.percentile(ALL.ba, [33, 66])

        sel_gals_ba = [
            np.less(ALL.ba, p33ba),
            np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba)),
            np.greater_equal(ALL.ba, p66ba)
        ]
        ba_labels = [
            r'b/a < %.2f' % p33ba,
            r'%.2f <= b/a < %.2f' % (p33ba, p66ba),
            r'b/a >= %.2f' % p66ba,
        ]

        versions = dict(
            Mtot=dict(var=np.ma.log10(ALL.Mtot), label=r'$\log$ M${}_\star}$'),
            CI_9050=dict(var=ALL.CI_9050, label=r'CI${}_{50}^{90}$'),
        )
        for k_ver, v_ver in versions.iteritems():
            page = 1
            sort_var = v_ver['var']
            for i_sel, selection in enumerate(sel_gals_ba):
                print k_ver, page
                sel_gals = np.asarray(gals)[selection]
                N_gals_sel = len(sel_gals)
                iS = np.argsort(sort_var[selection])
                N_rows, N_cols = 10, 12
                row, col = 0, 0
                # f = plt.figure(dpi=300, figsize=(N_cols * 2, N_rows * 2))
                # from mpl_toolkits.axes_grid1 import Grid
                # grid = Grid(f, rect=111, nrows_ncols=(N_rows, N_cols), axes_pad=0.0, label_mode='all',)
                f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
                f.suptitle('OH (%s) profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_name, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
                # for i in xrange(N_rows):
                #     for j in xrange(N_cols):
                #         # axArr[i, j].axis('off')
                ax_i = 0
                N_axes = N_cols * N_rows
                for g in sel_gals[iS]:
                    ax = np.ravel(axArr)[ax_i]
                    # ax = grid[i]
                    ax = axArr[row][col]
                    # ax.axis('off')

                    HLR_pix = ALL.get_gal_prop_unique(g, ALL.HLR_pix)
                    N_x = ALL.get_gal_prop_unique(g, ALL.N_x)
                    N_y = ALL.get_gal_prop_unique(g, ALL.N_y)
                    pixelDistance__yx = ALL.get_gal_prop(g, ALL.pixelDistance__yx).reshape(N_y, N_x)
                    pixelDistance_HLR__yx = pixelDistance__yx / HLR_pix
                    SB__lyx, HaHb_classif, cumsum_classif = cumulative_profiles(ALL, g, sel_WHa_DIG__gyx, sel_WHa_COMP__gyx, sel_WHa_HII__gyx)
                    SB_cumsum__lr = cumsum_classif['total__lr']['v']
                    SB_cumsum_HII__lr = cumsum_classif['HII__lr']['v']
                    tau_V_neb__yx = HaHb_classif['total']['tau_V_neb__yx']
                    tau_V_neb_cumsum__r = HaHb_classif['total']['tau_V_neb_cumsum__r']
                    tau_V_neb_cumsum_HII__r = HaHb_classif['HII']['tau_V_neb_cumsum__r']
                    ####################################
                    # O/H - Relative Oxygen abundances #
                    ####################################
                    #############
                    # O3N2 PP04 #
                    #############
                    O3Hb__yx = SB__lyx['5007']/SB__lyx['4861']
                    N2Ha__yx = SB__lyx['6583']/SB__lyx['6563']
                    OH_O3N2__yx = 8.73 - 0.32 * np.ma.log10(O3Hb__yx/N2Ha__yx)
                    O3Hb_cumsum__r = SB_cumsum__lr['5007']/SB_cumsum__lr['4861']
                    N2Ha_cumsum__r = SB_cumsum__lr['6583']/SB_cumsum__lr['6563']
                    OH_O3N2_cumsum__r = 8.73 - 0.32 * np.ma.log10(O3Hb_cumsum__r/N2Ha_cumsum__r)
                    O3Hb_cumsum_HII__r = SB_cumsum_HII__lr['5007']/SB_cumsum_HII__lr['4861']
                    N2Ha_cumsum_HII__r = SB_cumsum_HII__lr['6583']/SB_cumsum_HII__lr['6563']
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
                    mask = np.zeros((N_y, N_x), dtype='bool')
                    mask = np.bitwise_or(mask, SB__lyx['3727'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['4861'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['5007'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['6583'].mask)
                    logO23__yx = logO23(fOII=SB__lyx['3727'], fHb=SB__lyx['4861'], f5007=SB__lyx['5007'], tau_V=tau_V_neb__yx)
                    logN2O2__yx = logN2O2(fNII=SB__lyx['6583'], fOII=SB__lyx['3727'], tau_V=tau_V_neb__yx)
                    mask = np.bitwise_or(mask, logO23__yx.mask)
                    mask = np.bitwise_or(mask, logN2O2__yx.mask)
                    mlogO23__yx, mlogN2O2__yx = ma_mask_xyz(logO23__yx, logN2O2__yx, mask=mask)
                    OH_O23__yx = np.ma.masked_all((N_y, N_x))
                    OH_O23__yx[~mask] = OH_O23(logO23_ratio=mlogO23__yx.compressed(), logN2O2_ratio=mlogN2O2__yx.compressed())
                    OH_O23__yx[~np.isfinite(OH_O23__yx)] = np.ma.masked
                    # HII cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logO23_cumsum_HII__r = logO23(fOII=SB_cumsum_HII__lr['3727'], fHb=SB_cumsum_HII__lr['4861'], f5007=SB_cumsum_HII__lr['5007'], tau_V=tau_V_neb_cumsum_HII__r)
                    logN2O2_cumsum_HII__r = logN2O2(fNII=SB_cumsum_HII__lr['6583'], fOII=SB_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
                    mask = np.bitwise_or(mask, logO23_cumsum_HII__r.mask)
                    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
                    mlogO23_cumsum_HII__r, mlogN2O2_cumsum_HII__r = ma_mask_xyz(logO23_cumsum_HII__r, logN2O2_cumsum_HII__r, mask=mask)
                    OH_O23_cumsum_HII__r = np.ma.masked_all((N_R_bins))
                    OH_O23_cumsum_HII__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum_HII__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
                    OH_O23_cumsum_HII__r[~np.isfinite(OH_O23_cumsum_HII__r)] = np.ma.masked
                    # total cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logO23_cumsum__r = logO23(fOII=SB_cumsum__lr['3727'], fHb=SB_cumsum__lr['4861'], f5007=SB_cumsum__lr['5007'], tau_V=tau_V_neb_cumsum__r)
                    logN2O2_cumsum__r = logN2O2(fNII=SB_cumsum__lr['6583'], fOII=SB_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
                    mask = np.bitwise_or(mask, logO23_cumsum__r.mask)
                    mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
                    mlogO23_cumsum__r, mlogN2O2_cumsum__r = ma_mask_xyz(logO23_cumsum__r, logN2O2_cumsum__r, mask=mask)
                    OH_O23_cumsum__r = np.ma.masked_all((N_R_bins))
                    OH_O23_cumsum__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
                    OH_O23_cumsum__r[~np.isfinite(OH_O23_cumsum__r)] = np.ma.masked
                    #############################
                    # N2O2 Dopita et al. (2013) #
                    #############################
                    mask = np.zeros((N_y, N_x), dtype='bool')
                    mask = np.bitwise_or(mask, SB__lyx['3727'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['6583'].mask)
                    logN2O2__yx = logN2O2(fNII=SB__lyx['6583'], fOII=SB__lyx['3727'], tau_V=tau_V_neb__yx)
                    mask = np.bitwise_or(mask, logN2O2__yx.mask)
                    mlogN2O2__yx = np.ma.masked_array(logN2O2__yx, mask=mask)
                    OH_N2O2__yx = np.ma.masked_all((N_y, N_x))
                    OH_N2O2__yx[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2__yx.compressed())
                    OH_N2O2__yx[~np.isfinite(OH_N2O2__yx)] = np.ma.masked
                    # HII cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logN2O2_cumsum_HII__r = logN2O2(fNII=SB_cumsum_HII__lr['6583'], fOII=SB_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
                    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
                    mlogN2O2_cumsum_HII__r = np.ma.masked_array(logN2O2_cumsum_HII__r, mask=mask)
                    OH_N2O2_cumsum_HII__r = np.ma.masked_all((N_R_bins))
                    OH_N2O2_cumsum_HII__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
                    # total cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logN2O2_cumsum__r = logN2O2(fNII=SB_cumsum__lr['6583'], fOII=SB_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
                    mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
                    mlogN2O2_cumsum__r = np.ma.masked_array(logN2O2_cumsum__r, mask=mask)
                    OH_N2O2_cumsum__r = np.ma.masked_all((N_R_bins))
                    OH_N2O2_cumsum__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
                    #####################
                    y = OH_N2Ha__yx
                    y_cumsum = OH_N2Ha_cumsum__r
                    y_cumsum_HII = OH_N2Ha_cumsum_HII__r
                    sel_DIG__yx, sel_COMP__yx, sel_HII__yx = get_selections(ALL, g, sel_WHa_DIG__gyx, sel_WHa_COMP__gyx, sel_WHa_HII__gyx)

                    ax.scatter(pixelDistance_HLR__yx, y, s=5, color='silver', **dflt_kw_scatter)
                    ax.scatter(pixelDistance_HLR__yx[sel_HII__yx], y[sel_HII__yx], s=5, color='blue', **dflt_kw_scatter)
                    xm, ym = ma_mask_xyz(pixelDistance_HLR__yx, y, mask=None)
                    ax.plot(R_bin_center__r, y_cumsum, linewidth=2, linestyle='-', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.plot(R_bin_center__r, y_cumsum_HII, linewidth=2, linestyle='-', marker='*', markeredgewidth=0, markeredgecolor=colors_lines_DIG_COMP_HII[2], c='cyan', markersize=5)

                    min_npts = 10
                    yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
                    txt = '%s' % g
                    plot_text_ax(ax, txt, 0.02, 0.98, 16, 'top', 'left', color='k')
                    txt = r'%s: $%.2f$' % (v_ver['label'], ALL.get_gal_prop_unique(g, sort_var))
                    plot_text_ax(ax, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
                    if npts.any():
                        sel = npts > min_npts
                        ax.plot(bin_center, yPrc[2], '--', lw=1, c='gray')
                        ax.plot(bin_center[sel], yPrc[2][sel], '--', lw=1, c='k')
                        # ax.plot(bin_center[sel], yPrc[2][sel], linestyle='', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.yaxis.set_ticks([8, 8.25, 8.5, 8.75, 9, 9.25])
                    ax.xaxis.set_ticks([0, 1, 2])
                    ax.set_xlim(distance_range)
                    ax.set_ylim(OH_range)
                    ax.grid('on')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    if g == sel_gals[iS][-1]:
                        axArr[row][0].xaxis.set_visible(True)
                        plt.setp(axArr[row][0].get_xticklabels(), visible=True)
                        axArr[row][0].set_xlabel(r'R [HLR]')
                        for i_r in xrange(0, row+1):
                            axArr[i_r][0].yaxis.set_visible(True)
                            plt.setp(axArr[i_r][0].get_yticklabels(), visible=True)
                    col += 1
                    if col == N_cols:
                        col = 0
                        row += 1
                    ax_i += 1
                if ax_i < N_axes:
                    for i in xrange(ax_i, N_axes):
                        np.ravel(axArr)[i].set_axis_off()
                f.subplots_adjust(left=0.05, bottom=0.05, wspace=0, hspace=0, right=0.95, top=0.95)
                f.savefig('%s_%s_p%d.png' % (OH_name, k_ver, page), orientation='landscape', dpi=200)
                plt.close(f)
                page += 1

def fig6_profile_new(ALL, gals=None):
    sel_WHa_DIG__gz = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__gz = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__gz = (ALL.W6563__z >= HII_WHa_threshold).filled(False)
    sel_WHa_DIG__gyx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__gyx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__gyx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)

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
        OH_name = 'O23'
        p33ba, p66ba = np.percentile(ALL.ba, [33, 66])

        sel_gals_ba = [
            np.less(ALL.ba, p33ba),
            np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba)),
            np.greater_equal(ALL.ba, p66ba)
        ]
        ba_labels = [
            r'b/a < %.2f' % p33ba,
            r'%.2f <= b/a < %.2f' % (p33ba, p66ba),
            r'b/a >= %.2f' % p66ba,
        ]

        versions = dict(
            Mtot=dict(var=np.ma.log10(ALL.Mtot), label=r'$\log$ M${}_\star}$'),
            CI_9050=dict(var=ALL.CI_9050, label=r'CI${}_{50}^{90}$'),
        )
        for k_ver, v_ver in versions.iteritems():
            page = 1
            sort_var = v_ver['var']
            for i_sel, selection in enumerate(sel_gals_ba):
                print k_ver, page
                sel_gals = np.asarray(gals)[selection]
                N_gals_sel = len(sel_gals)
                iS = np.argsort(sort_var[selection])
                N_rows, N_cols = 10, 12
                row, col = 0, 0
                # f = plt.figure(dpi=300, figsize=(N_cols * 2, N_rows * 2))
                # from mpl_toolkits.axes_grid1 import Grid
                # grid = Grid(f, rect=111, nrows_ncols=(N_rows, N_cols), axes_pad=0.0, label_mode='all',)
                f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
                f.suptitle('OH (%s) profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_name, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
                # for i in xrange(N_rows):
                #     for j in xrange(N_cols):
                #         # axArr[i, j].axis('off')
                ax_i = 0
                N_axes = N_cols * N_rows
                for g in sel_gals[iS]:
                    ax = np.ravel(axArr)[ax_i]
                    # ax = grid[i]
                    ax = axArr[row][col]
                    # ax.axis('off')

                    HLR_pix = ALL.get_gal_prop_unique(g, ALL.HLR_pix)
                    N_x = ALL.get_gal_prop_unique(g, ALL.N_x)
                    N_y = ALL.get_gal_prop_unique(g, ALL.N_y)
                    pixelDistance__yx = ALL.get_gal_prop(g, ALL.pixelDistance__yx).reshape(N_y, N_x)
                    pixelDistance_HLR__yx = pixelDistance__yx / HLR_pix
                    SB__lyx, HaHb_classif, cumsum_classif = cumulative_profiles(ALL, g, sel_WHa_DIG__gyx, sel_WHa_COMP__gyx, sel_WHa_HII__gyx)
                    SB_cumsum__lr = cumsum_classif['total__lr']['v']
                    SB_cumsum_HII__lr = cumsum_classif['HII__lr']['v']
                    tau_V_neb__yx = HaHb_classif['total']['tau_V_neb__yx']
                    tau_V_neb_cumsum__r = HaHb_classif['total']['tau_V_neb_cumsum__r']
                    tau_V_neb_cumsum_HII__r = HaHb_classif['HII']['tau_V_neb_cumsum__r']
                    ####################################
                    # O/H - Relative Oxygen abundances #
                    ####################################
                    #############
                    # O3N2 PP04 #
                    #############
                    O3Hb__yx = SB__lyx['5007']/SB__lyx['4861']
                    N2Ha__yx = SB__lyx['6583']/SB__lyx['6563']
                    OH_O3N2__yx = 8.73 - 0.32 * np.ma.log10(O3Hb__yx/N2Ha__yx)
                    O3Hb_cumsum__r = SB_cumsum__lr['5007']/SB_cumsum__lr['4861']
                    N2Ha_cumsum__r = SB_cumsum__lr['6583']/SB_cumsum__lr['6563']
                    OH_O3N2_cumsum__r = 8.73 - 0.32 * np.ma.log10(O3Hb_cumsum__r/N2Ha_cumsum__r)
                    O3Hb_cumsum_HII__r = SB_cumsum_HII__lr['5007']/SB_cumsum_HII__lr['4861']
                    N2Ha_cumsum_HII__r = SB_cumsum_HII__lr['6583']/SB_cumsum_HII__lr['6563']
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
                    mask = np.zeros((N_y, N_x), dtype='bool')
                    mask = np.bitwise_or(mask, SB__lyx['3727'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['4861'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['5007'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['6583'].mask)
                    logO23__yx = logO23(fOII=SB__lyx['3727'], fHb=SB__lyx['4861'], f5007=SB__lyx['5007'], tau_V=tau_V_neb__yx)
                    logN2O2__yx = logN2O2(fNII=SB__lyx['6583'], fOII=SB__lyx['3727'], tau_V=tau_V_neb__yx)
                    mask = np.bitwise_or(mask, logO23__yx.mask)
                    mask = np.bitwise_or(mask, logN2O2__yx.mask)
                    mlogO23__yx, mlogN2O2__yx = ma_mask_xyz(logO23__yx, logN2O2__yx, mask=mask)
                    OH_O23__yx = np.ma.masked_all((N_y, N_x))
                    OH_O23__yx[~mask] = OH_O23(logO23_ratio=mlogO23__yx.compressed(), logN2O2_ratio=mlogN2O2__yx.compressed())
                    OH_O23__yx[~np.isfinite(OH_O23__yx)] = np.ma.masked
                    # HII cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logO23_cumsum_HII__r = logO23(fOII=SB_cumsum_HII__lr['3727'], fHb=SB_cumsum_HII__lr['4861'], f5007=SB_cumsum_HII__lr['5007'], tau_V=tau_V_neb_cumsum_HII__r)
                    logN2O2_cumsum_HII__r = logN2O2(fNII=SB_cumsum_HII__lr['6583'], fOII=SB_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
                    mask = np.bitwise_or(mask, logO23_cumsum_HII__r.mask)
                    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
                    mlogO23_cumsum_HII__r, mlogN2O2_cumsum_HII__r = ma_mask_xyz(logO23_cumsum_HII__r, logN2O2_cumsum_HII__r, mask=mask)
                    OH_O23_cumsum_HII__r = np.ma.masked_all((N_R_bins))
                    OH_O23_cumsum_HII__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum_HII__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
                    OH_O23_cumsum_HII__r[~np.isfinite(OH_O23_cumsum_HII__r)] = np.ma.masked
                    # total cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logO23_cumsum__r = logO23(fOII=SB_cumsum__lr['3727'], fHb=SB_cumsum__lr['4861'], f5007=SB_cumsum__lr['5007'], tau_V=tau_V_neb_cumsum__r)
                    logN2O2_cumsum__r = logN2O2(fNII=SB_cumsum__lr['6583'], fOII=SB_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
                    mask = np.bitwise_or(mask, logO23_cumsum__r.mask)
                    mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
                    mlogO23_cumsum__r, mlogN2O2_cumsum__r = ma_mask_xyz(logO23_cumsum__r, logN2O2_cumsum__r, mask=mask)
                    OH_O23_cumsum__r = np.ma.masked_all((N_R_bins))
                    OH_O23_cumsum__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
                    OH_O23_cumsum__r[~np.isfinite(OH_O23_cumsum__r)] = np.ma.masked
                    #############################
                    # N2O2 Dopita et al. (2013) #
                    #############################
                    mask = np.zeros((N_y, N_x), dtype='bool')
                    mask = np.bitwise_or(mask, SB__lyx['3727'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['6583'].mask)
                    logN2O2__yx = logN2O2(fNII=SB__lyx['6583'], fOII=SB__lyx['3727'], tau_V=tau_V_neb__yx)
                    mask = np.bitwise_or(mask, logN2O2__yx.mask)
                    mlogN2O2__yx = np.ma.masked_array(logN2O2__yx, mask=mask)
                    OH_N2O2__yx = np.ma.masked_all((N_y, N_x))
                    OH_N2O2__yx[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2__yx.compressed())
                    OH_N2O2__yx[~np.isfinite(OH_N2O2__yx)] = np.ma.masked
                    # HII cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logN2O2_cumsum_HII__r = logN2O2(fNII=SB_cumsum_HII__lr['6583'], fOII=SB_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
                    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
                    mlogN2O2_cumsum_HII__r = np.ma.masked_array(logN2O2_cumsum_HII__r, mask=mask)
                    OH_N2O2_cumsum_HII__r = np.ma.masked_all((N_R_bins))
                    OH_N2O2_cumsum_HII__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
                    # total cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logN2O2_cumsum__r = logN2O2(fNII=SB_cumsum__lr['6583'], fOII=SB_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
                    mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
                    mlogN2O2_cumsum__r = np.ma.masked_array(logN2O2_cumsum__r, mask=mask)
                    OH_N2O2_cumsum__r = np.ma.masked_all((N_R_bins))
                    OH_N2O2_cumsum__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
                    #####################
                    y = OH_O23__yx
                    y_cumsum = OH_O23_cumsum__r
                    y_cumsum_HII = OH_O23_cumsum_HII__r
                    sel_DIG__yx, sel_COMP__yx, sel_HII__yx = get_selections(ALL, g, sel_WHa_DIG__gyx, sel_WHa_COMP__gyx, sel_WHa_HII__gyx)

                    ax.scatter(pixelDistance_HLR__yx, y, s=5, color='silver', **dflt_kw_scatter)
                    ax.scatter(pixelDistance_HLR__yx[sel_HII__yx], y[sel_HII__yx], s=5, color='blue', **dflt_kw_scatter)
                    xm, ym = ma_mask_xyz(pixelDistance_HLR__yx, y, mask=None)
                    ax.plot(R_bin_center__r, y_cumsum, linewidth=2, linestyle='-', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.plot(R_bin_center__r, y_cumsum_HII, linewidth=2, linestyle='-', marker='*', markeredgewidth=0, markeredgecolor=colors_lines_DIG_COMP_HII[2], c='cyan', markersize=5)

                    min_npts = 10
                    yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
                    txt = '%s' % g
                    plot_text_ax(ax, txt, 0.02, 0.98, 16, 'top', 'left', color='k')
                    txt = r'%s: $%.2f$' % (v_ver['label'], ALL.get_gal_prop_unique(g, sort_var))
                    plot_text_ax(ax, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
                    if npts.any():
                        sel = npts > min_npts
                        ax.plot(bin_center, yPrc[2], '--', lw=1, c='gray')
                        ax.plot(bin_center[sel], yPrc[2][sel], '--', lw=1, c='k')
                        # ax.plot(bin_center[sel], yPrc[2][sel], linestyle='', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.yaxis.set_ticks([8, 8.25, 8.5, 8.75, 9, 9.25])
                    ax.xaxis.set_ticks([0, 1, 2])
                    ax.set_xlim(distance_range)
                    ax.set_ylim(OH_range)
                    ax.grid('on')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    if g == sel_gals[iS][-1]:
                        axArr[row][0].xaxis.set_visible(True)
                        plt.setp(axArr[row][0].get_xticklabels(), visible=True)
                        axArr[row][0].set_xlabel(r'R [HLR]')
                        for i_r in xrange(0, row+1):
                            axArr[i_r][0].yaxis.set_visible(True)
                            plt.setp(axArr[i_r][0].get_yticklabels(), visible=True)
                    col += 1
                    if col == N_cols:
                        col = 0
                        row += 1
                    ax_i += 1
                if ax_i < N_axes:
                    for i in xrange(ax_i, N_axes):
                        np.ravel(axArr)[i].set_axis_off()
                f.subplots_adjust(left=0.05, bottom=0.05, wspace=0, hspace=0, right=0.95, top=0.95)
                f.savefig('%s_%s_p%d.png' % (OH_name, k_ver, page), orientation='landscape', dpi=200)
                plt.close(f)
                page += 1


def fig7_profile_new(ALL, gals=None):
    sel_WHa_DIG__gz = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__gz = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__gz = (ALL.W6563__z >= HII_WHa_threshold).filled(False)
    sel_WHa_DIG__gyx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__gyx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__gyx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)

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
        OH_name = 'N2O2'
        p33ba, p66ba = np.percentile(ALL.ba, [33, 66])

        sel_gals_ba = [
            np.less(ALL.ba, p33ba),
            np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba)),
            np.greater_equal(ALL.ba, p66ba)
        ]
        ba_labels = [
            r'b/a < %.2f' % p33ba,
            r'%.2f <= b/a < %.2f' % (p33ba, p66ba),
            r'b/a >= %.2f' % p66ba,
        ]

        versions = dict(
            Mtot=dict(var=np.ma.log10(ALL.Mtot), label=r'$\log$ M${}_\star}$'),
            CI_9050=dict(var=ALL.CI_9050, label=r'CI${}_{50}^{90}$'),
        )
        for k_ver, v_ver in versions.iteritems():
            page = 1
            sort_var = v_ver['var']
            for i_sel, selection in enumerate(sel_gals_ba):
                print k_ver, page
                sel_gals = np.asarray(gals)[selection]
                N_gals_sel = len(sel_gals)
                iS = np.argsort(sort_var[selection])
                N_rows, N_cols = 10, 12
                row, col = 0, 0
                # f = plt.figure(dpi=300, figsize=(N_cols * 2, N_rows * 2))
                # from mpl_toolkits.axes_grid1 import Grid
                # grid = Grid(f, rect=111, nrows_ncols=(N_rows, N_cols), axes_pad=0.0, label_mode='all',)
                f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
                f.suptitle('OH (%s) profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_name, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
                # for i in xrange(N_rows):
                #     for j in xrange(N_cols):
                #         # axArr[i, j].axis('off')
                ax_i = 0
                N_axes = N_cols * N_rows
                for g in sel_gals[iS]:
                    ax = np.ravel(axArr)[ax_i]
                    # ax = grid[i]
                    ax = axArr[row][col]
                    # ax.axis('off')

                    HLR_pix = ALL.get_gal_prop_unique(g, ALL.HLR_pix)
                    N_x = ALL.get_gal_prop_unique(g, ALL.N_x)
                    N_y = ALL.get_gal_prop_unique(g, ALL.N_y)
                    pixelDistance__yx = ALL.get_gal_prop(g, ALL.pixelDistance__yx).reshape(N_y, N_x)
                    pixelDistance_HLR__yx = pixelDistance__yx / HLR_pix
                    SB__lyx, HaHb_classif, cumsum_classif = cumulative_profiles(ALL, g, sel_WHa_DIG__gyx, sel_WHa_COMP__gyx, sel_WHa_HII__gyx)
                    SB_cumsum__lr = cumsum_classif['total__lr']['v']
                    SB_cumsum_HII__lr = cumsum_classif['HII__lr']['v']
                    tau_V_neb__yx = HaHb_classif['total']['tau_V_neb__yx']
                    tau_V_neb_cumsum__r = HaHb_classif['total']['tau_V_neb_cumsum__r']
                    tau_V_neb_cumsum_HII__r = HaHb_classif['HII']['tau_V_neb_cumsum__r']
                    ####################################
                    # O/H - Relative Oxygen abundances #
                    ####################################
                    #############
                    # O3N2 PP04 #
                    #############
                    O3Hb__yx = SB__lyx['5007']/SB__lyx['4861']
                    N2Ha__yx = SB__lyx['6583']/SB__lyx['6563']
                    OH_O3N2__yx = 8.73 - 0.32 * np.ma.log10(O3Hb__yx/N2Ha__yx)
                    O3Hb_cumsum__r = SB_cumsum__lr['5007']/SB_cumsum__lr['4861']
                    N2Ha_cumsum__r = SB_cumsum__lr['6583']/SB_cumsum__lr['6563']
                    OH_O3N2_cumsum__r = 8.73 - 0.32 * np.ma.log10(O3Hb_cumsum__r/N2Ha_cumsum__r)
                    O3Hb_cumsum_HII__r = SB_cumsum_HII__lr['5007']/SB_cumsum_HII__lr['4861']
                    N2Ha_cumsum_HII__r = SB_cumsum_HII__lr['6583']/SB_cumsum_HII__lr['6563']
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
                    mask = np.zeros((N_y, N_x), dtype='bool')
                    mask = np.bitwise_or(mask, SB__lyx['3727'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['4861'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['5007'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['6583'].mask)
                    logO23__yx = logO23(fOII=SB__lyx['3727'], fHb=SB__lyx['4861'], f5007=SB__lyx['5007'], tau_V=tau_V_neb__yx)
                    logN2O2__yx = logN2O2(fNII=SB__lyx['6583'], fOII=SB__lyx['3727'], tau_V=tau_V_neb__yx)
                    mask = np.bitwise_or(mask, logO23__yx.mask)
                    mask = np.bitwise_or(mask, logN2O2__yx.mask)
                    mlogO23__yx, mlogN2O2__yx = ma_mask_xyz(logO23__yx, logN2O2__yx, mask=mask)
                    OH_O23__yx = np.ma.masked_all((N_y, N_x))
                    OH_O23__yx[~mask] = OH_O23(logO23_ratio=mlogO23__yx.compressed(), logN2O2_ratio=mlogN2O2__yx.compressed())
                    OH_O23__yx[~np.isfinite(OH_O23__yx)] = np.ma.masked
                    # HII cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logO23_cumsum_HII__r = logO23(fOII=SB_cumsum_HII__lr['3727'], fHb=SB_cumsum_HII__lr['4861'], f5007=SB_cumsum_HII__lr['5007'], tau_V=tau_V_neb_cumsum_HII__r)
                    logN2O2_cumsum_HII__r = logN2O2(fNII=SB_cumsum_HII__lr['6583'], fOII=SB_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
                    mask = np.bitwise_or(mask, logO23_cumsum_HII__r.mask)
                    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
                    mlogO23_cumsum_HII__r, mlogN2O2_cumsum_HII__r = ma_mask_xyz(logO23_cumsum_HII__r, logN2O2_cumsum_HII__r, mask=mask)
                    OH_O23_cumsum_HII__r = np.ma.masked_all((N_R_bins))
                    OH_O23_cumsum_HII__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum_HII__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
                    OH_O23_cumsum_HII__r[~np.isfinite(OH_O23_cumsum_HII__r)] = np.ma.masked
                    # total cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logO23_cumsum__r = logO23(fOII=SB_cumsum__lr['3727'], fHb=SB_cumsum__lr['4861'], f5007=SB_cumsum__lr['5007'], tau_V=tau_V_neb_cumsum__r)
                    logN2O2_cumsum__r = logN2O2(fNII=SB_cumsum__lr['6583'], fOII=SB_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
                    mask = np.bitwise_or(mask, logO23_cumsum__r.mask)
                    mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
                    mlogO23_cumsum__r, mlogN2O2_cumsum__r = ma_mask_xyz(logO23_cumsum__r, logN2O2_cumsum__r, mask=mask)
                    OH_O23_cumsum__r = np.ma.masked_all((N_R_bins))
                    OH_O23_cumsum__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
                    OH_O23_cumsum__r[~np.isfinite(OH_O23_cumsum__r)] = np.ma.masked
                    #############################
                    # N2O2 Dopita et al. (2013) #
                    #############################
                    mask = np.zeros((N_y, N_x), dtype='bool')
                    mask = np.bitwise_or(mask, SB__lyx['3727'].mask)
                    mask = np.bitwise_or(mask, SB__lyx['6583'].mask)
                    logN2O2__yx = logN2O2(fNII=SB__lyx['6583'], fOII=SB__lyx['3727'], tau_V=tau_V_neb__yx)
                    mask = np.bitwise_or(mask, logN2O2__yx.mask)
                    mlogN2O2__yx = np.ma.masked_array(logN2O2__yx, mask=mask)
                    OH_N2O2__yx = np.ma.masked_all((N_y, N_x))
                    OH_N2O2__yx[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2__yx.compressed())
                    OH_N2O2__yx[~np.isfinite(OH_N2O2__yx)] = np.ma.masked
                    # HII cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logN2O2_cumsum_HII__r = logN2O2(fNII=SB_cumsum_HII__lr['6583'], fOII=SB_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
                    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
                    mlogN2O2_cumsum_HII__r = np.ma.masked_array(logN2O2_cumsum_HII__r, mask=mask)
                    OH_N2O2_cumsum_HII__r = np.ma.masked_all((N_R_bins))
                    OH_N2O2_cumsum_HII__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
                    # total cumulative O/H
                    mask = np.zeros((N_R_bins), dtype='bool')
                    logN2O2_cumsum__r = logN2O2(fNII=SB_cumsum__lr['6583'], fOII=SB_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
                    mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
                    mlogN2O2_cumsum__r = np.ma.masked_array(logN2O2_cumsum__r, mask=mask)
                    OH_N2O2_cumsum__r = np.ma.masked_all((N_R_bins))
                    OH_N2O2_cumsum__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
                    #####################
                    y = OH_N2O2__yx
                    y_cumsum = OH_N2O2_cumsum__r
                    y_cumsum_HII = OH_N2O2_cumsum_HII__r
                    sel_DIG__yx, sel_COMP__yx, sel_HII__yx = get_selections(ALL, g, sel_WHa_DIG__gyx, sel_WHa_COMP__gyx, sel_WHa_HII__gyx)

                    ax.scatter(pixelDistance_HLR__yx, y, s=5, color='silver', **dflt_kw_scatter)
                    ax.scatter(pixelDistance_HLR__yx[sel_HII__yx], y[sel_HII__yx], s=5, color='blue', **dflt_kw_scatter)
                    xm, ym = ma_mask_xyz(pixelDistance_HLR__yx, y, mask=None)
                    ax.plot(R_bin_center__r, y_cumsum, linewidth=2, linestyle='-', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.plot(R_bin_center__r, y_cumsum_HII, linewidth=2, linestyle='-', marker='*', markeredgewidth=0, markeredgecolor=colors_lines_DIG_COMP_HII[2], c='cyan', markersize=5)

                    min_npts = 10
                    yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
                    txt = '%s' % g
                    plot_text_ax(ax, txt, 0.02, 0.98, 16, 'top', 'left', color='k')
                    txt = r'%s: $%.2f$' % (v_ver['label'], ALL.get_gal_prop_unique(g, sort_var))
                    plot_text_ax(ax, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
                    if npts.any():
                        sel = npts > min_npts
                        ax.plot(bin_center, yPrc[2], '--', lw=1, c='gray')
                        ax.plot(bin_center[sel], yPrc[2][sel], '--', lw=1, c='k')
                        # ax.plot(bin_center[sel], yPrc[2][sel], linestyle='', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.yaxis.set_ticks([8, 8.25, 8.5, 8.75, 9, 9.25])
                    ax.xaxis.set_ticks([0, 1, 2])
                    ax.set_xlim(distance_range)
                    ax.set_ylim(OH_range)
                    ax.grid('on')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    if g == sel_gals[iS][-1]:
                        axArr[row][0].xaxis.set_visible(True)
                        plt.setp(axArr[row][0].get_xticklabels(), visible=True)
                        axArr[row][0].set_xlabel(r'R [HLR]')
                        for i_r in xrange(0, row+1):
                            axArr[i_r][0].yaxis.set_visible(True)
                            plt.setp(axArr[i_r][0].get_yticklabels(), visible=True)
                    col += 1
                    if col == N_cols:
                        col = 0
                        row += 1
                    ax_i += 1
                if ax_i < N_axes:
                    for i in xrange(ax_i, N_axes):
                        np.ravel(axArr)[i].set_axis_off()
                f.subplots_adjust(left=0.05, bottom=0.05, wspace=0, hspace=0, right=0.95, top=0.95)
                f.savefig('%s_%s_p%d.png' % (OH_name, k_ver, page), orientation='landscape', dpi=200)
                plt.close(f)
                page += 1


# def fig4_cumulative_profile(ALL, gals=None):
#     sel_WHa_DIG__gz = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
#     sel_WHa_COMP__gz = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
#     sel_WHa_HII__gz = (ALL.W6563__z >= HII_WHa_threshold).filled(False)
#     sel_WHa_DIG__gyx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
#     sel_WHa_COMP__gyx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
#     sel_WHa_HII__gyx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)
#
#     if gals is None:
#         _, ind = np.unique(ALL.califaID__z, return_index=True)
#         gals = ALL.califaID__z[sorted(ind)]
#
#     sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')
#
#     N_gals = 0
#     for g in gals:
#         where_gals__gz = np.where(ALL.califaID__z == g)
#         if not where_gals__gz:
#             continue
#         sel_gals__gz[where_gals__gz] = True
#         N_gals += 1
#
#     if (sel_gals__gz).any():
#         OH_name = 'O3N2'
#
#         p33ba, p66ba = np.percentile(ALL.ba, [33, 66])
#
#         sel_gals_ba = [
#             np.less(ALL.ba, p33ba),
#             np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba)),
#             np.greater_equal(ALL.ba, p66ba)
#         ]
#
#         ba_labels = [
#             r'b/a < %.2f' % p33ba,
#             r'%.2f <= b/a < %.2f' % (p33ba, p66ba),
#             r'b/a >= %.2f' % p66ba,
#         ]
#
#         versions = dict(
#             Mtot=dict(var=np.ma.log10(ALL.Mtot), label=r'$\log$ M${}_\star}$'),
#             CI_9050=dict(var=ALL.CI_9050, label=r'CI${}_{50}^{90}$'),
#         )
#         for k_ver, v_ver in versions.iteritems():
#             page = 1
#             sort_var = v_ver['var']
#             for i_sel, selection in enumerate(sel_gals_ba):
#                 print k_ver, page
#                 sel_gals = np.asarray(gals)[selection]
#                 N_gals_sel = len(sel_gals)
#                 iS = np.argsort(sort_var[selection])
#                 N_rows, N_cols = 10, 12
#                 row, col = 0, 0
#                 f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
#                 f.suptitle('OH (%s) cumulative profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_name, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
#                 ax_i = 0
#                 N_axes = N_cols * N_rows
#                 for g in sel_gals[iS]:
#                     SB__lyx, HaHb_classif, cumsum_classif = cumulative_profiles(ALL, g, sel_WHa_DIG__gyx, sel_WHa_COMP__gyx, sel_WHa_HII__gyx)
#                     ax = np.ravel(axArr)[ax_i]
#                     ax = axArr[row][col]
#                     Hb_cumsum__r = cumsum_classif['total__lr']['v']['4861']
#                     O3_cumsum__r = cumsum_classif['total__lr']['v']['5007']
#                     Ha_cumsum__r = cumsum_classif['total__lr']['v']['6563']
#                     N2_cumsum__r = cumsum_classif['total__lr']['v']['6583']
#                     OH_O3N2 = 8.73 - 0.32 * np.ma.log10((O3_cumsum__r * Ha_cumsum__r)/(N2_cumsum__r * Hb_cumsum__r))
#                     Hb_cumsum_HII__r = cumsum_classif['HII__lr']['v']['4861']
#                     O3_cumsum_HII__r = cumsum_classif['HII__lr']['v']['5007']
#                     Ha_cumsum_HII__r = cumsum_classif['HII__lr']['v']['6563']
#                     N2_cumsum_HII__r = cumsum_classif['HII__lr']['v']['6583']
#                     OH_O3N2_HII = 8.73 - 0.32 * np.ma.log10((O3_cumsum_HII__r * Ha_cumsum_HII__r)/(N2_cumsum_HII__r * Hb_cumsum_HII__r))
#                     y = OH_O3N2
#                     y_HII = OH_O3N2_HII
#                     # sc = ax.scatter(zoneDistance_HLR, OH_O3N2, s=1, **dflt_kw_scatter)
#                     xm, ym = ma_mask_xyz(R_bin_center__r, y, mask=None)
#                     xm_HII, ym_HII = ma_mask_xyz(R_bin_center__r, y_HII, mask=None)
#                     txt = '%s' % g
#                     plot_text_ax(ax, txt, 0.02, 0.98, 16, 'top', 'left', color='k')
#                     txt = r'%s: $%.2f$' % (v_ver['label'], ALL.get_gal_prop_unique(g, sort_var))
#                     plot_text_ax(ax, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
#                     ax.plot(R_bin_center__r, y, '-', lw=2, c='k')
#                     ax.plot(R_bin_center__r, y_HII, '-', lw=2, c=colors_lines_DIG_COMP_HII[2])
#                     ax.yaxis.set_ticks([8, 8.25, 8.5, 8.75, 9, 9.25])
#                     ax.xaxis.set_ticks([0, 1, 2])
#                     ax.set_xlim(distance_range)
#                     ax.set_ylim(OH_range)
#                     ax.grid('on')
#                     plt.setp(ax.get_xticklabels(), visible=False)
#                     plt.setp(ax.get_yticklabels(), visible=False)
#                     if g == sel_gals[iS][-1]:
#                         axArr[row][0].xaxis.set_visible(True)
#                         plt.setp(axArr[row][0].get_xticklabels(), visible=True)
#                         axArr[row][0].set_xlabel(r'R [HLR]')
#                         for i_r in xrange(0, row+1):
#                             axArr[i_r][0].yaxis.set_visible(True)
#                             plt.setp(axArr[i_r][0].get_yticklabels(), visible=True)
#                     col += 1
#                     if col == N_cols:
#                         col = 0
#                         row += 1
#                     ax_i += 1
#                 if ax_i < N_axes:
#                     for i in xrange(ax_i, N_axes):
#                         np.ravel(axArr)[i].set_axis_off()
#                 f.subplots_adjust(left=0.05, bottom=0.05, wspace=0, hspace=0, right=0.95, top=0.95)
#                 f.savefig('%s_cumulative_%s_p%d.png' % (OH_name, k_ver, page), orientation='landscape', dpi=100)
#                 plt.close(f)
#                 page += 1

def fig5_profile(ALL, gals=None):
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
        OH_name = 'N2'
        p33ba, p66ba = np.percentile(ALL.ba, [33, 66])

        sel_gals_ba = [
            np.less(ALL.ba, p33ba),
            np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba)),
            np.greater_equal(ALL.ba, p66ba)
        ]
        ba_labels = [
            r'b/a < %.2f' % p33ba,
            r'%.2f <= b/a < %.2f' % (p33ba, p66ba),
            r'b/a >= %.2f' % p66ba,
        ]

        versions = dict(
            Mtot=dict(var=np.ma.log10(ALL.Mtot), label=r'$\log$ M${}_\star}$'),
            CI_9050=dict(var=ALL.CI_9050, label=r'CI${}_{50}^{90}$'),
        )
        for k_ver, v_ver in versions.iteritems():
            page = 1
            sort_var = v_ver['var']
            for i_sel, selection in enumerate(sel_gals_ba):
                print k_ver, page
                sel_gals = np.asarray(gals)[selection]
                N_gals_sel = len(sel_gals)
                iS = np.argsort(sort_var[selection])
                N_rows, N_cols = 10, 12
                row, col = 0, 0
                # f = plt.figure(dpi=300, figsize=(N_cols * 2, N_rows * 2))
                # from mpl_toolkits.axes_grid1 import Grid
                # grid = Grid(f, rect=111, nrows_ncols=(N_rows, N_cols), axes_pad=0.0, label_mode='all',)
                f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
                f.suptitle('OH (%s) profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_name, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
                # for i in xrange(N_rows):
                #     for j in xrange(N_cols):
                #         # axArr[i, j].axis('off')
                ax_i = 0
                N_axes = N_cols * N_rows
                for g in sel_gals[iS]:
                    ax = axArr[row][col]
                    ALL_N2Ha__z = ALL.get_gal_prop(g, ALL.f6583__z/ALL.f6563__z)
                    ALL_OH_N2Ha__z = 8.90 + 0.57 * np.ma.log10(ALL_N2Ha__z)
                    zoneDistance_HLR = ALL.get_gal_prop(g, ALL.zoneDistance_HLR)
                    sc = ax.scatter(zoneDistance_HLR, ALL_OH_N2Ha__z, s=1, **dflt_kw_scatter)
                    xm, ym = ma_mask_xyz(zoneDistance_HLR, ALL_OH_N2Ha__z, mask=None)
                    min_npts = 5
                    yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
                    txt = '%s' % g
                    plot_text_ax(ax, txt, 0.02, 0.98, 16, 'top', 'left', color='k')
                    txt = r'%s: $%.2f$' % (v_ver['label'], ALL.get_gal_prop_unique(g, sort_var))
                    plot_text_ax(ax, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
                    if npts.any():
                        sel = npts > min_npts
                        ax.plot(bin_center[sel], yPrc[2][sel], '-', lw=2, c='k')
                        ax.plot(bin_center, yPrc[2], '-', lw=1, c='gray')
                        ax.plot(bin_center[sel], yPrc[2][sel], linestyle='', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.yaxis.set_ticks([8, 8.25, 8.5, 8.75, 9, 9.25])
                    ax.xaxis.set_ticks([0, 1, 2])
                    ax.set_xlim(distance_range)
                    ax.set_ylim(OH_range)
                    ax.grid('on')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    if g == sel_gals[iS][-1]:
                        axArr[row][0].xaxis.set_visible(True)
                        plt.setp(axArr[row][0].get_xticklabels(), visible=True)
                        axArr[row][0].set_xlabel(r'R [HLR]')
                        for i_r in xrange(0, row+1):
                            axArr[i_r][0].yaxis.set_visible(True)
                            plt.setp(axArr[i_r][0].get_yticklabels(), visible=True)
                    col += 1
                    if col == N_cols:
                        col = 0
                        row += 1
                    ax_i += 1
                if ax_i < N_axes:
                    for i in xrange(ax_i, N_axes):
                        np.ravel(axArr)[i].set_axis_off()
                f.subplots_adjust(left=0.05, bottom=0.05, wspace=0, hspace=0, right=0.95, top=0.95)
                f.savefig('%s_%s_p%d.png' % (OH_name, k_ver, page), orientation='landscape', dpi=100)
                plt.close(f)
                page += 1


def fig5_cumulative_profile(ALL, gals=None):
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
        OH_name = 'N2'

        p33ba, p66ba = np.percentile(ALL.ba, [33, 66])

        sel_gals_ba = [
            np.less(ALL.ba, p33ba),
            np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba)),
            np.greater_equal(ALL.ba, p66ba)
        ]

        ba_labels = [
            r'b/a < %.2f' % p33ba,
            r'%.2f <= b/a < %.2f' % (p33ba, p66ba),
            r'b/a >= %.2f' % p66ba,
        ]

        versions = dict(
            Mtot=dict(var=np.ma.log10(ALL.Mtot), label=r'$\log$ M${}_\star}$'),
            CI_9050=dict(var=ALL.CI_9050, label=r'CI${}_{50}^{90}$'),
        )
        for k_ver, v_ver in versions.iteritems():
            page = 1
            sort_var = v_ver['var']
            for i_sel, selection in enumerate(sel_gals_ba):
                print k_ver, page
                sel_gals = np.asarray(gals)[selection]
                N_gals_sel = len(sel_gals)
                iS = np.argsort(sort_var[selection])
                N_rows, N_cols = 10, 12
                row, col = 0, 0
                f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
                f.suptitle('OH (%s) cumulative profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_name, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
                ax_i = 0
                N_axes = N_cols * N_rows
                for g in sel_gals[iS]:
                    ax = axArr[row][col]
                    Ha__z = ALL.get_gal_prop(g, ALL.f6563__z)
                    N2__z = ALL.get_gal_prop(g, ALL.f6583__z)
                    Ha_cumsum__z = Ha__z.filled(0.).cumsum()
                    N2_cumsum__z = N2__z.filled(0.).cumsum()
                    OH_N2 = 8.90 + 0.57 * np.ma.log10(N2_cumsum__z/Ha_cumsum__z)
                    y = OH_N2
                    zoneDistance_HLR = ALL.get_gal_prop(g, ALL.zoneDistance_HLR)
                    # sc = ax.scatter(zoneDistance_HLR, OH_O3N2, s=1, **dflt_kw_scatter)
                    xm, ym = ma_mask_xyz(zoneDistance_HLR, y, mask=None)
                    min_npts = 5
                    yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
                    txt = '%s' % g
                    plot_text_ax(ax, txt, 0.02, 0.98, 16, 'top', 'left', color='k')
                    txt = r'%s: $%.2f$' % (v_ver['label'], ALL.get_gal_prop_unique(g, sort_var))
                    plot_text_ax(ax, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
                    if npts.any():
                        sel = npts > min_npts
                        ax.plot(bin_center[sel], yPrc[2][sel], '-', lw=2, c='k')
                        ax.plot(bin_center, yPrc[2], '-', lw=1, c='gray')
                        ax.plot(bin_center[sel], yPrc[2][sel], linestyle='', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.yaxis.set_ticks([8, 8.25, 8.5, 8.75, 9, 9.25])
                    ax.xaxis.set_ticks([0, 1, 2])
                    ax.set_xlim(distance_range)
                    ax.set_ylim(OH_range)
                    ax.grid('on')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    if g == sel_gals[iS][-1]:
                        axArr[row][0].xaxis.set_visible(True)
                        plt.setp(axArr[row][0].get_xticklabels(), visible=True)
                        axArr[row][0].set_xlabel(r'R [HLR]')
                        for i_r in xrange(0, row+1):
                            axArr[i_r][0].yaxis.set_visible(True)
                            plt.setp(axArr[i_r][0].get_yticklabels(), visible=True)
                    col += 1
                    if col == N_cols:
                        col = 0
                        row += 1
                    ax_i += 1
                if ax_i < N_axes:
                    for i in xrange(ax_i, N_axes):
                        np.ravel(axArr)[i].set_axis_off()
                f.subplots_adjust(left=0.05, bottom=0.05, wspace=0, hspace=0, right=0.95, top=0.95)
                f.savefig('%s_cumulative_%s_p%d.png' % (OH_name, k_ver, page), orientation='landscape', dpi=100)
                plt.close(f)
                page += 1


def fig6_profile(ALL, gals=None):
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
        OH_name = 'O23'
        p33ba, p66ba = np.percentile(ALL.ba, [33, 66])

        sel_gals_ba = [
            np.less(ALL.ba, p33ba),
            np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba)),
            np.greater_equal(ALL.ba, p66ba)
        ]
        ba_labels = [
            r'b/a < %.2f' % p33ba,
            r'%.2f <= b/a < %.2f' % (p33ba, p66ba),
            r'b/a >= %.2f' % p66ba,
        ]

        versions = dict(
            Mtot=dict(var=np.ma.log10(ALL.Mtot), label=r'$\log$ M${}_\star}$'),
            CI_9050=dict(var=ALL.CI_9050, label=r'CI${}_{50}^{90}$'),
        )
        for k_ver, v_ver in versions.iteritems():
            page = 1
            sort_var = v_ver['var']
            for i_sel, selection in enumerate(sel_gals_ba):
                print k_ver, page
                sel_gals = np.asarray(gals)[selection]
                N_gals_sel = len(sel_gals)
                iS = np.argsort(sort_var[selection])
                N_rows, N_cols = 10, 12
                row, col = 0, 0
                # f = plt.figure(dpi=300, figsize=(N_cols * 2, N_rows * 2))
                # from mpl_toolkits.axes_grid1 import Grid
                # grid = Grid(f, rect=111, nrows_ncols=(N_rows, N_cols), axes_pad=0.0, label_mode='all',)
                f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
                f.suptitle('OH (%s) profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_name, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
                # for i in xrange(N_rows):
                #     for j in xrange(N_cols):
                #         # axArr[i, j].axis('off')
                ax_i = 0
                N_axes = N_cols * N_rows
                for g in sel_gals[iS]:
                    ax = axArr[row][col]
                    f3727__z = ALL.get_gal_prop(g, ALL.f3727__z)
                    f4861__z = ALL.get_gal_prop(g, ALL.f4861__z)
                    f5007__z = ALL.get_gal_prop(g, ALL.f5007__z)
                    f6563__z = ALL.get_gal_prop(g, ALL.f6563__z)
                    f6583__z = ALL.get_gal_prop(g, ALL.f6583__z)
                    N_zones = ALL.get_gal_prop_unique(g, ALL.N_zone)
                    tau_V_neb = lambda Ha, Hb: (1./qHbHa) * np.ma.log(Ha/Hb/2.86)
                    tau_V_neb__z = tau_V_neb(f6563__z, f4861__z)
                    tau_V_neb__z = np.where(np.less(tau_V_neb__z, 0), 0, tau_V_neb__z)
                    logO23__z = logO23(fOII=f3727__z, fHb=f4861__z, f5007=f5007__z, tau_V=tau_V_neb__z)
                    logN2O2__z = logN2O2(fNII=f6583__z, fOII=f3727__z, tau_V=tau_V_neb__z)
                    mask = np.zeros((N_zones), dtype='bool')
                    mask = np.bitwise_or(mask, f3727__z.mask)
                    mask = np.bitwise_or(mask, f4861__z.mask)
                    mask = np.bitwise_or(mask, f5007__z.mask)
                    mask = np.bitwise_or(mask, f6583__z.mask)
                    mask = np.bitwise_or(mask, logO23__z.mask)
                    mask = np.bitwise_or(mask, logN2O2__z.mask)
                    mlogO23__z, mlogN2O2__z = ma_mask_xyz(logO23__z, logN2O2__z, mask=mask)
                    OH_O23__z = np.ma.masked_all((N_zones))
                    OH_O23__z[~mask] = OH_O23(logO23_ratio=mlogO23__z.compressed(), logN2O2_ratio=mlogN2O2__z.compressed())
                    OH_O23__z[~np.isfinite(OH_O23__z)] = np.ma.masked
                    zoneDistance_HLR = ALL.get_gal_prop(g, ALL.zoneDistance_HLR)
                    sc = ax.scatter(zoneDistance_HLR, OH_O23__z, s=1, **dflt_kw_scatter)
                    xm, ym = ma_mask_xyz(zoneDistance_HLR, OH_O23__z, mask=None)
                    min_npts = 5
                    yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
                    txt = '%s' % g
                    plot_text_ax(ax, txt, 0.02, 0.98, 16, 'top', 'left', color='k')
                    txt = r'%s: $%.2f$' % (v_ver['label'], ALL.get_gal_prop_unique(g, sort_var))
                    plot_text_ax(ax, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
                    if npts.any():
                        sel = npts > min_npts
                        ax.plot(bin_center[sel], yPrc[2][sel], '-', lw=2, c='k')
                        ax.plot(bin_center, yPrc[2], '-', lw=1, c='gray')
                        ax.plot(bin_center[sel], yPrc[2][sel], linestyle='', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.yaxis.set_ticks([8, 8.25, 8.5, 8.75, 9, 9.25])
                    ax.xaxis.set_ticks([0, 1, 2])
                    ax.set_xlim(distance_range)
                    ax.set_ylim(OH_range)
                    ax.grid('on')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    if g == sel_gals[iS][-1]:
                        axArr[row][0].xaxis.set_visible(True)
                        plt.setp(axArr[row][0].get_xticklabels(), visible=True)
                        axArr[row][0].set_xlabel(r'R [HLR]')
                        for i_r in xrange(0, row+1):
                            axArr[i_r][0].yaxis.set_visible(True)
                            plt.setp(axArr[i_r][0].get_yticklabels(), visible=True)
                    col += 1
                    if col == N_cols:
                        col = 0
                        row += 1
                    ax_i += 1
                if ax_i < N_axes:
                    for i in xrange(ax_i, N_axes):
                        np.ravel(axArr)[i].set_axis_off()
                f.subplots_adjust(left=0.05, bottom=0.05, wspace=0, hspace=0, right=0.95, top=0.95)
                f.savefig('%s_%s_p%d.png' % (OH_name, k_ver, page), orientation='landscape', dpi=100)
                plt.close(f)
                page += 1


def fig6_cumulative_profile(ALL, gals=None):
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
        OH_name = 'O23'

        p33ba, p66ba = np.percentile(ALL.ba, [33, 66])

        sel_gals_ba = [
            np.less(ALL.ba, p33ba),
            np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba)),
            np.greater_equal(ALL.ba, p66ba)
        ]

        ba_labels = [
            r'b/a < %.2f' % p33ba,
            r'%.2f <= b/a < %.2f' % (p33ba, p66ba),
            r'b/a >= %.2f' % p66ba,
        ]

        versions = dict(
            Mtot=dict(var=np.ma.log10(ALL.Mtot), label=r'$\log$ M${}_\star}$'),
            CI_9050=dict(var=ALL.CI_9050, label=r'CI${}_{50}^{90}$'),
        )
        for k_ver, v_ver in versions.iteritems():
            page = 1
            sort_var = v_ver['var']
            for i_sel, selection in enumerate(sel_gals_ba):
                print k_ver, page
                sel_gals = np.asarray(gals)[selection]
                N_gals_sel = len(sel_gals)
                iS = np.argsort(sort_var[selection])
                N_rows, N_cols = 10, 12
                row, col = 0, 0
                f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
                f.suptitle('OH (%s) cumulative profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_name, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
                ax_i = 0
                N_axes = N_cols * N_rows
                for g in sel_gals[iS]:
                    ax = axArr[row][col]
                    f3727__z = ALL.get_gal_prop(g, ALL.f3727__z)
                    f4861__z = ALL.get_gal_prop(g, ALL.f4861__z)
                    f5007__z = ALL.get_gal_prop(g, ALL.f5007__z)
                    f6563__z = ALL.get_gal_prop(g, ALL.f6563__z)
                    f6583__z = ALL.get_gal_prop(g, ALL.f6583__z)
                    N_zones = ALL.get_gal_prop_unique(g, ALL.N_zone)
                    tau_V_neb = lambda Ha, Hb: (1./qHbHa) * np.ma.log(Ha/Hb/2.86)
                    tau_V_neb__z = tau_V_neb(f6563__z.filled(0.).cumsum(), f4861__z.filled(0.).cumsum())
                    tau_V_neb__z = np.where(np.less(tau_V_neb__z, 0), 0, tau_V_neb__z)
                    logO23__z = logO23(fOII=f3727__z.filled(0.).cumsum(), fHb=f4861__z.filled(0.).cumsum(), f5007=f5007__z.filled(0.).cumsum(), tau_V=tau_V_neb__z)
                    logN2O2__z = logN2O2(fNII=f6583__z.filled(0.).cumsum(), fOII=f3727__z.filled(0.).cumsum(), tau_V=tau_V_neb__z)
                    mask = np.zeros((N_zones), dtype='bool')
                    mask = np.bitwise_or(mask, f3727__z.mask)
                    mask = np.bitwise_or(mask, f4861__z.mask)
                    mask = np.bitwise_or(mask, f5007__z.mask)
                    mask = np.bitwise_or(mask, f6583__z.mask)
                    mask = np.bitwise_or(mask, logO23__z.mask)
                    mask = np.bitwise_or(mask, logN2O2__z.mask)
                    mlogO23__z, mlogN2O2__z = ma_mask_xyz(logO23__z, logN2O2__z, mask=mask)
                    OH_O23__z = np.ma.masked_all((N_zones))
                    OH_O23__z[~mask] = OH_O23(logO23_ratio=mlogO23__z.compressed(), logN2O2_ratio=mlogN2O2__z.compressed())
                    OH_O23__z[~np.isfinite(OH_O23__z)] = np.ma.masked
                    y = OH_O23__z
                    zoneDistance_HLR = ALL.get_gal_prop(g, ALL.zoneDistance_HLR)
                    # sc = ax.scatter(zoneDistance_HLR, OH_O3N2, s=1, **dflt_kw_scatter)
                    xm, ym = ma_mask_xyz(zoneDistance_HLR, y, mask=None)
                    min_npts = 5
                    yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
                    txt = '%s' % g
                    plot_text_ax(ax, txt, 0.02, 0.98, 16, 'top', 'left', color='k')
                    txt = r'%s: $%.2f$' % (v_ver['label'], ALL.get_gal_prop_unique(g, sort_var))
                    plot_text_ax(ax, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
                    if npts.any():
                        sel = npts > min_npts
                        ax.plot(bin_center[sel], yPrc[2][sel], '-', lw=2, c='k')
                        ax.plot(bin_center, yPrc[2], '-', lw=1, c='gray')
                        ax.plot(bin_center[sel], yPrc[2][sel], linestyle='', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.yaxis.set_ticks([8, 8.25, 8.5, 8.75, 9, 9.25])
                    ax.xaxis.set_ticks([0, 1, 2])
                    ax.set_xlim(distance_range)
                    ax.set_ylim(OH_range)
                    ax.grid('on')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    if g == sel_gals[iS][-1]:
                        axArr[row][0].xaxis.set_visible(True)
                        plt.setp(axArr[row][0].get_xticklabels(), visible=True)
                        axArr[row][0].set_xlabel(r'R [HLR]')
                        for i_r in xrange(0, row+1):
                            axArr[i_r][0].yaxis.set_visible(True)
                            plt.setp(axArr[i_r][0].get_yticklabels(), visible=True)
                    col += 1
                    if col == N_cols:
                        col = 0
                        row += 1
                    ax_i += 1
                if ax_i < N_axes:
                    for i in xrange(ax_i, N_axes):
                        np.ravel(axArr)[i].set_axis_off()
                f.subplots_adjust(left=0.05, bottom=0.05, wspace=0, hspace=0, right=0.95, top=0.95)
                f.savefig('%s_cumulative_%s_p%d.png' % (OH_name, k_ver, page), orientation='landscape', dpi=100)
                plt.close(f)
                page += 1


def fig7_profile(ALL, gals=None):
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
        OH_name = 'N2O2'
        p33ba, p66ba = np.percentile(ALL.ba, [33, 66])

        sel_gals_ba = [
            np.less(ALL.ba, p33ba),
            np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba)),
            np.greater_equal(ALL.ba, p66ba)
        ]
        ba_labels = [
            r'b/a < %.2f' % p33ba,
            r'%.2f <= b/a < %.2f' % (p33ba, p66ba),
            r'b/a >= %.2f' % p66ba,
        ]

        versions = dict(
            Mtot=dict(var=np.ma.log10(ALL.Mtot), label=r'$\log$ M${}_\star}$'),
            CI_9050=dict(var=ALL.CI_9050, label=r'CI${}_{50}^{90}$'),
        )
        for k_ver, v_ver in versions.iteritems():
            page = 1
            sort_var = v_ver['var']
            for i_sel, selection in enumerate(sel_gals_ba):
                print k_ver, page
                sel_gals = np.asarray(gals)[selection]
                N_gals_sel = len(sel_gals)
                iS = np.argsort(sort_var[selection])
                N_rows, N_cols = 10, 12
                row, col = 0, 0
                # f = plt.figure(dpi=300, figsize=(N_cols * 2, N_rows * 2))
                # from mpl_toolkits.axes_grid1 import Grid
                # grid = Grid(f, rect=111, nrows_ncols=(N_rows, N_cols), axes_pad=0.0, label_mode='all',)
                f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
                f.suptitle('OH (%s) profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_name, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
                # for i in xrange(N_rows):
                #     for j in xrange(N_cols):
                #         # axArr[i, j].axis('off')
                ax_i = 0
                N_axes = N_cols * N_rows
                for g in sel_gals[iS]:
                    ax = axArr[row][col]
                    f3727__z = ALL.get_gal_prop(g, ALL.f3727__z)
                    f4861__z = ALL.get_gal_prop(g, ALL.f4861__z)
                    f6563__z = ALL.get_gal_prop(g, ALL.f6563__z)
                    f6583__z = ALL.get_gal_prop(g, ALL.f6583__z)
                    N_zones = ALL.get_gal_prop_unique(g, ALL.N_zone)
                    tau_V_neb = lambda Ha, Hb: (1./qHbHa) * np.ma.log(Ha/Hb/2.86)
                    tau_V_neb__z = tau_V_neb(f6563__z, f4861__z)
                    tau_V_neb__z = np.where(np.less(tau_V_neb__z, 0), 0, tau_V_neb__z)
                    logN2O2__z = logN2O2(fNII=f6583__z, fOII=f3727__z, tau_V=tau_V_neb__z)
                    mask = np.zeros((N_zones), dtype='bool')
                    mask = np.bitwise_or(mask, f3727__z.mask)
                    mask = np.bitwise_or(mask, f6583__z.mask)
                    mask = np.bitwise_or(mask, logN2O2__z.mask)
                    mlogN2O2__z = np.ma.masked_array(logN2O2__z, mask=mask)
                    OH_N2O2__z = np.ma.masked_all((N_zones))
                    OH_N2O2__z[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2__z.compressed())
                    OH_N2O2__z[~np.isfinite(OH_N2O2__z)] = np.ma.masked
                    zoneDistance_HLR = ALL.get_gal_prop(g, ALL.zoneDistance_HLR)
                    sc = ax.scatter(zoneDistance_HLR, OH_N2O2__z, s=1, **dflt_kw_scatter)
                    xm, ym = ma_mask_xyz(zoneDistance_HLR, OH_N2O2__z, mask=None)
                    min_npts = 5
                    yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
                    txt = '%s' % g
                    plot_text_ax(ax, txt, 0.02, 0.98, 16, 'top', 'left', color='k')
                    txt = r'%s: $%.2f$' % (v_ver['label'], ALL.get_gal_prop_unique(g, sort_var))
                    plot_text_ax(ax, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
                    if npts.any():
                        sel = npts > min_npts
                        ax.plot(bin_center[sel], yPrc[2][sel], '-', lw=2, c='k')
                        ax.plot(bin_center, yPrc[2], '-', lw=1, c='gray')
                        ax.plot(bin_center[sel], yPrc[2][sel], linestyle='', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.yaxis.set_ticks([8, 8.25, 8.5, 8.75, 9, 9.25])
                    ax.xaxis.set_ticks([0, 1, 2])
                    ax.set_xlim(distance_range)
                    ax.set_ylim(OH_range)
                    ax.grid('on')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    if g == sel_gals[iS][-1]:
                        axArr[row][0].xaxis.set_visible(True)
                        plt.setp(axArr[row][0].get_xticklabels(), visible=True)
                        axArr[row][0].set_xlabel(r'R [HLR]')
                        for i_r in xrange(0, row+1):
                            axArr[i_r][0].yaxis.set_visible(True)
                            plt.setp(axArr[i_r][0].get_yticklabels(), visible=True)
                    col += 1
                    if col == N_cols:
                        col = 0
                        row += 1
                    ax_i += 1
                if ax_i < N_axes:
                    for i in xrange(ax_i, N_axes):
                        np.ravel(axArr)[i].set_axis_off()
                f.subplots_adjust(left=0.05, bottom=0.05, wspace=0, hspace=0, right=0.95, top=0.95)
                f.savefig('%s_%s_p%d.png' % (OH_name, k_ver, page), orientation='landscape', dpi=100)
                plt.close(f)
                page += 1


def fig7_cumulative_profile(ALL, gals=None):
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
        OH_name = 'N2O2'

        p33ba, p66ba = np.percentile(ALL.ba, [33, 66])

        sel_gals_ba = [
            np.less(ALL.ba, p33ba),
            np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba)),
            np.greater_equal(ALL.ba, p66ba)
        ]

        ba_labels = [
            r'b/a < %.2f' % p33ba,
            r'%.2f <= b/a < %.2f' % (p33ba, p66ba),
            r'b/a >= %.2f' % p66ba,
        ]

        versions = dict(
            Mtot=dict(var=np.ma.log10(ALL.Mtot), label=r'$\log$ M${}_\star}$'),
            CI_9050=dict(var=ALL.CI_9050, label=r'CI${}_{50}^{90}$'),
        )
        for k_ver, v_ver in versions.iteritems():
            page = 1
            sort_var = v_ver['var']
            for i_sel, selection in enumerate(sel_gals_ba):
                print k_ver, page
                sel_gals = np.asarray(gals)[selection]
                N_gals_sel = len(sel_gals)
                iS = np.argsort(sort_var[selection])
                N_rows, N_cols = 10, 12
                row, col = 0, 0
                f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
                f.suptitle('OH (%s) cumulative profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_name, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
                ax_i = 0
                N_axes = N_cols * N_rows
                for g in sel_gals[iS]:
                    ax = axArr[row][col]
                    f3727__z = ALL.get_gal_prop(g, ALL.f3727__z)
                    f4861__z = ALL.get_gal_prop(g, ALL.f4861__z)
                    f6563__z = ALL.get_gal_prop(g, ALL.f6563__z)
                    f6583__z = ALL.get_gal_prop(g, ALL.f6583__z)
                    N_zones = ALL.get_gal_prop_unique(g, ALL.N_zone)
                    tau_V_neb = lambda Ha, Hb: (1./qHbHa) * np.ma.log(Ha/Hb/2.86)
                    tau_V_neb__z = tau_V_neb(f6563__z.filled(0.).cumsum(), f4861__z.filled(0.).cumsum())
                    tau_V_neb__z = np.where(np.less(tau_V_neb__z, 0), 0, tau_V_neb__z)
                    logN2O2__z = logN2O2(fNII=f6583__z.filled(0.).cumsum(), fOII=f3727__z.filled(0.).cumsum(), tau_V=tau_V_neb__z)
                    mask = np.zeros((N_zones), dtype='bool')
                    mask = np.bitwise_or(mask, f3727__z.mask)
                    mask = np.bitwise_or(mask, f6583__z.mask)
                    mask = np.bitwise_or(mask, logN2O2__z.mask)
                    mlogN2O2__z = np.ma.masked_array(logN2O2__z, mask=mask)
                    OH_N2O2__z = np.ma.masked_all((N_zones))
                    OH_N2O2__z[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2__z.compressed())
                    OH_N2O2__z[~np.isfinite(OH_N2O2__z)] = np.ma.masked
                    y = OH_N2O2__z
                    zoneDistance_HLR = ALL.get_gal_prop(g, ALL.zoneDistance_HLR)
                    # sc = ax.scatter(zoneDistance_HLR, OH_O3N2, s=1, **dflt_kw_scatter)
                    xm, ym = ma_mask_xyz(zoneDistance_HLR, y, mask=None)
                    min_npts = 5
                    yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
                    txt = '%s' % g
                    plot_text_ax(ax, txt, 0.02, 0.98, 16, 'top', 'left', color='k')
                    txt = r'%s: $%.2f$' % (v_ver['label'], ALL.get_gal_prop_unique(g, sort_var))
                    plot_text_ax(ax, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
                    if npts.any():
                        sel = npts > min_npts
                        ax.plot(bin_center[sel], yPrc[2][sel], '-', lw=2, c='k')
                        ax.plot(bin_center, yPrc[2], '-', lw=1, c='gray')
                        ax.plot(bin_center[sel], yPrc[2][sel], linestyle='', marker='*', markeredgewidth=0, markeredgecolor='k', c='k', markersize=5)
                    ax.yaxis.set_ticks([8, 8.25, 8.5, 8.75, 9, 9.25])
                    ax.xaxis.set_ticks([0, 1, 2])
                    ax.set_xlim(distance_range)
                    ax.set_ylim(OH_range)
                    ax.grid('on')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    if g == sel_gals[iS][-1]:
                        axArr[row][0].xaxis.set_visible(True)
                        plt.setp(axArr[row][0].get_xticklabels(), visible=True)
                        axArr[row][0].set_xlabel(r'R [HLR]')
                        for i_r in xrange(0, row+1):
                            axArr[i_r][0].yaxis.set_visible(True)
                            plt.setp(axArr[i_r][0].get_yticklabels(), visible=True)
                    col += 1
                    if col == N_cols:
                        col = 0
                        row += 1
                    ax_i += 1
                if ax_i < N_axes:
                    for i in xrange(ax_i, N_axes):
                        np.ravel(axArr)[i].set_axis_off()
                f.subplots_adjust(left=0.05, bottom=0.05, wspace=0, hspace=0, right=0.95, top=0.95)
                f.savefig('%s_cumulative_%s_p%d.png' % (OH_name, k_ver, page), orientation='landscape', dpi=100)
                plt.close(f)
                page += 1


if __name__ == '__main__':
    main(sys.argv)
