import sys
import numpy as np
import matplotlib as mpl
from pytu.lines import Lines
from pytu.objects import runstats
from matplotlib import pyplot as plt
from pytu.functions import ma_mask_xyz, debug_var
from pycasso.util import radialProfile
from scipy.interpolate import interp1d
from pystarlight.util.constants import L_sun
from pystarlight.util.redenninglaws import calc_redlaw
from CALIFAUtils.scripts import get_NEDName_by_CALIFAID
from CALIFAUtils.objects import stack_gals, CALIFAPaths
from mpl_toolkits.axes_grid1 import make_axes_locatable
from CALIFAUtils.plots import DrawHLRCircleInSDSSImage, DrawHLRCircle
from pytu.plots import cmap_discrete, plot_text_ax, density_contour


mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'

debug = False
# CCM reddening law
q = calc_redlaw([4861, 6563], R_V=3.1, redlaw='CCM')
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

    Fig 5.
        Igual a Fig 4 mas usando O/H (N2)

    Fig 6.
        Igual a Fig 4 mas usando O/H (O23)

    Fig 7.
        Igual a Fig 4 mas usando O/H (N2O2)
    """
    figs4567(ALL, gals)


def figs4567(ALL, gals=None):
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
        [r'CI > %.2f - ba < %.2f' % (p66CI, p33ba), r'CI > %.2f - %.2f $\leq$ ba < %.2f' % (p66CI, p33ba, p66ba), r'CI > %.2f - ba $\geq$ %.2f' % (p66CI, p66ba)],
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
            cb.set_label(r'$\log$ M${}_{gal}^{tot}$ [M${}_\odot$]')
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
    # log(O/H) + 12 = log [1.54020 + 1.26602 R + 0.167977 R2] + 8.93, R=log [Nii]/[Oii]
    # x = np.linspace(7, 9.5, 1000) - 8.93
    # p = [0.167977, 1.26602, 1.54020]
    # interp = interp1d(np.log10(np.polyval(p, np.linspace(-2, 1, 1000))), x, kind='linear', bounds_error=False)
    # return interp(logN2O2_ratio) + 8.93


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
        tau_V_neb__yx = np.log(HaHb__yx / 2.86) / (q[0] - q[1])
        HaHb_cumsum__r = SB_sum__lr['6563'].filled(0.).cumsum()/SB_sum__lr['4861'].filled(0.).cumsum()
        tau_V_neb_cumsum__r = np.log(HaHb_cumsum__r / 2.86) / (q[0] - q[1])
        HaHb_sum_HII__r = SB_sum_HII__lr['6563'].filled(0.).cumsum()/SB_sum_HII__lr['4861'].filled(0.).cumsum()
        tau_V_neb_cumsum_HII__r = np.log(HaHb_sum_HII__r / 2.86) / (q[0] - q[1])

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
        # # AXIS 1
        # OH__yx = OH_O3N2__yx
        # OH_label = 'O3N2'
        # ax = ax1
        # plot_OH(ax, pixelDistance_HLR__yx, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range)
        # plot_text_ax(ax, 'a)', 0.02, 0.98, 16, 'top', 'left', 'k')
        # # AXIS 2
        # OH_cumsum__r = OH_O3N2_cumsum__r
        # OH_cumsum_HII__r = OH_O3N2_cumsum_HII__r
        # ax = ax2
        # plot_cumulative_OH(ax, R_bin_center__r, OH_cumsum__r, OH_cumsum_HII__r, OH_label, OH_range, distance_range)
        # plot_text_ax(ax, 'b)', 0.02, 0.98, 16, 'top', 'left', 'k')
        # # AXIS 3
        # OH__yx = OH_N2Ha__yx
        # OH_label = 'N2'
        # ax = ax3
        # plot_OH(ax, pixelDistance_HLR__yx, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range)
        # plot_text_ax(ax, 'c)', 0.02, 0.98, 16, 'top', 'left', 'k')
        # # AXIS 4
        # OH_cumsum__r = OH_N2Ha_cumsum__r
        # OH_cumsum_HII__r = OH_N2Ha_cumsum_HII__r
        # ax = ax4
        # plot_cumulative_OH(ax, R_bin_center__r, OH_cumsum__r, OH_cumsum_HII__r, OH_label, OH_range, distance_range)
        # plot_text_ax(ax, 'd)', 0.02, 0.98, 16, 'top', 'left', 'k')
        # # AXIS 5
        # OH__yx = OH_O23__yx
        # OH_label = 'O23'
        # ax = ax5
        # plot_OH(ax, pixelDistance_HLR__yx, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range)
        # plot_text_ax(ax, 'e)', 0.02, 0.98, 16, 'top', 'left', 'k')
        # # AXIS 6
        # OH_cumsum__r = OH_O23_cumsum__r
        # OH_cumsum_HII__r = OH_O23_cumsum_HII__r
        # ax = ax6
        # plot_cumulative_OH(ax, R_bin_center__r, OH_cumsum__r, OH_cumsum_HII__r, OH_label, OH_range, distance_range)
        # plot_text_ax(ax, 'f)', 0.02, 0.98, 16, 'top', 'left', 'k')
        # # AXIS 7
        # OH__yx = OH_N2O2__yx
        # OH_label = 'N2O2'
        # ax = ax7
        # plot_OH(ax, pixelDistance_HLR__yx, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range)
        # plot_text_ax(ax, 'g)', 0.02, 0.98, 16, 'top', 'left', 'k')
        # # AXIS 8
        # OH_cumsum__r = OH_N2O2_cumsum__r
        # OH_cumsum_HII__r = OH_N2O2_cumsum_HII__r
        # ax = ax8
        # plot_cumulative_OH(ax, R_bin_center__r, OH_cumsum__r, OH_cumsum_HII__r, OH_label, OH_range, distance_range)
        # plot_text_ax(ax, 'h)', 0.02, 0.98, 16, 'top', 'left', 'k')
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('%s-fig2.png' % califaID)
        plt.close(f)


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
        SB__z = {'%s' % L: ALL.get_gal_prop(califaID, 'SB%s__z' % L) for L in lines}
        f__z = {'%s' % L: ALL.get_gal_prop(califaID, 'f%s__z' % L) for L in lines}
        W6563__yx = ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x)
        W6563__z = ALL.get_gal_prop(califaID, ALL.W6563__z)

        N_cols = 3
        N_rows = 4
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        cmap = cmap_discrete()
        ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = axArr
        f.suptitle(r'%s - %s: %d pixels (%d zones)' % (califaID, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))

        # AXIS 1, 2, 3
        # 1
        ax1.set_axis_off()
        # 2
        galimg = plt.imread(P.get_image_file(califaID))[::-1, :, :]
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.imshow(galimg, origin='lower', aspect='equal')
        DrawHLRCircleInSDSSImage(ax2, HLR_pix, pa, ba, lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        txt = '%s' % califaID
        plot_text_ax(ax2, txt, 0.02, 0.98, 16, 'top', 'left', color='w')
        # 3
        ax3.set_axis_off()

        # AXIS 4, 5, 6
        L = Lines()
        O3Hb__yx = SB__lyx['5007']/SB__lyx['4861']
        N2Ha__yx = SB__lyx['6583']/SB__lyx['6563']
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
        O3Hb__z = f__z['5007']/f__z['4861']
        N2Ha__z = f__z['6583']/f__z['6563']
        # 5
        plotBPT(ax5, np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z), z=np.ma.log10(W6563__z), vmin=logWHa_range[0], vmax=logWHa_range[1], cb_label=r'W${}_{H\alpha}\ [\AA]$', cmap='viridis_r')
        # 6
        plotBPT(ax6, np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z), z=np.ma.log10(SB__z['6563']), vmin=logSBHa_range[0], vmax=logSBHa_range[1], cb_label=r'$\Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]', cmap='viridis_r')

        # AXIS 7, 8 e 9
        sel_DIG__yx = ALL.get_gal_prop(califaID, sel_WHa_DIG__yx).reshape(N_y, N_x)
        sel_COMP__yx = ALL.get_gal_prop(califaID, sel_WHa_COMP__yx).reshape(N_y, N_x)
        sel_HII__yx = ALL.get_gal_prop(califaID, sel_WHa_HII__yx).reshape(N_y, N_x)
        map__yx = np.ma.masked_all((N_y, N_x))
        map__yx[sel_DIG__yx] = 1
        map__yx[sel_COMP__yx] = 2
        map__yx[sel_HII__yx] = 3
        # 7
        im = ax7.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax7)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(['DIG', 'COMP', 'HII'])
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
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax8.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax8.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax8.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax8.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax8.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax8.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        # 9
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__lyx['6563']))
        ax9.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax9.set_xlim(distance_range)
        ax9.set_ylim(logSBHa_range)
        ax9.set_xlabel(r'R [HLR]')
        ax9.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax9.grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax9.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax9.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax9.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax9.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax9.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax9.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)

        # AXIS 10, 11 e 12
        sel_DIG__yx = ALL.get_gal_prop(califaID, sel_Zhang_DIG__yx).reshape(N_y, N_x)
        sel_COMP__yx = ALL.get_gal_prop(califaID, sel_Zhang_COMP__yx).reshape(N_y, N_x)
        sel_HII__yx = ALL.get_gal_prop(califaID, sel_Zhang_HII__yx).reshape(N_y, N_x)
        map__yx = np.ma.masked_all((N_y, N_x))
        map__yx[sel_DIG__yx] = 1
        map__yx[sel_COMP__yx] = 2
        map__yx[sel_HII__yx] = 3
        # 10
        im = ax10.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax10)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        cb.set_ticklabels(['DIG', 'COMP', 'HII'])
        ax10.set_title(r'classif. W${}_{H\alpha}$')
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
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax11.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax11.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax11.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax11.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax11.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax11.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        # 12
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__lyx['6563']))
        ax12.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax12.set_xlim(distance_range)
        ax12.set_ylim(logSBHa_range)
        ax12.set_xlabel(r'R [HLR]')
        ax12.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax12.grid()
        if sel_DIG__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax12.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax12.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
        if sel_COMP__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax12.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax12.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
        if sel_HII__yx.astype('int').sum():
            xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
            rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
            ax12.plot(rs.xS, rs.yS, 'k--', lw=2)
            ax12.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('%s-fig1.png' % califaID)
        plt.close(f)


if __name__ == '__main__':
    main(sys.argv)
