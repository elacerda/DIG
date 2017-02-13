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
HII_Zhang_threshold = 20
SF_WHa_threshold = HII_Zhang_threshold
HII_Zhang_threshold = 1e39/L_sun
SF_Zhang_threshold = HII_Zhang_threshold
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

    try:
        sample_choice = sys.argv[3]
    except IndexError:
        sample_choice = 'S0'

    gals, sel = samples(ALL, sample_choice, gals)
    summary(ALL, sel, gals, 'SEL %s' % sample_choice)
    # sys.exit(1)

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
    fig1(ALL, sel, gals)

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
    fig2(ALL, sel, gals)

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
    """
    Fig 5.
        Igual a Fig 4 mas usando O/H (N2)
    """
    """
    Fig 6.
        Igual a Fig 4 mas usando O/H (O23)
    """
    """
    Fig 7.
        Igual a Fig 4 mas usando O/H (N2O2)
    """
    figs4567(ALL, sel, gals)


def samples(ALL, sample_choice, gals=None):
    sel_WHa = dict(
        DIG=dict(
            z=(ALL.W6563__z < DIG_WHa_threshold).filled(False),
            yx=(ALL.W6563__yx < DIG_WHa_threshold).filled(False)
        ),
        COMP=dict(
            z=np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(
                False), (ALL.W6563__z < SF_WHa_threshold).filled(False)),
            yx=np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(
                False), (ALL.W6563__yx < SF_WHa_threshold).filled(False))
        ),
        HII=dict(
            z=(ALL.W6563__z >= SF_WHa_threshold).filled(False),
            yx=(ALL.W6563__yx >= SF_WHa_threshold).filled(False)
        ),
    )

    sel_Zhang = dict(
        DIG=dict(
            z=(ALL.SB6563__z < DIG_Zhang_threshold).filled(False),
            yx=(ALL.SB6563__yx < DIG_Zhang_threshold).filled(False)
        ),
        COMP=dict(
            z=np.bitwise_and((ALL.SB6563__z >= DIG_Zhang_threshold).filled(False), (ALL.SB6563__z < SF_Zhang_threshold).filled(False)),
            yx=np.bitwise_and((ALL.SB6563__yx >= DIG_Zhang_threshold).filled(False), (ALL.SB6563__yx < SF_Zhang_threshold).filled(False))
        ),
        HII=dict(
            z=(ALL.SB6563__z >= SF_Zhang_threshold).filled(False),
            yx=(ALL.SB6563__yx >= SF_Zhang_threshold).filled(False)
        ),
    )

    L = Lines()
    N2Ha__yx, N2Ha__z = np.ma.log10(ALL.f6583__yx/ALL.f6563__yx), np.ma.log10(ALL.f6583__z/ALL.f6563__z)
    O3Hb__yx, O3Hb__z = np.ma.log10(ALL.f5007__yx/ALL.f4861__yx), np.ma.log10(ALL.f5007__z/ALL.f4861__z)
    x__z, y__z = N2Ha__z, O3Hb__z
    x__yx, y__yx = N2Ha__yx, O3Hb__yx

    sel_below_S06__z = L.belowlinebpt('S06', x__z, y__z)
    sel_below_K03__z = L.belowlinebpt('K03', x__z, y__z)
    sel_below_K01__z = L.belowlinebpt('K01', x__z, y__z)
    sel_between_S06K03__z = np.bitwise_and(sel_below_K03__z, ~sel_below_S06__z)
    sel_between_K03K01__z = np.bitwise_and(~sel_below_K03__z, sel_below_K01__z)
    sel_above_K01__z = ~sel_below_K01__z
    sel_below_S06__yx = L.belowlinebpt('S06', x__yx, y__yx)
    sel_below_K03__yx = L.belowlinebpt('K03', x__yx, y__yx)
    sel_below_K01__yx = L.belowlinebpt('K01', x__yx, y__yx)
    sel_between_S06K03__yx = np.bitwise_and(sel_below_K03__yx, ~sel_below_S06__yx)
    sel_between_K03K01__yx = np.bitwise_and(~sel_below_K03__yx, sel_below_K01__yx)
    sel_above_K01__yx = ~sel_below_K01__yx

    sel_BPT = dict(
        S06=dict(z=sel_below_S06__z, yx=sel_below_S06__yx),
        K03=dict(z=sel_below_K03__z, yx=sel_below_K03__yx),
        K01=dict(z=sel_below_K01__z, yx=sel_below_K01__yx),
        betS06K03=dict(z=sel_between_S06K03__z, yx=sel_between_S06K03__yx),
        betK03K01=dict(z=sel_between_K03K01__z, yx=sel_between_K03K01__yx),
        aboK01=dict(z=sel_above_K01__z, yx=sel_above_K01__yx),
    )

    f__lgz = {'%s' % l: getattr(ALL, 'f%s__z' % l) for l in lines}
    ef__lgz = {'%s' % l: getattr(ALL, 'ef%s__z' % l) for l in lines}
    SN__lgz = {'%s' % l: f__lgz[l]/ef__lgz[l] for l in lines}
    f__lgyx = {'%s' % l: getattr(ALL, 'f%s__yx' % l) for l in lines}
    ef__lgyx = {'%s' % l: getattr(ALL, 'ef%s__yx' % l) for l in lines}
    SN__lgyx = {'%s' % l: f__lgyx[l]/ef__lgyx[l] for l in lines}

    sel_SN = dict(
        S1=dict(
            lz={'%s' % l: np.greater(SN__lgz[l].filled(0.), 1) for l in lines},
            lyx={'%s' % l: np.greater(SN__lgyx[l].filled(0.), 1) for l in lines},
        ),
        S3=dict(
            lz={'%s' % l: np.greater(SN__lgz[l].filled(0.), 3) for l in lines},
            lyx={'%s' % l: np.greater(SN__lgyx[l].filled(0.), 3) for l in lines},
        ),
        S5=dict(
            lz={'%s' % l: np.greater(SN__lgz[l].filled(0.), 5) for l in lines},
            lyx={'%s' % l: np.greater(SN__lgyx[l].filled(0.), 5) for l in lines},
        ),
    )

    sel_SN1__gz = sel_SN['S1']['lz']
    sel_SN1__gyx = sel_SN['S1']['lyx']
    sel_SN3__gz = sel_SN['S3']['lz']
    sel_SN3__gyx = sel_SN['S3']['lyx']
    sel_SN5__gz = sel_SN['S5']['lz']
    sel_SN5__gyx = sel_SN['S5']['lyx']

    sel_SN1__gz = np.bitwise_and(sel_SN1__gz['4861'], np.bitwise_and(sel_SN1__gz['5007'], np.bitwise_and(sel_SN1__gz['6563'], sel_SN1__gz['6583'])))
    sel_SN1__gyx = np.bitwise_and(sel_SN1__gyx['4861'], np.bitwise_and(sel_SN1__gyx['5007'], np.bitwise_and(sel_SN1__gyx['6563'], sel_SN1__gyx['6583'])))
    sel_SN3__gz = np.bitwise_and(sel_SN3__gz['4861'], np.bitwise_and(sel_SN3__gz['5007'], np.bitwise_and(sel_SN3__gz['6563'], sel_SN3__gz['6583'])))
    sel_SN3__gyx = np.bitwise_and(sel_SN3__gyx['4861'], np.bitwise_and(sel_SN3__gyx['5007'], np.bitwise_and(sel_SN3__gyx['6563'], sel_SN3__gyx['6583'])))
    sel_SN5__gz = np.bitwise_and(sel_SN5__gz['4861'], np.bitwise_and(sel_SN5__gz['5007'], np.bitwise_and(sel_SN5__gz['6563'], sel_SN5__gz['6583'])))
    sel_SN5__gyx = np.bitwise_and(sel_SN5__gyx['4861'], np.bitwise_and(sel_SN5__gyx['5007'], np.bitwise_and(sel_SN5__gyx['6563'], sel_SN5__gyx['6583'])))

    sel_SN_BPT = dict(
        S0=dict(z=np.ones((ALL.califaID__z.shape), dtype='bool'), yx=np.ones((ALL.califaID__yx.shape), dtype='bool')),
        S1=dict(z=sel_SN1__gz, yx=sel_SN1__gyx),
        S3=dict(z=sel_SN3__gz, yx=sel_SN3__gyx),
        S5=dict(z=sel_SN5__gz, yx=sel_SN5__gyx),
    )

    sel = dict(
        SN=sel_SN,
        WHa=sel_WHa,
        Zhang=sel_Zhang,
        BPT=sel_BPT,
        SN_BPT=sel_SN_BPT,
    )

    return gals_sample_choice(ALL, sel, gals, sample_choice)


def gals_sample_choice(ALL, sel, gals, sample_choice):
    try:
        sample__z = sel['SN_BPT'][sample_choice]['z']
        sample__yx = sel['SN_BPT'][sample_choice]['yx']
    except KeyError:
        sample_choice = 'S0'
        sample__z = sel['SN_BPT'][sample_choice]['z']
        sample__yx = sel['SN_BPT'][sample_choice]['yx']
        print 'sample_choice %s does not exists' % sample_choice
        print 'running for S0...' % sample_choice

    sel_gals = np.zeros((ALL.ba.shape), dtype='bool')
    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')
    sel_gals__gyx = np.zeros((ALL.califaID__yx.shape), dtype='bool')
    sel_gals_sample__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')
    sel_gals_sample__gyx = np.zeros((ALL.califaID__yx.shape), dtype='bool')

    new_gals = gals[:]
    for i, g in enumerate(gals):
        tmp_sel__gz = (ALL.califaID__z == g)
        tmp_sel_sample__gz = np.bitwise_and(sample__z, ALL.califaID__z == g)
        tmp_sel_gal_sample__z = ALL.get_gal_prop(g, tmp_sel_sample__gz)
        tmp_sel__gyx = (ALL.califaID__yx == g)
        tmp_sel_sample__gyx = np.bitwise_and(sample__yx, ALL.califaID__yx == g)
        tmp_sel_gal_sample__yx = ALL.get_gal_prop(g, tmp_sel_sample__gyx)
        N_zone_notmasked = tmp_sel_gal_sample__z.astype(int).sum()
        N_pixel_notmasked = tmp_sel_gal_sample__yx.astype(int).sum()
        if N_zone_notmasked == 0 or N_pixel_notmasked == 0:
            new_gals.remove(g)
            continue
        sel_gals[i] = True
        sel_gals__gz[tmp_sel__gz] = True
        sel_gals__gyx[tmp_sel__gyx] = True
        sel_gals_sample__gz[tmp_sel_sample__gz] = True
        sel_gals_sample__gyx[tmp_sel_sample__gyx] = True

    sel['gals'] = sel_gals
    sel['gals__z'] = sel_gals__gz
    sel['gals__yx'] = sel_gals__gyx
    sel['gals_sample__z'] = sel_gals_sample__gz
    sel['gals_sample__yx'] = sel_gals_sample__gyx

    return new_gals, sel


def summary(ALL, sel, gals, mask_name):
    import datetime
    print '# Summary - %s - {:%Y%m%d %H:%M:%S}'.format(datetime.datetime.today()) % mask_name

    sel_gals__gz = sel['gals__z']
    sel_gals__gyx = sel['gals__yx']
    sel_gals_sample__gz = sel['gals_sample__z']
    sel_gals_sample__gyx = sel['gals_sample__yx']

    print 'N_gals: %d' % len(gals)
    print 'N_zones: %d' % sel_gals__gz.astype('int').sum()
    print 'N_zones (not masked): %d' % sel_gals_sample__gz.astype('int').sum()
    print 'N_pixels: %d' % sel_gals__gyx.astype('int').sum()
    print 'N_pixels (not masked): %d' % sel_gals_sample__gyx.astype('int').sum()

    print '# WHa classif:'
    tmp_sel_DIG__z = np.bitwise_and(sel['WHa']['DIG']['z'], sel_gals_sample__gz)
    tmp_sel_COMP__z = np.bitwise_and(sel['WHa']['COMP']['z'], sel_gals_sample__gz)
    tmp_sel_HII__z = np.bitwise_and(sel['WHa']['HII']['z'], sel_gals_sample__gz)
    print '\tTotal zones DIG: %d' % tmp_sel_DIG__z.astype('int').sum()
    print '\tTotal zones COMP: %d' % tmp_sel_COMP__z.astype('int').sum()
    print '\tTotal zones HII: %d' % tmp_sel_HII__z.astype('int').sum()
    for g in gals:
        tmp_sel_gal__z = ALL.get_gal_prop(g, sel_gals_sample__gz)
        tmp_sel_gal__yx = ALL.get_gal_prop(g, sel_gals_sample__gyx)
        tmp_sel_gal_DIG__z = ALL.get_gal_prop(g, tmp_sel_DIG__z)
        tmp_sel_gal_COMP__z = ALL.get_gal_prop(g, tmp_sel_COMP__z)
        tmp_sel_gal_HII__z = ALL.get_gal_prop(g, tmp_sel_HII__z)

        N_zone = len(tmp_sel_gal__z)
        N_DIG = ALL.get_gal_prop(g, sel['WHa']['DIG']['z']).astype('int').sum()
        N_DIG_notmasked = tmp_sel_gal_DIG__z.astype('int').sum()
        N_COMP = ALL.get_gal_prop(g, sel['WHa']['COMP']['z']).astype('int').sum()
        N_COMP_notmasked = tmp_sel_gal_COMP__z.astype('int').sum()
        N_HII = ALL.get_gal_prop(g, sel['WHa']['HII']['z']).astype('int').sum()
        N_HII_notmasked = tmp_sel_gal_HII__z.astype('int').sum()
        N_TOT = N_DIG+N_HII+N_COMP
        N_TOT_notmasked = N_DIG_notmasked+N_HII_notmasked+N_COMP_notmasked
        DIG_perc_tot = 0.
        COMP_perc_tot = 0.
        HII_perc_tot = 0.
        DIG_perc = 0.
        HII_perc = 0.
        if N_TOT_notmasked > 0:
            DIG_perc_tot = 100. * N_DIG_notmasked/(N_TOT_notmasked)
            COMP_perc_tot = 100. * N_COMP_notmasked/(N_TOT_notmasked)
            HII_perc_tot = 100. * N_HII_notmasked/(N_TOT_notmasked)
        if N_HII_notmasked > 0 or N_DIG_notmasked > 0:
            DIG_perc = 100. * N_DIG_notmasked/(N_DIG_notmasked+N_HII_notmasked)
            HII_perc = 100. * N_HII_notmasked/(N_DIG_notmasked+N_HII_notmasked)
        print '%s - (Nz:%d - Ntot: %d of %d) - %d DIG (of %d) (%.1f%% [%.1f%%]) - %d COMP (of %d) (%.1f%%) - %d HII (of %d) (%.1f%% [%.1f%%])' % (g, N_zone, N_TOT_notmasked, N_TOT, N_DIG_notmasked, N_DIG, DIG_perc_tot, DIG_perc, N_COMP_notmasked, N_COMP, COMP_perc_tot, N_HII_notmasked, N_HII, HII_perc_tot, HII_perc)

    print '# Zhang classif:'
    tmp_sel_DIG__z = np.bitwise_and(sel['Zhang']['DIG']['z'], sel_gals_sample__gz)
    tmp_sel_COMP__z = np.bitwise_and(sel['Zhang']['COMP']['z'], sel_gals_sample__gz)
    tmp_sel_HII__z = np.bitwise_and(sel['Zhang']['HII']['z'], sel_gals_sample__gz)
    print '\tTotal zones DIG: %d' % tmp_sel_DIG__z.astype('int').sum()
    print '\tTotal zones COMP: %d' % tmp_sel_COMP__z.astype('int').sum()
    print '\tTotal zones HII: %d' % tmp_sel_HII__z.astype('int').sum()
    for g in gals:
        tmp_sel_gal__z = ALL.get_gal_prop(g, sel_gals_sample__gz)
        tmp_sel_gal__yx = ALL.get_gal_prop(g, sel_gals_sample__gyx)
        tmp_sel_gal_DIG__z = ALL.get_gal_prop(g, tmp_sel_DIG__z)
        tmp_sel_gal_COMP__z = ALL.get_gal_prop(g, tmp_sel_COMP__z)
        tmp_sel_gal_HII__z = ALL.get_gal_prop(g, tmp_sel_HII__z)

        N_zone = len(tmp_sel_gal__z)
        N_DIG = ALL.get_gal_prop(g, sel['Zhang']['DIG']['z']).astype('int').sum()
        N_DIG_notmasked = tmp_sel_gal_DIG__z.astype('int').sum()
        N_COMP = ALL.get_gal_prop(g, sel['Zhang']['COMP']['z']).astype('int').sum()
        N_COMP_notmasked = tmp_sel_gal_COMP__z.astype('int').sum()
        N_HII = ALL.get_gal_prop(g, sel['Zhang']['HII']['z']).astype('int').sum()
        N_HII_notmasked = tmp_sel_gal_HII__z.astype('int').sum()
        N_TOT = N_DIG+N_HII+N_COMP
        N_TOT_notmasked = N_DIG_notmasked+N_HII_notmasked+N_COMP_notmasked
        DIG_perc_tot = 0.
        COMP_perc_tot = 0.
        HII_perc_tot = 0.
        DIG_perc = 0.
        HII_perc = 0.
        if N_TOT_notmasked > 0:
            DIG_perc_tot = 100. * N_DIG_notmasked/(N_TOT_notmasked)
            COMP_perc_tot = 100. * N_COMP_notmasked/(N_TOT_notmasked)
            HII_perc_tot = 100. * N_HII_notmasked/(N_TOT_notmasked)
        if N_HII_notmasked > 0 or N_DIG_notmasked > 0:
            DIG_perc = 100. * N_DIG_notmasked/(N_DIG_notmasked+N_HII_notmasked)
            HII_perc = 100. * N_HII_notmasked/(N_DIG_notmasked+N_HII_notmasked)
        print '%s - (Nz:%d - Ntot: %d of %d) - %d DIG (of %d) (%.1f%% [%.1f%%]) - %d COMP (of %d) (%.1f%%) - %d HII (of %d) (%.1f%% [%.1f%%])' % (g, N_zone, N_TOT_notmasked, N_TOT, N_DIG_notmasked, N_DIG, DIG_perc_tot, DIG_perc, N_COMP_notmasked, N_COMP, COMP_perc_tot, N_HII_notmasked, N_HII, HII_perc_tot, HII_perc)


def cumulative_profiles(ALL, sel, califaID, negative_tauV=False):
    sel_sample__gyx = sel['gals_sample__yx']

    HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
    pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
    ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
    N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
    N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
    x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
    y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
    gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)
    f__lyx = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__yx' % L).reshape(N_y, N_x), mask=~gal_sample__yx) for L in lines}
    sel_DIG__yx, sel_COMP__yx, sel_HII__yx, _ = get_selections(ALL, califaID, sel['WHa'], sel_sample__gyx)

    f_sum__lr = {}
    f_cumsum__lr = {}
    f_npts__lr = {}
    f_DIG__lyx = {}
    f_sum_DIG__lr = {}
    f_cumsum_DIG__lr = {}
    f_npts_DIG__lr = {}
    f_COMP__lyx = {}
    f_sum_COMP__lr = {}
    f_cumsum_COMP__lr = {}
    f_npts_COMP__lr = {}
    f_HII__lyx = {}
    f_sum_HII__lr = {}
    f_cumsum_HII__lr = {}
    f_npts_HII__lr = {}
    for k, v in f__lyx.iteritems():
        f_sum__lr[k], f_npts__lr[k] = radialProfile(v, R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum', True)
        f_cumsum__lr[k] = f_sum__lr[k].filled(0.).cumsum()
        f_HII__lyx[k] = np.ma.masked_array(v, mask=~sel_HII__yx, copy=True)
        f_sum_HII__lr[k], f_npts_HII__lr[k] = radialProfile(f_HII__lyx[k], R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum', True)
        f_cumsum_HII__lr[k] = f_sum_HII__lr[k].filled(0.).cumsum()
        f_COMP__lyx[k] = np.ma.masked_array(v, mask=~sel_COMP__yx, copy=True)
        f_sum_COMP__lr[k], f_npts_COMP__lr[k] = radialProfile(f_COMP__lyx[k], R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum', True)
        f_cumsum_COMP__lr[k] = f_sum_COMP__lr[k].filled(0.).cumsum()
        f_DIG__lyx[k] = np.ma.masked_array(v, mask=~sel_DIG__yx, copy=True)
        f_sum_DIG__lr[k], f_npts_DIG__lr[k] = radialProfile(f_DIG__lyx[k], R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'sum', True)
        f_cumsum_DIG__lr[k] = f_sum_DIG__lr[k].filled(0.).cumsum()

    HaHb__yx = f__lyx['6563']/f__lyx['4861']
    tau_V_neb__yx = np.log(HaHb__yx / 2.86) / qHbHa
    HaHb_cumsum__r = f_sum__lr['6563'].filled(0.).cumsum()/f_sum__lr['4861'].filled(0.).cumsum()
    tau_V_neb_cumsum__r = np.log(HaHb_cumsum__r / 2.86) / qHbHa

    HaHb_DIG__yx = f_DIG__lyx['6563']/f_DIG__lyx['4861']
    HaHb_cumsum_DIG__r = f_sum_DIG__lr['6563'].filled(0.).cumsum()/f_sum_DIG__lr['4861'].filled(0.).cumsum()
    tau_V_neb_cumsum_DIG__r = np.log(HaHb_cumsum_DIG__r / 2.86) / qHbHa
    HaHb_COMP__yx = f_COMP__lyx['6563']/f_COMP__lyx['4861']
    HaHb_cumsum_COMP__r = f_sum_COMP__lr['6563'].filled(0.).cumsum()/f_sum_COMP__lr['4861'].filled(0.).cumsum()
    tau_V_neb_cumsum_COMP__r = np.log(HaHb_cumsum_COMP__r / 2.86) / qHbHa
    HaHb_HII__yx = f_HII__lyx['6563']/f_HII__lyx['4861']
    HaHb_cumsum_HII__r = f_sum_HII__lr['6563'].filled(0.).cumsum()/f_sum_HII__lr['4861'].filled(0.).cumsum()
    tau_V_neb_cumsum_HII__r = np.log(HaHb_cumsum_HII__r / 2.86) / qHbHa

    HaHb_classif = dict(
        total=dict(HaHb__yx=HaHb__yx, tau_V_neb__yx=tau_V_neb__yx, HaHb_cumsum__r=HaHb_cumsum__r, tau_V_neb_cumsum__r=tau_V_neb_cumsum__r),
        DIG=dict(HaHb__yx=HaHb_DIG__yx, HaHb_cumsum__r=HaHb_cumsum_DIG__r, tau_V_neb_cumsum__r=tau_V_neb_cumsum_DIG__r),
        COMP=dict(HaHb__yx=HaHb_COMP__yx, HaHb_cumsum__r=HaHb_cumsum_COMP__r, tau_V_neb_cumsum__r=tau_V_neb_cumsum_COMP__r),
        HII=dict(HaHb__yx=HaHb_HII__yx, HaHb_cumsum__r=HaHb_cumsum_HII__r, tau_V_neb_cumsum__r=tau_V_neb_cumsum_HII__r),
    )

    cumsum_classif = dict(
        total__lr=dict(v=f_cumsum__lr, vsum=f_sum__lr, npts=f_npts__lr, img=f__lyx),
        DIG__lr=dict(v=f_cumsum_DIG__lr, vsum=f_sum_DIG__lr, npts=f_npts_DIG__lr, img=f_DIG__lyx),
        COMP__lr=dict(v=f_cumsum_COMP__lr, vsum=f_sum_COMP__lr, npts=f_npts_COMP__lr, img=f_COMP__lyx),
        HII__lr=dict(v=f_cumsum_HII__lr, vsum=f_sum_HII__lr, npts=f_npts_HII__lr, img=f_HII__lyx),
    )

    return f__lyx, HaHb_classif, cumsum_classif,


def create_segmented_map(ALL, califaID, sel, sample_sel=None):
    N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
    N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
    sel_DIG__yx, sel_COMP__yx, sel_HII__yx, _ = get_selections(ALL, califaID, sel, sample_sel)
    map__yx = np.ma.masked_all((N_y, N_x))
    map__yx[sel_DIG__yx] = 1.
    map__yx[sel_COMP__yx] = 2.
    map__yx[sel_HII__yx] = 3.
    return map__yx


def create_segmented_map_zones(ALL, califaID, sel, sample_sel=None):
    N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
    sel_DIG__z, sel_COMP__z, sel_HII__z, _ = get_selections_zones(ALL, califaID, sel, sample_sel)
    map__z = np.ma.masked_all((N_zone))
    map__z[sel_DIG__z] = 1.
    map__z[sel_COMP__z] = 2.
    map__z[sel_HII__z] = 3.
    return map__z


def get_selections(ALL, califaID, sel, sample_sel=None):
    N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
    N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
    if sample_sel is None:
        sel__yx = np.ones((N_y, N_x), dtype='bool')
    else:
        sel__yx = ALL.get_gal_prop(califaID, sample_sel).reshape(N_y, N_x)
    sel_DIG__yx = np.bitwise_and(sel__yx, ALL.get_gal_prop(califaID, sel['DIG']['yx']).reshape(N_y, N_x))
    sel_COMP__yx = np.bitwise_and(sel__yx, ALL.get_gal_prop(califaID, sel['COMP']['yx']).reshape(N_y, N_x))
    sel_HII__yx = np.bitwise_and(sel__yx, ALL.get_gal_prop(califaID, sel['HII']['yx']).reshape(N_y, N_x))
    sel_gal_tot = sel__yx
    return sel_DIG__yx, sel_COMP__yx, sel_HII__yx, sel_gal_tot


def get_selections_zones(ALL, califaID, sel, sample_sel=None):
    N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
    if sample_sel is None:
        sel__z = np.ones((N_zone), dtype='bool')
    else:
        sel__z = ALL.get_gal_prop(califaID, sample_sel)
    sel_DIG__z = np.bitwise_and(sel__z, ALL.get_gal_prop(califaID, sel['DIG']['z']))
    sel_COMP__z = np.bitwise_and(sel__z, ALL.get_gal_prop(califaID, sel['COMP']['z']))
    sel_HII__z = np.bitwise_and(sel__z, ALL.get_gal_prop(califaID, sel['HII']['z']))
    sel_gal_tot = sel__z
    return sel_DIG__z, sel_COMP__z, sel_HII__z, sel_gal_tot


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


def OH(ALL, sel, califaID):
    sel_WHa = sel['WHa']
    sel_Zhang = sel['Zhang']
    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
    N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
    gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
    f__lyx, HaHb_classif, cumsum_classif = cumulative_profiles(ALL, sel, califaID)
    f__lz = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__z' % L), mask=~gal_sample__z) for L in lines}
    f_cumsum__lr = cumsum_classif['total__lr']['v']
    f_cumsum_HII__lr = cumsum_classif['HII__lr']['v']
    HaHb__z = f__lz['6563']/f__lz['4861']
    tau_V_neb__z = np.log(HaHb__z / 2.86) / qHbHa
    tau_V_neb__yx = HaHb_classif['total']['tau_V_neb__yx']

    tau_V_neb_cumsum__r = HaHb_classif['total']['tau_V_neb_cumsum__r']
    tau_V_neb_cumsum_HII__r = HaHb_classif['HII']['tau_V_neb_cumsum__r']
    ####################################
    # O/H - Relative Oxygen abundances #
    ####################################
    #############
    # O3N2 PP04 #
    #############
    O3Hb__yx = f__lyx['5007']/f__lyx['4861']
    N2Ha__yx = f__lyx['6583']/f__lyx['6563']
    OH_O3N2__yx = 8.73 - 0.32 * np.ma.log10(O3Hb__yx/N2Ha__yx)
    O3Hb__yx = f__lyx['5007']/f__lyx['4861']
    N2Ha__yx = f__lyx['6583']/f__lyx['6563']
    OH_O3N2__yx = 8.73 - 0.32 * np.ma.log10(O3Hb__yx/N2Ha__yx)
    O3Hb_cumsum__r = f_cumsum__lr['5007']/f_cumsum__lr['4861']
    N2Ha_cumsum__r = f_cumsum__lr['6583']/f_cumsum__lr['6563']
    OH_O3N2_cumsum__r = 8.73 - 0.32 * np.ma.log10(O3Hb_cumsum__r/N2Ha_cumsum__r)
    O3Hb_cumsum_HII__r = f_cumsum_HII__lr['5007']/f_cumsum_HII__lr['4861']
    N2Ha_cumsum_HII__r = f_cumsum_HII__lr['6583']/f_cumsum_HII__lr['6563']
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
    mask = np.bitwise_or(mask, f__lyx['3727'].mask)
    mask = np.bitwise_or(mask, f__lyx['4861'].mask)
    mask = np.bitwise_or(mask, f__lyx['5007'].mask)
    mask = np.bitwise_or(mask, f__lyx['6583'].mask)
    logO23__yx = logO23(fOII=f__lyx['3727'], fHb=f__lyx['4861'], f5007=f__lyx['5007'], tau_V=tau_V_neb__yx)
    logN2O2__yx = logN2O2(fNII=f__lyx['6583'], fOII=f__lyx['3727'], tau_V=tau_V_neb__yx)
    mask = np.bitwise_or(mask, logO23__yx.mask)
    mask = np.bitwise_or(mask, logN2O2__yx.mask)
    mlogO23__yx, mlogN2O2__yx = ma_mask_xyz(logO23__yx, logN2O2__yx, mask=mask)
    OH_O23__yx = np.ma.masked_all((N_y, N_x))
    OH_O23__yx[~mask] = OH_O23(logO23_ratio=mlogO23__yx.compressed(), logN2O2_ratio=mlogN2O2__yx.compressed())
    OH_O23__yx[~np.isfinite(OH_O23__yx)] = np.ma.masked
    # HII cumulative O/H
    mask = np.zeros((N_R_bins), dtype='bool')
    logO23_cumsum_HII__r = logO23(fOII=f_cumsum_HII__lr['3727'], fHb=f_cumsum_HII__lr['4861'], f5007=f_cumsum_HII__lr['5007'], tau_V=tau_V_neb_cumsum_HII__r)
    logN2O2_cumsum_HII__r = logN2O2(fNII=f_cumsum_HII__lr['6583'], fOII=f_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
    mask = np.bitwise_or(mask, logO23_cumsum_HII__r.mask)
    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
    mlogO23_cumsum_HII__r, mlogN2O2_cumsum_HII__r = ma_mask_xyz(logO23_cumsum_HII__r, logN2O2_cumsum_HII__r, mask=mask)
    OH_O23_cumsum_HII__r = np.ma.masked_all((N_R_bins))
    OH_O23_cumsum_HII__r[~mask] = OH_O23(logO23_ratio=mlogO23_cumsum_HII__r.compressed(), logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
    OH_O23_cumsum_HII__r[~np.isfinite(OH_O23_cumsum_HII__r)] = np.ma.masked
    # total cumulative O/H
    mask = np.zeros((N_R_bins), dtype='bool')
    logO23_cumsum__r = logO23(fOII=f_cumsum__lr['3727'], fHb=f_cumsum__lr['4861'], f5007=f_cumsum__lr['5007'], tau_V=tau_V_neb_cumsum__r)
    logN2O2_cumsum__r = logN2O2(fNII=f_cumsum__lr['6583'], fOII=f_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
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
    mask = np.bitwise_or(mask, f__lyx['3727'].mask)
    mask = np.bitwise_or(mask, f__lyx['6583'].mask)
    logN2O2__yx = logN2O2(fNII=f__lyx['6583'], fOII=f__lyx['3727'], tau_V=tau_V_neb__yx)
    mask = np.bitwise_or(mask, logN2O2__yx.mask)
    mlogN2O2__yx = np.ma.masked_array(logN2O2__yx, mask=mask)
    OH_N2O2__yx = np.ma.masked_all((N_y, N_x))
    OH_N2O2__yx[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2__yx.compressed())
    OH_N2O2__yx[~np.isfinite(OH_N2O2__yx)] = np.ma.masked
    # HII cumulative O/H
    mask = np.zeros((N_R_bins), dtype='bool')
    logN2O2_cumsum_HII__r = logN2O2(fNII=f_cumsum_HII__lr['6583'], fOII=f_cumsum_HII__lr['3727'], tau_V=tau_V_neb_cumsum_HII__r)
    mask = np.bitwise_or(mask, logN2O2_cumsum_HII__r.mask)
    mlogN2O2_cumsum_HII__r = np.ma.masked_array(logN2O2_cumsum_HII__r, mask=mask)
    OH_N2O2_cumsum_HII__r = np.ma.masked_all((N_R_bins))
    OH_N2O2_cumsum_HII__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum_HII__r.compressed())
    # total cumulative O/H
    mask = np.zeros((N_R_bins), dtype='bool')
    logN2O2_cumsum__r = logN2O2(fNII=f_cumsum__lr['6583'], fOII=f_cumsum__lr['3727'], tau_V=tau_V_neb_cumsum__r)
    mask = np.bitwise_or(mask, logN2O2_cumsum__r.mask)
    mlogN2O2_cumsum__r = np.ma.masked_array(logN2O2_cumsum__r, mask=mask)
    OH_N2O2_cumsum__r = np.ma.masked_all((N_R_bins))
    OH_N2O2_cumsum__r[~mask] = OH_N2O2(logN2O2_ratio=mlogN2O2_cumsum__r.compressed())
    #####################

    OH_dict = dict(
        O3N2=dict(yx=OH_O3N2__yx, cumsum=OH_O3N2_cumsum__r, cumsum_HII=OH_O3N2_cumsum_HII__r),
        N2Ha=dict(yx=OH_N2Ha__yx, cumsum=OH_N2Ha_cumsum__r, cumsum_HII=OH_N2Ha_cumsum_HII__r),
        O23=dict(yx=OH_O23__yx, cumsum=OH_O23_cumsum__r, cumsum_HII=OH_O23_cumsum_HII__r),
        N2O2=dict(yx=OH_N2O2__yx, cumsum=OH_N2O2_cumsum__r, cumsum_HII=OH_N2O2_cumsum_HII__r),
    )

    return OH_dict


def fig1(ALL, sel, gals):
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

    sel_WHa = sel['WHa']
    sel_Zhang = sel['Zhang']
    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        L = Lines()

        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
        gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)

        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        pixelDistance__yx = ALL.get_gal_prop(califaID, ALL.pixelDistance__yx).reshape(N_y, N_x)
        pixelDistance_HLR__yx = pixelDistance__yx / HLR_pix
        zoneDistance_HLR__z = ALL.zoneDistance_HLR
        SB__lyx, SB__lz, f__lz = {}, {}, {}
        for line in lines:
            SB__lyx[line] = np.ma.masked_array(ALL.get_gal_prop(califaID, 'SB%s__yx' % line).reshape(N_y, N_x), mask=~gal_sample__yx)
            SB__lz[line] = np.ma.masked_array(ALL.get_gal_prop(califaID, 'SB%s__z' % line), mask=~gal_sample__z)
            f__lz[line] = np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__z' % line), mask=~gal_sample__z)
        W6563__yx = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x), mask=~gal_sample__yx)
        W6563__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.W6563__z), mask=~gal_sample__z)
        O3Hb__yx = SB__lyx['5007']/SB__lyx['4861']
        O3Hb__z = f__lz['5007']/f__lz['4861']
        N2Ha__yx = SB__lyx['6583']/SB__lyx['6563']
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
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa, sel_sample__gz)
        print map__z.count()
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
        map__z = create_segmented_map_zones(ALL, califaID, sel['Zhang'], sel_sample__gz)
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
        # 5
        plotBPT(ax5, np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z), z=np.ma.log10(W6563__z), vmin=logWHa_range[0], vmax=logWHa_range[1], cb_label=r'W${}_{H\alpha}\ [\AA]$', cmap='viridis_r')
        # 6
        plotBPT(ax6, np.ma.log10(N2Ha__z), np.ma.log10(O3Hb__z), z=np.ma.log10(SB__lz['6563']), vmin=logSBHa_range[0], vmax=logSBHa_range[1], cb_label=r'$\Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]', cmap='viridis_r')

        # AXIS 7, 8 e 9
        map__yx = create_segmented_map(ALL, califaID, sel['WHa'], sel_sample__gyx)
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
        sel_DIG__yx, sel_COMP__yx, sel_HII__yx, _ = get_selections(ALL, califaID, sel['WHa'], sel_sample__gyx)
        min_npts = 10
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel_npts = npts_DIG > min_npts
            ax8.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], '--', lw=2, c=colors_lines_DIG_COMP_HII[0])
            ax8.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel_npts = npts_COMP > min_npts
            ax8.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], '--', lw=2, c=colors_lines_DIG_COMP_HII[1])
            ax8.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
        yMean_HII, yPrc_HII, bin_center_HII, npts_HII = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_HII.any():
            sel_npts = npts_HII > min_npts
            ax8.plot(bin_center_HII[sel_npts], yPrc_HII[2][sel_npts], 'k--', lw=2, c=colors_lines_DIG_COMP_HII[2])
            ax8.plot(bin_center_HII[sel_npts], yPrc_HII[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[2], markersize=10)
        # 9
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__lyx['6563']))
        ax9.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax9.set_xlim(distance_range)
        ax9.set_ylim(logSBHa_range)
        ax9.set_xlabel(r'R [HLR]')
        ax9.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax9.grid()
        sel_DIG__yx, sel_COMP__yx, sel_HII__yx, _ = get_selections(ALL, califaID, sel['WHa'], sel_sample__gyx)
        min_npts = 10
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel_npts = npts_DIG > min_npts
            ax9.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], '--', lw=2, c=colors_lines_DIG_COMP_HII[0])
            ax9.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel_npts = npts_COMP > min_npts
            ax9.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], '--', lw=2, c=colors_lines_DIG_COMP_HII[1])
            ax9.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
        yMean_HII, yPrc_HII, bin_center_HII, npts_HII = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_HII.any():
            sel_npts = npts_HII > min_npts
            ax9.plot(bin_center_HII[sel_npts], yPrc_HII[2][sel_npts], 'k--', lw=2, c=colors_lines_DIG_COMP_HII[2])
            ax9.plot(bin_center_HII[sel_npts], yPrc_HII[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[2], markersize=10)

        # AXIS 10, 11 e 12
        map__yx = create_segmented_map(ALL, califaID, sel['Zhang'], sel_sample__gyx)
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
        sel_DIG__yx, sel_COMP__yx, sel_HII__yx, _ = get_selections(ALL, califaID, sel['Zhang'], sel_sample__gyx)
        min_npts = 10
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel_npts = npts_DIG > min_npts
            ax11.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], '--', lw=2, c=colors_lines_DIG_COMP_HII[0])
            ax11.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel_npts = npts_COMP > min_npts
            ax11.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], '--', lw=2, c=colors_lines_DIG_COMP_HII[1])
            ax11.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
        yMean_HII, yPrc_HII, bin_center_HII, npts_HII = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_HII.any():
            sel_npts = npts_HII > min_npts
            ax11.plot(bin_center_HII[sel_npts], yPrc_HII[2][sel_npts], 'k--', lw=2, c=colors_lines_DIG_COMP_HII[2])
            ax11.plot(bin_center_HII[sel_npts], yPrc_HII[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[2], markersize=10)
        # 12
        x = np.ma.ravel(pixelDistance_HLR__yx)
        y = np.ma.ravel(np.ma.log10(SB__lyx['6563']))
        ax12.scatter(x, y, c=np.ravel(map__yx), cmap=cmap, s=2, **dflt_kw_scatter)
        ax12.set_xlim(distance_range)
        ax12.set_ylim(logSBHa_range)
        ax12.set_xlabel(r'R [HLR]')
        ax12.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax12.grid()
        sel_DIG__yx, sel_COMP__yx, sel_HII__yx, _ = get_selections(ALL, califaID, sel['Zhang'], sel_sample__gyx)
        min_npts = 10
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_DIG__yx))
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel_npts = npts_DIG > min_npts
            ax12.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], '--', lw=2, c=colors_lines_DIG_COMP_HII[0])
            ax12.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_COMP__yx))
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel_npts = npts_COMP > min_npts
            ax12.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], '--', lw=2, c=colors_lines_DIG_COMP_HII[1])
            ax12.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~np.ravel(sel_HII__yx))
        yMean_HII, yPrc_HII, bin_center_HII, npts_HII = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_HII.any():
            sel_npts = npts_HII > min_npts
            ax12.plot(bin_center_HII[sel_npts], yPrc_HII[2][sel_npts], 'k--', lw=2, c=colors_lines_DIG_COMP_HII[2])
            ax12.plot(bin_center_HII[sel_npts], yPrc_HII[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_HII[2], markersize=10)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('%s-fig1.png' % califaID)
        plt.close(f)


def fig2(ALL, sel, gals):
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
    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
        gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)
        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        N_pixel_notmasked = gal_sample__yx.astype('int').sum()
        N_zone_notmasked = gal_sample__z.astype('int').sum()
        pixelDistance__yx = ALL.get_gal_prop(califaID, ALL.pixelDistance__yx).reshape(N_y, N_x)
        pixelDistance_HLR__yx = pixelDistance__yx / HLR_pix
        map__yx = create_segmented_map(ALL, califaID, sel['WHa'], sel_sample__gyx)
        OH_dict = OH(ALL, sel, califaID)
        #####################
        # PLOT
        #####################
        N_cols = 4
        N_rows = 2
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        cmap = cmap_discrete(colors_DIG_COMP_HII)
        (axes_row1, axes_row2) = axArr
        f.suptitle(r'%s - %s: %d pixels [not masked - %d] (%d zones [not masked - %d]) - classif. W${}_{H\alpha}$' % (califaID, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_pixel_notmasked, N_zone, N_zone_notmasked))
        for i, k in enumerate(['O3N2', 'N2Ha', 'O23', 'N2O2']):
            ax1, ax2 = zip(axes_row1, axes_row2)[i]
            OH__yx = OH_dict[k]['yx']
            OH_label = k
            plot_OH(ax1, pixelDistance_HLR__yx, OH__yx, OH_label, map__yx, cmap, OH_range, distance_range)
            plot_text_ax(ax1, '%c)' % chr(97+i*2), 0.02, 0.98, 16, 'top', 'left', 'k')
            OH_cumsum__r = OH_dict[k]['cumsum']
            OH_cumsum_HII__r = OH_dict[k]['cumsum_HII']
            plot_cumulative_OH(ax2, R_bin_center__r, OH_cumsum__r, OH_cumsum_HII__r, OH_label, OH_range, distance_range)
            plot_text_ax(ax2, '%c)' % chr(97+(i*2+1)), 0.02, 0.98, 16, 'top', 'left', 'k')
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('%s-fig2.png' % califaID)
        plt.close(f)


def figs4567(ALL, sel, gals):
    N_gals = len(gals)

    gal_sample__gyx = sel['gals_sample__yx']

    for OH_method in ['O3N2', 'N2Ha', 'O23', 'N2O2']:
        p33ba, p66ba = np.percentile(ALL.ba, [33, 66])

        sel_gals_ba = [
            np.less(ALL.ba, p33ba)[sel['gals']],
            np.bitwise_and(np.greater_equal(ALL.ba, p33ba), np.less(ALL.ba, p66ba))[sel['gals']],
            np.greater_equal(ALL.ba, p66ba)[sel['gals']],
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
                print len(gals), len(selection)
                sel_gals = np.asarray(gals)[selection]
                N_gals_sel = len(sel_gals)
                iS = np.argsort(sort_var[selection])
                N_rows, N_cols = 10, 12
                row, col = 0, 0
                f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 3.2, N_rows * 2.4))
                f.suptitle('OH (%s) profiles - %d gals (%d this page) - %s - sort by: %s' % (OH_method, N_gals, N_gals_sel, ba_labels[i_sel], v_ver['label']), fontsize=20)
                ax_i = 0
                N_axes = N_cols * N_rows
                for g in sel_gals[iS]:
                    ax = np.ravel(axArr)[ax_i]
                    ax = axArr[row][col]
                    HLR_pix = ALL.get_gal_prop_unique(g, ALL.HLR_pix)
                    N_x = ALL.get_gal_prop_unique(g, ALL.N_x)
                    N_y = ALL.get_gal_prop_unique(g, ALL.N_y)
                    pixelDistance__yx = ALL.get_gal_prop(g, ALL.pixelDistance__yx).reshape(N_y, N_x)
                    pixelDistance_HLR__yx = pixelDistance__yx / HLR_pix
                    OH_dict = OH(ALL, sel, g)
                    y = OH_dict[OH_method]['yx']
                    y_cumsum = OH_dict[OH_method]['cumsum']
                    y_cumsum_HII = OH_dict[OH_method]['cumsum_HII']
                    sel_DIG__yx, sel_COMP__yx, sel_HII__yx, _ = get_selections(ALL, g, sel['WHa'], gal_sample__gyx)

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
                        sel_npts = npts > min_npts
                        ax.plot(bin_center, yPrc[2], '--', lw=1, c='gray')
                        ax.plot(bin_center[sel_npts], yPrc[2][sel_npts], '--', lw=1, c='k')
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
                f.savefig('%s_%s_p%d.png' % (OH_method, k_ver, page), orientation='landscape', dpi=200)
                plt.close(f)
                page += 1


if __name__ == '__main__':
    main(sys.argv)
