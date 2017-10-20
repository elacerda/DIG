import os
import sys
import ast
import time
import datetime
import itertools
import numpy as np
import matplotlib as mpl
from pytu.lines import Lines
from pycasso import fitsQ3DataCube
from matplotlib import pyplot as plt
from pycasso.util import radialProfile
import matplotlib.gridspec as gridspec
from pystarlight.util.constants import L_sun
from pytu.objects import readFileArgumentParser
from pytu.functions import ma_mask_xyz, debug_var
from pystarlight.util.redenninglaws import calc_redlaw
from CALIFAUtils.objects import stack_gals, CALIFAPaths
from mpl_toolkits.axes_grid1 import make_axes_locatable
from CALIFAUtils.plots import DrawHLRCircleInSDSSImage, DrawHLRCircle
from CALIFAUtils.scripts import get_NEDName_by_CALIFAID, spaxel_size_pc, F_to_L
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator, ScalarFormatter
from pytu.plots import cmap_discrete, plot_text_ax, density_contour, plot_scatter_histo, plot_histo_ax, stats_med12sigma, add_subplot_axes


# mpl.rcParams['font.size'] = 20
# mpl.rcParams['axes.labelsize'] = 16
# mpl.rcParams['axes.titlesize'] = 16
# mpl.rcParams['xtick.labelsize'] = 14
# mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
latex_ppi = 72.0
latex_column_width_pt = 240.0
latex_column_width = latex_column_width_pt/latex_ppi
latex_text_width_pt = 504.0
latex_text_width = latex_text_width_pt/latex_ppi
golden_mean = 0.5 * (1. + 5**0.5)
_SN_types = ['SN_HaHb', 'SN_BPT', 'SN_Ha']
cmap_R = plt.cm.copper
minorLocator = AutoMinorLocator(5)
_transp_choice = False
_dpi_choice = 300
# debug = True
# CCM reddening law
_q = calc_redlaw([4861, 6563], R_V=3.1, redlaw='CCM')
f_tauVneb = lambda Ha, Hb: np.ma.log(Ha / Hb / 2.86) / (_q[0] - _q[1])

# config variables
lineratios_range = {
    '6563/4861': [0.15, 1.],
    '6583/6563': [-0.7, 0.1],
    # '6300/6563': [-2, 0],
    '6717+6731/6563': [-0.8, 0.2],
}

distance_range = [0, 3]
tauVneb_range = [0, 5]
tauVneb_neg_range = [-1, 5]
logSBHa_range = [4, 6.5]
logWHa_range = [-0.5, 2]
DtauV_range = [-3, 3]
DeltatauV_range = [-1, 3]
DtauVnorm_range = [-1, 4]
x_Y_range = [0, 0.5]
OH_range = [8, 9.5]

# age to calc xY
# tY = 100e6

# read_one_cube() PyCASSO CALIFADataCube options`
config = -2
EL = True
elliptical = True
kw_cube = dict(EL=EL, config=config, elliptical=elliptical)
# setting directories
P = CALIFAPaths()

# Zhang SFc threshold
SFc_Zhang_threshold = 1e39/L_sun

# _lines = ['3727', '4363', '4861', '4959', '5007', '6300', '6563', '6583', '6717', '6731']
_lines = ['3727', '4363', '4861', '4959', '5007', '6563', '6583', '6717', '6731']

# Some defaults arguments
dflt_kw_scatter = dict(marker='o', edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, frac=0.07, debug=True)  # , tendency=True)
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal')


def parser_args(default_args_file='/Users/lacerda/dev/astro/dig/default.args'):
    '''
        Parse the command line args
        With fromfile_prefix_chars=@ we can read and parse command line args
        inside a file with @file.txt.
        default args inside default_args_file
    '''
    default_args = {
        'debug': False,
        'file': None,
        'gals': None,
        'rbinini': 0.,
        'rbinfin': 3.,
        'rbinstep': 0.2,
        'class_names': ['hDIG', 'mDIG', 'SFc'],
        'class_colors': ['brown', 'tomato', 'royalblue'],
        'class_linecolors': ['maroon', 'darkred', 'mediumblue'],
        'class_thresholds': [ 3, 12 ],
        'min_SN': 0,
        'SN_type': 'SN_Ha',
        'summary': False,
        # 'min_popx': 0,
    }
    parser = readFileArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--debug', '-D', action='store_true', default=default_args['debug'])
    parser.add_argument('--file', '-f', metavar='FILE', type=str, default=default_args['file'])
    parser.add_argument('--gals', metavar='FILE', type=str, default=default_args['gals'])
    parser.add_argument('--rbinini', metavar='HLR', type=float, default=default_args['rbinini'])
    parser.add_argument('--rbinfin', metavar='HLR', type=float, default=default_args['rbinfin'])
    parser.add_argument('--rbinstep', metavar='HLR', type=float, default=default_args['rbinstep'])
    parser.add_argument('--class_names', metavar='CLASSES', type=str, nargs='+', default=default_args['class_names'])
    parser.add_argument('--class_colors', metavar='COLORS', type=str, nargs='+', default=default_args['class_colors'])
    parser.add_argument('--class_linecolors', metavar='COLORS', type=str, nargs='+', default=default_args['class_linecolors'])
    parser.add_argument('--class_thresholds', metavar='FLOAT', type=float, nargs='+', default=default_args['class_thresholds'])
    parser.add_argument('--min_SN', metavar='FLOAT', type=float, default=default_args['min_SN'])
    # parser.add_argument('--min_popx', metavar='FLOAT', type=float, default=default_args['min_popx'])
    parser.add_argument('--SN_type', metavar='STR', type=str, default=default_args['SN_type'])
    parser.add_argument('--summary', '-S', action='store_true', default=default_args['summary'])
    args_list = sys.argv[1:]
    # if exists file default.args, load default args
    if os.path.isfile(default_args_file):
        args_list.insert(0, '@%s' % default_args_file)
    debug_var(True, args_list=args_list)
    args = parser.parse_args(args=args_list)
    assert len(args.class_thresholds) == (len(args.class_names) - 1), 'N(class_thresholds) must be N(class_names) - 1'
    assert len(args.class_names) == len(args.class_colors), 'N(class_colors) must be N(class_names)'
    assert len(args.class_names) == len(args.class_linecolors), 'N(class_linecolors) must be N(class_names)'
    assert args.SN_type in _SN_types, 'SN_type not implemented'
    args.R_bin__r = np.arange(args.rbinini, args.rbinfin + args.rbinstep, args.rbinstep)
    args.R_bin_center__r = (args.R_bin__r[:-1] + args.R_bin__r[1:]) / 2.0
    args.N_R_bins = len(args.R_bin_center__r)
    args.dict_colors = create_class_dict(args)
    debug_var(True, args=args)
    return args


def plot_setup(width, aspect, fignum=None, dpi=300, cmap=None):
    if cmap is None:
        cmap = 'inferno_r'
    plotpars = {
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'font.size': 8,
        'axes.titlesize': 10,
        'lines.linewidth': 0.5,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.8,
        'font.family': 'Times New Roman',
        'figure.subplot.left': 0.04,
        'figure.subplot.bottom': 0.04,
        'figure.subplot.right': 0.97,
        'figure.subplot.top': 0.95,
        'figure.subplot.wspace': 0.1,
        'figure.subplot.hspace': 0.25,
        'image.cmap': cmap,
    }
    plt.rcParams.update(plotpars)
    # plt.ioff()
    figsize = (width, width * aspect)
    return plt.figure(fignum, figsize, dpi=dpi)


def create_class_selection_WHa(args, W6563):
    class_names, class_thresholds = args.class_names, args.class_thresholds
    sel_WHa = {}
    print 0, class_names[0]
    sel_WHa[class_names[0]] = (W6563 <= class_thresholds[0])
    print '(W6563 <= %.1f)' % class_thresholds[0]
    n_class = len(class_names)
    n_th = len(class_thresholds)
    assert n_th == (n_class - 1), 'wrong number of thresholds'
    for i, c in enumerate(class_names[1:]):
        print i, c
        if i < (n_th - 1):
            sel_WHa[c] = ((W6563 > class_thresholds[i]) & (W6563 <= class_thresholds[i+1]))
            print '((W6563 > %.1f) & (W6563 <= %.1f))' % (class_thresholds[i], class_thresholds[i+1])
        else:
            sel_WHa[c] = ((W6563 > class_thresholds[i]))
            print '(W6563 > %.1f)' % class_thresholds[i]
    return sel_WHa


def create_class_dict(args):
    class_names, class_colors, class_linecolors = args.class_names, args.class_colors, args.class_linecolors
    d = {}
    for i, c in enumerate(class_names):
        d[c] = dict(c=class_colors[i], lc=class_linecolors[i])
    return d


def get_selections_spaxels(args, ALL, califaID, sel, sample_sel=None):
    sel_gal = {}
    N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
    N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
    if sample_sel is None:
        sel__yx = np.ones((N_y, N_x), dtype='bool')
    else:
        sel__yx = ALL.get_gal_prop(califaID, sample_sel).reshape(N_y, N_x)
    for i, c in enumerate(args.class_names):
        sel_gal[c] = sel__yx & ALL.get_gal_prop(califaID, sel[c]).reshape(N_y, N_x)
    sel_gal_tot = sel__yx
    return sel_gal, sel_gal_tot


def get_selections_zones(args, ALL, califaID, sel, sample_sel=None):
    sel_gal = {}
    N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
    if sample_sel is None:
        sel__z = np.ones((N_zone), dtype='bool')
    else:
        sel__z = ALL.get_gal_prop(califaID, sample_sel)
    for i, c in enumerate(args.class_names):
        sel_gal[c] = sel__z & ALL.get_gal_prop(califaID, sel[c])
    sel_gal_tot = sel__z
    return sel_gal, sel_gal_tot


def create_segmented_map_spaxels(args, ALL, califaID, sel, sample_sel=None):
    N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
    N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
    map__yx = np.ma.masked_all((N_y, N_x))
    sel_gal, _ = get_selections_spaxels(args, ALL, califaID, sel, sample_sel)
    for i, c in enumerate(args.class_names):
        map__yx[sel_gal[c]] = i + 1
    return map__yx


def create_segmented_map_zones(args, ALL, califaID, sel, sample_sel=None):
    N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
    map__z = np.ma.masked_all((N_zone))
    sel_gal, _ = get_selections_zones(args, ALL, califaID, sel, sample_sel)
    for i, c in enumerate(args.class_names):
        map__z[sel_gal[c]] = i + 1
    return map__z


def samples(args, ALL, sample_choice, gals=None):
    sel_WHa = {}
    sel_WHa['z'] = create_class_selection_WHa(args, ALL.W6563__z)
    sel_WHa['yx'] = create_class_selection_WHa(args, ALL.W6563__yx)

    sel_Zhang = {}
    sel_Zhang['z'] = (ALL.SB6563__z >= SFc_Zhang_threshold)
    sel_Zhang['yx'] = (ALL.SB6563__yx >= SFc_Zhang_threshold)

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
        z=dict(
            S06=sel_below_S06__z,
            K03=sel_below_K03__z,
            K01=sel_below_K01__z,
            betS06K03=sel_between_S06K03__z,
            betK03K01=sel_between_K03K01__z,
            aboK01=sel_above_K01__z,
        ),
        yx=dict(
            S06=sel_below_S06__yx,
            K03=sel_below_K03__yx,
            K01=sel_below_K01__yx,
            betS06K03=sel_between_S06K03__yx,
            betK03K01=sel_between_K03K01__yx,
            aboK01=sel_above_K01__yx,
        )
    )

    f__lgz = {'%s' % l: getattr(ALL, 'f%s__z' % l) for l in _lines}
    ef__lgz = {'%s' % l: getattr(ALL, 'ef%s__z' % l) for l in _lines}
    SN__lgz = {'%s' % l: f__lgz[l]/ef__lgz[l] for l in _lines}
    f__lgyx = {'%s' % l: getattr(ALL, 'f%s__yx' % l) for l in _lines}
    ef__lgyx = {'%s' % l: getattr(ALL, 'ef%s__yx' % l) for l in _lines}
    SN__lgyx = {'%s' % l: f__lgyx[l]/ef__lgyx[l] for l in _lines}

    choice_SN = float(sample_choice[1])
    # choice_SN = args.SN
    sel_SN = dict(
        lz={'%s' % l: np.greater(SN__lgz[l].filled(0.), choice_SN) for l in _lines},
        lyx={'%s' % l: np.greater(SN__lgyx[l].filled(0.), choice_SN) for l in _lines}
    )
    sel_SN3 = dict(
        lz={'%s' % l: np.greater(SN__lgz[l].filled(0.), 3) for l in _lines},
        lyx={'%s' % l: np.greater(SN__lgyx[l].filled(0.), 3) for l in _lines}
    )

    sel_SN_BPT = dict(
        z=np.bitwise_and(sel_SN['lz']['4861'], np.bitwise_and(sel_SN['lz']['5007'], np.bitwise_and(sel_SN['lz']['6563'], sel_SN['lz']['6583']))),
        yx=np.bitwise_and(sel_SN['lyx']['4861'], np.bitwise_and(sel_SN['lyx']['5007'], np.bitwise_and(sel_SN['lyx']['6563'], sel_SN['lyx']['6583'])))
    )
    sel_SN_BPT3 = dict(
        z=np.bitwise_and(sel_SN3['lz']['4861'], np.bitwise_and(sel_SN3['lz']['5007'], np.bitwise_and(sel_SN3['lz']['6563'], sel_SN3['lz']['6583']))),
        yx=np.bitwise_and(sel_SN3['lyx']['4861'], np.bitwise_and(sel_SN3['lyx']['5007'], np.bitwise_and(sel_SN3['lyx']['6563'], sel_SN3['lyx']['6583'])))
    )

    sel_SN_Ha = dict(
        z=sel_SN['lz']['6563'],
        yx=sel_SN['lyx']['6563']
    )

    sel_SN_HaHb = dict(
        z=np.bitwise_and(sel_SN['lz']['6563'], sel_SN['lz']['4861']),
        yx=np.bitwise_and(sel_SN['lyx']['6563'], sel_SN['lyx']['4861'])
    )
    sel_SN_HaHb3 = dict(
        z=np.bitwise_and(sel_SN3['lz']['6563'], sel_SN3['lz']['4861']),
        yx=np.bitwise_and(sel_SN3['lyx']['6563'], sel_SN3['lyx']['4861'])
    )

    sel = dict(
        SN=sel_SN,
        WHa=sel_WHa,
        Zhang=sel_Zhang,
        BPT=sel_BPT,
        SN_BPT=sel_SN_BPT,
        SN_HaHb=sel_SN_HaHb,
        SN_Ha=sel_SN_Ha,
        SN_BPT3=sel_SN_BPT3,
        SN_HaHb3=sel_SN_HaHb3,
    )

    return gals_sample_choice(args, ALL, sel, gals, sample_choice)


def gals_sample_choice(args, ALL, sel, gals, sample_choice):
    t_init = time.time()
    sample__z = sel[sample_choice[0]]['z']
    sample__yx = sel[sample_choice[0]]['yx']

    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')
    sel_gals__gyx = np.zeros((ALL.califaID__yx.shape), dtype='bool')
    sel_gals_sample__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')
    sel_gals_sample__gyx = np.zeros((ALL.califaID__yx.shape), dtype='bool')

    mt = np.hstack(list(itertools.chain(list(itertools.repeat(mt, ALL.N_zone[i])) for i, mt in enumerate(ALL.mt))))
    mt_keys = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']
    sel_gals_mt = {
        mt_keys[0]: (mt == -2),
        mt_keys[1]: (mt == -1),
        mt_keys[2]: np.bitwise_or(mt == 0, mt == 1),
        mt_keys[3]: mt == 2,
        mt_keys[4]: mt == 3,
        mt_keys[5]: np.bitwise_or(mt == 4, np.bitwise_or(mt == 5, np.bitwise_or(mt == 6, mt == 7))),
        # 'Sc+Scd+Sd+Irr'
    }

    sel_mt = {
        mt_keys[0]: (ALL.mt == -2),
        mt_keys[1]: (ALL.mt == -1),
        mt_keys[2]: np.bitwise_or(ALL.mt == 0, ALL.mt == 1),
        mt_keys[3]: ALL.mt == 2,
        mt_keys[4]: ALL.mt == 3,
        mt_keys[5]: np.bitwise_or(ALL.mt == 4, np.bitwise_or(ALL.mt == 5, np.bitwise_or(ALL.mt == 6, ALL.mt == 7))),
    }

    debug_var(args.debug, N_gals_in=len(gals))
    debug_var(args.debug, gals_in=gals)

    new_gals = gals[:]
    tmp_sel_gals = np.zeros(ALL.califaID__g.shape, dtype='bool')
    califaID__g_tolist = ALL.califaID__g.tolist()
    # print gals
    t_init_loop = time.time()
    for g in gals:
        if g in califaID__g_tolist:
            i_gal = califaID__g_tolist.index(g)
        else:
            print '%s: gal without data' % g
            new_gals.remove(g)
            continue
        tmp_sel__gz = (ALL.califaID__z == g)
        if tmp_sel__gz.any():
            tmp_sel_gals[i_gal] = True
            tmp_sel_sample__gz = sample__z & (ALL.califaID__z == g)
            tmp_sel_sample__gyx = sample__yx & (ALL.califaID__yx == g)
            # tmp_sel_gal_sample__z = ALL.get_gal_prop(g, tmp_sel_sample__gz)
            # tmp_sel_gal_sample__yx = ALL.get_gal_prop(g, tmp_sel_sample__gyx)
            # if N_zone_notmasked == 0 or N_pixel_notmasked == 0:
            # _Nz_tmp = ALL.get_gal_prop(g, tmp_sel_sample__gz)
            # _Nyx_tmp = ALL.get_gal_prop(g, tmp_sel_sample__gyx)
            if (~tmp_sel_sample__gz).all() or (~tmp_sel_sample__gyx).all():
                new_gals.remove(g)
                continue
            sel_gals__gz[tmp_sel__gz] = True
            sel_gals__gyx[ALL.califaID__yx == g] = True
            sel_gals_sample__gz[tmp_sel_sample__gz] = True
            sel_gals_sample__gyx[tmp_sel_sample__gyx] = True
        else:
            print '>>> %s: gal without data' % g

    print 'gals loop time: %.2f' % (time.time() - t_init_loop)
    debug_var(args.debug, N_gals_with_data=len(new_gals))
    debug_var(args.debug, gals_with_data=new_gals)

    ba_bins = [0.5, 0.87]
    ba_keys = ['all', 'ba_low', 'ba_mid', 'ba_high']
    sel_ba = {
        ba_keys[0]: tmp_sel_gals,  # all
        ba_keys[1]: (ALL.ba <= ba_bins[0]),
        ba_keys[2]: ((ALL.ba > ba_bins[0]) & (ALL.ba <= ba_bins[1])),
        ba_keys[3]: (ALL.ba > ba_bins[1]),
    }

    sel_ml_ba = {
        ba_keys[0]: (ALL.ml_ba >= 0) & tmp_sel_gals,  # all
        ba_keys[1]: (ALL.ml_ba >= 0) & (ALL.ml_ba <= ba_bins[0]),
        ba_keys[2]: ((ALL.ml_ba > ba_bins[0]) & (ALL.ml_ba <= ba_bins[1])),
        ba_keys[3]: (ALL.ml_ba > ba_bins[1]),
    }

    # report
    print 'pycasso ba'
    for k, v in sel_ba.iteritems():
        aux_gals = ALL.califaID__g[v]
        mt__g = np.array([ALL.mt[ALL.califaID__g == g][0] for g in aux_gals])
        print mt__g
        aux_sel_gals_mt = {
            'E': (mt__g == -2),
            'S0+S0a': (mt__g == -1),
            'Sa+Sab': np.bitwise_or(mt__g == 0, mt__g == 1),
            'Sb': mt__g == 2,
            'Sbc': mt__g == 3,
            '>= Sc': np.bitwise_or(mt__g == 4, np.bitwise_or(mt__g == 5, np.bitwise_or(mt__g == 6, mt__g == 7))),
        }
        print k, len(ALL.califaID__g[v])
        for mk in mt_keys:
            print mk, aux_sel_gals_mt[mk].astype('int').sum()

    print 'Masterlist ba'
    for k, v in sel_ml_ba.iteritems():
        aux_gals = ALL.califaID__g[v]
        mt__g = np.array([ALL.mt[ALL.califaID__g == g][0] for g in aux_gals])
        print mt__g
        aux_sel_gals_mt = {
            'E': (mt__g == -2),
            'S0+S0a': (mt__g == -1),
            'Sa+Sab': np.bitwise_or(mt__g == 0, mt__g == 1),
            'Sb': mt__g == 2,
            'Sbc': mt__g == 3,
            '>= Sc': np.bitwise_or(mt__g == 4, np.bitwise_or(mt__g == 5, np.bitwise_or(mt__g == 6, mt__g == 7))),
        }
        print k, len(ALL.califaID__g[v])
        for mk in mt_keys:
            print mk, aux_sel_gals_mt[mk].astype('int').sum()

    ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_zone[i])) for i, ba in enumerate(ALL.ba))))
    sel_gals_ba = {
        ba_keys[0]: sel_gals_sample__gz,
        ba_keys[1]: (ba <= ba_bins[0]),
        ba_keys[2]: ((ba > ba_bins[0]) & (ba <= ba_bins[1])),
        ba_keys[3]: (ba > ba_bins[1]),
    }

    ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_zone[i])) for i, ba in enumerate(ALL.ml_ba))))
    sel_gals_ml_ba = {
        ba_keys[0]: (ba >= 0) & sel_gals_sample__gz,
        ba_keys[1]: (ba >= 0) & (ba <= ba_bins[0]),
        ba_keys[2]: ((ba > ba_bins[0]) & (ba <= ba_bins[1])),
        ba_keys[3]: (ba > ba_bins[1]),
    }

    ALL.ba_bins = ba_bins
    ALL.mt_keys = mt_keys
    ALL.ba_keys = ba_keys
    sel['gals__mt'] = sel_mt
    sel['gals__mt_z'] = sel_gals_mt
    sel['gals__ba'] = sel_ba
    sel['gals__ba_z'] = sel_gals_ba
    sel['gals__mlba'] = sel_ml_ba
    sel['gals__mlba_z'] = sel_gals_ml_ba
    sel['gals'] = tmp_sel_gals
    sel['gals__z'] = sel_gals__gz
    sel['gals__yx'] = sel_gals__gyx
    sel['gals_sample__z'] = sel_gals_sample__gz
    # & (ALL.zoneDistance_HLR <= 2)
    # & (ALL.qSn__z > 10)
    sel['gals_sample__yx'] = sel_gals_sample__gyx
    # & (ALL.pixelDistance_HLR__yx <= 2)
    # & (ALL.qSn__yx > 10)

    print 'gals_sample_choice() time: %.2f' % (time.time() - t_init)
    return new_gals, sel, sample_choice


def create_fHa_cumul_per_WHa_bins(args):
    ALL, sel = args.ALL, args.sel
    sel_gals = sel['gals']
    gals = args.gals

    sel_sample__gz = sel['gals_sample__z']
    sel_radius_inside = (ALL.zoneDistance_HLR <= 0.5)
    sel_radius_middle = (ALL.zoneDistance_HLR > 0.5) & (ALL.zoneDistance_HLR <= 1)
    sel_radius_outside = (ALL.zoneDistance_HLR > 1)
    sel_gals_sample_Rin__gz = np.bitwise_and(sel_radius_inside, sel_sample__gz)
    sel_gals_sample_Rmid__gz = np.bitwise_and(sel_radius_middle, sel_sample__gz)
    sel_gals_sample_Rout__gz = np.bitwise_and(sel_radius_outside, sel_sample__gz)
    sels_R = [sel_sample__gz, sel_gals_sample_Rin__gz, sel_gals_sample_Rmid__gz, sel_gals_sample_Rout__gz]

    sel['radius_inside'] = sel_radius_inside
    sel['radius_middle'] = sel_radius_middle
    sel['radius_outside'] = sel_radius_outside
    sel['gals_sample_Rin__z'] = sel_gals_sample_Rin__gz
    sel['gals_sample_Rmid__z'] = sel_gals_sample_Rmid__gz
    sel['gals_sample_Rout__z'] = sel_gals_sample_Rout__gz

    cumulfHa__g_R = {}
    logWHa_bins = np.linspace(-1, 3, 50)
    # for k in ALL.mt_keys:
    #     v = sel['gals_mt'][k]
    #     for g in ALL.califaID__g[v]:
    for g in gals:
        aux_list_fHa = []
        for i, sel_R in enumerate(sels_R):
            sel_sample__z = ALL.get_gal_prop(g, sel_sample__gz)
            sel_R__z = ALL.get_gal_prop(g, sel_R)
            WHa = np.ma.masked_array(ALL.get_gal_prop(g, ALL.W6563__z), mask=~(sel_sample__z & sel_R__z))
            fHa = np.ma.masked_array(ALL.get_gal_prop(g, ALL.f6563__z), mask=~(sel_sample__z & sel_R__z))
            xm, ym = ma_mask_xyz(np.ma.log10(WHa), fHa)
            fHa_cumul, fHa_tot = cumul_bins(ym.compressed(), xm.compressed(), logWHa_bins)  # , fHa_tot)
            aux_list_fHa.append(fHa_cumul)
        cumulfHa__g_R[g] = aux_list_fHa
    ALL.cumulfHa__gRw = cumulfHa__g_R
    ALL.logWHa_bins__w = logWHa_bins

    # Some numbers
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    for k, v in sel['gals__mt'].iteritems():
        print k
        f_SFc = []
        f_mDIG = []
        f_hDIG = []
        for g in ALL.califaID__g[v]:
            if not g in gals:
                continue
            x = np.log10(args.class_thresholds[0])
            y1 = ALL.cumulfHa__gRw[g][0][ALL.logWHa_bins__w < x][-1]
            y2 = ALL.cumulfHa__gRw[g][0][ALL.logWHa_bins__w > x][0]
            x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
            x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
            y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
            # print x, y_th, x1, x2, y1, y2
            f_hDIG__g[g] = y_th
            x = np.log10(args.class_thresholds[1])
            y1 = ALL.cumulfHa__gRw[g][0][ALL.logWHa_bins__w < x][-1]
            y2 = ALL.cumulfHa__gRw[g][0][ALL.logWHa_bins__w > x][0]
            x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
            x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
            y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
            f_SFc__g[g] = 1. - y_th
            # print x, y_th, x1, x2, y1, y2
            f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
            f_hDIG.append(f_hDIG__g[g])
            f_mDIG.append(f_mDIG__g[g])
            f_SFc.append(f_SFc__g[g])
            # print g, f_hDIG__g[g], f_mDIG__g[g], f_SFc__g[g]
        # stats per mtype:
        if len(f_hDIG):
            mean_f_hDIG = np.mean(f_hDIG)
            median_f_hDIG = np.median(f_hDIG)
            std_f_hDIG = np.std(f_hDIG)
            max_f_hDIG = np.max(f_hDIG)
            min_f_hDIG = np.min(f_hDIG)
        else:
            mean_f_hDIG = 0.
            median_f_hDIG = 0.
            std_f_hDIG = 0.
            max_f_hDIG = 0.
            min_f_hDIG = 0.
        if len(f_mDIG):
            mean_f_mDIG = np.mean(f_mDIG)
            median_f_mDIG = np.median(f_mDIG)
            std_f_mDIG = np.std(f_mDIG)
            max_f_mDIG = np.max(f_mDIG)
            min_f_mDIG = np.min(f_mDIG)
        else:
            mean_f_mDIG = 0.
            median_f_mDIG = 0.
            std_f_mDIG = 0.
            max_f_mDIG = 0.
            min_f_mDIG = 0.
        aux_DIG = np.array(f_hDIG) + np.array(f_mDIG)
        if len(aux_DIG):
            mean_f_DIG = np.mean(aux_DIG)
            median_f_DIG = np.median(aux_DIG)
            std_f_DIG = np.std(aux_DIG)
            max_f_DIG = np.max(aux_DIG)
            min_f_DIG = np.min(aux_DIG)
        else:
            mean_f_DIG = 0.
            median_f_DIG = 0.
            std_f_DIG = 0.
            max_f_DIG = 0.
            min_f_DIG = 0.
        if len(f_SFc):
            mean_f_SFc = np.mean(f_SFc)
            median_f_SFc = np.median(f_SFc)
            std_f_SFc = np.std(f_SFc)
            max_f_SFc = np.max(f_SFc)
            min_f_SFc = np.min(f_SFc)
        else:
            mean_f_SFc = 0.
            median_f_SFc = 0.
            std_f_SFc = 0.
            max_f_SFc = 0.
            min_f_SFc = 0.
        print '<f_hDIG>:%.2f (med:%.2f sigma:%.2f max:%.2f min:%.2f)' % (mean_f_hDIG, median_f_hDIG, std_f_hDIG, max_f_hDIG, min_f_hDIG)
        print '<f_mDIG>:%.2f (med:%.2f sigma:%.2f max:%.2f min:%.2f)' % (mean_f_mDIG, median_f_mDIG, std_f_mDIG, max_f_mDIG, min_f_mDIG)
        print '<f_DIG>:%.2f (med:%.2f sigma:%.2f max:%.2f min:%.2f)' % (mean_f_DIG, median_f_DIG, std_f_DIG, max_f_DIG, min_f_DIG)
        print '<f_SFc>:%.2f (med:%.2f sigma:%.2f max:%.2f min:%.2f)' % (mean_f_SFc, median_f_SFc, std_f_SFc, max_f_SFc, min_f_SFc)
    f_hDIG = np.array([f_hDIG__g[g] for g in gals])
    f_mDIG = np.array([f_mDIG__g[g] for g in gals])
    f_DIG = f_hDIG + f_mDIG
    f_SFc = np.array([f_SFc__g[g] for g in gals])
    mean_f_hDIG = np.mean(f_hDIG)
    median_f_hDIG = np.median(f_hDIG)
    std_f_hDIG = np.std(f_hDIG)
    max_f_hDIG = np.max(f_hDIG)
    min_f_hDIG = np.min(f_hDIG)
    mean_f_mDIG = np.mean(f_mDIG)
    median_f_mDIG = np.median(f_mDIG)
    std_f_mDIG = np.std(f_mDIG)
    max_f_mDIG = np.max(f_mDIG)
    min_f_mDIG = np.min(f_mDIG)
    mean_f_DIG = np.mean(f_DIG)
    median_f_DIG = np.median(f_DIG)
    std_f_DIG = np.std(f_DIG)
    max_f_DIG = np.max(f_DIG)
    min_f_DIG = np.min(f_DIG)
    mean_f_SFc = np.mean(f_SFc)
    median_f_SFc = np.median(f_SFc)
    std_f_SFc = np.std(f_SFc)
    max_f_SFc = np.max(f_SFc)
    min_f_SFc = np.min(f_SFc)
    print 'all:'
    print '<f_hDIG>:%.2f (med:%.2f sigma:%.2f max:%.2f min:%.2f)' % (mean_f_hDIG, median_f_hDIG, std_f_hDIG, max_f_hDIG, min_f_hDIG)
    print '<f_mDIG>:%.2f (med:%.2f sigma:%.2f max:%.2f min:%.2f)' % (mean_f_mDIG, median_f_mDIG, std_f_mDIG, max_f_mDIG, min_f_mDIG)
    print '<f_DIG>:%.2f (med:%.2f sigma:%.2f max:%.2f min:%.2f)' % (mean_f_DIG, median_f_DIG, std_f_DIG, max_f_DIG, min_f_DIG)
    print '<f_SFc>:%.2f (med:%.2f sigma:%.2f max:%.2f min:%.2f)' % (mean_f_SFc, median_f_SFc, std_f_SFc, max_f_SFc, min_f_SFc)

    sel_sample__gz = sel['gals_sample__z']
    sel_ba_low = sel['gals__ba_z']['ba_low']
    sel_ba_mid = sel['gals__ba_z']['ba_mid']
    sel_ba_high = sel['gals__ba_z']['ba_high']
    sel_gals_ba_low__gz = np.bitwise_and(sel_ba_low, sel_sample__gz)
    sel_gals_ba_mid__gz = np.bitwise_and(sel_ba_mid, sel_sample__gz)
    sel_gals_ba_high__gz = np.bitwise_and(sel_ba_high, sel_sample__gz)
    sels_ba = [sel_sample__gz, sel_gals_ba_low__gz, sel_gals_ba_mid__gz, sel_gals_ba_high__gz]
    cumulfHa__g = {}
    for k in ALL.ba_keys:
        v = sel['gals__ba'][k]
        for g in ALL.califaID__g[v]:
            sel_sample__z = ALL.get_gal_prop(g, sel_sample__gz)
            WHa = np.ma.masked_array(ALL.get_gal_prop(g, ALL.W6563__z), mask=~sel_sample__z)
            fHa = np.ma.masked_array(ALL.get_gal_prop(g, ALL.f6563__z), mask=~sel_sample__z)
            xm, ym = ma_mask_xyz(np.ma.log10(WHa), fHa)
            fHa_cumul, fHa_tot = cumul_bins(ym.compressed(), xm.compressed(), logWHa_bins)  # , fHa_tot)
            cumulfHa__g[g] = fHa_cumul
    ALL.cumulfHa__gz = cumulfHa__g


def summary(args, ALL, sel, gals, mask_name):
    class_names = args.class_names
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

    tmp_sel_class = {'z':{}, 'yx':{}}
    print '# WHa classif: %s' % class_names
    for c in class_names:
        tmp_sel_class['z'][c] = (sel['WHa']['z'][c] & sel_gals_sample__gz)
        tmp_sel_class['yx'][c] = (sel['WHa']['yx'][c] & sel_gals_sample__gyx)
        print '\tTotal zones %s: %d' % (c, tmp_sel_class['z'][c].astype('int').sum())
        print '\tTotal spaxels %s: %d' % (c, tmp_sel_class['yx'][c].astype('int').sum())
    N_class = {}
    N_class_notmasked = {}
    N_tot = {}
    N_tot_notmasked = {}
    for g in gals:
        N_x = ALL.get_gal_prop_unique(g, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(g, ALL.N_y)
        N_spaxel = N_y * N_x
        N_zone = ALL.get_gal_prop_unique(g, ALL.N_zone)
        N_class[g] = {'z':{}, 'yx':{}}
        N_class_notmasked[g] = {'z':{}, 'yx':{}}
        N_tot[g] = {'z':0, 'yx':0}
        N_tot_notmasked[g] = {'z':0, 'yx':0}
        tmp_sel_gal__z = ALL.get_gal_prop(g, sel_gals_sample__gz)
        tmp_sel_gal__yx = ALL.get_gal_prop(g, sel_gals_sample__gyx)
        tmp_sel_gal_class = {'z':{}, 'yx':{}}
        for c in class_names:
            tmp_sel_gal_class['z'][c] = ALL.get_gal_prop(g, tmp_sel_class['z'][c])
            tmp_sel_gal_class['yx'][c] = ALL.get_gal_prop(g, tmp_sel_class['yx'][c])
            N_class[g]['z'][c] = ALL.get_gal_prop(g, sel['WHa']['z'][c]).astype('int').sum()
            N_class_notmasked[g]['z'][c] = tmp_sel_gal_class['z'][c].astype('int').sum()
            N_class[g]['yx'][c] = ALL.get_gal_prop(g, sel['WHa']['z'][c]).astype('int').sum()
            N_class_notmasked[g]['yx'][c] = tmp_sel_gal_class['z'][c].astype('int').sum()
            N_tot[g]['z'] += N_class[g]['z'][c]
            N_tot_notmasked[g]['z'] += N_class_notmasked[g]['z'][c]
            N_tot[g]['yx'] += N_class[g]['yx'][c]
            N_tot_notmasked[g]['yx'] += N_class_notmasked[g]['yx'][c]
        print g
        print '\tN_zones: %d | N_spaxels: %d' % (N_zone, N_spaxel)
        for c in class_names:
            print '\t%s:' % c
            print '\t\tN_zones: %d | N_zones_notmasked: %d' % (N_class[g]['z'][c], N_class_notmasked[g]['z'][c])
            print '\t\tN_spaxels: %d | N_spaxels_notmasked: %d' % (N_class[g]['yx'][c], N_class_notmasked[g]['yx'][c])
        print '\tNtot_zones_class: %d | Ntot_zones_class_notmasked: %d' % (N_tot[g]['z'], N_tot_notmasked[g]['z'])
        print '\tNtot_spaxels_class: %d | Ntot_spaxels_class_notmasked: %d' % (N_tot[g]['yx'], N_tot_notmasked[g]['yx'])


def cumul_bins(y, x, x_bins, y_tot=None):
    y_cumul = np.zeros_like(x_bins)
    if y_tot is None:
        y_tot = y.sum()
    for i, v in enumerate(x_bins):
        m = (x < v)
        if m.astype(int).sum() > 0:
            y_cumul[i] = y[m].sum()/y_tot
    return y_cumul, y_tot


def plot_text_classes_ax(ax, args, x=0.98, y_ini=0.98, y_spac=0.1, fs=14):
    plot_text_ax(ax, r'W${}_{H\alpha}$ $\leq$ %d $\AA$' % args.class_thresholds[0], x, y_ini, fs, 'top', 'left', args.class_colors[0])
    n_th = len(args.class_thresholds)
    y = y_ini
    for i, c in enumerate(args.class_names[1:]):
        y -= y_spac
        if i < (n_th - 1):
            plot_text_ax(ax, r'%d $\AA$ $<$ W${}_{H\alpha}$ $\leq$ %d $\AA$' % (args.class_thresholds[i], args.class_thresholds[i+1]), x, y, fs, 'top', 'left', args.class_colors[i+1])
        else:
            plot_text_ax(ax, r'W${}_{H\alpha}$ $>$ %d $\AA$' % args.class_thresholds[i], x, y, fs, 'top', 'left', args.class_colors[i+1])


def fig_logSBHa_logWHa_histograms(args):
    ALL, sel, gals = args.ALL, args.sel, args.gals
    sel_gals_sample__gz = sel['gals_sample__z']

    if (sel_gals_sample__gz).any():
        W6563__gz = ALL.W6563__z
        SB6563__gz = ALL.SB6563__z
        dist__gz = ALL.zoneDistance_HLR

        sel_WHa = {}
        for c in args.class_names:
            sel_WHa[c] = np.bitwise_and(sel['WHa']['z'][c], sel_gals_sample__gz)

        x = np.ma.log10(W6563__gz)
        y = np.ma.log10(SB6563__gz)
        z = dist__gz
        xm, ym, zm = ma_mask_xyz(x=x, y=y, z=z, mask=~sel_gals_sample__gz)
        # f = plot_setup(width=latex_column_width, aspect=1)
        f = plt.figure(figsize=(8, 8))

        x_ds = [xm[sel_WHa[c]].compressed() for c in args.class_names]
        y_ds = [ym[sel_WHa[c]].compressed() for c in args.class_names]
        axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logWHa_range, logSBHa_range, 50, 50,
                                             figure=f, c=args.class_colors, scatter=False, s=1,
                                             ylabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]',
                                             xlabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]', histtype='step')
        axS.xaxis.set_major_locator(MultipleLocator(0.5))
        axS.xaxis.set_minor_locator(MultipleLocator(0.1))
        axS.yaxis.set_major_locator(MultipleLocator(0.5))
        axS.yaxis.set_minor_locator(MultipleLocator(0.1))
        axH1.xaxis.set_major_locator(MultipleLocator(0.5))
        axH1.xaxis.set_minor_locator(MultipleLocator(0.1))
        axH2.xaxis.set_major_locator(MaxNLocator(3))
        axH1.yaxis.set_major_locator(MaxNLocator(3))
        axH2.yaxis.set_major_locator(MultipleLocator(0.5))
        axH2.yaxis.set_minor_locator(MultipleLocator(0.1))
        # aux_ax = axH2.twiny()
        # plot_histo_ax(aux_ax, ym.compressed(), stats_txt=False, kwargs_histo=dict(histtype='step', orientation='horizontal', normed=False, range=logSBHa_range, color='k', lw=2, ls='-'))
        # aux_ax.xaxis.set_major_locator(MaxNLocator(3))
        # plt.setp(aux_ax.xaxis.get_majorticklabels(), rotation=270)
        # plot_text_classes_ax(axH1, args, x=0.98, y_ini=0.98, y_spac=0.11, fs=14)
        # plot_text_ax(axH2, r'all zones', 0.98, 0.98, 14, 'top', 'right', 'k')
        scater_kwargs = dict(c=zm, s=3, vmax=3, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
        sc = axS.scatter(xm, ym, **scater_kwargs)
        cbaxes = f.add_axes([0.6, 0.17, 0.12, 0.02])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[0, 1, 2, 3], orientation='horizontal')
        cb.set_label(r'R [HLR]', fontsize=14)
        axS.axhline(y=np.log10(SFc_Zhang_threshold), color='k', linestyle='-.', lw=2)
        axS.set_xlim(logWHa_range)
        axS.set_ylim(logSBHa_range)
        axS.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        axS.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        xbins = np.linspace(logWHa_range[0], logWHa_range[-1], 40)
        yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), xbins)
        yMedian = prc[2]
        y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
        axS.plot(bin_center, yMedian, 'k-', lw=2)
        for y_prc in y_12sigma:
            axS.plot(bin_center, y_prc, 'k--', lw=2)
        axS.grid()
        f.savefig('fig_logSBHa_logWHa_histograms.png', dpi=_dpi_choice, transparent=_transp_choice)
        plt.close(f)


def fig_logxi_logWHa_histograms(args):
    ALL, sel, gals = args.ALL, args.sel, args.gals
    sel_gals_sample__gz = sel['gals_sample__z']

    if (sel_gals_sample__gz).any():
        W6563__gz = ALL.W6563__z
        # SB6563__gz = ALL.SB6563__z
        dist__gz = ALL.zoneDistance_HLR
        log_L6563__z = np.ma.log10(ALL.L6563__z)
        log_L6563_expected_hDIG__z = ALL.log_L6563_expected_HIG__z
        xi__z = log_L6563__z - log_L6563_expected_hDIG__z

        sel_WHa = {}
        for c in args.class_names:
            sel_WHa[c] = np.bitwise_and(sel['WHa']['z'][c], sel_gals_sample__gz)

        x = np.ma.log10(W6563__gz)
        y = xi__z
        z = dist__gz
        xm, ym, zm = ma_mask_xyz(x=x, y=y, z=z, mask=~sel_gals_sample__gz)
        # f = plt.figure(figsize=(8, 8))
        f = plot_setup(width=latex_column_width, aspect=1)
        gs = gridspec.GridSpec(4, 4, left=0.15, bottom=0.15, right=0.98, wspace=0., hspace=0.)
        axH1 = plt.subplot(gs[0, 0:3])
        axH2 = plt.subplot(gs[1:, -1])
        axS = plt.subplot(gs[1:, 0:-1])

        fs = 6
        x_ds = [xm[sel_WHa[c]].compressed() for c in args.class_names]
        y_ds = [ym[sel_WHa[c]].compressed() for c in args.class_names]

        axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logWHa_range, logWHa_range, 50, 50,
                                             axScatter=axS, axHistx=axH1, axHisty=axH2,
                                             figure=f, c=args.class_colors, scatter=False, s=0.2,
                                             ylabel=r'$\log\ \xi^{obs}$',
                                             xlabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]', histtype='step')

        axS.xaxis.set_major_locator(MultipleLocator(0.5))
        axS.xaxis.set_minor_locator(MultipleLocator(0.1))
        axS.yaxis.set_major_locator(MultipleLocator(0.5))
        axS.yaxis.set_minor_locator(MultipleLocator(0.1))

        axH1.xaxis.set_major_locator(MultipleLocator(0.5))
        axH1.xaxis.set_minor_locator(MultipleLocator(0.1))
        axH1.yaxis.set_major_locator(MaxNLocator(3, prune='lower'))
        axH1.yaxis.set_minor_locator(MaxNLocator(6))

        axH2.xaxis.set_major_locator(MaxNLocator(3, prune='lower'))
        axH2.xaxis.set_minor_locator(MaxNLocator(6))
        axH2.yaxis.set_major_locator(MultipleLocator(0.5))
        axH2.yaxis.set_minor_locator(MultipleLocator(0.1))

        # aux_ax = axH2.twiny()
        plot_histo_ax(axH2, ym.compressed(), stats_txt=False, kwargs_histo=dict(histtype='step', orientation='horizontal', normed=False, range=logWHa_range, color='k', lw=1, ls='-'))
        # axH2.xaxis.set_major_locator(MaxNLocator(3))
        plt.setp(axH2.xaxis.get_majorticklabels(), rotation=270)
        plot_text_classes_ax(axH1, args, x=1.01, y_ini=0.98, y_spac=0.15, fs=fs)
        plot_text_ax(axH2, r'all zones', 0.97, 0.98, fs+2, 'top', 'right', 'k')
        scater_kwargs = dict(rasterized=True, c=zm, s=1, vmax=2.5, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
        sc = axS.scatter(xm, ym, **scater_kwargs)
        cbaxes = f.add_axes([0.18, 0.69, 0.28, 0.04])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[0., 0.5, 1., 1.5, 2., 2.5], orientation='horizontal')
        cb.set_label(r'$R$ [HLR]', fontsize=fs+2)
        # axS.axhline(y=np.log10(SFc_Zhang_threshold), color='k', linestyle='-.', lw=2)
        axS.set_xlim(logWHa_range)
        axS.set_ylim(logWHa_range)
        axS.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        axS.set_ylabel(r'$\log\ \xi$')
        xbins = np.linspace(logWHa_range[0], logWHa_range[-1], 40)
        yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), xbins)
        yMedian = prc[2]
        y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
        axS.plot(bin_center, yMedian, 'k-', lw=1)
        for y_prc in y_12sigma:
            axS.plot(bin_center, y_prc, 'k--', lw=1)
        axS.grid()
        axS.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
        axH1.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='on', right='off', labelbottom='off', labeltop='off', labelleft='on', labelright='off')
        axH2.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='on', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')
        # f.set_tight_layout(True)
        f.savefig('fig_logxi_logWHa_histograms.pdf', dpi=_dpi_choice, transparent=_transp_choice)
        plt.close(f)


def fig_maps_Hafluxsum(args, gals=None, multi=False, suffix='', drawHLR=True, Hafluxsum=False, size=None):
    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    cmap = cmap_discrete(colors=args.class_colors)
    row = -1
    N_cols = 4
    N_rows = 1
    if Hafluxsum:
        suffix += '_Hafluxsum'
        N_cols = 5
    if multi:
        N_rows = len(gals)
        # f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 4.6, N_rows * 4.3))
        if size is None:
            size = (latex_text_width, 1.2)
        # plt.rcParams.update({'font.family': 'Times New Roman', 'lines.linewidth': 0.5})
        # f, axArr = plt.subplots(N_rows, N_cols, figsize=size)
        f = plot_setup(width=size[0], aspect=size[1])
        gs = gridspec.GridSpec(N_rows, N_cols)
        row = 0
    elif isinstance(gals, str):
        gals = [ gals ]
    if gals is None:
        gals = args.gals
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)
        gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        ml_ba = ALL.get_gal_prop_unique(califaID, ALL.ml_ba)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        f6563__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.f6563__z), mask=~gal_sample__z)
        f4861__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.f4861__z), mask=~gal_sample__z)
        W6563__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.W6563__z), mask=~gal_sample__z)
        f6563__yx = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.f6563__yx).reshape(N_y, N_x), mask=~gal_sample__yx)
        SB6563__yx = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.SB6563__yx).reshape(N_y, N_x), mask=~gal_sample__yx)
        W6563__yx = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x), mask=~gal_sample__yx)
        if multi:
            if Hafluxsum:
                ax1 = plt.subplot(gs[row, 0])
                ax2 = plt.subplot(gs[row, 1])
                ax3 = plt.subplot(gs[row, 2])
                ax4 = plt.subplot(gs[row, 3])
                ax5 = plt.subplot(gs[row, 4])
                # (ax1, ax2, ax3, ax4, ax5) = axArr[row]
            else:
                ax1 = plt.subplot(gs[row, 0])
                ax2 = plt.subplot(gs[row, 1])
                ax3 = plt.subplot(gs[row, 2])
                ax4 = plt.subplot(gs[row, 3])
                # (ax1, ax2, ax3, ax4) = axArr[row]
        else:
            N_rows = 1
            # plt.rcParams.update({'font.family': 'Times New Roman', 'lines.linewidth': 0.5})
            # f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(18, 4))
            f = plot_setup(width=latex_text_width, aspect=1./golden_mean)
            gs = gridspec.GridSpec(N_rows, N_cols)
            if Hafluxsum:
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1])
                ax3 = plt.subplot(gs[2])
                ax4 = plt.subplot(gs[3])
                ax5 = plt.subplot(gs[4])
                # ax1, ax2, ax3, ax4, ax5 = axArr
            else:
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1])
                ax3 = plt.subplot(gs[2])
                ax4 = plt.subplot(gs[3])
                # ax1, ax2, ax3, ax4 = axArr
            nedname = get_NEDName_by_CALIFAID(califaID)
            if nedname:
                nedname = nedname[0]
            else:
                fname = '/Users/lacerda/califa/legacy/q054/superfits/Bgstf6e/%s_synthesis_eBR_v20_q054.d22a512.ps03.k1.mE.CCM.Bgstf6e.fits' % califaID
                if os.path.isfile(fname):
                    nedname = fitsQ3DataCube(fname).galaxyName
                else:
                    fname = '/Users/lacerda/califa/legacy/q054/superfits/Bgstf6e/%s_synthesis_eBR_v20_q055.d22a512.ps03.k1.mE.CCM.Bgstf6e.fits' % califaID
                    if os.path.isfile(fname):
                        nedname = fitsQ3DataCube(fname).galaxyName
                    else:
                        nedname = 'not found'
            suptitle = '%s - %s ba:%.2f (ml_ba:%.2f) (%s): %d pixels (%d zones -' % (califaID, mto, ba, ml_ba, nedname, N_pixel, N_zone)
            map__yx = create_segmented_map_spaxels(args, ALL, califaID, sel['WHa']['yx'], sel_sample__gyx)
            tot_class = SB6563__yx.sum()
            frac_class = {}
            for i, c in enumerate(args.class_names):
                aux_sel = map__yx == i+1
                if aux_sel.sum() > 0:
                    N = SB6563__yx[aux_sel].sum()
                else:
                    N = 0
                frac_class[c] = 100. * N/tot_class
                # print c, frac_class[c]
                suptitle += ' %s: %.1f' % (c, frac_class[c])
            suptitle += ')'
            f.suptitle(suptitle)
        ax1.set_ylabel('%s' % mto, fontsize=10)
        # AXIS 1
        galimg = plt.imread(P.get_image_file(califaID))[::-1, :, :]
        print galimg.shape[0:-1]
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.imshow(galimg, origin='lower', aspect='equal')
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.02, 0.98, 10, 'top', 'left', color='w')
        ax1.xaxis.set_major_locator(MaxNLocator(4))
        ax1.xaxis.set_minor_locator(MaxNLocator(8))
        ax1.yaxis.set_major_locator(MaxNLocator(4))
        ax1.yaxis.set_minor_locator(MaxNLocator(8))
        ax1.tick_params(axis='both', which='both', direction='in', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        # AXIS 2
        x = np.ma.log10(SB6563__yx)
        im = ax2.imshow(x, vmin=logSBHa_range[0], vmax=6.5, cmap=plt.cm.copper_r, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        ax2.xaxis.set_major_locator(MaxNLocator(4))
        ax2.xaxis.set_minor_locator(MaxNLocator(8))
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        ax2.yaxis.set_minor_locator(MaxNLocator(8))
        ax2.tick_params(axis='both', which='both', direction='in', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        # cb.set_label(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        # AXIS 3
        x = np.ma.log10(W6563__yx)
        im = ax3.imshow(x, vmin=logWHa_range[0], vmax=logWHa_range[1], cmap=plt.cm.copper_r, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax3)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        ax3.xaxis.set_major_locator(MaxNLocator(4))
        ax3.xaxis.set_minor_locator(MaxNLocator(8))
        ax3.yaxis.set_major_locator(MaxNLocator(4))
        ax3.yaxis.set_minor_locator(MaxNLocator(8))
        ax3.tick_params(axis='both', which='both', direction='in', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        # cb.set_label(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        # AXIS 4
        x = W6563__yx
        im = ax4.imshow(x, vmin=3, vmax=args.class_thresholds[-1], cmap='Spectral', **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax4)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        ax4.xaxis.set_major_locator(MaxNLocator(4))
        ax4.xaxis.set_minor_locator(MaxNLocator(8))
        ax4.yaxis.set_major_locator(MaxNLocator(4))
        ax4.yaxis.set_minor_locator(MaxNLocator(8))
        ax4.tick_params(axis='both', which='both', direction='in', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        # cb.set_label(r'W${}_{H\alpha}$ [$\AA$]')
        if Hafluxsum:
            # iS = np.argsort(W6563__z)
            # xm, ym = ma_mask_xyz(W6563__z, f6563__z, mask=~gal_sample__z)
            # ytot = ym.sum()
            # ymcumsum = ym[iS].compressed().cumsum()
            ax5.plot(ALL.logWHa_bins__w, ALL.cumulfHa__gRw[califaID][0])
            # ax5.plot(np.ma.log10(xm[iS].compressed()), ymcumsum/ytot)
            ax5.set_ylim([0, 1])
            ax5.set_xlim([-0.5, 2])
            ax5.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
            ax5.set_ylabel(r'frac. F(H$\alpha$)')
            for th in args.class_thresholds:
                ax5.axvline(x=np.log10(th), c='k', ls='--')
            ax5.xaxis.set_major_locator(MultipleLocator(0.5))
            ax5.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax5.yaxis.set_major_locator(MultipleLocator(0.2))
            ax5.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax5.yaxis.grid(which='both')
            ax5.set_title('Cumulative Profiles')
            ax5.xaxis.set_major_locator(MaxNLocator(4))
            ax5.xaxis.set_minor_locator(MaxNLocator(8))
            ax5.yaxis.set_major_locator(MaxNLocator(4))
            ax5.yaxis.set_minor_locator(MaxNLocator(8))
            ax5.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        if drawHLR:
            DrawHLRCircleInSDSSImage(ax1, HLR_pix, pa, ba, lw=0.5, bins=[0.5, 1, 1.5, 2, 2.5, 3])  #, center_coord=(y0, x0))
            for ax in [ax2, ax3, ax4]:
                DrawHLRCircle(ax, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=0.5, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        if row <= 0:
            ax1.set_title('SDSS stamp', fontsize=10)
            # ax2.set_title(r'$\log$ W${}_{H\alpha}$ map')
            # ax3.set_title(r'$\log\ \Sigma_{H\alpha}$ map')
            ax2.set_title(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]', fontsize=10)
            ax3.set_title(r'$\log$ W${}_{H\alpha}$ [$\AA$]', fontsize=10)
            ax4.set_title(r'W${}_{H\alpha}$ [$\AA$]', fontsize=10)
            # ax4.set_title(r'classification map', fontsize=18)
        if multi:
            row += 1
        else:
            f.tight_layout(rect=[0, 0.01, 1, 0.98])
            # f.subplots_adjust(left=0.05, right=0.96, bottom=0.05, top=0.95, hspace=0.08, wspace=0.3)
            f.savefig('fig_maps-%s.pdf' % califaID, dpi=_dpi_choice, transparent=_transp_choice)
            plt.close(f)
    if multi:
        # f.subplots_adjust(left=0.05, right=0.96, bottom=0.05, top=0.95, hspace=0.08, wspace=0.35)
        gs.update(left=0.05, right=0.96, bottom=0.05, top=0.95, hspace=0.02, wspace=0.35)
        f.savefig('fig_maps_class%s.pdf' % suffix, dpi=_dpi_choice, transparent=_transp_choice)
        plt.close(f)


def fig_maps_xi(args, gals=None, multi=False, suffix='', drawHLR=True, xi=False):
    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    cmap = cmap_discrete(colors=args.class_colors)
    row = -1
    N_cols = 4
    N_rows = 1
    if xi:
        suffix += '_xi'
        N_cols = 5
    if multi:
        N_rows = len(gals)
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(18, 22))
        multi = True
        row = 0
    elif isinstance(gals, str):
        gals = [ gals ]
    if gals is None:
        gals = args.gals
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)
        gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        ml_ba = ALL.get_gal_prop_unique(califaID, ALL.ml_ba)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        f6563__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.f6563__z), mask=~gal_sample__z)
        f4861__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.f4861__z), mask=~gal_sample__z)
        W6563__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.W6563__z), mask=~gal_sample__z)
        f6563__yx = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.f6563__yx).reshape(N_y, N_x), mask=~gal_sample__yx)
        SB6563__yx = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.SB6563__yx).reshape(N_y, N_x), mask=~gal_sample__yx)
        W6563__yx = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x), mask=~gal_sample__yx)
        if multi:
            if xi:
                (ax1, ax2, ax3, ax4, ax5) = axArr[row]
            else:
                (ax1, ax2, ax3, ax4) = axArr[row]
        else:
            N_rows = 1
            f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(18, 4))
            if xi:
                ax1, ax2, ax3, ax4, ax5 = axArr
            else:
                ax1, ax2, ax3, ax4 = axArr
            nedname = get_NEDName_by_CALIFAID(califaID)
            if nedname:
                nedname = nedname[0]
            else:
                fname = '/Users/lacerda/califa/legacy/q054/superfits/Bgstf6e/%s_synthesis_eBR_v20_q054.d22a512.ps03.k1.mE.CCM.Bgstf6e.fits' % califaID
                if os.path.isfile(fname):
                    nedname = fitsQ3DataCube(fname).galaxyName
                else:
                    fname = '/Users/lacerda/califa/legacy/q054/superfits/Bgstf6e/%s_synthesis_eBR_v20_q055.d22a512.ps03.k1.mE.CCM.Bgstf6e.fits' % califaID
                    if os.path.isfile(fname):
                        nedname = fitsQ3DataCube(fname).galaxyName
                    else:
                        nedname = 'not found'
            suptitle = '%s - %s ba:%.2f (ml_ba:%.2f) (%s): %d pixels (%d zones -' % (califaID, mto, ba, ml_ba, nedname, N_pixel, N_zone)
            map__yx = create_segmented_map_spaxels(args, ALL, califaID, sel['WHa']['yx'], sel_sample__gyx)
            tot_class = SB6563__yx.sum()
            frac_class = {}
            for i, c in enumerate(args.class_names):
                aux_sel = map__yx == i+1
                if aux_sel.sum() > 0:
                    N = SB6563__yx[aux_sel].sum()
                else:
                    N = 0
                frac_class[c] = 100. * N/tot_class
                suptitle += ' %s: %.1f' % (c, frac_class[c])
            suptitle += ')'
            f.suptitle(suptitle)
        ax1.set_ylabel('%s' % mto, fontsize=24)
        # AXIS 1
        galimg = plt.imread(P.get_image_file(califaID))[::-1, :, :]
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.imshow(galimg, origin='lower', aspect='equal')
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.02, 0.98, 16, 'top', 'left', color='w')
        # AXIS 2
        x = np.ma.log10(SB6563__yx)
        im = ax2.imshow(x, vmin=logSBHa_range[0], vmax=6.5, cmap=plt.cm.copper_r, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        # AXIS 3
        x = np.ma.log10(W6563__yx)
        im = ax3.imshow(x, vmin=logWHa_range[0], vmax=logWHa_range[1], cmap=plt.cm.copper_r, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax3)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        # AXIS 4
        x = W6563__yx
        im = ax4.imshow(x, vmin=3, vmax=args.class_thresholds[-1], cmap='Spectral', **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax4)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        if xi:
            log_L6563__yx = np.ma.masked_array(np.ma.log10(ALL.get_gal_prop(califaID, ALL.L6563__yx).reshape(N_y, N_x)), mask=~gal_sample__yx)
            log_L6563_expected_hDIG__yx = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.log_L6563_expected_HIG__yx).reshape(N_y, N_x), mask=~gal_sample__yx)
            xi__yx = log_L6563__yx - log_L6563_expected_hDIG__yx
            x = xi__yx
            im = ax5.imshow(x, cmap=plt.cm.copper_r, vmin=logWHa_range[0], vmax=logWHa_range[1], **dflt_kw_imshow)
            the_divider = make_axes_locatable(ax5)
            color_axis = the_divider.append_axes('right', size='5%', pad=0)
            cb = plt.colorbar(im, cax=color_axis)
        if drawHLR:
            DrawHLRCircleInSDSSImage(ax1, HLR_pix, pa, ba, lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
            for ax in [ax2, ax3, ax4]:
                DrawHLRCircle(ax, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        if row <= 0:
            ax1.set_title('SDSS stamp', fontsize=18)
            ax2.set_title(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]', fontsize=18)
            ax3.set_title(r'$\log$ W${}_{H\alpha}$ [$\AA$]', fontsize=18)
            ax4.set_title(r'W${}_{H\alpha}$ [$\AA$]', fontsize=18)
            if xi:
                ax5.set_title(r'$\log\ \xi$', fontsize=18)
        if multi:
            row += 1
        else:
            f.tight_layout(rect=[0, 0.01, 1, 0.98])
            f.savefig('fig_maps-%s.png' % califaID, dpi=_dpi_choice, transparent=_transp_choice)
            plt.close(f)
    if multi:
        f.subplots_adjust(left=0.05, right=0.96, bottom=0.05, top=0.95, hspace=0.08, wspace=0.3)
        f.savefig('fig_maps_class%s.png' % suffix, dpi=_dpi_choice, transparent=_transp_choice)
        plt.close(f)


def fig_model_WHa_bimodal_distrib(args):
    from scipy.optimize import curve_fit
    ALL, sel, gals = args.ALL, args.sel, args.gals

    def gauss(x, mu, sigma, A):
        return A * np.exp(-(x-mu)**2./2. / sigma**2.)

    def bimodal(x, *p):
        mu1,sigma1,A1,mu2,sigma2,A2 = p
        # mu1 = 0.15
        return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)

    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~sel['gals_sample__z'])
    histo, bin_edges = np.histogram(xm.compressed(), bins=50, range=[-3, 3])
    bin_center = (bin_edges[:-1] + bin_edges[1:])/2.

    coeff, _ = curve_fit(bimodal, bin_center, histo, p0 = [0, 0.5, 20000, 1, 0.5, 55000], maxfev=5000)
    f = plot_setup(latex_column_width, 1./golden_mean)
    ax = plot_histo_ax(f.gca(), xm.compressed(), stats_txt=False, fs=15, va='top', ha='left', pos_x=0.01, ini_pos_y=0.79, first=True, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='grey', range=[-3, 3], lw=3))
    ax.plot(bin_center, bimodal(bin_center, *coeff))
    ax.set_xlim(-1, 2.5)
    print r'$\langle x \rangle_1$:%.2f, $\sigma_1$:%.2f, $A_1$:%d' % (coeff[0], coeff[1], coeff[2])
    print r'$\langle x \rangle_1$:%.2f, $\sigma_1$:%.2f, $A_1$:%d' % (coeff[3], coeff[4], coeff[5])
    plot_text_ax(plt.gca(), r'$\langle x \rangle_1$:%.2f, $\sigma_1$:%.2f, $A_1$:%d' % (coeff[0], coeff[1], coeff[2]), pos_x=0.01, pos_y=0.99, va='top', ha='left', fs=8)
    plot_text_ax(plt.gca(), r'$\langle x \rangle_2$:%.2f, $\sigma_2$:%.2f, $A_2$:%d' % (coeff[3], coeff[4], coeff[5]), pos_x=0.01, pos_y=0.89, va='top', ha='left', fs=8)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    f.tight_layout(rect=[0, 0.01, 1, 0.98])
    f.savefig('bimodalWHa_model.pdf', dpi=_dpi_choice, transparent=_transp_choice)


def fig_WHa_histograms_per_morftype_and_radius_FO(args, gals):
    print '#################################################'
    print '# fig_WHa_histograms_per_morftype_and_radius_FO #'
    print '#################################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']

    ml_ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_zone[i])) for i, ba in enumerate(ALL.ml_ba))))

    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']
    sel_ba_FO = (ml_ba > 0.7)
    # sel_ba_EO = (ml_ba < 0.4)
    sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z']
    sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z']
    sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z']
    N_rows, N_cols = 7, 4
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 10))
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    x_dataset__col_mt = [[],[],[],[]]
    x_dataset_FO__col_mt = [[],[],[],[]]
    for i, k in enumerate(mtype_labels):
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_sample__gz)
        x_dataset__col_mt[0].append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(m_aux, sel_ba_FO)
        x_dataset_FO__col_mt[0].append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())

        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_Rin__gz)
        x_dataset__col_mt[1].append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(m_aux, sel_ba_FO)
        x_dataset_FO__col_mt[1].append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())

        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_Rmid__gz)
        x_dataset__col_mt[2].append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(m_aux, sel_ba_FO)
        x_dataset_FO__col_mt[2].append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())

        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_Rout__gz)
        x_dataset__col_mt[3].append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(m_aux, sel_ba_FO)
        x_dataset_FO__col_mt[3].append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())

    N_zone = sel_sample__gz.astype('int').sum()
    (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43), (ax50, ax51, ax52, ax53), (ax60, ax61, ax62, ax63) = axArr
    axes_col0 = [ax00, ax10, ax20, ax30, ax40, ax50, ax60]
    axes_col1 = [ax01, ax11, ax21, ax31, ax41, ax51, ax61]
    axes_col2 = [ax02, ax12, ax22, ax32, ax42, ax52, ax62]
    axes_col3 = [ax03, ax13, ax23, ax33, ax43, ax53, ax63]

    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]
    logWHa_range = [-1, 2.5]
    ax = axes_cols[0][0]
    ax.set_title(r'%d galaxies - %d zones' % (len(gals), N_zone), fontsize=18, y=1.02)
    m_aux = sel_sample__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    m_aux = m_aux & sel_ba_FO
    plot_histo_ax(ax, np.ma.masked_array(xm, mask=~m_aux).compressed(), dataset_names='ba > 0.7', pos_x=0.02, ha='left', c='k', first=True, stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[1][0]
    ax.set_title(r'R $\leq$ 0.5 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rin__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    plot_histo_ax(ax, np.ma.masked_array(xm, mask=~(m_aux & sel_ba_FO)).compressed(), dataset_names='ba > 0.7', pos_x=0.02, ha='left', stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[2][0]
    ax.set_title(r'0.5 $<$ R $\leq$ 1 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rmid__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    plot_histo_ax(ax, np.ma.masked_array(xm, mask=~(m_aux & sel_ba_FO)).compressed(), dataset_names='ba > 0.7', pos_x=0.02, ha='left', stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[3][0]
    ax.set_title(r'R $>$ 1 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rout__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    plot_histo_ax(ax, np.ma.masked_array(xm, mask=~(m_aux & sel_ba_FO)).compressed(), dataset_names='ba > 0.7', pos_x=0.02, ha='left', stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    for col in range(N_cols):
        for i, mt_label in enumerate(mtype_labels):
            ax = axes_cols[col][i+1]
            if not col:
                if (mt_label == mtype_labels[-1]):
                    lab = r'$\geq$ Sc'
                else:
                    lab = mt_label
                ax.set_ylabel(lab, rotation=90)
            x = x_dataset__col_mt[col][i]
            x_FO = x_dataset_FO__col_mt[col][i]
            plot_histo_ax(ax, x, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=logWHa_range, lw=2))
            plot_histo_ax(ax, x_FO, ini_pos_y=0.97, pos_x=0.02, ha='left', fs=10, y_v_space=0.13, first=False, c=colortipo_darker[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo_darker[i], range=logWHa_range, lw=2))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.xaxis.set_minor_locator(MaxNLocator(25))
            ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            for th in args.class_thresholds:
                ax.axvline(x=np.ma.log10(th), c='k', ls='--')
            if i == (len(colortipo) - 1):
                ax.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
                if col == (N_cols - 1):
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))
            else:
                if not col:
                    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
                else:
                    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            ax.set_xlim(logWHa_range)

    f.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=20)
    f.savefig('fig_WHa_histograms_per_morftype_and_radius_FO.png', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'
    return ALL


def fig_SBHaWHa_scatter_per_morftype_and_ba(args, gals):
    print '###################################'
    print '# fig_WHa_per_morftype - ALL Gals #'
    print '###################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']

    ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_zone[i])) for i, ba in enumerate(ALL.ba))))
    ml_ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_zone[i])) for i, ba in enumerate(ALL.ml_ba))))

    N_zone = sel_sample__gz.astype('int').sum()
    z = ALL.zoneDistance_HLR
    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']
    sel_edgeon = (ml_ba <= 0.4)
    sel_middle = (ml_ba > 0.4) & (ml_ba <= 0.7)
    sel_faceon = (ml_ba > 0.7)
    sel_gals_sample_edgeon__gz = sel_sample__gz & sel_edgeon
    sel_gals_sample_mid__gz = sel_sample__gz & sel_middle
    sel_gals_sample_faceon__gz = sel_sample__gz & sel_faceon

    x_dataset__col_mt = [[],[],[],[]]
    y_dataset__col_mt = [[],[],[],[]]
    for i, k in enumerate(mtype_labels):
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_sample__gz)
        xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=np.ma.log10(ALL.SB6563__z), mask=~m_aux)
        x_dataset__col_mt[0].append(xm)
        y_dataset__col_mt[0].append(ym)
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_edgeon__gz)
        xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=np.ma.log10(ALL.SB6563__z), mask=~m_aux)
        x_dataset__col_mt[1].append(xm)
        y_dataset__col_mt[1].append(ym)
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_mid__gz)
        xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=np.ma.log10(ALL.SB6563__z), mask=~m_aux)
        x_dataset__col_mt[2].append(xm)
        y_dataset__col_mt[2].append(ym)
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_faceon__gz)
        xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=np.ma.log10(ALL.SB6563__z), mask=~m_aux)
        x_dataset__col_mt[3].append(xm)
        y_dataset__col_mt[3].append(ym)

    N_rows, N_cols = 7, 4
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 16))
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43), (ax50, ax51, ax52, ax53), (ax60, ax61, ax62, ax63) = axArr
    axes_col0 = [ax00, ax10, ax20, ax30, ax40, ax50, ax60]
    axes_col1 = [ax01, ax11, ax21, ax31, ax41, ax51, ax61]
    axes_col2 = [ax02, ax12, ax22, ax32, ax42, ax52, ax62]
    axes_col3 = [ax03, ax13, ax23, ax33, ax43, ax53, ax63]
    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]

    ax = axes_cols[0][0]
    ax.set_title(r'%d galaxies - %d zones' % (len(gals), N_zone), fontsize=18, y=1.02)
    m_aux = sel_sample__gz
    xm, ym, zm = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=np.ma.log10(ALL.SB6563__z), z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=3, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    cbaxes = f.add_axes([0.8, 0.96, 0.12, 0.02])
    cb = plt.colorbar(sc, cax=cbaxes, ticks=[0, 1, 2, 3], orientation='horizontal')
    cb.set_label(r'R [HLR]', fontsize=14)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlim(logWHa_range)
    ax.set_ylim(logSBHa_range)
    ax.grid()
    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')

    ax = axes_cols[1][0]
    ax.set_title(r'ba $\leq$ 0.4', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_edgeon__gz
    xm, ym, zm = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=np.ma.log10(ALL.SB6563__z), z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=3, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlim(logWHa_range)
    ax.set_ylim(logSBHa_range)
    ax.grid()
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[2][0]
    ax.set_title(r'0.4 $<$ ba $\leq$ 0.7 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_mid__gz
    xm, ym, zm = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=np.ma.log10(ALL.SB6563__z), z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=3, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlim(logWHa_range)
    ax.set_ylim(logSBHa_range)
    ax.grid()
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[3][0]
    ax.set_title(r'ba > 0.7', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_faceon__gz
    xm, ym, zm = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=np.ma.log10(ALL.SB6563__z), z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=3, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlim(logWHa_range)
    ax.set_ylim(logSBHa_range)
    ax.grid()
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    for col in range(N_cols):
        for i, mt_label in enumerate(mtype_labels):
            ax = axes_cols[col][i+1]
            x = x_dataset__col_mt[col][i]
            y = y_dataset__col_mt[col][i]
            xm, ym, zm = ma_mask_xyz(x, y, z)
            scater_kwargs = dict(c=zm, s=3, vmax=3, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
            sc = ax.scatter(xm, ym, **scater_kwargs)
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            plot_text_ax(ax, mt_label, 0.98, 0.01, 20, 'bottom', 'right', c=colortipo[i])
            if i == (len(colortipo) - 1):
                ax.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
                ax.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
                ax.set_xticks([-1, 0, 1, 2, 3])
            else:
                if not col:
                    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
                else:
                    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            ax.set_xlim(logWHa_range)
            ax.set_ylim(logSBHa_range)
            ax.grid()

    ax00.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
    f.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    f.savefig('fig_SBHaWHa_scatter_per_morftype_and_ba.png', dpi=_dpi_choice, transparent=_transp_choice)

    print '######################################################'
    print '# END ################################################'
    print '######################################################'
    return ALL


def fig_BPT_per_morftype_and_R(args, gals):
    print '##############################'
    print '# fig_BPT_per_morftype_and_R #'
    print '##############################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']

    ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_zone[i])) for i, ba in enumerate(ALL.ba))))
    ml_ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_zone[i])) for i, ba in enumerate(ALL.ml_ba))))

    N_zone = sel_sample__gz.astype('int').sum()
    z = ALL.W6563__z
    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']

    sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z']
    sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z']
    sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z']

    BPTLines = ['4861', '5007', '6563', '6583']
    f__lgz = {'%s' % l: np.ma.masked_array(getattr(ALL, 'f%s__z' % l), mask=~sel_sample__gz) for l in BPTLines}
    O3Hb__gz = np.ma.log10(f__lgz['5007']/f__lgz['4861'])
    N2Ha__gz = np.ma.log10(f__lgz['6583']/f__lgz['6563'])

    x_dataset__col_mt = [[],[],[],[]]
    y_dataset__col_mt = [[],[],[],[]]
    for i, k in enumerate(mtype_labels):
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_sample__gz)
        xm, ym = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, mask=~m_aux)
        x_dataset__col_mt[0].append(xm)
        y_dataset__col_mt[0].append(ym)
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_Rin__gz)
        xm, ym = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, mask=~m_aux)
        x_dataset__col_mt[1].append(xm)
        y_dataset__col_mt[1].append(ym)
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_Rmid__gz)
        xm, ym = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, mask=~m_aux)
        x_dataset__col_mt[2].append(xm)
        y_dataset__col_mt[2].append(ym)
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_Rout__gz)
        xm, ym = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, mask=~m_aux)
        x_dataset__col_mt[3].append(xm)
        y_dataset__col_mt[3].append(ym)

    N_rows, N_cols = 7, 4
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 16))
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43), (ax50, ax51, ax52, ax53), (ax60, ax61, ax62, ax63) = axArr
    axes_col0 = [ax00, ax10, ax20, ax30, ax40, ax50, ax60]
    axes_col1 = [ax01, ax11, ax21, ax31, ax41, ax51, ax61]
    axes_col2 = [ax02, ax12, ax22, ax32, ax42, ax52, ax62]
    axes_col3 = [ax03, ax13, ax23, ax33, ax43, ax53, ax63]
    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]

    extent = [-1.5, 1, -1.2, 1.2]
    # scater_kwargs = dict(c=zm, s=3, vmax=3, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
    # scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    # scater_kwargs = dict(s=3, marker='o', edgecolor='none')

    ax = axes_cols[0][0]
    ax.set_title(r'%d galaxies - %d zones' % (len(gals), N_zone), fontsize=18, y=1.02)
    m_aux = sel_sample__gz
    xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    cbaxes = f.add_axes([0.8, 0.96, 0.12, 0.02])
    cb = plt.colorbar(sc, cax=cbaxes, orientation='horizontal')  #, ticks=[0, 1, 2, 3])
    cb.set_label(r'W${}_{H\alpha}$ [$\AA$]', fontsize=14)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    # ax.grid()
    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')

    ax = axes_cols[1][0]
    ax.set_title(r'R $\leq$ 0.5 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rin__gz
    xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    # ax.grid()
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[2][0]
    ax.set_title(r'0.5 $<$ R $\leq$ 1 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rmid__gz
    xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    # ax.grid()
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[3][0]
    ax.set_title(r'R $>$ 1 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rout__gz
    xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    # ax.grid()
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    for col in range(N_cols):
        for i, mt_label in enumerate(mtype_labels):
            ax = axes_cols[col][i+1]
            x = x_dataset__col_mt[col][i]
            y = y_dataset__col_mt[col][i]
            xm, ym, zm = ma_mask_xyz(x, y, z)
            scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
            sc = ax.scatter(xm, ym, **scater_kwargs)
            ax.xaxis.set_minor_locator(minorLocator)
            ax.yaxis.set_minor_locator(minorLocator)
            plot_text_ax(ax, mt_label, 0.98, 0.01, 20, 'bottom', 'right', c=colortipo[i])
            if i == (len(colortipo) - 1):
                if not col:
                    ax.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
                else:
                    ax.tick_params(axis='both', which='both', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
                # ax.set_xlabel(r'$\log\ [NII]/H\alpha$')
                # ax.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
                ax.set_xticks([-1, 0, 1, 2, 3])
            else:
                if not col:
                    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
                else:
                    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            ax.set_xlim(extent[0:2])
            ax.set_ylim(extent[2:4])
            # ax.grid()

    L = Lines()
    for ax_col in axes_cols:
        for ax in ax_col:
            ax.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
            ax.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
            ax.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')

    # ax00.set_ylabel(r'$\log\ [OIII]/H\beta$')
    plot_text_ax(ax00, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax00, 'K03', 0.55, 0.02, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax00, 'K03', 0.85, 0.02, 20, 'bottom', 'left', 'k')

    f.subplots_adjust(left=0.06, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)

    f.text(0.5, 0.04, r'$\log\ [NII]/H\alpha$', ha='center', fontsize=20)
    f.text(0.01, 0.5, r'$\log\ [OIII]/H\beta$', va='center', rotation='vertical', fontsize=20)

    f.savefig('fig_BPT_per_morftype_and_R.png', dpi=_dpi_choice, transparent=_transp_choice)

    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_BPT_per_R(args, gals):
    print '#################'
    print '# fig_BPT_per_R #'
    print '#################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z'] & sel['SN_BPT3']['z']
    sel_sample__gyx = sel['gals_sample__yx'] & sel['SN_BPT3']['yx']

    N_zone = sel_sample__gz.astype('int').sum()
    z = ALL.W6563__z

    sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z'] & sel['SN_BPT3']['z']
    sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z'] & sel['SN_BPT3']['z']
    sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z'] & sel['SN_BPT3']['z']
    BPTLines = ['4861', '5007', '6563', '6583']
    f__lgz = {'%s' % l: np.ma.masked_array(getattr(ALL, 'f%s__z' % l), mask=~sel_sample__gz) for l in BPTLines}
    O3Hb__gz = np.ma.log10(f__lgz['5007']/f__lgz['4861'])
    N2Ha__gz = np.ma.log10(f__lgz['6583']/f__lgz['6563'])

    N_rows, N_cols = 4, 4
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(15, 15))
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    ax00, ax01, ax02, ax03 = axArr[0]
    ((ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33)) = axArr[1:]
    ax__R = [[ax10, ax20, ax30], [ax11, ax21, ax31], [ax12, ax22, ax32], [ax13, ax23, ax33]]

    extent = [-1.5, 0.7, -1.2, 1.2]

    ax = ax00
    ax.set_title(r'%d galaxies - %d zones' % (len(gals), N_zone), fontsize=18, y=1.02)
    m_aux = sel_sample__gz
    xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    cbaxes = f.add_axes([0.1, 0.95, 0.8, 0.03])
    cb = plt.colorbar(sc, cax=cbaxes, orientation='horizontal')  #, ticks=[0, 1, 2, 3])
    cb.set_label(r'W${}_{H\alpha}$ [$\AA$]', fontsize=16)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    # ax.grid()
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='on', right='off', labelbottom='off', labeltop='off', labelleft='on', labelright='off')
    ax.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))

    ax = ax01
    ax.set_title(r'R $\leq$ 0.5 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rin__gz
    xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    # ax.grid()
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    ax.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))

    ax = ax02
    ax.set_title(r'0.5 $<$ R $\leq$ 1 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rmid__gz
    xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    # ax.grid()
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    ax.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))

    ax = ax03
    ax.set_title(r'R $>$ 1 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rout__gz
    xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    # ax.grid()
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    # ax.set_xticks([-1, 0, 1, 2, 3])
    ax.xaxis.set_major_locator(MaxNLocator(5))

    sels_R = [sel['gals_sample__z'], sel['gals_sample_Rin__z'], sel['gals_sample_Rmid__z'], sel['gals_sample_Rout__z']]
    for iR, sel_R in enumerate(sels_R):
        axs = ax__R[iR]
        ax = axs[0]
        m_aux = sel_R & sel['WHa']['z']['hDIG']
        xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
        scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
        sc = ax.scatter(xm, ym, **scater_kwargs)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
        # ax.grid()
        ax.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))

        ax = axs[1]
        m_aux = sel_R & sel['WHa']['z']['mDIG']
        xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
        scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
        sc = ax.scatter(xm, ym, **scater_kwargs)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
        # ax.grid()
        ax.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))

        ax = axs[2]
        m_aux = sel_R & sel['WHa']['z']['SFc']
        xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
        scater_kwargs = dict(c=zm, s=3, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
        sc = ax.scatter(xm, ym, **scater_kwargs)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
        # ax.grid()
        # ax.set_xticks([-1, 0, 1, 2, 3])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        if not iR:
            axs[0].tick_params(axis='both', which='both', bottom='off', top='off', left='on', right='off', labelbottom='off', labeltop='off', labelleft='on', labelright='off')
            axs[1].tick_params(axis='both', which='both', bottom='off', top='off', left='on', right='off', labelbottom='off', labeltop='off', labelleft='on', labelright='off')
            axs[2].tick_params(axis='both', which='both', bottom='on', top='off', left='on', right='off', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
        else:
            axs[0].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            axs[1].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            axs[2].tick_params(axis='both', which='both', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')

    L = Lines()
    for ax in [ax00, ax01, ax02, ax03, ax10, ax11, ax12, ax13, ax20, ax21, ax22, ax23, ax30, ax31, ax32, ax33]:
        ax.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        ax.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')

    # ax00.set_ylabel(r'$\log\ [OIII]/H\beta$')
    plot_text_ax(ax00, 'S06', 0.32, 0.01, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax00, 'K03', 0.62, 0.01, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax00, 'K01', 0.99, 0.01, 20, 'bottom', 'right', 'k')

    f.subplots_adjust(left=0.06, right=0.95, bottom=0.10, top=0.88, hspace=0.0, wspace=0.0)

    f.text(0.5, 0.02, r'$\log\ [NII]/H\alpha$', ha='center', fontsize=20)
    f.text(0.01, 0.5, r'$\log\ [OIII]/H\beta$', va='center', rotation='vertical', fontsize=20)

    f.savefig('fig_BPT_per_R.png', dpi=_dpi_choice, transparent=_transp_choice)

    print '######################################################'
    print '# END ################################################'
    print '######################################################'
    sel_BPT_S06SFc__gz = np.bitwise_and(sel['BPT']['z']['S06'], sel_sample__gz)
    N_SFc_WHa_S06BPT__gz = np.bitwise_and(sel_BPT_S06SFc__gz, sel['WHa']['z']['SFc']).astype('int').sum()
    N_hDIG_WHa_S06BPT_zones = np.bitwise_and(~sel_BPT_S06SFc__gz, sel['WHa']['z']['hDIG']).astype('int').sum()
    print 'zones: N_SFc(S06 & WHa) = %d (%.2f%%)' % (N_SFc_WHa_S06BPT__gz, 100.*N_SFc_WHa_S06BPT__gz/sel['WHa']['z']['SFc'].astype('int').sum())
    print 'zones: N_hDIG(S06 & WHa) = %d (%.2f%%)' % (N_hDIG_WHa_S06BPT_zones, 100.*N_hDIG_WHa_S06BPT_zones/sel['WHa']['z']['hDIG'].astype('int').sum())
    sel_BPT_S06SFc__gyx = np.bitwise_and(sel['BPT']['yx']['S06'], sel_sample__gyx)
    N_SFc_WHa_S06BPT__gyx = np.bitwise_and(sel_BPT_S06SFc__gyx, sel['WHa']['yx']['SFc']).astype('int').sum()
    N_hDIG_WHa_S06BPT_pixels = np.bitwise_and(~sel_BPT_S06SFc__gyx, sel['WHa']['yx']['hDIG']).astype('int').sum()
    print 'pixels: N_SFc(S06 & WHa) = %d (%.2f%%)' % (N_SFc_WHa_S06BPT__gyx, 100.*N_SFc_WHa_S06BPT__gyx/sel['WHa']['yx']['SFc'].astype('int').sum())
    print 'pixels: N_hDIG(S06 & WHa) = %d (%.2f%%)' % (N_hDIG_WHa_S06BPT_pixels, 100.*N_hDIG_WHa_S06BPT_pixels/sel['WHa']['yx']['hDIG'].astype('int').sum())
    sel_BPT_K03SFc__gz = np.bitwise_and(sel['BPT']['z']['K03'], sel_sample__gz)
    N_SFc_WHa_K03BPT__gz = np.bitwise_and(sel_BPT_K03SFc__gz, sel['WHa']['z']['SFc']).astype('int').sum()
    N_hDIG_WHa_K03BPT_zones = np.bitwise_and(~sel_BPT_K03SFc__gz, sel['WHa']['z']['hDIG']).astype('int').sum()
    print 'zones: N_SFc(K03 & WHa) = %d (%.2f%%)' % (N_SFc_WHa_K03BPT__gz, 100.*N_SFc_WHa_K03BPT__gz/sel['WHa']['z']['SFc'].astype('int').sum())
    print 'zones: N_hDIG(K03 & WHa) = %d (%.2f%%)' % (N_hDIG_WHa_K03BPT_zones, 100.*N_hDIG_WHa_K03BPT_zones/sel['WHa']['z']['hDIG'].astype('int').sum())
    sel_BPT_K03SFc__gyx = np.bitwise_and(sel['BPT']['yx']['K03'], sel_sample__gyx)
    N_SFc_WHa_K03BPT__gyx = np.bitwise_and(sel_BPT_K03SFc__gyx, sel['WHa']['yx']['SFc']).astype('int').sum()
    N_hDIG_WHa_K03BPT_pixels = np.bitwise_and(~sel_BPT_K03SFc__gyx, sel['WHa']['yx']['hDIG']).astype('int').sum()
    print 'pixels: N_SFc(K03 & WHa) = %d (%.2f%%)' % (N_SFc_WHa_K03BPT__gyx, 100.*N_SFc_WHa_K03BPT__gyx/sel['WHa']['yx']['SFc'].astype('int').sum())
    print 'pixels: N_hDIG(K03 & WHa) = %d (%.2f%%)' % (N_hDIG_WHa_K03BPT_pixels, 100.*N_hDIG_WHa_K03BPT_pixels/sel['WHa']['yx']['hDIG'].astype('int').sum())


def fig_BPT_per_class(args, gals):
    print '#####################'
    print '# fig_BPT_per_class #'
    print '#####################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z'] & sel['SN_BPT3']['z']
    sel_sample__gyx = sel['gals_sample__yx'] & sel['SN_BPT3']['yx']

    N_zone = sel_sample__gz.astype('int').sum()
    z = ALL.W6563__z

    BPTLines = ['4861', '5007', '6563', '6583']
    f__lgz = {'%s' % l: np.ma.masked_array(getattr(ALL, 'f%s__z' % l), mask=~sel_sample__gz) for l in BPTLines}
    O3Hb__gz = np.ma.log10(f__lgz['5007']/f__lgz['4861'])
    N2Ha__gz = np.ma.log10(f__lgz['6583']/f__lgz['6563'])

    N_rows, N_cols = 1, 4
    fs = 6
    # f, axArr = plt.subplots(N_rows, N_cols, figsize=(15, 5))
    f = plot_setup(width=latex_text_width, aspect=1/3.)
    gs = gridspec.GridSpec(N_rows, N_cols, left=0.08, bottom=0.18, right=0.98, top=0.85, wspace=0., hspace=0.)
    # ax0, ax1, ax2, ax3 = axArr
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    ax_class = {
        '%s' % args.class_names[0]: ax3,
        '%s' % args.class_names[1]: ax2,
        '%s' % args.class_names[2]: ax1,
    }

    extent = [-1.5, 0.7, -1.2, 1.5]

    ax = ax0
    ax.set_title(r'%d galaxies - %d zones' % (len(gals), N_zone), fontsize=fs+4, y=1.02)
    m_aux = sel_sample__gz
    xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
    scater_kwargs = dict(rasterized=True, c=zm, s=1, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    cbaxes = add_subplot_axes(ax0, [0.04, 0.92, 0.50, 0.05])
    cb = plt.colorbar(sc, cax=cbaxes, orientation='horizontal')  #, ticks=[0, 1, 2, 3])
    cb.set_label(r'W${}_{H\alpha}$ [$\AA$]', fontsize=fs, labelpad=-0.5)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    # ax.grid()
    ax.xaxis.set_major_locator(MaxNLocator(5, prune='both'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom='on', labeltop='off', labelleft='on', labelright='off')

    z = ALL.zoneDistance_HLR
    for c in args.class_names:
        ax = ax_class[c]
        ax.set_title(r'%s' % c, fontsize=fs+4, y=1.02)
        m_aux = sel_sample__gz & sel['WHa']['z'][c]
        xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
        scater_kwargs = dict(rasterized=True, c=zm, s=1, vmax=2.5, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
        sc = ax.scatter(xm, ym, **scater_kwargs)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
        # ax.grid()
        ax.xaxis.set_major_locator(MaxNLocator(5, prune='both'))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom='on', labeltop='off', labelleft='off', labelright='off')

    cbaxes = add_subplot_axes(ax2, [0.06, 0.92, 0.50, 0.05])
    cb = plt.colorbar(sc, cax=cbaxes, ticks=[0., 0.5, 1., 1.5, 2., 2.5], orientation='horizontal')
    cb.set_label(r'$R$ [HLR]', fontsize=fs, labelpad=-0.5)

    L = Lines()
    for ax in [ax0, ax1, ax2, ax3]:
        ax.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        ax.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')

    # ax00.set_ylabel(r'$\log\ [OIII]/H\beta$')
    plot_text_ax(ax0, 'S06', 0.32, 0.01, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax0, 'K03', 0.6, 0.01, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax0, 'K01', 0.91, 0.01, fs+2, 'bottom', 'right', 'k')

    # f.subplots_adjust(left=0.06, right=0.95, bottom=0.15, top=0.90, hspace=0.0, wspace=0.0)
    # gs.update((left=0.06, right=0.95, bottom=0.15, top=0.90, hspace=0.0, wspace=0.0)

    f.text(0.5, 0.02, r'$\log\ [NII]/H\alpha$', ha='center', fontsize=fs+4)
    f.text(0.01, 0.5, r'$\log\ [OIII]/H\beta$', va='center', rotation='vertical', fontsize=fs+4)

    f.savefig('fig_BPT_per_class.pdf', dpi=_dpi_choice, transparent=_transp_choice)

    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def calcMixingLineOnBPT(N2Ha_1, O3Hb_1, HaHb_1, N2Ha_2, O3Hb_2, HaHb_2, df=0.1):
    f = np.arange(0.,1.+df,df)
    N2Ha = f * N2Ha_1 + (1. - f) * N2Ha_2
    HbHa = f * (1./HaHb_1) + (1. - f) * (1./HaHb_2)
    HaHb = 1. / HbHa
    O3Hb = (f * O3Hb_1 / HaHb_1 + (1. - f) * O3Hb_2 / HaHb_2) / (f * (1./HaHb_1) + (1. - f) * (1./HaHb_2))

    return f , N2Ha , O3Hb, HaHb


def fig_BPT_mixed(args, gals):
    print '#################'
    print '# fig_BPT_mixed #'
    print '#################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z'] & sel['SN_BPT3']['z']
    sel_sample__gyx = sel['gals_sample__yx'] & sel['SN_BPT3']['yx']

    N_zone = sel_sample__gz.astype('int').sum()
    z = ALL.W6563__z

    sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z'] & sel['SN_BPT3']['z']
    sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z'] & sel['SN_BPT3']['z']
    sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z'] & sel['SN_BPT3']['z']
    BPTLines = ['4861', '5007', '6563', '6583']
    f__lgz = {'%s' % l: np.ma.masked_array(getattr(ALL, 'f%s__z' % l), mask=~sel_sample__gz) for l in BPTLines}
    O3Hb__gz = np.ma.log10(f__lgz['5007']/f__lgz['4861'])
    N2Ha__gz = np.ma.log10(f__lgz['6583']/f__lgz['6563'])

    fs = 6
    # f = plt.figure(figsize=(8, 8))
    f = plot_setup(width=latex_column_width, aspect=1.)  #1./golden_mean)
    ax = f.gca()

    extent = [-1.5, 0.7, -1.2, 1.2]

    m_aux = sel_gals_sample_Rout__gz & sel['WHa']['z']['mDIG']
    xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, rasterized=True, s=1, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    L = Lines()
    ax.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
    ax.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
    ax.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
    N2Ha_2 = -0.0955  # -0.03  # -0.1
    O3Hb_2 = 0.1255  # 0.18  # 0.1
    _, N2Ha , O3Hb, _ = calcMixingLineOnBPT(-0.4, -0.7, 3.71, N2Ha_2, O3Hb_2, 2.86, 0.2)
    # ax.plot(N2Ha,O3Hb,'ko-', mec='none', ms=3.)
    _, N2Ha , O3Hb, _ = calcMixingLineOnBPT(-0.5, -0.3, 3.75, N2Ha_2, O3Hb_2, 2.86, 0.2)
    # ax.plot(N2Ha,O3Hb,'ko-', mec='none', ms=3.)
    _, N2Ha , O3Hb, _ = calcMixingLineOnBPT(-0.6, 0., 3.67, N2Ha_2, O3Hb_2, 2.86, 0.2)
    # ax.plot(N2Ha,O3Hb,'ko-', mec='none', ms=3.)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    plot_text_ax(ax, 'S06', 0.45, 0.01, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'K03', 0.60, 0.01, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'K01', 0.86, 0.01, fs+2, 'bottom', 'right', 'k')
    cbaxes = f.add_axes([0.69, 0.9, 0.25, 0.05])
    cb = plt.colorbar(sc, cax=cbaxes, orientation='horizontal')  #, ticks=[0, 1, 2, 3])
    cb.set_label(r'W${}_{H\alpha}$ [$\AA$]', fontsize=fs+2)

    ax.set_xlabel(r'$\log\ [NII]/H\alpha$')
    ax.set_ylabel(r'$\log\ [OIII]/H\beta$')
    f.tight_layout()
    f.savefig('fig_BPT_mixed.pdf', dpi=_dpi_choice, transparent=_transp_choice)


def fig_cumul_fHaWHa_per_morftype_and_R_gals(args, gals):
    print '#######################################'
    print '# fig_cumul_fHaWHa_per_morftype_and_R_gals #'
    print '#######################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_zone[i])) for i, ba in enumerate(ALL.ba))))
    ml_ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_zone[i])) for i, ba in enumerate(ALL.ml_ba))))
    mt = np.hstack(list(itertools.chain(list(itertools.repeat(mt, ALL.N_zone[i])) for i, mt in enumerate(ALL.mt))))
    N_zone = sel_sample__gz.astype('int').sum()
    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']

    N_rows, N_cols = 7, 4
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 16))
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43), (ax50, ax51, ax52, ax53), (ax60, ax61, ax62, ax63) = axArr
    axes_col0 = [ax00, ax10, ax20, ax30, ax40, ax50, ax60]
    axes_col1 = [ax01, ax11, ax21, ax31, ax41, ax51, ax61]
    axes_col2 = [ax02, ax12, ax22, ax32, ax42, ax52, ax62]
    axes_col3 = [ax03, ax13, ax23, ax33, ax43, ax53, ax63]
    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]

    extent = [-1.5, 1, -1.2, 1.2]

    col_titles = [
        r'%d galaxies - %d zones' % (len(gals), N_zone),
        r'R $\leq$ 0.5 HLR',
        r'0.5 $<$ R $\leq$ 1 HLR',
        r'R $>$ 1 HLR'
    ]

    sels_R = [sel['gals_sample__z'], sel['gals_sample_Rin__z'], sel['gals_sample_Rmid__z'], sel['gals_sample_Rout__z']]
    logWHa_bins = ALL.logWHa_bins__w
    cumulfHa__g_R = ALL.cumulfHa__gRw
    for col in range(N_cols):
        ax = axes_cols[col][0]
        x = np.hstack([logWHa_bins for g in ALL.califaID__g[sel['gals']]])
        y__gb = np.array([cumulfHa__g_R[g][col] for g in ALL.califaID__g[sel['gals']]])
        ax.plot(x, np.hstack(y__gb), c='darkgrey')
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.set_title(col_titles[col])
        ax.plot(logWHa_bins, np.median(y__gb, axis=0), ls='-', c='k')
        prc = np.percentile(y__gb, q=[16, 84], axis=0)
        ax.plot(logWHa_bins, prc[1], ls='--', c='k')
        ax.plot(logWHa_bins, prc[0], ls='--', c='k')
        for th in args.class_thresholds:
            ax.axvline(x=np.log10(th), c='k', ls='--')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.grid(which='both')
        if not col:
            ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
        else:
            ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(0, 1)
        for i, mt_label in enumerate(mtype_labels):
            ax = axes_cols[col][i+1]
            y__gb = np.array([cumulfHa__g_R[g][col] for g in ALL.califaID__g[sel['gals__mt'][mt_label]]])
            x = np.hstack([logWHa_bins for g in ALL.califaID__g[sel['gals__mt'][mt_label]]])
            ax.plot(x, np.hstack(y__gb), ls='', marker='o', c=colortipo_lighter[i])
            ax.plot(logWHa_bins, np.median(y__gb, axis=0), ls='-', c='k')
            prc = np.percentile(y__gb, q=[16, 84], axis=0)
            ax.plot(logWHa_bins, prc[1], ls='--', c='k')
            ax.plot(logWHa_bins, prc[0], ls='--', c='k')
            for th in args.class_thresholds:
                ax.axvline(x=np.log10(th), c='k', ls='--')
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.grid(which='both')
            plot_text_ax(ax, mt_label, 0.98, 0.01, 20, 'bottom', 'right', c=colortipo[i])
            if i == (len(colortipo) - 1):
                if not col:
                    ax.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
                else:
                    ax.tick_params(axis='both', which='both', bottom='on',
                                   top='off', left='off', right='off',
                                   labelbottom='on', labeltop='off', labelleft='off', labelright='off')
                if col == (N_cols - 1):
                    ax.xaxis.set_major_locator(MaxNLocator(6))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(6, prune='upper'))
            else:
                if not col:
                    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
                else:
                    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            ax.set_xlim(-0.5, 2)
            ax.set_ylim(0, 1)
    f.subplots_adjust(left=0.06, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=20)
    f.text(0.01, 0.5, r'H$\alpha$ flux fraction', va='center', rotation='vertical', fontsize=20)
    f.savefig('fig_cumul_fHaWHa_per_morftype_and_R_gals.png', dpi=_dpi_choice, transparent=_transp_choice)
    plt.close(f)

    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_integrated_BPT_color_Wfrac(args, gals, Wfrac=0.5):
    print '##################################'
    print '# fig_integrated_BPT_color_Wfrac #'
    print '##################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    iO3Hb__g = np.ma.masked_all(len(gals), dtype='float')
    iN2Ha__g = np.ma.masked_all(len(gals), dtype='float')
    O3Hb__g = np.ma.masked_all(len(gals), dtype='float')
    N2Ha__g = np.ma.masked_all(len(gals), dtype='float')
    Wfrac__g = np.ma.masked_all(len(gals), dtype='float')

    from pytu.functions import linearInterpol

    for i_g, califaID in enumerate(gals):
        # print califaID
        gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
        # gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)

        f4861__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.f4861__z), mask=~gal_sample__z)
        f5007__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.f5007__z), mask=~gal_sample__z)
        f6563__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.f6563__z), mask=~gal_sample__z)
        f6583__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.f6583__z), mask=~gal_sample__z)
        O3Hb__g[i_g] = np.ma.log10(f5007__z.sum()/f4861__z.sum())
        N2Ha__g[i_g] = np.ma.log10(f6583__z.sum()/f6563__z.sum())

        f4861 = ALL.get_gal_prop_unique(califaID, ALL.integrated_f4861)
        f5007 = ALL.get_gal_prop_unique(califaID, ALL.integrated_f5007)
        f6563 = ALL.get_gal_prop_unique(califaID, ALL.integrated_f6563)
        f6583 = ALL.get_gal_prop_unique(califaID, ALL.integrated_f6583)
        iO3Hb__g[i_g] = np.ma.log10(f5007/f4861)
        iN2Ha__g[i_g] = np.ma.log10(f6583/f6563)

        cumulfHa__w = ALL.cumulfHa__gRw[califaID][0]
        logWHa_bins__w = ALL.logWHa_bins__w
        sel_Wfrac = np.where(cumulfHa__w < Wfrac, True, False)
        N = sel_Wfrac.astype(int).sum()
        x1 = 0
        y1 = logWHa_bins__w[0]
        if N > 0:
            x1 = cumulfHa__w[sel_Wfrac][-1]
            y1 = logWHa_bins__w[sel_Wfrac][-1]
        x2 = 1
        y2 = logWHa_bins__w[-1]
        N = (~sel_Wfrac).astype(int).sum()
        if N > 0:
            x2 = cumulfHa__w[~sel_Wfrac][0]
            y2 = logWHa_bins__w[~sel_Wfrac][0]
        Wfrac__g[i_g] = 10**linearInterpol(x1, x2, y1, y2, Wfrac)
        # print 'Wfrac(%.1f): %.2f' % (Wfrac, Wperc__g[i_g])

    L = Lines()
    N_rows = 1
    N_cols = 2
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
    # ax
    #
    # f = plt.figure(dpi=200, figsize=(6, 5))
    ax = f.gca()
    _x = [N2Ha__g, iN2Ha__g]
    _y = [O3Hb__g, iO3Hb__g]
    _title = ['summed', 'integrated']
    sc = []

    for i, ax in enumerate(axArr):
        x = _x[i]
        y = _y[i]
        ax.set_title(_title[i])
        xm, ym, zm = ma_mask_xyz(x, y, Wfrac__g)
        extent = [-1.5, 0.5, -1, 1]
        sc_tmp = ax.scatter(xm, ym, c=zm, vmin=0.5, vmax=2, cmap='Spectral', s=20, marker='o', edgecolor='none')
        sc.append(sc_tmp)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
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
        ax.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        ax.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        plot_text_ax(ax, '%d %s' % (N, c), 0.01, 0.99, 15, 'top', 'left', 'k')
        plot_text_ax(ax, 'S06', 0.42, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax, 'K03', 0.67, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax, 'K01', 0.99, 0.02, 20, 'bottom', 'right', 'k')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    axArr[0].set_ylabel(r'$\log\ [OIII]/H\beta$')
    axArr[1].tick_params(axis='both', which='both', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')

    cbaxes = add_subplot_axes(axArr[1], [0.63, 0.98, 0.47, 0.06])
    cb = plt.colorbar(sc[0], cax=cbaxes, ticks=[-0.5, 0, 0.5, 1, 1.5, 2], orientation='horizontal')
    cb.ax.set_xlabel(r'$\log$ W${}_{%.0f}$ [$\AA$]' % (Wfrac * 100), fontsize=12)
    # cb.ax.xaxis.set_major_locator(MaxNLocator(4))

    f.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.92, wspace=0.0)

    f.text(0.5, 0.04, r'$\log\ [NII]/H\alpha$', ha='center', fontsize=20)

    f.savefig('fig_integrated_BPT_color_Wperc%.0f.png' % (Wfrac * 100), dpi=_dpi_choice, transparent=_transp_choice)
    plt.close(f)


def print_error_WHaSBHa(args, gals=None):
    print '#######################'
    print '# fig_WHaSBHa_profile #'
    print '#######################'
    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']

    W6563__gz = []
    elogf6563__gz = []
    mean_elogf6563_hDIG__g = []
    median_elogf6563_hDIG__g = []
    mean_elogf6563_SFc__g = []
    median_elogf6563_SFc__g = []

    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue
        gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
        W6563__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.W6563__z), mask=~gal_sample__z)
        f6563__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.f6563__z), mask=~gal_sample__z)
        ef6563__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.ef6563__z), mask=~gal_sample__z)
        elogf6563__z = np.log10(np.exp(1)) * ef6563__z/f6563__z
        W6563__gz.append(W6563__z)
        elogf6563__gz.append(elogf6563__z)
        N_hDIG = (W6563__z.compressed() <= args.class_thresholds[0]).astype('int').sum()
        N_SFc = (W6563__z.compressed() > args.class_thresholds[1]).astype('int').sum()
        if N_hDIG:
            sel_hDIG = (W6563__z <= args.class_thresholds[0])
            mean_elogf6563_hDIG = np.mean(elogf6563__z[sel_hDIG].compressed())
            median_elogf6563_hDIG = np.median(elogf6563__z[sel_hDIG].compressed())
            mean_elogf6563_hDIG__g.append(mean_elogf6563_hDIG)
            median_elogf6563_hDIG__g.append(median_elogf6563_hDIG)
            print califaID, ' - hDIG - ', mean_elogf6563_hDIG, median_elogf6563_hDIG
        if N_SFc:
            sel_SFc = (W6563__z > args.class_thresholds[1])
            mean_elogf6563_SFc = np.mean(elogf6563__z[sel_SFc].compressed())
            median_elogf6563_SFc = np.median(elogf6563__z[sel_SFc].compressed())
            mean_elogf6563_SFc__g.append(mean_elogf6563_SFc)
            median_elogf6563_SFc__g.append(median_elogf6563_SFc)
            print califaID, ' - SFc - ', mean_elogf6563_SFc, median_elogf6563_SFc
    stack_W6563__gz = np.hstack(W6563__gz)
    print '<ALL>:'
    print '     ERR logWHa - hDIG - <mean>: ', np.mean(mean_elogf6563_hDIG__g), ' - <median>: ', np.mean(median_elogf6563_hDIG__g), ' - med(median): ', np.median(median_elogf6563_hDIG__g)
    print '     ERR logWHa - SFc - <mean>: ', np.mean(mean_elogf6563_SFc__g), ' - <median>: ', np.mean(median_elogf6563_SFc__g), ' - med(median): ', np.median(median_elogf6563_SFc__g)


def fig_WHaSBHa_profile(args, gals=None, multi=False, size=None, suffix=''):
    print '#######################'
    print '# fig_WHaSBHa_profile #'
    print '#######################'
    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    cmap = cmap_discrete(colors=args.class_colors)
    row = -1
    N_cols = 2
    N_rows = 1
    fs = 6
    if multi:
        N_rows = len(gals)
        if size is None:
            size = (latex_column_width, 1.3)
        # plt.rcParams.update({'font.family': 'Times New Roman', 'lines.linewidth': 0.5})
        # f, axArr = plt.subplots(N_rows, N_cols, figsize=size)
        f = plot_setup(width=size[0], aspect=size[1])
        gs = gridspec.GridSpec(N_rows, N_cols)  #, hspace=0., wspace )
        # f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 4.4))
        row = 0
    elif isinstance(gals, str):
        gals = [ gals ]
    if gals is None:
        gals = args.gals
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue
        d_Mpc = ALL.get_gal_prop_unique(califaID, ALL.galDistance_Mpc)
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
        zoneDistance_HLR__z = ALL.get_gal_prop(califaID, ALL.zoneDistance_HLR)
        zoneArea_pc2 = ALL.get_gal_prop(califaID, ALL.zoneArea_pc2)
        SB6563__z = ALL.get_gal_prop(califaID, ALL.SB6563__z)
        W6563__z = ALL.get_gal_prop(califaID, ALL.W6563__z)
        W6563__yx = ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x)
        SB6563__yx = ALL.get_gal_prop(califaID, ALL.SB6563__yx).reshape(N_y, N_x)
        c6563__z = ALL.get_gal_prop(califaID, ALL.c6563__z)
        ef6563__z = ALL.get_gal_prop(califaID, ALL.ef6563__z)
        eL6563__z = F_to_L(ef6563__z, d_Mpc)/L_sun
        eSB6563__z = eL6563__z/(zoneArea_pc2 * 1e-6)

        mW6563__z, mSB6563__z = ma_mask_xyz(W6563__z, SB6563__z, mask=~gal_sample__z)
        mW6563__yx, mSB6563__yx = ma_mask_xyz(W6563__yx, SB6563__yx, mask=~gal_sample__yx)

        mc6563__z = np.ma.masked_array(c6563__z, mask=np.ma.getmaskarray(mW6563__z))
        mef6563__z = np.ma.masked_array(ef6563__z, mask=np.ma.getmaskarray(mW6563__z))
        meSB6563__z = np.ma.masked_array(eSB6563__z, mask=np.ma.getmaskarray(mW6563__z))

        meW6563__z = (mef6563__z/mc6563__z)
        melogW6563__z = np.log10(np.exp(1)) * meW6563__z/mW6563__z
        melogSB6563__z = np.log10(np.exp(1)) * meSB6563__z/mSB6563__z

        if multi:
            ax1 = plt.subplot(gs[row, 0])
            ax2 = plt.subplot(gs[row, 1])
            # (ax1, ax2) = axArr[row]
        else:
            N_rows = 1
            f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
            ax1, ax2 = axArr
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.98, 0.03, fs+2, 'bottom', 'right', color='k')

        x__z = zoneDistance_HLR__z
        y__z = np.ma.log10(mW6563__z)
        y__yx = np.ma.log10(mW6563__yx)
        y__r, npts = radialProfile(y__yx, args.R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'median', True)
        ax1.scatter(x__z, y__z, c=10**y__z, cmap='Spectral', s=1, vmin=args.class_thresholds[0], vmax=args.class_thresholds[1], **dflt_kw_scatter)
        # ax1.set_xlabel(r'R [HLR]', fontsize=fs+4)
        ax1.set_ylabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        ax1.grid()
        err_hDIG = np.median(melogW6563__z[mW6563__z <= args.class_thresholds[0]].compressed())
        err_SFc = np.median(melogW6563__z[mW6563__z > args.class_thresholds[1]].compressed())
        print califaID, 'ERR logWHa - hDIG: ', err_hDIG, ' - SFc: ', err_SFc
        # ax1.errorbar(2.25, 0.2, yerr=err_hDIG, fmt='.', capsize=3, c='r')
        # ax1.errorbar(2.25, 1.5, yerr=err_SFc, fmt='.', capsize=3, c='b')

        logW6563__z = y__z

        x__z = zoneDistance_HLR__z
        y__z = np.ma.log10(mSB6563__z)
        y__yx = np.ma.log10(mSB6563__yx)
        y__r, npts = radialProfile(y__yx, args.R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'median', True)
        sc = ax2.scatter(x__z, y__z, rasterized=True, c=10**logW6563__z, cmap='Spectral', s=1, vmin=args.class_thresholds[0], vmax=args.class_thresholds[1], **dflt_kw_scatter)
        ax2.axhline(y=np.log10(SFc_Zhang_threshold), c='k', ls='--')
        # ax2.set_xlabel(r'R [HLR]')
        ax2.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax2.grid()
        err_hDIG = np.median(melogSB6563__z[mW6563__z <= args.class_thresholds[0]].compressed())
        err_SFc = np.median(melogSB6563__z[mW6563__z > args.class_thresholds[1]].compressed())
        print '     ERR logSBHa - hDIG: ', err_hDIG, ' - SFc: ', err_SFc
        # ax2.errorbar(0.25, 4.25, yerr=err_hDIG, fmt='.', capsize=3, c='r')
        # ax2.errorbar(0.25, 6, yerr=err_SFc, fmt='.', capsize=3, c='b')

        if row <= 0:
            ax1.set_title(r'W${}_{H\alpha}$ profile', y=1.03)
            ax2.set_title(r'$\Sigma_{H\alpha}$ profile', y=1.03)
            # cbaxes = add_subplot_axes(ax1, [0.02, 1.19, 0.5, 0.09])
        ax1.set_xlim(0, 2.5)
        ax1.set_ylim(logWHa_range)
        ax2.set_xlim(0, 2.5)
        ax2.set_ylim(logSBHa_range)
        ax1.xaxis.set_major_locator(MaxNLocator(5))
        ax1.xaxis.set_minor_locator(MaxNLocator(10))
        ax1.yaxis.set_major_locator(MaxNLocator(5, prune='lower'))
        ax1.yaxis.set_minor_locator(MaxNLocator(10))
        ax2.xaxis.set_major_locator(MaxNLocator(5))
        ax2.xaxis.set_minor_locator(MaxNLocator(10))
        ax2.yaxis.set_major_locator(MaxNLocator(5, prune='lower'))
        ax2.yaxis.set_minor_locator(MaxNLocator(10))
        ax1.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='off', labeltop='off', labelleft='on', labelright='off')
        ax2.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='off', labeltop='off', labelleft='on', labelright='off')
        if row == (N_rows - 1):
            ax1.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
            ax2.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
            # cbaxes = f.add_axes([0.75, 0.30, 0.15, 0.02])
            # cb = plt.colorbar(sc, cax=cbaxes, ticks=[3.0, 8.5, 14.0], orientation='horizontal')
            # cb.ax.set_xlabel(r'W${}_{H\alpha}$ [$\AA$]', fontsize=fs+2)
        if multi:
            row += 1
        else:
            f.tight_layout(w_pad=0.05, h_pad=0)
            f.savefig('fig_WHaSBHa_profile-%s.pdf' % califaID, dpi=_dpi_choice, transparent=_transp_choice)
            plt.close(f)
    if multi:
        gs.update(hspace=0., wspace=0.37, left=0.15, right=0.97, bottom=0.1, top=0.94)
        # f.tight_layout(w_pad=0.05, h_pad=0)
        f.text(0.5, 0.03, r'$R$ [HLR]', ha='center', fontsize=fs+2)
        f.savefig('fig_WHaSBHa_profile%s.pdf' % suffix, dpi=_dpi_choice, transparent=_transp_choice)
        plt.close(f)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_cumul_fHaWHa_per_morftype(args, gals):
    print '#################################'
    print '# fig_cumul_fHaWHa_per_morftype #'
    print '#################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    N_zone = sel_sample__gz.astype('int').sum()
    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']

    # f = plt.figure(figsize=(8, 7))
    f = plot_setup(width=latex_column_width, aspect=1)  #1./golden_mean)
    # _, ind = np.unique(ALL.califaID__z, return_index=True)
    ax = f.gca()

    # sels_R = [sel['gals_sample__z'], sel['gals_sample_Rin__z'], sel['gals_sample_Rmid__z'], sel['gals_sample_Rout__z']]
    logWHa_bins = ALL.logWHa_bins__w
    cumulfHa__g_R = ALL.cumulfHa__gRw

    for i, mt_label in enumerate(mtype_labels):
        aux = []
        print mt_label
        for g in ALL.califaID__g[sel['gals__mt'][mt_label]]:
            if not g in args.gals:
                continue
            aux.append(cumulfHa__g_R[g][0])
        if len(aux) > 0:
            y__gb = np.array(aux)
            x = np.hstack([logWHa_bins for g in ALL.califaID__g[sel['gals__mt'][mt_label]]])
            # ax.plot(x, np.hstack(y__gb), ls='', marker='o', c=colortipo_lighter[i])
            ax.plot(logWHa_bins, np.median(y__gb, axis=0), ls='-', c=colortipo[i], lw=2)
            # ax.plot(logWHa_bins, np.mean(y__gb, axis=0), ls='--', c=colortipo[i], lw=3)
            # prc = np.percentile(y__gb, q=[16, 84], axis=0)
            # ax.plot(logWHa_bins, prc[1], ls='--', c=colortipo[i], alpha=0.5)
            # ax.plot(logWHa_bins, prc[0], ls='--', c=colortipo[i], alpha=0.5)
            y_pos = 0.98 - (i*0.1)
            if (i == len(mtype_labels) - 1):
                plot_text_ax(ax, r'$\geq$ Sc', 0.01, y_pos, 8, 'top', 'left', c=colortipo[i])
            else:
                plot_text_ax(ax, mt_label, 0.01, y_pos, 8, 'top', 'left', c=colortipo[i])
    for th in args.class_thresholds:
        ax.axvline(x=np.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.grid(which='both')
    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    # ax.set_title(mt_label, color=colortipo[i])
    ax.set_xlim(-1, 2.5)
    ax.set_ylim(0, 1)
    # f.subplots_adjust(left=0.06, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    ax.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    ax.set_ylabel(r'H$\alpha$ flux fraction')
    # f.subplots_adjust(left=0.open06, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    # f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=20)
    # f.text(0.01, 0.5, r'$\sum\ F_{H\alpha} (< $W${}_{H\alpha})$', va='center', rotation='vertical', fontsize=20)
    f.set_tight_layout(True)
    f.savefig('fig_cumul_fHaWHa_per_morftype.pdf', dpi=_dpi_choice, transparent=_transp_choice)
    plt.close(f)

    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_cumul_fHaWHa_per_morftype_and_R(args, gals):
    print '#################################'
    print '# fig_cumul_fHaWHa_per_morftype #'
    print '#################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    N_zone = sel_sample__gz.astype('int').sum()
    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']
    sels_R = [sel['gals_sample__z'], sel['gals_sample_Rin__z'], sel['gals_sample_Rout__z']]

    col_titles = [
        r'%d galaxies - %d zones' % (len(gals), N_zone),
        r'R $\leq$ 0.5 HLR',
        r'R $>$ 1 HLR'
    ]

    N_cols = len(sels_R)
    N_rows = 1
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))

    logWHa_bins = ALL.logWHa_bins__w
    cumulfHa__g_R = ALL.cumulfHa__gRw

    for iR, sel_R in enumerate(sels_R):
        ax = axArr[iR]
        for i, mt_label in enumerate(mtype_labels):
            y__gb = np.array([cumulfHa__g_R[g][iR] for g in ALL.califaID__g[sel['gals__mt'][mt_label]]])
            x = np.hstack([logWHa_bins for g in ALL.califaID__g[sel['gals__mt'][mt_label]]])
            ax.plot(logWHa_bins, np.median(y__gb, axis=0), ls='-', c=colortipo[i], lw=3)
            # ax.plot(x, np.hstack(y__gb), ls='', marker='o', c=colortipo_lighter[i])
            # ax.plot(logWHa_bins, np.mean(y__gb, axis=0), ls='--', c=colortipo[i], lw=3)
            # prc = np.percentile(y__gb, q=[16, 84], axis=0)
            # ax.plot(logWHa_bins, prc[1], ls='--', c=colortipo[i], alpha=0.5)
            # ax.plot(logWHa_bins, prc[0], ls='--', c=colortipo[i], alpha=0.5)
            if iR < (len(sels_R) - 1):
                ax.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))
            else:
                y_pos = -0.01 + (len(mtype_labels) - i - 1) * 0.1
                if (i == len(mtype_labels) - 1):
                    plot_text_ax(ax, r'$\geq$ Sc', 0.99, y_pos, 30, 'bottom', 'right', c=colortipo[i])
                else:
                    plot_text_ax(ax, mt_label, 0.99, y_pos, 30, 'bottom', 'right', c=colortipo[i])
                ax.xaxis.set_major_locator(MaxNLocator(5))
        for th in args.class_thresholds:
            ax.axvline(x=np.log10(th), c='k', ls='--')
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.grid(which='both')
        ax.set_title(col_titles[iR])
        ax.set_xlim(-1, 2.5)
        ax.set_ylim(0, 1)
    axArr[1].tick_params(axis='both', which='both', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')
    axArr[2].tick_params(axis='both', which='both', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')
    axArr[0].set_ylabel(r'H$\alpha$ flux fraction')
    f.subplots_adjust(left=0.06, right=0.95, bottom=0.15, top=0.9, wspace=0.0)
    f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=20)
    f.savefig('fig_cumul_fHaWHa_per_morftype_and_R.png', dpi=_dpi_choice, transparent=_transp_choice)
    plt.close(f)

    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_WHa_histograms_per_morftype_and_radius_cumulFHa(args, gals):
    print '#######################################################'
    print '# fig_WHa_histograms_per_morftype_and_radius_cumulFHa #'
    print '#######################################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']

    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']
    sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z']
    sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z']
    sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z']
    N_rows, N_cols = 7, 4
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 10))
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    N_zone = sel_sample__gz.astype('int').sum()
    (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43), (ax50, ax51, ax52, ax53), (ax60, ax61, ax62, ax63) = axArr
    axes_col0 = [ax00, ax10, ax20, ax30, ax40, ax50, ax60]
    axes_col1 = [ax01, ax11, ax21, ax31, ax41, ax51, ax61]
    axes_col2 = [ax02, ax12, ax22, ax32, ax42, ax52, ax62]
    axes_col3 = [ax03, ax13, ax23, ax33, ax43, ax53, ax63]

    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]
    logWHa_range = [-1, 2.5]
    ax = axes_cols[0][0]
    ax.set_title(r'All radii', fontsize=18, y=1.02)
    m_aux = sel_sample__gz
    xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=ALL.L6563__z, mask=~m_aux)
    tot_Ha = ym.sum()
    fracHa__c = {}
    for i, k in enumerate(args.class_names):
        y_pos = 0.98 - (i * 0.16)
        sel_aux = m_aux & sel['WHa']['z'][k]
        fracHa__c[k] = ALL.L6563__z[sel_aux].sum()/tot_Ha
        plot_text_ax(ax, r'$f_{%s}: %.3f$' % (k, fracHa__c[k]), 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
    plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    print '>>', xm.count()
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[1][0]
    ax.set_title(r'R $\leq$ 0.5 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rin__gz
    xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=ALL.L6563__z, mask=~m_aux)
    tot_Ha = ym.sum()
    fracHa__c = {}
    for i, k in enumerate(args.class_names):
        y_pos = 0.97 - (i * 0.16)
        sel_aux = m_aux & sel['WHa']['z'][k]
        fracHa__c[k] = ALL.L6563__z[sel_aux].sum()/tot_Ha
        plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
    plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    print '>>', xm.count()
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[2][0]
    ax.set_title(r'0.5 $<$ R $\leq$ 1 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rmid__gz
    xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=ALL.L6563__z, mask=~m_aux)
    tot_Ha = ym.sum()
    fracHa__c = {}
    for i, k in enumerate(args.class_names):
        y_pos = 0.97 - (i * 0.16)
        sel_aux = m_aux & sel['WHa']['z'][k]
        fracHa__c[k] = ALL.L6563__z[sel_aux].sum()/tot_Ha
        plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
    plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    print '>>', xm.count()
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[3][0]
    ax.set_title(r'R $>$ 1 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rout__gz
    xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=ALL.L6563__z, mask=~m_aux)
    tot_Ha = ym.sum()
    fracHa__c = {}
    for i, k in enumerate(args.class_names):
        y_pos = 0.97 - (i * 0.16)
        sel_aux = m_aux & sel['WHa']['z'][k]
        fracHa__c[k] = ALL.L6563__z[sel_aux].sum()/tot_Ha
        plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
    plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    print '>>', xm.count()
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    sels_R = [sel_sample__gz, sel_gals_sample_Rin__gz, sel_gals_sample_Rmid__gz, sel_gals_sample_Rout__gz]

    for col in range(N_cols):
        for i, mt_label in enumerate(mtype_labels):
            ax = axes_cols[col][i+1]
            if not col:
                if (mt_label == mtype_labels[-1]):
                    lab = r'$\geq$ Sc'
                else:
                    lab = mt_label
                ax.set_ylabel(lab, rotation=90)
            m_aux = np.bitwise_and(sel_gals_mt[mt_label], sels_R[col])
            xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=ALL.L6563__z, mask=~m_aux)
            print mt_label, col
            print ym.count()
            tot_Ha = ym.sum()
            fracHa__c = {}
            for j, k in enumerate(args.class_names):
                y_pos = 0.97 - (j * 0.16)
                sel_aux = m_aux & sel['WHa']['z'][k]
                print k, ALL.L6563__z[sel_aux].count()
                fracHa__c[k] = ALL.L6563__z[sel_aux].sum()/tot_Ha
                plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
            plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=logWHa_range, lw=2))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_minor_locator(MaxNLocator(20))
            ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            for th in args.class_thresholds:
                ax.axvline(x=np.ma.log10(th), c='k', ls='--')
            if i == (len(colortipo) - 1):
                ax.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
                if col == (N_cols - 1):
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
            else:
                if not col:
                    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
                else:
                    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            ax.set_xlim(logWHa_range)

    f.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=20)
    f.savefig('fig_WHa_histograms_per_morftype_and_radius_cumulFHa.png', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'
    return ALL


def fig_WHa_histograms_per_morftype_and_radius_cumulLHa(args, gals):
    print '#######################################################'
    print '# fig_WHa_histograms_per_morftype_and_radius_cumulFHa #'
    print '#######################################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']

    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']
    sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z']
    sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z']
    sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z']
    N_rows, N_cols = 7, 4
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 10))
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    N_zone = sel_sample__gz.astype('int').sum()
    (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43), (ax50, ax51, ax52, ax53), (ax60, ax61, ax62, ax63) = axArr
    axes_col0 = [ax00, ax10, ax20, ax30, ax40, ax50, ax60]
    axes_col1 = [ax01, ax11, ax21, ax31, ax41, ax51, ax61]
    axes_col2 = [ax02, ax12, ax22, ax32, ax42, ax52, ax62]
    axes_col3 = [ax03, ax13, ax23, ax33, ax43, ax53, ax63]

    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]
    logWHa_range = [-1, 2.5]
    ax = axes_cols[0][0]
    ax.set_title(r'All radii', fontsize=18, y=1.02)
    m_aux = sel_sample__gz
    xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=ALL.L6563__z, mask=~m_aux)
    tot_Ha = ym.sum()
    # y__gb = np.array([ALL.cumulfHa__gRw[g][0] for g in args.gals])
    # x = np.hstack([logWHa_bins for g in args.gals])
    # ax.plot(logWHa_bins, np.median(y__gb, axis=0), ls='-', c=colortipo[i], lw=3)
    for i, k in enumerate(args.class_names):
        y_pos = 0.98 - (i * 0.16)
        sel_aux = m_aux & sel['WHa']['z'][k]
        fracHa__c[k] = ALL.L6563__z[sel_aux].sum()/tot_Ha
        plot_text_ax(ax, r'$f_{%s}: %.3f$' % (k, fracHa__c[k]), 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
    plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[1][0]
    ax.set_title(r'R $\leq$ 0.5 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rin__gz
    xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=ALL.L6563__z, mask=~m_aux)
    tot_Ha = ym.sum()
    fracHa__c = {}
    for i, k in enumerate(args.class_names):
        y_pos = 0.97 - (i * 0.16)
        sel_aux = m_aux & sel['WHa']['z'][k]
        fracHa__c[k] = ALL.L6563__z[sel_aux].sum()/tot_Ha
        plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
    plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[2][0]
    ax.set_title(r'0.5 $<$ R $\leq$ 1 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rmid__gz
    xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=ALL.L6563__z, mask=~m_aux)
    tot_Ha = ym.sum()
    fracHa__c = {}
    for i, k in enumerate(args.class_names):
        y_pos = 0.97 - (i * 0.16)
        sel_aux = m_aux & sel['WHa']['z'][k]
        fracHa__c[k] = ALL.L6563__z[sel_aux].sum()/tot_Ha
        plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
    plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[3][0]
    ax.set_title(r'R $>$ 1 HLR', fontsize=18, y=1.02)
    m_aux = sel_gals_sample_Rout__gz
    xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=ALL.L6563__z, mask=~m_aux)
    tot_Ha = ym.sum()
    fracHa__c = {}
    for i, k in enumerate(args.class_names):
        y_pos = 0.97 - (i * 0.16)
        sel_aux = m_aux & sel['WHa']['z'][k]
        fracHa__c[k] = ALL.L6563__z[sel_aux].sum()/tot_Ha
        plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
    plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    sels_R = [sel_sample__gz, sel_gals_sample_Rin__gz, sel_gals_sample_Rmid__gz, sel_gals_sample_Rout__gz]

    for col in range(N_cols):
        for i, mt_label in enumerate(mtype_labels):
            ax = axes_cols[col][i+1]
            if not col:
                if (mt_label == mtype_labels[-1]):
                    lab = r'$\geq$ Sc'
                else:
                    lab = mt_label
                ax.set_ylabel(lab, rotation=90)
            m_aux = np.bitwise_and(sel_gals_mt[mt_label], sels_R[col])
            xm, ym = ma_mask_xyz(x=np.ma.log10(ALL.W6563__z), y=ALL.L6563__z, mask=~m_aux)
            print mt_label, col
            print ym.count()
            tot_Ha = ym.sum()
            fracHa__c = {}
            for j, k in enumerate(args.class_names):
                y_pos = 0.97 - (j * 0.16)
                sel_aux = m_aux & sel['WHa']['z'][k]
                print k, ALL.L6563__z[sel_aux].count()
                fracHa__c[k] = ALL.L6563__z[sel_aux].sum()/tot_Ha
                plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
            plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=logWHa_range, lw=2))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_minor_locator(MaxNLocator(20))
            ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            for th in args.class_thresholds:
                ax.axvline(x=np.ma.log10(th), c='k', ls='--')
            if i == (len(colortipo) - 1):
                ax.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
                if col == (N_cols - 1):
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
            else:
                if not col:
                    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
                else:
                    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            ax.set_xlim(logWHa_range)

    f.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=20)
    f.savefig('fig_WHa_histograms_per_morftype_and_radius_cumulFHa.png', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'
    return ALL


def maps_xi(args, gals):
    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    for califaID in gals:
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)


        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # AXIS 1
        ax1.set_title(r'$\log\ \xi$')
        log_L6563__yx = np.ma.masked_array(np.ma.log10(ALL.get_gal_prop(califaID, ALL.L6563__yx).reshape(N_y, N_x)), mask=~gal_sample__yx)
        log_L6563_expected_hDIG__yx = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.log_L6563_expected_HIG__yx).reshape(N_y, N_x), mask=~gal_sample__yx)
        xi = log_L6563__yx - log_L6563_expected_hDIG__yx
        im = ax1.imshow(xi, origin='lower')
        the_divider = make_axes_locatable(ax1)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)

        # AXIS 2
        ax2.set_title(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        log_W6563__yx = np.ma.masked_array(np.ma.log10(ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x)), mask=~gal_sample__yx)
        im = ax2.imshow(log_W6563__yx, origin='lower')
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)

        # AXIS 2
        ax3.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        ax3.set_ylabel(r'$\log\ \xi$')
        xm, ym, zm = ma_mask_xyz(log_W6563__yx, xi, )
        sc = ax3.scatter(log_W6563__yx, origin='lower')
        the_divider = make_axes_locatable(ax3)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        f.tight_layout(rect=[0, 0.01, 1, 0.98])

        f.tight_layout(rect=[0, 0.01, 1, 0.98])
        # f.subplots_adjust(left=0.05, right=0.96, bottom=0.05, top=0.95, hspace=0.08, wspace=0.3)
        f.savefig('fig_map_xi-%s.png' % califaID, dpi=_dpi_choice, transparent=_transp_choice)
        plt.close(f)


def fig_cumul_fHaSBHa_per_morftype(args, gals):
    print '##################################'
    print '# fig_cumul_fHaSBHa_per_morftype #'
    print '##################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']

    N_zone = sel_sample__gz.astype('int').sum()
    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']

    f = plt.figure(figsize=(8, 7))
    # _, ind = np.unique(ALL.califaID__z, return_index=True)
    ax = f.gca()

    sel_sample__gz = sel['gals_sample__z']
    cumulfHa__g = {}
    logSBHa_bins = np.linspace(2.5, 6.5, 50)
    for k, v in sel['gals__mt'].iteritems():
        for g in ALL.califaID__g[v]:
            sel_sample__z = ALL.get_gal_prop(g, sel_sample__gz)
            SBHa = np.ma.masked_array(ALL.get_gal_prop(g, ALL.SB6563__z), mask=~sel_sample__z)
            fHa = np.ma.masked_array(ALL.get_gal_prop(g, ALL.f6563__z), mask=~sel_sample__z)
            xm, ym = ma_mask_xyz(np.ma.log10(SBHa), fHa)
            fHa_cumul, fHa_tot = cumul_bins(ym.compressed(), xm.compressed(), logSBHa_bins)  # , fHa_tot)
            cumulfHa__g[g] = fHa_cumul
    for i, mt_label in enumerate(mtype_labels):
        aux = []
        y__gb = np.array([cumulfHa__g[g] for g in ALL.califaID__g[sel['gals__mt'][mt_label]]])
        x = np.hstack([logSBHa_bins for g in ALL.califaID__g[sel['gals__mt'][mt_label]]])
        # ax.plot(x, np.hstack(y__gb), ls='', marker='o', c=colortipo_lighter[i])
        ax.plot(logSBHa_bins, np.median(y__gb, axis=0), ls='-', c=colortipo[i], lw=3)
        # ax.plot(logSBHa_bins, np.mean(y__gb, axis=0), ls='--', c=colortipo[i], lw=3)
        # prc = np.percentile(y__gb, q=[16, 84], axis=0)
        # ax.plot(logSBHa_bins, prc[1], ls='--', c=colortipo[i], alpha=0.5)
        # ax.plot(logSBHa_bins, prc[0], ls='--', c=colortipo[i], alpha=0.5)
        y_pos = 0.99 - (i*0.1)
        if (i == len(mtype_labels) - 1):
            plot_text_ax(ax, r'$\geq$ Sc', 0.01, y_pos, 30, 'top', 'left', c=colortipo[i])
        else:
            plot_text_ax(ax, mt_label, 0.01, y_pos, 30, 'top', 'left', c=colortipo[i])
    ax.axvline(x=np.log10(SFc_Zhang_threshold), c='k', ls='--')
    # for th in args.class_thresholds:
    #     ax.axvline(x=np.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.grid(which='both')
    # ax.set_title(mt_label, color=colortipo[i])
    ax.set_xlim(2.5, 6.5)
    ax.set_ylim(0, 1)
    # f.subplots_adjust(left=0.06, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    ax.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
    ax.set_ylabel(r'H$\alpha$ flux fraction')
    # f.subplots_adjust(left=0.open06, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    # f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=20)
    # f.text(0.01, 0.5, r'$\sum\ F_{H\alpha} (< $W${}_{H\alpha})$', va='center', rotation='vertical', fontsize=20)
    f.savefig('fig_cumul_fHaSBHa_per_morftype.png', dpi=_dpi_choice, transparent=_transp_choice)
    plt.close(f)

    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_WHa_histograms_per_morftype_and_radius_cumulFHamed(args, gals, gals_sel=None, data_sel=None, fname_suffix='', histo_stats=False):
    print '##########################################################'
    print '# fig_WHa_histograms_per_morftype_and_radius_cumulFHamed #'
    print '##########################################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    if gals_sel is None:
        gals_sel = np.ones_like(gals, dtype='bool')
    sample_gals = sel['gals'] & gals_sel

    if data_sel is None:
        data_sel = np.ones_like(sel['gals_sample__z'], dtype='bool')
    sel_sample__gz = sel['gals_sample__z'] & data_sel
    sel_gals_mt = sel['gals__mt_z']

    print len(gals), len(sample_gals)
    gals = ALL.califaID__g[sample_gals]
    # for g in gals:
        # print g, ALL.ba[ALL.califaID__g == g]

    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']
    sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z'] & data_sel
    sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z'] & data_sel
    sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z'] & data_sel
    N_rows, N_cols = 7, 4
    f = plot_setup(width=latex_text_width, aspect=1./golden_mean)
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    N_zone = sel_sample__gz.astype('int').sum()

    gs = gridspec.GridSpec(N_rows, N_cols)
    axes_col0 = [
        plt.subplot(gs[0,0]), plt.subplot(gs[1,0]), plt.subplot(gs[2,0]),
        plt.subplot(gs[3,0]), plt.subplot(gs[4,0]), plt.subplot(gs[5,0]),
        plt.subplot(gs[6,0])
    ]
    axes_col1 = [
        plt.subplot(gs[0,1]), plt.subplot(gs[1,1]), plt.subplot(gs[2,1]),
        plt.subplot(gs[3,1]), plt.subplot(gs[4,1]), plt.subplot(gs[5,1]),
        plt.subplot(gs[6,1])
    ]
    axes_col2 = [
        plt.subplot(gs[0,2]), plt.subplot(gs[1,2]), plt.subplot(gs[2,2]),
        plt.subplot(gs[3,2]), plt.subplot(gs[4,2]), plt.subplot(gs[5,2]),
        plt.subplot(gs[6,2])
    ]
    axes_col3 = [
        plt.subplot(gs[0,3]), plt.subplot(gs[1,3]), plt.subplot(gs[2,3]),
        plt.subplot(gs[3,3]), plt.subplot(gs[4,3]), plt.subplot(gs[5,3]),
        plt.subplot(gs[6,3])
    ]

    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]
    fs = 6

    sels_R = [sel_sample__gz, sel_gals_sample_Rin__gz, sel_gals_sample_Rmid__gz, sel_gals_sample_Rout__gz]

    cumulfHa__g_R = {}
    logWHa_bins = np.linspace(-1, 3, 50)
    for k, v in sel['gals__mt'].iteritems():
        print k, len(v), len(sample_gals), np.sum(v), np.sum(v & sample_gals)
        aux_gals = ALL.califaID__g[sample_gals & v]
        for g in aux_gals:
            aux_list_fHa = []
            for i, sel_R in enumerate(sels_R):
                sel_sample__z = ALL.get_gal_prop(g, sel_sample__gz)
                sel_R__z = ALL.get_gal_prop(g, sel_R)
                WHa = np.ma.masked_array(ALL.get_gal_prop(g, ALL.W6563__z), mask=~(sel_sample__z & sel_R__z))
                fHa = np.ma.masked_array(ALL.get_gal_prop(g, ALL.f6563__z), mask=~(sel_sample__z & sel_R__z))
                xm, ym = ma_mask_xyz(np.ma.log10(WHa), fHa)
                fHa_cumul, fHa_tot = cumul_bins(ym.compressed(), xm.compressed(), logWHa_bins)  # , fHa_tot)
                # print 'iR:%d <WHa>:%.2f' % (i, WHa.mean())
                aux_list_fHa.append(fHa_cumul)
            cumulfHa__g_R[g] = aux_list_fHa

    N_tot = []
    logWHa_range = [-1, 2.5]
    ax = axes_cols[0][0]
    ax.set_title(r'All radii', fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    for g in gals:
        x = np.log10(args.class_thresholds[0])
        y1 = cumulfHa__g_R[g][0][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][0][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = cumulfHa__g_R[g][0][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][0][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in gals])
    f_SFc = np.mean([f_SFc__g[g] for g in gals])
    m_aux = sel_sample__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('hDIG', (100. * f_hDIG)), 0.99, 0.98, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('mDIG', (100. * f_mDIG)), 0.99, 0.82, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('SFc', (100. * f_SFc)), 0.99, 0.66, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    if histo_stats:
        plot_histo_ax(ax, xm.compressed(), first=True, stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    else:
        plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=False, ini_pos_y=0.97, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    N_hDIG = (xm <= np.log10(args.class_thresholds[0])).compressed().astype('int').sum()
    N_mDIG = ((xm > np.log10(args.class_thresholds[0])) & (xm <= np.log10(args.class_thresholds[1]))).compressed().astype('int').sum()
    N_SFc = (xm > np.log10(args.class_thresholds[1])).compressed().astype('int').sum()
    N_tot.append([N_hDIG, N_mDIG, N_SFc])
    # plot_text_ax(ax, r'%d' % N_hDIG, 0.02, 0.25, fs, 'top', 'left', c='k')
    # plot_text_ax(ax, r'%d' % N_mDIG, 0.44, 0.25, fs, 'top', 'left', c='k')
    # plot_text_ax(ax, r'%d' % N_SFc, 0.65, 0.25, fs, 'top', 'left', c='k')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[1][0]
    ax.set_title(r'$R\ \leq$ 0.5 HLR', fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    for g in gals:
        x = np.log10(args.class_thresholds[0])
        y1 = cumulfHa__g_R[g][1][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][1][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = cumulfHa__g_R[g][1][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][1][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in gals])
    f_SFc = np.mean([f_SFc__g[g] for g in gals])
    fs = 6
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_Rin__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    if histo_stats:
        plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    else:
        plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=False, ini_pos_y=0.97, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    N_hDIG = (xm <= np.log10(args.class_thresholds[0])).compressed().astype('int').sum()
    N_mDIG = ((xm > np.log10(args.class_thresholds[0])) & (xm <= np.log10(args.class_thresholds[1]))).compressed().astype('int').sum()
    N_SFc = (xm > np.log10(args.class_thresholds[1])).compressed().astype('int').sum()
    N_tot.append([N_hDIG, N_mDIG, N_SFc])
    # plot_text_ax(ax, r'%d' % N_hDIG, 0.02, 0.25, fs, 'top', 'left', c='k')
    # plot_text_ax(ax, r'%d' % N_mDIG, 0.44, 0.25, fs, 'top', 'left', c='k')
    # plot_text_ax(ax, r'%d' % N_SFc, 0.65, 0.25, fs, 'top', 'left', c='k')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[2][0]
    ax.set_title(r'0.5 $<\ R\ \leq$ 1 HLR', fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    for g in gals:
        x = np.log10(args.class_thresholds[0])
        y1 = cumulfHa__g_R[g][2][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][2][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = cumulfHa__g_R[g][2][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][2][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in gals])
    f_SFc = np.mean([f_SFc__g[g] for g in gals])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_Rmid__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    if histo_stats:
        plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    else:
        plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=False, ini_pos_y=0.97, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    N_hDIG = (xm <= np.log10(args.class_thresholds[0])).compressed().astype('int').sum()
    N_mDIG = ((xm > np.log10(args.class_thresholds[0])) & (xm <= np.log10(args.class_thresholds[1]))).compressed().astype('int').sum()
    N_SFc = (xm > np.log10(args.class_thresholds[1])).compressed().astype('int').sum()
    N_tot.append([N_hDIG, N_mDIG, N_SFc])
    # plot_text_ax(ax, r'%d' % N_hDIG, 0.02, 0.25, fs, 'top', 'left', c='k')
    # plot_text_ax(ax, r'%d' % N_mDIG, 0.44, 0.25, fs, 'top', 'left', c='k')
    # plot_text_ax(ax, r'%d' % N_SFc, 0.65, 0.25, fs, 'top', 'left', c='k')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[3][0]
    # ax.set_title(r'$R\ >$ 1 HLR', fontsize=fs+4, y=1.02)
    ax.set_title(r'1 $<\ R\ \leq$ 2 HLR', fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    for g in gals:
        x = np.log10(args.class_thresholds[0])
        y1 = cumulfHa__g_R[g][3][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][3][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = cumulfHa__g_R[g][3][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][3][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in gals])
    f_SFc = np.mean([f_SFc__g[g] for g in gals])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_Rout__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    if histo_stats:
        plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    else:
        plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=False, ini_pos_y=0.97, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    N_hDIG = (xm <= np.log10(args.class_thresholds[0])).compressed().astype('int').sum()
    N_mDIG = ((xm > np.log10(args.class_thresholds[0])) & (xm <= np.log10(args.class_thresholds[1]))).compressed().astype('int').sum()
    N_SFc = (xm > np.log10(args.class_thresholds[1])).compressed().astype('int').sum()
    N_tot.append([N_hDIG, N_mDIG, N_SFc])
    # plot_text_ax(ax, r'%d' % N_hDIG, 0.02, 0.25, fs, 'top', 'left', c='k')
    # plot_text_ax(ax, r'%d' % N_mDIG, 0.44, 0.25, fs, 'top', 'left', c='k')
    # plot_text_ax(ax, r'%d' % N_SFc, 0.65, 0.25, fs, 'top', 'left', c='k')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    for col in range(N_cols):
        for i, mt_label in enumerate(mtype_labels):
            ax = axes_cols[col][i+1]
            m_aux = sel_gals_mt[mt_label] & sels_R[col]
            f_SFc__g = {}
            f_mDIG__g = {}
            f_hDIG__g = {}
            _gals = ALL.califaID__g[sel['gals__mt'][mt_label] & sample_gals].tolist()
            for g in _gals:
                if not g in args.gals:
                    _gals.remove(g)
            if not col:
                if (mt_label == mtype_labels[-1]):
                    lab = r'$\geq$ Sc'
                else:
                    lab = mt_label
                # ax.set_ylabel('%s (%d)' % (lab, len(_gals)), rotation=90, fontsize=fs+1)
                ax.set_ylabel('%s' % lab, rotation=90, fontsize=fs+1)
            for g in _gals:
                # print mt_label, g, ALL.ba[ALL.califaID__g == g], ALL.mt[ALL.califaID__g == g]
                x = np.log10(args.class_thresholds[0])
                y1 = cumulfHa__g_R[g][col][ALL.logWHa_bins__w < x][-1]
                y2 = cumulfHa__g_R[g][col][ALL.logWHa_bins__w > x][0]
                x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
                x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
                y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
                # print x, y_th, x1, x2, y1, y2
                f_hDIG__g[g] = y_th
                x = np.log10(args.class_thresholds[1])
                y1 = cumulfHa__g_R[g][col][ALL.logWHa_bins__w < x][-1]
                y2 = cumulfHa__g_R[g][col][ALL.logWHa_bins__w > x][0]
                x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
                x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
                y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
                f_SFc__g[g] = 1. - y_th
                # print x, y_th, x1, x2, y1, y2
                f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
                # print g, f_hDIG__g[g], f_mDIG__g[g], f_SFc__g[g]
            f_hDIG = np.mean([f_hDIG__g[g] for g in _gals])
            f_mDIG = np.mean([f_mDIG__g[g] for g in _gals])
            f_SFc = np.mean([f_SFc__g[g] for g in _gals])
            plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
            plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
            plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
            xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
            # plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
            if histo_stats:
                plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=logWHa_range, lw=2))
            else:
                plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=logWHa_range, lw=2))
            N_hDIG = (xm <= np.log10(args.class_thresholds[0])).compressed().astype('int').sum()
            N_mDIG = ((xm > np.log10(args.class_thresholds[0])) & (xm <= np.log10(args.class_thresholds[1]))).compressed().astype('int').sum()
            N_SFc = (xm > np.log10(args.class_thresholds[1])).compressed().astype('int').sum()
            # print col, N_tot[col][0], N_tot[col][1], N_tot[col][2]
            # plot_text_ax(ax, r'%.2f%%' % (100.*N_hDIG/N_tot[col][0]), 0.02, 0.4, fs, 'top', 'left', c='k')
            # plot_text_ax(ax, r'%d' % N_hDIG, 0.02, 0.25, fs, 'top', 'left', c='k')
            # plot_text_ax(ax, r'%.2f%%' % (100.*N_mDIG/N_tot[col][1]), 0.44, 0.4, fs, 'top', 'left', c='k')
            # plot_text_ax(ax, r'%d' % N_mDIG, 0.44, 0.25, fs, 'top', 'left', c='k')
            # plot_text_ax(ax, r'%.2f%%' % (100.*N_SFc/N_tot[col][2]), 0.65, 0.4, fs, 'top', 'left', c='k')
            # plot_text_ax(ax, r'%d' % N_SFc, 0.65, 0.25, fs, 'top', 'left', c='k')
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_minor_locator(MaxNLocator(20))
            ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            # ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            for th in args.class_thresholds:
                ax.axvline(x=np.ma.log10(th), c='k', ls='--')
            if i == (len(colortipo) - 1):
                # ax.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
                ax.tick_params(axis='x', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')
                if col == (N_cols - 1):
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
            else:
                ax.tick_params(axis='both', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
                # if not col:
                #     ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
                # else:
                #     ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            ax.set_xlim(logWHa_range)

    # f.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    gs.update(left=0.05, right=0.95, bottom=0.14, top=0.9, hspace=0.0, wspace=0.0)
    f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=fs+4)
    # f.set_tight_layout(True)
    f.savefig('fig_WHa_histo_mt_and_R_cumulFHa%s.pdf' % fname_suffix, dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'
    return ALL


def get_extraplanar_regions(K, pup, plow):
    selection = np.zeros(K.N_zone, dtype='bool')
    for z in range(K.N_zone):
        yup = np.polyval(pup, K.zonePos[z][0])
        ylow = np.polyval(plow, K.zonePos[z][0])
        if (K.zonePos[z][-1] > yup) | (K.zonePos[z][-1] < ylow):
            selection[z] = True
    return selection


def extraplanar_xi_hDIG(args):
    from CALIFAUtils.scripts import read_one_cube
    W6563__gz = ALL.W6563__z
    log_L6563__gz = np.ma.log10(ALL.L6563__z)
    log_L6563_expected_hDIG__gz = ALL.log_L6563_expected_HIG__z
    xi__gz = log_L6563__gz - log_L6563_expected_hDIG__gz
    xi_extraplanar_hDIG = []
    WHa_extraplanar_hDIG = []

    g = 'K0791'
    K = read_one_cube(g, EL=True, config=-2, debug=True, elliptical=True)
    xi__z = ALL.get_gal_prop(g, xi__gz)
    sel_hDIG__z = ALL.get_gal_prop(g, sel['gals_sample__z'] & sel['WHa']['z']['hDIG'])
    sel_extrplan__z = get_extraplanar_regions(K, [1.8, -15.], [1.8, -53.])
    xi_extraplan__z = np.ma.masked_array(xi__z, mask=~(sel_extrplan__z & sel_hDIG__z), copy=True)
    xi_extraplanar_hDIG.append(xi_extraplan__z.compressed())
    WHa__z = ALL.get_gal_prop(g, W6563__gz)
    WHa_extraplan__z = np.ma.masked_array(WHa__z, mask=~(sel_extrplan__z & sel_hDIG__z), copy=True)
    WHa_extraplanar_hDIG.append(WHa_extraplan__z.compressed())

    g = 'K0822'
    K = read_one_cube(g, EL=True, config=-2, debug=True, elliptical=True)
    xi__z = ALL.get_gal_prop(g, xi__gz)
    sel_hDIG__z = ALL.get_gal_prop(g, sel['gals_sample__z'] & sel['WHa']['z']['hDIG'])
    sel_extrplan__z = get_extraplanar_regions(K, [0.7, 11.], [0.7, 0.])
    xi_extraplan__z = np.ma.masked_array(xi__z, mask=~(sel_extrplan__z & sel_hDIG__z), copy=True)
    xi_extraplanar_hDIG.append(xi_extraplan__z.compressed())
    WHa__z = ALL.get_gal_prop(g, W6563__gz)
    WHa_extraplan__z = np.ma.masked_array(WHa__z, mask=~(sel_extrplan__z & sel_hDIG__z), copy=True)
    WHa_extraplanar_hDIG.append(WHa_extraplan__z.compressed())

    g = 'K0077'
    K = read_one_cube(g, EL=True, config=-2, debug=True, elliptical=True)
    xi__z = ALL.get_gal_prop(g, xi__gz)
    sel_hDIG__z = ALL.get_gal_prop(g, sel['gals_sample__z'] & sel['WHa']['z']['hDIG'])
    sel_extrplan__z = get_extraplanar_regions(K, [-58./64., 72.], [-58./64., 58.])
    xi_extraplan__z = np.ma.masked_array(xi__z, mask=~(sel_extrplan__z & sel_hDIG__z), copy=True)
    xi_extraplanar_hDIG.append(xi_extraplan__z.compressed())
    WHa__z = ALL.get_gal_prop(g, W6563__gz)
    WHa_extraplan__z = np.ma.masked_array(WHa__z, mask=~(sel_extrplan__z & sel_hDIG__z), copy=True)
    WHa_extraplanar_hDIG.append(WHa_extraplan__z.compressed())

    g = 'K0936'
    K = read_one_cube(g, EL=True, config=-2, debug=True, elliptical=True)
    xi__z = ALL.get_gal_prop(g, xi__gz)
    sel_hDIG__z = ALL.get_gal_prop(g, sel['gals_sample__z'] & sel['WHa']['z']['hDIG'])
    sel_extrplan__z = get_extraplanar_regions(K, [-43./107., 49.], [-43./107., 43.])
    xi_extraplan__z = np.ma.masked_array(xi__z, mask=~(sel_extrplan__z & sel_hDIG__z), copy=True)
    xi_extraplanar_hDIG.append(xi_extraplan__z.compressed())
    WHa__z = ALL.get_gal_prop(g, W6563__gz)
    WHa_extraplan__z = np.ma.masked_array(WHa__z, mask=~(sel_extrplan__z & sel_hDIG__z), copy=True)
    WHa_extraplanar_hDIG.append(WHa_extraplan__z.compressed())


def fig_cumul_fHaWHa_per_morftype_group(args, gals):
    print '#################################'
    print '# fig_cumul_fHaWHa_per_morftype_group #'
    print '#################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    N_zone = sel_sample__gz.astype('int').sum()
    colortipo = ['red', 'green', 'blue']
    mtype_labels = ['E+S0+S0a', 'Sa+Sab+Sb', '>=Sbc']

    f = plt.figure(figsize=(8, 7))
    # _, ind = np.unique(ALL.califaID__z, return_index=True)
    ax = f.gca()

    # sels_R = [sel['gals_sample__z'], sel['gals_sample_Rin__z'], sel['gals_sample_Rmid__z'], sel['gals_sample_Rout__z']]
    logWHa_bins = ALL.logWHa_bins__w
    cumulfHa__g_R = ALL.cumulfHa__gRw
    sel_mt = {
        'E+S0+S0a': sel['gals__mt']['E'] | sel['gals__mt']['S0+S0a'],
        'Sa+Sab+Sb': sel['gals__mt']['Sa+Sab'] | sel['gals__mt']['Sb'],
        '>=Sbc': sel['gals__mt']['Sbc'] | sel['gals__mt']['>= Sc']
    }

    for i, mt_label in enumerate(mtype_labels):
        aux = []
        print mt_label
        print sel_mt[mt_label].sum()
        for g in ALL.califaID__g[sel_mt[mt_label]]:
            print g
            aux.append(cumulfHa__g_R[g][0])
        y__gb = np.array(aux)
        x = np.hstack([logWHa_bins for g in ALL.califaID__g[sel_mt[mt_label]]])
        # ax.plot(x, np.hstack(y__gb), ls='', marker='o', c=colortipo_lighter[i])
        ax.plot(logWHa_bins, np.median(y__gb, axis=0), ls='-', c=colortipo[i], lw=3)
        # ax.plot(logWHa_bins, np.mean(y__gb, axis=0), ls='--', c=colortipo[i], lw=3)
        # prc = np.percentile(y__gb, q=[16, 84], axis=0)
        # ax.plot(logWHa_bins, prc[1], ls='--', c=colortipo[i], alpha=0.5)
        # ax.plot(logWHa_bins, prc[0], ls='--', c=colortipo[i], alpha=0.5)
        y_pos = 0.99 - (i*0.1)
        if (i == len(mtype_labels) - 1):
            plot_text_ax(ax, r'$\geq$ Sbc', 0.01, y_pos, 30, 'top', 'left', c=colortipo[i])
        else:
            plot_text_ax(ax, mt_label, 0.01, y_pos, 30, 'top', 'left', c=colortipo[i])
    for th in args.class_thresholds:
        ax.axvline(x=np.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.grid(which='both')
    # ax.set_title(mt_label, color=colortipo[i])
    ax.set_xlim(-1, 2.5)
    ax.set_ylim(0, 1)
    # f.subplots_adjust(left=0.06, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    ax.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    ax.set_ylabel(r'H$\alpha$ flux fraction')
    # f.subplots_adjust(left=0.open06, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    # f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=20)
    # f.text(0.01, 0.5, r'$\sum\ F_{H\alpha} (< $W${}_{H\alpha})$', va='center', rotation='vertical', fontsize=20)
    f.savefig('fig_cumul_fHaWHa_per_morftype_group.png', dpi=_dpi_choice, transparent=_transp_choice)
    plt.close(f)

    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_BPT_mixed_xi(args, gals):
    print '####################'
    print '# fig_BPT_mixed_xi #'
    print '####################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z'] & sel['SN_BPT3']['z']
    sel_sample__gyx = sel['gals_sample__yx'] & sel['SN_BPT3']['yx']

    N_zone = sel_sample__gz.astype('int').sum()
    log_L6563__gz = np.ma.log10(ALL.L6563__z)
    log_L6563_expected_hDIG__gz = ALL.log_L6563_expected_HIG__z
    xi__gz = log_L6563__gz - log_L6563_expected_hDIG__gz
    z = 10**xi__gz  # ALL.W6563__z

    sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z'] & sel['SN_BPT3']['z']
    sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z'] & sel['SN_BPT3']['z']
    sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z'] & sel['SN_BPT3']['z']
    BPTLines = ['4861', '5007', '6563', '6583']
    f__lgz = {'%s' % l: np.ma.masked_array(getattr(ALL, 'f%s__z' % l), mask=~sel_sample__gz) for l in BPTLines}
    O3Hb__gz = np.ma.log10(f__lgz['5007']/f__lgz['4861'])
    N2Ha__gz = np.ma.log10(f__lgz['6583']/f__lgz['6563'])

    fs = 6
    # f = plt.figure(figsize=(8, 8))
    f = plot_setup(width=latex_column_width, aspect=1.)  #1./golden_mean)
    ax = f.gca()

    extent = [-1.5, 0.7, -1.2, 1.2]

    m_aux = sel_gals_sample_Rout__gz & sel['WHa']['z']['mDIG']
    xm, ym, zm = ma_mask_xyz(x=N2Ha__gz, y=O3Hb__gz, z=z, mask=~m_aux)
    scater_kwargs = dict(c=zm, s=1, vmax=14, vmin=3, cmap='Spectral', marker='o', edgecolor='none')
    sc = ax.scatter(xm, ym, **scater_kwargs)
    L = Lines()
    ax.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
    ax.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
    ax.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
    _, N2Ha , O3Hb, _ = calcMixingLineOnBPT(-0.4,-0.7,3.71,-0.1,0.1,2.86,0.2)
    ax.plot(N2Ha,O3Hb,'ko-')
    _, N2Ha , O3Hb, _ = calcMixingLineOnBPT(-0.5,-0.3,3.75,-0.1,0.1,2.86,0.2)
    ax.plot(N2Ha,O3Hb,'ko-')
    _, N2Ha , O3Hb, _ = calcMixingLineOnBPT(-0.6,0.,3.67,-0.1,0.1,2.86,0.2)
    ax.plot(N2Ha,O3Hb,'ko-')
    ax.tick_params(axis='both', which='both', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    # plot_text_ax(ax, 'S06', 0.32, 0.01, 20, 'bottom', 'left', 'k')
    # plot_text_ax(ax, 'K03', 0.62, 0.01, 20, 'bottom', 'left', 'k')
    # plot_text_ax(ax, 'K01', 0.99, 0.01, 20, 'bottom', 'right', 'k')
    plot_text_ax(ax, 'S06', 0.32, 0.01, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'K03', 0.62, 0.01, fs+2, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'K01', 0.99, 0.01, fs+2, 'bottom', 'right', 'k')
    cbaxes = f.add_axes([0.69, 0.9, 0.25, 0.05])
    cb = plt.colorbar(sc, cax=cbaxes, orientation='horizontal')  #, ticks=[0, 1, 2, 3])
    # cb.set_label(r'W${}_{H\alpha}$ [$\AA$]', fontsize=14)
    cb.set_label(r'$\xi_{H\alpha}$ [$\AA$]', fontsize=fs+2)

    ax.set_xlabel(r'$\log\ [NII]/H\alpha$')
    ax.set_ylabel(r'$\log\ [OIII]/H\beta$')
    f.set_tight_layout(True)
    f.savefig('fig_BPT_mixed_xi.pdf', dpi=_dpi_choice, transparent=_transp_choice)


def fig_SII_histogram(args, gals, minSNR=1):
    ###################################### MUSE ######################################
    ###################################### MUSE ######################################
    ###################################### MUSE ######################################
    # from astropy.io import fits
    # t = fits.open('/Users/lacerda/muse/flux_elines/flux_elines.NGC6754.cube.fits')
    # f6717 = np.hstack(t[0].data[24])
    # ef6717 = np.hstack(t[0].data[144])
    # f6731 = np.hstack(t[0].data[25])
    # ef6731 = np.hstack(t[0].data[145])
    # W6563 = -1. * np.hstack(t[0].data[110])
    # sel_SII = (f6717 > 0) & (f6731 > 0) & ~(np.isnan(f6717)) & ~(np.isnan(f6731)) & ~(np.isnan(ef6717)) & ~(np.isnan(ef6731))
    # SNR6717 = np.ma.masked_array(f6717/ef6717, mask=~sel_SII)
    # SNR6731 = np.ma.masked_array(f6731/ef6731, mask=~sel_SII)
    # SNRSII = 1./(1./SNR6717**2.0 + 1./SNR6731**2.0)**0.5
    # sel_muse = (W6563 > 0) & ~(np.isnan(W6563)) & (SNRSII > minSNR)
    # RSII = np.ma.masked_array(f6717/f6731, mask=~sel_muse)
    # mW6563 = np.ma.masked_array(W6563, mask=~sel_muse)
    # f = plot_setup(width=latex_column_width, aspect=1./golden_mean)
    # rangex = [0, 3]
    # rangey = [0.5, 2.5]
    # xm, ym = ma_mask_xyz(np.ma.log10(mW6563), RSII)
    # xbins = np.linspace(rangex[0], rangex[1], 30)
    # yMean, prc_muse, bin_center_muse, npts = stats_med12sigma(xm.compressed(), ym.compressed(), xbins)
    # f = plot_setup(width=latex_column_width, aspect=1./golden_mean)
    # ax1 = f.gca()
    # ax1.scatter(xm, ym, s=0.5, c=mW6563, vmax=14, vmin=3, cmap='Spectral', **dflt_kw_scatter)
    # ax1.set_xlim(rangex)
    # ax1.set_ylim(rangey)
    # yMedian_muse = prc_muse[2]
    # y_12sigma = [prc_muse[0], prc_muse[1], prc_muse[3], prc_muse[4]]
    # ax1.plot(bin_center_muse, yMedian_muse, 'k-', lw=0.7)
    # for y_prc in y_12sigma:
    #     ax1.plot(bin_center_muse, y_prc, 'k--', lw=0.7)
    # yerr = [yMedian_muse - prc_muse[1], prc_muse[3] - yMedian_muse]
    # ax1.errorbar(bin_center_muse, yMedian_muse, yerr=yerr, marker='o', ecolor='k', mew=0.7, mec='k', ms=2, mfc='w', capthick=0.7, capsize=0.7, ls='none')
    # ax1.grid()
    # ax1.set_title('MUSE - NGC6754')
    # ax1.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    # ax1.set_ylabel(r'$6716/6731$')
    # ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    # f.set_tight_layout(True)
    # f.savefig('fig_muse_NGC6754_SII_logWHa_SNR%d.png' % minSNR, dpi=300, transparent=_transp_choice)
    # t.close()
    #################################### FIM MUSE ####################################
    #################################### FIM MUSE ####################################
    #################################### FIM MUSE ####################################

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_SII = (ALL.f6717__z > 0.) & (ALL.f6731__z > 0.)
    SNR6717 = np.ma.masked_array(ALL.f6717__z/ALL.ef6717__z, mask=~sel_SII)
    SNR6731 = np.ma.masked_array(ALL.f6731__z/ALL.ef6731__z, mask=~sel_SII)
    # SNRSII is the lowest between SNR6717 & SNR6731
    # SNRSII = np.ma.where(SNR6717 < SNR6731, SNR6717, SNR6731)
    SNRSII = 1./(1./SNR6717**2.0 + 1./SNR6731**2.0)**0.5
    sel_sample__gz = sel['gals_sample__z'] & sel_SII & (SNRSII >= minSNR)
    RSII = np.ma.masked_array(ALL.f6731__z/ALL.f6717__z, mask=~sel_sample__gz)

    f = plot_setup(width=latex_column_width, aspect=1.)  #1./golden_mean)
    ax1 = f.gca()
    xDs = []
    for k in args.class_names:
        sel_aux = sel['WHa']['z'][k] & sel_sample__gz
        x = RSII[sel_aux].compressed()
        print k, x.shape
        xDs.append(x)
    range = [0, 2]
    _, text_list = plot_histo_ax(ax1, xDs, use_range_stats=False, stats_txt=False, return_text_list=True, y_v_space=0.06, y_h_space=0.25, first=False, c=args.class_colors, fs=8, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=1))
    x_ini = 0.02
    for j, k in enumerate(args.class_names):
        pos_y = 0.98
        for txt in text_list[j]:
            print k, txt
            plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=8, va='top', ha='left', c=args.class_colors[j]))
            pos_y -= 0.06
        x_ini += 0.15
    ax1.set_xlabel(r'$6731/6717$')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    f.set_tight_layout(True)
    f.savefig('fig_histo_SII_SNR%d.pdf' % minSNR, dpi=300, transparent=_transp_choice)

    rangex = [0, 3]
    rangey = [0.3, 1.5]
    W6563__z = np.ma.masked_array(ALL.W6563__z, mask=~sel_sample__gz)
    W6563__z[(W6563__z > 10**2.7) & (RSII > 0.9)] = np.ma.masked
    xm, ym = ma_mask_xyz(np.ma.log10(W6563__z), RSII)
    xbins = np.linspace(rangex[0], rangex[1], 20)
    yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), xbins)
    f = plot_setup(width=latex_column_width, aspect=1.)  #1./golden_mean)
    ax = f.gca()
    ax.scatter(xm, ym, rasterized=True, s=0.5, c=W6563__z, vmax=14, vmin=3, cmap='Spectral', **dflt_kw_scatter)
    print xm.count()
    # sel_bincenter = bin_center_muse > 1.1
    # yerr = [yMedian_muse[sel_bincenter] - prc_muse[1][sel_bincenter], prc_muse[3][sel_bincenter] - yMedian_muse[sel_bincenter]]
    # ax.errorbar(bin_center_muse[sel_bincenter], yMedian_muse[sel_bincenter], yerr=yerr, marker='o', ecolor='k', mew=0.7, mec='k', ms=2, mfc='w', capthick=0.7, capsize=0.7, ls='none')
    ax.set_xlim(rangex)
    ax.set_ylim(rangey)
    yMedian = prc[2]
    y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
    ax.plot(bin_center, yMedian, 'k-', lw=0.7)
    for y_prc in y_12sigma:
        ax.plot(bin_center, y_prc, 'k--', lw=0.7)
    ax.grid()
    # ax.set_title('CALIFA - %d zones' % xm.count())
    ax.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    ax.set_ylabel(r'$6731/6717$')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    f.set_tight_layout(True)
    f.savefig('fig_SII_logWHa_SNR%d.pdf' % minSNR, dpi=_dpi_choice, transparent=_transp_choice)

    # N_zone_tot = sel['gals_sample__z'].astype('int').sum()
    # N_zone_SII_tot = sel_sample__gz.astype('int').sum()
    # f6717_sum = {k:np.ma.masked_all((len(gals)), dtype='float') for k in args.class_names}
    # f6731_sum = {k:np.ma.masked_all((len(gals)), dtype='float') for k in args.class_names}
    # for i, g in enumerate(gals):
    #     print i, g
    #     sel_SII__z = ALL.get_gal_prop(g, sel_sample__gz)
    #     N_zone_SII = sel_SII__z.astype('int').sum()
    #     if N_zone_SII > 0:
    #         sel_classes__cz, _ = get_selections_zones(args, ALL, g, sel['WHa']['z'])
    #         for c in args.class_names:
    #             N_class = sel_classes__cz[c].astype('int').sum()
    #             print N_class
    #             f6717_sum[c][i] = np.ma.masked_array(ALL.get_gal_prop(g, ALL.f6717__z), mask=~(sel_SII__z & sel_classes__cz[c])).compressed().sum()
    #             f6731_sum[c][i] = np.ma.masked_array(ALL.get_gal_prop(g, ALL.f6731__z), mask=~(sel_SII__z & sel_classes__cz[c])).compressed().sum()
    #         N_zone = ALL.get_gal_prop_unique(g, ALL.N_zone)
    #         print g, N_zone, N_zone_SII, ' (%.2f%%)' % (100.*N_zone_SII/N_zone), f6717_sum[c][i]/f6731_sum[c][i]
    # print N_zone_tot, N_zone_SII_tot, ' (%.2f%%)' % (100.*N_zone_SII_tot/N_zone_tot)
    # f = plot_setup(width=latex_column_width, aspect=1./golden_mean)
    # ax1 = f.gca()
    # xDs = []
    # for k in args.class_names:
    #     RSII = f6717_sum[k]/f6731_sum[k]
    #     x = RSII.compressed()
    #     xDs.append(x)
    # range = [0, 2]
    # _, text_list = plot_histo_ax(ax1, xDs, use_range_stats=False, stats_txt=False, return_text_list=True, y_v_space=0.06, y_h_space=0.25, first=False, c=args.class_colors, fs=8, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=1))
    # x_ini = 0.02
    # for j, k in enumerate(args.class_names):
    #     pos_y = 0.98
    #     for txt in text_list[j]:
    #         print k, txt
    #         plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=8, va='top', ha='left', c=args.class_colors[j]))
    #         pos_y -= 0.06
    #     x_ini += 0.15
    # ax1.set_xlabel(r'$6717/6731$')
    # ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    # f.set_tight_layout(True)
    # f.savefig('fig_integrated_histo_SII_SNR%d.png' % minSNR, dpi=300, transparent=_transp_choice)


def fig_data_histograms_per_morftype_and_radius(args, gals, data, data_range, data_label, data_suffix, byclass=False, bins=None, data_sel=None):
    print '###############################################'
    print '# fig_data_histograms_per_morftype_and_radius #'
    print '###############################################'

    if bins is None:
        bins = 30

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    if data_sel is None:
        data_sel = np.ones(sel['gals_sample__z'].shape, dtype='bool')
    sel_sample__gz = sel['gals_sample__z'] & data_sel
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']

    colortipo = ['orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['Sa+Sab', 'Sb', 'Sbc', '>= Sc']
    sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z'] & data_sel
    sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z'] & data_sel
    sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z'] & data_sel
    N_rows, N_cols = 5, 4
    # f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 10))
    f = plot_setup(width=latex_text_width, aspect=1./golden_mean)
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    N_zone = sel_sample__gz.astype('int').sum()
    # (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43) = axArr
    # axes_col0 = [ax00, ax10, ax20, ax30, ax40]
    # axes_col1 = [ax01, ax11, ax21, ax31, ax41]
    # axes_col2 = [ax02, ax12, ax22, ax32, ax42]
    # axes_col3 = [ax03, ax13, ax23, ax33, ax43]

    gs = gridspec.GridSpec(N_rows, N_cols)
    axes_col0 = [
        plt.subplot(gs[0,0]), plt.subplot(gs[1,0]), plt.subplot(gs[2,0]),
        plt.subplot(gs[3,0]), plt.subplot(gs[4,0])
    ]
    axes_col1 = [
        plt.subplot(gs[0,1]), plt.subplot(gs[1,1]), plt.subplot(gs[2,1]),
        plt.subplot(gs[3,1]), plt.subplot(gs[4,1])
    ]
    axes_col2 = [
        plt.subplot(gs[0,2]), plt.subplot(gs[1,2]), plt.subplot(gs[2,2]),
        plt.subplot(gs[3,2]), plt.subplot(gs[4,2])
    ]
    axes_col3 = [
        plt.subplot(gs[0,3]), plt.subplot(gs[1,3]), plt.subplot(gs[2,3]),
        plt.subplot(gs[3,3]), plt.subplot(gs[4,3])
    ]

    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]
    fs = 6

    # logWHa = np.ma.log10(ALL.W6563__z)
    # logWHa_range = [-1, 2.5]
    # xlabel = r'$\log$ W${}_{H\alpha}$ [$\AA$]'

    ax = axes_cols[0][0]
    ax.set_title(r'All radii', fontsize=fs+4, y=1.02)

    m_aux = sel_sample__gz
    xm = np.ma.masked_array(data, mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.1, fs=fs, c='k', kwargs_histo=dict(bins=bins, histtype='stepfilled', color='darkgray', range=data_range, lw=1))
    if byclass:
        dataset = [np.ma.masked_array(data, mask=~(m_aux & sel['WHa']['z'][c])).compressed() for c in args.class_names]
        plot_histo_ax(ax, dataset, y_v_space=0.1, y_h_space=0.2, pos_x=0.99, ha='right', fs=fs, first=False, stats_txt=True, c=args.class_colors, kwargs_histo=dict(bins=bins, histtype='step', color=args.class_colors, normed=False, range=data_range, lw=1))
    ax.set_xlim(data_range)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    # ax.xaxis.grid()

    ax = axes_cols[1][0]
    ax.set_title(r'R $\leq$ 0.5 HLR', fontsize=fs+4, y=1.02)
    m_aux = sel_gals_sample_Rin__gz
    xm = np.ma.masked_array(data, mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.1, fs=fs, c='k', kwargs_histo=dict(bins=bins, histtype='stepfilled', color='darkgray', range=data_range, lw=1))
    if byclass:
        dataset = [np.ma.masked_array(data, mask=~(m_aux & sel['WHa']['z'][c])).compressed() for c in args.class_names]
        plot_histo_ax(ax, dataset, y_v_space=0.1, y_h_space=0.2, pos_x=0.99, ha='right', fs=fs, first=False, stats_txt=True, c=args.class_colors, kwargs_histo=dict(bins=bins, histtype='step', color=args.class_colors, normed=False, range=data_range, lw=1))
    ax.set_xlim(data_range)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    # ax.xaxis.grid()

    ax = axes_cols[2][0]
    ax.set_title(r'0.5 $<$ R $\leq$ 1 HLR', fontsize=fs+4, y=1.02)
    m_aux = sel_gals_sample_Rmid__gz
    xm = np.ma.masked_array(data, mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.1, fs=fs, c='k', kwargs_histo=dict(bins=bins, histtype='stepfilled', color='darkgray', range=data_range, lw=1))
    if byclass:
        dataset = [np.ma.masked_array(data, mask=~(m_aux & sel['WHa']['z'][c])).compressed() for c in args.class_names]
        plot_histo_ax(ax, dataset, y_v_space=0.1, y_h_space=0.2, pos_x=0.99, ha='right', fs=fs, first=False, stats_txt=True, c=args.class_colors, kwargs_histo=dict(bins=bins, histtype='step', color=args.class_colors, normed=False, range=data_range, lw=1))
    ax.set_xlim(data_range)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    # ax.xaxis.grid()

    ax = axes_cols[3][0]
    ax.set_title(r'R $>$ 1 HLR', fontsize=fs+4, y=1.02)
    m_aux = sel_gals_sample_Rout__gz
    xm = np.ma.masked_array(data, mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.1, fs=fs, c='k', kwargs_histo=dict(bins=bins, histtype='stepfilled', color='darkgray', range=data_range, lw=1))
    if byclass:
        dataset = [np.ma.masked_array(data, mask=~(m_aux & sel['WHa']['z'][c])).compressed() for c in args.class_names]
        plot_histo_ax(ax, dataset, y_v_space=0.1, y_h_space=0.2, pos_x=0.99, ha='right', fs=fs, first=False, stats_txt=True, c=args.class_colors, kwargs_histo=dict(bins=bins, histtype='step', color=args.class_colors, normed=False, range=data_range, lw=1))
    ax.set_xlim(data_range)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    # ax.xaxis.grid()

    sels_R = [sel_sample__gz, sel_gals_sample_Rin__gz, sel_gals_sample_Rmid__gz, sel_gals_sample_Rout__gz]

    for col in range(N_cols):
        for i, mt_label in enumerate(mtype_labels):
            ax = axes_cols[col][i+1]
            if not col:
                if (mt_label == mtype_labels[-1]):
                    lab = r'$\geq$ Sc'
                else:
                    lab = mt_label
                ax.set_ylabel(lab, rotation=90, fontsize=fs+4)
            m_aux = np.bitwise_and(sel_gals_mt[mt_label], sels_R[col])
            xm = np.ma.masked_array(data, mask=~m_aux)
            # plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
            plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.1, fs=fs, c='k', kwargs_histo=dict(bins=bins, histtype='stepfilled', alpha=0.5, color=colortipo[i], range=data_range, lw=1))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_minor_locator(MaxNLocator(20))
            if byclass:
                dataset = [np.ma.masked_array(data, mask=~(m_aux & sel['WHa']['z'][c])).compressed() for c in args.class_names]
                plot_histo_ax(ax, dataset, y_v_space=0.1, y_h_space=0.2, pos_x=0.99, ha='right', fs=fs, first=False, stats_txt=True, c=args.class_colors, kwargs_histo=dict(bins=bins, histtype='step', color=args.class_colors, normed=False, range=data_range, lw=1))
            ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            if i == (len(colortipo) - 1):
                ax.tick_params(axis='x', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')
                if col == (N_cols - 1):
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
            else:
                ax.tick_params(axis='both', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            ax.set_xlim(data_range)
            # ax.xaxis.grid()
    # f.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.9, hspace=0.0, wspace=0.0)
    gs.update(left=0.05, right=0.95, bottom=0.14, top=0.9, hspace=0.0, wspace=0.0)
    f.text(0.5, 0.04, data_label, ha='center', fontsize=fs+4)
    f.savefig('fig_histograms_per_morftype_and_radius_%s.png' % data_suffix, dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'
    return ALL


def calc_radial_profiles(args, gals, data_sel__z=None, data_sel__yx=None):  # calc_R_stuff
    ALL, sel = args.ALL, args.sel

    if data_sel__z is None:
        data_sel__z = np.ones_like(sel['gals_sample__z'])
    if data_sel__yx is None:
        data_sel__yx = np.ones_like(sel['gals_sample__yx'])

    sel_gals__mt = sel['gals__mt']
    sel_sample__gz = sel['gals_sample__z'] & data_sel__z
    sel_sample__gyx = sel['gals_sample__yx'] & data_sel__yx
    sel_gals__mt_gz = sel['gals__mt_z']
    sel_WHa__c_gz = sel['WHa']['z']
    sel_WHa__c_gyx = sel['WHa']['yx']
    N_gals = len(gals)
    N_R_bins = args.N_R_bins
    N_T = len(ALL.tSF__T)
    lines = ['4861', '6563']
    radial_mode = 'mean'

    if (sel_sample__gz).any():
        # # SYN
        tau_V__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        tau_V_npts__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        alogt_flux__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        alogt_flux_npts__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        alogZ_mass__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        alogZ_mass_npts__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        # SYN:tSF
        x_Y_32__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        x_Y_32_npts__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        aSFRSD_32__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        aSFRSD_32_npts__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        x_Y_300__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        x_Y_300_npts__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        aSFRSD_300__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        aSFRSD_300_npts__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}

        # EML
        mean_f4861__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        mean_f6563__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        mean_f4861_npts__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        mean_f6563_npts__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}

        mean_f4861__gr = np.ma.masked_all((N_gals, args.N_R_bins))
        mean_f4861_npts__gr = np.ma.masked_all((N_gals, args.N_R_bins))
        mean_f6563__gr = np.ma.masked_all((N_gals, args.N_R_bins))
        mean_f6563_npts__gr = np.ma.masked_all((N_gals, args.N_R_bins))
        tau_V_neb__gr = np.ma.masked_all((N_gals, args.N_R_bins))
        # tau_V_neb_npts__gr = np.ma.masked_all((N_gals, args.N_R_bins))
        tau_V_neb_sumGAL__gr = np.ma.masked_all((N_gals, args.N_R_bins))
        tau_V_neb__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        # tau_V_neb_npts__cgr = {k:np.ma.masked_all((N_gals, args.N_R_bins), dtype='float') for k in args.class_names}

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
            gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
            gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)
            f__lz = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__z' % L), mask=~gal_sample__z) for L in lines}
            f__lyx = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__yx' % L).reshape(N_y, N_x), mask=~gal_sample__yx) for L in lines}
            tau_V__z = ALL.get_gal_prop(califaID, ALL.tau_V__z)
            tau_V__yx = ALL.get_gal_prop(califaID, ALL.tau_V__yx).reshape(N_y, N_x)
            alogt_flux__z = ALL.get_gal_prop(califaID, ALL.at_flux__z)
            alogZ_mass__z = ALL.get_gal_prop(califaID, ALL.alogZ_mass__z)
            alogt_flux__yx = ALL.get_gal_prop(califaID, ALL.at_flux__yx).reshape(N_y, N_x)
            alogZ_mass__yx = ALL.get_gal_prop(califaID, ALL.alogZ_mass__yx).reshape(N_y, N_x)
            x_Y_32__yx = ALL.get_gal_prop(califaID, ALL.x_Y__Tyx[0]).reshape(N_y, N_x)
            x_Y_300__yx = ALL.get_gal_prop(califaID, ALL.x_Y__Tyx[1]).reshape(N_y, N_x)
            SFRSD_32__yx = ALL.get_gal_prop(califaID, ALL.SFRSD__Tyx[0]).reshape(N_y, N_x)
            SFRSD_300__yx = ALL.get_gal_prop(califaID, ALL.SFRSD__Tyx[1]).reshape(N_y, N_x)

            # get classes division (usually args.class_names is ['HIG', 'LIG', 'SF'])
            sel_classes_gal, _ = get_selections_spaxels(args, ALL, califaID, sel_WHa__c_gyx, sel_sample__gyx)

            # EML
            mean_f4861__gr[i_g], mean_f4861_npts__gr[i_g] = radialProfile(prop=f__lyx['4861'], bin_r=args.R_bin__r, x0=x0, y0=y0, pa=pa, ba=ba, rad_scale=HLR_pix, mask=gal_sample__yx, mode='mean', return_npts=True)
            mean_f6563__gr[i_g], mean_f6563_npts__gr[i_g] = radialProfile(prop=f__lyx['6563'], bin_r=args.R_bin__r, x0=x0, y0=y0, pa=pa, ba=ba, rad_scale=HLR_pix, mask=gal_sample__yx, mode='mean', return_npts=True)
            tau_V_neb__gr[i_g] = f_tauVneb(mean_f6563__gr[i_g], mean_f4861__gr[i_g])
            # m_aux = np.bitwise_or(~gal_sample__yx, np.bitwise_or(np.ma.getmaskarray(f__lyx['4861']), np.ma.getmaskarray(f__lyx['4861'])))
            # tau_V_neb_npts__gr[i_g] = (~m_aux).astype('int').sum()
            xm, ym = ma_mask_xyz(f__lyx['6563'], f__lyx['4861'])
            tau_V_neb_sumGAL__gr[i_g] = f_tauVneb(xm.sum(), ym.sum())  # np.ma.log(xm.sum() / ym.sum() / 2.86) / (q[0] - q[1])
            # print califaID
            # print mean_f4861_npts__gr[i_g]
            # print mean_f6563_npts__gr[i_g]

            # radial mean properties
            for k in args.class_names:
                # EML
                mean_f4861__cgr[k][i_g], mean_f4861_npts__cgr[k][i_g] = radialProfile(f__lyx['4861'], bin_r=args.R_bin__r, x0=x0, y0=y0, pa=pa, ba=ba, rad_scale=HLR_pix, mask=sel_classes_gal[k], mode='mean', return_npts=True)
                mean_f6563__cgr[k][i_g], mean_f6563_npts__cgr[k][i_g] = radialProfile(f__lyx['6563'], bin_r=args.R_bin__r, x0=x0, y0=y0, pa=pa, ba=ba, rad_scale=HLR_pix, mask=sel_classes_gal[k], mode='mean', return_npts=True)
                # print k
                # print mean_f4861_npts__cgr[k][i_g]
                # print mean_f6563_npts__cgr[k][i_g]
                tau_V_neb__cgr[k][i_g] = f_tauVneb(mean_f6563__cgr[k][i_g], mean_f4861__cgr[k][i_g])
                # m_aux = np.bitwise_or(~sel_classes_gal[k], np.bitwise_or(np.ma.getmaskarray(f__lyx['4861']), np.ma.getmaskarray(f__lyx['6563'])))
                # tau_V_neb_npts__cgr[k][i_g] = (~m_aux).astype('int').sum()

                # # SYN
                tau_V__cgr[k][i_g], tau_V_npts__cgr[k][i_g] = radialProfile(tau_V__yx, bin_r=args.R_bin__r, x0=x0, y0=y0, pa=pa, ba=ba, rad_scale=HLR_pix, mask=sel_classes_gal[k], mode='mean', return_npts=True)
                alogt_flux__cgr[k][i_g], alogt_flux_npts__cgr[k][i_g] = radialProfile(alogt_flux__yx, bin_r=args.R_bin__r, x0=x0, y0=y0, pa=pa, ba=ba, rad_scale=HLR_pix, mask=sel_classes_gal[k], mode='mean', return_npts=True)
                alogZ_mass__cgr[k][i_g], alogZ_mass_npts__cgr[k][i_g] = radialProfile(alogZ_mass__yx, bin_r=args.R_bin__r, x0=x0, y0=y0, pa=pa, ba=ba, rad_scale=HLR_pix, mask=sel_classes_gal[k], mode='mean', return_npts=True)

                # # SYN:tSF
                x_Y_32__cgr[k][i_g], x_Y_32_npts__cgr[k][i_g] = radialProfile(x_Y_32__yx, bin_r=args.R_bin__r, x0=x0, y0=y0, pa=pa, ba=ba, rad_scale=HLR_pix, mask=sel_classes_gal[k], mode='mean', return_npts=True)
                x_Y_300__cgr[k][i_g], x_Y_300_npts__cgr[k][i_g] = radialProfile(x_Y_300__yx, bin_r=args.R_bin__r, x0=x0, y0=y0, pa=pa, ba=ba, rad_scale=HLR_pix, mask=sel_classes_gal[k], mode='mean', return_npts=True)
                aSFRSD_32__cgr[k][i_g], aSFRSD_32_npts__cgr[k][i_g] = radialProfile(SFRSD_32__yx, bin_r=args.R_bin__r, x0=x0, y0=y0, pa=pa, ba=ba, rad_scale=HLR_pix, mask=sel_classes_gal[k], mode='mean', return_npts=True)
                aSFRSD_300__cgr[k][i_g], aSFRSD_300_npts__cgr[k][i_g] = radialProfile(SFRSD_300__yx, bin_r=args.R_bin__r, x0=x0, y0=y0, pa=pa, ba=ba, rad_scale=HLR_pix, mask=sel_classes_gal[k], mode='mean', return_npts=True)

        ALL.mean_f4861__gr = mean_f4861__gr
        ALL.mean_f4861_npts__gr = mean_f4861_npts__gr
        ALL.mean_f6563__gr = mean_f6563__gr
        ALL.mean_f6563_npts__gr = mean_f6563_npts__gr
        ALL.tau_V_neb__gr = tau_V_neb__gr
        # ALL.tau_V_neb_npts__gr = tau_V_neb_npts__gr
        ALL.tau_V_neb_sumGAL__g = tau_V_neb_sumGAL__gr

        ALL.mean_f4861__cgr = mean_f4861__cgr
        ALL.mean_f4861_npts__cgr = mean_f4861_npts__cgr
        ALL.mean_f6563__cgr = mean_f6563__cgr
        ALL.mean_f6563_npts__cgr = mean_f6563_npts__cgr
        ALL.tau_V_neb__cgr = tau_V_neb__cgr
        # ALL.tau_V_neb_npts__cgr = tau_V_neb_npts__cgr

        # SYN
        ALL.tau_V__cgr = tau_V__cgr
        ALL.tau_V_npts__cgr = tau_V_npts__cgr
        ALL.alogt_flux__cgr = alogt_flux__cgr
        ALL.alogt_flux_npts__cgr = alogt_flux_npts__cgr
        ALL.alogZ_mass__cgr = alogZ_mass__cgr
        ALL.alogZ_mass_npts__cgr = alogZ_mass_npts__cgr
        ALL.x_Y_32__cgr = x_Y_32__cgr
        ALL.x_Y_32_npts__cgr = x_Y_32_npts__cgr
        ALL.aSFRSD_32__cgr = aSFRSD_32__cgr
        ALL.aSFRSD_32_npts__cgr = aSFRSD_32_npts__cgr
        ALL.x_Y_300__cgr = x_Y_300__cgr
        ALL.x_Y_300_npts__cgr = x_Y_300_npts__cgr
        ALL.aSFRSD_300__cgr = aSFRSD_300__cgr
        ALL.aSFRSD_300_npts__cgr = aSFRSD_300_npts__cgr


def fig_tauVNeb_histo(args, gals, data_sel=None):
    print '#####################'
    print '# fig_tauVNeb_histo #'
    print '#####################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    if data_sel is None:
        data_sel = np.ones_like(sel['gals_sample__z'])
    sel_sample__gz = sel['gals_sample__z'] & data_sel
    sel_WHa = sel['WHa']
    N_gals = len(gals)

    Ha, Hb = ma_mask_xyz(ALL.f6563__z, ALL.f4861__z, mask=~sel_sample__gz)
    data__c = {}
    npts_6563__c = {}
    npts_4861__c = {}
    mask__c = {}
    for k in args.class_names:
        data__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.tau_V_neb__cgr[k]])
        npts_6563__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.mean_f6563_npts__cgr[k]])
        npts_4861__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.mean_f4861_npts__cgr[k]])
        mask__c[k] = np.hstack([np.ma.getmaskarray(x) for x in ALL.tau_V_neb__cgr[k]])
    # k='MIG'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]
    # k='SF'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]
    delta_tau_SFc_mDIG = np.ma.masked_array(data__c['SFc'] - data__c['mDIG'], mask=(mask__c['SFc'] | mask__c['mDIG']))
    delta_tau_SFc_hDIG = np.ma.masked_array(data__c['SFc'] - data__c['hDIG'], mask=(mask__c['SFc'] | mask__c['hDIG']))

    N_cols = 1
    N_rows = 3
    f = plot_setup(width=latex_column_width, aspect=2.2)
    # f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    gs = gridspec.GridSpec(N_rows, N_cols)
    # ax1, ax2, ax3 = axArr
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    fs = 6

    # AXIS 1
    x = f_tauVneb(Ha, Hb)
    # x = (1./(_q[0] - _q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
    range = [-2, 2]
    xDs = []
    for k in args.class_names:
        sel_aux = np.bitwise_and(sel_WHa['z'][k], sel_sample__gz)
        xDs.append(x[sel_aux].compressed())
    # plot_histo_ax(ax1, x.compressed(), stats_txt=False, histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, first=False, c=args.class_colors, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=1))
    x_ini = 0.98
    for j, k in enumerate(args.class_names):
        pos_y = 0.98
        for txt in text_list[j]:
            print k, txt
            plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=fs+2, va='top', ha='right', c=args.class_colors[j]))
            pos_y -= 0.08
        x_ini -= 0.25
    ax1.set_xlabel(r'$\tau_V^{neb}$')
    # ax1_top = ax1.twiny()
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.xaxis.set_minor_locator(MaxNLocator(20))
    ax1.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    # mn, mx = ax1.get_xlim()
    # unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
    # ax1_top.set_xlim(unit_converter(mn), unit_converter(mx))
    # # ax1_top.set_xlim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))
    # ax1_top.set_xlabel(r'$\log\ H\alpha/H\beta$')
    plot_text_ax(ax1, 'a)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    # AXIS 2
    x = delta_tau_SFc_mDIG
    range = DtauVnorm_range
    plot_histo_ax(ax2, x.compressed(), y_v_space=0.08, c='k', first=True, fs=fs+2, kwargs_histo=dict(normed=False, range=range))
    ax2.set_xlabel(r'$\Delta \tau\ =\ \tau_V^{%s}\ -\ \tau_V^{%s}$' % (args.class_names[-1], args.class_names[1]))
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_minor_locator(MaxNLocator(20))
    ax2.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    plot_text_ax(ax2, 'b)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    # AXIS 3
    x = delta_tau_SFc_hDIG
    range = DtauVnorm_range
    plot_histo_ax(ax3, x.compressed(), y_v_space=0.08, c='k', first=True, fs=fs+2, kwargs_histo=dict(normed=False, range=range))
    ax3.set_xlabel(r'$\Delta \tau\ =\ \tau_V^{%s}\ -\ \tau_V^{%s}}$' % (args.class_names[-1], args.class_names[0]))
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_minor_locator(MaxNLocator(20))
    ax3.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    plot_text_ax(ax3, 'c)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    # f.tight_layout(h_pad=0.05)
    gs.update(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.0)
    f.savefig('fig_tauVNeb_histograms.png', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_SFRSD_300_histo(args, gals, data_sel=None):
    print '#######################'
    print '# fig_SFRSD_300_histo #'
    print '#######################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    if data_sel is None:
        data_sel = np.ones_like(sel['gals_sample__z'])

    sel_sample__gz = sel['gals_sample__z'] & data_sel
    sel_gals_mt = sel['gals__mt_z']
    sel_WHa = sel['WHa']
    N_gals = len(gals)

    # Ha, Hb = ma_mask_xyz(ALL.f6563__z, ALL.f4861__z, mask=~sel_sample__gz)
    data__c = {}
    mask__c = {}
    data_npts__c = {}
    for k in args.class_names:
        data__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.aSFRSD_300__cgr[k]])
        mask__c[k] = np.hstack([np.ma.getmaskarray(x) for x in ALL.aSFRSD_300__cgr[k]])
        data_npts__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.aSFRSD_300_npts__cgr[k]])
    # k='MIG'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]
    # k='SF'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]
    delta_SFRSD_SFc_mDIG = np.ma.masked_array(np.ma.log10(data__c['SFc'] * 1e6) - np.ma.log10(data__c['mDIG'] * 1e6), mask=(mask__c['SFc'] | mask__c['mDIG']))
    delta_SFRSD_SFc_hDIG = np.ma.masked_array(np.ma.log10(data__c['SFc'] * 1e6) - np.ma.log10(data__c['hDIG'] * 1e6), mask=(mask__c['SFc'] | mask__c['hDIG']))

    N_cols = 1
    N_rows = 3
    f = plot_setup(width=latex_column_width, aspect=2.)
    # f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    gs = gridspec.GridSpec(N_rows, N_cols)
    # ax1, ax2, ax3 = axArr
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    fs = 6

    # AXIS 1
    x = np.ma.log10(ALL.SFRSD__Tz[1] * 1e6)
    # x = (1./(_q[0] - _q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
    range = [-5, -1]
    xDs = []
    for k in args.class_names:
        sel_aux = np.bitwise_and(sel_WHa['z'][k], sel_sample__gz)
        xDs.append(x[sel_aux].compressed())
    # plot_histo_ax(ax1, x.compressed(), stats_txt=False, histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, first=False, c=args.class_colors, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=1))
    x_ini = 0.98
    for j, k in enumerate(args.class_names):
        pos_y = 0.98
        for txt in text_list[j]:
            print k, txt
            plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=fs+2, va='top', ha='right', c=args.class_colors[j]))
            pos_y -= 0.08
        x_ini -= 0.25
    ax1.set_xlabel(r'$\log\ \Sigma_{SFR}^\star$ [$M_\odot$ yr${}^{-1}$ kpc${}^{-2}$]')
    ax1.set_title(r'$t_{SF}\ =\ 300\ Myrs$')
    ax1.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax1, 'a)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    # AXIS 2
    x = delta_SFRSD_SFc_mDIG
    range = [-1, 1]
    plot_histo_ax(ax2, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax2.set_xlabel(r'$\Delta \log\ \Sigma_{SFR}^\star(R)\ =\ %s - %s$' % (args.class_names[-1], args.class_names[1]))
    ax2.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax2, 'b)', 0.02, 0.98, fs, 'top', 'left', 'k')

    # AXIS 3
    x = delta_SFRSD_SFc_hDIG
    range = [-1, 1]
    plot_histo_ax(ax3, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax3.set_xlabel(r'$\Delta \log\ \Sigma_{SFR}^\star(R)\ =\ %s - %s$' % (args.class_names[-1], args.class_names[0]))
    ax3.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax3, 'c)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    gs.update(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.0)
    f.savefig('fig_SFRSD_300_histograms.png', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_SFRSD_32_histo(args, gals, data_sel=None):
    print '#######################'
    print '# fig_SFRSD_32_histo #'
    print '#######################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    if data_sel is None:
        data_sel = np.ones_like(sel['gals_sample__z'])

    sel_sample__gz = sel['gals_sample__z'] & data_sel
    sel_gals_mt = sel['gals__mt_z']
    sel_WHa = sel['WHa']
    N_gals = len(gals)

    # Ha, Hb = ma_mask_xyz(ALL.f6563__z, ALL.f4861__z, mask=~sel_sample__gz)
    data__c = {}
    mask__c = {}
    data_npts__c = {}
    for k in args.class_names:
        data__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.aSFRSD_32__cgr[k]])
        mask__c[k] = np.hstack([np.ma.getmaskarray(x) for x in ALL.aSFRSD_32__cgr[k]])
        data_npts__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.aSFRSD_32_npts__cgr[k]])
    # k='MIG'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]
    # k='SF'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]

    delta_SFRSD_SFc_mDIG = np.ma.masked_array(np.ma.log10(data__c['SFc'] * 1e6) - np.ma.log10(data__c['mDIG'] * 1e6), mask=(mask__c['SFc'] | mask__c['mDIG']))
    delta_SFRSD_SFc_hDIG = np.ma.masked_array(np.ma.log10(data__c['SFc'] * 1e6) - np.ma.log10(data__c['hDIG'] * 1e6), mask=(mask__c['SFc'] | mask__c['hDIG']))

    N_cols = 1
    N_rows = 3
    f = plot_setup(width=latex_column_width, aspect=2.)
    # f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    gs = gridspec.GridSpec(N_rows, N_cols)
    # ax1, ax2, ax3 = axArr
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    fs = 6

    # AXIS 1
    x = np.ma.log10(ALL.SFRSD__Tz[0] * 1e6)
    # x = (1./(_q[0] - _q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
    range = [-4, -0]
    xDs = []
    for k in args.class_names:
        sel_aux = np.bitwise_and(sel_WHa['z'][k], sel_sample__gz)
        xDs.append(x[sel_aux].compressed())
    # plot_histo_ax(ax1, x.compressed(), stats_txt=False, histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, first=False, c=args.class_colors, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=1))
    x_ini = 0.98
    for j, k in enumerate(args.class_names):
        pos_y = 0.98
        for txt in text_list[j]:
            print k, txt
            plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=fs+2, va='top', ha='right', c=args.class_colors[j]))
            pos_y -= 0.08
        x_ini -= 0.25
    ax1.set_xlabel(r'$\log\ \Sigma_{SFR}^\star$ [$M_\odot$ yr${}^{-1}$ kpc${}^{-2}$]')
    ax1.set_title(r'$t_{SF}\ =\ 32\ Myrs$')
    ax1.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax1, 'a)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    # AXIS 2
    x = delta_SFRSD_SFc_mDIG
    range = [-1, 1]
    plot_histo_ax(ax2, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax2.set_xlabel(r'$\Delta \log\ \Sigma_{SFR}^\star(R)\ =\ %s - %s$' % (args.class_names[-1], args.class_names[1]))
    ax2.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax2, 'b)', 0.02, 0.98, fs, 'top', 'left', 'k')

    # AXIS 3
    x = delta_SFRSD_SFc_hDIG
    range = [-1, 1]
    plot_histo_ax(ax3, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax3.set_xlabel(r'$\Delta \log\ \Sigma_{SFR}^\star(R)\ =\ %s - %s$' % (args.class_names[-1], args.class_names[0]))
    ax3.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax3, 'c)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    gs.update(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.0)
    f.savefig('fig_SFRSD_32_histograms.png', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_alogZ_mass_histo(args, gals):
    print '########################'
    print '# fig_alogZ_mass_histo #'
    print '########################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']
    sel_WHa = sel['WHa']
    N_gals = len(gals)

    # Ha, Hb = ma_mask_xyz(ALL.f6563__z, ALL.f4861__z, mask=~sel_sample__gz)
    data__c = {}
    mask__c = {}
    data_npts__c = {}
    for k in args.class_names:
        data__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.alogZ_mass__cgr[k]])
        aux = np.hstack([np.ma.getmaskarray(x) for x in ALL.alogZ_mass__cgr[k]])
        mask__c[k] = aux | np.isnan(data__c[k])
        data_npts__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.alogZ_mass_npts__cgr[k]])
    # k='MIG'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]
    # k='SF'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]

    delta_alogZ_mass_SFc_mDIG = np.ma.masked_array(data__c['SFc'] - data__c['mDIG'], mask=(mask__c['SFc'] | mask__c['mDIG']))
    delta_alogZ_mass_SFc_hDIG = np.ma.masked_array(data__c['SFc'] - data__c['hDIG'], mask=(mask__c['SFc'] | mask__c['hDIG']))
    N_cols = 1
    N_rows = 3
    f = plot_setup(width=latex_column_width, aspect=2.)
    # f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    gs = gridspec.GridSpec(N_rows, N_cols)
    # ax1, ax2, ax3 = axArr
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    fs = 6

    # AXIS 1
    x = np.ma.masked_array(ALL.alogZ_mass__z, mask=np.isnan(ALL.alogZ_mass__z))
    # x = (1./(_q[0] - _q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
    range = [-2.5, 0.5]
    xDs = []
    for k in args.class_names:
        sel_aux = np.bitwise_and(sel_WHa['z'][k], sel_sample__gz)
        xDs.append(x[sel_aux].compressed())
    # plot_histo_ax(ax1, x.compressed(), stats_txt=False, histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, first=False, c=args.class_colors, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=1))
    x_ini = 0.98
    for j, k in enumerate(args.class_names):
        pos_y = 0.98
        for txt in text_list[j]:
            print k, txt
            plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=fs+2, va='top', ha='right', c=args.class_colors[j]))
            pos_y -= 0.08
        x_ini -= 0.25
    ax1.set_xlabel(r'$\langle \log\ Z_\star \rangle_M\ [Z_\odot]$')
    # r'$\Sigma_{SFR}^\star$ [$M_\odot$ yr${}^{-1}$ kpc${}^{-2}$]')
    # ax1.set_title(r'$t_{SF}\ =\ 32\ Myrs$')
    ax1.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax1, 'a)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    # AXIS 2
    x = delta_alogZ_mass_SFc_mDIG
    range = [-0.5, 0.5]
    plot_histo_ax(ax2, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax2.set_xlabel(r'$\Delta \langle \log\ Z_\star \rangle_M\ =\ %s - %s$' % (args.class_names[-1], args.class_names[1]))
    ax2.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax2, 'b)', 0.02, 0.98, fs, 'top', 'left', 'k')

    # AXIS 3
    x = delta_alogZ_mass_SFc_hDIG
    range = [-0.5, 0.5]
    plot_histo_ax(ax3, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax3.set_xlabel(r'$\Delta \langle \log\ Z_\star \rangle_M\ =\ %s - %s$' % (args.class_names[-1], args.class_names[0]))
    ax3.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax3, 'c)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    gs.update(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.0)
    f.savefig('fig_alogZmass_histograms.png', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_alogt_flux_histo(args, gals):
    print '########################'
    print '# fig_alogt_flux_histo #'
    print '########################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']
    sel_WHa = sel['WHa']
    N_gals = len(gals)

    # Ha, Hb = ma_mask_xyz(ALL.f6563__z, ALL.f4861__z, mask=~sel_sample__gz)
    data__c = {}
    mask__c = {}
    data_npts__c = {}
    for k in args.class_names:
        data__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.alogt_flux__cgr[k]])
        aux = np.hstack([np.ma.getmaskarray(x) for x in ALL.alogt_flux__cgr[k]])
        mask__c[k] = aux | np.isnan(data__c[k])
        data_npts__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.alogt_flux_npts__cgr[k]])
    # k='MIG'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]
    # k='SF'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]

    delta_alogt_flux_SFc_mDIG = np.ma.masked_array(data__c['SFc'] - data__c['mDIG'], mask=(mask__c['SFc'] | mask__c['mDIG']))
    delta_alogt_flux_SFc_hDIG = np.ma.masked_array(data__c['SFc'] - data__c['hDIG'], mask=(mask__c['SFc'] | mask__c['hDIG']))
    N_cols = 1
    N_rows = 3
    f = plot_setup(width=latex_column_width, aspect=2.)
    # f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    gs = gridspec.GridSpec(N_rows, N_cols)
    # ax1, ax2, ax3 = axArr
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    fs = 6

    # AXIS 1
    x = np.ma.masked_array(ALL.at_flux__z, mask=np.isnan(ALL.at_flux__z))
    # x = (1./(_q[0] - _q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
    range = [7, 10]
    xDs = []
    for k in args.class_names:
        sel_aux = np.bitwise_and(sel_WHa['z'][k], sel_sample__gz)
        xDs.append(x[sel_aux].compressed())
    # plot_histo_ax(ax1, x.compressed(), stats_txt=False, histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, first=False, c=args.class_colors, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=1))
    x_ini = 0.98
    for j, k in enumerate(args.class_names):
        pos_y = 0.98
        for txt in text_list[j]:
            print k, txt
            plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=fs+2, va='top', ha='right', c=args.class_colors[j]))
            pos_y -= 0.08
        x_ini -= 0.25
    ax1.set_xlabel(r'$\langle \log\ Z_\star \rangle_M\ [Z_\odot]$')
    # r'$\Sigma_{SFR}^\star$ [$M_\odot$ yr${}^{-1}$ kpc${}^{-2}$]')
    # ax1.set_title(r'$t_{SF}\ =\ 32\ Myrs$')
    ax1.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax1, 'a)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    # AXIS 2
    x = delta_alogt_flux_SFc_mDIG
    range = [-1, 1]
    plot_histo_ax(ax2, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax2.set_xlabel(r'$\Delta \langle \log\ Z_\star \rangle_M\ =\ %s - %s$' % (args.class_names[-1], args.class_names[1]))
    ax2.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax2, 'b)', 0.02, 0.98, fs, 'top', 'left', 'k')

    # AXIS 3
    x = delta_alogt_flux_SFc_hDIG
    range = [-1, 1]
    plot_histo_ax(ax3, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax3.set_xlabel(r'$\Delta \langle \log\ Z_\star \rangle_M\ =\ %s - %s$' % (args.class_names[-1], args.class_names[0]))
    ax3.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax3, 'c)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    gs.update(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.0)
    f.savefig('fig_atflux_histograms.png', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_x_Y_32_histo(args, gals):
    print '####################'
    print '# fig_x_Y_32_histo #'
    print '####################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']
    sel_WHa = sel['WHa']
    N_gals = len(gals)

    # Ha, Hb = ma_mask_xyz(ALL.f6563__z, ALL.f4861__z, mask=~sel_sample__gz)
    data__c = {}
    mask__c = {}
    data_npts__c = {}
    for k in args.class_names:
        data__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.x_Y_32__cgr[k]])
        aux = np.hstack([np.ma.getmaskarray(x) for x in ALL.x_Y_32__cgr[k]])
        mask__c[k] = aux | np.isnan(data__c[k])
        data_npts__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.x_Y_32_npts__cgr[k]])
    # k='MIG'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]
    # k='SF'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]

    delta_x_Y_SFc_mDIG = np.ma.masked_array(data__c['SFc'] - data__c['mDIG'], mask=(mask__c['SFc'] | mask__c['mDIG']))
    delta_x_Y_SFc_hDIG = np.ma.masked_array(data__c['SFc'] - data__c['hDIG'], mask=(mask__c['SFc'] | mask__c['hDIG']))
    N_cols = 1
    N_rows = 3
    f = plot_setup(width=latex_column_width, aspect=2.)
    # f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    gs = gridspec.GridSpec(N_rows, N_cols)
    # ax1, ax2, ax3 = axArr
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    fs = 6

    # AXIS 1
    x = np.ma.masked_array(ALL.x_Y__Tz[0], mask=np.isnan(ALL.x_Y__Tz[0]))
    # x = (1./(_q[0] - _q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
    range = [0, 1]
    xDs = []
    for k in args.class_names:
        sel_aux = np.bitwise_and(sel_WHa['z'][k], sel_sample__gz)
        xDs.append(x[sel_aux].compressed())
    # plot_histo_ax(ax1, x.compressed(), stats_txt=False, histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, first=False, c=args.class_colors, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=1, bins=np.arange(0., 1.05, 0.05)))
    x_ini = 0.98
    for j, k in enumerate(args.class_names):
        pos_y = 0.98
        for txt in text_list[j]:
            print k, txt
            plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=fs+2, va='top', ha='right', c=args.class_colors[j]))
            pos_y -= 0.08
        x_ini -= 0.25
    ax1.set_xlabel(r'$x_Y$ [frac.]')
    # r'$\Sigma_{SFR}^\star$ [$M_\odot$ yr${}^{-1}$ kpc${}^{-2}$]')
    ax1.set_title(r'$t_{SF}\ =\ 32\ Myrs$')
    ax1.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax1, 'a)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    # AXIS 2
    x = delta_x_Y_SFc_mDIG
    range = [-0.3, 0.3]
    plot_histo_ax(ax2, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax2.set_xlabel(r'$\Delta \langle \log\ Z_\star \rangle_M\ =\ %s - %s$' % (args.class_names[-1], args.class_names[1]))
    ax2.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax2, 'b)', 0.02, 0.98, fs, 'top', 'left', 'k')

    # AXIS 3
    x = delta_x_Y_SFc_hDIG
    range = [-0.3, 0.3]
    plot_histo_ax(ax3, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax3.set_xlabel(r'$\Delta \langle \log\ Z_\star \rangle_M\ =\ %s - %s$' % (args.class_names[-1], args.class_names[0]))
    ax3.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax3, 'c)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    gs.update(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.0)
    f.savefig('fig_x_Y_32_histograms.png', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_x_Y_300_histo(args, gals):
    print '####################'
    print '# fig_x_Y_300_histo #'
    print '####################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']
    sel_WHa = sel['WHa']
    N_gals = len(gals)

    # Ha, Hb = ma_mask_xyz(ALL.f6563__z, ALL.f4861__z, mask=~sel_sample__gz)
    data__c = {}
    mask__c = {}
    data_npts__c = {}
    for k in args.class_names:
        data__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.x_Y_300__cgr[k]])
        aux = np.hstack([np.ma.getmaskarray(x) for x in ALL.x_Y_300__cgr[k]])
        mask__c[k] = aux | np.isnan(data__c[k])
        data_npts__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.x_Y_300_npts__cgr[k]])
    # k='MIG'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]
    # k='SF'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]

    delta_x_Y_SFc_mDIG = np.ma.masked_array(data__c['SFc'] - data__c['mDIG'], mask=(mask__c['SFc'] | mask__c['mDIG']))
    delta_x_Y_SFc_hDIG = np.ma.masked_array(data__c['SFc'] - data__c['hDIG'], mask=(mask__c['SFc'] | mask__c['hDIG']))
    N_cols = 1
    N_rows = 3
    f = plot_setup(width=latex_column_width, aspect=2.)
    # f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    gs = gridspec.GridSpec(N_rows, N_cols)
    # ax1, ax2, ax3 = axArr
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    fs = 6

    # AXIS 1
    x = np.ma.masked_array(ALL.x_Y__Tz[1], mask=np.isnan(ALL.x_Y__Tz[1]))
    # x = (1./(_q[0] - _q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
    range = [0, 1]
    xDs = []
    for k in args.class_names:
        sel_aux = np.bitwise_and(sel_WHa['z'][k], sel_sample__gz)
        xDs.append(x[sel_aux].compressed())
    # plot_histo_ax(ax1, x.compressed(), stats_txt=False, histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, first=False, c=args.class_colors, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=1, bins=np.arange(0., 1.05, 0.05)))
    x_ini = 0.98
    for j, k in enumerate(args.class_names):
        pos_y = 0.98
        for txt in text_list[j]:
            print k, txt
            plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=fs+2, va='top', ha='right', c=args.class_colors[j]))
            pos_y -= 0.08
        x_ini -= 0.25
    ax1.set_xlabel(r'$x_Y$ [frac.]')
    # r'$\Sigma_{SFR}^\star$ [$M_\odot$ yr${}^{-1}$ kpc${}^{-2}$]')
    ax1.set_title(r'$t_{SF}\ =\ 300\ Myrs$')
    ax1.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax1, 'a)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    # AXIS 2
    x = delta_x_Y_SFc_mDIG
    range = [-0.3, 0.3]
    plot_histo_ax(ax2, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax2.set_xlabel(r'$\Delta \langle \log\ Z_\star \rangle_M\ =\ %s - %s$' % (args.class_names[-1], args.class_names[1]))
    ax2.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax2, 'b)', 0.02, 0.98, fs, 'top', 'left', 'k')

    # AXIS 3
    x = delta_x_Y_SFc_hDIG
    range = [-0.3, 0.3]
    plot_histo_ax(ax3, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax3.set_xlabel(r'$\Delta \langle \log\ Z_\star \rangle_M\ =\ %s - %s$' % (args.class_names[-1], args.class_names[0]))
    ax3.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax3, 'c)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    gs.update(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.0)
    f.savefig('fig_x_Y_300_histograms.png', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_tau_V_histo(args, gals):
    print '###################'
    print '# fig_tau_V_histo #'
    print '###################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']
    sel_WHa = sel['WHa']
    N_gals = len(gals)

    # Ha, Hb = ma_mask_xyz(ALL.f6563__z, ALL.f4861__z, mask=~sel_sample__gz)
    data__c = {}
    mask__c = {}
    data_npts__c = {}
    for k in args.class_names:
        data__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.tau_V__cgr[k]])
        aux = np.hstack([np.ma.getmaskarray(x) for x in ALL.tau_V__cgr[k]])
        mask__c[k] = aux | np.isnan(data__c[k])
        data_npts__c[k] = np.hstack([np.ma.getdata(x) for x in ALL.tau_V_npts__cgr[k]])
    # k='MIG'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]
    # k='SF'
    # print k, mask__c[k].sum(), ALL.tau_V_neb__cgr[k]

    delta_tau_V_SFc_mDIG = np.ma.masked_array(data__c['SFc'] - data__c['mDIG'], mask=(mask__c['SFc'] | mask__c['mDIG']))
    delta_tau_V_SFc_hDIG = np.ma.masked_array(data__c['SFc'] - data__c['hDIG'], mask=(mask__c['SFc'] | mask__c['hDIG']))
    N_cols = 1
    N_rows = 3
    f = plot_setup(width=latex_column_width, aspect=2.)
    # f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    gs = gridspec.GridSpec(N_rows, N_cols)
    # ax1, ax2, ax3 = axArr
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    fs = 6

    # AXIS 1
    x = np.ma.masked_array(ALL.tau_V__z, mask=np.isnan(ALL.tau_V__z))
    # x = (1./(_q[0] - _q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
    range = [0, 1]
    xDs = []
    for k in args.class_names:
        sel_aux = np.bitwise_and(sel_WHa['z'][k], sel_sample__gz)
        xDs.append(x[sel_aux].compressed())
    # plot_histo_ax(ax1, x.compressed(), stats_txt=False, histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, first=False, c=args.class_colors, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=1))
    x_ini = 0.98
    for j, k in enumerate(args.class_names):
        pos_y = 0.98
        for txt in text_list[j]:
            print k, txt
            plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=fs+2, va='top', ha='right', c=args.class_colors[j]))
            pos_y -= 0.08
        x_ini -= 0.25
    ax1.set_xlabel(r'$\langle \log\ Z_\star \rangle_M\ [Z_\odot]$')
    # r'$\Sigma_{SFR}^\star$ [$M_\odot$ yr${}^{-1}$ kpc${}^{-2}$]')
    # ax1.set_title(r'$t_{SF}\ =\ 32\ Myrs$')
    ax1.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax1, 'a)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    # AXIS 2
    x = delta_tau_V_SFc_mDIG
    range = [-1, 2]
    plot_histo_ax(ax2, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax2.set_xlabel(r'$\Delta \langle \log\ Z_\star \rangle_M\ =\ %s - %s$' % (args.class_names[-1], args.class_names[1]))
    ax2.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax2, 'b)', 0.02, 0.98, fs, 'top', 'left', 'k')

    # AXIS 3
    x = delta_tau_V_SFc_hDIG
    range = [-1, 2]
    plot_histo_ax(ax3, x.compressed(), fs=fs+2, y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax3.set_xlabel(r'$\Delta \langle \log\ Z_\star \rangle_M\ =\ %s - %s$' % (args.class_names[-1], args.class_names[0]))
    ax3.tick_params(axis='both', which='both', direction='in', bottom='on', top='on', left='on', right='on', labelbottom='on', labeltop='off', labelleft='on', labelright='off')
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_minor_locator(MaxNLocator(20))
    plot_text_ax(ax3, 'c)', 0.02, 0.98, fs+2, 'top', 'left', 'k')

    gs.update(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.0)
    f.savefig('fig_tau_V_histograms.png', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_WHa_histograms_per_morftype_and_ba_cumulFHamed(args, gals):
    print '######################################################'
    print '# fig_WHa_histograms_per_morftype_and_ba_cumulFHamed #'
    print '######################################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_ba = sel['gals__ba_z']
    sel_gals_mt = sel['gals__mt_z']

    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ALL.mt_keys
    sel_gals_sample_balow__gz = sel_gals_ba['ba_low']
    sel_gals_sample_bamid__gz = sel_gals_ba['ba_mid']
    sel_gals_sample_bahigh__gz = sel_gals_ba['ba_high']
    N_rows, N_cols = 7, 4
    f = plot_setup(width=latex_text_width, aspect=1./golden_mean)
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    N_zone = sel_sample__gz.astype('int').sum()

    gs = gridspec.GridSpec(N_rows, N_cols)
    axes_col0 = [
        plt.subplot(gs[0,0]), plt.subplot(gs[1,0]), plt.subplot(gs[2,0]),
        plt.subplot(gs[3,0]), plt.subplot(gs[4,0]), plt.subplot(gs[5,0]),
        plt.subplot(gs[6,0])
    ]
    axes_col1 = [
        plt.subplot(gs[0,1]), plt.subplot(gs[1,1]), plt.subplot(gs[2,1]),
        plt.subplot(gs[3,1]), plt.subplot(gs[4,1]), plt.subplot(gs[5,1]),
        plt.subplot(gs[6,1])
    ]
    axes_col2 = [
        plt.subplot(gs[0,2]), plt.subplot(gs[1,2]), plt.subplot(gs[2,2]),
        plt.subplot(gs[3,2]), plt.subplot(gs[4,2]), plt.subplot(gs[5,2]),
        plt.subplot(gs[6,2])
    ]
    axes_col3 = [
        plt.subplot(gs[0,3]), plt.subplot(gs[1,3]), plt.subplot(gs[2,3]),
        plt.subplot(gs[3,3]), plt.subplot(gs[4,3]), plt.subplot(gs[5,3]),
        plt.subplot(gs[6,3])
    ]

    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]
    fs = 6

    logWHa_range = [-1, 2.5]
    ax = axes_cols[0][0]
    ax.set_title(r'All b/a', fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    aux_gals = ALL.califaID__g[sel['gals__ba']['all']].tolist()
    for g in aux_gals:
        x = np.log10(args.class_thresholds[0])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in aux_gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in aux_gals])
    f_SFc = np.mean([f_SFc__g[g] for g in aux_gals])
    m_aux = sel_sample__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('hDIG', (100. * f_hDIG)), 0.99, 0.98, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('mDIG', (100. * f_mDIG)), 0.99, 0.82, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('SFc', (100. * f_SFc)), 0.99, 0.66, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    plot_histo_ax(ax, xm.compressed(), first=True, stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=1))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[1][0]
    ax.set_title(r'$b/a\ \leq$ %.2f' % ALL.ba_bins[0], fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    aux_gals = ALL.califaID__g[sel['gals__ba']['ba_low']].tolist()
    for g in aux_gals:
        x = np.log10(args.class_thresholds[0])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in aux_gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in aux_gals])
    f_SFc = np.mean([f_SFc__g[g] for g in aux_gals])
    fs = 6
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_balow__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[2][0]
    ax.set_title(r'%.2f $<\ b/a\ \leq$ %.2f' % (ALL.ba_bins[0], ALL.ba_bins[1]), fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    aux_gals = ALL.califaID__g[sel['gals__ba']['ba_mid']].tolist()
    for g in aux_gals:
        x = np.log10(args.class_thresholds[0])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in aux_gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in aux_gals])
    f_SFc = np.mean([f_SFc__g[g] for g in aux_gals])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_bamid__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[3][0]
    ax.set_title(r'$b/a\ >$ %.2f' % ALL.ba_bins[1], fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    aux_gals = ALL.califaID__g[sel['gals__ba']['ba_high']].tolist()
    for g in aux_gals:
        x = np.log10(args.class_thresholds[0])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in aux_gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in aux_gals])
    f_SFc = np.mean([f_SFc__g[g] for g in aux_gals])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_bahigh__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    for col in range(N_cols):
        k = ALL.ba_keys[col]
        for i, mt_label in enumerate(mtype_labels):
            ax = axes_cols[col][i+1]

            if not col:
                if (mt_label == mtype_labels[-1]):
                    lab = r'$\geq$ Sc'
                else:
                    lab = mt_label
                ax.set_ylabel(lab, rotation=90, fontsize=fs+4)

            m_aux = np.bitwise_and(sel_gals_mt[mt_label], sel['gals__ba_z'][k])
            f_SFc__g = {}
            f_mDIG__g = {}
            f_hDIG__g = {}
            aux_gals_mt = ALL.califaID__g[sel['gals__mt'][mt_label]].tolist()
            aux_gals_ba = ALL.califaID__g[sel['gals__ba'][k]]
            _gals = ALL.califaID__g[sel['gals__mt'][mt_label] & sel['gals__ba'][k]].tolist()
            for g in _gals:
                if not g in args.gals:
                    _gals.remove(g)
            for g in _gals:
                x = np.log10(args.class_thresholds[0])
                y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
                y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
                x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
                x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
                y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
                print x, y_th, x1, x2, y1, y2
                f_hDIG__g[g] = y_th
                x = np.log10(args.class_thresholds[1])
                y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
                y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
                x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
                x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
                y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
                f_SFc__g[g] = 1. - y_th
                print x, y_th, x1, x2, y1, y2
                f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
                print g, f_hDIG__g[g], f_mDIG__g[g], f_SFc__g[g]
            f_hDIG = np.mean([f_hDIG__g[g] for g in _gals])
            f_mDIG = np.mean([f_mDIG__g[g] for g in _gals])
            f_SFc = np.mean([f_SFc__g[g] for g in _gals])
            plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
            plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
            plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
            xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
            # plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
            plot_histo_ax(ax, xm.compressed(), first=False, stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=logWHa_range, lw=2))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_minor_locator(MaxNLocator(20))
            ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            # ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            for th in args.class_thresholds:
                ax.axvline(x=np.ma.log10(th), c='k', ls='--')
            if i == (len(colortipo) - 1):
                # ax.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
                ax.tick_params(axis='x', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')
                if col == (N_cols - 1):
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
            else:
                ax.tick_params(axis='both', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
                # if not col:
                #     ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
                # else:
                #     ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            ax.set_xlim(logWHa_range)

    # f.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    gs.update(left=0.05, right=0.95, bottom=0.14, top=0.9, hspace=0.0, wspace=0.0)
    f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=fs+4)
    # f.set_tight_layout(True)
    f.savefig('fig_WHa_histograms_per_morftype_and_ba_cumulFHa.pdf', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'
    return ALL


def fig_WHa_histograms_per_morftype_and_mlba_cumulFHamed(args, gals):
    print '######################################################'
    print '# fig_WHa_histograms_per_morftype_and_mlba_cumulFHamed #'
    print '######################################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_ba = sel['gals__ba_z']
    sel_gals_mt = sel['gals__mt_z']

    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ALL.mt_keys
    sel_gals_sample_balow__gz = sel_gals_ba['ba_low']
    sel_gals_sample_bamid__gz = sel_gals_ba['ba_mid']
    sel_gals_sample_bahigh__gz = sel_gals_ba['ba_high']
    N_rows, N_cols = 7, 4
    f = plot_setup(width=latex_text_width, aspect=1./golden_mean)
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    N_zone = sel_sample__gz.astype('int').sum()

    gs = gridspec.GridSpec(N_rows, N_cols)
    axes_col0 = [
        plt.subplot(gs[0,0]), plt.subplot(gs[1,0]), plt.subplot(gs[2,0]),
        plt.subplot(gs[3,0]), plt.subplot(gs[4,0]), plt.subplot(gs[5,0]),
        plt.subplot(gs[6,0])
    ]
    axes_col1 = [
        plt.subplot(gs[0,1]), plt.subplot(gs[1,1]), plt.subplot(gs[2,1]),
        plt.subplot(gs[3,1]), plt.subplot(gs[4,1]), plt.subplot(gs[5,1]),
        plt.subplot(gs[6,1])
    ]
    axes_col2 = [
        plt.subplot(gs[0,2]), plt.subplot(gs[1,2]), plt.subplot(gs[2,2]),
        plt.subplot(gs[3,2]), plt.subplot(gs[4,2]), plt.subplot(gs[5,2]),
        plt.subplot(gs[6,2])
    ]
    axes_col3 = [
        plt.subplot(gs[0,3]), plt.subplot(gs[1,3]), plt.subplot(gs[2,3]),
        plt.subplot(gs[3,3]), plt.subplot(gs[4,3]), plt.subplot(gs[5,3]),
        plt.subplot(gs[6,3])
    ]

    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]
    fs = 6

    logWHa_range = [-1, 2.5]
    ax = axes_cols[0][0]
    ax.set_title(r'All b/a', fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    aux_gals = ALL.califaID__g[sel['gals__mlba']['all']].tolist()
    for g in aux_gals:
        x = np.log10(args.class_thresholds[0])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in aux_gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in aux_gals])
    f_SFc = np.mean([f_SFc__g[g] for g in aux_gals])
    m_aux = sel_sample__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('hDIG', (100. * f_hDIG)), 0.99, 0.98, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('mDIG', (100. * f_mDIG)), 0.99, 0.82, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('SFc', (100. * f_SFc)), 0.99, 0.66, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    plot_histo_ax(ax, xm.compressed(), first=True, stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[1][0]
    ax.set_title(r'$b/a\ \leq$ %.2f' % ALL.ba_bins[0], fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    aux_gals = ALL.califaID__g[sel['gals__mlba']['ba_low']].tolist()
    for g in aux_gals:
        x = np.log10(args.class_thresholds[0])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in aux_gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in aux_gals])
    f_SFc = np.mean([f_SFc__g[g] for g in aux_gals])
    fs = 6
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_balow__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[2][0]
    ax.set_title(r'%.2f $<\ b/a\ \leq$ %.2f' % (ALL.ba_bins[0], ALL.ba_bins[1]), fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    aux_gals = ALL.califaID__g[sel['gals__mlba']['ba_mid']].tolist()
    for g in aux_gals:
        x = np.log10(args.class_thresholds[0])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in aux_gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in aux_gals])
    f_SFc = np.mean([f_SFc__g[g] for g in aux_gals])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_bamid__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[3][0]
    ax.set_title(r'$b/a\ >$ %.2f' % ALL.ba_bins[1], fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    aux_gals = ALL.califaID__g[sel['gals__mlba']['ba_high']].tolist()
    for g in aux_gals:
        x = np.log10(args.class_thresholds[0])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
        y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in aux_gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in aux_gals])
    f_SFc = np.mean([f_SFc__g[g] for g in aux_gals])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_bahigh__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    for col in range(N_cols):
        k = ALL.ba_keys[col]
        for i, mt_label in enumerate(mtype_labels):
            ax = axes_cols[col][i+1]
            if not col:
                if (mt_label == mtype_labels[-1]):
                    lab = r'$\geq$ Sc'
                else:
                    lab = mt_label
                ax.set_ylabel(lab, rotation=90, fontsize=fs+4)
            m_aux = np.bitwise_and(sel_gals_mt[mt_label], sel['gals__mlba_z'][k])
            f_SFc__g = {}
            f_mDIG__g = {}
            f_hDIG__g = {}
            aux_gals_mt = ALL.califaID__g[sel['gals__mt'][mt_label]].tolist()
            aux_gals_ba = ALL.califaID__g[sel['gals__mlba'][k]]
            _gals = ALL.califaID__g[sel['gals__mt'][mt_label] & sel['gals__mlba'][k]].tolist()
            for g in _gals:
                if not g in args.gals:
                    _gals.remove(g)
            for g in _gals:
                x = np.log10(args.class_thresholds[0])
                y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
                y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
                x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
                x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
                y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
                print x, y_th, x1, x2, y1, y2
                f_hDIG__g[g] = y_th
                x = np.log10(args.class_thresholds[1])
                y1 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w < x][-1]
                y2 = ALL.cumulfHa__gz[g][ALL.logWHa_bins__w > x][0]
                x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
                x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
                y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
                f_SFc__g[g] = 1. - y_th
                print x, y_th, x1, x2, y1, y2
                f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
                print g, f_hDIG__g[g], f_mDIG__g[g], f_SFc__g[g]
            f_hDIG = np.mean([f_hDIG__g[g] for g in _gals])
            f_mDIG = np.mean([f_mDIG__g[g] for g in _gals])
            f_SFc = np.mean([f_SFc__g[g] for g in _gals])
            plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
            plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
            plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
            xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
            # plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
            plot_histo_ax(ax, xm.compressed(), first=False, stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=logWHa_range, lw=2))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_minor_locator(MaxNLocator(20))
            ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            # ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            for th in args.class_thresholds:
                ax.axvline(x=np.ma.log10(th), c='k', ls='--')
            if i == (len(colortipo) - 1):
                # ax.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
                ax.tick_params(axis='x', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')
                if col == (N_cols - 1):
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
            else:
                ax.tick_params(axis='both', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
                # if not col:
                #     ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
                # else:
                #     ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            ax.set_xlim(logWHa_range)

    # f.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    gs.update(left=0.05, right=0.95, bottom=0.14, top=0.9, hspace=0.0, wspace=0.0)
    f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=fs+4)
    # f.set_tight_layout(True)
    f.savefig('fig_WHa_histograms_per_morftype_and_mlba_cumulFHa.pdf', dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'
    return ALL


def fig_WHa_histograms_per_morftype_norm_pix(args, gals, gals_sel=None, data_sel=None, fname_suffix='', histo_stats=False):
    print '############################################'
    print '# fig_WHa_histograms_per_morftype_norm_pix #'
    print '############################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    if gals_sel is None:
        gals_sel = np.ones_like(gals, dtype='bool')
    sample_gals = sel['gals'] & gals_sel

    if data_sel is None:
        data_sel = np.ones_like(sel['gals_sample__z'], dtype='bool')
    sel_sample__gz = sel['gals_sample__z'] & data_sel
    sel_gals_mt = sel['gals__mt_z']

    print len(gals), len(sample_gals)
    gals = ALL.califaID__g[sample_gals]
    for g in gals:
        print g, ALL.ba[ALL.califaID__g == g]

    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', '>= Sc']
    sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z'] & data_sel
    sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z'] & data_sel
    sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z'] & data_sel
    N_rows, N_cols = 7, 4
    f = plot_setup(width=latex_text_width, aspect=1./golden_mean)
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    N_zone = sel_sample__gz.astype('int').sum()

    gs = gridspec.GridSpec(N_rows, N_cols)
    axes_col0 = [
        plt.subplot(gs[0,0]), plt.subplot(gs[1,0]), plt.subplot(gs[2,0]),
        plt.subplot(gs[3,0]), plt.subplot(gs[4,0]), plt.subplot(gs[5,0]),
        plt.subplot(gs[6,0])
    ]
    axes_col1 = [
        plt.subplot(gs[0,1]), plt.subplot(gs[1,1]), plt.subplot(gs[2,1]),
        plt.subplot(gs[3,1]), plt.subplot(gs[4,1]), plt.subplot(gs[5,1]),
        plt.subplot(gs[6,1])
    ]
    axes_col2 = [
        plt.subplot(gs[0,2]), plt.subplot(gs[1,2]), plt.subplot(gs[2,2]),
        plt.subplot(gs[3,2]), plt.subplot(gs[4,2]), plt.subplot(gs[5,2]),
        plt.subplot(gs[6,2])
    ]
    axes_col3 = [
        plt.subplot(gs[0,3]), plt.subplot(gs[1,3]), plt.subplot(gs[2,3]),
        plt.subplot(gs[3,3]), plt.subplot(gs[4,3]), plt.subplot(gs[5,3]),
        plt.subplot(gs[6,3])
    ]

    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]
    fs = 6

    sels_R = [sel_sample__gz, sel_gals_sample_Rin__gz, sel_gals_sample_Rmid__gz, sel_gals_sample_Rout__gz]

    logWHa_range = [-1, 2.5]
    ax = axes_cols[0][0]
    ax.set_title(r'All radii', fontsize=fs+4, y=1.02)
    EW_bin_edges = np.linspace(logWHa_range[0], logWHa_range[1], 51)
    EW_bin_center = (EW_bin_edges[:-1]+EW_bin_edges[1:])/2.
    H = np.zeros(EW_bin_center.shape, dtype='float')
    for g in gals:
        R = ALL.get_gal_prop(g, ALL.zoneDistance_HLR)
        sel_g = ALL.get_gal_prop(g, sel_sample__gz)
        # x = np.ma.masked_array(np.ma.log10(ALL.get_gal_prop(g, ALL.W6563__z)), mask=(R > 2))
        x = np.ma.masked_array(np.ma.log10(ALL.get_gal_prop(g, ALL.W6563__z)), mask=~sel_g)
        N = x.count()
        Hg, xe = np.histogram(x.compressed(), bins=EW_bin_edges)
        Hgnorm = 1.* Hg/N
        H += Hgnorm
    print H
    ax.bar(EW_bin_center, H, align='center', color='darkgrey', width=np.diff(EW_bin_edges))
    for x, y in zip(EW_bin_center, H):
        print x, y
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.set_ylim(0, H.max())
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    logWHa_range = [-1, 2.5]
    ax = axes_cols[1][0]
    ax.set_title(r'$R\ \leq$ 0.5 HLR', fontsize=fs+4, y=1.02)
    EW_bin_edges = np.linspace(logWHa_range[0], logWHa_range[1], 51)
    EW_bin_center = (EW_bin_edges[:-1]+EW_bin_edges[1:])/2.
    H = np.zeros(EW_bin_center.shape, dtype='float')
    for g in gals:
        R = ALL.get_gal_prop(g, ALL.zoneDistance_HLR)
        sel_g = ALL.get_gal_prop(g, sel_sample__gz)
        x = np.ma.masked_array(np.ma.log10(ALL.get_gal_prop(g, ALL.W6563__z)), mask=~((R <= 0.5) & sel_g))
        # x = np.ma.log10(ALL.get_gal_prop(g, ALL.W6563__z))
        N = x.count()
        Hg, xe = np.histogram(x.compressed(), bins=EW_bin_edges)
        Hgnorm = 1.* Hg/N
        H += Hgnorm
    print H
    ax.bar(EW_bin_center, H, align='center', color='darkgrey', width=np.diff(EW_bin_edges))
    for x, y in zip(EW_bin_center, H):
        print x, y
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.set_ylim(0, H.max())
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    logWHa_range = [-1, 2.5]
    ax = axes_cols[2][0]
    ax.set_title(r'0.5 $<\ R\ \leq$ 1 HLR', fontsize=fs+4, y=1.02)
    EW_bin_edges = np.linspace(logWHa_range[0], logWHa_range[1], 51)
    EW_bin_center = (EW_bin_edges[:-1]+EW_bin_edges[1:])/2.
    H = np.zeros(EW_bin_center.shape, dtype='float')
    for g in gals:
        sel_g = ALL.get_gal_prop(g, sel_sample__gz)
        R = ALL.get_gal_prop(g, ALL.zoneDistance_HLR)
        x = np.ma.masked_array(np.ma.log10(ALL.get_gal_prop(g, ALL.W6563__z)), mask=~((R > 0.5) & (R <= 1) & sel_g))
        # x = np.ma.log10(ALL.get_gal_prop(g, ALL.W6563__z))
        N = x.count()
        Hg, xe = np.histogram(x.compressed(), bins=EW_bin_edges)
        Hgnorm = 1.* Hg/N
        H += Hgnorm
    print H
    ax.bar(EW_bin_center, H, align='center', color='darkgrey', width=np.diff(EW_bin_edges))
    for x, y in zip(EW_bin_center, H):
        print x, y
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.set_ylim(0, H.max())
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    logWHa_range = [-1, 2.5]
    ax = axes_cols[3][0]
    ax.set_title(r'$R\ >$ 1 HLR', fontsize=fs+4, y=1.02)
    EW_bin_edges = np.linspace(logWHa_range[0], logWHa_range[1], 51)
    EW_bin_center = (EW_bin_edges[:-1]+EW_bin_edges[1:])/2.
    H = np.zeros(EW_bin_center.shape, dtype='float')
    for g in gals:
        sel_g = ALL.get_gal_prop(g, sel_sample__gz)
        R = ALL.get_gal_prop(g, ALL.zoneDistance_HLR)
        x = np.ma.masked_array(np.ma.log10(ALL.get_gal_prop(g, ALL.W6563__z)), mask=~((R > 1) & sel_g))
        # x = np.ma.log10(ALL.get_gal_prop(g, ALL.W6563__z))
        N = x.count()
        Hg, xe = np.histogram(x.compressed(), bins=EW_bin_edges)
        Hgnorm = 1.* Hg/N
        H += Hgnorm
    print H
    ax.bar(EW_bin_center, H, align='center', color='darkgrey', width=np.diff(EW_bin_edges))
    for x, y in zip(EW_bin_center, H):
        print x, y
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.set_ylim(0, H.max())
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    for i in xrange(4):
        # for j in xrange(6):
        ax = axes_cols[i][1]  # [j+1]
        ax.set_axis_off()
        # ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')


    ########################
    ########################
    ########################

    cumulfHa__g_R = {}
    logWHa_bins = np.linspace(-1, 3, 50)
    print ALL.mt[ALL.califaID__g == 'K0017'], ALL.ba[ALL.califaID__g == 'K0017']
    for k, v in sel['gals__mt'].iteritems():
        print k, len(v), len(sample_gals), np.sum(v), np.sum(v & sample_gals)
        aux_gals = ALL.califaID__g[sample_gals & v]
        for g in aux_gals:
            # if not g in gals:
            #     continue
            aux_list_fHa = []
            print g
            for i, sel_R in enumerate(sels_R):
                sel_sample__z = ALL.get_gal_prop(g, sel_sample__gz)
                sel_R__z = ALL.get_gal_prop(g, sel_R)
                WHa = np.ma.masked_array(ALL.get_gal_prop(g, ALL.W6563__z), mask=~(sel_sample__z & sel_R__z))
                fHa = np.ma.masked_array(ALL.get_gal_prop(g, ALL.f6563__z), mask=~(sel_sample__z & sel_R__z))
                xm, ym = ma_mask_xyz(np.ma.log10(WHa), fHa)
                fHa_cumul, fHa_tot = cumul_bins(ym.compressed(), xm.compressed(), logWHa_bins)  # , fHa_tot)
                # print 'iR:%d <WHa>:%.2f' % (i, WHa.mean())
                aux_list_fHa.append(fHa_cumul)
            cumulfHa__g_R[g] = aux_list_fHa

    logWHa_range = [-1, 2.5]
    ax = axes_cols[0][2]
    # ax.set_title(r'All radii', fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    for g in gals:
        # print g
        x = np.log10(args.class_thresholds[0])
        y1 = cumulfHa__g_R[g][0][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][0][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = cumulfHa__g_R[g][0][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][0][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in gals])
    f_SFc = np.mean([f_SFc__g[g] for g in gals])
    m_aux = sel_sample__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('hDIG', (100. * f_hDIG)), 0.99, 0.98, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('mDIG', (100. * f_mDIG)), 0.99, 0.82, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$f_{%s}: %.1f$%%' % ('SFc', (100. * f_SFc)), 0.99, 0.66, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    if histo_stats:
        plot_histo_ax(ax, xm.compressed(), first=True, stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    else:
        plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=False, ini_pos_y=0.97, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[1][2]
    # ax.set_title(r'$R\ \leq$ 0.5 HLR', fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    for g in gals:
        x = np.log10(args.class_thresholds[0])
        y1 = cumulfHa__g_R[g][1][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][1][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = cumulfHa__g_R[g][1][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][1][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in gals])
    f_SFc = np.mean([f_SFc__g[g] for g in gals])
    fs = 6
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_Rin__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    if histo_stats:
        plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    else:
        plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=False, ini_pos_y=0.97, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[2][2]
    # ax.set_title(r'0.5 $<\ R\ \leq$ 1 HLR', fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    for g in gals:
        x = np.log10(args.class_thresholds[0])
        y1 = cumulfHa__g_R[g][2][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][2][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = cumulfHa__g_R[g][2][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][2][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in gals])
    f_SFc = np.mean([f_SFc__g[g] for g in gals])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_Rmid__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    if histo_stats:
        plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    else:
        plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=False, ini_pos_y=0.97, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')

    ax = axes_cols[3][2]
    # ax.set_title(r'$R\ >$ 1 HLR', fontsize=fs+4, y=1.02)
    f_SFc__g = {}
    f_mDIG__g = {}
    f_hDIG__g = {}
    for g in gals:
        x = np.log10(args.class_thresholds[0])
        y1 = cumulfHa__g_R[g][3][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][3][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_hDIG__g[g] = y_th
        x = np.log10(args.class_thresholds[1])
        y1 = cumulfHa__g_R[g][3][ALL.logWHa_bins__w < x][-1]
        y2 = cumulfHa__g_R[g][3][ALL.logWHa_bins__w > x][0]
        x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
        x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
        y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
        f_SFc__g[g] = 1. - y_th
        f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    f_hDIG = np.mean([f_hDIG__g[g] for g in gals])
    f_mDIG = np.mean([f_mDIG__g[g] for g in gals])
    f_SFc = np.mean([f_SFc__g[g] for g in gals])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    m_aux = sel_gals_sample_Rout__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    if histo_stats:
        plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c='k', kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    else:
        plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=False, ini_pos_y=0.97, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=logWHa_range, lw=3))
    for th in args.class_thresholds:
        ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MaxNLocator(20))
    ax.set_xlim(logWHa_range)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')

    for i in xrange(4):
        for j in xrange(3, 7):
            ax = axes_cols[i][j]
            ax.set_axis_off()


    #
    # for col in range(N_cols):
    #     for i, mt_label in enumerate(mtype_labels):
    #         ax = axes_cols[col][i+1]
    #         m_aux = sel_gals_mt[mt_label] & sels_R[col]
    #         f_SFc__g = {}
    #         f_mDIG__g = {}
    #         f_hDIG__g = {}
    #         _gals = ALL.califaID__g[sel['gals__mt'][mt_label] & sample_gals].tolist()
    #         for g in _gals:
    #             if not g in args.gals:
    #                 _gals.remove(g)
    #         if not col:
    #             if (mt_label == mtype_labels[-1]):
    #                 lab = r'$\geq$ Sc'
    #             else:
    #                 lab = mt_label
    #             ax.set_ylabel('%s (%d)' % (lab, len(_gals)), rotation=90, fontsize=fs+1)
    #         for g in _gals:
    #             print mt_label, g, ALL.ba[ALL.califaID__g == g], ALL.mt[ALL.califaID__g == g]
    #             x = np.log10(args.class_thresholds[0])
    #             y1 = cumulfHa__g_R[g][col][ALL.logWHa_bins__w < x][-1]
    #             y2 = cumulfHa__g_R[g][col][ALL.logWHa_bins__w > x][0]
    #             x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
    #             x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
    #             y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
    #             # print x, y_th, x1, x2, y1, y2
    #             f_hDIG__g[g] = y_th
    #             x = np.log10(args.class_thresholds[1])
    #             y1 = cumulfHa__g_R[g][col][ALL.logWHa_bins__w < x][-1]
    #             y2 = cumulfHa__g_R[g][col][ALL.logWHa_bins__w > x][0]
    #             x1 = ALL.logWHa_bins__w[ALL.logWHa_bins__w < x][-1]
    #             x2 = ALL.logWHa_bins__w[ALL.logWHa_bins__w > x][0]
    #             y_th = (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)
    #             f_SFc__g[g] = 1. - y_th
    #             # print x, y_th, x1, x2, y1, y2
    #             f_mDIG__g[g] = 1. - f_SFc__g[g] - f_hDIG__g[g]
    #             # print g, f_hDIG__g[g], f_mDIG__g[g], f_SFc__g[g]
    #         f_hDIG = np.mean([f_hDIG__g[g] for g in _gals])
    #         f_mDIG = np.mean([f_mDIG__g[g] for g in _gals])
    #         f_SFc = np.mean([f_SFc__g[g] for g in _gals])
    #         plot_text_ax(ax, r'$%.1f$%%' % (100. * f_hDIG), 0.99, 0.97, fs, 'top', 'right', c=args.dict_colors['hDIG']['c'])
    #         plot_text_ax(ax, r'$%.1f$%%' % (100. * f_mDIG), 0.99, 0.81, fs, 'top', 'right', c=args.dict_colors['mDIG']['c'])
    #         plot_text_ax(ax, r'$%.1f$%%' % (100. * f_SFc), 0.99, 0.65, fs, 'top', 'right', c=args.dict_colors['SFc']['c'])
    #         xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    #         # plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
    #         if histo_stats:
    #             plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.13, fs=fs, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=logWHa_range, lw=2))
    #         else:
    #             plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=logWHa_range, lw=2))
    #         ax.xaxis.set_major_locator(MaxNLocator(4))
    #         ax.xaxis.set_minor_locator(MaxNLocator(20))
    #         ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    #         # ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    #         for th in args.class_thresholds:
    #             ax.axvline(x=np.ma.log10(th), c='k', ls='--')
    #         if i == (len(colortipo) - 1):
    #             # ax.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
    #             ax.tick_params(axis='x', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')
    #             if col == (N_cols - 1):
    #                 ax.xaxis.set_major_locator(MaxNLocator(4))
    #                 ax.xaxis.set_minor_locator(MaxNLocator(20))
    #             else:
    #                 ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
    #                 ax.xaxis.set_minor_locator(MaxNLocator(20))
    #         else:
    #             ax.tick_params(axis='both', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    #             # if not col:
    #             #     ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
    #             # else:
    #             #     ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    #         ax.set_xlim(logWHa_range)

    # f.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
    gs.update(left=0.05, right=0.95, bottom=0.14, top=0.9, hspace=0.0, wspace=0.0)
    f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=fs+4)
    # f.set_tight_layout(True)
    f.savefig('fig_WHa_histograms_per_morftype_norm_pix%s.pdf' % fname_suffix, dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'
    return ALL


def fig_WHa_histograms_per_morftype_and_radius_cumulFHamed_refreport(args, gals, gals_sel=None, data_sel=None, fname_suffix=''):
    print '##########################################################'
    print '# fig_WHa_histograms_per_morftype_and_radius_cumulFHamed #'
    print '##########################################################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    if gals_sel is None:
        gals_sel = np.ones_like(gals, dtype='bool')
    sample_gals = sel['gals'] & gals_sel

    if data_sel is None:
        data_sel = np.ones_like(sel['gals_sample__z'], dtype='bool')
    sel_sample__gz = sel['gals_sample__z'] & data_sel
    sel_gals_mt = sel['gals__mt_z']

    print len(gals), len(sample_gals)
    gals = ALL.califaID__g[sample_gals]

    colortipo = ['orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['Sa+Sab', 'Sb', 'Sbc', '>= Sc']

    sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z'] & data_sel
    sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z'] & data_sel
    sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z'] & data_sel
    N_rows, N_cols = 4, 4
    f = plot_setup(width=latex_text_width, aspect=1./golden_mean)
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    N_zone = sel_sample__gz.astype('int').sum()

    gs = gridspec.GridSpec(N_rows, N_cols)
    axes_col0 = [
        plt.subplot(gs[0,0]), plt.subplot(gs[1,0]), plt.subplot(gs[2,0]),
        plt.subplot(gs[3,0]),
    ]
    axes_col1 = [
        plt.subplot(gs[0,1]), plt.subplot(gs[1,1]), plt.subplot(gs[2,1]),
        plt.subplot(gs[3,1]),
    ]
    axes_col2 = [
        plt.subplot(gs[0,2]), plt.subplot(gs[1,2]), plt.subplot(gs[2,2]),
        plt.subplot(gs[3,2]),
    ]
    axes_col3 = [
        plt.subplot(gs[0,3]), plt.subplot(gs[1,3]), plt.subplot(gs[2,3]),
        plt.subplot(gs[3,3]),
    ]

    axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]
    fs = 6

    sels_R = [sel_sample__gz, sel_gals_sample_Rin__gz, sel_gals_sample_Rmid__gz, sel_gals_sample_Rout__gz]
    sels_ba = ['ba_low', 'ba_mid', 'ba_high']
    ba_names = [r'$b/a\ \leq$ %.2f' % ALL.ba_bins[0], r'%.2f $<\ b/a\ \leq$ %.2f' % (ALL.ba_bins[0], ALL.ba_bins[1]), r'$b/a\ >$ %.2f' % ALL.ba_bins[1]]

    color_ba = ['r', 'g', 'b']
    logWHa_range = [-1, 2.5]
    axes_col0[0].set_title(r'All radii', fontsize=fs+4, y=1.02)
    axes_col1[0].set_title(r'$R\ \leq$ 0.5 HLR', fontsize=fs+4, y=1.02)
    axes_col2[0].set_title(r'0.5 $<\ R\ \leq$ 1 HLR', fontsize=fs+4, y=1.02)
    axes_col3[0].set_title(r'$R\ >$ 1 HLR', fontsize=fs+4, y=1.02)

    for col in range(N_cols):
        for i, mt_label in enumerate(mtype_labels):
            ax = axes_cols[col][i]
            _gals = ALL.califaID__g[sel['gals__mt'][mt_label] & sample_gals].tolist()
            for g in _gals:
                if not g in args.gals:
                    _gals.remove(g)
            if not col:
                if (mt_label == mtype_labels[-1]):
                    lab = r'$\geq$ Sc'
                else:
                    lab = mt_label
                ax.set_ylabel('%s (%d)' % (lab, len(_gals)), rotation=90, fontsize=fs+1)
            for j, sel_ba in enumerate(sels_ba):
                m_aux = sel_gals_mt[mt_label] & sels_R[col] & sel['gals__ba_z'][sel_ba]
                xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
                plot_histo_ax(ax, xm.compressed(), dataset_names=ba_names[j], stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99 - (j * 0.2), y_v_space=0.1, fs=fs, c=color_ba[j], kwargs_histo=dict(bins=50, histtype='step', color=color_ba[j], range=logWHa_range, lw=1))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_minor_locator(MaxNLocator(20))
            ax.tick_params(axis='both', which='both', direction='in', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            for th in args.class_thresholds:
                ax.axvline(x=np.ma.log10(th), c='k', ls='--')
            if i == (len(colortipo) - 1):
                ax.tick_params(axis='x', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='on', labeltop='off', labelleft='off', labelright='off')
                if col == (N_cols - 1):
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
                    ax.xaxis.set_minor_locator(MaxNLocator(20))
            else:
                ax.tick_params(axis='both', direction='in', which='both', bottom='on', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
            ax.set_xlim(logWHa_range)

    gs.update(left=0.05, right=0.95, bottom=0.14, top=0.9, hspace=0.0, wspace=0.0)
    f.text(0.5, 0.04, r'$\log$ W${}_{H\alpha}$ [$\AA$]', ha='center', fontsize=fs+4)
    f.savefig('fig_WHa_histo_mt_and_R_cumulFHa_refreport%s.pdf' % fname_suffix, dpi=_dpi_choice, transparent=_transp_choice)
    print '######################################################'
    print '# END ################################################'
    print '######################################################'
    return ALL



if __name__ == '__main__':
    import time
    t_init = time.time()
    args = parser_args()
    print 'parser_args() time: %.2f' % (time.time() - t_init)

    sample_choice = [args.SN_type, args.min_SN]

    t_init = time.time()
    ALL = stack_gals().load(args.file)
    print 'stack_gals() time: %.2f' % (time.time() - t_init)

    if isinstance(ALL.mt[0], str):
        ALL.mt = np.array([eval(mt) for mt in ALL.mt], dtype='int')
    args.ALL = ALL
    if args.gals is not None:
        if os.path.isfile(args.gals):
            with open(args.gals) as f:
                gals = [line.strip() for line in f.xreadlines() if line.strip()[0] != '#']
        else:
            gals = args.gals
        if isinstance(gals, str):
            gals = [gals]
    else:
        gals = ALL.califaID__g.tolist()
        args.gals = gals

    t_init = time.time()
    gals, sel, sample_choice = samples(args, ALL, sample_choice, gals)
    print 'samples() time: %.2f' % (time.time() - t_init)
    args.gals = gals
    args.sel = sel
    args.sample_choice = sample_choice

    t_init = time.time()
    create_fHa_cumul_per_WHa_bins(args)
    print 'create_fHa_cumul_per_WHa_bins() time: %.2f' % (time.time() - t_init)

    if args.summary: summary(args, ALL, sel, gals, 'SEL %s' % sample_choice)

    # plots
    # fig3(args, gals=['K0072', 'K0886', 'K0010', 'K0073', 'K0813'], multi=True, suffix='_faceon_paper', Hafluxsum=True)
    # fig_cumul_fHaWHa_per_morftype(args, args.gals)
    # fig_cumul_fHaWHa_per_morftype_and_R_gals(args, args.gals)
    # fig2(args)
    # fig_model_WHa_bimodal_distrib(args)
    # fig3(args, args.gals)
    # fig_BPT_per_morftype_and_R(args, args.gals)
    # fig_SBHaWHa_scatter_per_morftype_and_ba(args, args.gals)
    # fig_maps(args, args.gals)

    # fig3(args, gals=['K0936', 'K0077', 'K0822', 'K0791', 'K0811'], multi=True, suffix='_edgeon_paper', drawHLR=False)
    # fig_WHaSBHa_profile(args, gals=['K0886', 'K0073', 'K0813'], multi=True, suffix='faceon_paper')
    # fig_BPT_per_R(args, args.gals)
    # fig_cumul_fHaWHa_per_morftype_and_R(args, args.gals)
    # fig_BPT_per_R(args, args.gals)
    # fig_maps_Hafluxsum(args, gals=['K0072', 'K0886', 'K0010', 'K0073', 'K0813', 'K0353'], multi=True, suffix='_faceon_paper', size=(18, 22))
    # fig_maps_Hafluxsum(args, gals=['K0936', 'K0077', 'K0822', 'K0791', 'K0811'], multi=True, suffix='_edgeon_paper', drawHLR=False, size=(18, 18.3))
    # fig_maps(args, gals=['K0857', 'K0151', 'K0274', 'K0867'], multi=True, suffix='_edgeon_paper', drawHLR=False)
    # maps_xi(args, args.gals)
    # fig_BPT_mixed(args, args.gals)
    # fig_WHa_histograms_per_morftype_and_radius_cumulFHa(args, args.gals)
    # fig_maps_xi(args, gals=['K0072', 'K0886', 'K0010', 'K0073', 'K0813', 'K0353'], multi=True, suffix='_faceon_paper', xi=True)
    # fig_logxi_logWHa_histograms(args)
    # fig_maps_xi(args, gals=args.gals, xi=True)
    # fig_logSBHa_logWHa_histograms(args)
    # fig_WHaSBHa_profile(args, gals=['K0886', 'K0073', 'K0813'], multi=True, suffix='_faceon_paper')
    # fig_cumul_fHaWHa_per_morftype_group(args, args.gals)
    # fig_BPT_mixed_xi(args, args.gals)
    # fig_BPT_per_morftype_and_R(args, args.gals)
    # fig_BPT_per_R(args, args.gals)
    # fig_BPT_per_class(args, args.gals)
    # fig_SII_histogram(args, args.gals, minSNR=1)
    # fig_SII_histogram(args, args.gals, minSNR=3)
    # fig_SII_histogram(args, args.gals, minSNR=5)
    # fig_BPT_mixed_xi(args, args.gals)

    # for g in args.gals:
    #     fig_maps_Hafluxsum(args, gals=g, multi=False)

    print ALL.galDistance_Mpc[sel['gals']].max(), ALL.galDistance_Mpc[sel['gals']].min(), ALL.galDistance_Mpc[sel['gals']].mean(), np.median(ALL.galDistance_Mpc[sel['gals']])
    print spaxel_size_pc(ALL.galDistance_Mpc[sel['gals']].max()), spaxel_size_pc(ALL.galDistance_Mpc[sel['gals']].max(), 2.5)
    print spaxel_size_pc(ALL.galDistance_Mpc[sel['gals']].min()), spaxel_size_pc(ALL.galDistance_Mpc[sel['gals']].min(), 2.5)
    print spaxel_size_pc(ALL.galDistance_Mpc[sel['gals']].mean()), spaxel_size_pc(ALL.galDistance_Mpc[sel['gals']].mean(), 2.5)
    print spaxel_size_pc(np.median(ALL.galDistance_Mpc[sel['gals']])), spaxel_size_pc(np.median(ALL.galDistance_Mpc[sel['gals']]), 2.5)
    # fig_maps_Hafluxsum(args, gals=['K0072', 'K0886', 'K0010', 'K0073', 'K0813', 'K0353'], multi=True, suffix='_faceon_paper')  #, size=(18, 22))
    fig_WHa_histograms_per_morftype_and_radius_cumulFHamed(args, args.gals)
    fig_logxi_logWHa_histograms(args)
    fig_cumul_fHaWHa_per_morftype(args, args.gals)
    # fig_maps_Hafluxsum(args, gals=['K0936', 'K0077', 'K0822', 'K0791', 'K0811'], multi=True, suffix='_edgeon_paper', drawHLR=False, size=(latex_text_width, 1.))
    # fig_WHaSBHa_profile(args, gals=['K0886', 'K0073', 'K0813'], multi=True, suffix='_faceon_paper')
    fig_SII_histogram(args, args.gals, minSNR=3)
    fig_BPT_per_class(args, args.gals)
    fig_BPT_mixed(args, args.gals)
    # print_error_WHaSBHa(args, gals)
    # fig_WHaSBHa_profile(args, gals=['K0072', 'K0017'], multi=True, suffix='_faceon_paper')
    # fig_WHa_histograms_per_morftype_and_ba_cumulFHamed(args, args.gals)
    # fig_WHa_histograms_per_morftype_and_mlba_cumulFHamed(args, args.gals)
    # fig_WHa_histograms_per_morftype_and_radius_cumulFHamed(args, args.gals, gals_sel=sel['gals__ba']['ba_low'], data_sel=sel['gals__ba_z']['ba_low'], fname_suffix='_ba_low', histo_stats=True)
    # fig_WHa_histograms_per_morftype_and_radius_cumulFHamed(args, args.gals, gals_sel=sel['gals__ba']['ba_mid'], data_sel=sel['gals__ba_z']['ba_mid'], fname_suffix='_ba_mid', histo_stats=True)
    # fig_WHa_histograms_per_morftype_and_radius_cumulFHamed(args, args.gals, gals_sel=sel['gals__ba']['ba_high'], data_sel=sel['gals__ba_z']['ba_high'], fname_suffix='_ba_high', histo_stats=True)
    # fig_WHa_histograms_per_morftype_and_radius_cumulFHamed(args, args.gals, gals_sel=sel['gals__mlba']['ba_low'], data_sel=sel['gals__mlba_z']['ba_low'], fname_suffix='_mlba_low', histo_stats=True)
    # fig_WHa_histograms_per_morftype_and_radius_cumulFHamed(args, args.gals, gals_sel=sel['gals__mlba']['ba_mid'], data_sel=sel['gals__mlba_z']['ba_mid'], fname_suffix='_mlba_mid', histo_stats=True)
    # fig_WHa_histograms_per_morftype_and_radius_cumulFHamed(args, args.gals, gals_sel=sel['gals__mlba']['ba_high'], data_sel=sel['gals__mlba_z']['ba_high'], fname_suffix='_mlba_high', histo_stats=True)
    # fig_WHa_histograms_per_morftype_norm_pix(args, args.gals)
    # fig_WHa_histograms_per_morftype_and_radius_cumulFHamed_refreport(args, args.gals)
    # paper 1
    # min_popx = 0.05

    # sel_popx_32 = np.greater_equal(ALL.x_Y__Tz[0], min_popx)
    # sel_popx_32__yx = np.greater_equal(ALL.x_Y__Tyx[0], min_popx)
    # calc_radial_profiles(args, args.gals, data_sel__z=sel_popx_32, data_sel__yx=sel_popx_32__yx)
    # fig_SFRSD_32_histo(args, args.gals, data_sel=sel_popx_32)
    # fig_data_histograms_per_morftype_and_radius(args, args.gals,
    #                                             data=np.ma.log10(ALL.SFRSD__Tz[0] * 1e6),
    #                                             data_range=[-4, 0],
    #                                             data_label=r'$\log\ \Sigma_{SFR}^\star(t_\star=32Myr)$ [$M_\odot$ yr${}^{-1}$ kpc${}^{-2}$]',
    #                                             data_suffix='SFRSD_32_minxY0.05',
    #                                             byclass=True,
    #                                             bins=30,
    #                                             data_sel=sel_popx_32)

    # sel_popx_300 = np.greater_equal(ALL.x_Y__Tz[1], min_popx)
    # sel_popx_300__yx = np.greater_equal(ALL.x_Y__Tyx[0], min_popx)
    # calc_radial_profiles(args, args.gals, data_sel__z=sel_popx_300, data_sel__yx=sel_popx_300__yx)
    # fig_SFRSD_300_histo(args, args.gals, data_sel=sel_popx_300)
    # fig_data_histograms_per_morftype_and_radius(args, args.gals,
    #                                             data=np.ma.log10(ALL.SFRSD__Tz[1] * 1e6),
    #                                             data_range=[-5, -1],
    #                                             data_label=r'$\Sigma_{SFR}^\star(t_\star=300Myr)$ [$M_\odot$ yr${}^{-1}$ kpc${}^{-2}$]',
    #                                             data_suffix='SFRSD_300_minxY0.05',
    #                                             byclass=True,
    #                                             bins=30,
    #                                             data_sel=sel_popx_300)

    # calc_radial_profiles(args, args.gals)

    # fig_alogZ_mass_histo(args, args.gals)
    # fig_data_histograms_per_morftype_and_radius(args, args.gals,
    #                                             data=np.ma.masked_array(ALL.alogZ_mass__z, mask=np.isnan(ALL.alogZ_mass__z)),
    #                                             data_range=[-2.5, 0.5],
    #                                             data_label=r'$\langle \log\ Z_\star \rangle_M\ [Z_\odot]$',
    #                                             data_suffix='alogZ_mass',
    #                                             byclass=True,
    #                                             bins=30)
    #
    # fig_alogt_flux_histo(args, args.gals)
    # fig_data_histograms_per_morftype_and_radius(args, args.gals,
    #                                             data=np.ma.masked_array(ALL.at_flux__z, mask=np.isnan(ALL.at_flux__z)),
    #                                             data_range=[7, 10],
    #                                             data_label=r'$\langle \log\ t_\star \rangle_L\ [yr]$',
    #                                             data_suffix='at_flux',
    #                                             byclass=True,
    #                                             bins=30)
    #
    # fig_x_Y_32_histo(args, args.gals)
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.x_Y__Tz[0],
    #                                             data_range=[0, 1],
    #                                             data_label=r'$x_Y$ [frac.]',
    #                                             data_suffix='xY_32_bin0.05',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.05, 0.05))
    #
    # fig_x_Y_300_histo(args, args.gals)
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.x_Y__Tz[1],
    #                                             data_range=[0, 1],
    #                                             data_label=r'$x_Y$ [frac.]',
    #                                             data_suffix='xY_300_bin0.05',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.05, 0.05))
    #
    # fig_tau_V_histo(args, args.gals)
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.tau_V__z,
    #                                             data_range=[0, 1.1],
    #                                             data_label=r'$\tau_V^\star$',
    #                                             data_suffix='tauV_minxY320.05_bin0.05',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.1, 0.1))

    # fig_tauVNeb_histo(args, args.gals)
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.tau_V_neb__z,
    #                                             data_range=[-1, 5],
    #                                             data_label=r'$\tau_V^{neb}$',
    #                                             data_suffix='tauVneb',
    #                                             byclass=True,
    #                                             bins=np.linspace(-1, 5, 30))

    ################
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.x_Y__Tz[1],
    #                                             data_range=[0, 1],
    #                                             data_label=r'$x_Y$ [frac.]',
    #                                             data_suffix='xY_bin0.01',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.01, 0.01))
    #
    #
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.x_Y__Tz[1],
    #                                             data_range=[0, 1],
    #                                             data_label=r'$x_Y$ [frac.]',
    #                                             data_suffix='xY_bin0.1',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.1, 0.1))
    #
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.x_Y__Tz[1],
    #                                             data_range=[0, 1],
    #                                             data_label=r'$x_Y$ [frac.]',
    #                                             data_suffix='xY_bin0.15',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.15, 0.15))
    #
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.x_Y__Tz[1],
    #                                             data_range=[0, 1],
    #                                             data_label=r'$x_Y$ [frac.]',
    #                                             data_suffix='xY_bin0.2',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.2, 0.2))
    #
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.x_Y__Tz[1],
    #                                             data_range=[0, 1],
    #                                             data_label=r'$x_Y$ [frac.]',
    #                                             data_suffix='xY_bin0.25',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.25, 0.25))
    #
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.at_flux__z,
    #                                             data_range=[6, 10.5],
    #                                             data_label=r'$\langle \log\ t_\star \rangle_L$ [yr]',
    #                                             data_suffix='atflux',
    #                                             byclass=True,
    #                                             bins=30)  #np.logspace(np.log10(6), np.log10(10.3), 30))
    #
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.alogZ_mass__z,
    #                                             data_range=[-2.5, 0.5],
    #                                             data_label=r'$\langle \log\ Z_\star \rangle_L$ [$Z_\odot$]',
    #                                             data_suffix='alogZmass',
    #                                             byclass=True,
    #                                             bins=np.linspace(-2.5, 0.5, 30))
    #
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.tau_V__z,
    #                                             data_range=[0, 1.05],
    #                                             data_label=r'$\tau_V^\star$',
    #                                             data_suffix='tauV_bin0.05',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.05, 0.05))
    #
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.tau_V__z,
    #                                             data_range=[0, 1.1],
    #                                             data_label=r'$\tau_V^\star$',
    #                                             data_suffix='tauV_bin0.1',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.1, 0.1))
    #
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.tau_V__z,
    #                                             data_range=[0, 1.2],
    #                                             data_label=r'$\tau_V^\star$',
    #                                             data_suffix='tauV_bin0.2',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.2, 0.2))
    #
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.tau_V_neb__z,
    #                                             data_range=[-1, 5],
    #                                             data_label=r'$\tau_V^{neb}$',
    #                                             data_suffix='tauVneb',
    #                                             byclass=True,
    #                                             bins=np.linspace(-1, 5, 30),
    #                                             data_sel=sel['SN_HaHb3']['z'])
