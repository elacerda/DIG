#######################################################################
# "Ad hoc Ad loc and qui pro quo... so little time, so much to know!" #
#######################################################################
import os
import sys
import ast
import time
import datetime
import itertools
import numpy as np
import matplotlib as mpl
from pytu.lines import Lines
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
from CALIFAUtils.scripts import get_NEDName_by_CALIFAID, spaxel_size_pc
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator, ScalarFormatter
from pytu.plots import cmap_discrete, plot_text_ax, density_contour, plot_scatter_histo, plot_histo_ax, stats_med12sigma, add_subplot_axes


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


def parser_args(args_list, default_args_file='/Users/lacerda/dev/astro/dig/default.args'):
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
        'class_thresholds': [3, 12],
        'min_SN': 0,
        'SN_type': 'SN_HaHb',
        'summary': False,
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
    parser.add_argument('--SN_type', metavar='STR', type=str, default=default_args['SN_type'])
    parser.add_argument('--summary', '-S', action='store_true', default=default_args['summary'])
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
    sel_gals_mt = {
        'E': (mt == -2),
        'S0+S0a': (mt == -1),
        'Sa+Sab': np.bitwise_or(mt == 0, mt == 1),
        'Sb': mt == 2,
        'Sbc': mt == 3,
        '>= Sc': np.bitwise_or(mt == 4, np.bitwise_or(mt == 5, np.bitwise_or(mt == 6, mt == 7))),
        # 'Sc+Scd+Sd+Irr'
    }

    sel_mt = {
        'E': (ALL.mt == -2),
        'S0+S0a': (ALL.mt == -1),
        'Sa+Sab': np.bitwise_or(ALL.mt == 0, ALL.mt == 1),
        'Sb': ALL.mt == 2,
        'Sbc': ALL.mt == 3,
        '>= Sc': np.bitwise_or(ALL.mt == 4, np.bitwise_or(ALL.mt == 5, np.bitwise_or(ALL.mt == 6, ALL.mt == 7))),
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

    sel_radius_inside = (ALL.zoneDistance_HLR <= 0.5)
    sel_radius_middle = (ALL.zoneDistance_HLR > 0.5) & (ALL.zoneDistance_HLR <= 1)
    sel_radius_outside = (ALL.zoneDistance_HLR > 1)
    sel_gals_sample_Rin__gz = np.bitwise_and(sel_radius_inside, sel_gals_sample__gz)
    sel_gals_sample_Rmid__gz = np.bitwise_and(sel_radius_middle, sel_gals_sample__gz)
    sel_gals_sample_Rout__gz = np.bitwise_and(sel_radius_outside, sel_gals_sample__gz)
    sels_R = [sel_gals_sample__gz, sel_gals_sample_Rin__gz, sel_gals_sample_Rmid__gz, sel_gals_sample_Rout__gz]
    sel['radius_inside'] = sel_radius_inside
    sel['radius_middle'] = sel_radius_middle
    sel['radius_outside'] = sel_radius_outside
    sel['gals_sample_Rin__z'] = sel_gals_sample_Rin__gz
    sel['gals_sample_Rmid__z'] = sel_gals_sample_Rmid__gz
    sel['gals_sample_Rout__z'] = sel_gals_sample_Rout__gz

    sel['gals__mt'] = sel_mt
    sel['gals__mt_z'] = sel_gals_mt
    sel['gals'] = tmp_sel_gals
    sel['gals__z'] = sel_gals__gz
    sel['gals__yx'] = sel_gals__gyx
    sel['gals_sample__z'] = sel_gals_sample__gz
    sel['gals_sample__yx'] = sel_gals_sample__gyx

    print 'gals_sample_choice() time: %.2f' % (time.time() - t_init)
    return new_gals, sel, sample_choice


def create_fHa_cumul_per_WHa_bins(args):
    ALL, sel = args.ALL, args.sel
    sel_gals = sel['gals']
    gals = args.gals
    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    mt = np.hstack(list(itertools.chain(list(itertools.repeat(mt, ALL.N_zone[i])) for i, mt in enumerate(ALL.mt))))
    sels_R = [sel_sample__gz, sel['gals_sample_Rin__z'], sel['gals_sample_Rmid__z'], sel['gals_sample_Rout__z']]
    cumulfHa__g_R = {}
    cumulLHa__g_R = {}
    logWHa_bins = np.linspace(-1, 3, 50)
    for k, v in sel['gals__mt'].iteritems():
        for g in ALL.califaID__g[v]:
            if not g in gals:
                continue
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


def fig2(args):
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
        aux_ax = axH2.twiny()
        plot_histo_ax(aux_ax, ym.compressed(), stats_txt=False, kwargs_histo=dict(histtype='step', orientation='horizontal', normed=False, range=logSBHa_range, color='k', lw=2, ls='-'))
        aux_ax.xaxis.set_major_locator(MaxNLocator(3))
        plt.setp(aux_ax.xaxis.get_majorticklabels(), rotation=270)
        plot_text_classes_ax(axH1, args, x=0.98, y_ini=0.98, y_spac=0.11, fs=14)
        plot_text_ax(axH2, r'all zones', 0.98, 0.98, 14, 'top', 'right', 'k')
        scater_kwargs = dict(c=zm, s=3, vmax=3, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
        sc = axS.scatter(xm, ym, **scater_kwargs)
        cbaxes = f.add_axes([0.6, 0.17, 0.12, 0.02])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[0, 1, 2, 3], orientation='horizontal')
        cb.set_label(r'R [HLR]', fontsize=14)
        axS.axhline(y=np.log10(SF_Zhang_threshold), color='k', linestyle='-.', lw=2)
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
        f.savefig('fig2.png', dpi=_dpi_choice, transparent=_transp_choice)
        plt.close(f)


def fig_maps(args, gals=None, multi=False, suffix='', drawHLR=True, tau_V_histo=False):
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
    if tau_V_histo:
        suffix += '_tau_V_histo'
        N_cols = 5
    if multi:
        N_rows = len(gals)
        # f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 4.6, N_rows * 4.3))
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
            if tau_V_histo:
                (ax1, ax2, ax3, ax4, ax5) = axArr[row]
            else:
                (ax1, ax2, ax3, ax4) = axArr[row]
        else:
            N_rows = 1
            f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(18, 4))
            if tau_V_histo:
                ax1, ax2, ax3, ax4, ax5 = axArr
            else:
                ax1, ax2, ax3, ax4 = axArr
            suptitle = '%s - %s ba:%.2f (ml_ba:%.2f) (%s): %d pixels (%d zones -' % (califaID, mto, ba, ml_ba, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone)
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
        # cb.set_label(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        # AXIS 3
        x = np.ma.log10(W6563__yx)
        im = ax3.imshow(x, vmin=logWHa_range[0], vmax=logWHa_range[1], cmap=plt.cm.copper_r, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax3)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        # cb.set_label(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        # AXIS 4
        x = W6563__yx
        im = ax4.imshow(x, vmin=3, vmax=args.class_thresholds[-1], cmap='Spectral', **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax4)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        # cb.set_label(r'W${}_{H\alpha}$ [$\AA$]')
        if tau_V_histo:
            xDs = []
            cmap_class_c = []
            for k in args.class_names:
                sel_aux = ALL.get_gal_prop(califaID, (sel_sample__gz & sel['WHa']['z'][k]))
                Hb, Ha = ma_mask_xyz(f4861__z, f6563__z, mask=~sel_aux)
                tau_tmp = f_tauVneb(Ha, Hb).compressed()
                # print k, tau_tmp
                xDs.append(tau_tmp)
                cmap_class_c.append(args.dict_colors[k]['c'])
            _, text_list = plot_histo_ax(ax5, xDs, stats_txt=False, return_text_list=True, y_v_space=0.06, y_h_space=0.25, first=False, c=cmap_class_c, kwargs_histo=dict(histtype='step', color=cmap_class_c, normed=False, range=logWHa_range, lw=3))
            pos_y = 0.9
            for txt in text_list[0]:
                plot_text_ax(ax5, txt, **dict(pos_x=0.98, pos_y=pos_y, fs=14, va='top', ha='right', c=cmap_class_c[0]))
                pos_y -= 0.06
            pos_y = 0.9
            for txt in text_list[1]:
                plot_text_ax(ax5, txt, **dict(pos_x=0.80, pos_y=pos_y, fs=14, va='top', ha='right', c=cmap_class_c[1]))
                pos_y -= 0.06
            pos_y = 0.9
            for txt in text_list[2]:
                plot_text_ax(ax5, txt, **dict(pos_x=0.4, pos_y=pos_y, fs=14, va='top', ha='right', c=cmap_class_c[2]))
                pos_y -= 0.06
            ax5.set_xlabel(r'$\tau_V^{neb}$')
            ax5.xaxis.set_minor_locator(minorLocator)
            ax5_top = ax5.twiny()
            mn, mx = ax5.get_xlim()
            unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
            ax5_top.set_xlim(unit_converter(mn), unit_converter(mx))
            #ax1_top.set_xlim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))
            ax5_top.set_xlabel(r'$\log\ H\alpha/H\beta$')
            ax5_top.xaxis.set_major_locator(MaxNLocator(4))
            # ax5_top.xaxis.set_minor_locator(minorLocator)
        if drawHLR:
            DrawHLRCircleInSDSSImage(ax1, HLR_pix, pa, ba, lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
            for ax in [ax2, ax3, ax4]:
                DrawHLRCircle(ax, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        if row <= 0:
            ax1.set_title('SDSS stamp', fontsize=18)
            # ax2.set_title(r'$\log$ W${}_{H\alpha}$ map')
            # ax3.set_title(r'$\log\ \Sigma_{H\alpha}$ map')
            ax2.set_title(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]', fontsize=18)
            ax3.set_title(r'$\log$ W${}_{H\alpha}$ [$\AA$]', fontsize=18)
            ax4.set_title(r'W${}_{H\alpha}$ [$\AA$]', fontsize=18)
            # ax4.set_title(r'classification map', fontsize=18)
        if multi:
            row += 1
        else:
            f.tight_layout(rect=[0, 0.01, 1, 0.98])
            # f.subplots_adjust(left=0.05, right=0.96, bottom=0.05, top=0.95, hspace=0.08, wspace=0.3)
            f.savefig('fig_maps-%s.png' % califaID, dpi=_dpi_choice, transparent=_transp_choice)
            plt.close(f)
    if multi:
        f.subplots_adjust(left=0.05, right=0.96, bottom=0.05, top=0.95, hspace=0.08, wspace=0.3)
        f.savefig('fig_maps_class%s.png' % suffix, dpi=_dpi_choice, transparent=_transp_choice)
        plt.close(f)


def calc_tau_V_neb(args, gals):  #calc_R_stuff
    ALL, sel = args.ALL, args.sel

    sel_gals__mt = sel['gals__mt']
    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
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
        x_Y__cTgr = {k:np.ma.masked_all((N_T, N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        SFRSD__cTgr = {k:np.ma.masked_all((N_T, N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        x_Y_npts__cTgr = {k:np.ma.masked_all((N_T, N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
        SFRSD_npts__cTgr = {k:np.ma.masked_all((N_T, N_gals, args.N_R_bins), dtype='float') for k in args.class_names}

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
                # for iT, tSF in enumerate(ALL.tSF__T):
                #     x_Y__cTgr = {k:np.ma.masked_all((N_T, N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
                #     SFRSD__cTgr = {k:np.ma.masked_all((N_T, N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
                #     x_Y_npts__cTgr = {k:np.ma.masked_all((N_T, N_gals, args.N_R_bins), dtype='float') for k in args.class_names}
                #     SFRSD_npts__cTgr = {k:np.ma.masked_all((N_T, N_gals, args.N_R_bins), dtype='float') for k in args.class_names}

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


def fig_tauVNeb_histo(args, gals):
    print '#####################'
    print '# fig_tauVNeb_histo #'
    print '#####################'

    ALL, sel = args.ALL, args.sel

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']
    sel_gals_mt = sel['gals__mt_z']
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
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    ax1, ax2, ax3 = axArr

    # AXIS 1
    x = f_tauVneb(Ha, Hb)
    # x = (1./(_q[0] - _q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
    range = [-2, 2]
    xDs = []
    for k in args.class_names:
        sel_aux = np.bitwise_and(sel_WHa['z'][k], sel_sample__gz)
        xDs.append(x[sel_aux].compressed())
    # plot_histo_ax(ax1, x.compressed(), stats_txt=False, histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, y_v_space=0.06, y_h_space=0.25, first=False, c=args.class_colors, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=3))
    x_ini = 0.98
    for j, k in enumerate(args.class_names):
        pos_y = 0.98
        for txt in text_list[j]:
            print k, txt
            plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=14, va='top', ha='right', c=args.class_colors[j]))
            pos_y -= 0.06
        x_ini -= 0.25
    ax1.set_xlabel(r'$\tau_V^{neb}$')
    ax1.xaxis.set_minor_locator(minorLocator)
    ax1_top = ax1.twiny()
    mn, mx = ax1.get_xlim()
    unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
    ax1_top.set_xlim(unit_converter(mn), unit_converter(mx))
    # ax1_top.set_xlim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))
    ax1_top.set_xlabel(r'$\log\ H\alpha/H\beta$')
    plot_text_ax(ax1, 'a)', 0.02, 0.98, 16, 'top', 'left', 'k')

    # AXIS 2
    x = delta_tau_SFc_mDIG
    range = DtauVnorm_range
    plot_histo_ax(ax2, x.compressed(), y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax2.set_xlabel(r'$\Delta \tau\ =\ \tau_V^{%s}\ -\ \tau_V^{%s}$' % (args.class_names[-1], args.class_names[1]))
    ax2.xaxis.set_minor_locator(minorLocator)
    plot_text_ax(ax2, 'b)', 0.02, 0.98, 16, 'top', 'left', 'k')

    # AXIS 3
    x = delta_tau_SFc_hDIG
    range = DtauVnorm_range
    plot_histo_ax(ax3, x.compressed(), y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax3.set_xlabel(r'$\Delta \tau\ =\ \tau_V^{%s}\ -\ \tau_V^{%s}}$' % (args.class_names[-1], args.class_names[0]))
    ax3.xaxis.set_minor_locator(minorLocator)
    plot_text_ax(ax3, 'b)', 0.02, 0.98, 16, 'top', 'left', 'k')

    f.tight_layout(h_pad=0.05)
    f.savefig('fig_tauVNeb_histograms.png', dpi=_dpi_choice, transparent=_transp_choice)


def fig_SFRSD_histograms(args, gals):
    print '########################'
    print '# fig_SFRSD_histograms #'
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
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    ax1, ax2, ax3 = axArr

    # AXIS 1
    x = f_tauVneb(Ha, Hb)
    # x = (1./(_q[0] - _q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
    range = [-2, 2]
    xDs = []
    for k in args.class_names:
        sel_aux = np.bitwise_and(sel_WHa['z'][k], sel_sample__gz)
        xDs.append(x[sel_aux].compressed())
    # plot_histo_ax(ax1, x.compressed(), stats_txt=False, histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, y_v_space=0.06, y_h_space=0.25, first=False, c=args.class_colors, kwargs_histo=dict(histtype='step', color=args.class_colors, normed=False, range=range, lw=3))
    x_ini = 0.98
    for j, k in enumerate(args.class_names):
        pos_y = 0.98
        for txt in text_list[j]:
            print k, txt
            plot_text_ax(ax1, txt, **dict(pos_x=x_ini, pos_y=pos_y, fs=14, va='top', ha='right', c=args.class_colors[j]))
            pos_y -= 0.06
        x_ini -= 0.25
    ax1.set_xlabel(r'$\tau_V^{neb}$')
    ax1.xaxis.set_minor_locator(minorLocator)
    ax1_top = ax1.twiny()
    mn, mx = ax1.get_xlim()
    unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
    ax1_top.set_xlim(unit_converter(mn), unit_converter(mx))
    # ax1_top.set_xlim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))
    ax1_top.set_xlabel(r'$\log\ H\alpha/H\beta$')
    plot_text_ax(ax1, 'a)', 0.02, 0.98, 16, 'top', 'left', 'k')

    # AXIS 2
    x = delta_tau_SFc_mDIG
    range = DtauVnorm_range
    plot_histo_ax(ax2, x.compressed(), y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax2.set_xlabel(r'$\Delta \tau\ =\ \tau_V^{%s}\ -\ \tau_V^{%s}$' % (args.class_names[-1], args.class_names[1]))
    ax2.xaxis.set_minor_locator(minorLocator)
    plot_text_ax(ax2, 'b)', 0.02, 0.98, 16, 'top', 'left', 'k')

    # AXIS 3
    x = delta_tau_SFc_hDIG
    range = DtauVnorm_range
    plot_histo_ax(ax3, x.compressed(), y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
    ax3.set_xlabel(r'$\Delta \tau\ =\ \tau_V^{%s}\ -\ \tau_V^{%s}$' % (args.class_names[-1], args.class_names[0]))
    ax3.xaxis.set_minor_locator(minorLocator)
    plot_text_ax(ax3, 'b)', 0.02, 0.98, 16, 'top', 'left', 'k')

    f.tight_layout(h_pad=0.05)
    f.savefig('fig_tauVNeb_histograms.png', dpi=_dpi_choice, transparent=_transp_choice)


# def fig_data_histograms_per_morftype_and_radius(args, gals, data, data_range, data_label, data_suffix):
#     print '###############################################'
#     print '# fig_data_histograms_per_morftype_and_radius #'
#     print '###############################################'
#
#     ALL, sel = args.ALL, args.sel
#
#     if gals is None:
#         _, ind = np.unique(ALL.califaID__z, return_index=True)
#         gals = ALL.califaID__z[sorted(ind)]
#
#     sel_sample__gz = sel['gals_sample__z']
#     sel_sample__gyx = sel['gals_sample__yx']
#     sel_gals_mt = sel['gals__mt_z']
#
#     colortipo = ['orange', 'green', '#00D0C9', 'blue']
#     colortipo_darker = ['orangered', 'darkgreen', '#11C0B3', 'darkblue']
#     colortipo_lighter = ['navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
#     mtype_labels = ['Sa+Sab', 'Sb', 'Sbc', '>= Sc']
#     sel_gals_sample_Rin__gz = sel['gals_sample_Rin__z']
#     sel_gals_sample_Rmid__gz = sel['gals_sample_Rmid__z']
#     sel_gals_sample_Rout__gz = sel['gals_sample_Rout__z']
#     N_rows, N_cols = 5, 4
#     f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 10))
#     _, ind = np.unique(ALL.califaID__z, return_index=True)
#     N_zone = sel_sample__gz.astype('int').sum()
#     (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43) = axArr
#     axes_col0 = [ax00, ax10, ax20, ax30, ax40]
#     axes_col1 = [ax01, ax11, ax21, ax31, ax41]
#     axes_col2 = [ax02, ax12, ax22, ax32, ax42]
#     axes_col3 = [ax03, ax13, ax23, ax33, ax43]
#
#     axes_cols = [axes_col0, axes_col1, axes_col2, axes_col3]
#
#     # logWHa = np.ma.log10(ALL.W6563__z)
#     # logWHa_range = [-1, 2.5]
#     # xlabel = r'$\log$ W${}_{H\alpha}$ [$\AA$]'
#
#     ax = axes_cols[0][0]
#     ax.set_title(r'All radii', fontsize=18, y=1.02)
#
#     m_aux = sel_sample__gz
#     xm = np.ma.masked_array(data, mask=~m_aux)
#     plot_histo_ax(ax, xm.compressed(), c='k', stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=data_range, lw=3))
#     ax.set_xlim(data_range)
#     ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#
#     ax = axes_cols[1][0]
#     ax.set_title(r'R $\leq$ 0.5 HLR', fontsize=18, y=1.02)
#     m_aux = sel_gals_sample_Rin__gz
#     xm = np.ma.masked_array(data, mask=~m_aux)
#     plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=data_range, lw=3))
#     ax.set_xlim(data_range)
#     ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#
#     ax = axes_cols[2][0]
#     ax.set_title(r'0.5 $<$ R $\leq$ 1 HLR', fontsize=18, y=1.02)
#     m_aux = sel_gals_sample_Rmid__gz
#     xm = np.ma.masked_array(data, mask=~m_aux)
#     plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=data_range, lw=3))
#     ax.set_xlim(data_range)
#     ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#
#     ax = axes_cols[3][0]
#     ax.set_title(r'R $>$ 1 HLR', fontsize=18, y=1.02)
#     m_aux = sel_gals_sample_Rout__gz
#     xm = np.ma.masked_array(data, mask=~m_aux)
#     plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=data_range, lw=3))
#     ax.set_xlim(data_range)
#     ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#
#     sels_R = [sel_sample__gz, sel_gals_sample_Rin__gz, sel_gals_sample_Rmid__gz, sel_gals_sample_Rout__gz]
#
#     for col in range(N_cols):
#         for i, mt_label in enumerate(mtype_labels):
#             ax = axes_cols[col][i+1]
#             if not col:
#                 if (mt_label == mtype_labels[-1]):
#                     lab = r'$\geq$ Sc'
#                 else:
#                     lab = mt_label
#                 ax.set_ylabel(lab, rotation=90)
#             m_aux = np.bitwise_and(sel_gals_mt[mt_label], sels_R[col])
#             xm = np.ma.masked_array(data, mask=~m_aux)
#             # plot_text_ax(ax, r'$%.3f$' % fracHa__c[k], 0.99, y_pos, 15, 'top', 'right', c=args.dict_colors[k]['c'])
#             plot_histo_ax(ax, xm.compressed(), stats_txt=False, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=data_range, lw=2))
#             ax.xaxis.set_major_locator(MaxNLocator(4))
#             ax.xaxis.set_minor_locator(MaxNLocator(20))
#             ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#             if i == (len(colortipo) - 1):
#                 ax.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
#                 if col == (N_cols - 1):
#                     ax.xaxis.set_major_locator(MaxNLocator(4))
#                     ax.xaxis.set_minor_locator(MaxNLocator(20))
#                 else:
#                     ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
#                     ax.xaxis.set_minor_locator(MaxNLocator(20))
#             else:
#                 if not col:
#                     ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
#                 else:
#                     ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#             ax.set_xlim(data_range)
#
#     f.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.0)
#     f.text(0.5, 0.04, data_label, ha='center', fontsize=20)
#     f.savefig('fig_histograms_per_morftype_and_radius_%s.png' % data_suffix, dpi=_dpi_choice, transparent=_transp_choice)
#     print '######################################################'
#     print '# END ################################################'
#     print '######################################################'
#     return ALL


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
            plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.01, ha='left', ini_pos_y=0.99, y_v_space=0.1, fs=fs, c='k', kwargs_histo=dict(bins=bins, histtype='stepfilled', color=colortipo[i], range=data_range, lw=1))
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


# def fig_data_radialprofiles(args, gals, data, data_range, data_label, data_suffix, byclass=False, bins=None, data_sel=None):
#     print '###########################'
#     print '# fig_data_radialprofiles #'
#     print '###########################'
#
#     if bins is None:
#         bins = args.R_bin_center__r
#
#     ALL, sel = args.ALL, args.sel
#
#     if gals is None:
#         _, ind = np.unique(ALL.califaID__z, return_index=True)
#         gals = ALL.califaID__z[sorted(ind)]
#
#     colortipo = ['orange', 'green', '#00D0C9', 'blue']
#     colortipo_darker = ['orangered', 'darkgreen', '#11C0B3', 'darkblue']
#     colortipo_lighter = ['navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
#     mtype_labels = ['Sa+Sab', 'Sb', 'Sbc', '>= Sc']
#
#     f = plot_setup(width=latex_text_width, aspect=1./golden_mean)
#
#     print '######################################################'
#     print '# END ################################################'
#     print '######################################################'
#     return ALL


if __name__ == '__main__':
    args = parser_args(sys.argv[1:])

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

    t_init = time.time()
    gals, sel, sample_choice = samples(args, ALL, sample_choice, gals)
    print 'samples() time: %.2f' % (time.time() - t_init)
    args.gals = gals
    args.sel = sel
    args.sample_choice = sample_choice

    t_init = time.time()
    create_fHa_cumul_per_WHa_bins(args)
    print 'create_fHa_cumul_per_WHa_bins() time: %.2f' % (time.time() - t_init)

    # t_init = time.time()
    # calc_tau_V_neb(args, args.gals)
    # print 'calc_tau_V_neb() time: %.2f' % (time.time() - t_init)

    # sys.exit(1)
    if args.summary: summary(args, ALL, sel, gals, 'SEL %s' % sample_choice)

    # fig_maps(args, args.gals, tau_V_histo=True)
    # fig_tauVNeb_histo(args, args.gals)
    # fig_logt_histograms_per_morftype_and_radius(args, args.gals)
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.at_flux__z,
    #                                             data_range=[7, 11],
    #                                             data_label=r'$\langle\log\ t\rangle_L$ [yr]',
    #                                             data_suffix='logt')
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=np.ma.log10(ALL.SFRSD__Tz[0]) + 6,
    #                                             data_range=[-4, 1],
    #                                             data_label=r'$\log\ \Sigma_{SFR}^\star(t_\star)\ [M_\odot yr^{-1} kpc^{-2}]$',
    #                                             data_suffix='SFRSD')
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=np.ma.log10(ALL.W6563__z),
    #                                             data_range=logWHa_range,
    #                                             data_label=r'$\log$ W${}_{H\alpha}$ [$\AA$]',
    #                                             data_suffix='logWHa',
    #                                             byclass=False)
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=np.ma.log10(ALL.SFRSD__Tz[0]) + 6,
    #                                             data_range=[-4, 1],
    #                                             data_label=r'$\log\ \Sigma_{SFR}^\star(t_\star)\ [M_\odot yr^{-1} kpc^{-2}]$',
    #                                             data_suffix='SFRSD',
    #                                             byclass=True)
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.at_flux__z,
    #                                             data_range=[7.5, 10.5],
    #                                             data_label=r'$\langle\log\ t\rangle_L$ [yr]',
    #                                             data_suffix='logt',
    #                                             byclass=True)
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.tau_V_neb__z,
    #                                             data_range=[-2, 4],
    #                                             data_label=r'$\tau_V^{neb}$',
    #                                             data_suffix='tauVneb',
    #                                             byclass=True)


    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.x_Y__Tz[1],
    #                                             data_range=[0, 1],
    #                                             data_label=r'$x_Y$ [frac.]',
    #                                             data_suffix='xY_bin0.01',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.01, 0.01))
    #
    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.x_Y__Tz[1],
    #                                             data_range=[0, 1],
    #                                             data_label=r'$x_Y$ [frac.]',
    #                                             data_suffix='xY_bin0.05',
    #                                             byclass=True,
    #                                             bins=np.arange(0., 1.05, 0.05))
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

    # fig_data_histograms_per_morftype_and_radius(args, gals,
    #                                             data=ALL.tau_V_neb__z,
    #                                             data_range=[-1, 5],
    #                                             data_label=r'$\tau_V^{neb}$',
    #                                             data_suffix='tauVneb',
    #                                             byclass=True,
    #                                             bins=np.linspace(-1, 5, 30),
    #                                             data_sel=sel['SN_HaHb3']['z'])
