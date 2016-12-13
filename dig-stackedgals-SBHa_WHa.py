import sys
import numpy as np
import matplotlib as mpl
from pytu.objects import runstats
from matplotlib import pyplot as plt
from CALIFAUtils.objects import stack_gals
from CALIFAUtils.scripts import loop_cubes
from CALIFAUtils.scripts import ma_mask_xyz
from pystarlight.util.constants import L_sun
from pytu.plots import plot_histo_ax


logSBHa_range = [3.5, 7]
logWHa_range = [0, 2.5]


mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
dflt_kw_scatter = dict(marker='o', s=5, edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, debug=True, gs_prc=True, poly1d=True)  # , tendency=True)


def cmap_discrete(colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)], n_bins=3, cmap_name='myMap'):
    cm = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cm


def plot_scatter_histo(x, y, xlim, ylim, xbins=30, ybins=30, xlabel='', ylabel='',
                       c=None, cmap=None, figure=None, axScatter=None, axHistx=None, axHisty=None,
                       scatter=True, histo=True):
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()  # no labels
    if axScatter is None:
        f = figure
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        axScatter = f.add_axes(rect_scatter)
        axHistx = f.add_axes(rect_histx)
        axHisty = f.add_axes(rect_histy)
        axHistx.xaxis.set_major_formatter(nullfmt)  # no labels
        axHisty.yaxis.set_major_formatter(nullfmt)  # no labels
    if scatter:
        if isinstance(x, list):
            for X, Y, C in zip(x, y, c):
                axScatter.scatter(X, Y, c=C, cmap=cmap, **dflt_kw_scatter)
    axScatter.set_xlim(xlim)
    axScatter.set_ylim(ylim)
    axScatter.set_xlabel(xlabel)
    axScatter.set_ylabel(ylabel)
    if histo:
        axHistx.hist(x, bins=xbins, range=xlim, color=c, histtype='barstacked')
        axHisty.hist(y, bins=ybins, range=ylim, orientation='horizontal', color=c, histtype='barstacked')
    plt.setp(axHisty.xaxis.get_majorticklabels(), rotation=90+180)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    return axScatter, axHistx, axHisty

if __name__ == '__main__':
    gals_file = sys.argv[1]

    f = open(gals_file, 'r')
    g = []
    for line in f.xreadlines():
        read_line = line.strip()
        if read_line[0] == '#':
            continue
        g.append(read_line)
    f.close()

    keys = ['tau_V_neb', 'tau_V', 'WHa', 'SBHa']
    ALL = stack_gals()
    for k in keys:
        ALL.new1d_masked(k)  # this way you can use ALL.key (example ALL.tau_V_neb)
    for i_gal, K in loop_cubes(g, EL=True, config=-2, elliptical=True):
        tau_V_neb__z = np.where((K.EL.tau_V_neb__z < 0).filled(0.0), 0, K.EL.tau_V_neb__z)
        ALL.append1d_masked(k='tau_V_neb', val=tau_V_neb__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        tau_V__z = K.tau_V__z
        ALL.append1d_masked(k='tau_V', val=tau_V__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        fHa_obs__z = K.EL.flux[K.EL.lines.index('6563')]
        SBHa__z = K.EL._F_to_L(fHa_obs__z)/(L_sun * K.zoneArea_pc2 * 1e-6)
        SBHa_mask__z = np.bitwise_or(~np.isfinite(fHa_obs__z), np.less(fHa_obs__z, 1e-40))
        ALL.append1d_masked(k='SBHa', val=SBHa__z, mask_val=SBHa_mask__z)
        WHa__z = K.EL.EW[K.EL.lines.index('6563')]
        ALL.append1d_masked(k='WHa', val=WHa__z, mask_val=SBHa_mask__z)
    # stack all lists
    ALL.stack()
    x = np.ma.log10(ALL.SBHa)
    y = np.ma.log10(ALL.WHa)

    DIG_WHa_threshold = 10
    HII_WHa_threshold = 20
    sel_WHa_DIG = (ALL.WHa < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP = np.bitwise_and((ALL.WHa >= DIG_WHa_threshold).filled(False), (ALL.WHa < HII_WHa_threshold).filled(False))
    sel_WHa_HII = (ALL.WHa >= HII_WHa_threshold).filled(False)
    f = plt.figure(figsize=(8, 8))
    x_ds = [x[sel_WHa_DIG], x[sel_WHa_COMP], x[sel_WHa_HII]]
    y_ds = [y[sel_WHa_DIG], y[sel_WHa_COMP], y[sel_WHa_HII]]
    axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logSBHa_range, logWHa_range, 30, 30,
                                         figure=f, c=['r', 'g', 'b'], scatter=False,
                                         xlabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]',
                                         ylabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logSBHa_range, logWHa_range, 30, 30,
                                         axScatter=axS, axHistx=axH1, axHisty=axH2, c=['r', 'g', 'b'], histo=False,
                                         xlabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]',
                                         ylabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_DIG)
    rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
    axS.plot(rs.xS, rs.yS, 'r', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
    xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_COMP)
    rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
    axS.plot(rs.xS, rs.yS, 'g', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
    xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_HII)
    rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
    axS.plot(rs.xS, rs.yS, 'b', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
    axS.grid()
    f.savefig('dig-stackedgals-SBHa_WHa-classifWHa.png')
    plt.close(f)

    HII_Zhang_threshold = 1e39/L_sun
    DIG_Zhang_threshold = 10**38.5/L_sun
    sel_Zhang_DIG = (ALL.SBHa < DIG_Zhang_threshold).filled(False)
    sel_Zhang_COMP = np.bitwise_and((ALL.SBHa >= DIG_Zhang_threshold).filled(False), (ALL.SBHa < HII_Zhang_threshold).filled(False))
    sel_Zhang_HII = (ALL.SBHa >= HII_Zhang_threshold).filled(False)
    f = plt.figure(figsize=(8, 8))
    x_ds = [x[sel_Zhang_DIG], x[sel_Zhang_COMP], x[sel_Zhang_HII]]
    y_ds = [y[sel_Zhang_DIG], y[sel_Zhang_COMP], y[sel_Zhang_HII]]
    axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logSBHa_range, logWHa_range, 30, 30,
                                         figure=f, c=['r', 'g', 'b'], scatter=False,
                                         xlabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]',
                                         ylabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logSBHa_range, logWHa_range, 30, 30,
                                         axScatter=axS, axHistx=axH1, axHisty=axH2, c=['r', 'g', 'b'], histo=False,
                                         xlabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]',
                                         ylabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_DIG)
    rs = runstats(ym.compressed(), xm.compressed(), **dflt_kw_runstats)
    axS.plot(rs.yS, rs.xS, 'r', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
    xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_COMP)
    rs = runstats(ym.compressed(), xm.compressed(), **dflt_kw_runstats)
    axS.plot(rs.yS, rs.xS, 'g', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
    xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_HII)
    rs = runstats(ym.compressed(), xm.compressed(), **dflt_kw_runstats)
    axS.plot(rs.yS, rs.xS, 'b', marker='*', markeredgewidth=1, markeredgecolor='k', markersize=10, lw=1)
    axS.grid()
    f.savefig('dig-stackedgals-SBHa_WHa-classifZhang.png')
    plt.close(f)
