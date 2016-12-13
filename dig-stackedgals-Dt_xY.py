import os
import sys
import numpy as np
import matplotlib as mpl
from pytu.objects import runstats
from matplotlib import pyplot as plt
from CALIFAUtils.scripts import calc_xY
from CALIFAUtils.objects import stack_gals
from CALIFAUtils.scripts import loop_cubes
from CALIFAUtils.objects import CALIFAPaths
from CALIFAUtils.scripts import ma_mask_xyz
from pystarlight.util.constants import L_sun
from mpl_toolkits.axes_grid1 import make_axes_locatable


logSBHa_range = [3.5, 7]
logWHa_range = [0, 2.5]
x_Y_range = [0, 0.6]
DtauV_range = [-1, 3]
# age to calc xY
tY = 32e6
config = -2
EL = True
elliptical = True


mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
dflt_kw_scatter = dict(marker='o', s=1, edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, debug=True, gs_prc=True, poly1d=True)  # , tendency=True)


def cmap_discrete(colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)], n_bins=3, cmap_name='myMap'):
    cm = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cm


# Trying to correctly load q055 cubes inside q054 directory
def try_q055_instead_q054(califaID, **kwargs):
    from pycasso import fitsQ3DataCube
    config = kwargs.get('config', -1)
    EL = kwargs.get('EL', False)
    elliptical = kwargs.get('elliptical', False)
    K = None
    P = CALIFAPaths()
    P.set_config(config)
    pycasso_file = P.get_pycasso_file(califaID)
    if not os.path.isfile(pycasso_file):
        P.pycasso_suffix = P.pycasso_suffix.replace('q054', 'q055')
        pycasso_file = P.get_pycasso_file(califaID)
        print pycasso_file
        if os.path.isfile(pycasso_file):
            K = fitsQ3DataCube(P.get_pycasso_file(califaID))
            if elliptical:
                K.setGeometry(*K.getEllipseParams())
            if EL:
                emlines_file = P.get_emlines_file(califaID)
                if not os.path.isfile(emlines_file):
                    P.emlines_suffix = P.emlines_suffix.replace('q054', 'q055')
                    emlines_file = P.get_emlines_file(califaID)
                    print emlines_file
                    if os.path.isfile(emlines_file):
                        K.loadEmLinesDataCube(P.get_pycasso_file(califaID))
    return K


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

    keys = ['tau_V_neb', 'tau_V', 'WHa', 'SBHa', 'x_Y', 'zones_map', 'califaID']
    ALL = stack_gals()
    for k in keys:
        ALL.new1d_masked(k)  # this way you can use ALL.key (example ALL.tau_V_neb)

    minSNR = 3

    for i_gal, K in loop_cubes(g, EL=EL, config=config, elliptical=elliptical):
        if K is None:
            print 'califaID:', g[i_gal], ' trying another qVersion...'
            K = try_q055_instead_q054(g[i_gal], EL=EL, config=config, elliptical=elliptical)
            if K is None or K.EL is None:
                print 'califaID:', g[i_gal], ' missing fits files...'
                continue
        maskSNR = K.EL._setMaskLineSNR('4861', minsnr=minSNR)
        maskSNR = np.bitwise_or(maskSNR, K.EL._setMaskLineSNR('6563', minsnr=minSNR))
        califaID__z = np.array([K.califaID for i in range(K.N_zone)], dtype='|S5')
        ALL.append1d_masked(k='califaID', val=califaID__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        zones_map__z = np.array(list(range(K.N_zone)), dtype='int')
        ALL.append1d_masked(k='zones_map', val=zones_map__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        tau_V_neb__z = np.where((K.EL.tau_V_neb__z < 0).filled(0.0), 0, K.EL.tau_V_neb__z)
        ALL.append1d_masked(k='tau_V_neb', val=tau_V_neb__z, mask_val=maskSNR)
        tau_V__z = K.tau_V__z
        ALL.append1d_masked(k='tau_V', val=tau_V__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        fHa_obs__z = K.EL.flux[K.EL.lines.index('6563')]
        SBHa__z = K.EL._F_to_L(fHa_obs__z)/(L_sun * K.zoneArea_pc2 * 1e-6)
        SBHa_mask__z = np.bitwise_or(~np.isfinite(fHa_obs__z), np.less(fHa_obs__z, 1e-40))
        SBHa_mask__z = np.bitwise_or(SBHa_mask__z, maskSNR)
        ALL.append1d_masked(k='SBHa', val=SBHa__z, mask_val=SBHa_mask__z)
        WHa__z = K.EL.EW[K.EL.lines.index('6563')]
        ALL.append1d_masked(k='WHa', val=WHa__z, mask_val=SBHa_mask__z)
        x_Y__z, _ = calc_xY(K, tY)
        ALL.append1d_masked(k='x_Y', val=x_Y__z, mask_val=np.zeros((K.N_zone), dtype='bool'))

    # stack all lists
    ALL.stack()

    # WHa DIG-COMP-HII decomposition
    DIG_WHa_threshold = 10
    HII_WHa_threshold = 20
    sel_WHa_DIG = (ALL.WHa < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP = np.bitwise_and((ALL.WHa >= DIG_WHa_threshold).filled(False), (ALL.WHa < HII_WHa_threshold).filled(False))
    sel_WHa_HII = (ALL.WHa >= HII_WHa_threshold).filled(False)

    # SBHa-Zhang DIG-COMP-HII decomposition
    HII_Zhang_threshold = 1e39/L_sun
    DIG_Zhang_threshold = 10**38.5/L_sun
    sel_Zhang_DIG = (ALL.SBHa < DIG_Zhang_threshold).filled(False)
    sel_Zhang_COMP = np.bitwise_and((ALL.SBHa >= DIG_Zhang_threshold).filled(False), (ALL.SBHa < HII_Zhang_threshold).filled(False))
    sel_Zhang_HII = (ALL.SBHa >= HII_Zhang_threshold).filled(False)

    f = plt.figure(dpi=100)
    cmap = cmap_discrete()
    x = ALL.x_Y
    y = ALL.tau_V_neb - ALL.tau_V
    classif = np.ma.masked_all(ALL.WHa.shape)
    classif[sel_WHa_DIG] = 1
    classif[sel_WHa_COMP] = 2
    classif[sel_WHa_HII] = 3
    xbin = np.linspace(0, 0.6, 30)
    ax = f.gca()
    sc = ax.scatter(x, y, c=classif, cmap=cmap, **dflt_kw_scatter)
    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(sc, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
    cb.set_ticklabels(['DIG', 'COMP', 'HII'])
    xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_DIG)
    rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
    ax.plot(rs.xS, rs.yS, 'k--', lw=3)
    ax.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c='r', markersize=10)
    xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_COMP)
    rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
    ax.plot(rs.xS, rs.yS, 'k--', lw=3)
    ax.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c='g', markersize=10)
    xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_HII)
    rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
    ax.plot(rs.xS, rs.yS, 'k--', lw=3)
    ax.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c='b', markersize=10)
    ax.set_xlim(x_Y_range)
    ax.set_ylim(DtauV_range)
    ax.set_xlabel(r'x${}_Y$ [frac.]')
    ax.set_ylabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
    ax.grid()
    f.tight_layout()
    f.savefig('dig-stackedgals-xY_Dt-classifWHa.png')

    f = plt.figure(dpi=100)
    cmap = cmap_discrete()
    x = ALL.x_Y
    y = ALL.tau_V_neb - ALL.tau_V
    classif = np.ma.masked_all(ALL.SBHa.shape)
    classif[sel_Zhang_DIG] = 1
    classif[sel_Zhang_COMP] = 2
    classif[sel_Zhang_HII] = 3
    xbin = np.linspace(0, 0.6, 30)
    ax = f.gca()
    sc = ax.scatter(x, y, c=classif, cmap=cmap, **dflt_kw_scatter)
    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(sc, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
    cb.set_ticklabels(['DIG', 'COMP', 'HII'])
    xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_DIG)
    rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
    ax.plot(rs.xS, rs.yS, 'k--', lw=2)
    ax.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c='r', markersize=10)
    xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_COMP)
    rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
    ax.plot(rs.xS, rs.yS, 'k--', lw=2)
    ax.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c='g', markersize=10)
    xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_HII)
    rs = runstats(xm.compressed(), ym.compressed(), xbin=xbin, **dflt_kw_runstats)
    ax.plot(rs.xS, rs.yS, 'k--', lw=2)
    ax.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c='b', markersize=10)
    ax.set_xlim(x_Y_range)
    ax.set_ylim(DtauV_range)
    ax.set_xlabel(r'x${}_Y$ [frac.]')
    ax.set_ylabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
    ax.grid()
    f.tight_layout()
    f.savefig('dig-stackedgals-xY_Dt-classifZhang.png')
