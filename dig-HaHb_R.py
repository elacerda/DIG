import sys
import numpy as np
import matplotlib as mpl
from pytu.objects import runstats
from matplotlib import pyplot as plt
from CALIFAUtils.scripts import calc_xY
from CALIFAUtils.objects import CALIFAPaths
from CALIFAUtils.scripts import ma_mask_xyz
from CALIFAUtils.plots import DrawHLRCircle
from pystarlight.util.constants import L_sun
from CALIFAUtils.scripts import read_one_cube
from CALIFAUtils.plots import plot_gal_img_ax
from CALIFAUtils.scripts import try_q055_instead_q054
from mpl_toolkits.axes_grid1 import make_axes_locatable


# config variables
logSBHa_range = [3.5, 7]
logWHa_range = [0, 2.5]
logHaHb_range = [0, 1]
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
rbinini = 0.
rbinfin = 3.
rbinstep = 0.2
R_bin__r = np.arange(rbinini, rbinfin + rbinstep, rbinstep)
R_bin_center__r = (R_bin__r[:-1] + R_bin__r[1:]) / 2.0
N_R_bins = len(R_bin_center__r)


mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
dflt_kw_scatter = dict(marker='o', s=1, edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, debug=True, gs_prc=True, poly1d=True)  # , tendency=True)
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal')


def cmap_discrete(colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)], n_bins=3, cmap_name='myMap'):
    cm = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cm


if __name__ == '__main__':
    P = CALIFAPaths()
    califaID = sys.argv[1]

    kw_cube = dict(EL=EL, config=config, elliptical=elliptical)
    K = read_one_cube(califaID, **kw_cube)
    if K is None:
        print 'califaID:', califaID, ' trying another qVersion...'
        K = try_q055_instead_q054(califaID, **kw_cube)
        if K is None or K.EL is None:
            print
            sys.exit('califaID:%s missing fits files...' % califaID)

    lines = K.EL.lines
    maskSNR = K.EL._setMaskLineSNR('4861', minsnr=minSNR)
    maskSNR = np.bitwise_or(maskSNR, K.EL._setMaskLineSNR('6563', minsnr=minSNR))

    f_obs__lz = {}
    f_obs__lyx = {}
    SB_obs__lyx = {}
    for i, l in enumerate(lines):
        f_obs__z = K.EL.flux[i]
        mask = np.bitwise_or(~np.isfinite(f_obs__z), np.less(f_obs__z, 1e-40))
        mask = np.bitwise_or(mask, maskSNR)
        f_obs__lz[l] = np.ma.masked_array(f_obs__z, mask=mask, copy=True)
        # SB per arcsec^2
        f_obs__lyx[l] = K.zoneToYX(f_obs__lz[l]/K.zoneArea_pix, extensive=False)
        # SB per kpc^2
        L_obs__z = K.EL._F_to_L(f_obs__lz[l])/L_sun
        SB_obs__lyx[l] = K.zoneToYX(L_obs__z/(K.zoneArea_pc2 * 1e-6), extensive=False)
    WHa__yx = np.ma.masked_array(K.zoneToYX(K.EL.EW[K.EL.lines.index('6563')], extensive=False), mask=f_obs__lyx['6563'].mask, copy=True)

    # zones
    x_Y__z, _ = calc_xY(K, tY)
    tau_V__z = K.tau_V__z
    tau_V_neb__z = np.where((K.EL.tau_V_neb__z < 0).filled(0.0), 0, K.EL.tau_V_neb__z)
    fHb_obs__z = K.EL.flux[K.EL.lines.index('4861')]
    SBHb__z = K.EL._F_to_L(fHb_obs__z)/(L_sun * K.zoneArea_pc2 * 1e-6)
    SBHb_mask__z = np.bitwise_or(~np.isfinite(fHb_obs__z), np.less(fHb_obs__z, 1e-40))
    SBHb_mask__z = np.bitwise_or(SBHb_mask__z, maskSNR)
    SBHb__z = np.ma.masked_array(SBHb__z, mask=SBHb_mask__z, copy=True)
    fHa_obs__z = K.EL.flux[K.EL.lines.index('6563')]
    SBHa__z = K.EL._F_to_L(fHa_obs__z)/(L_sun * K.zoneArea_pc2 * 1e-6)
    SBHa_mask__z = np.bitwise_or(~np.isfinite(fHa_obs__z), np.less(fHa_obs__z, 1e-40))
    SBHa_mask__z = np.bitwise_or(SBHa_mask__z, maskSNR)
    SBHa__z = np.ma.masked_array(SBHa__z, mask=SBHa_mask__z, copy=True)
    WHa__z = np.ma.masked_array(K.EL.EW[K.EL.lines.index('6563')], mask=SBHa_mask__z, copy=True)

    # WHa DIG-COMP-HII decomposition
    sel_WHa_DIG__z = (WHa__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__z = np.bitwise_and((WHa__z >= DIG_WHa_threshold).filled(False), (WHa__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__z = (WHa__z >= HII_WHa_threshold).filled(False)
    sel_WHa_DIG__yx = (WHa__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__yx = np.bitwise_and((WHa__yx >= DIG_WHa_threshold).filled(False), (WHa__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__yx = (WHa__yx >= HII_WHa_threshold).filled(False)

    # SBHa-Zhang DIG-COMP-HII decomposition
    sel_Zhang_DIG__z = (SBHa__z < DIG_Zhang_threshold).filled(False)
    sel_Zhang_COMP__z = np.bitwise_and((SBHa__z >= DIG_Zhang_threshold).filled(False), (SBHa__z < HII_Zhang_threshold).filled(False))
    sel_Zhang_HII__z = (SBHa__z >= HII_Zhang_threshold).filled(False)
    sel_Zhang_DIG__yx = (SB_obs__lyx['6563'] < DIG_Zhang_threshold).filled(False)
    sel_Zhang_COMP__yx = np.bitwise_and((SB_obs__lyx['6563'] >= DIG_Zhang_threshold).filled(False), (SB_obs__lyx['6563'] < HII_Zhang_threshold).filled(False))
    sel_Zhang_HII__yx = (SB_obs__lyx['6563'] >= HII_Zhang_threshold).filled(False)

    sel_DIG__z = sel_WHa_DIG__z
    sel_COMP__z = sel_WHa_COMP__z
    sel_HII__z = sel_WHa_HII__z
    sel_DIG__yx = sel_WHa_DIG__yx
    sel_COMP__yx = sel_WHa_COMP__yx
    sel_HII__yx = sel_WHa_HII__yx

    map__yx = np.ma.masked_all((K.N_y, K.N_x))
    map__yx[sel_DIG__yx] = 1
    map__yx[sel_COMP__yx] = 2
    map__yx[sel_HII__yx] = 3

    distance_range = [0, 3]
    N_cols = 2
    N_rows = 2
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(15, 5))
    cmap = cmap_discrete()
    ax1, ax2, ax3 = axArr
    # AXIS 1
    plot_gal_img_ax(ax1, P.get_image_file(califaID), califaID, 0.02, 0.98, 16, K, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    # AXIS 2
    im = ax2.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax2)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
    cb.set_ticklabels(['DIG', 'COMP', 'HII'])
    DrawHLRCircle(ax2, K, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    # AXIS 3
    x = K.pixelDistance__yx/K.pixelsPerHLR
    y = np.ma.log10(SB_obs__lyx['6563']/SB_obs__lyx['4861'])
    ax3.scatter(np.ma.ravel(x), np.ma.ravel(y), c=np.ma.ravel(map__yx), cmap=cmap, **dflt_kw_scatter)
    ax3.set_xlim(distance_range)
    ax3.set_ylim(logHaHb_range)
    ax3.set_xlabel(r'R [HLR]')
    ax3.set_ylabel(r'$\log\ H\alpha/H\beta$')
    ax3.grid()
    xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_DIG__yx)
    rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
    ax3.plot(rs.xS, rs.yS, 'k--', lw=2)
    ax3.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(0), markersize=10)
    xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_COMP__yx)
    rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
    ax3.plot(rs.xS, rs.yS, 'k--', lw=2)
    ax3.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(1), markersize=10)
    xm, ym = ma_mask_xyz(x, y, mask=~sel_Zhang_HII__yx)
    rs = runstats(xm.compressed(), ym.compressed(), **dflt_kw_runstats)
    ax3.plot(rs.xS, rs.yS, 'k--', lw=2)
    ax3.plot(rs.xS, rs.yS, linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=cmap(2), markersize=10)
    f.tight_layout()
    f.savefig('%s-HaHb_R.png' % califaID)
