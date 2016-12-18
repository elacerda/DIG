import sys
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from CALIFAUtils.scripts import calc_xY
from pytu.plots import plot_histo_ax
from CALIFAUtils.objects import stack_gals
from CALIFAUtils.scripts import loop_cubes
from pystarlight.util.constants import L_sun
from CALIFAUtils.scripts import try_q055_instead_q054

# Matplotlib config
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'


# config variables
logSBHa_range = [3.5, 7]
logWHa_range = [0, 2.5]
logHaHb_range = [0, 1]
DtauV_range = [-2, 3]
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
kw_cube = dict(EL=EL, config=config, elliptical=elliptical)
dflt_kw_scatter = dict(marker='o', s=1, edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, debug=True, gs_prc=True, poly1d=True)  # , tendency=True)
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal')


def histograms_HaHb_Dt(ALL):
    # WHa DIG-COMP-HII decomposition
    sel_WHa_DIG__z = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__z = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < HII_WHa_threshold).filled(False))
    sel_WHa_HII__z = (ALL.W6563__z >= HII_WHa_threshold).filled(False)
    sel_WHa_DIG__yx = (ALL.W6563__yx < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__yx = np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(False), (ALL.W6563__yx < HII_WHa_threshold).filled(False))
    sel_WHa_HII__yx = (ALL.W6563__yx >= HII_WHa_threshold).filled(False)

    # SBHa-Zhang DIG-COMP-HII decomposition
    sel_Zhang_DIG__z = (ALL.SB6563__z < DIG_Zhang_threshold).filled(False)
    sel_Zhang_COMP__z = np.bitwise_and((ALL.SB6563__z >= DIG_Zhang_threshold).filled(False), (ALL.SB6563__z < HII_Zhang_threshold).filled(False))
    sel_Zhang_HII__z = (ALL.SB6563__z >= HII_Zhang_threshold).filled(False)
    sel_Zhang_DIG__yx = (ALL.SB6563__yx < DIG_Zhang_threshold).filled(False)
    sel_Zhang_COMP__yx = np.bitwise_and((ALL.SB6563__yx >= DIG_Zhang_threshold).filled(False), (ALL.SB6563__yx < HII_Zhang_threshold).filled(False))
    sel_Zhang_HII__yx = (ALL.SB6563__yx >= HII_Zhang_threshold).filled(False)

    tau_V_neb__z = ALL.tau_V_neb__z
    tau_V_neb__yx = ALL.tau_V_neb__yx

    N_cols = 2
    N_rows = 2
    f, axArr = plt.subplots(N_rows, N_cols, dpi=100, figsize=(15, 10))
    ((ax1, ax2), (ax3, ax4)) = axArr
    x = np.ma.log10(ALL.f6563__z/ALL.f4861__z)
    range = logHaHb_range
    xDs = [x[sel_WHa_DIG__z].compressed(),  x[sel_WHa_COMP__z].compressed(),  x[sel_WHa_HII__z].compressed()]
    ax1.set_title('zones')
    print 'ax1'
    plot_histo_ax(ax1, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    plot_histo_ax(ax1, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
    ax1.set_xlabel(r'$\log\ H\alpha/H\beta$')
    x = tau_V_neb__z - ALL.tau_V__z
    xDs = [x[sel_WHa_DIG__z].compressed(),  x[sel_WHa_COMP__z].compressed(),  x[sel_WHa_HII__z].compressed()]
    range = DtauV_range
    ax2.set_title('zones')
    print 'ax2'
    plot_histo_ax(ax2, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    plot_histo_ax(ax2, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
    ax2.set_xlabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
    x = np.ma.log10(ALL.SB6563__yx/ALL.SB4861__yx)
    xDs = [x[sel_WHa_DIG__yx].compressed(),  x[sel_WHa_COMP__yx].compressed(),  x[sel_WHa_HII__yx].compressed()]
    range = logHaHb_range
    ax3.set_title('pixels')
    print 'ax3'
    plot_histo_ax(ax3, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    plot_histo_ax(ax3, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
    ax3.set_xlabel(r'$\log\ H\alpha/H\beta$')
    x = tau_V_neb__yx - ALL.tau_V__yx
    xDs = [x[sel_WHa_DIG__yx].compressed(),  x[sel_WHa_COMP__yx].compressed(),  x[sel_WHa_HII__yx].compressed()]
    range = DtauV_range
    ax4.set_title('pixels')
    print 'ax4'
    plot_histo_ax(ax4, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    plot_histo_ax(ax4, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
    ax4.set_xlabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
    f.tight_layout()
    f.savefig('dig-sample-histo-logHaHb_Dt_colorsWHa.png')

    N_cols = 2
    N_rows = 2
    f, axArr = plt.subplots(N_rows, N_cols, dpi=100, figsize=(15, 10))
    ((ax1, ax2), (ax3, ax4)) = axArr
    x = np.ma.log10(ALL.f6563__z/ALL.f4861__z)
    range = logHaHb_range
    xDs = [x[sel_Zhang_DIG__z].compressed(),  x[sel_Zhang_COMP__z].compressed(),  x[sel_Zhang_HII__z].compressed()]
    ax1.set_title('zones')
    plot_histo_ax(ax1, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    plot_histo_ax(ax1, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
    ax1.set_xlabel(r'$\log\ H\alpha/H\beta$')
    x = tau_V_neb__z - ALL.tau_V__z
    xDs = [x[sel_Zhang_DIG__z].compressed(),  x[sel_Zhang_COMP__z].compressed(),  x[sel_Zhang_HII__z].compressed()]
    range = DtauV_range
    ax2.set_title('zones')
    plot_histo_ax(ax2, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    plot_histo_ax(ax2, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
    ax2.set_xlabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
    x = np.ma.log10(ALL.SB6563__yx/ALL.SB4861__yx)
    xDs = [x[sel_Zhang_DIG__yx].compressed(),  x[sel_Zhang_COMP__yx].compressed(),  x[sel_Zhang_HII__yx].compressed()]
    range = logHaHb_range
    ax3.set_title('pixels')
    plot_histo_ax(ax3, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    plot_histo_ax(ax3, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
    ax3.set_xlabel(r'$\log\ H\alpha/H\beta$')
    x = tau_V_neb__yx - ALL.tau_V__yx
    xDs = [x[sel_Zhang_DIG__yx].compressed(),  x[sel_Zhang_COMP__yx].compressed(),  x[sel_Zhang_HII__yx].compressed()]
    range = DtauV_range
    ax4.set_title('pixels')
    plot_histo_ax(ax4, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
    plot_histo_ax(ax4, xDs, y_v_space=0.06, first=False, c=['r', 'g', 'b'], kwargs_histo=dict(histtype='barstacked', color=['r', 'g', 'b'], normed=False, range=range))
    ax4.set_xlabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
    f.tight_layout()
    f.savefig('dig-sample-histo-logHaHb_Dt_colorsZhang.png')


if __name__ == '__main__':
    filename = sys.argv[1]

    ALL = stack_gals().load(filename)

    histograms_HaHb_Dt(ALL)
