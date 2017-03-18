import sys
import numpy as np
import matplotlib as mpl
from pycasso import EmLinesDataCube
from matplotlib import pyplot as plt
from pytu.functions import ma_mask_xyz
from pytu.plots import plot_histo_ax, stats_med12sigma, plot_spearmanr_ax
from pystarlight.util.redenninglaws import calc_redlaw
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator, \
                              ScalarFormatter


mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
colors_DIG_COMP_SF = ['tomato', 'lightgreen', 'royalblue']
colors_lines_DIG_COMP_SF = ['darkred', 'olive', 'mediumblue']
classif_labels = ['DIG', 'COMP', 'SF']
cmap_R = plt.cm.copper
minorLocator = AutoMinorLocator(5)
debug = False
transp_choice = False
dpi_choice = 100
# debug = True
# CCM reddening law
q = calc_redlaw([4861, 6563], R_V=3.1, redlaw='CCM')
f_tauVneb = lambda Ha, Hb: np.ma.log(Ha / Hb / 2.86) / (q[0] - q[1])
BPTlines = ['4861', '5007', '6563', '6583']
dflt_kw_scatter = dict(s=5, marker='o', edgecolor='none')
logSNRrange = [0, 2.5]
logWHa_range = [0, 2.5]
SNRthreshold = 1


def main(argv=None):
    if len(sys.argv) <= 1:
        print 'usage: python compare_line_measurements.py K0073[,K0010[,...]]'
        print '       or'
        print '       python compare_line_measurements.py list-gals.txt'
        sys.exit(1)

    gals = []
    try:
        # read gals file
        with open(sys.argv[1]) as f:
            gals = [line.strip() for line in f.xreadlines() if line.strip()[0] != '#']
    except (IndexError, IOError):
        g_tmp = sys.argv[1].split(',')
        for g in g_tmp:
            gals.append(g)
    gals = sorted(gals)

    stacked_dict = fig1_compare_gal(gals)
    fig2_compare_allsample(gals, stacked_dict)


def fig1_compare_gal(gals):
    old_Hb__gz = []
    old_eHb__gz = []
    old_O3__gz = []
    old_eO3__gz = []
    old_Ha__gz = []
    old_eHa__gz = []
    old_N2__gz = []
    old_eN2__gz = []
    old_WHa__gz = []
    new_Hb__gz = []
    new_eHb__gz = []
    new_O3__gz = []
    new_eO3__gz = []
    new_Ha__gz = []
    new_eHa__gz = []
    new_N2__gz = []
    new_eN2__gz = []
    new_WHa__gz = []
    print gals
    for califaID in gals:
        oldELdir = '/Users/lacerda/califa/legacy/q054/EML/Bgstf6e'
        oldELsuf = '_synthesis_eBR_v20_q054.d22a512.ps03.k1.mE.CCM.Bgstf6e.EML.MC100.fits'
        oldEL = EmLinesDataCube('%s/%s%s' % (oldELdir, califaID, oldELsuf))
        oldf__lz = {l: oldEL.flux[oldEL.lines.index(l)] for l in BPTlines}
        oldef__lz = {l: oldEL.eflux[oldEL.lines.index(l)] for l in BPTlines}
        oldW6563__z = oldEL.EW[oldEL.lines.index('6563')]
        oldtauVneb__z = f_tauVneb(oldf__lz['6563'], oldf__lz['4861'])
        newELdir = '/Users/lacerda/califa/legacy/q054/EML/Bgsd6e'
        newELsuf = '_synthesis_eBR_v20_q054.d22a512.ps03.k1.mE.CCM.Bgsd6e.EML.MC100.fits'
        newEL = EmLinesDataCube('%s/%s%s' % (newELdir, califaID, newELsuf))
        newf__lz = {l: newEL.flux[newEL.lines.index(l)] for l in BPTlines}
        newef__lz = {l: newEL.eflux[newEL.lines.index(l)] for l in BPTlines}
        newW6563__z = newEL.EW[newEL.lines.index('6563')]
        newtauVneb__z = f_tauVneb(newf__lz['6563'], newf__lz['4861'])
        old_Hb__gz.append(oldf__lz['4861'])
        old_eHb__gz.append(oldef__lz['4861'])
        old_O3__gz.append(oldf__lz['5007'])
        old_eO3__gz.append(oldef__lz['5007'])
        old_Ha__gz.append(oldf__lz['6563'])
        old_eHa__gz.append(oldef__lz['6563'])
        old_N2__gz.append(oldf__lz['6583'])
        old_eN2__gz.append(oldef__lz['6583'])
        old_WHa__gz.append(oldW6563__z)
        new_Hb__gz.append(newf__lz['4861'])
        new_eHb__gz.append(newef__lz['4861'])
        new_O3__gz.append(newf__lz['5007'])
        new_eO3__gz.append(newef__lz['5007'])
        new_Ha__gz.append(newf__lz['6563'])
        new_eHa__gz.append(newef__lz['6563'])
        new_N2__gz.append(newf__lz['6583'])
        new_eN2__gz.append(newef__lz['6583'])
        new_WHa__gz.append(newW6563__z)
        ###############################################################
        ###############################################################
        ###############################################################
        N_cols = 2
        N_rows = 3
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 4))
        f.suptitle('%s' % califaID)
        ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axArr
        axArr = [ax1, ax2, ax3, ax4, ax5, ax6]
        for i_l, l in enumerate(BPTlines):
            old_f = oldf__lz[l]
            new_f = newf__lz[l]
            new_snr = new_f/newef__lz[l]
            sel_f = np.bitwise_and(np.bitwise_and(np.greater(old_f, 0), np.greater(new_f, 0)), np.greater_equal(new_snr, 1.))
            ax = axArr[i_l]
            x = np.ma.log10(1.+new_snr)
            y = (new_f - old_f)/new_f
            xm, ym = ma_mask_xyz(x, y, mask=~sel_f)
            plot_histo_ax(ax, ym.compressed(), ini_pos_y=0.45, y_v_space=0.07, first=True, histo=False, stats_txt=True, c='k')
            ax.scatter(xm, ym, **dflt_kw_scatter)
            ax.axhline(y=0, c='k', ls='--')
            ax.set_xlabel(r'$\log$ (1+SN${}_{NEW}$)')
            ax.set_ylabel(r'(F${}_{NEW}$ - F${}_{OLD}$)/F${}_{NEW}$')
            ax.set_title(l)
            ax.set_xlim(logSNRrange)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.xaxis.set_minor_locator(MaxNLocator(25))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_minor_locator(MaxNLocator(25))
        snrHb__z = newf__lz['4861']/newef__lz['4861']
        snrHa__z = newf__lz['6563']/newef__lz['6563']
        x = np.ma.log10(1.+snrHb__z)
        y = (newtauVneb__z - oldtauVneb__z)
        sel = np.bitwise_and(np.greater_equal(snrHa__z, SNRthreshold), np.greater_equal(snrHb__z, SNRthreshold))
        xm, ym = ma_mask_xyz(x, y, mask=~sel)
        ax5.scatter(xm, ym, **dflt_kw_scatter)
        ax5.set_xlabel(r'$\log$ (1+SN${}_{H\beta}$)')
        ax5.set_ylabel(r'($\tau_V^{NEW}$ - $\tau_V^{OLD}$)')
        ax5.axhline(y=0, c='k', ls='--')
        plot_histo_ax(ax5, ym.compressed(), ini_pos_y=0.98, y_v_space=0.07, first=True, histo=False, stats_txt=True, c='k')
        ax5.set_xlim(logSNRrange)
        ax5.xaxis.set_major_locator(MaxNLocator(5))
        ax5.xaxis.set_minor_locator(MaxNLocator(25))
        ax5.yaxis.set_major_locator(MaxNLocator(5))
        ax5.yaxis.set_minor_locator(MaxNLocator(25))
        x = np.ma.log10(newW6563__z)
        xm, ym = ma_mask_xyz(x, y, mask=~np.greater_equal(snrHa__z, SNRthreshold))
        ax6.scatter(xm, ym, **dflt_kw_scatter)
        ax6.axhline(y=0, c='k', ls='--')
        ax6.set_xlabel(r'$\log\ W_{H\alpha}$')
        ax6.set_ylabel(r'($\tau_V^{NEW}$ - $\tau_V^{OLD}$)')
        plot_histo_ax(ax6, ym.compressed(), ini_pos_y=0.98, y_v_space=0.07, first=True, histo=False, stats_txt=True, c='k')
        ax6.set_xlim(logWHa_range)
        ax6.xaxis.set_major_locator(MaxNLocator(5))
        ax6.xaxis.set_minor_locator(MaxNLocator(25))
        ax6.yaxis.set_major_locator(MaxNLocator(5))
        ax6.yaxis.set_minor_locator(MaxNLocator(25))
        f.tight_layout(w_pad=0.8, h_pad=0.8)
        f.savefig('%s_comparelines.png' % califaID, dpi=dpi_choice, transparent=transp_choice)
        plt.close(f)
        ###############################################################
        ###############################################################
        ###############################################################
    ret_dict = dict(
        old_O3__gz=np.ma.hstack(old_O3__gz),
        old_eO3__gz=np.ma.hstack(old_eO3__gz),
        old_Hb__gz=np.ma.hstack(old_Hb__gz),
        old_eHb__gz=np.ma.hstack(old_eHb__gz),
        old_Ha__gz=np.ma.hstack(old_Ha__gz),
        old_eHa__gz=np.ma.hstack(old_eHa__gz),
        old_N2__gz=np.ma.hstack(old_N2__gz),
        old_eN2__gz=np.ma.hstack(old_eN2__gz),
        old_WHa__gz=np.ma.hstack(old_WHa__gz),
        new_O3__gz=np.ma.hstack(new_O3__gz),
        new_eO3__gz=np.ma.hstack(new_eO3__gz),
        new_Hb__gz=np.ma.hstack(new_Hb__gz),
        new_eHb__gz=np.ma.hstack(new_eHb__gz),
        new_Ha__gz=np.ma.hstack(new_Ha__gz),
        new_eHa__gz=np.ma.hstack(new_eHa__gz),
        new_N2__gz=np.ma.hstack(new_N2__gz),
        new_eN2__gz=np.ma.hstack(new_eN2__gz),
        new_WHa__gz=np.ma.hstack(new_WHa__gz),
    )
    return ret_dict


def fig2_compare_allsample(gals, data_dict):
    NGals = len(gals)
    old_Hb__gz = data_dict['old_Hb__gz']
    old_eHb__gz = data_dict['old_eHb__gz']
    old_O3__gz = data_dict['old_O3__gz']
    old_eO3__gz = data_dict['old_eO3__gz']
    old_Ha__gz = data_dict['old_Ha__gz']
    old_eHa__gz = data_dict['old_eHa__gz']
    old_N2__gz = data_dict['old_N2__gz']
    old_eN2__gz = data_dict['old_eN2__gz']
    old_WHa__gz = data_dict['old_WHa__gz']
    new_Hb__gz = data_dict['new_Hb__gz']
    new_eHb__gz = data_dict['new_eHb__gz']
    new_O3__gz = data_dict['new_O3__gz']
    new_eO3__gz = data_dict['new_eO3__gz']
    new_Ha__gz = data_dict['new_Ha__gz']
    new_eHa__gz = data_dict['new_eHa__gz']
    new_N2__gz = data_dict['new_N2__gz']
    new_eN2__gz = data_dict['new_eN2__gz']
    new_WHa__gz = data_dict['new_WHa__gz']
    BPTLines_dict = {
        '4861': dict(old_flux=old_Hb__gz, old_eflux=old_eHb__gz, new_flux=new_Hb__gz, new_eflux=new_eHb__gz),
        '5007': dict(old_flux=old_O3__gz, old_eflux=old_eO3__gz, new_flux=new_O3__gz, new_eflux=new_eO3__gz),
        '6563': dict(old_flux=old_Ha__gz, old_eflux=old_eHa__gz, new_flux=new_Ha__gz, new_eflux=new_eHa__gz),
        '6583': dict(old_flux=old_N2__gz, old_eflux=old_eN2__gz, new_flux=new_N2__gz, new_eflux=new_eN2__gz),
    }
    old_tauVNeb = f_tauVneb(old_Ha__gz, old_Hb__gz)
    new_tauVNeb = f_tauVneb(new_Ha__gz, new_Hb__gz)
    N_cols = 4
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 4))
    f.suptitle('%d galaxies' % NGals)
    ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = axArr
    axArr = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    for i_l, l in enumerate(BPTlines):
        old_f = BPTLines_dict[l]['old_flux']
        new_f = BPTLines_dict[l]['new_flux']
        new_snr = new_f/BPTLines_dict[l]['new_eflux']
        sel_f = np.bitwise_and(np.bitwise_and(np.greater(old_f, 0), np.greater(new_f, 0)), np.greater_equal(new_snr, SNRthreshold))
        ax = axArr[i_l]
        x = np.ma.log10(1.+new_snr)
        y = (new_f - old_f)/new_f
        xm, ym = ma_mask_xyz(x, y, mask=~sel_f)
        ini_pos_y = 0.45
        if l[0] == '6':
            ini_pos_y = 0.98
        plot_histo_ax(ax, ym.compressed(), ini_pos_y=ini_pos_y, y_v_space=0.07, first=True, histo=False, stats_txt=True, c='k')
        ax.scatter(xm, ym, **dflt_kw_scatter)
        ax.axhline(y=0, c='k', ls='--')
        ax.set_xlabel(r'$\log$ (1+SN${}_{NEW}$)')
        ax.set_ylabel(r'(F${}_{NEW}$ - F${}_{OLD}$)/F${}_{NEW}$')
        ax.set_title(l)
        ax.set_xlim(logSNRrange)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_minor_locator(MaxNLocator(25))
        ax.grid()
    snrHa = new_Ha__gz/new_eHa__gz
    snrHb = new_Hb__gz/new_eHb__gz
    x = np.ma.log10(1.+snrHb)
    y = (new_tauVNeb - old_tauVNeb)
    sel = np.bitwise_and(np.greater_equal(snrHa, SNRthreshold), np.greater_equal(snrHb, SNRthreshold))
    xm, ym = ma_mask_xyz(x, y, mask=~sel)
    ax5.scatter(xm, ym, **dflt_kw_scatter)
    ax5.set_xlabel(r'$\log$ (1+SN${}_{H\beta}$)')
    ax5.set_ylabel(r'($\tau_V^{NEW}$ - $\tau_V^{OLD}$)')
    ax5.axhline(y=0, c='k', ls='--')
    plot_histo_ax(ax5, ym.compressed(), ini_pos_y=0.98, y_v_space=0.07, first=True, histo=False, stats_txt=True, c='k')
    ax5.set_xlim(logSNRrange)
    ax5.xaxis.set_major_locator(MaxNLocator(5))
    ax5.xaxis.set_minor_locator(MaxNLocator(25))
    ax5.yaxis.set_major_locator(MaxNLocator(5))
    ax5.yaxis.set_minor_locator(MaxNLocator(25))
    ax5.grid()
    x = np.ma.log10(new_WHa__gz)
    sel = np.bitwise_and(np.greater_equal(snrHa, SNRthreshold), np.greater_equal(snrHb, SNRthreshold))
    xm, ym = ma_mask_xyz(x, y, mask=~sel)
    ax6.scatter(xm, ym, **dflt_kw_scatter)
    ax6.axhline(y=0, c='k', ls='--')
    ax6.set_xlabel(r'$\log\ W_{H\alpha}$')
    ax6.set_ylabel(r'($\tau_V^{NEW}$ - $\tau_V^{OLD}$)')
    plot_histo_ax(ax6, ym.compressed(), ini_pos_y=0.98, y_v_space=0.07, first=True, histo=False, stats_txt=True, c='k')
    ax6.set_xlim(logWHa_range)
    ax6.xaxis.set_major_locator(MaxNLocator(5))
    ax6.xaxis.set_minor_locator(MaxNLocator(25))
    ax6.yaxis.set_major_locator(MaxNLocator(5))
    ax6.yaxis.set_minor_locator(MaxNLocator(25))
    ax6.grid()
    x = np.ma.log10(old_WHa__gz)
    y = old_tauVNeb
    sel = np.bitwise_and(np.greater_equal(snrHa, SNRthreshold), np.greater_equal(snrHb, SNRthreshold))
    xm, ym = ma_mask_xyz(x, y, mask=~sel)
    ax7.scatter(xm, ym, **dflt_kw_scatter)
    plot_spearmanr_ax(ax=ax7, x=xm.compressed(), y=ym.compressed(), horizontalalignment='right', pos_x=0.99, pos_y=0.99, fontsize=14)
    ax7.axhline(y=0, c='k', ls='--')
    ax7.set_xlabel(r'$\log\ W_{H\alpha}$ OLD')
    ax7.set_ylabel(r'$\tau_V^{neb}$ OLD')
    x_range = logWHa_range
    xbins = np.arange(x_range[0], x_range[1] + 0.2, 0.2)
    yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), xbins)
    yMedian = prc[2]
    y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
    ax7.plot(bin_center, yMedian, 'k-', lw=2)
    for y_prc in y_12sigma:
        ax7.plot(bin_center, y_prc, 'k--', lw=2)
    # plot_histo_ax(ax7, ym.compressed(), ini_pos_y=0.45, y_v_space=0.07, first=True, histo=False, stats_txt=True, c='k')
    ax7.set_xlim(logWHa_range)
    ax7.xaxis.set_major_locator(MaxNLocator(5))
    ax7.xaxis.set_minor_locator(MaxNLocator(25))
    ax7.yaxis.set_major_locator(MaxNLocator(5))
    ax7.yaxis.set_minor_locator(MaxNLocator(25))
    ax7.grid()
    x = np.ma.log10(new_WHa__gz)
    y = new_tauVNeb
    sel = np.bitwise_and(np.greater_equal(snrHa, SNRthreshold), np.greater_equal(snrHb, SNRthreshold))
    xm, ym = ma_mask_xyz(x, y, mask=~sel)
    ax8.scatter(xm, ym, **dflt_kw_scatter)
    plot_spearmanr_ax(ax=ax8, x=xm.compressed(), y=ym.compressed(), horizontalalignment='right', pos_x=0.99, pos_y=0.99, fontsize=14)
    ax8.axhline(y=0, c='k', ls='--')
    ax8.set_xlabel(r'$\log\ W_{H\alpha}$ NEW')
    ax8.set_ylabel(r'$\tau_V^{neb}$ NEW')
    x_range = logWHa_range
    xbins = np.arange(x_range[0], x_range[1] + 0.2, 0.2)
    yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), xbins)
    yMedian = prc[2]
    y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
    ax8.plot(bin_center, yMedian, 'k-', lw=2)
    for y_prc in y_12sigma:
        ax8.plot(bin_center, y_prc, 'k--', lw=2)
    # plot_histo_ax(ax8, ym.compressed(), ini_pos_y=0.45, y_v_space=0.07, first=True, histo=False, stats_txt=True, c='k')
    ax8.set_xlim(logWHa_range)
    ax8.xaxis.set_major_locator(MaxNLocator(5))
    ax8.xaxis.set_minor_locator(MaxNLocator(25))
    ax8.yaxis.set_major_locator(MaxNLocator(5))
    ax8.yaxis.set_minor_locator(MaxNLocator(25))
    ax8.grid()
    sel = np.bitwise_and(np.greater_equal(snrHa, SNRthreshold), np.greater_equal(snrHb, SNRthreshold))
    old_tauVNebm, new_tauVNebm = ma_mask_xyz(old_tauVNeb, new_tauVNeb, mask=~sel)
    plot_histo_ax(ax9, old_tauVNebm.compressed(), histo=True, stats=True, first=True, ini_pos_y=0.98, y_v_space=0.07, c='k', kwargs_histo=dict(normed=False, range=[-6, 6]))
    ax9.set_xlabel(r'$\tau_V^{neb}$ OLD')
    plot_histo_ax(ax10, new_tauVNebm.compressed(), histo=True, stats=True, first=True, ini_pos_y=0.98, y_v_space=0.07, c='k', kwargs_histo=dict(normed=False, range=[-6, 6]))
    ax10.set_xlabel(r'$\tau_V^{neb}$ NEW')
    WHamax = 8
    sel = np.bitwise_and(np.greater_equal(snrHa, SNRthreshold), np.greater_equal(snrHb, SNRthreshold))
    sel = np.bitwise_and(sel, np.less(new_WHa__gz, WHamax))
    old_tauVNebm, new_tauVNebm = ma_mask_xyz(old_tauVNeb, new_tauVNeb, mask=~sel)
    plot_histo_ax(ax11, old_tauVNebm.compressed(), histo=True, stats=True, first=True, pos_x=0.4, ini_pos_y=0.98, y_v_space=0.07, c='b', kwargs_histo=dict(normed=False, range=[-6, 6]))
    plot_histo_ax(ax12, new_tauVNebm.compressed(), histo=True, stats=True, first=True, pos_x=0.4, ini_pos_y=0.98, y_v_space=0.07, c='b', kwargs_histo=dict(normed=False, range=[-6, 6]))
    SNRmax = 3
    sel = np.bitwise_and(np.greater_equal(snrHa, SNRmax), np.greater_equal(snrHb, SNRmax))
    sel = np.bitwise_and(sel, np.less(new_WHa__gz, WHamax))
    old_tauVNebm, new_tauVNebm = ma_mask_xyz(old_tauVNeb, new_tauVNeb, mask=~sel)
    plot_histo_ax(ax11, old_tauVNebm.compressed(), histo=True, dataset_names='SN >= %d' % SNRmax, stats=True, ini_pos_y=0.98, y_v_space=0.07, c='k', kwargs_histo=dict(normed=False, color='k', alpha=0.5, range=[-6, 6]))
    plot_histo_ax(ax12, new_tauVNebm.compressed(), histo=True, dataset_names='SN >= %d' % SNRmax, stats=True, ini_pos_y=0.98, y_v_space=0.07, c='k', kwargs_histo=dict(normed=False, color='k', alpha=0.5, range=[-6, 6]))
    ax11.set_title(r'W${}_{H\alpha}\ <\ %d$' % WHamax)
    ax12.set_title(r'W${}_{H\alpha}\ <\ %d$' % WHamax)
    ax11.set_xlabel(r'$\tau_V^{neb}$ OLD')
    ax12.set_xlabel(r'$\tau_V^{neb}$ NEW')
    f.tight_layout(w_pad=0.8, h_pad=0.8)
    f.savefig('comparelines_allsample.png', dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


if __name__ == '__main__':
    main(sys.argv)
