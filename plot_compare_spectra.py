import os
import sys
import numpy as np
import matplotlib as mpl
from stackspectra import readFITS
from pytu.functions import debug_var
from matplotlib import pyplot as plt
from pytu.plots import plot_histo_ax, plot_text_ax
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
transp_choice = False
dpi_choice = 100
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal')
bad_threshold = 0.5
classif_labels = ['DIG', 'COMP', 'SF']
colors_DIG_COMP_SF = ['tomato', 'lightgreen', 'royalblue']
colors_lines_DIG_COMP_SF = ['darkred', 'olive', 'mediumblue']

# TODO XXX:
# def get_line_continuum(flux, wl_central, left=25, right=25):


def plot_bins_HbO3HaN2(data, R_bins_to_plot=None, debug=False, classif_labels=None):
    critical_bad_ratio_flag = 0.2
    initial_R_bins_to_plot = R_bins_to_plot
    if classif_labels is None:
        classif_labels = ['HIG', 'MIG', 'SF']
    for class_key in classif_labels:
        califaID = data._hdu[0].header['CALIFAID']
        wl = data.l_obs
        OI_window = np.bitwise_and(np.greater(wl, 6300-50), np.less(wl, 6300+50))
        HaN2_window = np.bitwise_and(np.greater(wl, 6563-50), np.less(wl, 6563+50))
        HbO3_window = np.bitwise_and(np.greater(wl, 4861-50), np.less(wl, 5007+50))
        S2_window = np.bitwise_and(np.greater(wl, 6724-50), np.less(wl, 6724+50))
        R_bin_center__R = data.R_bin_center__R
        rbinstep = data._hdu[0].header['RBINSTEP']
        if initial_R_bins_to_plot is None:
            R_bins_index = np.arange(len(data.classNtot__cR[class_key]))
            sel_nonzero_R_bins = data.classNtot__cR[class_key].astype('bool')
            R_bins_to_plot = R_bins_index[sel_nonzero_R_bins]
        N_R_bins_with_data = len(R_bins_to_plot)
        if N_R_bins_with_data == 0:
            print '%s does not have any %s zones' % (califaID, class_key)
            continue
        N_cols, N_rows = 4, N_R_bins_with_data
        f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 3))
        # f.suptitle(r'%s - %s' % (califaID, class_key))
        # print califaID, class_key, N_R_bins_with_data
        # print R_bins_to_plot
        for iax, iR in enumerate(R_bins_to_plot):
            # print iax, iR
            if N_R_bins_with_data == 1:
                ax_HbO3 = axArr[0]
                ax_OI = axArr[1]
                ax_HaN2 = axArr[2]
                ax_S2 = axArr[3]
            else:
                ax_HbO3 = axArr[iax, 0]
                ax_OI = axArr[iax, 1]
                ax_HaN2 = axArr[iax, 2]
                ax_S2 = axArr[iax, 3]
            f_res = data.O_rf__clR[class_key][:, iR] - data.M_rf__clR[class_key][:, iR]
            bad_ratio__l = data.bad_ratio__clR[class_key][:, iR]
            sel_badpix__l = (bad_ratio__l > 0)
            sel_crit_badpix__l = (bad_ratio__l > critical_bad_ratio_flag)
            c = 'b'
            # Hb OIII
            ax_HbO3.plot(wl, f_res, color=c)
            ax_HbO3.plot(wl[sel_badpix__l], f_res[sel_badpix__l], linestyle='', color='y', marker='+', lw=2)
            ax_HbO3.plot(wl[sel_crit_badpix__l], f_res[sel_crit_badpix__l], linestyle='', color='r', marker='^', lw=2)
            ax_HbO3.axvline(x='4861', c='k', ls='--')
            ax_HbO3.axvline(x='5007', c='k', ls='--')
            std_RGB = f_res[HbO3_window].std()
            ax_HbO3.set_ylim([f_res[HbO3_window].min() - std_RGB, f_res[HbO3_window].max() + std_RGB])
            ax_HbO3.set_xlim([4861-50, 5007+50])
            ax_HbO3.xaxis.set_major_locator(MultipleLocator(50))
            ax_HbO3.xaxis.set_minor_locator(MultipleLocator(10))
            ax2_HbO3 = ax_HbO3.twinx()
            ax2_HbO3.plot(wl, bad_ratio__l, color='g')
            ax2_HbO3.set_ylim([0, 1])
            ax2_HbO3.set_xlim([4861-50, 5007+50])
            ax2_HbO3.xaxis.set_major_locator(MultipleLocator(50))
            ax2_HbO3.xaxis.set_minor_locator(MultipleLocator(10))
            ax2_HbO3.yaxis.set_major_locator(MultipleLocator(0.25))
            ax2_HbO3.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax_HbO3.set_title(r'H$\beta$ - [OIII] (%.2f (i:%d) $\pm$ %.2f HLR)' % (R_bin_center__R[iR], iR, rbinstep))

            # OI
            ax_OI.plot(wl, f_res, color=c)
            ax_OI.plot(wl[sel_badpix__l], f_res[sel_badpix__l], linestyle='', color='y', marker='+', lw=2)
            ax_OI.plot(wl[sel_crit_badpix__l], f_res[sel_crit_badpix__l], linestyle='', color='r', marker='^', lw=2)
            ax_OI.axvline(x='6300', c='k', ls='--')
            std_RGB = f_res[OI_window].std()
            ax_OI.set_ylim([f_res[OI_window].min() - std_RGB, f_res[OI_window].max() + std_RGB])
            ax_OI.set_xlim([6300-50, 6300+50])
            ax_OI.xaxis.set_major_locator(MultipleLocator(25))
            ax_OI.xaxis.set_minor_locator(MultipleLocator(5))
            ax2_OI = ax_OI.twinx()
            ax2_OI.plot(wl, bad_ratio__l, color='g')
            ax2_OI.set_ylim([0, 1])
            ax2_OI.set_xlim([6300-50, 6300+50])
            ax2_OI.xaxis.set_major_locator(MultipleLocator(25))
            ax2_OI.xaxis.set_minor_locator(MultipleLocator(5))
            ax2_OI.yaxis.set_major_locator(MultipleLocator(0.25))
            ax2_OI.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax_OI.set_title(r'[OI] (%.2f (i:%d) $\pm$ %.2f HLR)' % (R_bin_center__R[iR], iR, rbinstep))

            # NII Ha
            ax_HaN2.plot(wl, f_res, color=c)
            ax_HaN2.plot(wl[sel_badpix__l], f_res[sel_badpix__l], linestyle='', color='y', marker='+', lw=2)
            ax_HaN2.plot(wl[sel_crit_badpix__l], f_res[sel_crit_badpix__l], linestyle='', color='r', marker='^', lw=2)
            ax_HaN2.axvline(x='6548', c='k', ls='--')
            ax_HaN2.axvline(x='6563', c='k', ls='--')
            ax_HaN2.axvline(x='6583', c='k', ls='--')
            std_RGB = f_res[HaN2_window].std()
            ax_HaN2.set_ylim([f_res[HaN2_window].min() - std_RGB, f_res[HaN2_window].max() + std_RGB])
            ax_HaN2.set_xlim([6563-50, 6563+50])
            ax_HaN2.xaxis.set_major_locator(MultipleLocator(25))
            ax_HaN2.xaxis.set_minor_locator(MultipleLocator(5))
            ax2_HaN2 = ax_HaN2.twinx()
            ax2_HaN2.plot(wl, bad_ratio__l, color='g')
            ax2_HaN2.set_ylim([0, 1])
            ax2_HaN2.set_xlim([6563-50, 6563+50])
            ax2_HaN2.xaxis.set_major_locator(MultipleLocator(25))
            ax2_HaN2.xaxis.set_minor_locator(MultipleLocator(5))
            ax2_HaN2.yaxis.set_major_locator(MultipleLocator(0.25))
            ax2_HaN2.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax_HaN2.set_title(r'H$\alpha$ - [NII] (%.2f (i:%d) $\pm$ %.2f HLR)' % (R_bin_center__R[iR], iR, rbinstep))

            # S2
            ax_S2.plot(wl, f_res, color=c)
            ax_S2.plot(wl[sel_badpix__l], f_res[sel_badpix__l], linestyle='', color='y', marker='+', lw=2)
            ax_S2.plot(wl[sel_crit_badpix__l], f_res[sel_crit_badpix__l], linestyle='', color='r', marker='^', lw=2)
            ax_S2.axvline(x='6717', c='k', ls='--')
            ax_S2.axvline(x='6731', c='k', ls='--')
            std_RGB = f_res[S2_window].std()
            ax_S2.set_ylim([f_res[S2_window].min() - std_RGB, f_res[S2_window].max() + std_RGB])
            ax_S2.set_xlim([6724-50, 6724+50])
            ax_S2.xaxis.set_major_locator(MultipleLocator(25))
            ax_S2.xaxis.set_minor_locator(MultipleLocator(5))
            ax2_S2 = ax_S2.twinx()
            ax2_S2.plot(wl, bad_ratio__l, color='g')
            ax2_S2.set_ylim([0, 1])
            ax2_S2.set_xlim([6724-50, 6724+50])
            ax2_S2.xaxis.set_major_locator(MultipleLocator(25))
            ax2_S2.xaxis.set_minor_locator(MultipleLocator(5))
            ax2_S2.yaxis.set_major_locator(MultipleLocator(0.25))
            ax2_S2.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax_S2.set_title(r'[SII] (%.2f (i:%d) $\pm$ %.2f HLR)' % (R_bin_center__R[iR], iR, rbinstep))

            plot_text_ax(ax_HbO3, '%s: %d zones' % (califaID, data.classNtot__cR[class_key][iR]), pos_x=0.01, pos_y=0.99, fs='11', va='top', ha='left', c='k')
            if debug:
                sel_lines = (sel_crit_badpix__l & (HbO3_window | HaN2_window | OI_window | S2_window))
                # print zip(wl, sel_crit_badpix__l & HbO3_window)
                # print np.sum(sel_lines)
                for l, flux, b in zip(wl[sel_lines], f_res[sel_lines], data.bad_ratio__clR[class_key][sel_lines, iR]):
                    Ntot = data.classNtot__cR[class_key][iR]
                    Nbad = Ntot * b
                    print '%s\t%s\t%d\t%d\t%e\t%f\t%d\t%d' % (califaID, class_key, iR, l, flux, b, Ntot, Nbad)
        f.tight_layout()
        f.savefig('%s_%s_spectra_withbadpixels.png' % (califaID, class_key), dpi=dpi_choice, transparent=transp_choice, orientation='landscape')
        plt.close(f)


def compare_plot(fitsA, fitsB, labelA='A', labelB='B', classif_labels=classif_labels):
    dataA = readFITS(fitsA)
    dataB = readFITS(fitsB)
    for class_key in classif_labels:
        califaID = dataA._hdu[0].header['CALIFAID']
        wl = dataA.l_obs
        norm_window = np.bitwise_and(np.greater(wl, 5635-45), np.less(wl, 5635+45))
        HaN2_window = np.bitwise_and(np.greater(wl, 6563-50), np.less(wl, 6563+50))
        HbO3_window = np.bitwise_and(np.greater(wl, 4861-50), np.less(wl, 5007+50))
        R_bin_center__R = dataB.R_bin_center__R
        N_R_bins = len(R_bin_center__R)  # dataB._hdu[0].header['NRBINS']
        rbinstep = dataB._hdu[0].header['RBINSTEP']
        R_bins_index_A = np.arange(len(dataA.classNtot__cR[class_key]))
        R_bins_index_B = np.arange(len(dataB.classNtot__cR[class_key]))
        sel_nonzero_R_bins_A = dataA.classNtot__cR[class_key].astype('bool')
        sel_nonzero_R_bins_B = dataB.classNtot__cR[class_key].astype('bool')
        set_A = {i for i in R_bins_index_A[sel_nonzero_R_bins_A]}
        set_B = {i for i in R_bins_index_B[sel_nonzero_R_bins_B]}
        N_A, N_B = np.sum(sel_nonzero_R_bins_A), np.sum(sel_nonzero_R_bins_B)
        print califaID, class_key
        print N_A, set_A, N_B, set_B
        R_bins_to_plot = list(set_A.union(set_B))
        print R_bins_to_plot
        N_R_bins_with_data = len(R_bins_to_plot)
        if N_R_bins_with_data == 0:
            continue
        N_cols, N_rows = 2, N_R_bins_with_data
        f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 3))
        first = True
        print 'NRBINS: ', N_R_bins_with_data
        for iax, iR in enumerate(R_bins_to_plot):
            c_A = 'r'
            c_B = 'b'
            # print iR
            if N_R_bins_with_data == 1:
                ax_HbO3 = axArr[0]
                ax_HaN2 = axArr[1]
            else:
                ax_HbO3 = axArr[iax, 0]
                ax_HaN2 = axArr[iax, 1]
            if iR in list(set_A):
                # print 'has_A'
                f_norm_A = dataA.O_rf__clR[class_key][norm_window, iR].mean()
                f_res_A = dataA.O_rf__clR[class_key][:, iR] - dataA.M_rf__clR[class_key][:, iR]
                f_res_A /= f_norm_A
                ax_HbO3.plot(wl, f_res_A, color=c_A, label='A: %s' % labelA)
                std_A = f_res_A[HbO3_window].std()
                ylim_min_HbO3 = f_res_A[HbO3_window].min() - std_A
                ylim_max_HbO3 = f_res_A[HbO3_window].max() + std_A
                ax_HaN2.plot(wl, f_res_A, color=c_A)
                std_A = f_res_A[HaN2_window].std()
                ylim_min_HaN2 = f_res_A[HaN2_window].min() - std_A
                ylim_max_HaN2 = f_res_A[HaN2_window].max() + std_A
                N_zones_A = dataA.classNtot__cR[class_key][iR]
            else:
                N_zones_A = 0
                ylim_min_HbO3 = 99999999.
                ylim_max_HbO3 = -99999999.
                ylim_min_HaN2 = 99999999.
                ylim_max_HaN2 = -99999999.
            if iR in list(set_B):
                # print 'has_B'
                f_norm_B = dataB.O_rf__clR[class_key][norm_window, iR].mean()
                f_res_B = dataB.O_rf__clR[class_key][:, iR] - dataB.M_rf__clR[class_key][:, iR]
                f_res_B /= f_norm_B
                ax_HbO3.plot(wl, f_res_B, color=c_B, label='B: %s' % labelB)
                ax_HaN2.plot(wl, f_res_B, color=c_B)
                std_B = f_res_B[HbO3_window].std()
                if ylim_min_HbO3 > (f_res_B[HbO3_window].min() - std_B):
                    ylim_min_HbO3 = f_res_B[HbO3_window].min() - std_B
                if ylim_max_HbO3 < (f_res_B[HbO3_window].max() + std_B):
                    ylim_max_HbO3 = f_res_B[HbO3_window].max() + std_B
                std_B = f_res_B[HaN2_window].std()
                if ylim_min_HaN2 > (f_res_B[HaN2_window].min() - std_B):
                    ylim_min_HaN2 = f_res_B[HaN2_window].min() - std_B
                if ylim_max_HaN2 < (f_res_B[HaN2_window].max() + std_B):
                    ylim_max_HaN2 = f_res_B[HaN2_window].max() + std_B
                N_zones_B = dataB.classNtot__cR[class_key][iR]
            else:
                N_zones_B = 0
            ax_HbO3.axvline(x='4861', c='k', ls='--')
            ax_HbO3.axvline(x='5007', c='k', ls='--')
            ax_HbO3.set_xlim([4861-50, 5007+50])
            ax_HbO3.set_ylim([ylim_min_HbO3, ylim_max_HbO3])
            ax_HbO3.xaxis.set_major_locator(MultipleLocator(50))
            ax_HbO3.xaxis.set_minor_locator(MultipleLocator(10))
            ax_HbO3.set_title(r'H$\beta$ and [OIII]  (%.2f (iR:%d) $\pm$ %.2f HLR)' % (R_bin_center__R[iR], iR, rbinstep))
            plot_text_ax(ax_HbO3, 'A/B:%d/%d zones' % (N_zones_A, N_zones_B), pos_x=0.01, pos_y=0.99, fs='11', va='top', ha='left', c='k')
            ax_HaN2.axvline(x='6548', c='k', ls='--')
            ax_HaN2.axvline(x='6563', c='k', ls='--')
            ax_HaN2.axvline(x='6583', c='k', ls='--')
            ax_HaN2.set_xlim([6563-50, 6563+50])
            ax_HaN2.set_ylim([ylim_min_HaN2, ylim_max_HaN2])
            ax_HaN2.xaxis.set_major_locator(MultipleLocator(25))
            ax_HaN2.xaxis.set_minor_locator(MultipleLocator(5))
            ax_HaN2.set_title(r'H$\alpha$ and [NII] (%.2f $\pm$ %.2f HLR)' % (R_bin_center__R[iR], rbinstep))
            plot_text_ax(ax_HaN2, 'A/B:%d/%d zones' % (N_zones_A, N_zones_B), pos_x=0.01, pos_y=0.99, fs='11', va='top', ha='left', c='k')
            plot_text_ax(ax_HaN2, '%s' % califaID, pos_x=0.99, pos_y=0.99, fs='15', va='top', ha='right', c='k')
            if first:
                ax_HbO3.legend(loc='upper right', frameon=False, fontsize=9)
                first = False
        f.tight_layout()
        f.savefig('%s_%s_spectra.png' % (califaID, class_key), dpi=dpi_choice, transparent=transp_choice, orientation='landscape')
        plt.close(f)


def plot_bad_frac_histogram(gals, stdata__g):
    ALL_DIG_bad_ratio__lR = np.hstack([d.bad_ratio__clR['DIG'] for d in stdata__g])
    ALL_COMP_bad_ratio__lR = np.hstack([d.bad_ratio__clR['COMP'] for d in stdata__g])
    ALL_SF_bad_ratio__lR = np.hstack([d.bad_ratio__clR['SF'] for d in stdata__g])
    wl = stdata__g[0].l_obs
    sel_Hb = ((wl > 4820) & (wl < 4900))
    sel_O3 = ((wl > 4967) & (wl < 5040))
    sel_N2Ha = ((wl > 6000) & (wl < 6700))
    sel = sel_Hb | sel_O3 | sel_N2Ha
    N_cols = 1
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    ax1, ax2, ax3 = axArr

    x = ALL_DIG_bad_ratio__lR[sel, :].flatten()
    range = [0, 0.5]
    plot_histo_ax(ax1, x, dataset_names='DIG', y_v_space=0.06, c='k', first=True, kwargs_histo=dict(histtype='bar', range=range, log=True, color=colors_DIG_COMP_SF[0], bins=100))
    plot_histo_ax(ax1, x, stats_txt=False, dataset_names='DIG', y_v_space=0.06, c='k', first=True, kwargs_histo=dict(histtype='step', range=range, log=True, color='k', bins=100))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)
    x = ALL_COMP_bad_ratio__lR[sel, :].flatten()
    plot_histo_ax(ax2, x, dataset_names='COMP', y_v_space=0.06, c='k', first=True, kwargs_histo=dict(histtype='bar', range=range, log=True, color=colors_DIG_COMP_SF[1], bins=100))
    plot_histo_ax(ax2, x, stats_txt=False, dataset_names='DIG', y_v_space=0.06, c='k', first=True, kwargs_histo=dict(histtype='step', range=range, log=True, color='k', bins=100))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.setp(ax2.xaxis.get_majorticklabels(), visible=False)
    x = ALL_SF_bad_ratio__lR[sel, :].flatten()
    plot_histo_ax(ax3, x, dataset_names='SF', y_v_space=0.06, c='k', first=True, kwargs_histo=dict(histtype='bar', range=range, log=True, color=colors_DIG_COMP_SF[2], bins=100))
    plot_histo_ax(ax3, x, stats_txt=False, dataset_names='DIG', y_v_space=0.06, c='k', first=True, kwargs_histo=dict(histtype='step', range=range, log=True, color='k', bins=100))
    ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax3.set_xlabel(r'$N_{BAD}/N_{TOTAL}$ (around BPT lines  )')
    # plot_text_ax(ax3, 'b)', 0.02, 0.98, 16, 'top', 'left', 'k')
    f.tight_layout(h_pad=0.0)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    f.savefig('bad_frac_histogram.png', dpi=dpi_choice, transparent=transp_choice)


if __name__ == '__main__':
    '''
        # legenda no grafico
        # numero de zonas por bin
        # limites em plt.xlim()
        pra frente: unidades do continuo
    '''
    Narg = len(sys.argv)
    if Narg > 1:
        if sys.argv[1] == 'C':
            fitsA, labelA = sys.argv[2], sys.argv[3]
            fitsB, labelB = sys.argv[4], sys.argv[5]
            compare_plot(fitsA, fitsB, labelA, labelB)
        elif sys.argv[1] == 'L':
            gals_file = sys.argv[2]
            fits_dir = sys.argv[3]
            with open(gals_file) as f:
                gals = [line.strip() for line in f.xreadlines() if line.strip()[0] != '#']
            stdata__g = [readFITS('%s/%s-RadBinStackedSpectra.fits' % (fits_dir, g)) for g in gals]
            plot_bad_frac_histogram(gals, stdata__g)
        elif sys.argv[1] == 'V':
            gals_file = sys.argv[2]
            with open(gals_file) as f:
                gals = [line.strip() for line in f.xreadlines() if line.strip()[0] != '#']
            labelA = '10-20'
            labelB = '8-15'
            for g in gals:
                fitsA = '/Users/lacerda/dev/astro/dig/runs/stackspectra/stack-fits/%s/%s-RadBinStackedSpectra.fits' % (labelA, g)
                fitsB = '/Users/lacerda/dev/astro/dig/runs/stackspectra/stack-fits/%s/%s-RadBinStackedSpectra.fits' % (labelB, g)
                compare_plot(fitsA, fitsB, labelA, labelB)
        elif sys.argv[1] == 'S':
            # plot_bins_HbO3HaN2
            gals_file = sys.argv[2]
            fits_dir = sys.argv[3]
            clabels = ['hDIG', 'mDIG', 'SFc']
            with open(gals_file) as f:
                gals = [line.strip() for line in f.xreadlines() if line.strip()[0] != '#']
            for g in gals:
                data_file = '%s/%s-RadBinStackedSpectra.fits' % (fits_dir, g)
                if os.path.isfile(data_file):
                    plot_bins_HbO3HaN2(readFITS(data_file, classif_labels=clabels), None, debug=True, classif_labels=clabels)
        else:
            fits = sys.argv[1]
            data = readFITS(fits)
            plot_bins_HbO3HaN2(data)
    else:
        dir = '/Users/lacerda/dev/astro/dig/'
        rundir = dir+'runs/stackspectra/stack-fits/20170426/8-15/'
        gals_file = dir+'gals_sim.txt'
        with open(gals_file) as f:
            gals = [line.strip() for line in f.xreadlines() if line.strip()[0] != '#']
        for g in gals:
            data = readFITS(rundir+g+'-RadBinStackedSpectra.fits')
            plot_bins_HbO3HaN2(data, None, debug=True)  # , classif_labels=['DIG'])
        ########################################################################
        ########################################################################
        ########################################################################
        # gal = 'K0025'
        # Rbins = [0]
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0028'
        # Rbins = [3]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0031'
        # Rbins = [2, 3, 4]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0034'
        # Rbins = [4]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0042'
        # Rbins = [3, 4]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0073'
        # Rbins = [5]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0140'
        # Rbins = [1]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0179'
        # Rbins = [1, 4, 5]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0183'
        # Rbins = [1]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0232'
        # Rbins = [0, 2, 6]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0277'
        # Rbins = [6]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0353'
        # Rbins = [3, 7, 9]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0414'
        # Rbins = [5, 9]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0436'
        # Rbins = [6]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0672'
        # Rbins = [0]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0707'
        # Rbins = [3, 4, 5, 6]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0748'
        # Rbins = [5, 6]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0769'
        # Rbins = [1, 2]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0836'
        # Rbins = [3]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
        # gal = 'K0924'
        # Rbins = [8, 9]
        # data = readFITS(rundir+gal+'-RadBinStackedSpectra.fits')
        # plot_bins_HbO3HaN2(data, Rbins, debug=True)
