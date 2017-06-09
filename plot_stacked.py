import os
import sys
import numpy as np
import matplotlib as mpl
from pytu.functions import debug_var
from matplotlib import pyplot as plt
from stackspectra import readFITS


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


def plot_spectra(args, wl, O_bin, M_bin, err_bin, b_bin, rbin, K, sel_zones, class_key, N_zones, O_zones=None):
    from pytu.plots import plot_histo_ax
    import matplotlib.gridspec as gridspec
    from CALIFAUtils.plots import DrawHLRCircle
    from matplotlib.ticker import MultipleLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    califaID = K.califaID
    line_window_edges = 25
    N2Ha_window = np.bitwise_and(np.greater(wl, 6563-(2.*line_window_edges)), np.less(wl, 6563+(2.*line_window_edges)))
    Hb_window = np.bitwise_and(np.greater(wl, 4861-line_window_edges), np.less(wl, 4861+line_window_edges))
    O3_window = np.bitwise_and(np.greater(wl, 5007-line_window_edges), np.less(wl, 5007+line_window_edges))
    N_cols, N_rows = 5, 2
    f = plt.figure(figsize=(N_cols * 5, N_rows * 5))
    gs = gridspec.GridSpec(N_rows, N_cols)
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, :], height_ratios=[4, 1], hspace=0.001, wspace=0.001)
    ax_spectra = plt.subplot(gs1[0])
    ax_b = plt.subplot(gs1[1])
    ax_map_v_0 = plt.subplot(gs[1, 0])
    ax_hist_v_0 = plt.subplot(gs[1, 1])
    ax_Hb = plt.subplot(gs[1, 2])
    ax_O3 = plt.subplot(gs[1, 3])
    ax_N2Ha = plt.subplot(gs[1, 4])
    Resid_bin = O_bin - M_bin
    ax_spectra.plot(wl, O_bin, '-k', lw=1)
    ax_spectra.plot(wl, M_bin, '-y', lw=1)
    ax_spectra.plot(wl, Resid_bin, '-r', lw=1)
    ax_b.bar(wl, b_bin, color='b')
    ax_Hb.plot(wl[Hb_window], O_bin[Hb_window], '-k', lw=1)
    ax_Hb.plot(wl[Hb_window], M_bin[Hb_window], '-y', lw=1)
    ax_Hb.plot(wl[Hb_window], Resid_bin[Hb_window], '-r', lw=1)
    ax_O3.plot(wl[O3_window], O_bin[O3_window], '-k', lw=1)
    ax_O3.plot(wl[O3_window], M_bin[O3_window], '-y', lw=1)
    ax_O3.plot(wl[O3_window], Resid_bin[O3_window], '-r', lw=1)
    ax_N2Ha.plot(wl[N2Ha_window], O_bin[N2Ha_window], '-k', lw=1)
    ax_N2Ha.plot(wl[N2Ha_window], M_bin[N2Ha_window], '-y', lw=1)
    ax_N2Ha.plot(wl[N2Ha_window], Resid_bin[N2Ha_window], '-r', lw=1)
    if O_zones is not None:
        ax_spectra.plot(wl, O_zones, '-c', alpha=0.3, lw=0.3)
        ax_Hb.plot(wl[Hb_window], O_zones[Hb_window, :], '-c', alpha=0.3, lw=0.3)
        ax_O3.plot(wl[O3_window], O_zones[O3_window, :], '-c', alpha=0.3, lw=0.3)
        ax_N2Ha.plot(wl[N2Ha_window], O_zones[N2Ha_window, :], '-c', alpha=0.3, lw=0.3)
    ax_spectra.xaxis.set_major_locator(MultipleLocator(250))
    ax_spectra.xaxis.set_minor_locator(MultipleLocator(50))
    ax_b.xaxis.set_major_locator(MultipleLocator(250))
    ax_b.xaxis.set_minor_locator(MultipleLocator(50))
    ax_b.yaxis.set_major_locator(MultipleLocator(100))
    ax_Hb.set_title(r'H$\beta$', y=1.1)
    ax_O3.set_title(r'[OIII]', y=1.1)
    ax_N2Ha.set_title(r'[NII] and H$\alpha$', y=1.1)
    ax_Hb.xaxis.set_major_locator(MultipleLocator(25))
    ax_Hb.xaxis.set_minor_locator(MultipleLocator(5))
    ax_O3.xaxis.set_major_locator(MultipleLocator(25))
    ax_O3.xaxis.set_minor_locator(MultipleLocator(5))
    ax_N2Ha.xaxis.set_major_locator(MultipleLocator(50))
    ax_N2Ha.xaxis.set_minor_locator(MultipleLocator(10))
    ax_spectra.get_xaxis().set_visible(False)
    ax_spectra.grid()
    ax_b.grid()
    ax_Hb.grid(which='both')
    ax_O3.grid(which='both')
    ax_N2Ha.grid(which='both')
    # v_0 map & histogram
    v_0_range = [K.v_0.min(), K.v_0.max()]
    plot_histo_ax(ax_hist_v_0, K.v_0[sel_zones], y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=v_0_range))
    v_0__yx = K.zoneToYX(np.ma.masked_array(K.v_0, mask=~sel_zones), extensive=False)
    im = ax_map_v_0.imshow(v_0__yx, vmin=v_0_range[0], vmax=v_0_range[1], cmap='RdBu', **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax_map_v_0)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'v${}_0$ [km/s]')
    DrawHLRCircle(ax_map_v_0, a=K.HLR_pix, pa=K.pa, ba=K.ba, x0=K.x0, y0=K.y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax_map_v_0.set_title(r'v${}_0$ map')
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.suptitle(r'%s - %s - R bin center: %.2f ($\pm$ %.2f) HLR - %d zones' % (califaID, class_key, rbin, args.rbinstep/2., N_zones))
    f.savefig('%s_%s_spectra_%.2fHLR.png' % (califaID, class_key, rbin), dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


# if __name__ == '__main__':
