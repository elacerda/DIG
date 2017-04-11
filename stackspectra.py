import os
import sys
import numpy as np
import matplotlib as mpl
from pycasso import fitsQ3DataCube
from pytu.functions import debug_var
from matplotlib import pyplot as plt
from CALIFAUtils.scripts import stack_spectra
from pytu.objects import tupperware_none, readFileArgumentParser


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


def parser_args(default_args_file='default.args'):
    '''
        Parse the command line args
        With fromfile_prefix_chars=@ we can read and parse command line args
        inside a file with @file.txt.
        default args inside default_args_file
    '''
    default_args = {
        'debug': False,
        'superfits': None,
        'emlfits': None,
        'rbinini': 0.,
        'rbinfin': 3.,
        'rbinstep': 0.2,
        'sfth': 15,
        'digth': 8,
    }
    parser = readFileArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--debug', '-D', action='store_true', default=default_args['debug'])
    parser.add_argument('--superfits', '-S', metavar='FILE', type=str, default=default_args['superfits'])
    parser.add_argument('--emlfits', '-E', metavar='FILE', type=str, default=default_args['emlfits'])
    parser.add_argument('--sfth', metavar='FLOAT', type=float, default=default_args['sfth'])
    parser.add_argument('--digth', metavar='FLOAT', type=float, default=default_args['digth'])
    parser.add_argument('--rbinini', metavar='HLR', type=float, default=default_args['rbinini'])
    parser.add_argument('--rbinfin', metavar='HLR', type=float, default=default_args['rbinfin'])
    parser.add_argument('--rbinstep', metavar='HLR', type=float, default=default_args['rbinstep'])
    args_list = sys.argv[1:]
    # if exists file default.args, load default args
    print default_args_file
    if os.path.isfile(default_args_file):
        args_list.insert(0, '@%s' % default_args_file)
    debug_var(True, args_list=args_list)
    args = parser.parse_args(args=args_list)
    args.R_bin__r = np.arange(args.rbinini, args.rbinfin + args.rbinstep, args.rbinstep)
    args.R_bin_center__r = (args.R_bin__r[:-1] + args.R_bin__r[1:]) / 2.0
    args.N_R_bins = len(args.R_bin_center__r)
    return args


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


def print_output(outdata):
    from CALIFAUtils.scripts import get_NEDName_by_CALIFAID
    for k in classif_labels:
        for iR in range(outdata.N_R_bins):
            filename = '%s_%s_bin%02d.txt' % (outdata.califaID, k, iR)
            with open(filename, 'w') as f:
                f.write('# CALIFAID: %s (%s)\n' % (outdata.califaID, get_NEDName_by_CALIFAID(outdata.califaID)))
                f.write('# classif %s\n' % k)
                f.write('# lambda\tobs\tsyn\terr\tflag\tgood2badratio\n')
                for il in range(len(outdata.l_obs)):
                    f.write('%04.1f\t%e\t%e\t%e\t%d\t%.3f\n' % (outdata.l_obs[il],
                                                                outdata.O_rf__lR[k].data[il, iR],
                                                                outdata.M_rf__lR[k].data[il, iR],
                                                                outdata.err_rf__lR[k].data[il, iR],
                                                                outdata.b_rf__lR[k][il, iR],
                                                                outdata.bad_ratio__lR[k][il, iR]
                                                                )
                            )


def saveFITS(K, outdata, overwrite=False):
    from astropy.io import fits
    hdu = fits.HDUList()
    # PrimaryHDU - HEADER
    header = fits.Header()
    header.append(fits.Card('EMLFN', value=K.EL._hdulist.filename().split('/')[-1], comment='EML FITS FILENAME'))
    header.append(fits.Card('SUPERFN', value=K._hdulist.filename().split('/')[-1], comment='SUPERFITS FILENAME'))
    header['CALIFAID'] = str.strip(K.califaID)
    header['NRBINS'] = outdata.N_R_bins
    header['RBININI'] = outdata.rbinini
    header['RBINFIN'] = outdata.rbinfin
    header['RBINSTEP'] = outdata.rbinstep
    header['NY'] = K.N_y
    header['NX'] = K.N_x
    header['SFTH'] = outdata.sfth
    header['DIGTH'] = outdata.digth
    header.append(fits.Card(keyword='CLABELS', value=len(classif_labels), comment='%s' % classif_labels))
    hdu.append(fits.PrimaryHDU(header=header))
    # Other HDUs
    hdu.append(fits.ImageHDU(data=outdata.R_bin__r, name='R_bin__r'))
    hdu.append(fits.ImageHDU(data=outdata.R_bin_center__r, name='R_bin_center__r'))
    hdu.append(fits.ImageHDU(data=K.l_obs, name='l_obs'))
    hdu.append(fits.ImageHDU(data=outdata.W6563__z.data, name='W6563__z'))
    hdu.append(fits.ImageHDU(data=outdata.W6563__yx.data, name='W6563_data__yx'))
    hdu.append(fits.ImageHDU(data=outdata.W6563__yx.mask.astype('int'), name='W6563_mask__yx'))
    hdu.append(fits.ImageHDU(data=outdata.v_0, name='v_0__z'))
    hdu.append(fits.ImageHDU(data=outdata.Ntot__R, name='Ntot__R'))
    hdu.append(fits.ImageHDU(data=outdata.bin_segmap__Ryx.astype('int'), name='bin_segmap__Ryx'))
    for k in classif_labels:
        hdu.append(fits.ImageHDU(data=outdata.O_rf__lR[k], name='%s_O_rf__lR' % k))
        hdu.append(fits.ImageHDU(data=outdata.M_rf__lR[k], name='%s_M_rf__lR' % k))
        hdu.append(fits.ImageHDU(data=outdata.err_rf__lR[k], name='%s_err_rf__lR' % k))
        hdu.append(fits.ImageHDU(data=outdata.b_rf__lR[k], name='%s_b_rf__lR' % k))
        hdu.append(fits.ImageHDU(data=outdata.bad_ratio__lR[k], name='%s_bad_ratio__lR' % k))
        hdu.append(fits.ImageHDU(data=outdata.classNtot__R[k], name='%s_Ntot__R' % k))
        hdu.append(fits.ImageHDU(data=outdata.classbin_segmap__Ryx[k].astype('int'), name='%s_bin_segmap__Ryx' % k))
    hdu.writeto('%s-RadBinStackedSpectra.fits' % K.califaID, clobber=overwrite)


def readFITS(fitsfile):
    from pytu.objects import tupperware_none
    from astropy.io import fits
    data = tupperware_none()
    hdu = fits.open(fitsfile)
    data._hdu = hdu
    classif_labels = eval(hdu[0].header.comments['CLABELS'])
    data.l_obs = hdu['L_OBS'].data
    data.R_bin_center__R = hdu['R_BIN_CENTER__R'].data
    data.O_rf__clR = {c: hdu['%s_O_RF__LR' % c].data for c in classif_labels}
    data.M_rf__clR = {c: hdu['%s_M_RF__LR' % c].data for c in classif_labels}
    data.err_rf__clR = {c: hdu['%s_ERR_RF__LR' % c].data for c in classif_labels}
    data.b_rf__clR = {c: hdu['%s_B_RF__LR' % c].data for c in classif_labels}
    data.bad_ratio__clR = {c: hdu['%s_BAD_RATIO__LR' % c].data for c in classif_labels}
    data.Ntot__R = hdu['NTOT__R'].data
    data.classNtot__cR = {c: hdu['%s_NTOT__R' % c].data for c in classif_labels}
    data.W6563__z = hdu['W6563__Z'].data
    data.v_0__z = hdu['V_0__Z'].data
    data.W6563__yx = np.ma.masked_array(hdu['W6563_DATA__YX'].data, mask=hdu['W6563_MASK__YX'].data)
    data.bin_segmap__Ryx = np.array(hdu['BIN_SEGMAP__RYX'].data, dtype='bool')
    data.classbin_segmap__cRyx = {c: np.array(hdu['%s_BIN_SEGMAP__RYX' % c].data, dtype='bool') for c in classif_labels}
    return data


def create_outdata(args, K):
    outdata = tupperware_none()
    outdata.N_R_bins = args.N_R_bins
    outdata.R_bin__r = args.R_bin__r
    outdata.R_bin_center__r = args.R_bin_center__r
    outdata.rbinini = args.rbinini
    outdata.rbinfin = args.rbinfin
    outdata.rbinstep = args.rbinstep
    outdata.sfth = args.sfth
    outdata.digth = args.digth
    outdata.O_rf__lR = {}
    outdata.M_rf__lR = {}
    outdata.err_rf__lR = {}
    # flags
    outdata.b_rf__lR = {}
    outdata.bad_ratio__lR = {}
    outdata.classbin_segmap__Ryx = {}
    outdata.classNtot__R = {}
    for k in classif_labels:
        outdata.O_rf__lR[k] = np.zeros((K.Nl_obs, args.N_R_bins), dtype='float')
        outdata.M_rf__lR[k] = np.zeros((K.Nl_obs, args.N_R_bins), dtype='float')
        outdata.err_rf__lR[k] = np.zeros((K.Nl_obs, args.N_R_bins), dtype='float')
        outdata.b_rf__lR[k] = np.zeros((K.Nl_obs, args.N_R_bins), dtype='float')
        outdata.bad_ratio__lR[k] = np.zeros((K.Nl_obs, args.N_R_bins), dtype='float')
        outdata.classbin_segmap__Ryx[k] = np.zeros((args.N_R_bins, K.N_y, K.N_x), dtype='bool')
        outdata.classNtot__R[k] = np.zeros((args.N_R_bins), dtype='int')
    iHa = K.EL.lines.index('6563')
    outdata.Ntot__R = np.zeros((args.N_R_bins), dtype='int')
    outdata.W6563__z = K.EL.EW[iHa]
    outdata.W6563__yx = K.zoneToYX(outdata.W6563__z, extensive=False)
    outdata.bin_segmap__Ryx = np.zeros((args.N_R_bins, K.N_y, K.N_x), dtype='bool')
    from astropy import constants as C
    c = C.c.to('km/s').value
    v_0_Ha = c * (K.EL.pos[iHa] - 6563.)/6563.
    outdata.v_0 = np.where(K.EL._setMaskLineSNR('6563', 1), K.v_0, v_0_Ha)
    # outdata.v_0 = K.v_0
    return outdata


if __name__ == '__main__':
    args = parser_args()
    K = fitsQ3DataCube(args.superfits)
    K.loadEmLinesDataCube(args.emlfits)
    # Set geometry
    K.setGeometry(*K.getEllipseParams())
    wl_of = K.l_obs
    outdata = create_outdata(args, K)
    W6563__z = outdata.W6563__z
    # Loop in radial bins
    for iR, (l_edge, r_edge) in enumerate(zip(args.R_bin__r[0:-1], args.R_bin__r[1:])):
        sel_zones = np.bitwise_and(np.greater_equal(K.zoneDistance_HLR, l_edge), np.less(K.zoneDistance_HLR, r_edge))
        Nsel = sel_zones.astype('int').sum()
        outdata.Ntot__R[iR] = Nsel
        # Segmentation map
        segmap_tmp = K.zoneToYX(np.ma.masked_array(K.v_0, mask=~sel_zones), extensive=False)
        outdata.bin_segmap__Ryx[iR] = np.invert(segmap_tmp.mask)
        if Nsel == 0:  # don't do anything in empty bins
            continue
        # classification selections
        sel_classif = {}
        sel_classif['DIG'] = np.bitwise_and(sel_zones, np.less(W6563__z, args.digth))
        sel_classif['COMP'] = np.bitwise_and(sel_zones, np.bitwise_and(np.greater_equal(W6563__z, args.digth), np.less(W6563__z, args.sfth)))
        sel_classif['SF'] = np.bitwise_and(sel_zones, np.greater_equal(W6563__z, args.sfth))
        for tmp_cl in classif_labels:
            print Nsel, tmp_cl, sel_classif[tmp_cl].astype('int').sum()
        # Segmented maps with classification
        segmap_tmp = K.zoneToYX(np.ma.masked_array(K.v_0, mask=~sel_classif['DIG']), extensive=False)
        outdata.classbin_segmap__Ryx['DIG'][iR] = np.invert(segmap_tmp.mask)
        segmap_tmp = K.zoneToYX(np.ma.masked_array(K.v_0, mask=~sel_classif['COMP']), extensive=False)
        outdata.classbin_segmap__Ryx['COMP'][iR] = np.invert(segmap_tmp.mask)
        segmap_tmp = K.zoneToYX(np.ma.masked_array(K.v_0, mask=~sel_classif['SF']), extensive=False)
        outdata.classbin_segmap__Ryx['SF'][iR] = np.invert(segmap_tmp.mask)
        for k in classif_labels:
            # zone classification selection
            sel = sel_classif[k]
            N = sel.astype('int').sum()
            outdata.classNtot__R[k][iR] = N
            if N == 0:  # don't do anything in empty selections
                continue
            # stack spectra
            O_rf__l, M_rf__l, err_rf__l, b_rf__l, bad_ratio__l, bindata = stack_spectra(K, sel)  # , outdata.v_0[sel])
            # save bindata
            outdata.b_rf__lR[k][:, iR] = b_rf__l
            outdata.bad_ratio__lR[k][:, iR] = bad_ratio__l
            outdata.O_rf__lR[k][:, iR] = O_rf__l
            outdata.M_rf__lR[k][:, iR] = M_rf__l
            outdata.err_rf__lR[k][:, iR] = err_rf__l
    saveFITS(K, outdata, overwrite=True)


# def main_print_O3Hb_N2Ha_bins():
