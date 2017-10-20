from mpl_toolkits.axes_grid1 import make_axes_locatable
from CALIFAUtils.scripts import read_one_cube
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from pytu.plots import plot_text_ax
import numpy as np
import sys

latex_ppi = 72.0
latex_column_width_pt = 240.0
latex_column_width = latex_column_width_pt/latex_ppi
latex_text_width_pt = 504.0
latex_text_width = latex_text_width_pt/latex_ppi
golden_mean = 0.5 * (1. + 5**0.5)

kw_cube = dict(debug=True, EL=True, config=-1, elliptical=True)
_dpi_choice = 300
_transp_choice = False

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

if __name__ == '__main__':
    g = sys.argv[1]
    Kpix = read_one_cube(g, debug=True, EL=True, config=-1, elliptical=True)
    Kv20 = read_one_cube(g, debug=True, EL=True, config=-2, elliptical=True)
    iHa_pix = Kpix.EL.lines.index('6563')
    iHa_v20 = Kv20.EL.lines.index('6563')
    fHapixsum__z = np.zeros((Kv20.N_zone))
    f6563__zpix_aux = Kpix.EL.flux[iHa_pix]
    mask = np.bitwise_or(~np.isfinite(f6563__zpix_aux), np.less(f6563__zpix_aux, 1e-40))
    f6563__zpix = np.ma.masked_array(f6563__zpix_aux, mask=mask).filled(0.)
    for i in xrange(Kv20.N_y):
        for j in xrange(Kv20.N_x):
            if Kv20.qZones[i, j] >= 0:
                z_i = Kv20.qZones[i, j]
                z_i_pix = Kpix.qZones[i, j]
                fHapixsum__z[z_i] += f6563__zpix[z_i_pix]

    # f = plot_setup(latex_column_width, 1.)
    # ax = f.gca()
    # ax.plot(np.ma.log10(Kv20.EL.flux[iHa_v20]), np.ma.log10(fHapixsum__z), '.')
    # ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', lw=0.5)
    # ax.set_xlabel(r'$\log\ F_{H\alpha}^{(z)}$ [erg/s/cm${}^2/\AA$]')
    # ax.set_ylabel(r'$\log\ \sum\ F_{H\alpha}^{(pix \in z)}$ [erg/s/cm${}^2/\AA$]')
    # ax.set_title(Kpix.califaID)
    # ax.tick_params(axis='both', which='both', direction='in', bottom='on', left='on', top='off', right='off', labelbottom='on', labelleft='on', labeltop='off', labelright='off')
    # ax.grid()
    # f.tight_layout(rect=[0, 0.01, 1, 0.98])
    # f.savefig('%s_f6563_px1vsv20.pdf' % Kpix.califaID, dpi=300, transparent=False)
    # plt.close(f)

    f = plot_setup(latex_column_width, 1.)
    ax = f.gca()
    ax.plot(Kv20.EL.EW[iHa_v20], fHapixsum__z/Kv20.EL.flux[iHa_v20], '.')
    # ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', lw=0.5)
    ax.set_xlabel(r'$W_{H\alpha}$ [$\AA$]')
    ax.set_ylabel(r'$\sum\ F_{H\alpha}^{(pix \in z)} / F_{H\alpha}^{(z)}$')
    ax.set_title(Kpix.califaID)
    ax.tick_params(axis='both', which='both', direction='in', bottom='on', left='on', top='off', right='off', labelbottom='on', labelleft='on', labeltop='off', labelright='off')
    ax.grid()
    f.tight_layout(rect=[0, 0.01, 1, 0.98])
    f.savefig('%s_fratio_WHa_px1vsv20.pdf' % Kpix.califaID, dpi=300, transparent=False)
    plt.close(f)

    # K = read_one_cube('K0017', **kw_cube)
    # W6563__yx = K.zoneToYX(K.EL.EW[K.EL.lines.index('6563')], extensive=False)
    # Hb__yx = K.zoneToYX(np.ma.masked_array(K.EL.Hb_obs__z, mask=np.bitwise_or(~np.isfinite(K.EL.Hb_obs__z), np.less(K.EL.Hb_obs__z, 1e-40)))/K.zoneArea_pix, extensive=False)
    # eHb__yx = K.zoneToYX(np.ma.masked_array(K.EL.eflux[K.EL.lines.index('4861')], mask=np.bitwise_or(~np.isfinite(K.EL.eflux[K.EL.lines.index('4861')]), np.less(K.EL.eflux[K.EL.lines.index('4861')], 1e-40)))/K.zoneArea_pix, extensive=False)
    # SNRHb__yx = Hb__yx/eHb__yx
    # O3__yx = K.zoneToYX(np.ma.masked_array(K.EL.O3_obs__z, mask=np.bitwise_or(~np.isfinite(K.EL.O3_obs__z), np.less(K.EL.O3_obs__z, 1e-40)))/K.zoneArea_pix, extensive=False)
    # eO3__yx = K.zoneToYX(np.ma.masked_array(K.EL.eflux[K.EL.lines.index('5007')], mask=np.bitwise_or(~np.isfinite(K.EL.eflux[K.EL.lines.index('5007')]), np.less(K.EL.eflux[K.EL.lines.index('5007')], 1e-40)))/K.zoneArea_pix, extensive=False)
    # SNRO3__yx = O3__yx/eO3__yx
    # Ha__yx = K.zoneToYX(np.ma.masked_array(K.EL.Ha_obs__z, mask=np.bitwise_or(~np.isfinite(K.EL.Ha_obs__z), np.less(K.EL.Ha_obs__z, 1e-40)))/K.zoneArea_pix, extensive=False)
    # eHa__yx = K.zoneToYX(np.ma.masked_array(K.EL.eflux[K.EL.lines.index('6563')], mask=np.bitwise_or(~np.isfinite(K.EL.eflux[K.EL.lines.index('6563')]), np.less(K.EL.eflux[K.EL.lines.index('6563')], 1e-40)))/K.zoneArea_pix, extensive=False)
    # SNRHa__yx = Ha__yx/eHa__yx
    # N2__yx = K.zoneToYX(np.ma.masked_array(K.EL.N2_obs__z, mask=np.bitwise_or(~np.isfinite(K.EL.O3_obs__z), np.less(K.EL.O3_obs__z, 1e-40)))/K.zoneArea_pix, extensive=False)
    # eN2__yx = K.zoneToYX(np.ma.masked_array(K.EL.eflux[K.EL.lines.index('6583')], mask=np.bitwise_or(~np.isfinite(K.EL.eflux[K.EL.lines.index('6583')]), np.less(K.EL.eflux[K.EL.lines.index('6583')], 1e-40)))/K.zoneArea_pix, extensive=False)
    # SNRN2__yx = N2__yx/eN2__yx
    # # ratios
    # O3Hb__yx = np.ma.log10(O3__yx/Hb__yx)
    # eO3Hb__yx = O3Hb__yx * (((O3__yx/eO3__yx)**(-2.) + (Hb__yx/eHb__yx)**(-2)))**0.5
    # SNRO3Hb__yx = O3Hb__yx/eO3Hb__yx
    # N2Ha__yx = np.ma.log10(N2__yx/Ha__yx)
    # eN2Ha__yx = N2Ha__yx * (((N2__yx/eN2__yx)**(-2.) + (Ha__yx/eHa__yx)**(-2)))**0.5
    # SNRN2Ha__yx = N2Ha__yx/eN2Ha__yx
    # Ha__yx = K.zoneToYX(np.ma.masked_array(K.EL.Ha_obs__z, mask=np.bitwise_or(~np.isfinite(K.EL.Ha_obs__z), np.less(K.EL.Ha_obs__z, 1e-40)))/K.zoneArea_pix, extensive=False)
    # eHa__yx = K.zoneToYX(np.ma.masked_array(K.EL.eflux[K.EL.lines.index('6563')], mask=np.bitwise_or(~np.isfinite(K.EL.eflux[K.EL.lines.index('6563')]), np.less(K.EL.eflux[K.EL.lines.index('6563')], 1e-40)))/K.zoneArea_pix, extensive=False)
    # SNRHa__yx = Ha__yx/eHa__yx
    # 
    # f = plot_setup(8, 1./4)
    # gs = gridspec.GridSpec(1, 4)
    #
    # SNRfloor = 1
    # # sel = (SNRHb__yx > SNRfloor) & (SNRO3__yx > SNRfloor) & (SNRHa__yx > SNRfloor) & (SNRN2__yx > SNRfloor)
    # sel = K.qSn > 10
    #
    # ax = plt.subplot(gs[0])
    # ax.set_title('SDSS stamp', fontsize=10)
    # galimg = plt.imread('/Users/lacerda/califa/images/K0017.jpg')[::-1, :, :]
    # plt.setp(ax.get_xticklabels(), visible=False)
    # plt.setp(ax.get_yticklabels(), visible=False)
    # ax.imshow(galimg, origin='lower', aspect='equal')
    # txt = 'CALIFA 0017'
    # plot_text_ax(ax, txt, 0.02, 0.98, 10, 'top', 'left', color='w')
    # ax.xaxis.set_major_locator(MaxNLocator(4))
    # ax.xaxis.set_minor_locator(MaxNLocator(8))
    # ax.yaxis.set_major_locator(MaxNLocator(4))
    # ax.yaxis.set_minor_locator(MaxNLocator(8))
    # ax.tick_params(axis='both', which='both', direction='in', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    #
    # ax = plt.subplot(gs[1])
    # x = O3Hb__yx
    # # sel = SNRO3Hb__yx > SNRfloor
    # # sel = (SNRHb__yx > SNRfloor) & (SNRO3__yx > SNRfloor)
    # xm = np.ma.masked_array(x, mask=(np.ma.getmaskarray(x) | ~sel))
    # ax.set_title(r'$\log\ [OIII]/H\beta$')
    # im = ax.imshow(xm, cmap=plt.cm.copper, origin='lower', interpolation='nearest', aspect='equal', vmin=-1.2, vmax=1.2)
    # the_divider = make_axes_locatable(ax)
    # color_axis = the_divider.append_axes('right', size='5%', pad=0)
    # cb = plt.colorbar(im, cax=color_axis)
    # ax.xaxis.set_major_locator(MaxNLocator(4))
    # ax.xaxis.set_minor_locator(MaxNLocator(8))
    # ax.yaxis.set_major_locator(MaxNLocator(4))
    # ax.yaxis.set_minor_locator(MaxNLocator(8))
    # ax.tick_params(axis='both', which='both', direction='in', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    #
    # ax = plt.subplot(gs[2])
    # x = N2Ha__yx
    # # sel = SNRN2Ha__yx > SNRfloor
    # # sel = (SNRHa__yx > SNRfloor) & (SNRN2__yx > SNRfloor)
    # xm = np.ma.masked_array(x, mask=(np.ma.getmaskarray(x) | ~sel))
    # ax.set_title(r'$\log\ [NII]/H\alpha$')
    # im = ax.imshow(xm, cmap=plt.cm.copper, origin='lower', interpolation='nearest', aspect='equal', vmin=-1.5, vmax=0.7)
    # the_divider = make_axes_locatable(ax)
    # color_axis = the_divider.append_axes('right', size='5%', pad=0)
    # cb = plt.colorbar(im, cax=color_axis)
    # ax.xaxis.set_major_locator(MaxNLocator(4))
    # ax.xaxis.set_minor_locator(MaxNLocator(8))
    # ax.yaxis.set_major_locator(MaxNLocator(4))
    # ax.yaxis.set_minor_locator(MaxNLocator(8))
    # ax.tick_params(axis='both', which='both', direction='in', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    #
    # ax = plt.subplot(gs[3])
    # x = W6563__yx
    # # sel = SNRHa__yx > SNRfloor
    # xm = np.ma.masked_array(x, mask=(np.ma.getmaskarray(x) | ~sel))
    # im = ax.imshow(xm, cmap='Spectral', origin='lower', interpolation='nearest', aspect='equal', vmin=3, vmax=14)
    # the_divider = make_axes_locatable(ax)
    # color_axis = the_divider.append_axes('right', size='5%', pad=0)
    # cb = plt.colorbar(im, cax=color_axis)
    # ax.xaxis.set_major_locator(MaxNLocator(4))
    # ax.xaxis.set_minor_locator(MaxNLocator(8))
    # ax.yaxis.set_major_locator(MaxNLocator(4))
    # ax.yaxis.set_minor_locator(MaxNLocator(8))
    # ax.set_title(r'W${}_{H\alpha}$ [$\AA$]', fontsize=10)
    # ax.tick_params(axis='both', which='both', direction='in', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    #
    # f.tight_layout(rect=[0, 0.01, 1, 0.98])
    # f.savefig('fig_GS.pdf', dpi=_dpi_choice, transparent=_transp_choice)
    # plt.close(f)
