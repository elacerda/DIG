import sys
import ast
import numpy as np
import itertools
import matplotlib as mpl
from pytu.lines import Lines
from matplotlib import pyplot as plt
from astropy.io import fits
from pycasso.util import radialProfile
from pystarlight.util.constants import L_sun
from pytu.functions import ma_mask_xyz, debug_var
from pystarlight.util.redenninglaws import calc_redlaw
from CALIFAUtils.objects import stack_gals, CALIFAPaths
from mpl_toolkits.axes_grid1 import make_axes_locatable
from CALIFAUtils.plots import DrawHLRCircleInSDSSImage, DrawHLRCircle
from CALIFAUtils.scripts import get_NEDName_by_CALIFAID, spaxel_size_pc
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator, \
                              ScalarFormatter
from pytu.plots import cmap_discrete, plot_text_ax, density_contour, \
                       plot_scatter_histo, plot_histo_ax, stats_med12sigma, \
                       add_subplot_axes


mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
# with comp
colors_DIG_COMP_SF = ['brown', 'tomato', 'lightgreen', 'royalblue']
colors_lines_DIG_COMP_SF = ['maroon', 'darkred', 'olive', 'mediumblue']
classif_labels = ['HIG', 'DIG', 'COMP', 'SF']
colors_DIG_COMP_SF_dict = {'HIG':'brown', 'DIG':'tomato', 'COMP':'lightgreen', 'SF':'royalblue'}
colors_lines_DIG_COMP_SF_dict = {'HIG':'maroon', 'DIG':'darkred','COMP':'olive','SF':'mediumblue'}
# without
colors_DIG_COMP_SF = ['brown', 'tomato', 'royalblue']
colors_lines_DIG_COMP_SF = ['maroon', 'darkred', 'mediumblue']
classif_labels = ['HIG', 'DIG', 'COMP', 'SF']
colors_DIG_COMP_SF_dict = {'HIG':'brown', 'DIG':'tomato', 'SF':'royalblue'}
colors_lines_DIG_COMP_SF_dict = {'HIG':'maroon', 'DIG':'darkred', 'SF':'mediumblue'}

cmap_R = plt.cm.copper
minorLocator = AutoMinorLocator(5)
debug = False
transp_choice = False
dpi_choice = 100
# debug = True
# CCM reddening law
q = calc_redlaw([4861, 6563], R_V=3.1, redlaw='CCM')
f_tauVneb = lambda Ha, Hb: np.ma.log(Ha / Hb / 2.86) / (q[0] - q[1])
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
logSBHa_range = [4, 7]
logWHa_range = [0, 2.5]
DtauV_range = [-3, 3]
DeltatauV_range = [-1, 3]
DtauVnorm_range = [-1, 4]
x_Y_range = [0, 0.5]
OH_range = [8, 9.5]
# age to calc xY
tY = 100e6
minSNR = 0
config = -2
EL = True
elliptical = True
HIG_WHa_threshold = 3
DIG_WHa_threshold = 12
SF_WHa_threshold = 12
SF_Zhang_threshold = 1e39/L_sun
# DIG_Zhang_threshold = 10**38.5/L_sun
# lines = ['3727', '4363', '4861', '4959', '5007', '6300', '6563', '6583', '6717', '6731']
lines = ['3727', '4363', '4861', '4959', '5007', '6563', '6583', '6717', '6731']
rbinini = 0.
rbinfin = 3.
rbinstep = 0.2
R_bin__r = np.arange(rbinini, rbinfin + rbinstep, rbinstep)
R_bin_center__r = (R_bin__r[:-1] + R_bin__r[1:]) / 2.0
N_R_bins = len(R_bin_center__r)
kw_cube = dict(EL=EL, config=config, elliptical=elliptical)
dflt_kw_scatter = dict(marker='o', edgecolor='none')
dflt_kw_runstats = dict(smooth=True, sigma=1.2, frac=0.07, debug=True)  # , tendency=True)
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal')
P = CALIFAPaths()


def main(argv=None):
    sample_filename = sys.argv[1]
    ALL = stack_gals().load(sample_filename)

    try:
        # read gals file
        with open(sys.argv[2]) as f:
            gals = [line.strip() for line in f.xreadlines() if line.strip()[0] != '#']
    except IndexError:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)].tolist()

    try:
        sample_choice = sys.argv[3].split(':')
    except IndexError:
        sample_choice = ['SN_HaHb', 'S0']

    # print sample_choice
    # sys.exit()
    gals, sel, sample_choice = samples(ALL, sample_choice, gals)
    summary(ALL, sel, gals, 'SEL %s' % sample_choice)
    # sel3gals1 = ['K0010', 'K0813', 'K0187']
    sel3gals_paper = ['K0010', 'K0073', 'K0813']
    # sel3gals1 = ['K0010', 'K0813', 'K0836']
    """
    Fig 1.
        Example fig (maps, images, spectral fit and a zoom on EML on resid. spectra)
    """
    # fig1(ALL, gals)

    """
    Fig 2.
        Scatter and histogram of SBHa vs WHa.
        Histograms should be colored by WHa classif.
        Scatter could be colored by zone distance from center. (grayscale?)
        -- WHa_SBHa_zones_sample_histograms() from diffextin-experiences.py
    """
    # fig2(ALL, sel, gals)

    """
    Fig 3.
        Maps: SDSS stamp, SBHa, WHa, galaxy map colored by WHa classif.
        TODO: choose 3 example galaxies (Sa, Sb and Sc??)
    """
    # fig3(ALL, gals)  # gals=['K0010', 'K0187', 'K0813', 'K0388'])
    # fig3_3gals(ALL, sel, gals=['K0010', 'K0023', 'K0025'], suffix='1')
    # fig3_3gals(ALL, sel, gals=['K0028', 'K0031', 'K0034'], suffix='2')
    # fig3_3gals(ALL, sel, gals=['K0042', 'K0073', 'K0140'], suffix='3')
    # fig3_3gals(ALL, sel, gals=['K0179', 'K0183', 'K0190'], suffix='4')
    # fig3_3gals(ALL, sel, gals=['K0232', 'K0277', 'K0353'], suffix='5')
    # fig3_3gals(ALL, sel, gals=['K0388', 'K0414', 'K0436'], suffix='6')
    # fig3_3gals(ALL, sel, gals=['K0515', 'K0518', 'K0672'], suffix='7')
    # fig3_3gals(ALL, sel, gals=['K0684', 'K0707', 'K0748'], suffix='8')
    # fig3_3gals(ALL, sel, gals=['K0769', 'K0813', 'K0823'], suffix='9')
    # fig3_3gals(ALL, sel, gals=['K0836', 'K0891', 'K0915'], suffix='10')
    # fig3_3gals(ALL, sel, gals=['K0924', 'K0924', 'K0924'], suffix='11')
    # fig3_3gals(ALL, sel, gals=sel3gals2, suffix='2')
    # fig3_3gals(ALL, sel, gals=sel3gals_paper, suffix='paper')

    """
    Fig 4.
        BPT diagrams: [OIII]/Hb vs (panel a:[NII]/Ha, panel b:[SII]/Ha, panel c:[OI]/Ha)
        Should be those same example galaxies from Fig. 3.
    """
    #  fig4(ALL, gals)  # gals=['K0010', 'K0187', 'K0813', 'K0388'])
    # fig4_3gals(ALL, sel, gals=sel3gals1, suffix='1')
    # fig4_3gals(ALL, sel, gals=sel3gals2, suffix='2')
    # fig4_3gals(ALL, sel, gals=['K0010', 'K0023', 'K0025'], suffix='1')
    # fig4_3gals(ALL, sel, gals=['K0028', 'K0031', 'K0034'], suffix='2')
    # fig4_3gals(ALL, sel, gals=['K0042', 'K0073', 'K0140'], suffix='3')
    # fig4_3gals(ALL, sel, gals=['K0179', 'K0183', 'K0190'], suffix='4')
    # fig4_3gals(ALL, sel, gals=['K0232', 'K0277', 'K0353'], suffix='5')
    # fig4_3gals(ALL, sel, gals=['K0388', 'K0414', 'K0436'], suffix='6')
    # fig4_3gals(ALL, sel, gals=['K0515', 'K0518', 'K0672'], suffix='7')
    # fig4_3gals(ALL, sel, gals=['K0684', 'K0707', 'K0748'], suffix='8')
    # fig4_3gals(ALL, sel, gals=['K0769', 'K0813', 'K0823'], suffix='9')
    # fig4_3gals(ALL, sel, gals=['K0836', 'K0891', 'K0915'], suffix='10')
    # fig4_3gals(ALL, sel, gals=['K0924', 'K0924', 'K0924'], suffix='11')
    # fig4_3gals(ALL, sel, gals=sel3gals_paper, suffix='paper')


    """
    Fig 5.
        panel a: [OIII]/Hb vs [NII]/Ha for entire sample
        panel b: [OIII]/Hb vs [NII]/Ha 2d histogram painting 2d cell choosing
            the color by the bootstrap classif. stats.
    """
    # fig5(ALL, gals)
    # fig5_3panels(ALL, sel, gals)

    """
    Fig 6.
        Ha/Hb (or tau_V_neb) vs R
        Should be those same example galaxies from Fig. 3.
    """
    # fig6(ALL, gals)  #  gals=['K0010', 'K0187', 'K0813', 'K0388'])
    # fig6_3gals(ALL, sel, gals=sel3gals1, suffix='1')
    # fig6_3gals(ALL, sel, gals=sel3gals2, suffix='2')
    # fig6_3gals(ALL, sel, gals=['K0010', 'K0023', 'K0025'], suffix='1')
    # fig6_3gals(ALL, sel, gals=['K0028', 'K0031', 'K0034'], suffix='2')
    # fig6_3gals(ALL, sel, gals=['K0042', 'K0073', 'K0140'], suffix='3')
    # fig6_3gals(ALL, sel, gals=['K0179', 'K0183', 'K0190'], suffix='4')
    # fig6_3gals(ALL, sel, gals=['K0232', 'K0277', 'K0353'], suffix='5')
    # fig6_3gals(ALL, sel, gals=['K0388', 'K0414', 'K0436'], suffix='6')
    # fig6_3gals(ALL, sel, gals=['K0515', 'K0518', 'K0672'], suffix='7')
    # fig6_3gals(ALL, sel, gals=['K0684', 'K0707', 'K0748'], suffix='8')
    # fig6_3gals(ALL, sel, gals=['K0769', 'K0813', 'K0823'], suffix='9')
    # fig6_3gals(ALL, sel, gals=['K0836', 'K0891', 'K0915'], suffix='10')
    # fig6_3gals(ALL, sel, gals=sel3gals_paper, suffix='paper')

    """
    Fig 7.
        panel a: Histogram of Ha/Hb (or tau_V_neb)
        panel b: Histogram of D_tau_classif (tau_SF - tau_DIG)
        panel c: Histogram of D_tau_classif (tau_SF - tau_DIG)/integrated_tau_V_neb
        All sample.
    """
    # fig7(ALL, sel, gals)

    """
    Fig 8.
        Histogram of D_tau (tau_V_neb - tau_V)
        All sample.
        -- histograms_HaHb_Dt() from diffextin-experiences.py
    """
    # fig8(ALL, sel, gals)

    """
    Fig 9.
        D_tau (tau_V_neb - tau_V) vs x_Y
        All sample.
        -- Dt_xY_profile_sample() from diffextin-experiences.py
    """
    # fig9(ALL, sel, gals)

    # fig_tauVNeb_WHaSBHa(ALL, sel, gals)
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, sel3gals1, suffix='1')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, sel3gals2, suffix='2')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=['K0010', 'K0023', 'K0025'], suffix='1')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=['K0028', 'K0031', 'K0034'], suffix='2')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=['K0042', 'K0073', 'K0140'], suffix='3')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=['K0179', 'K0183', 'K0190'], suffix='4')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=['K0232', 'K0277', 'K0353'], suffix='5')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=['K0388', 'K0414', 'K0436'], suffix='6')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=['K0515', 'K0518', 'K0672'], suffix='7')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=['K0684', 'K0707', 'K0748'], suffix='8')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=['K0769', 'K0813', 'K0823'], suffix='9')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=['K0836', 'K0891', 'K0915'], suffix='10')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=['K0924', 'K0924', 'K0924'], suffix='11')
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=sel3gals_paper, suffix='paper')

    #
    # fig_WHaSBHa_per_morftype(sample_choice)
    # # fig_WHaSBHa_profile_3gals(ALL, sel, sel3gals1, suffix='1')
    # # fig_WHaSBHa_profile_3gals(ALL, sel, sel3gals2, suffix='2')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=['K0010', 'K0023', 'K0025'], suffix='1')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=['K0028', 'K0031', 'K0034'], suffix='2')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=['K0042', 'K0073', 'K0140'], suffix='3')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=['K0179', 'K0183', 'K0190'], suffix='4')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=['K0232', 'K0277', 'K0353'], suffix='5')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=['K0388', 'K0414', 'K0436'], suffix='6')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=['K0515', 'K0518', 'K0672'], suffix='7')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=['K0684', 'K0707', 'K0748'], suffix='8')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=['K0769', 'K0813', 'K0823'], suffix='9')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=['K0836', 'K0891', 'K0915'], suffix='10')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=['K0924', 'K0924', 'K0924'], suffix='11')
    #
    # fig_compare_stacked_tauVNeb(ALL, sel, ['K0010', 'K0023', 'K0025'], suffix='1')
    #
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=['K0010', 'K0023', 'K0025'], suffix='1')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=['K0028', 'K0031', 'K0034'], suffix='2')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=['K0042', 'K0073', 'K0140'], suffix='3')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=['K0179', 'K0183', 'K0190'], suffix='4')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=['K0232', 'K0277', 'K0353'], suffix='5')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=['K0388', 'K0414', 'K0436'], suffix='6')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=['K0515', 'K0518', 'K0672'], suffix='7')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=['K0684', 'K0707', 'K0748'], suffix='8')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=['K0769', 'K0813', 'K0823'], suffix='9')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=['K0836', 'K0891', 'K0915'], suffix='10')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=['K0924', 'K0924', 'K0924'], suffix='11')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=sel3gals_paper, suffix='paper')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=sel3gals_paper, suffix='paper')
    # fig_compare_stacked_lines(ALL, sel, gals, sample_choice)
    # fig_stacked_statsalldata_histograms(ALL, sel, gals, sample_choice)
    # fig_stacks_BPT(gals, sample_choice)
    #fig7_stacks(gals, sample_choice)

    #######################################
    ################ PAPER ################
    #######################################
    # fig2(ALL, sel, gals)
    # fig3(ALL, sel, gals)
    # fig3_3gals(ALL, sel, gals=sel3gals_paper, suffix='paper')
    # fig4(ALL, sel, gals)
    # fig4_3gals(ALL, sel, gals=sel3gals_paper, suffix='paper')
    # fig5_3panels(ALL, sel, gals)
    # fig6(ALL, sel, gals)
    # fig6_3gals(ALL, sel, gals=sel3gals_paper, suffix='paper')
    # fig7(ALL, sel, gals)
    # fig8(ALL, sel, gals)
    # fig9(ALL, sel, gals)
    # fig_WHaSBHa_per_morftype(sample_choice)
    # fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals=sel3gals_paper, suffix='paper')
    # fig_WHaSBHa_profile_3gals(ALL, sel, gals=sel3gals_paper, suffix='paper')
    # fig_compare_stacked_tauVNeb(ALL, sel, gals=sel3gals_paper, suffix='paper')
    # fig_compare_stacked_lines(ALL, sel, gals, sample_choice)
    # fig_stacks_BPT(gals, sample_choice)
    # fig_HbHa_vs_SBHa_WHa(ALL, sel, gals)
    # fig_WHaSBHa_per_morftype_and_ba(sample_choice)
    fig3(ALL, sel, gals)


#############################################################################
## Emission Lines cube reader
#############################################################################
class EmLinesStackedDataCube(object):
    def __init__(self, emLinesFile=None, distance_Mpc=None):
        self._hdulist = fits.open(emLinesFile)
        self._qCCM = {
            '4861': 1.16427,
            '5007': 1.12022,
            '6563': 0.81775,
            '6583': 0.81466,
        }

        try:
            self._extensions = self._hdulist[2].data.names
        except AttributeError:
            raise Exception('This file is not a emission lines FITS.')
        except IndexError:
            raise Exception('This emission lines FITS appears to be broke.')

        self.lines = ast.literal_eval(self._getPrimaryHeaderExt('FLINES'))
        self.N_line = len(self._hdulist) - 2
        self.N_zone = self._hdulist['FITTEDID'].header['NAXIS2']

        # NOTE: It appears that this test below works for all files, but we
        # just have to keep in mind that in the future maybe this can fail.
        self.califaID = self._getPrimaryHeaderExt('FILENAME').split('/')[-1].split('_')[0]
        self.distance_Mpc = distance_Mpc

    def _getPrimaryHeaderExt(self, ext):
        try:
            return self._hdulist['PRIMARY'].header[ext]
        except AttributeError:
            raise AttributeError('The header does not have a %s extension.' % ext)

    def close(self):
        self._hdulist.close()
        self._hdulist = None

    def intrinsic_flux__z(self, line, ext_law=None, tau_V=None):
        if tau_V is None:
            tau_V = self.tau_V_neb__z
        if isinstance(line, str):
            flux = self._getLineFlux(line)
            if ext_law is None:
                ext_law = self._qCCM[line]
        else:
            flux = line
        return flux * np.ma.exp(ext_law * tau_V)

    def intrinsic_integrated_flux(self, line, ext_law=None, tau_V=None):
        if tau_V is None:
            tau_V = self.integrated_tau_V_neb
        if isinstance(line, str):
            flux = self._getIntegratedLineFlux(line)
            if ext_law is None:
                ext_law = self._qCCM[line]
        else:
            flux = line
        return flux * np.ma.exp(ext_law * tau_V)

    def _setMaskLineFluxNeg(self, line):
        '''
        Returns an array of booleans with value True if the measured zone
        line flux is negative.
        '''
        hdu = self._getLineHdu(line)
        arr = np.array(object=hdu.data['flux'], dtype=hdu.data.dtype['flux'])
        arr = np.ma.masked_array(arr, mask=~np.isfinite(arr))
        mask = (arr <= 0).filled(True)
        return mask

    def _setMaskLineSigma(self, line, maxsigma=3.5):
        hdu = self._getLineHdu(line)
        arr = np.array(object=hdu.data['sigma'], dtype=hdu.data.dtype['sigma'])
        arr = np.ma.masked_array(arr, mask=~np.isfinite(arr))
        mask = (arr > maxsigma).filled(True)
        return mask

    def _setMaskLineDisplacement(self, line, displacement=2.5):
        hdu = self._getLineHdu(line)
        arr = np.array(object=hdu.data['pos'], dtype=hdu.data.dtype['pos'])
        arr = np.ma.masked_array(arr, mask=~np.isfinite(arr))
        mask = (np.ma.abs(arr - float(line)) > displacement).filled(True)
        return mask

    def _setMaskLineSNR(self, line, minsnr=3.):
        hdu = self._getLineHdu(line)
        arr_flux = np.array(object=hdu.data['flux'], dtype=hdu.data.dtype['flux'])
        arr_eflux = np.array(object=hdu.data['eflux'], dtype=hdu.data.dtype['eflux'])
        arr = arr_flux/arr_eflux
        arr = np.ma.masked_array(arr, mask=~np.isfinite(arr))
        mask = (arr < minsnr).filled(True)
        return mask

    def _getLineHdu(self, line):
        try:
            return self._hdulist[line]
        except KeyError:
            raise KeyError('This fits does not have %s emission line.' % line)

    def _getExtMasked(self, ext, mask=None):
        arr = np.ma.zeros((self.N_line, self.N_zone))
        getMask = False

        if mask is None:
            getMask = True

        for i, line in enumerate(self.lines):
            hdu = self._getLineHdu(line)
            arr__z = np.copy(hdu.data[ext])
            if getMask:
                mask = ~np.isfinite(arr__z)
                # mask = self._setMaskFluxNeg(line)
                mask = np.zeros((self.N_zone), dtype=np.bool)
            else:
                mask = mask
            arr[i] = np.ma.masked_array(arr__z, mask=mask, dtype=hdu.data.dtype[ext])
        return arr

    def _F_to_L(self, flux, distance_Mpc=None):
        ''' Using distance, converts flux [erg cm^(-2) s^(-1)] to
            luminosity in [erg s^(-1)]. '''
        if not distance_Mpc:
            if not self.distance_Mpc:
                logger.warn('distance_Mpc: You have to set the distance.')
            else:
                distance_Mpc = self.distance_Mpc

        if distance_Mpc:
            # google: 1 Mpc = 3.08567758e24 cm
            Mpc_in_cm = 3.08567758e24  # cm
            solidAngle = 4. * np.pi * (distance_Mpc * Mpc_in_cm) ** 2.0
            lum = solidAngle * flux
        else:
            lum = None

        return lum

    def _getLineFlux(self, line_str):
        return self.flux[self.lines.index(line_str), :]

    def _getIntegratedLineFlux(self, line_str):
        return self.integrated_flux[self.lines.index(line_str)]

    @property
    def Ha_obs__z(self):
        return self._getLineFlux('6563')

    @property
    def Hb_obs__z(self):
        return self._getLineFlux('4861')

    @property
    def N2_obs__z(self):
        return self._getLineFlux('6583')

    @property
    def O3_obs__z(self):
        return self._getLineFlux('5007')

    @property
    def Ha_int__z(self):
        line = '6563'
        return self.intrinsic_flux__z(line=line, ext_law=self._qCCM[line])

    @property
    def Hb_int__z(self):
        line = '4861'
        return self.intrinsic_flux__z(line=line, ext_law=self._qCCM[line])

    @property
    def N2_int__z(self):
        line = '6583'
        return self.intrinsic_flux__z(line=line, ext_law=self._qCCM[line])

    @property
    def O3_int__z(self):
        line = '5007'
        return self.intrinsic_flux__z(line=line, ext_law=self._qCCM[line])

    @property
    def HaHb__z(self):
        return self.Ha_obs__z / self.Hb_obs__z

    @property
    def HaHb_err__z(self):
        i_Ha = self.lines.index('6563')
        i_Hb = self.lines.index('4861')
        a = self.eflux[i_Ha, :]
        b = (self.Ha_obs__z / self.Hb_obs__z) * self.eflux[i_Hb, :]
        return np.ma.sqrt(a ** 2. + b ** 2.) / self.Hb_obs__z

    @property
    def tau_V_neb__z(self):
        '''
        Balmer optical depth.

        Assumes that the ratio of intrinsic flux_Ha/flux_Hb is 2.86

        Lacerda@Saco - 25/Jun/2014

        * Units: Unitless
        * Shape: ``(N_zone)``
        '''
        return np.ma.log(self.HaHb__z / 2.86) / (self._qCCM['4861'] - self._qCCM['6563'])

    @property
    def tau_V_neb_err__z(self):
        '''
        Error propagation in Balmer optical depth calculation.

        Uses:

        Lacerda@Saco - 25/Jun/2014
        '''
        i_Ha = self.lines.index('6563')
        i_Hb = self.lines.index('4861')
        q = self._qCCM['4861'] - self._qCCM['6563']
        a = self.eflux[i_Ha, :] / self.Ha_obs__z
        b = self.eflux[i_Hb, :] / self.Hb_obs__z
        return np.sqrt(a ** 2.0 + b ** 2.0) / q

    @property
    def O3N2__z(self):
        '''
        Similar to O3N2 ratio but includes Ha and Hb
        The fluxes are dust corrected using Balmer optical depth.
        O3N2 = ([OIII]5007 * Ha) / ([NII]6583 * Hb)
        '''
        O3Hb = self.O3_int__z / self.Hb_int__z
        N2Ha = self.N2_int__z / self.Ha_int__z
        return O3Hb/N2Ha

    ### TODO: O3N2 err propagation
    #@property
    #def O3N2_err__z(self):

    @property
    def O3N2_ratio__z(self):
        '''
        O3N2 ratio.
        The fluxes are dust corrected using Balmer optical depth.
        O3N2_r = ([OIII]5007) / ([NII]6583)
        '''
        O3cor__z = self.O3_int__z
        N2cor__z = self.N2_int__z
        return O3cor__z / N2cor__z

    @property
    def O3N2_ratio_err__z(self):
        '''
        Error propagation in O3N2 ratio.
        '''
        i_Hb = self.lines.index('4861')
        i_O3 = self.lines.index('5007')
        i_Ha = self.lines.index('6563')
        i_N2 = self.lines.index('6583')
        e = np.ma.exp(self.tau_V_neb__z * (self._qCCM['5007'] - self._qCCM['6583'])) / self.flux[i_N2, :]
        a = self.eflux[i_O3, :]
        b = (self.eflux[i_O3, :] / self.flux[i_N2, :]) * self.eflux[i_N2, :]
        q = (self._qCCM['5007'] - self._qCCM['6583']) / (self._qCCM['4861'] - self._qCCM['6563'])
        c = (self.flux[i_O3, :] / self.flux[i_Ha, :]) * self.eflux[i_Ha, :]
        d = (self.flux[i_O3, :] / self.flux[i_Hb, :]) * self.eflux[i_Hb, :]
        return e * np.ma.sqrt(a ** 2. + b ** 2. + q ** 2. * (c ** 2.0 + d ** 2.))

    @property
    def logZ_neb_S06__z(self):
        '''
        Calculates log(Z_neb) using Asari et al (2007) with
        [OIII]5007 / [NII]6583 calibrated by Stasinska (2006).
        log(Z_neb) = log((O/H) / (O/H)_Sol) = - 0.14 - 0.25 log([OIII]5007 / [NII]6583)
        '''
        return - 0.14 - (0.25 * np.ma.log10(self.O3N2_ratio__z))

    @property
    def logZ_neb_S06_err__z(self):
        '''
        Error propagation in logZ_neb.
        '''
        return 0.25 * self.O3N2_ratio_err__z / (np.log(10.) * self.O3N2_ratio__z)

    @property
    def Zneb_M13__z(self):
        '''
        The 12 + log(O/H) from Marino et al (2013)
        8.533 - 0.214 * ([OIII]5007 * Ha) / ([NII]6583 * Hb)

        '''
        return 8.533 - 0.214 * np.ma.log10(self.O3N2__z)

    ### TODO: M13 Zneb error propagation

    @property
    def Zneb_PP04__z(self):
        '''
        The 12 + log(O/H) from Pettini & Pagel (2004)
        8.73 - 0.32 * ([OIII]5007 * Ha) / ([NII]6583 * Hb)

        '''
        return 8.73 - 0.32 * np.ma.log10(self.O3N2__z)

    @property
    def __version__(self):
        return self._getPrimaryHeaderExt('VERSION')

    @property
    def pos(self):
        '''
        Central wavelength.

        * Units: Angstroms
        * Shape: ``(N_line, N_zone)``
        '''
        return self._getExtMasked('pos')

    @property
    def posmean(self):
        '''
        Mean central wavelength..

        * Units: Angstroms
        * Shape: ``(N_line, N_zone)``
        '''
        return self._getExtMasked('posmean')

    @property
    def posmedian(self):
        '''
        Median central wavelength.

        * Units: Angstroms
        * Shape: ``(N_line, N_zone)``
        '''
        return self._getExtMasked('posmedian')

    @property
    def epos(self):
        '''
        Error in central wavelength measurement.

        * Units: Angstroms
        * Shape: ``(N_line, N_zone)``
        '''
        return self._getExtMasked('epos')

    # flux: total line flux
    @property
    def flux(self):
        return self._getExtMasked('flux')

    @property
    def fluxmean(self):
        return self._getExtMasked('fluxmean')

    @property
    def fluxmedian(self):
        return self._getExtMasked('fluxmedian')

    @property
    def eflux(self):
        return self._getExtMasked('eflux')

    @property
    def fluxStdError(self):
        return self._getExtMasked('fluxStdError')

    # ampl: peak amplitude
    @property
    def ampl(self):
        '''
        Amplitude of the line peak.

        * Units: :math:`[erg / s / cm^2 / \overset{\circ}{A}]`
        * Shape: ``(N_line, N_zone)``
        '''
        return self._getExtMasked('ampl')

    @property
    def amplmean(self):
        '''
        Mean amplitude of the line peak.

        * Units: :math:`[erg / s / cm^2 / \overset{\circ}{A}]`
        * Shape: ``(N_line, N_zone)``
        '''
        return self._getExtMasked('amplmean')

    @property
    def amplmedian(self):
        '''
        Median amplitude of the line peak.

        * Units: :math:`[erg / s / cm^2 / \overset{\circ}{A}]`
        * Shape: ``(N_line, N_zone)``
        '''
        return self._getExtMasked('amplmedian')

    @property
    def eampl(self):
        '''
        Error in amplitude of the line peak.

        * Units: :math:`[erg / s / cm^2 / \overset{\circ}{A}]`
        * Shape: ``(N_line, N_zone)``
        '''
        return self._getExtMasked('eampl')

    @property
    # fwhm: Full width at half maximum
    def fwhm(self):
        return self._getExtMasked('fwhm')

    @property
    def fwhmmean(self):
        return self._getExtMasked('fwhmmean')

    @property
    def fwhmmedian(self):
        return self._getExtMasked('fwhmmedian')

    @property
    def efwhm(self):
        return self._getExtMasked('efwhm')

    @property
    # sigma: standard deviation
    def sigma(self):
        return self._getExtMasked('sigma')

    @property
    def sigmamean(self):
        return self._getExtMasked('sigmamean')

    @property
    def esigma(self):
        return self._getExtMasked('esigma')

    @property
    # EW: Equivalent Width (flux / baseline)
    def EW(self):
        return self._getExtMasked('EW')

    # baseline: pseudo-continuum flux
    @property
    def baseline(self):
        return self._getExtMasked('baseline')

    @property
    def baseline_orig(self):
        return self._getExtMasked('baseline_orig')

    @property
    def baselineStd(self):
        return self._getExtMasked('baselineStd')

    @property
    def ebaseline(self):
        return self._getExtMasked('ebaseline')

    # c0 and c1: Line adjust polynomial constants
    @property
    def c0(self):
        return self._getExtMasked('c0')

    @property
    def c0mean(self):
        return self._getExtMasked('c0mean')

    @property
    def c0median(self):
        return self._getExtMasked('c0median')

    @property
    def ec0(self):
        return self._getExtMasked('ec0')

    @property
    def c1(self):
        return self._getExtMasked('c1')

    @property
    def c1mean(self):
        return self._getExtMasked('c1mean')

    @property
    def c1median(self):
        return self._getExtMasked('c1median')

    @property
    def ec1(self):
        return self._getExtMasked('ec1')

    @property
    def statval(self):
        return self._getExtMasked('statval')

    @property
    def dof(self):
        return self._getExtMasked('dof')

    @property
    def integrated_pos(self):
        return self._getIntegratedExt('pos')

    @property
    def snr__Lz(self):
        s = self.flux
        n = self.eflux
        return s/n


def samples(ALL, sample_choice, gals=None):
    sel_WHa = dict(
        HIG=dict(
            z=(ALL.W6563__z < HIG_WHa_threshold).filled(False),
            yx=(ALL.W6563__yx < HIG_WHa_threshold).filled(False)
        ),
        DIG=dict(
            z=np.bitwise_and((ALL.W6563__z >= HIG_WHa_threshold).filled(
                False), (ALL.W6563__z < DIG_WHa_threshold).filled(False)),
            yx=np.bitwise_and((ALL.W6563__yx >= HIG_WHa_threshold).filled(
                False), (ALL.W6563__yx < DIG_WHa_threshold).filled(False))
            # z=(ALL.W6563__z < DIG_WHa_threshold).filled(False),
            # yx=(ALL.W6563__yx < DIG_WHa_threshold).filled(False)
        ),
        COMP=dict(
            z=np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(
                False), (ALL.W6563__z < SF_WHa_threshold).filled(False)),
            yx=np.bitwise_and((ALL.W6563__yx >= DIG_WHa_threshold).filled(
                False), (ALL.W6563__yx < SF_WHa_threshold).filled(False))
        ),
        SF=dict(
            z=(ALL.W6563__z >= SF_WHa_threshold).filled(False),
            yx=(ALL.W6563__yx >= SF_WHa_threshold).filled(False)
        ),
    )

    sel_Zhang = dict(
        DIG=dict(
            z=(ALL.SB6563__z < SF_Zhang_threshold).filled(False),
            yx=(ALL.SB6563__yx < SF_Zhang_threshold).filled(False)
        ),
        SF=dict(
            z=(ALL.SB6563__z >= SF_Zhang_threshold).filled(False),
            yx=(ALL.SB6563__yx >= SF_Zhang_threshold).filled(False)
        ),
    )

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
        S06=dict(z=sel_below_S06__z, yx=sel_below_S06__yx),
        K03=dict(z=sel_below_K03__z, yx=sel_below_K03__yx),
        K01=dict(z=sel_below_K01__z, yx=sel_below_K01__yx),
        betS06K03=dict(z=sel_between_S06K03__z, yx=sel_between_S06K03__yx),
        betK03K01=dict(z=sel_between_K03K01__z, yx=sel_between_K03K01__yx),
        aboK01=dict(z=sel_above_K01__z, yx=sel_above_K01__yx),
    )

    f__lgz = {'%s' % l: getattr(ALL, 'f%s__z' % l) for l in lines}
    ef__lgz = {'%s' % l: getattr(ALL, 'ef%s__z' % l) for l in lines}
    SN__lgz = {'%s' % l: f__lgz[l]/ef__lgz[l] for l in lines}
    f__lgyx = {'%s' % l: getattr(ALL, 'f%s__yx' % l) for l in lines}
    ef__lgyx = {'%s' % l: getattr(ALL, 'ef%s__yx' % l) for l in lines}
    SN__lgyx = {'%s' % l: f__lgyx[l]/ef__lgyx[l] for l in lines}

    sel_SN = dict(
        S1=dict(
            lz={'%s' % l: np.greater(SN__lgz[l].filled(0.), 1) for l in lines},
            lyx={'%s' % l: np.greater(SN__lgyx[l].filled(0.), 1) for l in lines},
        ),
        S3=dict(
            lz={'%s' % l: np.greater(SN__lgz[l].filled(0.), 3) for l in lines},
            lyx={'%s' % l: np.greater(SN__lgyx[l].filled(0.), 3) for l in lines},
        ),
        S5=dict(
            lz={'%s' % l: np.greater(SN__lgz[l].filled(0.), 5) for l in lines},
            lyx={'%s' % l: np.greater(SN__lgyx[l].filled(0.), 5) for l in lines},
        ),
        S10=dict(
            lz={'%s' % l: np.greater(SN__lgz[l].filled(0.), 10) for l in lines},
            lyx={'%s' % l: np.greater(SN__lgyx[l].filled(0.), 10) for l in lines},
        ),
    )

    sel_SN1__gz = np.bitwise_and(sel_SN['S1']['lz']['4861'], np.bitwise_and(sel_SN['S1']['lz']['5007'], np.bitwise_and(sel_SN['S1']['lz']['6563'], sel_SN['S1']['lz']['6583'])))
    sel_SN1__gyx = np.bitwise_and(sel_SN['S1']['lyx']['4861'], np.bitwise_and(sel_SN['S1']['lyx']['5007'], np.bitwise_and(sel_SN['S1']['lyx']['6563'], sel_SN['S1']['lyx']['6583'])))
    sel_SN3__gz = np.bitwise_and(sel_SN['S3']['lz']['4861'], np.bitwise_and(sel_SN['S3']['lz']['5007'], np.bitwise_and(sel_SN['S3']['lz']['6563'], sel_SN['S3']['lz']['6583'])))
    sel_SN3__gyx = np.bitwise_and(sel_SN['S3']['lyx']['4861'], np.bitwise_and(sel_SN['S3']['lyx']['5007'], np.bitwise_and(sel_SN['S3']['lyx']['6563'], sel_SN['S3']['lyx']['6583'])))
    sel_SN5__gz = np.bitwise_and(sel_SN['S5']['lz']['4861'], np.bitwise_and(sel_SN['S5']['lz']['5007'], np.bitwise_and(sel_SN['S5']['lz']['6563'], sel_SN['S5']['lz']['6583'])))
    sel_SN5__gyx = np.bitwise_and(sel_SN['S5']['lyx']['4861'], np.bitwise_and(sel_SN['S5']['lyx']['5007'], np.bitwise_and(sel_SN['S5']['lyx']['6563'], sel_SN['S5']['lyx']['6583'])))
    sel_SN10__gz = np.bitwise_and(sel_SN['S5']['lz']['4861'], np.bitwise_and(sel_SN['S5']['lz']['5007'], np.bitwise_and(sel_SN['S5']['lz']['6563'], sel_SN['S5']['lz']['6583'])))
    sel_SN10__gyx = np.bitwise_and(sel_SN['S10']['lyx']['4861'], np.bitwise_and(sel_SN['S10']['lyx']['5007'], np.bitwise_and(sel_SN['S10']['lyx']['6563'], sel_SN['S10']['lyx']['6583'])))

    sel_SN_BPT = dict(
        S0=dict(z=np.ones((ALL.califaID__z.shape), dtype='bool'), yx=np.ones((ALL.califaID__yx.shape), dtype='bool')),
        S1=dict(z=sel_SN1__gz, yx=sel_SN1__gyx),
        S3=dict(z=sel_SN3__gz, yx=sel_SN3__gyx),
        S5=dict(z=sel_SN5__gz, yx=sel_SN5__gyx),
        S10=dict(z=sel_SN10__gz, yx=sel_SN10__gyx),
    )

    sel_SN1__gz = np.bitwise_and(sel_SN['S1']['lz']['4861'], sel_SN['S1']['lz']['6583'])
    sel_SN1__gyx = np.bitwise_and(sel_SN['S1']['lyx']['4861'], sel_SN['S1']['lyx']['6583'])
    sel_SN3__gz = np.bitwise_and(sel_SN['S3']['lz']['4861'], sel_SN['S3']['lz']['6583'])
    sel_SN3__gyx = np.bitwise_and(sel_SN['S3']['lyx']['4861'], sel_SN['S3']['lyx']['6583'])
    sel_SN5__gz = np.bitwise_and(sel_SN['S5']['lz']['4861'], sel_SN['S5']['lz']['6583'])
    sel_SN5__gyx = np.bitwise_and(sel_SN['S5']['lyx']['4861'], sel_SN['S5']['lyx']['6583'])
    sel_SN10__gz = np.bitwise_and(sel_SN['S5']['lz']['4861'], sel_SN['S5']['lz']['6583'])
    sel_SN10__gyx = np.bitwise_and(sel_SN['S10']['lyx']['4861'], sel_SN['S10']['lyx']['6583'])

    sel_SN_HaHb = dict(
        S0=dict(z=np.ones((ALL.califaID__z.shape), dtype='bool'), yx=np.ones((ALL.califaID__yx.shape), dtype='bool')),
        S1=dict(z=sel_SN1__gz, yx=sel_SN1__gyx),
        S3=dict(z=sel_SN3__gz, yx=sel_SN3__gyx),
        S5=dict(z=sel_SN5__gz, yx=sel_SN5__gyx),
        S10=dict(z=sel_SN10__gz, yx=sel_SN10__gyx),
    )

    sel = dict(
        SN=sel_SN,
        WHa=sel_WHa,
        Zhang=sel_Zhang,
        BPT=sel_BPT,
        SN_BPT=sel_SN_BPT,
        SN_HaHb=sel_SN_HaHb,
    )

    return gals_sample_choice(ALL, sel, gals, sample_choice)


def gals_sample_choice(ALL, sel, gals, sample_choice):
    try:
        sample__z = sel[sample_choice[0]][sample_choice[1]]['z']
        sample__yx = sel[sample_choice[0]][sample_choice[1]]['yx']
    except KeyError:
        sample_choice = 'SN_BPT:S0'.split(':')
        print sample_choice
        sample__z = sel[sample_choice[0]][sample_choice[1]]['z']
        sample__yx = sel[sample_choice[0]][sample_choice[1]]['yx']
        print 'sample_choice %s does not exists' % sample_choice
        print 'running for S0...' % sample_choice

    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')
    sel_gals__gyx = np.zeros((ALL.califaID__yx.shape), dtype='bool')
    sel_gals_sample__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')
    sel_gals_sample__gyx = np.zeros((ALL.califaID__yx.shape), dtype='bool')

    new_gals = gals[:]
    print ALL.ba.shape
    print len(gals)
    # print gals
    for g in gals:
        tmp_sel__gz = (ALL.califaID__z == g)
        if not tmp_sel__gz.any():
            print g
            new_gals.remove(g)
            continue
        tmp_sel_sample__gz = np.bitwise_and(sample__z, ALL.califaID__z == g)
        tmp_sel_gal_sample__z = ALL.get_gal_prop(g, tmp_sel_sample__gz)
        tmp_sel__gyx = (ALL.califaID__yx == g)
        tmp_sel_sample__gyx = np.bitwise_and(sample__yx, ALL.califaID__yx == g)
        tmp_sel_gal_sample__yx = ALL.get_gal_prop(g, tmp_sel_sample__gyx)
        N_zone_notmasked = tmp_sel_gal_sample__z.astype(int).sum()
        N_pixel_notmasked = tmp_sel_gal_sample__yx.astype(int).sum()
        if N_zone_notmasked == 0 or N_pixel_notmasked == 0:
            new_gals.remove(g)
            continue
        sel_gals__gz[tmp_sel__gz] = True
        sel_gals__gyx[tmp_sel__gyx] = True
        sel_gals_sample__gz[tmp_sel_sample__gz] = True
        sel_gals_sample__gyx[tmp_sel_sample__gyx] = True

    sel['gals__z'] = sel_gals__gz
    sel['gals__yx'] = sel_gals__gyx
    sel['gals_sample__z'] = sel_gals_sample__gz
    sel['gals_sample__yx'] = sel_gals_sample__gyx

    return new_gals, sel, sample_choice


def summary(ALL, sel, gals, mask_name):
    import datetime
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

    print '# WHa classif:'
    tmp_sel_HIG__gz = np.bitwise_and(sel['WHa']['HIG']['z'], sel_gals_sample__gz)
    tmp_sel_DIG__gz = np.bitwise_and(sel['WHa']['DIG']['z'], sel_gals_sample__gz)
    tmp_sel_COMP__gz = np.bitwise_and(sel['WHa']['COMP']['z'], sel_gals_sample__gz)
    tmp_sel_SF__gz = np.bitwise_and(sel['WHa']['SF']['z'], sel_gals_sample__gz)
    tmp_sel_HIG__gyx = np.bitwise_and(sel['WHa']['DIG']['yx'], sel_gals_sample__gyx)
    tmp_sel_DIG__gyx = np.bitwise_and(sel['WHa']['DIG']['yx'], sel_gals_sample__gyx)
    tmp_sel_COMP__gyx = np.bitwise_and(sel['WHa']['COMP']['yx'], sel_gals_sample__gyx)
    tmp_sel_SF__gyx = np.bitwise_and(sel['WHa']['SF']['yx'], sel_gals_sample__gyx)
    print '\tTotal zones HIG: %d' % tmp_sel_HIG__gz.astype('int').sum()
    print '\tTotal zones DIG: %d' % tmp_sel_DIG__gz.astype('int').sum()
    print '\tTotal zones COMP: %d' % tmp_sel_COMP__gz.astype('int').sum()
    print '\tTotal zones SF: %d' % tmp_sel_SF__gz.astype('int').sum()
    print '\tTotal spaxels HIG: %d' % tmp_sel_HIG__gyx.astype('int').sum()
    print '\tTotal spaxels DIG: %d' % tmp_sel_DIG__gyx.astype('int').sum()
    print '\tTotal spaxels COMP: %d' % tmp_sel_COMP__gyx.astype('int').sum()
    print '\tTotal spaxels SF: %d' % tmp_sel_SF__gyx.astype('int').sum()
    for g in gals:
        tmp_sel_gal__z = ALL.get_gal_prop(g, sel_gals_sample__gz)
        tmp_sel_gal__yx = ALL.get_gal_prop(g, sel_gals_sample__gyx)
        tmp_sel_gal_HIG__z = ALL.get_gal_prop(g, tmp_sel_HIG__gz)
        tmp_sel_gal_DIG__z = ALL.get_gal_prop(g, tmp_sel_DIG__gz)
        tmp_sel_gal_COMP__z = ALL.get_gal_prop(g, tmp_sel_COMP__gz)
        tmp_sel_gal_SF__z = ALL.get_gal_prop(g, tmp_sel_SF__gz)
        N_zone = len(tmp_sel_gal__z)
        N_HIG = ALL.get_gal_prop(g, sel['WHa']['HIG']['z']).astype('int').sum()
        N_HIG_notmasked = tmp_sel_gal_HIG__z.astype('int').sum()
        N_DIG = ALL.get_gal_prop(g, sel['WHa']['DIG']['z']).astype('int').sum()
        N_DIG_notmasked = tmp_sel_gal_DIG__z.astype('int').sum()
        N_COMP = ALL.get_gal_prop(g, sel['WHa']['COMP']['z']).astype('int').sum()
        N_COMP_notmasked = tmp_sel_gal_COMP__z.astype('int').sum()
        N_SF = ALL.get_gal_prop(g, sel['WHa']['SF']['z']).astype('int').sum()
        N_SF_notmasked = tmp_sel_gal_SF__z.astype('int').sum()
        N_TOT = N_HIG+N_DIG+N_SF+N_COMP
        N_TOT_notmasked = N_HIG_notmasked+N_DIG_notmasked+N_SF_notmasked+N_COMP_notmasked
        HIG_perc_tot = 0.
        DIG_perc_tot = 0.
        COMP_perc_tot = 0.
        SF_perc_tot = 0.
        DIG_perc = 0.
        SF_perc = 0.
        if N_TOT_notmasked > 0:
            HIG_perc_tot = 100. * N_HIG_notmasked/(N_TOT_notmasked)
            DIG_perc_tot = 100. * N_DIG_notmasked/(N_TOT_notmasked)
            COMP_perc_tot = 100. * N_COMP_notmasked/(N_TOT_notmasked)
            SF_perc_tot = 100. * N_SF_notmasked/(N_TOT_notmasked)
        if N_SF_notmasked > 0 or N_DIG_notmasked > 0 or N_HIG_notmasked > 0:
            tot_aux = N_DIG_notmasked+N_SF_notmasked+N_HIG_notmasked
            HIG_perc = 100. * N_HIG_notmasked/tot_aux
            DIG_perc = 100. * N_DIG_notmasked/tot_aux
            SF_perc = 100. * N_SF_notmasked/tot_aux
        if N_DIG_notmasked > 0 or N_HIG_notmasked > 0:
            totIG_aux = N_HIG_notmasked+N_DIG_notmasked
            HIG_to_totIG_perc = 100. * N_HIG_notmasked/totIG_aux
            DIG_to_totIG_perc = 100. * N_DIG_notmasked/totIG_aux
        print '%s - (Nz:%d - Ntot: %d of %d) - %d HIG (of %d) (%.1f%% [%.1f%%, %.1f%%]) - %d DIG (of %d) (%.1f%% [%.1f%%, %.1f%%]) - %d COMP (of %d) (%.1f%%) - %d SF (of %d) (%.1f%% [%.1f%%])' % (g, N_zone, N_TOT_notmasked, N_TOT, N_HIG_notmasked, N_HIG, HIG_perc_tot, HIG_perc, HIG_to_totIG_perc, N_DIG_notmasked, N_DIG, DIG_perc_tot, DIG_perc, DIG_to_totIG_perc, N_COMP_notmasked, N_COMP, COMP_perc_tot, N_SF_notmasked, N_SF, SF_perc_tot, SF_perc)

    # print '# Zhang classif:'
    # tmp_sel_SF__gz = np.bitwise_and(sel['Zhang']['SF']['z'], sel_gals_sample__gz)
    # tmp_sel_SF__gyx = np.bitwise_and(sel['Zhang']['SF']['yx'], sel_gals_sample__gyx)
    # print '\tTotal zones not SF: %d' % (~tmp_sel_SF__gz).astype('int').sum()
    # print '\tTotal zones SF: %d' % tmp_sel_SF__gz.astype('int').sum()
    # print '\tTotal spaxels not SF: %d' % (~tmp_sel_SF__gyx).astype('int').sum()
    # print '\tTotal spaxels SF: %d' % tmp_sel_SF__gyx.astype('int').sum()
    # for g in gals:
    #     tmp_sel_gal__z = ALL.get_gal_prop(g, sel_gals_sample__gz)
    #     tmp_sel_gal__yx = ALL.get_gal_prop(g, sel_gals_sample__gyx)
    #     tmp_sel_gal_SF__z = ALL.get_gal_prop(g, tmp_sel_SF__gz)
    #
    #     N_zone = len(tmp_sel_gal__z)
    #     N_SF = ALL.get_gal_prop(g, sel['Zhang']['SF']['z']).astype('int').sum()
    #     N_SF_notmasked = tmp_sel_gal_SF__z.astype('int').sum()
    #     N_NOTSF = ALL.get_gal_prop(g, ~sel['Zhang']['SF']['z']).astype('int').sum()
    #     N_NOTSF_notmasked = (~tmp_sel_gal_SF__z).astype('int').sum()
    #     N_TOT = N_NOTSF+N_SF
    #     N_TOT_notmasked = N_NOTSF_notmasked+N_SF_notmasked
    #     SF_perc = 0.
    #     NOTSF_perc = 0.
    #     if N_TOT_notmasked > 0:
    #         SF_perc = 100. * N_SF_notmasked/N_TOT_notmasked
    #         NOTSF_perc = 100. * N_NOTSF_notmasked/N_TOT_notmasked
    #     print '%s - (Nz:%d - Ntot: %d of %d) - %d NOT_SF (of %d) (%.1f%%) - %d SF (of %d) (%.1f%%)' % (g, N_zone, N_TOT_notmasked, N_TOT, N_NOTSF_notmasked, N_NOTSF, NOTSF_perc, N_SF_notmasked, N_SF, SF_perc)
    #
    FWHM = 2.5
    sel_HIG__gz = np.bitwise_and(sel['WHa']['HIG']['z'], sel_gals_sample__gz)
    sel_DIG__gz = np.bitwise_and(sel['WHa']['DIG']['z'], sel_gals_sample__gz)
    sel_COMP__gz = np.bitwise_and(sel['WHa']['COMP']['z'], sel_gals_sample__gz)
    sel_SF__gz = np.bitwise_and(sel['WHa']['SF']['z'], sel_gals_sample__gz)
    sel_HIG__gyx = np.bitwise_and(sel['WHa']['HIG']['yx'], sel_gals_sample__gyx)
    sel_DIG__gyx = np.bitwise_and(sel['WHa']['DIG']['yx'], sel_gals_sample__gyx)
    sel_COMP__gyx = np.bitwise_and(sel['WHa']['COMP']['yx'], sel_gals_sample__gyx)
    sel_SF__gyx = np.bitwise_and(sel['WHa']['SF']['yx'], sel_gals_sample__gyx)
    d_Mpc__g = []
    spaxel_size_pc__g = []
    FWHM_pc__g = []
    LHa_sum__g = []
    LHa_sum_HIG__g = []
    LHa_sum_DIG__g = []
    LHa_sum_COMP__g = []
    LHa_sum_SF__g = []
    mto__g = []
    table_paper = []
    table_paper_header = '''
 \\begin{table}
  \\centering
   \\begin{tabular}{@{}lcccccc@{}}
    \\hline
    '''
    table_paper.append(table_paper_header)
    table_paper_table_header = '''
    ID & Name & Type & pc/$^{\prime\prime}$ & L${}_{\Ha}^{\\rm obs}$ & \\%SF & \\%HIG & \\%DIG \\\\
    (1) & (2) & (3) & (4) & (5) & (6) & (7) & (8) \\\\
    \\hline
    '''
    table_paper.append(table_paper_table_header)
    N_pixel__g = []
    pixelDistance_pc__gyx = np.ma.masked_all(ALL.pixelDistance__yx.shape)
    for i_g, califaID in enumerate(gals):
        sel_gal__gyx = np.bitwise_and(sel_gals_sample__gyx, ALL.califaID__yx == califaID)
        HLR_pc = ALL.get_gal_prop_unique(califaID, ALL.HLR_pc)
        pixelDistance_pc__gyx[sel_gal__gyx] = (ALL.pixelDistance_HLR__yx[sel_gal__gyx] * HLR_pc)
        sel_gal__z = ALL.get_gal_prop(califaID, sel_gals_sample__gz)
        sel_gal_HIG__z = ALL.get_gal_prop(califaID, sel_HIG__gz)
        sel_gal_DIG__z = ALL.get_gal_prop(califaID, sel_DIG__gz)
        sel_gal_COMP__z = ALL.get_gal_prop(califaID, sel_COMP__gz)
        sel_gal_SF__z = ALL.get_gal_prop(califaID, sel_SF__gz)
        sel_gal__yx = ALL.get_gal_prop(califaID, sel_gals_sample__gyx)
        sel_gal_HIG__yx = ALL.get_gal_prop(califaID, sel_HIG__gyx)
        sel_gal_DIG__yx = ALL.get_gal_prop(califaID, sel_DIG__gyx)
        sel_gal_COMP__yx = ALL.get_gal_prop(califaID, sel_COMP__gyx)
        sel_gal_SF__yx = ALL.get_gal_prop(califaID, sel_SF__gyx)
        d_Mpc = ALL.get_gal_prop_unique(califaID, ALL.galDistance_Mpc)
        N_pixel = sel_gal__yx.astype('int').sum()
        N_pixel__g.append(N_pixel)
        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        spaxel_size = spaxel_size_pc(d_Mpc)
        FWHM_size = spaxel_size_pc(d_Mpc, FWHM)
        d_Mpc__g.append(d_Mpc)
        spaxel_size_pc__g.append(spaxel_size)
        FWHM_pc__g.append(FWHM_size)
        L6563__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.L6563__z), mask=~sel_gal__z)
        L6563_HIG__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.L6563__z), mask=~sel_gal_HIG__z)
        L6563_DIG__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.L6563__z), mask=~sel_gal_DIG__z)
        L6563_COMP__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.L6563__z), mask=~sel_gal_COMP__z)
        L6563_SF__z = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.L6563__z), mask=~sel_gal_SF__z)
        LHa_sum = L6563__z.sum()
        LHa_sum__g.append(LHa_sum)
        mto__g.append(mto)
        if L6563_HIG__z.count() > 0:
            LHa_sum_HIG = L6563_HIG__z.sum()
        else:
            LHa_sum_HIG = 0.
        if L6563_DIG__z.count() > 0:
            LHa_sum_DIG = L6563_DIG__z.sum()
        else:
            LHa_sum_DIG = 0.
        if L6563_COMP__z.count() > 0:
            LHa_sum_COMP = L6563_COMP__z.sum()
        else:
            LHa_sum_COMP = 0.
        if L6563_SF__z.count() > 0:
            LHa_sum_SF = L6563_SF__z.sum()
        else:
            LHa_sum_SF = 0
        LHa_sum_HIG__g.append(LHa_sum_HIG)
        LHa_sum_DIG__g.append(LHa_sum_DIG)
        LHa_sum_COMP__g.append(LHa_sum_COMP)
        LHa_sum_SF__g.append(LHa_sum_SF)

        if LHa_sum_DIG > 0 or LHa_sum_HIG > 0:
            ratio_HIG_DIGHIG = LHa_sum_HIG/(LHa_sum_DIG+LHa_sum_HIG)
            ratio_DIG_DIGHIG = LHa_sum_DIG/(LHa_sum_DIG+LHa_sum_HIG)
        else:
            ratio_HIG_DIGHIG = 0
            ratio_DIG_DIGHIG = 0

        ratio_SF_SFDIGHIG = LHa_sum_SF/(LHa_sum_DIG+LHa_sum_SF+LHa_sum_HIG)
        ratio_SF_TOT = LHa_sum_SF/(LHa_sum_DIG+LHa_sum_SF+LHa_sum_COMP+LHa_sum_HIG)

        ratio_DIG_SFDIGHIG = LHa_sum_DIG/(LHa_sum_DIG+LHa_sum_SF+LHa_sum_HIG)
        ratio_DIG_TOT = LHa_sum_DIG/(LHa_sum_DIG+LHa_sum_SF+LHa_sum_COMP+LHa_sum_HIG)

        ratio_HIG_SFDIGHIG = LHa_sum_HIG/(LHa_sum_DIG+LHa_sum_SF+LHa_sum_HIG)
        ratio_HIG_TOT = LHa_sum_HIG/(LHa_sum_DIG+LHa_sum_SF+LHa_sum_COMP+LHa_sum_HIG)

        nedname = get_NEDName_by_CALIFAID(califaID)[0]
        if nedname.find('NGC') == 0:
            nedname = 'NGC %s' % nedname.replace('NGC', '')
        elif nedname.find('UGC') == 0:
            nedname = 'UGC %s' % nedname.replace('UGC', '')
        table_paper_gal = '%03d & %s & %s & %.0f & %.2f & %.0f (%.0f) & %.0f (%.0f) & %.0f (%.0f) \\\\ ' % (int(califaID[1:]), nedname, mto, spaxel_size, LHa_sum/1e7, 100.*ratio_SF_SFDIGHIG, 100.*ratio_SF_TOT, 100.*ratio_HIG_SFDIGHIG, 100.*ratio_HIG_TOT, 100.*ratio_DIG_SFDIGHIG, 100.*ratio_DIG_TOT)
        table_paper.append(table_paper_gal)
    table_paper_bottom = '''
     \\hline
     \\end{tabular}
    \\caption{
    Columns 1 to 5 list the CALIFA ID number, name, morphological type (from **Walcher+15), linear scale, and the total observed \\Ha luminosity in units of $10^7 L_\\odot$. Columns 6 and 7 apply the definitions of section \\ref{sec:DIGxHIIcriterion} to estimate the percentage fraction of $L_{\\Ha}^{\\rm obs}$ which comes from SF and DIG dominated regions, respectively. These fractions are computed considering only SF, DIG and HIG spaxels (i.e., SF + DIG + HIG = 100\\%%). Fractions including also the composite class are given in between parenthesis.
    }
    \\label{tab:Data}
    \\end{table}
    '''
    table_paper.append(table_paper_bottom)
    for line in table_paper:
        print line
    print 'N_pixel: %d (%.2f per gal)' % (np.sum(N_pixel__g), np.mean(N_pixel__g))
    print '<spaxel size>: %.1f pc' % np.mean(spaxel_size_pc__g)
    print '<FWHM (2.5 arcsec) size>: %.1f pc' % np.mean(FWHM_pc__g)
    print '<d_Mpc>: %.2f' % np.mean(d_Mpc__g)
    print 'max(d_Mpc): %.2f [%s]' % (np.max(d_Mpc__g), gals[np.argmax(d_Mpc__g)])
    print 'min(d_Mpc): %.2f [%s]' % (np.min(d_Mpc__g), gals[np.argmin(d_Mpc__g)])
    N_SF = sel_SF__gyx.astype('int').sum()
    SBHa_SF__gyx = np.ma.masked_array(ALL.SB6563__yx, mask=~sel_SF__gyx)
    sel_low_SBHa_SF__gyx = np.ma.less(SBHa_SF__gyx, SF_Zhang_threshold)
    N_lowSBHa_SF = (sel_low_SBHa_SF__gyx.compressed()).astype('int').sum()
    print len(pixelDistance_pc__gyx), len(sel_low_SBHa_SF__gyx)
    mean_distance_lowSBHa_SF = pixelDistance_pc__gyx[sel_low_SBHa_SF__gyx].mean()
    print 'N_lowSBHa_SF: %d (of %d - %.0f%%) <distance>=%.2f kpc' % (N_lowSBHa_SF, N_SF, 100.*N_lowSBHa_SF/N_SF, mean_distance_lowSBHa_SF/1e3)
    sel_BPT_S06SF__gz = np.bitwise_and(sel['BPT']['S06']['z'], sel_gals_sample__gz)
    sel_BPT_S06SF__gyx = np.bitwise_and(sel['BPT']['S06']['yx'], sel_gals_sample__gyx)
    N_SF_WHa_S06BPT__gz = np.bitwise_and(sel_BPT_S06SF__gz, sel_SF__gz).astype('int').sum()
    N_SF_WHa_S06BPT__gyx = np.bitwise_and(sel_BPT_S06SF__gyx, sel_SF__gyx).astype('int').sum()
    N_DIG_WHa_S06BPT_zones = np.bitwise_and(~sel_BPT_S06SF__gz, sel_DIG__gz).astype('int').sum()
    N_DIG_WHa_S06BPT_pixels = np.bitwise_and(~sel_BPT_S06SF__gyx, sel_DIG__gyx).astype('int').sum()
    N_HIG_WHa_S06BPT_zones = np.bitwise_and(~sel_BPT_S06SF__gz, sel_HIG__gz).astype('int').sum()
    N_HIG_WHa_S06BPT_pixels = np.bitwise_and(~sel_BPT_S06SF__gyx, sel_HIG__gyx).astype('int').sum()
    print 'zones: N_SF(S06 & WHa) = %d (%.2f%%)' % (N_SF_WHa_S06BPT__gz, 100.*N_SF_WHa_S06BPT__gz/sel_SF__gz.astype('int').sum())
    print 'pixels: N_SF(S06 & WHa) = %d (%.2f%%)' % (N_SF_WHa_S06BPT__gyx, 100.*N_SF_WHa_S06BPT__gyx/sel_SF__gyx.astype('int').sum())
    print 'zones: N_DIG(S06 & WHa) = %d (%.2f%%)' % (N_DIG_WHa_S06BPT_zones, 100.*N_DIG_WHa_S06BPT_zones/sel_DIG__gz.astype('int').sum())
    print 'pixels: N_DIG(S06 & WHa) = %d (%.2f%%)' % (N_DIG_WHa_S06BPT_pixels, 100.*N_DIG_WHa_S06BPT_pixels/sel_DIG__gyx.astype('int').sum())
    print 'zones: N_HIG(S06 & WHa) = %d (%.2f%%)' % (N_HIG_WHa_S06BPT_zones, 100.*N_HIG_WHa_S06BPT_zones/sel_HIG__gz.astype('int').sum())
    print 'pixels: N_HIG(S06 & WHa) = %d (%.2f%%)' % (N_HIG_WHa_S06BPT_pixels, 100.*N_HIG_WHa_S06BPT_pixels/sel_HIG__gyx.astype('int').sum())
    sel_BPT_K03SF__gz = np.bitwise_and(sel['BPT']['K03']['z'], sel_gals_sample__gz)
    sel_BPT_K03SF__gyx = np.bitwise_and(sel['BPT']['K03']['yx'], sel_gals_sample__gyx)
    N_SF_WHa_K03BPT__gz = np.bitwise_and(sel_BPT_K03SF__gz, sel_SF__gz).astype('int').sum()
    N_SF_WHa_K03BPT__gyx = np.bitwise_and(sel_BPT_K03SF__gyx, sel_SF__gyx).astype('int').sum()
    N_DIG_WHa_K03BPT_zones = np.bitwise_and(~sel_BPT_K03SF__gz, sel_DIG__gz).astype('int').sum()
    N_DIG_WHa_K03BPT_pixels = np.bitwise_and(~sel_BPT_K03SF__gyx, sel_DIG__gyx).astype('int').sum()
    N_HIG_WHa_K03BPT_zones = np.bitwise_and(~sel_BPT_K03SF__gz, sel_HIG__gz).astype('int').sum()
    N_HIG_WHa_K03BPT_pixels = np.bitwise_and(~sel_BPT_K03SF__gyx, sel_HIG__gyx).astype('int').sum()
    print 'zones: N_SF(K03 & WHa) = %d (%.2f%%)' % (N_SF_WHa_K03BPT__gz, 100.*N_SF_WHa_K03BPT__gz/sel_SF__gz.astype('int').sum())
    print 'pixels: N_SF(K03 & WHa) = %d (%.2f%%)' % (N_SF_WHa_K03BPT__gyx, 100.*N_SF_WHa_K03BPT__gyx/sel_SF__gyx.astype('int').sum())
    print 'zones: N_DIG(K03 & WHa) = %d (%.2f%%)' % (N_DIG_WHa_K03BPT_zones, 100.*N_DIG_WHa_K03BPT_zones/sel_DIG__gz.astype('int').sum())
    print 'pixels: N_DIG(K03 & WHa) = %d (%.2f%%)' % (N_DIG_WHa_K03BPT_pixels, 100.*N_DIG_WHa_K03BPT_pixels/sel_DIG__gyx.astype('int').sum())
    print 'zones: N_HIG(K03 & WHa) = %d (%.2f%%)' % (N_HIG_WHa_K03BPT_zones, 100.*N_HIG_WHa_K03BPT_zones/sel_HIG__gz.astype('int').sum())
    print 'pixels: N_HIG(K03 & WHa) = %d (%.2f%%)' % (N_HIG_WHa_K03BPT_pixels, 100.*N_HIG_WHa_K03BPT_pixels/sel_HIG__gyx.astype('int').sum())


def create_segmented_map(ALL, califaID, sel, sample_sel=None):
    N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
    N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
    sel_HIG__yx, sel_DIG__yx, sel_COMP__yx, sel_SF__yx, _ = get_selections(ALL, califaID, sel, sample_sel)
    map__yx = np.ma.masked_all((N_y, N_x))
    map__yx[sel_HIG__yx] = 1
    map__yx[sel_DIG__yx] = 2
    map__yx[sel_COMP__yx] = 3
    map__yx[sel_SF__yx] = 4
    return map__yx


def create_segmented_map_zones(ALL, califaID, sel, sample_sel=None):
    N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
    sel_HIG__z, sel_DIG__z, sel_COMP__z, sel_SF__z, _ = get_selections_zones(ALL, califaID, sel, sample_sel)
    map__z = np.ma.masked_all((N_zone))
    map__z[sel_HIG__z] = 1
    map__z[sel_DIG__z] = 2
    map__z[sel_COMP__z] = 3
    map__z[sel_SF__z] = 4
    return map__z


def get_selections(ALL, califaID, sel, sample_sel=None):
    N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
    N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
    if sample_sel is None:
        sel__yx = np.ones((N_y, N_x), dtype='bool')
    else:
        sel__yx = ALL.get_gal_prop(califaID, sample_sel).reshape(N_y, N_x)
    sel_HIG__yx = np.bitwise_and(sel__yx, ALL.get_gal_prop(califaID, sel['HIG']['yx']).reshape(N_y, N_x))
    sel_DIG__yx = np.bitwise_and(sel__yx, ALL.get_gal_prop(califaID, sel['DIG']['yx']).reshape(N_y, N_x))
    sel_COMP__yx = np.bitwise_and(sel__yx, ALL.get_gal_prop(califaID, sel['COMP']['yx']).reshape(N_y, N_x))
    sel_SF__yx = np.bitwise_and(sel__yx, ALL.get_gal_prop(califaID, sel['SF']['yx']).reshape(N_y, N_x))
    sel_gal_tot = sel__yx
    return sel_HIG__yx, sel_DIG__yx, sel_COMP__yx, sel_SF__yx, sel_gal_tot


def get_selections_zones(ALL, califaID, sel, sample_sel=None):
    N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
    if sample_sel is None:
        sel__z = np.ones((N_zone), dtype='bool')
    else:
        sel__z = ALL.get_gal_prop(califaID, sample_sel)
    sel_HIG__z = np.bitwise_and(sel__z, ALL.get_gal_prop(califaID, sel['HIG']['z']))
    sel_DIG__z = np.bitwise_and(sel__z, ALL.get_gal_prop(califaID, sel['DIG']['z']))
    sel_COMP__z = np.bitwise_and(sel__z, ALL.get_gal_prop(califaID, sel['COMP']['z']))
    sel_SF__z = np.bitwise_and(sel__z, ALL.get_gal_prop(califaID, sel['SF']['z']))
    sel_gal_tot = sel__z
    return sel_HIG__z, sel_DIG__z, sel_COMP__z, sel_SF__z, sel_gal_tot


def plotBPT(ax, N2Ha, O3Hb, z=None, cmap='viridis', mask=None, labels=True,
            N=False, cb_label=r'R [HLR]', vmax=None, vmin=None, dcontour=True,
            s=10, extent=[-1.5, 1, -1.5, 1.5], bins=[30, 30],
            kwargs_scatter=dict(marker='o', edgecolor='none')):
    if mask is None:
        mask = np.zeros_like(O3Hb, dtype=np.bool_)
    if z is None:
        xm, ym = ma_mask_xyz(N2Ha, O3Hb, mask=mask)
        if dcontour:
            density_contour(xm.compressed(), ym.compressed(), bins[0], bins[1], ax,
                            range=[extent[0:2], extent[2:4]], colors=['b', 'y', 'r'])
        sc = ax.scatter(xm, ym, c='0.5', s=s, alpha=0.4, **kwargs_scatter)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
    else:
        xm, ym, z = ma_mask_xyz(N2Ha, O3Hb, z, mask=mask)
        sc = ax.scatter(xm, ym, c=z, cmap=cmap, vmin=vmin, vmax=vmax, s=s, **kwargs_scatter)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
        ax.set_aspect('equal', 'box')
        the_divider = make_axes_locatable(ax)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(sc, cax=color_axis)
        # cb = plt.colorbar(sc, ax=ax, ticks=[0, .5, 1, 1.5, 2, 2.5, 3], pad=0)
        cb.set_label(cb_label)
    if labels:
        ax.set_xlabel(r'$\log\ [NII]/H\alpha$')
        ax.set_ylabel(r'$\log\ [OIII]/H\beta$')
    L = Lines()
    if not N:
        N = xm.count()
    c = ''
    if (xm.compressed() < extent[0]).any():
        c += 'x-'
    if (xm.compressed() > extent[1]).any():
        c += 'x+'
    if (ym.compressed() < extent[2]).any():
        c += 'y-'
    if (ym.compressed() > extent[3]).any():
        c += 'y+'
    plot_text_ax(ax, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
    plot_text_ax(ax, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
    ax.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
    ax.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
    ax.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
    plot_text_ax(ax, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=38)  # 44.62)
    ax.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
    L.fixCF10('S06')
    return ax


def fig2(ALL, sel, gals):
    sel_gals_sample__gz = sel['gals_sample__z']

    if (sel_gals_sample__gz).any():
        W6563__gz = ALL.W6563__z
        SB6563__gz = ALL.SB6563__z
        dist__gz = ALL.zoneDistance_HLR

        # WHa HIG-DIG-COMP-SF decomposition
        sel_WHa_HIG__gz = np.bitwise_and(sel['WHa']['HIG']['z'], sel_gals_sample__gz)
        sel_WHa_DIG__gz = np.bitwise_and(sel['WHa']['DIG']['z'], sel_gals_sample__gz)
        sel_WHa_COMP__gz = np.bitwise_and(sel['WHa']['COMP']['z'], sel_gals_sample__gz)
        sel_WHa_SF__gz = np.bitwise_and(sel['WHa']['SF']['z'], sel_gals_sample__gz)

        x = np.ma.log10(W6563__gz)
        y = np.ma.log10(SB6563__gz)
        z = dist__gz
        xm, ym, zm = ma_mask_xyz(x=x, y=y, z=z, mask=~sel_gals_sample__gz)
        f = plt.figure(figsize=(8, 8))

        x_ds = [xm[sel_WHa_HIG__gz].compressed(), xm[sel_WHa_DIG__gz].compressed(), xm[sel_WHa_COMP__gz].compressed(), xm[sel_WHa_SF__gz].compressed()]
        y_ds = [ym[sel_WHa_HIG__gz].compressed(), ym[sel_WHa_DIG__gz].compressed(), ym[sel_WHa_COMP__gz].compressed(), ym[sel_WHa_SF__gz].compressed()]
        axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, logWHa_range, logSBHa_range, 50, 50,
                                             figure=f, c=colors_DIG_COMP_SF, scatter=False, s=1,
                                             ylabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]',
                                             xlabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]', histtype='step')
        axS.xaxis.set_major_locator(MultipleLocator(0.5))
        axS.xaxis.set_minor_locator(MultipleLocator(0.1))
        axH1.xaxis.set_major_locator(MultipleLocator(0.5))
        axH1.xaxis.set_minor_locator(MultipleLocator(0.1))
        axH2.xaxis.set_major_locator(MaxNLocator(3))
        axS.yaxis.set_major_locator(MultipleLocator(0.5))
        axS.yaxis.set_minor_locator(MultipleLocator(0.1))
        axH1.yaxis.set_major_locator(MaxNLocator(3))
        axH2.yaxis.set_major_locator(MultipleLocator(0.5))
        axH2.yaxis.set_minor_locator(MultipleLocator(0.1))
        aux_ax = axH2.twiny()
        plot_histo_ax(aux_ax, ym.compressed(), stats_txt=False, kwargs_histo=dict(histtype='step', orientation='horizontal', normed=False, range=logSBHa_range, color='k', lw=2, ls='-'))
        aux_ax.xaxis.set_major_locator(MaxNLocator(3))
        plt.setp(aux_ax.xaxis.get_majorticklabels(), rotation=270)
        plot_text_ax(axH1, r'W${}_{H\alpha}$ >= %d $\AA$' % SF_WHa_threshold, 0.98, 0.99, 14, 'top', 'right', colors_DIG_COMP_SF[3])
        plot_text_ax(axH1, r'%d $\AA$ <= W${}_{H\alpha}$ < %d $\AA$' % (DIG_WHa_threshold, SF_WHa_threshold), 0.98, 0.89, 14, 'top', 'right', colors_DIG_COMP_SF[2])
        plot_text_ax(axH1, r'%d $\AA$ <= W${}_{H\alpha}$ < %d $\AA$' % (HIG_WHa_threshold, DIG_WHa_threshold), 0.98, 0.79, 14, 'top', 'right', colors_DIG_COMP_SF[1])
        plot_text_ax(axH1, r'W${}_{H\alpha}$ < %d $\AA$' % HIG_WHa_threshold, 0.98, 0.69, 14, 'top', 'right', colors_DIG_COMP_SF[0])
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
        xbins = np.linspace(0.1, 2, 20)
        yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), xbins)
        yMedian = prc[2]
        y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
        axS.plot(bin_center, yMedian, 'k-', lw=2)
        for y_prc in y_12sigma:
            axS.plot(bin_center, y_prc, 'k--', lw=2)
        axS.grid()
        f.savefig('fig2.png', dpi=dpi_choice, transparent=transp_choice)
        plt.close(f)


def fig3(ALL, sel, gals=None):
    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)
        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        ml_ba = ALL.get_gal_prop_unique(califaID, ALL.ml_ba)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        SB__lyx = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'SB%s__yx' % L).reshape(N_y, N_x), mask=~gal_sample__yx) for L in lines}
        W6563__yx = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x), mask=~gal_sample__yx)
        N_cols = 4
        N_rows = 1
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        cmap = cmap_discrete(colors=colors_DIG_COMP_SF)
        ax1, ax2, ax3, ax4 = axArr
        map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)

        tot_class = map__z.count()
        frac_HIG = 100. * map__z[map__z == 1].count()/tot_class
        frac_DIG = 100. * map__z[map__z == 2].count()/tot_class
        frac_COMP = 100. * map__z[map__z == 3].count()/tot_class
        frac_SF = 100. * map__z[map__z == 4].count()/tot_class
        print frac_HIG, frac_DIG, frac_COMP, frac_SF
        f.suptitle(r'%s - %s ba:%.2f (ml_ba:%.2f) (%s): %d pixels (%d zones - HIG:%d DIG:%d COMP:%d SF:%d)' % (califaID, mto, ba, ml_ba, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone, frac_HIG, frac_DIG, frac_COMP, frac_SF))

        # AXIS 1
        galimg = plt.imread(P.get_image_file(califaID))[::-1, :, :]
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.imshow(galimg, origin='lower', aspect='equal')
        DrawHLRCircleInSDSSImage(ax1, HLR_pix, pa, ba, lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.02, 0.98, 16, 'top', 'left', color='w')
        ax1.set_ylabel('%s' % mto, fontsize=24)
        # AXIS 2
        x = np.ma.log10(SB__lyx['6563'])
        im = ax2.imshow(x, vmin=logSBHa_range[0], vmax=6.5, cmap=plt.cm.copper_r, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        # cb.set_label(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        DrawHLRCircle(ax2, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # AXIS 3
        x = np.ma.log10(W6563__yx)
        im = ax3.imshow(x, vmin=logWHa_range[0], vmax=logWHa_range[1], cmap=plt.cm.copper_r, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax3)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        # cb.set_label(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        DrawHLRCircle(ax3, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # AXIS 4
        map__yx = create_segmented_map(ALL, califaID, sel['WHa'], sel_sample__gyx)
        im = ax4.imshow(map__yx, cmap=cmap, vmax=4, vmin=1, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax4)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        divisions = np.linspace(1, 4, 5)
        ticks = (divisions[1:] - divisions[0:-1])/2. + divisions[0:-1]
        cb = plt.colorbar(im, cax=color_axis, ticks=ticks)
        cb.set_ticklabels(classif_labels)
        DrawHLRCircle(ax4, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        ax1.set_title('SDSS stamp', fontsize=18)
        # ax2.set_title(r'$\log$ W${}_{H\alpha}$ map')
        # ax3.set_title(r'$\log\ \Sigma_{H\alpha}$ map')
        ax2.set_title(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]', fontsize=18)
        ax3.set_title(r'$\log$ W${}_{H\alpha}$ [$\AA$]', fontsize=18)
        ax4.set_title(r'classification map', fontsize=18)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('fig3-%s.png' % califaID)
        plt.close(f)

        # HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        # mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        # pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        # ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        # N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        # N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        # x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        # y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        # N_pixel = N_x * N_y
        # N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        # SB__lyx = {'%s' % L: ALL.get_gal_prop(califaID, 'SB%s__yx' % L).reshape(N_y, N_x) for L in lines}
        # W6563__yx = ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x)
        #
        # N_cols = 4
        # N_rows = 1
        # f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        # cmap = cmap_discrete(colors=colors_DIG_COMP_SF)
        # ax1, ax2, ax3, ax4 = axArr
        # f.suptitle(r'%s - %s (%s): %d pixels (%d zones)' % (califaID, mto, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))
        # # AXIS 1
        # galimg = plt.imread(P.get_image_file(califaID))[::-1, :, :]
        # plt.setp(ax1.get_xticklabels(), visible=False)
        # plt.setp(ax1.get_yticklabels(), visible=False)
        # ax1.imshow(galimg, origin='lower', aspect='equal')
        # DrawHLRCircleInSDSSImage(ax1, HLR_pix, pa, ba, lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # txt = '%s' % califaID
        # plot_text_ax(ax1, txt, 0.02, 0.98, 16, 'top', 'left', color='w')
        # ax1.set_title('SDSS stamp')
        # # AXIS 2
        # x = np.ma.log10(W6563__yx)
        # im = ax2.imshow(x, vmin=logWHa_range[0], vmax=logWHa_range[1], cmap=cmap_R, **dflt_kw_imshow)
        # the_divider = make_axes_locatable(ax2)
        # color_axis = the_divider.append_axes('right', size='5%', pad=0)
        # cb = plt.colorbar(im, cax=color_axis)
        # cb.set_label(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        # DrawHLRCircle(ax2, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # ax2.set_title(r'$\log$ W${}_{H\alpha}$ map')
        # # AXIS 3
        # x = np.ma.log10(SB__lyx['6563'])
        # im = ax3.imshow(x, vmin=logSBHa_range[0], vmax=logSBHa_range[1], cmap=cmap_R, **dflt_kw_imshow)
        # the_divider = make_axes_locatable(ax3)
        # color_axis = the_divider.append_axes('right', size='5%', pad=0)
        # cb = plt.colorbar(im, cax=color_axis)
        # cb.set_label(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        # DrawHLRCircle(ax3, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # ax3.set_title(r'$\log\ \Sigma_{H\alpha}$ map')
        # # AXIS 4
        # map__yx = create_segmented_map(ALL, califaID, sel_WHa_DIG__yx, sel_WHa_COMP__yx, sel_WHa_SF__yx)
        # im = ax4.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
        # the_divider = make_axes_locatable(ax4)
        # color_axis = the_divider.append_axes('right', size='5%', pad=0)
        # cb = plt.colorbar(im, cax=color_axis, ticks=[1.+2/6., 2, 3-2/6.])
        # cb.set_ticklabels(classif_labels)
        # DrawHLRCircle(ax4, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # ax4.set_title(r'classif. map')
        # # FINAL
        # f.tight_layout(rect=[0, 0.03, 1, 0.95])
        # f.savefig('fig3-%s.png' % califaID)
        # plt.close(f)


def fig3_3gals(ALL, sel, gals, suffix):
    sel_sample__gyx = sel['gals_sample__yx']

    N_cols = 4
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    cmap = cmap_discrete(colors=colors_DIG_COMP_SF)

    row = 0
    for califaID in gals:
        print califaID
        (ax1, ax2, ax3, ax4) = axArr[row]
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)
        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        SB__lyx = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'SB%s__yx' % L).reshape(N_y, N_x), mask=~gal_sample__yx) for L in lines}
        W6563__yx = np.ma.masked_array(ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x), mask=~gal_sample__yx)
        # AXIS 1
        galimg = plt.imread(P.get_image_file(califaID))[::-1, :, :]
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.imshow(galimg, origin='lower', aspect='equal')
        DrawHLRCircleInSDSSImage(ax1, HLR_pix, pa, ba, lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.02, 0.98, 16, 'top', 'left', color='w')
        # ax1.set_ylabel('%s' % mto, fontsize=24)
        # AXIS 2
        x = np.ma.log10(SB__lyx['6563'])
        im = ax2.imshow(x, vmin=logSBHa_range[0], vmax=6.5, cmap=plt.cm.copper_r, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax2)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        # cb.set_label(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        DrawHLRCircle(ax2, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # AXIS 3
        x = np.ma.log10(W6563__yx)
        im = ax3.imshow(x, vmin=logWHa_range[0], vmax=logWHa_range[1], cmap=plt.cm.copper_r, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax3)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(im, cax=color_axis)
        # cb.set_label(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        DrawHLRCircle(ax3, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        # AXIS 4
        map__yx = create_segmented_map(ALL, califaID, sel['WHa'], sel_sample__gyx)
        im = ax4.imshow(map__yx, cmap=cmap, **dflt_kw_imshow)
        the_divider = make_axes_locatable(ax4)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        divisions = np.linspace(1, 4, 5)
        ticks = (divisions[1:] - divisions[0:-1])/2. + divisions[0:-1]
        cb = plt.colorbar(im, cax=color_axis, ticks=ticks)
        cb.set_ticklabels(classif_labels)
        DrawHLRCircle(ax4, a=HLR_pix, pa=pa, ba=ba, x0=x0, y0=y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
        if row == 0:
            ax1.set_title('SDSS stamp', fontsize=18)
            # ax2.set_title(r'$\log$ W${}_{H\alpha}$ map')
            # ax3.set_title(r'$\log\ \Sigma_{H\alpha}$ map')
            ax2.set_title(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]', fontsize=18)
            ax3.set_title(r'$\log$ W${}_{H\alpha}$ [$\AA$]', fontsize=18)
            ax4.set_title(r'classification map', fontsize=18)
        row += 1
    # FINAL
    f.tight_layout()
    f.savefig('fig3_%s.png' % suffix, dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


def fig4(ALL, sel, gals=None):
    sel_sample__gz = sel['gals_sample__z']

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        L = Lines()
        L.addLine('K01S2', L.linebpt, (1.30, 0.72, -0.32), np.linspace(-10.0, 0.32, L.xn + 1)[:-1])
        L.addLine('K01OI', L.linebpt, (1.33, 0.73, 0.59), np.linspace(-10.0, -0.59, L.xn + 1)[:-1])
        gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        f__lz = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__z' % L), mask=~gal_sample__z) for L in lines}
        O3Hb__z = np.ma.log10(f__lz['5007']/f__lz['4861'])
        N2Ha__z = np.ma.log10(f__lz['6583']/f__lz['6563'])
        S2Ha__z = np.ma.log10((f__lz['6717'] + f__lz['6731'])/f__lz['6563'])

        N_cols = 3
        N_rows = 1
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        cmap = cmap_discrete(colors=colors_DIG_COMP_SF)
        ax1, ax2, ax3 = axArr
        f.suptitle(r'%s - %s (%s): %d pixels (%d zones)' % (califaID, mto, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))

        # AXIS 1
        map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(N2Ha__z, O3Hb__z)
        sc = ax1.scatter(xm, ym, c=map__z, vmin=1, vmax=4, cmap=cmap, marker='o', s=20, edgecolor='none')
        ax1.set_xlim(extent[0:2])
        ax1.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax1, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax1, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
        ax1.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax1.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        ax1.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        plot_text_ax(ax1, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=35)  # 44.62)
        ax1.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
        L.fixCF10('S06')
        # AXIS 2
        map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(S2Ha__z, O3Hb__z)
        sc2 = ax2.scatter(xm, ym, c=map__z, vmin=1, vmax=4, cmap=cmap, marker='o', s=20, edgecolor='none')
        ax2.set_xlim(extent[0:2])
        ax2.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax2, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax2, 'K01', 0.60, 0.02, 20, 'bottom', 'right', 'k')
        ax2.plot(L.x['K01S2'], L.y['K01S2'], 'k-', label='K01')
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax2, txt, 0.98, 0.98, 16, 'top', 'right', color='k')
        # AXIS 3
        # map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        # extent = [-2.5, 0, -1.5, 1.5]
        # xm, ym = ma_mask_xyz(O1Ha__z, O3Hb__z)
        # sc = ax3.scatter(xm, ym, c=map__z, vmin=1, vmax=4, cmap=cmap, marker='o', s=20, edgecolor='none')
        # ax3.set_xlim(extent[0:2])
        # ax3.set_ylim(extent[2:4])
        # N = xm.count()
        # c = ''
        # if (xm.compressed() < extent[0]).any():
        #     c += 'x-'
        # if (xm.compressed() > extent[1]).any():
        #     c += 'x+'
        # if (ym.compressed() < extent[2]).any():
        #     c += 'y-'
        # if (ym.compressed() > extent[3]).any():
        #     c += 'y+'
        # plot_text_ax(ax3, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        # txt = 'CALIFA %s' % califaID[1:]
        # plot_text_ax(ax3, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
        # plot_text_ax(ax3, 'K01', 0.65, 0.02, 20, 'bottom', 'right', 'k')
        # ax3.plot(L.x['K01OI'], L.y['K01OI'], 'k-', label='K01')
        ax1.set_ylabel(r'$\log\ [OIII]/H\beta$')
        ax1.set_xlabel(r'$\log\ [NII]/H\alpha$')
        ax2.set_xlabel(r'$\log\ [SII]/H\alpha$')
        ax3.set_xlabel(r'$\log\ [OI]/H\alpha$')
        cbaxes = add_subplot_axes(ax3, [0.7, 0.93, 0.6, 0.07])
        divisions = np.linspace(1, 4, 5)
        ticks = (divisions[1:] - divisions[0:-1])/2. + divisions[0:-1]
        cb = plt.colorbar(sc, cax=cbaxes, ticks=ticks, orientation='horizontal')
        cb.ax.set_xticklabels(classif_labels, fontsize=12)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        ax2.xaxis.set_minor_locator(minorLocator)
        ax2.yaxis.set_minor_locator(minorLocator)
        ax3.xaxis.set_minor_locator(minorLocator)
        ax3.yaxis.set_minor_locator(minorLocator)
        # L = Lines()
        # L.addLine('K01S2', L.linebpt, (1.30, 0.72, -0.32), np.linspace(-10.0, 0.32, L.xn + 1)[:-1])
        # L.addLine('K01OI', L.linebpt, (1.33, 0.73, 0.59), np.linspace(-10.0, -0.59, L.xn + 1)[:-1])
        # mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        # N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        # N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        # N_pixel = N_x * N_y
        # N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        # f__lz = {'%s' % L: ALL.get_gal_prop(califaID, 'f%s__z' % L) for L in lines}
        # O3Hb__z = np.ma.log10(f__lz['5007']/f__lz['4861'])
        # N2Ha__z = np.ma.log10(f__lz['6583']/f__lz['6563'])
        # S2Ha__z = np.ma.log10((f__lz['6717'] + f__lz['6731'])/f__lz['6563'])
        # O1Ha__z = np.ma.log10(f__lz['6300']/f__lz['6563'])
        #
        # N_cols = 3
        # N_rows = 1
        # f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        # cmap = cmap_discrete(colors=colors_DIG_COMP_SF)
        # ax1, ax2, ax3 = axArr
        # f.suptitle(r'%s - %s (%s): %d pixels (%d zones)' % (califaID, mto, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))
        # map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        #
        # # AXIS 1
        # extent = [-1.5, 1, -1.5, 1.5]
        # xm, ym = ma_mask_xyz(N2Ha__z, O3Hb__z)
        # sc = ax1.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        # ax1.set_xlim(extent[0:2])
        # ax1.set_ylim(extent[2:4])
        # ax1.set_xlabel(r'$\log\ [NII]/H\alpha$')
        # ax1.set_ylabel(r'$\log\ [OIII]/H\beta$')
        # N = xm.count()
        # c = ''
        # if (xm.compressed() < extent[0]).any():
        #     c += 'x-'
        # if (xm.compressed() > extent[1]).any():
        #     c += 'x+'
        # if (ym.compressed() < extent[2]).any():
        #     c += 'y-'
        # if (ym.compressed() > extent[3]).any():
        #     c += 'y+'
        # plot_text_ax(ax1, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        # plot_text_ax(ax1, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        # plot_text_ax(ax1, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
        # plot_text_ax(ax1, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
        # ax1.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        # ax1.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        # ax1.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        # plot_text_ax(ax1, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=35)  # 44.62)
        # ax1.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
        # L.fixCF10('S06')
        #
        # # AXIS 2
        # extent = [-1.5, 1, -1.5, 1.5]
        # xm, ym = ma_mask_xyz(S2Ha__z, O3Hb__z)
        # sc = ax2.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        # ax2.set_xlim(extent[0:2])
        # ax2.set_ylim(extent[2:4])
        # ax2.set_xlabel(r'$\log\ [SII]/H\alpha$')
        # N = xm.count()
        # c = ''
        # if (xm.compressed() < extent[0]).any():
        #     c += 'x-'
        # if (xm.compressed() > extent[1]).any():
        #     c += 'x+'
        # if (ym.compressed() < extent[2]).any():
        #     c += 'y-'
        # if (ym.compressed() > extent[3]).any():
        #     c += 'y+'
        # plot_text_ax(ax2, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        # plot_text_ax(ax2, 'K01', 0.75, 0.02, 20, 'bottom', 'right', 'k')
        # ax2.plot(L.x['K01S2'], L.y['K01S2'], 'k-', label='K01')
        #
        # # AXIS 3
        # extent = [-2.5, 0, -1.5, 1.5]
        # xm, ym = ma_mask_xyz(O1Ha__z, O3Hb__z)
        # sc = ax3.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        # ax3.set_xlim(extent[0:2])
        # ax3.set_ylim(extent[2:4])
        # cbaxes = add_subplot_axes(ax3, [0.8, 0.91, 0.5, 0.04])
        # cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        # cb.ax.set_xticklabels(classif_labels, fontsize=9)
        # ax3.set_xlabel(r'$\log\ [OI]/H\alpha$')
        # N = xm.count()
        # c = ''
        # if (xm.compressed() < extent[0]).any():
        #     c += 'x-'
        # if (xm.compressed() > extent[1]).any():
        #     c += 'x+'
        # if (ym.compressed() < extent[2]).any():
        #     c += 'y-'
        # if (ym.compressed() > extent[3]).any():
        #     c += 'y+'
        # plot_text_ax(ax3, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        # plot_text_ax(ax3, 'K01', 0.80, 0.02, 20, 'bottom', 'right', 'k')
        # ax3.plot(L.x['K01OI'], L.y['K01OI'], 'k-', label='K01')
        #
        # plt.setp(ax2.get_yticklabels(), visible=False)
        # plt.setp(ax3.get_yticklabels(), visible=False)
        #
        # ax1.xaxis.set_minor_locator(minorLocator)
        # ax1.yaxis.set_minor_locator(minorLocator)
        # ax2.xaxis.set_minor_locator(minorLocator)
        # ax2.yaxis.set_minor_locator(minorLocator)
        # ax3.xaxis.set_minor_locator(minorLocator)
        # ax3.yaxis.set_minor_locator(minorLocator)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('fig4-%s.png' % califaID)
        plt.close(f)


def fig4_3gals(ALL, sel, gals, suffix):
    sel_sample__gz = sel['gals_sample__z']

    N_cols = 3
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
    cmap = cmap_discrete(colors=colors_DIG_COMP_SF)

    row = 0
    for califaID in gals:
        ax1, ax2, ax3 = axArr[row]

        L = Lines()
        L.addLine('K01S2', L.linebpt, (1.30, 0.72, -0.32), np.linspace(-10.0, 0.32, L.xn + 1)[:-1])
        L.addLine('K01OI', L.linebpt, (1.33, 0.73, 0.59), np.linspace(-10.0, -0.59, L.xn + 1)[:-1])
        gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        f__lz = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__z' % L), mask=~gal_sample__z) for L in lines}
        O3Hb__z = np.ma.log10(f__lz['5007']/f__lz['4861'])
        N2Ha__z = np.ma.log10(f__lz['6583']/f__lz['6563'])
        S2Ha__z = np.ma.log10((f__lz['6717'] + f__lz['6731'])/f__lz['6563'])
        print califaID
        print f__lz['6717']
        print f__lz['6731']
        # print O3Hb__z
        # print N2Ha__z
        # O1Ha__z = np.ma.log10(f__lz['6300']/f__lz['6563'])

        # AXIS 1
        map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(N2Ha__z, O3Hb__z)
        sc = ax1.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        ax1.set_xlim(extent[0:2])
        ax1.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax1, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax1, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
        ax1.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax1.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        ax1.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        plot_text_ax(ax1, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=35)  # 44.62)
        ax1.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
        L.fixCF10('S06')
        # AXIS 2
        map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(S2Ha__z, O3Hb__z)
        sc = ax2.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        ax2.set_xlim(extent[0:2])
        ax2.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax2, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax2, 'K01', 0.60, 0.02, 20, 'bottom', 'right', 'k')
        ax2.plot(L.x['K01S2'], L.y['K01S2'], 'k-', label='K01')
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax2, txt, 0.98, 0.98, 16, 'top', 'right', color='k')
        # AXIS 3
        # map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        # extent = [-2.5, 0, -1.5, 1.5]
        # xm, ym = ma_mask_xyz(O1Ha__z, O3Hb__z)
        # sc = ax3.scatter(xm, ym, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        # ax3.set_xlim(extent[0:2])
        # ax3.set_ylim(extent[2:4])
        # N = xm.count()
        # c = ''
        # if (xm.compressed() < extent[0]).any():
        #     c += 'x-'
        # if (xm.compressed() > extent[1]).any():
        #     c += 'x+'
        # if (ym.compressed() < extent[2]).any():
        #     c += 'y-'
        # if (ym.compressed() > extent[3]).any():
        #     c += 'y+'
        # plot_text_ax(ax3, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        # txt = 'CALIFA %s' % califaID[1:]
        # plot_text_ax(ax3, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')
        # plot_text_ax(ax3, 'K01', 0.65, 0.02, 20, 'bottom', 'right', 'k')
        # ax3.plot(L.x['K01OI'], L.y['K01OI'], 'k-', label='K01')
        if row < 2:
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax3.get_xticklabels(), visible=False)
        if row == 1:
            ax1.set_ylabel(r'$\log\ [OIII]/H\beta$')
        if row == 2:
            ax1.set_xlabel(r'$\log\ [NII]/H\alpha$')
            ax2.set_xlabel(r'$\log\ [SII]/H\alpha$')
            ax3.set_xlabel(r'$\log\ [OI]/H\alpha$')
            cbaxes = add_subplot_axes(ax3, [0.7, 0.99, 0.6, 0.07])
            cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
            cb.ax.set_xticklabels(classif_labels, fontsize=12)

        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        ax2.xaxis.set_minor_locator(minorLocator)
        ax2.yaxis.set_minor_locator(minorLocator)
        ax3.xaxis.set_minor_locator(minorLocator)
        ax3.yaxis.set_minor_locator(minorLocator)
        row += 1
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.tight_layout(w_pad=0, h_pad=0)
    f.savefig('fig4_%s.png' % suffix, dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


def fig5(ALL, gals=None):
    sel_WHa_DIG__gz = (ALL.W6563__z < DIG_WHa_threshold).filled(False)
    sel_WHa_COMP__gz = np.bitwise_and((ALL.W6563__z >= DIG_WHa_threshold).filled(False), (ALL.W6563__z < SF_WHa_threshold).filled(False))
    sel_WHa_SF__gz = (ALL.W6563__z >= SF_WHa_threshold).filled(False)

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]

    sel_gals__gz = np.zeros((ALL.califaID__z.shape), dtype='bool')

    N_gals = 0
    for g in gals:
        where_gals__gz = np.where(ALL.califaID__z == g)
        if not where_gals__gz:
            continue
        sel_gals__gz[where_gals__gz] = True
        N_gals += 1

    if (sel_gals__gz).any():
        W6563__gz = ALL.W6563__z[sel_gals__gz]

        O3Hb__gz = np.ma.log10(ALL.f5007__z/ALL.f4861__z)
        N2Ha__gz = np.ma.log10(ALL.f6583__z/ALL.f6563__z)
        S2Ha__gz = np.ma.log10((ALL.f6717__z+ALL.f6731__z)/ALL.f6563__z)
        OIHa__gz = np.ma.log10(ALL.f6300__z/ALL.f6563__z)
        L = Lines()
        L.addLine('K01S2', L.linebpt, (1.30, 0.72, -0.32), np.linspace(-10.0, 0.32, L.xn + 1)[:-1])
        L.addLine('K01OI', L.linebpt, (1.33, 0.73, 0.59), np.linspace(-10.0, -0.59, L.xn + 1)[:-1])

        classif = np.ma.masked_all(W6563__gz.shape)
        classif[sel_WHa_DIG__gz] = 1
        classif[sel_WHa_COMP__gz] = 2
        classif[sel_WHa_SF__gz] = 3

        N_cols = 3
        N_rows = 1
        # f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
        f = plt.figure(dpi=200, figsize=(6, 5))
        # ax1, ax2, ax3 = axArr
        ax1 = f.gca()
        cmap = cmap_discrete(colors_DIG_COMP_SF)
        # AXIS 1
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(N2Ha__gz, O3Hb__gz)
        # sc = ax1.scatter(xm, ym, c=np.ma.log10(W6563__gz), vmin=np.log10(DIG_WHa_threshold), vmax=np.log10(SF_WHa_threshold), cmap='rainbow_r', s=1, marker='o', edgecolor='none')
        sc = ax1.scatter(xm, ym, c=classif, cmap=cmap, s=1, marker='o', edgecolor='none')
        ax1.set_xlim(extent[0:2])
        ax1.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        plot_text_ax(ax1, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        plot_text_ax(ax1, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        # plot_text_ax(ax1, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K03', 0.55, 0.02, 20, 'bottom', 'left', 'k')
        # plot_text_ax(ax1, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
        ax1.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax1.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        # ax1.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
        # plot_text_ax(ax1, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=35)  # 44.62)
        # ax1.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
        # L.fixCF10('S06')
        # # AXIS 2
        # extent = [-1.5, 1, -1.5, 1.5]
        # xm, ym = ma_mask_xyz(S2Ha__gz, O3Hb__gz)
        # # sc = ax2.scatter(xm, ym, c=np.ma.log10(W6563__gz), vmin=np.log10(DIG_WHa_threshold), vmax=np.log10(SF_WHa_threshold), cmap='rainbow_r', s=1, marker='o', edgecolor='none')
        # sc = ax2.scatter(xm, ym, c=classif, cmap=cmap, s=1, marker='o', edgecolor='none')
        # ax2.set_xlim(extent[0:2])
        # ax2.set_ylim(extent[2:4])
        # N = xm.count()
        # c = ''
        # if (xm.compressed() < extent[0]).any():
        #     c += 'x-'
        # if (xm.compressed() > extent[1]).any():
        #     c += 'x+'
        # if (ym.compressed() < extent[2]).any():
        #     c += 'y-'
        # if (ym.compressed() > extent[3]).any():
        #     c += 'y+'
        # plot_text_ax(ax2, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        # plot_text_ax(ax2, 'K01', 0.75, 0.02, 20, 'bottom', 'right', 'k')
        # ax2.plot(L.x['K01S2'], L.y['K01S2'], 'k-', label='K01')
        # # AXIS 3
        # extent = [-2.5, 0, -1.5, 1.5]
        # xm, ym = ma_mask_xyz(OIHa__gz, O3Hb__gz)
        # # sc = ax3.scatter(xm, ym, c=np.ma.log10(W6563__gz), vmin=np.log10(DIG_WHa_threshold), vmax=np.log10(SF_WHa_threshold), cmap='rainbow_r', s=1, marker='o', edgecolor='none')
        # sc = ax3.scatter(xm, ym, c=classif, cmap=cmap, s=1, marker='o', edgecolor='none')
        # ax3.set_xlim(extent[0:2])
        # ax3.set_ylim(extent[2:4])
        # N = xm.count()
        # c = ''
        # if (xm.compressed() < extent[0]).any():
        #     c += 'x-'
        # if (xm.compressed() > extent[1]).any():
        #     c += 'x+'
        # if (ym.compressed() < extent[2]).any():
        #     c += 'y-'
        # if (ym.compressed() > extent[3]).any():
        #     c += 'y+'
        # plot_text_ax(ax3, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
        # plot_text_ax(ax3, 'K01', 0.80, 0.02, 20, 'bottom', 'right', 'k')
        # ax3.plot(L.x['K01OI'], L.y['K01OI'], 'k-', label='K01')
        ax1.set_ylabel(r'$\log\ [OIII]/H\beta$')
        ax1.set_xlabel(r'$\log\ [NII]/H\alpha$')
        # ax2.set_xlabel(r'$\log\ [SII]/H\alpha$')
        # ax3.set_xlabel(r'$\log\ [OI]/H\alpha$')
        # cbaxes = add_subplot_axes(ax1, [0.75, 0.95, 0.5, 0.05])
        cbaxes = add_subplot_axes(ax1, [0.51, 0.95, 0.5, 0.05])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        cb.ax.xaxis.labelpad = -29
        cb.ax.set_xticklabels(classif_labels, fontsize=12)
        # cb.ax.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]', fontsize=11)
        # plt.setp(ax2.get_yticklabels(), visible=False)
        # plt.setp(ax3.get_yticklabels(), visible=False)
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        # ax2.xaxis.set_minor_locator(minorLocator)
        # ax2.yaxis.set_minor_locator(minorLocator)
        # ax3.xaxis.set_minor_locator(minorLocator)
        # ax3.yaxis.set_minor_locator(minorLocator)

        f.tight_layout()
        f.savefig('fig5.png')
        plt.close(f)


def fig5_3panels(ALL, sel, gals):
    sel_gals_sample__gz = sel['gals_sample__z']
    sel_gals_sample__gyx = sel['gals_sample__yx']

    sel_tmp__gz = np.zeros_like(sel_gals_sample__gz, dtype='bool')
    sel_tmp__gyx = np.zeros_like(sel_gals_sample__gyx, dtype='bool')

    O3Hb__g = np.ma.masked_all(len(gals), dtype='float')
    N2Ha__g = np.ma.masked_all(len(gals), dtype='float')

    for i_g, califaID in enumerate(gals):
        sel_gal__gz = np.bitwise_and(ALL.califaID__z == califaID, sel_gals_sample__gz)
        sel_gal__gyx = np.bitwise_and(ALL.califaID__yx == califaID, sel_gals_sample__gyx)
        f4861__z = ALL.get_gal_prop(califaID, ALL.f4861__z)
        f5007__z = ALL.get_gal_prop(califaID, ALL.f5007__z)
        f6563__z = ALL.get_gal_prop(califaID, ALL.f6563__z)
        f6583__z = ALL.get_gal_prop(califaID, ALL.f6583__z)
        O3Hb__g[i_g] = np.ma.log10(f5007__z.sum()/f4861__z.sum())
        N2Ha__g[i_g] = np.ma.log10(f6583__z.sum()/f6563__z.sum())

    if (sel_gals_sample__gz).any():
        f__lgz = {'%s' % l: np.ma.masked_array(getattr(ALL, 'f%s__z' % l), mask=~sel_gals_sample__gz) for l in lines}
        sel_WHa_HIG__gz = np.bitwise_and(sel['WHa']['HIG']['z'], sel_gals_sample__gz)
        sel_WHa_DIG__gz = np.bitwise_and(sel['WHa']['DIG']['z'], sel_gals_sample__gz)
        sel_WHa_COMP__gz = np.bitwise_and(sel['WHa']['COMP']['z'], sel_gals_sample__gz)
        sel_WHa_SF__gz = np.bitwise_and(sel['WHa']['SF']['z'], sel_gals_sample__gz)
        W6563__gz = np.ma.masked_array(ALL.W6563__z, mask=~sel_gals_sample__gz)

        O3Hb__gz = np.ma.log10(f__lgz['5007']/f__lgz['4861'])
        N2Ha__gz = np.ma.log10(f__lgz['6583']/f__lgz['6563'])
        S2Ha__gz = np.ma.log10((f__lgz['6717']+f__lgz['6731'])/f__lgz['6563'])
        L = Lines()
        L.addLine('K01S2', L.linebpt, (1.30, 0.72, -0.32), np.linspace(-10.0, 0.32, L.xn + 1)[:-1])
        L.addLine('K01OI', L.linebpt, (1.33, 0.73, 0.59), np.linspace(-10.0, -0.59, L.xn + 1)[:-1])

        classif = np.ma.masked_all(W6563__gz.shape)
        classif[sel_WHa_HIG__gz] = 1
        classif[sel_WHa_DIG__gz] = 2
        classif[sel_WHa_COMP__gz] = 3
        classif[sel_WHa_SF__gz] = 4

        N_cols = 1
        N_rows = 3
        f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4))
        ax2, ax1, ax3 = axArr
        cmap = cmap_discrete(colors_DIG_COMP_SF)
        # AXIS 1
        extent = [-1.5, 1, -1.5, 1.5]
        xm, ym = ma_mask_xyz(N2Ha__gz, O3Hb__gz)
        sc = ax1.scatter(xm, ym, c=classif, cmap=cmap, s=1, marker='o', edgecolor='none')
        ax1.set_xlim(extent[0:2])
        ax1.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        ax1.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax1.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        plot_text_ax(ax1, '%d %s' % (N, c), 0.01, 0.99, 15, 'top', 'left', 'k')
        plot_text_ax(ax1, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'K03', 0.55, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax1, 'b)', 0.02, 0.02, 16, 'bottom', 'left', 'k')
        # cbaxes = add_subplot_axes(ax1, [0.54, 0.99, 0.46, 0.06])
        cbaxes = add_subplot_axes(ax1, [0.51, 1.06, 0.5, 0.06])
        divisions = np.linspace(1, 4, 5)
        ticks = (divisions[1:] - divisions[0:-1])/2. + divisions[0:-1]
        cb = plt.colorbar(sc, cax=cbaxes, ticks=ticks, orientation='horizontal')
        cb.ax.set_xticklabels(classif_labels, fontsize=12)
        # AXIS 2
        xm, ym = ma_mask_xyz(N2Ha__gz, O3Hb__gz)
        # sc = ax2.scatter(xm, ym, c=np.ma.log10(W6563__gz), vmin=np.log10(DIG_WHa_threshold), vmax=np.log10(SF_WHa_threshold), cmap='rainbow_r', s=1, marker='o', edgecolor='none')
        sc = ax2.scatter(xm, ym, c=np.ma.log10(W6563__gz), vmin=logWHa_range[0], vmax=logWHa_range[1], cmap='rainbow_r', s=1, marker='o', edgecolor='none')
        ax2.set_xlim(extent[0:2])
        ax2.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        ax2.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax2.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        plot_text_ax(ax2, '%d %s' % (N, c), 0.01, 0.99, 15, 'top', 'left', 'k')
        plot_text_ax(ax2, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax2, 'K03', 0.55, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax2, 'a)', 0.02, 0.02, 16, 'bottom', 'left', 'k')
        # cbaxes = add_subplot_axes(ax2, [0.51, 0.99, 0.49, 0.06])
        cbaxes = add_subplot_axes(ax2, [0.51, 1.06, 0.47, 0.06])
        # cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.0, 1.1, 1.2, 1.3], orientation='horizontal')
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[0, 1.25, 2.5], orientation='horizontal')
        cb.ax.xaxis.labelpad = -28
        cb.ax.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]', fontsize=11)

        xm, ym = ma_mask_xyz(N2Ha__g, O3Hb__g)
        # sc = ax2.scatter(xm, ym, c=np.ma.log10(W6563__gz), vmin=np.log10(DIG_WHa_threshold), vmax=np.log10(SF_WHa_threshold), cmap='rainbow_r', s=1, marker='o', edgecolor='none')
        extent = [-1.5, 1, -1.5, 1.5]
        sc = ax3.scatter(xm, ym, s=20, marker='*', edgecolor='none')
        print xm.count()
        ax3.set_xlim(extent[0:2])
        ax3.set_ylim(extent[2:4])
        N = xm.count()
        c = ''
        if (xm.compressed() < extent[0]).any():
            c += 'x-'
        if (xm.compressed() > extent[1]).any():
            c += 'x+'
        if (ym.compressed() < extent[2]).any():
            c += 'y-'
        if (ym.compressed() > extent[3]).any():
            c += 'y+'
        ax3.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
        ax3.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
        plot_text_ax(ax3, '%d %s' % (N, c), 0.01, 0.99, 15, 'top', 'left', 'k')
        plot_text_ax(ax3, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax3, 'K03', 0.55, 0.02, 20, 'bottom', 'left', 'k')
        plot_text_ax(ax3, 'c)', 0.02, 0.02, 16, 'bottom', 'left', 'k')
        # cbaxes = add_subplot_axes(ax3, [0.51, 0.99, 0.49, 0.06])
        # cbaxes = add_subplot_axes(ax3, [0.51, 1.06, 0.47, 0.06])
        # cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.0, 1.1, 1.2, 1.3], orientation='horizontal')

        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        ax2.xaxis.set_minor_locator(minorLocator)
        ax2.yaxis.set_minor_locator(minorLocator)
        ax3.xaxis.set_minor_locator(minorLocator)
        ax3.yaxis.set_minor_locator(minorLocator)

        ax1.set_ylabel(r'$\log\ [OIII]/H\beta$')
        ax3.set_xlabel(r'$\log\ [NII]/H\alpha$')

        f.tight_layout()
        f.savefig('fig5_3panels.png', dpi=dpi_choice, transparent=transp_choice)
        plt.close(f)


def fig6(ALL, sel, gals=None):
    sel_gals_sample__gz = sel['gals_sample__z']
    sel_WHa = sel['WHa']

    if gals is None:
        _, ind = np.unique(ALL.califaID__z, return_index=True)
        gals = ALL.califaID__z[sorted(ind)]
    for califaID in gals:
        if califaID not in ALL.califaID__z:
            print 'Fig_1: %s not in sample pickle' % califaID
            continue

        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        gal_sample__z = ALL.get_gal_prop(califaID, sel_gals_sample__gz)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        distance_HLR__z = ALL.get_gal_prop(califaID, ALL.zoneDistance_HLR)
        f__lz = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__z' % L), mask=~gal_sample__z) for L in lines}

        f = plt.figure(dpi=200, figsize=(6, 5))
        cmap = cmap_discrete(colors=colors_DIG_COMP_SF)
        f.suptitle(r'%s - %s (%s): %d pixels (%d zones)' % (califaID, mto, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone), fontsize=12)

        ax1 = f.gca()
        x = distance_HLR__z
        y = (1./(q[0] - q[1])) * (np.ma.log(f__lz['6563']/f__lz['4861']) - np.log(2.86))
        x_range = distance_range
        y_range = (1/0.34652) * np.log(10**(lineratios_range['6563/4861'][0])/2.86), (1/0.34652) * np.log(10**(lineratios_range['6563/4861'][1])/2.86)
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa, sel_gals_sample__gz)
        sc = ax1.scatter(x, y, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')

        sel_HIG__z, sel_DIG__z, sel_COMP__z, sel_SF__z, _ = get_selections_zones(ALL, califaID, sel_WHa, sel_gals_sample__gz)

        min_npts = 4
        xm, ym = ma_mask_xyz(x, y, mask=~sel_HIG__z)
        yMean_HIG, yPrc_HIG, bin_center_HIG, npts_HIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_HIG.any():
            sel = npts_HIG > min_npts
            ax1.plot(bin_center_HIG[sel], yPrc_HIG[2][sel], '-', lw=2, c=colors_lines_DIG_COMP_SF[0])
            ax1.plot(bin_center_HIG[sel], yPrc_HIG[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_DIG__z)
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel = npts_DIG > min_npts
            ax1.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], '-', lw=2, c=colors_lines_DIG_COMP_SF[1])
            ax1.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_COMP__z)
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel = npts_COMP > min_npts
            ax1.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], '-', lw=2, c=colors_lines_DIG_COMP_SF[2])
            ax1.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[2], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_SF__z)
        yMean_SF, yPrc_SF, bin_center_SF, npts_SF = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_SF.any():
            sel = npts_SF > min_npts
            ax1.plot(bin_center_SF[sel], yPrc_SF[2][sel], 'k-', lw=2, c=colors_lines_DIG_COMP_SF[3])
            ax1.plot(bin_center_SF[sel], yPrc_SF[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[3], markersize=10)

        ax2 = ax1.twinx()
        mn, mx = ax1.get_ylim()
        unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
        ax2.set_ylim(unit_converter(mn), unit_converter(mx))
        # ax2.set_ylim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))

        ax1.set_xlabel('R [HLR]')
        cbaxes = add_subplot_axes(ax1, [0.65, -0.05, 0.35, 0.07])
        divisions = np.linspace(1, 4, 5)
        ticks = (divisions[1:] - divisions[0:-1])/2. + divisions[0:-1]
        cb = plt.colorbar(sc, cax=cbaxes, ticks=ticks, orientation='horizontal')
        cb.ax.set_xticklabels(classif_labels, fontsize=12)
        ax1.set_ylabel(r'$\tau_V^{neb}$')
        ax2.set_ylabel(r'$\log\ H\alpha/H\beta$')

        # mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        # N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        # N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        # N_pixel = N_x * N_y
        # N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        # distance_HLR__z = ALL.get_gal_prop(califaID, ALL.zoneDistance_HLR)
        # f__lz = {'%s' % L: ALL.get_gal_prop(califaID, 'f%s__z' % L) for L in lines}
        #
        # f = plt.figure(dpi=200, figsize=(6, 5))
        # cmap = cmap_discrete(colors=colors_DIG_COMP_SF)
        # f.suptitle(r'%s - %s (%s): %d pixels (%d zones)' % (califaID, mto, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))
        #
        # ax1 = f.gca()
        # x = distance_HLR__z
        # y = np.ma.log10(f__lz['6563']/f__lz['4861'])
        # x_range = distance_range
        # y_range = lineratios_range['6563/4861']
        # map__z = create_segmented_map_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_SF__z)
        # sc = ax1.scatter(x, y, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        # cbaxes = add_subplot_axes(ax1, [0.59, 0.91, 0.35, 0.04])
        # cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        # cb.ax.set_xticklabels(classif_labels, fontsize=9)
        # ax1.set_ylabel(r'$\log\ H\alpha/H\beta$')
        # ax1.set_xlabel('R [HLR]')
        # ax1.set_xlim(x_range)
        # ax1.set_ylim(y_range)
        # ax1.xaxis.set_minor_locator(minorLocator)
        # ax1.yaxis.set_minor_locator(minorLocator)
        #
        # sel_DIG__z, sel_COMP__z, sel_SF__z = get_selections_zones(ALL, califaID, sel_WHa_DIG__z, sel_WHa_COMP__z, sel_WHa_SF__z)
        #
        # min_npts = 4
        # xm, ym = ma_mask_xyz(x, y, mask=~sel_DIG__z)
        # yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        # if npts_DIG.any():
        #     sel = npts_DIG > min_npts
        #     ax1.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_SF[0])
        #     ax1.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[0], markersize=10)
        # xm, ym = ma_mask_xyz(x, y, mask=~sel_COMP__z)
        # yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        # if npts_COMP.any():
        #     sel = npts_COMP > min_npts
        #     ax1.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], '--', lw=2, c=colors_lines_DIG_COMP_SF[1])
        #     ax1.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[1], markersize=10)
        # xm, ym = ma_mask_xyz(x, y, mask=~sel_SF__z)
        # yMean_SF, yPrc_SF, bin_center_SF, npts_SF = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        # if npts_SF.any():
        #     sel = npts_SF > min_npts
        #     ax1.plot(bin_center_SF[sel], yPrc_SF[2][sel], 'k--', lw=2, c=colors_lines_DIG_COMP_SF[2])
        #     ax1.plot(bin_center_SF[sel], yPrc_SF[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[2], markersize=10)
        #
        # ax2 = ax1.twinx()
        # mn, mx = ax1.get_ylim()
        # ax2.set_ylim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))
        # ax2.set_ylabel(r'$\tau_V$')
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig('fig6-%s.png' % califaID)
        plt.close(f)


def fig6_3gals(ALL, sel, gals, suffix):
    sel_gals_sample__gz = sel['gals_sample__z']
    sel_WHa = sel['WHa']

    N_cols = 1
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 6, N_rows * 4))
    cmap = cmap_discrete(colors=colors_DIG_COMP_SF)

    row = 0
    for califaID in gals:
        ax1 = axArr[row]

        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        gal_sample__z = ALL.get_gal_prop(califaID, sel_gals_sample__gz)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        distance_HLR__z = ALL.get_gal_prop(califaID, ALL.zoneDistance_HLR)
        f__lz = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__z' % L), mask=~gal_sample__z) for L in lines}

        x = distance_HLR__z
        y = (1./(q[0] - q[1])) * (np.ma.log(f__lz['6563']/f__lz['4861']) - np.log(2.86))
        x_range = distance_range
        y_range = (1/0.34652) * np.log(10**(lineratios_range['6563/4861'][0])/2.86), (1/0.34652) * np.log(10**(lineratios_range['6563/4861'][1])/2.86)
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa, sel_gals_sample__gz)
        sc = ax1.scatter(x, y, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=10, edgecolor='none')
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')

        sel_DIG__z, sel_COMP__z, sel_SF__z, _ = get_selections_zones(ALL, califaID, sel_WHa, sel_gals_sample__gz)

        min_npts = 4
        xm, ym = ma_mask_xyz(x, y, mask=~sel_DIG__z)
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel = npts_DIG > min_npts
            ax1.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], '-', lw=2, c=colors_lines_DIG_COMP_SF[0])
            ax1.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_COMP__z)
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            sel = npts_COMP > min_npts
            ax1.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], '-', lw=2, c=colors_lines_DIG_COMP_SF[1])
            ax1.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_SF__z)
        yMean_SF, yPrc_SF, bin_center_SF, npts_SF = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_SF.any():
            sel = npts_SF > min_npts
            ax1.plot(bin_center_SF[sel], yPrc_SF[2][sel], 'k-', lw=2, c=colors_lines_DIG_COMP_SF[2])
            ax1.plot(bin_center_SF[sel], yPrc_SF[2][sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[2], markersize=10)

        ax2 = ax1.twinx()
        mn, mx = ax1.get_ylim()
        unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
        ax2.set_ylim(unit_converter(mn), unit_converter(mx))
        # ax2.set_ylim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))

        if row < 2:
            plt.setp(ax1.get_xticklabels(), visible=False)
        if row == 2:
            ax1.set_xlabel('R [HLR]')
            cbaxes = add_subplot_axes(ax1, [0.08, -0.05, 0.35, 0.07])
            cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
            cb.ax.set_xticklabels(classif_labels, fontsize=12)
        if row == 1:
            ax1.set_ylabel(r'$\tau_V^{neb}$')
            ax2.set_ylabel(r'$\log\ H\alpha/H\beta$')
        row += 1

    f.tight_layout(h_pad=0)
    f.savefig('fig6_%s.png' % suffix, dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


def fig7(ALL, sel, gals):
    sel_WHa = sel['WHa']
    sel_gals_sample__gz = sel['gals_sample__z']
    sel_gals_sample__gyx = sel['gals_sample__yx']
    N_gals = len(gals)

    if (sel_gals_sample__gz).any():

        tau_V_neb = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_npts = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_HIG = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_HIG_npts = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_DIG = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_DIG_npts = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_SF = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_SF_npts = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_GAL = np.ma.masked_all((N_gals, N_R_bins))
        tau_V_neb_sumGAL = np.ma.masked_all((N_gals, N_R_bins))
        delta_tau = np.ma.masked_all((N_gals, N_R_bins))
        delta_tau_norm_GAL = np.ma.masked_all((N_gals, N_R_bins))
        delta_tau_norm_sumGAL = np.ma.masked_all((N_gals, N_R_bins))
        delta_tau_HIG = np.ma.masked_all((N_gals, N_R_bins))
        delta_tau_norm_HIG_GAL = np.ma.masked_all((N_gals, N_R_bins))
        delta_tau_norm_HIG_sumGAL = np.ma.masked_all((N_gals, N_R_bins))

        for i_g, califaID in enumerate(gals):
            if califaID not in ALL.califaID__z:
                print 'Fig_1: %s not in sample pickle' % califaID
                continue
            print califaID
            HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
            pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
            ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
            x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
            y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
            N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
            N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
            gal_sample__z = ALL.get_gal_prop(califaID, sel_gals_sample__gz)
            gal_sample__yx = ALL.get_gal_prop(califaID, sel_gals_sample__gyx).reshape(N_y, N_x)
            f__lz = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__z' % L), mask=~gal_sample__z) for L in lines}
            f__lyx = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__yx' % L).reshape(N_y, N_x), mask=~gal_sample__yx) for L in lines}
            sel_HIG__yx, sel_DIG__yx, sel_COMP__yx, sel_SF__yx, _ = get_selections(ALL, califaID, sel_WHa, sel_gals_sample__gyx)
            mean_f4861__r = radialProfile(f__lyx['4861'], R_bin__r, x0, y0, pa, ba, HLR_pix, gal_sample__yx, 'mean')
            mean_f6563__r = radialProfile(f__lyx['6563'], R_bin__r, x0, y0, pa, ba, HLR_pix, gal_sample__yx, 'mean')
            mean_f4861_HIG__r = radialProfile(f__lyx['4861'], R_bin__r, x0, y0, pa, ba, HLR_pix, sel_HIG__yx, 'mean')
            mean_f6563_HIG__r = radialProfile(f__lyx['6563'], R_bin__r, x0, y0, pa, ba, HLR_pix, sel_HIG__yx, 'mean')
            mean_f4861_DIG__r = radialProfile(f__lyx['4861'], R_bin__r, x0, y0, pa, ba, HLR_pix, sel_DIG__yx, 'mean')
            mean_f6563_DIG__r = radialProfile(f__lyx['6563'], R_bin__r, x0, y0, pa, ba, HLR_pix, sel_DIG__yx, 'mean')
            mean_f4861_SF__r = radialProfile(f__lyx['4861'], R_bin__r, x0, y0, pa, ba, HLR_pix, sel_SF__yx, 'mean')
            mean_f6563_SF__r = radialProfile(f__lyx['6563'], R_bin__r, x0, y0, pa, ba, HLR_pix, sel_SF__yx, 'mean')
            tau_V__gz = np.ma.masked_array(ALL.tau_V__z, mask=~sel_gals_sample__gz)
            tau_V_neb[i_g] = f_tauVneb(mean_f6563__r, mean_f4861__r)
            tau_V_neb_npts[i_g] = (~np.bitwise_or(~gal_sample__yx, np.bitwise_or(f__lyx['4861'].mask, f__lyx['6563'].mask))).astype('int').sum()
            tau_V_neb_HIG[i_g] = f_tauVneb(mean_f6563_HIG__r, mean_f4861_HIG__r)
            tau_V_neb_HIG_npts[i_g] = (~np.bitwise_or(~sel_HIG__yx, np.bitwise_or(f__lyx['4861'].mask, f__lyx['6563'].mask))).astype('int').sum()
            tau_V_neb_DIG[i_g] = f_tauVneb(mean_f6563_DIG__r, mean_f4861_DIG__r)
            # tau_V_neb_DIG[i_g] = np.ma.log(mean_f6563_DIG__r / mean_f4861_DIG__r / 2.86) / (q[0] - q[1])
            tau_V_neb_DIG_npts[i_g] = (~np.bitwise_or(~sel_DIG__yx, np.bitwise_or(f__lyx['4861'].mask, f__lyx['6563'].mask))).astype('int').sum()
            tau_V_neb_SF[i_g] = f_tauVneb(mean_f6563_SF__r, mean_f4861_SF__r)
            # tau_V_neb_SF[i_g] = np.ma.log(mean_f6563_SF__r / mean_f4861_SF__r / 2.86) / (q[0] - q[1])
            tau_V_neb_SF_npts[i_g] = (~np.bitwise_or(~sel_SF__yx, np.bitwise_or(f__lyx['4861'].mask, f__lyx['6563'].mask))).astype('int').sum()
            tau_V_neb_GAL[i_g] = ALL.get_gal_prop_unique(califaID, ALL.integrated_tau_V_neb)
            xm, ym = ma_mask_xyz(f__lyx['6563'], f__lyx['4861'])
            tau_V_neb_sumGAL[i_g] = f_tauVneb(xm.sum(), ym.sum())  # np.ma.log(xm.sum() / ym.sum() / 2.86) / (q[0] - q[1])
            delta_tau[i_g] = tau_V_neb_SF[i_g] - tau_V_neb_DIG[i_g]
            delta_tau_norm_GAL[i_g] = (tau_V_neb_SF[i_g] - tau_V_neb_DIG[i_g])/tau_V_neb_GAL[i_g]
            delta_tau_norm_sumGAL[i_g] = (tau_V_neb_SF[i_g] - tau_V_neb_DIG[i_g])/tau_V_neb_sumGAL[i_g]
            delta_tau_HIG[i_g] = tau_V_neb_SF[i_g] - tau_V_neb_HIG[i_g]
            delta_tau_norm_GAL[i_g] = (tau_V_neb_SF[i_g] - tau_V_neb_HIG[i_g])/tau_V_neb_GAL[i_g]
            delta_tau_norm_sumGAL[i_g] = (tau_V_neb_SF[i_g] - tau_V_neb_HIG[i_g])/tau_V_neb_sumGAL[i_g]
            print califaID
            print '\t<tauVneb> = %.2f (<<tauVNeb>_R>: %.2f)' % (f_tauVneb(f__lz['6563'], f__lz['4861']).mean(), tau_V_neb[i_g].mean())
            print '\t<tauVneb(SF)>_R = %.2f' % tau_V_neb_SF[i_g].mean()
            print '\t<tauVneb(DIG)>_R = %.2f' % tau_V_neb_DIG[i_g].mean()
            print '\t<tauVneb(HIG)>_R = %.2f' % tau_V_neb_HIG[i_g].mean()
            print '\t<delta_tauVneb(SF-DIG)>_R = %.2f' % delta_tau[i_g].mean()
            print '\t<delta_tauVneb(SF-DIG)_norm>_R = %.2f' % delta_tau_norm_GAL[i_g].mean()
            print '\t<delta_tauVneb(SF-HIG)>_R = %.2f' % delta_tau_HIG[i_g].mean()
            print '\t<delta_tauVneb(SF-HIG)_norm>_R = %.2f' % delta_tau_norm_HIG_GAL[i_g].mean()

        W6563__gz = np.ma.masked_array(ALL.W6563__z, mask=~sel_gals_sample__gz)
        SB4861__gz = np.ma.masked_array(ALL.SB4861__z, mask=~sel_gals_sample__gz)
        SB6563__gz = np.ma.masked_array(ALL.SB6563__z, mask=~sel_gals_sample__gz)

        f__lgz = {'%s' % l: np.ma.masked_array(getattr(ALL, 'f%s__z' % l), mask=~sel_gals_sample__gz) for l in lines}
        f__lgyx = {'%s' % l: np.ma.masked_array(getattr(ALL, 'f%s__yx' % l), mask=~sel_gals_sample__gyx) for l in lines}

        # WHa DIG-COMP-SF decomposition
        sel_WHa_HIG__gz = np.bitwise_and(sel['WHa']['HIG']['z'], sel_gals_sample__gz)
        sel_WHa_DIG__gz = np.bitwise_and(sel['WHa']['DIG']['z'], sel_gals_sample__gz)
        sel_WHa_COMP__gz = np.bitwise_and(sel['WHa']['COMP']['z'], sel_gals_sample__gz)
        sel_WHa_SF__gz = np.bitwise_and(sel['WHa']['SF']['z'], sel_gals_sample__gz)

        deltaTauVDIGStar = f_tauVneb(f__lgz['6563'][sel_WHa_DIG__gz], f__lgz['4861'][sel_WHa_DIG__gz]) - ALL.tau_V__z[sel_WHa_DIG__gz]
        deltaTauVHIGStar = f_tauVneb(f__lgz['6563'][sel_WHa_HIG__gz], f__lgz['4861'][sel_WHa_HIG__gz]) - ALL.tau_V__z[sel_WHa_HIG__gz]
        deltaTauVSFStar = f_tauVneb(f__lgz['6563'][sel_WHa_SF__gz], f__lgz['4861'][sel_WHa_SF__gz]) - ALL.tau_V__z[sel_WHa_SF__gz]
        print '<<tauVneb(DIG) - tauVneb(SF)>_R> = %.2f' % delta_tau.mean()
        print '<tauVneb(HIG) - tauVstar> = %.2f' % deltaTauVHIGStar.mean()
        print '<tauVneb(DIG) - tauVstar> = %.2f' % deltaTauVDIGStar.mean()
        print '<tauVneb(SF) - tauVstar> = %.2f' % deltaTauVSFStar.mean()

        N_cols = 1
        N_rows = 3
        f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
        ax1, ax2, ax3 = axArr

        # AXIS 1
        x = (1./(q[0] - q[1])) * np.ma.log(SB6563__gz/SB4861__gz/2.86)
        range = [-2, 2]
        xDs = [x[sel_WHa_HIG__gz].compressed(), x[sel_WHa_DIG__gz].compressed(),  x[sel_WHa_COMP__gz].compressed(),  x[sel_WHa_SF__gz].compressed()]
        # ax1.set_title('zones')
        plot_histo_ax(ax1, x.compressed(), stats_txt=False, histo=False, ini_pos_y=0.9, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, y_v_space=0.06, y_h_space=0.25, first=False, c=colors_lines_DIG_COMP_SF, kwargs_histo=dict(histtype='step', color=colors_DIG_COMP_SF, normed=False, range=range, lw=3))
        pos_y = 0.9
        for txt in text_list[0]:
            plot_text_ax(ax1, txt, **dict(pos_x=0.98, pos_y=pos_y, fs=14, va='top', ha='right', c=colors_lines_DIG_COMP_SF[0]))
            pos_y -= 0.06
        pos_y = 0.9
        for txt in text_list[1]:
            plot_text_ax(ax1, txt, **dict(pos_x=0.85, pos_y=pos_y, fs=14, va='top', ha='right', c=colors_lines_DIG_COMP_SF[1]))
            pos_y -= 0.06
        pos_y = 0.9
        for txt in text_list[2]:
            plot_text_ax(ax1, txt, **dict(pos_x=0.48, pos_y=pos_y, fs=14, va='top', ha='right', c=colors_lines_DIG_COMP_SF[2]))
            pos_y -= 0.06
        pos_y = 0.9
        for txt in text_list[3]:
            plot_text_ax(ax1, txt, **dict(pos_x=0.34, pos_y=pos_y, fs=14, va='top', ha='right', c=colors_lines_DIG_COMP_SF[3]))
            pos_y -= 0.06
        ax1.set_xlabel(r'$\tau_V^{neb}$')
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1_top = ax1.twiny()
        mn, mx = ax1.get_xlim()
        unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
        ax1_top.set_xlim(unit_converter(mn), unit_converter(mx))
        #ax1_top.set_xlim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))
        ax1_top.set_xlabel(r'$\log\ H\alpha/H\beta$')
        plot_text_ax(ax1, 'a)', 0.02, 0.98, 16, 'top', 'left', 'k')

        # AXIS 2
        x = delta_tau
        print x.shape
        range = DtauVnorm_range
        plot_histo_ax(ax2, x.compressed(), y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
        ax2.set_xlabel(r'$\Delta \tau\ =\ \tau_V^{SF}\ -\ \tau_V^{DIG}}$')
        ax2.xaxis.set_minor_locator(minorLocator)
        plot_text_ax(ax2, 'b)', 0.02, 0.98, 16, 'top', 'left', 'k')

        # AXIS 3
        x = delta_tau_HIG
        print x.shape
        range = DtauVnorm_range
        plot_histo_ax(ax3, x.compressed(), y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
        ax3.set_xlabel(r'$\Delta \tau\ =\ \tau_V^{SF}\ -\ \tau_V^{HIG}}$')
        ax3.xaxis.set_minor_locator(minorLocator)
        plot_text_ax(ax3, 'c)', 0.02, 0.98, 16, 'top', 'left', 'k')


        # x = delta_tau_norm_GAL
        # range = DtauVnorm_range
        # plot_histo_ax(ax3, x.compressed(), y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=range))
        # ax3.set_xlabel(r'$\Delta \tau\ =\ (\tau_V^{SF}\ -\ \tau_V^{DIG}) / \tau_V^{GAL}$')
        # ax3.xaxis.set_minor_locator(minorLocator)
        # plot_text_ax(ax3, 'c)', 0.02, 0.98, 16, 'top', 'left', 'k')

        f.tight_layout(h_pad=0.05)
        f.savefig('fig7.png', dpi=dpi_choice, transparent=transp_choice)


def fig8(ALL, sel, gals):
    sel_gals_sample__gz = sel['gals_sample__z']
    if (sel_gals_sample__gz).any():
        tau_V_neb__gz = np.ma.masked_array(ALL.tau_V_neb__z, mask=~sel_gals_sample__gz)
        tau_V__gz = np.ma.masked_array(ALL.tau_V__z, mask=~sel_gals_sample__gz)

        # WHa DIG-COMP-SF decomposition
        sel_WHa_DIG__gz = np.bitwise_and(sel['WHa']['DIG']['z'], sel_gals_sample__gz)
        sel_WHa_COMP__gz = np.bitwise_and(sel['WHa']['COMP']['z'], sel_gals_sample__gz)
        sel_WHa_SF__gz = np.bitwise_and(sel['WHa']['SF']['z'], sel_gals_sample__gz)

        f = plt.figure(dpi=200, figsize=(6, 5))
        ax1 = f.gca()
        x = tau_V_neb__gz - tau_V__gz
        xDs = [x[sel_WHa_DIG__gz].compressed(),  x[sel_WHa_COMP__gz].compressed(),  x[sel_WHa_SF__gz].compressed()]
        range = DtauV_range
        # ax1.set_title('zones')
        plot_histo_ax(ax1, x.compressed(), histo=False, y_v_space=0.06, ha='left', pos_x=0.02, c='k', first=True)
        plot_histo_ax(ax1, xDs, y_v_space=0.06, y_h_space=0.11, first=False, c=colors_lines_DIG_COMP_SF, kwargs_histo=dict(histtype='step', color=colors_DIG_COMP_SF, normed=False, range=range))
        ax1.set_xlabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        ax1.xaxis.set_minor_locator(minorLocator)
        f.tight_layout()
        f.savefig('fig8.png', dpi=dpi_choice, transparent=transp_choice)


def fig9(ALL, sel, gals):
    sel_gals_sample__gz = sel['gals_sample__z']

    if (sel_gals_sample__gz).any():
        W6563__gz = np.ma.masked_array(ALL.W6563__z, mask=~sel_gals_sample__gz)

        x_Y__gz = np.ma.masked_array(ALL.x_Y__z, mask=~sel_gals_sample__gz)
        tau_V_neb__gz = np.ma.masked_array(ALL.tau_V_neb__z, mask=~sel_gals_sample__gz)
        tau_V__gz = np.ma.masked_array(ALL.tau_V__z, mask=~sel_gals_sample__gz)

        # WHa DIG-COMP-SF decomposition
        sel_WHa_DIG__gz = np.bitwise_and(sel['WHa']['DIG']['z'], sel_gals_sample__gz)
        sel_WHa_COMP__gz = np.bitwise_and(sel['WHa']['COMP']['z'], sel_gals_sample__gz)
        sel_WHa_SF__gz = np.bitwise_and(sel['WHa']['SF']['z'], sel_gals_sample__gz)

        x = x_Y__gz
        y = tau_V_neb__gz - tau_V__gz
        xm, ym = ma_mask_xyz(x, y)
        f = plt.figure(figsize=(8, 8))
        cmap = cmap_discrete(colors_DIG_COMP_SF)
        x_dataset = [xm[sel_WHa_DIG__gz].compressed(), xm[sel_WHa_COMP__gz].compressed(), xm[sel_WHa_SF__gz].compressed()]
        y_dataset = [ym[sel_WHa_DIG__gz].compressed(), ym[sel_WHa_COMP__gz].compressed(), ym[sel_WHa_SF__gz].compressed()]
        axS, axH1, axH2 = plot_scatter_histo(x_dataset, y_dataset, x_Y_range, DtauV_range, 50, 50,
                                             figure=f, c=colors_DIG_COMP_SF, scatter=False, s=1,
                                             ylabel=r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$',
                                             xlabel=r'x${}_Y$ [frac.]', histtype='step')

        classif = np.ma.masked_all(W6563__gz.shape)
        classif[sel_WHa_DIG__gz] = 1.
        classif[sel_WHa_COMP__gz] = 2.
        classif[sel_WHa_SF__gz] = 3.
        sc = axS.scatter(xm, ym, c=classif, cmap=cmap, s=1, **dflt_kw_scatter)
        cbaxes = add_subplot_axes(axS, [0.59, 0.95, 0.4, 0.04])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        cb.ax.set_xticklabels(classif_labels, fontsize=9)
        axS.set_xlim(x_Y_range)
        axS.set_ylim(DtauV_range)
        axS.set_xlabel(r'x${}_Y$ [frac.]')
        axS.set_ylabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        axS.grid()
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_DIG__gz)
        xbin = np.linspace(0, 0.6, 30)
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), xbin)
        mask = (npts_DIG > 30)
        axS.plot(bin_center_DIG[mask], yPrc_DIG[2][mask], '--', lw=2, c=colors_lines_DIG_COMP_SF[0])
        axS.plot(bin_center_DIG[mask], yPrc_DIG[2][mask], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_COMP__gz)
        xbin = np.linspace(0, 0.6, 30)
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), xbin)
        mask = (npts_COMP > 30)
        axS.plot(bin_center_COMP[mask], yPrc_COMP[2][mask], '--', lw=2, c=colors_lines_DIG_COMP_SF[1])
        axS.plot(bin_center_COMP[mask], yPrc_COMP[2][mask], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_SF__gz)
        xbin = np.linspace(0, 0.6, 30)
        yMean_SF, yPrc_SF, bin_center_SF, npts_SF = stats_med12sigma(xm.compressed(), ym.compressed(), xbin)
        mask = (npts_SF > 30)
        axS.plot(bin_center_SF[mask], yPrc_SF[2][mask], '--', lw=2, c=colors_lines_DIG_COMP_SF[2])
        axS.plot(bin_center_SF[mask], yPrc_SF[2][mask], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[2], markersize=10)
        f.subplots_adjust(wspace=0.05)
        f.savefig('fig9.png', dpi=dpi_choice, transparent=transp_choice)
        # left, width = 0.1, 0.65
        # bottom, height = 0.1, 0.85
        # left_h = left + width + 0.02
        # rect_scatter = [left, bottom, width, height]
        # rect_histy = [left_h, bottom, 0.2, height]
        # ax1 = f.add_axes(rect_scatter)
        # ax2 = f.add_axes(rect_histy)
        # cmap = cmap_discrete(colors_DIG_COMP_SF)
        # # AXIS 1
        # classif = np.ma.masked_all(W6563__gz.shape)
        # classif[sel_WHa_DIG__gz] = 1
        # classif[sel_WHa_COMP__gz] = 2
        # classif[sel_WHa_SF__gz] = 3
        # sc = ax1.scatter(x, y, c=classif, cmap=cmap, s=1, **dflt_kw_scatter)
        # cbaxes = add_subplot_axes(ax1, [0.59, 0.95, 0.4, 0.04])
        # cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
        # cb.ax.set_xticklabels(classif_labels, fontsize=9)
        # ax1.set_xlim(x_Y_range)
        # ax1.set_ylim(DtauV_range)
        # ax1.set_xlabel(r'x${}_Y$ [frac.]')
        # ax1.set_ylabel(r'$\mathcal{D}_\tau\ =\ \tau_V^{neb}\ -\ \tau_V^\star$')
        # ax1.grid()
        # xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_DIG__gz)
        # xbin = np.linspace(0, 0.6, 30)
        # yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), xbin)
        # mask = (npts_DIG > 30)
        # ax1.plot(bin_center_DIG[mask], yPrc_DIG[2][mask], '--', lw=2, c=colors_lines_DIG_COMP_SF[0])
        # ax1.plot(bin_center_DIG[mask], yPrc_DIG[2][mask], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[0], markersize=10)
        # xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_COMP__gz)
        # xbin = np.linspace(0, 0.6, 30)
        # yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), xbin)
        # mask = (npts_COMP > 30)
        # ax1.plot(bin_center_COMP[mask], yPrc_COMP[2][mask], '--', lw=2, c=colors_lines_DIG_COMP_SF[1])
        # ax1.plot(bin_center_COMP[mask], yPrc_COMP[2][mask], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[1], markersize=10)
        # xm, ym = ma_mask_xyz(x, y, mask=~sel_WHa_SF__gz)
        # xbin = np.linspace(0, 0.6, 30)
        # yMean_SF, yPrc_SF, bin_center_SF, npts_SF = stats_med12sigma(xm.compressed(), ym.compressed(), xbin)
        # mask = (npts_SF > 30)
        # ax1.plot(bin_center_SF[mask], yPrc_SF[2][mask], '--', lw=2, c=colors_lines_DIG_COMP_SF[2])
        # ax1.plot(bin_center_SF[mask], yPrc_SF[2][mask], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[2], markersize=10)
        # # AXIS 2
        # yDs = [y[sel_WHa_DIG__gz].compressed(), y[sel_WHa_COMP__gz].compressed(), y[sel_WHa_SF__gz].compressed()]
        # ax2.hist(yDs, bins=30, range=DtauV_range, orientation='horizontal', color=colors_DIG_COMP_SF, histtype='step')
        # from matplotlib.ticker import NullFormatter
        # nullfmt = NullFormatter()  # no labels
        # ax2.yaxis.set_major_formatter(nullfmt)
        # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=270)
        # f.subplots_adjust(wspace=0.05)
        # f.savefig('fig9.png')


def fig_WHaSBHa_per_morftype_and_ba(sample_choice):
    '''
    2017-05-24 -
        TODO: Adicionar divisao por raio no mesmo histograma.
    '''
    print '######################################################'
    print '# fig_WHa_per_morftype - ALL Gals (not only spirals) #'
    print '######################################################'
    ALL = stack_gals().load('/Users/lacerda/dev/astro/dig/data/all/sample_all.pkl')
    # ALL = stack_gals().load('sample_all.pkl')
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    gals, sel, sample_choice = samples(ALL, sample_choice, ALL.califaID__z[sorted(ind)].tolist())
    # summary(ALL, sel, gals, 'SEL %s' % sample_choice)
    sel_gals_sample__gz = sel['gals_sample__z']
    mt = np.hstack(list(itertools.chain(list(itertools.repeat(mt, ALL.N_zone[i])) for i, mt in enumerate(ALL.mt))))
    ba = np.hstack(list(itertools.chain(list(itertools.repeat(ba, ALL.N_zone[i])) for i, ba in enumerate(ALL.ba))))
    sel_gals_mt = {
        'E': (mt == -2),
        'S0+S0a': (mt == -1),
        'Sa+Sab': np.bitwise_or(mt == 0, mt == 1),
        'Sb': mt == 2,
        'Sbc': mt == 3,
        'Sc+Scd+Sd+Irr': np.bitwise_or(mt == 4, np.bitwise_or(mt == 5, np.bitwise_or(mt == 6, mt == 7))),
    }
    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    colortipo_darker = ['maroon', 'darkred', 'orangered', 'darkgreen', '#11C0B3', 'darkblue']
    colortipo_lighter = ['rosybrown', 'salmon', 'navajowhite', 'lightgreen', '#00D0B9', 'lightblue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', 'Sc+Scd+Sd+Irr']
    # prc = np.percentile(ALL.ba, [33, 66])
    prc = [0.4, 0.7]
    bulge_size_HLR = 0.7
    sel_radius1 = np.greater(ALL.zoneDistance_HLR, bulge_size_HLR)
    sel_radius2 = np.less_equal(ALL.zoneDistance_HLR, bulge_size_HLR)
    sel_gals_sample_radius1__gz = np.bitwise_and(sel_radius1, sel_gals_sample__gz)
    sel_gals_sample_radius2__gz = np.bitwise_and(sel_radius2, sel_gals_sample__gz)
    N_rows, N_cols = 7, 4
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 10))
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    x_dataset1, x_dataset2, x_dataset3, x_dataset4 = [], [], [], []
    x_dataset11, x_dataset21, x_dataset31, x_dataset41 = [], [], [], []
    x_dataset12, x_dataset22, x_dataset32, x_dataset42 = [], [], [], []
    for k in mtype_labels:
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        x_dataset1.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_radius1__gz)
        x_dataset11.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_radius2__gz)
        x_dataset12.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        m_aux = np.bitwise_and(m_aux, np.less_equal(ba, prc[0]))
        x_dataset2.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_radius1__gz)
        m_aux = np.bitwise_and(m_aux, np.less_equal(ba, prc[0]))
        x_dataset21.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_radius2__gz)
        m_aux = np.bitwise_and(m_aux, np.less_equal(ba, prc[0]))
        x_dataset22.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        m_aux = np.bitwise_and(m_aux, ((ba > prc[0]) & (ba <= prc[1])))
        x_dataset3.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_radius1__gz)
        m_aux = np.bitwise_and(m_aux, ((ba > prc[0]) & (ba <= prc[1])))
        x_dataset31.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_radius2__gz)
        m_aux = np.bitwise_and(m_aux, ((ba > prc[0]) & (ba <= prc[1])))
        x_dataset32.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        m_aux = np.bitwise_and(m_aux, np.greater(ba, prc[1]))
        x_dataset4.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_radius1__gz)
        m_aux = np.bitwise_and(m_aux, np.greater(ba, prc[1]))
        x_dataset41.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_radius2__gz)
        m_aux = np.bitwise_and(m_aux, np.greater(ba, prc[1]))
        x_dataset42.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
    N_zone = sel_gals_sample__gz.astype('int').sum()
    (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43), (ax50, ax51, ax52, ax53), (ax60, ax61, ax62, ax63) = axArr
    axes_col0 = [ax00, ax10, ax20, ax30, ax40, ax50, ax60]
    axes_col1 = [ax01, ax11, ax21, ax31, ax41, ax51, ax61]
    axes_col2 = [ax02, ax12, ax22, ax32, ax42, ax52, ax62]
    axes_col3 = [ax03, ax13, ax23, ax33, ax43, ax53, ax63]
    ax = axes_col0[0]
    ax.set_title(r'%d galaxies - %d zones' % (len(gals), N_zone), fontsize=18, y=1.02)
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z))  #, mask=~sel_gals_sample_radius__gz)
    plot_histo_ax(ax, xm.compressed(), stats_txt=False, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=[-1, 3], lw=3))
    m_aux = sel_gals_sample_radius2__gz
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=False, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', range=[-1, 3], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
    # ax.axvline(x=np.log10(20), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    plt.setp(ax.yaxis.get_majorticklabels(), visible=False)
    ax = axes_col1[0]
    ax.set_title(r'ba <= %.2f (%d gals)' % (prc[0], (ALL.ba <= prc[0]).count()), fontsize=18, y=1.02)
    m_aux = np.bitwise_and(np.less_equal(ba, prc[0]), sel_gals_sample_radius1__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=[-1, 3], lw=3))
    m_aux = np.bitwise_and(np.less_equal(ba, prc[0]), sel_gals_sample_radius2__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.02, ha='left', ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', range=[-1, 3], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
    # ax.axvline(x=np.log10(20), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    plt.setp(ax.yaxis.get_majorticklabels(), visible=False)
    ax = axes_col2[0]
    ax.set_title(r'%.2f < ba <= %.2f (%d gals)' % (prc[0], prc[1], ), fontsize=18, y=1.02)
    m_aux = np.bitwise_and(((ba > prc[0]) & (ba <= prc[1])), sel_gals_sample_radius1__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=[-1, 3], lw=3))
    m_aux = np.bitwise_and(((ba > prc[0]) & (ba <= prc[1])), sel_gals_sample_radius2__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.02, ha='left', ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', range=[-1, 3], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
    # ax.axvline(x=np.log10(20), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    plt.setp(ax.yaxis.get_majorticklabels(), visible=False)
    ax = axes_col3[0]
    ax.set_title(r'ba > %.2f' % prc[1], fontsize=18, y=1.02)
    m_aux = np.bitwise_and(np.greater(ba, prc[1]), sel_gals_sample_radius1__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='darkgray', range=[-1, 3], lw=3))
    m_aux = np.bitwise_and(np.greater(ba, prc[1]), sel_gals_sample_radius2__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, pos_x=0.02, ha='left', ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', range=[-1, 3], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
    # ax.axvline(x=np.log10(20), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    plt.setp(ax.yaxis.get_majorticklabels(), visible=False)
    # ax.set_xscale('log')
    # ax.set_xlim(0.01, 1000)
    # ALL
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col0[i+1]
        plot_histo_ax(ax, x_dataset1[i], stats_txt=False, kwargs_histo=dict(bins=50, histtype='step', color='k', range=[-1, 3], lw=1))
        plot_histo_ax(ax, x_dataset11[i], dataset_names='%s:R > %.2f HLR' % (mt_label, bulge_size_HLR), ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=[-1, 3], lw=2))
        plot_histo_ax(ax, x_dataset12[i], dataset_names='R <= %.2f HLR' % bulge_size_HLR, pos_x=0.02, ha='left', ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo_darker[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo_darker[i], ls='-', range=[-1, 3], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        plt.setp(ax.yaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
        # ax.axvline(x=np.log10(20), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    ###########################################
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col1[i+1]
        plot_histo_ax(ax, x_dataset2[i], stats_txt=False, kwargs_histo=dict(bins=50, histtype='step', color='k', range=[-1, 3], lw=1))
        plot_histo_ax(ax, x_dataset21[i], dataset_names='R > %.2f HLR' % bulge_size_HLR, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=[-1, 3], lw=2))
        plot_histo_ax(ax, x_dataset22[i], dataset_names='R <= %.2f HLR' % bulge_size_HLR, pos_x=0.02, ha='left', ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo_darker[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo_darker[i], ls='-', range=[-1, 3], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        plt.setp(ax.yaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
        # ax.axvline(x=np.log10(20), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    ###########################################
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col2[i+1]
        plot_histo_ax(ax, x_dataset3[i], stats_txt=False, kwargs_histo=dict(bins=50, histtype='step', color='k', range=[-1, 3], lw=1))
        plot_histo_ax(ax, x_dataset31[i], dataset_names='R > %.2f HLR' % bulge_size_HLR, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=[-1, 3], lw=2))
        plot_histo_ax(ax, x_dataset32[i], dataset_names='R <= %.2f HLR' % bulge_size_HLR, pos_x=0.02, ha='left', ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo_darker[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo_darker[i], ls='-', range=[-1, 3], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        plt.setp(ax.yaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
        # ax.axvline(x=np.log10(20), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    ###########################################
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col3[i+1]
        plot_histo_ax(ax, x_dataset4[i], stats_txt=False, kwargs_histo=dict(bins=50, histtype='step', color='k', range=[-1, 3], lw=1))
        plot_histo_ax(ax, x_dataset41[i], dataset_names='R > %.2f HLR' % bulge_size_HLR, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], range=[-1, 3], lw=2))
        plot_histo_ax(ax, x_dataset42[i], dataset_names='R <= %.2f HLR' % bulge_size_HLR, pos_x=0.02, ha='left', ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo_darker[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo_darker[i], ls='-', range=[-1, 3], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        plt.setp(ax.yaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
        # ax.axvline(x=np.log10(20), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    plt.setp(ax60.xaxis.get_majorticklabels(), visible=True)
    plt.setp(ax61.xaxis.get_majorticklabels(), visible=True)
    plt.setp(ax62.xaxis.get_majorticklabels(), visible=True)
    plt.setp(ax63.xaxis.get_majorticklabels(), visible=True)
    ax60.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    ax61.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    ax62.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    ax63.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    # ax13.set_xticklabels([0.1, 1, 10, 100, 1000])
    # ax13.get_xaxis().set_major_formatter(ScalarFormatter())
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.subplots_adjust(hspace=0.05)
    f.savefig('fig_WHa_histo_bymorf_ba.png', dpi=dpi_choice, transparent=transp_choice)

    # f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 10))
    # _, ind = np.unique(ALL.califaID__z, return_index=True)
    # x_dataset1, x_dataset2, x_dataset3, x_dataset4 = [], [], [], []
    # for k in mtype_labels:
    #     m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
    #     x_dataset1.append(np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux).compressed())
    #     m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_radius__gz)
    #     m_aux = np.bitwise_and(m_aux, np.less_equal(ALL.zoneDistance_HLR, 0.5))
    #     x_dataset2.append(np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux).compressed())
    #     m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_radius__gz)
    #     m_aux = np.bitwise_and(m_aux, np.less_equal(ALL.zoneDistance_HLR, 1))
    #     x_dataset3.append(np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux).compressed())
    #     m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample_radius__gz)
    #     m_aux = np.bitwise_and(m_aux, np.greater(ALL.zoneDistance_HLR, 1))
    #     x_dataset4.append(np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux).compressed())
    # N_zone = sel_gals_sample__gz.astype('int').sum()
    # (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43), (ax50, ax51, ax52, ax53), (ax60, ax61, ax62, ax63) = axArr
    # axes_col0 = [ax00, ax10, ax20, ax30, ax40, ax50, ax60]
    # axes_col1 = [ax01, ax11, ax21, ax31, ax41, ax51, ax61]
    # axes_col2 = [ax02, ax12, ax22, ax32, ax42, ax52, ax62]
    # axes_col3 = [ax03, ax13, ax23, ax33, ax43, ax53, ax63]
    # ax = axes_col0[0]
    # ax.set_title(r'%d galaxies - %d zones' % (len(gals), N_zone), fontsize=18, y=1.02)
    # xm = np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~sel_gals_sample_radius__gz)
    # plot_histo_ax(ax, xm.compressed(), stats_txt=False, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[3, 7.5], lw=3))
    # ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    # ax.xaxis.set_major_locator(MaxNLocator(5))
    # ax.yaxis.set_major_locator(MaxNLocator(5))
    # ax.xaxis.set_minor_locator(MaxNLocator(25))
    # ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    # plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    # ax = axes_col1[0]
    # ax.set_title(r'ba <= %.2f' % prc[0], fontsize=18, y=1.02)
    # m_aux = np.bitwise_and(np.less_equal(ba, prc[0]), sel_gals_sample_radius__gz)
    # xm = np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux)
    # plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[3, 7.5], lw=3))
    # ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    # ax.xaxis.set_major_locator(MaxNLocator(5))
    # ax.yaxis.set_major_locator(MaxNLocator(5))
    # ax.xaxis.set_minor_locator(MaxNLocator(25))
    # ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    # plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    # ax = axes_col2[0]
    # ax.set_title(r'%.2f < ba <= %.2f' % (prc[0], prc[1]), fontsize=18, y=1.02)
    # m_aux = np.bitwise_and(((ba > prc[0]) & (ba <= prc[1])), sel_gals_sample_radius__gz)
    # xm = np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux)
    # plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[3, 7.5], lw=3))
    # ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    # ax.xaxis.set_major_locator(MaxNLocator(5))
    # ax.yaxis.set_major_locator(MaxNLocator(5))
    # ax.xaxis.set_minor_locator(MaxNLocator(25))
    # ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    # plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    # ax = axes_col3[0]
    # ax.set_title(r'ba > %.2f' % prc[0], fontsize=18, y=1.02)
    # m_aux = np.bitwise_and(np.greater(ba, prc[1]), sel_gals_sample_radius__gz)
    # xm = np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux)
    # plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[3, 7.5], lw=3))
    # ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    # ax.xaxis.set_major_locator(MaxNLocator(5))
    # ax.yaxis.set_major_locator(MaxNLocator(5))
    # ax.xaxis.set_minor_locator(MaxNLocator(25))
    # ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    # plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    # # ax.set_xscale('log')
    # # ax.set_xlim(0.01, 1000)
    # # ALL
    # for i, mt_label in enumerate(mtype_labels):
    #     ax = axes_col0[i+1]
    #     plot_histo_ax(ax, x_dataset1[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[3, 7.5], lw=2))
    #     ax.xaxis.set_major_locator(MaxNLocator(5))
    #     ax.yaxis.set_major_locator(MaxNLocator(5))
    #     ax.xaxis.set_minor_locator(MaxNLocator(25))
    #     plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    #     ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    #     # ax.set_xscale('log')
    #     # ax.set_xlim(0.01, 1000)
    # ###########################################
    # for i, mt_label in enumerate(mtype_labels):
    #     ax = axes_col1[i+1]
    #     plot_histo_ax(ax, x_dataset2[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[3, 7.5], lw=2))
    #     ax.xaxis.set_major_locator(MaxNLocator(5))
    #     ax.yaxis.set_major_locator(MaxNLocator(5))
    #     ax.xaxis.set_minor_locator(MaxNLocator(25))
    #     plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    #     ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    #     # ax.set_xscale('log')
    #     # ax.set_xlim(0.01, 1000)
    # ###########################################
    # for i, mt_label in enumerate(mtype_labels):
    #     ax = axes_col2[i+1]
    #     plot_histo_ax(ax, x_dataset3[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[3, 7.5], lw=2))
    #     ax.xaxis.set_major_locator(MaxNLocator(5))
    #     ax.yaxis.set_major_locator(MaxNLocator(5))
    #     ax.xaxis.set_minor_locator(MaxNLocator(25))
    #     plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    #     ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    #     # ax.set_xscale('log')
    #     # ax.set_xlim(0.01, 1000)
    # ###########################################
    # for i, mt_label in enumerate(mtype_labels):
    #     ax = axes_col3[i+1]
    #     plot_histo_ax(ax, x_dataset4[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[3, 7.5], lw=2))
    #     ax.xaxis.set_major_locator(MaxNLocator(5))
    #     ax.yaxis.set_major_locator(MaxNLocator(5))
    #     ax.xaxis.set_minor_locator(MaxNLocator(25))
    #     plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    #     ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    #     # ax.set_xscale('log')
    #     # ax.set_xlim(0.01, 1000)
    # plt.setp(ax60.xaxis.get_majorticklabels(), visible=True)
    # plt.setp(ax61.xaxis.get_majorticklabels(), visible=True)
    # plt.setp(ax62.xaxis.get_majorticklabels(), visible=True)
    # plt.setp(ax63.xaxis.get_majorticklabels(), visible=True)
    # ax60.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
    # ax61.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
    # ax62.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
    # ax63.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
    # # ax13.set_xticklabels([0.1, 1, 10, 100, 1000])
    # # ax13.get_xaxis().set_major_formatter(ScalarFormatter())
    # # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    # f.subplots_adjust(hspace=0.15)
    # f.savefig('fig_SBHa_histo_bymorf_ba.png', dpi=dpi_choice, transparent=transp_choice)

    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_WHaSBHa_per_morftype_and_R(sample_choice):
    print '######################################################'
    print '# fig_WHa_per_morftype - ALL Gals (not only spirals) #'
    print '######################################################'
    ALL = stack_gals().load('/Users/lacerda/dev/astro/dig/data/all/sample_all.pkl')
    # ALL = stack_gals().load('sample_all.pkl')
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    gals, sel, sample_choice = samples(ALL, sample_choice, ALL.califaID__z[sorted(ind)].tolist())
    # summary(ALL, sel, gals, 'SEL %s' % sample_choice)
    sel_gals_sample__gz = sel['gals_sample__z']
    mt = np.hstack(list(itertools.chain(list(itertools.repeat(mt, ALL.N_zone[i])) for i, mt in enumerate(ALL.mt))))
    sel_gals_mt = {
        'E': (mt == -2),
        'S0+S0a': (mt == -1),
        'Sa+Sab': np.bitwise_or(mt == 0, mt == 1),
        'Sb': mt == 2,
        'Sbc': mt == 3,
        'Sc+Scd+Sd+Irr': np.bitwise_or(mt == 4, np.bitwise_or(mt == 5, np.bitwise_or(mt == 6, mt == 7))),
    }
    colortipo = ['brown', 'red', 'orange', 'green', '#00D0C9', 'blue']
    mtype_labels = ['E', 'S0+S0a', 'Sa+Sab', 'Sb', 'Sbc', 'Sc+Scd+Sd+Irr']
    N_rows, N_cols = 7, 4
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 10))
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    x_dataset1, x_dataset2, x_dataset3, x_dataset4 = [], [], [], []
    for k in mtype_labels:
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        x_dataset1.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        m_aux = np.bitwise_and(m_aux, np.less_equal(ALL.zoneDistance_HLR, 0.5))
        x_dataset2.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        m_aux = np.bitwise_and(m_aux, np.less_equal(ALL.zoneDistance_HLR, 1))
        x_dataset3.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        m_aux = np.bitwise_and(m_aux, np.greater(ALL.zoneDistance_HLR, 1))
        x_dataset4.append(np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux).compressed())
    N_zone = sel_gals_sample__gz.astype('int').sum()
    (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43), (ax50, ax51, ax52, ax53), (ax60, ax61, ax62, ax63) = axArr
    axes_col0 = [ax00, ax10, ax20, ax30, ax40, ax50, ax60]
    axes_col1 = [ax01, ax11, ax21, ax31, ax41, ax51, ax61]
    axes_col2 = [ax02, ax12, ax22, ax32, ax42, ax52, ax62]
    axes_col3 = [ax03, ax13, ax23, ax33, ax43, ax53, ax63]
    ax = axes_col0[0]
    ax.set_title(r'%d galaxies - %d zones' % (len(gals), N_zone), fontsize=18, y=1.02)
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~sel_gals_sample__gz)
    plot_histo_ax(ax, xm.compressed(), stats_txt=False, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[-1, 3], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
    # ax.axvline(x=np.log10(20), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    ax = axes_col1[0]
    ax.set_title(r'R <= 0.5 HLR', fontsize=18, y=1.02)
    m_aux = np.bitwise_and(np.less_equal(ALL.zoneDistance_HLR, 0.5), sel_gals_sample__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[-1, 3], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
    # ax.axvline(x=np.log10(20), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    ax = axes_col2[0]
    ax.set_title(r'R <= 1 HLR', fontsize=18, y=1.02)
    m_aux = np.bitwise_and(np.less_equal(ALL.zoneDistance_HLR, 1), sel_gals_sample__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[-1, 3], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
    # ax.axvline(x=np.log10(20), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    ax = axes_col3[0]
    ax.set_title(r'R > 1', fontsize=18, y=1.02)
    m_aux = np.bitwise_and(np.greater(ALL.zoneDistance_HLR, 1), sel_gals_sample__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.W6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[-1, 3], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
    ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
    # ax.axvline(x=np.log10(20), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    # ax.set_xscale('log')
    # ax.set_xlim(0.01, 1000)
    # ALL
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col0[i+1]
        plot_histo_ax(ax, x_dataset1[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[-1, 3], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
        # ax.axvline(x=np.log10(20), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    ###########################################
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col1[i+1]
        plot_histo_ax(ax, x_dataset2[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[-1, 3], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
        # ax.axvline(x=np.log10(20), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    ###########################################
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col2[i+1]
        plot_histo_ax(ax, x_dataset3[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[-1, 3], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
        # ax.axvline(x=np.log10(20), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    ###########################################
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col3[i+1]
        plot_histo_ax(ax, x_dataset4[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[-1, 3], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(HIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(DIG_WHa_threshold), c='k', ls='--')
        ax.axvline(x=np.log10(SF_WHa_threshold), c='k', ls='--')
        # ax.axvline(x=np.log10(20), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    plt.setp(ax60.xaxis.get_majorticklabels(), visible=True)
    plt.setp(ax61.xaxis.get_majorticklabels(), visible=True)
    plt.setp(ax62.xaxis.get_majorticklabels(), visible=True)
    plt.setp(ax63.xaxis.get_majorticklabels(), visible=True)
    ax60.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    ax61.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    ax62.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    ax63.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
    # ax13.set_xticklabels([0.1, 1, 10, 100, 1000])
    # ax13.get_xaxis().set_major_formatter(ScalarFormatter())
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.subplots_adjust(hspace=0.15)
    f.savefig('fig_WHa_histo_bymorf_R.png', dpi=dpi_choice, transparent=transp_choice)

    f, axArr = plt.subplots(N_rows, N_cols, figsize=(16, 10))
    _, ind = np.unique(ALL.califaID__z, return_index=True)
    x_dataset1, x_dataset2, x_dataset3, x_dataset4 = [], [], [], []
    for k in mtype_labels:
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        x_dataset1.append(np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        m_aux = np.bitwise_and(m_aux, np.less_equal(ALL.zoneDistance_HLR, 0.5))
        x_dataset2.append(np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        m_aux = np.bitwise_and(m_aux, np.less_equal(ALL.zoneDistance_HLR, 1))
        x_dataset3.append(np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux).compressed())
        m_aux = np.bitwise_and(sel_gals_mt[k], sel_gals_sample__gz)
        m_aux = np.bitwise_and(m_aux, np.greater(ALL.zoneDistance_HLR, 1))
        x_dataset4.append(np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux).compressed())
    N_zone = sel_gals_sample__gz.astype('int').sum()
    (ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33), (ax40, ax41, ax42, ax43), (ax50, ax51, ax52, ax53), (ax60, ax61, ax62, ax63) = axArr
    axes_col0 = [ax00, ax10, ax20, ax30, ax40, ax50, ax60]
    axes_col1 = [ax01, ax11, ax21, ax31, ax41, ax51, ax61]
    axes_col2 = [ax02, ax12, ax22, ax32, ax42, ax52, ax62]
    axes_col3 = [ax03, ax13, ax23, ax33, ax43, ax53, ax63]
    ax = axes_col0[0]
    ax.set_title(r'%d galaxies - %d zones' % (len(gals), N_zone), fontsize=18, y=1.02)
    xm = np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~sel_gals_sample__gz)
    plot_histo_ax(ax, xm.compressed(), stats_txt=False, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[3, 7.5], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    ax = axes_col1[0]
    ax.set_title(r'R <= 0.5 HLR', fontsize=18, y=1.02)
    m_aux = np.bitwise_and(np.less_equal(ALL.zoneDistance_HLR, 0.5), sel_gals_sample__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[3, 7.5], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    ax = axes_col2[0]
    ax.set_title(r'R <= 1 HLR', fontsize=18, y=1.02)
    m_aux = np.bitwise_and(np.less_equal(ALL.zoneDistance_HLR, 1), sel_gals_sample__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[3, 7.5], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    ax = axes_col3[0]
    ax.set_title(r'R > 1', fontsize=18, y=1.02)
    m_aux = np.bitwise_and(np.greater(ALL.zoneDistance_HLR, 1), sel_gals_sample__gz)
    xm = np.ma.masked_array(np.ma.log10(ALL.SB6563__z), mask=~m_aux)
    plot_histo_ax(ax, xm.compressed(), stats_txt=True, ini_pos_y=0.97, fs=10, y_v_space=0.13, kwargs_histo=dict(bins=50, histtype='stepfilled', color='gray', normed=True, range=[3, 7.5], lw=3))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(MaxNLocator(25))
    ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    # ax.set_xscale('log')
    # ax.set_xlim(0.01, 1000)
    # ALL
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col0[i+1]
        plot_histo_ax(ax, x_dataset1[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[3, 7.5], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    ###########################################
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col1[i+1]
        plot_histo_ax(ax, x_dataset2[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[3, 7.5], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    ###########################################
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col2[i+1]
        plot_histo_ax(ax, x_dataset3[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[3, 7.5], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    ###########################################
    for i, mt_label in enumerate(mtype_labels):
        ax = axes_col3[i+1]
        plot_histo_ax(ax, x_dataset4[i], dataset_names=mt_label, ini_pos_y=0.97, fs=10, y_v_space=0.13, first=False, c=colortipo[i], kwargs_histo=dict(bins=50, histtype='stepfilled', color=colortipo[i], normed=True, range=[3, 7.5], lw=2))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(MaxNLocator(25))
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        ax.axvline(x=np.log10(SF_Zhang_threshold), c='k', ls='--')
        # ax.set_xscale('log')
        # ax.set_xlim(0.01, 1000)
    plt.setp(ax60.xaxis.get_majorticklabels(), visible=True)
    plt.setp(ax61.xaxis.get_majorticklabels(), visible=True)
    plt.setp(ax62.xaxis.get_majorticklabels(), visible=True)
    plt.setp(ax63.xaxis.get_majorticklabels(), visible=True)
    ax60.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
    ax61.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
    ax62.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
    ax63.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
    # ax13.set_xticklabels([0.1, 1, 10, 100, 1000])
    # ax13.get_xaxis().set_major_formatter(ScalarFormatter())
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.subplots_adjust(hspace=0.15)
    f.savefig('fig_SBHa_histo_bymorf_R.png', dpi=dpi_choice, transparent=transp_choice)

    print '######################################################'
    print '# END ################################################'
    print '######################################################'


def fig_WHaSBHa_profile_3gals(ALL, sel, gals, suffix):
    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    N_cols = 2
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 4.5))
    cmap = cmap_discrete(colors=colors_DIG_COMP_SF)

    row = 0
    for califaID in gals:
        ax1, ax2 = axArr[row]

        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.02, 0.98, 16, 'top', 'left', color='k')

        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
        gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)
        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        zoneDistance_HLR__z = ALL.get_gal_prop(califaID, ALL.zoneDistance_HLR)
        SB6563__z = ALL.get_gal_prop(califaID, ALL.SB6563__z)
        W6563__z = ALL.get_gal_prop(califaID, ALL.W6563__z)
        W6563__yx = ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x)
        SB6563__yx = ALL.get_gal_prop(califaID, ALL.SB6563__yx).reshape(N_y, N_x)

        mW6563__z, mSB6563__z = ma_mask_xyz(W6563__z, SB6563__z, mask=~gal_sample__z)
        mW6563__yx, mSB6563__yx = ma_mask_xyz(W6563__yx, SB6563__yx, mask=~gal_sample__yx)

        x__z = zoneDistance_HLR__z
        y__z = np.ma.log10(mW6563__z)
        y__yx = np.ma.log10(mW6563__yx)
        y__r, npts = radialProfile(y__yx, R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'median', True)
        map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        ax1.scatter(x__z, y__z, c=np.ravel(map__z), cmap=cmap, s=5, **dflt_kw_scatter)
        ax1.set_xlabel(r'R [HLR]')
        ax1.set_ylabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        ax1.grid()
        sel_DIG__z, sel_COMP__z, sel_SF__z, _ = get_selections_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        sel_DIG__yx, sel_COMP__yx, sel_SF__yx, _ = get_selections(ALL, califaID, sel['WHa'], sel_sample__gyx)
        min_npts = 20
        xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_DIG__z)
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel_npts = npts_DIG > min_npts
            # y_DIG__r, npts_DIG__r = radialProfile(y__yx, R_bin__r, x0, y0, pa, ba, HLR_pix, sel_DIG__yx, 'median', True)
            # print 'DIG npts (R):', npts_DIG__r.mean(), npts_DIG__r.std(), (npts_DIG__r > (npts_DIG__r.mean()-npts_DIG__r.std())).sum(), npts_DIG__r.max(), npts_DIG__r.min()
            ax1.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[0])
            # ax1.plot(R_bin_center__r, y_DIG__r, '-', lw=2, c='k')
            ax1.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[0], markersize=10)
        xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_COMP__z)
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            # y_COMP__r, npts_COMP__r = radialProfile(y__yx, R_bin__r, x0, y0, pa, ba, HLR_pix, sel_COMP__yx, 'median', True)
            # print 'COMP npts (R):', npts_COMP__r.mean(), npts_COMP__r.std(), (npts_COMP__r > (npts_COMP__r.mean()-npts_COMP__r.std())).sum(), npts_COMP__r.max(), npts_COMP__r.min()
            sel_npts = npts_COMP > min_npts
            ax1.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[1])
            # ax1.plot(R_bin_center__r, y_COMP__r, '-', lw=2, c='k')
            ax1.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[1], markersize=10)
        xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_SF__z)
        yMean_SF, yPrc_SF, bin_center_SF, npts_SF = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_SF.any():
            # y_SF__r, npts_SF__r = radialProfile(y__yx, R_bin__r, x0, y0, pa, ba, HLR_pix, sel_SF__yx, 'median', True)
            # print 'SF npts (R):', npts_SF__r.mean(), npts_SF__r.std(), (npts_SF__r > (npts_SF__r.mean()-npts_SF__r.std())).sum(), npts_SF__r.max(), npts_SF__r.min()
            sel_npts = npts_SF > min_npts
            ax1.plot(bin_center_SF[sel_npts], yPrc_SF[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[2])
            # ax1.plot(R_bin_center__r, y_SF__r, '-', lw=2, c='k')
            ax1.plot(bin_center_SF[sel_npts], yPrc_SF[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[2], markersize=10)
        x__z = zoneDistance_HLR__z
        y__z = np.ma.log10(mSB6563__z)
        y__yx = np.ma.log10(mSB6563__yx)
        y__r, npts = radialProfile(y__yx, R_bin__r, x0, y0, pa, ba, HLR_pix, None, 'median', True)
        map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        sc = ax2.scatter(x__z, y__z, c=np.ravel(map__z), cmap=cmap, s=5, **dflt_kw_scatter)
        ax2.axhline(y=np.log10(SF_Zhang_threshold), c='k', ls='--')
        ax2.set_xlabel(r'R [HLR]')
        ax2.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax2.grid()
        sel_DIG__z, sel_COMP__z, sel_SF__z, _ = get_selections_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        sel_DIG__yx, sel_COMP__yx, sel_SF__yx, _ = get_selections(ALL, califaID, sel['WHa'], sel_sample__gyx)
        min_npts = 20
        xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_DIG__z)
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_DIG.any():
            sel_npts = npts_DIG > min_npts
            # y_DIG__r, npts_DIG__r = radialProfile(y__yx, R_bin__r, x0, y0, pa, ba, HLR_pix, sel_DIG__yx, 'median', True)
            # print 'DIG npts (R):', npts_DIG__r.mean(), npts_DIG__r.std(), (npts_DIG__r > (npts_DIG__r.mean()-npts_DIG__r.std())).sum(), npts_DIG__r.max(), npts_DIG__r.min()
            ax2.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[0])
            # ax2.plot(R_bin_center__r, y_DIG__r, '-', lw=2, c='k')
            ax2.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[0], markersize=10)
        xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_COMP__z)
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_COMP.any():
            # y_COMP__r, npts_COMP__r = radialProfile(y__yx, R_bin__r, x0, y0, pa, ba, HLR_pix, sel_COMP__yx, 'median', True)
            # print 'COMP npts (R):', npts_COMP__r.mean(), npts_COMP__r.std(), (npts_COMP__r > (npts_COMP__r.mean()-npts_COMP__r.std())).sum(), npts_COMP__r.max(), npts_COMP__r.min()
            sel_npts = npts_COMP > min_npts
            ax2.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[1])
            # ax2.plot(R_bin_center__r, y_COMP__r, '-', lw=2, c='k')
            ax2.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[1], markersize=10)
        xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_SF__z)
        yMean_SF, yPrc_SF, bin_center_SF, npts_SF = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        if npts_SF.any():
            # y_SF__r, npts_SF__r = radialProfile(y__yx, R_bin__r, x0, y0, pa, ba, HLR_pix, sel_SF__yx, 'median', True)
            # print 'SF npts (R):', npts_SF__r.mean(), npts_SF__r.std(), (npts_SF__r > (npts_SF__r.mean()-npts_SF__r.std())).sum(), npts_SF__r.max(), npts_SF__r.min()
            sel_npts = npts_SF > min_npts
            ax2.plot(bin_center_SF[sel_npts], yPrc_SF[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[2])
            # ax2.plot(R_bin_center__r, y_SF__r, '-', lw=2, c='k')
            ax2.plot(bin_center_SF[sel_npts], yPrc_SF[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[2], markersize=10)
        if row == 0:
            ax1.set_title(r'W${}_{H\alpha}$ profile', fontsize=20, y=1.03)
            ax2.set_title(r'$\Sigma_{H\alpha}$ profile', fontsize=20, y=1.03)
            cbaxes = add_subplot_axes(ax2, [0.7, 1.12, 0.44, 0.07])
            cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
            cb.ax.set_xticklabels(classif_labels, fontsize=12)
        ax1.set_xlim(0, 2.5)
        ax1.set_ylim(0, 2.5)
        ax2.set_xlim(0, 2.5)
        ax2.set_ylim(4, 6.5)
        row += 1
    f.tight_layout(w_pad=0.05, h_pad=0)
    f.savefig('fig_WHaSBHa_profile_%s.png' % suffix, dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


def fig_tauVNeb_WHaSBHa_3gals(ALL, sel, gals, suffix):
    from pytu.plots import plot_spearmanr_ax
    sel_sample__gz = sel['gals_sample__z']
    sel_sample__gyx = sel['gals_sample__yx']

    N_cols = 2
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 4.5))
    cmap = cmap_discrete(colors=colors_DIG_COMP_SF)

    row = 0
    for califaID in gals:
        ax1, ax2 = axArr[row]

        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.02, 0.98, 16, 'top', 'left', color='k')

        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        gal_sample__z = ALL.get_gal_prop(califaID, sel_sample__gz)
        gal_sample__yx = ALL.get_gal_prop(califaID, sel_sample__gyx).reshape(N_y, N_x)
        HLR_pix = ALL.get_gal_prop_unique(califaID, ALL.HLR_pix)
        pa = ALL.get_gal_prop_unique(califaID, ALL.pa)
        ba = ALL.get_gal_prop_unique(califaID, ALL.ba)
        x0 = ALL.get_gal_prop_unique(califaID, ALL.x0)
        y0 = ALL.get_gal_prop_unique(califaID, ALL.y0)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        zoneDistance_HLR__z = ALL.get_gal_prop(califaID, ALL.zoneDistance_HLR)
        SB6563__z = ALL.get_gal_prop(califaID, ALL.SB6563__z)
        f6563__z = ALL.get_gal_prop(califaID, ALL.f6563__z)
        f4861__z = ALL.get_gal_prop(califaID, ALL.f4861__z)
        f6563__yx = ALL.get_gal_prop(califaID, ALL.f6563__yx).reshape(N_y, N_x)
        f4861__yx = ALL.get_gal_prop(califaID, ALL.f4861__yx).reshape(N_y, N_x)
        W6563__z = ALL.get_gal_prop(califaID, ALL.W6563__z)
        W6563__yx = ALL.get_gal_prop(califaID, ALL.W6563__yx).reshape(N_y, N_x)
        SB6563__yx = ALL.get_gal_prop(califaID, ALL.SB6563__yx).reshape(N_y, N_x)
        mW6563__z, mSB6563__z = ma_mask_xyz(W6563__z, SB6563__z, mask=~gal_sample__z)
        mW6563__yx, mSB6563__yx = ma_mask_xyz(W6563__yx, SB6563__yx, mask=~gal_sample__yx)
        mf6563__z, mf4861__z = ma_mask_xyz(f6563__z, f4861__z, mask=~gal_sample__z)
        mf6563__yx, mf4861__yx = ma_mask_xyz(f6563__yx, f4861__yx, mask=~gal_sample__yx)
        tau_V_neb__z = f_tauVneb(mf6563__z, mf4861__z)
        tau_V_neb__yx = f_tauVneb(mf6563__yx, mf4861__yx)

        x__z = tau_V_neb__z
        x_range = f_tauVneb(10**(lineratios_range['6563/4861'][0]), 10**(lineratios_range['6563/4861'][1]))
        x_range = [-0.5, 2.5]
        x_bins = np.arange(x_range[0], x_range[1] + 0.2, 0.2)
        y__z = np.ma.log10(mW6563__z)
        y__yx = np.ma.log10(mW6563__yx)
        map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        ax1.scatter(x__z, y__z, c=np.ravel(map__z), cmap=cmap, s=5, **dflt_kw_scatter)
        ax1.set_xlabel(r'$\tau_V^{neb}$')
        ax1.set_ylabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        ax1.grid()
        xm, ym = ma_mask_xyz(x__z, y__z)
        plot_spearmanr_ax(ax=ax1, x=xm.compressed(), y=ym.compressed(), pos_x=0.02, pos_y=0.90, fontsize=16)
        yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), x_bins)
        min_npts = 20
        if npts.any():
            sel_npts = npts > min_npts
            ax1.plot(bin_center[sel_npts], yPrc[2][sel_npts], '-', lw=2, c='k')
            ax1.plot(bin_center[sel_npts], yPrc[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c='k', markersize=10)
        # sel_DIG__z, sel_COMP__z, sel_SF__z, _ = get_selections_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        # min_npts = 20
        # xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_DIG__z)
        # yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), x_bins)
        # if npts_DIG.any():
        #     sel_npts = npts_DIG > min_npts
        #     ax1.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[0])
        #     ax1.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[0], markersize=10)
        # xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_COMP__z)
        # yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), x_bins)
        # if npts_COMP.any():
        #     sel_npts = npts_COMP > min_npts
        #     ax1.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[1])
        #     ax1.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[1], markersize=10)
        # xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_SF__z)
        # yMean_SF, yPrc_SF, bin_center_SF, npts_SF = stats_med12sigma(xm.compressed(), ym.compressed(), x_bins)
        # if npts_SF.any():
        #     sel_npts = npts_SF > min_npts
        #     ax1.plot(bin_center_SF[sel_npts], yPrc_SF[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[2])
        #     ax1.plot(bin_center_SF[sel_npts], yPrc_SF[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[2], markersize=10)
        x__z = tau_V_neb__z
        y__z = np.ma.log10(mSB6563__z)
        map__z = create_segmented_map_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        sc = ax2.scatter(x__z, y__z, c=np.ravel(map__z), cmap=cmap, s=5, **dflt_kw_scatter)
        ax2.axhline(y=np.log10(SF_Zhang_threshold), c='k', ls='--')
        ax2.set_xlabel(r'$\tau_V^{neb}$')
        ax2.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax2.grid()
        xm, ym = ma_mask_xyz(x__z, y__z)
        plot_spearmanr_ax(ax=ax2, x=xm.compressed(), y=ym.compressed(), pos_x=0.02, pos_y=0.98, fontsize=16)
        min_npts = 20
        yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), x_bins)
        if npts.any():
            sel_npts = npts > min_npts
            ax2.plot(bin_center[sel_npts], yPrc[2][sel_npts], '-', lw=2, c='k')
            ax2.plot(bin_center[sel_npts], yPrc[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c='k', markersize=10)
        # sel_DIG__z, sel_COMP__z, sel_SF__z, _ = get_selections_zones(ALL, califaID, sel['WHa'], sel_sample__gz)
        # xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_DIG__z)
        # yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), x_bins)
        # if npts_DIG.any():
        #     sel_npts = npts_DIG > min_npts
        #     ax2.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[0])
        #     ax2.plot(bin_center_DIG[sel_npts], yPrc_DIG[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[0], markersize=10)
        # xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_COMP__z)
        # yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), x_bins)
        # if npts_COMP.any():
        #     sel_npts = npts_COMP > min_npts
        #     ax2.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[1])
        #     ax2.plot(bin_center_COMP[sel_npts], yPrc_COMP[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[1], markersize=10)
        # xm, ym = ma_mask_xyz(x__z, y__z, mask=~sel_SF__z)
        # yMean_SF, yPrc_SF, bin_center_SF, npts_SF = stats_med12sigma(xm.compressed(), ym.compressed(), x_bins)
        # if npts_SF.any():
        #     sel_npts = npts_SF > min_npts
        #     ax2.plot(bin_center_SF[sel_npts], yPrc_SF[2][sel_npts], '-', lw=2, c=colors_lines_DIG_COMP_SF[2])
        #     ax2.plot(bin_center_SF[sel_npts], yPrc_SF[2][sel_npts], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[2], markersize=10)
        if row == 0:
            ax1.set_title(r'W${}_{H\alpha}$ profile', fontsize=20, y=1.03)
            ax2.set_title(r'$\Sigma_{H\alpha}$ profile', fontsize=20, y=1.03)
            cbaxes = add_subplot_axes(ax2, [0.7, 1.12, 0.44, 0.07])
            cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
            cb.ax.set_xticklabels(classif_labels, fontsize=12)
        x_range = (1/0.34652) * np.log(10**(lineratios_range['6563/4861'][0])/2.86), (1/0.34652) * np.log(10**(lineratios_range['6563/4861'][1])/2.86)
        ax1.set_xlim(x_range)
        ax1.set_ylim(0, 2.5)
        ax2.set_xlim(x_range)
        ax2.set_ylim(4, 6.5)
        ax1_twin = ax1.twinx()
        mn, mx = ax1.get_ylim()
        unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
        ax1_twin.set_ylim(unit_converter(mn), unit_converter(mx))
        row += 1
    f.tight_layout(w_pad=0.05, h_pad=0)
    f.savefig('fig_tauVNeb_WHaSBHa_profile_%s.png' % suffix, dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


def fig_tauVNeb_WHaSBHa(ALL, sel, gals):
    from pytu.plots import plot_spearmanr_ax
    sel_gals_sample__gz = sel['gals_sample__z']
    sel_center__gz = np.zeros(ALL.f6563__z.shape, dtype='bool')
    for g in gals:
        HLR_pc = ALL.get_gal_prop_unique(g, ALL.HLR_pc)
        distance_Mpc = ALL.get_gal_prop_unique(g, ALL.galDistance_Mpc)
        max_dist = spaxel_size_pc(distance_Mpc, arcsec=2.5)
        max_dist_HLR = max_dist/HLR_pc
        tmp_sel = np.bitwise_and(ALL.califaID__z == g, np.less_equal(ALL.zoneDistance_HLR, max_dist_HLR))
        sel_center__gz[tmp_sel] = True

    if (sel_gals_sample__gz).any():
        f6563__gz = ALL.f6563__z
        f4861__gz = ALL.f4861__z
        W6563__gz = ALL.W6563__z
        SB6563__gz = ALL.SB6563__z

        dist__gz = ALL.zoneDistance_HLR
        tau_V_neb__gz = f_tauVneb(f6563__gz, f4861__gz)
        # sel_gals_sample__gz = np.bitwise_and(sel_gals_sample__gz, sel_center__gz)

        # WHa DIG-COMP-SF decomposition
        sel_WHa_DIG__gz = np.bitwise_and(sel['WHa']['DIG']['z'], sel_gals_sample__gz)
        sel_WHa_COMP__gz = np.bitwise_and(sel['WHa']['COMP']['z'], sel_gals_sample__gz)
        sel_WHa_SF__gz = np.bitwise_and(sel['WHa']['SF']['z'], sel_gals_sample__gz)

        # x = np.ma.log10(tau_V_neb__gz)
        x = tau_V_neb__gz
        y = np.ma.log10(W6563__gz)
        z = dist__gz
        xm, ym, zm = ma_mask_xyz(x=x, y=y, z=z, mask=~sel_gals_sample__gz)
        #x_range = [-2, 0.5]
        x_range = [-0.5, 2]
        f = plt.figure(figsize=(8, 8))
        x_ds = [xm[sel_WHa_DIG__gz].compressed(), xm[sel_WHa_COMP__gz].compressed(), xm[sel_WHa_SF__gz].compressed()]
        y_ds = [ym[sel_WHa_DIG__gz].compressed(), ym[sel_WHa_COMP__gz].compressed(), ym[sel_WHa_SF__gz].compressed()]
        axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, x_range, logWHa_range, 50, 50,
                                             figure=f, c=colors_DIG_COMP_SF, scatter=False, s=1,
                                             ylabel=r'$\tau_V^{neb}$',
                                             xlabel=r'$\log$ W${}_{H\alpha}$ [$\AA$]', histtype='step')
        plot_spearmanr_ax(ax=axS, x=xm.compressed(), y=ym.compressed(), pos_x=0.01, pos_y=0.99, fontsize=14)
        axS.xaxis.set_major_locator(MultipleLocator(0.5))
        axS.xaxis.set_minor_locator(MultipleLocator(0.1))
        axH1.xaxis.set_major_locator(MultipleLocator(0.5))
        axH1.xaxis.set_minor_locator(MultipleLocator(0.1))
        axH2.xaxis.set_major_locator(MaxNLocator(3))
        axS.yaxis.set_major_locator(MultipleLocator(0.5))
        axS.yaxis.set_minor_locator(MultipleLocator(0.1))
        axH1.yaxis.set_major_locator(MaxNLocator(3))
        axH2.yaxis.set_major_locator(MultipleLocator(0.5))
        axH2.yaxis.set_minor_locator(MultipleLocator(0.1))
        aux_ax = axH2.twiny()
        plot_histo_ax(aux_ax, ym.compressed(), stats_txt=False, kwargs_histo=dict(histtype='step', orientation='horizontal', normed=False, range=logWHa_range, color='k', lw=2, ls='-'))
        aux_ax.xaxis.set_major_locator(MaxNLocator(3))
        plt.setp(aux_ax.xaxis.get_majorticklabels(), rotation=270)
        plot_text_ax(axH1, r'W${}_{H\alpha}$ >= %d $\AA$' % SF_WHa_threshold, 0.98, 0.98, 14, 'top', 'right', colors_DIG_COMP_SF[2])
        plot_text_ax(axH1, r'W${}_{H\alpha}$ < %d $\AA$' % DIG_WHa_threshold, 0.98, 0.85, 14, 'top', 'right', colors_DIG_COMP_SF[0])
        plot_text_ax(axH2, r'all zones', 0.98, 0.98, 14, 'top', 'right', 'k')
        scater_kwargs = dict(c=zm, s=3, vmax=3, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
        sc = axS.scatter(xm, ym, **scater_kwargs)
        cbaxes = f.add_axes([0.6, 0.17, 0.12, 0.02])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[0, 1, 2, 3], orientation='horizontal')
        cb.set_label(r'R [HLR]', fontsize=14)
        axS.axhline(y=np.log10(SF_Zhang_threshold), color='k', linestyle='-.', lw=2)
        axS.set_xlim(x_range)
        axS.set_ylim(logWHa_range)
        axS.set_xlabel(r'$\tau_V^{neb}$')
        axS.set_ylabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        # axS.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        xbins = np.arange(x_range[0], x_range[1] + 0.2, 0.2)
        yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), xbins)
        yMedian = prc[2]
        y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
        axS.plot(bin_center, yMedian, 'k-', lw=2)
        for y_prc in y_12sigma:
            axS.plot(bin_center, y_prc, 'k--', lw=2)
        axS.grid()
        f.savefig('fig_tauVNeb_logWHa.png', dpi=dpi_choice, transparent=transp_choice)
        plt.close(f)

        x = tau_V_neb__gz
        y = np.ma.log10(SB6563__gz)
        z = dist__gz
        xm, ym, zm = ma_mask_xyz(x=x, y=y, z=z, mask=~sel_gals_sample__gz)
        x_range = [-0.5, 2]
        f = plt.figure(figsize=(8, 8))
        x_ds = [xm[sel_WHa_DIG__gz].compressed(), xm[sel_WHa_COMP__gz].compressed(), xm[sel_WHa_SF__gz].compressed()]
        y_ds = [ym[sel_WHa_DIG__gz].compressed(), ym[sel_WHa_COMP__gz].compressed(), ym[sel_WHa_SF__gz].compressed()]
        axS, axH1, axH2 = plot_scatter_histo(x_ds, y_ds, x_range, logSBHa_range, 50, 50,
                                             figure=f, c=colors_DIG_COMP_SF, scatter=False, s=1,
                                             ylabel=r'$\tau_V^{neb}$',
                                             xlabel=r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]', histtype='step')
        plot_spearmanr_ax(ax=axS, x=xm.compressed(), y=ym.compressed(), pos_x=0.01, pos_y=0.99, fontsize=14)
        axS.xaxis.set_major_locator(MultipleLocator(0.5))
        axS.xaxis.set_minor_locator(MultipleLocator(0.1))
        axH1.xaxis.set_major_locator(MultipleLocator(0.5))
        axH1.xaxis.set_minor_locator(MultipleLocator(0.1))
        axH2.xaxis.set_major_locator(MaxNLocator(3))
        axS.yaxis.set_major_locator(MultipleLocator(0.5))
        axS.yaxis.set_minor_locator(MultipleLocator(0.1))
        axH1.yaxis.set_major_locator(MaxNLocator(3))
        axH2.yaxis.set_major_locator(MultipleLocator(0.5))
        axH2.yaxis.set_minor_locator(MultipleLocator(0.1))
        aux_ax = axH2.twiny()
        plot_histo_ax(aux_ax, ym.compressed(), stats_txt=False, kwargs_histo=dict(histtype='step', orientation='horizontal', normed=False, range=logSBHa_range, color='k', lw=2, ls='-'))
        aux_ax.xaxis.set_major_locator(MaxNLocator(3))
        plt.setp(aux_ax.xaxis.get_majorticklabels(), rotation=270)
        plot_text_ax(axH1, r'W${}_{H\alpha}$ >= %d $\AA$' % SF_WHa_threshold, 0.98, 0.98, 14, 'top', 'right', colors_DIG_COMP_SF[2])
        plot_text_ax(axH1, r'W${}_{H\alpha}$ < %d $\AA$' % DIG_WHa_threshold, 0.98, 0.85, 14, 'top', 'right', colors_DIG_COMP_SF[0])
        plot_text_ax(axH2, r'all zones', 0.98, 0.98, 14, 'top', 'right', 'k')
        scater_kwargs = dict(c=zm, s=3, vmax=3, vmin=0, cmap=cmap_R, marker='o', edgecolor='none')
        sc = axS.scatter(xm, ym, **scater_kwargs)
        cbaxes = f.add_axes([0.6, 0.17, 0.12, 0.02])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[0, 1, 2, 3], orientation='horizontal')
        cb.set_label(r'R [HLR]', fontsize=14)
        axS.axhline(y=np.log10(SF_Zhang_threshold), color='k', linestyle='-.', lw=2)
        axS.set_xlim(x_range)
        axS.set_ylim(logSBHa_range)
        axS.set_xlabel(r'$\tau_V^{neb}$')
        # axS.set_ylabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        axS.set_ylabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        xbins = np.arange(x_range[0], x_range[1] + 0.2, 0.2)
        yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), xbins)
        yMedian = prc[2]
        y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
        axS.plot(bin_center, yMedian, 'k-', lw=2)
        for y_prc in y_12sigma:
            axS.plot(bin_center, y_prc, 'k--', lw=2)
        axS.grid()
        f.savefig('fig_tauVNeb_logSBHa.png', dpi=dpi_choice, transparent=transp_choice)
        plt.close(f)


def fig_compare_stacked_tauVNeb(ALL, sel, gals, suffix):
    sel_gals_sample__gz = sel['gals_sample__z']
    sel_WHa = sel['WHa']

    N_cols = 1
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 6, N_rows * 4))
    cmap = cmap_discrete(colors=colors_DIG_COMP_SF)

    row = 0
    for califaID in gals:
        tmp_bin__R = np.arange(0., 3. + 0.2, 0.2)
        RbinCenterStacked__R = (tmp_bin__R[:-1] + tmp_bin__R[1:]) / 2.0
        NR = len(RbinCenterStacked__R)
        stdata__c = {}
        tau_V_neb_stacked__c = {}
        for k in classif_labels:
            stdata__c[k] = EmLinesStackedDataCube('/Users/lacerda/RGB/eml/v04/%d-%d/%s-RadBinStackedSpectra.EML.%s.MC100.fits' % (DIG_WHa_threshold, SF_WHa_threshold, califaID, k))
            iHa = stdata__c[k].lines.index('6563')
            iHb = stdata__c[k].lines.index('4861')
            tau_V_neb_stacked__c[k] = f_tauVneb(stdata__c[k].flux[iHa], stdata__c[k].flux[iHb])
        ax1 = axArr[row]

        mto = ALL.get_gal_prop_unique(califaID, ALL.mto)
        N_x = ALL.get_gal_prop_unique(califaID, ALL.N_x)
        N_y = ALL.get_gal_prop_unique(califaID, ALL.N_y)
        gal_sample__z = ALL.get_gal_prop(califaID, sel_gals_sample__gz)
        N_pixel = N_x * N_y
        N_zone = ALL.get_gal_prop_unique(califaID, ALL.N_zone)
        distance_HLR__z = ALL.get_gal_prop(califaID, ALL.zoneDistance_HLR)
        f__lz = {'%s' % L: np.ma.masked_array(ALL.get_gal_prop(califaID, 'f%s__z' % L), mask=~gal_sample__z) for L in lines}

        x = distance_HLR__z
        y = f_tauVneb(f__lz['6563'], f__lz['4861'])
        x_range = distance_range
        y_range = (1/0.34652) * np.log(10**(lineratios_range['6563/4861'][0])/2.86), (1/0.34652) * np.log(10**(lineratios_range['6563/4861'][1])/2.86)
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa, sel_gals_sample__gz)
        sc = ax1.scatter(x, y, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=6, edgecolor='none', alpha=0.9)
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        txt = 'CALIFA %s' % califaID[1:]
        plot_text_ax(ax1, txt, 0.98, 0.02, 16, 'bottom', 'right', color='k')

        sel_DIG__z, sel_COMP__z, sel_SF__z, _ = get_selections_zones(ALL, califaID, sel_WHa, sel_gals_sample__gz)

        min_npts = 0
        xm, ym = ma_mask_xyz(x, y, mask=~sel_DIG__z)
        yMean_DIG, yPrc_DIG, bin_center_DIG, npts_DIG = stats_med12sigma(xm.compressed(), ym.compressed(), tmp_bin__R)
        if npts_DIG.any():
            sel = npts_DIG > min_npts
            # ax1.plot(bin_center_DIG[sel], yPrc_DIG[2][sel], '-', lw=1, c=colors_lines_DIG_COMP_SF[0])
            ax1.plot(bin_center_DIG[sel], yMean_DIG[sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[0], markersize=10)
            # ax1.plot(RbinCenterStacked__R, tau_V_neb_stacked__c['DIG'], '-', lw=2, c=colors_lines_DIG_COMP_SF[0])
            ax1.plot(RbinCenterStacked__R, tau_V_neb_stacked__c['DIG'], linestyle='', marker='^', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[0], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_COMP__z)
        yMean_COMP, yPrc_COMP, bin_center_COMP, npts_COMP = stats_med12sigma(xm.compressed(), ym.compressed(), tmp_bin__R)
        if npts_COMP.any():
            sel = npts_COMP > min_npts
            # ax1.plot(bin_center_COMP[sel], yPrc_COMP[2][sel], '-', lw=1, c=colors_lines_DIG_COMP_SF[1])
            ax1.plot(bin_center_COMP[sel], yMean_COMP[sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[1], markersize=10)
            # ax1.plot(RbinCenterStacked__R, tau_V_neb_stacked__c['COMP'], '-', lw=2, c=colors_lines_DIG_COMP_SF[1])
            ax1.plot(RbinCenterStacked__R, tau_V_neb_stacked__c['COMP'], linestyle='', marker='^', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[1], markersize=10)
        xm, ym = ma_mask_xyz(x, y, mask=~sel_SF__z)
        yMean_SF, yPrc_SF, bin_center_SF, npts_SF = stats_med12sigma(xm.compressed(), ym.compressed(), tmp_bin__R)
        if npts_SF.any():
            sel = npts_SF > min_npts
            # ax1.plot(bin_center_SF[sel], yPrc_SF[2][sel], '-', lw=1, c=colors_lines_DIG_COMP_SF[2])
            ax1.plot(bin_center_SF[sel], yMean_SF[sel], linestyle='', marker='*', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[2], markersize=10)
            # ax1.plot(RbinCenterStacked__R, tau_V_neb_stacked__c['SF'], '-', lw=2, c=colors_lines_DIG_COMP_SF[2])
            ax1.plot(RbinCenterStacked__R, tau_V_neb_stacked__c['SF'], linestyle='', marker='^', markeredgewidth=1, markeredgecolor='k', c=colors_lines_DIG_COMP_SF[2], markersize=10)
        ax2 = ax1.twinx()
        mn, mx = ax1.get_ylim()
        unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
        ax2.set_ylim(unit_converter(mn), unit_converter(mx))
        # ax2.set_ylim((1/0.34652) * np.log(10**(mn)/2.86), (1/0.34652) * np.log(10**(mx)/2.86))

        if row < 2:
            plt.setp(ax1.get_xticklabels(), visible=False)
        if row == 2:
            ax1.set_xlabel('R [HLR]')
            cbaxes = add_subplot_axes(ax1, [0.08, -0.05, 0.35, 0.07])
            cb = plt.colorbar(sc, cax=cbaxes, ticks=[1.+2/6., 2, 3-2/6.], orientation='horizontal')
            cb.ax.set_xticklabels(classif_labels, fontsize=12)
        if row == 1:
            ax1.set_ylabel(r'$\tau_V^{neb}$')
            ax2.set_ylabel(r'$\log\ H\alpha/H\beta$')
        row += 1

    f.tight_layout(h_pad=0)
    f.savefig('compare_stacked_%s.png' % suffix, dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


def fig_compare_stacked_lines(ALL, sel, gals, sample_choice):
    sel_gals_sample__gz = sel['gals_sample__z']
    sel_WHa = sel['WHa']
    N_gals = len(gals)
    cmap = cmap_discrete(colors=colors_DIG_COMP_SF)
    SNHb__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    tau_V_neb_stacked__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    tau_V_neb_mean__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    tau_V_neb_median__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    for i_gal, califaID in enumerate(gals):
        sel_DIG__z, sel_COMP__z, sel_SF__z, _ = get_selections_zones(ALL, califaID, sel_WHa, sel_gals_sample__gz)
        sel_class = {'DIG': sel_DIG__z, 'COMP': sel_COMP__z, 'SF': sel_SF__z}
        distance_HLR__z = ALL.get_gal_prop(califaID, ALL.zoneDistance_HLR)
        gal_sample__z = ALL.get_gal_prop(califaID, sel_gals_sample__gz)
        fHb__z = np.ma.masked_array(ALL.get_gal_prop(califaID, 'f4861__z'), mask=~gal_sample__z)
        efHb__z = np.ma.masked_array(ALL.get_gal_prop(califaID, 'ef4861__z'), mask=~gal_sample__z)
        fHa__z = np.ma.masked_array(ALL.get_gal_prop(califaID, 'f6563__z'), mask=~gal_sample__z)
        tau_V_neb__z = f_tauVneb(fHa__z, fHb__z)
        SNR_min = eval(sample_choice[-1].replace('S', ''))
        # print SNR_min
        for k in classif_labels:
            tmp_stdata = EmLinesStackedDataCube('/Users/lacerda/RGB/eml/v04/%d-%d/%s-RadBinStackedSpectra.EML.%s.MC100.fits' % (DIG_WHa_threshold, SF_WHa_threshold, califaID, k))
            iHa = tmp_stdata.lines.index('6563')
            iHb = tmp_stdata.lines.index('4861')
            fHb__R = tmp_stdata.flux[iHb]
            SNHb__R = tmp_stdata.snr__Lz[iHb]
            fHa__R = tmp_stdata.flux[iHa]
            SNHb__cgR[k][i_gal, :] = SNHb__R
            # print fHb__R
            # print tmp_stdata.eflux[iHb]
            # print SNHb__R
            # print '--------'
            tmp_tauVneb = f_tauVneb(fHa__R, fHb__R)
            tau_V_neb_stacked__cgR[k][i_gal, :] = tmp_tauVneb
            xm, ym = ma_mask_xyz(distance_HLR__z, tau_V_neb__z, mask=(~sel_class[k]))
            yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
            tau_V_neb_median__cgR[k][i_gal, :] = yPrc[2]
            tau_V_neb_mean__cgR[k][i_gal, :] = yMean
            tmp_stdata.close()
        N_cols = 2
        N_rows = 2
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 6, N_rows * 4))
        (ax1, ax2), (ax3, ax4) = axArr
        cmap = cmap_discrete(colors=colors_DIG_COMP_SF)
        map__z = create_segmented_map_zones(ALL, califaID, sel_WHa, sel_gals_sample__gz)
        x = distance_HLR__z
        y = tau_V_neb__z
        y_stacked__cR = {k: tau_V_neb_stacked__cgR[k][i_gal, :] for k in classif_labels}
        y_mean__cR = {k: tau_V_neb_mean__cgR[k][i_gal, :] for k in classif_labels}
        y_median__cR = {k: tau_V_neb_median__cgR[k][i_gal, :] for k in classif_labels}
        sc = ax1.scatter(x, y, c=map__z, vmin=1, vmax=3, cmap=cmap, marker='o', s=5, edgecolor='none', alpha=0.5)
        ax1.plot(R_bin_center__r, y_stacked__cR['DIG'], '-', lw=2, c=colors_lines_DIG_COMP_SF[0], label='stacked')
        ax1.plot(R_bin_center__r, y_stacked__cR['COMP'], '-', lw=2, c=colors_lines_DIG_COMP_SF[1])
        ax1.plot(R_bin_center__r, y_stacked__cR['SF'], '-', lw=2, c=colors_lines_DIG_COMP_SF[2])
        ax1.plot(R_bin_center__r, y_mean__cR['DIG'], '--', lw=1, c=colors_lines_DIG_COMP_SF[0], label='mean')
        ax1.plot(R_bin_center__r, y_mean__cR['COMP'], '--', lw=1, c=colors_lines_DIG_COMP_SF[1])
        ax1.plot(R_bin_center__r, y_mean__cR['SF'], '--', lw=1, c=colors_lines_DIG_COMP_SF[2])
        ax1.plot(R_bin_center__r, y_median__cR['DIG'], '-.', lw=1, c=colors_lines_DIG_COMP_SF[0], label='median')
        ax1.plot(R_bin_center__r, y_median__cR['COMP'], '-.', lw=1, c=colors_lines_DIG_COMP_SF[1])
        ax1.plot(R_bin_center__r, y_median__cR['SF'], '-.', lw=1, c=colors_lines_DIG_COMP_SF[2])
        ax1.legend(loc='upper right', frameon=False, fontsize=9)
        ax1.set_xlim(distance_range)
        ax1.set_ylim([-2, 2.5])
        ax1.set_xlabel('R [HLR]')
        ax1.set_ylabel(r'$\tau_V^{neb}$')
        ax1.xaxis.set_major_locator(MultipleLocator(1))
        ax1.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax1.grid()
        ax__c = {'DIG': ax2, 'COMP': ax3, 'SF': ax4}
        logSNRrange = [0, 2.5]
        for k in classif_labels:
            ax = ax__c[k]
            x = np.ma.log10(1. + SNHb__cgR[k][i_gal, :])
            y_st_mean = y_stacked__cR[k] - y_mean__cR[k]
            y_st_median = y_stacked__cR[k] - y_median__cR[k]
            sc = ax.scatter(x, y_st_mean, c='k', marker='o', s=50, edgecolor='none', label='f = mean')
            sc = ax.scatter(x, y_st_median, c='b', marker='+', s=50, edgecolor='none', label='f = median')
            ax.legend(loc='upper right', frameon=False, fontsize=9)
            ax.set_xlim(logSNRrange)
            ax.set_ylim([-1, 1])
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            ax.grid(which='minor')
            ax.set_xlabel(r'$\log$ (1+SN${}_{H\beta}$) (stacked)')
            ax.set_ylabel(r'$\tau_V^{neb}$ - f')
            txt = 'CALIFA %s (%s)' % (califaID[1:], k)
            plot_text_ax(ax, txt, 0.01, 0.99, 16, 'top', 'left', color='k')
        f.tight_layout(h_pad=0)
        f.savefig('%s_compare_stacked_tauVstats_%s_%s.png' % (califaID, sample_choice[0], sample_choice[1]), dpi=dpi_choice, transparent=transp_choice)
        plt.close(f)
    N_cols = 3
    N_rows = 2
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 6, N_rows * 4))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axArr
    cmap = cmap_discrete(colors=colors_DIG_COMP_SF)

    x_stackeddata__c = {k: np.hstack(np.ma.getdata(SNHb__cgR[k])) for k in classif_labels}
    x_stackedmask__c = {k: np.hstack(np.ma.getmaskarray(SNHb__cgR[k])) for k in classif_labels}
    x_stacked__c = {k: np.ma.masked_array(x_stackeddata__c[k], mask=x_stackedmask__c[k]) for k in classif_labels}
    y_stackeddata__c = {k: np.hstack(np.ma.getdata(tau_V_neb_stacked__cgR[k])) for k in classif_labels}
    y_stackedmask__c = {k: np.hstack(np.ma.getmaskarray(tau_V_neb_stacked__cgR[k])) for k in classif_labels}
    y_stacked__c = {k: np.ma.masked_array(y_stackeddata__c[k], mask=y_stackedmask__c[k]) for k in classif_labels}
    y_meandata__c = {k: np.hstack(np.ma.getdata(tau_V_neb_mean__cgR[k])) for k in classif_labels}
    y_meanmask__c = {k: np.hstack(np.ma.getmaskarray(tau_V_neb_mean__cgR[k])) for k in classif_labels}
    y_mean__c = {k: np.ma.masked_array(y_meandata__c[k], mask=y_meanmask__c[k]) for k in classif_labels}
    y_mediandata__c = {k: np.hstack(np.ma.getdata(tau_V_neb_median__cgR[k])) for k in classif_labels}
    y_medianmask__c = {k: np.hstack(np.ma.getmaskarray(tau_V_neb_median__cgR[k])) for k in classif_labels}
    y_median__c = {k: np.ma.masked_array(y_mediandata__c[k], mask=y_medianmask__c[k]) for k in classif_labels}

    ax__c = {'DIG': (ax1, ax4), 'COMP': (ax2, ax5), 'SF': (ax3, ax6)}
    logSNRrange = [0, 2.5]
    for k in classif_labels:
        x_stacked__c[k][np.isnan(x_stacked__c[k])] = np.ma.masked
        y_stacked__c[k][np.isnan(y_stacked__c[k])] = np.ma.masked
        y_mean__c[k][np.isnan(y_mean__c[k])] = np.ma.masked
        y_median__c[k][np.isnan(y_median__c[k])] = np.ma.masked
        ax_R, ax_H = ax__c[k]
        x = np.ma.log10(1. + x_stacked__c[k])
        y_st_mean = y_stacked__c[k] - y_mean__c[k]
        y_st_median = y_stacked__c[k] - y_median__c[k]
        # print y_st_mean.compressed()
        # print y_st_median.compressed()
        ax_R.scatter(x, y_st_mean, c='k', marker='o', s=20, edgecolor='none', label='f = mean')
        ax_R.scatter(x, y_st_median, c='b', marker='+', s=25, edgecolor='none', label='f = median')
        ax_R.legend(loc='upper right', frameon=False, fontsize=9)
        ax_R.set_xlim(logSNRrange)
        ax_R.set_ylim([-1.5, 1.5])
        ax_R.xaxis.set_major_locator(MultipleLocator(0.5))
        ax_R.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax_R.yaxis.set_major_locator(MultipleLocator(0.5))
        ax_R.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax_R.grid(which='minor')
        ax_R.set_xlabel(r'$\log$ (1+SN${}_{H\beta}$) (stacked)')
        ax_R.set_ylabel(r'$\tau_V^{neb}$ - f')
        txt = '%s' % k
        plot_text_ax(ax_R, txt, 0.01, 0.99, 16, 'top', 'left', color='k')
        plot_histo_ax(ax_H, y_st_mean.compressed(), first=True, stats_txt=True, pos_x=0.38, dataset_names='f=mean', c='k', kwargs_histo=dict(bins=20, histtype='stepfilled', color='k', range=[-1.5, 1.5], lw=3))
        plot_histo_ax(ax_H, y_st_median.compressed(), stats_txt=True, dataset_names='f=median', c='b', kwargs_histo=dict(bins=20, histtype='stepfilled', color='b', range=[-1.5, 1.5], lw=3))
        # plot_histo_ax(ax_H, y_st_median.compressed(), stats_txt=True, kwargs_histo=dict(bins=50, histtype='stepfilled', color='b', range=[-1.5, 1.5], lw=3))
        ax_H.set_xlabel(r'$\tau_V^{neb}$ - f')
        ax_H.xaxis.set_major_locator(MultipleLocator(0.5))
        ax_H.xaxis.set_minor_locator(MultipleLocator(0.25))
    f.tight_layout(h_pad=0)
    f.savefig('ALL_compare_stacked_tauVstats_%s_%s.png' % (sample_choice[0], sample_choice[1]), dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


def fig_stacked_statsalldata_histograms(ALL, sel, gals, sample_choice):
    from astropy.io import fits
    from pycasso import fitsQ3DataCube
    radbinstacksfits_dir = '/Users/lacerda/dev/astro/dig/runs/stackspectra/stack-fits/20170426'
    radbinstacksemlfits_dir = '/Users/lacerda/RGB/eml/v04'
    pycassofits_dir = '/Users/lacerda/califa/legacy/q054/superfits/Bgstf6e'
    sel_gals_sample__gz = sel['gals_sample__z']
    sel_WHa = sel['WHa']
    N_gals = len(gals)
    cmap = cmap_discrete(colors=colors_DIG_COMP_SF)
    sum_Ha__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    sum_Hb__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    fHa_stack__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    fHb_stack__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    # sum_Ha__cgR, sum_Hb__cgR = get_radstacked_zone_data(ALL, sel, gals)
    tau_V_neb_sum__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    tau_V_neb_stacked__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    tau_V_neb_mean__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    tau_V_neb_median__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    for i_gal, califaID in enumerate(gals):
        sel_DIG__z, sel_COMP__z, sel_SF__z, _ = get_selections_zones(ALL, califaID, sel_WHa, sel_gals_sample__gz)
        sel_class = {'DIG': sel_DIG__z, 'COMP': sel_COMP__z, 'SF': sel_SF__z}
        distance_HLR__z = ALL.get_gal_prop(califaID, ALL.zoneDistance_HLR)
        gal_sample__z = ALL.get_gal_prop(califaID, sel_gals_sample__gz)
        fHa__z = np.ma.masked_array(ALL.get_gal_prop(califaID, 'f6563__z'), mask=~gal_sample__z)
        fHb__z = np.ma.masked_array(ALL.get_gal_prop(califaID, 'f4861__z'), mask=~gal_sample__z)
        efHb__z = np.ma.masked_array(ALL.get_gal_prop(califaID, 'ef4861__z'), mask=~gal_sample__z)
        tau_V_neb__z = f_tauVneb(fHa__z, fHb__z)
        SNR_min = eval(sample_choice[-1].replace('S', ''))
        # print SNR_min
        hdu = fits.open('%s/%d-%d/%s-RadBinStackedSpectra.fits' % (radbinstacksfits_dir, DIG_WHa_threshold, SF_WHa_threshold, califaID))
        K = fitsQ3DataCube('%s/%s_synthesis_eBR_v20_q054.d22a512.ps03.k1.mE.CCM.Bgstf6e.fits' % (pycassofits_dir, califaID))
        NRbins = hdu[0].header['NRBINS']
        N_cols = 3
        N_rows = 3
        f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 6, N_rows * 4))
        ax_dict = {'DIG': axArr[0], 'COMP': axArr[1], 'SF': axArr[2]}
        for k in classif_labels:
            ax_DIG = ax_dict[k][0]
            ax_COMP = ax_dict[k][1]
            ax_SF = ax_dict[k][2]
            bin_zones__R = [np.unique(K.qZones[hdu['%s_BIN_SEGMAP__RYX' % k].data[iR].astype('bool')]) for iR in range(NRbins)]
            # print califaID, k
            # print [len(m) for m in bin_zones__R]
            # print [fHa__z[m].count() for m in bin_zones__R]
            # print [fHb__z[m].count() for m in bin_zones__R]
            sum_Ha__cgR[k][i_gal, :] = np.ma.array([fHa__z[m].sum() for m in bin_zones__R])
            sum_Hb__cgR[k][i_gal, :] = np.ma.array([fHb__z[m].sum() for m in bin_zones__R])
            tmp_stdata = EmLinesStackedDataCube('%s/%d-%d/%s-RadBinStackedSpectra.EML.%s.MC100.fits' % (radbinstacksemlfits_dir, DIG_WHa_threshold, SF_WHa_threshold, califaID, k))
            iHa = tmp_stdata.lines.index('6563')
            iHb = tmp_stdata.lines.index('4861')
            fHa__R = tmp_stdata.flux[iHa]
            fHb__R = tmp_stdata.flux[iHb]
            fHa_stack__cgR[k][i_gal, :] = fHa__R
            fHb_stack__cgR[k][i_gal, :] = fHb__R
            tau_V_neb_sum__cgR[k][i_gal, :] = f_tauVneb(sum_Ha__cgR[k][i_gal], sum_Hb__cgR[k][i_gal])
            tau_V_neb_stacked__cgR[k][i_gal, :] = f_tauVneb(fHa__R, fHb__R)
            xm, ym = ma_mask_xyz(distance_HLR__z, tau_V_neb__z, mask=(~sel_class[k]))
            yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
            tau_V_neb_median__cgR[k][i_gal, :] = yPrc[2]
            tau_V_neb_mean__cgR[k][i_gal, :] = yMean
            tmp_stdata.close()
            ####################
            x = sum_Ha__cgR[k][i_gal]
            y = fHa__R
            ax_dict[k][0].scatter(x, y, c=colors_DIG_COMP_SF_dict[k], marker='o', s=50, edgecolor='none')
            ax_dict[k][0].set_xlabel(r'$\sum_{z \in R} F_{H\alpha}$')
            ax_dict[k][0].set_ylabel(r'$F_{H\alpha}^{stack}$')
            data_min = x.min()
            if y.min() < data_min:
                data_min = y.min()
            data_max = x.max()
            if y.max() > data_max:
                data_max = y.max()
            ax_dict[k][0].set_xlim([data_min, data_max])
            ax_dict[k][0].set_ylim([data_min, data_max])
            ax_dict[k][0].plot(ax_dict[k][0].get_xlim(), ax_dict[k][0].get_xlim(), '--k')
            plot_text_ax(ax_dict[k][0], k, 0.01, 0.99, 16, 'top', 'left', color=colors_DIG_COMP_SF_dict[k])
            ####################
            x = sum_Hb__cgR[k][i_gal]
            y = fHb__R
            ax_dict[k][1].scatter(x, y, c=colors_DIG_COMP_SF_dict[k], marker='o', s=50, edgecolor='none')
            ax_dict[k][1].set_xlabel(r'$\sum_{z \in R} F_{H\beta}$')
            ax_dict[k][1].set_ylabel(r'$F_{H\beta}^{stack}$')
            data_min = x.min()
            if y.min() < data_min:
                data_min = y.min()
            data_max = x.max()
            if y.max() > data_max:
                data_max = y.max()
            ax_dict[k][1].set_xlim([data_min, data_max])
            ax_dict[k][1].set_ylim([data_min, data_max])
            ax_dict[k][1].plot(ax_dict[k][1].get_xlim(), ax_dict[k][1].get_xlim(), '--k')
            ####################
            x = tau_V_neb_sum__cgR[k][i_gal]
            y = tau_V_neb_stacked__cgR[k][i_gal]
            ax_dict[k][2].scatter(x, y, c=colors_DIG_COMP_SF_dict[k], marker='o', s=50, edgecolor='none')
            ax_dict[k][2].set_xlabel(r'$\tau_V$ [$\sum_{z \in R} F_{H\alpha}$ and $F_{H\beta}$]')
            ax_dict[k][2].set_ylabel(r'$\tau_V^{stack}$')
            data_min = x.min()
            if y.min() < data_min:
                data_min = y.min()
            data_max = x.max()
            if y.max() > data_max:
                data_max = y.max()
            ax_dict[k][2].set_xlim([data_min, data_max])
            ax_dict[k][2].set_ylim([data_min, data_max])
            ax_dict[k][2].plot(ax_dict[k][2].get_xlim(), ax_dict[k][2].get_xlim(), '--k')
        del hdu
        K.close()
        del K
        f.tight_layout(h_pad=0)
        f.savefig('%s_stacked_statsalldata_histograms_%s_%s.png' % (califaID, sample_choice[0], sample_choice[1]), dpi=dpi_choice, transparent=transp_choice)
        plt.close(f)
    ####################
    N_cols = 3
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 6, N_rows * 4))
    ax_dict = {'DIG': axArr[0], 'COMP': axArr[1], 'SF': axArr[2]}
    for k in classif_labels:
        ax_DIG = ax_dict[k][0]
        ax_COMP = ax_dict[k][1]
        ax_SF = ax_dict[k][2]
        ####################
        x = np.ma.hstack(sum_Ha__cgR[k])
        y = np.ma.hstack(fHa_stack__cgR[k])
        ax_dict[k][0].scatter(x, y, c=colors_DIG_COMP_SF_dict[k], marker='o', s=50, edgecolor='none')
        ax_dict[k][0].set_xlabel(r'$\sum_{z \in R} F_{H\alpha}$')
        ax_dict[k][0].set_ylabel(r'$F_{H\alpha}^{stack}$')
        data_min = x.min()
        if y.min() < data_min:
            data_min = y.min()
        data_max = x.max()
        if y.max() > data_max:
            data_max = y.max()
        ax_dict[k][0].set_xlim([data_min, data_max])
        ax_dict[k][0].set_ylim([data_min, data_max])
        ax_dict[k][0].plot(ax_dict[k][0].get_xlim(), ax_dict[k][0].get_xlim(), '--k')
        plot_text_ax(ax_dict[k][0], k, 0.01, 0.99, 16, 'top', 'left', color=colors_DIG_COMP_SF_dict[k])
        ####################
        x = np.ma.hstack(sum_Hb__cgR[k])
        y = np.ma.hstack(fHb_stack__cgR[k])
        ax_dict[k][1].scatter(x, y, c=colors_DIG_COMP_SF_dict[k], marker='o', s=50, edgecolor='none')
        ax_dict[k][1].set_xlabel(r'$\sum_{z \in R} F_{H\beta}$')
        ax_dict[k][1].set_ylabel(r'$F_{H\beta}^{stack}$')
        data_min = x.min()
        if y.min() < data_min:
            data_min = y.min()
        data_max = x.max()
        if y.max() > data_max:
            data_max = y.max()
        ax_dict[k][1].set_xlim([data_min, data_max])
        ax_dict[k][1].set_ylim([data_min, data_max])
        ax_dict[k][1].plot(ax_dict[k][1].get_xlim(), ax_dict[k][1].get_xlim(), '--k')
        ####################
        x = np.ma.hstack(tau_V_neb_sum__cgR[k])
        y = np.ma.hstack(tau_V_neb_stacked__cgR[k])
        ax_dict[k][2].scatter(x, y, c=colors_DIG_COMP_SF_dict[k], marker='o', s=50, edgecolor='none')
        ax_dict[k][2].set_xlabel(r'$\tau_V$ [$\sum_{z \in R} F_{H\alpha}$ and $F_{H\beta}$]')
        ax_dict[k][2].set_ylabel(r'$\tau_V^{stack}$')
        data_min = x.min()
        if y.min() < data_min:
            data_min = y.min()
        data_max = x.max()
        if y.max() > data_max:
            data_max = y.max()
        ax_dict[k][2].set_xlim([data_min, data_max])
        ax_dict[k][2].set_ylim([data_min, data_max])
        ax_dict[k][2].plot(ax_dict[k][2].get_xlim(), ax_dict[k][2].get_xlim(), '--k')
    f.tight_layout(h_pad=0)
    f.savefig('ALL_stacked_statsalldata_%s_%s.png' % (sample_choice[0], sample_choice[1]), dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


def fig_stacks_BPT(gals, sample_choice):
    print 'fig_stacks_BPT'
    N_gals = len(gals)
    cmap = cmap_discrete(colors=colors_DIG_COMP_SF)
    SNHb__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    fHa__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    fHb__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    fO3__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    fN2__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    SNR_min = eval(sample_choice[-1].replace('S', ''))
    for i_gal, califaID in enumerate(gals):
        # print SNR_min
        for k in classif_labels:
            tmp_stdata = EmLinesStackedDataCube('/Users/lacerda/RGB/eml/v04/%d-%d/%s-RadBinStackedSpectra.EML.%s.MC100.fits' % (DIG_WHa_threshold, SF_WHa_threshold, califaID, k))
            iHb = tmp_stdata.lines.index('4861')
            iO3 = tmp_stdata.lines.index('5007')
            iHa = tmp_stdata.lines.index('6563')
            iN2 = tmp_stdata.lines.index('6583')
            fHb__cgR[k][i_gal, :] = np.ma.copy(tmp_stdata.flux[iHb])
            fO3__cgR[k][i_gal, :] = np.ma.copy(tmp_stdata.flux[iO3])
            fHa__cgR[k][i_gal, :] = np.ma.copy(tmp_stdata.flux[iHa])
            fN2__cgR[k][i_gal, :] = np.ma.copy(tmp_stdata.flux[iN2])
            del tmp_stdata
    fHb__c = {k: np.ma.hstack(fHb__cgR[k]) for k in classif_labels}
    fO3__c = {k: np.ma.hstack(fO3__cgR[k]) for k in classif_labels}
    fHa__c = {k: np.ma.hstack(fHa__cgR[k]) for k in classif_labels}
    fN2__c = {k: np.ma.hstack(fN2__cgR[k]) for k in classif_labels}

    # N_cols = 1
    # N_rows = 1
    # f, axArr = plt.subplots(N_rows, N_cols, dpi=200, figsize=(N_cols * 5, N_rows * 5))
    # cmap = cmap_discrete(colors=colors_DIG_COMP_SF)
    # ax1, ax2, ax3 = axArr
    # f.suptitle(r'%s - %s (%s): %d pixels (%d zones)' % (califaID, mto, get_NEDName_by_CALIFAID(califaID)[0], N_pixel, N_zone))
    f = plt.figure(figsize=(5,5 ))
    ax1 = f.gca()
    L = Lines()
    extent = [-1.5, 1, -1.5, 1.5]
    tmp_x = []
    tmp_y = []
    for k in classif_labels:
        N2Ha = np.ma.log10(fN2__c[k]/fHa__c[k])
        O3Hb = np.ma.log10(fO3__c[k]/fHb__c[k])
        xm, ym = ma_mask_xyz(N2Ha, O3Hb)
        sc = ax1.scatter(xm, ym, c=colors_DIG_COMP_SF_dict[k], vmin=1, vmax=3, marker='o', s=10, edgecolor='none')
        tmp_x.append(xm)
        tmp_y.append(ym)
        print k, xm.count()
    xm = np.ma.hstack(tmp_x)
    ym = np.ma.hstack(tmp_y)
    ax1.set_xlim(extent[0:2])
    ax1.set_ylim(extent[2:4])
    ax1.set_xlabel(r'$\log\ [NII]/H\alpha$')
    ax1.set_ylabel(r'$\log\ [OIII]/H\beta$')
    N = xm.count()
    c = ''
    if (xm.compressed() < extent[0]).any():
        c += 'x-'
    if (xm.compressed() > extent[1]).any():
        c += 'x+'
    if (ym.compressed() < extent[2]).any():
        c += 'y-'
    if (ym.compressed() > extent[3]).any():
        c += 'y+'
    plot_text_ax(ax1, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
    plot_text_ax(ax1, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax1, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax1, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
    ax1.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
    ax1.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
    ax1.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
    plot_text_ax(ax1, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=35)  # 44.62)
    ax1.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
    L.fixCF10('S06')
    f.tight_layout(h_pad=0)
    f.savefig('stacks_BPT_%s_%s.png' % (sample_choice[0], sample_choice[1]), dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


def fig7_stacks(gals, sample_choice):
    print '##############################################################'
    print '######################## FIG 7 STACKS ########################'
    print '##############################################################'
    N_gals = len(gals)
    SNR_min = eval(sample_choice[-1].replace('S', ''))
    fHa__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    fHb__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    tau_V_neb__cgR = {'DIG': np.ma.masked_all((N_gals, N_R_bins)), 'COMP': np.ma.masked_all((N_gals, N_R_bins)), 'SF': np.ma.masked_all((N_gals, N_R_bins))}
    delta_tau__gR = np.ma.masked_all((N_gals, N_R_bins))
    for i_gal, califaID in enumerate(gals):
        # print SNR_min
        print califaID
        for k in classif_labels:
            tmp_stdata = EmLinesStackedDataCube('/Users/lacerda/RGB/eml/v04/%d-%d/%s-RadBinStackedSpectra.EML.%s.MC100.fits' % (DIG_WHa_threshold, SF_WHa_threshold, califaID, k))
            iHb = tmp_stdata.lines.index('4861')
            iHa = tmp_stdata.lines.index('6563')
            fHb__cgR[k][i_gal, :] = np.ma.masked_array(np.ma.getdata(tmp_stdata.flux[iHb]), np.ma.getmaskarray(tmp_stdata.flux[iHb]), copy=True)
            fHa__cgR[k][i_gal, :] = np.ma.masked_array(np.ma.getdata(tmp_stdata.flux[iHa]), np.ma.getmaskarray(tmp_stdata.flux[iHa]), copy=True)
            fHb__cgR[k][i_gal, (fHb__cgR[k][i_gal] <= 0.0)] = np.ma.masked
            fHa__cgR[k][i_gal, (fHa__cgR[k][i_gal] <= 0.0)] = np.ma.masked
            tau_V_neb__cgR[k][i_gal, :] = f_tauVneb(fHa__cgR[k][i_gal, :], fHb__cgR[k][i_gal, :])
            # print type(tau_V_neb__cgR[k][i_gal])
            del tmp_stdata
        delta_tau__gR[i_gal, :] = tau_V_neb__cgR['SF'][i_gal] - tau_V_neb__cgR['DIG'][i_gal]
        print '\t<tauVneb(SF)>_R = %.2f' % tau_V_neb__cgR['SF'][i_gal].mean()
        print '\t<tauVneb(DIG)>_R = %.2f' % tau_V_neb__cgR['DIG'][i_gal].mean()
        print '\t<delta_tauVneb(SF-DIG)>_R = %.2f' % delta_tau__gR[i_gal].mean()
    N_cols = 1
    N_rows = 3
    f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 4.8))
    ax1, ax2, ax3 = axArr
    range = [-2, 2]
    xDs = [tau_V_neb__cgR['DIG'].compressed(), tau_V_neb__cgR['COMP'].compressed(), tau_V_neb__cgR['SF'].compressed()]
    _, text_list = plot_histo_ax(ax1, xDs, stats_txt=False, return_text_list=True, y_v_space=0.06, y_h_space=0.25, first=False, c=colors_lines_DIG_COMP_SF, kwargs_histo=dict(bins=20, histtype='step', color=colors_DIG_COMP_SF, normed=False, range=range, lw=3))
    pos_y = 0.9
    for txt in text_list[0]:
        plot_text_ax(ax1, txt, **dict(pos_x=0.98, pos_y=pos_y, fs=14, va='top', ha='right', c=colors_lines_DIG_COMP_SF[0]))
        pos_y -= 0.06
    pos_y = 0.9
    for txt in text_list[1]:
        plot_text_ax(ax1, txt, **dict(pos_x=0.48, pos_y=pos_y, fs=14, va='top', ha='right', c=colors_lines_DIG_COMP_SF[1]))
        pos_y -= 0.06
    pos_y = 0.9
    for txt in text_list[2]:
        plot_text_ax(ax1, txt, **dict(first=True, pos_x=0.30, pos_y=pos_y, fs=14, va='top', ha='right', c=colors_lines_DIG_COMP_SF[2]))
        pos_y -= 0.06
    ax1.set_xlabel(r'$\tau_V^{neb}$')
    ax1.xaxis.set_minor_locator(minorLocator)
    ax1_top = ax1.twiny()
    mn, mx = ax1.get_xlim()
    unit_converter = lambda x: np.log10(2.86 * np.exp(x * 0.34652))
    ax1_top.set_xlim(unit_converter(mn), unit_converter(mx))
    ax1_top.set_xlabel(r'$\log\ H\alpha/H\beta$')
    plot_text_ax(ax1, 'a)', 0.02, 0.98, 16, 'top', 'left', 'k')
    # AXIS 2
    x = delta_tau__gR
    range = DtauVnorm_range
    plot_histo_ax(ax2, x.compressed(), y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, bins=20, range=range))
    ax2.set_xlabel(r'$\Delta \tau\ =\ \tau_V^{SF}\ -\ \tau_V^{DIG}}$')
    ax2.xaxis.set_minor_locator(minorLocator)
    plot_text_ax(ax2, 'b)', 0.02, 0.98, 16, 'top', 'left', 'k')
    # AXIS 3
    y = delta_tau__gR
    x = np.ones(y.shape) * R_bin_center__r
    xm, ym = ma_mask_xyz(x, y)
    yMean, yPrc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
    ax3.scatter(x, y, c='gray', marker='o', s=10, edgecolor='none')
    ax3.plot(bin_center, yPrc[0], 'k--', lw=1)
    ax3.plot(bin_center, yPrc[1], 'k--', lw=1)
    ax3.plot(bin_center, yPrc[2], 'k-', lw=3)
    ax3.plot(bin_center, yMean, 'k*', lw=3)
    ax3.plot(bin_center, yPrc[3], 'k--', lw=1)
    ax3.plot(bin_center, yPrc[4], 'k--', lw=1)
    ax3.set_ylabel(r'$\Delta \tau(gal,R)\ =\ \tau_V^{SF}\ -\ \tau_V^{DIG}}$')
    ax3.set_xlabel(r'R [HLR]')
    ax3.grid()
    f.tight_layout(h_pad=0.05)
    f.savefig('fig7_stacks.png', dpi=dpi_choice, transparent=transp_choice)


def fig_HbHa_vs_SBHa_WHa(ALL, sel, gals):
    from pytu.plots import plot_spearmanr_ax
    sel_gals_sample__gz = sel['gals_sample__z']
    cmap = cmap_discrete(colors_DIG_COMP_SF)

    if (sel_gals_sample__gz).any():
        W6563__gz = ALL.W6563__z
        SB6563__gz = ALL.SB6563__z
        f__lgz = {'%s' % l: np.ma.masked_array(getattr(ALL, 'f%s__z' % l), mask=~sel_gals_sample__gz) for l in lines}
        dist__gz = ALL.zoneDistance_HLR

        # WHa DIG-COMP-SF decomposition
        sel_WHa_DIG__gz = np.bitwise_and(sel['WHa']['DIG']['z'], sel_gals_sample__gz)
        sel_WHa_COMP__gz = np.bitwise_and(sel['WHa']['COMP']['z'], sel_gals_sample__gz)
        sel_WHa_SF__gz = np.bitwise_and(sel['WHa']['SF']['z'], sel_gals_sample__gz)
        classif = np.ma.masked_all(W6563__gz.shape)
        classif[sel_WHa_DIG__gz] = 1
        classif[sel_WHa_COMP__gz] = 2
        classif[sel_WHa_SF__gz] = 3

        logWHa = np.ma.log10(W6563__gz)
        logSBHa = np.ma.log10(SB6563__gz)
        R = dist__gz

        N_cols = 2
        N_rows = 2
        f, axArr = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 5))
        (ax1, ax2), (ax3, ax4) = axArr

        x = logWHa
        y = np.ma.log10(f__lgz['4861']/f__lgz['6563'])
        z = R
        xm, ym, zm = ma_mask_xyz(x=x, y=y, z=z, mask=~sel_gals_sample__gz)
        sc = ax1.scatter(xm, ym, c=zm, cmap=cmap_R, vmax=2, s=2, marker='o', edgecolor='none')
        # cbaxes = f.add_axes([0.6, 0.17, 0.12, 0.02])
        # cb = plt.colorbar(sc, cax=cbaxes, ticks=[0, 1, 2, 3], orientation='horizontal')
        # cb.set_label(r'R [HLR]', fontsize=14)
        xbins = np.linspace(logWHa_range[0], logWHa_range[1], 20)
        yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), xbins)
        yMedian = prc[2]
        y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
        ax1.plot(bin_center, yMedian, 'k-', lw=2)
        for y_prc in y_12sigma:
            ax1.plot(bin_center, y_prc, 'k--', lw=2)
        ax1.set_xlabel(r'$\log$ W${}_{H\alpha}$ [$\AA$]')
        ax1.set_ylabel(r'$\log$ $H\beta / H\alpha$')
        ax1.set_ylim(-1, 0)
        ax1.set_xlim(logWHa_range)
        ax1.yaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax1.xaxis.set_major_locator(MultipleLocator(1))
        ax1.xaxis.set_minor_locator(MultipleLocator(0.2))
        # ax1.grid(which='both')
        plot_spearmanr_ax(ax=ax1, x=xm.compressed(), y=ym.compressed(), pos_x=0.02, pos_y=0.98, fontsize=16)

        x = logSBHa
        y = np.ma.log10(f__lgz['4861']/f__lgz['6563'])
        z = R
        xm, ym, zm = ma_mask_xyz(x=x, y=y, z=z, mask=~sel_gals_sample__gz)
        ax2.set_xlabel(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]')
        ax2.set_ylabel(r'$\log$ $H\beta / H\alpha$')
        sc = ax2.scatter(xm, ym, c=zm, cmap=cmap_R, s=2, vmax=2, marker='o', edgecolor='none')
        cbaxes = f.add_axes([0.83, 0.90, 0.12, 0.03])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[0, 1, 2], orientation='horizontal')
        cb.set_label(r'R [HLR]', fontsize=14)
        xbins = np.linspace(logSBHa_range[0], logSBHa_range[1], 20)
        yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), xbins)
        yMedian = prc[2]
        y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
        ax2.plot(bin_center, yMedian, 'k-', lw=2)
        for y_prc in y_12sigma:
            ax2.plot(bin_center, y_prc, 'k--', lw=2)
        ax2.set_ylim(-1, 0)
        ax2.set_xlim(logSBHa_range)
        ax2.yaxis.set_major_locator(MultipleLocator(0.5))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax2.xaxis.set_major_locator(MultipleLocator(1))
        ax2.xaxis.set_minor_locator(MultipleLocator(0.2))
        # ax2.grid(which='both')
        plot_spearmanr_ax(ax=ax2, x=xm.compressed(), y=ym.compressed(), pos_x=0.02, pos_y=0.98, fontsize=16)

        x = R
        y = np.ma.log10(f__lgz['4861']/f__lgz['6563'])
        z = logWHa
        xm, ym, zm = ma_mask_xyz(x=x, y=y, z=z, mask=~sel_gals_sample__gz)
        ax3.set_xlabel(r'R [HLR]')
        ax3.set_ylabel(r'$\log$ $H\beta / H\alpha$')
        sc = ax3.scatter(xm, ym, c=zm, cmap=cmap_R, s=2, vmin=logWHa_range[0], vmax=logWHa_range[1], marker='o', edgecolor='none')
        cbaxes = f.add_axes([0.33, 0.15, 0.12, 0.03])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[0, 1.25, 2.5], orientation='horizontal')
        cb.set_label(r'$\log$ W${}_{H\alpha}$ [$\AA$]', fontsize=14)
        yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        yMedian = prc[2]
        y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
        ax3.plot(bin_center, yMedian, 'k-', lw=2)
        for y_prc in y_12sigma:
            ax3.plot(bin_center, y_prc, 'k--', lw=2)
        ax3.set_ylim(-1, 0)
        ax3.set_xlim(0, 3)
        ax3.yaxis.set_major_locator(MultipleLocator(0.5))
        ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax3.xaxis.set_major_locator(MultipleLocator(1))
        ax3.xaxis.set_minor_locator(MultipleLocator(0.2))
        # ax3.grid(which='both')
        plot_spearmanr_ax(ax=ax3, x=xm.compressed(), y=ym.compressed(), pos_x=0.02, pos_y=0.98, fontsize=16)

        x = R
        y = np.ma.log10(f__lgz['4861']/f__lgz['6563'])
        z = logSBHa
        xm, ym, zm = ma_mask_xyz(x=x, y=y, z=z, mask=~sel_gals_sample__gz)
        ax4.set_xlabel(r'R [HLR]')
        ax4.set_ylabel(r'$\log$ $H\beta / H\alpha$')
        sc = ax4.scatter(xm, ym, c=zm, cmap=cmap_R, s=2, marker='o', edgecolor='none', vmin=logSBHa_range[0], vmax=logSBHa_range[1])
        cbaxes = f.add_axes([0.82, 0.15, 0.12, 0.03])
        cb = plt.colorbar(sc, cax=cbaxes, ticks=[4, 5, 6, 7], orientation='horizontal')
        cb.set_label(r'$\log\ \Sigma_{H\alpha}$ [L${}_\odot/$kpc${}^2$]', fontsize=14)
        yMean, prc, bin_center, npts = stats_med12sigma(xm.compressed(), ym.compressed(), R_bin__r)
        yMedian = prc[2]
        y_12sigma = [prc[0], prc[1], prc[3], prc[4]]
        ax4.plot(bin_center, yMedian, 'k-', lw=2)
        for y_prc in y_12sigma:
            ax4.plot(bin_center, y_prc, 'k--', lw=2)
        ax4.set_ylim(-1, 0)
        ax4.set_xlim(0, 3)
        ax4.yaxis.set_major_locator(MultipleLocator(0.5))
        ax4.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax4.xaxis.set_major_locator(MultipleLocator(1))
        ax4.xaxis.set_minor_locator(MultipleLocator(0.2))
        # ax4.grid(which='both')
        plot_spearmanr_ax(ax=ax4, x=xm.compressed(), y=ym.compressed(), pos_x=0.02, pos_y=0.98, fontsize=16)


        f.tight_layout()
        f.savefig('fig_HbHa_vs_SBHa_WHa.png', dpi=dpi_choice, transparent=transp_choice)
        plt.close(f)


if __name__ == '__main__':
    main(sys.argv)


# '''
# The different dust attenuation of star-forming regions and the diffuse ionized gas
# in CALIFA galaxies
#
# We use spatially resolved spectroscopy from the CALIFA survey to separate the
# emission from star-forming (SF) regions from that of the diffuse ionized gas (DIG)
# in an attempt to map variations in dust content among these two major contributors
# to emission lines in galaxies. The diagnosis of SF or DIG conditions is carried
# out exclusively in terms of the equivalent width of H\alpha (WH\alpha). Although
# inspired by theoretical considerations, our classification criterion is anchored
# on the strong bimodality of the WH\alpha distribution observed both within and
# among galaxies. Besides producing results in agreement with a classical
# diagnostic-diagram based analysis, this empirically calibrated method circumvents
# inconsistencies in DIG/SF separation criteria. A set of 31 face on spirals was
# culled from CALIFA and dissected into SF, composite, and DIG-dominated spaxels in
# order to study differences in dust attenuation among these components of the
# interstellar medium. We find that the H\alpha/H\beta ratio is systematically
# larger in SF regions than in the DIG, with composite regions in between. Quantifying
# dust attenuation by a foreground screen model we obtain V -band optical depth
# differences of typically \delta\tau_V^{neb} \sim 0.5 between SF and DIG spaxels.
# We further compare the nebular values of \tau_V with those derived from fits to the
# stellar continuum with the spectral synthesis code starlight. The stellar continuum
# and the DIG suffer similar attenuations, with an average \tau_V^{DIG} - \tau_V^\star
# difference of difference of just 0.07.
#
# The different dust attenuation of star-forming regions and the diffuse ionized gas
# in galaxies
#
# We use spatially resolved spectroscopy from the CALIFA survey to separate the
# emission from star-forming (SF) regions from that of the diffuse ionized gas (DIG)
# in an attempt to map variations in dust content among these two major contributors
# to emission lines in galaxies. A set of 31 face on spirals was culled from CALIFA
# and dissected into SF, composite, and DIG-dominated spaxels in order to study
# differences in dust attenuation among these components of the interstellar medium.
# We find that the H\alpha/H\beta ratio is systematically larger in SF regions than
# in the DIG, with composite regions in between. Quantifying dust attenuation by a
# foreground screen model we obtain V-band optical depth differences of typically
# \delta\tau_V^{neb} \sim 0.5 between SF and DIG spaxels. We further compare the
# nebular values of \tau_V with those derived from fits to the stellar continuum
# with the spectral synthesis code starlight. The stellar continuum and the DIG suffer
# similar attenuations, with an average \tau_V^{DIG} - \tau_V^\star difference of
# just 0.07.
# '''
