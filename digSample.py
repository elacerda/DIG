import os
import sys
import numpy as np
import astropy.table
from pytu.functions import debug_var
from CALIFAUtils.scripts import calc_xY
from pycasso.util import getGenFracRadius
from CALIFAUtils.objects import stack_gals
from CALIFAUtils.scripts import loop_cubes
from pystarlight.util.constants import L_sun
from CALIFAUtils.scripts import try_q055_instead_q054, calc_SFR, my_morf, get_morfologia, prop_Y


file_QH = '%s/LOCAL/data/BASE.gstf6e.square.QH_BC03' % os.getenv('HOME')
# tY = 2e7
tSF__T = np.array([0.032 , 0.3 , 1.5, 14.2]) * 1.e9
N_T = len(tSF__T)
config = -2
EL = None
elliptical = True
# kw_cube = dict(config=config, elliptical=elliptical)
debug = True
kw_cube = dict(debug=debug, EL=EL, config=config, elliptical=elliptical)


def get_zone_mean(prop, zones):
    N_zone = zones.max() + 1
    bins = np.arange(N_zone + 1)
    zones_flat = zones.ravel()
    if isinstance(prop, np.ma.MaskedArray):
        prop = prop.filled(0.0)
    prop_flat = prop.ravel()
    prop_mean,_ = np.histogram(zones_flat, weights=prop_flat, bins=bins)
    zone_area,_ = np.histogram(zones_flat, bins=bins)
    prop_mean /= zone_area
    return prop_mean


def calc_LHa_expected_HIG(K, HIG_ages_interval=None):
    if HIG_ages_interval is None:
        HIG_ages_interval = [9.99e7, 1.00e20]
    tab_QH = astropy.table.Table.read(file_QH, format = 'ascii.fixed_width_two_line')
    shape__tZ = (K.N_age, K.N_met)
    log_QH_base__tZ = np.copy(tab_QH['log_QH'].reshape(shape__tZ))
    tab_ageBase__tZ = tab_QH['age_base'].reshape(shape__tZ)
    tab_metBase__tZ = tab_QH['Z_base'].reshape(shape__tZ)
    check_t = np.all(np.isclose(K.ageBase, 10**tab_ageBase__tZ[..., 0], rtol=1.e-5, atol=1e-5))
    check_Z = np.all(np.isclose(K.metBase, tab_metBase__tZ[0, ...], rtol=1.e-5, atol=1e-5))
    if (not check_t) | (not check_Z):
        sys.exit('BASE CHECK: %s : please check your square base.' % califaID)

    # Calc the total percentage of popini (~100%) in each zone
    norm_popini__z = K.popmu_ini.sum(axis=(0, 1))
    popmu_ini_frac__tZz = K.popmu_ini/norm_popini__z

    # Calc the number of ionizing photons per stellar mass
    log_qH__z = np.ma.log10(np.sum(popmu_ini_frac__tZz * 10**log_QH_base__tZ[..., np.newaxis], axis=(0, 1)))

    flag_HIG__t = (K.ageBase >= HIG_ages_interval[0]) & (K.ageBase < HIG_ages_interval[1])
    log_qH_HIG__z = np.ma.log10(np.sum(popmu_ini_frac__tZz[flag_HIG__t] * 10**log_QH_base__tZ[flag_HIG__t, :, np.newaxis], axis=(0, 1)))

    # Calc Q by multiplying q by the mass
    log_QH_HIG__z = log_qH_HIG__z + np.log10(K.Mini__z)

    # Transform in Ha
    clight = 2.99792458  # * 10**18 angstrom/s
    hplanck = 6.6260755   # * 10**(-27) erg s
    lsun = 3.826       # * 10**33 erg/s
    _k_q = np.log10(lsun / (clight * hplanck)) + 33 + 27 - 18
    _k0 = 1. / (2.226 * 6562.80)

    LHa_expected_HIG__z = _k0 * 10**(log_QH_HIG__z - _k_q)
    log_LHa_expected_HIG__z = np.log10(LHa_expected_HIG__z)
    # Transform into xy
    log_LHa_expected_HIG__yx = np.ma.log10(K.zoneToYX(LHa_expected_HIG__z / K.zoneArea_pix, extensive=False))

    return log_LHa_expected_HIG__z, log_LHa_expected_HIG__yx


def gather_needed_data(g, mto, mt, dump=True, output_filename='ALL_HaHb.pkl'):
    lines = ['3727', '4363', '4861', '4959', '5007', '6300', '6563', '6583', '6717', '6731']
    keys2d = [
        ('x_Y__Tz', N_T), ('SFR__Tz', N_T), ('SFRSD__Tz', N_T), ('integrated_x_Y__T', N_T),
    ]
    keys2d_masked = [
        ('x_Y__Tyx', N_T), ('SFRSD__Tyx', N_T)
    ]
    keys1d = [
        'Mini__z', 'Mcor__z', 'McorSD__z', 'tau_V__z',
        'at_flux__z', 'at_mass__z', 'alogZ_flux__z', 'alogZ_mass__z', 'fobs_norm__z',
        'zones_map', 'califaID__g', 'califaID__z', 'califaID__yx', 'pixels_map',
        'x0', 'y0', 'N_x', 'N_y', 'N_zone', 'ml_ba', 'ba', 'pa',
        'mt', 'mto', 'HLR_pix', 'HLR_pc',
        'galDistance_Mpc', 'zoneArea_pc2', 'zoneArea_pix', 'redshift',
        'pixelDistance__yx', 'pixelDistance_HLR__yx',
        'zoneDistance_pc', 'zoneDistance_HLR', 'zoneDistance_pix',
        'lines', 'CI', 'CI_9050', 'Mtot',
        'qZones__yx',
        'integrated_tau_V', 'integrated_tau_V_neb', 'integrated_etau_V_neb',
        'integrated_alogZ_mass', 'integrated_at_flux', 'integrated_Dn4000',
        'integrated_vd', 'integrated_v0',
    ]
    keys1d_masked = [
        'log_L6563_expected_HIG__z', 'log_L6563_expected_HIG__yx',
        'at_flux__yx', 'at_mass__yx', 'alogZ_flux__yx', 'alogZ_mass__yx', 'fobs_norm__yx',
        'tau_V__yx', 'tau_V_neb__z', 'etau_V_neb__z', 'tau_V_neb__yx',
        'tau_V_neb_zeros__z', 'tau_V_neb_zeros__yx',
        'qSn__z', 'qSn__yx',
    ]
    for l in lines:
        for prop in ['c', 'W', 'f', 'ef', 'SB', 'L']:
            prop_name = 'integrated_%s%s' % (prop, l)
            # print prop_name
            keys1d.append(prop_name)
            for bin in ['z', 'yx']:
                prop_name = '%s%s__%s' % (prop, l, bin)
                # print prop_name
                keys1d_masked.append(prop_name)
    # print keys1d
    # print keys1d_masked
    # sys.exit(1)
    ##############################################################

    ALL = stack_gals(keys1d=keys1d, keys1d_masked=keys1d_masked, keys2d=keys2d, keys2d_masked=keys2d_masked)
    # for i_gal, califaID in enumerate(g):
        # from pycasso import fitsQ3DataCube
        # K = fitsQ3DataCube('/Users/lacerda/califa/legacy/q057/superfits/px1Bgstf6e/%s_synthesis_eBR_px1_q057.d22a512.ps03.k1.mE.CCM.Bgstf6e.fits' % califaID)
        # # elliptical-geometry
        # K.setGeometry(*K.getEllipseParams())
    for i_gal, K in loop_cubes(g, **kw_cube):
        califaID = g[i_gal]

        if K is None:
            print 'califaID:', califaID, ' trying another qVersion...'
            K = try_q055_instead_q054(califaID, **kw_cube)
            # if (K is None) or (K.EL is None):
            if K is None:
                # print 'califaID:%s missing fits files...' % califaID
                print 'SUPERFITS: %s : missing fits file.' % califaID
                continue
        # EMLDataCube_file = '/Users/lacerda/califa/legacy/q057/EML/px1Bgstf6e/%s_synthesis_eBR_px1_q057.d22a512.ps03.k1.mE.CCM.Bgstf6e.EML.MC100.fits' % califaID
        EMLDataCube_file = '%s/RGB/Bgstf6e/v04/%s_synthesis_eBR_v20_q054.d22a512.ps03.k1.mE.CCM.Bgstf6e.EML.MC100.fits'% (os.getenv('HOME'), califaID)
        if os.path.isfile(EMLDataCube_file):
            K.loadEmLinesDataCube(EMLDataCube_file)
        else:
            print 'EML:', califaID, ' trying another qVersion...'
            # continue
            EMLDataCube_file = EMLDataCube_file.replace('q054', 'q055')
            if os.path.isfile(EMLDataCube_file):
                K.loadEmLinesDataCube(EMLDataCube_file)
            else:
                print 'EML: %s : missing fits file.' % califaID
                continue
        log_LHa_expected_HIG__z, log_LHa_expected_HIG__yx = calc_LHa_expected_HIG(K, [9.99e7, 1.00e20])
        ALL.append1d_masked('log_L6563_expected_HIG__z', log_LHa_expected_HIG__z, np.ma.getmaskarray(log_LHa_expected_HIG__z))
        ALL.append1d_masked('log_L6563_expected_HIG__yx', np.ravel(log_LHa_expected_HIG__yx), np.ravel(np.ma.getmaskarray(log_LHa_expected_HIG__yx)))
        ageBase = K.ageBase
        metBase = K.metBase
        # ALL.append1d_masked(k='qZones__yx', val=np.ravel(K.qZones), mask_val=K.qMask)
        ALL.append1d('qZones__yx', np.ravel(K.qZones))
        ALL.append1d_masked(k='qSn__yx', val=np.ravel(K.qZonesSnOrig), mask_val=np.ravel(np.ma.getmaskarray(K.qZonesSnOrig)))
        qSn__z = get_zone_mean(K.qSn, K.qZones)
        ALL.append1d_masked(k='qSn__z', val=qSn__z, mask_val=np.zeros(K.N_zone, dtype='bool'))
        zones_map__z = np.array(list(range(K.N_zone)), dtype='int')
        ALL.append1d('zones_map', zones_map__z)
        califaID__z = np.array([K.califaID for i in range(K.N_zone)], dtype='|S5')
        ALL.append1d('califaID__z', califaID__z)
        califaID__yx = np.array([K.califaID for i in range(K.N_y * K.N_x)], dtype='|S5')
        ALL.append1d('califaID__yx', califaID__yx)
        pixels_map__yx = np.array(list(range(K.N_y * K.N_x)), dtype='int')
        # mto = get_morfologia(K.califaID)[0]
        ALL.append1d('redshift', K.redshift)
        ALL.append1d('califaID__g', califaID)
        ALL.append1d('mto', mto[i_gal])
        ALL.append1d('mt', mt[i_gal])
        ALL.append1d('pixels_map', pixels_map__yx)
        ALL.append1d('x0', K.x0)
        ALL.append1d('y0', K.y0)
        ALL.append1d('N_x', K.N_x)
        ALL.append1d('N_y', K.N_y)
        ALL.append1d('N_zone', K.N_zone)
        if 'ba' in K.masterListData.keys():
            ALL.append1d('ml_ba', eval(K.masterListData['ba']))
        elif 'gc_ba' in K.masterListData.keys():
            if K.masterListData['gc_ba'] is None:
                ml_ba = -1
            else:
                ml_ba = eval(K.masterListData['gc_ba'])
            ALL.append1d('ml_ba', ml_ba)
        ALL.append1d('ba', K.ba)
        ALL.append1d('pa', K.pa)
        ALL.append1d('HLR_pix', K.HLR_pix)
        ALL.append1d('HLR_pc', K.HLR_pc)
        ALL.append1d('galDistance_Mpc', K.distance_Mpc)
        ALL.append1d('zoneArea_pc2', K.zoneArea_pc2)
        ALL.append1d('zoneArea_pix', K.zoneArea_pix)
        ALL.append1d('pixelDistance__yx', np.ravel(K.pixelDistance__yx))
        ALL.append1d('pixelDistance_HLR__yx', np.ravel(K.pixelDistance__yx / K.HLR_pix))
        ALL.append1d('zoneDistance_pc', K.zoneDistance_pc)
        ALL.append1d('zoneDistance_HLR', K.zoneDistance_HLR)
        ALL.append1d('zoneDistance_pix', K.zoneDistance_pix)
        ALL.append1d('lines', lines)
        ALL.append1d('Mini__z', K.Mini__z)
        ALL.append1d('fobs_norm__z', K.fobs_norm)
        fobs_norm__yx = K.zoneToYX(K.fobs_norm/K.zoneArea_pix, extensive=False)
        ALL.append1d_masked('fobs_norm__yx', np.ravel(fobs_norm__yx), np.ravel(np.ma.getmaskarray(fobs_norm__yx)))
        ALL.append1d('Mtot', K.Mcor_tot.sum())
        ALL.append1d('Mcor__z', K.Mcor__z)
        ALL.append1d('McorSD__z', K.Mcor__z / K.zoneArea_pc2)
        ALL.append1d('at_flux__z', K.at_flux__z)
        ALL.append1d_masked('at_flux__yx', np.ravel(K.at_flux__yx), np.ravel(np.ma.getmaskarray(K.at_flux__yx)))
        ALL.append1d('at_mass__z', K.at_mass__z)
        ALL.append1d_masked('at_mass__yx', np.ravel(K.at_mass__yx), np.ravel(np.ma.getmaskarray(K.at_mass__yx)))
        ALL.append1d('tau_V__z', K.tau_V__z)
        ALL.append1d('integrated_tau_V', K.integrated_tau_V)
        ALL.append1d('alogZ_flux__z', K.alogZ_flux__z)
        ALL.append1d_masked('alogZ_flux__yx', np.ravel(K.alogZ_flux__yx), np.ravel(np.ma.getmaskarray(K.alogZ_flux__yx)))
        ALL.append1d('alogZ_mass__z', K.alogZ_mass__z)
        ALL.append1d_masked('alogZ_mass__yx', np.ravel(K.alogZ_mass__yx), np.ravel(np.ma.getmaskarray(K.alogZ_mass__yx)))
        ALL.append1d('integrated_alogZ_mass', K.integrated_alogZ_mass)
        ALL.append1d('integrated_at_flux', K.integrated_at_flux)
        ALL.append1d('integrated_Dn4000', K.integrated_Dn4000)
        ALL.append1d('integrated_vd', K.integrated_keywords['V_D'])
        ALL.append1d('integrated_v0', K.integrated_keywords['V_0'])
        ########################
        # tSF things
        for iT, tSF in enumerate(tSF__T):
            x_Y__z, integrated_x_Y = calc_xY(K, tSF)
            x_Y__yx, _ = calc_xY(tY=tSF, ageBase=K.ageBase, popx=K.popx__tZyx)
            ALL.append2d(k='x_Y__Tz', i=iT, val=x_Y__z)
            ALL.append2d_masked(k='x_Y__Tyx', i=iT, val=np.ravel(x_Y__yx), mask_val=np.ravel(np.ma.getmaskarray(x_Y__yx)))
            ALL.append2d(k='integrated_x_Y__T', i=iT, val=integrated_x_Y)
            SFR__z, SFRSD__z = calc_SFR(K, tSF)
            SFRSD__yx = prop_Y(K.MiniSD__tZyx, tSF, K.ageBase)/tSF
            ALL.append2d(k='SFR__Tz', i=iT, val=SFR__z)
            ALL.append2d(k='SFRSD__Tz', i=iT, val=SFRSD__z)
            ALL.append2d_masked(k='SFRSD__Tyx', i=iT, val=np.ravel(SFRSD__yx), mask_val=np.ravel(np.ma.getmaskarray(SFRSD__yx)))
        ########################
        '''
        CI: Calc. usando a equacao que ta no paper do Conselice
        - http://iopscience.iop.org/article/10.1086/375001/pdf, pagina 7 -
        e r80 e r20 calculando usando o lambda de normalizacao do CALIFA, ou seja, da mesma forma que
        o HLR (r50) e calculado mas utilizando 0.8 e 0.2 como fracao ao inves de 0.5 (half).
        '''
        r90 = getGenFracRadius(K.qSignal[K.qMask], K.pixelDistance__yx[K.qMask], None, frac=0.9)
        r80 = getGenFracRadius(K.qSignal[K.qMask], K.pixelDistance__yx[K.qMask], None, frac=0.8)
        r50 = getGenFracRadius(K.qSignal[K.qMask], K.pixelDistance__yx[K.qMask], None, frac=0.5)
        r20 = getGenFracRadius(K.qSignal[K.qMask], K.pixelDistance__yx[K.qMask], None, frac=0.2)
        CI = 5. * np.log10(r80/r20)
        CI_9050 = 5. * np.log10(r90/r50)
        ALL.append1d('CI', CI)
        ALL.append1d('CI_9050', CI_9050)
        debug_var(debug, CALIFAID=califaID, CI=CI, CI_9050=CI_9050)
        tau_V__yx = K.A_V__yx / (2.5 * np.log10(np.exp(1.)))
        ALL.append1d_masked(k='tau_V__yx', val=np.ravel(tau_V__yx), mask_val=np.ravel(np.ma.getmaskarray(tau_V__yx)))
        # EML
        ALL.append1d_masked(k='tau_V_neb__z', val=K.EL.tau_V_neb__z.data, mask_val=np.ma.getmaskarray(K.EL.tau_V_neb__z))
        tau_V_neb_zeros__z = np.where((K.EL.tau_V_neb__z < 0).filled(True), 0, K.EL.tau_V_neb__z)
        ALL.append1d_masked(k='tau_V_neb_zeros__z', val=tau_V_neb_zeros__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        etau_V_neb__z = K.EL.tau_V_neb_err__z
        ALL.append1d_masked(k='etau_V_neb__z', val=etau_V_neb__z, mask_val=np.ma.getmaskarray(K.EL.tau_V_neb_err__z))
        tau_V_neb__yx = K.zoneToYX(K.EL.tau_V_neb__z, extensive=False)
        ALL.append1d_masked(k='tau_V_neb__yx', val=np.ravel(tau_V_neb__yx), mask_val=np.ravel(np.ma.getmaskarray(tau_V_neb__yx)))
        tau_V_neb__yx = K.zoneToYX(tau_V_neb_zeros__z, extensive=False)
        ALL.append1d_masked(k='tau_V_neb_zeros__yx', val=np.ravel(tau_V_neb__yx), mask_val=np.ravel(np.ma.getmaskarray(tau_V_neb__yx)))
        ALL.append1d('integrated_tau_V_neb', K.EL.integrated_tau_V_neb)
        ALL.append1d('integrated_etau_V_neb', K.EL.integrated_tau_V_neb_err)
        ########################
        for l in lines:
            if l not in K.EL.lines:
                zeros__z = np.ma.masked_all((K.N_zone))
                zeros__yx = np.ma.masked_all((K.N_y * K.N_x))
                ALL.append1d_masked('f%s__z' % l, zeros__z, np.ma.getmaskarray(zeros__z))
                ALL.append1d_masked('ef%s__z' % l, zeros__z, np.ma.getmaskarray(zeros__z))
                ALL.append1d_masked('SB%s__z' % l, zeros__z, np.ma.getmaskarray(zeros__z))
                ALL.append1d_masked('L%s__z' % l, zeros__z, np.ma.getmaskarray(zeros__z))
                ALL.append1d_masked('W%s__z' % l, zeros__z, np.ma.getmaskarray(zeros__z))
                ALL.append1d_masked('c%s__z' % l, zeros__z, np.ma.getmaskarray(zeros__z))
                ALL.append1d_masked('f%s__yx' % l, zeros__yx, np.ma.getmaskarray(zeros__yx))
                ALL.append1d_masked('ef%s__yx' % l, zeros__yx, np.ma.getmaskarray(zeros__yx))
                ALL.append1d_masked('SB%s__yx' % l, zeros__yx, np.ma.getmaskarray(zeros__yx))
                ALL.append1d_masked('L%s__yx' % l, zeros__yx, np.ma.getmaskarray(zeros__yx))
                ALL.append1d_masked('W%s__yx' % l, zeros__yx, np.ma.getmaskarray(zeros__yx))
                ALL.append1d_masked('c%s__yx' % l, zeros__yx, np.ma.getmaskarray(zeros__yx))
                ALL.append1d('integrated_f%s' % l, 0.)
                ALL.append1d('integrated_ef%s' % l, 0.)
                ALL.append1d('integrated_L%s' % l, 0.)
                ALL.append1d('integrated_SB%s' % l, 0.)
                ALL.append1d('integrated_W%s' % l, 0.)
                ALL.append1d('integrated_c%s' % l, 0.)
            else:
                i = K.EL.lines.index(l)
                mask = np.bitwise_or(~np.isfinite(K.EL.flux[i]), np.less(K.EL.flux[i], 1e-40))
                fl_obs__z = np.ma.masked_array(K.EL.flux[i], mask=mask, copy=True)
                ALL.append1d_masked('f%s__z' % l, fl_obs__z, mask)
                efl_obs__z = np.ma.masked_array(K.EL.eflux[i], mask=mask, copy=True)
                ALL.append1d_masked('ef%s__z' % l, efl_obs__z, mask)
                Ll_obs__z = K.EL._F_to_L(fl_obs__z)/L_sun
                integrated_Ll_obs = K.EL._F_to_L(K.EL.integrated_flux[i])/L_sun
                ALL.append1d_masked('L%s__z' % l, Ll_obs__z, np.ma.getmaskarray(Ll_obs__z))
                SBl_obs__z = Ll_obs__z/(K.zoneArea_pc2 * 1e-6)
                integrated_SBl_obs = integrated_Ll_obs/(K.zoneArea_pc2.sum() * 1e-6)
                ALL.append1d_masked('SB%s__z' % l, SBl_obs__z, np.ma.getmaskarray(SBl_obs__z))
                W__z = K.EL.EW[i]
                integrated_W = K.EL.integrated_EW[i]
                ALL.append1d_masked('W%s__z' % l, W__z, np.ma.getmaskarray(W__z))
                c__z = K.EL.baseline[i]
                integrated_c = K.EL.integrated_baseline[i]
                ALL.append1d_masked('c%s__z' % l, c__z, np.ma.getmaskarray(c__z))
                fl_obs__yx = K.zoneToYX(fl_obs__z/K.zoneArea_pix, extensive=False)
                ALL.append1d_masked('f%s__yx' % l, np.ravel(fl_obs__yx), np.ravel(np.ma.getmaskarray(fl_obs__yx)))
                efl_obs__yx = K.zoneToYX(efl_obs__z/K.zoneArea_pix, extensive=False)
                ALL.append1d_masked('ef%s__yx' % l, np.ravel(efl_obs__yx), np.ravel(np.ma.getmaskarray(efl_obs__yx)))
                Ll_obs__yx = K.EL._F_to_L(fl_obs__yx)/L_sun
                ALL.append1d_masked('L%s__yx' % l, np.ravel(Ll_obs__yx), np.ravel(np.ma.getmaskarray(Ll_obs__yx)))
                SBl_obs__yx = K.zoneToYX(Ll_obs__z/(K.zoneArea_pc2 * 1e-6), extensive=False)
                ALL.append1d_masked('SB%s__yx' % l, np.ravel(SBl_obs__yx), np.ravel(np.ma.getmaskarray(SBl_obs__yx)))
                W__yx = K.zoneToYX(W__z, extensive=False)
                ALL.append1d_masked('W%s__yx' % l, np.ravel(W__yx), np.ravel(np.ma.getmaskarray(W__yx)))
                c__yx = K.zoneToYX(c__z/K.zoneArea_pix, extensive=False)
                ALL.append1d_masked('c%s__yx' % l, np.ravel(c__yx), np.ravel(np.ma.getmaskarray(c__yx)))
                ALL.append1d('integrated_f%s' % l, K.EL.integrated_flux[i])
                ALL.append1d('integrated_ef%s' % l, K.EL.integrated_eflux[i])
                ALL.append1d('integrated_L%s' % l, integrated_Ll_obs)
                ALL.append1d('integrated_SB%s' % l, integrated_SBl_obs)
                ALL.append1d('integrated_W%s' % l, integrated_W)
                ALL.append1d('integrated_c%s' % l, integrated_c)
        ########################
    ALL.stack()
    ALL.ageBase = ageBase
    ALL.metBase = metBase
    ALL.tSF__T = tSF__T
    ALL.N_T = N_T
    if dump:
        ALL.dump(output_filename)
    return ALL


if __name__ == '__main__':
    f = open(sys.argv[1], 'r')
    g = []
    mt = []
    mto = []
    for line in f.xreadlines():
        read_line = line.strip()
        if read_line[0] == '#':
            continue
        seg_line = read_line.split(':')
        g.append(seg_line[0])
        # mto.append(seg_line[1])
        # mt.append(seg_line[2])
        mto_tmp = get_morfologia(seg_line[0])[0]
        mto.append(mto_tmp)
        mt.append(my_morf(mto_tmp))
    f.close()
    try:
        output_filename = sys.argv[2]
    except IndexError:
        output_filename = 'debug'
    ALL = gather_needed_data(g, mto, mt, dump=True, output_filename=output_filename)
