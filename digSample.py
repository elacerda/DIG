import sys
import numpy as np
from CALIFAUtils.scripts import calc_xY
from CALIFAUtils.objects import stack_gals
from CALIFAUtils.scripts import loop_cubes
from pystarlight.util.constants import L_sun
from pycasso.util import getGenFracRadius
from CALIFAUtils.scripts import try_q055_instead_q054, calc_SFR, my_morf, get_morfologia


tY = 32e6
config = -2
EL = True
elliptical = True
kw_cube = dict(EL=EL, config=config, elliptical=elliptical)


def gather_needed_data(filename, dump=True, output_filename='ALL_HaHb.pkl'):
    lines = ['3727', '4363', '4861', '4959', '5007', '6300', '6563', '6583', '6717', '6731']
    keys1d = [
        'zones_map', 'califaID__z', 'califaID__yx', 'pixels_map',
        'x0', 'y0', 'N_x', 'N_y', 'N_zone', 'ml_ba', 'ba', 'pa',
        'mt', 'mto', 'HLR_pix', 'HLR_pc',
        'galDistance_Mpc', 'zoneArea_pc2', 'zoneArea_pix',
        'pixelDistance__yx', 'pixelDistance_HLR__yx',
        'zoneDistance_pc', 'zoneDistance_HLR', 'zoneDistance_pix',
        'lines', 'CI', 'Mtot',
        'integrated_x_Y',
        'integrated_tau_V', 'integrated_tau_V_neb', 'integrated_etau_V_neb',
        'integrated_W6563',
        'integrated_f3727', 'integrated_ef3727', 'integrated_SB3727', 'integrated_L3727',
        'integrated_f4363', 'integrated_ef4363', 'integrated_SB4363', 'integrated_L4363',
        'integrated_f4861', 'integrated_ef4861', 'integrated_SB4861', 'integrated_L4861',
        'integrated_f4959', 'integrated_ef4959', 'integrated_SB4959', 'integrated_L4959',
        'integrated_f5007', 'integrated_ef5007', 'integrated_SB5007', 'integrated_L5007',
        'integrated_f6300', 'integrated_ef6300', 'integrated_SB6300', 'integrated_L6300',
        'integrated_f6563', 'integrated_ef6563', 'integrated_SB6563', 'integrated_L6563',
        'integrated_f6583', 'integrated_ef6583', 'integrated_SB6583', 'integrated_L6583',
        'integrated_f6717', 'integrated_ef6717', 'integrated_SB6717', 'integrated_L6717',
        'integrated_f6731', 'integrated_ef6731', 'integrated_SB6731', 'integrated_L6731',
    ]
    keys1d_masked = [
        'Mini__z', 'SFR32__z', 'SFR100__z',
        'tau_V__z', 'tau_V__yx', 'x_Y__z', 'x_Y__yx',
        'tau_V_neb__z', 'etau_V_neb__z', 'tau_V_neb__yx',
        'tau_V_neb_zeros__z', 'tau_V_neb_zeros__yx',
        'W6563__z', 'W6563__yx',
        'f3727__z', 'ef3727__z', 'SB3727__z', 'L3727__z',
        'f4363__z', 'ef4363__z', 'SB4363__z', 'L4363__z',
        'f4861__z', 'ef4861__z', 'SB4861__z', 'L4861__z',
        'f4959__z', 'ef4959__z', 'SB4959__z', 'L4959__z',
        'f5007__z', 'ef5007__z', 'SB5007__z', 'L5007__z',
        'f6300__z', 'ef6300__z', 'SB6300__z', 'L6300__z',
        'f6563__z', 'ef6563__z', 'SB6563__z', 'L6563__z',
        'f6583__z', 'ef6583__z', 'SB6583__z', 'L6583__z',
        'f6717__z', 'ef6717__z', 'SB6717__z', 'L6717__z',
        'f6731__z', 'ef6731__z', 'SB6731__z', 'L6731__z',
        'f3727__yx', 'SB3727__yx', 'L3727__yx',
        'f4363__yx', 'SB4363__yx', 'L4363__yx',
        'f4861__yx', 'SB4861__yx', 'L4861__yx',
        'f4959__yx', 'SB4959__yx', 'L4959__yx',
        'f5007__yx', 'SB5007__yx', 'L5007__yx',
        'f6300__yx', 'SB6300__yx', 'L6300__yx',
        'f6563__yx', 'SB6563__yx', 'L6563__yx',
        'f6583__yx', 'SB6583__yx', 'L6583__yx',
        'f6717__yx', 'SB6717__yx', 'L6717__yx',
        'f6731__yx', 'SB6731__yx', 'L6731__yx',
    ]
    ALL = stack_gals(keys1d=keys1d, keys1d_masked=keys1d_masked)
    for i_gal, K in loop_cubes(g, **kw_cube):
        califaID = g[i_gal]
        if K is None:
            print 'califaID:', califaID, ' trying another qVersion...'
            K = try_q055_instead_q054(califaID, **kw_cube)
            if K is None or K.EL is None:
                print 'califaID:%s missing fits files...' % califaID
                continue
        zones_map__z = np.array(list(range(K.N_zone)), dtype='int')
        ALL.append1d('zones_map', zones_map__z)
        califaID__z = np.array([K.califaID for i in range(K.N_zone)], dtype='|S5')
        ALL.append1d('califaID__z', califaID__z)
        califaID__yx = np.array([K.califaID for i in range(K.N_y * K.N_x)], dtype='|S5')
        ALL.append1d('califaID__yx', califaID__yx)
        pixels_map__yx = np.array(list(range(K.N_y * K.N_x)), dtype='int')
        mto = get_morfologia(K.califaID)[0]
        ALL.append1d('mto', mto)
        ALL.append1d('mt', my_morf(mto))
        ALL.append1d('pixels_map', pixels_map__yx)
        ALL.append1d('x0', K.x0)
        ALL.append1d('y0', K.y0)
        ALL.append1d('N_x', K.N_x)
        ALL.append1d('N_y', K.N_y)
        ALL.append1d('N_zone', K.N_zone)
        ALL.append1d('ml_ba', eval(K.masterListData['ba']))
        ALL.append1d('ba', K.ba)
        ALL.append1d('pa', K.pa)
        ALL.append1d('HLR_pix', K.HLR_pix)
        ALL.append1d('HLR_pc', K.HLR_pc)
        ALL.append1d('galDistance_Mpc', K.distance_Mpc)
        ALL.append1d('zoneArea_pc2', K.zoneArea_pc2)
        ALL.append1d('zoneArea_pix', K.zoneArea_pix)
        ALL.append1d('pixelDistance__yx', np.ravel(K.pixelDistance__yx))
        ALL.append1d('pixelDistance_HLR__yx', np.ravel(K.pixelDistance__yx / K.HLR_pix))
        ALL.append1d('zoneDistance_pc', K.zoneDistance_HLR)
        ALL.append1d('zoneDistance_HLR', K.zoneDistance_HLR)
        ALL.append1d('zoneDistance_pix', K.zoneDistance_pix)
        ALL.append1d('lines', lines)
        ALL.append1d('Mtot', K.Mcor_tot.sum())
        r80 = getGenFracRadius(K.qSignal[K.qMask], K.pixelDistance__yx[K.qMask], None, frac=0.8)
        r20 = getGenFracRadius(K.qSignal[K.qMask], K.pixelDistance__yx[K.qMask], None, frac=0.2)
        CI = 5. * np.log10(r80/r20)
        ALL.append1d('CI', CI)
        tau_V__z = K.tau_V__z
        ALL.append1d_masked(k='tau_V__z', val=tau_V__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        tau_V__yx = K.A_V__yx / (2.5 * np.log10(np.exp(1.)))
        ALL.append1d_masked(k='tau_V__yx', val=np.ravel(tau_V__yx), mask_val=np.ravel(tau_V__yx.mask))
        ALL.append1d('integrated_tau_V', K.integrated_tau_V)
        x_Y__z, integrated_x_Y = calc_xY(K, tY)
        ALL.append1d_masked(k='x_Y__z', val=x_Y__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        x_Y__yx, _ = calc_xY(tY=tY, ageBase=K.ageBase, popx=K.popx__tZyx)
        ALL.append1d_masked(k='x_Y__yx', val=np.ravel(x_Y__yx), mask_val=np.zeros((K.N_y*K.N_x), dtype='bool'))
        ALL.append1d('integrated_x_Y', integrated_x_Y)
        ALL.append1d_masked(k='Mini__z', val=K.Mini__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        SFR32__z, SFRSD32__z = calc_SFR(K, 3.2e7)
        SFR100__z, SFRSD100__z = calc_SFR(K, 1e8)
        ALL.append1d_masked(k='SFR32__z', val=SFR32__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        ALL.append1d_masked(k='SFR100__z', val=SFR100__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        # EML
        ALL.append1d_masked(k='tau_V_neb__z', val=K.EL.tau_V_neb__z.data, mask_val=K.EL.tau_V_neb__z.mask)
        tau_V_neb_zeros__z = np.where((K.EL.tau_V_neb__z < 0).filled(True), 0, K.EL.tau_V_neb__z)
        ALL.append1d_masked(k='tau_V_neb_zeros__z', val=tau_V_neb_zeros__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        etau_V_neb__z = K.EL.tau_V_neb_err__z
        ALL.append1d_masked(k='etau_V_neb__z', val=etau_V_neb__z, mask_val=K.EL.tau_V_neb_err__z.mask)
        tau_V_neb__yx = K.zoneToYX(K.EL.tau_V_neb__z, extensive=False)
        ALL.append1d_masked(k='tau_V_neb__yx', val=np.ravel(tau_V_neb__yx), mask_val=np.ravel(tau_V_neb__yx.mask))
        tau_V_neb__yx = K.zoneToYX(tau_V_neb_zeros__z, extensive=False)
        ALL.append1d_masked(k='tau_V_neb_zeros__yx', val=np.ravel(tau_V_neb__yx), mask_val=np.ravel(tau_V_neb__yx.mask))
        ALL.append1d('integrated_tau_V_neb', K.EL.integrated_tau_V_neb)
        ALL.append1d('integrated_etau_V_neb', K.EL.integrated_tau_V_neb_err)
        for l in lines:
            if l not in K.EL.lines:
                zeros__z = np.ma.masked_all((K.N_zone))
                zeros__yx = np.ma.masked_all((K.N_y * K.N_x))
                ALL.append1d_masked('f%s__z' % l, zeros__z, zeros__z.mask)
                ALL.append1d_masked('ef%s__z' % l, zeros__z, zeros__z.mask)
                ALL.append1d_masked('SB%s__z' % l, zeros__yx, zeros__yx.mask)
                ALL.append1d_masked('L%s__z' % l, zeros__z, zeros__z.mask)
                ALL.append1d_masked('f%s__yx' % l, zeros__z, zeros__z.mask)
                ALL.append1d_masked('SB%s__yx' % l, zeros__yx, zeros__yx.mask)
                ALL.append1d_masked('L%s__yx' % l, zeros__yx, zeros__yx.mask)
                ALL.append1d('integrated_f%s' % l, 0.)
                ALL.append1d('integrated_ef%s' % l, 0.)
                ALL.append1d('integrated_L%s' % l, 0.)
                ALL.append1d('integrated_SB%s' % l, 0.)
                if l is lines[lines.index('6563')]:
                    W6563__z = K.EL.EW[i]
                    ALL.append1d_masked('W%s__z' % l, zeros__z, zeros__z.mask)
                    ALL.append1d_masked('W%s__yx' % l, zeros__yx, zeros__yx.mask)
                    ALL.append1d('integrated_W%s' % l, 0.)
                continue
            else:
                i = K.EL.lines.index(l)
                mask = np.bitwise_or(~np.isfinite(K.EL.flux[i]), np.less(K.EL.flux[i], 1e-40))
                fl_obs__z = np.ma.masked_array(K.EL.flux[i], mask=mask, copy=True)
                ALL.append1d_masked('f%s__z' % l, fl_obs__z, mask)
                efl_obs__z = np.ma.masked_array(K.EL.eflux[i], mask=mask, copy=True)
                ALL.append1d_masked('ef%s__z' % l, efl_obs__z, mask)
                Ll_obs__z = K.EL._F_to_L(fl_obs__z)/L_sun
                integrated_Ll_obs = K.EL._F_to_L(K.EL.integrated_flux[i])/L_sun
                ALL.append1d_masked('L%s__z' % l, Ll_obs__z, Ll_obs__z.mask)
                SBl_obs__z = Ll_obs__z/(K.zoneArea_pc2 * 1e-6)
                integrated_SBl_obs = integrated_Ll_obs/(K.zoneArea_pc2 * 1e-6)
                ALL.append1d_masked('SB%s__z' % l, SBl_obs__z, SBl_obs__z.mask)
                fl_obs__yx = K.zoneToYX(fl_obs__z/K.zoneArea_pix, extensive=False)
                ALL.append1d_masked('f%s__yx' % l, np.ravel(fl_obs__yx), np.ravel(fl_obs__yx.mask))
                Ll_obs__yx = K.EL._F_to_L(fl_obs__yx)/L_sun
                ALL.append1d_masked('L%s__yx' % l, np.ravel(Ll_obs__yx), np.ravel(Ll_obs__yx.mask))
                SBl_obs__yx = K.zoneToYX(Ll_obs__z/(K.zoneArea_pc2 * 1e-6), extensive=False)
                ALL.append1d_masked('SB%s__yx' % l, np.ravel(SBl_obs__yx), np.ravel(SBl_obs__yx.mask))
                ALL.append1d('integrated_f%s' % l, K.EL.integrated_flux[i])
                ALL.append1d('integrated_ef%s' % l, K.EL.integrated_eflux[i])
                ALL.append1d('integrated_L%s' % l, integrated_Ll_obs)
                ALL.append1d('integrated_SB%s' % l, integrated_SBl_obs)
                if l is lines[lines.index('6563')]:
                    W6563__z = K.EL.EW[i]
                    integrated_W6563 = K.EL.integrated_EW[i]
                    ALL.append1d_masked('W%s__z' % l, W6563__z, fl_obs__z.mask)
                    ALL.append1d_masked('W%s__yx' % l, np.ravel(K.zoneToYX(W6563__z, extensive=False)), np.ravel(SBl_obs__yx.mask))
                    ALL.append1d('integrated_W%s' % l, integrated_W6563)

    ALL.stack()
    if dump:
        ALL.dump(output_filename)
    return ALL


if __name__ == '__main__':
    f = open(sys.argv[1], 'r')
    g = []
    for line in f.xreadlines():
        read_line = line.strip()
        if read_line[0] == '#':
            continue
        g.append(read_line)
    f.close()
    try:
        ALL = gather_needed_data(g, dump=True, output_filename=sys.argv[2])
    except IndexError:
        ALL = gather_needed_data(g, dump=True)
