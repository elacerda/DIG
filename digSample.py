import sys
import numpy as np
from CALIFAUtils.scripts import calc_xY
from CALIFAUtils.objects import stack_gals
from CALIFAUtils.scripts import loop_cubes
from pystarlight.util.constants import L_sun
from CALIFAUtils.scripts import try_q055_instead_q054


tY = 32e6
config = -2
EL = True
elliptical = True
kw_cube = dict(EL=EL, config=config, elliptical=elliptical)


def gather_needed_data(filename, dump=True, output_filename='ALL_HaHb.pkl'):
    lines = ['3727', '4363', '4861', '4959', '5007', '6300', '6563', '6583', '6717', '6731']
    keys1d = [
        'zones_map', 'califaID__z', 'califaID__yx', 'pixels_map',
        'x0', 'y0', 'N_x', 'N_y', 'N_zone', 'ml_ba', 'ba', 'pa', 'HLR_pix', 'HLR_pc',
        'galDistance_Mpc', 'zoneArea_pc2', 'zoneArea_pix',
        'pixelDistance__yx', 'zoneDistance_pc', 'zoneDistance_HLR', 'zoneDistance_pix',
        'lines',
    ]
    keys1d_masked = [
        'tau_V__z', 'tau_V__yx', 'x_Y__z', 'x_Y__yx',
        'tau_V_neb__z', 'etau_V_neb__z', 'tau_V_neb__yx',
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
        ALL.append1d('pixels_map', pixels_map__yx)
        ALL.append1d('x0', K.x0)
        ALL.append1d('y0', K.y0)
        ALL.append1d('N_x', K.N_x)
        ALL.append1d('N_y', K.N_y)
        ALL.append1d('N_zone', K.N_zone)
        ALL.append1d('ml_ba', K.masterListData['ba'])
        ALL.append1d('ba', K.ba)
        ALL.append1d('pa', K.pa)
        ALL.append1d('HLR_pix', K.HLR_pix)
        ALL.append1d('HLR_pc', K.HLR_pc)
        ALL.append1d('galDistance_Mpc', K.distance_Mpc)
        ALL.append1d('zoneArea_pc2', K.zoneArea_pc2)
        ALL.append1d('zoneArea_pix', K.zoneArea_pix)
        ALL.append1d('pixelDistance__yx', np.ravel(K.pixelDistance__yx))
        ALL.append1d('zoneDistance_pc', K.zoneDistance_HLR)
        ALL.append1d('zoneDistance_HLR', K.zoneDistance_HLR)
        ALL.append1d('zoneDistance_pix', K.zoneDistance_pix)
        ALL.append1d('lines', lines)
        tau_V__z = K.tau_V__z
        ALL.append1d_masked(k='tau_V__z', val=tau_V__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        tau_V__yx = K.A_V__yx / (2.5 * np.log10(np.exp(1.)))
        ALL.append1d_masked(k='tau_V__yx', val=np.ravel(tau_V__yx), mask_val=np.ravel(tau_V__yx.mask))
        x_Y__z, _ = calc_xY(K, tY)
        ALL.append1d_masked(k='x_Y__z', val=x_Y__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        x_Y__yx, _ = calc_xY(tY=tY, ageBase=K.ageBase, popx=K.popx__tZyx)
        ALL.append1d_masked(k='x_Y__yx', val=np.ravel(x_Y__yx), mask_val=np.zeros((K.N_y*K.N_x), dtype='bool'))
        # EML
        tau_V_neb__z = np.where((K.EL.tau_V_neb__z < 0).filled(True), 0, K.EL.tau_V_neb__z)
        ALL.append1d_masked(k='tau_V_neb__z', val=tau_V_neb__z, mask_val=np.zeros((K.N_zone), dtype='bool'))
        etau_V_neb__z = K.EL.tau_V_neb_err__z
        ALL.append1d_masked(k='etau_V_neb__z', val=etau_V_neb__z, mask_val=K.EL.tau_V_neb_err__z.mask)
        tau_V_neb__yx = K.zoneToYX(tau_V_neb__z, extensive=False)
        ALL.append1d_masked(k='tau_V_neb__yx', val=np.ravel(tau_V_neb__yx), mask_val=np.ravel(tau_V_neb__yx.mask))
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
                if l is lines[lines.index('6563')]:
                    W6563__z = K.EL.EW[i]
                    ALL.append1d_masked('W%s__z' % l, zeros__z, zeros__z.mask)
                    ALL.append1d_masked('W%s__yx' % l, zeros__yx, zeros__yx.mask)
                continue
            else:
                i = K.EL.lines.index(l)
                mask = np.bitwise_or(~np.isfinite(K.EL.flux[i]), np.less(K.EL.flux[i], 1e-40))
                fl_obs__z = np.ma.masked_array(K.EL.flux[i], mask=mask, copy=True)
                ALL.append1d_masked('f%s__z' % l, fl_obs__z, mask)
                efl_obs__z = np.ma.masked_array(K.EL.eflux[i], mask=mask, copy=True)
                ALL.append1d_masked('ef%s__z' % l, efl_obs__z, mask)
                Ll_obs__z = K.EL._F_to_L(fl_obs__z)/L_sun
                ALL.append1d_masked('L%s__z' % l, Ll_obs__z, Ll_obs__z.mask)
                SBl_obs__z = Ll_obs__z/(K.zoneArea_pc2 * 1e-6)
                ALL.append1d_masked('SB%s__z' % l, SBl_obs__z, SBl_obs__z.mask)
                fl_obs__yx = K.zoneToYX(fl_obs__z/K.zoneArea_pix, extensive=False)
                ALL.append1d_masked('f%s__yx' % l, np.ravel(fl_obs__yx), np.ravel(fl_obs__yx.mask))
                Ll_obs__yx = K.EL._F_to_L(fl_obs__yx)/L_sun
                ALL.append1d_masked('L%s__yx' % l, np.ravel(Ll_obs__yx), np.ravel(Ll_obs__yx.mask))
                SBl_obs__yx = K.zoneToYX(Ll_obs__z/(K.zoneArea_pc2 * 1e-6), extensive=False)
                ALL.append1d_masked('SB%s__yx' % l, np.ravel(SBl_obs__yx), np.ravel(SBl_obs__yx.mask))
                if l is lines[lines.index('6563')]:
                    W6563__z = K.EL.EW[i]
                    ALL.append1d_masked('W%s__z' % l, W6563__z, fl_obs__z.mask)
                    ALL.append1d_masked('W%s__yx' % l, np.ravel(K.zoneToYX(W6563__z, extensive=False)), np.ravel(SBl_obs__yx.mask))
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
