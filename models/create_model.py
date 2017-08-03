import numpy as np
import matplotlib.pyplot as plt
import pyCloudy as pc


pc.config.cloudy_exe = 'cloudy.exe'
models_dir = '/Users/lacerda/dev/astro/dig/models/'


options = ('no molecules',
           'no level2 lines',
           'no fine opacities',
           'atom h-like levels small',
           'atom he-like levels small',
           'element limit off -7',
            )

Z_metal = np.arange(7, 9.5, 0.2) / 8.69

def make_mod(name, logU, Z_str):
    assert Z_str in ('0.001', '0.004', '0.008', '0.020', '0.040')
    Z = np.float(Z_str)
    NH = 100
    ff = 1.0
    abund = abund_Asplund_2009.copy()
    delta_O = np.log10(Z/0.020)
    for elem in abund:
        if elem != 'He':
            abund[elem] += delta_O
    c_input = pc.CloudyInput('{0}/{1}'.format(models_dir, name))
    c_input.set_star(SED='table star "ISB_{}.mod"'.format(Z_str.split('.')[1]),
                     SED_params=(1e6),
                     lumi_unit = 'ionization parameter', lumi_value = logU)
    c_input.set_cste_density(np.log10(NH), ff = ff)
    c_input.set_abund(ab_dict = abund)
    c_input.set_distance(dist=1., unit='kpc', linear=True)
    c_input.set_other(options)
    c_input.set_stop(('temperature off', 'pfrac 0.02'))
    c_input.set_emis_tab(['H  1  4861.36A', 'H  1  6562.85A',
                          'N  2  6583.45A', 'O  3  5006.84A'])
    c_input.print_input()
