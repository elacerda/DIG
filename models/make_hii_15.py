'''
Created on 9 mai 2014

@author: christophemorisset
'''
import numpy as np
import pyCloudy as pc
from pyCloudy.db import use3MdB


OVN_dic = {
    'host' : '132.248.1.102',
    'user_name' : 'OVN_admin2',
    'user_passwd' : 'getenv',
    'base_name' : 'OVN2',
    'pending_table' : '`pending`',
    'master_table' : '`tab`',
    'teion_table' : '`teion`',
    'abion_table' : '`abion`',
    'temis_table' : '`temis`',
    'lines_table' : '`lines`'
}

MdB = pc.MdB(OVN_dic)
models_dir = 'GridHII_15/'
from pyCloudy.db.MdB import MdB_subproc
name = 'HII_15'
alpha_B = 2.6e-13
options = ('no molecules',
           'no level2 lines',
           'no fine opacities',
           'COSMIC RAY BACKGROUND',
           )
NH = 1e2
ff = 1.0
Dopita_log12pOH = np.array([ 7.39,  7.5 ,  7.69,  7.99,  8.17,  8.39,  8.69,  8.8 ,  8.99, 9.17,  9.39])
Dopita_logNH = np.array([-6.61, -6.47, -6.23, -5.79, -5.51, -5.14, -4.6 , -4.4 , -4.04, -3.67, -3.17])
Dopita_logNO = Dopita_logNH - Dopita_log12pOH + 12
p = np.polyfit(Dopita_log12pOH, Dopita_logNO, 2)

tab_lU_mean = np.asarray([-1., -1.5, -2, -2.5, -3, -3.5, -4.0])
tab_O = np.arange(7, 9.5, 0.2) - 12
tab_NO = np.polyval(p, tab_O)  # np.asarray([-2, -1.5, -1, -0.5, 0.])
tab_age = np.asarray([1., 4.]) * 1e6
tab_fr = [3.00]

pc.log_.level = 2

def get_He(logO):
    Y = 0.2514 + 29.81 * 10**logO
    Z = (8.64 * (logO + 12) - 47.44) * 10**logO
    He = np.log10(Y / (4 * (1. - Z - Y)))
    return He

def test_inputs(all_tabs, d_law = 'CSTE', SED='sp_cha', Cversion = '13'):
    i = 1
    for lU_mean, ab_O, NO, age, fr in all_tabs:
        xt = 8.10
        x = 12 + ab_O
        a = 2.21 # log(G/D) solar
        if x > xt:
            y = a + 1.00 * (8.69 - x)
        else:
            y = 0.96 + 3.10 * (8.69 - x)
        solar_GoD = 10**a
        this_GoD = 10**y # y is log(G/D)
        this_relative_GoD =  this_GoD / solar_GoD#  we want the correction to apply to solar Dust abundance
        this_relative_DoG = 1.0 / this_relative_GoD
        print('{} {} {} {} {} {} {} '.format(i, lU_mean, ab_O, NO, age, fr, this_relative_DoG))
        i += 1

def make_inputs(all_tabs, d_law = 'CSTE', SED='sp_cha', Cversion = '13'):
    wP = use3MdB.writePending(MdB, OVN_dic)
    wP.set_ref(name)
    wP.set_user('Grazyna minions')
    if Cversion == '13':
        wP.set_C_version('13.03')
    elif Cversion == '10':
        wP.set_C_version('10.00')
    elif Cversion == '17':
        wP.set_C_version('17.00')
    wP.set_iterate(1)
    wP.set_file(name)
    wP.set_dir(models_dir)
    wP.set_cloudy_others(options)
    wP.set_N_Hb_cut(4)
    wP.set_geometry('Sphere')
    wP.set_stop(('temperature 20', 'pfrac 0.02'))
    c = pc.CST.CLIGHT
    # Starting the main loop on the 4 parameters.
    for lU_mean, ab_O, NO, age, fr in all_tabs:
        U_mean = 10**lU_mean
        w = (1 + fr**3.)**(1./3) - fr
        Q0 = 4. * np.pi * c**3 * U_mean**3 / (3. * NH * ff**2 * alpha_B**2 * w**3)
        R_str = (3. * Q0 / (4 * np.pi * NH**2 * alpha_B * ff))**(1./3)
        R_in = fr * R_str
        if fr < 1.0:
            wP.set_priority(5)
        else:
            wP.set_priority(15)
        wP.set_radius(r_in = np.log10(R_in))
        wP.set_cste_density(dens = np.log10(NH))
        ###GS  for Ne/O, S/O, Cl/O, Ar/O, Fe/O we take the average ot the ratios
        ###GS in the DR7-025-clean + DR10 samples from the O3O2 paper (Stasinska et al 2014)
        ###GS we do not take into account the observed variation of Fe/O
        ###GS for Si/O and Mg/O we take the CEL Orion values from Simon-Diaz & Stasinska 2011

        abund = {'N'  :   ab_O + NO,
                 'O'  :   ab_O,
                 'Ne' :   ab_O - 0.73,
                 'Mg' :   ab_O - 2.02,
                 'Si' :   ab_O - 2.02,
                 'S'  :   ab_O - 1.66,
                 'Cl' :   ab_O - 3.54,
                 'Ar' :   ab_O - 2.32,
                 'Fe' :   ab_O - 1.83}
        ab_O_12 = 12 + ab_O
        if ab_O_12 > 8.1:
            abund['Si'] -= 1
            abund['Mg'] -= 1
            abund['Fe'] -= 1.5
        abund['He'] = get_He(abund['O'])
        ###GS C is inspired by CEL vales of C and N in Orion (SDS11) and fig 11 from Esteban et al 2014
        abund['C'] = 8.40 - 7.92 + abund['N']
        wP.set_abund(ab_dict = abund)

        ###Remy-Ruyer et al 2014, broken power-law XCO,z case (as recommended by them)
        xt = 8.10
        x = 12 + ab_O
        a = 2.21 # log(G/D) solar
        if x > xt:
            y = a + 1.00 * (8.69 - x)
        else:
            y = 0.96 + 3.10 * (8.69 - x)
        Draine_fact = 2./3. # Draine 2011,
        solar_GoD = 10**a
        this_GoD = 10**y # y is log(G/D)
        this_relative_GoD =  this_GoD / solar_GoD#  we want the correction to apply to solar Dust abundance
        this_relative_DoG = 1.0 / this_relative_GoD
        wP.set_dust('ism {0}'.format(Draine_fact * this_relative_DoG))

        wP.set_comments(('lU_mean = {0}'.format(lU_mean),
                        'fr = {0}'.format(fr),
                        'age = {0}'.format(age),
                        'ab_O = {0}'.format(ab_O),
                        'NO = {0}'.format(NO),
                        'SED = {0}'.format(SED)))
        if SED == 'sp_cha':
            metallicity = min(max(-1.87 + (ab_O - -3.3), -3.99), -1.31)
            wP.set_star('table stars', atm_file='sp_cha_c{0}.mod'.format(Cversion), atm1=age, atm2=metallicity,
                        lumi_unit= 'q(H)', lumi_value = np.log10(Q0))
            wP.insert_model()
        elif SED == 'test':
            pass
        else:
            pc.log_.error('Unknown SED {0}'.format(SED))


def get_data():
    import pymysql
    co = pymysql.connect(host='132.248.1.102', db='3MdB', user='OVN_user', passwd='oiii5007')
    # db = MdB_subproc(OVN_dic)
    select_ = """dens,substr(com2,5) as fr,radius,rout,substr(com3,6) as age,logU_in as log_U0,logU_mean,
substr(com1,10) as logUin, logQ0,logQ1,logQ2,logQ,ff,HELIUM,CARBON,NITROGEN,OXYGEN,NEON,SULPHUR,ARGON,IRON,
log10(H__1__4861A) as H__1__4861A,
log10(H__1__6563A) as H__1__6563A,
log10(H__1__4340A) as H__1__4340A,
log10(CA_B__5876A) as CA_B__5876A,
log10(HE_1__4471A) as HE_1__4471A,
log10(HE_2__4686A) as HE_2__4686A,
log10(N__2__6584A) as N__2__6584A,
log10(N__2__5755A) as N__2__5755A,
log10(N_2R__5755A) as N_2R__5755A,
log10(O__1__6300A) as O__1__6300A,
log10(O_II__3726A) as O_II__3726A,
log10(O_II__3729A) as O_II__3729A,
log10(O_II__7323A) as O_II__7323A,
log10(O_II__7332A) as O_II__7332A,
log10(O_2R__3726A) as O_2R__3726A,
log10(O_2R__3729A) as O_2R__3729A,
log10(O_2R__7323A) as O_2R__7323A,
log10(O_2R__7332A) as O_2R__7332A,
log10(TOTL__3727A) as TOTL__3727A,
log10(TOTL__7325A) as TOTL__7325A,
log10(O__3__5007A) as O__3__5007A,
log10(TOTL__4363A) as TOTL__4363A,
log10(NE_3__3869A) as NE_3__3869A,
log10(S_II__4070A) as S_II__4070A,
log10(S_II__4078A) as S_II__4078A,
log10(S_II__6716A) as S_II__6716A,
log10(S_II__6731A) as S_II__6731A,
log10(S__3__9069A) as S__3__9069A,
log10(S__3__6312A) as S__3__6312A,
log10(CL_3__5518A) as CL_3__5518A,
log10(CL_3__5538A) as CL_3__5538A,
log10(AR_3__7135A) as AR_3__7135A,
log10(AR_4__4711A) as AR_4__4711A,
log10(AR_4__4740A) as AR_4__4740A,
log10(FE_3__4659A) as FE_3__4659A,
A_HYDROGEN_vol_1,
A_NITROGEN_vol_1, A_OXYGEN_vol_1,A_OXYGEN_vol_2,T_HYDROGEN_vol_1,T_OXYGEN_vol_1,T_OXYGEN_vol_2, atm2 as metallicity """
    from_ = "tab,abion,teion"
    where_ = "tab.ref = 'HII_15' and tab.N=abion.N and tab.N=teion.N"
    group_ = "OXYGEN, NITROGEN-OXYGEN, age, fr, logUin"
    import pandas as pd
    pd.read_sql('SELECT %s FROM %s WHERE %s GROUP BY %s' % (select_, from_, where_, group_), con=co)

    co.select_dB(select_ = select_, from_ = from_, where_ = where_, order_ = None, group_ = group_,
                 limit_ = None,outfile='tab_HII_15_All.dat')
    co.select_dB(select_ = select_, from_ = from_, where_ = where_, order_ = None, group_ = group_,
                 limit_ = None,outfile='tab_HII_15_All.dat')

all_tabs =  [(lU_mean, ab_O, NO, age, fr)
             for lU_mean in tab_lU_mean
             for ab_O in tab_O
             for NO in tab_NO
             for age in tab_age
             for fr in tab_fr]

make_inputs(all_tabs, SED='sp_cha', Cversion='17')
