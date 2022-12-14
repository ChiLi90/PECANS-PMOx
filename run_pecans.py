import os
import numpy as np
import argparse
from pecans.utilities.config import load_config_file
from pecans.main import Domain
from pecans.ensembles.api import EnsembleRunner
import pdb

_mydir = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
config_file = os.path.join(os.path.dirname(__file__), 'pecans_config.cfg')


def name_nox_pan_test_width_output_files(index, **config_opts):
    emissions_width_km = config_opts['EMISSIONS/no_emission_opts/width_x'] / 1000
    ho2_ppt = 6e7 / 2e19 * 1e12
    return 'pecans_ens_ho2-{}ppt_emwidth-{}km'.format(ho2_ppt, emissions_width_km)

def name_nox_pan_test_ho2_output_files(index, **config_opts):
    ho2_ppt = config_opts['CHEMISTRY/const_species/ACO3'] / 2e19 * 1e12
    return 'pecans_ens_ho2-{}ppt_emwidth-{}km'.format(ho2_ppt, 36)

def name_nox_pan_test_ho2_pointsource_output_files(index, **config_opts):
    ho2_ppt = config_opts['CHEMISTRY/const_species/ACO3'] / 2e19 * 1e12
    return 'pecans_ens_ho2-{}ppt_pointsource'.format(ho2_ppt, 36)

def name_nox_pan_test_ho_output_files(index, **config_opts):
    ho_ppt = config_opts['CHEMISTRY/const_species/HO'] / 2e19 * 1e12
    return 'pecans_ens_ho-{}ppt_emwidth-{}km'.format(ho_ppt, 36)

def name_nox_voc_output_files(index, **config_opts):
    p_hox = config_opts['CHEMISTRY/const_species/P_HOx'] / 1e6
    rvoc = config_opts['CHEMISTRY/const_species/rVOC'] /2e19 *1e9
    return 'pecans_ens_p_hox-{}e6_rvoc-{}ppb'.format(p_hox, rvoc)

def name_nox_voc_output_files_for_emis_test(index, **config_opts):
    total = config_opts['EMISSIONS/no_emission_opts/total'] / 1e25
    return 'pecans_ens_nox_emis_{:.1f}e25_ovoc_0ppt_loc_o1'.format(total)

def name_nox_voc_output_files_for_bkg_emis_test(index, **config_opts):
    total = config_opts['EMISSIONS/no2_emission_opts/total'] / 1e23
    return 'pecans_ens_nox_bkg_emis_{:.1f}e23_loc_01'.format(total)

def nox_run():
    ho = np.arange(1e6, 15e6, 1e6)
    ensemble_variables = {'CHEMISTRY/const_species/HO': ho}
    member_naming_fxn = name_nox_pan_test_ho_output_files
    ens = EnsembleRunner(config_file,
                         ensemble_variables=ensemble_variables,
                         ensemble_mode='combinations',
                         save_in_individual_dirs=False,
                         save_final_output_only=True,
                         member_naming_fxn=member_naming_fxn,
                         root_output_dir=os.path.join(_mydir, '../../MATLAB/PAN_Data', 'Workspaces', 'PECANS',
                                                      'lifetime-ensemble-nox-odetest'))

    ens.run()

def nox_voc_ox_run(test):
    if test == 'p_hox_rvoc':
        p_hox = np.arange(1.6e8, 2.4e8, 0.1e8)
        rvoc = np.array([6e10, 7e10, 8e10])
        ensemble_variables = {'CHEMISTRY/const_species/P_HOx': p_hox, 'CHEMISTRY/const_species/rVOC': rvoc}
        member_naming_fxn = name_nox_voc_output_files
    if test == 'emis':
        total = np.append(np.arange(1.e23,3.e24,2.e23),np.arange(3.5e24,1.e25,4.e23))
        ensemble_variables = {'EMISSIONS/no_emission_opts/total': total}
        member_naming_fxn = name_nox_voc_output_files_for_emis_test
    if test == 'bkg_no2':
        no2_emis = np.arange(0.5e23, 15e24, 0.5e23)
        ensemble_variables = {'EMISSIONS/no2_emission_opts/total': no2_emis}
        member_naming_fxn = name_nox_voc_output_files_for_bkg_emis_test
    print('NO   NO2     O3  HOx     NOx     Ox  VOCR')
    ens = EnsembleRunner(config_file,
                         ensemble_variables=ensemble_variables,
                         ensemble_mode='combinations',
                         save_in_individual_dirs=False,
                         save_final_output_only=False,
                         member_naming_fxn=member_naming_fxn,
                         root_output_dir=os.path.join(_mydir, 'output'))

    ens.run()

def nox_voc_run(test):
    if test == 'p_hox_rvoc':
        p_hox = np.arange(1.6e8, 2.4e8, 0.1e8)
        rvoc = np.array([6e10, 7e10, 8e10])
        ensemble_variables = {'CHEMISTRY/const_species/P_HOx': p_hox, 'CHEMISTRY/const_species/rVOC': rvoc}
        member_naming_fxn = name_nox_voc_output_files
    if test == 'emis':
        total = np.arange(0.1e26, 1.1e26, 0.1e26)
        ensemble_variables = {'EMISSIONS/no_emission_opts/total': total}
        member_naming_fxn = name_nox_voc_output_files_for_emis_test
    if test == 'bkg_no2':
        no2_emis = np.arange(0.5e23, 15e24, 0.5e23)
        ensemble_variables = {'EMISSIONS/no2_emission_opts/total': no2_emis}
        member_naming_fxn = name_nox_voc_output_files_for_bkg_emis_test
    ens = EnsembleRunner(config_file,
                         ensemble_variables=ensemble_variables,
                         ensemble_mode='combinations',
                         save_in_individual_dirs=False,
                         save_final_output_only=False,
                         member_naming_fxn=member_naming_fxn,
                         root_output_dir=os.path.join(_mydir, 'output'))

    ens.run()

def nox_pan_run(test, emis):
    if test == 'emwidth':
        widths = np.arange(3600, 9 * 3600 + 1, 3600) * 5
        widths = np.concatenate(([3000], widths))
        ensemble_variables = {'EMISSIONS/no_emission_opts/width_x': widths}
        member_naming_fxn = name_nox_pan_test_width_output_files
    elif test == 'HO2':
        ho2 = np.arange(1e7, 30e7, 2e7)
        ensemble_variables = {'CHEMISTRY/const_species/ACO3': ho2}
        if emis == 'point':
            member_naming_fxn = name_nox_pan_test_ho2_pointsource_output_files
        else:
            member_naming_fxn = name_nox_pan_test_ho2_output_files
    elif test == 'HO':
        ho = np.arange(1e6, 15e6, 1e6)
        ensemble_variables = {'CHEMISTRY/const_species/HO': ho}
        member_naming_fxn = name_nox_pan_test_ho_output_files
    ens = EnsembleRunner(config_file,
                         ensemble_variables=ensemble_variables,
                         ensemble_mode='combinations',
                         save_in_individual_dirs=False,
                         save_final_output_only=True,
                         member_naming_fxn=member_naming_fxn,
                         root_output_dir=os.path.join(_mydir, '../../MATLAB/PAN_Data', 'Workspaces', 'PECANS',
                                                      'lifetime-ensemble-nox-pan'))

    ens.run()


def main():
    parser = argparse.ArgumentParser(description='Choose one of the chemical solvers')
    parser.add_argument('solver', type=str, help='What the chemical solver is. Default is "first_order"')
    parser.add_argument('--test', '-t', default='HO2', help='What parameter we are testing. Default is "ACO3"')
    parser.add_argument('--emis', '-e', default='gaussian', help='What is the shape of NO emission. Default is'
                                                                 ' "Gaussian"')
    args = parser.parse_args()
    print(args.solver)
    if args.solver == 'nox':
        nox_run()
    elif args.solver == 'nox_pan':
        nox_pan_run(args.test, args.emis)
    elif args.solver == 'nox_voc':
        nox_voc_run(args.test)
    elif args.solver == 'nox_voc_ox':
        nox_voc_ox_run(args.test)
    elif args.solver == 'single_test':
        config_file_name = os.path.join(os.path.dirname(__file__), 'pecans_config.cfg')
        config = load_config_file(config_file_name)
        model_domain = Domain(config, output_dir='test/')
        model_domain.execute()
    else:
        print("The chemical solver is not implemented.")
        quit()


if __name__ == '__main__':
    main()
