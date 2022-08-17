from libc.math cimport exp, sqrt, log, log10
import numpy as np

alpha=0.04

def rhs(f, double t, param):
    [this_conc_NO ,this_conc_NO2 ,this_conc_O3 ,this_conc_HO ,this_conc_HO2 ,this_conc_HNO3 ,this_conc_rVOCRO2 ,this_conc_RONO2 ,this_conc_ROOH ,this_conc_OVOCRO2 ,this_conc_PAN ,this_conc_H2O2 ,this_conc_P_HOx ] = f
    [TEMP, CAIR, this_conc_OVOC ,this_conc_PM ,this_conc_rVOC ] = param
    cdef double dconc_NO_dt = dNO_dt( TEMP,  CAIR,  this_conc_NO,  this_conc_HO2,  this_conc_rVOCRO2,  this_conc_NO2,  this_conc_OVOCRO2,  this_conc_O3)
    cdef double dconc_NO2_dt = dNO2_dt( TEMP,  CAIR,  this_conc_rVOCRO2,  this_conc_HO,  this_conc_PAN,  this_conc_NO,  this_conc_HO2,  this_conc_NO2,  this_conc_OVOCRO2,  this_conc_O3)
    cdef double dconc_O3_dt = dO3_dt( TEMP,  CAIR,  this_conc_NO2,  this_conc_NO,  this_conc_O3)
    cdef double dconc_HO_dt = dHO_dt( TEMP,  CAIR,  this_conc_HO2,  this_conc_NO,  this_conc_P_HOx,  this_conc_rVOC,  this_conc_OVOC,  this_conc_NO2,  this_conc_HO)
    cdef double dconc_HO2_dt = dHO2_dt( TEMP,  CAIR,  this_conc_NO,  this_conc_HO2,  this_conc_PM,  this_conc_P_HOx,  this_conc_rVOCRO2,  this_conc_OVOCRO2)
    cdef double dconc_HNO3_dt = dHNO3_dt( TEMP,  CAIR,  this_conc_NO2,  this_conc_HO)
    cdef double dconc_rVOCRO2_dt = drVOCRO2_dt( TEMP,  CAIR,  this_conc_NO,  this_conc_HO2,  this_conc_rVOC,  this_conc_rVOCRO2,  this_conc_HO,  this_conc_OVOCRO2)
    cdef double dconc_RONO2_dt = dRONO2_dt( TEMP,  CAIR,  this_conc_rVOCRO2,  this_conc_NO)
    cdef double dconc_ROOH_dt = dROOH_dt( TEMP,  CAIR,  this_conc_HO2,  this_conc_rVOCRO2,  this_conc_OVOCRO2)
    cdef double dconc_OVOCRO2_dt = dOVOCRO2_dt( TEMP,  CAIR,  this_conc_NO,  this_conc_HO2,  this_conc_OVOC,  this_conc_rVOCRO2,  this_conc_NO2,  this_conc_HO,  this_conc_OVOCRO2,  this_conc_PAN)
    cdef double dconc_PAN_dt = dPAN_dt( TEMP,  CAIR,  this_conc_NO2,  this_conc_OVOCRO2,  this_conc_PAN)
    cdef double dconc_H2O2_dt = dH2O2_dt( TEMP,  CAIR,  this_conc_HO2,  this_conc_NO,  this_conc_PM,  this_conc_NO2,  this_conc_O3)
    cdef double dconc_P_HOx_dt = dP_HOx_dt( TEMP,  CAIR,  this_conc_P_HOx,  this_conc_O3)
    return np.array([dconc_NO_dt, dconc_NO2_dt, dconc_O3_dt, dconc_HO_dt, dconc_HO2_dt, dconc_HNO3_dt, dconc_rVOCRO2_dt, dconc_RONO2_dt, dconc_ROOH_dt, dconc_OVOCRO2_dt, dconc_PAN_dt, dconc_H2O2_dt, dconc_P_HOx_dt])

def chem_solver(double dt, double TEMP, double CAIR, double conc_NO, double conc_NO2, double conc_O3, double conc_HO, double conc_HO2, double conc_HNO3, double conc_rVOC, double conc_rVOCRO2, double conc_RONO2, double conc_ROOH, double conc_OVOC, double conc_OVOCRO2, double conc_PAN, double conc_H2O2, double conc_PM, double conc_P_HOx):
    cdef double dconc_NO = dNO_dt( TEMP,  CAIR,  conc_NO,  conc_HO2,  conc_rVOCRO2,  conc_NO2,  conc_OVOCRO2,  conc_O3) * dt
    cdef double dconc_NO2 = dNO2_dt( TEMP,  CAIR,  conc_rVOCRO2,  conc_HO,  conc_PAN,  conc_NO,  conc_HO2,  conc_NO2,  conc_OVOCRO2,  conc_O3) * dt
    cdef double dconc_O3 = dO3_dt( TEMP,  CAIR,  conc_NO2,  conc_NO,  conc_O3) * dt
    cdef double dconc_HO = dHO_dt( TEMP,  CAIR,  conc_HO2,  conc_NO,  conc_P_HOx,  conc_rVOC,  conc_OVOC,  conc_NO2,  conc_HO) * dt
    cdef double dconc_HO2 = dHO2_dt( TEMP,  CAIR,  conc_NO,  conc_HO2,  conc_PM,  conc_P_HOx,  conc_rVOCRO2,  conc_OVOCRO2) * dt
    cdef double dconc_HNO3 = dHNO3_dt( TEMP,  CAIR,  conc_NO2,  conc_HO) * dt
    cdef double dconc_rVOC = drVOC_dt( TEMP,  CAIR,  conc_HO,  conc_rVOC) * dt
    cdef double dconc_rVOCRO2 = drVOCRO2_dt( TEMP,  CAIR,  conc_NO,  conc_HO2,  conc_rVOC,  conc_rVOCRO2,  conc_HO,  conc_OVOCRO2) * dt
    cdef double dconc_RONO2 = dRONO2_dt( TEMP,  CAIR,  conc_rVOCRO2,  conc_NO) * dt
    cdef double dconc_ROOH = dROOH_dt( TEMP,  CAIR,  conc_HO2,  conc_rVOCRO2,  conc_OVOCRO2) * dt
    cdef double dconc_OVOC = dOVOC_dt( TEMP,  CAIR,  conc_HO,  conc_OVOC) * dt
    cdef double dconc_OVOCRO2 = dOVOCRO2_dt( TEMP,  CAIR,  conc_NO,  conc_HO2,  conc_OVOC,  conc_rVOCRO2,  conc_NO2,  conc_HO,  conc_OVOCRO2,  conc_PAN) * dt
    cdef double dconc_PAN = dPAN_dt( TEMP,  CAIR,  conc_NO2,  conc_OVOCRO2,  conc_PAN) * dt
    cdef double dconc_H2O2 = dH2O2_dt( TEMP,  CAIR,  conc_HO2,  conc_NO,  conc_PM,  conc_NO2,  conc_O3) * dt
    cdef double dconc_PM = dPM_dt( TEMP,  CAIR,  conc_HO2,  conc_PM) * dt
    cdef double dconc_P_HOx = dP_HOx_dt( TEMP,  CAIR,  conc_P_HOx,  conc_O3) * dt
    return { "NO": conc_NO + dconc_NO, "NO2": conc_NO2 + dconc_NO2, "O3": conc_O3 + dconc_O3, "HO": conc_HO + dconc_HO, "HO2": conc_HO2 + dconc_HO2, "HNO3": conc_HNO3 + dconc_HNO3, "rVOC": conc_rVOC + dconc_rVOC, "rVOCRO2": conc_rVOCRO2 + dconc_rVOCRO2, "RONO2": conc_RONO2 + dconc_RONO2, "ROOH": conc_ROOH + dconc_ROOH, "OVOC": conc_OVOC + dconc_OVOC, "OVOCRO2": conc_OVOCRO2 + dconc_OVOCRO2, "PAN": conc_PAN + dconc_PAN, "H2O2": conc_H2O2 + dconc_H2O2, "PM": conc_PM + dconc_PM, "P_HOx": conc_P_HOx + dconc_P_HOx }

cdef double dNO_dt(double TEMP, double CAIR, double conc_NO, double conc_HO2, double conc_rVOCRO2, double conc_NO2, double conc_OVOCRO2, double conc_O3):
    cdef double dNO = -1.0*ARR2( 1.40e-12, 1310.0, TEMP )*conc_NO*conc_O3 + 1.0*0.01*conc_NO2 + -1.0*(1-alpha) * ARR2(2.7e-12, -360.0, TEMP)*conc_NO*conc_rVOCRO2 + -1.0*alpha * ARR2(2.7e-12, -360.0, TEMP)*conc_rVOCRO2*conc_NO + -1.0*ARR2(8.1e-12, -270.0, TEMP)*conc_OVOCRO2*conc_NO + -1.0*ARR2(3.5e-12, -250.0, TEMP)*conc_HO2*conc_NO + -1.0*1.e-5*conc_NO
    return dNO


cdef double dNO2_dt(double TEMP, double CAIR, double conc_rVOCRO2, double conc_HO, double conc_PAN, double conc_NO, double conc_HO2, double conc_NO2, double conc_OVOCRO2, double conc_O3):
    cdef double dNO2 = 1.0*ARR2( 1.40e-12, 1310.0, TEMP )*conc_NO*conc_O3 + -1.0*0.01*conc_NO2 + -1.0*TROE( 1.49e-30 , 1.8, 2.58e-11 , 0.0, TEMP, CAIR)*conc_HO*conc_NO2 + 1.0*(1-alpha) * ARR2(2.7e-12, -360.0, TEMP)*conc_NO*conc_rVOCRO2 + 1.0*ARR2(8.1e-12, -270.0, TEMP)*conc_OVOCRO2*conc_NO + -1.0*TROE( 9.70e-29 , 5.6 , 9.30e-12 , 1.5 , TEMP, CAIR)*conc_NO2*conc_OVOCRO2 + 1.0*TROEE(1.11e28,14000.0, 9.70e-29 , 5.6 , 9.30e-12 , 1.5 , TEMP, CAIR)*conc_PAN + 1.0*ARR2(3.5e-12, -250.0, TEMP)*conc_HO2*conc_NO + -1.0*1.e-5*conc_NO2
    return dNO2


cdef double dO3_dt(double TEMP, double CAIR, double conc_NO2, double conc_NO, double conc_O3):
    cdef double dO3 = -1.0*ARR2( 1.40e-12, 1310.0, TEMP )*conc_NO*conc_O3 + 1.0*0.01*conc_NO2 + -1.0*1.e-5*conc_O3 + -1.0*1.5e-6*conc_O3
    return dO3


cdef double dHO_dt(double TEMP, double CAIR, double conc_HO2, double conc_NO, double conc_P_HOx, double conc_rVOC, double conc_OVOC, double conc_NO2, double conc_HO):
    cdef double dHO = -1.0*TROE( 1.49e-30 , 1.8, 2.58e-11 , 0.0, TEMP, CAIR)*conc_HO*conc_NO2 + -1.0*3e-11*conc_rVOC*conc_HO + -1.0*ARR2(4.7e-12, -345.0, TEMP)*conc_HO*conc_OVOC + 1.0*ARR2(3.5e-12, -250.0, TEMP)*conc_HO2*conc_NO + 0.8*1*conc_P_HOx
    return dHO


cdef double dHO2_dt(double TEMP, double CAIR, double conc_NO, double conc_HO2, double conc_PM, double conc_P_HOx, double conc_rVOCRO2, double conc_OVOCRO2):
    cdef double dHO2 = 1.0*(1-alpha) * ARR2(2.7e-12, -360.0, TEMP)*conc_NO*conc_rVOCRO2 + -1.0*ARR2(2.9e-13, -1300.0, TEMP)*conc_HO2*conc_rVOCRO2 + 1.0*ARR2(8.1e-12, -270.0, TEMP)*conc_OVOCRO2*conc_NO + -1.0*ARR2(4.3e-13, -1040.0, TEMP)*conc_HO2*conc_OVOCRO2 + -1.0*ARR2(3.5e-12, -250.0, TEMP)*conc_HO2*conc_NO + -2.0*ARR2(2.2e-13, -266.0, TEMP)*conc_HO2**2.0 + -1.0*21.87*conc_PM*conc_HO2 + 0.2*1*conc_P_HOx
    return dHO2


cdef double dHNO3_dt(double TEMP, double CAIR, double conc_NO2, double conc_HO):
    cdef double dHNO3 = 1.0*TROE( 1.49e-30 , 1.8, 2.58e-11 , 0.0, TEMP, CAIR)*conc_HO*conc_NO2
    return dHNO3


cdef double drVOC_dt(double TEMP, double CAIR, double conc_HO, double conc_rVOC):
    cdef double drVOC = -1.0*3e-11*conc_rVOC*conc_HO
    return drVOC


cdef double drVOCRO2_dt(double TEMP, double CAIR, double conc_NO, double conc_HO2, double conc_rVOC, double conc_rVOCRO2, double conc_HO, double conc_OVOCRO2):
    cdef double drVOCRO2 = 1.0*3e-11*conc_rVOC*conc_HO + -1.0*(1-alpha) * ARR2(2.7e-12, -360.0, TEMP)*conc_NO*conc_rVOCRO2 + -1.0*alpha * ARR2(2.7e-12, -360.0, TEMP)*conc_rVOCRO2*conc_NO + -1.0*ARR2(2.9e-13, -1300.0, TEMP)*conc_HO2*conc_rVOCRO2 + -2.0*2.4e-12*conc_rVOCRO2**2.0 + -1.0*2.4e-12*conc_rVOCRO2*conc_OVOCRO2 + -1.0*ARR2(2.0e-12, -500.0, TEMP)*conc_OVOCRO2*conc_rVOCRO2
    return drVOCRO2


cdef double dRONO2_dt(double TEMP, double CAIR, double conc_rVOCRO2, double conc_NO):
    cdef double dRONO2 = 1.0*alpha * ARR2(2.7e-12, -360.0, TEMP)*conc_rVOCRO2*conc_NO
    return dRONO2


cdef double dROOH_dt(double TEMP, double CAIR, double conc_HO2, double conc_rVOCRO2, double conc_OVOCRO2):
    cdef double dROOH = 1.0*ARR2(2.9e-13, -1300.0, TEMP)*conc_HO2*conc_rVOCRO2 + 1.0*2.4e-12*conc_rVOCRO2**2.0 + 1.0*2.4e-12*conc_rVOCRO2*conc_OVOCRO2 + 1.0*ARR2(4.3e-13, -1040.0, TEMP)*conc_HO2*conc_OVOCRO2 + 1.0*ARR2(2.0e-12, -500.0, TEMP)*conc_OVOCRO2*conc_rVOCRO2 + 1.0*ARR2(2.0e-12, -500.0, TEMP)*conc_OVOCRO2**2.0
    return dROOH


cdef double dOVOC_dt(double TEMP, double CAIR, double conc_HO, double conc_OVOC):
    cdef double dOVOC = -1.0*ARR2(4.7e-12, -345.0, TEMP)*conc_HO*conc_OVOC
    return dOVOC


cdef double dOVOCRO2_dt(double TEMP, double CAIR, double conc_NO, double conc_HO2, double conc_OVOC, double conc_rVOCRO2, double conc_NO2, double conc_HO, double conc_OVOCRO2, double conc_PAN):
    cdef double dOVOCRO2 = -1.0*2.4e-12*conc_rVOCRO2*conc_OVOCRO2 + 1.0*ARR2(4.7e-12, -345.0, TEMP)*conc_HO*conc_OVOC + -1.0*ARR2(8.1e-12, -270.0, TEMP)*conc_OVOCRO2*conc_NO + -1.0*ARR2(4.3e-13, -1040.0, TEMP)*conc_HO2*conc_OVOCRO2 + -1.0*ARR2(2.0e-12, -500.0, TEMP)*conc_OVOCRO2*conc_rVOCRO2 + -2.0*ARR2(2.0e-12, -500.0, TEMP)*conc_OVOCRO2**2.0 + -1.0*TROE( 9.70e-29 , 5.6 , 9.30e-12 , 1.5 , TEMP, CAIR)*conc_NO2*conc_OVOCRO2 + 1.0*TROEE(1.11e28,14000.0, 9.70e-29 , 5.6 , 9.30e-12 , 1.5 , TEMP, CAIR)*conc_PAN
    return dOVOCRO2


cdef double dPAN_dt(double TEMP, double CAIR, double conc_NO2, double conc_OVOCRO2, double conc_PAN):
    cdef double dPAN = 1.0*TROE( 9.70e-29 , 5.6 , 9.30e-12 , 1.5 , TEMP, CAIR)*conc_NO2*conc_OVOCRO2 + -1.0*TROEE(1.11e28,14000.0, 9.70e-29 , 5.6 , 9.30e-12 , 1.5 , TEMP, CAIR)*conc_PAN
    return dPAN


cdef double dH2O2_dt(double TEMP, double CAIR, double conc_HO2, double conc_NO, double conc_PM, double conc_NO2, double conc_O3):
    cdef double dH2O2 = 1.0*ARR2(2.2e-13, -266.0, TEMP)*conc_HO2**2.0 + 1.0*1.e-5*conc_O3 + 1.0*1.e-5*conc_NO2 + 1.0*1.e-5*conc_NO + 1.0*21.87*conc_PM*conc_HO2
    return dH2O2


cdef double dPM_dt(double TEMP, double CAIR, double conc_HO2, double conc_PM):
    cdef double dPM = -1.0*21.87*conc_PM*conc_HO2
    return dPM


cdef double dP_HOx_dt(double TEMP, double CAIR, double conc_P_HOx, double conc_O3):
    cdef double dP_HOx = 1.0*1.5e-6*conc_O3 + -1.0*1*conc_P_HOx
    return dP_HOx


####################
# RATE EXPRESSIONS #
####################

cdef ARR2(double A0, double B0, double TEMP):
    return A0 * exp(-B0 / TEMP)


cdef TROE(double k0_300K, double n, double kinf_300K, double m, double TEMP, double CAIR):
    cdef double zt_help
    cdef double k0_T
    cdef double kinf_T
    cdef double k_ratio
    zt_help = 300.0 / TEMP;
    k0_T    = k0_300K   * zt_help ** n * CAIR   # k_0   at current T
    kinf_T  = kinf_300K * zt_help ** m          # k_inf at current T
    k_ratio = k0_T/kinf_T
    return k0_T/(1.0 + k_ratio)*0.6 ** (1.0 / (1.0+log10(k_ratio)**2))


cdef TROEE(double A, double B, double k0_300K, double n, double kinf_300K, double m, double TEMP, double CAIR):
    cdef double zt_help
    cdef double k0_T
    cdef double kinf_T
    cdef double k_ratio
    cdef double troe
    zt_help = 300.0 / TEMP;
    k0_T    = k0_300K   * zt_help ** n * CAIR   # k_0   at current T
    kinf_T  = kinf_300K * zt_help ** m          # k_inf at current T
    k_ratio = k0_T/kinf_T
    troe = k0_T/(1.0 + k_ratio)*0.6 ** (1.0 / (1.0+log10(k_ratio)**2))
    return A * exp(- B / TEMP) * troe


