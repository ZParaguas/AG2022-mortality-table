# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:58:11 2022

@author: Eigenaar
"""

import numpy as np
import pandas as pd
  
INITIALIZE_YEAR = 2020

def import_systems():
    
    """
    Desc:
        Import AG2022 mortality model parameters
    """
    
    dfParamsM = pd.read_excel(r"data\AG2022_parameters.xlsx", sheet_name = "ParamsM", header=1, engine='openpyxl')
    dfParamsV = pd.read_excel(r"data\AG2022_parameters.xlsx", sheet_name = "ParamsV", header=1, engine='openpyxl')
    dfParamsCov = pd.read_excel(r"data\AG2022_parameters.xlsx", sheet_name = "Covariance", engine='openpyxl')
    
    return dfParamsM, dfParamsV, dfParamsCov 

def simulate_errors(dfParamsCov):
    
    """
    Desc:
        Given the Covariance Matrix, draw 173 times from iid normal distribution for Monte Carlo
    """
    
    mCov = dfParamsCov.iloc[1:3, 1:3].to_numpy()
    mErrors = np.random.multivariate_normal([0,0], mCov, 173).T

    return mErrors

def construct_time_effects(vKappaEU_temp, vKappaNL_temp, vFracKappa_temp):
    
    """
    Desc:
        Combine mortality trend vectors into time effects matrix
    """
    
    vKappaEU = vKappaEU_temp[1:]
    vKappaNL= vKappaNL_temp[1:]
    vFracKappa = vFracKappa_temp[:-2]
    
    vOnes = np.ones(172)
    mTimeEffects = np.array([
        vOnes,
        vKappaEU,
        vOnes,
        vKappaNL,
        vFracKappa
    ])
    
    return mTimeEffects

def unpack_parameters(dfParams):
        
    """
    Desc:
        Rearrange model paramaters for later use
    """
    
    Theta = dfParams['K(t)'][55]
    a = dfParams['K(t)'][56]
    c = dfParams['K(t)'][57]
    eta = dfParams['K(t)'][58]
    mAgeEffects = dfParams[['A(x)','B(x)','alpha(x)','beta(x)','frak-beta(x)']].to_numpy()
    
    K_0 = dfParams['K(t)'][49]
    kappa_0 = dfParams['kappa(t)'][49]
    fracKappa_0 = dfParams['frak-kappa(t)'][50]
    fracKappa_1 = dfParams['frak-kappa(t)'][51]

    coeff = {'Theta' : Theta,
             'a' : a,
             'c' : c,
             'eta' : eta,
             'mAgeEffects' : mAgeEffects,
             'K(0)': K_0,
             'kappa(0)' : kappa_0,
             'frak-kappa(0)' : fracKappa_0,
             'frak-kappa(1)' : fracKappa_1
        }    
    
    return coeff

def generate_mortality_time_effects(dfCoeff, mErrors):
    
    """
    Desc:
        Generate vector of mortality trends in mortality model by calculating point estimate at time t
    """

    # Initialize time series
    vKappaNL = []
    vKappaEU = []
    vFracKappa = []

    vKappaEU.append(dfCoeff['K(0)'])
    vKappaNL.append(dfCoeff['kappa(0)'])
    vFracKappa.append(dfCoeff['frak-kappa(0)'])
    vFracKappa.append(dfCoeff['frak-kappa(1)'])
    
    # Forcast mortality dynamics with ARIMA model structure 
    for t in range(1,173):
        KappaEU_t = vKappaEU[t-1] + dfCoeff['Theta'] + mErrors[0, t-1] 
        KappaNL_t = dfCoeff['a'] * vKappaNL[t-1] + dfCoeff['c'] + mErrors[1, t-1] 
        FracKappa_t = vFracKappa[t-1] * 1/2 
    
        vKappaEU.append(KappaEU_t)
        vKappaNL.append(KappaNL_t)
        vFracKappa.append(FracKappa_t)
        
    mTimeEffects = construct_time_effects(vKappaEU, vKappaNL, vFracKappa)
    
    return mTimeEffects

def mortality_table(mTimeEffects, dfCoeff):
    
    """
    Desc:
        Generate AG2022 mortality table. 1-year mortality rates 
    """

    mAgeEffects = dfCoeff['mAgeEffects']
    mHazardRate = np.exp(mTimeEffects.T @ mAgeEffects.T)
    mMortalityTable = 1 - np.exp(-mHazardRate).T 
    
    return mMortalityTable
    
def generate_tables(number_sims, dfParams, dfParamsCov, dfExperimentalCorrection, iStochastic):
    
    """
    Desc:
        Given provided number of simulations, generate period life tabes
    """   
    
    dfCoeff = unpack_parameters(dfParams)
    vMortality_tables = []
    
    for _ in range(number_sims):
        mErrors = simulate_errors(dfParamsCov) * iStochastic        
        mTimeEffects = generate_mortality_time_effects(dfCoeff, mErrors)    
        mMortality_table = mortality_table(mTimeEffects, dfCoeff)
        
        vMortality_tables.append(mMortality_table)
    
    return vMortality_tables

def generate_prognosis_data(number_sims):
    
    """
    Desc:
        Parse life expectancy Best Estimates and stochastic data
    """   
        
    dfParamsM, dfParamsV, dfParamsCov = import_systems()
    
    mBestEstimate_temp = generate_tables(1, dfParamsM, dfParamsCov, iStochastic = 0)
    mSimulatedTables_temp = generate_tables(number_sims, dfParamsM, dfParamsCov, iStochastic = 1)
    
    mBestEstimate = mBestEstimate_temp[0]
    mSimulatedTables = np.array(mSimulatedTables_temp)
        
    return mBestEstimate, mSimulatedTables

