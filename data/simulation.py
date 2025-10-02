"""
Data simulation for the causal inference example.
This module generates a semi-synthetic dataset that includes
confounders and a collider to demonstrate common challenges in
causal estimation.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

def simulate_adr_data(n_patients=2000, seed=42):
    """
    Simulates a dataset for an adverse drug reaction (ADR) study.

    The simulation includes:
    - Multiple confounders (disease severity, age, sex, comorbidities)
    - A collider (hospitalization)

    Args:
        n_patients (int): Number of patients to simulate.
        seed (int): Random seed for reproducibility.

    Returns:
        pandas.DataFrame: A dataframe containing the simulated patient data.
    """
    np.random.seed(seed)

    # 1. Simulate confounders
    # L = disease severity (binary, 1=severe)
    L = np.random.binomial(1, 0.2, n_patients)
    # Age (continuous, centered and scaled)
    age = (np.random.normal(65, 10, n_patients) - 65) / 10
    # Sex (binary, 1=female)
    sex = np.random.binomial(1, 0.5, n_patients)
    # Comorbidities (count)
    comorbidities = np.random.poisson(1.5, n_patients)

    # 2. Simulate treatment assignment (A) based on confounders
    # Older, sicker patients are more likely to get the drug
    p_A_given_L = expit(-1 + 2.5 * L + 0.2 * age + 0.1 * sex + 0.3 * comorbidities)
    A = np.random.binomial(1, p_A_given_L, n_patients)

    # 3. Simulate outcome (Y - ADR) based on treatment and confounders
    # True causal effect of drug A on ADR is log(OR) = 0.8
    theta_true_adr = 0.8
    p_Y = expit(-2 + theta_true_adr * A + 1.5 * L + 0.3 * age + 0.2 * sex + 0.4 * comorbidities)
    Y = np.random.binomial(1, p_Y, n_patients)

    # 4. Simulate a collider (hospitalization)
    # Hospitalization is caused by both treatment and having an ADR
    p_hospitalized = expit(-1.5 + 1.0 * A + 1.5 * Y)
    hospitalized = np.random.binomial(1, p_hospitalized, n_patients)

    # 5. Assemble DataFrame
    df = pd.DataFrame({
        'L': L,
        'age': age,
        'sex': sex,
        'comorbidities': comorbidities,
        'A': A,
        'Y': Y,
        'hospitalized': hospitalized,
        'theta_true_adr': theta_true_adr
    })

    return df