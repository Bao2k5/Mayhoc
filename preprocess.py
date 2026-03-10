import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Unified feature engineering for both training and inference.
    Creates 52 features as required by the best-performing models.
    """
    # 1. Payment Delay Features
    pay_cols = [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]
    df['MAX_PAY_DELAY'] = df[pay_cols].max(axis=1)
    df['MIN_PAY_DELAY'] = df[pay_cols].min(axis=1)
    df['AVG_PAY_DELAY'] = df[pay_cols].mean(axis=1)
    df['PAY_DELAY_SUM'] = df[pay_cols].sum(axis=1)
    df['STD_PAY_DELAY'] = df[pay_cols].std(axis=1)
    df['PAY_DELAY_TREND'] = df['PAY_0'] - df['PAY_6']
    
    # 2. Bill Amount Features
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)
    df['MAX_BILL_AMT'] = df[bill_cols].max(axis=1)
    df['MIN_BILL_AMT'] = df[bill_cols].min(axis=1)
    df['STD_BILL_AMT'] = df[bill_cols].std(axis=1)
    df['BILL_AMT_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
    
    # 3. Payment Amount Features
    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    df['AVG_PAY_AMT'] = df[pay_amt_cols].mean(axis=1)
    df['MAX_PAY_AMT'] = df[pay_amt_cols].max(axis=1)
    df['MIN_PAY_AMT'] = df[pay_amt_cols].min(axis=1)
    df['STD_PAY_AMT'] = df[pay_amt_cols].std(axis=1)
    df['PAY_AMT_TREND'] = df['PAY_AMT1'] - df['PAY_AMT6']
    
    # 4. Ratios and Consistency
    df['UTILIZATION_RATE'] = (df['AVG_BILL_AMT'] / df['LIMIT_BAL'] * 100).clip(upper=100)
    df['MAX_UTILIZATION'] = (df['MAX_BILL_AMT'] / df['LIMIT_BAL'] * 100).clip(upper=100)
    # Correcting trend for utilization
    df['UTILIZATION_TREND'] = (df['BILL_AMT1'] / df['LIMIT_BAL']) - (df['BILL_AMT6'] / df['LIMIT_BAL'])
    
    df['PAYMENT_RATIO'] = np.where(df['AVG_BILL_AMT'] > 0, (df['AVG_PAY_AMT'] / df['AVG_BILL_AMT'] * 100).clip(upper=200), 0)
    df['TIMES_DELAYED'] = (df[pay_cols] > 0).sum(axis=1)
    
    # 5. Flags (NEVER/ALWAYS DELAYED)
    df['NEVER_DELAYED'] = (df['TIMES_DELAYED'] == 0).astype(int)
    df['ALWAYS_DELAYED'] = (df['TIMES_DELAYED'] == len(pay_cols)).astype(int)
    
    # 6. Interaction Features
    df['AGE_LIMIT'] = df['AGE'] * df['LIMIT_BAL'] / 1000000
    df['EDUCATION_LIMIT'] = df['EDUCATION'] * df['LIMIT_BAL'] / 100000
    df['AGE_UTILIZATION'] = df['AGE'] * df['UTILIZATION_RATE'] / 100
    df['PAY_TO_LIMIT_RATIO'] = df['AVG_PAY_AMT'] / df['LIMIT_BAL']
    
    # 7. Recent Trends (Month 0 vs Month 2)
    df['RECENT_PAY_TREND'] = df['PAY_0'] - df['PAY_2']
    df['RECENT_BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT2']
    df['RECENT_PAYMENT_TREND'] = df['PAY_AMT1'] - df['PAY_AMT2']
    
    return df
