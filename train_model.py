
import os
from zipfile import ZipFile
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json


ZIP_PATH = './Capstone Project (2).zip' 
XLS_IN_ZIP = 'Updated Problem Statement Packs/3. Credit Card Delinquency Pack/Credit Card Delinquency Watch.xlsx'
SHEET_NAME = 'Sample'  
OUT_DIR = './cc_delinquency_model'
RANDOM_STATE = 42
RF_ESTIMATORS = 200

os.makedirs(OUT_DIR, exist_ok=True)

print('Loading data from zip...')
with ZipFile(ZIP_PATH, 'r') as z:
    with z.open(XLS_IN_ZIP) as f:
        df = pd.read_excel(f, sheet_name=SHEET_NAME)

print('Raw columns:', list(df.columns))


df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={
    'Customer ID': 'Customer_ID',
    'Credit Limit': 'Credit_Limit',
    'Utilisation %': 'Utilisation_%',
    'Avg Payment Ratio': 'Avg_Payment_Ratio',
    'Min Due Paid Frequency': 'Min_Due_Paid_Frequency',
    'Merchant Mix Index': 'Merchant_Mix_Index',
    'Cash Withdrawal %': 'Cash_Withdrawal_%',
    'Recent Spend Change %': 'Recent_Spend_Change_%',
    'DPD Bucket Next Month': 'DPD_Bucket_Next_Month'
})


print('\nData sample:')
print(df.head().to_string(index=False))


print(f'\nDataset shape: {df.shape}')
if 'DPD_Bucket_Next_Month' not in df.columns:
    raise ValueError('Target column DPD_Bucket_Next_Month not found.')


df = df.dropna(subset=['DPD_Bucket_Next_Month'])

percent_cols = ['Utilisation_%','Avg_Payment_Ratio','Min_Due_Paid_Frequency','Cash_Withdrawal_%','Recent_Spend_Change_%']
for c in percent_cols:
    if c in df.columns:
       
        df[c] = df[c].astype(str).str.rstrip('%').replace('', '0').astype(float)


X = df.drop(columns=['Customer_ID','DPD_Bucket_Next_Month'], errors='ignore')
y = df['DPD_Bucket_Next_Month']

print('\nFeatures used:', list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)


rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE))
])

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

print('\nTraining RandomForestRegressor...')
rf_pipeline.fit(X_train, y_train)
print('Training LinearRegression...')
lr_pipeline.fit(X_train, y_train)


rf_pred = rf_pipeline.predict(X_test)
lr_pred = lr_pipeline.predict(X_test)


from math import sqrt
rf_mse = mean_squared_error(y_test, rf_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
rf_rmse = sqrt(rf_mse)
lr_rmse = sqrt(lr_mse)

rf_r2 = r2_score(y_test, rf_pred)
lr_r2 = r2_score(y_test, lr_pred)


print('\n--- Evaluation on test set ---')
print(f'RandomForest RMSE: {rf_rmse:.4f}  R2: {rf_r2:.4f}')
print(f'LinearRegression RMSE: {lr_rmse:.4f}  R2: {lr_r2:.4f}')


rf_model = rf_pipeline.named_steps['rf']
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print('\nFeature importances (RandomForest):')
print(feat_imp.to_string())


print(f'\nSaving artifacts to {OUT_DIR} ...')
joblib.dump(rf_pipeline, os.path.join(OUT_DIR, 'rf_pipeline.pkl'))
joblib.dump(lr_pipeline, os.path.join(OUT_DIR, 'lr_pipeline.pkl'))
X_train.to_csv(os.path.join(OUT_DIR,'X_train_sample.csv'), index=False)
X_test.to_csv(os.path.join(OUT_DIR,'X_test_sample.csv'), index=False)
y_test.to_csv(os.path.join(OUT_DIR,'y_test_sample.csv'), index=False)
feat_imp.to_csv(os.path.join(OUT_DIR,'feature_importance.csv'))


summary = {
    'rf_rmse': float(rf_rmse),
    'lr_rmse': float(lr_rmse),
    'rf_r2': float(rf_r2),
    'lr_r2': float(lr_r2),
    'feature_importance': feat_imp.to_dict(),
    'models': {
        'rf': os.path.join(OUT_DIR, 'rf_pipeline.pkl'),
        'lr': os.path.join(OUT_DIR, 'lr_pipeline.pkl')
    }
}
with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('Done. Artifacts saved:')
print(os.listdir(OUT_DIR))
