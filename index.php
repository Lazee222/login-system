<?php
header("Location: login.php");
exit();
?>

http://localhost/login_system/login.php


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

messy_data = {
'patient_name': ['Dela Cruz', 'Santos', 'Reyes', 'bautista', 'GARCIA ', ' Mendoza', 'Torres', 'Ramos',
    'Aquino', 'Fernandez', 'delaCruz', 'Santos', 'Villanueva', 'Domingo', 'Castro',
    None, 'Reyes', 'Navarro', None, 'Pascual', 'Torres', 'Aquino', 'Bautista', 'Cruz', 'Mercado'],
'sex': ['M','F','Male','female','M','F','m','F','M','Female','F','M','F','M','F',
    'M','F','M','F','M','F','M','F','M','F'],
'barangay': ['San Roque','Bagumbayan','san roque','Poblacion','Bagumbayan ','San Jose',
    'Poblacion','Sto. Nino','San Roque','Bagumbayan','Poblacion','San Jose',
    'Sto Nino','San Roque','Bagumbayan','Poblacion','San Roque','San Jose',
    'Bagumbayan','Poblacion','Sto. Nino','San Roque','San Jose','Bagumbayan','Poblacion'],
'visit_date': ['2026-01-05','01/06/2026','2026-1-7','2026-01-08','08-01-2026','2026-01-09',
    '2026-01-09','2026/01/10','2026-01-11','01-12-2026','2026-01-13','2026-01-13',
    '2026-01-14','2026-01-15',None,'2026-01-16','2026-01-17','2026-01-18',
    '2026-01-19','2026-01-20','2026-01-20','2026-01-21','2026-01-22','2026-01-23','2026-01-24'],
'age': [34, 5, 150, 27, np.nan, 62, 41, 8, np.nan, 29, 34, 5, 71, 45, 19, -3, 55, np.nan, 22, 38, 41, np.nan, 60, 34, 47],
    'weight_kg': ['65kg','18.5','58 kg','70.2','80','45.0kg', '150', '20', '55.5', '62', '65kg', '18.5',
    '68', '75', np.nan, '-40', '90kg', '52', '30', '77', '68', '58', '95', '65kg', '61'],
    'height_cm': [165, 110, 170, np.nan, 175, 150, 168, 128, 160, 172, 165, 110, 155, 180, 140,
    169, 158, np.nan, 132, 162, 168, 173, 5.9, 165, 159],
    'bp_systolic': ['120','95','210','130','118','160','122','100','300','125','120','95','140',
    '135','108','119','150','127','98','131','122','128','145','120','133'],
    'temperature_c': [36.5, 37.0, 39.8, 36.7, 36.6, 38.2, 36.9, 36.8, 42.5, 36.5, 36.5, 37.0,
    37.1, 36.4, 36.9, 36.6, 40.1, 36.7, 36.8, 36.9, 36.5, 37.2, 39.0, 36.5, 36.6],
    'diagnosis': ['Hypertension','Common Cold','hypertension','Diabetes','COMMON COLD','Flu',
    'Hypertension ','Asthma','Diabetes Type 2','common cold','Hypertension','Common Cold',
    'Migraine','Flu','UTI','Hypertension','diabetes','Asthma','Common Cold','Flu',
    'Hypertension','Diabetes','Migrain','Hypertension','UTI'],
    'follow_up_needed': ['Yes','No','yes','NO','Y','N','Yes','No','yes','No','Yes','No','No',
    'Yes','No','Yes','yes','No','N','Yes','Yes','No','Y','Yes','No'],
    'senior_citizen': [0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,np.nan]
}

df = pd.DataFrame(messy_data)
print('Original shape:', df.shape)


TEST 2 Missing
missing = df.isna().sum()

missing[missing > 0]



TEST 3Clean Text data 
"df_clean = df.copy()
df_clean['patient_name'] = (df_clean['patient_name'].astype('string')
                            .str.strip()
                            .str.replace(r'\s+', ' ', regex=True))
df_clean['sex'] = (df_clean['sex'].str.strip().str.lower()
                  .map({'m':'Male', 'male':'Male', 'f':'Female', 'female':'Female'}))
df_clean['barangay'] = (df_clean['barangay'].str.strip()
                        .str.replace('.', '', regex=False)
                        .str.replace(r'\s+', ' ', regex=True)
                        .str.title())

print('Unique barangays before cleaning:', df['barangay'].nunique())
print('Unique barangays after cleaning:', df_clean['barangay'].nunique())
print(sorted(df_clean['barangay'].unique()))"


TEST 4 CHECK VISIT DATES
def clean_date(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if s == '08-01-2026':
        return pd.Timestamp('2026-01-08')
    if s == '01-12-2026':
        return pd.Timestamp('2026-01-12')
    if s.startswith('2026/'):
        return pd.to_datetime(s, format='%Y/%m/%d', errors='coerce')
    if '/' in s:
        return pd.to_datetime(s, format='%m/%d/%Y', errors='coerce')
    return pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')

df_clean['visit_date'] = df['visit_date'].map(clean_date)
df_clean = df_clean.sort_values('visit_date', na_position='last').reset_index(drop=True)

visits_per_day = df_clean.groupby('visit_date').size()
print(visits_per_day)
print('\nDays with more than one visit:')
print(visits_per_day[visits_per_day > 1])


TEST 5  Clean numeric measurements
df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
df_clean.loc[~df_clean['age'].between(0, 120), 'age'] = np.nan

df_clean['weight_kg'] = pd.to_numeric(
    df_clean['weight_kg'].astype('string').str.extract(r'(-?\d+(?:\.\d+)?)')[0],
    errors='coerce'
)
df_clean.loc[df_clean['weight_kg'] <= 0, 'weight_kg'] = np.nan

df_clean['height_cm'] = pd.to_numeric(df_clean['height_cm'], errors='coerce')

df_clean.loc[df_clean['height_cm'] == 5.9, 'height_cm'] = 5.9 * 30.48
df_clean.loc[~df_clean['height_cm'].between(50, 250), 'height_cm'] = np.nan

df_clean['bp_systolic'] = pd.to_numeric(df_clean['bp_systolic'], errors='coerce')
df_clean.loc[~df_clean['bp_systolic'].between(70, 250), 'bp_systolic'] = np.nan

df_clean['temperature_c'] = pd.to_numeric(df_clean['temperature_c'], errors='coerce')
df_clean.loc[~df_clean['temperature_c'].between(34, 42), 'temperature_c'] = np.nan

print(df_clean[['age','weight_kg','height_cm','bp_systolic','temperature_c']].describe())

  

 TEST 6 STANDARDIZE DIAGNOSIS AND FOLLOW-UP
df_clean['diagnosis'] = (df_clean['diagnosis'].astype('string').str.strip().str.lower()
                        .str.replace(r'\s+', ' ', regex=True).str.title()
                        .replace({'Migrain':'Migraine'}))

df_clean['follow_up_needed'] = (df_clean['follow_up_needed'].astype('string').str.strip().str.lower()
                               .map({'yes':'Yes','y':'Yes','no':'No','n':'No'}))

print(df_clean['diagnosis'].value_counts())


TEST 7 Create the final df_clean
df_clean = df_clean[df_clean['patient_name'].notna()].copy()
df_clean = df_clean.reset_index(drop=True)

valid_patients = len(df_clean)
average_age = df_clean['age'].mean()
most_common_diagnosis = df_clean['diagnosis'].mode()[0]
follow_up_percentage = (df_clean['follow_up_needed'].eq('Yes').mean()) * 100

print('Total valid patients:', valid_patients)
print('Average age:', round(average_age, 2))
print('Most common diagnosis:', most_common_diagnosis)
print('Percentage needing follow-up:', round(follow_up_percentage, 2), '%')

print('\nFinal df_clean:')
display(df_clean)


TEST 8 VISUALIZATIONS
date_counts = df_clean.dropna(subset=['visit_date']).groupby('visit_date').size()
plt.figure(figsize=(10,5))
date_counts.plot(marker='o')
plt.title('Patient Visits by Date')
plt.xlabel('Visit Date')
plt.ylabel('Number of Visits')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
