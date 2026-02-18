import pandas as pd
import numpy as np

# Загрузка данных
data_clean = pd.read_csv('wells_info_na.csv', dtype=str)

# Копия для работы

# Нормализуем пустые строки -> NaN
data_clean = data_clean.replace({'': np.nan})

# Заполняем числовые значения медианой (сначала приводим к числовому типу)
num_cols = ['LatWGS84', 'LonWGS84', 'PROP_PER_FOOT']
for col in num_cols:
    if col in data_clean.columns:
        data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
        data_clean[col] = data_clean[col].fillna(data_clean[col].median())

# Заполняем категориальные/дата модой
cat_cols = ['CompletionDate', 'FirstProductionDate', 'formation', 'BasinName', 'StateName', 'CountyName']
for col in cat_cols:
    if col in data_clean.columns:
        mode_vals = data_clean[col].mode(dropna=True)
        if not mode_vals.empty:
            data_clean[col] = data_clean[col].fillna(mode_vals.iloc[0])

# Сохраняем результат
data_clean.to_csv('wells_info_na_filled.csv', index=False)
