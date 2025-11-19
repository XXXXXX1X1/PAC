import pandas as pd
import numpy as np

# Загружаем данные
titanic_data = pd.read_csv(
    'titanic_with_labels.csv',
    sep=None,
    engine='python',
    header=None,
    names=['idx', 'sex', 'row_number', 'liters_drunk', 'age', 'drink', 'check_number', 'label']
)

# --- 1. Пол (sex) ---

# Нормализуем строку пола
sex_norm = titanic_data['sex'].astype(str).str.strip().str.lower()

# Заменяем явные "нет пола" на NaN
sex_norm = sex_norm.replace({
    '-', 'не указан', 'неуказан', 'не указано', 'nan', 'none', ''
}, np.nan)

# Маппинг в 0/1
sex_map = {
    'м': 1, 'муж': 1, 'мужчина': 1, 'мужской': 1,
    'm': 1, 'male': 1, 'man': 1,
    'ж': 0, 'жен': 0, 'женщина': 0, 'женский': 0,
    'f': 0, 'female': 0, 'woman': 0
}

titanic_data['sex'] = sex_norm.map(sex_map)

# Отфильтровываем строки, где пол не удалось распознать
titanic_data = titanic_data.dropna(subset=['sex'])

# Приводим к целочисленному типу 0/1
titanic_data['sex'] = titanic_data['sex'].astype(int)

# --- 2. Номер ряда (row_number) ---

titanic_data['row_number'] = pd.to_numeric(titanic_data['row_number'], errors='coerce')
max_row_number = titanic_data['row_number'].max(skipna=True)
titanic_data['row_number'] = titanic_data['row_number'].fillna(max_row_number).astype(int)

# --- 3. Количество выпитого (liters_drunk) ---


titanic_data['liters_drunk'] = pd.to_numeric(
    titanic_data['liters_drunk'],
    errors='coerce'
).astype(float)

liters_mean = titanic_data['liters_drunk'].mean()
liters_std = titanic_data['liters_drunk'].std()
upper_bound = liters_mean + 3 * liters_std

mask_bad = (titanic_data['liters_drunk'] < 0) | (titanic_data['liters_drunk'] > upper_bound)
titanic_data.loc[mask_bad, 'liters_drunk'] = liters_mean


# Сохраняем результат
titanic_data.to_csv('titanic_processed.csv', index=False)
