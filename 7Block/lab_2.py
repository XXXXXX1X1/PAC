import pandas as pd
import numpy as np

# ============================================================
#  ЛАБОРАТОРНАЯ РАБОТА 7.1
#  ОЧИСТКА И ПРЕДОБРАБОТКА ДАННЫХ TITANIC
# ============================================================

# === 1. Загрузка необработанного Titanic ===
titanic = pd.read_csv(
    'titanic_with_labels.csv',
    sep=None,
    engine='python',
    header=None,
    names=[
        'idx', 'sex', 'row_number', 'liters_drunk',
        'age', 'drink', 'check_number', 'label'
    ]
)

# удаляем мусорные строки, например те где label = 'label'
titanic = titanic[titanic['label'] != 'label'].copy()

# === 2. Пол (sex): фильтрация и бинаризация ===

sex_norm = titanic['sex'].astype(str).str.strip().str.lower()

sex_norm = sex_norm.replace({
    '-', 'не указан', 'неуказан', 'не указано',
    'nan', 'none', ''
}, np.nan)

sex_map = {
    'м': 1, 'муж': 1, 'мужчина': 1, 'мужской': 1,
    'm': 1, 'male': 1, 'man': 1,
    'ж': 0, 'жен': 0, 'женщина': 0, 'женский': 0,
    'f': 0, 'female': 0, 'woman': 0
}

titanic['sex'] = sex_norm.map(sex_map)
titanic = titanic.dropna(subset=['sex'])
titanic['sex'] = titanic['sex'].astype(int)

# === 3. row_number: NAN → максимальный ряд ===

titanic['row_number'] = pd.to_numeric(titanic['row_number'], errors='coerce')
max_row = titanic['row_number'].max(skipna=True)
titanic['row_number'] = titanic['row_number'].fillna(max_row).astype(int)

# === 4. liters_drunk: отрицательные и выбросы → среднее ===

titanic['liters_drunk'] = pd.to_numeric(titanic['liters_drunk'], errors='coerce').astype(float)

mean_lit = titanic['liters_drunk'].mean()
std_lit = titanic['liters_drunk'].std()
upper = mean_lit + 3 * std_lit

mask_bad = (titanic['liters_drunk'] < 0) | (titanic['liters_drunk'] > upper)
titanic.loc[mask_bad, 'liters_drunk'] = mean_lit

# === Сохранение результата 7.1 ===
titanic.to_csv('titanic_processed.csv', index=False)
print('7.1: сохранено в titanic_processed.csv, строк:', len(titanic))


# ============================================================
#  ЛАБОРАТОРНАЯ РАБОТА 7.2
# ============================================================

titanic = pd.read_csv('titanic_processed.csv')

# === 5. age → age_child / age_adult / age_senior ===

titanic['age'] = pd.to_numeric(titanic['age'], errors='coerce')

titanic['age_child']  = (titanic['age'] < 18).astype(int)
titanic['age_adult']  = ((titanic['age'] >= 18) & (titanic['age'] <= 50)).astype(int)
titanic['age_senior'] = (titanic['age'] > 50).astype(int)

# удаляем старый age
titanic = titanic.drop(columns=['age'])

# === 6. drink → 0/1 ===

alcoholic = {'Beerbeer', 'Bugbeer', 'Strong beer'}

titanic['drink'] = (
    titanic['drink'].astype(str).str.strip()
    .apply(lambda x: 1 if x in alcoholic else 0)
)

# === 7. check_number → число ===
titanic['check_number'] = pd.to_numeric(titanic['check_number'], errors='coerce')

# === 8. Чтение cinema_sessions ===

cinema = pd.read_csv(
    'cinema_sessions.csv',
    sep=r'\s+',
    header=None,
    names=['idx', 'check_number', 'session_start'],
    engine='python',
    dtype=str
)

cinema = cinema.drop(columns=['idx'])

cinema['check_number'] = pd.to_numeric(cinema['check_number'], errors='coerce')
cinema['session_start'] = cinema['session_start'].astype(str).str.strip()

# убираем мусорные строки
cinema = cinema[
    cinema['check_number'].notna() &
    (cinema['session_start'].str.lower() != 'session_start')
]

# === 9. Merge ===

merged = pd.merge(titanic, cinema, on='check_number', how='left')

# === 10. session_start → morning/day/evening ===

merged['session_dt'] = pd.to_datetime(
    merged['session_start'],
    format='%H:%M:%S.%f',
    errors='coerce'
)

merged['session_hour'] = merged['session_dt'].dt.hour

merged['morning'] = ((merged['session_hour'] >= 6)  & (merged['session_hour'] < 12)).astype(int)
merged['day']     = ((merged['session_hour'] >= 12) & (merged['session_hour'] < 18)).astype(int)
merged['evening'] = ((merged['session_hour'] >= 18) & (merged['session_hour'] < 24)).astype(int)

merged = merged.drop(columns=['session_dt'])

# === Сохранение результата 7.2 ===

merged.to_csv('titanic_final_processed.csv', index=False)
print('7.2: сохранено в titanic_final_processed.csv, строк:', len(merged))
