import pandas as pd

wells = pd.read_csv('wells_info.csv')
wells['SpudDate'] = pd.to_datetime(wells['SpudDate'], errors='coerce')
wells['CompletionDate'] = pd.to_datetime(wells['CompletionDate'], errors='coerce')

def full_months_between(start, end):
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    s = start.date()
    e = end.date()
    months = (e.year - s.year) * 12 + (e.month - s.month)
    if e.day < s.day:
        months -= 1
    return max(0, months)

wells['months_duration'] = wells.apply(lambda r: full_months_between(r['SpudDate'], r['CompletionDate']), axis=1)
print(wells[['API','operatorNameIHS','months_duration']].head())
