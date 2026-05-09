import pandas as pd
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from itertools import product

print("⏳ Inicjalizacja systemu predykcyjnego PLON Market...")

# 1. Ładowanie i czyszczenie plików CSV
def load_and_clean(filename):
    path = os.path.join('assets', filename) if os.path.exists(os.path.join('assets', filename)) else filename
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip() 
    return df

sales = load_and_clean('Sprzedaż_dzienna.csv')
events = load_and_clean('Eventy_lokalne.csv')
promos = load_and_clean('Promocje.csv')

# --- NOWE: Wczytywanie cennika ---
ceny_df = load_and_clean('Konkurencja_-_ceny.csv')
# Wyciągamy ostatnią (najnowszą) cenę PLON dla każdego produktu (Kolumna E)
ceny_aktualne = ceny_df.sort_values('Data początku tygodnia').groupby('SKU')['Cena PLON (PLN)'].last()
# ---------------------------------

sales['Data'] = pd.to_datetime(sales['Data'])
events['Data'] = pd.to_datetime(events['Data'])
promos['Data od'] = pd.to_datetime(promos['Data od'])
promos['Data do'] = pd.to_datetime(promos['Data do'])

# Naprawa ID Sklepu
sales['ID Sklepu'] = sales['ID Sklepu'].astype(str).str.replace('PLN-', '', regex=False).astype(int)

# 2. Wybór kategorii Fresh
fresh_categories = ['Warzywa i owoce', 'Nabiał i jaja', 'Mięso', 'Pieczywo']
fresh_sales = sales[sales['Kategoria'].isin(fresh_categories)].copy()

sku_meta = fresh_sales[['ID_SKU', 'Nazwa SKU', 'Kategoria']].drop_duplicates().set_index('ID_SKU')

# 3. Przetworzenie Promocji
promos_daily = []
for _, row in promos.iterrows():
    dates = pd.date_range(row['Data od'], row['Data do'])
    for d in dates:
        promos_daily.append({'Data': d, 'Kategoria': row['Kategoria'], 'Is_Promo': 1})
promos_df = pd.DataFrame(promos_daily).drop_duplicates()

# 4. Łączenie danych treningowych
train_df = fresh_sales.copy()
train_df = train_df.merge(events[['Data', 'Wpływ szacowany (1-5)']], on='Data', how='left')
train_df = train_df.merge(promos_df, on=['Data', 'Kategoria'], how='left')
train_df.fillna({'Is_Promo': 0, 'Wpływ szacowany (1-5)': 0}, inplace=True)

def add_features(df_feat):
    df_feat['day_of_week'] = df_feat['Data'].dt.dayofweek
    df_feat['month'] = df_feat['Data'].dt.month
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    df_feat['ID_SKU'] = df_feat['ID_SKU'].astype('category')
    df_feat['ID Sklepu'] = df_feat['ID Sklepu'].astype('category')
    return df_feat

train_df = add_features(train_df)
features = ['day_of_week', 'month', 'is_weekend', 'ID_SKU', 'ID Sklepu', 'Is_Promo', 'Wpływ szacowany (1-5)']
target = 'Sztuki sprzedane'

# Walidacja (ostatnie 7 dni)
split_date = train_df['Data'].max() - pd.Timedelta(days=7)
train_mask = train_df['Data'] <= split_date
val_mask = train_df['Data'] > split_date

X_train, y_train = train_df[train_mask][features], train_df[train_mask][target]
X_val, y_val = train_df[val_mask][features], train_df[val_mask][target]

# 5. Trenowanie modelu
print("🚀 Trenowanie modelu ML...")
model = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
    random_state=42, enable_categorical=True, objective='reg:absoluteerror'
)
model.fit(X_train, np.log1p(y_train))
val_preds = np.maximum(0, np.expm1(model.predict(X_val)))
mape = mean_absolute_percentage_error(y_val, val_preds)
pewnosc_str = f"Średni błąd modelu (MAPE) to {mape:.1%}"
model.fit(train_df[features], np.log1p(train_df[target]))

# 6. Generowanie siatki predykcyjnej (NAJBLIŻSZE 7 DNI)
ostatnia_data = train_df['Data'].max()
next_7_days = pd.date_range(ostatnia_data + pd.Timedelta(days=1), periods=7)

unique_stores = fresh_sales['ID Sklepu'].unique()
unique_skus = fresh_sales['ID_SKU'].unique()

grid = list(product(next_7_days, unique_stores, unique_skus))
future_df = pd.DataFrame(grid, columns=['Data', 'ID Sklepu', 'ID_SKU'])

future_df['Kategoria'] = future_df['ID_SKU'].map(sku_meta['Kategoria'])
future_df = future_df.merge(events[['Data', 'Wpływ szacowany (1-5)']], on='Data', how='left')
future_df = future_df.merge(promos_df, on=['Data', 'Kategoria'], how='left')
future_df.fillna({'Is_Promo': 0, 'Wpływ szacowany (1-5)': 0}, inplace=True)
future_df = add_features(future_df)

# 7. Predykcja ilościowa
future_preds_log = model.predict(future_df[features])
future_df['Przewidywana_Sprzedaz'] = np.expm1(future_preds_log)
future_df['Przewidywana_Sprzedaz'] = future_df['Przewidywana_Sprzedaz'].apply(lambda x: max(0, int(np.ceil(x))))

# 8. --- ZMIANA: Kalkulacja Łącznego Kosztu z IMPUTACJĄ (Uzupełnianiem) DANYCH ---
future_df['Nazwa produktu'] = future_df['ID_SKU'].map(sku_meta['Nazwa SKU'])

name_mapping = {
    'Jabłka 1kg': 'Jabłka Ligol 1kg',
    'Masło 200g': 'Masło ekstra 200g',
    'Chleb pszenny 500g': 'Chleb wiejski 500g'
}

future_df['Nazwa do cennika'] = future_df['Nazwa produktu'].astype(str).replace(name_mapping)
future_df['Cena jednostkowa'] = future_df['Nazwa do cennika'].map(ceny_aktualne)

# NOWE: Zastępowanie braków w cenniku
# Wyliczamy średnią cenę dla danej kategorii wprost z dostępnych na ten moment danych
srednie_kategorii = future_df.groupby('Kategoria')['Cena jednostkowa'].transform('mean')

# Wypełniamy "puste" miejsca (NaN) średnią ceną z ich kategorii. 
# Gdyby cała kategoria nie miała ani jednej ceny, rezerwowo wpisujemy np. 6.00 PLN.
future_df['Cena jednostkowa'] = future_df['Cena jednostkowa'].fillna(srednie_kategorii).fillna(6.0)

# Matematyka: Ilość * Cena (zera już tu nie wystąpią!)
future_df['Łączny koszt (PLN)'] = (future_df['Przewidywana_Sprzedaz'] * future_df['Cena jednostkowa']).round(2)

def przypisz_jednostke(kat):
    if kat == 'Mięso': return 'kg'
    elif kat in ['Warzywa i owoce']: return 'kg/szt.'
    else: return 'szt.'

future_df['Jednostka'] = future_df['Kategoria'].apply(przypisz_jednostke)
future_df['Zalecana ilość do zamówienia'] = future_df['Przewidywana_Sprzedaz'].astype(str) + ' ' + future_df['Jednostka']
future_df['Pewność predykcji'] = pewnosc_str

# 9. --- ZMIANA: Zostawiamy cały asortyment i filtrujemy WYBRANY SKLEP ---

WYBRANY_SKLEP = 20

# Usunęliśmy filtr wykluczający, bo poradziły z nim sobie średnie kategorie!
future_df_filtered = future_df[future_df['ID Sklepu'] == WYBRANY_SKLEP].copy()

# Sortujemy rosnąco (najtańsze koszty zamówienia na górze)
filtered_df = (
    future_df_filtered.sort_values(['Data', 'Łączny koszt (PLN)'], ascending=[True, True])
    .groupby('Data')
    .head(10) # Zawsze wyłapie 10 produktów, bo znów ma ich w puli aż 13
)

# Wybór ostatecznych kolumn
output_columns = ['Data', 'ID Sklepu', 'Nazwa produktu', 'Kategoria', 'Zalecana ilość do zamówienia', 'Łączny koszt (PLN)']
final_output = filtered_df[output_columns].copy()

# Upiększenie kosztu o walutę
final_output['Łączny koszt (PLN)'] = final_output['Łączny koszt (PLN)'].astype(str) + ' PLN'

out_path = os.path.join('assets', f'rekomendacje_zamowien_sklep_{WYBRANY_SKLEP}.csv') if os.path.exists('assets') else f'rekomendacje_zamowien_sklep_{WYBRANY_SKLEP}.csv'
final_output.to_csv(out_path, index=False, sep=';', encoding='utf-8-sig')

print(f"✅ Sukces! Plik CSV wygenerowany dla sklepu {WYBRANY_SKLEP}. Braki uzupełniono średnimi, Top 10 jest pełne.")