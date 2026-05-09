import pandas as pd
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from itertools import product

print("⏳ Inicjalizacja systemu predykcyjnego PLON Market...")

# 1. Ładowanie i czyszczenie
def load_and_clean(filename):
    path = os.path.join('assets', filename)
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip() 
    return df

sales = load_and_clean('Sprzedaż_dzienna.csv')
events = load_and_clean('Eventy_lokalne.csv')
promos = load_and_clean('Promocje.csv')

sales['Data'] = pd.to_datetime(sales['Data'])
events['Data'] = pd.to_datetime(events['Data'])
promos['Data od'] = pd.to_datetime(promos['Data od'])
promos['Data do'] = pd.to_datetime(promos['Data do'])

# 2. Wybór kategorii Fresh i utworzenie słownika z nazwami
fresh_categories = ['Warzywa i owoce', 'Nabiał i jaja', 'Mięso', 'Pieczywo']
fresh_sales = sales[sales['Kategoria'].isin(fresh_categories)].copy()

# Zapisujemy mapowanie, by przywrócić nazwy w pliku wyjściowym
sku_meta = fresh_sales[['ID_SKU', 'Nazwa SKU', 'Kategoria']].drop_duplicates().set_index('ID_SKU')

# 3. Przetworzenie Promocji
promos_daily = []
for _, row in promos.iterrows():
    dates = pd.date_range(row['Data od'], row['Data do'])
    for d in dates:
        promos_daily.append({'Data': d, 'Kategoria': row['Kategoria'], 'Is_Promo': 1})
promos_df = pd.DataFrame(promos_daily).drop_duplicates()

# 4. Łączenie danych treningowych i inżynieria cech
train_df = fresh_sales.copy()
train_df = train_df.merge(events[['Data', 'Wpływ szacowany (1-5)']], on='Data', how='left')
train_df = train_df.merge(promos_df, on=['Data', 'Kategoria'], how='left')
train_df.fillna({'Is_Promo': 0, 'Wpływ szacowany (1-5)': 0}, inplace=True)

def add_features(df_feat):
    df_feat['day_of_week'] = df_feat['Data'].dt.dayofweek
    df_feat['month'] = df_feat['Data'].dt.month
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    # Rzutowanie na typ kategoryczny dla XGBoost
    df_feat['ID_SKU'] = df_feat['ID_SKU'].astype('category')
    df_feat['ID Sklepu'] = df_feat['ID Sklepu'].astype('category')
    return df_feat

train_df = add_features(train_df)
features = ['day_of_week', 'month', 'is_weekend', 'ID_SKU', 'ID Sklepu', 'Is_Promo', 'Wpływ szacowany (1-5)']
target = 'Sztuki sprzedane'

# Walidacja do obliczenia wskaźnika pewności (MAPE)
split_date = train_df['Data'].max() - pd.Timedelta(days=7)
train_mask = train_df['Data'] <= split_date
val_mask = train_df['Data'] > split_date

X_train, y_train = train_df[train_mask][features], train_df[train_mask][target]
X_val, y_val = train_df[val_mask][features], train_df[val_mask][target]

# 5. Trenowanie modelu
print("🚀 Trenowanie modelu ML...")
model = XGBRegressor(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=6, 
    random_state=42, 
    enable_categorical=True,
    objective='reg:absoluteerror' # <--- Wymuszenie optymalizacji MAE
)
model.fit(X_train, np.log1p(y_train))

# Wyliczenie błędu na potrzeby kolumny z pewnością predykcji
val_preds = np.expm1(model.predict(X_val))
val_preds = np.maximum(0, val_preds)
mape = mean_absolute_percentage_error(y_val, val_preds)
pewnosc_str = f"Średni błąd modelu z ostatnich 7 dni wynosi {mape:.1%}"

# Retrening na całości danych, aby wyciągnąć maksimum informacji do predykcji przyszłości
model.fit(train_df[features], train_df[target])

# 6. Generowanie siatki predykcyjnej (Target: Kolejny dzień)
next_day = train_df['Data'].max() + pd.Timedelta(days=1)
unique_stores = fresh_sales['ID Sklepu'].unique()
unique_skus = fresh_sales['ID_SKU'].unique()

# Tworzymy kombinacje dla wszystkich sklepów i wszystkich produktów świeżych
grid = list(product([next_day], unique_stores, unique_skus))
future_df = pd.DataFrame(grid, columns=['Data', 'ID Sklepu', 'ID_SKU'])

# Przygotowanie siatki pod model
future_df['Kategoria'] = future_df['ID_SKU'].map(sku_meta['Kategoria'])
future_df = future_df.merge(events[['Data', 'Wpływ szacowany (1-5)']], on='Data', how='left')
future_df = future_df.merge(promos_df, on=['Data', 'Kategoria'], how='left')
future_df.fillna({'Is_Promo': 0, 'Wpływ szacowany (1-5)': 0}, inplace=True)
future_df = add_features(future_df)

# 7. Predykcja 
future_df['Przewidywana_Sprzedaz'] = model.predict(future_df[features])
future_df['Przewidywana_Sprzedaz'] = future_df['Przewidywana_Sprzedaz'].apply(lambda x: max(0, int(round(x))))

# 8. Formatowanie wyjścia zgodnie z wymaganiami
future_df['Nazwa produktu'] = future_df['ID_SKU'].map(sku_meta['Nazwa SKU'])

def przypisz_jednostke(kat):
    if kat == 'Mięso': return 'kg'
    elif kat in ['Warzywa i owoce']: return 'kg/szt.'
    else: return 'szt.'

future_df['Jednostka'] = future_df['Kategoria'].apply(przypisz_jednostke)
future_df['Zalecana ilość do zamówienia'] = future_df['Przewidywana_Sprzedaz'].astype(str) + ' ' + future_df['Jednostka']
future_df['Pewność predykcji'] = pewnosc_str

# Wyciągnięcie dokładnie 10 najlepszych rekomendacji per Sklep
top_10_daily = (
    future_df.sort_values(['ID Sklepu', 'Przewidywana_Sprzedaz'], ascending=[True, False])
    .groupby('ID Sklepu')
    .head(10)
)

output_columns = ['Data', 'ID Sklepu', 'Nazwa produktu', 'Kategoria', 'Zalecana ilość do zamówienia', 'Pewność predykcji']
final_output = top_10_daily[output_columns].copy()

out_path = os.path.join('assets', 'rekomendacje_zamowien_fresh.csv')
final_output.to_csv(out_path, index=False, sep=';', encoding='utf-8-sig')

print(f"✅ Sukces! Plik wyjściowy (Top 10 rekomendacji per sklep) zapisany w: {out_path}")
print(f"📊 {pewnosc_str}")