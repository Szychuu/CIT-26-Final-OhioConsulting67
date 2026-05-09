import pandas as pd
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

print("⏳ Inicjalizacja systemu predykcyjnego PLON Market...")

# 1. Ładowanie i czyszczenie nazw kolumn (rozwiązuje błąd KeyError)
def load_and_clean(filename):
    path = os.path.join('assets', filename)
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip() # Usuwa ukryte spacje z nagłówków
    return df

sales = load_and_clean('Sprzedaż_dzienna.csv')
events = load_and_clean('Eventy_lokalne.csv')
promos = load_and_clean('Promocje.csv')

# Konwersja dat na format systemowy
sales['Data'] = pd.to_datetime(sales['Data'])
events['Data'] = pd.to_datetime(events['Data'])
promos['Data od'] = pd.to_datetime(promos['Data od'])
promos['Data do'] = pd.to_datetime(promos['Data do'])

# 2. Wybór Top 10 SKU z kategorii Fresh (zgodnie z zadaniem) [cite: 36, 51]
fresh_categories = ['Warzywa i owoce', 'Nabiał i jaja', 'Mięso', 'Pieczywo']
fresh_sales = sales[sales['Kategoria'].isin(fresh_categories)].copy()

top_10_sku = (
    fresh_sales.groupby('ID_SKU')['Sztuki sprzedane']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)
df = fresh_sales[fresh_sales['ID_SKU'].isin(top_10_sku)].copy()

# 3. Przetworzenie Promocji (rozbicie zakresów dat na pojedyncze dni)
promos_daily = []
for _, row in promos.iterrows():
    dates = pd.date_range(row['Data od'], row['Data do'])
    for d in dates:
        promos_daily.append({'Data': d, 'Kategoria': row['Kategoria'], 'Is_Promo': 1})
promos_df = pd.DataFrame(promos_daily).drop_duplicates()

# 4. Łączenie danych (Merging)
# Dołączamy eventy i promocje (po dacie i kategorii)
df = df.merge(events[['Data', 'Wpływ szacowany (1-5)']], on='Data', how='left')
df = df.merge(promos_df, on=['Data', 'Kategoria'], how='left')
df['Is_Promo'] = df['Is_Promo'].fillna(0)
df['Wpływ szacowany (1-5)'] = df['Wpływ szacowany (1-5)'].fillna(0)

# 5. Inżynieria Cech (Feature Engineering) [cite: 52]
df['day_of_week'] = df['Data'].dt.dayofweek
df['month'] = df['Data'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Kodowanie zmiennych kategorycznych
df_encoded = pd.get_dummies(df, columns=['ID_SKU', 'ID Sklepu'])

# Przygotowanie zbiorów X i y
target = 'Sztuki sprzedane'
# Usuwamy kolumny tekstowe i pomocnicze przed treningiem
drop_cols = ['Data', 'Nazwa SKU', 'Miasto', 'Kategoria', 'Cena jedn. (PLN)', 'Sprzedaż (PLN)', target]
X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
y = df_encoded[target]

# 6. Podział na Train/Test (ostatnie 7 dni danych historycznych) [cite: 53]
split_date = df['Data'].max() - pd.Timedelta(days=7)
X_train, y_train = X[df['Data'] <= split_date], y[df['Data'] <= split_date]
X_test, y_test = X[df['Data'] > split_date], y[df['Data'] > split_date]

# 7. Model ML (XGBoost)
print(f"🚀 Trenowanie modelu dla {len(top_10_sku)} produktów...")
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 8. Predykcja i zapis wyników 
preds = model.predict(X_test)
preds = [max(0, round(p)) for p in preds] # Wynik nie może być ujemny ani ułamkowy

# Tworzenie pliku wyjściowego zgodnego z wymogami
output = df[df['Data'] > split_date][['Data', 'ID_SKU', 'ID Sklepu']].copy()
output.rename(columns={'ID Sklepu': 'ID_Sklepu'}, inplace=True)
output['Prognoza_Sprzedazy'] = preds

out_path = os.path.join('assets', 'predykcje_fresh.csv')
output.to_csv(out_path, index=False, sep=';')

mape = mean_absolute_percentage_error(y_test, preds)
print(f"✅ Sukces! Plik zapisany w: {out_path}")
print(f"📊 Wynik modelu (MAPE): {mape:.2%}")