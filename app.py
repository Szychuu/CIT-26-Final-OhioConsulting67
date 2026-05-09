import pandas as pd
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

# 1. Poprawne ładowanie danych
sales = pd.read_csv(os.path.join('assets', 'Sprzedaż_dzienna.csv'), sep=';')
events = pd.read_csv(os.path.join('assets', 'Eventy_lokalne.csv'), sep=';')
promos = pd.read_csv(os.path.join('assets', 'Promocje.csv'), sep=';')

# Zakładam, że w assets wygenerował się też arkusz z produktami:
products = pd.read_csv(os.path.join('assets', 'Produkty.csv'), sep=';')

# 2. Filtracja kategorii Fresh i wyłonienie Top 10 SKU
# Dołączamy kategorię do sprzedaży
sales_with_cat = sales.merge(products[['ID_SKU', 'Kategoria']], on='ID_SKU', how='left')

# Filtrujemy tylko 'Fresh'
fresh_sales = sales_with_cat[sales_with_cat['Kategoria'] == 'Fresh']

# Znajdujemy 10 najlepiej sprzedających się SKU (sumując całą historię)
top_10_sku = (
    fresh_sales.groupby('ID_SKU')['Sprzedaz_Sztuki'] # Podmień na właściwą nazwę kolumny ze sztukami
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

print(f"Wybrane Top 10 SKU do modelowania: {top_10_sku}")

# Zawężamy dane sprzedażowe tylko do tej złotej dziesiątki
df_top10 = fresh_sales[fresh_sales['ID_SKU'].isin(top_10_sku)]

# 3. Tworzenie cech (Złączenie eventów i promocji)
df = df_top10.merge(events, on='Data', how='left').merge(promos, on=['Data', 'ID_SKU'], how='left')
df['day_of_week'] = pd.to_datetime(df['Data']).dt.dayofweek

# Obejrzenie wynikowej tabeli
df.head()