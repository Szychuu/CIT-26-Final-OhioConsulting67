import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

# 1. Ładowanie danych
sales = pd.read_csv('Sprzedaż dzienna.csv')
events = pd.read_csv('Eventy lokalne.csv')
promos = pd.read_csv('Promocje.csv')

# 2. Tworzenie cech (Feature Engineering)
df = sales.merge(events, on='Data', how='left').merge(promos, on=['Data', 'ID_SKU'], how='left')
df['day_of_week'] = pd.to_datetime(df['Data']).dt.dayofweek

# 3. Modelowanie - trening na danych historycznych, test na ostatnich 7 dniach
train = df[df['Data'] < '2026-05-01']
test = df[df['Data'] >= '2026-05-01']

model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(train.drop(columns=['Prognoza_Sprzedazy', 'Data']), train['Prognoza_Sprzedazy'])

# 4. Ewaluacja (Metryka MAPE)
predictions = model.predict(test.drop(columns=['Prognoza_Sprzedazy', 'Data']))
mape = mean_absolute_percentage_error(test['Prognoza_Sprzedazy'], predictions)
print(f"Wynik MAPE: {mape}")