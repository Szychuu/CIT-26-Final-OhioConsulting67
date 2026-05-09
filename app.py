import pandas as pd
import os

# 1. Konfiguracja ścieżek
folder_name = 'assets'
input_file_name = 'PLON_Market_dane.xlsx'  # Nazwa pliku z instrukcji 
input_path = os.path.join(folder_name, input_file_name)

# Upewniamy się, że folder assets istnieje, zanim zaczniemy
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Utworzono brakujący folder: {folder_name}")

try:
    # 2. Wczytujemy plik z folderu assets
    excel_data = pd.ExcelFile(input_path)
    sheet_names = excel_data.sheet_names
    print(f"Znaleziono {len(sheet_names)} arkuszy w {input_path}")

    # 3. Iteracja i zapis do assets
    for sheet in sheet_names:
        df = pd.read_excel(excel_data, sheet_name=sheet)
        
        # Przygotowanie nazwy pliku wyjściowego
        csv_name = f"{sheet.replace(' ', '_')}.csv"
        output_path = os.path.join(folder_name, csv_name)
        
        # Zapis do CSV w folderze assets
        df.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
        print(f"Wyeksportowano: {output_path}")

    print("\n✅ Wszystkie pliki zostały zapisane w folderze assets.")

except FileNotFoundError:
    print(f"❌ Błąd: Nie znaleziono pliku {input_file_name} w folderze {folder_name}!")