import pandas as pd
import numpy as np
from minisom import MiniSom
import pdfplumber
import docx
import os
import re

# --- CZĘŚĆ 1: NARZĘDZIA POMOCNICZE (Parsowanie kwot i tekstów) ---

def clean_currency(value):
    """Zamienia polskie formaty kwot (np. '1 200,50') na float."""
    if isinstance(value, (int, float)):
        return float(value)
    
    value = str(value).strip()
    # Usuń spacje (np. 1 000) i symbole walut
    value = re.sub(r'[^\d,-]', '', value)
    # Zamień przecinek na kropkę
    value = value.replace(',', '.')
    
    try:
        return float(value)
    except ValueError:
        return 0.0

def clean_paragraph(value):
    """Standaryzuje zapis paragrafu (tylko cyfry, jako string)."""
    return str(value).split('.')[0].strip()

# --- CZĘŚĆ 2: FUNKCJE WCZYTUJĄCE PLIKI ---

def load_from_excel(filepath):
    """Wczytuje dane z Excela. Zakładamy kolumny: Konto, Paragraf, Kwota."""
    try:
        df = pd.read_excel(filepath)
        # Dostosuj nazwy kolumn jeśli masz inne w plikach
        df.columns = [c.lower() for c in df.columns] 
        # Szukamy kolumn pasujących do nazw
        return df[['konto', 'paragraf', 'kwota']].to_dict('records')
    except Exception as e:
        print(f"Błąd Excel ({filepath}): {e}")
        return []

def load_from_word(filepath):
    """Wczytuje pierwszą tabelę z dokumentu Word."""
    data = []
    try:
        doc = docx.Document(filepath)
        for table in doc.tables:
            for i, row in enumerate(table.rows):
                if i == 0: continue # Pomijamy nagłówek
                cells = [c.text.strip() for c in row.cells]
                # Zakładamy strukturę tabeli w Wordzie: Kol 0=Konto, Kol 1=Paragraf, Kol 2=Kwota
                if len(cells) >= 3:
                    data.append({
                        'konto': cells[0],
                        'paragraf': cells[1],
                        'kwota': cells[2]
                    })
    except Exception as e:
        print(f"Błąd Word ({filepath}): {e}")
    return data

def load_from_pdf(filepath):
    """Wczytuje tabele z PDF używając pdfplumber."""
    data = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    for i, row in enumerate(table):
                        if i == 0: continue # Pomijamy nagłówek
                        # Filtrujemy puste wiersze i zakładamy strukturę jak w Wordzie
                        row = [x for x in row if x is not None]
                        if len(row) >= 3:
                            data.append({
                                'konto': row[0],
                                'paragraf': row[1],
                                'kwota': row[2]
                            })
    except Exception as e:
        print(f"Błąd PDF ({filepath}): {e}")
    return data

# --- CZĘŚĆ 3: AGREGACJA DANYCH ---

# Lista plików do przetworzenia (tutaj wpisz swoje ścieżki)
# Możesz użyć os.listdir(), żeby wczytać wszystko z folderu
files_to_process = [
    ('dane_excel.xlsx', 'excel'),
    # ('faktura.pdf', 'pdf'),   <-- odkomentuj gdy będziesz miał pliki
    # ('raport.docx', 'word')
]

raw_data = []

# Symulacja danych (gdyby nie było plików, żeby kod zadziałał od razu)
raw_data = [
    {'konto': 'Konto_A', 'paragraf': '4210', 'kwota': 5000},
    {'konto': 'Konto_A', 'paragraf': '4300', 'kwota': 100},
    {'konto': 'Konto_B', 'paragraf': '4300', 'kwota': 4000},
    {'konto': 'Konto_B', 'paragraf': '4210', 'kwota': 100},
    {'konto': 'Konto_C', 'paragraf': '4210', 'kwota': 4800}, # Podobne do A
    {'konto': 'Konto_D', 'paragraf': '4300', 'kwota': 4200}, # Podobne do B
]

# Pętla po plikach (jeśli istnieją)
for path, ftype in files_to_process:
    if not os.path.exists(path): continue
    
    if ftype == 'excel':
        raw_data.extend(load_from_excel(path))
    elif ftype == 'word':
        raw_data.extend(load_from_word(path))
    elif ftype == 'pdf':
        raw_data.extend(load_from_pdf(path))

# --- CZĘŚĆ 4: PRZETWARZANIE I PIVOT ---

df = pd.DataFrame(raw_data)

# Czyszczenie typów danych
df['kwota'] = df['kwota'].apply(clean_currency)
df['paragraf'] = df['paragraf'].apply(clean_paragraph)

# PIVOT TABLE: Zamiana z formatu "długiego" na macierz dla sieci
# Wiersze = Konta, Kolumny = Paragrafy, Wartości = Suma Kwot
pivot_df = df.pivot_table(index='konto', columns='paragraf', values='kwota', aggfunc='sum', fill_value=0)

print(f"Przetworzono {len(df)} operacji dla {len(pivot_df)} unikalnych kont.")
print("Macierz wejściowa do sieci (fragment):")
print(pivot_df.head())

# --- CZĘŚĆ 5: SIEĆ KOHONENA (SOM) ---

# Przygotowanie danych (Normalizacja Min-Max)
data_values = pivot_df.values
data_normalized = (data_values - data_values.min(axis=0)) / (data_values.max(axis=0) - data_values.min(axis=0) + 1e-10)

# Ustalenie rozmiaru mapy (zależne od liczby kont - heurystyka: 5*sqrt(N))
map_size = int(np.sqrt(5 * np.sqrt(len(pivot_df)))) + 2
som = MiniSom(map_size, map_size, data_values.shape[1], sigma=1.0, learning_rate=0.5)

som.random_weights_init(data_normalized)
som.train_random(data_normalized, num_iteration=1000)

# --- CZĘŚĆ 6: WYNIKI ---

grouped_accounts = {}
for account_name, row_data in zip(pivot_df.index, data_normalized):
    winner = som.winner(row_data)
    if winner not in grouped_accounts:
        grouped_accounts[winner] = []
    grouped_accounts[winner].append(account_name)

print("\n--- WYNIKI GRUPOWANIA KONT ---")
for coords, accounts in grouped_accounts.items():
    print(f"Grupa {coords}: {accounts}")
    
    # Prosta analiza grupy (co ją wyróżnia?)
    # Pobieramy średnie wydatki dla kont w tej grupie
    group_vectors = pivot_df.loc[accounts]
    top_para = group_vectors.mean().idxmax()
    print(f"   -> Dominujący paragraf w tej grupie: {top_para}")