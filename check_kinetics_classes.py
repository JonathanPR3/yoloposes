"""
Verifica qué clases están disponibles en Kinetics-700
"""
import pandas as pd

print("Leyendo CSV...")
df_train = pd.read_csv('data/k700_train.csv')
df_val = pd.read_csv('data/k700_val.csv')
df = pd.concat([df_train, df_val])

# Detectar nombre de columna
label_col = 'label' if 'label' in df.columns else 'class'

print(f"\nColumnas disponibles: {df.columns.tolist()}\n")

# Todas las clases únicas
all_classes = sorted(df[label_col].unique())

print(f"Total de clases en Kinetics-700: {len(all_classes)}\n")

# Buscar clases relacionadas con lo que necesitamos
keywords = ['climb', 'kick', 'throw', 'peek', 'peep', 'window', 'look']

print("Clases que podrían servir:\n")

for keyword in keywords:
    matches = [c for c in all_classes if keyword.lower() in c.lower()]
    if matches:
        print(f"🔍 '{keyword}':")
        for match in matches:
            count = len(df[df[label_col] == match])
            print(f"   - {match:50s} ({count:,} videos)")
        print()

# Guardar todas las clases en un archivo
with open('data/kinetics_all_classes.txt', 'w', encoding='utf-8') as f:
    for cls in all_classes:
        count = len(df[df[label_col] == cls])
        f.write(f"{cls}\t{count}\n")

print(f"✅ Lista completa guardada en: data/kinetics_all_classes.txt")