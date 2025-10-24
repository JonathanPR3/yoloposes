"""
PASO 1: Descargar videos de Kinetics-700
VERSIÓN FINAL - Con clases exactas confirmadas
"""
import os
import pandas as pd
import subprocess
from tqdm import tqdm
import urllib.request

print("="*70)
print("PASO 1: DESCARGA DE KINETICS-700")
print("="*70)

# ==========================================
# CONFIGURACIÓN - CLASES EXACTAS
# ==========================================

# Nombres EXACTOS según el CSV de Kinetics-700
KINETICS_TO_POSES = {
    # Escalamiento
    'climbing ladder': 'escalamiento',
    'climbing tree': 'escalamiento',
    'rock climbing': 'escalamiento',
    
    # Patada
    'side kick': 'patada',
    'high kick': 'patada',

    
    # Lanzar
    'throwing ball (not baseball or American football)': 'lanzar',
    'throwing knife': 'lanzar',
    'throwing axe': 'lanzar',
    
    # Mirar ventana (clase más similar disponible)
    'looking in mirror': 'mirar_ventana'
}

MAX_VIDEOS_PER_POSE = 50
OUTPUT_DIR = "data/kinetics_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n📋 Configuración:")
print(f"   Clases de Kinetics: {len(KINETICS_TO_POSES)}")
print(f"   Videos por pose: {MAX_VIDEOS_PER_POSE}")
print(f"   Carpeta salida: {OUTPUT_DIR}\n")

# ==========================================
# PASO 1.1: Verificar archivos CSV
# ==========================================

train_csv = "data/k700_train.csv"
val_csv = "data/k700_val.csv"

if not os.path.exists(train_csv) or not os.path.exists(val_csv):
    print("❌ Error: No se encontraron los archivos CSV")
    print("   Descargándolos ahora...")
    
    TRAIN_URL = "https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv"
    VAL_URL = "https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv"
    
    os.makedirs("data", exist_ok=True)
    
    print("   Descargando train.csv...")
    urllib.request.urlretrieve(TRAIN_URL, train_csv)
    
    print("   Descargando val.csv...")
    urllib.request.urlretrieve(VAL_URL, val_csv)
    
    print("   ✅ CSVs descargados\n")

# ==========================================
# PASO 1.2: Leer y filtrar
# ==========================================

print("📊 Leyendo dataset...")

df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)
df = pd.concat([df_train, df_val], ignore_index=True)

print(f"   Total de videos: {len(df):,}")

# Detectar columna de clase
label_col = 'label' if 'label' in df.columns else 'class'

# Filtrar solo las clases que necesitamos
df_filtered = df[df[label_col].isin(KINETICS_TO_POSES.keys())]

print(f"   Videos de clases objetivo: {len(df_filtered):,}\n")

# Mostrar cuántos videos hay por clase
print("🔍 Videos disponibles por clase:")
for kinetics_class, our_pose in KINETICS_TO_POSES.items():
    count = len(df[df[label_col] == kinetics_class])
    if count > 0:
        print(f"   {kinetics_class:50s} → {our_pose:20s}: {count:4,} videos")

# Limitar por pose (no por clase individual)
print(f"\n🎯 Limitando a {MAX_VIDEOS_PER_POSE} videos por pose...")

df_final = pd.DataFrame()
pose_counts = {}

for kinetics_class, our_pose in KINETICS_TO_POSES.items():
    # Videos de esta clase de Kinetics
    class_videos = df[df[label_col] == kinetics_class]
    
    if len(class_videos) == 0:
        continue
    
    # Calcular cuántos videos ya tenemos de esta pose
    current_count = pose_counts.get(our_pose, 0)
    
    # Cuántos más necesitamos
    needed = MAX_VIDEOS_PER_POSE - current_count
    
    if needed <= 0:
        continue
    
    # Tomar los que necesitamos
    to_add = class_videos.head(needed)
    
    df_final = pd.concat([df_final, to_add], ignore_index=True)
    pose_counts[our_pose] = current_count + len(to_add)

print(f"\n📊 Videos seleccionados por pose:")
for pose, count in sorted(pose_counts.items()):
    print(f"   {pose:20s}: {count:3d} videos")

print(f"\n   Total a descargar: {len(df_final)} videos")

if len(df_final) == 0:
    print("\n❌ No hay videos para descargar")
    exit(1)

# ==========================================
# PASO 1.3: Verificar yt-dlp
# ==========================================

print(f"\n🔧 Verificando yt-dlp...")

try:
    result = subprocess.run(['yt-dlp', '--version'], 
                          capture_output=True, text=True, 
                          check=True, timeout=5)
    print(f"   ✅ yt-dlp instalado: {result.stdout.strip()}")
except:
    print(f"   📦 Instalando yt-dlp...")
    subprocess.run(['pip', 'install', '-U', 'yt-dlp'], check=True)
    print(f"   ✅ yt-dlp instalado")

# ==========================================
# FUNCIÓN DE DESCARGA
# ==========================================

def download_video(row, kinetics_class, output_dir):
    """Descarga un video de YouTube."""
    our_pose = KINETICS_TO_POSES[kinetics_class]
    
    # Carpeta por pose
    pose_dir = os.path.join(output_dir, our_pose)
    os.makedirs(pose_dir, exist_ok=True)
    
    youtube_id = row['youtube_id']
    time_start = int(row['time_start'])
    time_end = int(row['time_end'])
    
    output_file = os.path.join(pose_dir, f"{youtube_id}_{time_start}-{time_end}.mp4")
    
    # Si ya existe, skip
    if os.path.exists(output_file) and os.path.getsize(output_file) > 50000:
        return True
    
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    
    cmd = [
        'yt-dlp',
        '-f', 'worst[height>=360]',
        '--download-sections', f'*{time_start}-{time_end}',
        '-o', output_file,
        '--no-playlist',
        '--quiet',
        '--no-warnings',
        url
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 50000:
            return True
        else:
            if os.path.exists(output_file):
                os.remove(output_file)
            return False
    except:
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass
        return False

# ==========================================
# DESCARGAR
# ==========================================

print(f"\n" + "="*70)
print("📹 DESCARGANDO VIDEOS DESDE YOUTUBE")
print("="*70)
print("⏱️  Esto tardará 30-60 minutos...")
print("💡 Puedes cancelar con Ctrl+C y continuar después\n")

stats = {pose: {'success': 0, 'failed': 0} for pose in set(KINETICS_TO_POSES.values())}

for idx, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Descargando", ncols=80):
    kinetics_class = row[label_col]
    our_pose = KINETICS_TO_POSES[kinetics_class]
    
    success = download_video(row, kinetics_class, OUTPUT_DIR)
    
    if success:
        stats[our_pose]['success'] += 1
    else:
        stats[our_pose]['failed'] += 1

# ==========================================
# RESUMEN
# ==========================================

print(f"\n" + "="*70)
print("✅ DESCARGA COMPLETADA")
print("="*70)

total_success = sum(s['success'] for s in stats.values())
total_failed = sum(s['failed'] for s in stats.values())

print(f"\n📊 Resumen:")
print(f"   Videos descargados: {total_success}")
print(f"   Videos fallidos: {total_failed}")

if (total_success + total_failed) > 0:
    print(f"   Tasa de éxito: {total_success/(total_success+total_failed)*100:.1f}%")

print(f"\n📂 Por pose:")
total_size_mb = 0

for pose in sorted(stats.keys()):
    stat = stats[pose]
    pose_dir = os.path.join(OUTPUT_DIR, pose)
    
    if os.path.exists(pose_dir):
        videos = [f for f in os.listdir(pose_dir) if f.endswith('.mp4')]
        size_mb = sum(os.path.getsize(os.path.join(pose_dir, f)) 
                     for f in videos) / (1024*1024)
        total_size_mb += size_mb
        
        print(f"   {pose:20s}: {len(videos):3d} videos ({size_mb:6.1f} MB)")
    else:
        print(f"   {pose:20s}: {stat['success']:3d} OK, {stat['failed']:3d} fallidos")

print(f"\n💾 Espacio total: {total_size_mb:.1f} MB")

if total_success > 0:
    print(f"\n🎯 SIGUIENTE PASO:")
    print(f"   python 2_extract_keypoints.py")
else:
    print(f"\n❌ No se descargó ningún video.")
    print(f"   Algunos videos pueden no estar disponibles en YouTube.")