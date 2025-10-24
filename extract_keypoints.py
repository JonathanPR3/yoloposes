"""
PASO 2: Extraer keypoints de los videos de Kinetics usando YOLO
"""
import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

print("="*70)
print("PASO 2: EXTRACCIÓN DE KEYPOINTS")
print("="*70)

# ==========================================
# CONFIGURACIÓN
# ==========================================

VIDEO_DIR = "data/kinetics_videos"
OUTPUT_FILE = "data/pose_dataset_kinetics.json"

# Mapeo de clases
KINETICS_TO_POSES = {
    'escalamiento': 'escalamiento',
    'patada': 'patada',
    'lanzar': 'lanzar',
    'mirar_ventana': 'mirar_ventana'
}

print(f"\n📋 Configuración:")
print(f"   Carpeta de videos: {VIDEO_DIR}")
print(f"   Archivo de salida: {OUTPUT_FILE}")

# Verificar que exista la carpeta de videos
if not os.path.exists(VIDEO_DIR):
    print(f"\n❌ Error: No se encontró la carpeta {VIDEO_DIR}")
    print(f"   Ejecuta primero: python 1_download_kinetics.py")
    exit(1)

# ==========================================
# CARGAR MODELO YOLO
# ==========================================

print(f"\n🤖 Cargando modelo YOLO...")
try:
    model = YOLO('yolov8n-pose.pt')
    print(f"   ✅ Modelo YOLO cargado correctamente")
except Exception as e:
    print(f"   ❌ Error cargando modelo: {e}")
    exit(1)

# ==========================================
# FUNCIÓN DE EXTRACCIÓN
# ==========================================

def extract_keypoints_from_video(video_path, kinetics_class):
    """
    Extrae keypoints de un video.
    
    Returns:
        Lista de samples con keypoints
    """
    our_class = KINETICS_TO_POSES.get(kinetics_class)
    if not our_class:
        return []
    
    cap = cv2.VideoCapture(video_path)
    
    # Metadata del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    samples = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Procesar cada 3 frames (optimización)
        if frame_idx % 3 != 0:
            continue
        
        # Detectar keypoints con YOLO
        try:
            results = model(frame, conf=0.5, verbose=False)
        except:
            continue
        
        for result in results:
            if result.keypoints is None or len(result.keypoints.data) == 0:
                continue
            
            # Procesar cada persona detectada
            for person_idx, kp_data in enumerate(result.keypoints.data):
                kp = kp_data.cpu().numpy()
                
                # Calcular calidad del sample
                valid_kp = np.sum(kp[:, 2] > 0.3)  # Keypoints con confianza > 0.3
                avg_conf = np.mean(kp[:, 2])
                
                # Filtrar samples de baja calidad
                if valid_kp < 12:  # Mínimo 12 keypoints válidos de 17
                    continue
                
                if avg_conf < 0.4:  # Confianza promedio mínima
                    continue
                
                # Crear sample
                sample = {
                    'keypoints': kp.tolist(),
                    'label': our_class,
                    'source': 'kinetics',
                    'kinetics_class': kinetics_class,
                    'video': os.path.basename(video_path),
                    'frame': frame_idx,
                    'person_idx': person_idx,
                    
                    # Metadata de calidad
                    'quality': {
                        'valid_keypoints': int(valid_kp),
                        'avg_confidence': float(avg_conf),
                        'video_width': width,
                        'video_height': height,
                        'video_fps': float(fps) if fps > 0 else 30.0
                    },
                    
                    # Peso para entrenamiento (Kinetics tiene ángulos variables)
                    'weight': 0.7
                }
                
                samples.append(sample)
    
    cap.release()
    return samples

# ==========================================
# PROCESAR TODOS LOS VIDEOS
# ==========================================

print(f"\n" + "="*70)
print("🎬 PROCESANDO VIDEOS")
print("="*70)
print("⏱️  Esto tardará 20-40 minutos...\n")

all_samples = []
stats = {}

for kinetics_class in KINETICS_TO_POSES.keys():
    class_dir = os.path.join(VIDEO_DIR, kinetics_class)
    
    if not os.path.exists(class_dir):
        print(f"⚠️  Carpeta no encontrada: {class_dir}")
        continue
    
    # Listar videos
    videos = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
    
    if len(videos) == 0:
        print(f"⚠️  No hay videos en: {class_dir}")
        continue
    
    our_class = KINETICS_TO_POSES[kinetics_class]
    
    print(f"📹 Procesando: {kinetics_class} → {our_class}")
    print(f"   Videos encontrados: {len(videos)}")
    
    class_samples = []
    
    for video in tqdm(videos, desc=f"   {kinetics_class:20s}", ncols=80):
        video_path = os.path.join(class_dir, video)
        
        try:
            samples = extract_keypoints_from_video(video_path, kinetics_class)
            class_samples.extend(samples)
        except Exception as e:
            # Continuar con el siguiente video si hay error
            continue
    
    all_samples.extend(class_samples)
    stats[our_class] = len(class_samples)
    
    print(f"   ✅ {len(class_samples):,} samples extraídos\n")

# ==========================================
# GUARDAR DATASET
# ==========================================

print(f"💾 Guardando dataset...")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_samples, f, indent=2)

file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

print(f"   ✅ Dataset guardado: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")

# ==========================================
# ESTADÍSTICAS FINALES
# ==========================================

print(f"\n" + "="*70)
print("✅ EXTRACCIÓN COMPLETADA")
print("="*70)

print(f"\n📊 Estadísticas:")
print(f"   Total de samples: {len(all_samples):,}")

print(f"\n📂 Distribución por clase:")
for pose in sorted(stats.keys()):
    count = stats[pose]
    percentage = (count / len(all_samples)) * 100 if len(all_samples) > 0 else 0
    print(f"   {pose:20s}: {count:5,} samples ({percentage:5.1f}%)")

# Calidad promedio
if len(all_samples) > 0:
    avg_valid_kp = np.mean([s['quality']['valid_keypoints'] for s in all_samples])
    avg_confidence = np.mean([s['quality']['avg_confidence'] for s in all_samples])
    
    print(f"\n📈 Calidad promedio:")
    print(f"   Keypoints válidos: {avg_valid_kp:.1f} / 17")
    print(f"   Confianza: {avg_confidence:.1%}")

if len(all_samples) > 0:
    print(f"\n🎯 SIGUIENTE PASO:")
    print(f"   python 3_train_classifier.py")
else:
    print(f"\n❌ No se extrajo ningún sample.")
    print(f"   Verifica que los videos se hayan descargado correctamente.")