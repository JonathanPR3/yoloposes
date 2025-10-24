"""
Configuración del sistema de detección de poses sospechosas
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURACIÓN DE CÁMARA
# ==========================================
# Fuente de video: 0 para webcam, o URL para cámara IP
# Ejemplos:
# - Webcam: 0
# - RTSP: "rtsp://usuario:password@192.168.1.100:554/stream"
# - HTTP: "http://192.168.1.100:8080/video"
CAMERA_SOURCE = 3
CAMERA_INDEX = CAMERA_SOURCE  # Alias para compatibilidad

# Resolución de procesamiento (reduce para mayor velocidad)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Procesar 1 de cada N frames (optimización)
PROCESS_EVERY_N_FRAMES = 1 # Procesar todos los frames

# ==========================================
# CONFIGURACIÓN DE YOLO
# ==========================================
# Modelo a usar (yolov8n-pose es el más rápido)
YOLO_MODEL = "yolov8n-pose.pt"  # Opciones: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

# Confianza mínima para detección de personas
CONFIDENCE_THRESHOLD = 0.5

# Confianza mínima de keypoints individuales
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5

# Mínimo de keypoints válidos requeridos para analizar pose
MIN_KEYPOINTS_REQUIRED = 10

# ==========================================
# CONFIGURACIÓN DEL SISTEMA ML
# ==========================================
# Cooldown entre alertas de la misma pose (segundos)
ALERT_COOLDOWN_SECONDS = 5
GLOBAL_COOLDOWN = ALERT_COOLDOWN_SECONDS  # Alias

# Mostrar video en tiempo real
DISPLAY_VIDEO = True

# Dibujar skeleton en el video
DRAW_SKELETON = True

# Habilitar notificaciones
ENABLE_NOTIFICATIONS = True

# ==========================================
# ZONAS DE DETECCIÓN (coordenadas normalizadas 0-1)
# ==========================================
# Estas coordenadas se ajustan según tu cámara
# Formato: (x_min, y_min, x_max, y_max)

# Zona donde está la puerta (ajustar según tu setup)
ZONA_PUERTA = {
    'x_min': 0.3,  # 30% desde la izquierda
    'y_min': 0.4,  # 40% desde arriba
    'x_max': 0.7,  # 70% desde la izquierda
    'y_max': 0.9,  # 90% desde arriba
}

# Zona donde está la barda/muro (ajustar según tu setup)
ZONA_BARDA = {
    'x_min': 0.0,
    'y_min': 0.0,
    'x_max': 1.0,
    'y_max': 0.4,  # Parte superior del frame (barda alta)
}

# Línea del suelo (Y normalizado, 0=arriba, 1=abajo)
LINEA_SUELO = 0.85

# ==========================================
# MODO DE OPERACIÓN
# ==========================================
# True = Usar zonas fijas (menos falsos positivos, requiere calibración)
# False = Detección generalizada (funciona en cualquier lugar, más falsos positivos)
USE_ZONES = False

# ==========================================
# UMBRALES DE DETECCIÓN - 5 POSES (para heurísticas legacy)
# ==========================================

# --- POSE 1: ESCALAMIENTO DE BARDA ---
ESCALAMIENTO = {
    'altura_minima_pies': 0.30,
    'extension_brazos_min': 140,
    'duracion_minima': 2.5,
    'cooldown': 30,
    'descripcion': 'Persona escalando barda/muro con brazos arriba y pies elevados'
}

# --- POSE 2: PATADA A PUERTA ---
PATADA = {
    'altura_minima_pie': 0.40,
    'extension_pierna_min': 140,
    'velocidad_minima': 60,
    'duracion_minima': 0.3,
    'cooldown': 15,
    'descripcion': 'Patada hacia puerta/ventana (intento de forzar entrada)'
}

# --- POSE 3: LANZAR OBJETO ---
LANZAR_OBJETO = {
    'extension_brazo_min': 150,
    'altura_mano_min': 0.3,
    'velocidad_mano_min': 100,
    'aceleracion_min': 50,
    'duracion_minima': 0.3,
    'cooldown': 15,
    'descripcion': 'Lanzamiento de objeto (piedra, botella) hacia propiedad'
}

# --- POSE 4: MIRAR A TRAVÉS DE VENTANA ---
MIRAR_VENTANA = {
    'altura_cabeza_min': 0.35,
    'altura_cabeza_max': 0.65,
    'inclinacion_cabeza_min': 15,
    'distancia_manos_cara': 150,
    'duracion_minima': 3.5,
    'cooldown': 30,
    'descripcion': 'Persona asomándose/mirando por ventana de forma sospechosa'
}

# --- POSE 5: FORZAR CERRADURA ---
FORZAR_CERRADURA = {
    'altura_maxima_cabeza': 0.65,
    'altura_minima_cabeza': 0.45,
    'inclinacion_torso_min': 25,
    'inclinacion_torso_max': 75,
    'manos_juntas': True,
    'distancia_maxima_manos': 100,
    'movimiento_manos_repetitivo': True,
    'duracion_minima': 3.0,
    'cooldown': 20,
    'descripcion': 'Persona agachada manipulando cerradura con ambas manos'
}

# ==========================================
# CONFIGURACIÓN DE NOTIFICACIONES
# ==========================================
API_URL = os.getenv('API_URL', 'http://localhost:8000/api/alerts')
API_TOKEN = os.getenv('API_TOKEN', '')
WEBHOOK_URL = API_URL  # Alias para compatibilidad

# ==========================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ==========================================
# Mostrar ventana de preview en tiempo real
SHOW_PREVIEW = True

# Mostrar información de debug en pantalla
SHOW_DEBUG_INFO = True

# Dibujar zonas de detección
DRAW_ZONES = False

# Colores (BGR format para OpenCV)
COLOR_NORMAL = (0, 255, 0)      # Verde
COLOR_SOSPECHOSO = (0, 165, 255)  # Naranja
COLOR_ALERTA = (0, 0, 255)      # Rojo
COLOR_ZONA = (255, 255, 0)      # Cyan

# ==========================================
# KEYPOINTS DE YOLO-POSE (17 puntos)
# ==========================================
# Índices de los keypoints detectados por YOLO
KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

# ==========================================
# LOGGING
# ==========================================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = 'detections.log'