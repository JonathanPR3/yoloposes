"""
Funciones auxiliares para procesamiento de keypoints
"""
import numpy as np
import math
from typing import Tuple, Optional, List

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos.
    
    Args:
        point1: (x, y) del primer punto
        point2: (x, y) del segundo punto
    
    Returns:
        Distancia entre los puntos
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def calculate_angle(point1: Tuple[float, float], 
                    point2: Tuple[float, float], 
                    point3: Tuple[float, float]) -> float:
    """
    Calcula el ángulo formado por tres puntos (point2 es el vértice).
    
    Args:
        point1: Primer punto de la línea
        point2: Vértice (punto central)
        point3: Tercer punto de la línea
    
    Returns:
        Ángulo en grados (0-180)
    """
    # Vectores
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])
    
    # Producto punto y magnitudes
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    
    # Evitar división por cero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    # Calcular ángulo
    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = max(-1, min(1, cos_angle))  # Limitar al rango [-1, 1]
    
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def is_point_in_zone(point: Tuple[float, float], 
                     zone: dict,
                     frame_width: int,
                     frame_height: int) -> bool:
    """
    Verifica si un punto está dentro de una zona.
    
    Args:
        point: (x, y) en coordenadas de píxeles
        zone: Diccionario con x_min, y_min, x_max, y_max (normalizados 0-1)
        frame_width: Ancho del frame en píxeles
        frame_height: Alto del frame en píxeles
    
    Returns:
        True si el punto está dentro de la zona
    """
    # Normalizar coordenadas del punto
    x_norm = point[0] / frame_width
    y_norm = point[1] / frame_height
    
    return (zone['x_min'] <= x_norm <= zone['x_max'] and 
            zone['y_min'] <= y_norm <= zone['y_max'])


def get_keypoint(keypoints: np.ndarray, index: int) -> Optional[Tuple[float, float, float]]:
    """
    Obtiene un keypoint específico con validación de confianza.
    
    Args:
        keypoints: Array de keypoints de YOLO (17 x 3: x, y, confidence)
        index: Índice del keypoint deseado
    
    Returns:
        Tupla (x, y, confidence) o None si no es válido
    """
    if index >= len(keypoints):
        return None
    
    kp = keypoints[index]
    x, y, conf = kp[0], kp[1], kp[2]
    
    # Validar que tenga confianza suficiente y coordenadas válidas
    if conf < 0.3 or x == 0 or y == 0:
        return None
    
    return (float(x), float(y), float(conf))


def calculate_torso_angle(keypoints: np.ndarray, keypoint_indices: dict) -> Optional[float]:
    """
    Calcula el ángulo de inclinación del torso (vertical = 0°, horizontal = 90°).
    
    Args:
        keypoints: Array de keypoints
        keypoint_indices: Diccionario con índices de keypoints
    
    Returns:
        Ángulo del torso en grados o None si no se puede calcular
    """
    # Obtener hombro y cadera promedio
    left_shoulder = get_keypoint(keypoints, keypoint_indices['left_shoulder'])
    right_shoulder = get_keypoint(keypoints, keypoint_indices['right_shoulder'])
    left_hip = get_keypoint(keypoints, keypoint_indices['left_hip'])
    right_hip = get_keypoint(keypoints, keypoint_indices['right_hip'])
    
    if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
        return None
    
    # Punto medio de hombros y caderas
    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2)
    hip_mid = ((left_hip[0] + right_hip[0]) / 2,
               (left_hip[1] + right_hip[1]) / 2)
    
    # Calcular ángulo respecto a la vertical
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    
    # Ángulo con respecto al eje Y (vertical)
    angle = abs(math.degrees(math.atan2(dx, dy)))
    
    return angle


def calculate_limb_extension(keypoints: np.ndarray, 
                             joint1_idx: int,
                             joint2_idx: int, 
                             joint3_idx: int) -> Optional[float]:
    """
    Calcula el ángulo de extensión de una extremidad (brazo o pierna).
    
    Args:
        keypoints: Array de keypoints
        joint1_idx: Índice de la articulación proximal (hombro/cadera)
        joint2_idx: Índice de la articulación media (codo/rodilla)
        joint3_idx: Índice de la articulación distal (muñeca/tobillo)
    
    Returns:
        Ángulo de extensión o None si no se puede calcular
    """
    joint1 = get_keypoint(keypoints, joint1_idx)
    joint2 = get_keypoint(keypoints, joint2_idx)
    joint3 = get_keypoint(keypoints, joint3_idx)
    
    if not all([joint1, joint2, joint3]):
        return None
    
    point1 = (joint1[0], joint1[1])
    point2 = (joint2[0], joint2[1])
    point3 = (joint3[0], joint3[1])
    
    return calculate_angle(point1, point2, point3)


def get_body_height_ratio(keypoints: np.ndarray, 
                          keypoint_indices: dict,
                          frame_height: int) -> Optional[float]:
    """
    Calcula la altura del cuerpo normalizada (0-1) dentro del frame.
    
    Args:
        keypoints: Array de keypoints
        keypoint_indices: Diccionario con índices
        frame_height: Alto del frame
    
    Returns:
        Ratio de altura (0 = parte superior, 1 = parte inferior) o None
    """
    # Usar la nariz como punto más alto
    nose = get_keypoint(keypoints, keypoint_indices['nose'])
    
    if not nose:
        return None
    
    return nose[1] / frame_height


def get_foot_position(keypoints: np.ndarray, 
                     keypoint_indices: dict,
                     frame_height: int) -> Optional[Tuple[float, float]]:
    """
    Obtiene la posición promedio de los pies normalizada.
    
    Args:
        keypoints: Array de keypoints
        keypoint_indices: Diccionario con índices
        frame_height: Alto del frame
    
    Returns:
        Tupla (x, y) normalizada o None
    """
    left_ankle = get_keypoint(keypoints, keypoint_indices['left_ankle'])
    right_ankle = get_keypoint(keypoints, keypoint_indices['right_ankle'])
    
    ankles = [a for a in [left_ankle, right_ankle] if a is not None]
    
    if not ankles:
        return None
    
    avg_x = sum(a[0] for a in ankles) / len(ankles)
    avg_y = sum(a[1] for a in ankles) / len(ankles)
    
    # Normalizar Y respecto al alto del frame
    y_norm = avg_y / frame_height
    
    return (avg_x, y_norm)


def calculate_point_velocity(current: Tuple[float, float],
                             previous: Optional[Tuple[float, float]],
                             fps: float) -> float:
    """
    Calcula la velocidad de movimiento de un punto.
    
    Args:
        current: Posición actual (x, y)
        previous: Posición previa (x, y) o None
        fps: Frames por segundo del video
    
    Returns:
        Velocidad en píxeles/segundo
    """
    if previous is None:
        return 0.0
    
    distance = calculate_distance(previous, current)
    velocity = distance * fps
    
    return velocity


def smooth_value(current: float, 
                 history: List[float], 
                 window_size: int = 5) -> float:
    """
    Suaviza un valor usando promedio móvil.
    
    Args:
        current: Valor actual
        history: Lista de valores históricos
        window_size: Tamaño de la ventana de suavizado
    
    Returns:
        Valor suavizado
    """
    history.append(current)
    
    # Mantener solo los últimos N valores
    if len(history) > window_size:
        history.pop(0)
    
    return sum(history) / len(history)


def normalize_keypoints(keypoints: np.ndarray,
                       frame_width: int,
                       frame_height: int) -> np.ndarray:
    """
    Normaliza las coordenadas de keypoints al rango 0-1.
    
    Args:
        keypoints: Array de keypoints (17 x 3)
        frame_width: Ancho del frame
        frame_height: Alto del frame
    
    Returns:
        Keypoints normalizados
    """
    normalized = keypoints.copy()
    normalized[:, 0] = keypoints[:, 0] / frame_width   # X
    normalized[:, 1] = keypoints[:, 1] / frame_height  # Y
    # Confidence se mantiene igual
    
    return normalized