"""
Lógica de detección de poses sospechosas
"""
import time
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import config
import joblib

from utils import (
    get_keypoint,
    calculate_torso_angle,
    calculate_limb_extension,
    get_foot_position,
    is_point_in_zone,
    calculate_point_velocity,
    get_body_height_ratio,
    calculate_distance  # AGREGAR ESTA LÍNEA
)

logger = logging.getLogger(__name__)


class PoseDetector:
    """
    Detector de poses sospechosas basado en análisis de keypoints.
    """
    
    def __init__(self):
        # Historial de detecciones para tracking temporal
        self.escalamiento_start_time: Optional[float] = None
        self.patada_start_time: Optional[float] = None
        self.lanzar_objeto_start_time: Optional[float] = None
        self.mirar_ventana_start_time: Optional[float] = None
        self.forzar_cerradura_start_time: Optional[float] = None
        
        # Historial de posiciones para calcular velocidad
        self.previous_foot_position: Optional[Tuple[float, float]] = None
        self.previous_wrist_position: Optional[Tuple[float, float]] = None
        self.previous_frame_time: float = time.time()
        
        # FPS estimado
        self.fps = 30.0
        
    def update_fps(self):
        """Actualiza el cálculo de FPS."""
        current_time = time.time()
        time_diff = current_time - self.previous_frame_time
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        self.previous_frame_time = current_time
    
    def detect_escalamiento(self, 
                          keypoints: np.ndarray,
                          frame_width: int,
                          frame_height: int) -> Tuple[bool, float, Dict]:
        """
        Detecta si una persona está escalando.
        
        Criterios:
        - Pies elevados por encima del umbral normal
        - Brazos extendidos hacia arriba
        - Posición en zona de barda
        - Duración mínima sosteniendo la pose
        
        Args:
            keypoints: Array de keypoints (17 x 3)
            frame_width: Ancho del frame
            frame_height: Alto del frame
        
        Returns:
            Tupla (detectado, confianza, metadata)
        """
        metadata = {}
        
        # 1. Verificar posición de los pies
        foot_pos = get_foot_position(keypoints, config.KEYPOINTS, frame_height)
        
        if not foot_pos:
            self.escalamiento_start_time = None
            return False, 0.0, metadata
        
        feet_height = 1.0 - foot_pos[1]  # Invertir (1 = arriba, 0 = abajo)
        metadata['feet_height'] = round(feet_height, 2)
        
        # 2. Verificar si los pies están elevados
        if feet_height < config.ESCALAMIENTO['altura_minima_pies']:
            self.escalamiento_start_time = None
            return False, 0.0, metadata
        
        # 3. Verificar extensión de brazos
        left_arm_extension = calculate_limb_extension(
            keypoints,
            config.KEYPOINTS['left_shoulder'],
            config.KEYPOINTS['left_elbow'],
            config.KEYPOINTS['left_wrist']
        )
        
        right_arm_extension = calculate_limb_extension(
            keypoints,
            config.KEYPOINTS['right_shoulder'],
            config.KEYPOINTS['right_elbow'],
            config.KEYPOINTS['right_wrist']
        )
        
        # Usar el mejor brazo detectado
        arm_extensions = [e for e in [left_arm_extension, right_arm_extension] if e is not None]
        
        if not arm_extensions:
            self.escalamiento_start_time = None
            return False, 0.0, metadata
        
        max_arm_extension = max(arm_extensions)
        metadata['arm_extension'] = round(max_arm_extension, 1)
        
        if max_arm_extension < config.ESCALAMIENTO['extension_brazos_min']:
            self.escalamiento_start_time = None
            return False, 0.0, metadata
        
        # 4. Verificar si está en zona de barda
        nose = get_keypoint(keypoints, config.KEYPOINTS['nose'])
        if nose:
            in_barda_zone = is_point_in_zone(
                (nose[0], nose[1]),
                config.ZONA_BARDA,
                frame_width,
                frame_height
            )
            metadata['in_barda_zone'] = in_barda_zone
        else:
            in_barda_zone = False
        
        # 5. Verificar duración
        current_time = time.time()
        
        if self.escalamiento_start_time is None:
            self.escalamiento_start_time = current_time
            duration = 0
        else:
            duration = current_time - self.escalamiento_start_time
        
        metadata['duration'] = round(duration, 1)
        
        # 6. Calcular confianza
        confidence = 0.0
        
        # Contribución por altura de pies (0-0.4)
        confidence += min(feet_height / config.ESCALAMIENTO['altura_minima_pies'], 1.0) * 0.4
        
        # Contribución por extensión de brazos (0-0.3)
        confidence += min(max_arm_extension / 180.0, 1.0) * 0.3
        
        # Contribución por estar en zona (0-0.2)
        if in_barda_zone:
            confidence += 0.2
        
        # Contribución por duración (0-0.1)
        if duration >= config.ESCALAMIENTO['duracion_minima']:
            confidence += 0.1
        
        # 7. Decidir si alertar
        if duration >= config.ESCALAMIENTO['duracion_minima'] and confidence > 0.6:
            return True, confidence, metadata
        
        return False, confidence, metadata
    
    def detect_agachado(self,
                       keypoints: np.ndarray,
                       frame_width: int,
                       frame_height: int) -> Tuple[bool, float, Dict]:
        """
        Detecta si una persona está agachada cerca de la puerta.
        
        Criterios:
        - Cabeza a altura baja
        - Torso inclinado
        - Posición en zona de puerta
        - Duración mínima
        
        Args:
            keypoints: Array de keypoints (17 x 3)
            frame_width: Ancho del frame
            frame_height: Alto del frame
        
        Returns:
            Tupla (detectado, confianza, metadata)
        """
        metadata = {}
        
        # 1. Verificar altura de la cabeza
        body_height = get_body_height_ratio(keypoints, config.KEYPOINTS, frame_height)
        
        if not body_height:
            self.agachado_start_time = None
            return False, 0.0, metadata
        
        metadata['head_height'] = round(body_height, 2)
        
        if body_height < config.AGACHADO['altura_maxima_cabeza']:
            self.agachado_start_time = None
            return False, 0.0, metadata
        
        # 2. Calcular inclinación del torso
        torso_angle = calculate_torso_angle(keypoints, config.KEYPOINTS)
        
        if not torso_angle:
            self.agachado_start_time = None
            return False, 0.0, metadata
        
        metadata['torso_angle'] = round(torso_angle, 1)
        
        if not (config.AGACHADO['inclinacion_torso_min'] <= torso_angle <= config.AGACHADO['inclinacion_torso_max']):
            self.agachado_start_time = None
            return False, 0.0, metadata
        
        # 3. Verificar si está en zona de puerta
        nose = get_keypoint(keypoints, config.KEYPOINTS['nose'])
        if nose:
            in_door_zone = is_point_in_zone(
                (nose[0], nose[1]),
                config.ZONA_PUERTA,
                frame_width,
                frame_height
            )
            metadata['in_door_zone'] = in_door_zone
        else:
            in_door_zone = False
            self.agachado_start_time = None
            return False, 0.0, metadata
        
        if not in_door_zone:
            self.agachado_start_time = None
            return False, 0.0, metadata
        
        # 4. Verificar duración
        current_time = time.time()
        
        if self.agachado_start_time is None:
            self.agachado_start_time = current_time
            duration = 0
        else:
            duration = current_time - self.agachado_start_time
        
        metadata['duration'] = round(duration, 1)
        
        # 5. Calcular confianza
        confidence = 0.0
        
        # Contribución por altura (0-0.3)
        confidence += (body_height / 1.0) * 0.3
        
        # Contribución por inclinación del torso (0-0.4)
        angle_range = config.AGACHADO['inclinacion_torso_max'] - config.AGACHADO['inclinacion_torso_min']
        angle_score = 1.0 - abs(torso_angle - 50) / (angle_range / 2)  # Óptimo en 50°
        confidence += max(0, angle_score) * 0.4
        
        # Contribución por estar en zona de puerta (0-0.2)
        confidence += 0.2
        
        # Contribución por duración (0-0.1)
        if duration >= config.AGACHADO['duracion_minima']:
            confidence += 0.1
        
        # 6. Decidir si alertar
        if duration >= config.AGACHADO['duracion_minima'] and confidence > 0.6:
            return True, confidence, metadata
        
        return False, confidence, metadata
    
    def detect_patada(self,
                     keypoints: np.ndarray,
                     frame_width: int,
                     frame_height: int) -> Tuple[bool, float, Dict]:
        """
        Detecta si una persona está pateando la puerta.
        
        Criterios:
        - Pie elevado
        - Pierna extendida
        - Movimiento rápido del pie
        - Posición cerca de la puerta
        
        Args:
            keypoints: Array de keypoints (17 x 3)
            frame_width: Ancho del frame
            frame_height: Alto del frame
        
        Returns:
            Tupla (detectado, confianza, metadata)
        """
        metadata = {}
        
        # 1. Verificar posición del pie
        foot_pos = get_foot_position(keypoints, config.KEYPOINTS, frame_height)
        
        if not foot_pos:
            self.patada_start_time = None
            self.previous_foot_position = None
            return False, 0.0, metadata
        
        foot_height = 1.0 - foot_pos[1]  # Invertir
        metadata['foot_height'] = round(foot_height, 2)
        
        if foot_height < config.PATADA['altura_minima_pie']:
            self.patada_start_time = None
            self.previous_foot_position = None
            return False, 0.0, metadata
        
        # 2. Calcular velocidad del pie
        velocity = 0.0
        if self.previous_foot_position:
            velocity = calculate_point_velocity(
                foot_pos,
                self.previous_foot_position,
                self.fps
            )
        
        self.previous_foot_position = foot_pos
        metadata['foot_velocity'] = round(velocity, 1)
        
        # 3. Verificar extensión de piernas
        left_leg_extension = calculate_limb_extension(
            keypoints,
            config.KEYPOINTS['left_hip'],
            config.KEYPOINTS['left_knee'],
            config.KEYPOINTS['left_ankle']
        )
        
        right_leg_extension = calculate_limb_extension(
            keypoints,
            config.KEYPOINTS['right_hip'],
            config.KEYPOINTS['right_knee'],
            config.KEYPOINTS['right_ankle']
        )
        
        leg_extensions = [e for e in [left_leg_extension, right_leg_extension] if e is not None]
        
        if not leg_extensions:
            self.patada_start_time = None
            return False, 0.0, metadata
        
        max_leg_extension = max(leg_extensions)
        metadata['leg_extension'] = round(max_leg_extension, 1)
        
        if max_leg_extension < config.PATADA['extension_pierna_min']:
            self.patada_start_time = None
            return False, 0.0, metadata
        
        # 4. Verificar si está cerca de la puerta
        nose = get_keypoint(keypoints, config.KEYPOINTS['nose'])
        if nose:
            in_door_zone = is_point_in_zone(
                (nose[0], nose[1]),
                config.ZONA_PUERTA,
                frame_width,
                frame_height
            )
            metadata['in_door_zone'] = in_door_zone
        else:
            in_door_zone = False
        
        # 5. Verificar duración (la patada es rápida, pero queremos capturarla)
        current_time = time.time()
        
        if self.patada_start_time is None:
            self.patada_start_time = current_time
            duration = 0
        else:
            duration = current_time - self.patada_start_time
        
        metadata['duration'] = round(duration, 1)
        
        # Reset si pasa mucho tiempo (patada fallida)
        if duration > 2.0:
            self.patada_start_time = None
            return False, 0.0, metadata
        
        # 6. Calcular confianza
        confidence = 0.0
        
        # Contribución por altura del pie (0-0.3)
        confidence += min(foot_height / config.PATADA['altura_minima_pie'], 1.0) * 0.3
        
        # Contribución por velocidad (0-0.4)
        confidence += min(velocity / config.PATADA['velocidad_minima'], 1.0) * 0.4
        
        # Contribución por extensión de pierna (0-0.2)
        confidence += min(max_leg_extension / 180.0, 1.0) * 0.2
        
        # Contribución por estar en zona de puerta (0-0.1)
        if in_door_zone:
            confidence += 0.1
        
        # 7. Decidir si alertar (patada requiere menos duración)
        if duration >= config.PATADA['duracion_minima'] and confidence > 0.5:
            self.patada_start_time = None  # Reset para próxima detección
            return True, confidence, metadata
        
        return False, confidence, metadata
    
    def detect_lanzar_objeto(self,
                            keypoints: np.ndarray,
                            frame_width: int,
                            frame_height: int) -> Tuple[bool, float, Dict]:
        """
        Detecta si una persona está lanzando un objeto.
        
        Criterios:
        - Brazo extendido hacia adelante/arriba
        - Mano elevada por encima del hombro
        - Movimiento rápido del brazo/mano
        - Aceleración detectada
        
        Returns:
            Tupla (detectado, confianza, metadata)
        """
        metadata = {}
        
        # 1. Verificar extensión de brazos
        left_arm_extension = calculate_limb_extension(
            keypoints,
            config.KEYPOINTS['left_shoulder'],
            config.KEYPOINTS['left_elbow'],
            config.KEYPOINTS['left_wrist']
        )
        
        right_arm_extension = calculate_limb_extension(
            keypoints,
            config.KEYPOINTS['right_shoulder'],
            config.KEYPOINTS['right_elbow'],
            config.KEYPOINTS['right_wrist']
        )
        
        arm_extensions = [e for e in [left_arm_extension, right_arm_extension] if e is not None]
        
        if not arm_extensions:
            self.lanzar_objeto_start_time = None
            return False, 0.0, metadata
        
        max_arm_extension = max(arm_extensions)
        metadata['arm_extension'] = round(max_arm_extension, 1)
        
        if max_arm_extension < config.LANZAR_OBJETO['extension_brazo_min']:
            self.lanzar_objeto_start_time = None
            return False, 0.0, metadata
        
        # 2. Verificar altura de la mano (por encima del hombro)
        left_wrist = get_keypoint(keypoints, config.KEYPOINTS['left_wrist'])
        right_wrist = get_keypoint(keypoints, config.KEYPOINTS['right_wrist'])
        left_shoulder = get_keypoint(keypoints, config.KEYPOINTS['left_shoulder'])
        right_shoulder = get_keypoint(keypoints, config.KEYPOINTS['right_shoulder'])
        
        hand_above_shoulder = False
        if left_wrist and left_shoulder:
            if left_wrist[1] < left_shoulder[1]:  # Y menor = más arriba
                hand_above_shoulder = True
        if right_wrist and right_shoulder:
            if right_wrist[1] < right_shoulder[1]:
                hand_above_shoulder = True
        
        metadata['hand_above_shoulder'] = hand_above_shoulder
        
        if not hand_above_shoulder:
            self.lanzar_objeto_start_time = None
            return False, 0.0, metadata
        
        # 3. Calcular velocidad de la mano
        # Usar la muñeca con mejor detección
        current_wrist = left_wrist if left_wrist and left_wrist[2] > 0.5 else right_wrist
        
        velocity = 0.0
        if current_wrist and hasattr(self, 'previous_wrist_position'):
            velocity = calculate_point_velocity(
                (current_wrist[0], current_wrist[1]),
                self.previous_wrist_position,
                self.fps
            )
        
        if current_wrist:
            self.previous_wrist_position = (current_wrist[0], current_wrist[1])
        
        metadata['hand_velocity'] = round(velocity, 1)
        
        # 4. Verificar duración
        current_time = time.time()
        
        if self.lanzar_objeto_start_time is None:
            self.lanzar_objeto_start_time = current_time
            duration = 0
        else:
            duration = current_time - self.lanzar_objeto_start_time
        
        metadata['duration'] = round(duration, 1)
        
        # Reset si pasa mucho tiempo
        if duration > 2.0:
            self.lanzar_objeto_start_time = None
            return False, 0.0, metadata
        
        # 5. Calcular confianza
        confidence = 0.0
        
        # Extensión de brazo (0-0.3)
        confidence += min(max_arm_extension / 180.0, 1.0) * 0.3
        
        # Mano arriba (0-0.3)
        if hand_above_shoulder:
            confidence += 0.3
        
        # Velocidad (0-0.4)
        confidence += min(velocity / config.LANZAR_OBJETO['velocidad_mano_min'], 1.0) * 0.4
        
        # 6. Decidir si alertar
        if duration >= config.LANZAR_OBJETO['duracion_minima'] and confidence > 0.6:
            self.lanzar_objeto_start_time = None
            return True, confidence, metadata
        
        return False, confidence, metadata
    
    def detect_mirar_ventana(self,
                            keypoints: np.ndarray,
                            frame_width: int,
                            frame_height: int) -> Tuple[bool, float, Dict]:
        """
        Detecta si una persona está mirando a través de una ventana.
        
        Criterios:
        - Cabeza a altura de ventana típica
        - Cabeza inclinada hacia adelante
        - Manos cerca de la cara (opcional: binoculares o proteger vista)
        - Duración prolongada
        
        Returns:
            Tupla (detectado, confianza, metadata)
        """
        metadata = {}
        
        # 1. Verificar altura de la cabeza
        nose = get_keypoint(keypoints, config.KEYPOINTS['nose'])
        
        if not nose:
            self.mirar_ventana_start_time = None
            return False, 0.0, metadata
        
        head_height = nose[1] / frame_height
        metadata['head_height'] = round(head_height, 2)
        
        if not (config.MIRAR_VENTANA['altura_cabeza_min'] <= head_height <= config.MIRAR_VENTANA['altura_cabeza_max']):
            self.mirar_ventana_start_time = None
            return False, 0.0, metadata
        
        # 2. Verificar inclinación de la cabeza (hacia adelante)
        # Usando posición de nariz vs orejas
        left_ear = get_keypoint(keypoints, config.KEYPOINTS['left_ear'])
        right_ear = get_keypoint(keypoints, config.KEYPOINTS['right_ear'])
        
        head_forward = False
        if nose and (left_ear or right_ear):
            ear = left_ear if left_ear else right_ear
            # Si la nariz está significativamente adelante de la oreja
            if nose[0] > ear[0] + 20:  # 20 píxeles adelante
                head_forward = True
        
        metadata['head_forward'] = head_forward
        
        # 3. Verificar manos cerca de la cara (opcional pero aumenta confianza)
        left_wrist = get_keypoint(keypoints, config.KEYPOINTS['left_wrist'])
        right_wrist = get_keypoint(keypoints, config.KEYPOINTS['right_wrist'])
        
        hands_near_face = False
        if nose and (left_wrist or right_wrist):
            wrists = [w for w in [left_wrist, right_wrist] if w is not None]
            for wrist in wrists:
                distance = calculate_distance((nose[0], nose[1]), (wrist[0], wrist[1]))
                if distance < config.MIRAR_VENTANA['distancia_manos_cara']:
                    hands_near_face = True
                    break
        
        metadata['hands_near_face'] = hands_near_face
        
        # 4. Verificar duración
        current_time = time.time()
        
        if self.mirar_ventana_start_time is None:
            self.mirar_ventana_start_time = current_time
            duration = 0
        else:
            duration = current_time - self.mirar_ventana_start_time
        
        metadata['duration'] = round(duration, 1)
        
        # 5. Calcular confianza
        confidence = 0.0
        
        # Altura correcta (0-0.3)
        height_score = 1.0 - abs(head_height - 0.45) / 0.15  # Óptimo en 45%
        confidence += max(0, height_score) * 0.3
        
        # Cabeza hacia adelante (0-0.3)
        if head_forward:
            confidence += 0.3
        
        # Manos cerca de cara (0-0.2)
        if hands_near_face:
            confidence += 0.2
        
        # Duración (0-0.2)
        if duration >= config.MIRAR_VENTANA['duracion_minima']:
            confidence += 0.2
        
        # 6. Decidir si alertar
        if duration >= config.MIRAR_VENTANA['duracion_minima'] and confidence > 0.5:
            return True, confidence, metadata
        
        return False, confidence, metadata
    
    def detect_forzar_cerradura(self,
                                keypoints: np.ndarray,
                                frame_width: int,
                                frame_height: int) -> Tuple[bool, float, Dict]:
        """
        Detecta si una persona está forzando una cerradura.
        
        Criterios:
        - Agachado a altura de cerradura (45-65% del frame)
        - Torso inclinado
        - Ambas manos juntas/cerca (manipulando cerradura)
        - Movimientos pequeños repetitivos de las manos
        - Duración prolongada
        
        Returns:
            Tupla (detectado, confianza, metadata)
        """
        metadata = {}
        
        # 1. Verificar altura de la cabeza (a nivel de cerradura)
        body_height = get_body_height_ratio(keypoints, config.KEYPOINTS, frame_height)
        
        if not body_height:
            self.forzar_cerradura_start_time = None
            return False, 0.0, metadata
        
        metadata['head_height'] = round(body_height, 2)
        
        if not (config.FORZAR_CERRADURA['altura_minima_cabeza'] <= body_height <= config.FORZAR_CERRADURA['altura_maxima_cabeza']):
            self.forzar_cerradura_start_time = None
            return False, 0.0, metadata
        
        # 2. Verificar inclinación del torso
        torso_angle = calculate_torso_angle(keypoints, config.KEYPOINTS)
        
        if not torso_angle:
            self.forzar_cerradura_start_time = None
            return False, 0.0, metadata
        
        metadata['torso_angle'] = round(torso_angle, 1)
        
        if not (config.FORZAR_CERRADURA['inclinacion_torso_min'] <= torso_angle <= config.FORZAR_CERRADURA['inclinacion_torso_max']):
            self.forzar_cerradura_start_time = None
            return False, 0.0, metadata
        
        # 3. Verificar que ambas manos estén juntas
        left_wrist = get_keypoint(keypoints, config.KEYPOINTS['left_wrist'])
        right_wrist = get_keypoint(keypoints, config.KEYPOINTS['right_wrist'])
        
        hands_together = False
        hands_distance = 0
        
        if left_wrist and right_wrist:
            hands_distance = calculate_distance(
                (left_wrist[0], left_wrist[1]),
                (right_wrist[0], right_wrist[1])
            )
            
            if hands_distance < config.FORZAR_CERRADURA['distancia_maxima_manos']:
                hands_together = True
        
        metadata['hands_distance'] = round(hands_distance, 1)
        metadata['hands_together'] = hands_together
        
        if not hands_together:
            self.forzar_cerradura_start_time = None
            return False, 0.0, metadata
        
        # 4. Verificar posición en zona (si USE_ZONES está activo)
        in_door_zone = True
        if config.USE_ZONES:
            nose = get_keypoint(keypoints, config.KEYPOINTS['nose'])
            if nose:
                in_door_zone = is_point_in_zone(
                    (nose[0], nose[1]),
                    config.ZONA_PUERTA,
                    frame_width,
                    frame_height
                )
                metadata['in_door_zone'] = in_door_zone
            
            if not in_door_zone:
                self.forzar_cerradura_start_time = None
                return False, 0.0, metadata
        
        # 5. Verificar duración
        current_time = time.time()
        
        if self.forzar_cerradura_start_time is None:
            self.forzar_cerradura_start_time = current_time
            duration = 0
        else:
            duration = current_time - self.forzar_cerradura_start_time
        
        metadata['duration'] = round(duration, 1)
        
        # 6. Calcular confianza
        confidence = 0.0
        
        # Altura correcta (0-0.3)
        height_score = 1.0 - abs(body_height - 0.55) / 0.1  # Óptimo en 55%
        confidence += max(0, height_score) * 0.3
        
        # Inclinación del torso (0-0.3)
        angle_range = config.FORZAR_CERRADURA['inclinacion_torso_max'] - config.FORZAR_CERRADURA['inclinacion_torso_min']
        angle_score = 1.0 - abs(torso_angle - 50) / (angle_range / 2)
        confidence += max(0, angle_score) * 0.3
        
        # Manos juntas (0-0.3)
        if hands_together:
            distance_score = 1.0 - (hands_distance / config.FORZAR_CERRADURA['distancia_maxima_manos'])
            confidence += distance_score * 0.3
        
        # Duración (0-0.1)
        if duration >= config.FORZAR_CERRADURA['duracion_minima']:
            confidence += 0.1
        
        # 7. Decidir si alertar
        if duration >= config.FORZAR_CERRADURA['duracion_minima'] and confidence > 0.6:
            return True, confidence, metadata
        
        return False, confidence, metadata
    
    def analyze_pose(self,
                    keypoints: np.ndarray,
                    frame_width: int,
                    frame_height: int) -> Dict:
        """
        Analiza todas las poses sospechosas en un frame.
        
        Args:
            keypoints: Array de keypoints (17 x 3)
            frame_width: Ancho del frame
            frame_height: Alto del frame
        
        Returns:
            Diccionario con resultados de todas las detecciones
        """
        self.update_fps()
        
        results = {
            'escalamiento': self.detect_escalamiento(keypoints, frame_width, frame_height),
            'patada': self.detect_patada(keypoints, frame_width, frame_height),
            'lanzar_objeto': self.detect_lanzar_objeto(keypoints, frame_width, frame_height),
            'mirar_ventana': self.detect_mirar_ventana(keypoints, frame_width, frame_height),
            'forzar_cerradura': self.detect_forzar_cerradura(keypoints, frame_width, frame_height),
            'fps': round(self.fps, 1)
        }
        
        return results