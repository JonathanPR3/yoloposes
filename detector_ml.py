"""
Detector de poses usando modelo de Machine Learning entrenado
Reemplaza las heurísticas manuales con predicciones del clasificador
"""
import numpy as np
import pickle
import logging
from typing import Tuple, Dict, Optional
from collections import deque
import time

logger = logging.getLogger(__name__)


class PoseDetectorML:
    """
    Detector de poses basado en modelo de Machine Learning.
    Usa el clasificador entrenado para identificar poses sospechosas.
    """
    
    def __init__(self, model_path: str = "data/pose_classifier.pkl"):
        """
        Inicializa el detector con el modelo entrenado.
        
        Args:
            model_path: Ruta al archivo .pkl del modelo entrenado
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
        
        # Mapeo de clases del modelo a nombres de poses
        self.class_names = {
            'escalamiento': 'escalamiento',
            'patada': 'patada',
            'lanzar': 'lanzar_objeto',
            'mirar_ventana': 'mirar_ventana'
        }
        
        # Buffer para suavizar predicciones (evitar falsos positivos)
        self.prediction_buffers = {
            'escalamiento': deque(maxlen=5),
            'patada': deque(maxlen=3),      # Más corto porque es rápida
            'lanzar': deque(maxlen=3),      # Más corto porque es rápida
            'mirar_ventana': deque(maxlen=10)  # Más largo porque es sostenida
        }
        
        # Umbrales de confianza por pose
        self.confidence_thresholds = {
            'escalamiento': 0.60,      # Umbral moderado
            'patada': 0.65,            # Umbral más alto (rápida, puede confundirse)
            'lanzar': 0.65,            # Umbral más alto (rápida)
            'mirar_ventana': 0.55      # Umbral más bajo (sostenida, menos ambigua)
        }
        
        # Contadores de tiempo para poses sostenidas
        self.pose_timers = {
            'mirar_ventana': {'start': None, 'duration': 0}
        }
        
        # Duraciones mínimas para poses sostenidas
        self.min_durations = {
            'mirar_ventana': 2.0  # segundos
        }
        
        logger.info("Detector ML inicializado correctamente")
    
    def load_model(self):
        """Carga el modelo entrenado desde disco."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Modelo cargado desde: {self.model_path}")
        except FileNotFoundError:
            logger.error(f"No se encontró el modelo en: {self.model_path}")
            logger.error("Ejecuta primero: python 3_train_classifier.py")
            raise
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def preprocess_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Preprocesa keypoints para que coincidan con el formato de entrenamiento.
        
        Args:
            keypoints: Array de shape (17, 3) con [x, y, conf]
        
        Returns:
            Array de shape (34,) con [x1, y1, x2, y2, ..., x17, y17]
        """
        # Extraer solo x, y (ignorar confianza)
        xy_coords = keypoints[:, :2]  # Shape: (17, 2)
        
        # Aplanar a un vector de 34 elementos
        flattened = xy_coords.flatten()  # Shape: (34,)
        
        return flattened
    
    def predict(self, keypoints: np.ndarray) -> Tuple[str, float]:
        """
        Predice la pose a partir de keypoints.
        
        Args:
            keypoints: Array de shape (17, 3) con [x, y, conf]
        
        Returns:
            Tupla (clase_predicha, confianza)
        """
        # Preprocesar
        features = self.preprocess_keypoints(keypoints)
        
        # Reshape para predicción (modelo espera 2D)
        features = features.reshape(1, -1)
        
        # Predecir clase
        predicted_class = self.model.predict(features)[0]
        
        # Obtener probabilidades
        probabilities = self.model.predict_proba(features)[0]
        
        # Encontrar confianza de la clase predicha
        class_index = list(self.model.classes_).index(predicted_class)
        confidence = probabilities[class_index]
        
        return predicted_class, confidence
    
    def smooth_prediction(self, pose_name: str, confidence: float) -> Tuple[bool, float]:
        """
        Suaviza predicciones usando un buffer temporal.
        Evita falsos positivos por detecciones esporádicas.
        
        Args:
            pose_name: Nombre de la pose
            confidence: Confianza de la predicción actual
        
        Returns:
            Tupla (debería_detectar, confianza_promedio)
        """
        buffer = self.prediction_buffers[pose_name]
        threshold = self.confidence_thresholds[pose_name]
        
        # Agregar predicción actual al buffer
        buffer.append(confidence)
        
        # Calcular confianza promedio
        avg_confidence = np.mean(buffer)
        
        # Detectar si el promedio supera el umbral
        should_detect = avg_confidence >= threshold
        
        return should_detect, avg_confidence
    
    def check_sustained_pose(self, pose_name: str, is_detected: bool) -> bool:
        """
        Verifica si una pose sostenida ha durado lo suficiente.
        
        Args:
            pose_name: Nombre de la pose
            is_detected: Si la pose está siendo detectada actualmente
        
        Returns:
            True si cumple la duración mínima
        """
        if pose_name not in self.pose_timers:
            return is_detected  # No requiere duración mínima
        
        timer = self.pose_timers[pose_name]
        min_duration = self.min_durations[pose_name]
        current_time = time.time()
        
        if is_detected:
            # Iniciar timer si no está activo
            if timer['start'] is None:
                timer['start'] = current_time
                timer['duration'] = 0
            else:
                # Actualizar duración
                timer['duration'] = current_time - timer['start']
            
            # Verificar si cumple duración mínima
            return timer['duration'] >= min_duration
        else:
            # Resetear timer si ya no se detecta
            timer['start'] = None
            timer['duration'] = 0
            return False
    
    def analyze_pose(
        self, 
        keypoints: np.ndarray, 
        frame_width: int, 
        frame_height: int
    ) -> Dict[str, Tuple[bool, float, Dict]]:
        """
        Analiza keypoints y detecta poses sospechosas.
        
        Args:
            keypoints: Array de shape (17, 3) con [x, y, conf]
            frame_width: Ancho del frame (para normalización si es necesario)
            frame_height: Alto del frame
        
        Returns:
            Dict con resultados para cada pose:
            {
                'escalamiento': (detected, confidence, metadata),
                'patada': (detected, confidence, metadata),
                ...
            }
        """
        # Predecir con el modelo
        predicted_class, raw_confidence = self.predict(keypoints)
        
        # Inicializar resultados
        results = {}
        
        # Procesar cada pose
        for pose_name in self.class_names.keys():
            if predicted_class == pose_name:
                # Esta es la clase predicha
                should_detect, smoothed_conf = self.smooth_prediction(
                    pose_name, 
                    raw_confidence
                )
                
                # Verificar duración para poses sostenidas
                if should_detect:
                    should_detect = self.check_sustained_pose(pose_name, True)
                else:
                    self.check_sustained_pose(pose_name, False)
                
                # Metadata
                metadata = {
                    'raw_confidence': raw_confidence,
                    'smoothed_confidence': smoothed_conf,
                    'buffer_size': len(self.prediction_buffers[pose_name]),
                    'threshold': self.confidence_thresholds[pose_name]
                }
                
                # Agregar duración si aplica
                if pose_name in self.pose_timers:
                    metadata['duration'] = self.pose_timers[pose_name]['duration']
                    metadata['min_duration'] = self.min_durations[pose_name]
                
                results[pose_name] = (should_detect, smoothed_conf, metadata)
            else:
                # No es la clase predicha
                self.check_sustained_pose(pose_name, False)  # Reset timer
                results[pose_name] = (False, 0.0, {})
        
        return results
    
    def reset_buffers(self):
        """Limpia todos los buffers de predicción."""
        for buffer in self.prediction_buffers.values():
            buffer.clear()
        
        for timer in self.pose_timers.values():
            timer['start'] = None
            timer['duration'] = 0
        
        logger.info("Buffers y timers reseteados")
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del detector."""
        return {
            'model_path': self.model_path,
            'classes': list(self.class_names.keys()),
            'thresholds': self.confidence_thresholds,
            'buffer_sizes': {k: len(v) for k, v in self.prediction_buffers.items()},
            'min_durations': self.min_durations
        }