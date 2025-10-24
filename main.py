"""
Sistema de detección de poses sospechosas usando YOLO + ML Classifier
Versión con Machine Learning
"""
import cv2
import logging
import time
import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# Importar módulos del proyecto
import config
from detector_ml import PoseDetectorML  # ← USAR DETECTOR ML
from notifier import AlertNotifier
from utils import get_keypoint

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pose_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PoseDetectionSystem:
    """Sistema principal de detección de poses."""
    
    def __init__(self):
        """Inicializa el sistema de detección."""
        self.model = None
        self.cap = None
        self.pose_detector = None
        self.notifier = None
        self.running = False
        
        # Estadísticas
        self.frame_count = 0
        self.alert_count = 0
        self.start_time = None
        
        # Control de notificaciones (cooldown)
        self.last_alert_time = {}
        self.alert_cooldown = config.GLOBAL_COOLDOWN
        
    def initialize(self):
        """Inicializa todos los componentes del sistema."""
        try:
            # Cargar modelo YOLO
            logger.info(f"Cargando modelo {config.YOLO_MODEL}...")
            self.model = YOLO(config.YOLO_MODEL)
            logger.info("Modelo cargado correctamente")
            
            # Inicializar detector ML
            logger.info("Inicializando detector ML...")
            self.pose_detector = PoseDetectorML(
                model_path="data/pose_classifier.pkl"
            )
            
            # Inicializar notificador
            try:
                # Intenta con webhook_url (nuevo formato)
                self.notifier = AlertNotifier(
                    webhook_url=config.WEBHOOK_URL,
                    enable_notifications=config.ENABLE_NOTIFICATIONS
                )
            except TypeError:
                # Si falla, usa el formato viejo (sin argumentos o con api_url)
                try:
                    self.notifier = AlertNotifier()
                except TypeError:
                    self.notifier = AlertNotifier(
                        api_url=config.API_URL,
                        api_token=config.API_TOKEN
                    )
            
            # Conectar a cámara
            logger.info(f"Conectando a camara: {config.CAMERA_INDEX}")
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"No se pudo abrir la cámara {config.CAMERA_INDEX}")
            
            # Configurar resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            
            # Verificar que se puede leer frames
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("No se pudo leer frame de la cámara")
            
            logger.info("Verificando lectura de frames...")
            logger.info("Sistema iniciado correctamente")
            logger.info(f"Resolucion de captura: {frame.shape[1]}x{frame.shape[0]}")
            
            # Mostrar estadísticas del detector
            stats = self.pose_detector.get_stats()
            logger.info(f"Detector ML stats: {stats}")
            
            self.running = True
            return True
            
        except Exception as e:
            logger.error(f"Error en inicialización: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un frame y detecta poses.
        
        Args:
            frame: Frame de video en formato BGR
            
        Returns:
            Frame procesado con visualizaciones
        """
        height, width = frame.shape[:2]
        
        # Detectar personas y keypoints con YOLO
        results = self.model(
            frame, 
            conf=config.CONFIDENCE_THRESHOLD,
            verbose=False
        )
        
        # Procesar cada persona detectada
        for result in results:
            if result.keypoints is None or len(result.keypoints.data) == 0:
                continue
            
            # Obtener keypoints de la primera persona
            keypoints = result.keypoints.data[0].cpu().numpy()
            
            # Verificar que hay suficientes keypoints válidos
            valid_keypoints = np.sum(keypoints[:, 2] > config.KEYPOINT_CONFIDENCE_THRESHOLD)
            if valid_keypoints < config.MIN_KEYPOINTS_REQUIRED:
                continue
            
            # Dibujar skeleton
            if config.DRAW_SKELETON:
                self.draw_skeleton(frame, keypoints)
            
            # Analizar poses con ML
            pose_results = self.pose_detector.analyze_pose(keypoints, width, height)
            
            # Procesar resultados
            alerts_to_send = []
            
            # Revisar cada pose
            for pose_name, (detected, conf, meta) in pose_results.items():
                if detected:
                    logger.warning(f"{pose_name.upper()} detectado (conf: {conf:.2f})")
                    self.draw_alert(frame, pose_name, conf, meta)
                    alerts_to_send.append((pose_name, conf, meta))
            
            # Enviar alertas (respetando cooldown)
            for pose_name, conf, meta in alerts_to_send:
                self.send_alert_if_allowed(pose_name, conf, meta, frame)
        
        # Dibujar información en pantalla
        self.draw_info(frame)
        
        return frame
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray):
        """Dibuja el skeleton sobre el frame."""
        # Conexiones del skeleton (formato COCO)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Cabeza
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Brazos
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Piernas
        ]
        
        # Dibujar conexiones
        for start_idx, end_idx in connections:
            start = keypoints[start_idx]
            end = keypoints[end_idx]
            
            if start[2] > 0.5 and end[2] > 0.5:
                start_point = (int(start[0]), int(start[1]))
                end_point = (int(end[0]), int(end[1]))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Dibujar keypoints
        for kp in keypoints:
            if kp[2] > 0.5:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)
    
    def draw_alert(self, frame: np.ndarray, pose_name: str, confidence: float, metadata: dict):
        """Dibuja alerta visual en el frame."""
        # Configurar colores según severidad
        if confidence > 0.8:
            color = (0, 0, 255)  # Rojo (alta confianza)
        elif confidence > 0.6:
            color = (0, 165, 255)  # Naranja
        else:
            color = (0, 255, 255)  # Amarillo
        
        # Texto de alerta
        text = f"ALERTA: {pose_name.upper()}"
        conf_text = f"Confianza: {confidence:.1%}"
        
        # Dibujar fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Dibujar texto
        cv2.putText(frame, text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, conf_text, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar metadata adicional
        if 'duration' in metadata:
            duration_text = f"Duracion: {metadata['duration']:.1f}s"
            cv2.putText(frame, duration_text, (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def draw_info(self, frame: np.ndarray):
        """Dibuja información del sistema en el frame."""
        info_text = f"FPS: {self.get_fps():.1f} | Frames: {self.frame_count} | Alertas: {self.alert_count}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def send_alert_if_allowed(self, pose_name: str, confidence: float, metadata: dict, frame: np.ndarray):
        """Envía alerta si ha pasado el cooldown."""
        current_time = time.time()
        
        # Verificar cooldown
        if pose_name in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[pose_name]
            if time_since_last < self.alert_cooldown:
                return
        
        # Enviar alerta
        def send_alert_if_allowed(self, pose_name: str, confidence: float, metadata: dict, frame: np.ndarray):
            if self.notifier is None:  # ← ESTA LÍNEA DEBE ESTAR
                return
        self.last_alert_time[pose_name] = current_time
        self.alert_count += 1
    
    def get_fps(self) -> float:
        """Calcula FPS actual."""
        if self.start_time is None or self.frame_count == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def run(self):
        """Loop principal de detección."""
        logger.info("Iniciando deteccion...")
        logger.info("Presiona 'q' para salir, 'r' para resetear cooldowns")
        
        self.start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("No se pudo leer frame")
                break
            
            # Procesar frame
            frame = self.process_frame(frame)
            self.frame_count += 1
            
            # Mostrar frame
            if config.DISPLAY_VIDEO:
                cv2.imshow('Pose Detection', frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Saliendo...")
                break
            elif key == ord('r'):
                logger.info("Reseteando cooldowns...")
                self.last_alert_time.clear()
                self.pose_detector.reset_buffers()
        
        self.cleanup()
    
    def cleanup(self):
        """Libera recursos."""
        logger.info("Liberando recursos...")
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Mostrar estadísticas finales
        logger.info("Estadisticas finales:")
        logger.info(f"   Frames procesados: {self.frame_count}")
        logger.info(f"   Alertas enviadas: {self.alert_count}")
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"   FPS promedio: {self.get_fps():.1f}")
        
        logger.info("Sistema detenido")


def main():
    """Función principal."""
    try:
        logger.info("Iniciando sistema de deteccion de poses...")
        
        system = PoseDetectionSystem()
        
        if not system.initialize():
            logger.error("No se pudo inicializar el sistema")
            return 1
        
        system.run()
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupcion por usuario")
        return 0
    except Exception as e:
        logger.error(f"Error fatal: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())