"""
Sistema de notificaciones a API REST
"""
import requests
import logging
import time
from datetime import datetime
from typing import Dict, Optional
import config

logger = logging.getLogger(__name__)


class AlertNotifier:
    """
    Maneja el envío de alertas a la API REST con cooldown.
    """
    
    def __init__(self):
        self.last_alert_times: Dict[str, float] = {}
        self.last_global_alert = 0
        self.alert_count = 0
        
    def can_send_alert(self, alert_type: str, cooldown: int) -> bool:
        """
        Verifica si se puede enviar una alerta según el cooldown.
        
        Args:
            alert_type: Tipo de alerta ('escalamiento', 'agachado', 'patada')
            cooldown: Tiempo de cooldown específico en segundos
        
        Returns:
            True si se puede enviar la alerta
        """
        current_time = time.time()
        
        # Verificar cooldown global
        if current_time - self.last_global_alert < config.GLOBAL_COOLDOWN:
            return False
        
        # Verificar cooldown específico del tipo de alerta
        if alert_type in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[alert_type]
            if time_since_last < cooldown:
                logger.debug(f"Cooldown activo para {alert_type}: {cooldown - time_since_last:.1f}s restantes")
                return False
        
        return True
    
    def send_alert(self, 
                   alert_type: str,
                   confidence: float,
                   cooldown: int,
                   metadata: Optional[Dict] = None) -> bool:
        """
        Envía una alerta a la API REST.
        
        Args:
            alert_type: Tipo de alerta ('escalamiento', 'agachado', 'patada')
            confidence: Nivel de confianza de la detección (0-1)
            cooldown: Tiempo de cooldown en segundos
            metadata: Información adicional opcional
        
        Returns:
            True si la alerta se envió exitosamente
        """
        # Verificar cooldown
        if not self.can_send_alert(alert_type, cooldown):
            return False
        
        # Preparar payload
        payload = {
            'alert_type': alert_type,
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().isoformat(),
            'device_id': 'camera_01',  # Ajustar según tu setup
        }
        
        # Agregar metadata si existe
        if metadata:
            payload['metadata'] = metadata
        
        try:
            # Headers con autenticación si es necesaria
            headers = {'Content-Type': 'application/json'}
            
            if config.API_TOKEN:
                headers['Authorization'] = f'Bearer {config.API_TOKEN}'
            
            # Enviar request
            response = requests.post(
                config.API_URL,
                json=payload,
                headers=headers,
                timeout=5
            )
            
            # Verificar respuesta
            if response.status_code in [200, 201]:
                logger.info(f"✅ Alerta enviada: {alert_type} (confianza: {confidence:.2f})")
                
                # Actualizar timestamps
                current_time = time.time()
                self.last_alert_times[alert_type] = current_time
                self.last_global_alert = current_time
                self.alert_count += 1
                
                return True
            else:
                logger.error(f"❌ Error al enviar alerta: HTTP {response.status_code}")
                logger.error(f"Respuesta: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("❌ Timeout al enviar alerta a la API")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("❌ Error de conexión con la API")
            return False
        except Exception as e:
            logger.error(f"❌ Error inesperado al enviar alerta: {e}")
            return False
    
    def send_escalamiento_alert(self, confidence: float, metadata: Optional[Dict] = None) -> bool:
        """Envía alerta de escalamiento."""
        return self.send_alert(
            'escalamiento',
            confidence,
            config.ESCALAMIENTO['cooldown'],
            metadata
        )
    
    def send_patada_alert(self, confidence: float, metadata: Optional[Dict] = None) -> bool:
        """Envía alerta de patada."""
        return self.send_alert(
            'patada',
            confidence,
            config.PATADA['cooldown'],
            metadata
        )
    
    def send_lanzar_objeto_alert(self, confidence: float, metadata: Optional[Dict] = None) -> bool:
        """Envía alerta de lanzamiento de objeto."""
        return self.send_alert(
            'lanzar_objeto',
            confidence,
            config.LANZAR_OBJETO['cooldown'],
            metadata
        )
    
    def send_mirar_ventana_alert(self, confidence: float, metadata: Optional[Dict] = None) -> bool:
        """Envía alerta de persona mirando por ventana."""
        return self.send_alert(
            'mirar_ventana',
            confidence,
            config.MIRAR_VENTANA['cooldown'],
            metadata
        )
    
    def send_forzar_cerradura_alert(self, confidence: float, metadata: Optional[Dict] = None) -> bool:
        """Envía alerta de forzamiento de cerradura."""
        return self.send_alert(
            'forzar_cerradura',
            confidence,
            config.FORZAR_CERRADURA['cooldown'],
            metadata
        )
    
    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas de alertas enviadas.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'total_alerts': self.alert_count,
            'last_alerts': self.last_alert_times,
            'global_cooldown_active': (time.time() - self.last_global_alert) < config.GLOBAL_COOLDOWN
        }
    
    def reset_cooldowns(self):
        """Resetea todos los cooldowns (útil para testing)."""
        self.last_alert_times.clear()
        self.last_global_alert = 0
        logger.info("🔄 Cooldowns reseteados")