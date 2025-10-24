import cv2

print("Buscando camaras disponibles...\n")

# Probar indices del 0 al 10
for i in range(10):
    print(f"Probando camara {i}...", end=" ")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DSHOW es para Windows
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"FUNCIONA! - Resolucion: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
        else:
            print("Abre pero no lee frames")
            cap.release()
    else:
        print("No disponible")

print("\nSi ninguna funciono, verifica:")
print("1. Que la webcam este conectada")
print("2. Que ningun otro programa la este usando (Zoom, Teams, etc.)")
print("3. Permisos de camara en Windows (Configuracion > Privacidad > Camara)")