import cv2 as cv
import numpy as np
from classdetector import *
from cv2 import aruco

#using iPCAM because i dont use WEBCAM
ipcam = '192.168.1.17:8080'
url = f'http://{ipcam}/video'

# PARAMETROS Y DICCIONARIO 
parametros = cv.aruco.DetectorParameters()
diccionario = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)


# Crear instancias de los detectores
detector = DetectorFondoHomogeneo()
cap = cv.VideoCapture(url)
cap.set(3, 640)
cap.set(4, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame.")
            break

        esquinas, _, _ = cv.aruco.detectMarkers(frame, diccionario, parameters=parametros)

        if esquinas and len(esquinas) > 0:
            esquinas_ent = np.int0(esquinas)
            cv.polylines(frame, esquinas_ent, True, (0, 0, 255), 5)

            perimetro_aruco = cv.arcLength(esquinas_ent[0], True)
            proporcion_cm = perimetro_aruco / 16

            # Usar el detector de fondohomogeneo
            contornos = detector.deteccion_objetos(frame)

            for cont in contornos:

                rectangulo = cv.minAreaRect(cont)
                (x, y), (an, al), angulo = rectangulo

                ancho = an / proporcion_cm
                alto = al / proporcion_cm

                cv.circle(frame, (int(x), int(y)), 5, (255, 255, 0), -1)

                rect = cv.boxPoints(rectangulo)
                rect = np.int0(rect)
                
                cv.polylines(frame, [rect], True, (0,255,0),2)

                # Dibuja el rectángulo alrededor del objeto
                cv.drawContours(frame, [rect], 0, (0, 255, 0), 2)

                # Muestra las dimensiones del objeto en el frame
                dimensiones_texto = f'Ancho: {ancho:.2f} cm, Alto: {alto:.2f} cm'
                cv.putText(frame, dimensiones_texto, (int(x) - 50, int(y) - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Muestra el frame resultante
        cv.imshow('Medidor de Objetos', frame)

        # Rompe el bucle si se presiona la tecla 'q'
        key = cv.waitKey(1)
        if key == 27:  # 27 es el código ASCII para la tecla 'Esc'
            break

except Exception as e:
    print(f"Error inesperado: {e}")

finally:
    # Libera los recursos y cierra las ventanas
    cap.release()
    cv.destroyAllWindows()


