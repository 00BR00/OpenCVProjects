import cv2 as cv
import numpy as np

class DetectorFondoHomogeneo():
    def __init__(self) -> None:
        pass
    
    def deteccion_objetos(self, frame):
        # Convertir imagen a escala de grises
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Aplicar umbral adaptativo
        mask = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 19, 5)
        
        # Encontrar contornos
        contornos, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área
        objetos_contornos = [contorno for contorno in contornos if cv.contourArea(contorno) > 2000]
        
        return objetos_contornos
                