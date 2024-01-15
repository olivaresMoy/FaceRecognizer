#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:39:44 2023
@Proyecto: Reconocimiento facial
@Módulo: face_recognizer
@Descripción: Realiza el reconocimiento facial
"""
import cv2
import os

def face_recognizer(folderPath):
    """
    Realiza el reconocimiento facial utilizando OpenCV.

    Parameters:
        folderPath (str): Ruta del directorio que contiene las imágenes de entrenamiento.
    """
    # Obtiene la lista de nombres de archivos de imágenes en el directorio especificado
    imagePaths = os.listdir(folderPath)
    print('imagePaths = ', folderPath)

    # Crea el objeto de reconocimiento facial utilizando el algoritmo LBPH
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

    # Lee el modelo entrenado previamente
    faceRecognizer.read('ModeloFaceFrontalData.xml')

    # Inicializa la captura de video desde la cámara
    cap = cv2.VideoCapture(0)  # También se puede utilizar cv2.CAP_DSHOW en Windows
    # Crea un objeto de clasificador de rostros utilizando el archivo haarcascade_frontalface_default.xml
    faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Bucle para capturar y procesar continuamente imágenes desde la cámara
    while True:
        # Captura un fotograma desde la cámara
        ret, frame = cap.read()

        # Sale del bucle si la captura no fue exitosa
        if ret == False:
            break

        # Convierte el fotograma a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Realiza la detección de rostros en el fotograma en escala de grises
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        # Bucle para procesar cada rostro detectado en el fotograma
        for (x, y, w, h) in faces:
            # Recorta y redimensiona la región del rostro
            rostro = gray[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            # Realiza la predicción del rostro utilizando el modelo entrenado
            result = faceRecognizer.predict(rostro)

            # Muestra la etiqueta de la predicción en el fotograma
            cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            # Si la confianza de la predicción es menor a 82, se considera una coincidencia
            if result[1] < 70:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                # Si la confianza es mayor, se considera como "Desconocido"
                cv2.putText(frame, 'Desconocido.', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Muestra el fotograma con las detecciones en una ventana llamada 'frame'
        cv2.imshow('frame', frame)

        # Espera una tecla durante 1 milisegundo y sale si se presiona la tecla Esc (código 27)
        k = cv2.waitKey(1)
        if k == 27:
            break

    # Libera la cámara y cierra todas las ventanas
    cap.release()
    cv2.destroyAllWindows()
