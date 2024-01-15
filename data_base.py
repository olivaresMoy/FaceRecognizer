#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:51:17 2023
@Proyecto: Reconocimiento facial
@Módulo: save_image
@Descripción: Guarda data en folder
"""

# Importación de bibliotecas necesarias
from tkinter import messagebox
import cv2
import imutils
import os
import tkinter as tk
from PIL import Image, ImageTk

# Función que crea un directorio si no existe
def crear_carpeta(path_parameter):
    if not os.path.exists(path_parameter):
        print('Carpeta creada', path_parameter)
        os.makedirs(path_parameter)
        
# Función principal que guarda las imágenes del usuario
def save_images(folder_path, person_name):
    
    # Creación del path de la carpeta del usuario
    person_path = folder_path + '/' + person_name
    # Llamada a la función para crear la carpeta
    crear_carpeta(person_path)
    
    # Inicialización de la captura de video
    cap = cv2.VideoCapture(0) #, cv2.CAP_DSHOW
    # Carga del clasificador de rostros
    face_classif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Inicialización del contador
    count = 0
    
    # Bucle para almacenar las imágenes del usuario
    while True:
        # Captura de video
        ret, frame = cap.read()
        # Condición de salida si la captura no tiene éxito
        if ret is False:
            break
        
        # Llamada a la función Iniciar
        frame = imutils.resize(frame, width=866)
        
        # Conversión de la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Copia de la imagen original
        aux_frame = frame.copy()
        # Detección de rostros en la imagen en escala de grises
        faces = face_classif.detectMultiScale(gray, 1.3, 5)
        
        # Bucle para crear rectángulos alrededor de los rostros detectados
        for (x, y, w, h) in faces:
            # Creación de un rectángulo en la imagen original
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Recorte y redimensionamiento de la región del rostro
            rostro = aux_frame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
            # Guardado de la imagen del rostro en el directorio en formato JPG
            cv2.imwrite(person_path + '/rostro_{}.jpg'.format(count), rostro)
            # Incremento del contador en cada iteración
            count += 1
    
        # Mostrar la imagen con los rectángulos
        cv2.imshow('frame', frame)
        
        # Salir si se presiona la tecla escape o se alcanza el límite de imágenes
        k = cv2.waitKey(1)
        if k == 27 or count >= 200:
            break
    
    # Liberar la captura de video
    cap.release()
    
    messagebox.showinfo("Warning", "Registrando datos, espere.")
    # Cerrar todas las ventanas de visualización
    cv2.destroyAllWindows()
