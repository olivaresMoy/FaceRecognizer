#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:14:37 2023
@Proyecto: Reconocimiento facial
@Módulo: entrenamiento
@Descripción: Realiza el entrenamiento para el reconoc facial
"""

from tkinter import messagebox
import cv2
import os
import numpy as np

def train(folderPath):
    # Almacena lista de personas guardadas en directorio
    peopleList = os.listdir(folderPath)
    # Mensaje con la lista de personas
    print('Lista de personas: ', peopleList)
    #vars arrays vacíos que guardan etiquetas e imagenes
    labels = []
    facesData = []
    # Contador label inicializado en 0
    label = 0
    
    for nameDir in peopleList:
        personPath = folderPath + '/' + nameDir
        
        print('Leyendo imágenes')
        
        for fileName in os.listdir(personPath):
            print('Rostros: ', nameDir + '/ ' + fileName)
            labels.append(label)
            
            facesData.append(cv2.imread(personPath + '/' + fileName, 0))

        label = label + 1
        
        #cv2.destroyAllWindows()
    
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    print('Entrenando modelo...')
    
    faceRecognizer.train(facesData, np.array(labels))
    faceRecognizer.write('ModeloFaceFrontalData.xml')
    print('Modelo guardado.')
    messagebox.showinfo("Warning", "Registro Completado")