#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Proyecto: Reconocimiento Facial
# Asignatura: PROGRAMACIÓN DE DISPOSITIVOS MOVILES
# Equipo: Luis Riano, Antonio Gallegos, Miguel Fajardo, Moises Olivares, Azarel Villanueva

import data_base
import entrenamiento
import reconocimiento
import tkinter as tk
import tkinter.messagebox as messagebox

folder_path = 'Data'

def guardar_imagenes_y_entrenar(person_name):
    # Llama a la función que guarda las imágenes
    data_base.save_images(folder_path, person_name)
    # Llama a la función de entrenamiento
    entrenamiento.train(folder_path)

def reconocer_usuario():
    # Llama a la función de reconocimiento facial
    reconocimiento.face_recognizer(folder_path)

def on_registrar_click(entry_nombre_usuario):
    person_name = entry_nombre_usuario.get()
    if person_name:
        guardar_imagenes_y_entrenar(person_name)
        # ventana_principal.destroy()
    else:
        messagebox.showinfo("Error", "Por favor, ingrese un nombre.")


# Crear la interfaz principal
ventana_principal = tk.Tk()
ventana_principal.geometry("866x768+200+10")
ventana_principal.title("Reconocimiento Facial")
ventana_principal.resizable(width=False, height=False)

# Imagen de fondo
imagen_fondo = tk.PhotoImage(file="inicio.png")
fondo_label = tk.Label(ventana_principal, image=imagen_fondo).place(x=0, y=0, relheight=1, relwidth=1)

# Campo de entrada para el nombre de usuario
entry_nombre_usuario = tk.Entry(ventana_principal, font=("Calisto MT", 12))
entry_nombre_usuario.place(x=90, y=210)

# Botón para registrar usuario
btn_registro = tk.Button(ventana_principal, text="Registrar Usuario", command=lambda: on_registrar_click(entry_nombre_usuario), bg="#fff", relief="flat", cursor="hand2", width=16, height=3,
                       font=("Calisto MT", 13))
btn_registro.place(x=140, y=345)

# Botón para reconocer usuario
btn_iniciar = tk.Button(ventana_principal, text="Reconocer Usuario", command=lambda: reconocer_usuario(), bg="#fff", relief="flat", cursor="hand2", width=16, height=3,
                       font=("Calisto MT", 13))
btn_iniciar.place(x=136, y=488)

ventana_principal.mainloop()
