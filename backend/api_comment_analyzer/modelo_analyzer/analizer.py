from asyncio import exceptions
import os
import pickle
import re
from googletrans import Translator
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from modelo_analyzer.consume_modelo import ClasificadorTexto
# from modelo_analyzer.a import predecir

class ModeloClass:
    def __init__(self):
        # instancio el modelo 
        self.modelo = ClasificadorTexto()
    

    def traductor(self, comentario):
        translator = Translator()

        # Verificar si el comentario es una cadena
        if not isinstance(comentario, str):
            return comentario

        # Eliminar caracteres especiales y convertir a minúsculas
        comentario = re.sub(r'[^a-zA-Z0-9\s]', '', comentario.lower())

        try:
            # Verificar si el comentario no está vacío
            if not comentario:
                return comentario

            # Intentar realizar la traducción
            traduccion = translator.translate(comentario, dest='en')

            # Verificar si la traducción fue exitosa
            if traduccion is not None and hasattr(traduccion, 'text'):
                return traduccion.text
            else:
                # Si no se pudo traducir, devolver el comentario original
                return comentario
        except Exception as e:
            # Manejar la excepción imprimiendo un mensaje
            print(f"Error al traducir: {e}")
            return comentario

        




    def predictor(self, comentario):

        # Analizar el sentimiento del comentario
        prediccion = self.modelo.predecir(comentario)
    
        # obtengo la respuesta de la prediccion en 0 y 1
        if prediccion == '0':
            prediccion = "Positivo"
        else:
            prediccion = "Negativo"
        return prediccion

    def analizar_comentarios(self, comentarios_data):
        if comentarios_data:
            comentarios = comentarios_data
            # Lista para almacenar comentarios y sus análisis de sentimiento
            comentarios_con_sentimiento = []

            # Itero a través de los comentarios de varios videos
            for video in comentarios:
                video_comentarios = video['comentario_data']
                # video_comentarios = self.traductor(video_comentarios)

                for comentario in video_comentarios:
                    img_autor = comentario['img_autor']
                    link_autor = comentario['link_autor']
                    autor = comentario["autor"]
                    comentario_texto = comentario["comentario"]
                    # comentario_texto = self.traductor(comentario["comentario"])
                    link_video = comentario['link_video']
                    # Crear un diccionario que almacena el comentario y su análisis de sentimiento
                    comentario_con_sentimiento = {
                        "img_autor": img_autor,
                        "link_autor": link_autor,
                        "autor": autor,
                        "comentario":  comentario_texto,
                        "link_video": link_video,
                        "sentimiento": self.predictor(comentario_texto)
                    }
                    comentarios_con_sentimiento.append(comentario_con_sentimiento)
            return comentarios_con_sentimiento
        else:
            print("No se recibieron comentarios de la API")
