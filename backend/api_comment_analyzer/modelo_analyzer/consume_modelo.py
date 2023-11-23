import os
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class ClasificadorTexto:
    def __init__(self, ):
        self.ruta_relativa = 'modelo_analyzer/mymodelo'
        self.ruta_absoluta =os.path.abspath(self.ruta_relativa)
        self.model_directory = self.ruta_absoluta.replace('\\', '/')
        # print(self.model_directory)
        # self.model_directory="C:/Users/maikol/Desktop/curso inteligencia artificial/proyectos grupales/Proyecto_NLP_Grupo_3/principal/backend/api_comment_analyzer/modelo_analyzer/mymodelo"
        self.model = DistilBertForSequenceClassification.from_pretrained(str(self.model_directory))
        self.tokenizer = DistilBertTokenizer.from_pretrained(str(self.model_directory))

        

    def predecir(self, texto):
        # Tokenizar el texto
        texto = texto
        tokens = self.tokenizer(texto, return_tensors="pt")

        # Realizar la predicción
        outputs = self.model(**tokens)

        # Obtener las probabilidades de predicción
        probabilities = outputs.logits.softmax(dim=1)

        # Obtener la clase predicha (la que tiene la probabilidad más alta)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Obtener el nombre de la clase predicha
        class_names = ["0", "1"]  # Reemplaza con tus nombres de clases reales
        predicted_class_name = class_names[predicted_class]

        # Devolver la clase predicha
        return predicted_class_name
    


clasificador = ClasificadorTexto()
texto_a_predecir = "fuck"
prediccion = clasificador.predecir(texto_a_predecir)
print("Predicción:", prediccion)