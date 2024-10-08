# ALGORITMO GENETICO CON REALES PARA LA CORRECCION   
# DE IMAGENES CON LA FUNCION SIGMOIDE
import random
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Funcion para leer una imagen
def leer_imagen(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"La imagen en la ruta '{path}' no se encontro")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    img = img.astype(np.float64) / 255.0  # Normalizar la imagen [0, 1]
    return img

# Funcion Objetivo para Maximizar la Entropia de una imagen
def calcularAptitud(imagenOriginal, variables):
    alpha = variables[0]  # Factor de contraste
    delta = variables[1]  # Punto medio de la curva
    imagenConSigmoide = aplicarSigmoide(imagenOriginal, alpha, delta) # Primero aplicar la funcion sigmoide a la imagen original
    aptitud = calcularEntropia(imagenConSigmoide) # Calcular la entropia de la imagen con sigmoide generada anteriormente
    return - aptitud # Se cambia el signo de la aptitud para que en lugar de minimizar se haga la maximizacion de la FO

# Funcion para aplicar la Funcion Sigmoide a una imagen
def aplicarSigmoide(img, alpha, delta):
    output_img = 1 / (1 + np.exp(alpha * (delta - img)))
    output_img = cv2.normalize(output_img, None, 0, 1, cv2.NORM_MINMAX) # Reescalar la imagen [0, 1]
    return output_img

# Funcion para calcular la Entropia de Shanon
def calcularEntropia(img):   
    img_flat = img.flatten() # Convertir la imagen a un arreglo 1D
    hist_counts, _ = np.histogram(img_flat, bins=256, range=(0, 1), density=False) # Calcular el histograma
    total_pixels = np.sum(hist_counts) # Normalizar para obtener probabilidades
    probabilities = hist_counts / total_pixels
    probabilities = probabilities[probabilities > 0]
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities)) # Calcular la entropia
    return shannon_entropy

#############################################################################################################################################################
# PARAMETROS PARA LA FUNCION DEL ALGORITMO GENETICO
ruta_imagen = "C:/Users/S ALBERT FC/Documents/bones_hand_two.jpg"
imagenOriginal = leer_imagen(ruta_imagen)
li = [0, 0]
ls = [10, 1]
maxGeneracion = 40
numIndiv = 150
probaCruzam = 0.9
probaMutaci = 0.05
numVar = 2
nC = 2
nM = 20
#############################################################################################################################################################
# GENERAR POBLACION INICIAL
poblacion = []
for i in range(numIndiv):
    arregloVariables = [] # Arreglo para guardar los valores de las variables
    for j in range(numVar):
        rand = random.random()
        x = li[j] + (rand) * (ls[j] - li[j])
        arregloVariables.append(x)
    aptitud = calcularAptitud(imagenOriginal, arregloVariables) # Evaluacion de los individuos en FO
    poblacion.append((arregloVariables, aptitud))
#############################################################################################################################################################
# CICLO PRINCIPAL DEL ALGORITMO GENETICO
for generacion in range(maxGeneracion):
    print("Generacion Actual:", generacion + 1)
    #########################################################################################################################################################
    # SELECCION DE PADRES POR TORNEO 
    padres = [] # Arreglo para guardar a los padres seleccionados en el torneo
    a = random.sample(poblacion, len(poblacion)) # Hacer la primer permutacion
    b = random.sample(poblacion, len(poblacion)) # Hacer la segunda permutacion
    for i in range(len(poblacion)):
        aptitud_a = a[i][1] # Aptitud del primer individuo de la permutacion a
        aptitud_b = b[i][1] # Aptitud del primer individuo de la permutacion b
        if aptitud_a < aptitud_b:
            padres.append(a[i][0]) # Agregar los valores del primer individuo de la permutacion a como padre
        else:
            padres.append(b[i][0]) # Agregar los valores del primer individuo de la permutacion b como padre
    #########################################################################################################################################################
    # CRUZAMIENTO SBX
    hijos = [] # Arreglo para guardar a los hijos generador por los padres 
    randCruza = round(random.uniform(0, 1), 1)  # Generar un numero random para comparar con probaCruzam
    for i in range(0, len(padres), 2):  
        if randCruza <= probaCruzam:
            u = random.random()  
            hijo1 = []
            hijo2 = []
            for j in range(numVar):
                P1 = padres[i][j]
                P2 = padres[i + 1][j]

                if P2 != P1: # Mientras ambos padres no sean el mismo individuo
                    beta = 1 + (2 / (P2 - P1)) * min((P1 - li[j]), (ls[j] - P2)) 
                else:
                    beta = 1 # En caso de que la permutacion empareje al mismo individuo para el cruzamiento

                alpha = 2 - math.pow(abs(beta), -(nC + 1))
                if u <= 1 / alpha:
                    beta_C = math.pow(u * alpha, 1 / (nC + 1))
                else:
                    beta_C = math.pow(1 / (2 - u * alpha), 1 / (nC + 1))

                # Generar a los hijos
                hijo1_val = 0.5 * ((P1 + P2) - beta_C * abs(P2 - P1))
                hijo2_val = 0.5 * ((P1 + P2) + beta_C * abs(P2 - P1))
                        
                hijo1.append(hijo1_val)
                hijo2.append(hijo2_val)

            hijos.append(hijo1)  # Guardar el primer hijo
            hijos.append(hijo2)  # Guardar el segundo hijo
        else:
            # Si no se cumple se pasan los padres
            hijos.append(padres[i])
            hijos.append(padres[i + 1])
    #########################################################################################################################################################
    # MUTACION POLINOMIAL
    for i in range(len(hijos)):
        for j in range(numVar):
            randMutac = random.uniform(0, 1) # Generar un numero random para comparar con probaMutaci
            if randMutac <= probaMutaci: # Si se cumple la condicion para la mutacion
                r = round(random.uniform(0, 1), 1)
                delta = min(ls[j] - hijos[i][j], hijos[i][j] - li[j]) / (ls[j] - li[j])
                if r <= 0.5:
                    deltaQ = math.pow((2*r + (1 - 2*r)*math.pow((1 - delta), nM + 1)), 1/(nM+1)) - 1
                else:
                    deltaQ = 1 - math.pow(2*(1 - r) + 2 * (r - 0.5)*math.pow((1 - delta), nM + 1), 1/(nM+1))
                hijos[i][j] = hijos[i][j] + deltaQ*(ls[j] - li[j])
    #########################################################################################################################################################
    # ELITISMO
    aptitudesPoblacion = [individuo[1] for individuo in poblacion]  # Obtener las aptitudes de la poblacion actual
    indiceMenorAptitud = aptitudesPoblacion.index(min(aptitudesPoblacion)) # Obtener el indice del individuo con la menor aptitud
    individuoMenorAptitud = poblacion[indiceMenorAptitud] # Guardar al individuo con la menor aptitud
    #########################################################################################################################################################
    # EVALUACION DE LOS HIJOS Y SUSTITUCION
    nuevaPoblacion = [] # Arreglo para guardar a la nueva generacion
    for i in range(len(hijos)):
        aptitudHijo = calcularAptitud(imagenOriginal, hijos[i]) # Calcular la aptitud de los hijos
        nuevaPoblacion.append((hijos[i], aptitudHijo)) # Guardar a cada hijo con su aptitud
    indiceAleatorio = random.randint(0, len(nuevaPoblacion) - 1)  # Generar un indice aleatorio del individuo de la nuevaPoblacion para sustituir por Elitismo
    nuevaPoblacion[indiceAleatorio] = individuoMenorAptitud # Reemplazar el individuo aleatorio por el individuo de menor aptitud del Elitismo
  
    poblacion = nuevaPoblacion # Actualiza el cumulo con el cumuloFinal

#############################################################################################################################################################
# MOSTRAR EL RESULTADO FINAL DE TODAS LAS GENERACIONES
entropiaOriginal = calcularEntropia(imagenOriginal)

# Obtener los mejores valores de alfa y delta del mejor individuo
aptitudes = [individuo[1] for individuo in poblacion]
indice_MenorAptitud = aptitudes.index(min(aptitudes))
mejoresValores = poblacion[indice_MenorAptitud][0]
mejorAlfa = mejoresValores[0]
mejorDelta = mejoresValores[1]

# Aplicar la funcin sigmoide a la imagen original con los valores de alfa y delta
imagenResultante = aplicarSigmoide(imagenOriginal, mejorAlfa, mejorDelta)
entropiaMejorada = calcularEntropia(imagenResultante)

# Mostrar la imagen original y la imagen resultante
plt.figure(figsize=(10, 5))

# Imagen original
plt.subplot(1, 2, 1)
plt.imshow(imagenOriginal)
plt.title(f"Imagen Original\nValor de Entropia: {round(entropiaOriginal, 7)}")
plt.axis("off")

# Imagen con sigmoide
plt.subplot(1, 2, 2)
plt.imshow(imagenResultante)
plt.title(f"Imagen con Sigmoide\nValor de Entropia: {round(entropiaMejorada, 8)}\nAlfa: {round(mejorAlfa, 5)}, Delta: {round(mejorDelta, 5)}")
plt.axis("off")

# Mostrar el resultado
plt.show()
