
import string
from functools import reduce



alfabeto = {'A', 'G', 'D', 'L', 'H', 'I', 'O'}
estado_inicial = 'q0'
estados_finales = {'q31', 'q30', 'q28', 'q4', 'q27', 'q26', 'q19', 'q25', 'q32', 'q34', 'q12', 'q23', 'q17', 'q18', 'q24', 'q11', 'q21', 'q29', 'q35', 'q7', 'q10', 'q33', 'q20', 'q15'}
funcion_transicion = {'q25': {'D': 'q30', 'A': 'q25', 'G': 'q25', 'H': 'q25', 'I': 'q25', 'O': 'q25', 'L': 'q31'}, 'q22': {'D': 'q27', 'L': 'q29', 'G': 'q22', 'H': 'q22', 'I': 'q22', 'O': 'q22', 'A': 'q30'}, 'q23': {'D': 'q28', 'L': 'q30', 'A': 'q23', 'G': 'q23', 'H': 'q23', 'I': 'q23', 'O': 'q23'}, 'q11': {'L': 'q19', 'A': 'q20', 'D': 'q11', 'G': 'q11', 'H': 'q11', 'I': 'q11', 'O': 'q11'}, 'q28': {'L': 'q33', 'A': 'q28', 'D': 'q28', 'G': 'q28', 'H': 'q28', 'I': 'q28', 'O': 'q28'}, 'q14': {'A': 'q23', 'G': 'q14', 'H': 'q14', 'I': 'q14', 'O': 'q14', 'D': 'q20', 'L': 'q22'}, 'q16': {'L': 'q24', 'G': 'q16', 'H': 'q16', 'I': 'q16', 'O': 'q16', 'D': 'q22', 'A': 'q25'}, 'q0': {'L': 'q2', 'G': 'q0', 'H': 'q0', 'I': 'q0', 'O': 'q0', 'D': 'q1', 'A': 'q3'}, 'q2': {'D': 'q5', 'G': 'q2', 'H': 'q2', 'I': 'q2', 'O': 'q2', 'L': 'q7', 'A': 'q8'}, 'q10': {'L': 'q18', 'D': 'q10', 'G': 'q10', 'H': 'q10', 'I': 'q10', 'O': 'q10', 'A': 'q19'}, 'q9': {'D': 'q14', 'G': 'q9', 'H': 'q9', 'I': 'q9', 'O': 'q9', 'A': 'q17', 'L': 'q16'}, 'q6': {'A': 'q14', 'G': 'q6', 'H': 'q6', 'I': 'q6', 'O': 'q6', 'L': 'q13', 'D': 'q11'}, 'q30': {'D': 'q33', 'A': 'q30', 'G': 'q30', 'H': 'q30', 'I': 'q30', 'O': 'q30', 'L': 'q34'}, 'q4': {'A': 'q11', 'D': 'q4', 'G': 'q4', 'H': 'q4', 'I': 'q4', 'O': 'q4', 'L': 'q10'}, 'q24': {'D': 'q29', 'A': 'q31', 'G': 'q24', 'H': 'q24', 'I': 'q24', 'L': 'q24', 'O': 'q24'}, 'q17': {'A': 'q17', 'D': 'q23', 'G': 'q17', 'H': 'q17', 'I': 'q17', 'O': 'q17', 'L': 'q25'}, 'q35': {'A': 'q35', 'D': 'q35', 'G': 'q35', 'H': 'q35', 'I': 'q35', 'L': 'q35', 'O': 'q35'}, 'q31': {'A': 'q31', 'G': 'q31', 'H': 'q31', 'I': 'q31', 'L': 'q31', 'O': 'q31', 'D': 'q34'}, 'q33': {'A': 'q33', 'D': 'q33', 'G': 'q33', 'H': 'q33', 'I': 'q33', 'O': 'q33', 'L': 'q35'}, 'q34': {'A': 'q34', 'G': 'q34', 'H': 'q34', 'I': 'q34', 'L': 'q34', 'O': 'q34', 'D': 'q35'}, 'q29': {'D': 'q32', 'G': 'q29', 'H': 'q29', 'I': 'q29', 'L': 'q29', 'O': 'q29', 'A': 'q34'}, 'q32': {'D': 'q32', 'G': 'q32', 'H': 'q32', 'I': 'q32', 'L': 'q32', 'O': 'q32', 'A': 'q35'}, 'q18': {'D': 'q18', 'G': 'q18', 'H': 'q18', 'I': 'q18', 'L': 'q18', 'O': 'q18', 'A': 'q26'}, 'q19': {'D': 'q19', 'G': 'q19', 'H': 'q19', 'I': 'q19', 'O': 'q19', 'A': 'q27', 'L': 'q26'}, 'q20': {'D': 'q20', 'G': 'q20', 'H': 'q20', 'I': 'q20', 'O': 'q20', 'L': 'q27', 'A': 'q28'}, 'q27': {'D': 'q27', 'G': 'q27', 'H': 'q27', 'I': 'q27', 'O': 'q27', 'L': 'q32', 'A': 'q33'}, 'q26': {'D': 'q26', 'G': 'q26', 'H': 'q26', 'I': 'q26', 'L': 'q26', 'O': 'q26', 'A': 'q32'}, 'q21': {'G': 'q21', 'H': 'q21', 'I': 'q21', 'L': 'q21', 'O': 'q21', 'D': 'q26', 'A': 'q29'}, 'q3': {'G': 'q3', 'H': 'q3', 'I': 'q3', 'O': 'q3', 'L': 'q8', 'D': 'q6', 'A': 'q9'}, 'q13': {'G': 'q13', 'H': 'q13', 'I': 'q13', 'O': 'q13', 'L': 'q21', 'D': 'q19', 'A': 'q22'}, 'q1': {'G': 'q1', 'H': 'q1', 'I': 'q1', 'O': 'q1', 'A': 'q6', 'D': 'q4', 'L': 'q5'}, 'q7': {'G': 'q7', 'H': 'q7', 'I': 'q7', 'L': 'q7', 'D': 'q12', 'O': 'q7', 'A': 'q15'}, 'q15': {'G': 'q15', 'H': 'q15', 'I': 'q15', 'L': 'q15', 'O': 'q15', 'A': 'q24', 'D': 'q21'}, 'q12': {'G': 'q12', 'H': 'q12', 'I': 'q12', 'L': 'q12', 'O': 'q12', 'D': 'q18', 'A': 'q21'}, 'q5': {'G': 'q5', 'H': 'q5', 'I': 'q5', 'O': 'q5', 'D': 'q10', 'A': 'q13', 'L': 'q12'}, 'q8': {'G': 'q8', 'H': 'q8', 'I': 'q8', 'O': 'q8', 'D': 'q13', 'A': 'q16', 'L': 'q15'}}


def leer_palabras_test(ruta_archivo):

    palabras = []

    # Abrimos el archivo en modo lectura
    f = open(ruta_archivo, 'r', encoding="utf8")

    # Recorremos cada linea y sacamos las palabras sin signos de puntuacion
    for linea in f:
        for palabra in linea.split():
            palabra = palabra.translate(str.maketrans('', '', string.punctuation))
            palabra = palabra.replace(u'\xad', '')
            palabra = palabra.replace('¡', '')
            palabra = palabra.replace('¿', '')
            palabras.append(palabra.upper())

    f.close()

    return palabras

def automata_parte_1(palabras, alfabeto):
    aceptadas = []

    for palabra in palabras:
        letras = [c for c in palabra]

        # Filtramos las letras que no estan en el alfabeto
        no_contenidas = lambda suma, letra: suma + 1 if letra not in alfabeto else suma
        cuenta_no_contenidas = reduce(no_contenidas, letras, 0)

        # Filtamos aquellas que tengan todas las letras del alfabeto
        if cuenta_no_contenidas == 0:
            aceptadas.append(palabra)

    return aceptadas

def test(palabra, estado_actual, alfabeto, funcion_transicion, estados_finales):

    estado = estado_actual

    for letra in palabra:
        if letra in alfabeto:
            estado = funcion_transicion[estado][letra]

    return estado in estados_finales

def automata_parte_2(palabras, alfabeto, estado_inicial, estados_finales, funcion_transicion):

    aceptadas = []

    for palabra in palabras:
        if not test(palabra, estado_inicial, alfabeto, funcion_transicion, estados_finales):
            aceptadas.append(palabra)

    return aceptadas

if __name__ == "__main__":

    """
    Este script hace uso del DFA desarrollado con JFLAP para ahorrar tiempo codificando manualmente:
    la funcion de transicion, estados, estados finales...
    
    HACE FALTA INSTALAR DOS LIBRERÍAS PARA PARSEAR EL ARCHIVO JFLAP (es un archivo xml):
    pip install lxml
    pip install beautifulsoup4
    """

    ruta_archivo_jflap = "./ej2_dfa_minimizado.jff"
    ruta_archivo_prueba = "./ElQuijote.txt"

    # Leemos el archivo con las palabras
    palabras = leer_palabras_test(ruta_archivo_prueba)

    # Pasamos la primera parte (que todas las palabras tengan las letras que aparecen en el alfabeto)
    aceptadas_parte_1 = automata_parte_1(palabras, alfabeto)

    # Ahora, para la segunda parte, empleamos el DFA desarrollado en JFLAP
    aceptadas_parte_2 = automata_parte_2(aceptadas_parte_1, alfabeto, estado_inicial, estados_finales, funcion_transicion)

    """print(aceptadas_parte_1)
    print(len(aceptadas_parte_1))
    print(aceptadas_parte_2)"""
    print('Número de palabras que han sido aceptadas por el autómata: ' + str(len(aceptadas_parte_2)))