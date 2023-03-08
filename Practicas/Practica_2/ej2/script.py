from bs4 import BeautifulSoup
from functools import reduce
import string

"""alfabeto = {"H", "I", "D", "A", "L", "G", "O"}
estados = {"q0","q1","q2","q3","q4","q5","q6","q7"}
estado_inicial = "q0"
estados_finales = {"q3", "q5", "q7"}

funcion_transicion = {
    "q0": {
        "A": ["q1", "q0"]
    }
}"""


class estado:

    def __init__(self, id, nombre, es_inicial=False, es_final=False):
        self.id = id
        self.nombre = nombre
        self.es_inicial = es_inicial
        self.es_final = es_final


class transicion:

    def __init__(self, id_origen, id_destino, simbolo):
        self.id_origen = id_origen
        self.id_destino = id_destino
        self.simbolo = simbolo


def lee_xml(ruta_archivo):
    with open(ruta_archivo, 'r') as f:
        datos = f.read()

    return datos


def states_to_estados(states):
    estados = []
    for state in states:
        id = state.get('id')
        nombre = state.get('name')
        es_inicial = True if state.find("initial") else False
        es_final = True if state.find("final") else False

        estados.append(estado(id, nombre, es_inicial, es_final))

    return estados


def transtitions_to_transiciones(transitions):
    transiciones = []
    for transition in transitions:
        id_origen = transition.find_next('from').get_text()
        id_destino = transition.find_next('to').get_text()
        simbolo = transition.find_next('read').get_text()

        transiciones.append(transicion(id_origen, id_destino, simbolo))

    return transiciones


def obtener_alfabeto(transiciones):
    alfabeto = set()

    for transicion in transiciones:
        if transicion.simbolo not in alfabeto:
            alfabeto.add(transicion.simbolo)

    return set(list(alfabeto))


def obtener_estado_inicial(estados):
    for estado in estados:
        if estado.es_inicial:
            return estado.nombre

    return None


def obtener_estados_finales(estados):
    finales = set()

    for estado in estados:
        if estado.es_final:
            finales.add(estado.nombre)

    return set(list(finales))


def generar_funcion_transicion(transiciones):
    funcion_transicion = {}

    for transicion in transiciones:
        origen = 'q{}'.format(transicion.id_origen)
        destino = 'q{}'.format(transicion.id_destino)
        simbolo = '{}'.format(transicion.simbolo)

        # No estaba la etiqueta de origen en la funcion de transicion
        if origen not in funcion_transicion:
            funcion_transicion[origen] = {}

        # Guardamos la transicion
        funcion_transicion[origen][simbolo] = destino

    return funcion_transicion


def parsear_dfa_jflap(ruta_archivo):
    xml = BeautifulSoup(lee_xml(ruta_archivo), 'xml')

    states = xml.find_all('state')
    transitions = xml.find_all('transition')

    transiciones = transtitions_to_transiciones(transitions)
    estados = states_to_estados(states)

    estados_por_nombre = set([estado.nombre for estado in estados])
    alfabeto = obtener_alfabeto(transiciones)
    estado_inicial = obtener_estado_inicial(estados)
    estados_finales = obtener_estados_finales(estados)
    funcion_transicion = generar_funcion_transicion(transiciones)

    return (estados_por_nombre, alfabeto, estado_inicial, estados_finales, funcion_transicion)

def leer_palabras_test(ruta_archivo):

    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    palabras = []

    f = open(ruta_archivo, 'r', encoding="utf8")


    i = 1
    for linea in f:
        for palabra in linea.split():
            palabra = palabra.translate(str.maketrans('', '', string.punctuation))
            palabra = palabra.replace(u'\xad', ' ')
            palabras.append(palabra.upper())

        i+=1

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
        if test(palabra, estado_inicial, alfabeto, funcion_transicion, estados_finales):
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

    ruta_archivo_jflap = "./ej2_minimizado.jff"
    ruta_archivo_prueba = "./ElQuijote.txt"

    print('--- HACE FALTA INSTALAR UN PAR DE LIBRERÍAS PARA PARSEAR EL ARCHIVO JFLAP ---')

    # Obtiene toda la información del DFA creado en JFLAP
    (estados, alfabeto, estado_inicial, estados_finales, funcion_transicion) = parsear_dfa_jflap(ruta_archivo_jflap)

    # Leemos el archivo con las palabras
    palabras = leer_palabras_test(ruta_archivo_prueba)

    # Pasamos la primera parte (que todas las palabras tengan las letras que aparecen en el alfabeto)
    aceptadas_parte_1 = automata_parte_1(palabras, alfabeto)

    # Ahora, para la segunda parte, empleamos el DFA desarrollado en JFLAP
    aceptadas_parte_2 = automata_parte_2(aceptadas_parte_1, alfabeto, estado_inicial, estados_finales, funcion_transicion)

    print('Palabras que han pasado el autómata: ' + str(len(aceptadas_parte_2)))