class AutomataFinito():

    def __init__(self, tabla_transiciones={}, alfabeto=[], estado_inicial='', estados_finales=set()):
        self.tabla_transiciones = tabla_transiciones
        self.alfabeto = alfabeto
        self.estado_inicial = estado_inicial
        self.estados_finales = estados_finales

    def __str__(self):
        msg = 'alfabeto: ' + str(self.alfabeto) + '\n'
        msg += 'estado inicial: ' + str(self.estado_inicial) + '\n'
        msg += 'estados finales: ' + str(self.estados_finales) + '\n'
        msg += 'conjunto de estados: ' + str(list(self.tabla_transiciones.keys())) + '\n'
        msg += 'tabla de transiciones: ' + str(self.tabla_transiciones)
        return msg

    def acepta(self, palabra):

        estado_actual = self.estado_inicial

        for simbolo in palabra:
            if simbolo in self.alfabeto:
                estado_actual = self.tabla_transiciones[estado_actual][simbolo]
            else:
                return False
        return estado_actual in self.estados_finales

alfabeto = ['A', 'C', 'G', 'T']
estado_inicial = 'q10'
estados_finales = ['q7']
tabla_de_transiciones = {'q0': {'T': 'q12', 'A': 'q5', 'C': 'q5', 'G': 'q3'}, 'q1': {'A': 'q5', 'C': 'q5', 'G': 'q3', 'T': 'q1'}, 'q2': {'A': 'q5', 'C': 'q5', 'G': 'q11', 'T': 'q1'}, 'q3': {'A': 'q5', 'C': 'q5', 'G': 'q2', 'T': 'q1'}, 'q4': {'A': 'q5', 'C': 'q5', 'G': 'q5', 'T': 'q0'}, 'q5': {'A': 'q5', 'C': 'q5', 'G': 'q5', 'T': 'q1'}, 'q6': {'T': 'q1', 'G': 'q8', 'A': 'q8', 'C': 'q8'}, 'q7': {'A': 'q7', 'C': 'q7', 'T': 'q7', 'G': 'q7'}, 'q8': {'G': 'q8', 'A': 'q8', 'C': 'q8', 'T': 'q8'}, 'q9': {'G': 'q8', 'A': 'q5', 'C': 'q8', 'T': 'q8'}, 'q10': {'G': 'q8', 'A': 'q6', 'T': 'q9', 'C': 'q8'}, 'q11': {'C': 'q13', 'G': 'q5', 'T': 'q1', 'A': 'q5'}, 'q12': {'A': 'q5', 'C': 'q5', 'G': 'q3', 'T': 'q7'}, 'q13': {'C': 'q5', 'T': 'q1', 'G': 'q4', 'A': 'q5'}}


dfa = AutomataFinito(tabla_de_transiciones, alfabeto, estado_inicial, estados_finales)


def test_automata():
    ARCHIVO_DATOS_PRUEBAS_ACEPTABLES = 'aceptadas.txt'
    ARCHIVO_DATOS_PRUEBAS_DENEGABLES = 'denegadas.txt'

    aceptadas = 0
    totales = 0
    with open(ARCHIVO_DATOS_PRUEBAS_ACEPTABLES, 'r') as f:
        for palabra in f:
            totales += 1

            palabra = palabra.replace('\n', '')

            if dfa.acepta(palabra):
                aceptadas += 1

    print('(TEST) El automata ha aceptado {}'.format(aceptadas) + ' cadenas de {}'.format(totales) + ' totales')


    denegadas = 0
    totales = 0
    with open(ARCHIVO_DATOS_PRUEBAS_DENEGABLES, 'r') as f:
        for palabra in f:
            totales += 1

            palabra = palabra.replace('\n', '')

            if not dfa.acepta(palabra):
                denegadas += 1

    print('(TEST) El automata ha denegado {}'.format(denegadas) + ' cadenas de {}'.format(totales) + ' totales')


def ejecutar():
    ARCHIVO_DATOS = 'dataset.csv'

    aceptadas = 0
    listado_aceptadas = []
    totales = 0
    with open(ARCHIVO_DATOS, 'r') as f:
        for palabra in f:
            totales += 1

            palabra = palabra.replace('\n', '')

            if dfa.acepta(palabra):
                aceptadas += 1
                listado_aceptadas.append(palabra)

    print('El automata ha aceptado {}'.format(aceptadas) + ' cadenas de {}'.format(totales) + ' totales')
    print(listado_aceptadas)

if __name__ == '__main__':

    # Ejecuta pruebas sobre el automata
    #test_automata()

    # Usa la base de datos del campus
    ejecutar()