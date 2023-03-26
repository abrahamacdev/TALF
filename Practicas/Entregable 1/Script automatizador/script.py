import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import functools

"""
El archivo a parsear tiene que estar separado por tabulaciones.

Aquellos símbolos para los que no exista una función de transición dado un determinado estado, en los datos debe de aparecer
como un punto (.) o un espacio ( ).

La primera línea indicará:
    - Primera 	columna: Puede estar vacía o contener algo (se ignorará).
    - 2..(N-2) 	columa: Símbolos que componen el alfabeto.
    - N-1 		columna: Texto 'eps' que indicará las transiciones epsilon.
    - N			columna: {-1, 0, 1} que indica si el estado es (en orden): inicial, 'normal', final

El resto de líneas irán en consonancia. Ejemplo:
    q 	A 	B 	eps
    q0	.	q1	q2
    q1	.	.	.
    q2	q0	q1	q1

    En este ejemplo, el alfabeto está compuesto por sólo dos símbolos: A y B. Los posibles estados son: q0, q1 y q2.
    La última columna, 'eps', indica las transiciones epsilon.
"""

ARCHIVO_INICIAL_EPS_NFA = 'tabla_con_epsilons_inicial.txt'
ARCHIVO_INICIAL_DFA = 'tabla_dfa_inicial.txt'


ARCHIVO_CLAUSURAS = 'tabla_clausuras_script.txt'
ARCHIVO_NFA = 'tabla_nfa_script.txt'
ARCHIVO_DFA = 'tabla_dfa_script.txt'
ARCHIVO_DFA_MINIMIZADO = 'tabla_dfa_minimizado_script.txt'

ARCHIVO_TEST_ACEPTADAS_AUTOMATA_1 = './test_automata_1/aceptadas.txt'
ARCHIVO_TEST_DENEGADAS_AUTOMATA_1 = './test_automata_1/denegadas.txt'

ARCHIVO_TEST_ACEPTADAS_AUTOMATA_2 = './test_automata_2/aceptadas.txt'
ARCHIVO_TEST_DENEGADAS_AUTOMATA_2 = './test_automata_2/denegadas.txt'

ARCHIVO_TEST_ACEPTADAS_AUTOMATA_3 = './test_automata_3/aceptadas.txt'
ARCHIVO_TEST_DENEGADAS_AUTOMATA_3 = './test_automata_3/denegadas.txt'

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

class Particion():

    def __init__(self, n):
        self.__arbol = np.ones((n,), dtype=int) * -1

    def unir(self, n1, n2):

        # n1 es más alto
        if self.__arbol[n1] < self.__arbol[n2]:
            self.__arbol[n2] = n1

        # n2 es más alto
        elif self.__arbol[n1] > self.__arbol[n2]:
            self.__arbol[n1] = n2

        # Son iguales, da igual la unión
        else:
            self.__arbol[n2] = n1
            self.__arbol[n1] -= 1

    def encontrar(self, n1):
        n = n1
        temp = self.__arbol[n1]

        while (temp >= 0):
            n = temp
            temp = self.__arbol[temp]

        return n


# --- Lee el archivo inicial y extrae la información ---
def lee_eps_nfa_de_archivo(archivo):
    tabla_transiciones = {}
    alfabeto = []
    estado_inicial = []
    estados_finales = set()

    primera = True
    with open(archivo, 'r') as f:

        # Leemos cada línea
        for linea in f:
            linea = linea.replace('\n', '').split('\t')

            # Guardamos el alfabeto
            if primera:
                primera = False
                alfabeto = linea[1:len(linea) - 2]

            # Guardamos los datos de cada estado
            else:
                parsea_datos_eps_nfa(tabla_transiciones, alfabeto, linea, estado_inicial, estados_finales)

    return [tabla_transiciones, alfabeto, estado_inicial[0], estados_finales]

def parsea_datos_eps_nfa(tabla_transiciones, alfabeto, columnas, estados_iniciales, estados_finales):
    """
    Procesa el archivo recibido por parámetros con la tabla de eps-transiciones.

    :param tabla_transiciones: Diccionario que servirá para guardar las distintas transicciones.
    :param alfabeto: Listado con los símbolos que componen el alfabeto. El orden será igual que el de la primera línea del archivo.
    :param columnas: Vector con los datos del estado a procesar.
    :param estados_iniciales: Set con el conjunto de estados iniciales
    :param estados_finales: Set con el conjunto de estados finales
    :return:
    """

    estado = columnas[0]
    transiciones = columnas[1:len(columnas) - 2]
    transiciones_epsilon = columnas[len(columnas) - 2]
    tipo_estado = columnas[len(columnas) - 1]

    # Creamos un diccionario para el estado
    if estado not in tabla_transiciones:
        tabla_transiciones[estado] = {}

    # Añadiremos las transiciones para cada símbolo del alfabeto
    for indx_transicion in range(len(transiciones)):
        simbolo = alfabeto[indx_transicion]
        estados_transicion_simbolo = transiciones[indx_transicion]

        # No había ninguna transición para dicho símbolo
        if len(estados_transicion_simbolo) == 0 or estados_transicion_simbolo == '.':
            tabla_transiciones[estado][simbolo] = set()

        # Había una o más transiciones para dicho símbolo
        else:
            tabla_transiciones[estado][simbolo] = set(estados_transicion_simbolo.split(','))

    # No hay transiciones epsilon
    if len(transiciones_epsilon) == 0 or transiciones_epsilon == '.':
        tabla_transiciones[estado]['eps'] = set()

    # Había una o más transiciones epsilon
    else:
        tabla_transiciones[estado]['eps'] = set(transiciones_epsilon.split(','))

    # El estado es inicial
    if tipo_estado == '-1':
        estados_iniciales.append(estado)

    # El estado es final
    elif tipo_estado == '1':
        estados_finales.add(estado)

def lee_dfa_de_archivo(archivo):
    """
    Lee la tabla de transiciones de un DFA de un archivo delimitado por tabulaciones.
    :param archivo:
    :return:
    """

    tabla_transiciones = {}
    alfabeto = []
    estado_inicial = []
    estados_finales = set()

    primera = True
    with open(archivo, 'r') as f:

        # Leemos cada línea
        for linea in f:
            linea = linea.replace('\n', '').split('\t')

            # Guardamos el alfabeto
            if primera:
                primera = False
                alfabeto = linea[1:len(linea) - 1]

            # Guardamos los datos de cada estado
            else:
                parsea_datos_dfa(tabla_transiciones, alfabeto, linea, estado_inicial, estados_finales)

    return [tabla_transiciones, alfabeto, estado_inicial[0], estados_finales]

def parsea_datos_dfa(tabla_transiciones, alfabeto, columnas, estados_iniciales, estados_finales):
    """
    Procesa el archivo recibido por parámetros con la tabla de eps-transiciones.

    :param tabla_transiciones: Diccionario que servirá para guardar las distintas transicciones.
    :param alfabeto: Listado con los símbolos que componen el alfabeto. El orden será igual que el de la primera línea del archivo.
    :param columnas: Vector con los datos del estado a procesar.
    :param estados_iniciales: Listado en el que guardaremos el estado inicial.
    :param estados_finales: Set con el conjunto de estados finales
    :return:
    """

    estado = columnas[0]
    transiciones = columnas[1:len(columnas) - 1]
    tipo_estado = columnas[len(columnas) - 1]

    # Creamos un diccionario para el estado
    if estado not in tabla_transiciones:
        tabla_transiciones[estado] = {}

    # Añadiremos las transiciones para cada símbolo del alfabeto
    for indx_transicion in range(len(transiciones)):
        simbolo = alfabeto[indx_transicion]
        tabla_transiciones[estado][simbolo] = transiciones[indx_transicion]

    # El estado es inicial
    if tipo_estado == '-1':
        estados_iniciales.append(estado)

    # El estado es final
    elif tipo_estado == '1':
        estados_finales.add(estado)
# ------------------------------------------------------


# --- Obtentión de cierres de eps-NFA ---
def anade_clausura_rec(tabla_transiciones, listadoEps, listadoClausura):
    """
    Genera la tabla de clausura de manera recursiva.

    :param tabla_transiciones: Tabla con las transiciones de cada estado.
    :param listadoEps: Listado con los estados a visitar de manera recursiva.
    :param listadoClausura: Set final con las clausuras.
    :return:
    """

    if len(listadoEps) == 0:
        return listadoClausura

    else:
        actual = listadoEps.pop()  # Cogemos el primero del listado y lo eliminamos

        # Evitamos recursiones infinitas
        if actual not in listadoClausura:

            listadoClausura.append(actual)  # Añadimos el estado a la clausura
            listadoEps.extend(list(tabla_transiciones[actual]['eps']))

            # Procesamos recursivamente
            return anade_clausura_rec(tabla_transiciones, listadoEps, listadoClausura)

        else:
            return listadoClausura


def genera_clausuras(tabla_transiciones):
    clausuras = {}

    for estado in tabla_transiciones.keys():
        listado_eps = [estado]
        clausuras[estado] = anade_clausura_rec(tabla_transiciones, listado_eps, [])

    return clausuras


def guarda_clausuras(clausuras, archivo):
    """
    Crea un archivo con las clausuras de cada estado en el mismo orden que en el archivo inicial.

    :param clausuras:
    :param archivo:
    :return:
    """

    with open(archivo, 'w') as f:

        for estado in clausuras.keys():
            msg = '.'

            # Hay estados
            if len(clausuras[estado]) > 0:
                msg = ''
                n = len(clausuras[estado])
                clausuras_ordenadas = sorted(clausuras[estado], key=functools.cmp_to_key(comparar_estados))
                for indx in range(n):
                    msg += clausuras_ordenadas[indx]

                    if indx < n - 1:
                        msg += ','

                # Escribimos en el archivo
                msg += '\n'
                f.write(msg)


# ---------------------------------------


# --- Obtiene NFA a partir de eps-NFA y sus cierres ---
def genera_nfa(automata_finito: AutomataFinito, clausuras):
    columnas_df = ['estados']
    columnas_df.extend(automata_finito.alfabeto)

    df = pd.DataFrame(columns=columnas_df)

    # Creamos el df con todos los datos
    for estado in automata_finito.tabla_transiciones.keys():
        tupla = {
            'estados': estado,
        }
        for simbolo in automata_finito.alfabeto:
            tupla[simbolo] = list(automata_finito.tabla_transiciones[estado][simbolo])

        df = df.append(tupla, ignore_index=True)

    # Vamos haciendo las transiciones
    nueva_tabla_transiciones = {}
    for estado in automata_finito.tabla_transiciones.keys():
        nueva_tabla_transiciones[estado] = {}
        for simbolo in automata_finito.alfabeto:
            nueva_tabla_transiciones[estado][simbolo] = np.concatenate(
                df.loc[df['estados'].isin(clausuras[estado])][simbolo].values).ravel().tolist()

    # Calculamos los nuevos estados finales
    nuevos_estados_finales = set()
    for estado in clausuras:
        for siguiente_estado in clausuras[estado]:
            if siguiente_estado in automata_finito.estados_finales:
                nuevos_estados_finales.add(estado)

    return AutomataFinito(nueva_tabla_transiciones, automata_finito.alfabeto, automata_finito.estado_inicial, nuevos_estados_finales)

def guarda_nfa(nfa: AutomataFinito, archivo):
    with open(archivo, 'w') as f:

        # Escribimos la cabecera del archivo
        cabecera = 'q\t'
        for simbolo in nfa.alfabeto:
            cabecera += simbolo + '\t'

        cabecera += 'tipo_estado'
        cabecera += '\n'
        f.write(cabecera)

        for estado in nfa.tabla_transiciones.keys():
            msg = estado + '\t'

            # Hay estados
            for simbolo in nfa.alfabeto:
                if len(nfa.tabla_transiciones[estado][simbolo]) > 0:
                    msg += ','.join(nfa.tabla_transiciones[estado][simbolo])

                else:
                    msg += '.'


                msg += '\t'

            # Añadimos el tipo de estado
            es_inicial = estado == nfa.estado_inicial
            es_final = estado in nfa.estados_finales
            if es_inicial and not es_final:
                msg += '-1'
            elif es_final and not es_inicial:
                msg += '1'
            elif es_inicial and es_final:
                msg += '2'
            else:
                msg += '0'

            # Escribimos en el archivo
            msg += '\n'

            f.write(msg)
# -----------------------------------------------------


# --- NFA->DFA ---
def componer_estado(estados):
    nuevo_estado = 'q'

    estados_ordenados = sorted(estados, key=functools.cmp_to_key(comparar_estados))

    for estado in estados_ordenados:
        nuevo_estado += estado.replace('q', '') + '_'

    return nuevo_estado[0:len(nuevo_estado) - 1]


def compuesto_por_estados_finales(conjunto_componedor, estados_finales):
    for estado in estados_finales:
        if estado in conjunto_componedor:
            return True

    return False


def comparar_estados(q1, q2):
    return int(q1[1:len(q1)]) - int(q2[1:len(q2)])


def procesa_paso_nfa_2_dfa(automata_finito, nueva_tabla_transiciones, nuevos_estados_finales, cola_procesamiento):
    """

    :param automata_finito:
    :param nueva_tabla_transiciones:
    :param nuevos_estados_finales: Set vació sobre el que se irán añadiendo los nuevos estados finales.
    :param cola_procesamiento:
    :return:
    """

    # No hay nada más que procesar
    if len(cola_procesamiento) == 0:
        return [nueva_tabla_transiciones, nuevos_estados_finales]

    else:

        estado_actual = cola_procesamiento.pop(0)  # Estado que procesaremos. Puede ser simple (q1) o compuesto (q2_3_4)
        estado_actual_composicion = estado_actual.split(
            '_')  # Obtenemos los estados que lo componen (para saber si es simple o no)

        # El estado aun no ha sido explorado
        if estado_actual not in nueva_tabla_transiciones:

            # Inicializamos el diccionario
            nueva_tabla_transiciones[estado_actual] = {}

            # Es un estado 'simple'
            if len(estado_actual_composicion) == 1:

                # Recorremos cada símbolo del alfabeto
                for simbolo in automata_finito.tabla_transiciones[estado_actual]:
                    siguientes_estados = automata_finito.tabla_transiciones[estado_actual][simbolo]
                    n = len(siguientes_estados)

                    nuevo_estado = siguientes_estados[0]

                    # Sólo hay transicion a un estado
                    if n == 1:
                        cola_procesamiento.append(siguientes_estados[0])
                        nueva_tabla_transiciones[estado_actual][simbolo] = siguientes_estados[0]

                    # Hay transicion a multiples estados
                    else:
                        nuevo_estado = componer_estado(
                            siguientes_estados)  # Componemos los estados. Ej: q1,q2,q3 = q1_2_3
                        cola_procesamiento.append(nuevo_estado)  # Añadimos el nuevo estado compuesto a la cola
                        nueva_tabla_transiciones[estado_actual][
                            simbolo] = nuevo_estado  # Guardamos la transicion al nuevo estado compuesto

                    # El nuevo estado será también final
                    if compuesto_por_estados_finales(siguientes_estados, automata_finito.estados_finales):
                        nuevos_estados_finales.add(nuevo_estado)


            # Es un estado 'compuesto'
            else:

                # Añadimos una 'q' a cada 'numerito' para que sean de la forma 'qx'. Ejemplo: q1_2_3 = q1 q2 q3
                for i in range(1, len(estado_actual_composicion)):
                    estado_actual_composicion[i] = 'q' + estado_actual_composicion[i]

                # Recorremos cada símbolo del alfabeto
                for simbolo in automata_finito.alfabeto:

                    # Set en el que iremos metiendo todos los estados para el símbolo #simbolo
                    estados_transicion_compuestos = set()

                    # Recorremos cada uno de los estados que componen el estado actual
                    for estado_que_compone in estado_actual_composicion:

                        # Hay una transicion para el símbolo
                        if simbolo in automata_finito.tabla_transiciones[estado_que_compone]:

                            # Estados a los que transita el #estado_que_compone dado el símbolo #simbolo
                            siguientes_estados = automata_finito.tabla_transiciones[estado_que_compone][simbolo]

                            # Añadimos los estados para luego componerlos en uno solo
                            for s in siguientes_estados:
                                estados_transicion_compuestos.add(s)

                    # Hay transiciones con el símbolo
                    if len(estados_transicion_compuestos) > 0:

                        # Componemos un nuevo estado 'multiple'
                        nuevo_estado = componer_estado(estados_transicion_compuestos)

                        # Añadimos el nuevo estado compuesto a la cola de procesamiento
                        cola_procesamiento.append(nuevo_estado)

                        # Añadimos el nuevo estado compuesto como transición para el estado compuesto actual con el simbolo #simbolo
                        nueva_tabla_transiciones[estado_actual][simbolo] = nuevo_estado

                        # El nuevo estado compuesto será también final
                        if compuesto_por_estados_finales(estados_transicion_compuestos,
                                                         automata_finito.estados_finales):
                            nuevos_estados_finales.add(nuevo_estado)

            return procesa_paso_nfa_2_dfa(automata_finito, nueva_tabla_transiciones, nuevos_estados_finales,
                                          cola_procesamiento)

        # El estado ya se exploró
        else:
            return procesa_paso_nfa_2_dfa(automata_finito, nueva_tabla_transiciones, nuevos_estados_finales,
                                          cola_procesamiento)


def generar_traduccion(tabla_transiciones, alfabeto, estados_finales, estado_inicial):
    traducciones = {}

    nueva_tabla_transiciones = {}
    nuevos_estados_finales = set()

    q_gradual = 0
    q_muerto = 'q_'
    hay_transiciones_a_estado_muerto = False

    # Los estados por procesar
    cola_procesamiento = [estado_inicial]
    conjunto_procesados = set()

    # Traducimos la tabla de transiciones
    while len(cola_procesamiento) > 0:

        # Cogemos el primer elemento de la cola
        estado = cola_procesamiento.pop(0)
        conjunto_procesados.add(estado)

        for simbolo in alfabeto:

            # Transicion con el simbolo y estado actual
            estado_siguiente = tabla_transiciones[estado][simbolo]

            traduccion_actual = None

            # Guardamos el siguiente estado para procesarlo si no se proceso antes
            if estado_siguiente not in conjunto_procesados:
                cola_procesamiento.append(estado_siguiente)


            # Ya se ha traducido antes
            if estado in traducciones:
                traduccion_actual = traducciones[estado]

            # No se ha traducido, lo hacemos ahora
            else:

                # Creamos su traduccion
                traduccion_actual = 'q{}'.format(q_gradual)
                traducciones[estado] = traduccion_actual
                q_gradual += 1
                nueva_tabla_transiciones[traduccion_actual] = {}

            # Hay una transicion
            if simbolo in tabla_transiciones[estado]:
                estado_siguiente = tabla_transiciones[estado][simbolo]

                traduccion_siguiente = None

                # Ya se ha traducido antes
                if estado_siguiente in traducciones:
                    traduccion_siguiente = traducciones[estado_siguiente]

                # No se ha traducido, lo hacemos ahora
                else:
                    traduccion_siguiente = 'q{}'.format(q_gradual)
                    traducciones[estado_siguiente] = traduccion_siguiente
                    q_gradual += 1
                    nueva_tabla_transiciones[traduccion_siguiente] = {}

                # Guardamos en la nueva tabla la transicion con las traducciones
                nueva_tabla_transiciones[traduccion_actual][simbolo] = traduccion_siguiente

            # No hay transicion -> va a estado muerto
            else:
                hay_transiciones_a_estado_muerto = True
                nueva_tabla_transiciones[traduccion_actual][simbolo] = q_muerto


    # El estado muerto sólo tendrá transiciones a sí mismo
    if hay_transiciones_a_estado_muerto:
        q_gradual = 'q{}'.format(q_gradual)
        nueva_tabla_transiciones[q_gradual] = {}
        for simbolo in alfabeto:
            nueva_tabla_transiciones[q_gradual][simbolo] = q_gradual

    # Renombramos las transiciones al estado muerto 'q_' para que vayan a un nuevo qx (simbolicamente
    # es lo mismo
    for simbolo in alfabeto:
        for estado in nueva_tabla_transiciones:
            if nueva_tabla_transiciones[estado][simbolo] == q_muerto:
                nueva_tabla_transiciones[estado][simbolo] = q_gradual

    # Eliminamos el antiguo estado muerto 'q_'

    # Traducimos los estados finales
    for estado_final in estados_finales:

        if estado_final in traducciones:
            traduccion = traducciones[estado_final]
            nuevos_estados_finales.add(traduccion)

    return [nueva_tabla_transiciones, nuevos_estados_finales, traducciones[estado_inicial]]

def genera_dfa(automata_finito: AutomataFinito):

    # Creamos la cola y añadimos el primer estado
    cola_procesamiento = [automata_finito.estado_inicial]

    # Creamos el dfa
    [nueva_tabla_transiciones, nuevos_estados_finales] = procesa_paso_nfa_2_dfa(automata_finito, {},
                                                                                automata_finito.estados_finales,
                                                                                cola_procesamiento)


    # Traducimos los nombres y creamos el estado muerto
    [nueva_tabla_transiciones, nuevos_estados_finales, nuevo_estado_inicial] = generar_traduccion(nueva_tabla_transiciones,
                                                                            automata_finito.alfabeto,
                                                                            nuevos_estados_finales,
                                                                            automata_finito.estado_inicial)

    return AutomataFinito(nueva_tabla_transiciones, automata_finito.alfabeto, nuevo_estado_inicial,
                          nuevos_estados_finales)

def guarda_dfa(automata_finito:AutomataFinito, archivo):
    with open(archivo, 'w') as f:

        # Escribimos la cabecera del archivo
        cabecera = 'q\t'
        for simbolo in automata_finito.alfabeto:
            cabecera += simbolo + '\t'

        cabecera += 'tipo_estado'
        cabecera += '\n'
        f.write(cabecera)

        for estado in automata_finito.tabla_transiciones:
            msg = estado + '\t'

            # Hay estados
            for simbolo in automata_finito.alfabeto:
                msg += automata_finito.tabla_transiciones[estado][simbolo]
                msg += '\t'

            # Añadimos el tipo de estado
            es_inicial = estado == automata_finito.estado_inicial
            es_final = estado in automata_finito.estados_finales
            if es_inicial and not es_final:
                msg += '-1'
            elif es_final and not es_inicial:
                msg += '1'
            elif es_inicial and es_final:
                msg += '2'
            else:
                msg += '0'

            # Escribimos en el archivo
            msg += '\n'

            f.write(msg)
# -----------------------------------

# --- Minimización DFA ---
def unir_estados_sinonimos(matriz):

    n = matriz.shape[0]

    particion = Particion(n)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if matriz[i][j] == 0 and particion.encontrar(i) != particion.encontrar(j):
                particion.unir(i, j)

    return particion

def obtener_listado_sinonimos(particion, n):

    sinonimos = {}

    for i in range(n):
        representante = particion.encontrar(i)

        if representante not in sinonimos:
            sinonimos[representante] = []

        sinonimos[representante].append(i)

    return [sinonimos[x] for x in sinonimos.keys()]

def traducir_sinonimos(listado_sinonimo, automata_finito):

    traduccion = {}
    q_actual = 0

    nueva_tabla_transiciones = {}
    nuevos_estados_finales = set()
    nuevo_estado_inicial = ''

    # Generamos las traducciones
    for i in range(len(listado_sinonimo)):
        estado = listado_sinonimo[i]

        # No tiene sinonimos
        if len(estado) == 1:
            traduccion['q{}'.format(estado[0])] = 'q{}'.format(q_actual)
            q_actual += 1

        # Tiene sinonimos, todos apuntarán al mismo estado
        else:
            for composicion in estado:
                traduccion['q{}'.format(composicion)] = 'q{}'.format(q_actual)
            q_actual+=1

    # Generamos la nueva tabla de transiciones a partir de las traducciones
    # y la antigua tabla de transiciones
    for estado in automata_finito.tabla_transiciones:
        estado_traducido = traduccion[estado]

        if estado_traducido not in nueva_tabla_transiciones:
            nueva_tabla_transiciones[estado_traducido] = {}

        for simbolo in automata_finito.alfabeto:
            nueva_tabla_transiciones[estado_traducido][simbolo] = traduccion[automata_finito.tabla_transiciones[estado][simbolo]]

    # Traducimos el listado de estados finales
    for estado_final in automata_finito.estados_finales:
        estado_final_traducido = traduccion[estado_final]
        nuevos_estados_finales.add(estado_final_traducido)

    # Traducimos el estado inicial
    nuevo_estado_inicial = traduccion[automata_finito.estado_inicial]

    return AutomataFinito(nueva_tabla_transiciones, automata_finito.alfabeto, nuevo_estado_inicial, nuevos_estados_finales)

def minimiza_dfa(automata_finito: AutomataFinito):
    tabla_transicion = automata_finito.tabla_transiciones
    estados = list(automata_finito.tabla_transiciones)
    estados_finales = automata_finito.estados_finales

    n = len(estados)

    # Creamos la matriz para minimizar
    matriz = np.zeros((n, n))
    matriz = matriz + np.eye(n) * -1

    # Marcamos los estados iniciales distinguibles (uno final y el otro no)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            estado_i = estados[i]
            estado_j = estados[j]

            # Son distinguibles
            if (estado_i in estados_finales and estado_j not in estados_finales) or \
                    (estado_j in estados_finales and estado_i not in estados_finales):
                matriz[i][j] = matriz[j][i] = 1

    # Aplicamos el algoritmo para distinguir estados
    continuar = True
    while continuar:
        alguno_marcado = False

        for i in range(0, n - 1):
            for j in range(i + 1, n):
                estado_i = estados[i]
                estado_j = estados[j]

                # Susceptible a distinguir (estaba vacío)
                if matriz[i][j] == 0:

                    # Comprobamos para cada símbolo, los estados a los que transitan el estado_i y estado_j
                    for simbolo in automata_finito.alfabeto:
                        i_siguiente = int(tabla_transicion[estado_i][simbolo].replace('q', ''))
                        j_siguiente = int(tabla_transicion[estado_j][simbolo].replace('q', ''))

                        # Si esos estados a los que transitan están distinguidos en la matriz, distinguimos tambien
                        # estado_i y estado_j
                        if matriz[i_siguiente][j_siguiente] == 1:
                            matriz[i][j] = 1
                            alguno_marcado = True
        continuar = alguno_marcado

    # Unimos los estados sinonimos
    particion = unir_estados_sinonimos(matriz)

    # Obtenemos listado con estados sinonimos
    listado_sinonimos = obtener_listado_sinonimos(particion, n)

    return traducir_sinonimos(listado_sinonimos, automata_finito)

def componer_estado_multiplicacion(q1, q2):
    return q1 + '_' + q2[1:]

def multiplica_dfa(dfa1: AutomataFinito, dfa2: AutomataFinito, renombrar_estados = False):
    """
    Calcula un nuevo DFA a partir del producto de otros dos DFA.
    Ambos tiene que tener el mismo alfabeto.
    :param dfa1:
    :param dfa2:
    :return:
    """

    alfabeto = dfa1.alfabeto

    estados_dfa1 = dfa1.tabla_transiciones.keys()
    estados_dfa2 = dfa2.tabla_transiciones.keys()

    estados_finales_dfa1 = dfa1.estados_finales
    estados_finales_dfa2 = dfa2.estados_finales

    estado_inicial_dfa1 = dfa1.estado_inicial
    estado_inicial_dfa2 = dfa2.estado_inicial

    nueva_tabla_transiciones = {}
    nuevos_estados_finales = set()
    nuevo_estado_inicial = None

    for simbolo in alfabeto:
        for estado_dfa1 in estados_dfa1:
            for estado_dfa2 in estados_dfa2:

                # Creamos el nuevo estado
                nuevo_estado = componer_estado_multiplicacion(estado_dfa1, estado_dfa2)

                # El nuevo estado será final si y solo si, uno de los estados que lo compone es final
                if (estado_dfa1 in estados_finales_dfa1 and estado_dfa2 not in estados_finales_dfa2) or \
                    (estado_dfa2 in estados_finales_dfa2 and estado_dfa1 not in estados_finales_dfa1):
                    nuevos_estados_finales.add(nuevo_estado)

                # El nuevo estado inicial será la combinación de los estados iniciales de ambos DFA's
                if estado_dfa1 == estado_inicial_dfa1 and estado_dfa2 == estado_inicial_dfa2:
                    nuevo_estado_inicial = nuevo_estado

                # Vemos al siguiente estado al que apuntarían
                siguiente_estado = componer_estado_multiplicacion(dfa1.tabla_transiciones[estado_dfa1][simbolo], dfa2.tabla_transiciones[estado_dfa2][simbolo])

                # Inicializamos el nuevo estado compuesto en la tabla de transiciones
                if nuevo_estado not in nueva_tabla_transiciones:
                    nueva_tabla_transiciones[nuevo_estado] = {}

                nueva_tabla_transiciones[nuevo_estado][simbolo] = siguiente_estado

    # Renombramos los estados
    if renombrar_estados:
        [nueva_tabla_transiciones, nuevos_estados_finales, nuevo_estado_inicial] = generar_traduccion(nueva_tabla_transiciones, alfabeto, nuevos_estados_finales, nuevo_estado_inicial)

    return AutomataFinito(nueva_tabla_transiciones, alfabeto, nuevo_estado_inicial, nuevos_estados_finales)
# ------------------------

# --- Testing ---
def comprueba_igualdad_tabla_transiciones_dfa(tabla1, tabla2, alfabeto):
    estados_tabla1 = set(tabla1.keys())
    estados_tabla2 = set(tabla2.keys())

    # Ambas tablas deben de tener los mismos estados
    for estado in estados_tabla1:
        if estado not in estados_tabla2:
            return False

    # Comprobamos que las transiciones para todos los simbolos y estados sean los mismos en ambas tablas
    for simbolo in alfabeto:
        for estado in estados_tabla1:

            # Hay transición para un estado y un simbolo en una tabla pero en la otra no
            if (simbolo not in tabla1[estado] and simbolo in tabla2[estado]) or (
                    simbolo in tabla1[estado] and simbolo not in tabla2[estado]):
                return False

            # El estado al que llevan es distinto
            if tabla1[estado][simbolo] != tabla2[estado][simbolo]:
                return False

    return True


def test_nfa_2_dfa():
    estado_inicial = 'q0'
    estados_finales = {'q3'}
    alfabeto = ['A', 'B']
    tabla_transiciones_prueba = {
        'q0': {
            'A': ['q1', 'q2'],
            'B': ['q3']
        },
        'q1': {
            'A': ['q2'],
            'B': ['q1', 'q3']
        },
        'q2': {
            'B': ['q1']
        },
        'q3': {
            'B': ['q3']
        }
    }

    num_estados_correcto = 7
    num_estados_finales_correcto = 2

    # DFA generado por mi algoritmo
    nfa = AutomataFinito(tabla_transiciones_prueba, alfabeto, estado_inicial, estados_finales)
    dfa_mio = genera_dfa(nfa)

    # Comprobaciones
    assert num_estados_correcto == len(dfa_mio.tabla_transiciones.keys())
    assert num_estados_finales_correcto == len(dfa_mio.estados_finales)

def test_minimiza_dfa():
    estado_inicial = 'q0'
    estados_finales = {'q4', 'q5', 'q6'}
    alfabeto = ['a', 'b']
    funcion_transicion = {
        'q0': {
            'a': 'q1',
            'b': 'q4'
        },
        'q1': {
            'a': 'q1',
            'b': 'q2'
        },
        'q2': {
            'a': 'q1',
            'b': 'q3'
        },
        'q3': {
            'a': 'q3',
            'b': 'q2'
        },
        'q4': {
            'a': 'q4',
            'b': 'q5'
        },
        'q5': {
            'a': 'q6',
            'b': 'q4'
        },
        'q6': {
            'a': 'q6',
            'b': 'q6'
        }
    }

    dfa = AutomataFinito(funcion_transicion, alfabeto, estado_inicial, estados_finales)

    dfa_minimizado = minimiza_dfa(dfa)

def test_producto_dfas():

    alfabeto = ['0', '1']

    estado_inicial_1 = 'q0'
    estados_finales_1 = {'q1'}
    tabla_transiciones_1 = {
        'q0': {             # Estado A del DFA de superior
            '0': 'q0',
            '1': 'q1'
        },
        'q1': {             # Estado B del DFA de superior
            '0': 'q0',
            '1': 'q0'
        }
    }

    estado_inicial_2 = 'q0'
    estados_finales_2 = {'q0'}
    tabla_transiciones_2 = {
        'q0': {  # Estado C del DFA de inferior
            '0': 'q1',
            '1': 'q0'
        },
        'q1': {  # Estado D del DFA de inferior
            '0': 'q1',
            '1': 'q0'
        }
    }

    dfa_1 = AutomataFinito(tabla_transiciones_1, alfabeto, estado_inicial_1, estados_finales_1)
    dfa_2 = AutomataFinito(tabla_transiciones_2, alfabeto, estado_inicial_2, estados_finales_2)

    dfa_producto = multiplica_dfa(dfa_1, dfa_2, True)

    tabla_transiciones_final = {'q0': {'0': 'q1', '1': 'q2'}, 'q1': {'0': 'q1', '1': 'q2'}, 'q2': {'0': 'q1', '1': 'q0'}}

    assert comprueba_igualdad_tabla_transiciones_dfa(tabla_transiciones_final, dfa_producto.tabla_transiciones, alfabeto)

def test_automata_parte_1():
    """
        Comprueba que el primer autómata reconozca correctamente las cadenas con la secuencia TGGGCGTTT
    """

    dfa_minimizado = obtiene_dfa_de_eps_nfa(ARCHIVO_INICIAL_EPS_NFA)

    from pythomata import SimpleDFA
    dfa_test = SimpleDFA(set(dfa_minimizado.tabla_transiciones.keys()), dfa_minimizado.alfabeto,
                         dfa_minimizado.estado_inicial, dfa_minimizado.estados_finales,
                         dfa_minimizado.tabla_transiciones)

    aceptadas = 0
    totales_aceptables = 0
    with open(ARCHIVO_TEST_ACEPTADAS_AUTOMATA_1, 'r') as f:
        palabras = f.read().splitlines()
    for palabra in palabras:
        totales_aceptables += 1
        if dfa_test.accepts(palabra):
            aceptadas += 1

    denegadas = 0
    totales_denegables = 0
    with open(ARCHIVO_TEST_DENEGADAS_AUTOMATA_1, 'r') as f:
        palabras = f.read().splitlines()
    for palabra in palabras:
        totales_denegables += 1
        if not dfa_test.accepts(palabra):
            denegadas += 1

    print('Habia un total de ' + str(totales_aceptables) + ' que aceptar y hemos aceptado ' + str(aceptadas))
    print('Habia un total de ' + str(totales_denegables) + ' que denegar y hemos denegado ' + str(denegadas))

    assert aceptadas == totales_aceptables
    assert denegadas == totales_denegables

def test_automata_parte_2():
    """
    Comprueba que el segundo autómata reconozca correctamente las cadenas que empiecen por AT o TA
    """

    [tabla_transiciones, alfabeto, estado_inicial, estados_finales] = lee_dfa_de_archivo(ARCHIVO_INICIAL_DFA)
    dfa = AutomataFinito(tabla_transiciones, alfabeto, estado_inicial, estados_finales)
    dfa_minimizado = minimiza_dfa(dfa)

    from pythomata import SimpleDFA
    dfa_test = SimpleDFA(set(dfa_minimizado.tabla_transiciones.keys()), dfa_minimizado.alfabeto,
                         dfa_minimizado.estado_inicial, dfa_minimizado.estados_finales,
                         dfa_minimizado.tabla_transiciones)

    aceptadas = 0
    totales_aceptables = 0
    with open(ARCHIVO_TEST_ACEPTADAS_AUTOMATA_2, 'r') as f:
        palabras = f.read().splitlines()
    for palabra in palabras:
        totales_aceptables += 1
        if dfa_test.accepts(palabra):
            aceptadas += 1

    denegadas = 0
    totales_denegables = 0
    with open(ARCHIVO_TEST_DENEGADAS_AUTOMATA_2, 'r') as f:
        palabras = f.read().splitlines()
    for palabra in palabras:
        totales_denegables += 1
        if not dfa_test.accepts(palabra):
            denegadas += 1

    print('Habia un total de ' + str(totales_aceptables) + ' que aceptar y hemos aceptado ' + str(aceptadas))
    print('Habia un total de ' + str(totales_denegables) + ' que denegar y hemos denegado ' + str(denegadas))

    assert aceptadas == totales_aceptables
    assert denegadas == totales_denegables

def test_automata_parte_3():
    """
    Test final. Obtiene los dos DFA necesarios y los multiplica para obtener el DFA final pedido en la
    práctica.
    """

    dfa_secuencia = obtiene_dfa_de_eps_nfa(ARCHIVO_INICIAL_EPS_NFA)
    dfa_at = obtiene_dfa_de_dfa(ARCHIVO_INICIAL_DFA)

    dfa_final = multiplica_dfa(dfa_secuencia, dfa_at, renombrar_estados=True)
    dfa_final_minimizado = minimiza_dfa(dfa_final)

    from pythomata import SimpleDFA
    dfa_test = SimpleDFA(set(dfa_final_minimizado.tabla_transiciones.keys()), dfa_final_minimizado.alfabeto,
                         dfa_final_minimizado.estado_inicial, dfa_final_minimizado.estados_finales,
                         dfa_final_minimizado.tabla_transiciones)

    aceptadas = 0
    totales_aceptables = 0
    with open(ARCHIVO_TEST_ACEPTADAS_AUTOMATA_3, 'r') as f:
        palabras = f.read().splitlines()
    for palabra in palabras:
        totales_aceptables += 1
        if dfa_test.accepts(palabra):
            aceptadas += 1

    denegadas = 0
    totales_denegables = 0
    with open(ARCHIVO_TEST_DENEGADAS_AUTOMATA_3, 'r') as f:
        palabras = f.read().splitlines()
    for palabra in palabras:
        totales_denegables += 1
        if not dfa_test.accepts(palabra):
            denegadas += 1

    print('Habia un total de ' + str(totales_aceptables) + ' que aceptar y hemos aceptado ' + str(aceptadas))
    print('Habia un total de ' + str(totales_denegables) + ' que denegar y hemos denegado ' + str(denegadas))

    assert aceptadas == totales_aceptables
    assert denegadas == totales_denegables

# ---------------

def obtiene_dfa_de_dfa(archivo):

    [tabla_transiciones, alfabeto, estado_inicial, estados_finales] = lee_dfa_de_archivo(ARCHIVO_INICIAL_DFA)
    dfa = AutomataFinito(tabla_transiciones, alfabeto, estado_inicial, estados_finales)
    dfa_minimizado = minimiza_dfa(dfa)
    return dfa_minimizado

def obtiene_dfa_de_eps_nfa(archivo):

    # Lee el archivo con la tabla de eps-transiciones y crea el eps-nfa
    [tabla_transiciones, alfabeto, estado_inicial, estados_finales] = lee_eps_nfa_de_archivo(archivo)
    eps_nfa = AutomataFinito(tabla_transiciones, alfabeto, estado_inicial, estados_finales)

    # Obtenemos las clausuras de cada elemento y los guarda en un archivo en orden
    clausuras_nfa = genera_clausuras(tabla_transiciones)
    guarda_clausuras(clausuras_nfa, ARCHIVO_CLAUSURAS)

    # Generamos el NFA a partir del epsilon-nfa
    nfa = genera_nfa(eps_nfa, clausuras_nfa)
    guarda_nfa(nfa, ARCHIVO_NFA)

    # Convertimos el NFA a un DFA
    dfa = genera_dfa(nfa)
    guarda_dfa(dfa, ARCHIVO_DFA)

    # Minimizamos el dfa
    dfa_minimizado = minimiza_dfa(dfa)
    guarda_dfa(dfa_minimizado, ARCHIVO_DFA_MINIMIZADO)

    return dfa_minimizado

def guarda_automata_finito(automata_finito, archivo):
    """
        Crea un archivo .csv en el que se almacena el automata finito
        :param automata_finito
        :param archivo
        :return:
    """

    with open(archivo, 'w') as f:

        # Añadimos la cabecera al archivo
        cabecera = ','
        for simbolo in automata_finito.alfabeto:
            cabecera += simbolo + ','
        cabecera += 'tipo_estado'
        f.write(cabecera + '\n')

        for estado in automata_finito.tabla_transiciones:

            linea = estado + ','

            for simbolo in automata_finito.alfabeto:

                # Hay una transicion para el simbolo
                if simbolo in automata_finito.tabla_transiciones[estado]:
                    linea += automata_finito.tabla_transiciones[estado][simbolo] + ','

                # No hay una transicion para el simbolo
                else:
                    linea += '.,'

            if estado == automata_finito.estado_inicial:
                linea += '-1'

            elif estado in automata_finito.estados_finales:
                linea += '1'

            else:
                linea += '0'

            # Escribimos en el archivo
            linea += '\n'
            f.write(linea)


if __name__ == "__main__":

    # Autómata para la cadena que contenta ...TGGGCGTTT...
    #test_automata_parte_1()

    # Autómata para cadenas que empicen por AT ó TA
    #test_automata_parte_2()

    # Autómata final
    test_automata_parte_3()