from pythomata import SimpleDFA
import os
import json
import jsonpickle
from json import JSONEncoder

def prueba(alfabeto, cadena, funcion_transicion, estado = 'q0', debug = False):
    if len(cadena) == 0:
        return estado
    else:
        letra = cadena[0]
        cadena = cadena[1:]

        if letra in alfabeto:
            if debug:
                print(estado + ' - ' + letra + ' -> ' + funcion_transicion[estado][letra])
            
            estado = funcion_transicion[estado][letra]

            return prueba(alfabeto, cadena, funcion_transicion, estado)

def test(dfa, palabras):
    bien = 0
    total = 0
    for palabra in palabras:
        if len(palabra) > 0:
            total += 1
            res = dfa.accepts(palabra)
            resp = 'Rechazada'

            if res:
                bien += 1
                resp = 'Aceptada'
            
            print(palabra + ': ' + resp)
            #print(prueba(alphabet, palabra, transition_function))

    print(str(bien) + ' bien')
    print(str(total) + ' en total')

alphabet = {"E", ",", "+", "-"}
for i in range(0, 10):
    alphabet.add("{}".format(str(i)))

states = {"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16"}
initial_state = "q0"
accepting_states = {"q9", "q13", "q16"}
transition_function = {
    "q0": {
        "0" : "q3",
        "5" : "q2",
        "E" : "q14",
        "," : "q14",
        "+" : "q1",
    },
    "q1": {
        "0" : "q5",
        "5" : "q4",
        "E" : "q14",
        "," : "q14",
        "+" : "q14"
    },
    "q2": {
        "0" : "q14",
        "5" : "q14",
        "E" : "q14",
        "," : "q15",
        "+" : "q14"
    },
    "q3": {
        "0" : "q14",
        "5" : "q14",
        "E" : "q14",
        "," : "q12",
        "+" : "q14"
    },
    "q4": {
        "0" : "q14",
        "5" : "q14",
        "E" : "q14",
        "," : "q15",
        "+" : "q14"
    },
    "q5": {
        "0" : "q14",
        "5" : "q14",
        "E" : "q14",
        "," : "q10",
        "+" : "q14"
    },
    "q6": {
        "0" : "q6",
        "5" : "q6",
        "E" : "q7",
        "+" : "q14",
        "," : "q14"
    },
    "q7": {
        "0" : "q9",
        "5" : "q9",
        "E" : "q14",
        "+" : "q8",
        "," : "q14"
    },
    "q8": {
        "0" : "q16",
        "5" : "q16",
        "E" : "q14",
        "+" : "q14",
        "," : "q14"
    },
    "q9": {
        "0" : "q9",
        "5" : "q9",
        "E" : "q14",
        "+" : "q14",
        "," : "q14"
    },
    "q10": {
        "0" : "q10",
        "5" : "q11",
        "E" : "q14",
        "+" : "q14",
        "," : "q14"
    },
    "q11": {
        "0" : "q11",
        "5" : "q11",
        "E" : "q7",
        "+" : "q14",
        "," : "q14"
    },
    "q12": {
        "0" : "q12",
        "5" : "q14",
        "E" : "q13",
        "+" : "q14",
        "," : "q14"
    },
    "q13": {
        "0" : "q13",
        "5" : "q14",
        "E" : "q14",
        "+" : "q14",
        "," : "q14"
    },
    "q14": {
        "0" : "q14",
        "5" : "q14",
        "E" : "q14",
        "+" : "q14",
        "," : "q14"
    },
    "q15": {
        "0" : "q6",
        "5" : "q6",
        "E" : "q14",
        "+" : "q14",
        "," : "q14"
    },
    "q16": {
        "0" : "q16",
        "5" : "q16",
        "E" : "q14",
        "+" : "q14",
        "," : "q14"
    }
}

# Rellenamos para símbolos 1-9 y -
for estado in transition_function:
    for i in range(1,10):
        transition_function[estado]["{}".format(str(i))] = transition_function[estado]["5"]
    transition_function[estado]["-"] = transition_function[estado]["+"]        

dfa = SimpleDFA(states, alphabet, initial_state, accepting_states, transition_function)

# Minimizamos el automata
dfa_minimized = dfa.minimize()

# Comprobamos que pasa todos los test
with open('test.txt', 'r') as f:
    archivo = f.read()
    palabras = archivo.split('\n')

# Test antes
test(dfa, palabras)
print('\n\n')


# Guardamos las transiciones finales
transiciones_finales = dfa_minimized.get_transitions()

# Test despues
test(dfa_minimized, palabras)

#with open('./salida.txt', 'w') as f:
#    for situacion in transiciones_finales:
#        f.write(str(situacion[0]) + ' va a ' + str(situacion[2]) + ' con ' + str(situacion[1]))
#        f.write('\n')



# Guarda el diagrama del autómata
#graph = dfa.minimize().to_graphviz()
#graph.render("./ejemplo")
#os.remove('./ejemplo')