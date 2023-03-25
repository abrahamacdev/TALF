
import numpy as np
import random

estados = {'q6', 'q3', 'q0', 'q10', 'q4', 'q7', 'q2', 'q1', 'q5', 'q8', 'q9'}
alfabeto = {'6', '2', '8', '4', '5'}
estado_inicial = 'q0'
estados_finales = {'q5'}
funcion_transicion = {'q2': {'8': 'q10', '6': 'q10', '5': 'q10', '2': 'q10', '4': 'q3'}, 'q4': {'8': 'q10', '2': 'q10', '6': 'q10', '4': 'q10', '5': 'q5'}, 'q0': {'5': 'q6', '8': 'q1', '6': 'q10', '4': 'q10', '2': 'q10'}, 'q9': {'8': 'q5', '2': 'q10', '5': 'q10', '4': 'q10', '6': 'q10'}, 'q5': {'2': 'q5', '4': 'q5', '5': 'q5', '6': 'q5', '8': 'q5'}, 'q10': {'2': 'q10', '4': 'q10', '5': 'q10', '6': 'q10', '8': 'q10'}, 'q7': {'8': 'q10', '2': 'q10', '6': 'q10', '5': 'q10', '4': 'q8'}, 'q1': {'8': 'q10', '4': 'q10', '5': 'q10', '6': 'q10', '2': 'q2'}, 'q3': {'8': 'q10', '4': 'q10', '5': 'q10', '2': 'q10', '6': 'q4'}, 'q8': {'6': 'q9', '8': 'q10', '5': 'q10', '4': 'q10', '2': 'q10'}, 'q6': {'6': 'q10', '4': 'q10', '5': 'q10', '8': 'q10', '2': 'q7'}}

def test(cadena):
	"""
	Comprueba si una cadena dada pasa el automata.

	:return: False Si no pasa el automata.
			 True en caso contrario.
	"""

	estado_actual = estado_inicial

	for c in cadena:

		# El caracter tiene que estar en el alfabeto si no, no será consumido
		if c not in alfabeto:
			return False

		# 
		else:
			estado_actual = funcion_transicion[estado_actual][c]


	return estado_actual in estados_finales

def generar_permutaciones(posibles, n):
	"""
	Genera permutaciones aleatorias del alfabeto para generar casos de prueba.

	:param posibles: Valor posibles del alfabeto.
	:param n: Cantidad de permutaciones.
	:return: Vector de permutaciones generadas.
	"""

	# Probabilidad de añadir algo al final
	alfa = 0.3

	perms = []

	for i in range(n):
		perm = np.random.permutation(posibles)

		if random.uniform(0, 1) < alfa:
			perm = np.append(perm, np.random.permutation((posibles)))

		perms.append(perm)

	return perms


if __name__ == "__main__":

	posibles = [int(x) for x in alfabeto]

	N = 1000

	perms = generar_permutaciones(posibles, N)

	pasan_automata = []

	for perm in perms:
		input = ''.join(perm[:].astype(str).tolist())
		res_test = test(input)

		if res_test:
			pasan_automata.append(perm)

	print('De {}'.format(str(N)) + ' permutaciones {}'.format(str(len(pasan_automata))) + ' han pasado el autómata. Algunos ejemplos:')

	for i in range(10):
		print(pasan_automata[i])