import numpy as np
import matplotlib.pyplot as plt


estados = {'q2', 'q1', 'q10', 'q8', 'q4', 'q7', 'q9', 'q3', 'q6', 'q5', 'q0'}
alfabeto = {'0', '2', '1'}
estado_inicial = 'q0'
estados_finales = {'q9', 'q5'}
funcion_transicion = {'q0': {'0': 'q6', '2': 'q1', '1': 'q10'}, 'q6': {'2': 'q10', '1': 'q10', '0': 'q7'}, 'q4': {'1': 'q10', '0': 'q10', '2': 'q5'}, 'q7': {'0': 'q8', '1': 'q7', '2': 'q7'}, 'q8': {'1': 'q7', '2': 'q7', '0': 'q9'}, 'q1': {'1': 'q2', '0': 'q10', '2': 'q10'}, 'q3': {'1': 'q4', '2': 'q10', '0': 'q10'}, 'q9': {'1': 'q7', '2': 'q7', '0': 'q9'}, 'q5': {'0': 'q5', '1': 'q5', '2': 'q5'}, 'q10': {'0': 'q10', '1': 'q10', '2': 'q10'}, 'q2': {'0': 'q3', '1': 'q10', '2': 'q10'}}


def test(cadena):
	"""
	Comprueba si una cadena dada pasa el automata.

	:return: -1 Si no pasa el automata. 
			 0 Si el estado final es q5 (caso A). 
			 1 Si el estado final es q9 (caso B).
	"""

	estado_actual = estado_inicial

	for c in cadena:

		# El caracter tiene que estar en el alfabeto si no, no será consumido
		if c not in alfabeto:
			return -1

		# 
		else:
			estado_actual = funcion_transicion[estado_actual][c]


	return 0 if estado_actual == 'q5' else 1 if estado_actual == 'q9' else -1


def leer_test(nombre_archivo):

	palabras = []

	with open(nombre_archivo, 'r') as f:
		
		for linea in f:
			palabras.append(linea.replace('\n', ''))

	return palabras

def realizar_pruebas():

	archivo_test = 'test.txt'

	palabras = leer_test(archivo_test)

	for palabra in palabras:
		
		resultado = test(palabra)
		msg = 'Aceptada' if resultado != -1 else 'Rechazada'
		print(palabra + ' -- ' + msg)

if __name__ == "__main__":
	
	imgtest= np.random.randint(0,3, (160,144))

	res = np.copy(imgtest)

	plt.imshow(imgtest) # Check initial image

	for row in range(imgtest.shape[0]):
		input = ''.join(imgtest[row,:].astype(str).tolist())
		res_test = test(input)

		# Caso A
		if res_test == 0:
			res[row,:] = 2

		# Caso B
		if res_test == 1:
			res[row,:] = 0
	
	for col in range(imgtest.shape[1]):
		input = ''.join(imgtest[:,col].astype(str).tolist())
		res_test = test(input)		

		# Caso A
		if res_test == 0:
			res[row,:] = 2

		# Caso B
		if res_test == 1:
			res[row,:] = 0
	
	# After treatment by automata acceptance test
	plt.figure()
	plt.imshow(res) # Check initial image
	plt.show()