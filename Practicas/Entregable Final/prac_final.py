import enum
import time

import graphviz
from keras.utils import to_categorical
from sly import Lexer, Parser
import pandas as pd
import numpy as np
from typing import List
from tensorflow import keras
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz
from scipy import stats as st


class ENSEMBLE_MODES(enum.Enum):
    MODE = 1
    MEAN = 2


class EnsembleModel:

    def __init__(self, models):
        self.models = models

    def fit(self, X, y, epochs):
        for model in self.models:
            model.fit(X, y, epochs=epochs)

    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return predictions

    def get_models(self):
        return self.models

    def add_model(self, model):
        return self.models.append(model)


class BaggingModel(EnsembleModel):

    def __init__(self, models, mode: ENSEMBLE_MODES):
        super().__init__(models)
        self.mode = mode

    def get_mode(self):
        return self.mode

    def predict(self, X):
        predictions = EnsembleModel.predict(self, X)

        # final_predictions = np.zeros(predictions[0].shape[0], dtype=np.float)

        # Hay que calcular la moda
        if self.mode == ENSEMBLE_MODES.MODE:

            intermediate_arr = np.zeros((predictions[0].shape[0], len(predictions)), dtype=np.float)

            # Calculamos la prediccion (clase) por modelo y la almacenamos temporalmente
            colIndx = 0
            for prediction in predictions:
                intermediate_arr[:, colIndx] = np.argmax(prediction, axis=1) + 1
                colIndx += 1

            final_predictions = st.mode(intermediate_arr, axis=1, keepdims=True)

            return final_predictions.mode

        # Hay que calcular la media
        else:

            # Creamos la matriz intermedia en la que iremos guardando los resultados
            intermediate_arr = np.stack(predictions)
            output_dims = predictions[0].shape[1]

            # Creamos una matriz en la cual cada fila tendrá la media de la probabilidad de cada clase
            # segun cada predictor.
            final_predictions = np.zeros(predictions[0].shape, dtype=np.float)

            for i in range(output_dims):
                # Localizamos la columna a traves de todos los canales (profundidades).
                # profundidad, fila, columna
                t = intermediate_arr[:, :, i]

                # Calculamos la media para la clase i
                final_predictions[:, i] = np.mean(t, axis=0)

            # Devolvemos la clase finalmente elegida
            # return np.argmax(final_predictions, axis=1)+1

            # Devolvemos las probabilidades de cada clase
            return final_predictions


class StackingModel(EnsembleModel):

    def __init__(self, models, final_predictor: keras.models.Sequential):
        super().__init__(models)
        self.final_predictor = final_predictor

    def fit(self, X, y, epochs):
        # Ajustamos los modelos
        EnsembleModel.fit(self, X, y, epochs)

        # Hacemos predicciones sobre los datos de entrenamiento
        predictions = np.hstack(EnsembleModel.predict(self, X))

        # Ajustamos el predictor final dandole como entrada las salidas
        # de los modelos
        self.final_predictor.fit(predictions, y, epochs=epochs)

    def predict(self, X):
        # Hacemos las predicciones individuales de cada modelo
        pre_predictions = np.hstack(EnsembleModel.predict(self, X))

        # Hacemos la prediccion final en base a la de los modelos
        final_predictions = self.final_predictor.predict(pre_predictions)

        # Devolvemos la clase finalmente elegida
        # return np.argmax(final_predictions, axis=1) + 1

        # Devolvemos las probabilidades de cada clase
        return final_predictions


class Node:
    def execute(self):
        pass


class OP_TYPE(enum.Enum):
    UNKNOWN = 0,
    BASIC = 1,
    READ = 2,
    SPLIT = 3


class OperationNode(Node):

    def __init__(self, value, op_type: OP_TYPE = OP_TYPE.UNKNOWN):
        self.value = value
        self.op_type = op_type

    def execute(self):
        return self.value


class AssignNode(Node):

    def __init__(self, vars: List[str], value: OperationNode):
        self.vars = vars
        self.value = value

    def set_vars(self, vars):
        self.vars = vars

    def get_vars(self):
        return self.vars

    def __read_assignment(self):

        data: np.ndarray = self.value.execute()

        # Esta leyendo y guardando todo en una única variable
        if len(self.vars) == 1:
            return {self.vars[0]: data}

        # Esta leyendo y guardando en dos variables
        elif len(self.vars) == 2:
            y = data[:, -1]
            t = {
                self.vars[0]: data[:, 0:-1],
                self.vars[1]: to_categorical(y)
            }
            return t

    def __split_assignment(self):
        """
        Ligamos cada variable a su respectivo valor
        :return:
        """
        if len(self.vars) == 4:
            return {var: value for var, value in zip(self.vars, self.value.execute())}

    def __basic_assignment(self):
        temp = {}
        value = self.value.execute()
        for var in self.vars:
            temp[var] = value
        return temp

    def execute(self) -> dict:

        op_type = self.value.op_type

        if op_type is OP_TYPE.READ:
            return self.__read_assignment()

        elif op_type is OP_TYPE.SPLIT:
            return self.__split_assignment()

        elif op_type is OP_TYPE.BASIC:
            return self.__basic_assignment()

        elif op_type is OP_TYPE.UNKNOWN:
            return {}


class KerasLexer(Lexer):
    tokens = {READ, DATASET, NORM, SPLIT, SEQUENTIAL, DENSE, COMPILE, FIT, PREDICT,
              BAGGING, STACKING, VISUALIZE, INT, FLOAT, VARIABLE,
              ACTIVATION, OPTIMIZER, LOSS, METRIC, MEAN, MODE, PRINT}

    literals = {'\'', '(', ')', '[', ']', '=', ','}

    INT = r'[0-9]+'
    FLOAT = r'[0-9]+\.[0-9]+'

    READ = r'read_csv'
    DATASET = r'Iris|iris|Image|image|Pima|pima|Wine|wine'

    NORM = r'normalize'
    SPLIT = r'split'

    SEQUENTIAL = r'Sequential'
    DENSE = r'Dense'
    COMPILE = r'compile'
    FIT = r'fit'
    PREDICT = r'predict'

    BAGGING = r'Bagging'
    STACKING = r'Stacking'

    ACTIVATION = r'relu|softmax'
    OPTIMIZER = r'adam|sgd'
    LOSS = r'categorical_crossentropy'
    METRIC = r'accuracy'

    VISUALIZE = r'visualize'

    PRINT = r'print'

    MEAN = r'((?<![a-zA-Z0-9])(mean|Mean)(?![a-zA-Z0-9]))'
    MODE = r'((?<![a-zA-Z0-9])(mode|Mode)(?![a-zA-Z0-9]))'

    VARIABLE = r'[a-zA-Z_][a-zA-Z0-9_]*'

    # String containing ignored characters
    ignore_comment = r'\#.*'
    ignore = '\t'
    ignore_newline = r'\n+'

    def INT(self, t):
        t.value = int(t.value)
        return t

    def FLOAT(self, t):
        t.value = float(t.value)
        return t

    # Line number tracking
    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        # print('Line %d: Bad character %r' % (self.lineno, t.value[0]))
        self.index += 1


class KerasParser(Parser):
    # Get the token list from the lexer (required)
    tokens = KerasLexer.tokens

    variables = {}

    debugfile = 'parser.log'

    @_('Inst')
    def S(self, p):
        for k in self.variables:
            # print(k + ': ' + str(self.variables[k]))
            # print(str(self.variables[k].shape))
            pass
        return None

    @_('Asig Inst')
    def Inst(self, p):
        return p.Inst

    @_('Ejec Inst')
    def Inst(self, p):
        return p.Inst

    @_('')
    def Inst(self, p):
        return None

    @_('VARIABLE "=" Ejec')
    def Asig(self, p):

        vars = [p.VARIABLE]
        assignNode = AssignNode(vars, p.Ejec)

        # Asignamos el valor a la variable
        for key, value in assignNode.execute().items():
            self.variables[key] = value

        return None

    @_('"[" Vars_1 "]" "=" Ejec')
    def Asig(self, p):
        vars = p.Vars_1
        assignNode = AssignNode(vars, p.Ejec)

        print(vars)
        print(p.Ejec)

        # Asignamos cada valor a su respectiva variable
        for key, value in assignNode.execute().items():
            self.variables[key] = value

        return None

    @_('VARIABLE "," Vars_2')
    def Vars_1(self, p):
        temp = [p.VARIABLE]
        temp.extend(p.Vars_2)
        return temp

    @_('VARIABLE "," Vars_2')
    def Vars_2(self, p):
        temp = [p.VARIABLE]
        temp.extend(p.Vars_2)
        return temp

    @_('VARIABLE')
    def Vars_2(self, p):
        return [p.VARIABLE]

    @_('Read')
    def Ejec(self, p):
        return p.Read

    @_('Norm')
    def Ejec(self, p):
        return p.Norm

    @_('Split')
    def Ejec(self, p):
        return p.Split

    @_('Sequential')
    def Ejec(self, p):
        return p.Sequential

    @_('Compile')
    def Ejec(self, p):
        return p.Compile

    @_('Fit')
    def Ejec(self, p):
        print('Fitting...')
        return p.Fit

    @_('Predict')
    def Ejec(self, p):
        return p.Predict

    @_('Bagging')
    def Ejec(self, p):
        return p.Bagging

    @_('Stacking')
    def Ejec(self, p):
        return p.Stacking

    @_('Visualize')
    def Ejec(self, p):
        return p.Visualize

    @_('Print')
    def Ejec(self, p):
        return p.Print

    @_('READ "(" "\'" DATASET "\'" ")"')
    def Read(self, p):
        return OperationNode(pd.read_csv('datasets/{}.csv'.format(p.DATASET.lower())).to_numpy(), OP_TYPE.READ)

    @_('NORM "(" VARIABLE ")"')
    def Norm(self, p):

        data: np.ndarray = self.variables[p.VARIABLE]
        min = data.min()
        data = (data - min) / (data.max() - min)

        return OperationNode(data, OP_TYPE.BASIC)

    @_('SPLIT "(" VARIABLE "," VARIABLE ")"')
    def Split(self, p):

        # Localizamos los dos conjuntos de datos (independientes y dependientes)
        x_data = self.variables[p.VARIABLE0]
        y_data = self.variables[p.VARIABLE1]

        # Elegimos los indxs de las filas que seran para train y para test
        nrows = x_data.shape[0]
        idx = list(np.arange(0, nrows))
        train_idx = list(np.random.choice(idx, size=round(0.70 * nrows), replace=False))
        test_idx = list(set(idx) - set(train_idx))

        # Creamos los dos conjuntos separados
        xtrn, ytrn = x_data[train_idx], y_data[train_idx]
        xtst, ytst = x_data[test_idx], y_data[test_idx]

        return OperationNode([xtrn, ytrn, xtst, ytst], OP_TYPE.SPLIT)

    @_('SEQUENTIAL "[" Denses "]" ')
    def Sequential(self, p):

        temp = keras.models.Sequential()

        # Añadimos las capas al modelo
        for layer in p.Denses:
            temp.add(layer)

        return OperationNode(temp, OP_TYPE.BASIC)

    @_('Dense')
    def Denses(self, p):
        return [p.Dense]

    @_('Dense "," Denses')
    def Denses(self, p):
        t = [p.Dense]
        t.extend(p.Denses)
        return t

    @_('DENSE "(" INT "," "\'" ACTIVATION "\'" ")"')
    def Dense(self, p):
        return Dense(p.INT, activation='{}'.format(p.ACTIVATION))

    @_('COMPILE "(" VARIABLE "," "\'" OPTIMIZER "\'" "," "\'" LOSS "\'" "," "[" Metrics "]" ")"')
    def Compile(self, p):

        self.variables[p.VARIABLE].compile(
            optimizer=p.OPTIMIZER,
            loss=p.LOSS,
            metrics=list(set(p.Metrics))
        )
        return OperationNode(None, OP_TYPE.BASIC)

    @_('"\'" METRIC "\'" ')
    def Metrics(self, p):
        return [p.METRIC]

    @_('"\'" METRIC "\'" "," Metrics')
    def Metrics(self, p):
        t = [p.METRIC]
        t.extend(p.Metrics)
        return t

    @_('FIT "(" VARIABLE "," VARIABLE "," VARIABLE "," INT ")"')
    def Fit(self, p):

        # TODO Comprobar si la variable es un modelo o es un Ensemble
        modelo = self.variables[p.VARIABLE0]

        x_trn = self.variables[p.VARIABLE1]
        y_trn = self.variables[p.VARIABLE2]
        epochs = p.INT

        # Ajustamos el modelo
        modelo.fit(x_trn, y_trn, epochs=epochs)

        return OperationNode(Node, OP_TYPE.BASIC)

    @_('PREDICT "(" VARIABLE "," VARIABLE ")"')
    def Predict(self, p):

        modelo = self.variables[p.VARIABLE0]
        x_pred = self.variables[p.VARIABLE1]

        y_pred = modelo.predict(x_pred)

        # Hacemos la prediccions
        return OperationNode(y_pred, OP_TYPE.BASIC)

    @_('BAGGING "(" "[" Vars_1 "]" "," "\'" Bagging_Mode "\'" ")"')
    def Bagging(self, p):

        args_models_vars = p.Vars_1

        # Metemos todos los modelos en un array
        models = []
        for var_name in set(args_models_vars):
            models.append(self.variables[var_name])

        # Miramos el modo que tendrá el model
        mode = p.Bagging_Mode

        # Creamos el modelo compuesto
        return OperationNode(BaggingModel(models, mode), OP_TYPE.BASIC)

    @_('MEAN')
    def Bagging_Mode(self, p):

        # Creamos el modelo compuesto
        return ENSEMBLE_MODES.MEAN

    @_('MODE')
    def Bagging_Mode(self, p):

        # Creamos el modelo compuesto
        return ENSEMBLE_MODES.MODE

    @_('STACKING "(" "[" Vars_1 "]" "," VARIABLE ")"')
    def Stacking(self, p):

        args_models_vars = p.Vars_1

        # Metemos todos los modelos en un array
        models = []
        for var_name in set(args_models_vars):
            models.append(self.variables[var_name])

        # Creamos el modelo compuesto
        return OperationNode(StackingModel(models, self.variables[p.VARIABLE]), OP_TYPE.BASIC)

    @_('VISUALIZE "(" VARIABLE ")"')
    def Visualize(self, p):

        # Localizamos el modelo
        modelo = self.variables[p.VARIABLE]

        # Mostramos el modelo
        ann_viz(modelo)

        return OperationNode(Node, OP_TYPE.BASIC)

    @_('PRINT "(" VARIABLE ")"')
    def Print(self, p):
        print(str(self.variables[p.VARIABLE]))
        return None


def execute(text):
    lexer = KerasLexer()
    parser = KerasParser()
    parser.parse(lexer.tokenize(text))


def execute_test():
    texts = []

    # Carga el archivo de ejemplo
    with open('ejemplo1_prac_final.txt', 'r') as f:
        text = f.read()
        texts = text.split("@")

    texts = texts[0:2]

    # Ejecuta los programas
    for program in texts:
        print("----- Programa a interpretar -----")
        print(program)
        print('----- Salida (si hay prints) -----')
        execute(program)
        print('*'*25)
        print('')


if __name__ == '__main__':

    # Ejecuta algunos programas de prueba
    execute_test()
