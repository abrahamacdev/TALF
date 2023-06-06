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

        return predictions

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

class PruebecitaLexer(Lexer):
    tokens = {BAGGING, VARIABLE, MEAN, MODE}

    literals = {'\'', '(', ')', '[', ']', '=', ','}


    BAGGING = r'Bagging'

    MEAN = r'((?<![a-zA-Z0-9])(mean|Mean)(?![a-zA-Z0-9]))'
    MODE = r'((?<![a-zA-Z0-9])(mode|Mode)(?![a-zA-Z0-9]))'

    VARIABLE = r'[a-zA-Z][a-zA-Z0-9]*'

    # String containing ignored characters
    ignore = ' \t'
    ignore_newline = r'\n+'

    # Line number tracking
    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        # print('Line %d: Bad character %r' % (self.lineno, t.value[0]))
        self.index += 1


class PruebecitaParse(Parser):
    # Get the token list from the lexer (required)
    tokens = PruebecitaLexer.tokens

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

    @_('VARIABLE "," Vars_2')
    def Vars_2(self, p):
        temp = [p.VARIABLE]

        if p.Vars_2:
            temp.extend(p.Vars_2)

        return temp

    @_('VARIABLE')
    def Vars_2(self, p):
        return None


    @_('Bagging')
    def Ejec(self, p):
        return p.Bagging

    @_('BAGGING "(" "[" Vars_2 "]" "," "\'" Bagging_Mode "\'" ")"')
    def Bagging(self, p):

        print(p.Vars_2)
        print(p.Bagging_Mode)
        #print(p.MEAN)

        """
        args_models_vars = p.Vars_2

        # Metemos todos los modelos en un array
        models = []
        for var_name in args_models_vars:
            models.append(self.variables[var_name])

        # Miramos el modo que tendrá el model
        mode = p.Bagging_Mode
        
        """

        # Creamos el modelo compuesto
        return OperationNode(None, OP_TYPE.BASIC)

    @_('MEAN')
    def Bagging_Mode(self, p):

        # Creamos el modelo compuesto
        return ENSEMBLE_MODES.MEAN

    @_('MODE')
    def Bagging_Mode(self, p):

        # Creamos el modelo compuesto
        return ENSEMBLE_MODES.MODE


if __name__ == '__main__':
    texts = []
    with open('ejemplo1_prac_final.txt', 'r') as f:
        text = f.read()
        texts = text.split("#")

    lexer = PruebecitaLexer()
    parser = PruebecitaParse()

    indxsPrograma = 1

    print("Programa a interpretar:")
    print(texts[indxsPrograma])
    for token in lexer.tokenize(texts[indxsPrograma]):
        print('Tipo {} -- Value {}'.format(token.type, token.value))

    parser.parse(lexer.tokenize(texts[indxsPrograma]))