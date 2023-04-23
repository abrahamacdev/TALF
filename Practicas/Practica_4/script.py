import re
from sly import Lexer

PALABRAS_RESERVADAS =[]
#NUM_INT = []
#NUM_FLOAT = []
NUM_IDENTIFICADOR = []
CONSTANTES = []

class Procesador_C(Lexer):

    # Set of token names.   This is always required
    #tokens = {CODE, IF, ELSE, SIMPLE_COMMENT, MULT_COMMENT, INT, FLOAT, CHAR, FOR, WHILE, RETURN}
    #tokens = {CONSTANT, CODE, IF, ELSE, INT, FLOAT, CHAR, FOR, WHILE, RETURN}
    tokens = {ID, CONSTANT, IF, ELSE, INT, FLOAT, CHAR, FOR, WHILE, RETURN}

    # Ignoramos espacios y tabulaciones
    ignore = '( )+|(\t)+'
    ignore_comment = r'(\/\*[a-zA-Z_0-9\s\t\n]*\*\/)|(\/\/([a-zA-Z_0-9\s\t]* | \n{0,1}))'
    ignore_newline = r'\n+'

    # Regular expression rules for tokens
    #MULT_COMMENT = r'\/\*[a-zA-Z_0-9\s\t\n]*\*\/'
    #SIMPLE_COMMENT = r'\/\/([a-zA-Z_0-9\s\t]* | \n{0,1})'


    #FOR = r'for( )*\([\w\.=+-_()\"\s\t\n]+\)'
    FOR = r'for'

    #WHILE = r'while( )*\([\w\.=+-_()\"\s\t\n]+\)'
    WHILE = r'while'

    INT = r'int'
    FLOAT = r'float'
    CHAR = r'char'

    RETURN = r'return'

    IF = r'if'
    ELSE = r'else'

    CONSTANT = r'\d+\.\d+|\d+'

    ID = r'[A-Za-z_]+[A-Za-z0-9_]*[ ]*=[ ]*'

    #CODE = r'[\w\.=_()\"\'\s\t\n]+'

    # Pasamos del resto de cosas
    def error(self, t):
        self.index += 1

    def CONSTANT(self, t):
        global CONSTANTES
        CONSTANTES.append(self.lineno)
        return t

    def ID(self, t):
        global NUM_IDENTIFICADOR
        NUM_IDENTIFICADOR.append(self.lineno)
        return t

    # palabras reservadas
    def INT(self, t):
        global PALABRAS_RESERVADAS
        PALABRAS_RESERVADAS.append(self.lineno)
        return t
    def FLOAT(self, t):
        global PALABRAS_RESERVADAS
        PALABRAS_RESERVADAS.append(self.lineno)
        return t
    def CHAR(self, t):
        global PALABRAS_RESERVADAS
        PALABRAS_RESERVADAS.append(self.lineno)
        return t
    def FOR(self, t):
        global PALABRAS_RESERVADAS
        PALABRAS_RESERVADAS.append(self.lineno)
        return t
    def WHILE(self, t):
        global PALABRAS_RESERVADAS
        PALABRAS_RESERVADAS.append(self.lineno)

    def IF(self, t):
        global PALABRAS_RESERVADAS
        PALABRAS_RESERVADAS.append(self.lineno)
        return t

    def ELSE(self, t):
        global PALABRAS_RESERVADAS
        PALABRAS_RESERVADAS.append(self.lineno)
        return t

    def RETURN(self, t):
        global PALABRAS_RESERVADAS
        PALABRAS_RESERVADAS.append(self.lineno)
        return t

    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += t.value.count('\n')

    def ignore_comment(self, t):
        self.lineno += t.value.count('\n')

    """
    def CODE(self, t):
        if re.search(r'[a-zA-Z]+[a-zA-Z_0-9]*[ ]*=(?!=)', t.value):
            global NUM_IDENTIFICADOR
            NUM_IDENTIFICADOR.append(self.lineno)
        return t
    """

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    lexer = Procesador_C()

    ARCHIVO_C = './ejemplo.c'

    with open(ARCHIVO_C, 'r') as f:
        str = f.read()

    detected = []
    for token in lexer.tokenize(str):
        valor = token.value
        valor = valor.replace('\n', ' ')
        detected.append(f'Tipo: {token.type}; Valor: {valor}; Linea: {token.lineno};')

    print('Hay palabras reservadas en las lineas: {}'.format(PALABRAS_RESERVADAS))
    #print('Hay enteros (int) en las lineas: {}'.format(NUM_INT))
    #print('Hay flotantes (float) en las lineas: {}'.format(NUM_FLOAT))
    print('Hay constantes (int o float) en las lineas: {}'.format(CONSTANTES))
    print('Hay identificadores en las lineas: {}'.format(NUM_IDENTIFICADOR))

    for d in detected:
        print(d)