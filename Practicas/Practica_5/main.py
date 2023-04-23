from sly import Lexer, Parser


class CalcLexer(Lexer):
    tokens = {NUMERO, VARIABLE}

    VARIABLE = r'[A-Za-z]+'
    NUMERO = r'\d+'

    #ABRE_PARENTESIS = r'\('
    #CIERRA_PARENTESIS = r'\)'

    literals = {'+', '-', '*', '/', '=', '(', ')'}

    # String containing ignored characters
    ignore = ' \t'
    ignore_newline = r'\n+'

    def NUMERO(self, t):
        t.value = int(t.value)
        return t

    # Line number tracking
    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        # print('Line %d: Bad character %r' % (self.lineno, t.value[0]))
        self.index += 1


class CalcParser(Parser):
    tokens = CalcLexer.tokens

    def __init__(self):
        self.vars = {}

    # Get the token list from the lexer (required)
    tokens = CalcLexer.tokens


    @_('VAR "=" V')
    def V(self, p):
        self.vars[p.VAR] = p.V
        return p.V

    @_('E')
    def V(self, p):
        return p.E

    @_('VARIABLE')
    def VAR(self, p):
        return p.VARIABLE

    @_('E "+" SUMA')
    def E(self, p):
        return p.E + p.SUMA

    @_('E "-" SUMA')
    def E(self, p):
        return p.E - p.SUMA

    @_('SUMA')
    def E(self, p):
        return p.SUMA

    @_('SUMA "*" FACTO')
    def SUMA(self, p):
        return p.SUMA * p.FACTO

    @_('SUMA "/" FACTO')
    def SUMA(self, p):
        return p.SUMA / p.FACTO

    @_('FACTO')
    def SUMA(self, p):
        return p.FACTO

    @_('"-" UNA')
    def FACTO(self, p):
        return -p.UNA

    @_('UNA')
    def FACTO(self, p):
        return p.UNA

    @_('"(" E ")"')
    def UNA(self, p):
        return p.E

    @_('PAREN')
    def UNA(self, p):
        return p.PAREN

    @_('NUMERO')
    def PAREN(self, p):
        return p.NUMERO

    @_('VARIABLE')
    def PAREN(self, p):
        return self.vars[p.VARIABLE]


if __name__ == '__main__':

    t1 = '2 - 3 * -5 + 7 / 2'  # 20.5
    t2 = '(2 - 3) * -(5 + 7) / 2'  # 6
    t3 = 'z = 5'  # 5
    t4 = 'x = y = 2 * -(3 + 5) * z'  # -80
    t5 = 'x'  # -80
    t6 = 'y'  # -80

    tests = {
        t1: 20.5,
        t2: 6,
        t3: 5,
        t4: -80,
        t5: -80,
        t6: -80
    }

    lexer = CalcLexer()
    parser = CalcParser()

    for test, valor in tests.items():
        print(test)
        try:
            print('Calculado: ' + str(parser.parse(lexer.tokenize(test))))
            print('Real: ' + str(valor))
        except Exception as e:
            print('No se pudo calcular.')
        print('-' * 20)