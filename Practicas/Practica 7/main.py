import time
import turtle

import graphviz
from sly import Lexer, Parser


class TurtleLexer(Lexer):
    tokens = {NUMBER, BEGIN, END, FORWARD, RIGHT, LEFT, BACK, REPEAT}

    NUMBER = r'\d+'

    BEGIN = r'BEGIN'
    END = r'END'

    FORWARD = r'FORWARD'
    RIGHT = r'RIGHT'
    LEFT = r'LEFT'
    BACK = r'BACK'

    REPEAT = r'REPEAT'

    # ABRE_PARENTESIS = r'\('
    # CIERRA_PARENTESIS = r'\)'

    literals = {';', '[', ']'}

    # String containing ignored characters
    ignore = ' \t'
    ignore_newline = r'\n+'

    def N(self, t):
        t.value = int(t.value)
        return t

    # Line number tracking
    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        # print('Line %d: Bad character %r' % (self.lineno, t.value[0]))
        self.index += 1


class NumberNode():
    def __init__(self, num):
        self.num = num

    def __str__(self) -> str:
        return str(self.num)

    def execute(self):
        return self.num


class UnaryNode():
    def __init__(self, tipo, numberNode, turt):
        self.numberNode = numberNode
        self.tipo = tipo
        self.turt = turt

    def to_string(self):
        return '{} {}'.format(self.tipo, self.numberNode)


    def execute(self):
        if self.tipo == 'FORWARD':
            self.turt.forward(self.numberNode.execute())

        elif self.tipo == 'BACK':
            self.turt.back(self.numberNode.execute())

        elif self.tipo == 'RIGHT':
            self.turt.right(self.numberNode.execute())

        else:
            self.turt.left(self.numberNode.execute())


class MultiNode():

    def __init__(self, numberNode, ops):
        self.ops = ops
        self.numberNode = numberNode

    def execute(self):
        for i in range(self.numberNode.execute()):
            for op in self.ops:
                op.execute()

    def to_string(self):
        res = 'REPEAT {} ['.format(self.numberNode.execute())

        for op in self.ops:
            res += op.to_string() + ', '

        res = res[:-2]
        res += ']'

        return res

class TurtleParser(Parser):

    def __init__(self):
        self.vars = {}
        self.turtle = turtle.Turtle()
        self.turtle.speed(7)
        self.ordenes = []
        self.nodes = []

    # Get the token list from the lexer (required)
    tokens = TurtleLexer.tokens

    debugfile = 'parser.log'

    @_('BEGIN A END')
    def S(self, p):

        for instr in p.A:
            instr.execute()
            self.ordenes.append(instr.to_string())

        return self.ordenes

    @_('F A')
    def A(self, p):

        res = [p.F]

        # Añadimos recursivamente la solucion de A si la hubiese
        res_a = p.A
        if res_a is not None:
            res = [p.F, *p.A]

        return res

    @_('')
    def A(self, p):
        return None

    @_('FORWARD N ";"')
    def F(self, p):
        return UnaryNode('FORWARD', p.N, self.turtle)

    @_('BACK N ";"')
    def F(self, p):
        return UnaryNode('BACK', p.N, self.turtle)

    @_('LEFT N ";"')
    def F(self, p):
        return UnaryNode('LEFT', p.N, self.turtle)

    @_('RIGHT N ";"')
    def F(self, p):
        return UnaryNode('RIGHT', p.N, self.turtle)

    @_('REPEAT N "[" A "]"')
    def F(self, p):
        return MultiNode(p.N, p.A)

    @_('NUMBER')
    def N(self, p):
        return NumberNode(int(p.NUMBER))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    """
    ejemplo1 = "BEGIN" \
               "FORWARD 100;" \
               "BACK 40;" \
               "END"

    ejemplo2 = "BEGIN" \
               "FORWARD 100;" \
               "LEFT 45;" \
               "FORWARD 50;" \
               "END"

    ejemplo3 = "BEGIN" \
               "RIGHT 30;" \
               "FORWARD 50;" \
               "REPEAT 20 [" \
               "FORWARD 40;" \
               "LEFT 15;]" \
               "END"

    ejemplo4 = "BEGIN" \
               "REPEAT 10 [" \
               "FORWARD 15;" \
               "LEFT 15;" \
               "]" \
               "END"

    ejemplos = [ejemplo1, ejemplo2, ejemplo3, ejemplo4]

    lexer = TurtleLexer()
    parser = TurtleParser()

    print(parser.parse(lexer.tokenize(ejemplo3)))
    """

    with open('ejemplo.txt', 'r') as f:
        text = f.read()

    lexer = TurtleLexer()
    parser = TurtleParser()

    print('La ruta ha sido: {}'.format(parser.parse(lexer.tokenize(text))))

    time.sleep(5)