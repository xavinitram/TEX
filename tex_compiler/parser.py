"""
TEX Parser — recursive-descent parser producing an AST from tokens.

Grammar (simplified):
  program     = statement*
  statement   = var_decl | assignment | if_else | for_loop | expr_stmt
  var_decl    = type_kw IDENT ('=' expr)? ';'
  assignment  = lvalue '=' expr ';'
               | lvalue '+=' expr ';'   (desugars to lvalue = lvalue + expr)
               | lvalue '-=' expr ';'
               | lvalue '*=' expr ';'
               | lvalue '/=' expr ';'
               | lvalue '++'  ';'       (desugars to lvalue = lvalue + 1)
               | lvalue '--'  ';'
  if_else     = 'if' '(' expr ')' block ('else' (if_else | block))?
  for_loop    = 'for' '(' for_init ';' expr ';' for_update ')' block
  block       = '{' statement* '}'
  expr        = ternary
  ternary     = logic_or ('?' expr ':' ternary)?
  logic_or    = logic_and ('||' logic_and)*
  logic_and   = equality ('&&' equality)*
  equality    = comparison (('==' | '!=') comparison)*
  comparison  = addition (('<' | '>' | '<=' | '>=') addition)*
  addition    = multiply (('+' | '-') multiply)*
  multiply    = unary (('*' | '/' | '%') unary)*
  unary       = ('-' | '!') unary | postfix
  postfix     = primary ('.' IDENT | '(' args ')')*
  primary     = NUMBER | IDENT | AT_BINDING | '(' expr ')'
               | vec_constructor | cast_expr
"""
from __future__ import annotations
from .lexer import Token, TokenType, LexerError
from .ast_nodes import (
    SourceLoc, Program, VarDecl, Assignment, IfElse, ForLoop, ExprStatement,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor, CastExpr, ASTNode,
    ArrayDecl, ArrayIndexAccess, ArrayLiteral,
)

TYPE_KEYWORDS = {TokenType.KW_FLOAT, TokenType.KW_INT, TokenType.KW_VEC3, TokenType.KW_VEC4, TokenType.KW_STRING}

COMPOUND_ASSIGN_OPS = {
    TokenType.PLUS_ASSIGN: "+",
    TokenType.MINUS_ASSIGN: "-",
    TokenType.STAR_ASSIGN: "*",
    TokenType.SLASH_ASSIGN: "/",
}


class ParseError(Exception):
    def __init__(self, message: str, loc: SourceLoc):
        self.loc = loc
        super().__init__(f"[{loc}] {message}")


class Parser:
    """Recursive-descent parser for TEX."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # -- Helpers --------------------------------------------------------

    def current(self) -> Token:
        return self.tokens[self.pos]

    def peek(self) -> TokenType:
        return self.tokens[self.pos].type

    def peek_ahead(self, offset: int = 1) -> TokenType:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return TokenType.EOF
        return self.tokens[idx].type

    def loc(self) -> SourceLoc:
        return self.current().loc

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return tok

    def expect(self, tt: TokenType, msg: str = "") -> Token:
        if self.peek() != tt:
            tok = self.current()
            what = msg or f"Expected {tt.name}"
            raise ParseError(f"{what}, got {tok.type.name} ({tok.value!r})", tok.loc)
        return self.advance()

    def match(self, *types: TokenType) -> Token | None:
        if self.peek() in types:
            return self.advance()
        return None

    # -- Program --------------------------------------------------------

    def parse(self) -> Program:
        """Parse a complete TEX program."""
        loc = self.loc()
        stmts: list[ASTNode] = []
        while self.peek() != TokenType.EOF:
            stmts.append(self.parse_statement())
        return Program(loc=loc, statements=stmts)

    # -- Statements -----------------------------------------------------

    def parse_statement(self) -> ASTNode:
        # Variable or array declaration: type_kw IDENT ...
        if self.peek() in TYPE_KEYWORDS:
            # But not if it's a cast like float(x) used as expression
            if self.peek_ahead() == TokenType.IDENT:
                # Array declaration: type IDENT '[' ...
                if self.peek_ahead(2) == TokenType.LBRACKET:
                    return self.parse_array_decl()
                return self.parse_var_decl()

        # if/else
        if self.peek() == TokenType.KW_IF:
            return self.parse_if_else()

        # for loop
        if self.peek() == TokenType.KW_FOR:
            return self.parse_for_loop()

        # Assignment or expression statement
        return self.parse_assignment_or_expr()

    def parse_var_decl(self) -> VarDecl:
        loc = self.loc()
        type_tok = self.advance()  # consume type keyword
        type_name = type_tok.value
        name_tok = self.expect(TokenType.IDENT, "Expected variable name after type")

        initializer = None
        if self.match(TokenType.ASSIGN):
            initializer = self.parse_expr()

        self.expect(TokenType.SEMI, "Expected ';' after variable declaration")
        return VarDecl(loc=loc, type_name=type_name, name=name_tok.value, initializer=initializer)

    def parse_array_decl(self) -> ArrayDecl:
        """Parse: type IDENT '[' INT? ']' ('=' ('{' expr_list '}' | IDENT))? ';'"""
        loc = self.loc()
        type_tok = self.advance()  # consume type keyword
        name_tok = self.expect(TokenType.IDENT, "Expected array name after type")
        self.expect(TokenType.LBRACKET, "Expected '[' after array name")

        size = None
        if self.peek() == TokenType.INT_LIT:
            size_tok = self.advance()
            size = int(size_tok.value)
            if size <= 0:
                raise ParseError(f"Array size must be positive, got {size}", size_tok.loc)

        self.expect(TokenType.RBRACKET, "Expected ']' after array size")

        initializer = None
        if self.match(TokenType.ASSIGN):
            if self.peek() == TokenType.LBRACE:
                initializer = self.parse_array_literal()
            else:
                # Array copy: float b[3] = a;
                initializer = self.parse_expr()

        self.expect(TokenType.SEMI, "Expected ';' after array declaration")

        if size is None and initializer is None:
            raise ParseError("Array must have explicit size or initializer", loc)

        return ArrayDecl(
            loc=loc,
            element_type_name=type_tok.value,
            name=name_tok.value,
            size=size,
            initializer=initializer,
        )

    def parse_array_literal(self) -> ArrayLiteral:
        """Parse: '{' expr (',' expr)* '}'"""
        loc = self.loc()
        self.expect(TokenType.LBRACE, "Expected '{' for array literal")
        elements: list[ASTNode] = []
        if self.peek() != TokenType.RBRACE:
            elements.append(self.parse_expr())
            while self.match(TokenType.COMMA):
                elements.append(self.parse_expr())
        self.expect(TokenType.RBRACE, "Expected '}' after array literal")
        return ArrayLiteral(loc=loc, elements=elements)

    def parse_if_else(self) -> IfElse:
        loc = self.loc()
        self.expect(TokenType.KW_IF)
        self.expect(TokenType.LPAREN, "Expected '(' after 'if'")
        condition = self.parse_expr()
        self.expect(TokenType.RPAREN, "Expected ')' after if condition")

        then_body = self.parse_block()

        else_body: list[ASTNode] = []
        if self.match(TokenType.KW_ELSE):
            if self.peek() == TokenType.KW_IF:
                # else if
                else_body = [self.parse_if_else()]
            else:
                else_body = self.parse_block()

        return IfElse(loc=loc, condition=condition, then_body=then_body, else_body=else_body)

    def parse_for_loop(self) -> ForLoop:
        """Parse: for (init; condition; update) { body }"""
        loc = self.loc()
        self.expect(TokenType.KW_FOR)
        self.expect(TokenType.LPAREN, "Expected '(' after 'for'")

        # Init clause: var_decl or assignment (without trailing semicolon — we handle it)
        init = self._parse_for_init()
        self.expect(TokenType.SEMI, "Expected ';' after for-loop initializer")

        # Condition
        condition = self.parse_expr()
        self.expect(TokenType.SEMI, "Expected ';' after for-loop condition")

        # Update clause: assignment, ++, --, compound assign (no semicolon)
        update = self._parse_for_update()
        self.expect(TokenType.RPAREN, "Expected ')' after for-loop update")

        # Body
        body = self.parse_block()

        return ForLoop(loc=loc, init=init, condition=condition, update=update, body=body)

    def _parse_for_init(self) -> ASTNode:
        """Parse the initializer part of a for loop (no trailing semicolon)."""
        if self.peek() in TYPE_KEYWORDS and self.peek_ahead() == TokenType.IDENT:
            loc = self.loc()
            type_tok = self.advance()
            name_tok = self.expect(TokenType.IDENT, "Expected variable name")
            initializer = None
            if self.match(TokenType.ASSIGN):
                initializer = self.parse_expr()
            return VarDecl(loc=loc, type_name=type_tok.value, name=name_tok.value, initializer=initializer)

        # Otherwise, parse as assignment expression
        loc = self.loc()
        expr = self.parse_expr()
        if self.match(TokenType.ASSIGN):
            value = self.parse_expr()
            return Assignment(loc=loc, target=expr, value=value)
        return ExprStatement(loc=loc, expr=expr)

    def _parse_for_update(self) -> ASTNode:
        """Parse the update part of a for loop (no trailing semicolon).

        Supports: i = i + 1, i++, i--, i += 1, etc.
        """
        loc = self.loc()
        expr = self.parse_expr()

        # Check for postfix ++ / --
        if self.match(TokenType.PLUS_PLUS):
            return self._desugar_increment(expr, "+", loc)
        if self.match(TokenType.MINUS_MINUS):
            return self._desugar_increment(expr, "-", loc)

        # Check for compound assignment
        for tok_type, op in COMPOUND_ASSIGN_OPS.items():
            if self.match(tok_type):
                value = self.parse_expr()
                return Assignment(
                    loc=loc,
                    target=expr,
                    value=BinOp(loc=loc, op=op, left=expr, right=value),
                )

        # Check for plain assignment
        if self.match(TokenType.ASSIGN):
            value = self.parse_expr()
            return Assignment(loc=loc, target=expr, value=value)

        return ExprStatement(loc=loc, expr=expr)

    def _desugar_increment(self, target: ASTNode, op: str, loc: SourceLoc) -> Assignment:
        """Desugar i++ -> i = i + 1, i-- -> i = i - 1"""
        one = NumberLiteral(loc=loc, value=1.0, is_int=True)
        return Assignment(
            loc=loc,
            target=target,
            value=BinOp(loc=loc, op=op, left=target, right=one),
        )

    def parse_block(self) -> list[ASTNode]:
        self.expect(TokenType.LBRACE, "Expected '{'")
        stmts: list[ASTNode] = []
        while self.peek() != TokenType.RBRACE and self.peek() != TokenType.EOF:
            stmts.append(self.parse_statement())
        self.expect(TokenType.RBRACE, "Expected '}'")
        return stmts

    def parse_assignment_or_expr(self) -> ASTNode:
        """Parse an expression; if followed by `=`, `+=`, `++`, etc., treat as assignment."""
        loc = self.loc()
        expr = self.parse_expr()

        # Postfix ++ / --
        if self.match(TokenType.PLUS_PLUS):
            stmt = self._desugar_increment(expr, "+", loc)
            self.expect(TokenType.SEMI, "Expected ';' after '++'")
            return stmt
        if self.match(TokenType.MINUS_MINUS):
            stmt = self._desugar_increment(expr, "-", loc)
            self.expect(TokenType.SEMI, "Expected ';' after '--'")
            return stmt

        # Compound assignment operators (+=, -=, *=, /=)
        for tok_type, op in COMPOUND_ASSIGN_OPS.items():
            if self.match(tok_type):
                value = self.parse_expr()
                self.expect(TokenType.SEMI, f"Expected ';' after '{op}=' assignment")
                return Assignment(
                    loc=loc,
                    target=expr,
                    value=BinOp(loc=loc, op=op, left=expr, right=value),
                )

        # Plain assignment
        if self.match(TokenType.ASSIGN):
            value = self.parse_expr()
            self.expect(TokenType.SEMI, "Expected ';' after assignment")
            return Assignment(loc=loc, target=expr, value=value)

        self.expect(TokenType.SEMI, "Expected ';' after expression")
        return ExprStatement(loc=loc, expr=expr)

    # -- Expressions (precedence climbing) ------------------------------

    def parse_expr(self) -> ASTNode:
        return self.parse_ternary()

    def parse_ternary(self) -> ASTNode:
        expr = self.parse_logic_or()
        if self.match(TokenType.QUESTION):
            loc = self.current().loc
            true_expr = self.parse_expr()
            self.expect(TokenType.COLON, "Expected ':' in ternary expression")
            false_expr = self.parse_ternary()
            return TernaryOp(loc=loc, condition=expr, true_expr=true_expr, false_expr=false_expr)
        return expr

    def parse_logic_or(self) -> ASTNode:
        left = self.parse_logic_and()
        while self.peek() == TokenType.OR:
            tok = self.advance()
            right = self.parse_logic_and()
            left = BinOp(loc=tok.loc, op="||", left=left, right=right)
        return left

    def parse_logic_and(self) -> ASTNode:
        left = self.parse_equality()
        while self.peek() == TokenType.AND:
            tok = self.advance()
            right = self.parse_equality()
            left = BinOp(loc=tok.loc, op="&&", left=left, right=right)
        return left

    def parse_equality(self) -> ASTNode:
        left = self.parse_comparison()
        while self.peek() in (TokenType.EQ, TokenType.NEQ):
            tok = self.advance()
            right = self.parse_comparison()
            left = BinOp(loc=tok.loc, op=tok.value, left=left, right=right)
        return left

    def parse_comparison(self) -> ASTNode:
        left = self.parse_addition()
        while self.peek() in (TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            tok = self.advance()
            right = self.parse_addition()
            left = BinOp(loc=tok.loc, op=tok.value, left=left, right=right)
        return left

    def parse_addition(self) -> ASTNode:
        left = self.parse_multiply()
        while self.peek() in (TokenType.PLUS, TokenType.MINUS):
            tok = self.advance()
            right = self.parse_multiply()
            left = BinOp(loc=tok.loc, op=tok.value, left=left, right=right)
        return left

    def parse_multiply(self) -> ASTNode:
        left = self.parse_unary()
        while self.peek() in (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            tok = self.advance()
            right = self.parse_unary()
            left = BinOp(loc=tok.loc, op=tok.value, left=left, right=right)
        return left

    def parse_unary(self) -> ASTNode:
        if self.peek() in (TokenType.MINUS, TokenType.NOT):
            tok = self.advance()
            operand = self.parse_unary()
            return UnaryOp(loc=tok.loc, op=tok.value, operand=operand)
        return self.parse_postfix()

    def parse_postfix(self) -> ASTNode:
        expr = self.parse_primary()
        while True:
            if self.peek() == TokenType.DOT:
                self.advance()  # consume .
                # Read channel/swizzle name
                name_tok = self.expect(TokenType.IDENT, "Expected channel name after '.'")
                expr = ChannelAccess(loc=name_tok.loc, object=expr, channels=name_tok.value)
            elif self.peek() == TokenType.LBRACKET:
                # Array indexing: expr[index]
                loc = self.loc()
                self.advance()  # consume [
                index = self.parse_expr()
                self.expect(TokenType.RBRACKET, "Expected ']' after array index")
                expr = ArrayIndexAccess(loc=loc, array=expr, index=index)
            elif self.peek() == TokenType.LPAREN and isinstance(expr, Identifier):
                # Function call: name(args)
                expr = self._parse_call(expr)
            else:
                break
        return expr

    def _parse_call(self, callee: Identifier) -> FunctionCall:
        loc = callee.loc
        self.expect(TokenType.LPAREN)
        args: list[ASTNode] = []
        if self.peek() != TokenType.RPAREN:
            args.append(self.parse_expr())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expr())
        self.expect(TokenType.RPAREN, "Expected ')' after function arguments")
        return FunctionCall(loc=loc, name=callee.name, args=args)

    def parse_primary(self) -> ASTNode:
        tok = self.current()

        # Numeric literals
        if tok.type == TokenType.INT_LIT:
            self.advance()
            if tok.value.startswith("0x") or tok.value.startswith("0X"):
                val = int(tok.value, 16)
            else:
                val = int(tok.value)
            return NumberLiteral(loc=tok.loc, value=float(val), is_int=True)

        if tok.type == TokenType.FLOAT_LIT:
            self.advance()
            return NumberLiteral(loc=tok.loc, value=float(tok.value), is_int=False)

        # String literals
        if tok.type == TokenType.STRING_LIT:
            self.advance()
            return StringLiteral(loc=tok.loc, value=tok.value)

        # Identifiers (variable refs or function calls handled in postfix)
        if tok.type == TokenType.IDENT:
            self.advance()
            return Identifier(loc=tok.loc, name=tok.value)

        # @ bindings
        if tok.type == TokenType.AT_BINDING:
            self.advance()
            return BindingRef(loc=tok.loc, name=tok.value)

        # Vector constructors: vec3(...) / vec4(...)
        if tok.type in (TokenType.KW_VEC3, TokenType.KW_VEC4):
            return self._parse_vec_constructor()

        # Cast expressions: float(...) / int(...) / string(...)
        if tok.type in (TokenType.KW_FLOAT, TokenType.KW_INT, TokenType.KW_STRING):
            if self.peek_ahead() == TokenType.LPAREN:
                return self._parse_cast()
            # Otherwise it's a type keyword used where an expression is expected
            raise ParseError(f"Unexpected type keyword '{tok.value}' in expression", tok.loc)

        # Parenthesized expression
        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN, "Expected ')'")
            return expr

        raise ParseError(f"Unexpected token: {tok.type.name} ({tok.value!r})", tok.loc)

    def _parse_vec_constructor(self) -> VecConstructor:
        tok = self.advance()  # vec3 or vec4
        size = 3 if tok.type == TokenType.KW_VEC3 else 4
        self.expect(TokenType.LPAREN, f"Expected '(' after {tok.value}")
        args: list[ASTNode] = []
        if self.peek() != TokenType.RPAREN:
            args.append(self.parse_expr())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expr())
        self.expect(TokenType.RPAREN, f"Expected ')' after {tok.value} arguments")
        return VecConstructor(loc=tok.loc, size=size, args=args)

    def _parse_cast(self) -> CastExpr:
        tok = self.advance()  # float or int
        self.expect(TokenType.LPAREN, f"Expected '(' after {tok.value}")
        expr = self.parse_expr()
        self.expect(TokenType.RPAREN, f"Expected ')' after cast")
        return CastExpr(loc=tok.loc, target_type=tok.value, expr=expr)
