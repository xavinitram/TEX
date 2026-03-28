"""
TEX Parser — recursive-descent parser producing an AST from tokens.

Grammar (simplified):
  program     = statement*
  statement   = func_def | var_decl | param_decl | assignment | if_else | for_loop | while_loop | return_stmt | expr_stmt
  func_def    = type_kw IDENT '(' param_list ')' block
  param_list  = (type_kw IDENT (',' type_kw IDENT)*)?
  var_decl    = type_kw IDENT ('=' expr)? ';'
  param_decl  = ('$'|typed_'$') IDENT ('=' expr)? ';'
  assignment  = lvalue '=' expr ';'
               | lvalue '+=' expr ';'   (desugars to lvalue = lvalue + expr)
               | lvalue '-=' expr ';'
               | lvalue '*=' expr ';'
               | lvalue '/=' expr ';'
               | lvalue '++'  ';'       (desugars to lvalue = lvalue + 1)
               | lvalue '--'  ';'
  if_else     = 'if' '(' expr ')' block ('else' (if_else | block))?
  for_loop    = 'for' '(' for_init ';' expr ';' for_update ')' block
  return_stmt = 'return' expr ';'
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
  postfix     = primary ('.' IDENT | '[' args ']' | '(' args ')')*
  primary     = NUMBER | IDENT | AT_BINDING | DOLLAR_BINDING
               | TYPED_AT_BINDING | TYPED_DOLLAR_BINDING
               | '(' expr ')' | vec_constructor | cast_expr
"""
from __future__ import annotations
from .lexer import Token, TokenType
from .ast_nodes import (
    SourceLoc, Program, VarDecl, Assignment, IfElse, ForLoop, WhileLoop, ExprStatement,
    BreakStmt, ContinueStmt, FunctionDef, ReturnStmt,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor, CastExpr, ASTNode,
    ArrayDecl, ArrayIndexAccess, ArrayLiteral, MatConstructor, ParamDecl,
    BindingIndexAccess, BindingSampleAccess,
)

TYPE_KEYWORDS = {TokenType.KW_FLOAT, TokenType.KW_INT, TokenType.KW_VEC2, TokenType.KW_VEC3, TokenType.KW_VEC4, TokenType.KW_STRING, TokenType.KW_MAT3, TokenType.KW_MAT4}

COMPOUND_ASSIGN_OPS = {
    TokenType.PLUS_ASSIGN: "+",
    TokenType.MINUS_ASSIGN: "-",
    TokenType.STAR_ASSIGN: "*",
    TokenType.SLASH_ASSIGN: "/",
}


class ParseError(Exception):
    def __init__(self, message: str, loc: SourceLoc, *, source: str = "",
                 code: str = "E2000", hint: str = "", end_col: int | None = None):
        self.loc = loc
        self.diagnostic = None  # Built lazily
        self._raw_message = message
        self._source = source
        self._code = code
        self._hint = hint
        self._end_col = end_col
        super().__init__(f"[{loc}] {message}")

    def _build_diagnostic(self):
        if self.diagnostic is not None:
            return
        from .diagnostics import make_diagnostic
        self.diagnostic = make_diagnostic(
            code=self._code,
            message=self._raw_message,
            loc=self.loc,
            source=self._source,
            end_col=self._end_col,
            hint=self._hint,
            phase="parser",
        )


class Parser:
    """Recursive-descent parser for TEX."""

    def __init__(self, tokens: list[Token], source: str = ""):
        self.tokens = tokens
        self.pos = 0
        self._source = source
        self._errors: list[ParseError] = []

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

    def expect(self, tt: TokenType, msg: str = "", *,
               code: str = "E2000", hint: str = "") -> Token:
        if self.peek() != tt:
            tok = self.current()
            if msg:
                what = msg
            elif tt == TokenType.SEMI:
                what = "I expected a semicolon here"
            else:
                what = f"I expected {tt.name} here"
            # Describe what we got in plain language
            if tok.type == TokenType.EOF:
                got_desc = "but the program ends here"
            else:
                got_desc = f"but found `{tok.value}` instead"
            raise self._make_error(f"{what}, {got_desc}.", tok.loc,
                                   code=code, hint=hint)
        return self.advance()

    def match(self, *types: TokenType) -> Token | None:
        if self.peek() in types:
            return self.advance()
        return None

    def _make_error(self, message: str, loc: SourceLoc, *,
                    code: str = "E2000", hint: str = "") -> ParseError:
        """Create a ParseError with source context."""
        return ParseError(message, loc, source=self._source, code=code, hint=hint)

    def _synchronize(self):
        """Panic-mode recovery: skip tokens until a synchronization point."""
        while self.peek() != TokenType.EOF:
            # Stop AFTER consuming a semicolon
            if self.peek() == TokenType.SEMI:
                self.advance()
                return
            # Stop BEFORE a statement-starting token (don't consume it)
            if self.peek() in (TokenType.RBRACE, TokenType.KW_IF,
                               TokenType.KW_FOR, TokenType.KW_WHILE,
                               TokenType.KW_BREAK, TokenType.KW_CONTINUE,
                               TokenType.KW_RETURN,
                               *TYPE_KEYWORDS):
                return
            self.advance()

    # -- Program --------------------------------------------------------

    def parse(self) -> Program:
        """Parse a complete TEX program with error recovery."""
        from .ast_nodes import ErrorNode
        from .diagnostics import TEXMultiError
        loc = self.loc()
        stmts: list[ASTNode] = []
        while self.peek() != TokenType.EOF:
            try:
                stmts.append(self.parse_statement())
            except ParseError as e:
                self._errors.append(e)
                stmts.append(ErrorNode(loc=e.loc, error_message=str(e)))
                self._synchronize()

        if self._errors:
            # Build diagnostics for all collected errors
            for e in self._errors:
                e._build_diagnostic()
            if len(self._errors) == 1:
                raise self._errors[0]
            diagnostics = [e.diagnostic for e in self._errors if e.diagnostic]
            raise TEXMultiError(diagnostics)

        return Program(loc=loc, statements=stmts)

    # -- Statements -----------------------------------------------------

    def parse_statement(self) -> ASTNode:
        # const qualifier: const type_kw IDENT = expr;
        if self.peek() == TokenType.KW_CONST:
            return self._parse_const_decl()

        # Variable or array declaration: type_kw IDENT ...
        if self.peek() in TYPE_KEYWORDS:
            # But not if it's a cast like float(x) used as expression
            if self.peek_ahead() == TokenType.IDENT:
                # Array declaration: type IDENT '[' ...
                if self.peek_ahead(2) == TokenType.LBRACKET:
                    return self.parse_array_decl()
                # Function definition: type IDENT '(' ...
                if self.peek_ahead(2) == TokenType.LPAREN:
                    return self.parse_function_def()
                return self.parse_var_decl()

        # Parameter declaration: f$name = 0.5; or $name; or i$count = 10;
        if self.peek() == TokenType.TYPED_DOLLAR_BINDING:
            return self.parse_param_decl()
        if self.peek() == TokenType.DOLLAR_BINDING:
            if self.peek_ahead() in (TokenType.ASSIGN, TokenType.SEMI):
                return self.parse_param_decl()

        # if/else
        if self.peek() == TokenType.KW_IF:
            return self.parse_if_else()

        # for loop
        if self.peek() == TokenType.KW_FOR:
            return self.parse_for_loop()

        # while loop
        if self.peek() == TokenType.KW_WHILE:
            return self.parse_while_loop()

        # break / continue
        if self.peek() == TokenType.KW_BREAK:
            loc = self.loc()
            self.advance()
            self.expect(TokenType.SEMI, "I need a semicolon after `break`",
                       code="E2010", hint="Add `;` after break.")
            return BreakStmt(loc=loc)
        if self.peek() == TokenType.KW_CONTINUE:
            loc = self.loc()
            self.advance()
            self.expect(TokenType.SEMI, "I need a semicolon after `continue`",
                       code="E2010", hint="Add `;` after continue.")
            return ContinueStmt(loc=loc)

        # return statement
        if self.peek() == TokenType.KW_RETURN:
            loc = self.loc()
            self.advance()
            value = self.parse_expr()
            self.expect(TokenType.SEMI, "I need a semicolon after `return`",
                       code="E2010", hint="Add `;` after the return value.")
            return ReturnStmt(loc=loc, value=value)

        # Check for foreign keywords (const, let, var, etc.)
        if self.peek() == TokenType.IDENT:
            from .diagnostics import get_keyword_hint
            kw_hint = get_keyword_hint(self.current().value)
            if kw_hint is not None:
                tok = self.current()
                raise self._make_error(
                    f"Unexpected keyword '{tok.value}'.",
                    tok.loc, code="E2001", hint=kw_hint
                )

        # Assignment or expression statement
        return self.parse_assignment_or_expr()

    def parse_var_decl(self) -> VarDecl:
        loc = self.loc()
        type_tok = self.advance()  # consume type keyword
        type_name = type_tok.value
        name_tok = self.expect(TokenType.IDENT, "I expected a variable name after the type")

        initializer = None
        if self.match(TokenType.ASSIGN):
            initializer = self.parse_expr()

        self.expect(TokenType.SEMI, "It looks like there's a missing semicolon after this variable declaration",
                   code="E2010", hint="Every statement in TEX ends with `;`.")
        return VarDecl(loc=loc, type_name=type_name, name=name_tok.value, initializer=initializer)

    def _parse_const_decl(self) -> VarDecl:
        loc = self.loc()
        self.advance()  # consume 'const'
        if self.peek() not in TYPE_KEYWORDS:
            raise self._make_error(
                "Expected a type after 'const'.",
                self.current().loc, code="E2001",
                hint="Use: const float x = 1.0; or const vec3 color = vec3(1.0);",
            )
        type_tok = self.advance()
        name_tok = self.expect(TokenType.IDENT, "Expected a variable name after the type")
        if not self.match(TokenType.ASSIGN):
            raise self._make_error(
                "A 'const' variable must be initialized.",
                name_tok.loc, code="E2010",
                hint="Add an initializer: const float x = 1.0;",
            )
        initializer = self.parse_expr()
        self.expect(TokenType.SEMI, "Missing semicolon after const declaration",
                   code="E2010", hint="Every statement in TEX ends with `;`.")
        return VarDecl(loc=loc, type_name=type_tok.value, name=name_tok.value,
                       initializer=initializer, is_const=True)

    def parse_function_def(self) -> FunctionDef:
        loc = self.loc()
        return_type = self.advance().value  # consume type keyword

        name_tok = self.expect(TokenType.IDENT, "I expected a function name")
        self.expect(TokenType.LPAREN, "I expected '(' after function name")

        params: list[tuple[str, str]] = []
        if self.peek() in TYPE_KEYWORDS:
            type_tok = self.advance()
            p_name = self.expect(TokenType.IDENT, "I expected a parameter name")
            params.append((type_tok.value, p_name.value))
            while self.match(TokenType.COMMA):
                if self.peek() not in TYPE_KEYWORDS:
                    raise self._make_error(
                        "I expected a type for this parameter.",
                        self.loc(), code="E2020",
                        hint="Each parameter needs a type: float x, vec3 color, etc.",
                    )
                type_tok = self.advance()
                p_name = self.expect(TokenType.IDENT, "I expected a parameter name")
                params.append((type_tok.value, p_name.value))

        self.expect(TokenType.RPAREN, "I expected ')' to close the parameter list")
        body = self.parse_block()
        return FunctionDef(loc=loc, return_type=return_type, name=name_tok.value,
                           params=params, body=body)

    def parse_param_decl(self) -> ParamDecl:
        """Parse: ($name | prefix$name) ('=' expr)? ';'"""
        loc = self.loc()
        tok = self.advance()  # TYPED_DOLLAR_BINDING or DOLLAR_BINDING
        default_expr = None
        if self.match(TokenType.ASSIGN):
            default_expr = self.parse_expr()
        self.expect(TokenType.SEMI, "It looks like there's a missing semicolon after this parameter declaration",
                   code="E2010", hint="Every statement in TEX ends with `;`.")
        return ParamDecl(
            loc=loc,
            name=tok.value,
            type_hint=tok.prefix,
            default_expr=default_expr,
        )

    def parse_array_decl(self) -> ArrayDecl:
        """Parse: type IDENT '[' INT? ']' ('=' ('{' expr_list '}' | IDENT))? ';'"""
        loc = self.loc()
        type_tok = self.advance()  # consume type keyword
        name_tok = self.expect(TokenType.IDENT, "I expected an array name after the type")
        self.expect(TokenType.LBRACKET, "I expected `[` after the array name")

        size = None
        if self.peek() == TokenType.INT_LIT:
            size_tok = self.advance()
            size = int(size_tok.value)
            if size <= 0:
                raise self._make_error(f"Array size must be positive, got {size}.",
                                      size_tok.loc, code="E2004")

        self.expect(TokenType.RBRACKET, "I expected `]` to close the array size")

        initializer = None
        if self.match(TokenType.ASSIGN):
            if self.peek() == TokenType.LBRACE:
                initializer = self.parse_array_literal()
            else:
                # Array copy: float b[3] = a;
                initializer = self.parse_expr()

        self.expect(TokenType.SEMI, "It looks like there's a missing semicolon after this array declaration",
                   code="E2010", hint="Every statement in TEX ends with `;`.")

        if size is None and initializer is None:
            raise self._make_error("Array must have explicit size or initializer. For example: float arr[3]; or float arr[] = {1.0, 2.0};",
                                  loc, code="E2005")

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
        self.expect(TokenType.LBRACE, "I expected `{` to start the array literal")
        elements: list[ASTNode] = []
        if self.peek() != TokenType.RBRACE:
            elements.append(self.parse_expr())
            while self.match(TokenType.COMMA):
                elements.append(self.parse_expr())
        self.expect(TokenType.RBRACE, "I expected `}` to close the array literal")
        return ArrayLiteral(loc=loc, elements=elements)

    def parse_if_else(self) -> IfElse:
        loc = self.loc()
        self.expect(TokenType.KW_IF)
        self.expect(TokenType.LPAREN, "I expected `(` after `if`")
        condition = self.parse_expr()
        self.expect(TokenType.RPAREN, "I expected `)` after the if condition")

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
        self.expect(TokenType.LPAREN, "I expected `(` after `for`")

        # Init clause: var_decl or assignment (without trailing semicolon — we handle it)
        init = self._parse_for_init()
        self.expect(TokenType.SEMI, "I need a semicolon after the for-loop initializer",
                   code="E2010", hint="for-loops use the pattern: for (init; condition; update) { ... }")

        # Condition
        condition = self.parse_expr()
        self.expect(TokenType.SEMI, "I need a semicolon after the for-loop condition",
                   code="E2010", hint="for-loops use the pattern: for (init; condition; update) { ... }")

        # Update clause: assignment, ++, --, compound assign (no semicolon)
        update = self._parse_for_update()
        self.expect(TokenType.RPAREN, "I expected `)` to close the for-loop header")

        # Body
        body = self.parse_block()

        return ForLoop(loc=loc, init=init, condition=condition, update=update, body=body)

    def parse_while_loop(self) -> WhileLoop:
        """Parse: while (condition) { body }"""
        loc = self.loc()
        self.expect(TokenType.KW_WHILE)
        self.expect(TokenType.LPAREN, "I expected `(` after `while`")
        condition = self.parse_expr()
        self.expect(TokenType.RPAREN, "I expected `)` after the while condition")
        body = self.parse_block()
        return WhileLoop(loc=loc, condition=condition, body=body)

    def _parse_for_init(self) -> ASTNode:
        """Parse the initializer part of a for loop (no trailing semicolon)."""
        if self.peek() in TYPE_KEYWORDS and self.peek_ahead() == TokenType.IDENT:
            loc = self.loc()
            type_tok = self.advance()
            name_tok = self.expect(TokenType.IDENT, "I expected a variable name here")
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
        self.expect(TokenType.LBRACE, "I expected `{` to start a block")
        stmts: list[ASTNode] = []
        while self.peek() != TokenType.RBRACE and self.peek() != TokenType.EOF:
            stmts.append(self.parse_statement())
        self.expect(TokenType.RBRACE, "I expected `}` to close this block")
        return stmts

    def parse_assignment_or_expr(self) -> ASTNode:
        """Parse an expression; if followed by `=`, `+=`, `++`, etc., treat as assignment."""
        loc = self.loc()
        expr = self.parse_expr()

        # Postfix ++ / --
        if self.match(TokenType.PLUS_PLUS):
            stmt = self._desugar_increment(expr, "+", loc)
            self.expect(TokenType.SEMI, "I need a semicolon after `++`",
                       code="E2010", hint="Add `;` to end this statement.")
            return stmt
        if self.match(TokenType.MINUS_MINUS):
            stmt = self._desugar_increment(expr, "-", loc)
            self.expect(TokenType.SEMI, "I need a semicolon after `--`",
                       code="E2010", hint="Add `;` to end this statement.")
            return stmt

        # Compound assignment operators (+=, -=, *=, /=)
        for tok_type, op in COMPOUND_ASSIGN_OPS.items():
            if self.match(tok_type):
                value = self.parse_expr()
                self.expect(TokenType.SEMI, f"It looks like there's a missing semicolon after this `{op}=` assignment",
                           code="E2010", hint="Every statement in TEX ends with `;`.")
                # For scatter targets (@OUT[x,y] += val), preserve op for scatter_add_
                if isinstance(expr, BindingIndexAccess):
                    return Assignment(loc=loc, target=expr, value=value, op=op)
                return Assignment(
                    loc=loc,
                    target=expr,
                    value=BinOp(loc=loc, op=op, left=expr, right=value),
                )

        # Plain assignment
        if self.match(TokenType.ASSIGN):
            value = self.parse_expr()
            self.expect(TokenType.SEMI, "It looks like there's a missing semicolon after this assignment",
                       code="E2010", hint="Every statement in TEX ends with `;`.")
            return Assignment(loc=loc, target=expr, value=value)

        self.expect(TokenType.SEMI, "It looks like there's a missing semicolon after this expression",
                   code="E2010", hint="Every statement in TEX ends with `;`.")
        return ExprStatement(loc=loc, expr=expr)

    # -- Expressions (precedence climbing) ------------------------------

    def parse_expr(self) -> ASTNode:
        return self.parse_ternary()

    def parse_ternary(self) -> ASTNode:
        expr = self.parse_logic_or()
        if self.match(TokenType.QUESTION):
            loc = expr.loc
            true_expr = self.parse_expr()
            self.expect(TokenType.COLON, "I expected `:` in this ternary expression (condition ? then : else)")
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
                name_tok = self.expect(TokenType.IDENT, "I expected a channel name after `.`")
                expr = ChannelAccess(loc=name_tok.loc, object=expr, channels=name_tok.value)
            elif self.peek() == TokenType.LBRACKET and isinstance(expr, BindingRef):
                # Binding fetch: @Image[ix, iy] or @Image[ix, iy, frame]
                loc = self.loc()
                self.advance()  # consume [
                args = [self.parse_expr()]
                while self.match(TokenType.COMMA):
                    args.append(self.parse_expr())
                self.expect(TokenType.RBRACKET, "I expected `]` to close the binding index access")
                expr = BindingIndexAccess(loc=loc, binding=expr, args=args)
            elif self.peek() == TokenType.LPAREN and isinstance(expr, BindingRef):
                # Binding sample: @Image(u, v) or @Image(u, v, frame)
                loc = self.loc()
                self.advance()  # consume (
                args = [self.parse_expr()]
                while self.match(TokenType.COMMA):
                    args.append(self.parse_expr())
                self.expect(TokenType.RPAREN, "I expected `)` to close the binding sample access")
                expr = BindingSampleAccess(loc=loc, binding=expr, args=args)
            elif self.peek() == TokenType.LBRACKET:
                # Array indexing: expr[index]
                loc = self.loc()
                self.advance()  # consume [
                index = self.parse_expr()
                self.expect(TokenType.RBRACKET, "I expected `]` to close the array index")
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
        self.expect(TokenType.RPAREN, "I expected `)` to close the function call")
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

        # @ bindings (untyped)
        if tok.type == TokenType.AT_BINDING:
            self.advance()
            return BindingRef(loc=tok.loc, name=tok.value)

        # Typed @ bindings (e.g. f@threshold, img@result)
        if tok.type == TokenType.TYPED_AT_BINDING:
            self.advance()
            return BindingRef(loc=tok.loc, name=tok.value, kind="wire", type_hint=tok.prefix)

        # $ parameter bindings (untyped)
        if tok.type == TokenType.DOLLAR_BINDING:
            self.advance()
            return BindingRef(loc=tok.loc, name=tok.value, kind="param")

        # Typed $ parameter bindings (e.g. f$strength, i$count)
        if tok.type == TokenType.TYPED_DOLLAR_BINDING:
            self.advance()
            return BindingRef(loc=tok.loc, name=tok.value, kind="param", type_hint=tok.prefix)

        # Vector constructors: vec3(...) / vec4(...)
        if tok.type in (TokenType.KW_VEC2, TokenType.KW_VEC3, TokenType.KW_VEC4):
            return self._parse_vec_constructor()

        # Matrix constructors: mat3(...) / mat4(...)
        if tok.type in (TokenType.KW_MAT3, TokenType.KW_MAT4):
            return self._parse_mat_constructor()

        # Cast expressions: float(...) / int(...) / string(...)
        if tok.type in (TokenType.KW_FLOAT, TokenType.KW_INT, TokenType.KW_STRING):
            if self.peek_ahead() == TokenType.LPAREN:
                return self._parse_cast()
            # Otherwise it's a type keyword used where an expression is expected
            raise self._make_error(
                f"Unexpected type keyword '{tok.value}' in expression.",
                tok.loc, code="E2002",
                hint=f"To cast a value, use {tok.value}(expr). To declare a variable, write: {tok.value} name = expr;")

        # Parenthesized expression
        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN, "I expected `)` here")
            return expr

        raise self._make_error(f"Unexpected token: {tok.type.name} ({tok.value!r}).",
                               tok.loc, code="E2003")

    def _parse_vec_constructor(self) -> VecConstructor:
        tok = self.advance()  # vec3 or vec4
        size = 2 if tok.type == TokenType.KW_VEC2 else 3 if tok.type == TokenType.KW_VEC3 else 4
        self.expect(TokenType.LPAREN, f"Expected '(' after {tok.value}")
        args: list[ASTNode] = []
        if self.peek() != TokenType.RPAREN:
            args.append(self.parse_expr())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expr())
        self.expect(TokenType.RPAREN, f"Expected ')' after {tok.value} arguments")
        return VecConstructor(loc=tok.loc, size=size, args=args)

    def _parse_mat_constructor(self) -> MatConstructor:
        tok = self.advance()  # mat3 or mat4
        size = 3 if tok.type == TokenType.KW_MAT3 else 4
        self.expect(TokenType.LPAREN, f"Expected '(' after {tok.value}")
        args: list[ASTNode] = []
        if self.peek() != TokenType.RPAREN:
            args.append(self.parse_expr())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expr())
        self.expect(TokenType.RPAREN, f"Expected ')' after {tok.value} arguments")
        return MatConstructor(loc=tok.loc, size=size, args=args)

    def _parse_cast(self) -> CastExpr:
        tok = self.advance()  # float or int
        self.expect(TokenType.LPAREN, f"Expected '(' after {tok.value}")
        expr = self.parse_expr()
        self.expect(TokenType.RPAREN, f"Expected ')' after cast")
        return CastExpr(loc=tok.loc, target_type=tok.value, expr=expr)
