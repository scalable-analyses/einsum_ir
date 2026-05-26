"""Recursive-descent parser for the textual TEIR format."""

from __future__ import annotations

from etops.diag import TeirEmissionError
from etops.ir import First, Guard, Last, Teir, TeirBuilder, guard
from etops.textir.lexer import Token, TokenKind, tokenize

__all__ = ["parse"]


SUPPORTED_FORMAT_VERSIONS = ("1.0",)


def parse(text: str, *, validate: bool = True) -> Teir:
    """Parse a textual TEIR document into a `Teir`.

    Args:
        text: The textual TEIR source.
        validate: When True (the default), run global validation after
            building. Pass False to obtain a structurally consistent but
            potentially semantically invalid `Teir` — for example, in
            interactive editors that want to pretty-print malformed files.
    """

    tokens = tokenize(text)
    parser = _Parser(tokens)
    return parser.parse_document(validate=validate)


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ----- token helpers -----

    @property
    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.kind is not TokenKind.EOF:
            self._pos += 1
        return tok

    def _accept_kw(self, keyword: str) -> bool:
        tok = self._peek
        if tok.kind is TokenKind.IDENT and tok.text == keyword:
            self._advance()
            return True
        return False

    def _expect_kw(self, keyword: str) -> Token:
        if not self._accept_kw(keyword):
            tok = self._peek
            raise TeirEmissionError(
                f"expected keyword {keyword!r} at line {tok.line}, col {tok.col}; got {tok.text!r}"
            )
        return self._tokens[self._pos - 1]

    def _expect(self, kind: TokenKind) -> Token:
        tok = self._peek
        if tok.kind is not kind:
            raise TeirEmissionError(
                f"expected {kind.name} at line {tok.line}, col {tok.col}; got {tok.kind.name} {tok.text!r}"
            )
        return self._advance()

    def _accept(self, kind: TokenKind) -> bool:
        if self._peek.kind is kind:
            self._advance()
            return True
        return False

    def _expect_int(self) -> int:
        tok = self._expect(TokenKind.INT)
        try:
            return int(tok.text)
        except ValueError as exc:  # pragma: no cover - lexer prevents this
            raise TeirEmissionError(
                f"invalid integer {tok.text!r} at line {tok.line}, col {tok.col}"
            ) from exc

    # ----- document -----

    def parse_document(self, *, validate: bool = True) -> Teir:
        self._maybe_format_header()
        builder = TeirBuilder()
        self._expect_kw("teir")
        # Optional @name
        if self._peek.kind is TokenKind.SIGIL_AT:
            builder.set_name(self._advance().text)
        self._expect(TokenKind.LBRACE)
        while self._peek.kind is not TokenKind.RBRACE:
            if self._peek.kind is TokenKind.EOF:
                raise TeirEmissionError("unexpected end of input inside teir block")
            self._parse_top_level_decl(builder)
        self._expect(TokenKind.RBRACE)
        return builder.finish(validate=validate)

    def _maybe_format_header(self) -> None:
        # Header is optional: 'teir-format <version>' must appear before the
        # 'teir' keyword.
        tok = self._peek
        if not (tok.kind is TokenKind.IDENT and tok.text == "teir-format"):
            return
        self._advance()
        version_tok = self._advance()
        if version_tok.kind not in (TokenKind.FLOAT, TokenKind.INT):
            raise TeirEmissionError(
                f"teir-format expects a version literal at line {version_tok.line}, col {version_tok.col}"
            )
        version = version_tok.text
        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise TeirEmissionError(
                f"unsupported textir format version {version!r};"
                f" supported: {list(SUPPORTED_FORMAT_VERSIONS)}"
            )

    def _parse_top_level_decl(self, b: TeirBuilder) -> None:
        tok = self._peek
        if tok.kind is not TokenKind.IDENT:
            raise TeirEmissionError(
                f"expected declaration keyword at line {tok.line}, col {tok.col};"
                f" got {tok.kind.name} {tok.text!r}"
            )
        kw = tok.text
        if kw == "tensor":
            self._parse_tensor(b)
        elif kw == "axis":
            self._parse_axis(b)
        elif kw == "primitive":
            self._parse_primitive(b)
        elif kw == "schedule":
            self._parse_schedule(b)
        else:
            raise TeirEmissionError(
                f"unknown top-level keyword {kw!r} at line {tok.line}, col {tok.col}"
            )

    # ----- tensor -----

    def _parse_tensor(self, b: TeirBuilder) -> None:
        self._expect_kw("tensor")
        name_tok = self._expect(TokenKind.SIGIL_PERCENT)
        self._expect(TokenKind.COLON)
        dtype_tok = self._expect(TokenKind.IDENT)
        b.add_tensor(name_tok.text, dtype=dtype_tok.text)

    # ----- axis -----

    def _parse_axis(self, b: TeirBuilder) -> None:
        self._expect_kw("axis")
        name_tok = self._expect(TokenKind.SIGIL_AT)
        self._expect_kw("extent")
        extent = self._expect_int()
        strides: dict[str, int] = {}
        offsets: dict[str, int] = {}
        # Optional 'strides { ... }' and 'offsets { ... }' in either order.
        while True:
            if self._accept_kw("strides"):
                strides = self._parse_tensor_int_map()
            elif self._accept_kw("offsets"):
                offsets = self._parse_tensor_int_map()
            else:
                break
        b.add_axis(
            name_tok.text,
            extent=extent,
            strides_by_tensor=strides,
            offsets_by_tensor=offsets,
        )

    def _parse_tensor_int_map(self) -> dict[str, int]:
        result: dict[str, int] = {}

        def _parse_entry() -> None:
            key_tok = self._expect(TokenKind.IDENT)
            self._expect(TokenKind.COLON)
            if key_tok.text in result:
                raise TeirEmissionError(
                    f"duplicate tensor key {key_tok.text!r} in stride/offset map"
                    f" at line {key_tok.line}, col {key_tok.col}"
                )
            result[key_tok.text] = self._expect_int()

        self._parse_comma_separated(TokenKind.LBRACE, TokenKind.RBRACE, _parse_entry)
        return result

    def _parse_comma_separated(
        self,
        open_tok: TokenKind,
        close_tok: TokenKind,
        parse_item: object,
    ) -> None:
        """Parse a ``{ a, b, c }`` / ``[ a, b, c ]`` style comma list.

        Calls ``parse_item()`` once per element. Trailing commas are not
        accepted (the textual IR's grammar reserves space for them in a
        future version revision).
        """

        self._expect(open_tok)
        first = True
        while self._peek.kind is not close_tok:
            if not first:
                self._expect(TokenKind.COMMA)
            first = False
            parse_item()  # type: ignore[operator]
        self._expect(close_tok)

    # ----- primitive -----

    def _parse_primitive(self, b: TeirBuilder) -> None:
        self._expect_kw("primitive")
        name_tok = self._expect(TokenKind.SIGIL_AT)
        self._expect(TokenKind.COLON)
        op_tok = self._expect(TokenKind.IDENT)
        roles: dict[str, list[str]] = {}
        metadata: dict[str, object] = {}
        if self._accept_kw("axes"):
            roles = self._parse_role_map()
        if self._accept_kw("metadata"):
            metadata = self._parse_metadata_map()
        b.add_primitive(
            name_tok.text,
            operation=op_tok.text,
            axes=roles,
            metadata=metadata,
        )

    def _parse_role_map(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}

        def _parse_entry() -> None:
            role_tok = self._expect(TokenKind.IDENT)
            self._expect(TokenKind.COLON)
            if role_tok.text in result:
                raise TeirEmissionError(
                    f"duplicate role key {role_tok.text!r} in primitive axes map"
                    f" at line {role_tok.line}, col {role_tok.col}"
                )
            axes: list[str] = []

            def _parse_axis() -> None:
                axes.append(self._expect(TokenKind.SIGIL_AT).text)

            self._parse_comma_separated(TokenKind.LBRACK, TokenKind.RBRACK, _parse_axis)
            result[role_tok.text] = axes

        self._parse_comma_separated(TokenKind.LBRACE, TokenKind.RBRACE, _parse_entry)
        return result

    def _parse_metadata_map(self) -> dict[str, object]:
        result: dict[str, object] = {}

        def _parse_entry() -> None:
            key_tok = self._expect(TokenKind.IDENT)
            self._expect(TokenKind.COLON)
            if key_tok.text in result:
                raise TeirEmissionError(
                    f"duplicate metadata key {key_tok.text!r}"
                    f" at line {key_tok.line}, col {key_tok.col}"
                )
            value_tok = self._peek
            if value_tok.kind is TokenKind.INT:
                self._advance()
                result[key_tok.text] = int(value_tok.text)
            elif value_tok.kind is TokenKind.FLOAT:
                self._advance()
                result[key_tok.text] = float(value_tok.text)
            elif value_tok.kind is TokenKind.STRING:
                self._advance()
                result[key_tok.text] = value_tok.text
            elif value_tok.kind is TokenKind.IDENT:
                self._advance()
                if value_tok.text == "true":
                    result[key_tok.text] = True
                elif value_tok.text == "false":
                    result[key_tok.text] = False
                else:
                    result[key_tok.text] = value_tok.text
            else:
                raise TeirEmissionError(
                    f"unexpected metadata value at line {value_tok.line},"
                    f" col {value_tok.col}"
                )

        self._parse_comma_separated(TokenKind.LBRACE, TokenKind.RBRACE, _parse_entry)
        return result

    # ----- schedule -----

    def _parse_schedule(self, b: TeirBuilder) -> None:
        self._expect_kw("schedule")
        self._expect(TokenKind.LBRACE)
        # Phase 1: support the flat form only; tree form is a Phase-1.x
        # follow-up and parser accepts the same node syntax under indent.
        roots: list[str] | None = None
        # Collect all declarations first, then add iter / invoke in
        # topological order (children before parents).
        iter_decls: list[
            tuple[str, str, str, list[str], Guard | None, dict[str, object]]
        ] = []
        invoke_decls: list[tuple[str, str, Guard | None, dict[str, object]]] = []
        while self._peek.kind is not TokenKind.RBRACE:
            if self._peek.kind is TokenKind.EOF:
                raise TeirEmissionError("unexpected end of input inside schedule block")
            if self._accept_kw("roots"):
                if roots is not None:
                    tok = self._peek
                    raise TeirEmissionError(
                        f"duplicate 'roots' declaration at line {tok.line}, col {tok.col}"
                    )
                roots = self._parse_id_list()
                continue
            if self._accept_kw("iter"):
                iter_decls.append(self._parse_iter_decl())
                continue
            if self._accept_kw("invoke"):
                invoke_decls.append(self._parse_invoke_decl())
                continue
            tok = self._peek
            raise TeirEmissionError(
                f"expected 'roots', 'iter', or 'invoke' at line {tok.line}, col {tok.col};"
                f" got {tok.kind.name} {tok.text!r}"
            )
        self._expect(TokenKind.RBRACE)

        # The builder requires children to exist before their parents. We add
        # invocations first, then iterations in dependency order.
        node_kinds: dict[str, str] = {}
        for nid, *_ in iter_decls:
            node_kinds[nid] = "iter"
        for nid, *_ in invoke_decls:
            node_kinds[nid] = "invoke"

        for nid, prim, inv_guard, metadata in invoke_decls:
            b.add_invocation(nid, primitive=prim, guard=inv_guard, metadata=metadata)

        added: set[str] = {nid for nid, *_ in invoke_decls}
        remaining = list(iter_decls)
        while remaining:
            progress = False
            still: list[
                tuple[str, str, str, list[str], Guard | None, dict[str, object]]
            ] = []
            for decl in remaining:
                nid, axis, policy, children, it_guard, metadata = decl
                if all(c in added for c in children):
                    b.add_iteration(
                        nid,
                        axis=axis,
                        policy=policy,
                        children=children,
                        guard=it_guard,
                        metadata=metadata,
                    )
                    added.add(nid)
                    progress = True
                else:
                    still.append(decl)
            if not progress:
                missing = [
                    f"{nid} (missing child {c!r})"
                    for nid, _, _, kids, _, _ in still
                    for c in kids
                    if c not in added
                ]
                raise TeirEmissionError(
                    "cannot resolve iteration node dependencies; possible cycle "
                    "or dangling child references: " + ", ".join(sorted(missing))
                )
            remaining = still

        if roots is None:
            raise TeirEmissionError("schedule is missing 'roots' declaration")
        b.set_roots(roots)

    def _parse_iter_decl(
        self,
    ) -> tuple[str, str, str, list[str], Guard | None, dict[str, object]]:
        nid = self._expect(TokenKind.SIGIL_AT).text
        self._expect_kw("axis")
        axis = self._expect(TokenKind.SIGIL_AT).text
        self._expect_kw("policy")
        policy_tok = self._expect(TokenKind.IDENT)
        if policy_tok.text not in ("sequential", "parallel"):
            raise TeirEmissionError(
                f"iteration policy must be 'sequential' or 'parallel'; got {policy_tok.text!r}"
                f" at line {policy_tok.line}, col {policy_tok.col}"
            )
        self._expect_kw("children")
        children = self._parse_id_list()
        guard, metadata = self._parse_node_suffix()
        return nid, axis, policy_tok.text, children, guard, metadata

    def _parse_invoke_decl(self) -> tuple[str, str, Guard | None, dict[str, object]]:
        nid = self._expect(TokenKind.SIGIL_AT).text
        self._expect_kw("primitive")
        prim = self._expect(TokenKind.SIGIL_AT).text
        guard, metadata = self._parse_node_suffix()
        return nid, prim, guard, metadata

    def _parse_node_suffix(
        self,
    ) -> tuple[Guard | None, dict[str, object]]:
        """Parse the optional ``guard`` and ``metadata`` clauses on a node.

        The two clauses may appear in either order so the parser is symmetric
        with the printer's emission policy.
        """

        guard: Guard | None = None
        metadata: dict[str, object] | None = None
        while True:
            if self._accept_kw("guard"):
                if guard is not None:
                    tok = self._peek
                    raise TeirEmissionError(
                        f"duplicate 'guard' clause at line {tok.line}, col {tok.col}"
                    )
                guard = self._parse_guard()
            elif self._accept_kw("metadata"):
                if metadata is not None:
                    tok = self._peek
                    raise TeirEmissionError(
                        f"duplicate 'metadata' clause at line {tok.line}, col {tok.col}"
                    )
                metadata = self._parse_metadata_map()
            else:
                break
        return guard, metadata if metadata is not None else {}

    def _parse_id_list(self) -> list[str]:
        items: list[str] = []

        def _parse_id() -> None:
            items.append(self._expect(TokenKind.SIGIL_AT).text)

        self._parse_comma_separated(TokenKind.LBRACK, TokenKind.RBRACK, _parse_id)
        return items

    def _parse_guard(self) -> Guard:
        # 'first(@a)' or 'last(@b)', optionally chained with 'and'.
        terms: list[First | Last] = []
        while True:
            kind_tok = self._expect(TokenKind.IDENT)
            if kind_tok.text not in ("first", "last"):
                raise TeirEmissionError(
                    f"guard term must be 'first' or 'last'; got {kind_tok.text!r}"
                    f" at line {kind_tok.line}, col {kind_tok.col}"
                )
            self._expect(TokenKind.LPAREN)
            axis_tok = self._expect(TokenKind.SIGIL_AT)
            self._expect(TokenKind.RPAREN)
            term_cls = First if kind_tok.text == "first" else Last
            terms.append(term_cls(axis_tok.text))
            if not self._accept_kw("and"):
                break
        return guard(*terms)
