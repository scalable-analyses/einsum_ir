"""Tokenizer for the textual TEIR format."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum

from etops.diag import TeirEmissionError

__all__ = ["Token", "TokenKind", "tokenize"]


class TokenKind(IntEnum):
    IDENT = 1
    INT = 2
    SIGIL_PERCENT = 3
    SIGIL_AT = 4
    LBRACE = 5
    RBRACE = 6
    LBRACK = 7
    RBRACK = 8
    LPAREN = 9
    RPAREN = 10
    COLON = 11
    COMMA = 12
    STRING = 13  # double-quoted, backslash-escaped string literal.
    FLOAT = 14  # IEEE-754 float literal (e.g. "1.5", "-2e-3").
    EOF = 15


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    line: int
    col: int


_PUNCT = {
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    "[": TokenKind.LBRACK,
    "]": TokenKind.RBRACK,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    ":": TokenKind.COLON,
    ",": TokenKind.COMMA,
}

_IDENT_HEAD = re.compile(r"[A-Za-z_]")
_IDENT_BODY = re.compile(r"[A-Za-z0-9_\-.]")
_INT_HEAD = re.compile(r"[+\-]?[0-9]")


def tokenize(text: str) -> list[Token]:
    """Return a list of tokens from ``text``. Comments are stripped.

    Raises:
        TeirEmissionError: On invalid characters or malformed numbers.
    """

    tokens: list[Token] = []
    line, col = 1, 1
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        # Whitespace
        if ch == "\n":
            line += 1
            col = 1
            i += 1
            continue
        if ch == "\r":
            # Treat CR as a newline; CRLF is handled by also bumping the
            # subsequent LF below.
            line += 1
            col = 1
            i += 1
            if i < n and text[i] == "\n":
                i += 1
            continue
        if ch.isspace():
            i += 1
            col += 1
            continue
        # Comment to end-of-line.
        if ch == "#":
            while i < n and text[i] != "\n":
                i += 1
                col += 1
            continue
        # Punctuation
        if ch in _PUNCT:
            tokens.append(Token(_PUNCT[ch], ch, line, col))
            i += 1
            col += 1
            continue
        # Sigils
        if ch == "%":
            i, col = _consume_sigil(text, i, line, col, tokens, TokenKind.SIGIL_PERCENT)
            continue
        if ch == "@":
            i, col = _consume_sigil(text, i, line, col, tokens, TokenKind.SIGIL_AT)
            continue
        # String literal: double-quoted with backslash escapes.
        if ch == '"':
            i, col, line = _consume_string(text, i, line, col, tokens)
            continue
        # Integer / float / dotted version literal.
        if _INT_HEAD.match(text, i):
            start = i
            start_col = col
            sign_only = False
            if text[i] in "+-":
                i += 1
                col += 1
                sign_only = True
            while i < n and text[i].isdigit():
                i += 1
                col += 1
                sign_only = False
            if start == i or sign_only:
                raise TeirEmissionError(
                    f"malformed numeric literal at line {line}, col {start_col}"
                )
            has_fraction = False
            if i < n and text[i] == "." and i + 1 < n and text[i + 1].isdigit():
                has_fraction = True
                i += 1
                col += 1
                while i < n and text[i].isdigit():
                    i += 1
                    col += 1
            has_exponent = False
            if i < n and text[i] in "eE":
                has_exponent = True
                i += 1
                col += 1
                if i < n and text[i] in "+-":
                    i += 1
                    col += 1
                exp_digits_start = i
                while i < n and text[i].isdigit():
                    i += 1
                    col += 1
                if i == exp_digits_start:
                    raise TeirEmissionError(
                        f"malformed exponent in numeric literal at line"
                        f" {line}, col {start_col}"
                    )
            if has_exponent or has_fraction:
                tokens.append(Token(TokenKind.FLOAT, text[start:i], line, start_col))
                continue
            tokens.append(Token(TokenKind.INT, text[start:i], line, start_col))
            continue
        # Identifier (or keyword)
        if _IDENT_HEAD.match(ch):
            start = i
            start_col = col
            i += 1
            col += 1
            while i < n and _IDENT_BODY.match(text[i]):
                i += 1
                col += 1
            tokens.append(Token(TokenKind.IDENT, text[start:i], line, start_col))
            continue
        raise TeirEmissionError(
            f"unexpected character {ch!r} at line {line}, col {col}"
        )
    tokens.append(Token(TokenKind.EOF, "", line, col))
    return tokens


def _consume_sigil(
    text: str,
    i: int,
    line: int,
    col: int,
    tokens: list[Token],
    kind: TokenKind,
) -> tuple[int, int]:
    sigil_col = col
    i += 1
    col += 1
    n = len(text)
    if i >= n or not _IDENT_HEAD.match(text[i]):
        raise TeirEmissionError(
            f"sigil '{text[i - 1]}' at line {line}, col {sigil_col} must be followed by an identifier"
        )
    start = i
    while i < n and _IDENT_BODY.match(text[i]):
        i += 1
        col += 1
    tokens.append(Token(kind, text[start:i], line, sigil_col))
    return i, col


_ESCAPE_MAP = {
    "\\": "\\",
    '"': '"',
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "0": "\0",
}


def _consume_string(
    text: str,
    i: int,
    line: int,
    col: int,
    tokens: list[Token],
) -> tuple[int, int, int]:
    """Consume a double-quoted string literal and append a STRING token.

    Returns the updated ``(i, col, line)`` cursor. Recognized escape
    sequences are ``\\\\``, ``\\"``, ``\\n``, ``\\t``, ``\\r``, ``\\0``;
    any other escape is rejected.
    """

    start_line = line
    start_col = col
    n = len(text)
    i += 1
    col += 1
    parts: list[str] = []
    while i < n and text[i] != '"':
        ch = text[i]
        if ch == "\\":
            i += 1
            col += 1
            if i >= n:
                raise TeirEmissionError(
                    f"unterminated escape in string at line {start_line},"
                    f" col {start_col}"
                )
            esc = text[i]
            mapped = _ESCAPE_MAP.get(esc)
            if mapped is None:
                raise TeirEmissionError(
                    f"unknown escape sequence '\\{esc}' in string at line {line},"
                    f" col {col}"
                )
            parts.append(mapped)
            i += 1
            col += 1
        else:
            parts.append(ch)
            if ch == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1
    if i >= n:
        raise TeirEmissionError(
            f"unterminated string literal starting at line {start_line},"
            f" col {start_col}"
        )
    i += 1  # consume the closing quote
    col += 1
    tokens.append(Token(TokenKind.STRING, "".join(parts), start_line, start_col))
    return i, col, line
