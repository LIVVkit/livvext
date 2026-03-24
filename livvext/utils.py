# coding=utf-8
"""Utilities for generating LIVVkit reports of LIVVext results."""

import operator as op
from collections.abc import Iterable
from functools import singledispatch

import pybtex.database
import pybtex.io
from pybtex.backends.html import Backend as BaseBackend
from pybtex.style.formatting.plain import Style as PlainStyle


class HTMLBackend(BaseBackend):
    """Extends ``pybtex.backends.html.Backend``"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._html = ""

    def output(self, html):
        """Append HTML to the _html attribute."""
        self._html += html

    def format_protected(self, text):
        if text[:4] == "http":
            return self.format_href(text, text)
        return f'<span class="bibtex-protected">{text}</span>'

    def write_prologue(self):
        self.output('<div class="bibliography"><dl>')

    def write_epilogue(self):
        self.output("</dl></div>")

    def _repr_html(self, formatted_bibliography):
        self.write_prologue()
        for entry in formatted_bibliography:
            self.write_entry(entry.key, entry.label, entry.text.render(self))
        self.write_epilogue()

        return self._html.replace("\n", " ").replace("\\url <a", "<a")


@singledispatch
def bib2html(bib, style=None, backend=None):
    """
    Convert a BibTeX bibliography to HTML.

    Parameters
    ----------
    bib : `str`, `Iterable`, ``pybtex.database.BibliographyData``
        Location of bibliograph(y, ies), or a ``pybtex.database.BibliographyData``
    style : _type_, optional
        Bibliography style to output, by default None, which uses
        ``pybtex.style.formatting.plain.Style``
    backend : , optional
        HTML backend to format HTML output, by default None, which uses
        ``pybtex.backends.html.Backend``

    Returns
    -------
    bib_html : str
        Bibliography in HTML format as a string

    Raises
    ------
    NotImplementedError
        If ``bib`` is not a `str`, `Iterable`, or ``pybtex.database.BibliographyData``,
        raise `NotImplementedError`

    """
    raise NotImplementedError(
        f"I do not now how to convert a {type(bib)} type to a bibliography"
    )


@bib2html.register
def _bib2html_string(bib: str, style=None, backend=None):
    if style is None:
        style = PlainStyle()
    if backend is None:
        backend = HTMLBackend()

    formatted_bib = style.format_bibliography(pybtex.database.parse_file(bib))

    return backend._repr_html(formatted_bib)


@bib2html.register
def _bib2html_list(bib: Iterable, style=None, backend=None):
    if style is None:
        style = PlainStyle()
    if backend is None:
        backend = HTMLBackend()

    bibliography = pybtex.database.BibliographyData()
    for bib_file in bib:
        temp_bib = pybtex.database.parse_file(bib_file)
        for key, entry in temp_bib.entries.items():
            try:
                bibliography.add_entry(key, entry)
            except pybtex.database.BibliographyDataError:
                continue

    formatted_bib = style.format_bibliography(bibliography)

    return backend._repr_html(formatted_bib)


@bib2html.register
def _bib2html_bibdata(bib: pybtex.database.BibliographyData, style=None, backend=None):
    if style is None:
        style = PlainStyle()
    if backend is None:
        backend = HTMLBackend()

    formatted_bib = style.format_bibliography(bib)

    return backend._repr_html(formatted_bib)


def eval_expr(expr):
    r"""
    Evaluate a mathematical expression stored as a string in a JSON file.

    Parameters
    ----------
    expr : str
        Mathematical expression, stored as a single string: e.g.
        "1 + 1 \* 3", "24 \* 3600 \* 365", etc.
        Will be checked to contain _only_ numbers and mathematical symbols:
        [0-9], +, -, \*, /, ^, (, and )
    Returns
    -------
    result : float
        Result of mathematical expression

    """
    if isinstance(expr, str):
        _code = compile(expr, "<string>", "eval")
        if _code.co_names:
            raise ValueError(
                f"EXPRESSION {expr} MUST ONLY CONTAIN MATHEMATICAL EXPRESSION"
            )
        _expr_out = eval(_code, {"__builtins__": {}})
    elif isinstance(expr, (int, float)):
        _expr_out = expr
    else:
        raise NotImplementedError(
            f"EXPRESSION OF TYPE ({type(expr)}) COULD NOT BE PARSED"
        )
    return _expr_out


def extract_ds(expr, dset, name=False):
    """
    Recursively extract a formula from a list of operands and dataset fields.

    Parameters
    ----------
    expr : list
        List of expressions where first element is operator, subsequent
        two elements are operands, either numeric values, fields within ``dset``,
        or an expression of this kind.
        (e.g. ["^", ["+", ["*", "U", "U"], ["*", "V", "V"]], "0.5"] for the wind velocity)
    dset : xarray.Dataset
        Dataset which contains the fields described in ``expr``
    name : bool, optional
        If true, output the string of the interpreterd expression rather than its result

    Returns
    -------
    output : xarray.DataArray
        Evaluated expression

    """
    if len(expr) >= 3:
        operator, *operands = expr
    else:
        operator, operands = expr

    operand_queue = []
    for _operand in operands:
        if isinstance(_operand, (list, tuple)):
            operand_queue.append(extract_ds(_operand, dset, name))

        elif isinstance(_operand, str) and not name:
            _number = will_it_float(_operand)
            if _number:
                operand_queue.append(_number)
            else:
                operand_queue.append(dset[_operand])
        else:
            operand_queue.append(_operand)

    _out = apply_operator(operand_queue, operator, name)

    return _out


def extract_vars(expr):
    """
    Extract a list of unique variable names from an expression.

    """
    if len(expr) >= 3:
        _, *operands = expr
    else:
        _, operands = expr

    operand_queue = []
    for _operand in operands:
        if isinstance(_operand, (list, tuple)):
            operand_queue.extend(extract_vars(_operand))

        elif isinstance(_operand, str):
            _number = will_it_float(_operand)
            if not _number:
                operand_queue.append(_operand)

    return list(set(operand_queue))


def apply_operator(operands, operator, name=False):
    r"""
    Recursively apply mathematical operator(s).

    Parameters
    ----------
    operands : list
        List of operands, either two field names, or a field name and another list
        of field names (e.g. [U, V], or [T, [U, V]], or [T, [U, [V, X]]])
    operator : str
        Operator to apply recursively to the pairs of operands (one of +, -, \*, /, or ^)
    name : bool
        If true, return a string of the applied operation, if false, return the value.

    Returns
    -------
    output : (type(``operands``), str)
        Returns the result of the mathematical expression, with the same type as
        ``operands[0]`` or a string representation of the expression

    """
    ops = {
        "+": op.add,
        "-": op.sub,
        "*": op.mul,
        "/": op.truediv,
        "^": op.pow,
    }
    if len(operands) >= 3:
        op0, *op1 = operands
    else:
        op0, op1 = operands

    if isinstance(op1, (list, tuple)):
        op1 = apply_operator(op1, operator, name)

    if not name:
        _out = ops[operator](op0, op1)
    else:
        _out = f"({op0} {operator} {op1})"
    return _out


def extract_name(expr):
    """
    Recursively extract a formula from a list of operands and dataset fields.

    Parameters
    ----------
    expr : list
        List of expressions where first element is operator, subsequent
        two elements are operands, either numeric values, fields within ``dset``,
        or an expression of this kind.
        (e.g. ["^", ["+", ["*", "U", "U"], ["*", "V", "V"]], "0.5"] for the wind velocity)

    Returns
    -------
    output : str
        Human redable text of evaluated expression

    """
    return extract_ds(expr, {}, name=True)


def will_it_float(test_str: str) -> bool:
    """
    Test a string to determine if it will convert to a float.

    Parameters
    ----------
    test_str : str
        String to be tested

    Returns
    -------
    _will : float, bool
        If the string converts to a float, return the float, if not return False

    """
    try:
        _will = float(test_str)
    except ValueError:
        _will = False
    return _will
