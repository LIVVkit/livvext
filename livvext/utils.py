# coding=utf-8

from __future__ import absolute_import, print_function, unicode_literals

import operator as op

import pybtex.database
import pybtex.io
import six
from pybtex.backends.html import Backend as BaseBackend
from pybtex.style.formatting.plain import Style as PlainStyle


class HTMLBackend(BaseBackend):
    def __init__(self, *args, **kwargs):
        super(HTMLBackend, self).__init__(*args, **kwargs)
        self._html = ""

    def output(self, html):
        self._html += html

    def format_protected(self, text):
        if text[:4] == "http":
            return self.format_href(text, text)
        else:
            return r'<span class="bibtex-protected">{}</span>'.format(text)

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


# FIXME: For python 3.7+ only...
# from functools import singledispatch
# from collections.abc import Iterable
#
# # noinspection PyUnusedLocal
# @singledispatch
# def bib2html(bib, style=None, backend=None):
#     raise NotImplementedError('I do not now how to convert a {} type to a bibliography'.format(type(bib)))
def bib2html(bib, style=None, backend=None):
    if isinstance(bib, six.string_types):
        return _bib2html_string(bib, style=style, backend=backend)
    if isinstance(bib, (list, set, tuple)):
        return _bib2html_list(bib, style=style, backend=backend)
    if isinstance(bib, pybtex.database.BibliographyData):
        return _bib2html_bibdata(bib, style=style, backend=backend)
    else:
        raise NotImplementedError(
            "I do not now how to convert a {} type to a bibliography".format(type(bib))
        )


# FIXME: For python 3.7+ only...
# @bib2html.register
# def _bib2html_string(bib: str, style=None, backend=None):
def _bib2html_string(bib, style=None, backend=None):
    if style is None:
        style = PlainStyle()
    if backend is None:
        backend = HTMLBackend()

    formatted_bib = style.format_bibliography(pybtex.database.parse_file(bib))

    return backend._repr_html(formatted_bib)


# FIXME: For python 3.7+ only...
# @bib2html.register
# def _bib2html_list(bib: Iterable, style=None, backend=None):
def _bib2html_list(bib, style=None, backend=None):
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


# FIXME: For python 3.7+ only...
# @bib2html.register
# def _bib2html_bibdata(bib: pybtex.database.BibliographyData, style=None, backend=None):
def _bib2html_bibdata(bib, style=None, backend=None):
    if style is None:
        style = PlainStyle()
    if backend is None:
        backend = HTMLBackend()

    formatted_bib = style.format_bibliography(bib)

    return backend._repr_html(formatted_bib)


def eval_expr(expr):
    """
    Evaluate a mathematical expression stored as a string in a JSON file.

    Parameters
    ----------
    expr : str
        Mathematical expression, stored as a single string: e.g.
        "1 + 1 * 3", "24 * 3600 * 365", etc.
        Will be checked to contain _only_ numbers and mathematical symbols:
        [0-9], +, -, *, /, ^, (, and )
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
        else:
            _expr_out = eval(_code, {"__builtins__": {}})
    elif isinstance(expr, (int, float)):
        _expr_out = expr
    return _expr_out


def extract_ds(expr, dset, name=False):
    """
    Recursively extract a formula from a list of operands and dataset fields.

    Parameters
    ----------
    expr : list
        List of expressions where first element is operator, subsequent
        two elements are operands, either numeric values, fields within `dset`,
        or an expression of this kind.
        (e.g. ["^", ["+", ["*", "U", "U"], ["*", "V", "V"]], "0.5"] for the wind velocity)
    dset : xarray.Dataset
        Dataset which contains the fields described in `expr`
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
        operator, *operands = expr
    else:
        operator, operands = expr

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
        two elements are operands, either numeric values, fields within `dset`,
        or an expression of this kind.
        (e.g. ["^", ["+", ["*", "U", "U"], ["*", "V", "V"]], "0.5"] for the wind velocity)
    dset : xarray.Dataset
        Dataset which contains the fields described in `expr`

    Returns
    -------
    output : xarray.DataArray
        Evaluated expression

    """
    return extract_ds(expr, {}, name=True)


def will_it_float(test_str: str) -> bool:
    try:
        _will = float(test_str)
    except ValueError:
        _will = False
    return _will
