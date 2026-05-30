"""
tools/calculator.py — Safe math expression evaluator
=====================================================
Uses numexpr for fast, safe evaluation of mathematical expressions.
Falls back to a restricted eval() if numexpr is not installed.

Install: pip install numexpr  (strongly recommended)

Supported:
  Basic arithmetic  : 2 + 3 * 4, 10 / 3, 2 ** 8
  Math functions    : sqrt(x), log(x), sin(x), cos(x), abs(x), exp(x)
  Constants         : pi, e
  Comparisons       : 3 > 2, 100 == 100
"""

import math
from langchain_core.tools import tool

# Safe names available in fallback eval
_SAFE_NS = {
    "sqrt": math.sqrt, "log": math.log,   "log10": math.log10,
    "sin":  math.sin,  "cos": math.cos,   "tan":   math.tan,
    "exp":  math.exp,  "abs": abs,        "round": round,
    "pi":   math.pi,   "e":  math.e,
    "pow":  pow,       "min": min,        "max":   max,
    "floor": math.floor, "ceil": math.ceil,
}

try:
    import numexpr as _ne
    _HAS_NUMEXPR = True
except ImportError:
    _HAS_NUMEXPR = False


def _evaluate(expression: str) -> str:
    """Core evaluation logic, used by both the tool and direct calls."""
    expr = expression.strip()

    if _HAS_NUMEXPR:
        try:
            result = _ne.evaluate(expr)
            # numexpr returns numpy scalar — convert to plain Python
            value = result.item() if hasattr(result, "item") else float(result)
            # Show integer without decimal if it's a whole number
            return str(int(value)) if value == int(value) else str(value)
        except Exception:
            pass  # fall through to restricted eval

    # Restricted eval fallback
    try:
        result = eval(expr, {"__builtins__": {}}, _SAFE_NS)
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except (SyntaxError, NameError) as exc:
        return f"Error: invalid expression — {exc}"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result as a string.

    Args:
        expression: A math expression, e.g. "2 ** 10", "sqrt(144)", "(3+4)*8/2"

    Returns:
        The computed result as a string, or an error message.

    Examples:
        calculator("2 ** 10")           -> "1024"
        calculator("sqrt(144)")         -> "12"
        calculator("sin(pi / 2)")       -> "1.0"
        calculator("(100 + 200) * 1.5") -> "450"
    """
    return _evaluate(expression)
