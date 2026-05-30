"""
tools/code_executor.py — Python code execution sandbox
=======================================================
Executes Python code and returns stdout output + any exceptions.

⚠️  SECURITY WARNING:
    This runs real Python code on your machine.
    Safe for local / development use only.
    For production: use a proper sandbox (Docker, e2b.dev, etc.)

The executor gives the code access to:
    - All stdlib modules (via __import__)
    - print, range, len, common builtins
    - Captures stdout automatically
"""

import sys
import io
import traceback
from langchain_core.tools import tool


@tool
def execute_python(code: str) -> str:
    """
    Execute Python code and return the printed output or any error.

    The code runs in an isolated namespace but with access to Python's
    standard library. Captured output (print statements) is returned
    as a string. If the code raises an exception, the traceback is
    returned instead.

    Args:
        code: Valid Python code to execute.

    Returns:
        stdout output from the code, or a traceback string on error.

    Examples:
        execute_python("print(2 ** 10)")              -> "1024"
        execute_python("import math; print(math.pi)") -> "3.141592653589793"
    """
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf

    try:
        # Full builtins available — needed for import, type(), etc.
        exec(code, {"__builtins__": __builtins__})
        output = stdout_buf.getvalue()
        return output if output else "[Code executed successfully — no output]"

    except SystemExit:
        return "[Code called sys.exit() — execution stopped]"

    except Exception:
        return f"[Runtime error]\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout
