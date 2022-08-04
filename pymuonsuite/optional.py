"""
optional.py

Contains imports for libraries considered optional, along with decorators to
include checks for them.
"""


from functools import wraps

try:
    from euphonic import QpointPhononModes as _euphonic_qpm
except ImportError:
    _euphonic_qpm = None


"""
The decorator checks if the module is available, if not print
an error message, if yes pass it as a variable to the function itself.
They all take the name one desires the library to have within the function
as an argument. The function needs to have a named variable of the same name
in its interface.
"""


def requireEuphonicQPM(import_name="euphonic_qpm"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            if _euphonic_qpm is None:
                raise RuntimeError(
                    """
                Can't use castep phonon interface due to Euphonic not being
                installed. Please install Euphonic using your preferred package manager:

                pip install euphonic
                OR
                conda install euphonic

                and try again. Help can be found in the Euphonic install documentation:
                https://euphonic.readthedocs.io/en/latest/installation.html"""
                )
            else:
                kwargs[import_name] = _euphonic_qpm
                return func(*args, **kwargs)

        return wrapper

    return decorator
