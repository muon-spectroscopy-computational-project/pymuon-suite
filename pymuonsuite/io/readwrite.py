"""Base class for reading and writing input files for other software

Author: Laura Murgatroyd
"""


class ReadWrite(object):
    def __init__(self, params={}, script=None, calc=None):
        """
        |   params (dict)           parameters for writing input files
        |   script (str):           Path to a file containing a submission
        |                           script to copy to the input folder. The
        |                           script can contain the argument
        |                           {seedname} in curly braces, and it will
        |                           be appropriately replaced.
        |   calc (ase.Calculator):  Calculator to attach to Atoms. If
        |                           present, the pre-existent one will
        |                           be ignored.
        """
        self._calc = calc
        self.script = script
        self.params = params

    def read(self, folder, sname=None):
        raise (
            NotImplementedError(
                "read method is not implemented for" " ReadWrite baseclass."
            )
        )

    def write(self, a, folder, sname=None, calc_type=None):
        raise (
            NotImplementedError(
                "write method is not implemented for" " ReadWrite baseclass."
            )
        )

    def set_params(self, params):
        """
        |   params (dict)           Contains muon symbol, parameter file,
        |                           k_points_grid.
        """
        if not (isinstance(params, dict)):
            raise ValueError("params should be a dict, not ", type(params))
            return
        else:
            self.params = params

    def set_script(self, script):
        """
        |   script (str):           Path to a file containing a submission
        |                           script to copy to the input folder. The
        |                           script can contain the argument
        |                           {seedname} in curly braces, and it will
        |                           be appropriately replaced.
        """
        self.script = script
