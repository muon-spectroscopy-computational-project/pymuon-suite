import os
from ase import io


class ReadWrite(object):
    def __init__(self, params={}, script=None, calc=None):
        '''
        |   params (dict)           parameters for writing input files
        |   script (str):           Path to a file containing a submission
        |                           script to copy to the input folder. The
        |                           script can contain the argument
        |                           {seedname} in curly braces, and it will
        |                           be appropriately replaced.
        |   calc (ase.Calculator):  Calculator to attach to Atoms. If
        |                           present, the pre-existent one will
        |                           be ignored.
        '''
        self.__calc = calc
        self.script = script
        self.params = params

    def read(self, folder, sname=None):
        try:
            if sname is not None:
                sfile = os.path.join(folder, sname + '.xyz')
            else:
                sfile = glob.glob(os.path.join(folder, '*.xyz'))[0]
                sname = seedname(sfile)
                print(sfile)
            atoms = io.read(sfile)
            atoms.info['name'] = sname
            return atoms

        except IndexError:
            raise IOError("ERROR: No .xyz files found in {}."
                          .format(os.path.abspath(folder)))
        except OSError as e:
            raise IOError("ERROR: {}".format(e))
        except Exception as e:
            raise IOError("ERROR: Could not read {file}"
                          .format(file=sname + '.xyz'))

    def write(self, a, folder, sname=None, calc_type=None):
        if sname is None:
            sname = os.path.split(folder)[-1]
        if self.__calc is not None:
            a.calc = self.__calc
        fname = os.path.join(folder, sname + ".xyz")
        io.write(fname, a)
