# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pymuonsuite.schemas import load_input_file, PhononHfccSchema

def phonon_hfcc(param_file):
    params = load_input_file(param_file, PhononHfccSchema)
    print(params)
    return
