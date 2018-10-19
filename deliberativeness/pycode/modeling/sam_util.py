from __future__ import absolute_import, print_function, division
from copy import copy, deepcopy
from sys import getsizeof
import sys
import traceback
import numpy as np

import theano
from theano.compat import izip
from six import reraise
from six.moves import StringIO
from theano.gof import utils
from theano.gof import graph
from theano.gof.type import Type

# from .utils import undef

__excepthook = sys.excepthook

def trace_apply_node(node):

    # Print node backtraces
    tr = getattr(node.outputs[0].tag, 'trace', [])
    if isinstance(tr, list) and len(tr) > 0:
        message = "\nBacktrace when the node is created(use Theano flag traceback.limit=N to make it longer):\n"

        # Print separate message for each element in the list of batcktraces
        sio = StringIO()
        for subtr in tr:
            traceback.print_list(subtr, sio)
        message += str(sio.getvalue())
    else:
        message = None

    return message

def sprint(variable, name):
    return theano.printing.Print('Variable {} shape: '.format(name),attrs=['shape'])(variable)