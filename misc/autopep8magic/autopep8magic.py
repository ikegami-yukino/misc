import os
import sys
import tempfile
from IPython.core.magic import register_cell_magic
from autopep8 import main


@register_cell_magic
def autopep8(line, cell):
    try:
        (_, tmp_path) = tempfile.mkstemp()
        with open(tmp_path, 'w') as fd:
            fd.write(cell)
        old_sys_argv = sys.argv.copy()
        sys.argv = ['']
        if line:
            sys.argv += line.split()
        sys.argv.append(tmp_path)
        main()
    finally:
        sys.argv = old_sys_argv.copy()
        os.remove(tmp_path)


def load_ipython_extension(ipython):
    ipython.register_magic_function(autopep8, 'cell')
