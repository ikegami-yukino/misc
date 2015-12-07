#/usr/bin/env bash

Z3_DL_PATH=`curl https://github.com/Z3Prover/z3/releases | grep -m 1 -o -e '/Z3.*osx.*.zip'`
filename=`basename ${Z3_DL_PATH}`
echo ${Z3_DL_PATH}
echo ${filename}
curl -L -o /tmp/${filename} https://github.com${Z3_DL_PATH}
unzip /tmp/${filename}
cd z3-*
cp bin/libz3.a /usr/local/lib/
cp bin/libz3.dylib /usr/local/lib/
cp bin/z3 /usr/local/bin/
cp include/* /usr/local/include/

PY_SITEPKG_DIR=`python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"`
mkdir ${PY_SITEPKG_DIR}/z3
grep -l 'from z3' bin/*.py | xargs sed -i.bak -e 's/from z3/from \.z3/g'
grep -l 'import ctypes, z3core' bin/*.py | xargs sed -i.bak -e 's/import ctypes, z3core/import ctypes/g'
sed -e '10i\
  from . import z3core
' /work/venv/atango/lib/python3.5/site-packages/z3/z3types.py
cp bin/*.py ${PY_SITEPKG_DIR}/z3
cp bin/z3.py ${PY_SITEPKG_DIR}/z3/__init__.py
