#!/usr/bin/env sh
set -eu

cp autopep8magic.py ${HOME}/.ipython/extensions

DEFALUT_CFG_FILE=${HOME}/.ipython/profile_default/ipython_config.py

if [ ! -e ${DEFALUT_CFG_FILE} ]
then
  echo 'c = get_config()' > ${DEFALUT_CFG_FILE}
fi

if [ ! `grep autopep8magic ${DEFALUT_CFG_FILE}` ]
then
  echo 'c.InteractiveShellApp.extensions.append("autopep8magic")' >> ${DEFALUT_CFG_FILE}
fi
