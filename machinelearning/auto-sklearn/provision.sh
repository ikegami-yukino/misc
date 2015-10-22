#/usr/bin/env bash

apt-get update

apt-get -y install git python-dev python-pip

# for matplotlib
apt-get -y install libfreetype6-dev libpng-dev
ln -s /usr/include/freetype2/ft2build.h /usr/include/

# for numpy
pip install cython numpy==1.9.0
np_include_path=`find / -name arrayobject.h -print 2> /dev/null | xargs dirname | xargs -I {} dirname {}../`
export C_INCLUDE_PATH=${np_include_path}:${C_INCLUDE_PATH}

# for scipy
apt-get -y install gfortran libblas-dev  liblapack-dev
pip install scipy==0.14.0

pip install scikit-learn==0.15.2 pandas
pip install git+https://github.com/mfeurer/HPOlibConfigSpace
pip install git+https://git@bitbucket.org/mfeurer/paramsklearn.git --no-deps
pip install --editable git+https://bitbucket.org/mfeurer/pymetalearn/#egg=pyMetaLearn --no-deps
pip install git+https://github.com/automl/HPOlib

pip install git+https://github.com/automl/auto-sklearn

# for smac
apt-get install -y default-jre
export PATH=`dirname /usr/local/lib/python2.7/dist-packages/AutoSklearn*/autosklearn/binaries/smac*/smac*/smac`:${PATH}
