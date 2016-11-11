MECABDIR = /tmp/mecab
MOZCDIR = /tmp/mozc
OUTPUTDIR = /tmp/converted

all:
	@if [ ! -d $(MECABDIR) ]; \
      then git clone --depth 1 https://github.com/taku910/mecab.git $(MECABDIR); \
    fi
	@if [ ! -d $(MOZCDIR) ]; \
      then git clone --depth 1 https://github.com/google/mozc.git $(MOZCDIR); \
    fi
	@if [ ! -d $(OUTPUTDIR) ]; \
      then mkdir $(OUTPUTDIR); \
    fi
	pip install jaconv
	python mozc2ipadic.py --mozc $(MOZCDIR) --ipadic $(MECABDIR)/mecab-ipadic --out $(OUTPUTDIR)
