# RNNLM for Japanese

## Plain text to tokenized tsv
```
$ ./preprocessing -i train.txt -o train.tsv
$ ./preprocessing -i dev.txt -o dev.tsv
```

- `-i` option indicates inout file (plain text) [required]
- `-o` option indicates output file (csv format) [required]
- `-m` option indicates MeCab arguments [optional]

## Train
```
$ ./train.py
```

See more detail by `--help` option.

## Predict
```
$ ./predict.py
```

See more detail by `--help` option.
