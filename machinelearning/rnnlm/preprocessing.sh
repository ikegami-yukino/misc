#!/bin/bash -eu

MECAB_ARGS=""

while getopts i:o:m: OPT
do
  case $OPT in
    "i" ) INPUT_FILE="$OPTARG" ;;
    "o" ) OUTPUT_FILE="$OPTARG" ;;
    "m" ) MECAB_ARGS="$OPTARG" ;;
      * ) echo "Usage: preprocessing.sh [-i INPUT_FILE] [-o OUTPUT_FILE] [-m MECAB_ARGS]" 1>&2
          exit 1 ;;
  esac
done

tr -s "\n" < ${INPUT_FILE} | mecab ${MECAB_ARGS} -F "%m\t" --eos-format="\n" | tr "\t\n" "\n" > ${OUTPUT_FILE}
