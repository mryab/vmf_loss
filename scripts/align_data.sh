#!/usr/bin/env bash

declare -a lang_pairs=("de-en" "en-fr" "fr-en")

for pair in ${lang_pairs[@]}; do
  echo ${pair}
  mkdir -p ${pair}/align
  src=$(echo ${pair}| cut -f1 -d'-')
  dst=$(echo ${pair}| cut -f2 -d'-')
  paste ${pair}/truecased/train.${pair}.${src} ${pair}/truecased/train.${pair}.${dst} | sed 's/\t/ ||| /' > ${pair}/align/train.parallel
  ~/fast_align/build/fast_align -i ${pair}/align/train.parallel -d -o -v > ${pair}/align/train.aligned
done
