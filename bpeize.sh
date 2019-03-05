#!/usr/bin/env bash
declare -a lang_pairs=("de-en" "en-fr" "fr-en")

for pair in ${lang_pairs[@]}; do
  echo ${pair}
  mkdir -p ${pair}/bpe
  langp=$(echo ${pair} | tr "-" " ")
  for lang in ${langp}; do
    echo ${lang}
    subword-nmt learn-bpe -s 16000 < ${pair}/truecased/train.${pair}.${lang} > ${pair}/bpe/${lang}.codes
    subword-nmt apply-bpe -c ${pair}/bpe/${lang}.codes < ${pair}/truecased/train.${pair}.${lang} > ${pair}/bpe/train.${pair}.${lang}
    subword-nmt apply-bpe -c ${pair}/bpe/${lang}.codes < ${pair}/truecased/dev.${lang} > ${pair}/bpe/dev.${lang}
    subword-nmt apply-bpe -c ${pair}/bpe/${lang}.codes < ${pair}/truecased/test.${lang} > ${pair}/bpe/test.${lang}
  done
done
