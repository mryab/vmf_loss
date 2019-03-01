#!/bin/bash
if [ ! -d ./de-en/bpe ]; then
  mkdir -p ./de-en/bpe;
fi
if [ ! -d ./en-fr/bpe ]; then
  mkdir -p ./en-fr/bpe;
fi
if [ ! -d ./fr-en/bpe ]; then
  mkdir -p ./fr-en/bpe;
fi

declare -a lang_pairs=("de-en" "en-fr" "fr-en")

for pair in ${lang_pairs[@]}; do
  echo $pair
  langp=$(echo $pair | tr "-" " ")
  for lang in $langp; do
    echo $lang
    subword-nmt learn-bpe -s 50000 < $pair/truecased/train.$pair.$lang > $pair/bpe/$lang.codes
    subword-nmt apply-bpe -c $pair/bpe/$lang.codes < $pair/truecased/train.$pair.$lang > $pair/bpe/train.$pair.$lang
    subword-nmt apply-bpe -c $pair/bpe/$lang.codes < $pair/truecased/dev.$lang > $pair/bpe/dev.$lang
    subword-nmt apply-bpe -c $pair/bpe/$lang.codes < $pair/truecased/test.$lang > $pair/bpe/test.$lang
  done
done
