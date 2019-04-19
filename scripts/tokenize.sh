#!/usr/bin/env bash

python3 parse.py

pair_lgs="de en fr"

declare -a lang_pairs=("de-en" "en-fr" "fr-en")

for pair in ${lang_pairs[@]}; do
    mkdir -p ${pair}/tokenized
    mkdir -p ${pair}/truecased
    src=$(echo ${pair}| cut -f1 -d'-')
    dst=$(echo ${pair}| cut -f2 -d'-')
    lang_list="$src $dst"
    for lang in ${lang_list}; do
        cat "$pair/parsed/IWSLT16.TED.tst2013.$pair.$lang.xml" "$pair/parsed/IWSLT16.TED.tst2014.$pair.$lang.xml" > "$pair/parsed/dev.$lang"
        sacremoses tokenize -j 8 -l ${lang} < "$pair/parsed/dev.$lang" > "$pair/tokenized/dev.$lang"

        cat "$pair/parsed/IWSLT16.TED.tst2015.$pair.$lang.xml" "$pair/parsed/IWSLT16.TED.tst2016.$pair.$lang.xml" > "$pair/parsed/test.$lang"
        sacremoses tokenize -j 8 -l ${lang} < "$pair/parsed/test.$lang" > "$pair/tokenized/test.$lang"

        sacremoses tokenize -j 8 -l ${lang} < "$pair/parsed/train.tags.$pair.$lang" > "$pair/tokenized/train.tags.$pair.$lang"

        sacremoses train-truecase -m "$pair/truecased/model.$lang" -j 8 < "$pair/tokenized/train.tags.$pair.$lang"
        sacremoses truecase -m "$pair/truecased/model.$lang" -j 8 < "$pair/tokenized/train.tags.$pair.$lang" > "$pair/truecased/train.$pair.$lang"
        sacremoses truecase -m "$pair/truecased/model.$lang" -j 8 < "$pair/tokenized/dev.$lang" > "$pair/truecased/dev.$lang"
        sacremoses truecase -m "$pair/truecased/model.$lang" -j 8 < "$pair/tokenized/test.$lang" > "$pair/truecased/test.$lang"
    done
    rm -rf "$pair/parsed"
    rm -rf "$pair/tokenized"
done
