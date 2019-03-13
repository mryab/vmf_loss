#!/bin/bash

lgs="en fr"


for lg in ${lgs}; do
    if [[ ! -d "./wmt-$lg/tok" ]]; then
      mkdir -p "./wmt-$lg/tok";
    fi
    dir="./wmt-$lg/*"
    for file in ${dir}; do
        sacremoses tokenize -j 20 -l ${lg} < "$file" > "./wmt-$lg/tok/$(basename "$file")"
    done
    
    if [[ ! -d "./wmt-$lg/truecased" ]]; then
      mkdir -p "./wmt-$lg/truecased";
    fi
    dir="./wmt-$lg/tok/*"
    for file in ${dir}; do
        sacremoses truecase -m "./en-fr/truecased/model.$lg" -j 20 < "$file" > "./wmt-$lg/truecased/$(basename "$file")"
    done
    
    rm -rf "./wmt-$lg/tok"
    cat ./wmt-${lg}/truecased/* > "./corpus.$lg"
    rm -rf "./wmt-$lg/truecased"
done