#!/usr/bin/env bash
python3 parse.py

if [[ ! -d ./de-en/tokenized ]]; then
  mkdir -p ./de-en/tokenized;
fi
if [[ ! -d ./en-fr/tokenized ]]; then
  mkdir -p ./en-fr/tokenized;
fi
if [[ ! -d ./fr-en/tokenized ]]; then
  mkdir -p ./fr-en/tokenized;
fi

if [[ ! -d ./de-en/truecased ]]; then
  mkdir -p ./de-en/truecased;
fi
if [[ ! -d ./fr-en/truecased ]]; then
  mkdir -p ./fr-en/truecased;
fi
if [[ ! -d ./en-fr/truecased ]]; then
  mkdir -p ./en-fr/truecased;
fi

pair_lgs="de en fr"


for lg1 in ${pair_lgs}; do
    for lg2 in ${pair_lgs}; do
        #tokenizing
        cat "$lg1-$lg2/parsed/IWSLT16.TED.tst2013.$lg1-$lg2.$lg1.xml" "$lg1-$lg2/parsed/IWSLT16.TED.tst2014.$lg1-$lg2.$lg1.xml" > "$lg1-$lg2/parsed/dev.$lg1" || continue
        sacremoses tokenize -j 4 < "$lg1-$lg2/parsed/dev.$lg1" > "$lg1-$lg2/tokenized/dev.$lg1"
        cat "$lg1-$lg2/parsed/IWSLT16.TED.tst2013.$lg1-$lg2.$lg2.xml" "$lg1-$lg2/parsed/IWSLT16.TED.tst2014.$lg1-$lg2.$lg2.xml" > "$lg1-$lg2/parsed/dev.$lg2"
        sacremoses tokenize -j 4 < "$lg1-$lg2/parsed/dev.$lg2" > "$lg1-$lg2/tokenized/dev.$lg2"
        
        cat "$lg1-$lg2/parsed/IWSLT16.TED.tst2015.$lg1-$lg2.$lg1.xml" "$lg1-$lg2/parsed/IWSLT16.TED.tst2016.$lg1-$lg2.$lg1.xml" > "$lg1-$lg2/parsed/test.$lg1"
        sacremoses tokenize -j 4 < "$lg1-$lg2/parsed/test.$lg1" > "$lg1-$lg2/tokenized/test.$lg1"
        cat "$lg1-$lg2/parsed/IWSLT16.TED.tst2015.$lg1-$lg2.$lg2.xml" "$lg1-$lg2/parsed/IWSLT16.TED.tst2016.$lg1-$lg2.$lg2.xml" > "$lg1-$lg2/parsed/test.$lg2"
        sacremoses tokenize -j 4 < "$lg1-$lg2/parsed/test.$lg2" > "$lg1-$lg2/tokenized/test.$lg2"
        
        sacremoses tokenize -j 4 < "$lg1-$lg2/parsed/train.tags.$lg1-$lg2.$lg1" > "$lg1-$lg2/tokenized/train.tags.$lg1-$lg2.$lg1"
        sacremoses tokenize -j 4 < "$lg1-$lg2/parsed/train.tags.$lg1-$lg2.$lg2" > "$lg1-$lg2/tokenized/train.tags.$lg1-$lg2.$lg2"
        rm -rf "./$lg1-$lg2/parsed"
        
        #truecasing
        sacremoses train-truecase -m "./$lg1-$lg2/truecased/model.$lg1" -j 4 < "./$lg1-$lg2/tokenized/train.tags.$lg1-$lg2.$lg1"
        sacremoses train-truecase -m "./$lg1-$lg2/truecased/model.$lg2" -j 4 < "./$lg1-$lg2/tokenized/train.tags.$lg1-$lg2.$lg2"
        
        sacremoses truecase -m "./$lg1-$lg2/truecased/model.$lg1" -j 4 < "./$lg1-$lg2/tokenized/train.tags.$lg1-$lg2.$lg1" > "./$lg1-$lg2/truecased/train.$lg1-$lg2.$lg1"
        sacremoses truecase -m "./$lg1-$lg2/truecased/model.$lg2" -j 4 < "./$lg1-$lg2/tokenized/train.tags.$lg1-$lg2.$lg2" > "./$lg1-$lg2/truecased/train.$lg1-$lg2.$lg2"
        
        sacremoses truecase -m "./$lg1-$lg2/truecased/model.$lg1" -j 4 < "./$lg1-$lg2/tokenized/dev.$lg1" > "./$lg1-$lg2/truecased/dev.$lg1"
        sacremoses truecase -m "./$lg1-$lg2/truecased/model.$lg2" -j 4 < "./$lg1-$lg2/tokenized/dev.$lg2" > "./$lg1-$lg2/truecased/dev.$lg2"
        
        sacremoses truecase -m "./$lg1-$lg2/truecased/model.$lg1" -j 4 < "./$lg1-$lg2/tokenized/test.$lg1" > "./$lg1-$lg2/truecased/test.$lg1"
        sacremoses truecase -m "./$lg1-$lg2/truecased/model.$lg2" -j 4 < "./$lg1-$lg2/tokenized/test.$lg2" > "./$lg1-$lg2/truecased/test.$lg2"
        rm -rf "./$lg1-$lg2/tokenized"
    done
done
