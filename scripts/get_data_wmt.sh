#!/usr/bin/env bash

mkdir -p wmt-en
mkdir -p wmt-fr

lgs="en fr"

for lg in ${lgs}; do
    for year in $(seq 2007 2013); do
        wget "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.$year.$lg.shuffled.gz"
        gzip -dc "news.$year.$lg.shuffled.gz" > "wmt-$lg/news.$year.$lg.shuffled"
    done
done


for lg in ${lgs}; do
    wget "http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.$lg.shuffled.v2.gz"
    gzip -dc "news.2014.$lg.shuffled.v2.gz" > "wmt-$lg/news.2014.$lg.shuffled.v2"

    wget "http://www.statmt.org/wmt14/training-monolingual-europarl-v7/europarl-v7.$lg.gz"
    gzip -dc "europarl-v7.$lg.gz" > "wmt-$lg/europarl-v7.$lg"

    wget "http://www.statmt.org/wmt15/news-discuss-v1.$lg.txt.gz"
    gzip -dc "news-discuss-v1.$lg.txt.gz" > "wmt-$lg/news-discuss-v1.$lg.txt"
done

wget "http://data.statmt.org/wmt16/translation-task/news-commentary-v11.en.gz"
gzip -dc "news-commentary-v11.en.gz" > "wmt-en/news-commentary-v11.en"

wget "http://data.statmt.org/wmt16/translation-task/news.2015.en.shuffled.gz"
gzip -dc "news.2015.en.shuffled.gz" > "wmt-en/news.2015.en.shuffled"

wget "http://www.statmt.org/wmt15/training-monolingual-nc-v10/news-commentary-v10.fr.gz"
gzip -dc "news-commentary-v10.fr.gz" > "wmt-fr/news-commentary-v10.fr"
