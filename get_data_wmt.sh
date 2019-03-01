if [ ! -d ./wmt-fr ]; then
  mkdir -p ./wmt-en;
fi

if [ ! -d ./wmt-fr ]; then
  mkdir -p ./wmt-fr;
fi


lgs="en fr"
years="2007 2008 2009 2010 2011 2012 2013"

for lg in $lgs; do
    for year in $years; do
        URL="http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.$year.$lg.shuffled.gz"
        wget "$URL"
        gzip -d "news.$year.$lg.shuffled.gz"
        mv "news.$year.$lg.shuffled" "./wmt-$lg"
    done
done


for lg in $lgs; do
    URL="http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.$lg.shuffled.v2.gz"
    wget "$URL"
    gzip -d "news.2014.$lg.shuffled.v2.gz"
    mv "news.2014.$lg.shuffled.v2" "./wmt-$lg"
    
    URL="http://www.statmt.org/wmt14/training-monolingual-europarl-v7/europarl-v7.$lg.gz"
    wget "$URL"
    gzip -d "europarl-v7.$lg.gz"
    mv "europarl-v7.$lg" "./wmt-$lg"
    
    URL="http://www.statmt.org/wmt15/news-discuss-v1.$lg.txt.gz"
    wget "$URL"
    gzip -d "news-discuss-v1.$lg.txt.gz"
    mv "news-discuss-v1.$lg.txt" "./wmt-$lg"
done

EN_URL="http://data.statmt.org/wmt16/translation-task/news-commentary-v11.en.gz"
wget "$EN_URL"
gzip -d "news-commentary-v11.en.gz"
mv "news-commentary-v11.en" ./wmt-en

EN_URL="http://data.statmt.org/wmt16/translation-task/news.2015.en.shuffled.gz"
wget "$EN_URL"
gzip -d "news.2015.en.shuffled.gz"
mv "news.2015.en.shuffled" ./wmt-en

FR_URL="http://www.statmt.org/wmt15/training-monolingual-nc-v10/news-commentary-v10.fr.gz"
wget "$EN_URL"
gzip -d "news-commentary-v10.fr.gz"
mv "news-commentary-v10.fr" ./wmt-fr