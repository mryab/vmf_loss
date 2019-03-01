#!/usr/bin/env bash
DEEN_URL="https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz"
FREN_URL="https://wit3.fbk.eu/archive/2016-01//texts/fr/en/fr-en.tgz"
ENFR_URL="https://wit3.fbk.eu/archive/2016-01//texts/en/fr/en-fr.tgz"
wget "$DEEN_URL"
wget "$FREN_URL"
wget "$ENFR_URL"
tar zxvf de-en.tgz
tar zxvf fr-en.tgz
tar zxvf en-fr.tgz
rm de-en.tgz
rm fr-en.tgz
rm en-fr.tgz


DEEN_URL="https://wit3.fbk.eu/archive/2016-01-test//texts/de/en/de-en.tgz"
FREN_URL="https://wit3.fbk.eu/archive/2016-01-test//texts/fr/en/fr-en.tgz"
ENFR_URL="https://wit3.fbk.eu/archive/2016-01-test//texts/en/fr/en-fr.tgz"
wget "$DEEN_URL"
wget "$FREN_URL"
wget "$ENFR_URL"
tar zxvf de-en.tgz
tar zxvf fr-en.tgz
tar zxvf en-fr.tgz
rm de-en.tgz

DEEN_URL="https://wit3.fbk.eu/archive/2016-01-test//texts/en/de/en-de.tgz"
wget "$DEEN_URL"
tar zxvf en-de.tgz -C ./de-en
tar zxvf fr-en.tgz -C ./en-fr
tar zxvf en-fr.tgz -C ./fr-en
rm en-de.tgz
rm fr-en.tgz
rm en-fr.tgz
mv de-en/en-de/IWSLT16.TED.tst2015.en-de.en.xml de-en/IWSLT16.TED.tst2015.de-en.en.xml
mv de-en/en-de/IWSLT16.TED.tst2016.en-de.en.xml de-en/IWSLT16.TED.tst2016.de-en.en.xml

mv fr-en/en-fr/IWSLT16.TED.tst2015.en-fr.en.xml fr-en/IWSLT16.TED.tst2015.fr-en.en.xml
mv fr-en/en-fr/IWSLT16.TED.tst2016.en-fr.en.xml fr-en/IWSLT16.TED.tst2016.fr-en.en.xml

mv en-fr/fr-en/IWSLT16.TED.tst2015.fr-en.fr.xml en-fr/IWSLT16.TED.tst2015.en-fr.fr.xml
mv en-fr/fr-en/IWSLT16.TED.tst2016.fr-en.fr.xml en-fr/IWSLT16.TED.tst2016.en-fr.fr.xml

rm -rf de-en/en-de
rm -rf fr-en/en-fr
rm -rf en-fr/fr-en