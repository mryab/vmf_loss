DEEN_URL="https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz"
FREN_URL="https://wit3.fbk.eu/archive/2016-01//texts/fr/en/fr-en.tgz"
ENFR_URL="https://wit3.fbk.eu/archive/2016-01//texts/en/fr/en-fr.tgz"
wget "$DEEN_URL"
wget "$FREN_URL"
wget "$ENFR_URL"
tar zxvf de-en.tgz
tar zxvf fr-en.tgz
tar zxvf en-fr.tgz
