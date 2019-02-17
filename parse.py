import argparse
import os
from os import listdir
from os.path import isfile


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./', type=str)

args = parser.parse_args()

lg_pairs = ["de-en/", "en-fr/", "fr-en/"]

for pair in lg_pairs:
    mypath = args.data_path + pair
    if not os.path.exists(mypath + 'parsed'):
        os.makedirs(mypath + 'parsed')
        
    onlyfiles = [f for f in listdir(mypath) if isfile(mypath + f)]
    for file in onlyfiles:
        with open(mypath + file, 'r') as rd:
            with open(mypath + 'parsed/' + file, 'w') as wr:
                for line in rd:
                    if line.startswith('<desc'):
                        wr.write(line[len('<description>') : -len('</description>\n')] + '\n')
                    if line.startswith('<ti'):
                        wr.write(line[len('<title>') : -len('</title>\n')] + '\n')
                    if line.startswith('<seg'):
                        wr.write(line[line.find('">') + 2 : -len('</seg>\n')] + '\n')
                    if line[0] != '<':
                        wr.write(line)
