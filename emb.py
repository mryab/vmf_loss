import argparse

from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument('--corpus-path', required=True)
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--num-workers', default=20, type=int)
parser.add_argument('--model-type', choices=['w2v', 'fasttext'], required=True)
args = parser.parse_args()

if args.model_type == 'w2v':
    model_corp_file = Word2Vec(corpus_file=args.corpus_path, iter=5, size=300, workers=args.num_workers)
else:
    model_corp_file = FastText(corpus_file=args.corpus_path, iter=5, size=300, workers=args.num_workers)
model_corp_file.wv.save_word2vec_format(args.model_type + '.' + args.lang)
