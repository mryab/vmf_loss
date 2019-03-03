import argparse

from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument('--tmp-corpus-path', default='./corpus.', type=str)
parser.add_argument('--lang', required=True)
parser.add_argument('--num-workers', default=10, type=int)
parser.add_argument('--iter', default=5, type=int)
parser.add_argument('--model-type', choices=['w2v', 'fasttext'], required=True)
args = parser.parse_args()

if args.model_type == 'w2v':
    model_corp_file = Word2Vec(corpus_file=args.tmp_corpus_path + args.lang, iter=args.iter, size=300, workers=args.num_workers, min_count=0)
else:
    model_corp_file = FastText(corpus_file=args.tmp_corpus_path + args.lang, iter=args.iter, size=300, workers=args.num_workers, min_count=0)
model_corp_file.wv.save_word2vec_format(args.model_type + '.' + args.lang)
