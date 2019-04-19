import argparse

import gensim.models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp-corpus-path', default='./corpus.', type=str)
    parser.add_argument('--lang', required=True)
    parser.add_argument('--num-workers', default=10, type=int)
    parser.add_argument('--iter', default=5, type=int)
    parser.add_argument('--model-type', choices=['w2v', 'fasttext'], required=True)
    args = parser.parse_args()

    model_map = {
            'w2v': gensim.models.word2vec.Word2Vec,
            'fasttext': gensim.models.fasttext.FastText,
    }

    model_corp_file = model_map[args.model_type](
            corpus_file=args.tmp_corpus_path + args.lang,
            iter=args.iter,
            size=300,
            workers=args.num_workers,
            # min_count=0,
    )

    model_corp_file.wv.save_word2vec_format(f'{args.model_type}.{args.lang}')
    model_corp_file.save(f'{args.model_type}.{args.lang}.model')


if __name__ == '__main__':
    main()
