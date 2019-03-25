import pathlib
from collections import Counter

import torch

from decode import get_postprocess_func, translate_checkpoint
from model import Model
from util import data, misc
import options


def calc_f_scores(args):
    misc.fix_seed()
    device = torch.device('cuda', args.device_id)

    test_iter, src_field, tgt_field, path_dst, src_lang, tgt_lang = data.setup_fields_and_iters(args, train=False)

    cnt = Counter()
    with open(pathlib.Path(args.dataset) / path_dst / f'train.{args.dataset}.{src_lang}') as test_file:
        lines = test_file.read().splitlines()
        for words in lines:
            if args.token_type in ['bpe', 'word_bpe']:
                words = words.replace('@@ ', '')
            cnt.update(words.split())

    words_by_group = {
            '1': [word for word, count in cnt.items() if count == 1],
            '2': [word for word, count in cnt.items() if count == 2],
            '3': [word for word, count in cnt.items() if count == 3],
            '4': [word for word, count in cnt.items() if count == 4],
            '5-9': [word for word, count in cnt.items() if 5 <= count <= 9],
            '10-99': [word for word, count in cnt.items() if 10 <= count <= 99],
            '100-999': [word for word, count in cnt.items() if 100 <= count <= 999],
            '1000+': [word for word, count in cnt.items() if 1000 <= count],
    }
    word_to_group = {
            word: i for i, (group, words) in enumerate(words_by_group.items()) for word in words
    }
    if args.loss == 'xent':
        out_dim = len(tgt_field.vocab)
    else:
        data.load_tgt_vectors(args, tgt_field)
        out_dim = tgt_field.vocab.vectors.size(1)
    model = Model(1024, 512, out_dim, src_field, tgt_field,
                  dropout=0.3 if args.loss == 'xent' else 0.0, tied=args.tied).to(device)
    src_raw = []
    gt = []
    with open(pathlib.Path(args.dataset) / path_dst / f'test.{src_lang}') as test_file:
        lines = test_file.read().splitlines()
        for words in lines:
            src_raw.append([src_field.init_token] + words.split() + [src_field.eos_token])
    with open(pathlib.Path(args.dataset) / path_dst / f'test.{tgt_lang}') as test_file:
        lines = test_file.read().splitlines()
        for words in lines:
            if args.token_type in ['bpe', 'word_bpe']:
                words = words.replace('@@ ', '')
            gt.append(words)
    word_dict = {}
    if args.token_type == 'word':
        with open(f'{args.dataset}/align/dict') as f:
            for line in f:
                src_word, dst_word = line.strip().split()
                word_dict[src_word] = dst_word

    path = misc.get_path(args) / 'checkpoint_last.pt'
    postprocess_func = get_postprocess_func(args, src_field, tgt_field, lambda x: x,
                                            lambda x: x, word_dict)
    res, checkpoint = translate_checkpoint(model, path, test_iter, args, src_raw, postprocess_func)
    match_stats = [[0, 0, 0] for _ in words_by_group]
    for sent_pred, sent_gt in zip(res, gt):
        tokenized_pred = sent_pred.split()
        tokenized_gt = sent_gt.split()
        in_pred = Counter(tokenized_pred)
        in_gt = Counter(tokenized_gt)
        for word in set(in_pred.keys()) | set(in_gt.keys()):
            if word in word_to_group:
                ind = word_to_group[word]
                match_stats[ind][0] += min(in_pred[word], in_gt[word])
                match_stats[ind][1] += in_pred[word]
                match_stats[ind][2] += in_gt[word]
    for i, group in enumerate(words_by_group):
        recall = match_stats[i][0] / match_stats[i][1]
        precision = match_stats[i][0] / match_stats[i][2]
        print(f'{group}\t{2 * precision * recall / (precision + recall):.2f}')


def main():
    parser = options.create_evaluation_parser()
    args = parser.parse_args()
    calc_f_scores(args)


if __name__ == '__main__':
    main()
