import argparse
import os
import pathlib
import random
from collections import Counter

import torch
import torch.nn as nn
from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset
from torchtext.vocab import Vectors
from tqdm import tqdm

from model import Model


class MeanInit:
    def __init__(self, init_vector):
        self.init_vector = init_vector

    def __call__(self, tensor):
        return tensor.zero_() + self.init_vector


def filter_pred(example):
    return len(example.src) <= 100 and len(example.trg) <= 100


def decode(args):
    src_field = Field(
            batch_first=True,
            include_lengths=True,
            fix_length=None,
            init_token='<BOS>',
            eos_token='<EOS>',
    )
    tgt_field = Field(
            batch_first=True,
            include_lengths=True,
            fix_length=None,
            init_token='<BOS>',
            eos_token='<EOS>',
    )
    src_lang, tgt_lang = args.dataset.split('-')
    if args.token_type == 'word':
        path_src = path_dst = pathlib.Path('truecased')
        inp_vocab_size = out_vocab_size = 50000
    elif args.token_type == 'word_bpe':
        path_src = pathlib.Path('truecased')
        path_dst = pathlib.Path('bpe')
        inp_vocab_size = 50000
        out_vocab_size = 16000  # inferred from the paper
    else:
        path_src = path_dst = pathlib.Path('bpe')
        inp_vocab_size = out_vocab_size = 16000
    path_field_pairs = list(zip((path_src, path_dst), (src_lang, tgt_lang)))
    train_dataset = TranslationDataset(
            args.dataset + '/',
            exts=list(map(lambda x: str(x[0] / f'train.{args.dataset}.{x[1]}'), path_field_pairs)),
            fields=(src_field, tgt_field),
            filter_pred=filter_pred,
    )
    test_dataset = TranslationDataset(
            args.dataset + '/',
            exts=list(map(lambda x: str(x[0] / f'test.{x[1]}'), path_field_pairs)),
            fields=(src_field, tgt_field),
    )

    random.seed(args.device_id)
    torch.manual_seed(args.device_id)
    device = torch.device('cuda', args.device_id)
    torch.cuda.set_device(device)
    src_field.build_vocab(train_dataset, max_size=inp_vocab_size - 4)  # 4 special tokens are added automatically
    tgt_field.build_vocab(train_dataset, max_size=out_vocab_size - 4)

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
    test_iter = BucketIterator(
            test_dataset,
            batch_size=args.batch_size,
            train=False,
            sort=False,
            sort_within_batch=False,  # ensure correct order
            device=device,
    )
    out_dim = out_vocab_size
    if args.loss != 'xent':
        # assign pretrained embeddings to trg_field
        vectors = Vectors(name=args.emb_type + '.' + tgt_lang, cache=args.emb_dir)
        mean = torch.zeros((vectors.dim,))
        num = 0
        for word, ind in vectors.stoi.items():
            if tgt_field.vocab.stoi.get(word) is None:
                mean += vectors.vectors[ind]
                num += 1
        mean /= num
        tgt_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim, unk_init=MeanInit(mean))
        tgt_field.vocab.vectors[tgt_field.vocab.stoi['<EOS>']] = torch.ones(vectors.dim)
        if args.loss != 'l2':
            tgt_field.vocab.vectors = nn.functional.normalize(tgt_field.vocab.vectors, p=2, dim=-1)
        out_dim = vectors.dim
    model = Model(1024, 512, out_dim, src_field, tgt_field,
                  0.3 if args.loss == 'xent' else 0.0, tied=args.tied).to(device)
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
    if args.token_type == 'word':
        word_dict = {}
        with open(f'{args.dataset}/align/dict') as f:
            for line in f:
                src_word, dst_word = line.strip().split()
                word_dict[src_word] = dst_word

    def replace_unk_word(current_word, aligned_src, gt_src):
        if current_word == tgt_field.unk_token:
            aligned_word = src_field.vocab.itos[aligned_src]
            repl = word_dict.get(aligned_word)
            if repl is None:
                if gt_src in [src_field.init_token, src_field.eos_token]:
                    return ''
                return gt_src
            return repl
        return current_word

    path = pathlib.Path('checkpoints') / args.dataset / args.token_type / args.loss
    if args.loss != 'xent':
        path /= args.emb_type
    if args.tied:
        path /= 'tied'
    if args.loss in ['vmfapprox_paper', 'vmfapprox_fixed', 'vmf']:
        path /= f'reg1{args.reg_1}_reg2{args.reg_2}'
    path /= 'checkpoint_last.pt'
    assert os.path.exists(path), 'No checkpoint exists at a given path: {}'.format(path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    res = []
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(test_iter)):
            src, src_len = batch.src
            order = sorted(range(len(src_len)), key=src_len.__getitem__, reverse=True)
            src = src[order]
            src_len = src_len[order]
            rev_order = sorted(range(len(order)), key=order.__getitem__)
            preds, attn = model.translate_greedy(src, src_len, max_len=100, loss_type=args.loss)
            max_attn, alignments = attn.max(2)
            preds = preds[rev_order]
            alignments = alignments[rev_order]
            words_for_alignments = src[rev_order][torch.arange(src.size(0))[:, None], alignments]
            for sent_num, (sent, align) in enumerate(zip(preds, words_for_alignments)):
                words = [tgt_field.vocab.itos[token] for token in sent]
                if tgt_field.eos_token in words:
                    cut_ind = words.index(tgt_field.eos_token)
                    words = words[:cut_ind]
                else:
                    cut_ind = len(words)
                if args.token_type == 'word':
                    alignments_cut = alignments[sent_num][:cut_ind]
                    gt_for_sent = src_raw[batch_num * args.batch_size + sent_num]
                    gt_words_for_sent = [gt_for_sent[ind] for ind in alignments_cut]
                    words = [replace_unk_word(word, align_for_word, gt_word_for_sent)
                             for word, align_for_word, gt_word_for_sent in zip(words, align, gt_words_for_sent)]
                words = ' '.join(words)
                if args.token_type in ['bpe', 'word_bpe']:
                    words = words.replace('@@ ', '')
                res.append(words)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['de-en', 'en-fr', 'fr-en'], required=True)
    parser.add_argument('--token-type', choices=['word', 'bpe', 'word_bpe'], required=True)
    parser.add_argument('--loss',
                        choices=['xent', 'l2', 'cosine', 'maxmarg', 'vmfapprox_paper', 'vmfapprox_fixed', 'vmf'],
                        required=True)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--emb-type', choices=['w2v', 'fasttext'], required=False)
    parser.add_argument('--emb-dir', type=str, required=False)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--eval-checkpoint', default='best', type=str)
    parser.add_argument('--reg_1', default=0, type=float)
    parser.add_argument('--reg_2', default=1, type=float)
    parser.add_argument('--tied', action='store_true')
    args = parser.parse_args()
    decode(args)


if __name__ == '__main__':
    main()
