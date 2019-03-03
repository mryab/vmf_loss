import argparse
import os
import pathlib
import random

import torch
from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset
from torchtext.vocab import Vectors
from tqdm import tqdm

from model import Model

from sacremoses import MosesDetokenizer, MosesDetruecaser
from sacrebleu import corpus_bleu


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
        vocab_size = 50000
    elif args.token_type == 'word_bpe':
        path_src = pathlib.Path('truecased')
        path_dst = pathlib.Path('bpe')
        vocab_size = 50000
    else:
        path_src = path_dst = pathlib.Path('bpe')
        vocab_size = 50000  # should be 100k for bpe, but some corpora don't have this many words
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
    src_field.build_vocab(train_dataset, max_size=vocab_size)
    tgt_field.build_vocab(train_dataset, max_size=vocab_size)

    test_iter = BucketIterator(
        test_dataset,
        batch_size=args.batch_size,
        train=False,
        sort=False,
        sort_within_batch=False,  # ensure correct order
        device=device,
    )
    out_dim = len(tgt_field.vocab)
    if args.loss != 'xent':
        # assign pretrained embeddings to trg_field
        vectors = Vectors(name='corpus.fasttext.txt', cache=args.emb_dir)  # temporal path
        mean = torch.zeros((vectors.dim,))
        num = 0
        for word, ind in vectors.stoi.items():
            if tgt_field.vocab.stoi.get(word) is None:
                mean += vectors.vectors[ind]
                num += 1
        mean /= num
        tgt_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim, unk_init=MeanInit(mean))
        tgt_field.vocab.vectors[tgt_field.vocab.stoi['<EOS>']] = torch.ones(vectors.dim)
        tgt_field.vocab.vectors = F.normalize(tgt_field.vocab.vectors, p=2, dim=-1)
        out_dim = vectors.dim
    model = Model(1024, 512, out_dim, src_field, tgt_field, 0.2).to(device)
    path = pathlib.Path('checkpoints') / args.dataset / args.token_type / args.loss
    if args.loss != 'xent':
        path /= args.emb_type
    path /= f'checkpoint_{args.eval_checkpoint}.pt'
    assert os.path.exists(path), 'No checkpoint exists at a given path'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    total_unk = 0
    if args.token_type == 'word':
        word_dict = {}
        with open(f'{args.dataset}/align/dict') as f:
            for line in f:
                src_word, dst_word = line.strip().split()
                word_dict[src_word] = dst_word
    res = []
    detokenizer = MosesDetokenizer()
    detruecaser = MosesDetruecaser()

    def replace_unk_word(current_word, aligned_src):
        if current_word == tgt_field.unk_token:
            aligned_word = src_field.vocab.itos[aligned_src]
            repl = word_dict.get(aligned_word)
            if repl is None:
                return aligned_word
            else:
                return repl
        else:
            return current_word

    with torch.no_grad():
        for batch in tqdm(test_iter):
            src, src_len = batch.src
            order = sorted(range(len(src_len)), key=src_len.__getitem__, reverse=True)
            src = src[order]
            src_len = src_len[order]
            rev_order = sorted(range(len(order)), key=order.__getitem__)
            preds, attn = model.translate_greedy(src, src_len, max_len=150, loss_type=args.loss)
            print(attn.sum(2))
            max_attn, alignments = attn.max(2)
            total_unk += (preds == tgt_field.vocab.stoi[tgt_field.unk_token]).sum().item()
            preds = preds[rev_order]
            alignments = alignments[rev_order]
            words_for_alignments = src[rev_order][torch.arange(src.size(0))[:, None], alignments]
            for sent, align in zip(preds, words_for_alignments):
                words = [tgt_field.vocab.itos[token] for token in sent]
                if tgt_field.eos_token in words:
                    words = words[:words.index(tgt_field.eos_token)]
                if args.token_type == 'word':
                    words = [replace_unk_word(word, align_for_word) for word, align_for_word in zip(words, align)]
                words = ' '.join(words)
                if args.token_type in ['bpe', 'word_bpe']:
                    words = words.replace('@@ ', '')
                words = detruecaser.detruecase(words)
                words = detokenizer.detokenize(words)
                res.append(words)
    gt = []
    with open(pathlib.Path(args.dataset) / path_dst / f'test.{tgt_lang}') as test_file:
        lines = test_file.read().splitlines()
        for words in tqdm(lines):
            if args.token_type in ['bpe', 'word_bpe']:
                words = words.replace('@@ ', '')
            words = detruecaser.detruecase(words)
            words = detokenizer.detokenize(words)
            gt.append(words)

    print(corpus_bleu(res, [gt]))
    print(total_unk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['de-en', 'en-fr', 'fr-en'], required=True)
    parser.add_argument('--token-type', choices=['word', 'bpe', 'word_bpe'], required=True)
    parser.add_argument('--loss', choices=['xent', 'l2', 'cosine', 'maxmarg', 'vmf'], required=True)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--emb-type', choices=['w2v', 'fasttext'], required=False)
    parser.add_argument('--emb-dir', type=str, required=False)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--eval-checkpoint', default='best', type=str)
    args = parser.parse_args()
    decode(args)


if __name__ == '__main__':
    main()
