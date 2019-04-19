import pathlib

import torch
import torch.nn as nn
from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset
from torchtext.vocab import Vectors


class MeanInit:
    def __init__(self, init_vector):
        self.init_vector = init_vector

    def __call__(self, tensor):
        return tensor.zero_() + self.init_vector


def filter_pred(example):
    return len(example.src) <= 100 and len(example.trg) <= 100


def load_tgt_vectors(args, tgt_field):
    src_lang, tgt_lang = args.dataset.split("-")
    vectors = Vectors(name=f"{args.emb_type}.{tgt_lang}", cache=args.emb_dir)
    mean = torch.zeros((vectors.dim,))
    num = 0
    for word, ind in vectors.stoi.items():
        if tgt_field.vocab.stoi.get(word) is None:
            mean += vectors.vectors[ind]
            num += 1
    mean /= num
    tgt_field.vocab.set_vectors(
        vectors.stoi, vectors.vectors, vectors.dim, unk_init=MeanInit(mean)
    )
    tgt_field.vocab.vectors[tgt_field.vocab.stoi["<EOS>"]] = torch.ones(vectors.dim)
    if args.loss != "l2":
        tgt_field.vocab.vectors = nn.functional.normalize(
            tgt_field.vocab.vectors, p=2, dim=-1
        )


def setup(args, train=True):
    src_field = Field(
        batch_first=True,
        include_lengths=True,
        fix_length=None,
        init_token="<BOS>",
        eos_token="<EOS>",
    )
    tgt_field = Field(
        batch_first=True,
        include_lengths=True,
        fix_length=None,
        init_token="<BOS>",
        eos_token="<EOS>",
    )
    src_lang, tgt_lang = args.dataset.split("-")
    if args.token_type == "word":
        path_src = path_dst = pathlib.Path("truecased")
        inp_vocab_size = out_vocab_size = 50000
    elif args.token_type == "word_bpe":
        path_src = pathlib.Path("truecased")
        path_dst = pathlib.Path("bpe")
        inp_vocab_size = 50000
        out_vocab_size = 16000  # inferred from the paper
    else:
        path_src = path_dst = pathlib.Path("bpe")
        inp_vocab_size = out_vocab_size = 16000
    path_field_pairs = list(zip((path_src, path_dst), (src_lang, tgt_lang)))
    train_dataset = TranslationDataset(
        args.dataset + "/",
        exts=list(
            map(lambda x: str(x[0] / f"train.{args.dataset}.{x[1]}"), path_field_pairs)
        ),
        fields=(src_field, tgt_field),
        filter_pred=filter_pred,
    )
    if train:
        val_dataset = TranslationDataset(
            args.dataset + "/",
            exts=list(map(lambda x: str(x[0] / f"dev.{x[1]}"), path_field_pairs)),
            fields=(src_field, tgt_field),
            filter_pred=filter_pred,
        )
    else:
        test_dataset = TranslationDataset(
            args.dataset + "/",
            exts=list(map(lambda x: str(x[0] / f"test.{x[1]}"), path_field_pairs)),
            fields=(src_field, tgt_field),
        )
    src_field.build_vocab(
        train_dataset, max_size=inp_vocab_size - 4
    )  # 4 special tokens added automatically
    tgt_field.build_vocab(train_dataset, max_size=out_vocab_size - 4)
    device = torch.device("cuda", args.device_id)
    if train:
        train_iter = BucketIterator(
            train_dataset,
            batch_size=args.batch_size,
            sort_key=lambda x: (len(x.src), len(x.trg)),
            sort_within_batch=True,
            device=device,
        )
        val_iter = BucketIterator(
            val_dataset,
            batch_size=args.batch_size,
            train=False,
            sort_key=lambda x: (len(x.src), len(x.trg)),
            sort_within_batch=True,
            device=device,
        )
        return train_iter, val_iter, src_field, tgt_field
    else:
        test_iter = BucketIterator(
            test_dataset,
            batch_size=args.batch_size,
            train=False,
            sort=False,
            sort_within_batch=False,  # ensure correct order
            device=device,
        )
        return test_iter, src_field, tgt_field, path_dst, src_lang, tgt_lang
