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
from loss import *


class MeanInit:
    def __init__(self, init_vector):
        self.init_vector = init_vector

    def __call__(self, tensor):
        return tensor.zero_() + self.init_vector


def filter_pred(example):
    return len(example.src) <= 100 and len(example.trg) <= 100


def train_dummy(model, criterion, optimizer, dummy_batch):
    model.train()
    dummy_src, dummy_src_lengths, dummy_dst = dummy_batch
    outputs_voc = model(dummy_src, dummy_src_lengths, dummy_dst[:, :-1])
    loss = criterion(outputs_voc, dummy_dst[:, 1:])
    loss.backward()
    optimizer.zero_grad()


def compute_loss(model, batch, criterion, optimizer=None):
    src, src_lengths = batch.src
    dst, dst_lengths = batch.trg
    src = src.cuda()
    dst = dst.cuda()
    src_lengths = src_lengths.cuda()
    outputs_voc = model(src, src_lengths, dst[:, :-1])
    target = dst[:, 1:]
    loss = criterion(outputs_voc, target)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def train_epoch(model, train_iter, optimizer, criterion):
    model.train()
    pbar = tqdm(train_iter)
    total_loss = 0
    for batch in pbar:
        loss = compute_loss(model, batch, criterion, optimizer)
        # torch.cuda.empty_cache()
        pbar.set_postfix(loss=loss)
        total_loss += loss
    print(f'Train loss: {total_loss / len(train_iter):.5f}')


def validate(model, val_iter, criterion):
    model.eval()
    pbar = tqdm(val_iter)
    total_loss = 0
    with torch.no_grad():
        for batch in pbar:
            loss = compute_loss(model, batch, criterion)
            total_loss += loss
            pbar.set_postfix(loss=loss)
    res = total_loss / len(val_iter)
    print(f'Validation loss: {res:.5f}')
    return res


def train(args, init_distributed=False):
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
        filter_pred=filter_pred
    )
    val_dataset = TranslationDataset(
        args.dataset + '/',
        exts=list(map(lambda x: str(x[0] / f'dev.{x[1]}'), path_field_pairs)),
        fields=(src_field, tgt_field),
        filter_pred=filter_pred
    )

    random.seed(args.device_id)
    torch.manual_seed(args.device_id)
    device = torch.device('cuda', args.device_id)
    torch.cuda.set_device(device)
    src_field.build_vocab(train_dataset, max_size=vocab_size)
    tgt_field.build_vocab(train_dataset, max_size=vocab_size)

    train_iter = BucketIterator(
        train_dataset,
        batch_size=args.batch_size,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        sort_within_batch=True,
        # device=device,
    )
    val_iter = BucketIterator(
        val_dataset,
        batch_size=args.batch_size,
        train=False,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        sort_within_batch=True,
        # device=device,
    )
    out_dim = len(tgt_field.vocab)
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
        out_dim = vectors.dim
    model = Model(1024, 512, out_dim, src_field, tgt_field, 0.2).to(device)
    # TODO change criterion (and output dim) depending on args; inp_dim for tied embeddings too
    if args.loss == 'xent':
        criterion = nn.CrossEntropyLoss(ignore_index=1).to(device)
    if args.loss == 'l2':
        criterion = EmbeddingLoss(tgt_field, out_dim, L2Loss).to(device)
    if args.loss == 'cosine':
        criterion = EmbeddingLoss(tgt_field, out_dim, CosineLoss).to(device)

    if init_distributed:
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=torch.cuda.device_count(),
            init_method=args.distributed_init_method,
            rank=args.distributed_rank,
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=False,
        )
    if args.distributed_rank == 0:
        print('Starting training')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    path = pathlib.Path('checkpoints') / args.dataset / args.token_type / args.loss
    if args.loss != 'xent':
        path /= args.emb_type
    os.makedirs(path, exist_ok=True)
    init_epoch = 0

    dummy_src = torch.zeros((args.batch_size, 110), dtype=torch.long, device=device)
    dummy_src[:, -1] = 3
    dummy_src_lengths = torch.full((args.batch_size,), 110, dtype=torch.long, device=device)
    dummy_dst = torch.zeros((args.batch_size, 110), dtype=torch.long, device=device)
    dummy_dst[:, -1] = 3

    train_dummy(model, criterion, optimizer, (dummy_src, dummy_src_lengths, dummy_dst))
    if os.path.exists(path / 'checkpoint_last.pt'):
        checkpoint = torch.load(path / 'checkpoint_last.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        init_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
    else:
        best_val_loss = validate(model, val_iter, criterion)
    for epoch in range(init_epoch, 15):
        train_epoch(model, train_iter, optimizer, criterion)
        val_loss = validate(model, val_iter, criterion)
        if args.distributed_rank == 0:
            best_val_loss = min(best_val_loss, val_loss)
            checkpoint = {
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, path / f'checkpoint_{epoch}.pt')
            if val_loss == best_val_loss:
                torch.save(checkpoint, path / 'checkpoint_best.pt')
            torch.save(checkpoint, path / 'checkpoint_last.pt')


def distributed_train(i, args):
    args.device_id = i
    if args.distributed_rank is None:
        args.distributed_rank = i
    train(args, init_distributed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['de-en', 'en-fr', 'fr-en'], required=True)
    parser.add_argument('--token-type', choices=['word', 'bpe', 'word_bpe'], required=True)
    parser.add_argument('--loss', choices=['xent', 'l2', 'cosine'], required=True)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--emb-type', choices=['w2v', 'fasttext'], required=False)
    parser.add_argument('--emb-dir', type=str, required=False)
    parser.add_argument('--device-id', default=0, type=int)
    num_gpus = torch.cuda.device_count()
    args = parser.parse_args()
    if num_gpus > 1:
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        torch.multiprocessing.spawn(fn=distributed_train, args=(args,), nprocs=num_gpus)
    else:
        args.distributed_rank = 0
        train(args, init_distributed=False)


if __name__ == '__main__':
    main()
