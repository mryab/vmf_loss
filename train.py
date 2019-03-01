import argparse
import os
import pathlib
import random

import torch
import torch.nn as nn
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


def train_dummy(model, args, criterion, optimizer):
    dummy_src = torch.zeros((args.batch_size, 110), dtype=torch.long).cuda()
    dummy_src[:, -1] = 3
    dummy_src_lengths = torch.full((args.batch_size,), 110, dtype=torch.long).cuda()
    dummy_dst = torch.zeros((args.batch_size, 110), dtype=torch.long).cuda()
    dummy_dst[:, -1] = 3
    outputs_voc = model(dummy_src, dummy_src_lengths, dummy_dst[:, :-1]).transpose(1, 2)
    loss = criterion(outputs_voc, dummy_dst[:, 1:])
    loss.backward()
    optimizer.zero_grad()


def train_step(model, batch, optimizer, criterion):
    src, src_lengths = batch.src
    dst, dst_lengths = batch.trg
    src = src.cuda()
    dst = dst.cuda()
    src_lengths = src_lengths.cuda()
    outputs_voc = model(src, src_lengths, dst[:, :-1]).transpose(1, 2)
    target = dst[:, 1:]
    loss = criterion(outputs_voc, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(model, train_iter, optimizer, criterion):
    model.train()
    pbar = tqdm(train_iter)
    total_loss = 0
    count = 0
    for batch in pbar:
        loss = train_step(model, batch, optimizer, criterion)
        pbar.set_postfix(loss=loss)
        torch.cuda.empty_cache()
        total_loss += loss
        count += 1
    print(f'Train loss: {total_loss / count:.5f}')


def validate(model, val_iter, criterion):
    model.eval()
    with torch.no_grad():
        pbar = tqdm(val_iter)
        total_loss = 0
        count = 0
        for batch in pbar:
            src, src_lengths = batch.src
            dst, dst_lengths = batch.trg
            src = src.cuda()
            dst = dst.cuda()
            src_lengths = src_lengths.cuda()
            outputs_voc = model(src, src_lengths, dst[:, :-1]).transpose(1, 2)
            target = dst[:, 1:]
            loss = criterion(outputs_voc, target).item()
            total_loss += loss
            count += 1
            pbar.set_postfix(loss=loss)
    print(f'Validation loss: {total_loss / count:.5f}')
    return total_loss / count


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
        path_src = pathlib.Path('truecased')
        path_dst = pathlib.Path('truecased')
        vocab_size = 50000
    elif args.token_type == 'word_bpe':
        path_src = pathlib.Path('truecased')
        path_dst = pathlib.Path('bpe')
        vocab_size = 50000
    else:
        path_src = pathlib.Path('bpe')
        path_dst = pathlib.Path('bpe')
        vocab_size = 50000  # should be 100k for bpe, but some corpora don't have this many words
    train, val, test = TranslationDataset.splits(
        (path_src / src_lang, path_dst / tgt_lang),
        (src_field, tgt_field),
        path=args.dataset,
        train=f'train.{args.dataset}.',
        validation='dev.',
        test='test.',
        filter_pred=filter_pred,
    )

    random.seed(args.device_id)
    torch.manual_seed(args.device_id)
    torch.cuda.set_device(args.device_id)
    src_field.build_vocab(train, max_size=vocab_size)
    tgt_field.build_vocab(train, max_size=vocab_size)

    train_iter = BucketIterator(
        train,
        batch_size=args.batch_size,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        sort_within_batch=True,
    )
    val_iter = BucketIterator(
        val,
        batch_size=args.batch_size,
        train=False,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        sort_within_batch=True,
    )
    #assign pretrained embeddings to trg_field
    vectors = Vectors(name=args.emb_type + '.' + tgt_lang, cache=args.emb_dir)
    mean = torch.zeros((vectors.dim,))
    num = 0
    for word, ind in vectors.stoi.items():
        if tgt_field.vocab.stoi.get(word) is None:
            mean += vectors.vectors[ind]
            num += 1
    mean /= num
    tgt_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim, unk_init=MeanInit(mean))
    
    out_dim = len(tgt_field.vocab) if args.emb_type is None else vectors.dim
    model = Model(1024, 512, out_dim, src_field, tgt_field, 0.2).cuda()
    # TODO change criterion (and output dim) depending on args; inp_dim for tied embeddings too
    if args.loss == 'xent':
        criterion = nn.CrossEntropyLoss(ignore_index=1).cuda()
    if args.loss == 'l2':
        criterion = EmbeddingLoss(tgt_field, out_dim, L2Loss).cuda()
    if args.loss == 'cosine':
        criterion = EmbeddingLoss(tgt_field, out_dim, CosineLoss).cuda()
        
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
    os.makedirs(path, exist_ok=True)
    init_epoch = 0
    if os.path.exists(path / 'checkpoint_last.pt'):
        checkpoint = torch.load(path / 'checkpoint_last.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        init_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
    else:
        best_val_loss = validate(model, val_iter, criterion)
    train_dummy(model, args, criterion, optimizer)
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
