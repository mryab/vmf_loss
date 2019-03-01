import argparse
import os
import pathlib
import random

import torch
import torch.nn as nn
from torchtext.data import BucketIterator, Field, interleave_keys
from torchtext.datasets import TranslationDataset
from tqdm import tqdm

from model import Model


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


def compute_loss(model, batch, criterion):
    src, src_lengths = batch.src
    dst, dst_lengths = batch.trg
    src = src
    dst = dst
    src_lengths = src_lengths
    outputs_voc = model(src, src_lengths, dst[:, :-1])
    target = dst[:, 1:]
    loss = criterion(outputs_voc, target)
    return loss


def train_epoch(model, train_iter, optimizer, criterion):
    model.train()
    pbar = tqdm(train_iter)
    total_loss = 0
    for batch in pbar:
        loss = compute_loss(model, batch, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        pbar.set_postfix(loss=loss)
        torch.cuda.empty_cache()
        total_loss += loss
    print(f'Train loss: {total_loss / len(train_iter):.5f}')


def validate(model, val_iter, criterion):
    model.eval()
    pbar = tqdm(val_iter)
    total_loss = 0
    with torch.no_grad():
        for batch in pbar:
            loss = compute_loss(model, batch, criterion).item()
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
    train_dataset, val_dataset = TranslationDataset.splits(
        (path_src / src_lang, path_dst / tgt_lang),
        (src_field, tgt_field),
        path=args.dataset,
        train=f'train_dataset.{args.dataset}.',
        validation='dev.',
        test=None,
        filter_pred=filter_pred,
    )

    random.seed(args.device_id)
    torch.manual_seed(args.device_id)
    torch.cuda.set_device(args.device_id)
    src_field.build_vocab(train, max_size=vocab_size)
    tgt_field.build_vocab(train, max_size=vocab_size)

    train_iter = BucketIterator(
        train_dataset,
        batch_size=args.batch_size,
        sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)),
        sort_within_batch=True,
    )
    val_iter = BucketIterator(
        val_dataset,
        batch_size=args.batch_size,
        train=False,
        sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)),
        sort_within_batch=True,
    )

    model = Model(1024, 512, len(tgt_field.vocab), src_field, tgt_field, 0.2).cuda()
    # TODO change criterion (and output dim) depending on args
    criterion = nn.CrossEntropyLoss(ignore_index=1).cuda()
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
    parser.add_argument('--loss', choices=['xent'], required=True)
    parser.add_argument('--batch-size', default=64, type=int)
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
