import random

import torch
import torch.nn as nn
from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset

from tqdm import tqdm
from model import Model

import argparse
import os
import pathlib
import gc


def filter_pred(example):
    return len(example.src) <= 100 and len(example.trg) <= 100


def train_step(model, batch, optimizer, criterion):
    src, src_lengths = batch.src
    dst, dst_lengths = batch.trg
    src = src.cuda()
    dst = dst.cuda()
    src_lengths = src_lengths.cuda()
    optimizer.zero_grad()
    outputs_voc = model(src, src_lengths, dst[:, :-1]).transpose(1, 2)
    target = dst[:, 1:]
    loss = criterion(outputs_voc, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(model, train_iter, optimizer, criterion):
    model.train()
    pbar = tqdm(train_iter)
    for batch in pbar:
        loss = train_step(model, batch, optimizer, criterion)
        pbar.set_postfix(loss=loss)


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
    print(total_loss / count)
    gc.collect()
    return total_loss / count


def main(args, init_distributed=False):
    src_field = Field(batch_first=True,
                      include_lengths=True,
                      fix_length=None,
                      init_token='<BOS>',
                      eos_token='<EOS>'
                      )
    tgt_field = Field(batch_first=True,
                      include_lengths=True,
                      fix_length=None,
                      init_token='<BOS>',
                      eos_token='<EOS>'
                      )
    src_lang, tgt_lang = args.dataset.split('-')
    if args.token_type == 'word':
        path = f'{args.dataset}/truecased/'
    else:
        path = f'{args.dataset}/truecased_bpe/'
    train, val, test = TranslationDataset.splits((src_lang, tgt_lang),
                                                 (src_field, tgt_field),
                                                 path=path,
                                                 train=f'train.{args.dataset}.',
                                                 validation='dev.',
                                                 test='test.',
                                                 filter_pred=filter_pred
                                                 )

    random.seed(args.device_id)
    torch.manual_seed(args.device_id)
    torch.cuda.set_device(args.device_id)
    src_field.build_vocab(train, max_size=50000)
    tgt_field.build_vocab(train, max_size=50000)

    batch_size = 32
    train_iter = BucketIterator(train, batch_size=batch_size,
                                sort_key=lambda x: (len(x.src), len(x.trg)), sort_within_batch=True)
    val_iter = BucketIterator(val, batch_size=batch_size, train=False,
                              sort_key=lambda x: (len(x.src), len(x.trg)), sort_within_batch=True)

    model = Model(1024, 512, len(tgt_field.vocab), src_field, tgt_field, 0.2).cuda()
    # TODO change criterion (and output dim) depending on args
    criterion = nn.CrossEntropyLoss(ignore_index=1).cuda()
    if init_distributed:
        torch.distributed.init_process_group(backend='nccl',
                                             world_size=torch.cuda.device_count(),
                                             init_method=args.distributed_init_method,
                                             rank=args.distributed_rank)

        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.device_id],
                                                          output_device=args.device_id,
                                                          broadcast_buffers=False)
    if args.distributed_rank == 0:
        print('Starting training')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    best_val_loss = validate(model, val_iter, criterion)
    path = pathlib.Path('checkpoints') / args.dataset / args.token_type / args.loss
    os.makedirs(path, exist_ok=True)
    init_epoch = 0
    if os.path.exists(path / 'checkpoint_last.pt'):
        checkpoint = torch.load(path / 'checkpoint_last.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        init_epoch = checkpoint['epoch']
    for epoch in range(init_epoch, 15):
        train_epoch(model, train_iter, optimizer, criterion)
        val_loss = validate(model, val_iter, criterion)
        if args.distributed_rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(checkpoint, path / f'checkpoint_{epoch}.pt')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, path / 'checkpoint_best.pt')
            torch.save(checkpoint, path / 'checkpoint_last.pt')


def distributed_main(i, args):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = i
    main(args, init_distributed=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['de-en', 'en-fr', 'fr-en'], required=True)
    parser.add_argument('--token-type', choices=['word', 'bpe'], required=True)
    parser.add_argument('--loss', choices=['xent'], required=True)
    parser.add_argument('--device-id', default=0)
    num_gpus = torch.cuda.device_count()
    args = parser.parse_args()
    if num_gpus > 1:
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        torch.multiprocessing.spawn(fn=distributed_main, args=(args,), nprocs=num_gpus)
    else:
        args.distributed_rank = 0
        main(args, init_distributed=False)
