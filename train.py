import argparse
import os
import pathlib
import random
import time

import torch
import torch.nn as nn
from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset
from torchtext.vocab import Vectors
from tqdm import tqdm

from model import Model
import losses


class MeanInit:
    def __init__(self, init_vector):
        self.init_vector = init_vector

    def __call__(self, tensor):
        return tensor.zero_() + self.init_vector


class TimeMeter(object):

    def __init__(self):
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    def avg(self):
        return self.n / self.elapsed_time()

    def elapsed_time(self):
        return time.time() - self.start


class StopwatchMeter(object):

    def __init__(self, sum_=0):
        self.sum = sum_
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.start_time = None


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


def train_epoch(model, train_iter, optimizer, criterion, wall_timer):
    wall_timer.start()
    model.train()
    pbar = tqdm(train_iter)
    total_loss = 0
    samples_per_sec = TimeMeter()
    time_per_batch = TimeMeter()
    for batch in pbar:
        loss = compute_loss(model, batch, criterion, optimizer)
        torch.cuda.empty_cache()
        pbar.set_postfix(loss=loss)
        total_loss += loss
        samples_per_sec.update(len(batch))
        time_per_batch.update()
    wall_timer.stop()
    print(
            f'Train loss: {total_loss / len(train_iter):.5f} '
            f'Samples per second: {samples_per_sec.avg():.3f} '
            f'Time per batch: {1 / time_per_batch.avg():.3f} '
            f'Time elapsed: {wall_timer.sum:.3f}'
    )


def validate(model, val_iter, criterion, wall_timer):
    wall_timer.start()
    model.eval()
    pbar = tqdm(val_iter)
    total_loss = 0
    with torch.no_grad():
        for batch in pbar:
            loss = compute_loss(model, batch, criterion)
            total_loss += loss
            pbar.set_postfix(loss=loss)
    res = total_loss / len(val_iter)
    wall_timer.stop()
    print(
            f'Validation loss: {res:.5f} '
            f'Time elapsed: {wall_timer.sum:.3f}'
    )
    return res


def train(args):
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
    val_dataset = TranslationDataset(
            args.dataset + '/',
            exts=list(map(lambda x: str(x[0] / f'dev.{x[1]}'), path_field_pairs)),
            fields=(src_field, tgt_field),
            filter_pred=filter_pred,
    )

    random.seed(args.device_id)
    torch.manual_seed(args.device_id)
    device = torch.device('cuda', args.device_id)
    torch.cuda.set_device(device)
    src_field.build_vocab(train_dataset, max_size=inp_vocab_size - 4)  # 4 special tokens are added automatically
    tgt_field.build_vocab(train_dataset, max_size=out_vocab_size - 4)

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
    out_dim = out_vocab_size
    if args.loss != 'xent':
        # assign pretrained embeddings to trg_field
        vectors = Vectors(name='corpus.fasttext.txt', cache=args.emb_dir)  # temporary path
        mean = torch.zeros((vectors.dim,))
        num = 0
        for word, ind in vectors.stoi.items():
            if tgt_field.vocab.stoi.get(word) is None:
                mean += vectors.vectors[ind]
                num += 1
        mean /= num
        tgt_field.vocab.set_vectors(
                vectors.stoi,
                vectors.vectors,
                vectors.dim,
                unk_init=MeanInit(mean))
        tgt_field.vocab.vectors[tgt_field.vocab.stoi['<EOS>']] = torch.ones(vectors.dim)
        if args.loss != 'l2':
            tgt_field.vocab.vectors = nn.functional.normalize(tgt_field.vocab.vectors, p=2, dim=-1)
        out_dim = vectors.dim
    model = Model(1024, 512, out_dim, src_field, tgt_field, 0.3 if args.loss == 'xent' else 0.0).to(device)
    # TODO change inp_dim for tied embeddings
    if args.loss == 'xent':
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_field.vocab.stoi[tgt_field.pad_token]).to(device)
    elif args.loss == 'l2':
        criterion = losses.L2Loss(tgt_field, out_dim).to(device)
    elif args.loss == 'cosine':
        criterion = losses.CosineLoss(tgt_field, out_dim).to(device)
    elif args.loss == 'maxmarg':
        criterion = losses.MaxMarginLoss(tgt_field, out_dim).to(device)
    elif args.loss == 'vmfapprox_paper':
        criterion = losses.NLLvMFApproxPaper(tgt_field, out_dim, args.reg_1, args.reg_2).to(device)
    elif args.loss == 'vmfapprox_fixed':
        criterion = losses.NLLvMFApproxFixed(tgt_field, out_dim, args.reg_1, args.reg_2).to(device)
    elif args.loss == 'vmf':
        criterion = losses.NLLvMF(tgt_field, out_dim, args.reg_1, args.reg_2).to(device)
    else:
        raise ValueError
    print('Starting training')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
        wall_timer = StopwatchMeter(checkpoint['train_wall'])
    else:
        wall_timer = StopwatchMeter()
        best_val_loss = validate(model, val_iter, criterion, wall_timer)
    for epoch in range(init_epoch, args.num_epoch):
        train_epoch(model, train_iter, optimizer, criterion, wall_timer)
        val_loss = validate(model, val_iter, criterion, wall_timer)
        best_val_loss = min(best_val_loss, val_loss)
        checkpoint = {
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'train_wall': wall_timer.sum,
        }
        torch.save(checkpoint, path / f'checkpoint_{epoch}.pt')
        if val_loss == best_val_loss:
            torch.save(checkpoint, path / 'checkpoint_best.pt')
        torch.save(checkpoint, path / 'checkpoint_last.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['de-en', 'en-fr', 'fr-en'], required=True)
    parser.add_argument('--token-type', choices=['word', 'bpe', 'word_bpe'], required=True)
    parser.add_argument('--loss', choices=['xent', 'l2', 'cosine', 'maxmarg', 'vmfapprox_paper', 'vmfapprox_fixed', 'vmf'],
                        required=True)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-epoch', default=15, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--emb-type', choices=['w2v', 'fasttext'], required=False)
    parser.add_argument('--emb-dir', type=str, required=False)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--reg_1', default=0, type=float)
    parser.add_argument('--reg_2', default=1, type=float)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
