import os

import torch
from tqdm import tqdm

import losses
import options
from model import Model
from util import data, misc


def compute_loss(model, batch, criterion, optimizer=None):
    src, src_lengths = batch.src
    dst, dst_lengths = batch.trg
    outputs_voc = model(src, src_lengths, dst[:, :-1])
    loss = criterion(outputs_voc, dst[:, 1:])
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
    samples_per_sec = misc.TimeMeter()
    time_per_batch = misc.TimeMeter()
    for batch in pbar:
        loss = compute_loss(model, batch, criterion, optimizer)
        pbar.set_postfix(loss=loss)
        total_loss += loss
        samples_per_sec.update(len(batch))
        time_per_batch.update()
    wall_timer.stop()
    print(
        f"Train loss: {total_loss / len(train_iter):.5f} "
        f"Samples per second: {samples_per_sec.avg():.3f} "
        f"Time per batch: {1 / time_per_batch.avg():.3f} "
        f"Time elapsed: {wall_timer.sum:.3f}"
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
    print(f"Validation loss: {res:.5f} " f"Time elapsed: {wall_timer.sum:.3f}")
    return res


def train(args):
    misc.fix_seed()
    device = torch.device("cuda", args.device_id)
    train_iter, val_iter, src_field, tgt_field = data.setup(
        args, train=True
    )
    if args.loss == "xent":
        out_dim = len(tgt_field.vocab)
    else:
        data.load_tgt_vectors(args, tgt_field)
        out_dim = tgt_field.vocab.vectors.size(1)
    model = Model(
        1024,
        512,
        out_dim,
        src_field,
        tgt_field,
        dropout=0.3 if args.loss == "xent" else 0.0,
        tied=args.tied,
    ).to(device)
    criterion = losses.get_loss(args, tgt_field).to(device)
    print("Starting training")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    path = misc.get_path(args)
    os.makedirs(path, exist_ok=True)

    if os.path.exists(path / "checkpoint_last.pt"):
        checkpoint = torch.load(path / "checkpoint_last.pt")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optim"])
        init_epoch = checkpoint["epoch"]
        wall_timer = misc.StopwatchMeter(checkpoint["train_wall"])
        best_val_loss = checkpoint["best_val_loss"]
    else:
        init_epoch = 0
        wall_timer = misc.StopwatchMeter()
        best_val_loss = validate(model, val_iter, criterion, wall_timer)
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)
    for epoch in range(init_epoch, args.num_epoch):
        train_epoch(model, train_iter, optimizer, criterion, wall_timer)
        val_loss = validate(model, val_iter, criterion, wall_timer)
        best_val_loss = min(best_val_loss, val_loss)
        checkpoint = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "train_wall": wall_timer.sum,
        }
        torch.save(checkpoint, path / f"checkpoint_{epoch}.pt")
        if val_loss == best_val_loss:
            torch.save(checkpoint, path / "checkpoint_best.pt")
        torch.save(checkpoint, path / "checkpoint_last.pt")


def main():
    parser = options.create_training_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
