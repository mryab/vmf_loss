from argparse import ArgumentParser

from losses import get_available_losses


def add_general_args(parser):
    group = parser.add_argument_group("General options")
    group.add_argument("--dataset", choices=["de-en", "en-fr", "fr-en"], required=True)
    group.add_argument(
        "--token-type", choices=["word", "bpe", "word_bpe"], required=True
    )
    group.add_argument("--loss", choices=get_available_losses(), required=True)
    group.add_argument("--batch-size", default=64, type=int)
    group.add_argument("--num-epoch", default=15, type=int)
    group.add_argument("--emb-type", choices=["w2v", "fasttext"], required=False)
    group.add_argument("--emb-dir", default=".", type=str)
    group.add_argument("--device-id", default=0, type=int)
    group.add_argument("--reg_1", default=0, type=float)
    group.add_argument("--reg_2", default=1, type=float)
    group.add_argument("--tied", action="store_true")


def add_training_args(parser):
    group = parser.add_argument_group("Training options")
    group.add_argument("--lr", default=0.0002, type=float)


def add_evaluation_args(parser):
    group = parser.add_argument_group("Evaluation options")
    group.add_argument("--eval-checkpoint", default="best", type=str)


def create_training_parser():
    parser = ArgumentParser()
    add_general_args(parser)
    add_training_args(parser)
    return parser


def create_evaluation_parser():
    parser = ArgumentParser()
    add_general_args(parser)
    add_evaluation_args(parser)
    return parser
