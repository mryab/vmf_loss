import os
import pathlib

import torch
from sacrebleu import corpus_bleu
from sacremoses import MosesDetokenizer, MosesDetruecaser
from tqdm import tqdm

from model import Model
from util import data, misc
import options


def replace_unk(words, align, gt_words_for_sent, src_field, tgt_field, word_dict):
    result = []
    for word, align_for_word, gt_word_for_sent in zip(words, align, gt_words_for_sent):
        if word == tgt_field.unk_token:
            aligned_word = src_field.vocab.itos[align_for_word]
            repl = word_dict.get(aligned_word)
            if repl is None:
                if gt_word_for_sent in [src_field.init_token, src_field.eos_token]:
                    result.append('')
                else:
                    result.append(gt_word_for_sent)
            else:
                result.append(repl)
        else:
            result.append(word)
    return result


def get_postprocess_func(args, src_field, tgt_field, detruecase, detokenize, word_dict):
    def postprocess_prediction(sent, alignment, gt_for_sent):
        words = [tgt_field.vocab.itos[token] for token in sent]
        if tgt_field.eos_token in words:
            cut_ind = words.index(tgt_field.eos_token)
            words = words[:cut_ind]
        else:
            cut_ind = len(words)
        if args.token_type == 'word':
            align_cut = alignment[:cut_ind]
            gt_words_for_sent = [gt_for_sent[ind] for ind in align_cut]
            words = replace_unk(words, align_cut, gt_words_for_sent, src_field, tgt_field, word_dict)
        words = ' '.join(words)
        if args.token_type in ['bpe', 'word_bpe']:
            words = words.replace('@@ ', '')
        words = detruecase(words)
        words = detokenize(words)
        return words

    return postprocess_prediction


def translate_checkpoint(model, path, test_iter, args, src_raw, postprocess_prediction):
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
                words = postprocess_prediction(sent, align, src_raw[batch_num * args.batch_size + sent_num])
                res.append(words)
    return res, checkpoint


def decode(args):
    misc.fix_seed()
    device = torch.device('cuda', args.device_id)

    test_iter, src_field, tgt_field, path_dst, src_lang, tgt_lang = data.setup_fields_and_iters(args, train=False)

    if args.loss == 'xent':
        out_dim = len(tgt_field.vocab)
    else:
        data.load_tgt_vectors(args, tgt_field)
        out_dim = tgt_field.vocab.vectors.size(1)
    model = Model(1024, 512, out_dim, src_field, tgt_field,
                  dropout=0.3 if args.loss == 'xent' else 0.0, tied=args.tied).to(device)

    detokenizer = MosesDetokenizer(lang=tgt_lang)
    detruecaser = MosesDetruecaser()
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
            words = detruecaser.detruecase(words)
            words = detokenizer.detokenize(words)
            gt.append(words)
    word_dict = {}
    if args.token_type == 'word':
        with open(f'{args.dataset}/align/dict') as f:
            for line in f:
                src_word, dst_word = line.strip().split()
                word_dict[src_word] = dst_word
    path = misc.get_path(args)
    if args.eval_checkpoint != 'all':
        paths = [path / f'checkpoint_{args.eval_checkpoint}.pt']
    else:
        paths = sorted(list(path.glob('checkpoint_*.pt')))
        paths.remove(path / 'checkpoint_last.pt')
    result_dict = {}
    time_dict = {}
    postprocess_func = get_postprocess_func(args, src_field, tgt_field, detruecaser.detruecase,
                                            detokenizer.detokenize, word_dict)
    for path in tqdm(paths):
        res, checkpoint = translate_checkpoint(model, path, test_iter, args, src_raw, postprocess_func)
        result_dict[path.stem.split('_')[1]] = corpus_bleu(res, [gt]).score
        time_dict[path.stem.split('_')[1]] = checkpoint['train_wall']
    print('')
    for checkpoint, bleu in sorted(result_dict.items(), key=lambda x: x[0] if len(x[0]) > 1 else f'0{x[0]}'):
        print(f'{checkpoint}\tBLEU={bleu:.3f}\tTime={time_dict[checkpoint]:.3f}')


def main():
    parser = options.create_evaluation_parser()
    args = parser.parse_args()
    decode(args)


if __name__ == '__main__':
    main()
