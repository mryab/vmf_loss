import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hid_dim, inp_emb_dim, inp_voc, dropout):
        super().__init__()
        self.input_emb = nn.Embedding(
                len(inp_voc.vocab),
                inp_emb_dim,
                padding_idx=inp_voc.vocab.stoi[inp_voc.pad_token],
        )
        self.encoder = nn.LSTM(
                inp_emb_dim,
                hid_dim // 2,
                num_layers=2,
                bidirectional=True,
                dropout=dropout,
        )
        self.hid_dim = hid_dim // 2

    def forward(self, src_tokens, src_lengths):
        src_emb = self.input_emb(src_tokens)
        src_packed = nn.utils.rnn.pack_padded_sequence(src_emb, src_lengths)
        seqlen, bsz = src_tokens.size()
        h0 = src_emb.new_zeros((4, bsz, self.hid_dim))
        c0 = src_emb.new_zeros((4, bsz, self.hid_dim))
        encoder_output, (enc_h, enc_c) = self.encoder(src_packed, (h0, c0))
        encoder_output, _ = nn.utils.rnn.pad_packed_sequence(encoder_output)

        def combine_bidir(outs):
            out = outs.view(2, 2, bsz, -1).transpose(1, 2).contiguous()
            return out.view(2, bsz, -1)

        return encoder_output, combine_bidir(enc_h), combine_bidir(enc_c)


class AttentionLayer(nn.Module):
    def __init__(self, output_embed_dim, bias=False):
        super().__init__()
        self.output_proj = nn.Linear(2 * output_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input_, source_hids, encoder_padding_mask):
        x = input_
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        attn_scores = torch.softmax(attn_scores.masked_fill_(encoder_padding_mask, float('-inf')), dim=0)

        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        x = torch.tanh(self.output_proj(torch.cat((x, input_), dim=1)))
        return x, attn_scores


class Decoder(nn.Module):
    def __init__(self, inp_emb_dim, hid_dim, out_dim, out_voc, dropout):
        super().__init__()
        self.output_emb = nn.Embedding(
                len(out_voc.vocab),
                inp_emb_dim,
                padding_idx=out_voc.vocab.stoi[out_voc.pad_token],
        )
        self.layers = nn.ModuleList(
                [nn.LSTMCell(
                        hid_dim + inp_emb_dim if i == 0 else hid_dim,
                        hid_dim
                ) for i in range(2)])
        self.attn = AttentionLayer(hid_dim)
        self.pred_proj = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim

    def forward(self, enc_out, enc_hid, enc_memory, encoder_padding_mask, dst):
        dst_emb = self.output_emb(dst)
        seqlen, bsz = dst.size()
        inp_feed = dst_emb.new_zeros(bsz, self.hid_dim)
        decoder_hidden = [enc_hid[i] for i in range(2)]
        decoder_memory = [enc_memory[i] for i in range(2)]
        decoder_outputs = []
        for step, cur_emb in enumerate(dst_emb):
            input_ = torch.cat((cur_emb, inp_feed), dim=1)
            for i, layer in enumerate(self.layers):
                new_hidden, new_memory = layer(input_, (decoder_hidden[i], decoder_memory[i]))
                input_ = self.dropout(new_hidden)
                decoder_hidden[i] = new_hidden
                decoder_memory[i] = new_memory
            out, attn_scores = self.attn(new_hidden, enc_out, encoder_padding_mask)
            out = self.dropout(out)
            decoder_outputs.append(out)
            inp_feed = out
        res = torch.stack(decoder_outputs).transpose(1, 0)  # bsz x seqlen x hid_dim
        res = self.pred_proj(res).transpose(1, 2)  # bsz x hid_dim x seqlen
        return res


class Model(nn.Module):
    def __init__(self, hid_dim, inp_emb_dim, out_dim, inp_voc, out_voc, dropout):
        super().__init__()
        self.encoder = Encoder(hid_dim, inp_emb_dim, inp_voc, dropout)
        self.decoder = Decoder(inp_emb_dim, hid_dim, out_dim, out_voc, dropout)
        self.pad_idx = inp_voc.vocab.stoi[inp_voc.pad_token]
        self.out_voc = out_voc

    def forward(self, src_tokens, src_lengths, dst):
        src_tokens = src_tokens.transpose(0, 1)
        dst = dst.transpose(0, 1)
        encoder_output, enc_h, enc_c = self.encoder(src_tokens, src_lengths)
        enc_mask = src_tokens.eq(self.pad_idx)
        output = self.decoder(encoder_output, enc_h, enc_c, enc_mask, dst)
        return output

    def translate_greedy(self, src_tokens, src_lengths, max_len=100, loss_type='xent'):
        src_tokens = src_tokens.transpose(0, 1)
        enc_out, enc_h, enc_c = self.encoder(src_tokens, src_lengths)
        enc_mask = src_tokens.eq(self.pad_idx)
        bsz = src_tokens.size(1)
        cur_word = torch.full((bsz,), self.out_voc.vocab.stoi[self.out_voc.init_token],
                              device=src_tokens.device, dtype=torch.long)
        cur_emb = self.decoder.output_emb(cur_word)
        inp_feed = cur_emb.new_zeros(bsz, self.decoder.hid_dim)
        decoder_hidden = [enc_h[i] for i in range(2)]
        decoder_memory = [enc_c[i] for i in range(2)]
        decoder_outputs = []
        attention_scores = []
        for step in range(max_len):
            input_ = torch.cat((cur_emb, inp_feed), dim=1)
            for i, layer in enumerate(self.layers):
                new_hidden, new_memory = layer(input_, (decoder_hidden[i], decoder_memory[i]))
                input_ = self.dropout(new_hidden)
                decoder_hidden[i] = new_hidden
                decoder_memory[i] = new_memory
            out, attn_scores = self.attn(new_hidden, enc_out, enc_mask)
            out = self.dropout(out)
            inp_feed = out
            if loss_type == 'xent':
                pred_words = self.decoder.pred_proj(out).max(1)[1]
            else:
                distances = self.compute_distances(self.decoder.pred_proj(out), loss_type)
                pred_words = distances.min(1)[1]
            cur_emb = self.decoder.output_emb(pred_words.to(src_tokens.device))
            decoder_outputs.append(pred_words)
            attention_scores.append(attn_scores)
        res = torch.stack(decoder_outputs).transpose(1, 0)
        attn = torch.stack(attention_scores).permute(2, 0, 1)  # seq_len x nsrc x bsz -> bsz x seq_len x nsrc
        return res, attn

    def compute_distances(self, output_vecs, loss_type):
        # bsz x dim, voc_size x dim -> bsz x voc_size
        # bsz x dim and dim x voc_size ->

        device = output_vecs.device
        vecs = self.out_voc.vocab.vectors.to(device)
        if loss_type == 'l2':
            r_out = torch.norm(output_vecs, dim=1).unsqueeze(1) ** 2

            r_voc = torch.norm(vecs, dim=1).unsqueeze(0) ** 2
            r_scal_prod = 2 * output_vecs.matmul(vecs.transpose(0, 1))
            return r_out + r_voc - r_scal_prod
        else:
            out_ves_norm = nn.functional.normalize(output_vecs, p=2, dim=1)
            return 1 - out_ves_norm.matmul(vecs.transpose(0, 1))
