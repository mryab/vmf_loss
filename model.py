import torch
import torch.nn as nn


class StackedLSTMCell(nn.Module):
    def __init__(self, input_size, rnn_size):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(2):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            h_0[i] = h_1_i
            c_0[i] = c_1_i
        return input, (h_0, c_0)


class Encoder(nn.Module):
    def __init__(self, hid_dim, inp_emb_dim, inp_voc):
        super().__init__()
        self.input_emb = nn.Embedding(len(inp_voc.vocab), inp_emb_dim,
                                      padding_idx=inp_voc.vocab.stoi[inp_voc.pad_token])
        self.encoder = nn.LSTM(inp_emb_dim, hid_dim // 2, num_layers=2, bidirectional=True)

    def forward(self, src_tokens, src_lengths):
        src_emb = self.input_emb(src_tokens)
        src_packed = nn.utils.rnn.pack_padded_sequence(src_emb, src_lengths)
        encoder_output, (enc_h, enc_c) = self.encoder(src_packed)
        encoder_output, _ = nn.utils.rnn.pad_packed_sequence(encoder_output)
        seqlen, bsz = src_tokens.size()

        def combine_bidir(outs):
            out = outs.view(2, 2, bsz, -1).transpose(1, 2).contiguous()
            return out.view(2, bsz, -1)

        return encoder_output, combine_bidir(enc_h), combine_bidir(enc_c)


class AttentionLayer(nn.Module):
    def __init__(self, output_embed_dim, bias=False):
        super().__init__()
        self.output_proj = nn.Linear(2 * output_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim
        x = input
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)

        attn_scores = torch.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class Decoder(nn.Module):
    def __init__(self, inp_emb_dim, hid_dim, out_dim, out_voc):
        super().__init__()
        self.decoder = StackedLSTMCell(inp_emb_dim + hid_dim, hid_dim)
        self.output_emb = nn.Embedding(len(out_voc.vocab), inp_emb_dim,
                                       padding_idx=out_voc.vocab.stoi[out_voc.pad_token])
        self.attn = AttentionLayer(hid_dim)
        self.pred_proj = nn.Linear(hid_dim, out_dim)
        self.hid_dim = hid_dim

    def forward(self, enc_out, enc_hid, enc_memory, encoder_padding_mask, dst):
        dst_emb = self.output_emb(dst)
        seqlen, bsz = dst.size()
        inp_feed = dst_emb.new_zeros(bsz, self.hid_dim)
        decoder_hidden = ([enc_hid[i] for i in range(2)], [enc_memory[i] for i in range(2)])
        decoder_outputs = []
        for step, cur_emb in enumerate(dst_emb):
            rnn_input = torch.cat((cur_emb, inp_feed), dim=1)
            output, decoder_hidden = self.decoder(rnn_input, decoder_hidden)
            out, attn_scores = self.attn(output, enc_out, encoder_padding_mask)
            decoder_outputs.append(out)
            inp_feed = out
        res = torch.stack(decoder_outputs).transpose(1, 0)
        res = self.pred_proj(res)
        return res


class Model(nn.Module):
    def __init__(self, hid_dim, inp_emb_dim, out_dim, inp_voc, out_voc):
        super().__init__()
        self.encoder = Encoder(hid_dim, inp_emb_dim, inp_voc)
        self.decoder = Decoder(inp_emb_dim, hid_dim, out_dim, out_voc)
        self.pad_idx = inp_voc.vocab.stoi[inp_voc.pad_token]

    def forward(self, src_tokens, src_lengths, dst):
        src_tokens = src_tokens.transpose(0, 1)
        dst = dst.transpose(0, 1)
        encoder_output, enc_h, enc_c = self.encoder(src_tokens, src_lengths)
        enc_mask = src_tokens.eq(self.pad_idx)
        output = self.decoder(encoder_output, enc_h, enc_c, enc_mask, dst)
        return output