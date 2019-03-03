import torch
import torch.nn as nn


class StackedLSTMCell(nn.Module):
    def __init__(self, input_size, rnn_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(2):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_, hidden):
        h_0, c_0 = hidden
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_, (h_0[i], c_0[i]))
            input_ = self.dropout(h_1_i)
            h_0[i] = h_1_i
            c_0[i] = c_1_i
        return input_, (h_0, c_0)


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

        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)

        attn_scores = torch.softmax(attn_scores, dim=0)

        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        x = torch.tanh(self.output_proj(torch.cat((x, input_), dim=1)))
        return x, attn_scores


class Decoder(nn.Module):
    def __init__(self, inp_emb_dim, hid_dim, out_dim, out_voc, dropout):
        super().__init__()
        self.decoder = StackedLSTMCell(inp_emb_dim + hid_dim, hid_dim, dropout)
        self.output_emb = nn.Embedding(
            len(out_voc.vocab),
            inp_emb_dim,
            padding_idx=out_voc.vocab.stoi[out_voc.pad_token],
        )
        self.attn = AttentionLayer(hid_dim)
        self.pred_proj = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
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
        decoder_hidden = ([enc_h[i] for i in range(2)], [enc_c[i] for i in range(2)])
        decoder_outputs = []
        attention_scores = []
        for step in range(max_len):
            rnn_input = torch.cat((cur_emb, inp_feed), dim=1)
            output, decoder_hidden = self.decoder.decoder(rnn_input, decoder_hidden)
            out, attn_scores = self.decoder.attn(output, enc_out, enc_mask)
            inp_feed = out
            if loss_type is 'xent':
                pred_words = self.decoder.pred_proj(out).max(1)[1]
            else:
                scores = self._get_scores(self.decoder.pred_proj(out), loss_type, device=src_tokens.device)
                pred_words = scores.max(1)[1]
            cur_emb = self.decoder.output_emb(pred_words.to(src_tokens.device))
            decoder_outputs.append(pred_words)
            attention_scores.append(attn_scores)
        res = torch.stack(decoder_outputs).transpose(1, 0)
        attn = torch.stack(attention_scores).permute(2, 0, 1)  # seq_len x nsrc x bsz -> bsz x seq_len x nsrc
        return res, attn
    
    def _get_scores(self, output_vecs, loss_type, device):
        # bsz x dim, voc_size x dim -> bsz x voc_size
        # bsz x dim and dim x voc_size -> 
        if loss_type == 'l2':
            #voc_emb = self.out_voc.vocab.vectors
            #return ((output_vecs.unsqueeze(2).to(voc_emb.device) - voc_emb.transpose(0,1)) ** 2).sum(1)
            r_out = (output_vecs ** 2).sum(dim=1).unsqueeze(1)
            r_voc = (self.out_voc.vocab.vectors ** 2).sum(dim=1).unsqueeze(0).to(device)
            r_scal_prod = 2 * output_vecs.matmul(self.out_voc.vocab.vectors.t().to(device))
            return - r_out - r_voc + r_scal_prod
            
        elif loss_type == 'cosine':
            out_ves_norm = nn.functional.normalize(output_vecs, p=2, dim=1)
            voc_emb_norm = nn.functional.normalize(self.out_voc.vocab.vectors.to(device), dim=1)
            return output_vecs.matmul(self.out_voc.vocab.vectors.t())
        
