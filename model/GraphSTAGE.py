from layers.STAGE_modules import temporal_embedding_4, Encoder
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_nodes = configs.enc_in
        # num_nodes = 307
        self.in_len = configs.seq_len
        self.out_len = configs.pred_len
        self.seg_len = configs.patch_seg_len
        self.stride = configs.patch_stride_len
        self.enable_inter_GrAG = configs.enable_inter_GrAG
        self.enable_intra_GrAG = configs.enable_intra_GrAG
        self.baseline = configs.add_res
        self.d_model = configs.d_model
        self.d_ffn = int(configs.d_model / 2)
        self.n_heads = configs.n_heads
        self.num_blocks = configs.e_layers
        self.dropout = configs.dropout
        self.data = configs.data
        self.freq = configs.freq
        self.use_norm = configs.use_norm
        self.plot_graphs = configs.plot_graphes

    # padding
        self.padding = True
        self.pad_len = 0
        if (self.in_len - self.seg_len) % self.stride != 0:
            self.pad_len = self.stride - ((self.in_len - self.seg_len) % self.stride)
            self.padding = True
        self.patch_num = (self.in_len + self.pad_len - self.seg_len) // self.stride + 1

        # Embedding
        self.enc_value_embedding = temporal_embedding_4(self.patch_num, self.seg_len, self.stride, self.d_model, self.num_nodes, self.freq)

        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.num_nodes, self.patch_num, self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)

        # Encoder
        self.encoder = Encoder(self.num_blocks, self.d_model, self.n_heads, self.d_ffn,
                               self.dropout, self.num_nodes, self.patch_num, self.enable_inter_GrAG, self.enable_intra_GrAG)

        self.output_layers = nn.ModuleList()
        for i in range(self.num_blocks + 1):
            self.output_layers.append(nn.Linear(self.patch_num, self.out_len))

        self.ln = nn.Linear(self.patch_num * self.d_model, self.out_len)

        self.end = nn.Linear(self.d_model, 1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, training=True):
        if self.use_norm:
            # print(self.use_norm)
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        # x_enc B T N
        # x_mark_enc B T F (F=4 :tod dow dom doy)
        x_seq = x_enc
        time_seq = x_mark_enc.unsqueeze(1).repeat(1, self.num_nodes, 1, 1)
        # x_seq  B T N
        # time_seq B N T F

        # baseline
        if self.baseline:
            base = x_seq.mean(dim=1, keepdim=True).permute(0, 2, 1).unsqueeze(-1)
            # base B T N-> B 1 N -> B N 1 1
        else:
            base = 0

        # padding x_seq and time_seq before Patch
        if self.padding:
            x_seq = torch.cat((x_seq[:, [-1], :].expand(-1, self.pad_len, -1), x_seq), dim=1)
            time_seq = torch.cat((time_seq[:, :, [-1], :].expand(-1, -1, self.pad_len, -1), time_seq), dim=2)

        # embedding
        x_seq = self.enc_value_embedding(x_seq, time_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        # x_seq B N P D; P:patch_num D:d_model

        # encoder
        enc_out, dy_A_Tem, dy_A_Spa = self.encoder(x_seq, training)
        batch_size = enc_out.shape[0]
        num_nodes = enc_out.shape[1]

        enc_out = enc_out.view(batch_size, num_nodes, -1)
        # enc_out B N P*D
        output = self.ln(enc_out).view(batch_size, num_nodes, -1, 1)
        res = output + base
        res = res.view(batch_size, num_nodes, -1).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            res = res * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(
                          1, self.out_len, 1))
            res = res + \
                      (means[:, 0, :].unsqueeze(1).repeat(
                          1, self.out_len, 1))
        # res B Pred_len N
        return res, dy_A_Tem, dy_A_Spa

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, training=True):
        if self.plot_graphs:
            dec_out, dy_A_Tem, dy_A_Spa = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, training=training)
            return dec_out[:, -self.out_len:, :], dy_A_Tem, dy_A_Spa  # [B, L, D]
        else:
            dec_out, dy_A_Tem, dy_A_Spa = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.out_len:, :]# [B, L, D]
