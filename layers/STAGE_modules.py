import torch
import torch.nn as nn

class temporal_embedding_4(nn.Module):
    def __init__(self, patch_num, seg_len, stride, d_model, num_nodes, freq):
        super(temporal_embedding_4, self).__init__()
        self.seg_len = seg_len
        self.stride = stride
        self.linear = nn.Linear(seg_len, d_model)  # 3->d_model

        # Dynamically calculate num_timestamps based on freq
        num_timestamps = self.calculate_num_timestamps(freq)
        self.tod_embedding = nn.Embedding(num_timestamps, d_model)
        dow_size = 7
        self.dow_embedding = nn.Embedding(dow_size, d_model)
        self.dropout = nn.Dropout(0.2)

        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(patch_num, num_nodes, d_model))
        )  # 6 N d_model

    @staticmethod  # cal L2
    def calculate_num_timestamps(freq):
        """ Calculate number of timestamps based on the frequency string. """
        if 'min' in freq:
            minutes_per_day = 24 * 60
            interval = int(freq.replace('min', ''))
            return minutes_per_day // interval
        elif 'h' in freq:
            hours_per_day = 24
            interval = int(freq.replace('h', ''))
            return hours_per_day // interval
        else:
            raise ValueError(f"Unsupported frequency: {freq}")

    def forward(self, x, x_tem):
        Batch = x.shape[0]
        x_patch = x.unfold(dimension=1, size=self.seg_len, step=self.stride)  # B Patch_num N Patch_len
        x_embed = self.linear(x_patch).permute(0, 2, 1, 3)

        # Use TimestampsOfDay embedding instead of hod_embedding and moh_embedding
        tod_emb = self.tod_embedding(x_tem[:, :, :, 0].long())
        x_embed += tod_emb.unfold(dimension=2, size=self.seg_len, step=self.stride).mean(-1)

        dow_emb = self.dow_embedding(x_tem[:, :, :, 1].long())
        x_embed += dow_emb.unfold(dimension=2, size=self.seg_len, step=self.stride).mean(-1)

        adp_emb = self.adaptive_embedding.expand(size=(Batch, *self.adaptive_embedding.shape)).permute(0, 2, 1, 3)
        x_embed += adp_emb
        x_embed = self.dropout(x_embed)
        return x_embed


class Encoder(nn.Module):
    def __init__(self, num_blocks, d_model, n_heads, d_ff, dropout, num_nodes,
                 patch_num, enable_Inter_GrAG, enable_Intra_GrAG):
        super(Encoder, self).__init__()

        self.encode_blocks = nn.ModuleList(
            [STAGE(patch_num, enable_Inter_GrAG, enable_Intra_GrAG, d_model, n_heads,
                                     num_nodes, d_ff, dropout) for _ in range(num_blocks)])
        self.gate_ln = GateLayer(dim=d_model)

    def forward(self, x, training=True):
        # x: B N P D
        for block in self.encode_blocks:
            x, dy_A_Tem, dy_A_Spa = block(x, training=training)
        out = self.gate_ln(x)
        return out, dy_A_Tem, dy_A_Spa


class GateLayer(nn.Module):
    def __init__(self, dim, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        gate_value = self.gate(x)
        return gate_value.sigmoid() * x


class STAGE(nn.Module):
    def __init__(self, patch_num, enable_Inter_GrAG, enable_Intra_GrAG, d_model, n_heads, num_nodes, d_ff, dropout):
        super(STAGE, self).__init__()
        print(f'enable_Intra_GrAG:{enable_Intra_GrAG},enable_Inter_GrAG:{enable_Inter_GrAG}')

        self.node_embedding_dim = 12
        # Intra-series GCN
        self.enable_Intra_GrAG = enable_Intra_GrAG
        self.Intra_GrAG = Intra_GrAG(d_model, num_nodes, dropout, patch_num, node_embedding=self.node_embedding_dim,
                                   enable_Intra_GrAG=self.enable_Intra_GrAG)
        # Inter-series GCN
        self.enable_Inter_GrAG = enable_Inter_GrAG
        self.Inter_GrAG = Inter_GrAG(d_model, patch_num, dropout, num_nodes, node_embedding=self.node_embedding_dim,
                                   enable_Inter_GrAG=self.enable_Inter_GrAG)

        self.dropout = nn.Dropout(dropout)
        self.gate_ln = GateLayer(dim=d_model)

    def forward(self, x, training=True):
        batch = x.shape[0]
        num_nodes = x.shape[1]
        seg_num = x.shape[2]
        d_model = x.shape[3]

        # GraphSTAGE
        dim_in = x
        dy_A_Tem = 1
        dy_A_Spa = 1
        if self.enable_Intra_GrAG:
            dim_in, dy_A_Tem, e1, e2 = self.Intra_GrAG(dim_in.transpose(1, 2).contiguous(), training=training)
            dim_in = self.gate_ln(dim_in)

        if self.enable_Inter_GrAG:
            dim_in, dy_A_Spa, e1, e2 = self.Inter_GrAG(dim_in, training=training)
            dim_in = self.gate_ln(dim_in)

        return dim_in, dy_A_Tem, dy_A_Spa

class Intra_GrAG(nn.Module):
    def __init__(self, d_model_gcn, num_nodes, dropout, patch_num, node_embedding, enable_Intra_GrAG):
        nn.Module.__init__(self)
        self.d_model_gcn = d_model_gcn
        self.dyA_G = Learn_A_Generator(d_model_gcn, 'mean', node_embedding, dropout)
        self.dff = int(d_model_gcn / 2)
        self.Pruned_GrAG = Pruned_GrAG(d_model_gcn, self.dff, num_nodes)
        self.enable_Intra_GrAG = enable_Intra_GrAG

        self.FFN = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.dff, d_model_gcn),
            nn.ReLU(),
            nn.Linear(d_model_gcn, d_model_gcn),
        )

        self.FFN2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model_gcn, int(d_model_gcn * 0.5)),
            nn.ReLU(),
            nn.Linear(int(d_model_gcn * 0.5), d_model_gcn),
        )

        self.LN = nn.LayerNorm(normalized_shape=[patch_num, d_model_gcn], elementwise_affine=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training=True):
        # x, dim_send: B N P D

        batch_size = x.shape[0]
        num_nodes = x.shape[2]
        time_len = x.shape[1]

        sublayer1_out = x
        dy_A, e1, e2 = self.dyA_G(sublayer1_out, training=training)

        sublayer1_out = sublayer1_out.transpose(1, 2).contiguous()
        sublayer1_out = sublayer1_out.view(batch_size * num_nodes, time_len, -1)
        # sublayer1_out B*P N D

        if self.enable_Intra_GrAG:
            dy_gcn_out = self.Pruned_GrAG(dy_A, sublayer1_out)
            ffn_out = self.FFN(dy_gcn_out)
        else:
            ffn_out = self.FFN2(sublayer1_out)

        sublayer2_out = self.dropout(ffn_out) + sublayer1_out
        sublayer2_out = sublayer2_out.view(batch_size, num_nodes, time_len, -1)
        sublayer2_out = self.LN(sublayer2_out)
        sublayer2_out = sublayer2_out + self.dropout(self.FFN2(sublayer2_out))
        sublayer2_out = self.LN(sublayer2_out)

        return sublayer2_out, dy_A, e1, e2


class Inter_GrAG(nn.Module):
    def __init__(self, d_model_gcn, seg_num_gcn, dropout, num_nodes, node_embedding, enable_Inter_GrAG):
        nn.Module.__init__(self)
        self.d_model_gcn = d_model_gcn
        self.dyA_G = Learn_A_Generator(d_model_gcn, 'mean', node_embedding, dropout)
        self.dff = int(d_model_gcn / 2)
        self.Pruned_GrAG = Pruned_GrAG(d_model_gcn, self.dff, seg_num_gcn)
        self.enable_Inter_GrAG = enable_Inter_GrAG

        self.FFN = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.dff, d_model_gcn),
            nn.ReLU(),
            nn.Linear(d_model_gcn, d_model_gcn),
        )

        self.FFN2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model_gcn, int(d_model_gcn * 0.5)),
            nn.ReLU(),
            nn.Linear(int(d_model_gcn * 0.5), d_model_gcn),
        )

        self.LN = nn.LayerNorm(normalized_shape=[num_nodes, d_model_gcn], elementwise_affine=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training=True):
        # x, dim_send: B N P D

        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        time_len = x.shape[2]

        sublayer1_out = x
        dy_A, e1, e2 = self.dyA_G(sublayer1_out, training=training)

        sublayer1_out = sublayer1_out.transpose(1, 2).contiguous()
        sublayer1_out = sublayer1_out.view(batch_size * time_len, num_nodes, -1)
        # sublayer1_out B*P N D

        if self.enable_Inter_GrAG:
            dy_gcn_out = self.Pruned_GrAG(dy_A, sublayer1_out)
            ffn_out = self.FFN(dy_gcn_out)
        else:
            ffn_out = self.FFN2(sublayer1_out)

        sublayer2_out = self.dropout(ffn_out) + sublayer1_out
        sublayer2_out = sublayer2_out.view(batch_size, time_len, num_nodes, -1)
        sublayer2_out = self.LN(sublayer2_out)
        sublayer2_out = sublayer2_out + self.dropout(self.FFN2(sublayer2_out))
        sublayer2_out = self.LN(sublayer2_out)
        sublayer2_out = sublayer2_out.transpose(1, 2).contiguous()

        return sublayer2_out, dy_A, e1, e2


class Learn_A_Generator(nn.Module):
    def __init__(self, out_channels, pooling='sum', node_embedding_dim=12, dropout=0.0, ):
        nn.Module.__init__(self)
        self.e1_ln = nn.Linear(out_channels, node_embedding_dim, bias=False)
        self.e2_ln = nn.Linear(out_channels, node_embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.pooling = pooling
        nn.init.kaiming_uniform_(self.e1_ln.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.e2_ln.weight, nonlinearity='relu')

    @staticmethod  # cal L2 |emb|
    def mapEmbedding(emb):
        e_len = torch.sqrt(torch.sum(emb ** 2, dim=-1, keepdim=True))
        emb = emb / (e_len + 1e-3)  # avoid divide zero
        return emb

    @staticmethod
    def Prune(A):
        # A: B x N x N
        B, N, _ = A.shape
        k = int(N*0.8)
        topk_values, topk_indices = torch.topk(A, k, dim=-1)
        mask = torch.zeros_like(A)
        mask.scatter_(-1, topk_indices, topk_values)
        return mask

    def forward(self, x, training=True):
        # x: B N P D
        if self.pooling == 'mean':
            x = x.mean(2)
        elif self.pooling == 'sum':
            x = x.sum(2)
        else:
            x = x[:, :, -1, :]

        e1 = self.mapEmbedding(self.relu(self.e1_ln(x)))
        e2 = self.mapEmbedding(self.relu(self.e2_ln(x)))

        if training:
            A = torch.matmul(e1, e2.transpose(1, 2))
            A = self.relu(A)
            A = self.dropout(A)
        else:
            A = torch.matmul(e1, e2.transpose(1, 2))
            A = self.relu(A)
        # return self.Prune(A), e1, e2
        return A, e1, e2

class Pruned_GrAG(nn.Module):
    def __init__(self, d_model_gcn, out_channels, seg_num_gcn):
        nn.Module.__init__(self)

        self.weights = nn.Parameter(torch.zeros(3, 1, d_model_gcn, out_channels))
        self.seg_num_gcn = seg_num_gcn
        nn.init.xavier_normal_(self.weights)  # from zeros to Xavier matrix

    def forward(self, A, x):
        """
        A: B N N
        x: B*P N D
        return out: B*P N F
        """
        x = x.view(A.shape[0], -1, x.shape[1], x.shape[2])  # B N P D

        A_1 = (A / (A.sum(dim=2, keepdims=True) + 1e-3)).unsqueeze(1)  # A
        A_2 = A.transpose(1, 2).contiguous()
        A_2 = (A_2 / (A_2.sum(dim=2, keepdims=True) + 1e-3)).unsqueeze(1)  # A^T
        #  A_1: B 1 N N; x: B P N D     torch.matmul(A_1, x) = broadcasting(A_1)*x = B P N D
        out = torch.stack([x, torch.matmul(A_1, x), torch.matmul(A_2, x)], dim=2)  # B P 3 N D

        weights = torch.repeat_interleave(self.weights, self.seg_num_gcn, dim=1)
        out = torch.einsum('bpknd, kpdf-> bpknf', (out, weights))
        out = out.sum(dim=2)
        out = out.view(-1, out.shape[2], out.shape[3])
        return out