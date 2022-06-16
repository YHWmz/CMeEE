from collections import defaultdict
import pdb
from typing import Set
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax

def get_pos_encoding(embed_size, seq_len):
    """生成的embedding是相对位置的embedding
        由于相对位置有正有负，则Encoding的大小也应该为(-seq_len,seq_len)
        若使用绝对位置，则Encoding的大小为(0,seq_len)
    """
    assert embed_size % 2 == 0
    num_embeddings = 2*seq_len+1
    half_dim = embed_size // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)

    #emb = torch.FloatTensor(2/embed_size * np.log(10000) * np.arange(0, embed_size // 2))
    # 在这里可以设置是pos是要从(0,2*seq_len)还是(-seq_len,seq_len)
    emb = torch.arange(-seq_len,seq_len+1, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)

    # sin_emb, cos_emb = torch.sin(emb).unsqueeze(2), torch.cos(emb).unsqueeze(2)
    # emb = torch.cat([sin_emb, cos_emb],2).view(num_embeddings, -1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    return emb


class Multihead_Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_dropout) -> None:
        super(Multihead_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.dropout = attn_dropout

        if hidden_size % num_heads:
            RuntimeError(f"hidden size {hidden_size} 应该要能被 num_heads {num_heads} 除尽")
        self.per_size = self.hidden_size // self.num_heads

        self.k_mat = nn.Linear(self.hidden_size, self.hidden_size)
        self.q_mat = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_mat = nn.Linear(self.hidden_size, self.hidden_size)
        self.r_mat = nn.Linear(self.hidden_size, self.hidden_size)

        self.u = nn.Parameter(torch.rand((self.num_heads, self.per_size)))
        self.v = nn.Parameter(torch.rand((self.num_heads, self.per_size)))

        self.final = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(attn_dropout)


    def forward(self, feature, rel_pos : torch, mask):
        """相对位置可以在一开始就计算好，不用多次重复计算
            mask也可以
        """
        batch_size, seq_len, _ = feature.shape

        key = self.k_mat(feature).reshape((batch_size, seq_len, self.num_heads, self.per_size))
        query = self.q_mat(feature).reshape((batch_size, seq_len, self.num_heads, self.per_size))
        value = self.v_mat(feature).reshape((batch_size, seq_len, self.num_heads, self.per_size))

        rel_pos = self.r_mat(rel_pos).reshape((batch_size, seq_len, seq_len, self.num_heads, self.per_size))
        
        # # N x H x P x S 
        # key = key.permute((0, 2, 3, 1))
        # query = query.transpose(1, 2)
        # value = value.transpose(1, 2)
        
        # rel_pos = rel_pos.permute((0, 3, 1, 4, 2))

        # # N x H x S x P
        # query_u = self.u + query

        # # N X H x S x 1 x S
        # query_v = self.v + query.unsqueeze(-2)
        # #pdb.set_trace()
        
        # # N x H X S X S
        # A = torch.matmul(query_u, key) + torch.matmul(query_v, rel_pos).squeeze(-2)

        # A = A  / np.sqrt(self.per_size)

        # attn_score_raw_masked = A.masked_fill(~mask, -1e15)
        # attn_score = self.dropout(softmax(attn_score_raw_masked, -1))

        # result = torch.matmul(attn_score, value).reshape((batch_size, seq_len, self.hidden_size))
        # result = self.final(result)
        # #pdb.set_trace()
        # return result

         # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)



        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)
        # #A
        # A_ = torch.matmul(query,key)
        # #C
        # # key: batch * n_head * d_head * key_len
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        # u_for_c: 1(batch broadcast) * num_heads * 1 *per_head_size
        # key_for_c = key
        # C_ = torch.matmul(u_for_c, key)
        query_and_u_for_c = query + u_for_c
        
        A_C = torch.matmul(query_and_u_for_c, key)

        #B
        rel_pos_embedding_for_b = rel_pos.permute(0, 3, 1, 4, 2)
        # after above, rel_pos_embedding: batch * num_head * query_len * per_head_size * key_len
        query_for_b = query.view([batch_size, self.num_heads, seq_len, 1, self.per_size])
        # after above, query_for_b: batch * num_head * query_len * 1 * per_head_size
        # print('query for b:{}'.format(query_for_b.size()))
        # print('rel_pos_embedding_for_b{}'.format(rel_pos_embedding_for_b.size()))
        # B_ = torch.matmul(query_for_b,rel_pos_embedding_for_b).squeeze(-2)

        #D
        # rel_pos_embedding_for_d = rel_pos_embedding.unsqueeze(-2)
        # after above, rel_pos_embedding: batch * query_seq_len * key_seq_len * num_heads * 1 *per_head_size
        # v_for_d = self.v.unsqueeze(-1)
        # v_for_d: num_heads * per_head_size * 1
        # D_ = torch.matmul(rel_pos_embedding_for_d,v_for_d).squeeze(-1).squeeze(-1).permute(0,3,1,2)

        query_for_b_and_v_for_d = query_for_b + self.v.view(1,self.num_heads,1,1,self.per_size)
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)
        #att_score: Batch * num_heads * query_len * key_len
        # A, B C and D is exactly the shape
        
            # print_info('D:{}'.format(D_.size()))
        attn_score_raw = A_C + B_D

        
        attn_score_raw  = attn_score_raw / np.sqrt(self.per_size)

        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)

        attn_score = softmax(attn_score_raw_masked,dim=-1)

        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1,2).contiguous(). \
            reshape(batch_size, seq_len, self.hidden_size)


        
        result = self.final(result)

        return result

        
class Encoder_Layer(nn.Module):
    def __init__(self, hidden_size, ff_size, num_heads, dropout : dict):
        super(Encoder_Layer, self).__init__()

        self.attn_layer = Multihead_Attention(hidden_size, num_heads, dropout['attn'])
        self.dropout1 = nn.Dropout(dropout['res_1'])
        self.norm1 = nn.LayerNorm(hidden_size)


        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.Dropout(dropout['ff_1']),
            nn.ReLU(),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout['ff_2']),
        )
        self.dropout2 = nn.Dropout(dropout['res_2'])
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, inp, rel_pos : torch, mask):
        x = inp
        inp = self.attn_layer.forward(inp, rel_pos, mask)
        inp = self.norm1(x + self.dropout1(inp))

        x = inp
        inp = self.norm2(self.dropout2(self.FFN(inp)) + x)
        return inp


class Transformer_Encoder(nn.Module):
    def __init__(self, hidden_size, ff_size, num_layers, num_heads, max_len, shared_pos_encoding = True, dropout = defaultdict(int), pos_norm = True):
        super(Transformer_Encoder, self).__init__()

        self.hidden_size = hidden_size 
        self.ff_size = ff_size

        self.dropout = dropout
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len # 输入句子的长度的最长的一个

        self.pos_norm = pos_norm

        # 按照原论文直接使用相对位置的embedding
        pe = get_pos_encoding(self.hidden_size, self.max_len)
        pe_sum = pe.sum(dim=-1,keepdim=True)
        if self.pos_norm:
            with torch.no_grad():
                pe = pe/pe_sum

        self.share_pos_encoding = shared_pos_encoding

        if shared_pos_encoding:
            # 由于在论文中这几个position encoding是可以训练权重的，所以可以通过设置是否共享权重来调整训练方式
            self.pe = nn.Embedding(2 * self.max_len + 1, self.hidden_size, _weight = pe)
        else:
            self.s2s_pe = nn.Embedding(2 * self.max_len + 1, self.hidden_size, _weight = pe.detach())
            self.s2e_pe = nn.Embedding(2 * self.max_len + 1, self.hidden_size, _weight = pe.detach())
            self.e2s_pe = nn.Embedding(2 * self.max_len + 1, self.hidden_size, _weight = pe.detach())
            self.e2e_pe = nn.Embedding(2 * self.max_len + 1, self.hidden_size, _weight = pe.detach())

        self.rel_fusion = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
                nn.ReLU(0.1),
                nn.Linear(self.hidden_size * 4 , self.hidden_size),
        )

        self.layers = nn.ModuleDict()

        for i in range(self.num_layers):
            self.layers[f'layer_{i}'] = Encoder_Layer(self.hidden_size, self.ff_size, self.num_heads, dropout)

    def get_rel_fusion(self, pos_s, pos_e, max_len):
        pos_ss = pos_s.unsqueeze(-1)-pos_s.unsqueeze(-2) + self.max_len
        pos_se = pos_s.unsqueeze(-1)-pos_e.unsqueeze(-2) + self.max_len
        pos_es = pos_e.unsqueeze(-1)-pos_s.unsqueeze(-2) + self.max_len
        pos_ee = pos_e.unsqueeze(-1)-pos_e.unsqueeze(-2) + self.max_len


        pe_ss = pos_ss.view(size=[-1, max_len, max_len, 1])
        pe_se = pos_se.view(size=[-1, max_len, max_len, 1])
        pe_es = pos_es.view(size=[-1, max_len, max_len, 1])
        pe_ee = pos_ee.view(size=[-1, max_len, max_len, 1])

        if self.share_pos_encoding:
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            pe_4 = pe_4.view(size=[-1, 4])
            pe_unique, inverse_indices = torch.unique(pe_4, sorted=True, return_inverse=True, dim=0)
            
            pos_unique_embedding = self.pe(pe_unique)
            pos_unique_embedding = pos_unique_embedding.view([pos_unique_embedding.size(0), -1])
            
            pos_unique_embedding_after_fusion = self.rel_fusion(pos_unique_embedding)
            
            rel_pos = pos_unique_embedding_after_fusion[inverse_indices]
            return rel_pos.view(size=[-1,max_len, max_len, self.hidden_size])

        else:
            pe_ss, pe_se, pe_es, pe_ee = self.s2s_pe(pe_ss), self.s2e_pe(pe_se), self.e2s_pe(pe_es), self.e2e_pe(pe_ee)
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            return self.rel_fusion(pe_4)

    def generate_mask(self, sen_len, max_len):
        batch_size = sen_len.shape[0]
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(sen_len)

        mask = broad_cast_seq_len.lt(sen_len.unsqueeze(1))
        return mask.unsqueeze(1).unsqueeze(1)

    def forward(self, inp, pos_s, pos_e, sen_len, lat_len):
        """input 是经过了embedding的
            pos_s, pos_e, sen_len, lat_len : LongTensor
        """
        max_len = inp.shape[1]
        rel_pos = self.get_rel_fusion(pos_s, pos_e, max_len)
        mask = self.generate_mask(sen_len + lat_len, max_len).requires_grad_(False)
        for i in range(self.num_layers):
            inp = self.layers[f'layer_{i}'].forward(inp, rel_pos, mask)
            
        """ TODO: 还有最后一个处理的layer，回去看
            Done:   默认是没有的
        """
        return inp

if __name__ == "__main__":
    inp = torch.rand((10, 18, 120))
    pos_s = torch.randint(0, 18, (10,18))
    pos_e = torch.randint(0, 18, (10,18))
    sen_len = torch.randint(0, 9, (10,))
    lat_len = torch.randint(0, 9, (10,))

    dropout = {
        'attn' : 0.2, # Attention 层的dropout
        'res_1' : 0.2, # residual 层的dropout
        'res_2' : 0.2, # 因为每个encode模块有两个残差链接
        'ff_1' : 0.2, # FFN层的dropout
        'ff_2' : 0.2, # FFN层的第二个dropout
    }
    model = Transformer_Encoder(120, 480, 3, 4, 20,dropout=dropout)

    print(model)
    print(next(model.parameters()).device)


    k = model(inp, pos_s, pos_e, sen_len, lat_len)
    print(k[0,:,:10])


                
