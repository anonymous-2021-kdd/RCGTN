''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from entmax import sparsemax, entmax15, entmax_bisect

def get_subsequent_mask(x):
    ''' For masking out the subsequent info. '''
    # (num_samples, num_nodes, input_length, d_model)
    num_samples, num_nodes, input_len, d_model = x.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((num_samples, num_nodes, input_len, input_len), device=x.device), diagonal=1)).bool()
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)


    def forward(self, q, k, v, mask=None):
        # b x n_nodes x n_head x lq x dk

        attn = torch.matmul(q / self.temperature, k.transpose(3, 4)) #b x n_nodes x n_head x lq x lq

        if mask is not None:
            # print('attention:',attn.size())
            # print('mask:',mask.size())
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1)) # b x n_nodes x n_head x lq x lq
        attn = torch.matmul(attn, v) # b x n_nodes x n_head x lq x dv

        return attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k=16, d_v=16, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False) #(num_samples, num_nodes, input_length, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False) #(num_samples, num_nodes, input_length, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False) #(num_samples, num_nodes, input_length, n_head * d_k)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5,attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        #dim of q,k,v : (num_samples, num_nodes, input_length, d_model)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, num_nodes, len_q, len_k, len_v = q.size(0), q.size(1), q.size(2), k.size(2), v.size(2)

        residual = q

        # Pass through the pre-attention projection: b x N * lq x (n*dv)
        # Separate different heads: (batch_size, num_nodes, input_length, n_head , d_k)
        q = self.w_qs(q).view(sz_b, num_nodes, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, num_nodes, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, num_nodes, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n_nodes x n_head x lq x dv
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)

        if mask is not None:
            mask = mask.unsqueeze(2)   # For head axis broadcasting.

        # dim of q:  b x n_nodes x n_head x lq x dv
        # dim of attn: b x n_nodes x n_head x lq x lq
        q = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x n_nodes x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x n_nodes x lq x (n*dv)
        q = q.transpose(2, 3).contiguous().view(sz_b, num_nodes, len_q, -1)
        q = self.dropout(self.fc(q)) # b x n_nodes x lq x d_model
        q = q + residual

        q = self.layer_norm(q)


        return q


class LocalRangeConvAttention(nn.Module):
    def __init__(self,d_model,dropout=0.3):
        super().__init__()

        ''' Causal Convolutions '''
        self.conv_1 = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=1,stride=1,padding=0)
        self.conv_2 = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=2,stride=1,padding=1)
        self.conv_3 = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=3,stride=1,padding=2)
        self.conv_4 = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=4,stride=1,padding=3)


        # self.slf_attn = MultiHeadAttention_spa(n_head, d_model, d_k, d_v, dropout=dropout)
        self.attn_conv1 = MultiHeadAttention(2, d_model, d_model, d_model, dropout=dropout)
        self.attn_conv2 = MultiHeadAttention(2, d_model, d_model, d_model, dropout=dropout)
        self.attn_conv3 = MultiHeadAttention(2, d_model, d_model, d_model, dropout=dropout)
        self.attn_conv4 = MultiHeadAttention(2, d_model, d_model, d_model, dropout=dropout)

        self.fc = nn.Linear(in_features=4*d_model,out_features=d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self,x,cross_input,mask=None,cross = False):

        residual = x
        # x :(num_samples, num_nodes, input_length, d_model)
        num_samples, num_nodes, input_length, d_model = x.shape

        ''' Causal Convolutions '''

        x = x.reshape(-1,input_length, d_model)
        x = x.permute(0,2,1)    # b*n, d_model, input_len
        conv1 = self.conv_1(x).permute(0,2,1)[:,:input_length,:]
        # print('conv1',conv1.shape)
        conv1 = conv1.reshape(num_samples, num_nodes, input_length,-1)   # b*n, 1, input_len -> b*n, input_len, 1 -> b, n , input_Len, 1
        conv2 = self.conv_2(x).permute(0,2,1)[:,:input_length,:]
        # print('conv2',conv2.shape)
        conv2 = conv2.reshape(num_samples, num_nodes, int(input_length),-1)   # b*n, 1, input_len/2
        conv3 = self.conv_3(x).permute(0,2,1)[:,:input_length,:]
        # print('conv3',conv3.shape)
        conv3 = conv3.reshape(num_samples, num_nodes, int(input_length),-1)   # b*n, 1, input_len/3
        conv4 = self.conv_4(x).permute(0,2,1)[:,:input_length,:]
        conv4 = conv4.reshape(num_samples, num_nodes, int(input_length),-1)   # b*n, 1, input_len/4

        if cross:
            conv1 = self.attn_conv1(conv1,cross_input,cross_input,mask) # b*n, input_len, 1
            conv2 = self.attn_conv2(conv2,cross_input,cross_input,mask)
            conv3 = self.attn_conv3(conv3,cross_input,cross_input,mask)
            conv4 = self.attn_conv4(conv4,cross_input,cross_input,mask)
        else:
            conv1 = self.attn_conv1(conv1, conv1, conv1,mask)  # b*n, input_len, 1
            conv2 = self.attn_conv2(conv2, conv2, conv2,mask)
            conv3 = self.attn_conv3(conv3, conv3, conv3,mask)
            conv4 = self.attn_conv4(conv4, conv4, conv4,mask)


        cat_conv = torch.cat([conv1,conv2,conv3,conv4],dim = -1) # b*n, input_len, 4


        cat_conv = self.fc(cat_conv)   # b*n, input_len, d_model
        cat_conv = cat_conv.reshape(num_samples, num_nodes, input_length,-1)

        # cat_conv = self.dropout(cat_conv)
        cat_conv = cat_conv + residual
        cat_conv = F.relu(cat_conv)
        cat_conv = self.layer_norm(cat_conv)

        return cat_conv


class GroupRangeConvAttention(nn.Module):
    def __init__(self,d_model,num_nodes,range_size,dropout=0.3):
        super().__init__()

        self.range_size = range_size
        self.pd_size = (range_size - num_nodes % range_size) if num_nodes % range_size > 0 else 0
        self.range_conv1 = nn.Conv1d(in_channels=d_model,out_channels=d_model,
                                    kernel_size=self.range_size,stride=self.range_size,
                                    padding = self.pd_size)
        self.range_conv2 = nn.Conv1d(in_channels=d_model,out_channels=d_model,
                                    kernel_size=self.range_size,stride=self.range_size,
                                     padding=self.pd_size)
        self.range_conv3 = nn.Conv1d(in_channels=d_model,out_channels=d_model,
                                    kernel_size=self.range_size,stride=self.range_size,
                                    padding = self.pd_size)
        self.range_conv4 = nn.Conv1d(in_channels=d_model,out_channels=d_model,
                                    kernel_size=self.range_size,stride=self.range_size,
                                    padding = self.pd_size)

        # self.slf_attn = MultiHeadAttention_spa(n_head, d_model, d_k, d_v, dropout=dropout)
        self.attn_conv1 = MultiHeadAttention(2, d_model, d_model, d_model, dropout=dropout)
        self.attn_conv2 = MultiHeadAttention(2, d_model, d_model, d_model, dropout=dropout)
        self.attn_conv3 = MultiHeadAttention(2, d_model, d_model, d_model, dropout=dropout)
        self.attn_conv4 = MultiHeadAttention(2, d_model, d_model, d_model, dropout=dropout)

        self.fc = nn.Linear(in_features=4*d_model,out_features=d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self,x,cross_input,cross =False):
        residual = x

        perm2 = torch.randperm(x.size(2))
        sort,index2 = torch.sort(perm2)
        perm3 = torch.randperm(x.size(2))
        sort,index3 = torch.sort(perm3)
        perm4 = torch.randperm(x.size(2))
        sort,index4 = torch.sort(perm4)

        # x :(num_samples, input_length, num_nodes, d_model)
        num_samples, input_length, num_nodes, d_model = x.shape

        n_ranges = int(math.ceil((num_nodes/self.range_size)))

        x = x.reshape(-1,num_nodes,d_model)
        x = x.permute(0,2,1)  #b*T, d_model, num_nodes

        conv1 = self.range_conv1(x).permute(0,2,1)
        # print('conv1',conv1.size())

        conv1 =  conv1.reshape(num_samples,input_length,
                                                           n_ranges,-1)
        conv2 = self.range_conv2(x[:,:,perm2]).permute(0, 2, 1).reshape(num_samples,input_length,
                                                           n_ranges,-1)
        conv3 = self.range_conv3(x[:,:,perm3]).permute(0, 2, 1).reshape(num_samples,input_length,
                                                           n_ranges,-1)
        conv4 = self.range_conv4(x[:,:,perm4]).permute(0, 2, 1).reshape(num_samples,input_length,
                                                           n_ranges,-1)


        if cross:
            # print('con1:',conv1.size())
            # print('cross_input:',cross_input.size())
            conv1 = self.attn_conv1(conv1,cross_input,cross_input) # b , T, n_ranges, d_model
            conv2 = self.attn_conv1(conv2,cross_input,cross_input)
            conv3 = self.attn_conv1(conv3,cross_input,cross_input)
            conv4 = self.attn_conv1(conv4,cross_input,cross_input)
        else:
            conv1 = self.attn_conv1(conv1, conv1, conv1)  # b , T, N/range_size, d_model
            conv2 = self.attn_conv1(conv2, conv2, conv2)
            conv3 = self.attn_conv1(conv3, conv3, conv3)
            conv4 = self.attn_conv1(conv4, conv4, conv4)

        conv1 = conv1.repeat_interleave(self.range_size,dim=2)[:,:,:num_nodes,:]  # b , T, N, d_model
        conv2 = conv2.repeat_interleave(self.range_size,dim=2)[:,:,:num_nodes,:]
        conv3 = conv3.repeat_interleave(self.range_size,dim=2)[:,:,:num_nodes,:]
        conv4 = conv4.repeat_interleave(self.range_size,dim=2)[:,:,:num_nodes,:]

        conv2 = conv2[:,:,index2,:]
        conv3 = conv3[:,:,index3,:]
        conv4 = conv4[:,:,index4,:]


        cat_conv = torch.cat([conv1,conv2,conv3,conv4],dim = -1) # b , T, N/range_size, 4*d_model
        # print('cat_conv',cat_conv.size())

        cat_conv = self.fc(cat_conv)   # b , T, N/range_size, d_model


        cat_conv = cat_conv + residual
        cat_conv = F.relu(cat_conv)
        cat_conv = self.layer_norm(cat_conv)

        return cat_conv


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, seq_len, n_position=50):
        super(PositionalEncoding, self).__init__()

        self.seq_len = seq_len

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, in_decode = False ):
        #dim of x: (batch_size, seq_len, d_model)

        if in_decode:
            return x + self.pos_table[:, self.seq_len:(self.seq_len+x.size(1))].clone().detach()
        else:
            return x + self.pos_table[:, :x.size(1)].clone().detach()

