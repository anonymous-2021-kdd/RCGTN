import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from sublayers import *


class TemporalEncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, dropout, if_ffn):
        super(TemporalEncoderLayer, self).__init__()

        self.if_ffn = if_ffn

        self.slf_attn = LocalRangeConvAttention(d_model,dropout=dropout) #contains add & layer_norm
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, x,slf_attn_mask=None):

        x = self.slf_attn(x,x,mask=slf_attn_mask,cross=False)

        if self.if_ffn:
            x = self.pos_ffn(x)
        else:
            x = F.relu(x)

        return x


class TemporalEncoder(nn.Module):

    def __init__(self, d_model, seq_len, n_layers, d_inner, dropout=0.3, if_ffn=False):
        super(TemporalEncoder, self).__init__()


        self.position_enc = PositionalEncoding(d_model, seq_len, n_position=100)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            TemporalEncoderLayer(d_model, d_inner, dropout, if_ffn)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, ):
        # (num_samples, num_nodes, input_length, d_model)
        num_samples, num_nodes, input_length, d_model = x.shape

        input_mask = get_subsequent_mask(x)

        # -- Forward
        enc_output = self.position_enc(x.reshape(-1,input_length, d_model),in_decode = False).reshape(num_samples, num_nodes, input_length, d_model)
        # enc_output = self.dropout(x)
        # enc_output = self.layer_norm(enc_output)

        enc_inputs_list = []
        enc_inputs_list.append(enc_output)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, input_mask)
            enc_inputs_list.append(enc_output)

        enc_inputs_list = enc_inputs_list[:-1]
        return enc_output,enc_inputs_list




class SpatialEncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner,num_nodes, range_size, dropout,if_ffn):
        super(SpatialEncoderLayer, self).__init__()

        self.if_ffn = if_ffn
        self.dropout = nn.Dropout(p=dropout)

        self.slf_attn = GroupRangeConvAttention(d_model, num_nodes,range_size,dropout=dropout)  # 里面包含了add & layer_norm
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, x):
        x = self.slf_attn(x,x,cross=False)

        if self.if_ffn:
            x = self.pos_ffn(x)
        else:
            x = F.relu(x)

        return x



class SpatialEncoder(nn.Module):
    def __init__(self, d_model, n_layers, d_inner,num_nodes,range_size, dropout=0.3,if_ffn =False):
        super(SpatialEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            SpatialEncoderLayer(d_model,d_inner,num_nodes,range_size, dropout,if_ffn)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x,):
        # (num_samples, num_nodes, input_length, d_model)

        enc_output = x.permute(0,2,1,3) # (num_samples, input_len, num_nodes, d_model)

        # enc_output = self.dropout(enc_output)
        # enc_output = self.layer_norm(enc_output) #(num_samples, num_nodes, input_len, d_model)

        for enc_layer in self.layer_stack:

            ## dim of attn: b x n_nodes x n_head x lq x lq
            enc_output = enc_layer(enc_output)

        enc_output = enc_output.permute(0,2,1,3) # num_sample, N, input_len,out_features

        return enc_output



class fusion_spatial_temporal(nn.Module):
    def __init__(self,d_model):
        super(fusion_spatial_temporal, self).__init__()

        self.fuse_conv = nn.Conv2d(in_channels=d_model*2,
                                    out_channels=d_model,
                                    kernel_size=(1, 1))

        self.fuse_linear = nn.Linear(in_features=d_model*2,out_features=d_model)

    def forward(self,temporal_enc_output,spatial_enc_output):
        # spa_enc_output: num_samples, num_nodes, input_length, d_model

        x = torch.cat((temporal_enc_output,spatial_enc_output),dim=-1) #(num_samples, num_nodes, input_length, 2*d_model)
        x = x.permute(0,3,1,2) # (num_samples, 2*d_model, num_nodes, input_length)
        x = self.fuse_conv(x).permute(0,2,3,1) # (num_samples, num_nodes, input_len, d_model)

        return x


class trend_controller(nn.Module):
    '''
    trend:   (num_samples, num_nodes, out_len,d_model)
    fuse_input: (num_samples, num_nodes, out_len,d_model)
    return: (num_samples, num_nodes, out_len,d_model)
    '''
    def __init__(self,d_model,seq_len):
        super(trend_controller, self).__init__()

        self.trends_conv = nn.Conv2d(in_channels=1,
                                    out_channels=d_model,
                                    kernel_size=(1, 1),bias=True)

        self.linear = nn.Linear(in_features=d_model,out_features=d_model)
        self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_((random.randint(1,5)/10))

        self.position_enc = PositionalEncoding(d_model, seq_len, n_position=100)


    def forward(self,trend,fuse_input):
        trend = torch.unsqueeze(trend, 1) # (num_samples, 1, num_nodes, output_length,)
        trend = self.trends_conv(trend).permute(0,2,3,1)  # (num_samples, num_nodes, out_len,d_model,)

        num_samples, num_nodes, out_len, d_model = trend.shape
        # consistent position encoding
        trend = self.position_enc(trend.reshape(-1, out_len, d_model), in_decode=True).reshape(num_samples, num_nodes, out_len,
                                                                                       d_model)

        fuse_input = torch.sigmoid(self.linear(fuse_input))
        fuse_input = -self.alpha * fuse_input

        return torch.multiply(trend,fuse_input)



class TemporalDecoderLayer(nn.Module):
    def __init__(
            self, d_model, d_inner, dropout=0.3,if_ffn=False):
        super(TemporalDecoderLayer, self).__init__()

        self.if_ffn = if_ffn
        self.tem_attn = LocalRangeConvAttention(d_model,dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,dec_input,enc_output,tem_inputs,slf_attn_mask=None):
        # dim of dec_input : (num_samples, num_nodes, out_len, d_model)
        # dim of tem_inputs : (num_samples, num_nodes, out_len, d_model)
        num_samples, num_nodes, out_len, d_model = dec_input.shape


        dec_input = torch.cat([tem_inputs,dec_input],dim=2)
        # temporal attention
        dec_out = self.tem_attn(
            dec_input, dec_input,mask=slf_attn_mask,cross=False)  #(num_samples, num_nodes, out_len, d_model)

        dec_out = dec_out[:,:,-out_len:,:]


        if self.if_ffn:
            dec_out = self.pos_ffn(dec_out)
        else:
            dec_out = F.relu(dec_out)

        return dec_out

class TemporalDecoder(nn.Module):
    def __init__(
            self,d_model, seq_len, n_layers,d_inner, dropout=0.3,if_ffn=False):

        super(TemporalDecoder, self).__init__()

        self.position_enc = PositionalEncoding(d_model, seq_len, n_position=100)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.layer_stack = nn.ModuleList([
            TemporalDecoderLayer(d_model, d_inner, dropout, if_ffn)
            for _ in range(n_layers)])

    def forward(self,x, fuse_enc_output,tem_inputs_list):
        # trend(x): (num_samples, num_nodes, out_len, d_model,)
        # fuse_end_output: num_samples, num_nodes, input_length, d_model

        input_mask = get_subsequent_mask(torch.cat([x,tem_inputs_list[0]],dim=2))

        for index,dec_layer in enumerate(self.layer_stack):
            x = dec_layer(x,fuse_enc_output,tem_inputs_list[index],input_mask)

        return x


class SpatialDecoderLayer(nn.Module):
    def __init__(
            self, d_model, d_inner, num_nodes, range_size, dropout=0.3, if_ffn=False):
        super(SpatialDecoderLayer, self).__init__()

        self.if_ffn = if_ffn

        self.spa_attn = GroupRangeConvAttention(d_model, num_nodes, range_size,dropout=dropout)
        self.enc_dec_spa_attn = GroupRangeConvAttention(d_model,num_nodes,range_size, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, dec_input, enc_output):
        # dim of dec_input : (num_samples, num_nodes, out_len, d_model)

        # spatial attention
        enc_output = enc_output.permute(0,2,1,3)
        dec_out = dec_input.permute(0,2,1,3) #(num_samples,out_len, N ,d_model)
        dec_out = self.spa_attn(
            dec_out, dec_out, cross=False)  # (num_samples, num_nodes, out_len, d_model)

        # encode_decode temporal attention
        dec_out = self.enc_dec_spa_attn(
            dec_out, enc_output, cross=True)

        dec_out = dec_out.permute(0, 2, 1, 3)  # (num_samples,out_len, N ,d_model)

        # feed foward, add & norm
        if self.if_ffn:
            dec_out = self.pos_ffn(dec_out)
        else:
            dec_out = F.relu(dec_out)

        return dec_out


class SpatialDecoder(nn.Module):
    def __init__(
            self, d_model, seq_len, n_layers, d_inner, num_nodes, range_size, dropout=0.3, if_ffn=False):
        super(SpatialDecoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.layer_stack = nn.ModuleList([
            SpatialDecoderLayer(d_model, d_inner, num_nodes, range_size, dropout, if_ffn)
            for _ in range(n_layers)])

    def forward(self, x, fuse_enc_output):
        # trend(x): (num_samples, num_nodes, out_len, d_model,)
        # fuse_end_output: num_samples, num_nodes, input_length, d_model

        # x = self.dropout(x)
        # x = self.layer_norm(x) # (num_samples, num_nodes, out_len, d_model)

        for dec_layer in self.layer_stack:
            x = dec_layer(x, fuse_enc_output, )

        return x
