import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import TemporalEncoder, SpatialEncoder, fusion_spatial_temporal, trend_controller, TemporalDecoder, SpatialDecoder

class Trendformer(nn.Module):

    def __init__(self,in_dim=2,out_dim=1,seq_len=12,out_len=12,
                 end_channels = 32, d_inner=64,
                 en_layers=3, de_layers=3,
                 d_model=32, num_nodes=321,range_size = 25,
                 dropout=0.3,if_ffn=False):

        super(Trendformer, self).__init__()

        self.dropout = dropout
        self.seq_len = seq_len
        self.d_model = d_model

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=d_model,
                                    kernel_size=(1, 1),bias=True)


        self.temporal_encoder = TemporalEncoder(
                                       d_model=d_model,seq_len=seq_len,n_layers=en_layers,
                                       d_inner=d_inner,dropout=dropout,if_ffn=if_ffn)

        self.spatial_encoder = SpatialEncoder(
                                       d_model=d_model,n_layers=en_layers,
                                       d_inner=d_inner, num_nodes=num_nodes, range_size = range_size,
                                       dropout=dropout,if_ffn=if_ffn)

        self.fuse_ST = fusion_spatial_temporal(d_model=d_model)


        self.time_conv = nn.Conv2d(in_channels=seq_len,
                                  out_channels=out_len,
                                  kernel_size=(1,1),
                                  bias=True)

        self.trend_controller = trend_controller(d_model = d_model,seq_len = seq_len)


        self.temporal_decoder = TemporalDecoder(
                                d_model=d_model, seq_len=seq_len, n_layers=de_layers,
                                d_inner=d_inner,  dropout=dropout,if_ffn=if_ffn)

        self.spatial_decoder = SpatialDecoder(
                                d_model=d_model, seq_len=seq_len, n_layers=de_layers,
                                d_inner=d_inner, num_nodes=num_nodes,range_size = range_size,
                                dropout=dropout,if_ffn=if_ffn)


        self.end_conv_1 = nn.Conv2d(in_channels=d_model,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.end_linear1 = nn.Linear(in_features=d_model,out_features=end_channels)
        self.end_linear2 = nn.Linear(in_features=end_channels,out_features=out_dim)


    def forward(self, x, trend, ):
        # x: (num_samples, input_dim, num_nodes, input_length)
        # y: (num_samples, num_nodes, input_length)
        # trend : (num_samples, num_nodes, output_length,)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # print('x_shape',x.shape)
        seq_length = x.size(3)
        assert seq_length == self.seq_len, 'input sequence length not equal to preset sequence length'

        x = self.start_conv(x)  #(num_samples, d_model, num_nodes, input_length)
        x = x.permute(0,2,3,1).contiguous()# (num_samples, num_nodes, input_length, d_model )

        """ Encoder """

        tem_enc_output, tem_inputs_list = self.temporal_encoder(x) # (num_samples, num_nodes, input_length, d_model)

        # spa_enc_output: num_samples, num_nodes, input_length, d_model
        # attention : batch_size, num_nodes, num_nodes
        spa_enc_output = self.spatial_encoder(x)


        """ Temporal Spatial Fusion """

        fuse_enc_output = self.fuse_ST(tem_enc_output, spa_enc_output)
        fuse_enc_output = self.time_conv(fuse_enc_output.permute(0,2,1,3)).permute(0,2,1,3) # num_samples, num_nodes, input_len,d_model,) -> num_samples, num_nodes, out_len,d_model,)

        """ Gate Selection Mechanism """
        trend = self.trend_controller(trend,fuse_enc_output)

        """ Decoder """

        dec_out = self.temporal_decoder(trend, fuse_enc_output,tem_inputs_list) # (num_samples, num_nodes, out_len,d_model,)
        dec_out = self.spatial_decoder(dec_out, fuse_enc_output)  # (num_samples, num_nodes, out_len,d_model,)
        dec_out = self.end_conv_1(dec_out.permute(0,3,1,2))  # (num_samples, out_channels, num_nodes, out_len)
        dec_out = F.relu(dec_out)
        dec_out = self.end_conv_2(dec_out) ## (num_samples, 1, num_nodes, out_len)

        # dec_out = self.end_linear1(dec_out)
        # dec_out = F.relu(dec_out)
        # dec_out = self.end_linear2(dec_out)


        return dec_out