import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mdm_forstyledataset import MotionEncoder
import numpy as np
from torch import autograd

def get_conv_pad(kernel_size, stride, padding=nn.ReflectionPad1d):
    pad_l = (kernel_size - stride) // 2
    pad_r = (kernel_size - stride) - pad_l
    return padding((pad_l, pad_r))


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode = self.mode)


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)  # batch size & channels
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


def ZeroPad1d(sizes):
    return nn.ConstantPad1d(sizes, 0)


def ConvLayers(kernel_size, in_channels, out_channels, stride=1, pad_type='reflect', use_bias=True):

    """
    returns a list of [pad, conv] => should be += to some list, then apply sequential
    """

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = ZeroPad1d
    else:
        assert 0, "Unsupported padding type: {}".format(pad_type)

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return [pad((pad_l, pad_r)),
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride, bias=use_bias)]


def get_acti_layer(acti='relu', inplace=True):

    if acti == 'relu':
        return [nn.ReLU(inplace=inplace)]
    elif acti == 'lrelu':
        return [nn.LeakyReLU(0.2, inplace=inplace)]
    elif acti == 'tanh':
        return [nn.Tanh()]
    elif acti == 'none':
        return []
    else:
        assert 0, "Unsupported activation: {}".format(acti)


def get_norm_layer(norm='none', norm_dim=None):

    if norm == 'bn':
        return [nn.BatchNorm1d(norm_dim)]
    elif norm == 'in':
        # return [nn.InstanceNorm1d(norm_dim, affine=False)]  # for rt42!
        return [nn.InstanceNorm1d(norm_dim, affine=True)]
    elif norm == 'adain':
        return [AdaptiveInstanceNorm1d(norm_dim)]
    elif norm == 'none':
        return []
    else:
        assert 0, "Unsupported normalization: {}".format(norm)


def get_dropout_layer(dropout=None):
    if dropout is not None:
        return [nn.Dropout(p=dropout)]
    else:
        return []


def ConvBlock(kernel_size, in_channels, out_channels, stride=1, pad_type='reflect', dropout=None,
              norm='none', acti='lrelu', acti_first=False, use_bias=True, inplace=True):
    """
    returns a list of [pad, conv, norm, acti] or [acti, pad, conv, norm]
    """

    layers = ConvLayers(kernel_size, in_channels, out_channels, stride=stride, pad_type=pad_type, use_bias=use_bias)
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_channels)
    acti_layers = get_acti_layer(acti, inplace=inplace)

    if acti_first:
        return acti_layers + layers
    else:
        return layers + acti_layers


def LinearBlock(in_dim, out_dim, dropout=None, norm='none', acti='relu'):

    use_bias = True
    layers = []
    layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_dim)
    layers += get_acti_layer(acti)

    return layers


class ResBlock(nn.Module):

    def __init__(self, kernel_size, channels, stride=1, pad_type='zero', norm='none', acti='relu'):
        super(ResBlock, self).__init__()
        layers = []
        layers += ConvBlock(kernel_size, channels, channels,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti=acti)
        layers += ConvBlock(kernel_size, channels, channels,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti='none')

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ShallowResBlock(nn.Module):

    def __init__(self, kernel_size, channels, stride=1, pad_type='zero', norm='none', acti='relu', inplace=True):
        super(ShallowResBlock, self).__init__()
        layers = []
        layers += ConvBlock(kernel_size, channels, channels,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti=acti, inplace=inplace)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ActiFirstResBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1, pad_type='reflect', norm='none', acti='lrelu'):
        super(ActiFirstResBlock, self).__init__()

        self.learned_shortcut = (in_channels != out_channels)
        self.c_in = in_channels
        self.c_out = out_channels
        self.c_mid = min(in_channels, out_channels)

        layers = []
        layers += ConvBlock(kernel_size, self.c_in, self.c_mid,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti=acti, acti_first=True)
        layers += ConvBlock(kernel_size, self.c_mid, self.c_out,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti=acti, acti_first=True)
        self.conv_model = nn.Sequential(*layers)

        if self.learned_shortcut:
            self.conv_s = nn.Sequential(*ConvBlock(kernel_size, self.c_in, self.c_out,
                                        stride=stride, pad_type='zero',
                                        norm='none', acti='none', use_bias=False))

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_model(x)
        out = x_s + dx
        return out


class BottleNeckResBlock(nn.Module):
    def __init__(self, kernel_size, c_in, c_mid, c_out, stride=1, pad_type='reflect', norm='none', acti='lrelu'):
        super(BottleNeckResBlock, self).__init__()

        self.learned_shortcut = (c_in != c_out)
        self.c_in = c_in
        self.c_out = c_out
        self.c_mid = c_mid

        layers = []
        layers += ConvBlock(kernel_size, self.c_in, self.c_mid,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti=acti)
        layers += ConvBlock(kernel_size, self.c_mid, self.c_out,
                            stride=stride, pad_type=pad_type,
                            norm='none', acti='none') # !! no norm here
        self.conv_model = nn.Sequential(*layers)

        if self.learned_shortcut:
            self.conv_s = nn.Sequential(*ConvBlock(kernel_size, self.c_in, self.c_out,
                                        stride=stride, pad_type='zero',
                                        norm='none', acti='none', use_bias=False))

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_model(x)
        out = x_s + dx
        return out


class EncoderContent(nn.Module):
    def __init__(self):
        super(EncoderContent, self).__init__()
        channels = [181, 144] # style xia
        kernel_size = 5
        stride = 2

        layers = []
        n_convs = 1
        n_resblk = 1
        acti = 'lrelu'

        assert n_convs + 1 == len(channels)

        for i in range(n_convs):
            layers += ConvBlock(kernel_size, channels[i], channels[i + 1],
                                stride=stride, norm='none', acti=acti)

        for i in range(n_resblk):
            layers.append(ResBlock(kernel_size, channels[-1], stride=1,
                                   pad_type='reflect', norm='none', acti=acti))

        self.conv_model = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x):
        # x: batch, feature, 1, seq
        x = x.squeeze(-2)
        x = self.conv_model(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        channels = [144, 181]
        kernel_size = 5
        stride = 1

        res_norm = 'none' # no adain in res
        norm = 'none'
        pad_type = 'reflect'
        acti = 'lrelu'

        layers = []
        n_resblk = 1
        n_conv = 1
        bt_channel = 144 # #channels at the bottleneck

        # layers += get_norm_layer('adain', channels[0]) # adain before everything

        for i in range(n_resblk):
            layers.append(BottleNeckResBlock(kernel_size, channels[0], bt_channel, channels[0],
                                             pad_type=pad_type, norm=res_norm, acti=acti))

        for i in range(n_conv):
            layers.append(Upsample(scale_factor=2, mode='nearest'))
            cur_acti = 'none' if i == n_conv - 1 else acti
            cur_norm = 'none' if i == n_conv - 1 else norm
            layers += ConvBlock(kernel_size, channels[i], channels[i + 1], stride=stride,
                                pad_type=pad_type, norm=cur_norm, acti=cur_acti)

        self.model = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x):
        x = self.model(x)
        x = x.unsqueeze(-2)
        return x

class AE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = EncoderContent()
        self.dec = Decoder()

    def forward(self, x):
        hidden = self.enc(x)
        output = self.dec(hidden)
        return output
    
    def enc_forward(self, x):
        return self.enc(x)
    
    def dec_reverse(self, x):
        return self.dec(x)
    


class StyleAE(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()
        
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)
        self.kargs = kargs

        self.input_feats = self.njoints * self.nfeats

        self.arch = arch
        
        self.enc = EncoderContent()
        self.dec = Decoder()
        self.motion_enc = MotionEncoder(modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim, ff_size, num_layers, num_heads, dropout,
                 ablation, activation, legacy, data_rep, dataset, clip_dim,
                 arch, emb_trans_dec, clip_version, **kargs)
        
        self.load_motion_enc()

    def load_motion_enc(self):
        print("load motion_enc from checkpoint {}".format(self.kargs["motion_enc_path"]))
        self.load_model(self.motion_enc, torch.load(self.kargs["motion_enc_path"]))
        self.motion_enc = self.motion_enc.eval()  

        for p in self.motion_enc.parameters():
            p.requires_grad = False

        assert all([not para.requires_grad for para in self.motion_enc.parameters()])

    def load_model(self, model, state_dict):
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        # print([k for k in missing_keys if not k.startswith('clip_model.')])
        assert len(unexpected_keys) == 0
        assert all([k.startswith('mdm_model.') for k in missing_keys])

    def parameters_wo_enc(self):
        return [p for name, p in self.named_parameters() if not name.startswith('motion_enc.')]

        
    def forward(self, x, timesteps, y=None):
        output = self.dec(self.enc(x))
        return output
    

class StyleDis(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()
        
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)
        self.kargs = kargs

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.emb_trans_dec = emb_trans_dec
        
        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.motion_enc = MotionEncoder(modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim, ff_size, num_layers, num_heads, dropout,
                 ablation, activation, legacy, data_rep, dataset, clip_dim,
                 arch, emb_trans_dec, clip_version, **kargs)
        
        self.dis = PatchDis()
        
        self.load_motion_enc()

    def load_motion_enc(self):
        print("load motion_enc from checkpoint {}".format(self.kargs["motion_enc_path"]))
        self.load_model(self.motion_enc, torch.load(self.kargs["motion_enc_path"]))
        self.motion_enc = self.motion_enc.eval()  

        self.dis.load_state_dict(torch.load(self.kargs["inpainting_model_path"]))
        self.dis = self.dis.eval()

        for p in self.motion_enc.parameters():
            p.requires_grad = False

        for p in self.dis.parameters():
            p.requires_grad = False

        assert all([not para.requires_grad for para in self.motion_enc.parameters()])

    def load_model(self, model, state_dict):
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        # print([k for k in missing_keys if not k.startswith('clip_model.')])
        assert len(unexpected_keys) == 0
        # assert all([k.startswith('input_process.') for k in unexpected_keys])
        # assert all([k.startswith('clip_model.') for k in missing_keys])
        assert all([k.startswith('mdm_model.') for k in missing_keys])


    def parameters_wo_enc(self):
        return [p for name, p in self.named_parameters() if not name.startswith('motion_enc.')]


    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def forward(self, x, timesteps, y=None):
    
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.motion_enc.mdm_model.embed_timestep(timesteps)  # [1, bs, d]
        
        force_mask = y.get('uncond', False)
        x_mu = self.motion_enc.mdm_model.encode_text(y['text'])
        
        # input_mu = x_mu # for debug
        emb += self.motion_enc.mdm_model.embed_text(self.mask_cond(x_mu, force_mask=force_mask))

        x = self.motion_enc.mdm_model.input_process(x)

        if self.arch == 'trans_enc':
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.motion_enc.mdm_model.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        
        output = self.motion_enc.mdm_model.output_process(output)
        return output


    
class PatchDis(nn.Module):
    def __init__(self):
        super(PatchDis, self).__init__()

        channels = [181, 96, 144] # for stylexia dataset
        down_n = 2
        ks = 4
        stride = 1
        pool_ks = 3
        pool_stride = 2

        out_dim = 8

        assert down_n + 1 == len(channels)

        cnn_f = ConvLayers(kernel_size=ks, in_channels=channels[0], out_channels=channels[0])

        for i in range(down_n):
            cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[i], out_channels=channels[i], stride=stride, acti='lrelu', norm='none')]
            cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[i], out_channels=channels[i + 1], stride=stride, acti='lrelu', norm='none')]
            cnn_f += [get_conv_pad(pool_ks, pool_stride)]
            cnn_f += [nn.AvgPool1d(kernel_size=pool_ks, stride=pool_stride)]

        cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[-1], out_channels=channels[-1], stride=stride, acti='lrelu', norm='none')]
        cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[-1], out_channels=channels[-1], stride=stride, acti='lrelu', norm='none')]

        cnn_c = ConvBlock(kernel_size=ks, in_channels=channels[-1], out_channels = out_dim,
                          stride=1, norm='none', acti='lrelu', acti_first=True)

        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        

    def forward(self, x, y):
        x = x.squeeze(  -2)
        assert(x.size(0) == y.size(0))
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)
        index = torch.LongTensor(range(out.size(0))).to(x.device)
        out = out[index, y, :]
        return out, feat

    def calc_dis_fake_loss(self, input_fake, input_label):
        resp_fake, gan_feat = self.forward(input_fake, input_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).to(input_fake.device)
        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean()
        correct_count = (resp_fake < 0).sum()
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss(self, input_real, input_label):
        resp_real, gan_feat = self.forward(input_real, input_label)
        total_count = torch.tensor(np.prod(resp_real.size()),
                                   dtype=torch.float).to(input_real.device)
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label):
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).to(input_fake.device)
        loss = -resp_fake.mean()
        correct_count = (resp_fake >= 0).sum()
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()/batch_size
        return reg

    
        

if __name__ == '__main__':
    enc = EncoderContent()
    dec = Decoder()
    
    a = torch.ones([128, 181, 76])
    x = enc(a)
    print(x.shape) # 128, 144, 38
    a_1 = dec(x)
    print(a_1.shape) # 128, 181, 76

    dis = PatchDis()
    dis_a, dis_feat = dis(a, torch.ones([128]).long())
    print(dis_a.shape, dis_feat.shape)
    # dis_a: 128 19, dis_feat: 128 144 19
