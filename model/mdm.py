import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from typing import Optional

class StyleTransformerLayer(nn.TransformerEncoderLayer):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=0.00001, batch_first=False, norm_first=False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        self.ins_norm = nn.InstanceNorm1d(d_model, affine=False)

    def forward(self, src: torch.Tensor,  adaIN_para: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x, adaIN_para))

        return x

    # feed forward block
    def _ff_block(self, x: torch.Tensor, adaIN_para=None) -> torch.Tensor: # x: seq bs d
        """
        x: content motion code: [seq, bs, d]
        sty_x: style motion code: [batch_size, njoints, nfeats, max_frames]
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        if adaIN_para is not None:
            x = x.permute(1, 2, 0)
            gamma, beta = adaIN_para
            x = (gamma + 1) * self.ins_norm(x) + beta
            x = x.permute(2, 0, 1)
   
        return self.dropout2(x)
    

class StyleTransformerEncoder(nn.TransformerEncoder):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)   

    def forward(self, src: torch.Tensor, adaIN_para: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None, 
                src_key_padding_mask: Optional[torch.Tensor] = None,
                middle_trans = False, layer_residual = None) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        
        layer_len = len(self.layers)
        for i, mod in enumerate(self.layers):
            if middle_trans:
                if i >= layer_len // 2:
                    adaIN_para = None
            output = mod(output, adaIN_para, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            if layer_residual is not None:
                output += layer_residual[i]

        if self.norm is not None:
            output = self.norm(output)

        return output
    
    def outEachLayer(self, src: torch.Tensor, adaIN_para: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None, 
                src_key_padding_mask: Optional[torch.Tensor] = None,
                middle_trans = False) -> torch.Tensor:
        output = src
        
        layer_len = len(self.layers)
        all_output = []
        for i, mod in enumerate(self.layers):
            if middle_trans:
                if i >= layer_len // 2:
                    adaIN_para = None
            output = mod(output, adaIN_para, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            all_output.append(output) 

        if self.norm is not None:
            output = self.norm(output)

        return all_output



class StyEncoder(nn.Module):
    def __init__(self, data_rep, njoints, nfeats, latent_dim=512) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.data_rep = data_rep
        self.njoints = njoints
        self.nfeats = nfeats
        self.input_feats = self.njoints * self.nfeats
        
        # self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim) # Seq B d
        
        self.conv1 = nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=9, stride=1, 
                               padding='same', padding_mode='reflect')
        self.conv2 = nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=9, stride=1, 
                               padding='same', padding_mode='reflect')
        self.maxpool = nn.MaxPool1d(2, 2)
        self.activation = nn.GELU()

    def forward(self, x, y=None):
        """
        x: [Seq, Batch_size, latent_dim], denoted style motion representation
        y: conditons include motion mask
        """
        # x = self.input_process(x) # Seq B d
        x = x.permute(1, 2, 0)
        if y is not None:
            motion_mask = y["mask"].squeeze(1) # B 1 1 Maxleng
            motion_mask = self.maxpool(self.maxpool(motion_mask))
        else:
            motion_mask = None
        x = self.maxpool(self.activation(self.conv1(x)))
        x = self.maxpool(self.activation(self.conv2(x)))
   
        return x, motion_mask


# class AdaIN(nn.Module):
#     def __init__(self, latent_dim, num_features):
#         super().__init__()
        
#         self.adaptpool = nn.AdaptiveAvgPool1d(1)
#         self.to_latent = nn.Sequential(
#                                        nn.Conv1d(num_features, latent_dim, 1, 1, 0),
#                                        nn.GELU())
#         self.inject = nn.Sequential(nn.Linear(latent_dim, latent_dim),
#                                     nn.GELU(),
#                                     nn.Linear(latent_dim, num_features*2))
#         self.norm = nn.InstanceNorm1d(num_features, affine=False)

#     def forward(self, x, s, motion_mask=None):
#         """
#         Args:
#             x: B C Seq
#             s: B C Seq
#         Returns:

#         """
#         if motion_mask is None:
#             s = self.adaptpool(s)
#         else:
#             s = s * motion_mask
#             s = s.sum(-1, keepdim=True)/motion_mask.sum(-1, keepdim=True)
#         s = self.to_latent(s).squeeze(-1)   # B C
#         h = self.inject(s)
#         h = h.view(h.size(0), h.size(1), 1)
#         gamma, beta = torch.chunk(h, chunks=2, dim=1)
#         return (1 + gamma) * self.norm(x) + beta


class AdaIN(nn.Module):
    def __init__(self, latent_dim, num_features):
        super().__init__()
        
        self.adaptpool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Sequential(
                                       nn.Conv1d(num_features, latent_dim, 1, 1, 0),
                                       nn.GELU())
        self.inject = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                    nn.GELU(),
                                    nn.Linear(latent_dim, num_features*2))

    def forward(self, s, motion_mask=None):
        """
        Args:
            x: B C Seq
            s: B C Seq
        Returns:

        """
        if motion_mask is None:
            s = self.adaptpool(s)
        else:
            s = s * motion_mask
            s = s.sum(-1, keepdim=True)/motion_mask.sum(-1, keepdim=True)
        s = self.to_latent(s).squeeze(-1)   # B C
        h = self.inject(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (gamma, beta)
        


class MDM(nn.Module):
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

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.sequence_pos_encoder_shift = PositionalEncoding_shift(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        
        self.sty_enc = StyEncoder(self.data_rep, self.njoints, self.nfeats, self.latent_dim)
        self.adaIN = AdaIN(self.latent_dim, self.latent_dim)
        
        
        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = StyleTransformerLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = StyleTransformerEncoder(seqTransEncoderLayer,
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

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None, sty_x=None, sty_y=None, layer_residual=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        x = self.input_process(x)
        if sty_x is not None:
            sty_x = self.input_process(sty_x) # seq bs d
            sty_feat, adaIN_mask = self.sty_enc(sty_x, sty_y)
            adaIN_para = self.adaIN(sty_feat, adaIN_mask)
        else:
            adaIN_para = None 

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

       
        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if sty_y is not None:
                middle_trans = sty_y.get('middle_trans', False)
            else:
                middle_trans = False
            output = self.seqTransEncoder(xseq, adaIN_para, middle_trans=middle_trans, layer_residual=layer_residual)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


    def style_forward(self, sty_x=None, sty_y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        sty_x = self.input_process(sty_x) # seq bs d
        sty_feat, adaIN_mask = self.sty_enc(sty_x, sty_y)
        adaIN_para = self.adaIN(sty_feat, adaIN_mask)
        return adaIN_para
    
    def re_encode(self, output):
        re_input = self.input_process(output)
        zero_times = torch.zeros(re_input.shape[1]).to(re_input.device).long()
        emb = self.embed_timestep(zero_times)  # [1, bs, d]
        if self.arch == 'trans_enc':
            re_input = torch.cat((emb, re_input), axis=0)  # [seqlen+1, bs, d]
            re_input = self.sequence_pos_encoder(re_input)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(re_input)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        output = self.output_process(output)
        return output
    
    def getLayerLatent(self, x):

        x = self.input_process(x)  
        zero_times = torch.zeros(x.shape[1]).to(x.device).long()
        emb = self.embed_timestep(zero_times)  # [1, bs, d]

        if self.arch == 'trans_enc':
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output_features = self.seqTransEncoder.outEachLayer(xseq, None, middle_trans=False)
        
        return output_features

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    

class PositionalEncoding_shift(nn.Module):
    # shift one position when positional encoding for omit the text and time influence.
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding_shift, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[1:x.shape[0]+1, :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
    

if __name__ == "__main__":
    style_enc = StyEncoder('hml_vec', 1, 263, 512)
    adain = AdaIN(512, 512)
    x = torch.randn([32, 1, 263, 196])
    cond_x = torch.randn((32, 512, 196))
    x_mask = {"mask":torch.from_numpy(np.ones((32,196))).float().unsqueeze(1).unsqueeze(1)}
    print(x_mask["mask"].dtype)
    y, motion_mask = style_enc(x, x_mask)
    print(y.shape, motion_mask.shape) 
    y = adain(cond_x, y, motion_mask)
    print(y.shape)