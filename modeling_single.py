import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BigBirdForTokenClassification, BertPreTrainedModel, RobertaPreTrainedModel, BertModel, BertForTokenClassification
from transformers import AutoModelForTokenClassification, AutoModel
from modules.transformer import TransformerEncoder, MultiHeadAttn, TransformerLayer
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange
from torch import Tensor
import math

num_layers = 2
n_heads = 8
head_dims = 128
d_model = n_heads * head_dims
feedforward_dim = int(2 * d_model)
dropout = 0.45
dropout_attn = None
after_norm = 1
attn_type = 'transformer'
scale = attn_type == 'transformer'
pos_embed = 'sin'
fc_dropout = 0.4

#class BertForTokenClassification_(BigBirdForTokenClassification):
#class BertForTokenClassification_(BertForTokenClassification):
#class AutoModelForTokenClassification_(AutoModelForTokenClassification):
#    r"""
#        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
#            Labels for computing the token classification loss.
#            Indices should be in ``[0, ..., config.num_labels - 1]``.
#
#    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
#        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
#            Classification loss.
#        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
#            Classification scores (before SoftMax).
#        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
#            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
#            of shape ``(batch_size, sequence_length, hidden_size)``:
#            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
#            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
#            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
#
#    Examples::
#
#        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
#        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
#        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
#        outputs = model(input_ids, labels=labels)
#        loss, scores = outputs[:2]
#
#    """
#    def __init__(self, config):
#      super(BertForTokenClassification, self).__init__(config)
#
#
#      self.num_labels = config.num_labels
#  
#      self.bert = AutoModel(config)
#      self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
#      self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
#  
#      self.init_weights()


class DocFormerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(DocFormerEmbeddings, self).__init__()

        self.config = config

        self.position_embeddings_v = PositionalEncoding(
            d_model=config["hidden_size"],
            dropout=0.1,
            max_len=config["max_position_embeddings"],
        )

        self.x_topleft_position_embeddings_v = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])
        self.x_bottomright_position_embeddings_v = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])
        self.w_position_embeddings_v = nn.Embedding(config["max_2d_position_embeddings"], config["shape_size"])
        self.x_topleft_distance_to_prev_embeddings_v = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.x_bottomleft_distance_to_prev_embeddings_v = nn.Embedding(2*config["max_2d_position_embeddings"]  + 1, config["shape_size"])
        self.x_topright_distance_to_prev_embeddings_v = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.x_bottomright_distance_to_prev_embeddings_v = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.x_centroid_distance_to_prev_embeddings_v = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])

        self.y_topleft_position_embeddings_v = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])
        self.y_bottomright_position_embeddings_v = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])
        self.h_position_embeddings_v = nn.Embedding(config["max_2d_position_embeddings"], config["shape_size"])
        self.y_topleft_distance_to_prev_embeddings_v = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.y_bottomleft_distance_to_prev_embeddings_v = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.y_topright_distance_to_prev_embeddings_v = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.y_bottomright_distance_to_prev_embeddings_v = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.y_centroid_distance_to_prev_embeddings_v = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])

        self.position_embeddings_t = PositionalEncoding(
            d_model=config["hidden_size"],
            dropout=0.1,
            max_len=config["max_position_embeddings"],
        )

        self.x_topleft_position_embeddings_t = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])
        self.x_bottomright_position_embeddings_t = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])
        self.w_position_embeddings_t = nn.Embedding(config["max_2d_position_embeddings"], config["shape_size"])
        self.x_topleft_distance_to_prev_embeddings_t = nn.Embedding(2*config["max_2d_position_embeddings"]+1, config["shape_size"])
        self.x_bottomleft_distance_to_prev_embeddings_t = nn.Embedding(2*config["max_2d_position_embeddings"]+1, config["shape_size"])
        self.x_topright_distance_to_prev_embeddings_t = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.x_bottomright_distance_to_prev_embeddings_t = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.x_centroid_distance_to_prev_embeddings_t = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])

        self.y_topleft_position_embeddings_t = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])
        self.y_bottomright_position_embeddings_t = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])
        self.h_position_embeddings_t = nn.Embedding(config["max_2d_position_embeddings"], config["shape_size"])
        self.y_topleft_distance_to_prev_embeddings_t = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.y_bottomleft_distance_to_prev_embeddings_t = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.y_topright_distance_to_prev_embeddings_t = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.y_bottomright_distance_to_prev_embeddings_t = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])
        self.y_centroid_distance_to_prev_embeddings_t = nn.Embedding(2*config["max_2d_position_embeddings"] + 1, config["shape_size"])

        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])



    def forward(self, x_feature, y_feature):

        """
        Arguments:
        x_features of shape, (batch size, seq_len, 8)
        y_features of shape, (batch size, seq_len, 8)
        Outputs:
        (V-bar-s, T-bar-s) of shape (batch size, 512,768),(batch size, 512,768)
        What are the features:
        0 -> top left x/y
        1 -> bottom right x/y
        2 -> width/height
        3 -> diff top left x/y
        4 -> diff bottom left x/y
        5 -> diff top right x/y
        6 -> diff bottom right x/y
        7 -> centroids diff x/y
        """


        batch, seq_len = x_feature.shape[:-1]
        hidden_size = self.config["hidden_size"]
        num_feat = x_feature.shape[-1]
        sub_dim = hidden_size // num_feat
        
        # Clamping and adding a bias for handling negative values
        x_feature[:,:,3:] = torch.clamp(x_feature[:,:,3:],-self.config["max_2d_position_embeddings"],self.config["max_2d_position_embeddings"])
        x_feature[:,:,3:]+= self.config["max_2d_position_embeddings"]

        y_feature[:,:,3:] = torch.clamp(y_feature[:,:,3:],-self.config["max_2d_position_embeddings"],self.config["max_2d_position_embeddings"])
        y_feature[:,:,3:]+= self.config["max_2d_position_embeddings"]
        
        x_topleft_position_embeddings_v = self.x_topleft_position_embeddings_v(x_feature[:,:,0])
        x_bottomright_position_embeddings_v = self.x_bottomright_position_embeddings_v(x_feature[:,:,1])
        w_position_embeddings_v = self.w_position_embeddings_v(x_feature[:,:,2])
        x_topleft_distance_to_prev_embeddings_v = self.x_topleft_distance_to_prev_embeddings_v(x_feature[:,:,3])
        x_bottomleft_distance_to_prev_embeddings_v = self.x_bottomleft_distance_to_prev_embeddings_v(x_feature[:,:,4])
        x_topright_distance_to_prev_embeddings_v = self.x_topright_distance_to_prev_embeddings_v(x_feature[:,:,5])
        x_bottomright_distance_to_prev_embeddings_v = self.x_bottomright_distance_to_prev_embeddings_v(x_feature[:,:,6])
        x_centroid_distance_to_prev_embeddings_v = self.x_centroid_distance_to_prev_embeddings_v(x_feature[:,:,7])

        x_calculated_embedding_v = torch.cat(
            [
             x_topleft_position_embeddings_v,
             x_bottomright_position_embeddings_v,
             w_position_embeddings_v,
             x_topleft_distance_to_prev_embeddings_v,
             x_bottomleft_distance_to_prev_embeddings_v,
             x_topright_distance_to_prev_embeddings_v,
             x_bottomright_distance_to_prev_embeddings_v ,
             x_centroid_distance_to_prev_embeddings_v
            ],
            dim = -1
        )

        y_topleft_position_embeddings_v = self.y_topleft_position_embeddings_v(y_feature[:,:,0])
        y_bottomright_position_embeddings_v = self.y_bottomright_position_embeddings_v(y_feature[:,:,1])
        h_position_embeddings_v = self.h_position_embeddings_v(y_feature[:,:,2])
        y_topleft_distance_to_prev_embeddings_v = self.y_topleft_distance_to_prev_embeddings_v(y_feature[:,:,3])
        y_bottomleft_distance_to_prev_embeddings_v = self.y_bottomleft_distance_to_prev_embeddings_v(y_feature[:,:,4])
        y_topright_distance_to_prev_embeddings_v = self.y_topright_distance_to_prev_embeddings_v(y_feature[:,:,5])
        y_bottomright_distance_to_prev_embeddings_v = self.y_bottomright_distance_to_prev_embeddings_v(y_feature[:,:,6])
        y_centroid_distance_to_prev_embeddings_v = self.y_centroid_distance_to_prev_embeddings_v(y_feature[:,:,7])

        x_calculated_embedding_v = torch.cat(
            [
             x_topleft_position_embeddings_v,
             x_bottomright_position_embeddings_v,
             w_position_embeddings_v,
             x_topleft_distance_to_prev_embeddings_v,
             x_bottomleft_distance_to_prev_embeddings_v,
             x_topright_distance_to_prev_embeddings_v,
             x_bottomright_distance_to_prev_embeddings_v ,
             x_centroid_distance_to_prev_embeddings_v
            ],
            dim = -1
        )

        y_calculated_embedding_v = torch.cat(
            [
             y_topleft_position_embeddings_v,
             y_bottomright_position_embeddings_v,
             h_position_embeddings_v,
             y_topleft_distance_to_prev_embeddings_v,
             y_bottomleft_distance_to_prev_embeddings_v,
             y_topright_distance_to_prev_embeddings_v,
             y_bottomright_distance_to_prev_embeddings_v ,
             y_centroid_distance_to_prev_embeddings_v
            ],
            dim = -1
        )

        v_bar_s = x_calculated_embedding_v + y_calculated_embedding_v + self.position_embeddings_v()



        x_topleft_position_embeddings_t = self.x_topleft_position_embeddings_t(x_feature[:,:,0])
        x_bottomright_position_embeddings_t = self.x_bottomright_position_embeddings_t(x_feature[:,:,1])
        w_position_embeddings_t = self.w_position_embeddings_t(x_feature[:,:,2])
        x_topleft_distance_to_prev_embeddings_t = self.x_topleft_distance_to_prev_embeddings_t(x_feature[:,:,3])
        x_bottomleft_distance_to_prev_embeddings_t = self.x_bottomleft_distance_to_prev_embeddings_t(x_feature[:,:,4])
        x_topright_distance_to_prev_embeddings_t = self.x_topright_distance_to_prev_embeddings_t(x_feature[:,:,5])
        x_bottomright_distance_to_prev_embeddings_t = self.x_bottomright_distance_to_prev_embeddings_t(x_feature[:,:,6])
        x_centroid_distance_to_prev_embeddings_t = self.x_centroid_distance_to_prev_embeddings_t(x_feature[:,:,7])

        x_calculated_embedding_t = torch.cat(
            [
             x_topleft_position_embeddings_t,
             x_bottomright_position_embeddings_t,
             w_position_embeddings_t,
             x_topleft_distance_to_prev_embeddings_t,
             x_bottomleft_distance_to_prev_embeddings_t,
             x_topright_distance_to_prev_embeddings_t,
             x_bottomright_distance_to_prev_embeddings_t ,
             x_centroid_distance_to_prev_embeddings_t
            ],
            dim = -1
        )

        y_topleft_position_embeddings_t = self.y_topleft_position_embeddings_t(y_feature[:,:,0])
        y_bottomright_position_embeddings_t = self.y_bottomright_position_embeddings_t(y_feature[:,:,1])
        h_position_embeddings_t = self.h_position_embeddings_t(y_feature[:,:,2])
        y_topleft_distance_to_prev_embeddings_t = self.y_topleft_distance_to_prev_embeddings_t(y_feature[:,:,3])
        y_bottomleft_distance_to_prev_embeddings_t = self.y_bottomleft_distance_to_prev_embeddings_t(y_feature[:,:,4])
        y_topright_distance_to_prev_embeddings_t = self.y_topright_distance_to_prev_embeddings_t(y_feature[:,:,5])
        y_bottomright_distance_to_prev_embeddings_t = self.y_bottomright_distance_to_prev_embeddings_t(y_feature[:,:,6])
        y_centroid_distance_to_prev_embeddings_t = self.y_centroid_distance_to_prev_embeddings_t(y_feature[:,:,7])

        x_calculated_embedding_t = torch.cat(
            [
             x_topleft_position_embeddings_t,
             x_bottomright_position_embeddings_t,
             w_position_embeddings_t,
             x_topleft_distance_to_prev_embeddings_t,
             x_bottomleft_distance_to_prev_embeddings_t,
             x_topright_distance_to_prev_embeddings_t,
             x_bottomright_distance_to_prev_embeddings_t ,
             x_centroid_distance_to_prev_embeddings_t
            ],
            dim = -1
        )

        y_calculated_embedding_t = torch.cat(
            [
             y_topleft_position_embeddings_t,
             y_bottomright_position_embeddings_t,
             h_position_embeddings_t,
             y_topleft_distance_to_prev_embeddings_t,
             y_bottomleft_distance_to_prev_embeddings_t,
             y_topright_distance_to_prev_embeddings_t,
             y_bottomright_distance_to_prev_embeddings_t ,
             y_centroid_distance_to_prev_embeddings_t
            ],
            dim = -1
        )

        t_bar_s = x_calculated_embedding_t + y_calculated_embedding_t + self.position_embeddings_t()
        
        return v_bar_s, t_bar_s



# fmt: off
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        # Fig 1: http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttn(nn.Module):
    def __init__(self, dim, fn):
        # Fig 1: http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf
        super().__init__()

        self.norm_t_bar = nn.LayerNorm(dim)
        self.norm_v_bar = nn.LayerNorm(dim)
        self.norm_t_bar_s = nn.LayerNorm(dim)
        self.norm_v_bar_s = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, t_bar, v_bar, t_bar_s, v_bar_s, **kwargs):
        return self.fn(self.norm_t_bar(t_bar),
                       self.norm_v_bar(v_bar),
                       self.norm_t_bar_s(t_bar_s),
                       self.norm_v_bar_s(v_bar_s), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position, max_seq_length):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        self.max_length = max_seq_length
        range_vec_q = torch.arange(max_seq_length)
        range_vec_k = torch.arange(max_seq_length)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        self.final_mat = torch.LongTensor(final_mat)
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        embeddings = self.embeddings_table[self.final_mat[:length_q, :length_k]]
        return embeddings


class MultiModalAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, max_relative_position, max_seq_length, dropout):
        super().__init__()
        assert embed_dim % n_heads == 0

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.relative_positions_text = RelativePosition(self.head_dim, max_relative_position, max_seq_length)
        self.relative_positions_img = RelativePosition(self.head_dim, max_relative_position, max_seq_length)

        # text qkv embeddings
        self.fc_k_text = nn.Linear(embed_dim, embed_dim)
        self.fc_q_text = nn.Linear(embed_dim, embed_dim)
        self.fc_v_text = nn.Linear(embed_dim, embed_dim)

        # image qkv embeddings
        self.fc_k_img = nn.Linear(embed_dim, embed_dim)
        self.fc_q_img = nn.Linear(embed_dim, embed_dim)
        self.fc_v_img = nn.Linear(embed_dim, embed_dim)

        # spatial qk embeddings (shared for visual and text)
        self.fc_k_spatial = nn.Linear(embed_dim, embed_dim)
        self.fc_q_spatial = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))

    def forward(self, text_feat, img_feat, text_spatial_feat, img_spatial_feat):
        text_feat = text_feat
        img_feat = img_feat
        text_spatial_feat = text_spatial_feat
        img_spatial_feat = img_spatial_feat
        seq_length = text_feat.shape[1]

        # self attention of text
        # b -> batch, t -> time steps (l -> length has same meaning), head -> # of heads, k -> head dim.
        key_text_nh = rearrange(self.fc_k_text(text_feat), 'b t (head k) -> head b t k', head=self.n_heads).to(DEVICE)
        query_text_nh = rearrange(self.fc_q_text(text_feat), 'b l (head k) -> head b l k', head=self.n_heads).to(DEVICE)
        value_text_nh = rearrange(self.fc_v_text(text_feat), 'b t (head k) -> head b t k', head=self.n_heads).to(DEVICE)
        dots_text = torch.einsum('hblk,hbtk->hblt', query_text_nh, key_text_nh) / self.scale.to(DEVICE)

        # 1D relative positions (query, key)
        rel_pos_embed_text = self.relative_positions_text(seq_length, seq_length)
        rel_pos_key_text = torch.einsum('bhrd,lrd->bhlr', key_text_nh, rel_pos_embed_text)
        rel_pos_query_text = torch.einsum('bhld,lrd->bhlr', query_text_nh, rel_pos_embed_text)

        # shared spatial <-> text hidden features
        key_spatial_text = self.fc_k_spatial(text_spatial_feat)
        query_spatial_text = self.fc_q_spatial(text_spatial_feat)
        key_spatial_text_nh = rearrange(key_spatial_text, 'b t (head k) -> head b t k', head=self.n_heads)
        query_spatial_text_nh = rearrange(query_spatial_text, 'b l (head k) -> head b l k', head=self.n_heads)
        dots_text_spatial = torch.einsum('hblk,hbtk->hblt', query_spatial_text_nh, key_spatial_text_nh) / self.scale.to(DEVICE)

        # Line 38 of pseudo-code
        text_attn_scores = dots_text + rel_pos_key_text + rel_pos_query_text + dots_text_spatial

        # self-attention of image
        key_img_nh = rearrange(self.fc_k_img(img_feat), 'b t (head k) -> head b t k', head=self.n_heads).to(DEVICE)
        query_img_nh = rearrange(self.fc_q_img(img_feat), 'b l (head k) -> head b l k', head=self.n_heads).to(DEVICE)
        value_img_nh = rearrange(self.fc_v_img(img_feat), 'b t (head k) -> head b t k', head=self.n_heads).to(DEVICE)
        dots_img = torch.einsum('hblk,hbtk->hblt', query_img_nh, key_img_nh) / self.scale.to(DEVICE)

        # 1D relative positions (query, key)
        rel_pos_embed_img = self.relative_positions_img(seq_length, seq_length)
        rel_pos_key_img = torch.einsum('bhrd,lrd->bhlr', key_img_nh, rel_pos_embed_text)
        rel_pos_query_img = torch.einsum('bhld,lrd->bhlr', query_img_nh, rel_pos_embed_text)

        # shared spatial <-> image features
        key_spatial_img = self.fc_k_spatial(img_spatial_feat)
        query_spatial_img = self.fc_q_spatial(img_spatial_feat)
        key_spatial_img_nh = rearrange(key_spatial_img, 'b t (head k) -> head b t k', head=self.n_heads)
        query_spatial_img_nh = rearrange(query_spatial_img, 'b l (head k) -> head b l k', head=self.n_heads)
        dots_img_spatial = torch.einsum('hblk,hbtk->hblt', query_spatial_img_nh, key_spatial_img_nh) / self.scale.to(DEVICE)

        # Line 59 of pseudo-code
        img_attn_scores = dots_img + rel_pos_key_img + rel_pos_query_img + dots_img_spatial

        text_attn_probs = self.dropout(torch.softmax(text_attn_scores, dim=-1))
        img_attn_probs = self.dropout(torch.softmax(img_attn_scores, dim=-1))

        text_context = torch.einsum('hblt,hbtv->hblv', text_attn_probs, value_text_nh)
        img_context = torch.einsum('hblt,hbtv->hblv', img_attn_probs, value_img_nh)

        context = text_context + img_context

        embeddings = rearrange(context, 'head b t d -> b t (head d)')
        return self.to_out(embeddings)

class DocFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([])
        for _ in range(config['num_hidden_layers']):
            encoder_block = nn.ModuleList([
                PreNormAttn(config['hidden_size'],
                            MultiModalAttentionLayer(config['hidden_size'],
                                                     config['num_attention_heads'],
                                                     config['max_relative_positions'],
                                                     config['max_position_embeddings'],
                                                     config['hidden_dropout_prob'],
                                                     )
                            ),
                PreNorm(config['hidden_size'],
                        FeedForward(config['hidden_size'],
                                    config['hidden_size'] * config['intermediate_ff_size_factor'],
                                    dropout=config['hidden_dropout_prob']))
            ])
            self.layers.append(encoder_block)

    def forward(
            self,
            text_feat,  # text feat or output from last encoder block
            img_feat,
            text_spatial_feat,
            img_spatial_feat,
    ):
        # Fig 1 encoder part (skip conn for both attn & FF): https://arxiv.org/abs/1706.03762
        # TODO: ensure 1st skip conn (var "skip") in such a multimodal setting makes sense (most likely does)
        for attn, ff in self.layers:
            skip = text_feat + img_feat + text_spatial_feat + img_spatial_feat
            x = attn(text_feat, img_feat, text_spatial_feat, img_spatial_feat) + skip
            x = ff(x) + x
            text_feat = x
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        

    def forward(self) -> Tensor:
        x = self.pe[0, : self.max_len]
        return self.dropout(x).unsqueeze(0)
    
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Making the resnet 50 model, which was used in the docformer for the purpose of visual feature extraction

        self.resnet50 = models.resnet50(pretrained=True)
        for param in self.resnet50.parameters():
            param.requires_grad = True
        
        modules = list(self.resnet50.children())[:-2]
        self.resnet50 = nn.Sequential(*modules)
        for param in self.resnet50.parameters():
            param.requires_grad = True

        # Applying convolution and linear layer

        self.conv1 = nn.Conv2d(2048, 768, 1)
        self.relu1 = F.relu
        self.linear1 = nn.Linear(49, 128)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = self.resnet50(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = rearrange(x, "b e w h -> b e (w h)")  # b -> batch, e -> embedding dim, w -> width, h -> height
        x = self.linear1(x)
        x = rearrange(x, "b e s -> b s e")  # b -> batch, e -> embedding dim, s -> sequence length
        return x


config = {
  "coordinate_size": 96,
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "image_feature_pool_shape": [
    7,
    7,
    256
  ],
  "intermediate_ff_size_factor": 3,  # default ought to be 4
  "max_2d_position_embeddings": 1024,
  "max_position_embeddings": 512,
  "max_relative_positions": 8,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "shape_size": 96,
  "vocab_size": 30522,
  "layer_norm_eps": 1e-12,
  "batch_size":9
}


from transformers import ViTFeatureExtractor, ViTModel

IMAGE_MODEL = 'google/vit-base-patch16-224'

#class BertForTokenClassification_(BertPreTrainedModel):
class BertForTokenClassification_(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, max_len=128):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.max_len = max_len
        
        self.language_feature = BertModel(config, add_pooling_layer=False)
        self.visual_feature = ResNetFeatureExtractor()
        self.visual_transformer = ViTModel.from_pretrained(IMAGE_MODEL)
        # self.spatial_feature = DocFormerEmbeddings(config)
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, config.num_labels)
        # self.classifier = torch.nn.Linear(1300, config.num_labels)

        self.in_fc = torch.nn.Linear(self.config.hidden_size, d_model)
        self.transformer = TransformerEncoder(num_layers, d_model, n_heads, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.fc_dropout = torch.nn.Dropout(fc_dropout)
        # Initialize weights and apply final processing
#        self.post_init()

        self.position_embeddings_v = PositionalEncoding(
            d_model=1536,
            dropout=0.1,
            max_len=self.max_len
        )
        
    def forward(self, input_ids, src_probs=None, attention_mask=None, token_type_ids=None, image_ids=None,
                position_ids=None, head_mask=None, labels=None, loss_ignore_index=-100):
        
      
#        mask = attention_mask.ne(0)
        outputs = self.language_feature(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        image_output = self.visual_transformer(image_ids)
        image_output = image_output['pooler_output']
        
#        sequence_output = self.in_fc(sequence_output)
#        sequence_output = self.transformer(sequence_output, mask)
#        sequence_output = self.fc_dropout(sequence_output)
        # output = torch.cat([image_output['last_hidden_state'], sequence_output], 1)
        visual_feature = self.visual_feature(image_ids)
        # import pdb;pdb.set_trace()
        
        # sequence_output = torch.cat([sequence_output, image_output.unsqueeze(1).repeat(1, 128, 1)], 2)
        sequence_output = torch.cat([sequence_output, visual_feature], 2)
        # output = torch.cat([sequence_output, image_output.unsqueeze(1)], 1)
        # print(output.shape)
        
        sequence_output = sequence_output + self.position_embeddings_v()
        # import pdb;pdb.set_trace()
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        if src_probs is not None:
            # ## KL Divergence
            # loss_KD_fct = KLDivLoss(reduction="mean")
            # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_log_probs = log_probs.view(-1, self.num_labels)[active_loss]
            #     active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]
            #
            #     loss_KD = loss_KD_fct(active_log_probs, active_src_probs)
            # else:
            #     loss_KD = loss_KD_fct(log_probs, src_probs)

            # ## CrossEntropy
            # loss_KD_fct = CrossEntropyLoss()
            # src_labels = torch.argmax(src_probs.view(-1, self.num_labels), dim=-1)
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)[active_loss]
            #     active_src_labels = src_labels[active_loss]
            #
            #     loss_KD = loss_KD_fct(active_logits, active_src_labels)
            # else:
            #     loss_KD = loss_KD_fct(logits.view(-1, self.num_labels), src_labels)

            ## L2 Norm
            loss_KD_fct = MSELoss(reduction="mean")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                inactive_subword = labels.view(-1) == loss_ignore_index
                active_loss[inactive_subword] = 0
                active_probs = probs.view(-1, self.num_labels)[active_loss]
                active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]

                loss_KD = loss_KD_fct(active_probs, active_src_probs)
            else:
                loss_KD = loss_KD_fct(probs, src_probs)

            outputs = (loss_KD,) + outputs

        return outputs  # (loss_KD), (loss), scores, (hidden_states), (attentions)


#class BaseModel(BertPreTrainedModel):
#    def __init__(self, config):
#        super(BaseModel, self).__init__(config)
#
#        self.bert = BertModel(config)
#        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
#
#        self.init_weights()
#
#    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
#
#        outputs = self.bert(input_ids,
#                            attention_mask=attention_mask,
#                            token_type_ids=token_type_ids,
#                            position_ids=position_ids,
#                            head_mask=head_mask)
#
#        # sequence_output = outputs[0]
#        # pooled_output = outputs[1]
#
#        return outputs
#

class LIOutputLayer(torch.nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, n_langs, gr_lambda=-1.0):
        super(LIOutputLayer, self).__init__()
        self.n_langs = n_langs

        if gr_lambda > 0:
            self.dropout = torch.nn.Sequential(GradientReversal(lambda_=gr_lambda), torch.nn.Dropout(hidden_dropout_prob))
        else:
            self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, n_langs)

    def forward(self, pooled_output, labels=None):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,)

        if labels is not None:
            if self.n_langs == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.n_langs), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits


class TaskOutputLayer(torch.nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, n_labels):
        super(TaskOutputLayer, self).__init__()
        self.n_labels = n_labels

        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, n_labels)

    def forward(self, sequence_output, src_probs=None, attention_mask=None, labels=None): # src_probs: batch_size x n_src_langs x n_labels
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # logits: batch_size x seq_len x n_labels

        outputs = (logits,)

        # compute the supervised loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.n_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # compute the li_probs weighted KD loss (L2 norm)
        if src_probs is not None:
            loss_KD_fct = MSELoss(reduction="mean")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_probs = probs.view(-1, self.n_labels)[active_loss]
                active_src_probs = src_probs.view(-1, self.n_labels)[active_loss]

                loss_KD = loss_KD_fct(active_probs, active_src_probs)
            else:
                loss_KD = loss_KD_fct(probs, src_probs)

            outputs = (loss_KD,) + outputs

        return outputs  # (loss_KD), (loss), logits


# class LanguageIdentifier(torch.nn.Module):
#     def __init__(self, base_model, output_layer):
#         super(LanguageIdentifier, self).__init__()
#
#         self.base_model = base_model
#         self.output_layer = output_layer
#
# class TaskModel(torch.nn.Module):
#     def __init__(self, base_model, output_layer):
#         super(TaskModel, self).__init__()
#
#         self.base_model = base_model
#         self.output_layer = output_layer


class GRFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from: https://github.com/jvanvugt/pytorch-domain-adaptation
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GRFunction.apply(x, self.lambda_)


class DomainLearner(torch.nn.Module):
    # def __init__(self, domain_vocab_size, domain_hidden_size, feature_hidden_size, low_rank_size, weights_init=None):
    #     super(DomainLearner, self).__init__()
    #
    #     self.domain_embed = torch.nn.Parameter(torch.randn(domain_vocab_size, domain_hidden_size))
    #     if weights_init is not None:
    #         self.domain_embed.data.copy_(weights_init)
    #
    #     self.simU = torch.nn.Linear(feature_hidden_size, low_rank_size)
    #     self.simV = torch.nn.Linear(domain_hidden_size, low_rank_size)
    def __init__(self, domain_vocab_size, hidden_size, low_rank_size, weights_init=None, gamma=0.00001):
        super(DomainLearner, self).__init__()
        self.gamma = gamma
        self.hidden_size = hidden_size

        self.domain_embed = torch.nn.Parameter(torch.randn(domain_vocab_size, hidden_size))
        if weights_init is not None:
            self.domain_embed.data.copy_(weights_init)

        self.simU = torch.nn.Linear(hidden_size, low_rank_size)
        self.simV = torch.nn.Linear(hidden_size, low_rank_size)

    def forward(self, features, labels=None, device="cuda"):

        U_fi = self.simU(features) # batch_size x low_rank_size
        V_mu_all = self.simV(self.domain_embed).transpose(0, 1) # vocab_size x low_rank_size = > low_rank_size x vocab_size

        logits = torch.mm(U_fi, V_mu_all) # batch_size x vocab_size

        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss() # nn.LogSoftmax() + nn.NLLLoss()
            loss_f = loss_fct(logits, labels)

            R = torch.mm(self.domain_embed.transpose(0, 1), self.domain_embed) - torch.eye(self.hidden_size).to(device)
            loss_R = self.gamma * torch.sum(R * R)

            loss = loss_f + loss_R

            outputs = (loss, loss_f, loss_R) + outputs

        return outputs # (loss), logits

    def get_domain_embeds(self):
        return self.domain_embed.detach()

    def get_domain_similarity(self, domain_idx, domain_idy, method="default"):
        if method == "default":
            f_x = self.simU(self.domain_embed[domain_idx]) # low_rank_size
            f_y = self.simV(self.domain_embed[domain_idy]) # low_rank_size

            sim = torch.sum(f_x * f_y).detach().item()
        elif method == "cosine":
            sim = torch.nn.functional.cosine_similarity(self.domain_embed[domain_idx], self.domain_embed[domain_idy],
                                                        dim=0, eps=1e-8)
        elif method == "l2":
            sim = torch.norm(self.domain_embed[domain_idx] - self.domain_embed[domain_idy])
        else:
            sim = -1.0

        return sim