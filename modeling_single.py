import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (BigBirdForTokenClassification, 
                          BertPreTrainedModel, RobertaPreTrainedModel, 
                          BertModel, BertForTokenClassification)
from transformers import AutoModelForTokenClassification, AutoModel
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange
from torch import Tensor
import math

num_layers = 2
n_heads = 6
head_dims = 128
d_model = n_heads * head_dims
feedforward_dim = int(2 * d_model)
dropout = 0.45
dropout_attn = 0.0
after_norm = 1
attn_type = 'transformer'
scale = attn_type == 'transformer'
pos_embed = 'sin'
fc_dropout = 0.4

from transformers import ViTFeatureExtractor, ViTModel
from transformers import ImageClassificationPipeline, PerceiverForImageClassificationConvProcessing, PerceiverFeatureExtractor, PerceiverForImageClassificationLearned


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

#      self.init_weights()
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Making the resnet 50 model, which was used in the docformer for the purpose of visual feature extraction
        self.resnet50 = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")
        # self.resnet50 = models.resnet50(pretrained=True)
        for param in self.resnet50.parameters():
            param.requires_grad = True
        
        modules = list(self.resnet50.children())[:-3]
        self.resnet50 = nn.Sequential(*modules)
        for param in self.resnet50.parameters():
            param.requires_grad = True

        # Applying convolution and linear layer

        self.conv1 = nn.Conv2d(224, 768, 1)
        self.relu1 = F.relu
        self.linear1 = nn.Linear(672, 128)

    def forward(self, x):
        x = self.resnet50(x)
        # import pdb;pdb.set_trace()
        x = rearrange(x, "b e w h -> b h w e")
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = rearrange(x, "b e w h -> b e (w h)")  # b -> batch, e -> embedding dim, w -> width, h -> height
        x = self.linear1(x)
        x = rearrange(x, "b e s -> b s e")  # b -> batch, e -> embedding dim, s -> sequence length
        return x

from transformers import ViTFeatureExtractor, ViTModel
from modules.transformer import TransformerEncoder, MultiHeadAttn, TransformerLayer, RelativeMultiHeadAttn
from copy import deepcopy

IMAGE_MODEL = 'google/vit-base-patch16-224'

#class BertForTokenClassification_(BertPreTrainedModel):
class BertForTokenClassification_(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.max_len = 128

        self.bert = BertModel(config, add_pooling_layer=False)

        self.vit = ViTModel.from_pretrained(IMAGE_MODEL)
        self.visual_feature = ResNetFeatureExtractor()

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size * 3, config.num_labels)
        self.aux_classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        # self.classifier = torch.nn.Linear(1300, config.num_labels)

        # self.in_fc = torch.nn.Linear(self.config.hidden_size, d_model)
        # self.transformer = TransformerEncoder(num_layers, d_model, n_heads, feedforward_dim, dropout,
        #                                       after_norm=after_norm, attn_type=attn_type,
        #                                       scale=scale, dropout_attn=dropout_attn,
        #                                       pos_embed=pos_embed)
        # self.fc_dropout = torch.nn.Dropout(fc_dropout)
        # Initialize weights and apply final processing
#        self.post_init()

        attn_type = 'adatrans'
        if attn_type == 'transformer':
            self.self_attention_text = MultiHeadAttn(d_model, n_heads, dropout_attn, scale=scale)
            self.self_attention_image = MultiHeadAttn(d_model, n_heads, dropout_attn, scale=scale)
        elif attn_type == 'adatrans':
            self.self_attention_text = RelativeMultiHeadAttn(d_model, n_heads, dropout_attn, scale=scale)
            self.self_attention_image = RelativeMultiHeadAttn(d_model, n_heads, dropout_attn, scale=scale)
            
        self.text_transformer_layer = TransformerLayer(d_model, deepcopy(self.self_attention_text), feedforward_dim, after_norm, dropout)
        self.image_transformer_layer = TransformerLayer(d_model, deepcopy(self.self_attention_image), feedforward_dim, after_norm, dropout)
        
        self.position_embeddings_v = PositionalEncoding(
            d_model=config.hidden_size * 2,
            dropout=0.1,
            max_len=self.max_len
        )
        
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, input_ids, src_probs=None, attention_mask=None, token_type_ids=None, image_ids=None,
                position_ids=None, head_mask=None, labels=None, loss_ignore_index=-100):
        
      
#        mask = attention_mask.ne(0)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        aux_addon_sequence_encoder = self.self_attention_text(sequence_output, attention_mask)
        aux_addon_sequence_output = aux_addon_sequence_encoder[-1]
        
        text_layer = self.text_transformer_layer(aux_addon_sequence_encoder, attention_mask)

        aux_text_feats = self.aux_classifier(aux_addon_sequence_output)

        image_output = self.visual_feature(image_ids)
        # image_output = image_output['pooler_output']

        aux_addon_image_encoder = self.self_attention_image(image_output, attention_mask)
        aux_addon_image_output = aux_addon_image_encoder[-1]
        
        image_layer = self.image_transformer_layer(aux_addon_image_encoder, attention_mask)

#        sequence_output = self.in_fc(sequence_output)
#        sequence_output = self.transformer(sequence_output, mask)
#        sequence_output = self.fc_dropout(sequence_output)
        # output = torch.cat([image_output['last_hidden_state'], sequence_output], 1)
        # visual_feature = self.visual_feature(image_ids)
        # import pdb;pdb.set_trace()
        
        merge_representation = torch.cat((aux_addon_sequence_encoder, aux_addon_image_encoder), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        gated_converted_att_vis_embed = torch.mul(gate_value, aux_addon_image_encoder)
        
        aux_addon_gated_output = torch.cat((aux_addon_sequence_encoder, gated_converted_att_vis_embed), dim=-1)
        
        # sequence_output = torch.cat([sequence_output, image_output.unsqueeze(1).repeat(1, 128, 1)], 2)
        # output = torch.cat([sequence_output, image_output.unsqueeze(1)], 1)
        # print(output.shape)
        # import pdb;pdb.set_trace()
        final_output = torch.cat((text_layer, image_layer, gated_converted_att_vis_embed), dim=-1)
        
        logits = self.classifier(final_output)

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