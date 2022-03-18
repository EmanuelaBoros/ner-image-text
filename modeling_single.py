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
num_layers = 1
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
# from biggan import BigGAN

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

def bi_modal_attention(x, y):
    
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
    {} stands for concatenation
        
    m1 = x . transpose(y) ||  m2 = y . transpose(x) 
    n1 = softmax(m1)      ||  n2 = softmax(m2)
    o1 = n1 . y           ||  o2 = m2 . x
    a1 = o1 * x           ||  a2 = o2 * y
       
    return {a1, a2}
        
    '''
     
    y1 = rearrange(x, "b e s -> b s e")
    m1 = torch.matmul(x, y1)
    n1 = torch.nn.functional.softmax(m1, -1)
    y1 = rearrange(x, "b e s -> b s e")
    
    o1 = torch.matmul(n1, y)
    
    a1 = torch.multiply(o1, x)
    # m1 = torch.dot([x, y], axes=[2, 2])
    # n1 = torch.nn.functional.softmax(m1)
    # o1 = torch.dot([n1, y], axes=[2, 1])
    # a1 = torch.multiply([o1, x])

    m2 = torch.dot([y, x], axes=[2, 2])
    n2 = torch.nn.functional.softmax(m2)
    o2 = torch.dot([n2, x], axes=[2, 1])
    a2 = torch.multiply([o2, y])

    return torch.concatenate([a1, a2])


def self_attention(x):
    
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
        
    m = x . transpose(x)
    n = softmax(m)
    o = n . x  
    a = o * x           
       
    return a
        
    '''
    x1 = rearrange(x, "b e s -> b s e")
    m = torch.matmul(x, x1)
    # m = torch.dot([x, x], axes=[2,2])
    # n = torch.nn.functional.softmax(m)
    n1 = torch.nn.functional.softmax(m, -1)
    o = torch.matmul(n1, x)
    # o = rearrange(o, "b e s -> b s e")
    # o = torch.dot([n, x], axes=[2,1])
    # a = torch.multiply([o, x])
    a1 = torch.multiply(o, x)
        
    return a1

import torchvision

class CNN_Encoder(nn.Module):
    """
    CNN_Encoder.
    """

    # def __init__(self, encoded_image_size=14, attention_method="ByPixel"):
    def __init__(self, max_len=256, encoded_image_size=14, attention_method="ByChannel"):
        super(CNN_Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.attention_method = attention_method

        resnet = torchvision.models.resnet152(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        # Specifically, Remove: AdaptiveAvgPool2d(output_size=(1, 1)), Linear(in_features=2048, out_features=1000, bias=True)]
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        if self.attention_method == "ByChannel":
            self.cnn1 = nn.Conv2d(in_channels=2048, out_channels=768, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.bn1 = nn.BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu = nn.ReLU(inplace=True)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        self.conv1 = nn.Conv2d(2048, 768, 1)
        self.linear1 = nn.Linear(64, max_len)
        
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images [batch_size, encoded_image_size=14, encoded_image_size=14, 2048]
        """
        x = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        if self.attention_method == "ByChannel":  # [batch_size, 2048, 8, 8] -> # [batch_size, 512, 8, 8]
            x = self.relu(self.bn1(self.cnn1(x)))
        
        # import pdb;pdb.set_trace()
        x = rearrange(x, "b e w h -> b e (w h)")
        
        # out = self.adaptive_pool(out)  # [batch_size, 2048/512, 8, 8] -> [batch_size, 2048/512, 14, 14]
        x = self.linear1(x)
        
        x = rearrange(x, "b e s -> b s e")
        # out = out.permute(0, 2, 3, 1)
        return x

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
                
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Making the resnet 50 model, which was used in the docformer for the purpose of visual feature extraction
        # self.resnet50 = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")ResNet101V2
        
        self.resnet101 = models.resnet152(pretrained=False)
        
        # self.resnet50 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        for param in self.resnet101.parameters():
            param.requires_grad = True
        
        modules = list(self.resnet101.children())[:-2]
        self.resnet101 = nn.Sequential(*modules)
        # for param in self.resnet50.parameters():
        #     param.requires_grad = True

        # Applying convolution and linear layer

        self.relu1 = F.relu
        # self.conv1 = nn.Conv2d(800, 800, 1)
        # self.linear1 = nn.Linear(49, 200)
        self.conv1 = nn.Conv2d(2048, 768, 1)
        self.linear1 = nn.Linear(49, 200)
        
    def forward(self, x):

        # import pdb;pdb.set_trace()
        # x = self.resnet50[:-2](x)
        x = self.resnet101(x)
        print(x.shape)
        # import pdb;pdb.set_trace()
        # x = rearrange(x, "b e w h -> b h w e")
        x = self.conv1(x)
        print(x.shape)
        
        x = self.relu1(x)
        x = rearrange(x, "b e w h -> b e (w h)")  # b -> batch, e -> embedding dim, w -> width, h -> height
        print(x.shape)
        
        x = self.linear1(x)
        print(x.shape)
        
        x = rearrange(x, "b e s -> b s e")  # b -> batch, e -> embedding dim, s -> sequence length
        print(x.shape)
        
        return x

    
from transformers import ViTFeatureExtractor, ViTModel
from modules.transformer import (TransformerEncoder, MultiHeadAttn, 
                                 TransformerLayer, RelativeMultiHeadAttn, 
                                 MultiHeadAttnImage, TransformerEncoderImage,
                                 TransformerLayerImage)
from copy import deepcopy
from attentions import  ScaledDotProductAttention, CustomizingAttention, MultiHeadAttention
    
IMAGE_MODEL = 'google/vit-base-patch16-224'

#class BertForTokenClassification_(BertPreTrainedModel):
class BertForTokenClassification_(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.max_len = 256

        self.bert = BertModel(config, add_pooling_layer=False)

        # self.vit = ViTModel.from_pretrained(IMAGE_MODEL)
        # self.visual_feature = ResNetFeatureExtractor()
        self.cnn_encoder = CNN_Encoder(max_len=self.max_len)
        # self.biggan = BigGAN.from_pretrained('biggan-deep-256')

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, config.num_labels)
        self.aux_classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        # self.classifier = torch.nn.Linear(1300, config.num_labels)

        # self.in_fc = torch.nn.Linear(self.config.hidden_size, d_model)
        # self.transformer = TransformerEncoder(num_layers, d_model, n_heads, feedforward_dim, dropout,
        #                                       after_norm=after_norm, attn_type=attn_type,
        #                                       scale=scale, dropout_attn=dropout_attn,
        #                                       pos_embed=pos_embed)
        # self.fc_dropout = torch.nn.Dropout(fc_dropout)
        # Initialize weights and apply final processing
        self.qv_linear2_1 = nn.Linear(d_model, d_model * 2, bias=False)
        self.qv_linear2_2 = nn.Linear(d_model, d_model * 2, bias=False)
        self.qv_linear3 = nn.Linear(d_model, d_model * 3, bias=False)


        attn_type = 'transformer'
        if attn_type == 'transformer':
            self.self_attention_text = MultiHeadAttn(d_model, n_heads, dropout_attn, scale=scale)
            self.self_attention_image = MultiHeadAttn(d_model, n_heads, dropout_attn, scale=scale)
        elif attn_type == 'adatrans':
            self.self_attention_text = RelativeMultiHeadAttn(d_model, n_heads, dropout_attn, scale=scale)
            self.self_attention_image = RelativeMultiHeadAttn(d_model, n_heads, dropout_attn, scale=scale)
        
        self.self_attention_text_image = MultiHeadAttnImage(d_model, n_heads, dropout_attn, scale=scale)
            
        self.text_transformer_layer = TransformerLayer(d_model, deepcopy(self.self_attention_text), feedforward_dim, after_norm, dropout)
        self.image_transformer_layer = TransformerLayer(d_model, deepcopy(self.self_attention_image), feedforward_dim, after_norm, dropout)
        
        
        # self.text_encoder = TransformerEncoder(num_layers, d_model, n_heads, feedforward_dim, dropout)
        self.image_encoder = TransformerEncoder(num_layers, d_model, n_heads, feedforward_dim, dropout)
        
        self.text_image_encoder = TransformerEncoderImage(num_layers, d_model, n_heads, feedforward_dim, dropout)
        self.text_encoder = TransformerEncoderImage(num_layers, d_model, n_heads, feedforward_dim, dropout)
        
        self.simple_text_image_encoder = TransformerEncoder(num_layers, d_model, n_heads, feedforward_dim, dropout)
        
        self.text_image_transformer_layer = TransformerLayerImage(d_model, deepcopy(self.self_attention_text_image), feedforward_dim, after_norm, dropout)
        
        self.position_embeddings_v = PositionalEncoding(
            d_model=config.hidden_size * 2,
            dropout=0.1,
            max_len=self.max_len
        )
        
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.scaled_attention = ScaledDotProductAttention(dim=config.hidden_size)
        self.multihead_attention = MultiHeadAttention(d_model, n_heads)
        self.customizing_attention = CustomizingAttention(hidden_dim = config.hidden_size, num_heads = 4, conv_out_channel = 10)
        
        self.gate_image_text = nn.Linear(self.max_len * 2, self.max_len)
        self.relu1 = F.relu
        # self.linear1 = nn.Linear(672, 128)
        
        self.conv1 = nn.Conv1d(self.max_len * 2, self.max_len, 1)
                
        encoded_image_size = 14
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.decode_step = nn.LSTMCell(config.hidden_size, 200, bias=True)  #
        
    def forward(self, input_ids, src_probs=None, attention_mask=None, image_attention_mask=None, token_type_ids=None, image_ids=None,
                position_ids=None, head_mask=None, labels=None, loss_ignore_index=-100):
        
      
#        mask = attention_mask.ne(0)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # image_output = self.visual_feature(image_ids)
        # image_encoded = self.image_encoder(image_output, image_attention_mask)
        
        image_encoded = self.cnn_encoder(image_ids)
        truncation = 0.4

        # import pdb;pdb.set_trace()
        # self.biggan(image_encoded, sequence_output, truncation)
        # self.biggan(image_encoded, rearrange(sequence_output, "b e s -> b s e"), truncation)
        
        ### change weight in regards to the image relevance
        q = image_encoded
        
        kv = self.qv_linear2_1(sequence_output)
        k, v = torch.chunk(kv, chunks=2, dim=-1)
        image_text, weighted_attns, alphas = self.text_image_encoder(image_encoded, image_attention_mask, q, k, v)
        image_text_alphas = self.adaptive_pool(alphas[0])
        
        q = sequence_output

        kv = self.qv_linear2_2(image_encoded)
        k, v = torch.chunk(kv, chunks=2, dim=-1)
        text_image, _, _ = self.text_image_encoder(sequence_output, attention_mask, q, k, v)
        text_image_alphas = self.adaptive_pool(alphas[0])


        # qkv = self.qv_linear3(sequence_output)
        # q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        # text, text_weighted_attns, text_alphas = self.text_encoder(sequence_output, attention_mask, q, k, v)
        # text_alphas = self.adaptive_pool(text_alphas[0])
        
        # qkv = self.qv_linear3(image_encoded)
        # q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        # image, image_weighted_attns, image_alphas = self.text_encoder(image_encoded, image_attention_mask, q, k, v)
        # image_alphas = self.adaptive_pool(image_alphas[0])
        
        # alphas = alphas + image_alphas
        alphas = image_text_alphas
        
        logits = self.classifier(torch.cat((image_text, text_image), dim=-1))

        # aux_addon_sequence_encoder = self.self_attention_text(sequence_output, attention_mask)
        # aux_addon_sequence_output = aux_addon_sequence_encoder[-1]
        # q, k, v x = rearrange(x, "b e s -> b s e")
        # import pdb;pdb.set_trace()
        # scores = F.log_softmax(logits, dim=1)
        
        # import pdb;pdb.set_trace()
        
        # image_path = ''
        # rev_word_map = {}
        # # import pdb;pdb.set_trace()
        # visualize_att(image_path, sequence_output, alphas, rev_word_map, smooth=True)
        # # text_layer = self.text_transformer_layer(aux_addon_sequence_encoder, attention_mask)
        # text_layer = self.text_transformer_layer(sequence_output, attention_mask)

        # aux_text_feats = self.aux_classifier(aux_addon_sequence_output)

        # text_layer = self.text_image_transformer_layer(sequence_output, attention_mask)
        # image_layer = self.text_image_transformer_layer(sequence_output, image_attention_mask)
        # text_image = torch.cat((sequence_output, image_output), dim=1)
        # text_image_mask = torch.cat((attention_mask, image_attention_mask), dim=1)
        
        # text_image = self.simple_text_image_encoder(text_image, text_image_mask)
        # text_image = self.conv1(text_image)
        # text_image = self.relu1(text_image)
        
        # import pdb;pdb.set_trace()
        # text_image = rearrange(text_image, "b e s -> b s e")
        # text_image = self.gate_image_text(text_image)
        
        # text_image = rearrange(text_image, "b e s -> b s e")

        # import pdb;pdb.set_trace()
        # self.simple_text_image_encoder
        # context, attn = self.multihead_attention(image_layer, text_layer, sequence_output)

        # import pdb;pdb.set_trace()
        # image_output = image_output['pooler_output']
        
        # final_output = torch.cat((text_layer, text_image_layer), dim=-1)
        
        # aux_addon_image_encoder = self.self_attention_image(image_output, image_attention_mask)
        # aux_addon_image_output = aux_addon_image_encoder[-1]
        
        # image_layer = self.image_transformer_layer(aux_addon_image_encoder, image_attention_mask)

# #        sequence_output = self.in_fc(sequence_output)
# #        sequence_output = self.transformer(sequence_output, mask)
# #        sequence_output = self.fc_dropout(sequence_output)
#         # output = torch.cat([image_output['last_hidden_state'], sequence_output], 1)
#         # visual_feature = self.visual_feature(image_ids)
#         # import pdb;pdb.set_trace()
        
#         merge_representation = torch.cat((aux_addon_sequence_encoder, aux_addon_image_encoder), dim=-1)
#         gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
#         gated_converted_att_vis_embed = torch.mul(gate_value, aux_addon_image_encoder)
        
#         # aux_addon_gated_output = torch.cat((aux_addon_sequence_encoder, gated_converted_att_vis_embed), dim=-1)
        
#         # sequence_output = torch.cat([sequence_output, image_output.unsqueeze(1).repeat(1, 128, 1)], 2)
#         # output = torch.cat([sequence_output, image_output.unsqueeze(1)], 1)
#         # print(output.shape)
        # import pdb;pdb.set_trace()
        
        # final_output = torch.cat((sequence_output, image_output), dim=-1)
#         final_output = torch.cat((text_layer, image_layer, gated_converted_att_vis_embed), dim=-1)
        
        # final_output = torch.cat((text_layer, image_layer), dim=-1)
        # final_mask = torch.cat((attention_mask, image_attention_mask), dim=-1)

        # import pdb;pdb.set_trace()
        # final_output = self.self_attention_text_image(final_output, final_mask)
        # print(final_output.shape)
        # final_output = self.self_attention_text_image(final_output)
        
        

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

        return outputs, weighted_attns, alphas   # (loss_KD), (loss), scores, (hidden_states), (attentions)




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