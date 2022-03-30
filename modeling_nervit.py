import torch
import torch.nn as nn

#import layers
from torch.nn import CrossEntropyLoss
from modeling_single import ResNetFeatureExtractor, PositionalEncoding
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertPooler, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaEncoder, RobertaPooler, RobertaModel, RobertaConfig
from transformers import ViTFeatureExtractor, ViTModel

IMAGE_MODEL = 'google/vit-base-patch16-224'

class NERVitBase(RobertaPreTrainedModel):

    def __init__(self, config, max_len=128):
        super().__init__(config)
        self.max_len = max_len

        #self.language_feature = RobertaModel.from_pretrained(config.name_or_path,
        #                                                     output_attentions=config.output_attentions,
        #                                                     output_hidden_states=config.output_hidden_states)
        self.language_feature = BertModel(config, add_pooling_layer=False)
        self.visual_feature = ResNetFeatureExtractor()
        self.visual_transformer = ViTModel.from_pretrained(IMAGE_MODEL)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)

        self.position_embeddings_v = PositionalEncoding(
            d_model=config.hidden_size * 2,
            dropout=0.1,
            max_len=self.max_len
        )

    def forward_base(self, input_ids, src_probs=None, attention_mask=None, token_type_ids=None, image_ids=None,
                     position_ids=None, head_mask=None, loss_ignore_index=-100):

        outputs = self.language_feature(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        image_output = self.visual_transformer(image_ids)
        image_output_pooled = image_output['pooler_output']

        visual_feature = self.visual_feature(image_ids)
        sequence_output = torch.cat([sequence_output, visual_feature], 2)

        return sequence_output


class NERVitModel(NERVitBase):

    def __init__(self, config):
        super().__init__(config)

        #self.classifier = layers.InferenceLayer(config.hidden_size*2, config.num_labels+1, use_crf=True)
        self.num_labels = config.num_labels
        self.classifier = torch.nn.Linear(config.hidden_size * 2, config.num_labels)

    def ner_loss(self, classifier=None, sequence_output=None, labels=None, attention_mask=None):
        if labels is None:
            loss = torch.tensor(0, dtype=torch.float, device=labels.device)
        else:
            if attention_mask is None:
                attention_mask = torch.ones(labels.shape, device=labels.device)

            #loss, logits = classifier(sequence_output, labels, attention_mask)
            logits = classifier(sequence_output)
            loss_fct = CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

            return loss, logits

    def forward(self, input_ids, src_probs=None, attention_mask=None, token_type_ids=None, image_ids=None,
                position_ids=None, head_mask=None, labels=None, loss_ignore_index=-100):

        sequence_output = self.forward_base(input_ids=input_ids, attention_mask=attention_mask,
                                            token_type_ids=token_type_ids, image_ids=image_ids,
                                            position_ids=position_ids, head_mask=head_mask)

        loss, logits = self.ner_loss(classifier=self.classifier, sequence_output=sequence_output,
                                     labels=labels, attention_mask=attention_mask)

        return loss, logits
