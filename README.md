 NER Image Text

Multimodal named entity recognition experiments that combine text representations with visual features generated for entity mentions or phrases.

The project builds on Hugging Face Transformers for token classification and adds image-aware modeling components for experiments where visual context may help named entity recognition. It includes both a text-only baseline-style entry point and multimodal variants that combine BERT/RoBERTa-style language encoders with CNN or ViT-based image features.

## What This Repository Contains

| Path | Purpose |
| --- | --- |
| `main_single.py` | Main training/evaluation script for the image-text NER model built around `BertForTokenClassification_`. |
| `main_nervit.py` | Alternative multimodal NER training script using the `NERVitModel` architecture. |
| `modeling_single.py` | Model definitions for token classification with visual features, attention, and image-text fusion. |
| `modeling_nervit.py` | NER-ViT model definition combining language features with a ViT visual encoder. |
| `utils_ner.py` | CoNLL-style data loading, label handling, tokenization, image loading, and feature conversion. |
| `attentions.py`, `layers.py`, `modules/` | Attention layers and transformer utilities used by the multimodal models. |
| `image_gen/imagine_multiconer.py` | Utility script for generating images from text phrases using Big Sleep. |
| `data/conll2003_images/` | Small CoNLL-style sample files. |

