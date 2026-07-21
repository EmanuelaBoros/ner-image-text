# NER Image Text

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


## Task

The code is designed for token-level NER with CoNLL-style labels:

```text
O
B-PER / I-PER
B-LOC / I-LOC
B-MISC / I-MISC
B-ORG / I-ORG
```

The default loader expects one token per line and a blank line between sentences. The final column is treated as the NER label.

Example:

```text
EU NNP I-NP I-ORG
rejects VBZ I-VP O
German JJ I-NP I-MISC
call NN I-NP O
```

## Data Layout

The training scripts expect a data directory with CoNLL files named by split:

```text
data/conll2003_images/
  train.conll
  dev.conll
  test.conll
```

For each sentence/example, `utils_ner.py` also looks for a corresponding image under:

```text
data/conll2003_images/<split>/doc_<doc_index>_ph_<phrase_index>.final.png
```

For example:

```text
data/conll2003_images/train/doc_1_ph_1.final.png
data/conll2003_images/dev/doc_1_ph_1.final.png
data/conll2003_images/test/doc_1_ph_1.final.png
```

The repository includes `train_small.conll` and `test_small.conll` as sample text files, but it does not include the full generated image folders. To run the default scripts without modifying the loader, prepare the expected `train.conll`, `dev.conll`, and `test.conll` files and matching images.

## Installation

Create an environment with PyTorch and the main Python dependencies:

```bash
pip install torch torchvision transformers seqeval tensorboardX tqdm numpy pillow opencv-python albumentations scikit-image matplotlib einops
```

The image generation utility also depends on the bundled Big Sleep code and its own deep learning stack. Use it separately from model training if you need to generate phrase images.

## Training

A typical training command for `main_single.py` is:

```bash
TOKENIZERS_PARALLELISM=false python main_single.py \
  --data_dir data/conll2003_images/ \
  --output_dir outputs/ner-image-text \
  --do_train \
  --evaluate_during_training \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 4 \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --gpu_ids 0 \
  --logging_steps 100 \
  --overwrite_cache
```

To train the NER-ViT variant:

```bash
TOKENIZERS_PARALLELISM=false python main_nervit.py \
  --data_dir data/conll2003_images/ \
  --output_dir outputs/ner-vit \
  --do_train \
  --evaluate_during_training \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 4 \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --gpu 0 \
  --logging_steps 100 \
  --overwrite_cache
```

Both scripts save model checkpoints, tokenizer files, logs, and training arguments under the selected `--output_dir`.





