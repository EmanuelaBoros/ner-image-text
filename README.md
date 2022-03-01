# ner_image_text

-- no GPU for tests for now

```
TOKENIZERS_PARALLELISM=false python main_single.py --data_dir data/conll2003_images/ 
--output_dir data/conll2003_images/ 
--do_train --evaluate_during_training --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 4 
--model_name_or_path bert-base-cased --gpu_ids 0 --no_cuda --logging_steps 1132
```
