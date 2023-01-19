## API Sequence Generation Training
### annotation
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python single_run.py --max_length 64 --norm True --batch_size 32 --output_dir <output_directory> --epoch 30 --training_file <training_data_path> --valid_file <validation_data_path>
```

### annotation + SO Title
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python dual_run.py --max_length 64 --norm True --batch_size 32 --output_dir <output_directory> --epoch 30 --training_file <training_data_path> --valid_file <validation_data_path>
```

### annotation + SO Title + SO API
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python dual_api_run.py --max_length 64 --norm True --batch_size 32 --output_dir <output_directory> --epoch 30 --training_file <training_data_path> --valid_file <validation_data_path>
```


## Evaluation
### single-annotation
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python evaluation.py --model_type 0 --max_length 64 --load_model_path  <model_path>/pytorch_model.bin --test_filename <test_file_path> --output_dir <output_path>
```

### annotation + SO Title
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python evaluation.py --model_type 1 --max_length 64 --load_model_path  <model_path>/pytorch_model.bin --test_filename <test_file_path> --output_dir <output_path>
```

### annotation + SO Title + SO API
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python evaluate.py --model_type 2 --load_model_path 13-Oct-dual/checkpoint-best-bleu/pytorch_model.bin --test_filename <test_file_path> --output_dir <output_path>
```


## Calculate the results
### single-annotation
```bash
python calculate_bleu_score.py --reference <evaluation_output_path>/test_ref.csv --candidate  <evaluation_output_path>/test_hyp.csv
```
