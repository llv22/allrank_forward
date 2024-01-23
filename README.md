# allRank_forward : Learning to Rank in PyTorch (forked from [https://github.com/allegro/allRank](https://github.com/allegro/allRank))

## About

allRank is a PyTorch-based framework for training neural Learning-to-Rank (LTR) models, featuring implementations of:

* common pointwise, pairwise and listwise loss functions
* fully connected and Transformer-like scoring functions
* commonly used evaluation metrics like Normalized Discounted Cumulative Gain (NDCG) and Mean Reciprocal Rank (MRR)
* click-models for experiments on simulated click-through data

We forked the original repository and added the following modifications:

* support AP@K, precision@K
* support {metric}_None to support get MRR, P
* add tracing for failure and success cases (still refine code to dynamical determine the threshold)
* fix the bug of MRR
* support statistics report on the test set

### Motivation

allRank provides an easy and flexible way to experiment with various LTR neural network models and loss functions.
It is easy to add a custom loss, and to configure the model and the training procedure.  
We hope that allRank will facilitate both research in neural LTR and its industrial applications.

## Features

### Implemented loss functions:  

 1. ListNet (for binary and graded relevance)
 2. ListMLE
 3. RankNet
 4. Ordinal loss
 5. LambdaRank
 6. LambdaLoss
 7. ApproxNDCG
 8. RMSE
 9. NeuralNDCG (introduced in https://arxiv.org/pdf/2102.07831)

### Getting started guide

To help you get started, we provide a ```run_example.sh``` script which generates dummy ranking data in libsvm format and trains
 a Transformer model on the data using provided example ```config.json``` config file. Once you run the script, the dummy data can be found in `dummy_data` directory
 and the results of the experiment in `test_run` directory. To run the example, Docker is required.

### Getting the right architecture version (GPU vs CPU-only)

Since torch binaries are different for GPU and CPU and GPU version doesn't work on CPU - one must select & build appropriate docker image version.

To do so pass `gpu` or `cpu` as `arch_version` build-arg in 

```docker build --build-arg arch_version=${ARCH_VERSION}```

When calling `run_example.sh` you can select the proper version by a first cmd line argument e.g. 

```run_example.sh gpu ...```

with `cpu` being the default if not specified.

### Configuring your model & training

To train your own model, configure your experiment in ```config.json``` file and run  

```python allrank/main.py --config_file_name allrank/config.json --run_id <the_name_of_your_experiment> --job_dir <the_place_to_save_results>```

All the hyperparameters of the training procedure: i.e. model defintion, data location, loss and metrics used, training hyperparametrs etc. are controlled
by the ```config.json``` file. We provide a template file ```config_template.json``` where supported attributes, their meaning and possible values are explained.
 Note that following MSLR-WEB30K convention, your libsvm file with training data should be named `train.txt`. You can specify the name of the validation dataset 
 (eg. valid or test) in the config. Results will be saved under the path ```<job_dir>/results/<run_id>```
 
Google Cloud Storage is supported in allRank as a place for data and job results.


### Implementing custom loss functions

To experiment with your own custom loss, you need to implement a function that takes two tensors (model prediction and ground truth) as input
 and put it in the `losses` package, making sure it is exposed on a package level.
To use it in training, simply pass the name (and args, if your loss method has some hyperparameters) of your function in the correct place in the config file:

```json
"loss": {
    "name": "yourLoss",
    "args": {
        "arg1": val1,
        "arg2: val2
    }
  }
```

### Applying click-model

To apply a click model you need to first have an allRank model trained.
Next, run:

```python allrank/rank_and_click.py --input-model-path <path_to_the_model_weights_file> --roles <comma_separated_list_of_ds_roles_to_process e.g. train,valid> --config_file_name allrank/config.json --run_id <the_name_of_your_experiment> --job_dir <the_place_to_save_results>``` 

The model will be used to rank all slates from the dataset specified in config. Next - a click model configured in config will be applied and the resulting click-through dataset will be written under ```<job_dir>/results/<run_id>``` in a libSVM format.
The path to the results directory may then be used as an input for another allRank model training.

## Continuous integration

You should run `scripts/ci.sh` to verify that code passes style guidelines and unit tests.

## Research

This framework was developed to support the research project [Context-Aware Learning to Rank with Self-Attention](https://arxiv.org/abs/2005.10084). If you use allRank in your research, please cite:
```
@article{Pobrotyn2020ContextAwareLT,
  title={Context-Aware Learning to Rank with Self-Attention},
  author={Przemyslaw Pobrotyn and Tomasz Bartczak and Mikolaj Synowiec and Radoslaw Bialobrzeski and Jaroslaw Bojar},
  journal={ArXiv},
  year={2020},
  volume={abs/2005.10084}
}
```

Additionally, if you use the NeuralNDCG loss function, please cite the corresponding work, [NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting](https://arxiv.org/abs/2102.07831):
```
@article{Pobrotyn2021NeuralNDCG,
  title={NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting},
  author={Przemyslaw Pobrotyn and Radoslaw Bialobrzeski},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.07831}
}
```

* Visual features# allRank : Learning to Rank in PyTorch

## About

allRank is a PyTorch-based framework for training neural Learning-to-Rank (LTR) models, featuring implementations of:

* common pointwise, pairwise and listwise loss functions
* fully connected and Transformer-like scoring functions
* commonly used evaluation metrics like Normalized Discounted Cumulative Gain (NDCG) and Mean Reciprocal Rank (MRR)
* click-models for experiments on simulated click-through data

### Motivation

allRank provides an easy and flexible way to experiment with various LTR neural network models and loss functions.
It is easy to add a custom loss, and to configure the model and the training procedure.  
We hope that allRank will facilitate both research in neural LTR and its industrial applications.

## Features

### Implemented loss functions:  

 1. ListNet (for binary and graded relevance)
 2. ListMLE
 3. RankNet
 4. Ordinal loss
 5. LambdaRank
 6. LambdaLoss
 7. ApproxNDCG
 8. RMSE
 9. NeuralNDCG (introduced in https://arxiv.org/pdf/2102.07831)

### Getting started guide

To help you get started, we provide a ```run_example.sh``` script which generates dummy ranking data in libsvm format and trains
 a Transformer model on the data using provided example ```config.json``` config file. Once you run the script, the dummy data can be found in `dummy_data` directory
 and the results of the experiment in `test_run` directory. To run the example, Docker is required.

### Getting the right architecture version (GPU vs CPU-only)

Since torch binaries are different for GPU and CPU and GPU version doesn't work on CPU - one must select & build appropriate docker image version.

To do so pass `gpu` or `cpu` as `arch_version` build-arg in 

```docker build --build-arg arch_version=${ARCH_VERSION}```

When calling `run_example.sh` you can select the proper version by a first cmd line argument e.g. 

```run_example.sh gpu ...```

with `cpu` being the default if not specified.

### Configuring your model & training

To train your own model, configure your experiment in ```config.json``` file and run  

```python allrank/main.py --config_file_name allrank/config.json --run_id <the_name_of_your_experiment> --job_dir <the_place_to_save_results>```

All the hyperparameters of the training procedure: i.e. model defintion, data location, loss and metrics used, training hyperparametrs etc. are controlled
by the ```config.json``` file. We provide a template file ```config_template.json``` where supported attributes, their meaning and possible values are explained.
 Note that following MSLR-WEB30K convention, your libsvm file with training data should be named `train.txt`. You can specify the name of the validation dataset 
 (eg. valid or test) in the config. Results will be saved under the path ```<job_dir>/results/<run_id>```
 
Google Cloud Storage is supported in allRank as a place for data and job results.


### Implementing custom loss functions

To experiment with your own custom loss, you need to implement a function that takes two tensors (model prediction and ground truth) as input
 and put it in the `losses` package, making sure it is exposed on a package level.
To use it in training, simply pass the name (and args, if your loss method has some hyperparameters) of your function in the correct place in the config file:

```json
"loss": {
    "name": "yourLoss",
    "args": {
        "arg1": val1,
        "arg2: val2
    }
  }
```

### Applying click-model

To apply a click model you need to first have an allRank model trained.
Next, run:

```python allrank/rank_and_click.py --input-model-path <path_to_the_model_weights_file> --roles <comma_separated_list_of_ds_roles_to_process e.g. train,valid> --config_file_name allrank/config.json --run_id <the_name_of_your_experiment> --job_dir <the_place_to_save_results>``` 

The model will be used to rank all slates from the dataset specified in config. Next - a click model configured in config will be applied and the resulting click-through dataset will be written under ```<job_dir>/results/<run_id>``` in a libSVM format.
The path to the results directory may then be used as an input for another allRank model training.

## Continuous integration

You should run `scripts/ci.sh` to verify that code passes style guidelines and unit tests.

## Research

This framework was developed to support the research project [Context-Aware Learning to Rank with Self-Attention](https://arxiv.org/abs/2005.10084). If you use allRank in your research, please cite:
```
@article{Pobrotyn2020ContextAwareLT,
  title={Context-Aware Learning to Rank with Self-Attention},
  author={Przemyslaw Pobrotyn and Tomasz Bartczak and Mikolaj Synowiec and Radoslaw Bialobrzeski and Jaroslaw Bojar},
  journal={ArXiv},
  year={2020},
  volume={abs/2005.10084}
}
```

Additionally, if you use the NeuralNDCG loss function, please cite the corresponding work, [NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting](https://arxiv.org/abs/2102.07831):
```
@article{Pobrotyn2021NeuralNDCG,
  title={NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting},
  author={Przemyslaw Pobrotyn and Radoslaw Bialobrzeski},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.07831}
}
```

## Reproduce on Set1-5 on dataset/MSLR-WEB30K based on allRank

Refer to

1. [NDCG](https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0)
2. [MAP](https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map)
3. [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
4. [MAP@K](https://towardsdatascience.com/mean-average-precision-at-k-map-k-clearly-explained-538d8e032d2)
5. [torch.gather](https://cloud.tencent.com/developer/article/1761613) and [torch.gather official guide](https://pytorch.org/docs/stable/generated/torch.gather.html)
6. [Markdown table generator](https://www.tablesgenerator.com/markdown_tables)

### Prepare feature for MSLR-WEB30K

```bash
conda activate metaGUI
python reproducibility/normalize_features.py --ds_path dataset/MSLR-WEB30K/Fold1
python reproducibility/normalize_features.py --ds_path dataset/MSLR-WEB30K/Fold2
python reproducibility/normalize_features.py --ds_path dataset/MSLR-WEB30K/Fold3
python reproducibility/normalize_features.py --ds_path dataset/MSLR-WEB30K/Fold4
python reproducibility/normalize_features.py --ds_path dataset/MSLR-WEB30K/Fold5
```

### Run ranallrank_NeuralNDCG

```bash
conda activate metaGUI
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=1 python allrank/main.py  --config-file-name allrank/settings/neuralndcg_atmax_Fold1_normalized.json --run-id ranallrank_NeuralNDCG_Fold1_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold1_normalized # ongoing
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/settings/neuralndcg_atmax_Fold2_normalized.json --run-id ranallrank_NeuralNDCG_Fold2_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold2_normalized # DONE
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/settings/neuralndcg_atmax_Fold3_normalized.json --run-id ranallrank_NeuralNDCG_Fold3_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold3_normalized # DONE
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/settings/neuralndcg_atmax_Fold4_normalized.json --run-id ranallrank_NeuralNDCG_Fold4_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold4_normalized # DONE
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/settings/neuralndcg_atmax_Fold5_normalized.json --run-id ranallrank_NeuralNDCG_Fold5_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold5_normalized # DONE
```

NeuralNDCG on MSLR-WEB30K Fold1 to Fold5

| Dataset | Train Loss          | Train NDCG@1       | Train NDCG@5       | Train NDCG@10      | Train NDCG@30      | Train NDCG@60      | Val Loss            | Val NDCG@1         | Val NDCG@5         | Val NDCG@10        | Val NDCG@30        | Val NDCG@60        |
|---------|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| Fold1   | -0.7669206603150948 | 0.6319156289100647 | 0.5951358675956726 | 0.595418393611908  | 0.6312896013259888 | 0.6902117133140564 | -0.7294725948502213 | 0.517143189907074  | 0.5089449882507324 | 0.5275285840034485 | 0.5840967297554016 | 0.6481217741966248 |
| Fold2   | -0.7665537890652948 | 0.6281455755233765 | 0.5941371917724609 | 0.5944378972053528 | 0.6313827037811279 | 0.6902371048927307 | -0.7269774124779981 | 0.5167470574378967 | 0.5107281804084778 | 0.527868390083313  | 0.5821496248245239 | 0.6455816030502319 |
| Fold3   | -0.7668627891849042 | 0.6241025328636169 | 0.5938913822174072 | 0.5951226353645325 | 0.6322952508926392 | 0.6909143924713135 | -0.732415835629786  | 0.532313883304596  | 0.5211151838302612 | 0.5354329347610474 | 0.5869977474212646 | 0.6501902341842651 |
| Fold4   | -0.7662994338049748 | 0.6262935996055603 | 0.5946605801582336 | 0.5965979099273682 | 0.6328622698783875 | 0.6911565065383911 | -0.7308251556804208 | 0.5148641467094421 | 0.5154050588607788 | 0.5328453779220581 | 0.5857040286064148 | 0.6489840745925903 |
| Fold5   | -0.7675414259737834 | 0.6310990452766418 | 0.5953048467636108 | 0.5956495404243469 | 0.632238507270813  | 0.6907401084899902 | -0.727307596510643  | 0.5146557092666626 | 0.5060291290283203 | 0.5266775488853455 | 0.5825557708740234 | 0.646621823310852  |

### Feature extraction for execution models

* Text features of web page, search query, instruction from the web page

| Text Feature                              | Id | Definition | Supported by Data |
|-------------------------------------------|----|------------|-------------------|
| covered query term ratio in instructions  | 1  |            |                   |
| title_search_query_similarity             | 2  |            |                   |
| keyword_hitting_ratio                     | 3  |            |                   |

* Visual features

| Visual Feature                                     | Id | Definition                                                                                | Supported by Data |
|----------------------------------------------------|----|-------------------------------------------------------------------------------------------|-------------------|
| Instructions completion degree                     | 4  |                                                                                           | 1, unfinished     |
| Action term ratio in instruction (avg)             | 5  | percentage of query term appearing in instructions                                        | 1                 |
| Action term ratio in instruction (min)             | 6  |                                                                                           | 1                 |
| Action term ratio in instruction (max)             | 7  |                                                                                           | 1                 |
| Action term ratio in instruction (var)             | 8  |                                                                                           | 1                 |
| UI text term matching in instruction (avg)         | 9  | Whether ui text appearing in instructions                                                 | 1                 |
| UI text term matching in instruction (min)         | 10 |                                                                                           | 1                 |
| UI text term matching in instruction (max)         | 11 |                                                                                           | 1                 |
| UI text term matching in instruction (var)         | 12 |                                                                                           | 1                 |
| Match UI text term frequency in control list (avg) | 13 | percentage of control with ui text in current instruction                                 | 1                 |
| Match UI text term frequency in control list (min) | 14 |                                                                                           | 1                 |
| Match UI text term frequency in control list (max) | 15 |                                                                                           | 1                 |
| Match UI text term frequency in control list (var) | 16 |                                                                                           | 1                 |
| The position of the last instruction term          | 17 | relative position of the lastly matched instruction term                                  | 1                 |
| The moving distancing of instruction terms         | 18 | distance from the nearest matched instruction term to the farest matched instruction term | 1                 |

* Temporal disabled features

#### Text Feature

| Text Feature                                          | Id | Definition | Formula |
|-------------------------------------------------------|----|------------|---------|
| attributing instruction coverage ratio                |    |            |         |
| covered query term number in instructions             |    |            |         |
| covered query term number (title)                     |    |            |         |
| covered query term number (body)                      |    |            |         |
| covered query term ratio (title)                      |    |            |         |
| covered query term ratio (body)                       |    |            |         |
| term frequency (min)                                  |    |            |         |
| term frequency (max)                                  |    |            |         |
| term frequency (sum)                                  |    |            |         |
| term frequency (variance)                             |    |            |         |
| similarity between action description and instruction |    |            |         |
| attributing instruction coverage ratio                |    |            |         |

#### Visual Feature

| Visual Feature                         | Id | Definition                                                        | Supported by Data |
|----------------------------------------|----|-------------------------------------------------------------------|-------------------|
| Step repetition degree                 |    | how many time the same UI appears                                 |                   |
| Step disorder degree                   |    | assign UI with order Id, how many disorder exist                  |                   |
| Action term ratio in instruction (sum) |    |                                                                   |                   |
| Stepping MAE of instruction term (avg) |    | whether each step is moving from previous instruction to the next | 1, disordered     |
| Stepping MAE of instruction term (min) |    |                                                                   | 1, disordered     |
| Stepping MAE of instruction term (max) |    |                                                                   | 1, disordered     |
| Stepping MAE of instruction term (sum) |    |                                                                   | 1, disordered     |
| Stepping MAE of instruction term (var) |    |                                                                   | 1, disordered     |

### ranallrank_NeuralNDCG given statistics of execution models

```bash
conda activate metaGUI
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Multimodal_Feature18_label3.json --run-id ranallrank_NeuralNDCG_mm_Label3_Feature18 --job-dir experiments/ranallrank_NeuralNDCG_mm_Label3_Feature18 #DONE
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=1 python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Multimodal_Feature18_label2.json --run-id ranallrank_NeuralNDCG_mm_Label2_Feature18 --job-dir experiments/ranallrank_NeuralNDCG_mm_Label2_Feature18 #DONE
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Multimodal_Feature18_label2_on_ground_truth.json --run-id ranallrank_NeuralNDCG_mm_Label2_Feature18_on_ground_truth --job-dir experiments/ranallrank_NeuralNDCG_mm_Label2_Feature18_on_ground_truth #DONE
```

* NeuralNDCG on Train/Val dataset

Case 1: From the same data distribution

| Dataset | Train Loss          | Train NDCG@1       | Train NDCG@5       | Train NDCG@10      | Train NDCG@30      | Train NDCG@60      | Val Loss            | Val NDCG@1         | Val NDCG@5         | Val NDCG@10        | Val NDCG@30        | Val NDCG@60        |
|---------|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| Label3  | -0.8995515780741942 | 0.8795496821403503 | 0.9445069432258606 | 0.9503862261772156 | 0.9515350461006165 | 0.9515953063964844 | -0.568975856273471  | 0.8961373567581177 | 0.9549955725669861 | 0.9585631489753723 | 0.959588348865509  | 0.9597274661064148 |
| Label2  | -0.8606304213653889 | 0.9028130173683167 | 0.9484454989433289 | 0.9527173638343811 | 0.9535864591598511 | 0.9536436200141907 | -0.5326286527408298 | 0.9087750315666199 | 0.9546259045600891 | 0.9579418897628784 | 0.9588051438331604 | 0.9588857889175415 |

| Dataset | Train Loss | Train NDCG@1 | Train NDCG@5 | Train NDCG@10 | Train MRR@1 | Train MRR@5 | Train MRR@10 | Train AP@1 | Train AP@5 | Train AP@10 | Val Loss  | Val NDCG@1 | Val NDCG@5 | Val NDCG@10 | Val MRR@1 | Val MRR@5 | Val MRR@10 | Val AP@1 | Val AP@5 | Val AP@10 |
|---------|------------|--------------|--------------|---------------|-------------|-------------|--------------|------------|------------|-------------|-----------|------------|------------|-------------|-----------|-----------|------------|----------|----------|-----------|
| Label2  | -0.765341  | 0.6223       | 0.5927       | 0.5944        | 0.4951      | 0.5902      | 0.6000       | 0.8431     | 0.8597     | 0.8217      | -0.733317 | 0.5179     | 0.5167     | 0.5343      | 0.3880    | 0.4892    | 0.5031     | 0.7766   | 0.8131   | 0.7819    |

* Convergence Analysis

1. Label 3 - From Epoch 0 to Epoch 0, then keep the best result
2. Label 2 - From Epoch 0 to Epoch 2, then keep the best result

Case 2: From the different data distribution (Train from metaGUI dataset, Val from WeCollect dataset)

| Dataset                           | Train Loss | Train NDCG@1 | Train NDCG@5 | Train NDCG@10 | Train MRR@1 | Train MRR@5 | Train MRR@10 | Train AP@1 | Train AP@5 | Train AP@10 | Val Loss  | Val NDCG@1 | Val NDCG@5 | Val NDCG@10 | Val MRR@1 | Val MRR@5 | Val MRR@10 | Val AP@1 | Val AP@5 | Val AP@10 |
|-----------------------------------|------------|--------------|--------------|---------------|-------------|-------------|--------------|------------|------------|-------------|-----------|------------|------------|-------------|-----------|-----------|------------|----------|----------|-----------|
| Label2 (168 Ground-Truth as test) | -0.859941  | 0.9027       | 0.9484       | 0.9529        | 0.9027      | 0.9384      | 0.9392       | 0.2390     | 0.2723     | 0.2725      | -0.532629 | 0.9088     | 0.9546     | 0.9579      | 0.5195    | 0.5557    | 0.5560     | 0.2928   | 0.3274   | 0.3269    |

| Dataset                           | Test NDCG@1 | Test NDCG@5 | Test NDCG@10 | Test MRR@1 | Test MRR@5 | Test MRR@10 | Test AP@1  | Test AP@5  | Test AP@10 |
|-----------------------------------|-------------|-------------|--------------|------------|------------|-------------|------------|------------|------------|
| Label2 (168 Ground-Truth as test) | 0.90625     | 0.94897854  | 0.94897854   | 0.90625    | 0.90625    | 0.94921875  | 0.94921875 | 0.94921875 | 0.94921875 |

* Convergence Analysis

1. Label 2 - From Epoch 0 to Epoch 3, then keep the best result

* ApproxNDCG on Train/Val dataset

Case 3: From the different data distribution (Train from metaGUI dataset, Val from WeCollect dataset)

| Dataset                           | Train Loss | Train NDCG@1 | Train NDCG@5 | Train NDCG@10 | Train MRR@1 | Train MRR@5 | Train MRR@10 | Train AP@1 | Train AP@5 | Train AP@10 | Val Loss  | Val NDCG@1 | Val NDCG@5 | Val NDCG@10 | Val MRR@1 | Val MRR@5 | Val MRR@10 | Val AP@1 | Val AP@5 | Val AP@10 |
|-----------------------------------|------------|--------------|--------------|---------------|-------------|-------------|--------------|------------|------------|-------------|-----------|------------|------------|-------------|-----------|-----------|------------|----------|----------|-----------|
| Label2 (168 Ground-Truth as test) | -0.334683  | 0.9978       | 0.9991       | 0.9991        | 0.9982      | 0.9991      | 0.9991       | 0.3344     | 0.3350     | 0.3350      | -0.357475 | 0.9991     | 0.9993     | 0.9996      | 0.6099    | 0.6102    | 0.6102     | 0.3831   | 0.3834   | 0.3833    |

| Dataset                           | Test NDCG@1 | Test NDCG@5 | Test NDCG@10 | Test MRR@1 | Test MRR@5 | Test MRR@10 | Test AP@1 | Test AP@5 | Test AP@10 |
|-----------------------------------|-------------|-------------|--------------|------------|------------|-------------|-----------|-----------|------------|
| Label2 (168 Ground-Truth as test) | 0.875       | 0.9346039   | 0.94449234   | 0.875      | 0.9375     | 0.9375      | 0.78125   | 0.8275824 | 0.82520723 |

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=2 python allrank/main.py  --config-file-name allrank/approxndc_Multimodal_Feature18_label2_on_ground_truth.json --run-id approxndc_Multimodal_Feature18_label2_on_ground_truth --job-dir experiments/approxndc_Multimodal_Feature18_label2_on_ground_truth #DONE
```

* Convergence Analysis

1. Label 2 - From Epoch 0 to Epoch 3, then keep the best result

* LambdaRank on Train/Val dataset

Case 4: From the different data distribution (Train from metaGUI dataset, Val from WeCollect dataset)

| Dataset                           | Train Loss | Train NDCG@1 | Train NDCG@5 | Train NDCG@10 | Train MRR@1 | Train MRR@5 | Train MRR@10 | Train AP@1 | Train AP@5 | Train AP@10 | Val Loss | Val NDCG@1 | Val NDCG@5 | Val NDCG@10 | Val MRR@1 | Val MRR@5 | Val MRR@10 | Val AP@1 | Val AP@5 | Val AP@10 |
|-----------------------------------|------------|--------------|--------------|---------------|-------------|-------------|--------------|------------|------------|-------------|----------|------------|------------|-------------|-----------|-----------|------------|----------|----------|-----------|
| Label2 (168 Ground-Truth as test) | 0.254771   | 0.9995       | 0.9998       | 0.9998        | 0.9997      | 0.9999      | 0.9999       | 0.3360     | 0.3361     | 0.3361      | 0.222945 | 1.0000     | 1.0000     | 1.0000      | 0.6108    | 0.6108    | 0.6108     | 0.3840   | 0.3840   | 0.3840    |

| Dataset                           | Test NDCG@1 | Test NDCG@5 | Test NDCG@10 | Test MRR@1 | Test MRR@5 | Test MRR@10 | Test AP@1 | Test AP@5  | Test AP@10 |
|-----------------------------------|-------------|-------------|--------------|------------|------------|-------------|-----------|------------|------------|
| Label2 (168 Ground-Truth as test) | 0.859375    | 0.9340519   | 0.9457377    | 0.859375   | 0.9296875  | 0.9296875   | 0.765625  | 0.83305126 | 0.82924366 |

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=3 python allrank/main.py  --config-file-name allrank/lambdarank_atmax_Multimodal_Feature18_label2_on_ground_truth.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_ground_truth --job-dir experiments/lambdarank_atmax_Multimodal_Feature18_label2_on_ground_truth #ongoing
```

* Convergence Analysis

1. Label 2 - From Epoch 0 to Epoch 26(0.9501168727874756) Epoch 78(0.9514542818069458), then keep the best result

* ListNet loss by NeuralNDCG on Train/Val dataset

Case 5: From the different data distribution (Train from metaGUI dataset, Val from WeCollect dataset)

| Dataset                           | Train Loss | Train NDCG@1 | Train NDCG@5 | Train NDCG@10 | Train MRR@1 | Train MRR@5 | Train MRR@10 | Train AP@1 | Train AP@5 | Train AP@10 | Val Loss | Val NDCG@1 | Val NDCG@5 | Val NDCG@10 | Val MRR@1 | Val MRR@5 | Val MRR@10 | Val AP@1 | Val AP@5 | Val AP@10 |
|-----------------------------------|------------|--------------|--------------|---------------|-------------|-------------|--------------|------------|------------|-------------|----------|------------|------------|-------------|-----------|-----------|------------|----------|----------|-----------|
| Label2 (168 Ground-Truth as test) | 0.000000   | 1.0000       | 1.0000       | 1.0000        | 1.0000      | 1.0000      | 1.0000       | 0.3363     | 0.7679     | 0.9850      | 0.512866 | 0.9444     | 0.9737     | 0.9752      | 0.5552    | 0.5799    | 0.5801     | 0.3284   | 0.3504   | 0.3494    |

| Dataset                           | Test NDCG@1 | Test NDCG@5 | Test NDCG@10 | Test MRR@1 | Test MRR@5 | Test MRR@10 | Test AP@1 | Test AP@5  | Test AP@10 |
|-----------------------------------|-------------|-------------|--------------|------------|------------|-------------|-----------|------------|------------|
| Label2 (168 Ground-Truth as test) | 0.828125    | 0.9209986   | 0.92807746    | 0.828125   | 0.9067709  | 0.9067709   | 0.734375  | 0.8098959 | 0.80627537 |

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=3 python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_ground_truth --job-dir experiments/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_ground_truth #ongoing
```

* Convergence Analysis

1. Label 2 - From Epoch 0 to Epoch 26(0.9501168727874756) Epoch 78(0.9514542818069458), then keep the best result

* Cohere Re-ranking result

```bash
ln -s /data/orlando/workspace/metaGUI_forward/repo_upload/ground_truth/rerank_input_raw_data_label2.txt /data/orlando/workspace/allRank/mmdataset/Feature_18_coherent_label2/test.txt
ln -s /data/orlando/workspace/metaGUI_forward/repo_upload/ground_truth/rerank_input_raw_data_label2.json /data/orlando/workspace/allRank/mmdataset/Feature_18_coherent_label2/test_qid_label2.json
conda activate metaGUI
```

* NeurlNDCG

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth #DONE
```

### 1. 18 features

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth #DONE
```

[INFO] 2024-01-22 09:40:52 - Epoch : 19 Train loss: -0.9063339655131206 Val loss: -0.9046579749734552 Train ndcg_1 0.9942656755447388 Train ndcg_5 0.9941678643226624 Train ndcg_10 0.9949824810028076 Train mrr 0.9963098764419556 Train ap 0.9907324910163879 Train precision_1 0.9928321242332458 Train precision_5 0.29735586047172546 Val ndcg_1 0.9898648858070374 Val ndcg_5 0.9933796525001526 Val ndcg_10 0.9941566586494446 Val mrr 0.9947447180747986 Val ap 0.9904809594154358 Val precision_1 0.9898648858070374 Val precision_5 0.2797297239303589
[INFO] 2024-01-22 09:40:52 - Current:0.9933796525001526 Best:1.0
[INFO] 2024-01-22 09:40:53 - Test metrics: {'ndcg_1': 0.26495728, 'ndcg_5': 0.28738827, 'ndcg_10': 0.2960714, 'mrr': 0.29110706, 'ap': 0.27940682, 'precision_1': 0.26495728, 'precision_5': 0.107692316}

### 2. Without feature 0 "Query term ratio in instructions"

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth #DONE
```

### 3. Without feature 1 "Relevance between title and search query"

### 4. Without feature 2 "Keyword ratio"

### 5. Without feature 3 "Instructions completion degree"

### 6. Without feature 4-7 "Action term ratio in instructions (avg, min, max, var)"

### 7. Without feature 8-11 "UI term matching ratio in instructions (avg, min, max, var)"

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_without_feature9_10_11_12.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_without_feature9_10_11_12 --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_without_feature9_10_11_12 #DONE
```

### 8. Without feature 12-15 "Matched UI term frequency on UI (avg, min, max, var)"

### 9.  Without feature 16 "Relative position of the last matched instruction term"

### 10.  Without feature 17 "Moving distancing of instruction terms"

* ListNet

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth --job-dir experiments/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth #DONE
```

* LambdaRank

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth #DONE
```

* ApproxNDCG

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth #DONE
```

### neuralNDCG, listNet, lambdarank, approxndc on dataset "How-to" metaGUI

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_metagui_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_metagui_ground_truth --job-dir experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_metagui_ground_truth && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth --job-dir experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/approxndc/approxndc_Multimodal_Feature18_label2_on_metagui_ground_truth.json --run-id approxndc_Multimodal_Feature18_label2_on_metagui_ground_truth --job-dir experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_metagui_ground_truth
```

neuralNDCG Time 1:

[INFO] 2024-01-22 09:33:37 - Epoch : 19 Train loss: -0.9063339655131206 Val loss: -0.4570321904288398 Train ndcg_1 0.9942656755447388 Train ndcg_5 0.9941678643226624 Train ndcg_10 0.9949824810028076 Train mrr 0.9963098764419556 Train ap 0.9907324910163879 Train precision_1 0.9928321242332458 Train precision_5 0.29735586047172546 Val ndcg_1 0.2326388955116272 Val ndcg_5 0.27378058433532715 Val ndcg_10 0.27711227536201477 Val mrr 0.26250413060188293 Val ap 0.26316431164741516 Val precision_1 0.2326388955116272 Val precision_5 0.20104166865348816
[INFO] 2024-01-22 09:33:37 - Current:0.27378058433532715 Best:0.29597318172454834
[INFO] 2024-01-22 09:33:41 - Test metrics: {'ndcg_1': 0.26434782, 'ndcg_5': 0.32604983, 'ndcg_10': 0.33070648, 'mrr': 0.3112029, 'ap': 0.3126666, 'precision_1': 0.26434782, 'precision_5': 0.17878261}

listNet Time 1:

[INFO] 2024-01-22 10:52:17 - Epoch : 19 Train loss: 0.8899635142602522 Val loss: 0.49296434058083427 Train ndcg_1 0.9899649620056152 Train ndcg_5 0.9947431087493896 Train ndcg_10 0.9950008988380432 Train mrr 0.9945312142372131 Train ap 0.9926384687423706 Train precision_1 0.9906021356582642 Train precision_5 0.29802486300468445 Val ndcg_1 0.21875 Val ndcg_5 0.26425012946128845 Val ndcg_10 0.2694321572780609 Val mrr 0.2517836391925812 Val ap 0.2535860538482666 Val precision_1 0.21875 Val precision_5 0.19895833730697632
[INFO] 2024-01-22 10:52:17 - Current:0.26425012946128845 Best:0.2708679735660553
[INFO] 2024-01-22 10:52:20 - Test metrics: {'ndcg_1': 0.24695653, 'ndcg_5': 0.31522298, 'ndcg_10': 0.32162762, 'mrr': 0.29837102, 'ap': 0.3009607, 'precision_1': 0.24695653, 'precision_5': 0.17704348}

lambdarank Time 1:

2024-01-22 08:52:31 - Epoch : 19 Train loss: 0.06408040969733145 Val loss: 19.628841824001736 Train ndcg_1 0.9988850355148315 Train ndcg_5 0.9991609454154968 Train ndcg_10 0.9991609454154968 Train mrr 0.9992035627365112 Train ap 0.9991106986999512 Train precision_1 0.9990442991256714 Train precision_5 0.2986301779747009 Val ndcg_1 0.265625 Val ndcg_5 0.2896327078342438 Val ndcg_10 0.2924577593803406 Val mrr 0.2857060134410858 Val ap 0.2823547422885895 Val precision_1 0.265625 Val precision_5 0.20243056118488312
2024-01-22 08:52:31 - Current:0.2896327078342438 Best:0.30427107214927673
2024-01-22 08:52:34 - Test metrics: {'ndcg_1': 0.29391304, 'ndcg_5': 0.34495965, 'ndcg_10': 0.34703442, 'mrr': 0.3343768, 'ap': 0.33268175, 'precision_1': 0.29391304, 'precision_5': 0.1826087}

approxndc Time 1:

[INFO] 2024-01-22 10:03:20 - Epoch : 19 Train loss: -0.9955557235185793 Val loss: -0.26082850164837307 Train ndcg_1 0.9947435855865479 Train ndcg_5 0.9971022009849548 Train ndcg_10 0.997189462184906 Train mrr 0.9968276023864746 Train ap 0.9961496591567993 Train precision_1 0.9941064119338989 Train precision_5 0.2983434498310089 Val ndcg_1 0.2326388955116272 Val ndcg_5 0.27194035053253174 Val ndcg_10 0.2761770486831665 Val mrr 0.25984206795692444 Val ap 0.2622193694114685 Val precision_1 0.2326388955116272 Val precision_5 0.20173612236976624
[INFO] 2024-01-22 10:03:20 - Current:0.27194035053253174 Best:0.3040275573730469
[INFO] 2024-01-22 10:03:23 - Test metrics: {'ndcg_1': 0.26956522, 'ndcg_5': 0.32612193, 'ndcg_10': 0.3310262, 'mrr': 0.31213045, 'ap': 0.31199834, 'precision_1': 0.26956522, 'precision_5': 0.18017393}

### neuralNDCG, listNet, lambdarank, approxndc on dataset "How-to" WeWeb

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_weweb_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_weweb_ground_truth --job-dir experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_weweb_ground_truth && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth --job-dir experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/approxndc/approxndc_Multimodal_Feature18_label2_on_weweb_ground_truth.json --run-id approxndc_Multimodal_Feature18_label2_on_weweb_ground_truth --job-dir experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_weweb_ground_truth
```

neuralNDCG Time 1:

listNet Time 1:

lambdarank Time 1:

approxndc Time 1:

### neuralNDCG, listNet, lambdarank, approxndc on train and validate on "How-to" metaGUI and test on Extrated instruction for "How-to" WeWeb

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth --job-dir experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/approxndc/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth
```

* neuralNDCG without Google Rank:

2024-01-21 12:13:05,230 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.9063339655131206 Val loss: -0.9046579749734552 Train ndcg_1 0.9942656755447388 Train ndcg_5 0.9941678643226624 Train ndcg_10 0.9949824810028076 Train mrr 0.9963098764419556 Train ap 0.9907324910163879 Train precision_1 0.9928321242332458 Train precision_5 0.29735586047172546 Val ndcg_1 0.9898648858070374 Val ndcg_5 0.9933796525001526 Val ndcg_10 0.9941566586494446 Val mrr 0.9947447180747986 Val ap 0.9904809594154358 Val precision_1 0.9898648858070374 Val precision_5 0.2797297239303589
2024-01-21 12:13:05,234 - allrank.utils.ltr_logging - INFO - Current:0.9933796525001526 Best:1.0
2024-01-21 12:13:05,593 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.26495728, 'ndcg_5': 0.28738827, 'ndcg_10': 0.2960714, 'mrr': 0.29110706, 'ap': 0.27940682, 'precision_1': 0.26495728, 'precision_5': 0.107692316} -> 117/167
{'ndcg_1': 0.18562875305389223,
 'ndcg_5': 0.20134387778443116,
 'ndcg_10': 0.20742726826347305,
 'mrr': 0.2039492576047904,
 'ap': 0.19575208347305392,
 'precision_1': 0.18562875305389223,
 'precision_5': 0.07544910761676647}

* neuralNDCG with Google Rank adjustment:

```bash
python end_to_end_statistics.py --rerank_result experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth/results/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth//predicted_result.txt
```

python end_to_end_statistics.py --rerank_result experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth/results/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth//predicted_result.txt
How to pause history tracking in YouTube {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.16666666666666666, 'p1': 0.0, 'p5': 0.0, 'ndcg1': 0.0, 'ndcg5': 0.0}
How to share an article link in FlipBoard {'mrr': 0.5, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.6309297535714575} {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306}
google[mrr] > execution_statistics[mrr]: 27, google[mrr] < execution_statistics[mrr]: 2, google[mrr] = execution_statistics[mrr]: 88, zero_cnt_execution_statistics: 86, non_zero_cnt_execution_statistics: 31, total_query: 117
google average statistics: {'mrr': 0.10238170502348029, 'p1': 0.05389221556886228, 'p5': 0.05389221556886228, 'ndcg1': 0.05389221556886228, 'ndcg5': 0.10419093916874589}
execution average statistics: {'mrr': 0.20309381237524948, 'p1': 0.18562874251497005, 'p5': 0.07904191616766469, 'ndcg1': 0.18562874251497005, 'ndcg5': 0.2032960374207608}

* listNet without Google Rank:


[INFO] 2024-01-22 11:23:24 - Epoch : 19 Train loss: 0.8899635142602522 Val loss: 0.8315677615973327 Train ndcg_1 0.9899649620056152 Train ndcg_5 0.9947431087493896 Train ndcg_10 0.9950008988380432 Train mrr 0.9945312142372131 Train ap 0.9926384687423706 Train precision_1 0.9906021356582642 Train precision_5 0.29802486300468445 Val ndcg_1 0.9853603839874268 Val ndcg_5 0.9939936995506287 Val ndcg_10 0.9942396283149719 Val mrr 0.9926801919937134 Val ap 0.9916432499885559 Val precision_1 0.9853603839874268 Val precision_5 0.2804054319858551
[INFO] 2024-01-22 11:23:24 - Current:0.9939936995506287 Best:1.0
[INFO] 2024-01-22 11:23:24 - Test metrics: {'ndcg_1': 0.25641027, 'ndcg_5': 0.28601643, 'ndcg_10': 0.2935485, 'mrr': 0.29039478, 'ap': 0.2788039, 'precision_1': 0.26495728, 'precision_5': 0.11282053} -> 117/167

{'ndcg_1': 0.17964072808383233,
 'ndcg_5': 0.2003827683233533,
 'ndcg_10': 0.20565972754491016,
 'mrr': 0.20345023508982038,
 'ap': 0.19532967844311377,
 'precision_1': 0.18562875305389223,
 'precision_5': 0.07904192820359282}

* listNet wih Google Rank Adjustment:

```bash
python end_to_end_statistics.py --rerank_result  experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth/results/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth/predicted_result.txt
```

How do I reply to a specific message? in Google Chat {'mrr': 0.3333333333333333, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.5} {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306}
How do I reply to a specific message? in Messenger {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.2, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.38685280723454163}
How do I turn on/off my active status? in Messenger {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.09090909090909091, 'p1': 0.0, 'p5': 0.0, 'ndcg1': 0.0, 'ndcg5': 0.0}
How do I unsend a message? in Messenger {'mrr': 1.0, 'p1': 1.0, 'p5': 0.4, 'ndcg1': 1.0, 'ndcg5': 0.9197207891481876} {'mrr': 0.3333333333333333, 'p1': 0.0, 'p5': 0.4, 'ndcg1': 0.0, 'ndcg5': 0.5706417189553201}
How to add Home or Work in Google Maps {'mrr': 1.0, 'p1': 1.0, 'p5': 0.6, 'ndcg1': 1.0, 'ndcg5': 0.6843515475204855} {'mrr': 0.5, 'p1': 0.0, 'p5': 0.4, 'ndcg1': 0.0, 'ndcg5': 0.38356636737133565}
How to apply coupon? in eBay {'mrr': 0.3333333333333333, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.3065735963827292} {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.2640681225725909}
How to pause history tracking in YouTube {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.16666666666666666, 'p1': 0.0, 'p5': 0.0, 'ndcg1': 0.0, 'ndcg5': 0.0}
google[mrr] > execution_statistics[mrr]: 22, google[mrr] < execution_statistics[mrr]: 7, google[mrr] = execution_statistics[mrr]: 88, zero_cnt_execution_statistics: 98, non_zero_cnt_execution_statistics: 19, total_query: 117
google average statistics: {'mrr': 0.10238170502348029, 'p1': 0.05389221556886228, 'p5': 0.05389221556886228, 'ndcg1': 0.05389221556886228, 'ndcg5': 0.10419093916874589}
execution average statistics: {'mrr': 0.1524859372164761, 'p1': 0.11377245508982035, 'p5': 0.06467065868263475, 'ndcg1': 0.11377245508982035, 'ndcg5': 0.15288695620307285}

* approxndc without Google Rank:


[INFO] 2024-01-22 10:15:15 - Epoch : 19 Train loss: -0.9966119197987948 Val loss: -0.9938635546881873 Train ndcg_1 0.9988850355148315 Train ndcg_5 0.9990139007568359 Train ndcg_10 0.9990105628967285 Train mrr 0.9990442991256714 Train ap 0.9980828166007996 Train precision_1 0.9982478618621826 Train precision_5 0.2986301779747009 Val ndcg_1 1.0 Val ndcg_5 1.0 Val ndcg_10 1.0 Val mrr 1.0 Val ap 1.0 Val precision_1 1.0 Val precision_5 0.2806306481361389
[INFO] 2024-01-22 10:15:15 - Current:1.0 Best:1.0
[INFO] 2024-01-22 10:15:16 - Test metrics: {'ndcg_1': 0.26495728, 'ndcg_5': 0.2888895, 'ndcg_10': 0.29544598, 'mrr': 0.29017097, 'ap': 0.27788368, 'precision_1': 0.26495728, 'precision_5': 0.109401725}

2024-01-21 12:16:56,040 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.9966119197987948 Val loss: -0.9938635546881873 Train ndcg_1 0.9988850355148315 Train ndcg_5 0.9990139007568359 Train ndcg_10 0.9990105628967285 Train mrr 0.9990442991256714 Train ap 0.9980828166007996 Train precision_1 0.9982478618621826 Train precision_5 0.2986301779747009 Val ndcg_1 1.0 Val ndcg_5 1.0 Val ndcg_10 1.0 Val mrr 1.0 Val ap 1.0 Val precision_1 1.0 Val precision_5 0.2806306481361389
2024-01-21 12:16:56,042 - allrank.utils.ltr_logging - INFO - Current:1.0 Best:1.0
2024-01-21 12:16:56,454 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.26495728, 'ndcg_5': 0.2888895, 'ndcg_10': 0.29544598, 'mrr': 0.29017097, 'ap': 0.27788368, 'precision_1': 0.26495728, 'precision_5': 0.109401725} -> 117/167
{'ndcg_1': 0.18562875305389223,
 'ndcg_5': 0.2023956377245509,
 'ndcg_10': 0.20698909976047905,
 'mrr': 0.2032934340718563,
 'ap': 0.19468497341317365,
 'precision_1': 0.18562875305389223,
 'precision_5': 0.07664671751497007}

* approxndc with Google Rank adjustment:

```bash
python end_to_end_statistics.py --rerank_result experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth/results/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth/predicted_result.txt
```

How to add Home or Work in Google Maps {'mrr': 1.0, 'p1': 1.0, 'p5': 0.6, 'ndcg1': 1.0, 'ndcg5': 0.6843515475204855} {'mrr': 0.5, 'p1': 0.0, 'p5': 0.4, 'ndcg1': 0.0, 'ndcg5': 0.38356636737133565}
How to pause history tracking in YouTube {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.2, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.38685280723454163}
How to share an article link in FlipBoard {'mrr': 0.5, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.6309297535714575} {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306}
google[mrr] > execution_statistics[mrr]: 27, google[mrr] < execution_statistics[mrr]: 3, google[mrr] = execution_statistics[mrr]: 87, zero_cnt_execution_statistics: 86, non_zero_cnt_execution_statistics: 31, total_query: 117
google average statistics: {'mrr': 0.10238170502348029, 'p1': 0.05389221556886228, 'p5': 0.05389221556886228, 'ndcg1': 0.05389221556886228, 'ndcg5': 0.10419093916874589}
execution average statistics: {'mrr': 0.20329341317365268, 'p1': 0.18562874251497005, 'p5': 0.07664670658682635, 'ndcg1': 0.18562874251497005, 'ndcg5': 0.20239561662786645}

* lambdarank without Google Rank:

[INFO] 2024-01-22 08:57:06 - Current learning rate: 0.001
[INFO] 2024-01-22 08:57:10 - Epoch : 19 Train loss: 0.06408040969733145 Val loss: 0.002021976289347754 Train ndcg_1 0.9988850355148315 Train ndcg_5 0.9991609454154968 Train ndcg_10 0.9991609454154968 Train mrr 0.9992035627365112 Train ap 0.9991106986999512 Train precision_1 0.9990442991256714 Train precision_5 0.2986301779747009 Val ndcg_1 1.0 Val ndcg_5 1.0 Val ndcg_10 1.0 Val mrr 1.0 Val ap 1.0 Val precision_1 1.0 Val precision_5 0.2806306481361389
[INFO] 2024-01-22 08:57:10 - Current:1.0 Best:1.0
[INFO] 2024-01-22 08:57:10 - Test metrics: {'ndcg_1': 0.2905983, 'ndcg_5': 0.30285743, 'ndcg_10': 0.30491447, 'mrr': 0.304416, 'ap': 0.29099444, 'precision_1': 0.2905983, 'precision_5': 0.116239324}

2024-01-21 12:15:27,868 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 0.06408040969733145 Val loss: 0.002021976289347754 Train ndcg_1 0.9988850355148315 Train ndcg_5 0.9991609454154968 Train ndcg_10 0.9991609454154968 Train mrr 0.9992035627365112 Train ap 0.9991106986999512 Train precision_1 0.9990442991256714 Train precision_5 0.2986301779747009 Val ndcg_1 1.0 Val ndcg_5 1.0 Val ndcg_10 1.0 Val mrr 1.0 Val ap 1.0 Val precision_1 1.0 Val precision_5 0.2806306481361389
2024-01-21 12:15:27,870 - allrank.utils.ltr_logging - INFO - Current:1.0 Best:1.0
2024-01-21 12:15:28,285 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.2905983, 'ndcg_5': 0.30285743, 'ndcg_10': 0.30491447, 'mrr': 0.304416, 'ap': 0.29099444, 'precision_1': 0.2905983, 'precision_5': 0.116239324} -> 117/167
{'ndcg_1': 0.20359282095808381,
 'ndcg_5': 0.21218155275449102,
 'ndcg_10': 0.21362271251497006,
 'mrr': 0.21327348502994012,
 'ap': 0.2038703561676647,
 'precision_1': 0.20359282095808381,
 'precision_5': 0.08143713118562874}

* lambdarank with Google Rank adjustment:

```bash
python end_to_end_statistics.py --rerank_result experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth/results/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth/predicted_result.txt
```

How to pause history tracking in YouTube {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.2, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.38685280723454163}
How to share an article link in FlipBoard {'mrr': 0.5, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.6309297535714575} {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306}
google[mrr] > execution_statistics[mrr]: 27, google[mrr] < execution_statistics[mrr]: 2, google[mrr] = execution_statistics[mrr]: 88, zero_cnt_execution_statistics: 83, non_zero_cnt_execution_statistics: 34, total_query: 117
google average statistics: {'mrr': 0.10238170502348029, 'p1': 0.05389221556886228, 'p5': 0.05389221556886228, 'ndcg1': 0.05389221556886228, 'ndcg5': 0.10419093916874589}
execution average statistics: {'mrr': 0.21327345309381235, 'p1': 0.20359281437125748, 'p5': 0.08143712574850301, 'ndcg1': 0.20359281437125748, 'ndcg5': 0.2121815454375664}

### neuralNDCG, listNet, lambdarank, approxndc on dataset "How-to" META-GUI via Sigmoid MLP

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth_no_trans --job-dir experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans --job-dir experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/approxndc/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans.json --run-id approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans --job-dir experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans # METAGUI -> WeWeb
```

* neuralNDCG without Google Rank:

2024-01-21 12:22:43,086 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.9063806808765772 Val loss: -0.9025418189194825 Train ndcg_1 0.9992035627365112 Train ndcg_5 0.9992708563804626 Train ndcg_10 0.9992708563804626 Train mrr 0.9992832541465759 Train ap 0.9992088675498962 Train precision_1 0.9992035627365112 Train precision_5 0.29863014817237854 Val ndcg_1 1.0 Val ndcg_5 1.0 Val ndcg_10 1.0 Val mrr 1.0 Val ap 1.0 Val precision_1 1.0 Val precision_5 0.2806306481361389
2024-01-21 12:22:43,089 - allrank.utils.ltr_logging - INFO - Current:1.0 Best:1.0
2024-01-21 12:22:43,493 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.25641027, 'ndcg_5': 0.28600442, 'ndcg_10': 0.2917161, 'mrr': 0.28418806, 'ap': 0.2748781, 'precision_1': 0.25641027, 'precision_5': 0.11282053} -> 117/167
{'ndcg_1': 0.17964072808383233,
 'ndcg_5': 0.2003743541317365,
 'ndcg_10': 0.20437595029940117,
 'mrr': 0.19910181449101796,
 'ap': 0.19257926766467065,
 'precision_1': 0.17964072808383233,
 'precision_5': 0.07904192820359282}

* neuralNDCG with Google Rank adjustment:

```bash
python end_to_end_statistics.py --rerank_result experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans/results/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans/predicted_result.txt
```

python end_to_end_statistics.py --rerank_result experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans/results/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans/predicted_result.txt
How to add Home or Work in Google Maps {'mrr': 1.0, 'p1': 1.0, 'p5': 0.6, 'ndcg1': 1.0, 'ndcg5': 0.6843515475204855} {'mrr': 0.5, 'p1': 0.0, 'p5': 0.6, 'ndcg1': 0.0, 'ndcg5': 0.5296347172140421}
How to pause history tracking in YouTube {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.16666666666666666, 'p1': 0.0, 'p5': 0.0, 'ndcg1': 0.0, 'ndcg5': 0.0}
How to share an article link in FlipBoard {'mrr': 0.5, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.6309297535714575} {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306}
google[mrr] > execution_statistics[mrr]: 27, google[mrr] < execution_statistics[mrr]: 3, google[mrr] = execution_statistics[mrr]: 87, zero_cnt_execution_statistics: 87, non_zero_cnt_execution_statistics: 30, total_query: 117
google average statistics: {'mrr': 0.10238170502348029, 'p1': 0.05389221556886228, 'p5': 0.05389221556886228, 'ndcg1': 0.05389221556886228, 'ndcg5': 0.10419093916874589}
execution average statistics: {'mrr': 0.19910179640718562, 'p1': 0.17964071856287422, 'p5': 0.07904191616766468, 'ndcg1': 0.17964071856287422, 'ndcg5': 0.20037433800518994}

* approxndc without Google Rank:

2024-01-21 12:26:14,507 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.707638318389037 Val loss: -0.712489253765828 Train ndcg_1 0.9992035627365112 Train ndcg_5 0.9992371797561646 Train ndcg_10 0.9992564916610718 Train mrr 0.9992832541465759 Train ap 0.9991770386695862 Train precision_1 0.9992035627365112 Train precision_5 0.2985982894897461 Val ndcg_1 1.0 Val ndcg_5 0.9999096393585205 Val ndcg_10 0.9999096393585205 Val mrr 1.0 Val ap 0.9998123645782471 Val precision_1 1.0 Val precision_5 0.2806306481361389
2024-01-21 12:26:14,511 - allrank.utils.ltr_logging - INFO - Current:0.9999096393585205 Best:0.999819278717041
2024-01-21 12:26:14,922 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.25641027, 'ndcg_5': 0.28610703, 'ndcg_10': 0.2919317, 'mrr': 0.28418806, 'ap': 0.27532113, 'precision_1': 0.25641027, 'precision_5': 0.11282053} -> 117/167
{'ndcg_1': 0.17964072808383233,
 'ndcg_5': 0.2004462425748503,
 'ndcg_10': 0.20452699940119762,
 'mrr': 0.19910181449101796,
 'ap': 0.19288965395209584,
 'precision_1': 0.17964072808383233,
 'precision_5': 0.07904192820359282}

* approxndc with Google Rank adjustment:

```bash
python end_to_end_statistics.py --rerank_result experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans/results/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans/predicted_result.txt
```

How to add Home or Work in Google Maps {'mrr': 1.0, 'p1': 1.0, 'p5': 0.6, 'ndcg1': 1.0, 'ndcg5': 0.6843515475204855} {'mrr': 0.5, 'p1': 0.0, 'p5': 0.6, 'ndcg1': 0.0, 'ndcg5': 0.5147714448836774}
How to pause history tracking in YouTube {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.16666666666666666, 'p1': 0.0, 'p5': 0.0, 'ndcg1': 0.0, 'ndcg5': 0.0}
How to share an article link in FlipBoard {'mrr': 0.5, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.6309297535714575} {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306}
google[mrr] > execution_statistics[mrr]: 27, google[mrr] < execution_statistics[mrr]: 3, google[mrr] = execution_statistics[mrr]: 87, zero_cnt_execution_statistics: 87, non_zero_cnt_execution_statistics: 30, total_query: 117
google average statistics: {'mrr': 0.10238170502348029, 'p1': 0.05389221556886228, 'p5': 0.05389221556886228, 'ndcg1': 0.05389221556886228, 'ndcg5': 0.10419093916874589}
execution average statistics: {'mrr': 0.19910179640718562, 'p1': 0.17964071856287422, 'p5': 0.07904191616766468, 'ndcg1': 0.17964071856287422, 'ndcg5': 0.20044623703197398}

* lambdarank without Google Rank:

2024-01-21 12:24:55,555 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 95.47729390865301 Val loss: 90.27464586550052 Train ndcg_1 0.9992035627365112 Train ndcg_5 0.999198317527771 Train ndcg_10 0.9992368817329407 Train mrr 0.9992832541465759 Train ap 0.9991318583488464 Train precision_1 0.9992035627365112 Train precision_5 0.29856646060943604 Val ndcg_1 1.0 Val ndcg_5 0.999728798866272 Val ndcg_10 0.999728798866272 Val mrr 1.0 Val ap 0.9994369745254517 Val precision_1 1.0 Val precision_5 0.2806306481361389
2024-01-21 12:24:55,556 - allrank.utils.ltr_logging - INFO - Current:0.999728798866272 Best:0.9995905756950378
2024-01-21 12:24:55,955 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.25641027, 'ndcg_5': 0.28712347, 'ndcg_10': 0.29279655, 'mrr': 0.28561255, 'ap': 0.27622625, 'precision_1': 0.25641027, 'precision_5': 0.11282053} -> 117/167
{'ndcg_1': 0.17964072808383233,
 'ndcg_5': 0.2011583592215569,
 'ndcg_10': 0.2051329122754491,
 'mrr': 0.20009981047904193,
 'ap': 0.19352377994011977,
 'precision_1': 0.17964072808383233,
 'precision_5': 0.07904192820359282}

* lambdarank with Google Rank adjustment:

```bash
python end_to_end_statistics.py --rerank_result experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans/results/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans/predicted_result.txt
```

How to add Home or Work in Google Maps {'mrr': 1.0, 'p1': 1.0, 'p5': 0.6, 'ndcg1': 1.0, 'ndcg5': 0.6843515475204855} {'mrr': 0.5, 'p1': 0.0, 'p5': 0.6, 'ndcg1': 0.0, 'ndcg5': 0.5296347172140421}
How to pause history tracking in YouTube {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.16666666666666666, 'p1': 0.0, 'p5': 0.0, 'ndcg1': 0.0, 'ndcg5': 0.0}
How to share an article link in FlipBoard {'mrr': 0.5, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.6309297535714575} {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306}
google[mrr] > execution_statistics[mrr]: 27, google[mrr] < execution_statistics[mrr]: 3, google[mrr] = execution_statistics[mrr]: 87, zero_cnt_execution_statistics: 87, non_zero_cnt_execution_statistics: 30, total_query: 117
google average statistics: {'mrr': 0.10238170502348029, 'p1': 0.05389221556886228, 'p5': 0.05389221556886228, 'ndcg1': 0.05389221556886228, 'ndcg5': 0.10419093916874589}
execution average statistics: {'mrr': 0.20009980039920158, 'p1': 0.17964071856287422, 'p5': 0.07904191616766468, 'ndcg1': 0.17964071856287422, 'ndcg5': 0.20115834850561787}

* listNet without Google Rank:

2024-01-21 12:23:36,726 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 0.0 Val loss: 1.3319710439389891 Train ndcg_1 0.9993628859519958 Train ndcg_5 0.9993628859519958 Train ndcg_10 0.9993628859519958 Train mrr 0.9993628859519958 Train ap 0.9993628859519958 Train precision_1 0.9993628859519958 Train precision_5 0.19987258315086365 Val ndcg_1 0.8220720887184143 Val ndcg_5 0.9075261354446411 Val ndcg_10 0.9129072427749634 Val mrr 0.8934175372123718 Val ap 0.8783726096153259 Val precision_1 0.8220720887184143 Val precision_5 0.2704955041408539
2024-01-21 12:23:36,728 - allrank.utils.ltr_logging - INFO - Current:0.9075261354446411 Best:0.9075261354446411
2024-01-21 12:23:37,121 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.21367523, 'ndcg_5': 0.24847877, 'ndcg_10': 0.26463625, 'mrr': 0.24744609, 'ap': 0.24046874, 'precision_1': 0.21367523, 'precision_5': 0.10256411} -> 117/167
{'ndcg_1': 0.14970061023952094,
 'ndcg_5': 0.17408392868263473,
 'ndcg_10': 0.1854038398203593,
 'mrr': 0.17336043431137724,
 'ap': 0.168472111257485,
 'precision_1': 0.14970061023952094,
 'precision_5': 0.07185629263473053}

* listNet with Google Rank adjustment:

```bash
python end_to_end_statistics.py --rerank_result experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth_no_trans/results/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth_no_trans/predicted_result.txt
```

How do I reply to a specific message? in Messenger {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.2, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.38685280723454163}
How do I turn on/off my active status? in Messenger {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.1, 'p1': 0.0, 'p5': 0.0, 'ndcg1': 0.0, 'ndcg5': 0.0}
How to pause history tracking in YouTube {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.125, 'p1': 0.0, 'p5': 0.0, 'ndcg1': 0.0, 'ndcg5': 0.0}
How to scan item? in eBay {'mrr': 0.5, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.6309297535714575} {'mrr': 0.3333333333333333, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.5}
How to share an article link in FlipBoard {'mrr': 0.5, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.6309297535714575} {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306}
google[mrr] > execution_statistics[mrr]: 23, google[mrr] < execution_statistics[mrr]: 5, google[mrr] = execution_statistics[mrr]: 89, zero_cnt_execution_statistics: 92, non_zero_cnt_execution_statistics: 25, total_query: 117
google average statistics: {'mrr': 0.10238170502348029, 'p1': 0.05389221556886228, 'p5': 0.05389221556886228, 'ndcg1': 0.05389221556886228, 'ndcg5': 0.10419093916874589}
execution average statistics: {'mrr': 0.17336042201311663, 'p1': 0.1497005988023952, 'p5': 0.07185628742514971, 'ndcg1': 0.1497005988023952, 'ndcg5': 0.17408391507708737}

### MLP + Loss

#### neuralNDCG

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans # METAGUI -> WeWeb
```

"How-to" META-GUI

2024-01-21 23:13:16,626 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.9094336908423396 Val loss: -0.38125332196553546 Train ndcg_1 0.9993628859519958 Train ndcg_5 0.9993501305580139 Train ndcg_10 0.9993501305580139 Train mrr 0.9993628859519958 Train ap_10 0.9993363618850708 Train precision_1 0.9993628859519958 Train precision_5 0.29863014817237854 Val ndcg_1 0.2065972238779068 Val ndcg_5 0.25481468439102173 Val ndcg_10 0.26165100932121277 Val mrr 0.2391844391822815 Val ap_10 0.2440851479768753 Val precision_1 0.2065972238779068 Val precision_5 0.19722221791744232
2024-01-21 23:13:16,627 - allrank.utils.ltr_logging - INFO - Current:0.25481468439102173 Best:0.28698232769966125
2024-01-21 23:13:23,082 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.21739131, 'ndcg_5': 0.30020073, 'ndcg_10': 0.30700687, 'mrr': 0.2761398, 'ap_10': 0.28196734, 'precision_1': 0.21739131, 'precision_5': 0.17669566}

"How-to" WeWeb

Zero-shot

#### listNet

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_metagui_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_metagui_ground_truth_no_trans --job-dir experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_metagui_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_weweb_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_weweb_ground_truth_no_trans --job-dir experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_weweb_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth_no_trans --job-dir experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth_no_trans # METAGUI -> WeWeb
```

"How-to" META-GUI"

2024-01-21 23:14:22,826 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 0.0 Val loss: 0.4932870931095547 Train ndcg_1 0.9993628859519958 Train ndcg_5 0.9993628859519958 Train ndcg_10 0.9993628859519958 Train mrr 0.9993628859519958 Train ap 0.9993628859519958 Train precision_1 0.9993628859519958 Train precision_5 0.19987258315086365 Val ndcg_1 0.234375 Val ndcg_5 0.27244213223457336 Val ndcg_10 0.27637332677841187 Val mrr 0.2634197473526001 Val ap 0.2623297870159149 Val precision_1 0.234375 Val precision_5 0.19930556416511536
2024-01-21 23:14:22,827 - allrank.utils.ltr_logging - INFO - Current:0.27244213223457336 Best:0.27244213223457336
2024-01-21 23:14:31,703 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.24695653, 'ndcg_5': 0.31612018, 'ndcg_10': 0.32168564, 'mrr': 0.29968533, 'ap': 0.30061364, 'precision_1': 0.24695653, 'precision_5': 0.17808697}

"How-to" WeWeb

Zero-shot

#### lambdarank

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans --job-dir experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans --job-dir experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans --job-dir experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans # METAGUI -> WeWeb
```

"How-to" META-GUI

2024-01-21 23:16:01,858 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 47.04511473990808 Val loss: 10.773070318831337 Train ndcg_1 0.9992035627365112 Train ndcg_5 0.9992766380310059 Train ndcg_10 0.9992766380310059 Train mrr 0.9992832541465759 Train ap 0.9992221593856812 Train precision_1 0.9992035627365112 Train precision_5 0.29863014817237854 Val ndcg_1 0.2065972238779068 Val ndcg_5 0.2546476721763611 Val ndcg_10 0.2614840269088745 Val mrr 0.23895835876464844 Val ap 0.24439819157123566 Val precision_1 0.2065972238779068 Val precision_5 0.19722221791744232
2024-01-21 23:16:01,859 - allrank.utils.ltr_logging - INFO - Current:0.2546476721763611 Best:0.2713039219379425
2024-01-21 23:16:10,475 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.21565218, 'ndcg_5': 0.2995025, 'ndcg_10': 0.30612543, 'mrr': 0.27493352, 'ap': 0.28152198, 'precision_1': 0.21565218, 'precision_5': 0.17669566}

"How-to" WeWeb

Zero-shot

#### approxndc

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/approxndc/approxndc_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans.json --run-id approxndc_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans --job-dir experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/approxndc/approxndc_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans.json --run-id approxndc_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans --job-dir experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/approxndc/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans.json --run-id approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans --job-dir experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans # METAGUI -> WeWeb
```

"How-to" META-GUI

[INFO] 2024-01-22 13:11:58 - Epoch : 19 Train loss: -0.707638318389037 Val loss: -0.24607786536216736 Train ndcg_1 0.9992035627365112 Train ndcg_5 0.9992371797561646 Train ndcg_10 0.9992564916610718 Train mrr 0.9992832541465759 Train ap 0.9991770386695862 Train precision_1 0.9992035627365112 Train precision_5 0.2985982894897461 Val ndcg_1 0.2065972238779068 Val ndcg_5 0.2558635175228119 Val ndcg_10 0.26212114095687866 Val mrr 0.23985923826694489 Val ap 0.2451452761888504 Val precision_1 0.2065972238779068 Val precision_5 0.19756944477558136
[INFO] 2024-01-22 13:11:58 - Current:0.2558635175228119 Best:0.28453683853149414
[INFO] 2024-01-22 13:12:01 - Test metrics: {'ndcg_1': 0.2226087, 'ndcg_5': 0.3035813, 'ndcg_10': 0.31035638, 'mrr': 0.28074643, 'ap': 0.28681475, 'precision_1': 0.2226087, 'precision_5': 0.17669566}

"How-to" WeWeb

Zero-shot

[INFO] 2024-01-22 13:28:41 - Epoch : 19 Train loss: -0.707638318389037 Val loss: -0.712489253765828 Train ndcg_1 0.9992035627365112 Train ndcg_5 0.9992371797561646 Train ndcg_10 0.9992564916610718 Train mrr 0.9992832541465759 Train ap 0.9991770386695862 Train precision_1 0.9992035627365112 Train precision_5 0.2985982894897461 Val ndcg_1 1.0 Val ndcg_5 0.9999096393585205 Val ndcg_10 0.9999096393585205 Val mrr 1.0 Val ap 0.9998123645782471 Val precision_1 1.0 Val precision_5 0.2806306481361389
[INFO] 2024-01-22 13:28:41 - Current:0.9999096393585205 Best:0.999819278717041
[INFO] 2024-01-22 13:28:42 - Test metrics: {'ndcg_1': 0.25641027, 'ndcg_5': 0.28610703, 'ndcg_10': 0.2919317, 'mrr': 0.28418806, 'ap': 0.27532113, 'precision_1': 0.25641027, 'precision_5': 0.11282053}

### Pointwise Loss

pointwise bce loss

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/bce/pointwise_bce_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model.json --run-id pointwise_bce_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model --job-dir experiments/bce/pointwise_bce_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/bce/pointwise_bce_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model.json --run-id pointwise_bce_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model --job-dir experiments/bce/pointwise_bce_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/bce/pointwise_bce_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model.json --run-id pointwise_bce_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model --job-dir experiments/bce/pointwise_bce_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model # on METAGUI -> WeWeb
```

* Zero-shot

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/bce/pointwise_bce_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model.json --run-id pointwise_bce_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model --job-dir experiments/bce/pointwise_bce_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model
```

BCE without Google Rank:

2024-01-21 12:57:34,792 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 2.745351061846808 Val loss: 2.6144230193919964 Train ndcg_1 0.8442178964614868 Train ndcg_5 0.9182736277580261 Train ndcg_10 0.9227007031440735 Train mrr 0.9101946949958801 Train ap 0.8890700936317444 Train precision_1 0.8442178964614868 Train precision_5 0.28999683260917664 Val ndcg_1 0.8400900959968567 Val ndcg_5 0.9131476879119873 Val ndcg_10 0.9186704158782959 Val mrr 0.9035566449165344 Val ap 0.8846280574798584 Val precision_1 0.8400900959968567 Val precision_5 0.2713964283466339
2024-01-21 12:57:34,794 - allrank.utils.ltr_logging - INFO - Current:0.9131476879119873 Best:0.9128440022468567
2024-01-21 12:57:35,107 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.23931625, 'ndcg_5': 0.27728394, 'ndcg_10': 0.28498685, 'mrr': 0.27735046, 'ap': 0.26439884, 'precision_1': 0.23931625, 'precision_5': 0.109401725} ->117/167
{'ndcg_1': 0.16766467814371258,
 'ndcg_5': 0.19426479628742516,
 'ndcg_10': 0.19966144580838324,
 'mrr': 0.19431140011976047,
 'ap': 0.18523751065868266,
 'precision_1': 0.16766467814371258,
 'precision_5': 0.07664671751497007}

BCE with Google Rank adjustment:

```bash
python end_to_end_statistics.py --rerank_result experiments/bce/pointwise_bce_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model/results/pointwise_bce_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model/predicted_result.txt
```

How to add multi-stops in Here WeGo {'mrr': 1.0, 'p1': 1.0, 'p5': 0.2, 'ndcg1': 1.0, 'ndcg5': 1.0} {'mrr': 0.5, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.6309297535714575}
How to pause history tracking in YouTube {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306} {'mrr': 0.2, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.38685280723454163}
How to share an article link in FlipBoard {'mrr': 0.5, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.6309297535714575} {'mrr': 0.25, 'p1': 0.0, 'p5': 0.2, 'ndcg1': 0.0, 'ndcg5': 0.43067655807339306}
how to connect to wifi step by step on Android {'mrr': 1.0, 'p1': 1.0, 'p5': 0.6, 'ndcg1': 1.0, 'ndcg5': 0.6548086577531307} {'mrr': 0.5, 'p1': 0.0, 'p5': 0.4, 'ndcg1': 0.0, 'ndcg5': 0.36005461457723403}
google[mrr] > execution_statistics[mrr]: 27, google[mrr] < execution_statistics[mrr]: 4, google[mrr] = execution_statistics[mrr]: 86, zero_cnt_execution_statistics: 89, non_zero_cnt_execution_statistics: 28, total_query: 117
google average statistics: {'mrr': 0.10238170502348029, 'p1': 0.05389221556886228, 'p5': 0.05389221556886228, 'ndcg1': 0.05389221556886228, 'ndcg5': 0.10419093916874589}
execution average statistics: {'mrr': 0.194311377245509, 'p1': 0.16766467065868262, 'p5': 0.07664670658682636, 'ndcg1': 0.16766467065868262, 'ndcg5': 0.19426477609123186}

* META-GUI

[INFO] 2024-01-22 08:26:25 - Epoch : 19 Train loss: 1.9755361210430227 Val loss: 2.419004612498813 Train ndcg_1 0.9424976110458374 Train ndcg_5 0.9684039354324341 Train ndcg_10 0.9700592756271362 Train mrr 0.967902660369873 Train ap 0.9554974436759949 Train precision_1 0.9424976110458374 Train precision_5 0.2954125702381134 Val ndcg_1 0.2152777761220932 Val ndcg_5 0.26506659388542175 Val ndcg_10 0.269875168800354 Val mrr 0.2508721947669983 Val ap 0.2541239559650421 Val precision_1 0.2152777761220932 Val precision_5 0.20000000298023224
[INFO] 2024-01-22 08:26:25 - Current:0.26506659388542175 Best:0.2648300230503082
[INFO] 2024-01-22 08:26:28 - Test metrics: {'ndcg_1': 0.23826088, 'ndcg_5': 0.3143243, 'ndcg_10': 0.31855163, 'mrr': 0.29377642, 'ap': 0.29690367, 'precision_1': 0.23826088, 'precision_5': 0.17913043}

* WeWeb

pointwise rmse loss

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model.json --run-id pointwise_rmse_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model --job-dir experiments/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model.json --run-id pointwise_rmse_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model --job-dir experiments/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model.json --run-id pointwise_rmse_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model --job-dir experiments/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model # METAGUI -> WeWeb
```

* META-GUI

* WeWeb

* Zero-shot

## Failure analysis

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_ground_truth_extra --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_failure_analysis
```

## License

Apache 2 License
