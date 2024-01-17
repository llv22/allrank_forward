# allRank : Learning to Rank in PyTorch

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
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=1 python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Fold1_normalized.json --run-id ranallrank_NeuralNDCG_Fold1_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold1_normalized # ongoing
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Fold2_normalized.json --run-id ranallrank_NeuralNDCG_Fold2_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold2_normalized # DONE
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Fold3_normalized.json --run-id ranallrank_NeuralNDCG_Fold3_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold3_normalized # DONE
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Fold4_normalized.json --run-id ranallrank_NeuralNDCG_Fold4_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold4_normalized # DONE
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Fold5_normalized.json --run-id ranallrank_NeuralNDCG_Fold5_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold5_normalized # DONE
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
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth #DONE
```

### 1. 18 features

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth #DONE
```

Time 1:
[INFO] 2024-01-11 16:55:13 - Epoch : 19 Train loss: -0.8616754800851613 Val loss: -0.5326286527408298 Train ndcg_1 0.9028130173683167 Train ndcg_5 0.9484610557556152 Train ndcg_10 0.9529234766960144 Train mrr_1 0.9026862978935242 Train mrr_5 0.9384377598762512 Train mrr_10 0.9391921162605286 Train ap_1 0.23897619545459747 Train ap_5 0.2722625434398651 Train ap_10 0.27252301573753357 Val ndcg_1 0.9087750315666199 Val ndcg_5 0.9546259045600891 Val ndcg_10 0.9579418897628784 Val mrr_1 0.519548237323761 Val mrr_5 0.5557485818862915 Val mrr_10 0.5559899806976318 Val ap_1 0.29278889298439026 Val ap_5 0.32736751437187195 Val ap_10 0.3269151747226715
[INFO] 2024-01-11 16:55:13 - Current:0.9546259045600891 Best:0.9546259045600891
[INFO] 2024-01-11 16:55:13 - Test metrics: {'ndcg_1': 0.98290604, 'ndcg_5': 0.98323363, 'ndcg_10': 0.9909921, 'mrr_1': 0.52991456, 'mrr_5': 0.5370371, 'mrr_10': 0.5370371, 'ap_1': 0.17094018, 'ap_5': 0.17539175, 'ap_10': 0.17233358}

Time 2:
[INFO] 2024-01-11 17:01:23 - Epoch : 19 Train loss: -0.8616754800851613 Val loss: -0.5326286527408298 Train ndcg_1 0.9028130173683167 Train ndcg_5 0.9484610557556152 Train ndcg_10 0.9529234766960144 Train mrr_1 0.9026862978935242 Train mrr_5 0.9384377598762512 Train mrr_10 0.9391921162605286 Train ap_1 0.23897619545459747 Train ap_5 0.2722625434398651 Train ap_10 0.27252301573753357 Val ndcg_1 0.9087750315666199 Val ndcg_5 0.9546259045600891 Val ndcg_10 0.9579418897628784 Val mrr_1 0.519548237323761 Val mrr_5 0.5557485818862915 Val mrr_10 0.5559899806976318 Val ap_1 0.29278889298439026 Val ap_5 0.32736751437187195 Val ap_10 0.3269151747226715
[INFO] 2024-01-11 17:01:23 - Current:0.9546259045600891 Best:0.9546259045600891
[INFO] 2024-01-11 17:01:23 - Test metrics: {'ndcg_1': 0.98290604, 'ndcg_5': 0.98323363, 'ndcg_10': 0.9909921, 'mrr_1': 0.52991456, 'mrr_5': 0.5370371, 'mrr_10': 0.5370371, 'ap_1': 0.17094018, 'ap_5': 0.17539175, 'ap_10': 0.17233358}

Time 3:
[INFO] 2024-01-11 17:23:30 - Epoch : 19 Train loss: -0.8616754800851613 Val loss: -0.5326286527408298 Train ndcg_1 0.9028130173683167 Train ndcg_5 0.9484610557556152 Train ndcg_10 0.9529234766960144 Train mrr_1 0.9026862978935242 Train mrr_5 0.9384377598762512 Train mrr_10 0.9391921162605286 Train ap_1 0.23897619545459747 Train ap_5 0.2722625434398651 Train ap_10 0.27252301573753357 Val ndcg_1 0.9087750315666199 Val ndcg_5 0.9546259045600891 Val ndcg_10 0.9579418897628784 Val mrr_1 0.519548237323761 Val mrr_5 0.5557485818862915 Val mrr_10 0.5559899806976318 Val ap_1 0.29278889298439026 Val ap_5 0.32736751437187195 Val ap_10 0.3269151747226715
[INFO] 2024-01-11 17:23:30 - Current:0.9546259045600891 Best:0.9546259045600891
[INFO] 2024-01-11 17:23:30 - Test metrics: {'ndcg_1': 0.98290604, 'ndcg_5': 0.98323363, 'ndcg_10': 0.9909921, 'mrr_1': 0.52991456, 'mrr_5': 0.5370371, 'mrr_10': 0.5370371, 'ap_1': 0.17094018, 'ap_5': 0.17539175, 'ap_10': 0.17233358}

Time 4:
2024-01-11 22:13:02,779 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.8616754800851613 Val loss: -0.5326286527408298 Train ndcg_1 0.9028130173683167 Train ndcg_5 0.9484610557556152 Train ndcg_10 0.9529234766960144 Train mrr_1 0.9026862978935242 Train mrr_5 0.9384377598762512 Train mrr_10 0.9391921162605286 Train ap_1 0.23897619545459747 Train ap_5 0.2722625434398651 Train ap_10 0.27252301573753357 Val ndcg_1 0.9087750315666199 Val ndcg_5 0.9546259045600891 Val ndcg_10 0.9579418897628784 Val mrr_1 0.519548237323761 Val mrr_5 0.5557485818862915 Val mrr_10 0.5559899806976318 Val ap_1 0.29278889298439026 Val ap_5 0.32736751437187195 Val ap_10 0.3269151747226715
2024-01-11 22:13:02,781 - allrank.utils.ltr_logging - INFO - Current:0.9546259045600891 Best:0.9546259045600891
2024-01-11 22:13:03,032 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.98290604, 'ndcg_5': 0.98323363, 'ndcg_10': 0.9909921, 'mrr_1': 0.52991456, 'mrr_5': 0.5370371, 'mrr_10': 0.5370371, 'ap_1': 0.17094018, 'ap_5': 0.17539175, 'ap_10': 0.17233358}

### 2. Without feature 0 "Query term ratio in instructions"

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth #DONE
```

### 3. Without feature 1 "Relevance between title and search query"

Time 1:
2024-01-11 22:16:50,964 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.8599708624980924 Val loss: -0.5326286527408298 Train ndcg_1 0.9001520872116089 Train ndcg_5 0.9470900297164917 Train ndcg_10 0.9515610933303833 Train mrr_1 0.8996452689170837 Train mrr_5 0.9365096092224121 Train mrr_10 0.9372721314430237 Train ap_1 0.23618856072425842 Train ap_5 0.27152130007743835 Train ap_10 0.27147695422172546 Val ndcg_1 0.9087750315666199 Val ndcg_5 0.9546259045600891 Val ndcg_10 0.9579418897628784 Val mrr_1 0.519548237323761 Val mrr_5 0.5557485818862915 Val mrr_10 0.5559899806976318 Val ap_1 0.29278889298439026 Val ap_5 0.32736751437187195 Val ap_10 0.3269151747226715
2024-01-11 22:16:50,964 - allrank.utils.ltr_logging - INFO - Current:0.9546259045600891 Best:0.9546259045600891
2024-01-11 22:16:51,218 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.97435904, 'ndcg_5': 0.9850009, 'ndcg_10': 0.9902063, 'mrr_1': 0.52136755, 'mrr_5': 0.5320513, 'mrr_10': 0.5320513, 'ap_1': 0.16239317, 'ap_5': 0.17229345, 'ap_10': 0.1737746}

### 4. Without feature 2 "Keyword ratio"

Time 1:

### 5. Without feature 3 "Instructions completion degree"

Time 1:
2024-01-11 22:18:45,700 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.8599706298194437 Val loss: -0.5326286527408298 Train ndcg_1 0.9059807658195496 Train ndcg_5 0.9502905607223511 Train ndcg_10 0.9547591209411621 Train mrr_1 0.9086416959762573 Train mrr_5 0.9428598880767822 Train mrr_10 0.943534791469574 Train ap_1 0.24303092062473297 Train ap_5 0.2759297788143158 Train ap_10 0.27584463357925415 Val ndcg_1 0.9087750315666199 Val ndcg_5 0.9546259045600891 Val ndcg_10 0.9579418897628784 Val mrr_1 0.519548237323761 Val mrr_5 0.5557485818862915 Val mrr_10 0.5559899806976318 Val ap_1 0.29278889298439026 Val ap_5 0.32736751437187195 Val ap_10 0.3269151747226715
2024-01-11 22:18:45,700 - allrank.utils.ltr_logging - INFO - Current:0.9546259045600891 Best:0.9607665538787842
2024-01-11 22:18:45,953 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.98290604, 'ndcg_5': 0.98323363, 'ndcg_10': 0.9909921, 'mrr_1': 0.52991456, 'mrr_5': 0.5370371, 'mrr_10': 0.5370371, 'ap_1': 0.17094018, 'ap_5': 0.17539175, 'ap_10': 0.17233358}

### 6. Without feature 4-7 "Action term ratio in instructions (avg, min, max, var)"

### 7. Without feature 8-11 "UI term matching ratio in instructions (avg, min, max, var)"

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

[INFO] 2024-01-16 19:01:20 - Epoch : 19 Train loss: -0.8611217010643539 Val loss: -0.38119486967722577 Train ndcg_1 0.9045869708061218 Train ndcg_5 0.9499666094779968 Train ndcg_10 0.9542075395584106 Train mrr_1 0.901165783405304 Train mrr_5 0.9375000596046448 Train mrr_10 0.9383748769760132 Train ap_1 0.2355550080537796 Train ap_5 0.2709049582481384 Train ap_10 0.2710127532482147 Train precision_1 0.23808921873569489 Train precision_5 0.17067918181419373 Val ndcg_1 0.9236111044883728 Val ndcg_5 0.9552834033966064 Val ndcg_10 0.9603060483932495 Val mrr_1 0.3680555522441864 Val mrr_5 0.39207175374031067 Val mrr_10 0.3933945298194885 Val ap_1 0.2413194477558136 Val ap_5 0.2645905911922455 Val ap_10 0.26458829641342163 Val precision_1 0.2413194477558136 Val precision_5 0.19930556416511536
[INFO] 2024-01-16 19:01:20 - Current:0.9552834033966064 Best:0.999628484249115
[INFO] 2024-01-16 19:01:21 - Test metrics: {'ndcg_1': 0.8730435, 'ndcg_5': 0.9395627, 'ndcg_10': 0.9429519, 'mrr_1': 0.6504348, 'mrr_5': 0.6989275, 'mrr_10': 0.6998765, 'ap_1': 0.2573913, 'ap_5': 0.30639133, 'ap_10': 0.3077341, 'precision_1': 0.2573913, 'precision_5': 0.1798261}

listNet Time 1:

lambdarank Time 1:

approxndc Time 1:

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

neuralNDCG Time 1:

[INFO] 2024-01-16 18:51:10 - Epoch : 19 Train loss: -0.8611217010643539 Val loss: -0.5326286494265837 Train ndcg_1 0.9045869708061218 Train ndcg_5 0.9499666094779968 Train ndcg_10 0.9542075395584106 Train mrr_1 0.901165783405304 Train mrr_5 0.9375000596046448 Train mrr_10 0.9383748769760132 Train ap_1 0.2355550080537796 Train ap_5 0.2709049582481384 Train ap_10 0.2710127532482147 Train precision_1 0.23808921873569489 Train precision_5 0.17067918181419373 Val ndcg_1 0.8792354464530945 Val ndcg_5 0.9410944581031799 Val ndcg_10 0.945366382598877 Val mrr_1 0.49000871181488037 Val mrr_5 0.5360845923423767 Val mrr_10 0.5363162755966187 Val ap_1 0.26324936747550964 Val ap_5 0.3101940453052521 Val ap_10 0.31034404039382935 Val precision_1 0.26324936747550964 Val precision_5 0.21112076938152313
[INFO] 2024-01-16 18:51:10 - Current:0.9410944581031799 Best:0.9991475343704224
[INFO] 2024-01-16 18:51:10 - Test metrics: {'ndcg_1': 0.97435904, 'ndcg_5': 0.98032117, 'ndcg_10': 0.986086, 'mrr_1': 0.52136755, 'mrr_5': 0.53418803, 'mrr_10': 0.53418803, 'ap_1': 0.16239317, 'ap_5': 0.16716525, 'ap_10': 0.16532335, 'precision_1': 0.16239317, 'precision_5': 0.10256412}

listNet Time 1:

lambdarank Time 1:

approxndc Time 1:


### neuralNDCG, listNet, lambdarank, approxndc on dataset "How-to" META-GUI via Sigmoid LTR

#### neuralNDCG

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans # METAGUI -> WeWeb
```

"How-to" META-GUI

2024-01-13 20:43:03,597 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.9543615473567443 Val loss: -0.42135993639628094 Train ndcg_1 0.9996199011802673 Train ndcg_5 0.9998076558113098 Train ndcg_10 0.9998289346694946 Train mrr_1 0.9996199011802673 Train mrr_5 0.9997571110725403 Train mrr_10 0.9997571110725403 Train ap_1 0.33590978384017944 Train ap_5 0.3360460102558136 Train ap_10 0.33605340123176575 Val ndcg_1 1.0 Val ndcg_5 0.999628484249115 Val ndcg_10 0.9998435974121094 Val mrr_1 0.4444444477558136 Val mrr_5 0.4444444477558136 Val mrr_10 0.4444444477558136 Val ap_1 0.3177083432674408 Val ap_5 0.31736835837364197 Val ap_10 0.31739211082458496
2024-01-13 20:43:03,603 - allrank.utils.ltr_logging - INFO - Current:0.999628484249115 Best:0.9997056126594543
2024-01-13 20:43:03,942 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.9982609, 'ndcg_5': 0.9992387, 'ndcg_10': 0.9990704, 'mrr_1': 0.77565217, 'mrr_5': 0.77652174, 'mrr_10': 0.77652174, 'ap_1': 0.3826087, 'ap_5': 0.3836232, 'ap_10': 0.38342446}

"How-to" WeWeb

2024-01-13 20:43:17,182 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.9183958172798157 Val loss: -0.8798767924308777 Train ndcg_1 0.9555555582046509 Train ndcg_5 0.9794867634773254 Train ndcg_10 0.984341025352478 Train mrr_1 0.9555555582046509 Train mrr_5 0.9777777791023254 Train mrr_10 0.9777777791023254 Train ap_1 0.3333333432674408 Train ap_5 0.3548148274421692 Val ndcg_1 1.0 Val ndcg_5 0.9799532294273376 Val ndcg_10 0.9892464280128479 Val mrr_1 1.0 Val mrr_5 1.0 Val mrr_10 1.0 Val ap_1 0.38461539149284363 Val ap_5 0.3619658052921295
2024-01-13 20:43:17,185 - allrank.utils.ltr_logging - INFO - Current:0.9799532294273376 Best:0.9862615466117859
2024-01-13 20:43:17,386 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.9830508, 'ndcg_5': 0.9920277, 'ndcg_10': 0.9956743, 'mrr_1': 0.9830508, 'mrr_5': 0.9915254, 'mrr_10': 0.9915254, 'ap_1': 0.016949153, 'ap_5': 0.027777776}

Zero-shot

2024-01-13 20:45:50,433 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.9543615473567443 Val loss: -0.5785580262943105 Train ndcg_1 0.9996199011802673 Train ndcg_5 0.9998076558113098 Train ndcg_10 0.9998289346694946 Train mrr_1 0.9996199011802673 Train mrr_5 0.9997571110725403 Train mrr_10 0.9997571110725403 Train ap_1 0.33590978384017944 Train ap_5 0.3360460102558136 Train ap_10 0.33605340123176575 Val ndcg_1 1.0 Val ndcg_5 0.9997694492340088 Val ndcg_10 0.9998878240585327 Val mrr_1 0.6107732653617859 Val mrr_5 0.6107732653617859 Val mrr_10 0.6107732653617859 Val ap_1 0.38401392102241516 Val ap_5 0.3837822377681732 Val ap_10 0.38372915983200073
2024-01-13 20:45:50,434 - allrank.utils.ltr_logging - INFO - Current:0.9997694492340088 Best:0.9995319247245789
2024-01-13 20:45:50,660 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.99145305, 'ndcg_5': 0.9853995, 'ndcg_10': 0.9911009, 'mrr_1': 0.53846157, 'mrr_5': 0.54273504, 'mrr_10': 0.54273504, 'ap_1': 0.17948718, 'ap_5': 0.17526118, 'ap_10': 0.17198345}

#### listNet

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_metagui_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_metagui_ground_truth_no_trans --job-dir experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_metagui_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_weweb_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_weweb_ground_truth_no_trans --job-dir experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_weweb_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth_no_trans.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth_no_trans --job-dir experiments/listNet/neuralndcg_atmax_Multimodal_Feature18_label2_listNet_on_cohere_ground_truth_no_trans # METAGUI -> WeWeb
```

"How-to" META-GUI

2024-01-13 21:38:03,781 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 0.0 Val loss: 0.4932870931095547 Train ndcg_1 1.0 Train ndcg_5 1.0 Train ndcg_10 1.0 Train mrr_1 1.0 Train mrr_5 1.0 Train mrr_10 1.0 Train ap_1 0.3362899422645569 Train ap_5 0.7678619623184204 Train ap_10 0.9849826097488403 Val ndcg_1 0.9166666865348816 Val ndcg_5 0.9547337293624878 Val ndcg_10 0.9586650133132935 Val mrr_1 0.3611111044883728 Val mrr_5 0.38952547311782837 Val mrr_10 0.3900318443775177 Val ap_1 0.234375 Val ap_5 0.26288822293281555 Val ap_10 0.2621814012527466
2024-01-13 21:38:03,783 - allrank.utils.ltr_logging - INFO - Current:0.9547337293624878 Best:0.9547337293624878
2024-01-13 21:38:04,127 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.86260873, 'ndcg_5': 0.93177235, 'ndcg_10': 0.9373378, 'mrr_1': 0.64, 'mrr_5': 0.6914783, 'mrr_10': 0.69272876, 'ap_1': 0.24695653, 'ap_5': 0.29928988, 'ap_10': 0.300237}

"How-to" WeWeb

2024-01-13 21:38:16,115 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 0.0 Val loss: 0.6879475116729736 Train ndcg_1 1.0 Train ndcg_5 1.0 Train ndcg_10 1.0 Train mrr_1 1.0 Train mrr_5 1.0 Train mrr_10 1.0 Train ap_1 0.3777777850627899 Train ap_5 0.8625925779342651 Val ndcg_1 0.9230769872665405 Val ndcg_5 0.9526470899581909 Val ndcg_10 0.9619402885437012 Val mrr_1 0.9230769872665405 Val mrr_5 0.9615384936332703 Val mrr_10 0.9615384936332703 Val ap_1 0.3076923191547394 Val ap_5 0.32467949390411377
2024-01-13 21:38:16,119 - allrank.utils.ltr_logging - INFO - Current:0.9526470899581909 Best:0.9526470899581909
2024-01-13 21:38:16,299 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.9830508, 'ndcg_5': 0.9913773, 'ndcg_10': 0.9931907, 'mrr_1': 0.9830508, 'mrr_5': 0.9915254, 'mrr_10': 0.9915254, 'ap_1': 0.016949153, 'ap_5': 0.0259887}

Zero-shot

2024-01-13 21:39:04,894 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 0.0 Val loss: 0.5133136443722569 Train ndcg_1 1.0 Train ndcg_5 1.0 Train ndcg_10 1.0 Train mrr_1 1.0 Train mrr_5 1.0 Train mrr_10 1.0 Train ap_1 0.3362899422645569 Train ap_5 0.7678619623184204 Train ap_10 0.9849826097488403 Val ndcg_1 0.8757602572441101 Val ndcg_5 0.9325862526893616 Val ndcg_10 0.9399206638336182 Val mrr_1 0.4865334630012512 Val mrr_5 0.5290616750717163 Val mrr_10 0.5305718183517456 Val ap_1 0.2597741186618805 Val ap_5 0.30236393213272095 Val ap_10 0.30362439155578613
2024-01-13 21:39:04,898 - allrank.utils.ltr_logging - INFO - Current:0.9325862526893616 Best:0.9325862526893616
2024-01-13 21:39:05,126 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.99145305, 'ndcg_5': 0.9868452, 'ndcg_10': 0.99194443, 'mrr_1': 0.53846157, 'mrr_5': 0.54273504, 'mrr_10': 0.54273504, 'ap_1': 0.17948718, 'ap_5': 0.17757599, 'ap_10': 0.17502747}

#### lambdarank

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans --job-dir experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans --job-dir experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans.json --run-id lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans --job-dir experiments/lambdarank/lambdarank_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans # METAGUI -> WeWeb
```

"How-to" META-GUI

2024-01-13 21:43:56,487 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 8.478063752671758 Val loss: 10.180869572692448 Train ndcg_1 0.9996199011802673 Train ndcg_5 0.9998396039009094 Train ndcg_10 0.9998387098312378 Train mrr_1 0.9996199011802673 Train mrr_5 0.9997888207435608 Train mrr_10 0.9997888207435608 Train ap_1 0.33590978384017944 Train ap_5 0.3360583484172821 Train ap_10 0.33605530858039856 Val ndcg_1 1.0 Val ndcg_5 0.999628484249115 Val ndcg_10 0.9998435974121094 Val mrr_1 0.4444444477558136 Val mrr_5 0.4444444477558136 Val mrr_10 0.4444444477558136 Val ap_1 0.3177083432674408 Val ap_5 0.31736835837364197 Val ap_10 0.31739211082458496
2024-01-13 21:43:56,490 - allrank.utils.ltr_logging - INFO - Current:0.999628484249115 Best:1.0
2024-01-13 21:43:56,839 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.9982609, 'ndcg_5': 0.99946684, 'ndcg_10': 0.99946684, 'mrr_1': 0.77565217, 'mrr_5': 0.77652174, 'mrr_10': 0.77652174, 'ap_1': 0.3826087, 'ap_5': 0.3836232, 'ap_10': 0.3836232}

"How-to" WeWeb

2024-01-13 21:44:09,981 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 6.458059310913086 Val loss: 3.359440803527832 Train ndcg_1 0.9555555582046509 Train ndcg_5 0.9632382392883301 Train ndcg_10 0.9759992957115173 Train mrr_1 0.9555555582046509 Train mrr_5 0.9666666984558105 Train mrr_10 0.9703704118728638 Train ap_1 0.3333333432674408 Train ap_5 0.3406481444835663 Val ndcg_1 1.0 Val ndcg_5 0.9815351366996765 Val ndcg_10 0.9908283352851868 Val mrr_1 1.0 Val mrr_5 1.0 Val mrr_10 1.0 Val ap_1 0.38461539149284363 Val ap_5 0.3658119738101959
2024-01-13 21:44:09,983 - allrank.utils.ltr_logging - INFO - Current:0.9815351366996765 Best:0.9844529628753662
2024-01-13 21:44:10,191 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.9830508, 'ndcg_5': 0.9920277, 'ndcg_10': 0.9957573, 'mrr_1': 0.9830508, 'mrr_5': 0.9915254, 'mrr_10': 0.9915254, 'ap_1': 0.016949153, 'ap_5': 0.027777776}

Zero-shot

2024-01-13 21:45:32,167 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 8.478063752671758 Val loss: 7.569999424920302 Train ndcg_1 0.9996199011802673 Train ndcg_5 0.9998396039009094 Train ndcg_10 0.9998387098312378 Train mrr_1 0.9996199011802673 Train mrr_5 0.9997888207435608 Train mrr_10 0.9997888207435608 Train ap_1 0.33590978384017944 Train ap_5 0.3360583484172821 Train ap_10 0.33605530858039856 Val ndcg_1 1.0 Val ndcg_5 0.9997962117195129 Val ndcg_10 0.99993896484375 Val mrr_1 0.6107732653617859 Val mrr_5 0.6107732653617859 Val mrr_10 0.6107732653617859 Val ap_1 0.38401392102241516 Val ap_5 0.38401392102241516 Val ap_10 0.38385945558547974
2024-01-13 21:45:32,167 - allrank.utils.ltr_logging - INFO - Current:0.9997962117195129 Best:0.9997962117195129
2024-01-13 21:45:32,400 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.98290604, 'ndcg_5': 0.98445654, 'ndcg_10': 0.9903541, 'mrr_1': 0.52991456, 'mrr_5': 0.53846157, 'mrr_10': 0.53846157, 'ap_1': 0.17094018, 'ap_5': 0.17384854, 'ap_10': 0.17181525}

#### approxndc

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/approxndc/approxndc_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans.json --run-id approxndc_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans --job-dir experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_metagui_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/approxndc/approxndc_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans.json --run-id approxndc_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans --job-dir experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_weweb_ground_truth_no_trans && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/approxndc/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans.json --run-id approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans --job-dir experiments/approxndc/approxndc_Multimodal_Feature18_label2_on_cohere_ground_truth_no_trans # METAGUI -> WeWeb
```

"How-to" META-GUI

2024-01-13 21:54:36,743 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.2674705851549312 Val loss: -0.2544657289981842 Train ndcg_1 0.9996199011802673 Train ndcg_5 0.9998497366905212 Train ndcg_10 0.9998497366905212 Train mrr_1 0.9996199011802673 Train mrr_5 0.9997888207435608 Train mrr_10 0.9997888207435608 Train ap_1 0.33590978384017944 Train ap_5 0.3360794484615326 Train ap_10 0.3360794484615326 Val ndcg_1 1.0 Val ndcg_5 0.999628484249115 Val ndcg_10 0.9998435974121094 Val mrr_1 0.4444444477558136 Val mrr_5 0.4444444477558136 Val mrr_10 0.4444444477558136 Val ap_1 0.3177083432674408 Val ap_5 0.31736835837364197 Val ap_10 0.31739211082458496
2024-01-13 21:54:36,743 - allrank.utils.ltr_logging - INFO - Current:0.999628484249115 Best:0.9996328353881836
2024-01-13 21:54:37,088 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.9982609, 'ndcg_5': 0.99946684, 'ndcg_10': 0.99945486, 'mrr_1': 0.77565217, 'mrr_5': 0.77652174, 'mrr_10': 0.77652174, 'ap_1': 0.3826087, 'ap_5': 0.3836232, 'ap_10': 0.3835818}

"How-to" WeWeb

2024-01-13 21:54:49,756 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.3151623606681824 Val loss: -0.2801719903945923 Train ndcg_1 0.9555555582046509 Train ndcg_5 0.9635684490203857 Train ndcg_10 0.9752347469329834 Train mrr_1 0.9555555582046509 Train mrr_5 0.9666666984558105 Train mrr_10 0.9694444537162781 Train ap_1 0.3333333432674408 Train ap_5 0.34175926446914673 Val ndcg_1 0.9230769872665405 Val ndcg_5 0.9682123064994812 Val ndcg_10 0.9775055050849915 Val mrr_1 0.9230769872665405 Val mrr_5 0.9615384936332703 Val mrr_10 0.9615384936332703 Val ap_1 0.3076923191547394 Val ap_5 0.3529914617538452
2024-01-13 21:54:49,760 - allrank.utils.ltr_logging - INFO - Current:0.9682123064994812 Best:0.9810370802879333
2024-01-13 21:54:49,956 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.9830508, 'ndcg_5': 0.989552, 'ndcg_10': 0.99532926, 'mrr_1': 0.9830508, 'mrr_5': 0.9915254, 'mrr_10': 0.9915254, 'ap_1': 0.016949153, 'ap_5': 0.026836159}

Zero-shot

2024-01-13 21:56:11,050 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.2674705851549312 Val loss: -0.3119836721235933 Train ndcg_1 0.9996199011802673 Train ndcg_5 0.9998497366905212 Train ndcg_10 0.9998497366905212 Train mrr_1 0.9996199011802673 Train mrr_5 0.9997888207435608 Train mrr_10 0.9997888207435608 Train ap_1 0.33590978384017944 Train ap_5 0.3360794484615326 Train ap_10 0.3360794484615326 Val ndcg_1 1.0 Val ndcg_5 0.9997962117195129 Val ndcg_10 0.9999414086341858 Val mrr_1 0.6107732653617859 Val mrr_5 0.6107732653617859 Val mrr_10 0.6107732653617859 Val ap_1 0.38401392102241516 Val ap_5 0.38401392102241516 Val ap_10 0.38386908173561096
2024-01-13 21:56:11,053 - allrank.utils.ltr_logging - INFO - Current:0.9997962117195129 Best:0.9999539256095886
2024-01-13 21:56:11,279 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.98290604, 'ndcg_5': 0.98412853, 'ndcg_10': 0.99006146, 'mrr_1': 0.52991456, 'mrr_5': 0.53846157, 'mrr_10': 0.53846157, 'ap_1': 0.17094018, 'ap_5': 0.17294635, 'ap_10': 0.17127392}

### Pointwise Loss

pointwise bce loss

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/bce/pointwise_bce_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model.json --run-id pointwise_bce_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model --job-dir experiments/bce/pointwise_bce_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/bce/pointwise_bce_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model.json --run-id pointwise_bce_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model --job-dir experiments/bce/pointwise_bce_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/bce/pointwise_bce_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model.json --run-id pointwise_bce_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model --job-dir experiments/bce/pointwise_bce_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model # on METAGUI -> WeWeb
```

* META-GUI

2024-01-13 18:20:39,951 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 2.4061221820838066 Val loss: 2.915162728892432 Train ndcg_1 0.8812721967697144 Train ndcg_5 0.9380508661270142 Train ndcg_10 0.9435099959373474 Train mrr_1 0.8812721967697144 Train mrr_5 0.924353837966919 Train mrr_10 0.9253323674201965 Train ap_1 0.21756209433078766 Train ap_5 0.25966590642929077 Train ap_10 0.2601567804813385 Val ndcg_1 0.9027777910232544 Val ndcg_5 0.9486153721809387 Val ndcg_10 0.9538309574127197 Val mrr_1 0.347222238779068 Val mrr_5 0.38029518723487854 Val mrr_10 0.3813698887825012 Val ap_1 0.220486119389534 Val ap_5 0.2557339668273926 Val ap_10 0.2561280429363251
2024-01-13 18:20:39,954 - allrank.utils.ltr_logging - INFO - Current:0.9486153721809387 Best:0.9485266208648682
2024-01-13 18:20:40,272 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.85391307, 'ndcg_5': 0.93252105, 'ndcg_10': 0.93622977, 'mrr_1': 0.6313044, 'mrr_5': 0.68811595, 'mrr_10': 0.6889856, 'ap_1': 0.23826088, 'ap_5': 0.2977778, 'ap_10': 0.29917645}

* WeWeb

2024-01-13 18:20:52,383 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 2.3240435123443604 Val loss: 3.482133388519287 Train ndcg_1 0.9333333373069763 Train ndcg_5 0.9540493488311768 Train ndcg_10 0.9639526605606079 Train mrr_1 0.9333333373069763 Train mrr_5 0.9562963247299194 Train mrr_10 0.9562963247299194 Train ap_1 0.31111112236976624 Train ap_5 0.32484567165374756 Val ndcg_1 0.8461538553237915 Val ndcg_5 0.9268097877502441 Val ndcg_10 0.9361029267311096 Val mrr_1 0.8461538553237915 Val mrr_5 0.9102565050125122 Val mrr_10 0.9102565050125122 Val ap_1 0.23076924681663513 Val ap_5 0.2974359095096588
2024-01-13 18:20:52,386 - allrank.utils.ltr_logging - INFO - Current:0.9268097877502441 Best:0.9268097877502441
2024-01-13 18:20:52,581 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 1.0, 'ndcg_5': 0.9912751, 'ndcg_10': 0.99513614, 'mrr_1': 1.0, 'mrr_5': 1.0, 'mrr_10': 1.0, 'ap_1': 0.033898305, 'ap_5': 0.029661017}

* Zero-shot

2024-01-13 18:22:14,064 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 2.4061221820838066 Val loss: 2.4725071565676107 Train ndcg_1 0.8812721967697144 Train ndcg_5 0.9380508661270142 Train ndcg_10 0.9435099959373474 Train mrr_1 0.8812721967697144 Train mrr_5 0.924353837966919 Train mrr_10 0.9253323674201965 Train ap_1 0.21756209433078766 Train ap_5 0.25966590642929077 Train ap_10 0.2601567804813385 Val ndcg_1 0.8627281188964844 Val ndcg_5 0.9305370450019836 Val ndcg_10 0.936080813407898 Val mrr_1 0.4735013246536255 Val mrr_5 0.5213003158569336 Val mrr_10 0.5220381617546082 Val ap_1 0.24674196541309357 Val ap_5 0.29723551869392395 Val ap_10 0.29832491278648376
2024-01-13 18:22:14,067 - allrank.utils.ltr_logging - INFO - Current:0.9305370450019836 Best:0.9299390912055969
2024-01-13 18:22:14,277 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.98290604, 'ndcg_5': 0.9853858, 'ndcg_10': 0.9923209, 'mrr_1': 0.52991456, 'mrr_5': 0.53846157, 'mrr_10': 0.53846157, 'ap_1': 0.17094018, 'ap_5': 0.17891738, 'ap_10': 0.17555182}

pointwise rmse loss

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model.json --run-id pointwise_rmse_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model --job-dir experiments/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_metagui_ground_truth_lr_model && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model.json --run-id pointwise_rmse_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model --job-dir experiments/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_weweb_ground_truth_lr_model && PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model.json --run-id pointwise_rmse_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model --job-dir experiments/pointwise/pointwise_rmse_Multimodal_Feature18_label2_on_cohere_ground_truth_lr_model # METAGUI -> WeWeb
```

* META-GUI

2024-01-13 18:32:29,822 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 0.27578800019430016 Val loss: 0.2783466309515966 Train ndcg_1 0.8616320490837097 Train ndcg_5 0.9245414137840271 Train ndcg_10 0.9321139454841614 Train mrr_1 0.8616320490837097 Train mrr_5 0.9066480994224548 Train mrr_10 0.9086328148841858 Train ap_1 0.1979219615459442 Train ap_5 0.24395503103733063 Train ap_10 0.24624794721603394 Val ndcg_1 0.8940972089767456 Val ndcg_5 0.9430345892906189 Val ndcg_10 0.9482466578483582 Val mrr_1 0.3385416567325592 Val mrr_5 0.37051501870155334 Val mrr_10 0.37187913060188293 Val ap_1 0.2118055522441864 Val ap_5 0.24760320782661438 Val ap_10 0.24946369230747223
2024-01-13 18:32:29,823 - allrank.utils.ltr_logging - INFO - Current:0.9430345892906189 Best:0.9439597725868225
2024-01-13 18:32:30,147 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.8486957, 'ndcg_5': 0.9265971, 'ndcg_10': 0.9315074, 'mrr_1': 0.62608695, 'mrr_5': 0.6799421, 'mrr_10': 0.6818158, 'ap_1': 0.23304348, 'ap_5': 0.29092756, 'ap_10': 0.29350153}

* WeWeb

2024-01-13 18:32:42,568 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 0.4141063392162323 Val loss: 0.4083878695964813 Train ndcg_1 0.8888888955116272 Train ndcg_5 0.946821928024292 Train ndcg_10 0.9567252397537231 Train mrr_1 0.8888888955116272 Train mrr_5 0.9370371103286743 Train mrr_10 0.9370371103286743 Train ap_1 0.2666666805744171 Train ap_5 0.31577160954475403 Val ndcg_1 0.8461538553237915 Val ndcg_5 0.9268097877502441 Val ndcg_10 0.9361029267311096 Val mrr_1 0.8461538553237915 Val mrr_5 0.9102565050125122 Val mrr_10 0.9102565050125122 Val ap_1 0.23076924681663513 Val ap_5 0.2974359095096588
2024-01-13 18:32:42,570 - allrank.utils.ltr_logging - INFO - Current:0.9268097877502441 Best:0.9268097877502441
2024-01-13 18:32:42,763 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 1.0, 'ndcg_5': 0.9910232, 'ndcg_10': 0.99498695, 'mrr_1': 1.0, 'mrr_5': 1.0, 'mrr_10': 1.0, 'ap_1': 0.033898305, 'ap_5': 0.02881356}

* Zero-shot

2024-01-13 18:34:03,623 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: 0.27578800019430016 Val loss: 0.33086358092330415 Train ndcg_1 0.8616320490837097 Train ndcg_5 0.9245414137840271 Train ndcg_10 0.9321139454841614 Train mrr_1 0.8616320490837097 Train mrr_5 0.9066480994224548 Train mrr_10 0.9086328148841858 Train ap_1 0.1979219615459442 Train ap_5 0.24395503103733063 Train ap_10 0.24624794721603394 Val ndcg_1 0.8540399670600891 Val ndcg_5 0.9252707362174988 Val ndcg_10 0.9321764707565308 Val mrr_1 0.464813232421875 Val mrr_5 0.5138285756111145 Val mrr_10 0.5149456262588501 Val ap_1 0.23805387318134308 Val ap_5 0.2915242910385132 Val ap_10 0.29392173886299133
2024-01-13 18:34:03,623 - allrank.utils.ltr_logging - INFO - Current:0.9252707362174988 Best:0.9257491230964661
2024-01-13 18:34:03,835 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.948718, 'ndcg_5': 0.9772547, 'ndcg_10': 0.9833162, 'mrr_1': 0.49572653, 'mrr_5': 0.51994306, 'mrr_10': 0.51994306, 'ap_1': 0.13675214, 'ap_5': 0.16342592, 'ap_10': 0.16355799}

## Failure analysis

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_ground_truth_extra --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth_failure_analysis
```

## License

Apache 2 License
