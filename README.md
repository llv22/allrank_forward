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
2024-01-11 22:04:52,944 - allrank.utils.ltr_logging - INFO - Epoch : 19 Train loss: -0.8616754800851613 Val loss: -0.5326286527408298 Train ndcg_1 0.9028130173683167 Train ndcg_5 0.9484610557556152 Train ndcg_10 0.9529234766960144 Train mrr_1 0.9026862978935242 Train mrr_5 0.9384377598762512 Train mrr_10 0.9391921162605286 Train ap_1 0.23897619545459747 Train ap_5 0.2722625434398651 Train ap_10 0.27252301573753357 Val ndcg_1 0.9087750315666199 Val ndcg_5 0.9546259045600891 Val ndcg_10 0.9579418897628784 Val mrr_1 0.519548237323761 Val mrr_5 0.5557485818862915 Val mrr_10 0.5559899806976318 Val ap_1 0.29278889298439026 Val ap_5 0.32736751437187195 Val ap_10 0.3269151747226715
2024-01-11 22:04:52,945 - allrank.utils.ltr_logging - INFO - Current:0.9546259045600891 Best:0.9546259045600891
2024-01-11 22:04:53,195 - allrank.utils.ltr_logging - INFO - Test metrics: {'ndcg_1': 0.98290604, 'ndcg_5': 0.98323363, 'ndcg_10': 0.9909921, 'mrr_1': 0.52991456, 'mrr_5': 0.5370371, 'mrr_10': 0.5370371, 'ap_1': 0.17094018, 'ap_5': 0.17539175, 'ap_10': 0.17233358}

### 2. Without feature 0 "Query term ratio in instructions"

```bash
PYTHONPATH=.:${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json --run-id neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth --job-dir experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth #DONE
```


### 3. Without feature 1 "Relevance between title and search query"

### 4. Without feature 2 "Keyword ratio"

### 5. Without feature 3 "Instructions completion degree"

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

## License

Apache 2 License
