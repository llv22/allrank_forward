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

```
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

## Reproduce on Set1 on dataset/MSLR-WEB30K/Fold1

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
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Fold1_normalized.json --run-id ranallrank_NeuralNDCG_Fold1_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold1_normalized # DONE
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Fold2_normalized.json --run-id ranallrank_NeuralNDCG_Fold2_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold2_normalized
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Fold3_normalized.json --run-id ranallrank_NeuralNDCG_Fold3_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold3_normalized
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Fold4_normalized.json --run-id ranallrank_NeuralNDCG_Fold4_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold4_normalized
PYTHONPATH=.:${PYTHONPATH} python allrank/main.py  --config-file-name allrank/neuralndcg_atmax_Fold5_normalized.json --run-id ranallrank_NeuralNDCG_Fold5_normalized --job-dir experiments/ranallrank_NeuralNDCG_Fold5_normalized
```

NeuralNDCG on MSLR-WEB30K Fold1 to Fold5

| Dataset | Train Loss          | Train NDCG@1       | Train NDCG@5       | Train NDCG@10      | Train NDCG@30      | Train NDCG@60      | Val Loss            | Val NDCG@1         | Val NDCG@5         | Val NDCG@10        | Val NDCG@30        | Val NDCG@60        |
|---------|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| Fold1   | -0.7669206603150948 | 0.6319156289100647 | 0.5951358675956726 | 0.595418393611908  | 0.6312896013259888 | 0.6902117133140564 | -0.7294725948502213 | 0.517143189907074  | 0.5089449882507324 | 0.5275285840034485 | 0.5840967297554016 | 0.6481217741966248 |
| Fold2   | -0.7665537890652948 | 0.6281455755233765 | 0.5941371917724609 | 0.5944378972053528 | 0.6313827037811279 | 0.6902371048927307 | -0.7269774124779981 | 0.5167470574378967 | 0.5107281804084778 | 0.527868390083313  | 0.5821496248245239 | 0.6455816030502319 |
| Fold3   | -0.7668627891849042 | 0.6241025328636169 | 0.5938913822174072 | 0.5951226353645325 | 0.6322952508926392 | 0.6909143924713135 | -0.732415835629786  | 0.532313883304596  | 0.5211151838302612 | 0.5354329347610474 | 0.5869977474212646 | 0.6501902341842651 |
| Fold4   | -0.7662994338049748 | 0.6262935996055603 | 0.5946605801582336 | 0.5965979099273682 | 0.6328622698783875 | 0.6911565065383911 | -0.7308251556804208 | 0.5148641467094421 | 0.5154050588607788 | 0.5328453779220581 | 0.5857040286064148 | 0.6489840745925903 |
| Fold5   | -0.7675414259737834 | 0.6310990452766418 | 0.5953048467636108 | 0.5956495404243469 | 0.632238507270813  | 0.6907401084899902 | -0.727307596510643  | 0.5146557092666626 | 0.5060291290283203 | 0.5266775488853455 | 0.5825557708740234 | 0.646621823310852  |

### Prepare features

refer to [Scalable, neural learning to rank (LTR) models](https://www.tensorflow.org/ranking) and [Neural Learning to Rank using TensorFlow ICTIR 2019](https://bendersky.github.io/res/TF-Ranking-ICTIR-2019.pdf)

```bash
python reproducibility/tfrecord_features.py --ds_path dataset/MSLR-WEB30K/Fold1 --out_tf_path dataset/MSLR-WEB30K/Fold1_tfrecord
python reproducibility/tfrecord_features.py --ds_path dataset/MSLR-WEB30K/Fold2 --out_tf_path dataset/MSLR-WEB30K/Fold2_tfrecord
python reproducibility/tfrecord_features.py --ds_path dataset/MSLR-WEB30K/Fold3 --out_tf_path dataset/MSLR-WEB30K/Fold3_tfrecord
python reproducibility/tfrecord_features.py --ds_path dataset/MSLR-WEB30K/Fold4 --out_tf_path dataset/MSLR-WEB30K/Fold4_tfrecord
python reproducibility/tfrecord_features.py --ds_path dataset/MSLR-WEB30K/Fold5 --out_tf_path dataset/MSLR-WEB30K/Fold5_tfrecord
```

## License

Apache 2 License
