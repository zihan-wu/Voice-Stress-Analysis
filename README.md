# Testing data-driven and DSP-based features on downstream Voice Stress Analysis (VSA) datasets:

This is the code base for the paper: Speaker Embeddings as Individuality Proxy for Voice Stress Detection https://www.isca-speech.org/archive/pdfs/interspeech_2023/wu23i_interspeech.pdf 

This directory includes:
* Main script (evaluate_clf) that load data, extract embeddings, train classifier and get UAR (unweighted average recall) scores for train and test sets
* Utils file with helper functions
* Settings file including some constants (e.g: datasets paths 'change it based on where your data is located', datasets stats used for normalization in BYOL-S models). Also, it defines all models (data-driven or dsp-based) for testing.
* nn_trainer file with MLP and Transformer classifier. You can specify 
* BYOL-A utility functions in byola directory.
* BYOL-A and tranformer-inspired models in byola directory.
## Environment:
```
conda env create -f environment.yml
```

## Run Script:
1. First make sure to have selected models (you can select multiple) specified in `_MODEL_DICT` in `settings.py`. Then, specify the checkpoint path in `_WEIGHT_DICT` in `settings.py`.

2. In `settings.py`, please also set `_VSA_PATH` to your path to the VSA data and `_EMBED_PATH` to path where you want to store embeddings.

3. If your model outputs timestamp embeddings, make sure `MAX_WINS` > `CLIP_LEN`/(hop_length) in `settings.py`, whether receptive field is the hop_length (in second) of timestamp embeddings.

4. By default, data is chunked into `CLIP_LEN` in `settings.py`. If you want to remove it, set `clip_audio` to false in the configuration file.

5. By default, the training batch size for MLP or Transformer is 128, and the number of classes is 2. You can change them in the configuration file.

6. Training Process Explained:
* The command below will load the dataset of your choice, iterate over the models defined in the settings.py file generate the embeddings of the dataset for each of these models and save them as npy file named 'speaker_embed_name + model_name + .npy' in a directory named '{EMBED_PATH}/Dataset_{dataset_name}_{speaker_embed_name}_{split_name (train/val/test)}'. Where EMBED_PATH is the one stored in settings.py
* Then, it will train a classifier: SVC, MLP, or Transformer. If you use SVC classifier, the code performs a 5-fold CV and a grid search over SVC hyperparameters on a train set and test the best model on the test dataset. If MLP or Transformer is used, validation set is required in the data split. The validation set is used for early stopping and hyperparameter search.
* After hyperparameter search, the best model will be saved in `clf_results/{sklearn or nn}_models` directory. The inferences on the test dataset will be stored in `clf_results/inferences` directory.
* The reported scores will be saved in `clf_results/summary` directory as a csv file (each column represents a model and two rows for train and test UAR as well as two rows for the 95% CI of test UAR).


7. To Run, use the following code
```console
CUBLAS_WORKSPACE_CONFIG=:4096:8 python evaluate_clf.py --dataset_name {cog1, cog2, cog3, cog4, phy, allcog, all, or all_split} --clf {svc, mlp, transf} --speaker_embed {ecapa, resemblyzer, nospeaker} --config_path{default: config.yaml}
# Milos's replication
CUBLAS_WORKSPACE_CONFIG=:4096:8 python src/VSA/evaluate_clf.py --dataset_name all_split --clf transf --speaker_embed nospeaker --config_path src/VSA/config.yaml
```
* --dataset_name specifies which dataset to use. 'all' combines all five datasets and do train/test split using a 75/25 ratio. 'all_split' also combined all five datasets, but do train/val/test split using a 60/15/25 split.

* --clf specifies which classifier to use. If 'mlp' or 'transf' is used, only 'all_split' dataset_name is supported. 

* --speaker_embed specifies the speaker embedding to use

* config_path specifies the path to configuration file.


