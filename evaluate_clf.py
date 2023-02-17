import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from joblib import dump, load
from shutil import copyfile

from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, PredefinedSplit
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import ParameterGrid
from torchaudio.transforms import MelSpectrogram

from byol_a.common import load_yaml_config
from byol_a.augmentations import PrecomputedNorm
from settings import _CLF_STATS_DICT, _MODELS_DICT, _CLF_GRID, _RANDOM_SEED, _REQUIRED_SAMPLE_RATE, _SCORING, _SPEAKER_EMBED, CLIP_LEN
from nn_trainer import hyparam_search, test_pred
from utils import *

def parse_args():
    """Parse main arguments."""
    parser = ArgumentParser(description='benchmark arguments')

    parser.add_argument(
        '-dataset_name', '--dataset_name',
        type=str, default='all_split',
        help='The dataset used for classification (cog1, cog2, cog3, cog4, phy, allcog, all, or all_split)'
    )

    parser.add_argument(
        '-clf', '--clf',
        type=str, default='mlp',
        help='The classifier used for classification (svc, mlp, or transf)'
    )

    parser.add_argument(
        '-speaker_embed', '--speaker_embed',
        type=str, default='ecapa',
        help='The model for speaker embedding (resemblyzer, ecapa, or nospeaker)'
    )

    parser.add_argument(
        '-config_path', '--config_path',
        type=str, default='config.yaml',
        help='The path to config file'
    )

    return parser.parse_args()

def save_results(fname, results_df, results_folder):
    save_path = os.path.join(results_folder, fname)

    os.makedirs(results_folder, exist_ok=True)

    if not(os.path.exists(save_path)):
        print(f'File {fname} does not exist yet. Creating a new results file.')
        results_df.to_csv(save_path)
    else:
        all_results_df = pd.read_csv(save_path, index_col=0)
        new_all_results_df = all_results_df.join(results_df, how='right', lsuffix='_old')
        columns = new_all_results_df.columns.to_list()
        to_remove = [c for c in columns if c.endswith('_old')]
        new_all_results_df.drop(columns = to_remove, inplace=True)
        new_all_results_df = new_all_results_df.reindex(sorted(new_all_results_df.columns), axis=1)
        new_all_results_df.to_csv(save_path)


def get_embed_name(args, cfg):
    filename = args.dataset_name
    if cfg.clip_audio:
        filename += f'_clip{CLIP_LEN}'
    if args.speaker_embed in _SPEAKER_EMBED:
        prefix = '_'
        if cfg.speaker_only:
            prefix += 'only'
        if cfg.clip_speaker:
            prefix += 'clip'
        filename += prefix + args.speaker_embed

    return filename

def get_sklearn_model():

    svc = SVC(max_iter=1e5, random_state=42)
    params_svc = {
        'estimator__kernel': ['linear'],
        'estimator__C': np.logspace(5, -5, num=11)
    }

    estimator_list = svc
    log_list = 'SVC'
    param_list = params_svc
    return log_list, estimator_list, param_list

def svc_model(embed_data, model_save_path: Path):
    '''
    train svc model
    Args: 
        embed_data: a dictionary of dictionary
            It contains keys 'train', 'test'
            Each embed_data[key] is a dictionary containing keys 'embed', 'label'
        model_save_path: Path
            Path to save model
    Return:
        result: a dictionary of train and test scores

    '''
    print(f"Splitted Embedding Shapes of Train/Test are {embed_data['train']['embed'].shape}, {embed_data['test']['embed'].shape}")
    print(f"Splitted Label Shapes of Train/Test are {embed_data['train']['label'].shape}, {embed_data['test']['label'].shape}")
    
    # Load classifiers
    log, estimator, param = get_sklearn_model()
    clf = estimator

    # Apply 5-fold CV with grid search over SVM clf hyperparameters
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    pipeline = Pipeline([('transformer', StandardScaler()), ('estimator', clf)])
    grid_search = GridSearchCV(pipeline, param_grid=param, n_jobs=-1, cv=cv, verbose=2, scoring=_SCORING)
    grid_result = grid_search.fit(embed_data['train']['embed'], embed_data['train']['label'])
    
    # Collect Result
    test_score = grid_result.score(embed_data['test']['embed'], embed_data['test']['label']) 
    y_pred = grid_result.predict(embed_data['test']['embed'])
    y_true = embed_data['test']['label']
    test_accu= accuracy_score(y_true, y_pred)
    bootstrap_stats = bootstrap(y_true, y_pred)
    print(f'Best {log} UAR: {grid_result.best_score_*100: .2f} using {grid_result.best_params_}')
    print(f'    Test {log} UAR: {test_score*100: .2f}')
    print(f'    Test {log} ACCURACY: {test_accu*100: .2f}')

    # Save Prediction and Model
    inference_file = Path('clf_results/inferences').joinpath(model_save_path.stem + '.csv')
    inference_file.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({'True': y_true, 'Pred': y_pred}).to_csv(inference_file)
    model_save_path.parent.mkdir(exist_ok=True, parents=True)
    dump(grid_result.best_estimator_, model_save_path)
    
    return {
        'train_uar': round(grid_result.best_score_ * 100, 2),
        'test_uar': round(test_score * 100, 2),
        'test_uar_low': round(bootstrap_stats[1] * 100, 2),
        'test_uar_high': round(bootstrap_stats[2] * 100, 2),
    }
    
def nn_model(embed_data, clf_name, model_save_name, norm_embed):
    '''
    train nerual network classifier
    Args: 
        embed_data: a dictionary of dictionary
            It contains keys 'train', 'val', 'test'
            Each embed_data[key] is a dictionary containing keys 'embed', 'label', and 'window' if using transformer
        clf_name: str
            name of classifier
        model_save_name: 
            Name of saved model
        norm_embed: bool
            Whether to normalize embedding before feed into nn models
    Return:
        result: a dictionary of train and test scores

    '''

    seed_everything(_RANDOM_SEED, workers=True)
    print(f"Splitted Embedding Shapes of Train/Val/Test are {embed_data['train']['embed'].shape}, {embed_data['val']['embed'].shape}, {embed_data['test']['embed'].shape}")
    confs = list(ParameterGrid(_CLF_GRID[clf_name]))
    
    if embed_data['train']['label'].ndim == 1:
        # One hot encoding
        label_enc = OneHotEncoder(sparse=False)
        embed_data['train']['label'] = label_enc.fit_transform(embed_data['train']['label'].reshape((-1, 1)))
        embed_data['val']['label'] = label_enc.transform(embed_data['val']['label'].reshape((-1, 1)))
        embed_data['test']['label'] = label_enc.transform(embed_data['test']['label'].reshape((-1, 1)))
        print(f"Encoded Label Shapes of Train/Val/Test are {embed_data['train']['label'].shape}, {embed_data['val']['label'].shape}, {embed_data['test']['label'].shape}")
        print(f"Sample labels: {embed_data['train']['label'][:10]}")
        
    if norm_embed:
        train_mean = np.mean(embed_data['train']['embed'])
        train_std = np.std(embed_data['train']['embed'])
        embed_data['train']['embed'] = (embed_data['train']['embed'] - train_mean)/train_std
        embed_data['val']['embed'] = (embed_data['val']['embed'] - train_mean)/train_std
        embed_data['test']['embed'] = (embed_data['test']['embed'] - train_mean)/train_std
        model_save_name = 'normed_' + model_save_name

    # Load classifiers and Do predictions
    train_loader = embedding_loader(embed_data['train']['embed'], embed_data['train']['label'], n_wins=embed_data['train']['windows'], n_batch=cfg.train_bs, shuffle=True)
    val_loader = embedding_loader(embed_data['val']['embed'], embed_data['val']['label'], n_wins=embed_data['val']['windows'], n_batch=cfg.train_bs, shuffle=False)
    test_loader = embedding_loader(embed_data['test']['embed'], embed_data['test']['label'], n_wins=embed_data['test']['windows'], n_batch=cfg.train_bs, shuffle=False)
    best_result = hyparam_search(embed_data['train']['embed'].shape[-1], embed_data['train']['label'].shape[-1], confs, train_loader, val_loader, test_loader)
    print('BEST Model with score {} logged in path {}'.format(best_result['val_score'], best_result['model_path']))
    y_true, y_pred = test_pred(best_result, test_loader)
    bootstrap_stats = bootstrap(y_true, y_pred)
    test_accu= accuracy_score(y_true, y_pred)
    test_recall = recall_score(y_true, y_pred, average='macro')
    print(f'MODEL has test recall {test_recall} and test_accuracy {test_accu}')

    # Save Prediction and checkpoint
    inference_file = Path('clf_results/inferences').joinpath(model_save_name+ '.csv')
    inference_file.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({'True': y_true, 'Pred': y_pred}).to_csv(inference_file)
    model_save_path = Path('clf_results/nn_models').joinpath(model_save_name+ '.ckpt')
    model_save_path.parent.mkdir(exist_ok=True, parents=True)
    copyfile(best_result['model_path'], model_save_path) # copy saved checkpoint to target

    return {
        'train_score': round(best_result['val_score'] * 100, 2),
        'test_uar': round(test_recall * 100, 2),
        'test_uar_low': round(bootstrap_stats[1] * 100, 2),
        'test_uar_high': round(bootstrap_stats[2] * 100, 2)
    }


if __name__ == '__main__':
    # Load config
    args = parse_args()
    seed_everything(_RANDOM_SEED)
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_yaml_config(args.config_path)
    print(cfg)
    to_melspec = MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )
    results = {}
    embed_name = get_embed_name(args, cfg)

    #Get the datasets directories
    data_tuple = get_data_path(args.dataset_name)
    
    #Split data to train and test (speaker independent)
    if args.clf == 'svc':
        assert args.dataset_name != 'all_split', 'all_split dataset is not for SVC'
        dataset, splits = split_data(data_tuple, ratio=0.75)
    elif args.clf in ['mlp', 'transf']:
        assert args.dataset_name != 'all', 'all dataset is not used for NN'
        dataset, splits = split_data(data_tuple, ratio=0.6, val_ratio=0.15)
    else:
        raise ValueError(f'Invalid Classifier Name {args.clf}')

    #Shuffle training set with the same seed
    random.Random(_RANDOM_SEED).shuffle(dataset['train']['audio'])
    random.Random(_RANDOM_SEED).shuffle(dataset['train']['gender'])
    random.Random(_RANDOM_SEED).shuffle(dataset['train']['id'])
    random.Random(_RANDOM_SEED).shuffle(dataset['train']['label'])
    # encode labels
    for split in splits:
        dataset[split]['label'] = np.array([1 if l == 'with' else 0 for l in dataset[split]['label']])

    # Load data statistics
    try:
        stats = _CLF_STATS_DICT[args.dataset_name]
        print(stats)
    except KeyError:
        print(f'Did not find mean/std stats for {args.dataset_name}.')
        stats = compute_norm_stats(dataset['train']['audio'], to_melspec)

        _CLF_STATS_DICT[args.dataset_name] = stats

        print(_CLF_STATS_DICT)
    normalizer = PrecomputedNorm(stats)

    for model_name in tqdm(_MODELS_DICT):
        # Generate byols cvt embeddings
        embed_data = {}
        
        for split in splits:
            folder_name = embed_name + '_' + split
            embed_data[split] = generate_embeddings(model_name, args.speaker_embed, dataset[split], cfg.clip_audio, cfg.clip_speaker, to_melspec, normalizer, device, folder_name, cfg.speaker_only)

        if args.clf == 'svc':
            assert splits == ['train', 'test'], 'we use train/test split for SVC'
            model_save_path = Path('clf_results/sklearn_models').joinpath(args.clf + '_' + embed_name + '_' + model_name + '.joblib')
            result = svc_model(embed_data, model_save_path)
        elif args.clf in ['mlp', 'transf']:
            assert splits == ['train', 'val', 'test'], 'we use train/val/test split for NN'
            model_save_name = args.clf + '_' + embed_name + '_' + model_name 
            result = nn_model(embed_data, args.clf, model_save_name, cfg.norm_embed)
        else:
            raise ValueError(f'Invalid Classifier Name {args.clf}')

        results[model_name] = result
    
    # Save results
    results_df = pd.DataFrame(results)

    print(results_df)

    results_folder = 'clf_results/summary/'
    summary_name = args.clf + '_' + embed_name + '.csv'
    if cfg.norm_embed:
        summary_name = 'normed_' + summary_name
    save_results(summary_name, results_df, results_folder)
