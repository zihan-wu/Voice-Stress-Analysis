import os
import pathlib
import pickle
import random
import collections
from glob import glob
import soundfile as sf
import numpy as np
from pathlib import Path
from resemblyzer import preprocess_wav
from tqdm import tqdm
from sklearn.metrics import recall_score, accuracy_score
import torch
import torchaudio
import librosa
from torch.utils.data import DataLoader, Dataset
from settings import _REQUIRED_SAMPLE_RATE, _VSA_PATH, _COG1_LOAD_PATH, _COG2_LOAD_PATH, _COG3_LOAD_PATH, _COG4_LOAD_PATH, _PHY_LOAD_PATH
from settings import _WEIGHT_DICT, _MODELS_DICT, _SPEAKER_EMBED, MAX_WINS, CLIP_LEN, _EMBED_PATH
from settings import NUM_WORKERS

def get_data_path(data_name):
    '''This function returns the selected datasets directory'''
    if data_name.startswith('cog1'):
        data_dir = [_COG1_LOAD_PATH+'/results/*/*']
        data_name = ['cog1']
    elif data_name.startswith('cog2'):
        data_dir = [_COG2_LOAD_PATH+'/raw/*/*/*']
        data_name = ['cog2']
    elif data_name.startswith('cog3'):
        data_dir = [_COG3_LOAD_PATH+'/*/*']
        data_name = ['cog3']
    elif data_name.startswith('cog4'):
        data_dir = [_COG4_LOAD_PATH+'/*']
        data_name = ['cog4']
    elif data_name.startswith('phy'):
        data_dir = [_PHY_LOAD_PATH+'/*']
        data_name = ['phy']
    elif data_name == 'allcog':
        data_dir = [_COG1_LOAD_PATH+'/results/*/*', _COG2_LOAD_PATH+'/raw/*/*/*', _COG3_LOAD_PATH+'/*/*', _COG4_LOAD_PATH+'/*']
        data_name = ['cog1','cog2','cog3','cog4']
    else:
        data_dir = [_COG1_LOAD_PATH+'/results/*/*', _COG2_LOAD_PATH+'/raw/*/*/*', _COG3_LOAD_PATH+'/*/*', _COG4_LOAD_PATH+'/*', _PHY_LOAD_PATH+'/*']
        data_name = ['cog1','cog2','cog3','cog4','phy']
    return zip(data_name, data_dir)

def _parse_label_for_cog1(file):
    key = os.path.splitext(os.path.basename(file))[0].replace('_mono', '')
    metadata_file = glob(f'{os.path.dirname(file)}/metadata*.pkl')[0]
    
    metadata = pickle.load(open(metadata_file,'rb'))
    
    syls = metadata[key]['syls']
    
    if syls == 0:
        try:
            assert(metadata[key]['block'] == 0)  # Sanity check
        except KeyError:
            assert(metadata[key]['step'] == 0)  # Sanity check
        return 'without'
    else:
        try:
            assert(metadata[key]['block'] > 0)  # Sanity check
        except KeyError:
            assert(metadata[key]['step'] > 0)  # Sanity check
        return 'with'

def _parse_label_for_cog2(file):
    if 'without' in file:
        return 'without'
    elif 'with' in file and not('without' in file):
        return 'with'
    else:
        raise ValueError

def _parse_label_for_phy(file):
    if 'after' in file:
        return 'with'
    elif 'before' in file:
        return 'without'
    else:
        raise ValueError

def _parse_label_for_bic(file):
    if 'cog_load' in file:
        return 'with'
    elif 'baseline' in file:
        return 'without'
    else:
        raise ValueError

def _parse_gender(file):
    speaker_id = pathlib.PurePath(file).parent.name
    if speaker_id[0] == 'M':
        return 'M'
    elif speaker_id[0] == 'F':
        return 'F'
    else:
        raise ValueError('Gender [M/F] not found.')

def _parse_gender_for_phy(file):
    if 'sf' in file:
        return 'F'
    elif 'sm' in file:
        return 'M'
    else:
        raise ValueError

def _parse_gender_for_bic(file):
    speaker_id = pathlib.PurePath(file).parent.name
    if '_Female' in speaker_id:
        return 'F'
    elif '_Male' in speaker_id:
        return 'M'
    else:
        raise ValueError

def _compute_split_boundaries(split_probs, n_items):
    """Computes boundary indices for each of the splits in split_probs.
    Args:
      split_probs: List of (split_name, prob), e.g. [('train', 0.6), ('dev', 0.2),
        ('test', 0.2)]
      n_items: Number of items we want to split.
    Returns:
      The item indices of boundaries between different splits. For the above
      example and n_items=100, these will be
      [('train', 0, 60), ('dev', 60, 80), ('test', 80, 100)].
    """
    if len(split_probs) > n_items:
        raise ValueError('Not enough items for the splits. There are {splits} '
                         'splits while there are only {items} items'.format(splits=len(split_probs), items=n_items))
    total_probs = sum(p for name, p in split_probs)
    if abs(1 - total_probs) > 1E-8:
        raise ValueError('Probs should sum up to 1. probs={}'.format(split_probs))
    split_boundaries = []
    sum_p = 0.0
    for name, p in split_probs:
        prev = sum_p
        sum_p += p
        split_boundaries.append((name, int(prev * n_items), int(sum_p * n_items)))

    # Guard against rounding errors.
    split_boundaries[-1] = (split_boundaries[-1][0], split_boundaries[-1][1],
                            n_items)
    return split_boundaries

def items_to_split_datasets(items_and_groups, split_probs, split_to_ids):
    groups = sorted(set(gender_id+'_'+group_id for _, _, group_id, gender_id in items_and_groups))
    print(groups)
    gender_id = {'F': list(filter(lambda k: 'F_' in k, groups)),
                'M': list(filter(lambda k: 'M_' in k, groups))}
    group_id_to_split = {}
    for gender in ['M', 'F']:
        split_boundaries = _compute_split_boundaries(split_probs, len(gender_id[gender]))
        for split_name, i_start, i_end in split_boundaries:
            for i in range(i_start, i_end):
                i = i - len(gender_id[gender])
                group_id_to_split[gender_id[gender][i]] = split_name
    print(group_id_to_split)
    for item, label, group_id, gender_id in items_and_groups:
        split = group_id_to_split[gender_id+'_'+group_id]
        split_to_ids[split]['audio'].append(item)
        split_to_ids[split]['label'].append(label)
        split_to_ids[split]['id'].append(group_id)
        split_to_ids[split]['gender'].append(gender_id)
    return split_to_ids

def split_data(data_tuple, ratio=0.75, val_ratio=None):
    if val_ratio is None:
        split_probs = [('train', ratio), ('test', 1-ratio)]
        splits = ['train', 'test']
    else:
        split_probs = [('train', ratio), ('val', val_ratio), ('test', 1-ratio-val_ratio)]
        splits = ['train', 'val', 'test']
    split_to_ids = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for key, data_dir in data_tuple:
        data = sorted(glob(f'{data_dir}/*_mono.wav'))
        if key == 'cog1':
            names = list(map(os.path.basename, sorted(glob(data_dir))))
            speaker_ids = list(map(lambda x: pathlib.PurePath(x).parent.name, data))
            labels = list(map(_parse_label_for_cog1, data))
            genders = list(map(_parse_gender, data))
        elif key == 'cog2' or key == 'cog3':
            names = list(map(os.path.basename, sorted(glob(data_dir))))
            speaker_ids = list(map(lambda x: pathlib.PurePath(x).parent.name, data))
            labels = list(map(_parse_label_for_cog2, data))
            genders = ['F'] * 400 + ['M'] * 400
        elif key == 'cog4':
            speaker_ids = list(map(lambda x: pathlib.PurePath(x).parent.name.split('_')[0], data))
            names = list(map(lambda x: os.path.basename(x), sorted(glob(data_dir))))
            labels = list(map(_parse_label_for_bic, data))
            genders = list(map(_parse_gender_for_bic, data))
        else:
            data = sorted(glob(f'{data_dir}/*.wav'))
            names = list(map(os.path.basename, sorted(glob(data_dir))))
            speaker_ids = list(map(lambda x: pathlib.PurePath(x).parent.name, data))
            labels = list(map(_parse_label_for_phy, data))
            genders = list(map(_parse_gender_for_phy, data))
        items_and_groups = list(zip(data, labels, speaker_ids, genders))
        split_to_ids = items_to_split_datasets(items_and_groups, split_probs, split_to_ids)
        print(f'Data {key}: {len(data)} samples, {len(names)} subjects.')
    return split_to_ids, splits

def process_audio(audio, orig_sr):
    # Convert to float if necessary
    if np.issubdtype(audio.dtype, np.integer):
        float_audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    elif audio.dtype == np.float64:
        float_audio = np.float32(audio)
    else:
        float_audio = audio

    # Resample if needed
    if orig_sr != _REQUIRED_SAMPLE_RATE:
        float_audio = librosa.core.resample(
            float_audio,
            orig_sr=orig_sr,
            target_sr=_REQUIRED_SAMPLE_RATE,
            res_type='kaiser_best'
        )
    return float_audio

def compute_norm_stats(files, to_melspec):
    mean = 0.
    std = 0.

    for file in tqdm(files, desc=f'Computing stats...', total=len(files)):
        audio, orig_sr = sf.read(file) 
        float_audio = process_audio(audio, orig_sr)

        # Convert to tensor
        float_audio = torch.tensor(float_audio).unsqueeze(0)

        # Compute log-mel-spectrogram
        lms = (to_melspec(float_audio) + torch.finfo(torch.float).eps).log()

        # Compute mean, std
        mean += lms.mean()
        std += lms.std()

    mean /= len(files)
    std /= len(files)

    stats = [mean.item(), std.item()]

    print(f'Finished')
    return stats

def get_byolas_weights(model_name, device):
    # Load pretrained weights.
    model = _MODELS_DICT[model_name]
    weight_file = _WEIGHT_DICT[model_name]
    print('Loaded Weights: {}'.format(weight_file))
    state_dict = torch.load(weight_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # Disable parameter tuning
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_speaker_embedding(encoder, encoder_name, file_path, use_mean = False):
    # speaker embedding of each file needs to be generated first before using the mean
    rel_path = os.path.relpath(file_path, _VSA_PATH)
    assert rel_path[-4:] == '.wav', 'Audio File must be in wav format'
    save_path = Path(_EMBED_PATH).joinpath(encoder_name).joinpath(rel_path[:-4] +'.npy')
    mean_path = save_path.parent.joinpath('mean_embed.npy')
    if save_path.exists():
        embed = np.load(save_path)
        if use_mean:
            if mean_path.exists():
                embed = np.load(mean_path)
            else:
                embeds = 0
                n = 0
                for sub_embed in tqdm(os.listdir(mean_path.parent), desc='Computing Average Speaker Embed'):
                    embeds += np.load(mean_path.parent.joinpath(sub_embed))
                    n += 1
                embed = embeds/n
                np.save(mean_path, embed)
    else:
        if encoder_name == 'resemblyzer':
            fpath = Path(file_path)
            wav = preprocess_wav(fpath)
            embed = encoder.embed_utterance(wav)
        elif encoder_name == 'ecapa':
            wav, fs = torchaudio.load(file_path)
            embedding = encoder.encode_batch(wav)
            embed = embedding.squeeze().detach().cpu().numpy()
        else:
            raise ValueError('speaker encoder name not available')
        save_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(save_path, embed)
    return embed

def get_clip_speaker(encoder, encoder_name, wav, k, rel_path):
    save_path = Path(_EMBED_PATH).joinpath(encoder_name + f'_clip{CLIP_LEN}').joinpath(rel_path[:-4] + f'-{k}.npy')
    if save_path.exists():
        embed = np.load(save_path)
    else:
        if encoder_name == 'resemblyzer':
            wav = preprocess_wav(wav.detach().cpu().numpy())
            embed = encoder.embed_utterance(wav)
        elif encoder_name == 'ecapa':
            embedding = encoder.encode_batch(wav.unsqueeze(0))
            embed = embedding.squeeze().detach().cpu().numpy()
        else:
            raise ValueError('speaker encoder name not available')
        save_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(save_path, embed)
    return embed


def concat_embedding(model_embedding, speaker_embedding):
    # concat model embeddings and speaker embeddings
    if model_embedding.ndim == 3: # for timestamp embedding
        padded_embedding = np.zeros((MAX_WINS, model_embedding.shape[-1]))
        padded_embedding[:model_embedding.shape[1], :] = model_embedding[0]
        n_win = model_embedding.shape[1]
        if speaker_embedding is not None:
            tiled_speaker = np.tile(speaker_embedding, (padded_embedding.shape[0], 1))
            embedding = np.append(padded_embedding, tiled_speaker, axis=1)
        else:
            embedding = padded_embedding
    elif model_embedding.ndim == 2:
        n_win = None
        if speaker_embedding is not None:
            embedding = np.append(np.squeeze(model_embedding), speaker_embedding)
        else:
            embedding = np.squeeze(model_embedding)
    else:
        raise ValueError('Incorrect embedding dimension, must be 2 for scene or 3 for timestamp')
    return embedding, n_win

def split_large_audio(audio, clip_len, min_tail_len):
    # split the long audios to smaller segments
    count = 1
    audios = [audio[:clip_len]]
    start = clip_len
    while start + clip_len < len(audio):
        audios.append(audio[start : start + clip_len])
        start = start + clip_len
        count += 1
    if len(audio) - start > min_tail_len:
        audios.append(audio[start:])
        count += 1

    return audios

def generate_embeddings(
    model_name,
    speaker_embed, 
    data,
    clip_audio,
    clip_speaker,
    to_melspec,
    normalizer,
    device,
    folder_name,
    speaker_only
):
    """
    Generate audio embeddings from a pretrained feature extractor.

    Converts audios to float, resamples them to the desired learning_rate, (clip into chunks)
    and produces the embeddings from a pre-trained model.

    It should adjust the shape based on whether receiving a timestamp or scene embedding

    Parameters
    ----------
    model_name : str
        Name of the model
    speaker_embed: str
        Name of speaker embedding
    data : Dict
        Dictionary containing 'audio', 'label'
    clip_audio: boolean
        Whether to chunk data
    clip_audio: boolean
        Whether to compute speaker embedding on each separate chunk
    to_melspec : torchaudio.transforms.MelSpectrogram object
        Mel-spectrogram transform to create a spectrogram from an audio signal
    normalizer : nn.Module
        Pre-normalization transform
    device : torch.device object
        Used device (CPU or GPU)
    folder_name: str
        name of the folder
    speaker_only: bool
        True if use only speaker embedding

    Returns
    ----------
    embeddings: numpy.ndarray
        2D or 3D Array of embeddings for each audio of size (N, M) ro (N, T, M). N = number of samples, M = embedding dimension, T = number of temporal dimension
    """
    n_wins = None
    if speaker_embed in _SPEAKER_EMBED:
        speaker_encoder = _SPEAKER_EMBED[speaker_embed]
        name_prefix = speaker_embed + '_'
    else:
        speaker_encoder = None
        print(f'No Speaker Embedding used for {folder_name}')
        name_prefix = ''
    to_file = Path(_EMBED_PATH)/(f'Dataset_{folder_name}') / (name_prefix + model_name + '.npy')
    to_labelfile = Path(_EMBED_PATH)/(f'Dataset_{folder_name}') / (name_prefix + model_name + '_label.npy')
    to_winsfile = Path(_EMBED_PATH)/(f'NWins_{folder_name}') / (name_prefix + model_name + '.npy')# The wins files stores temporal length of timestamp embeddings before padding

    if to_file.exists() and to_labelfile.exists(): # if embeddings and labels file exists, load existing files
        embeddings = np.load(to_file)
        labels = np.load(to_labelfile)
        print(f'Loaded embeddings from existing {to_file} and labels from {to_labelfile}')
        if to_winsfile.exists(): # if wins info file exists, load existing files
            n_wins = np.load(to_winsfile)
            print(f'Loaded windows info from existing {to_winsfile}')
    else: # if embeddings file not exist, generate new embedding and wins file
        print(f'{to_file} or {to_labelfile} is not found, need to generate embedding')
        embeddings = []
        files = data['audio']
        old_labels = data['label']
        new_labels = []
        clip_len = int(_REQUIRED_SAMPLE_RATE * CLIP_LEN)
        min_tail_len = int(_REQUIRED_SAMPLE_RATE * 2) # remove utterance less than 2s
        n_wins = []
        print(f'Loading Model {model_name}')
        model = get_byolas_weights(model_name, device)

        for i, file in tqdm(enumerate(files), desc=f'Generating embeddings..', total=len(files), ascii=True):
            # load audio and extract embedding
            audio, orig_sr = sf.read(file) 
            float_audio = process_audio(audio, orig_sr)
            label =  old_labels[i]
            if speaker_encoder and not clip_speaker:
                speaker_embedding = get_speaker_embedding(speaker_encoder, speaker_embed, file)
            else: 
                speaker_embedding = None
            if len(float_audio) > clip_len and clip_audio:
                audios = split_large_audio(torch.tensor(float_audio), clip_len, min_tail_len) # chunk audio
            else:
                audios = torch.tensor(float_audio).unsqueeze(0)
            for k, x in enumerate(audios):

                # get speaker embedding clipped
                if speaker_encoder and clip_speaker:
                    rel_path = os.path.relpath(file, _VSA_PATH)
                    assert rel_path[-4:] == '.wav', 'Audio File must be in wav format'
                    speaker_embedding = get_clip_speaker(speaker_encoder, speaker_embed, x, k, rel_path)

                if speaker_only:
                    assert speaker_encoder is not None, 'You must specify a type of speaker embedding if you only use speaker_embed'
                    embeddings.append(speaker_embedding)
                else:
                    lms = normalizer((to_melspec(x.unsqueeze(0)) + torch.finfo(torch.float).eps).log()).unsqueeze(1)
                    model_embedding = model(lms.to(device)).cpu().detach().numpy()
                    embedding, n_win = concat_embedding(model_embedding, speaker_embedding)
                    if n_win is not None:
                        n_wins.append(n_win) 
                    embeddings.append(embedding)

                new_labels.append(label)


        embeddings = np.array(embeddings, dtype=np.float32)
        labels = np.array(new_labels, dtype=np.float32)
        assert len(embeddings) == len(labels), f'number of embeddings {len(embeddings)} should be same as labels {len(labels)}'
        print(f'{model_name} Finished, with embedding shape {embeddings.shape} and label shape {labels.shape}' )
        # Saving embeddings in case you need to reuse them without generating the embeddings again.
        to_file.parent.mkdir(exist_ok=True, parents=True)
        np.save(to_file, embeddings)
        to_labelfile.parent.mkdir(exist_ok=True, parents=True)
        np.save(to_labelfile, labels)
        print(f'Saved embeddings as {to_file} and labels as {to_labelfile}')
        if len(n_wins) > 0:
            assert len(n_wins) == len(labels), f'number of windows {len(n_wins)} should be same as labels {len(labels)}'
            to_winsfile.parent.mkdir(exist_ok=True, parents=True)
            n_wins = np.array(n_wins, dtype=np.float32)
            np.save(to_winsfile, n_wins)
            print(f'Saved windows info as {to_winsfile}')
        else:
            n_wins = None

    return {'embed': embeddings, 'label': labels, 'windows': n_wins}

def generate_byols_embeddings(
    model_name,
    speaker_embed, 
    data,
    to_melspec,
    normalizer,
    device,
    split,
    dataset_name,
):
    """
    Generate audio embeddings from a pretrained feature extractor.

    Converts audios to float, resamples them to the desired learning_rate,
    and produces the embeddings from a pre-trained model.

    Adapted from https://github.com/google-research/google-research/tree/master/non_semantic_speech_benchmark

    Parameters
    ----------
    model_name : str
        Name of the model
    data : Dict
        Dictionary containing 'audio', 'label'
    split : str
        Dataset split, can be 'train', 'validation' or 'test'.
    orig_sr : int
        Original sample rate in the dataset.
    to_melspec : torchaudio.transforms.MelSpectrogram object
        Mel-spectrogram transform to create a spectrogram from an audio signal
    normalizer : nn.Module
        Pre-normalization transform
    device : torch.device object
        Used device (CPU or GPU)

    Returns
    ----------
    embeddings: numpy.ndarray
        2D Array of embeddings for each audio of size (N, M). N = number of samples, M = embedding dimension
    """
    # Get speaker embedding encoder and file names of the embeddings and windows info
    files = data['audio']
    use_wins = False # indicate whether to return windows info
    if speaker_embed in _SPEAKER_EMBED:
        speaker_encoder = _SPEAKER_EMBED[speaker_embed]
        to_file = Path(_EMBED_PATH)/(f'Dataset_{dataset_name}_{len(files)}_{split}') / (speaker_embed + '_' + model_name + '.npy')
        to_winsfile = Path(_EMBED_PATH)/(f'NWins_{dataset_name}_{len(files)}_{split}') / (speaker_embed + '_' + model_name + '.npy')
    else:
        speaker_encoder = None
        to_file = Path(_EMBED_PATH)/(f'Dataset_{dataset_name}_{len(files)}_{split}') / (model_name + '.npy')
        to_winsfile = Path(_EMBED_PATH)/(f'NWins_{dataset_name}_{len(files)}_{split}') / (model_name + '.npy')
    

    if to_winsfile.exists():
        n_wins = np.load(to_winsfile)
        use_wins = True
        print(f'Loaded windows info from existing {to_winsfile}')

    if to_file.exists():
        embeddings = np.load(to_file)
        print(f'Loaded embeddings from existing {to_file}')
    else: # if embeddings file not exist, generate new embedding and wins file
        embeddings = []
        n_wins = []
        print(f'Loading Model {model_name}')
        model, _ = get_byolas_weights(model_name, device)

        for file in tqdm(files, desc=f'Generating embeddings..', total=len(files), ascii=True):
            # load audio and extract embedding
            audio, orig_sr = sf.read(file) 
            float_audio = process_audio(audio, orig_sr)
            float_audio = torch.tensor(float_audio).unsqueeze(0)
            lms = normalizer((to_melspec(float_audio) + torch.finfo(torch.float).eps).log()).unsqueeze(0)
            embedding = model(lms.to(device)).cpu().detach().numpy()

            if embedding.ndim == 3:
                padded_embedding = np.zeros((MAX_WINS, embedding.shape[-1]))
                padded_embedding[:embedding.shape[1], :] = embedding[0]
                n_wins.append(embedding.shape[1])
                if speaker_encoder:
                    speaker_embedding = get_speaker_embedding(speaker_encoder, speaker_embed, file)
                    speaker_embedding = np.tile(speaker_embedding, (padded_embedding.shape[0], 1))
                    embeddings.append(np.append(padded_embedding, speaker_embedding, axis=1))
                    # print(embedding.shape, padded_embedding.shape)
                else:
                    embeddings.append(padded_embedding)
            elif embedding.ndim == 2:
                if speaker_encoder:
                    speaker_embedding = get_speaker_embedding(speaker_encoder, speaker_embed, file)
                    embeddings.append(np.append(np.squeeze(embedding), speaker_embedding))
                else:
                    embeddings.append(np.squeeze(embedding))
            else:
                raise ValueError('Incorrect embedding dimension, must be 2 for scene or 3 for timestamp')
            

        embeddings = np.array(embeddings, dtype=np.float32)
        print(f'{model_name} Finished', embeddings.shape)
        # Saving embeddings in case you need to reuse them without generating the embeddings again.
        to_file.parent.mkdir(exist_ok=True, parents=True)
        np.save(to_file, embeddings)
        print(f'Saved embeddings as {to_file}')
        if embeddings.ndim == 3:
            use_wins = True
            to_winsfile.parent.mkdir(exist_ok=True, parents=True)
            np.save(to_winsfile, np.array(n_wins, dtype=np.float32))
            print(f'Saved windows info as {to_winsfile}')
    
    if use_wins:
        return embeddings, n_wins
    else:
        return embeddings

def generate_byols_clip_embeddings(
    model_name,
    model_weight,
    speaker_embed, 
    data,
    to_melspec,
    normalizer,
    device,
    split,
    dataset_name,
):
    """
    Generate audio embeddings from a pretrained feature extractor.

    Converts audios to float, resamples them to the desired learning_rate,
    and produces the embeddings from a pre-trained model.

    Adapted from https://github.com/google-research/google-research/tree/master/non_semantic_speech_benchmark

    Parameters
    ----------
    model_name : str
        Name of the model
    model_weight : str
        Weight of the model
    data : Dict
        Dictionary containing 'audio', 'label'
    split : str
        Dataset split, can be 'train', 'validation' or 'test'.
    orig_sr : int
        Original sample rate in the dataset.
    to_melspec : torchaudio.transforms.MelSpectrogram object
        Mel-spectrogram transform to create a spectrogram from an audio signal
    normalizer : nn.Module
        Pre-normalization transform
    device : torch.device object
        Used device (CPU or GPU)

    Returns
    ----------
    embeddings: numpy.ndarray
        2D Array of embeddings for each audio of size (N, M). N = number of samples, M = embedding dimension
    """
    # Get speaker embedding encoder and file names of the embeddings and windows info
    use_wins = False # indicate whether to return windows info
    if speaker_embed in _SPEAKER_EMBED:
        speaker_encoder = _SPEAKER_EMBED[speaker_embed]
        to_file = Path(_EMBED_PATH)/(f'Dataset_{dataset_name}_clip{CLIP_LEN}_{split}') / (speaker_embed + '_' + model_name + '.npy')
        to_labelfile = Path(_EMBED_PATH)/(f'Dataset_{dataset_name}_clip{CLIP_LEN}_{split}') / (speaker_embed + '_' + model_name + '_label.npy')
        to_winsfile = Path(_EMBED_PATH)/(f'NWins_{dataset_name}_clip{CLIP_LEN}_{split}') / (speaker_embed + '_' + model_name + '.npy')
    else:
        speaker_encoder = None
        to_file = Path(_EMBED_PATH)/(f'Dataset_{dataset_name}_clip{CLIP_LEN}_{split}') / (model_name + '.npy')
        to_labelfile = Path(_EMBED_PATH)/(f'Dataset_{dataset_name}_clip{CLIP_LEN}_{split}') / (model_name + '_label.npy')
        to_winsfile = Path(_EMBED_PATH)/(f'NWins_{dataset_name}_clip{CLIP_LEN}_{split}') / (model_name + '.npy')
    

    if to_file.exists() and to_labelfile.exists():
        embeddings = np.load(to_file)
        labels = np.load(to_labelfile)
        print(f'Loaded embeddings from existing {to_file} and labels from {to_labelfile}')
        if to_winsfile.exists():
            n_wins = np.load(to_winsfile)
            use_wins = True
            print(f'Loaded windows info from existing {to_winsfile}')
    else: # if embeddings file not exist, generate new embedding and wins file
        embeddings = []
        files = data['audio']
        old_labels = data['label']
        new_labels = []
        clip_len = int(_REQUIRED_SAMPLE_RATE * CLIP_LEN)
        min_tail_len = int(_REQUIRED_SAMPLE_RATE * 2) # remove utterance less than 2s
        n_wins = []
        print(f'Loading Model {model_name}')
        model, _ = get_byolas_weights(model_name, device)

        for i, file in tqdm(enumerate(files), desc=f'Generating embeddings..', total=len(files), ascii=True):
            # load audio and extract embedding
            audio, orig_sr = sf.read(file) 
            float_audio = process_audio(audio, orig_sr)
            label =  old_labels[i]
            if speaker_encoder:
                speaker_embedding = get_speaker_embedding(speaker_encoder, speaker_embed, file)
            if len(float_audio) > clip_len:
                audios = split_large_audio(torch.tensor(float_audio), clip_len, min_tail_len)
            else:
                audios = torch.tensor(float_audio).unsqueeze(0)
            for x in audios:
                lms = normalizer((to_melspec(x.unsqueeze(0)) + torch.finfo(torch.float).eps).log()).unsqueeze(1)
                embedding = model(lms.to(device)).cpu().detach().numpy()
                new_labels.append(label)
                if embedding.ndim == 3:
                    padded_embedding = np.zeros((MAX_WINS, embedding.shape[-1]))
                    padded_embedding[:embedding.shape[1], :] = embedding[0]
                    n_wins.append(embedding.shape[1])
                    if speaker_encoder:
                        tiled_speaker = np.tile(speaker_embedding, (padded_embedding.shape[0], 1))
                        embeddings.append(np.append(padded_embedding, tiled_speaker, axis=1))
                        # print(embedding.shape, padded_embedding.shape)
                    else:
                        embeddings.append(padded_embedding)
                elif embedding.ndim == 2:
                    if speaker_encoder:
                        embeddings.append(np.append(np.squeeze(embedding), speaker_embedding))
                    else:
                        embeddings.append(np.squeeze(embedding))
                else:
                    raise ValueError('Incorrect embedding dimension, must be 2 for scene or 3 for timestamp')
            

        embeddings = np.array(embeddings, dtype=np.float32)
        labels = np.array(new_labels, dtype=np.float32)
        print(f'{model_name} Finished, with embedding shape {embeddings.shape} and label shape {labels.shape}' )
        # Saving embeddings in case you need to reuse them without generating the embeddings again.
        to_file.parent.mkdir(exist_ok=True, parents=True)
        np.save(to_file, embeddings)
        to_labelfile.parent.mkdir(exist_ok=True, parents=True)
        np.save(to_labelfile, labels)
        print(f'Saved embeddings as {to_file} and labels as {to_labelfile}')
        if embeddings.ndim == 3:
            use_wins = True
            to_winsfile.parent.mkdir(exist_ok=True, parents=True)
            np.save(to_winsfile, np.array(n_wins, dtype=np.float32))
            print(f'Saved windows info as {to_winsfile}')
    
    if use_wins:
        return embeddings, labels, n_wins
    else:
        return embeddings, labels


class EmbedDataset(Dataset):
    """
    Create Embed dataset for training
    """

    def __init__(self, embeds, labels, n_wins):

        self.embeds = embeds
        self.labels = labels
        self.n_wins = n_wins if n_wins is not None else np.zeros(labels.shape[0])
        print(f'DATASET INFO: Shapes of Embed {self.embeds.shape}, labels {self.labels.shape}, windows {self.n_wins.shape}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeds[idx], self.labels[idx], self.n_wins[idx]


def embedding_loader(embeds, labels, n_wins, n_batch, shuffle=True):
    return DataLoader(
            EmbedDataset(embeds, labels, n_wins),
            batch_size=n_batch,
            shuffle=shuffle,
            pin_memory=False,
            num_workers=NUM_WORKERS
        )


def bootstrap(y_true, y_pred, seed=42, n_rep = 1000, alpha=95):
    # score func a score function taking (y_true, y_pred)
    low_cut = (100 - alpha)/2
    high_cut = (100 + alpha)/2
    np.random.seed(seed)
    n_sample = len(y_true)
    scores = np.zeros(n_rep)
    for i in tqdm(range(n_rep), desc='Compute Bootstrap Stats ...'):
        sample_ind = np.random.choice(range(n_sample), size=n_sample, replace=True)
        scores[i] = recall_score(y_true=y_true[sample_ind], y_pred=y_pred[sample_ind], average='macro')

    return (np.mean(scores), np.percentile(scores, low_cut), np.percentile(scores, high_cut))