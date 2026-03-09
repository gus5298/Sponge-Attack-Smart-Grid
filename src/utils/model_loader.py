import torch
import numpy as np
import pandas as pd

from config import (NUM_FEATURES, PREDICTION_LEN, HIDDEN_SIZE, RNN_LAYERS,
                    ACT_HIDDEN_SIZE, DEEPAR_MODEL_PATH, ACT_MODEL_PATH,
                    CONTEXT_LEN, DATA_PATH, ALL_FEATURES)
from models.deepar import DeepARLSTM
from models.act import ACTModel
from utils.data_loader import load_seed_data, get_normalization_params


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_deepar(checkpoint_path=None, device=None):
    device = device or get_device()
    path = checkpoint_path or DEEPAR_MODEL_PATH
    checkpoint = torch.load(path, map_location=device)
    model = DeepARLSTM(
        input_size=NUM_FEATURES,
        hidden_size=checkpoint.get('hidden_size', HIDDEN_SIZE),
        num_layers=checkpoint.get('rnn_layers', RNN_LAYERS),
        output_size=PREDICTION_LEN,
        dropout=0.1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def load_act(checkpoint_path=None, device=None):
    device = device or get_device()
    path = checkpoint_path or ACT_MODEL_PATH
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        hidden = state.get('hidden_size', ACT_HIDDEN_SIZE)
        model = ACTModel(NUM_FEATURES, hidden, PREDICTION_LEN).to(device)
        model.load_state_dict(state['model_state_dict'])
    else:
        model = ACTModel(NUM_FEATURES, ACT_HIDDEN_SIZE, PREDICTION_LEN).to(device)
        model.load_state_dict(state)
    model.eval()
    return model


def load_seed(data_path=None, context_len=None, checkpoint=None):
    data_path = data_path or DATA_PATH
    context_len = context_len or CONTEXT_LEN
    norm_params = checkpoint.get('norm_params', None) if checkpoint else None
    return load_seed_data(data_path, context_len, norm_params)


def load_seed_from_csv(data_path=None, context_len=None):
    data_path = data_path or DATA_PATH
    context_len = context_len or CONTEXT_LEN
    df = pd.read_csv(data_path).sort_values('Time').reset_index(drop=True)
    data = df[ALL_FEATURES].values.astype(np.float32)
    mean, std = get_normalization_params(data)
    seed_data = (data[:context_len] - mean) / std
    return seed_data, mean, std


def load_chronos(device=None):
    from chronos import ChronosPipeline
    device = device or get_device()
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map=device, torch_dtype=torch.float32
    )
    return pipeline


def make_predictor(model, device=None):
    device = device or get_device()

    def predict(input_array):
        x = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(x)

    return predict
