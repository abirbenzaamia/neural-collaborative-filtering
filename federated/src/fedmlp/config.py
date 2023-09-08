import torch
#DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PARAMETERS = {
    'FedMLP': {
        'mf_dim': 16,
        'layers': [48, 24, 12, 6],
        'reg_layers': [0, 0, 0, 0],
        'reg_mf': 0
    },
}

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_NEGATIVES = 4
TOPK = 20
VALIDATION_STEPS = 10
NEG_DATA = 99