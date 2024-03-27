"""
Predict
"""
from modules.utils import load_yaml
from modules.datasets import CustomDataset, csv_preprocessing
from models.utils import get_model

from torch.utils.data import DataLoader
import albumentations as A

from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import numpy as np
import random
import os
import torch
import pandas as pd
import yaml

# Config
PROJECT_DIR = os.path.dirname(__file__)
predict_config = load_yaml(os.path.join(PROJECT_DIR, 'config', 'predict_config.yaml'))


# Serial
train_serial = predict_config['TRAIN']['train_serial']
kst = timezone(timedelta(hours=9))
predict_timestamp = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
predict_serial = train_serial + '_' + predict_timestamp

# Predict directory
PREDICT_DIR = os.path.join(PROJECT_DIR, 'results', 'predict', predict_serial)
os.makedirs(PREDICT_DIR, exist_ok=True)

# Data Directory
IMAGE_DIR = predict_config['DIRECTORY']['image_dir']
META_DIR = predict_config['DIRECTORY']['meta_dir']

########################################################################################
# Recorder Directory
RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)

# Train config
train_config = load_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'))
########################################################################################
# SEED
torch.manual_seed(predict_config['PREDICT']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(predict_config['PREDICT']['seed'])
random.seed(predict_config['PREDICT']['seed'])

# Gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(predict_config['PREDICT']['gpu'])

if __name__ == '__main__':

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    # Load dataframe & preprocessing
    df = pd.read_csv(META_DIR, encoding="utf-8", dtype={"no": "str"})
    df = csv_preprocessing(df, IMAGE_DIR)

    # Dataset
    test_transform = A.Compose([A.Resize(224, 224), A.Normalize()])

    test_dataset = CustomDataset(df, transform=test_transform, mode='test')

    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=train_config['DATALOADER']['batch_size'],
                                num_workers=train_config['DATALOADER']['num_workers'], 
                                shuffle=False,
                                pin_memory=train_config['DATALOADER']['pin_memory'],
                                drop_last=train_config['DATALOADER']['drop_last'])

    # Load model
    model_name = train_config['TRAINER']['model']
    model_args = train_config['MODEL'][model_name]
    model = get_model(model_name=model_name, model_args=model_args).to(device)

    checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'))
    model.load_state_dict(checkpoint['model'])

    model.eval()
    
    # Make predictions
    y_preds = []
    filenames = []
    bar = tqdm(test_dataloader)
    for batch_index, (x, meta_info, filename) in enumerate(bar):
        x = x.to(device, dtype=torch.float)
        meta_info = meta_info.to(device, dtype=torch.float)
        y_pred = model(x, meta_info).squeeze(dim=-1)

        filenames.extend(filename)
        y_preds.append(y_pred.detach().cpu())
    
    y_preds = torch.cat(y_preds, dim=0).tolist()

    # Save Submission File
    sub_df = pd.DataFrame({'filename':filenames, 'time_min':y_preds})
    sub_df = sub_df.sort_values(by=['filename']).reset_index(drop=True)
    sub_df.to_csv(os.path.join(PREDICT_DIR, 'submission.csv'), index=False)
    
    ### ------------------------------------------------------------------- ###
    output_yaml = {'DIR' : PREDICT_DIR}
    
    with open('/USER/baseline/config/output_config.yaml', 'w') as file:
        yaml.dump(output_yaml, file, default_flow_style=False)