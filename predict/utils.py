import numpy as np 
import torch
import json 
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from model import Encoder, Decoder, Seq2Seq
MAX_TR , MIN_TR = 116, 0 
MEAN_CURRENT_LEVELS , STD_CURRENT_LEVELS = 488.3017127799736, 996.1384988591653
STATICOL="CURRENT_LEVEL"
LISTMONTHS = "listMonths"
INPUT_SEQUENCE="inputSeq"
OUTPUT_SEQUENCE="outputSeq"
NORM_STATICOL = f"normalized{STATICOL}"
IDCOL = 'SKU'
TARGET_WEEKS = "targetWeeks"
TARGET = 'QUANTITY_SOLD'
DATECOL = 'DATE'
WEEKOFYEAR = "weekOfYear"
YEARCOL = "year"
LISTWEEKSOFYEAR = f"list{WEEKOFYEAR}"
LISTTARGET=f'list_{TARGET}'
NORM_STATICOL = f"normalized{STATICOL}"


class InsufficientDataError(Exception):
    """Exception raised when the input data is insufficient."""
    def __init__(self, message="The input data is insufficient. The data length must be at least 4."):
        self.message = message
        super().__init__(self.message)

def encode_week(week):
    return np.cos(2 * np.pi * week / 52), np.sin(2 * np.pi * week / 52)

def min_max_normalize(data, min_val=MIN_TR, max_val=MAX_TR):
    data = torch.clamp(data, min_val, max_val)
    return (data - min_val) / (max_val - min_val)

def min_max_denormalize(data, min_val=MIN_TR, max_val=MAX_TR):
    return data * (max_val - min_val) + min_val

def normalize_current_levels(data, mean=MEAN_CURRENT_LEVELS , std=STD_CURRENT_LEVELS):
    return (data-mean)/std


def generate_pair_samples(weeks, data, lookback, horizon ):
    samples_input = []
    samples_output = []
    samples_target_weeks=[]
    for i in range(len(data) - lookback - horizon + 1):
        input_seq = []
        for j in range(lookback):
            week = weeks[i + j] % 52
            cos_week, sin_week = encode_week(week)
            input_seq.append([data[i + j], cos_week, sin_week])
        
        output_seq = data[i + lookback:i + lookback + horizon]
        target_weeks = [encode_week(wk) for wk in weeks[i + lookback:i + lookback + horizon]]
        samples_input.append(input_seq)
        samples_output.append(output_seq)
        samples_target_weeks.append(target_weeks)
    
    return [samples_input, samples_output,samples_target_weeks]

def generate_future_weeks(start_week, count):
    # Generate `count` future weeks starting from `start_week`
    return [(start_week + i) % 52 for i in range(count)]

def format_inference_sample(weeks,data,lookback=4):
    if len(data)<4:
        raise InsufficientDataError()
    input_seq = []
    for j in range(lookback):
        week = weeks[ j] % 52
        cos_week, sin_week = encode_week(week)
        input_seq.append([data[ j], cos_week, sin_week])
    
    output_seq = data[lookback:] if len(data)>4 else []

    target_weeks = [encode_week(wk) for wk in weeks[lookback:]]
    
    if len(target_weeks) < 48:
        last_week = weeks[-1]
        future_weeks_needed = 48 - len(target_weeks)
        future_weeks = generate_future_weeks(last_week, future_weeks_needed)
        future_encodings = [encode_week(wk) for wk in future_weeks]
        target_weeks.extend(future_encodings)
    
    target_weeks = [encode_week(wk) for wk in weeks[lookback:]]

    #normalise
    output_seq = torch.tensor(output_seq).reshape(1,-1).float()
    input_seq=torch.tensor(input_seq).reshape(1,4,3).float()
    target_weeks = torch.tensor(target_weeks).reshape(1,-1,2).float()

    output_seq = min_max_normalize(output_seq) if len(output_seq)>0 else None
    input_seq = min_max_normalize(input_seq)
    return input_seq , output_seq,target_weeks

def write_json_to_directory(json_data, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write JSON data to file
    with open(file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def load_model(path):
    device = torch.device("cpu")
    hidden_dim=128
    num_layers=2
    dropout=.5
    output_dim = 1
    teacher_forcing_ratio=0
    attention_enabled=True
    scalar_input_dim = 1
    seq_input_dim = 3 
    

    # Recreate the model
    encoder = Encoder(seq_input_dim, scalar_input_dim, hidden_dim, num_layers, dropout)
    decoder = Decoder(output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, attention_enabled=True)
    model = Seq2Seq(encoder, decoder, device, teacher_forcing_ratio).to(device)

    # Load the trained model state
    model.load_state_dict(torch.load(path, map_location=device))
    print( 'MODEL :\n----------\n',model,'\n----------')
    return encoder ,decoder , model
