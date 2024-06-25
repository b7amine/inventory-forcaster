
import pandas as pd
import numpy as np 
import os 
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils import *
from utils import normalize_current_levels ,format_inference_sample , min_max_denormalize , write_json_to_directory , InsufficientDataError
import matplotlib.pyplot as plt
from scipy.stats import entropy
import torch

device = torch.device("cpu")

INPUT_SAMPLES_DIR = "samples"
PRED_OUTPUT_DIR = "output"
MODEL_PATH = '../models/model_dim_128_att_True_tfr_0_layers_2.pt'  

def preprocess(df):
    if len(df)<4:
        raise InsufficientDataError()
    df[DATECOL]=pd.to_datetime(df[DATECOL])
    df[YEARCOL]=df[DATECOL].dt.year
    df.drop_duplicates([IDCOL,DATECOL],inplace=True)
    df[WEEKOFYEAR]=df[DATECOL].dt.isocalendar().week
    df = df.sort_values([DATECOL]).reset_index(drop=True)
    if df[TARGET].isna().sum():
        df[TARGET] = df[TARGET].interpolate(method="linear")
    df = df.groupby([IDCOL,YEARCOL,WEEKOFYEAR,STATICOL],as_index=False)[TARGET].sum()
    df = df.groupby([IDCOL,STATICOL], as_index=False).agg(
        {WEEKOFYEAR: list,
        TARGET: list,}
    ).rename(columns={WEEKOFYEAR: LISTWEEKSOFYEAR, TARGET: LISTTARGET})
    df[NORM_STATICOL] = normalize_current_levels(df[STATICOL])

    input_seq , output_seq , target_weeks = format_inference_sample(weeks=df[LISTWEEKSOFYEAR].values[0], data=df[LISTTARGET].values[0])
    input_scalar = torch.tensor(df[NORM_STATICOL].values[0]).reshape(1,1).float()
    input_seq , output_seq , target_weeks =input_seq.float() , output_seq.float() , target_weeks.float()
    return input_seq, input_scalar, target_weeks, output_seq

def predict_with_uncertainty(model, inputs_seq, inputs_scalar, targets_weeks, targets=None, n_iterations=100):
    model.train()  # Ensure dropout is enabled
    
    predictions = []

    for _ in range(n_iterations):
        with torch.no_grad():
            seq_inputs = inputs_seq.clone().to(device)
            scalar_inputs = inputs_scalar.clone().to(device)
            target_weeks = targets_weeks.clone().to(device)
            batch_size = seq_inputs.shape[0]

            outputs = torch.zeros(batch_size, target_weeks.shape[1], 1).to(device)
            encoder_outputs, hidden = model.encoder(seq_inputs, scalar_inputs)

            decoder_input = seq_inputs[:, -1, 0].unsqueeze(1).unsqueeze(2)  # Adding feature dimension
            steps_per_chunk = 4

            for t in range(0, target_weeks.shape[1], steps_per_chunk):
                for step in range(steps_per_chunk):
                    current_idx = t + step
                    if current_idx >= target_weeks.shape[1]:
                        break

                    week_encoding = target_weeks[:, current_idx, :].unsqueeze(1)  # Shape (batch_size, 1, 2)
                    output, hidden = model.decoder(decoder_input, hidden, encoder_outputs, week_encoding)
                    outputs[:, current_idx] = output

                    # Prepare next input for decoder
                    if targets is not None and current_idx < targets.shape[1]:
                        decoder_input = targets[:, current_idx].unsqueeze(1).unsqueeze(2)
                    else:
                        decoder_input = output.unsqueeze(1)
                
                if t + steps_per_chunk < target_weeks.shape[1]:
                    actual_idx_end = t + steps_per_chunk

                    if targets is not None and actual_idx_end <= targets.shape[1]:
                        actual_quantity = targets[:, t:actual_idx_end].unsqueeze(2)
                        actual_weeks = target_weeks[:, t:actual_idx_end, :]

                        actual_sequence = torch.cat((actual_quantity, actual_weeks), dim=2)
                        seq_inputs = torch.cat((seq_inputs, actual_sequence), dim=1)
                        encoder_outputs, hidden = model.encoder(seq_inputs, scalar_inputs)
                        decoder_input = targets[:, actual_idx_end - 1].unsqueeze(1).unsqueeze(2)

                    else:
                        # Update seq_inputs with own predictions if targets are not provided
                        own_predictions = outputs[:, t:actual_idx_end, :]
                        actual_weeks = target_weeks[:, t:actual_idx_end, :]

                        own_sequence = torch.cat((own_predictions, actual_weeks), dim=2)
                        seq_inputs = torch.cat((seq_inputs, own_sequence), dim=1)
                        encoder_outputs, hidden = model.encoder(seq_inputs, scalar_inputs)
                        decoder_input = outputs[:, actual_idx_end - 1].unsqueeze(1)

            predictions.append(min_max_denormalize(outputs.cpu().numpy().reshape(-1)))

    predictions = np.array(predictions)
    pred_mean = predictions.mean(axis=0).flatten()
    pred_std = predictions.std(axis=0).flatten()
    # Normalized Entropy
    norm_entropies = []
    for pred in predictions:
        pred_hist, _ = np.histogram(pred, bins=10, density=True)
        norm_entropies.append(entropy(pred_hist) / np.log(len(pred_hist)))

    if targets is not None:
        gt = min_max_denormalize(targets).detach().numpy().flatten().tolist()
    else:
        gt = []  # If targets are not available, ground truth cannot be determined
    result = {
        'ground_truth': gt,
        'predictions': pred_mean.tolist(),
        'pred_mean':pred_mean.tolist(),
        'pred_upper_band_95_confidence': (pred_mean + 1.96 * pred_std).tolist(),
        'pred_lower_band_95_confidence': (pred_mean - 1.96 * pred_std).tolist(),
        'uncertainty_scores(std)': pred_std.tolist(),
        'uncertainty_score(normalized_entropy)':norm_entropies,
        'mean_uncertainty_score(normalized_entropy)':np.mean(norm_entropies),
    }
    #plot_prediction(result=result)
    return result

def plot_prediction(result):
    df = pd.DataFrame({
        'pred_mean': result['pred_mean'],
        'pred_upper': result['pred_upper_band_95_confidence'],
        'pred_lower': result['pred_lower_band_95_confidence']
    })
    
    if len(result['ground_truth']) > 0:
        df['gt'] = result['ground_truth']

    plt.figure(figsize=(20, 4))
    if len(result['ground_truth']) > 0:
        plt.plot(df['gt'], label='Ground Truth')
    plt.plot(df['pred_mean'], label='Prediction Mean')
    plt.fill_between(df.index, df['pred_upper'], df['pred_lower'], color='green', alpha=0.3, label='Uncertainty Band (95%)')
    plt.legend()
    plt.title(f'Predictions with Uncertainty for Sample')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(.4)
    plt.show()

def predict_plot_attention(model, inputs_seq, inputs_scalar, targets_weeks, targets=None):
    model.eval()  # Ensure dropout is disabled

    with torch.no_grad():
        seq_inputs = inputs_seq.clone().to(device)
        scalar_inputs = inputs_scalar.clone().to(device)
        target_weeks = targets_weeks.clone().to(device)
        batch_size = seq_inputs.shape[0]

        outputs = torch.zeros(batch_size, target_weeks.shape[1], 1).to(device)
        encoder_outputs, hidden = model.encoder(seq_inputs, scalar_inputs)

        decoder_input = seq_inputs[:, -1, 0].unsqueeze(1).unsqueeze(2)  # Adding feature dimension
        steps_per_chunk = 4

        all_attention_scores = []  # To store attention scores

        for t in range(0, target_weeks.shape[1], steps_per_chunk):
            for step in range(steps_per_chunk):
                current_idx = t + step
                if current_idx >= target_weeks.shape[1]:
                    break

                week_encoding = target_weeks[:, current_idx, :].unsqueeze(1)  # Shape (batch_size, 1, 2)
                output, hidden = model.decoder(decoder_input, hidden, encoder_outputs, week_encoding)
                outputs[:, current_idx] = output

                # Extract and store attention scores
                if model.decoder.attention_enabled:
                    attn_weights = model.decoder.attention(hidden, encoder_outputs).cpu().numpy()
                    all_attention_scores.append(attn_weights.squeeze(0))

                # Prepare next input for decoder
                if targets is not None and current_idx < targets.shape[1]:
                    decoder_input = targets[:, current_idx].unsqueeze(1).unsqueeze(2)
                else:
                    decoder_input = output.unsqueeze(1)

            if t + steps_per_chunk < target_weeks.shape[1]:
                actual_idx_end = t + steps_per_chunk

                if targets is not None and actual_idx_end <= targets.shape[1]:
                    actual_quantity = targets[:, t:actual_idx_end].unsqueeze(2)
                    actual_weeks = target_weeks[:, t:actual_idx_end, :]

                    actual_sequence = torch.cat((actual_quantity, actual_weeks), dim=2)
                    seq_inputs = torch.cat((seq_inputs, actual_sequence), dim=1)
                    encoder_outputs, hidden = model.encoder(seq_inputs, scalar_inputs)
                    decoder_input = targets[:, actual_idx_end - 1].unsqueeze(1).unsqueeze(2)
                else:
                    # Update seq_inputs with own predictions if targets are not provided
                    own_predictions = outputs[:, t:actual_idx_end, :]
                    actual_weeks = target_weeks[:, t:actual_idx_end, :]

                    own_sequence = torch.cat((own_predictions, actual_weeks), dim=2)
                    seq_inputs = torch.cat((seq_inputs, own_sequence), dim=1)
                    encoder_outputs, hidden = model.encoder(seq_inputs, scalar_inputs)
                    decoder_input = outputs[:, actual_idx_end - 1].unsqueeze(1)
    attention_scores =all_attention_scores
    max_seq_len = max(len(attn) for attn in attention_scores)
    
    # Create a matrix to hold the attention scores
    attn_matrix = np.full((len(attention_scores), max_seq_len), np.nan)
    
    for i, attn in enumerate(attention_scores):
        attn_matrix[i, :len(attn)] = attn

    # Plot the attention scores
    fig, ax = plt.subplots(figsize=(20, 5))
    cax = ax.matshow(attn_matrix, cmap='viridis', aspect='auto', vmin=0)
    fig.colorbar(cax)

    # Set x and y ticks
    x_ticks = np.arange(1, max_seq_len + 1, 4)
    y_ticks = np.arange(5, len(attention_scores) + 5, 4)
    ax.set_xticks(x_ticks - 1)
    ax.set_yticks(y_ticks - 5)
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    
    # Highlight rows with a thin red box
    for i in range(len(attention_scores)):
        ax.add_patch(plt.Rectangle((-.5, i-.5), max_seq_len, 1, fill=False, edgecolor='red', lw=.2))

    ax.set_xlabel('Input Sequence Position')
    ax.set_ylabel('Prediction Step')
    ax.set_title('Attention Scores Across Prediction Steps')
    plt.show()
    return min_max_denormalize(outputs.detach().numpy().flatten())


if __name__ == "__main__" : 
    encoder ,decoder , model = load_model(path=MODEL_PATH)
    # Load the trained model state
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print( 'MODEL :\n----------\n',model,'\n----------')


    samples_paths = os.listdir(INPUT_SAMPLES_DIR)
    output={}
    for path in samples_paths:
        try:
            df = pd.read_json(os.path.join(INPUT_SAMPLES_DIR,path))
            input_seq, input_scalar, target_weeks, output_seq = preprocess(df)
            output_sample = predict_with_uncertainty(model, input_seq, input_scalar, target_weeks, targets=output_seq)
            output.update({path:output_sample})
            print(f'Processed successfully : {path} ')

        except Exception as e: 
            print(f'Failed processing sample {path} , error : \n',e)
    write_json_to_directory(output, os.path.join(PRED_OUTPUT_DIR,f'predictions_{samples_paths[0]}_to_{samples_paths[-1]}.json'))