import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, seq_input_dim, scalar_input_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(seq_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.scalar_fc = nn.Linear(scalar_input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, seq_inputs, scalar_inputs):
        outputs, hidden = self.gru(seq_inputs)
        scalar_hidden = self.scalar_fc(scalar_inputs).unsqueeze(0).repeat(self.gru.num_layers, 1, 1)  # Repeat for num_layers
        combined_hidden = hidden + scalar_hidden  # Combine the hidden states
        combined_hidden = self.dropout(combined_hidden)
        return outputs, combined_hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1, dropout=0.5, attention_enabled=True):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(hidden_dim + output_dim + 2, hidden_dim, num_layers, batch_first=True, dropout=dropout)  # +2 for week encoding
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention_enabled = attention_enabled
        if attention_enabled:
            self.attention = Attention(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden, encoder_outputs, week_encoding):
        if self.attention_enabled:
            attn_weights = self.attention(hidden, encoder_outputs)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            gru_input = torch.cat((x, attn_applied, week_encoding), dim=2)  # Include week encoding in input
        else:
            last_encoder_output = encoder_outputs[:, -1, :].unsqueeze(1)  # Use the last encoder output state
            gru_input = torch.cat((x, last_encoder_output, week_encoding), dim=2)  # Adjust dimensions as needed
        
        output, hidden = self.gru(gru_input, hidden)
        output = self.dropout(output)
        prediction = self.relu(self.fc(output.squeeze(1)))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.train_teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, seq_inputs, scalar_inputs, trg, target_weeks, trainBool):
        pass_tfr = self.train_teacher_forcing_ratio if trainBool else 0
        batch_size = seq_inputs.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = 1  # Since trg is of shape (batch_size, horizon)
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(seq_inputs, scalar_inputs)
        
        # Initial input to the decoder is the last observed element in the sequence
        decoder_input = seq_inputs[:, -1, 0].unsqueeze(1).unsqueeze(2)  # Adding feature dimension
        
        steps_per_chunk = 4
        
        for t in range(0, trg_len, steps_per_chunk):
            # Decode for 4 steps using predictions
            for step in range(steps_per_chunk):
                current_idx = t + step
                if current_idx >= trg_len:
                    break

                week_encoding = target_weeks[:, current_idx, :].unsqueeze(1)  # Shape (batch_size, 1, 2)
                output, hidden = self.decoder(decoder_input, hidden, encoder_outputs, week_encoding)
                outputs[:, current_idx] = output
                
                # Prepare next input for decoder
                teacher_force = torch.rand(1).item() < pass_tfr ## DEBUG WHEN CALLED IN TRAIN VS EVAL
                decoder_input = trg[:, current_idx].unsqueeze(1).unsqueeze(2) if teacher_force else output.unsqueeze(1)
            
            # After 4 steps, use the actual values to update the context
            if t + steps_per_chunk < trg_len:
                actual_idx_end = t + steps_per_chunk
                
                # Extract quantitySoldWeekly and corresponding week_encoding
                actual_quantity = trg[:, t:actual_idx_end].unsqueeze(2)  # Shape (batch_size, steps_per_chunk, 1)
                actual_weeks = target_weeks[:, t:actual_idx_end, :]  # Shape (batch_size, steps_per_chunk, 2)
            
                # Concatenate to form the new sequence part
                actual_sequence = torch.cat((actual_quantity, actual_weeks), dim=2)  # Shape (batch_size, 4, 3)
                
                # Update the input sequence for the encoder
                seq_inputs = torch.cat((seq_inputs, actual_sequence), dim=1)
                
                # Encode the updated sequence
                encoder_outputs, hidden = self.encoder(seq_inputs, scalar_inputs)

                decoder_input = trg[:, actual_idx_end - 1].unsqueeze(1).unsqueeze(2)
              
        return outputs.squeeze(2)