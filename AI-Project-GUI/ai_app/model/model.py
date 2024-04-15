from torch import nn
import torch.nn.functional as F
import torch
import random
from .yolo import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        encoder_outputs = encoder_outputs.transpose(0,1)

        # Calculate the attention scores.
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)

        # multiply attention weights to encoder outputs to obtain context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)
        return context_vector, attn_weights

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        self.dropout(src.float())
        outputs, (hidden, cell) = self.rnn(src)
        return outputs, hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.attention = Attention()
        self.n_layers = n_layers
        self.rnn = nn.LSTM(output_dim + hidden_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, encoder_outputs, hidden, cell):
        self.dropout(input.float())
        context_vector, attn_weights = self.attention(encoder_outputs, hidden[-1])  # using the last layer's hidden state
        rnn_input = torch.cat([input.transpose(0,1), context_vector.unsqueeze(1)], dim=2)  # (batch_size, 1, emb_dim + hidden_dim)
        rnn_input = rnn_input.transpose(0,1)
        output, (hidden, cell) = self.rnn(rnn_input.float(), (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class LSTMSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, print_output = False, teacher_forcing_ratio = 0):
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        outputs = torch.zeros(trg_length, batch_size, self.decoder.output_dim).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)

        input = trg[0,:]
        input = input[None,:]
        pred = []
        for t in range(1, trg_length):
            output, hidden, cell = self.decoder(input, encoder_outputs, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            top1_vals = top1.tolist()
            one_hot_encoded_top1 = torch.tensor([
                [
                  1 if i == val
                  else 0
                  for i in range(self.decoder.output_dim)
              ] for val in top1_vals]
            )
            pred.append(top1_vals)
            input = trg[t] if teacher_force else one_hot_encoded_top1
            encoded_input = []
            input = torch.tensor(input).float().to(device)
            input = input[None,:]
        pred_0 = []
        for i in range(len(pred)):
          pred_0.append(pred[i][0])
        return outputs
    

def get_LSTM_model():
    input_dim = 61
    output_dim = 63
    hidden_dim = 512
    n_layers = 1
    encoder_dropout = 0.2
    decoder_dropout = 0.2
    
    lstm_encoder = LSTMEncoder(
        input_dim,
        hidden_dim,
        n_layers,
        encoder_dropout,
    )

    lstm_decoder = LSTMDecoder(
        output_dim,
        hidden_dim,
        n_layers,
        decoder_dropout,
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "c_lstm_1_layer_512_with_attention_and_params_batch_1234567_epoch")
                
    lstm_model = LSTMSeq2Seq(lstm_encoder, lstm_decoder, device).to(device)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    lstm_model.load_state_dict(state_dict)

    return lstm_model



