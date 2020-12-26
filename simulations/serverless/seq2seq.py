import math
import torch
import random
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder module
    """
    def __init__(
        self, 
        input_size, 
        hidden_size,
        num_layers=1, 
        dropout=0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.rnn = nn.GRU(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            dropout=self.dropout, 
        )

    def forward(self, x, hidden=None):
        outputs, hidden = self.rnn(x, hidden)
        return outputs, hidden


class Attention(nn.Module):
    """
    Attention module
    """
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)

        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]

        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    """
    Decoder module
    """
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        output_size,
        num_layers=1, 
        dropout=0.2
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention = Attention(hidden_size)
        
        self.rnn = nn.GRU(
            input_size=self.input_size + self.hidden_size, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers, 
            dropout=self.dropout
        )
        self.out = nn.Linear(hidden_size * 2, output_size)

    def embed_action(self, action):
        action = action.long().unsqueeze(0)
        action = F.one_hot(action, self.input_size).float()

        return action

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        input = self.embed_action(input)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine input and attended context, run through RNN
        rnn_input = torch.cat([input, context], 2)
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    """
    Sequence to Sequence model
    """
    def __init__(self, encoder, decoder, is_actor):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.is_actor = is_actor

    def get_mask_list(self, observation, vocab_size):
        max_len = observation.size(0)
        mask_list = []

        for i in range(max_len):
            observation_i = observation[i, :].squeeze(0)
            invoke_num = observation_i[-1].long()
            # Inactive functions
            if invoke_num == 0: 
                cpu = observation_i[-3].long()
                memory = observation_i[-2].long()
                encode_position = (cpu - 1) * 8 + memory - 1
                mask = torch.ones(1, vocab_size) * -10e6
                mask[:, encode_position] = 0
            else:
                mask = torch.zeros(1, vocab_size)

            mask_list.append(mask)

        return mask_list

    def forward(self, observation, action):
        batch_size = observation.size(1)
        # Output the same length of operations if actor, otherwise the value
        if self.is_actor is True:
            max_len = observation.size(0)
        else:
            max_len = 1

        vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocab_size)

        encoder_output, hidden = self.encoder(observation)
        hidden = hidden[:self.decoder.num_layers]
        output = action[0, :]  # <predict>

        # Get mask list if actor
        if self.is_actor is True:
            mask_list = self.get_mask_list(observation, vocab_size)

        # Inference
        for t in range(0, max_len):
            output, hidden, attn_weights = self.decoder(
                output, 
                hidden, 
                encoder_output
            )
            # Apply mask to output if actor
            if self.is_actor is True:
                output = output + mask_list[t]

            outputs[t] = output
            top1 = output.data.max(1)[1]
            output = top1
            
        return outputs