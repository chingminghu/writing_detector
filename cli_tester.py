import torch
import torch.nn as nn
from typing import List, Tuple
import torch.nn.utils.rnn as rnn_utils
import numpy as np

class GRU(nn.Module):
    def __init__(self, input_size = 2, hidden_size = 32, num_layers = 2, output_size = 10):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,  
            batch_first=True,
            dropout=0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterison = nn.CrossEntropyLoss()

    def forward(self, input):
        seq_length = torch.tensor([len(input[0])], dtype = torch.int64)
        packed_input = rnn_utils.pack_padded_sequence(input, seq_length, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output[:, -1, :])
        return output
    
def embed_str(s: str) -> List[List[int]]:
    kb_map = [['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
           ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']'],
           ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'"],
           ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/']]
    ret = []
    for c in s:
        for row in range(len(kb_map)):
            if c in kb_map[row]:
                ret.append([row, kb_map[row].index(c)])
                break
    return ret

def eval(model, s: str):
    model.eval()
    input = torch.tensor(embed_str(s), dtype = torch.float32)
    with torch.no_grad():
        output = model(input.unsqueeze(0))[0]
        _, predicted = torch.max(output, 0)
        prob = torch.nn.functional.softmax(output, dim=0).numpy()
        print("digit |      0       1       2       3       4       5       6       7       8       9 ")
        print("-----------------------------------------------------------------------------------------")
        print("prob. |", end = ' ')
        for i in range(10):
            print(" {:.4f}".format(prob[i]), end = ' ')
        print('\n')
    return predicted.item()
    
if __name__ == '__main__':
    model = GRU()
    model.load_state_dict(torch.load('model.ckpt', map_location = 'cpu', weights_only = True))
    while True:
        try:
            test = input('Enter test data: ')
            print()
            pred = eval(model, test)
            print("----------------------")
            print("| Predicted Digit: {} |".format(pred))
            print("----------------------")
            print('\n=========================================================================================\n')
        except EOFError:
            break