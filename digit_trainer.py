from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def load_file(filename: str) -> List[List[List[int]]]:
    with open(filename) as f:
        data = f.read().splitlines()
    return [embed_str(s) for s in data], len(data)

def load_data(dirpath: str) -> Tuple[List[int], List[List[List[int]]]]:
    num_repeat = [2, 4, 1, 1, 1, 1, 1, 1, 1, 1]
    target, feature = [], []
    for i in range(10):
        filename = os.path.join(dirpath, str(i))
        data, n = load_file(filename)
        target += [i] * n * num_repeat[i]
        feature += data * num_repeat[i]
    return target, feature

class DigitDataset(Dataset):
    def __init__(self, data_set: Tuple[List[int], List[List[List[int]]]]):
        self.target = torch.tensor(data_set[0])
        
        sequences = data_set[1]
        sequences_tensor = [torch.tensor(seq, dtype = torch.float32) for seq in sequences]
        self.sequence_lengths = torch.tensor([len(seq) for seq in sequences])
        padded_sequences = rnn_utils.pad_sequence(sequences_tensor, batch_first = True)
        self.data = padded_sequences
        return
        
    def __getitem__(self, index):
        return self.data[index], self.sequence_lengths[index], self.target[index]
    
    def __len__(self):
        return len(self.target)
    
def get_dataloader(batch_size = 5):
    data_set = load_data('./writing_data/digit')
    train_set = DataLoader(DigitDataset(data_set), batch_size, shuffle = True)
    return train_set

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

    def forward(self, input, seq_length):
        packed_input = rnn_utils.pack_padded_sequence(input, seq_length, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output[:, -1, :])
        return output
    
    def loss(self, output, target):
        return self.criterison(output, target)
    
    def score(self, output, target):
        _, predicted = torch.max(output.data, 1)
        return (predicted == target).sum().cpu().item()
    
def train(model, train_loader, num_epochs = 5, learning_rate = 0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    total_step = len(train_loader)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_score = 0
        for i, (data, seq_length, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data, seq_length)
            loss = model.loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item() * data.size(0)
            total_score += model.score(output, target)
        scheduler.step()
        total_loss /= len(train_loader.dataset)
        total_score /= len(train_loader.dataset)
        print ('Epoch [{}/{}], Loss: {:.4f}, score: {:.4f}'.format(epoch+1, num_epochs, total_loss, total_score))
    model.eval().cpu()
    torch.save(model.state_dict(), 'model.ckpt')
    return
                
if __name__ == '__main__':
    model = GRU().to(device)
    train_loader = get_dataloader()
    train(model, train_loader, num_epochs = 500)