import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ds = load_from_disk("stories_dataset")
id_to_char = {idx:chr(dec) for idx, dec in enumerate(range(32, 126))}
id_to_char[10] = chr(10)
char_to_id = {v:k for k, v in id_to_char.items()}

vocab_size = len(id_to_char)

ndim = 256
hidden_size = ndim * 4
input_size = ndim
output_size = vocab_size
n_cells = 8
n_steps = 512



class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  
        self.w1 = nn.Linear(input_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, output_size)

    def forward(self, input, state):
        out = self.w1(input)
        state = self.w2(state)
        state = F.tanh(out + state)
        out = self.w3(state)
        return out, state
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_cells):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, input_size)
        input_rnn = RNNCell(input_size=input_size, hidden_size=hidden_size, output_size=hidden_size)
        output_rnn = RNNCell(input_size=hidden_size, hidden_size=hidden_size, output_size=output_size)
        self.rnncells = [RNNCell(input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size) for _ in range(n_cells-2)]
        self.rnncells = [input_rnn] + self.rnncells + [output_rnn]
        self.rnncells = nn.ModuleList(self.rnncells)

    def forward(self, idx, states):
        input = self.embed(idx)
        for i, cell in enumerate(self.rnncells):
            input, states[i] = cell(input, states[i])
        return input, states

model = RNN(input_size, hidden_size, output_size, n_cells)
model = model.to(device)
# for name, param in model.named_parameters():
#     print(name, param.shape, param.requires_grad)

def collate_fn(batch):
    for story in batch:
        pass



optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,          # learning rate
    betas=(0.9, 0.99),
    eps=1e-8,
    weight_decay=1e-6 # tiny L2 regularization
)

for step in range(1000):
    story = ds['train'][step]['text']
    optimizer.zero_grad()
    states = [torch.zeros(hidden_size, device=device) for _ in range(n_cells)]
    total_loss = 0.0
    for idx, char in enumerate(story):
        if idx == len(story)-2:
            break
        char_id = torch.tensor(char_to_id[char], device=device)
        next_char = char_to_id[story[idx+1]]
        out, states = model(char_id, states)
        loss = F.cross_entropy(out, torch.tensor(next_char, device=device))   
        total_loss = total_loss + loss

    total_loss = total_loss / len(story)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print(f"Step: {step} - ", f"Loss: {total_loss.item():.4f}")





        


#check the vocab
# og = ds['train'][0]['text']
# fake = ""
# tokens = [char_to_id[char] for char in og]

# print(og, end="\n")
# for id in tokens:
#     fake += id_to_char[id]
# print("this is fake")
# print(fake)

#fallback characters     