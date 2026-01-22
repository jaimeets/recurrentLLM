import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ds = load_from_disk("stories_dataset")
# Byte-level vocab: 0..255
vocab_size = 256

ndim = 256
hidden_size = ndim * 4
input_size = ndim
output_size = vocab_size
n_cells = 8
n_steps = 512


class StoryDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        story = self.dataset[idx]["text"]
        tokens = list(story.encode("utf-8", errors="replace"))
        return torch.tensor(tokens, dtype=torch.long)


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  
        self.ln = nn.LayerNorm(hidden_size)
        self.w1 = nn.Linear(input_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, output_size)

    def forward(self, input, state):
        out = self.w1(input)
        state = self.w2(state)
        state = F.tanh(self.ln(out + state))
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
    lengths = torch.tensor([len(story) for story in batch], dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    for i, story in enumerate(batch):
        padded[i, : len(story)] = story
    # inputs are all but last char; targets are all but first char
    inputs = padded[:, :-1]
    targets = padded[:, 1:]
    lengths = lengths - 1
    return inputs, targets, lengths


train_dataset = StoryDataset(ds["train"])
loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,          # learning rate
    betas=(0.9, 0.99),
    eps=1e-8,
    weight_decay=1e-6 # tiny L2 regularization
)

for step, (inputs, targets, lengths) in enumerate(loader):
    if step >= 100:
        break
    inputs = inputs.to(device)
    targets = targets.to(device)
    lengths = lengths.to(device)
    batch_size, seq_len = inputs.shape
    if seq_len == 0:
        continue

    optimizer.zero_grad()
    states = [torch.zeros(batch_size, hidden_size, device=device) for _ in range(n_cells)]
    total_loss = 0.0
    total_tokens = lengths.sum().clamp_min(1)

    for t in range(seq_len):
        active = lengths > t
        if not active.any():
            break
        idx_t = inputs[:, t]
        out, new_states = model(idx_t, states)
        for i in range(n_cells):
            states[i] = torch.where(active.unsqueeze(1), new_states[i], states[i])
        loss = F.cross_entropy(out[active], targets[:, t][active], reduction="sum")
        total_loss = total_loss + loss

    total_loss = total_loss / total_tokens
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print(f"Step: {step} - ", f"Loss: {total_loss.item():.4f}")

save_path = "char_rnn.pt"
torch.save(model.state_dict(), save_path)
print(f"Saved model to {save_path}")




        


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
