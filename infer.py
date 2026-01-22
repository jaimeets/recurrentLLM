import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

id_to_char = {idx: chr(dec) for idx, dec in enumerate(range(32, 126))}
id_to_char[10] = chr(10)
char_to_id = {v: k for k, v in id_to_char.items()}

vocab_size = len(id_to_char)

ndim = 256
hidden_size = ndim * 4
input_size = ndim
output_size = vocab_size
n_cells = 8


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
        self.rnncells = [RNNCell(input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size) for _ in range(n_cells - 2)]
        self.rnncells = [input_rnn] + self.rnncells + [output_rnn]
        self.rnncells = nn.ModuleList(self.rnncells)

    def forward(self, idx, states):
        input = self.embed(idx)
        for i, cell in enumerate(self.rnncells):
            input, states[i] = cell(input, states[i])
        return input, states


def sample(model, start_text, max_new_chars=200, temperature=1.0):
    model.eval()
    states = [torch.zeros(1, hidden_size, device=device) for _ in range(n_cells)]
    for ch in start_text:
        idx = torch.tensor([char_to_id.get(ch, 0)], device=device)
        _, states = model(idx, states)
    current = start_text
    last_ch = start_text[-1] if len(start_text) > 0 else " "
    for _ in range(max_new_chars):
        idx = torch.tensor([char_to_id.get(last_ch, 0)], device=device)
        logits, states = model(idx, states)
        logits = logits.squeeze(0) / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        next_ch = id_to_char.get(next_id, " ")
        current += next_ch
        last_ch = next_ch
    return current


def main():
    model = RNN(input_size, hidden_size, output_size, n_cells).to(device)
    model.load_state_dict(torch.load("char_rnn.pt", map_location=device))

    with torch.no_grad():
        prompt = "Once upon a time"
        out = sample(model, prompt, max_new_chars=200, temperature=0.9)
        print(out)


if __name__ == "__main__":
    main()
