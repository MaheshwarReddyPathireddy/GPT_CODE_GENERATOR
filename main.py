import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_seq_len):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_seq_len, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.max_seq_len, "Cannot forward sequence length larger than model's maximum"

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.attn = nn.MultiheadAttention(embed_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x
class DSADataset(Dataset):
    def __init__(self, problems, solutions, tokenizer, max_seq_len):
        self.problems = problems
        self.solutions = solutions
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]
        solution = self.solutions[idx]
    
        input_ids = self.tokenizer.encode(problem + " [SEP] " + solution)
    
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
        
        return torch.tensor(input_ids), torch.tensor(input_ids)

def train(model, train_dataset, val_dataset, epochs, batch_size, learning_rate):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss / len(val_loader)}")

vocab_size = 10000 
embed_size = 256
num_heads = 8
num_layers = 6
max_seq_len = 512

model = GPT(vocab_size, embed_size, num_heads, num_layers, max_seq_len)
class SimpleTokenizer:
    def encode(self, text):

        pass

    def decode(self, ids):

        pass

tokenizer = SimpleTokenizer()


problems = ["Implement quicksort", "Write a function to reverse a linked list"]
solutions = ["def quicksort(arr): ...", "def reverse_linked_list(head): ..."]

train_dataset = DSADataset(problems[:800], solutions[:800], tokenizer, max_seq_len)
val_dataset = DSADataset(problems[800:], solutions[800:], tokenizer, max_seq_len)

train(model, train_dataset, val_dataset, epochs=10, batch_size=32, learning_rate=3e-4)

def generate_solution(model, tokenizer, problem, max_tokens=100):
    model.eval()
    input_ids = tokenizer.encode(problem + " [SEP] ")
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            next_token = torch.argmax(logits[0, -1, :])
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            if next_token == tokenizer.encode("[END]")[0]:
                break
    
    return tokenizer.decode(input_ids[0].tolist())

test_problem = "Implement a binary search function"
print(generate_solution(model, tokenizer, test_problem))
print("check the sensitive information once again")
# prime number problem
prime_problem = "Write a function to find if a number is prime"
prime_solution = """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
"""

problems.append(prime_problem)
solutions.append(prime_solution)

train_dataset = DSADataset(problems[:800], solutions[:800], tokenizer, max_seq_len)
val_dataset = DSADataset(problems[800:], solutions[800:], tokenizer, max_seq_len)

train(model, train_dataset, val_dataset, epochs=10, batch_size=32, learning_rate=3e-4)
