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
    def __init__(self):
        
        self.vocab = {
            "[SEP]": 0, "[END]": 1, "def": 2, "function": 3, "return": 4,
            "for": 5, "in": 6, "if": 7, "else": 8, "true": 9, "false": 10,
            "range": 11, "max": 12, "min": 13, "sorted": 14, "set": 15,
            "len": 16, "append": 17, "list": 18, "x": 19, "y": 20, "arr": 21,
            "0": 22, "1": 23, "2": 24, "3": 25, "4": 26, "5": 27, "6": 28,
            "7": 29, "8": 30, "9": 31, "": 32,  # Add more as needed
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        tokens = text.split()
        return [self.vocab.get(token, len(self.vocab)) for token in tokens]  

    def decode(self, ids):
        return " ".join([self.inverse_vocab.get(id, "[UNK]") for id in ids])


tokenizer = SimpleTokenizer()

array_problems = [
    {
        "problem": "Find the maximum element in an array.",
        "solution": """
def find_maximum(arr):
    return max(arr)
"""
    },
    {
        "problem": "Reverse an array.",
        "solution": """
def reverse_array(arr):
    return arr[::-1]
"""
    },
    {
        "problem": "Check if an array contains duplicates.",
        "solution": """
def contains_duplicate(arr):
    return len(arr) != len(set(arr))
"""
    },
    {
        "problem": "Find the index of the first occurrence of a target element.",
        "solution": """
def first_occurrence(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
"""
    },
    {
        "problem": "Merge two sorted arrays.",
        "solution": """
def merge_sorted_arrays(arr1, arr2):
    return sorted(arr1 + arr2)
"""
    },
    {
        "problem": "Find the minimum element in a rotated sorted array.",
        "solution": """
def find_min_in_rotated(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > arr[right]:
            left = mid + 1
        else:
            right = mid
    return arr[left]
"""
    },
    {
        "problem": "Find the intersection of two arrays.",
        "solution": """
def intersection(arr1, arr2):
    return list(set(arr1) & set(arr2))
"""
    },
    {
        "problem": "Remove duplicates from a sorted array.",
        "solution": """
def remove_duplicates(arr):
    return list(set(arr))
"""
    },
    {
        "problem": "Rotate an array to the right by k steps.",
        "solution": """
def rotate_array(arr, k):
    k %= len(arr)
    return arr[-k:] + arr[:-k]
"""
    },
    {
        "problem": "Find the longest increasing subsequence.",
        "solution": """
def longest_increasing_subsequence(arr):
    lis = []
    for num in arr:
        if not lis or num > lis[-1]:
            lis.append(num)
        else:
            idx = next(i for i, x in enumerate(lis) if x >= num)
            lis[idx] = num
    return len(lis)
"""
    },
    {
        "problem": "Check if an array is sorted.",
        "solution": """
def is_sorted(arr):
    return arr == sorted(arr)
"""
    },
    {
        "problem": "Find the second largest element in an array.",
        "solution": """
def second_largest(arr):
    unique_arr = list(set(arr))
    unique_arr.sort()
    return unique_arr[-2] if len(unique_arr) > 1 else None
"""
    },
    {
        "problem": "Find the kth largest element in an array.",
        "solution": """
def kth_largest(arr, k):
    return sorted(arr)[-k]
"""
    },
    {
        "problem": "Move all zeroes to the end of an array.",
        "solution": """
def move_zeroes(arr):
    non_zeroes = [x for x in arr if x != 0]
    return non_zeroes + [0] * (len(arr) - len(non_zeroes))
"""
    },
    {
        "problem": "Find the majority element in an array.",
        "solution": """
def majority_element(arr):
    count = {}
    for num in arr:
        count[num] = count.get(num, 0) + 1
    return max(count, key=count.get)
"""
    },
    {
        "problem": "Find the first missing positive integer.",
        "solution": """
def first_missing_positive(arr):
    arr = set(arr)
    i = 1
    while i in arr:
        i += 1
    return i
"""
    },
    {
        "problem": "Find the longest substring without repeating characters.",
        "solution": """
def longest_substring(s):
    char_index = {}
    start = max_length = 0
    for i, char in enumerate(s):
        if char in char_index:
            start = max(start, char_index[char] + 1)
        char_index[char] = i
        max_length = max(max_length, i - start + 1)
    return max_length
"""
    },
    {
        "problem": "Check if two arrays are equal.",
        "solution": """
def arrays_equal(arr1, arr2):
    return sorted(arr1) == sorted(arr2)
"""
    },
]

problems = [item["problem"] for item in array_problems]
solutions = [item["solution"] for item in array_problems]

split_point = int(0.8 * len(problems))

train_dataset = DSADataset(problems[:split_point], solutions[:split_point], tokenizer, max_seq_len)
val_dataset = DSADataset(problems[split_point:], solutions[split_point:], tokenizer, max_seq_len)

train(model, train_dataset, val_dataset, epochs=10, batch_size=32, learning_rate=3e-4)

def generate_solution(model, tokenizer, problem, max_tokens=100):
    model.eval() 
    input_ids = tokenizer.encode(problem + " [SEP] ")
    input_ids = torch.tensor(input_ids).unsqueeze(0) 
    
    generated_ids = input_ids.clone()  
    
    with torch.no_grad(): 
        for _ in range(max_tokens):
            logits = model(generated_ids)
            next_token = torch.argmax(logits[0, -1, :])
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            print(f"Generated token ID: {next_token.item()}, Token: {tokenizer.inverse_vocab.get(next_token.item(), '[UNK]')}")
            if next_token == tokenizer.encode("[END]")[0]:
                break

    return tokenizer.decode(generated_ids[0].tolist())


test_problem = "Find the maximum element in an array."
predicted_solution = generate_solution(model, tokenizer, test_problem)

print("Generated solution for the maximum element problem:")
print(predicted_solution)
