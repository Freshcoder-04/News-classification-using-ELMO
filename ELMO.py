import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import brown
from tqdm import tqdm  # progress bar

from tokenizer import custom_nlp_tokenizer

# Set environment variable if needed (Kaggle usually auto-detects GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Download Brown corpus if not already available
nltk.download('brown')

# Enable CuDNN benchmark for improved performance (if input sizes remain constant)
torch.backends.cudnn.benchmark = True

# --- Vocabulary utilities ---
def build_vocab(sentences, min_freq=1):
    freq = {}
    for sent in sentences:
        for word in sent:
            word = word.lower()
            freq[word] = freq.get(word, 0) + 1
    # Special tokens: <pad> for padding and <unk> for unknown words.
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, count in freq.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def sentence_to_indices(sentence, vocab):
    return [vocab.get(word.lower(), vocab["<unk>"]) for word in sentence]

# --- Dataset for Language Modeling ---
class LMDataset(Dataset):
    def __init__(self, tokenized_sentences, vocab):
        self.vocab = vocab
        self.data = []
        # Each tokenized sentence is used to create a training sample.
        for sent in tokenized_sentences:
            if len(sent) < 2:
                continue
            indices = sentence_to_indices(sent, vocab)
            # For forward LM: input = sentence[:-1], target = sentence[1:]
            # For backward LM: input = sentence[1:], target = sentence[:-1]
            self.data.append(torch.tensor(indices))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sent = self.data[idx]
        forward_input = sent[:-1]
        forward_target = sent[1:]
        backward_input = sent[1:]
        backward_target = sent[:-1]
        return forward_input, forward_target, backward_input, backward_target

def lm_collate_fn(batch):
    # Collate a batch of LM samples.
    forward_inputs, forward_targets, backward_inputs, backward_targets = zip(*batch)
    forward_inputs = nn.utils.rnn.pad_sequence(forward_inputs, batch_first=True, padding_value=0)
    forward_targets = nn.utils.rnn.pad_sequence(forward_targets, batch_first=True, padding_value=-100)
    backward_inputs = nn.utils.rnn.pad_sequence(backward_inputs, batch_first=True, padding_value=0)
    backward_targets = nn.utils.rnn.pad_sequence(backward_targets, batch_first=True, padding_value=-100)
    return forward_inputs, forward_targets, backward_inputs, backward_targets

# --- ELMo Model Definition ---
class ELMoModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        This model creates three representations per token:
         • e0: non-contextual embedding (dimension = embed_size)
         • e1: output from the first Bi-LSTM layer (dimension = hidden_size*2)
         • e2: output from the second Bi-LSTM layer (dimension = hidden_size*2)
        For language modeling, we use e2’s forward part to predict next tokens and
        its backward part to predict previous tokens.
        """
        super(ELMoModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm1 = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.fc_forward = nn.Linear(hidden_size, vocab_size)
        self.fc_backward = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        e0 = self.embedding(x)  # (batch, seq_len, embed_size)
        out1, _ = self.lstm1(e0)  # (batch, seq_len, hidden_size*2)
        out2, _ = self.lstm2(out1)  # (batch, seq_len, hidden_size*2)
        hidden_size = out2.size(2) // 2
        forward_hidden = out2[:, :, :hidden_size]   # for predicting next word
        backward_hidden = out2[:, :, hidden_size:]   # for predicting previous word
        pred_forward = self.fc_forward(forward_hidden)
        pred_backward = self.fc_backward(backward_hidden)
        return e0, out1, out2, pred_forward, pred_backward

def train_elmo():
    # Hyperparameters
    embed_size = 100
    hidden_size = 256
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001

    # Cache file for tokenized Brown sentences
    cache_file = "brown_tokenized.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            tokenized_sentences = pickle.load(f)
        print("Loaded cached tokenized Brown corpus.")
    else:
        tokenized_sentences = []
        print("Tokenizing Brown corpus...")
        # Use tqdm to show progress over Brown sentences.
        for sent in tqdm(brown.sents(), desc="Tokenizing sentences"):
            raw_text = " ".join(sent)
            tokenized_list = custom_nlp_tokenizer(raw_text)
            for tokenized_sent in tokenized_list:
                if len(tokenized_sent) >= 2:
                    tokenized_sentences.append(tokenized_sent)
        with open(cache_file, "wb") as f:
            pickle.dump(tokenized_sentences, f)
        print("Tokenized Brown corpus and saved to cache.")

    # Build vocabulary (using words that occur at least twice)
    vocab = build_vocab(tokenized_sentences, min_freq=2)
    vocab_size = len(vocab)
    print("Vocabulary size:", vocab_size)

    # Prepare dataset and dataloader with improved performance settings.
    dataset = LMDataset(tokenized_sentences, vocab)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lm_collate_fn,
        num_workers=4,     # Adjust number of workers as per Kaggle's environment
        pin_memory=True    # Pin memory for faster GPU transfers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = ELMoModel(vocab_size, embed_size, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for forward_inputs, forward_targets, backward_inputs, backward_targets in progress_bar:
            forward_inputs = forward_inputs.to(device, non_blocking=True)
            forward_targets = forward_targets.to(device, non_blocking=True)
            backward_inputs = backward_inputs.to(device, non_blocking=True)
            backward_targets = backward_targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            # Forward LM pass
            _, _, _, pred_forward, _ = model(forward_inputs)
            pred_forward = pred_forward.permute(0, 2, 1)  # (batch, vocab_size, seq_len)
            loss_forward = criterion(pred_forward, forward_targets)
            # Backward LM pass
            _, _, _, _, pred_backward = model(backward_inputs)
            pred_backward = pred_backward.permute(0, 2, 1)
            loss_backward = criterion(pred_backward, backward_targets)
            
            loss = loss_forward + loss_backward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        print(f"Epoch: {epoch+1} | Average Loss: {total_loss/len(dataloader):.4f}")
    
    # Save the model along with the vocabulary.
    torch.save({'model_state_dict': model.state_dict(), 'vocab': vocab}, "bilstm.pt")
    print("Saved ELMo model to bilstm.pt")

if __name__ == "__main__":
    train_elmo()
