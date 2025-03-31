import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
from tqdm import tqdm

from tokenizer import custom_nlp_tokenizer
from ELMO import ELMoModel

# ------------------------
# Classifier Definitions
# ------------------------

class FrozenLambdasClassifier(nn.Module):
    """
    Uses fixed (frozen) lambda weights to combine ELMo embeddings.
    Dimensions match: e0_dim=100, e1_dim=512, e2_dim=512.
    """
    def __init__(self, elmo_model, num_classes, combined_dim=256, 
                 rnn_hidden_size=128, rnn_num_layers=1, bidirectional=True,
                 e0_dim=100, e1_dim=512, e2_dim=512):
        super(FrozenLambdasClassifier, self).__init__()
        self.elmo_model = elmo_model
        self.elmo_model.eval()
        for param in self.elmo_model.parameters():
            param.requires_grad = False

        # Projection layers to common dimension.
        self.proj_e0 = nn.Linear(e0_dim, combined_dim)
        self.proj_e1 = nn.Linear(e1_dim, combined_dim)
        self.proj_e2 = nn.Linear(e2_dim, combined_dim)
        
        rand_lambdas = torch.rand(3)
        self.register_buffer("lambdas", rand_lambdas / rand_lambdas.sum())
        
        # RNN for processing combined representations.
        self.rnn = nn.LSTM(input_size=combined_dim, hidden_size=rnn_hidden_size, 
                           num_layers=rnn_num_layers, batch_first=True, bidirectional=bidirectional)
        final_dim = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.classifier = nn.Linear(final_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) token indices
        e0, e1, e2, _, _ = self.elmo_model(x)
        # Detach to prevent gradients flowing back.
        e0, e1, e2 = e0.detach(), e1.detach(), e2.detach()
        
        # Project each embedding to the common space.
        proj_e0 = self.proj_e0(e0)  # (batch, seq_len, combined_dim)
        proj_e1 = self.proj_e1(e1)
        proj_e2 = self.proj_e2(e2)
        
        # Combine with fixed lambda weights.
        combined = (self.lambdas[0] * proj_e0 +
                    self.lambdas[1] * proj_e1 +
                    self.lambdas[2] * proj_e2)
        
        rnn_out, (h_n, _) = self.rnn(combined)
        # For bidirectional, concatenate the last forward and backward hidden states.
        if self.rnn.bidirectional:
            final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final = h_n[-1]
        
        logits = self.classifier(final)
        return logits


class TrainableLambdasClassifier(nn.Module):
    """
    Uses trainable lambda weights to combine ELMo embeddings.
    """
    def __init__(self, elmo_model, num_classes, combined_dim=256, 
                 rnn_hidden_size=128, rnn_num_layers=1, bidirectional=True,
                 e0_dim=100, e1_dim=512, e2_dim=512):
        super(TrainableLambdasClassifier, self).__init__()
        self.elmo_model = elmo_model
        self.elmo_model.eval()
        for param in self.elmo_model.parameters():
            param.requires_grad = False

        self.proj_e0 = nn.Linear(e0_dim, combined_dim)
        self.proj_e1 = nn.Linear(e1_dim, combined_dim)
        self.proj_e2 = nn.Linear(e2_dim, combined_dim)
        
        # Trainable lambda weights (initialized to 1/3 each)
        self.lambdas = nn.Parameter(torch.tensor([1/3, 1/3, 1/3], dtype=torch.float))
        
        self.rnn = nn.LSTM(input_size=combined_dim, hidden_size=rnn_hidden_size, 
                           num_layers=rnn_num_layers, batch_first=True, bidirectional=bidirectional)
        final_dim = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.classifier = nn.Linear(final_dim, num_classes)

    def forward(self, x):
        e0, e1, e2, _, _ = self.elmo_model(x)
        e0, e1, e2 = e0.detach(), e1.detach(), e2.detach()
        
        proj_e0 = self.proj_e0(e0)
        proj_e1 = self.proj_e1(e1)
        proj_e2 = self.proj_e2(e2)
        
        combined = (self.lambdas[0] * proj_e0 +
                    self.lambdas[1] * proj_e1 +
                    self.lambdas[2] * proj_e2)
        
        rnn_out, (h_n, _) = self.rnn(combined)
        if self.rnn.bidirectional:
            final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final = h_n[-1]
        
        logits = self.classifier(final)
        return logits


class LearnableFunctionClassifier(nn.Module):
    """
    Uses a learnable function (an MLP) to combine the three projected ELMo embeddings.
    """
    def __init__(self, elmo_model, num_classes, combined_dim=256, 
                 rnn_hidden_size=128, rnn_num_layers=1, bidirectional=True,
                 e0_dim=100, e1_dim=512, e2_dim=512):
        super(LearnableFunctionClassifier, self).__init__()
        self.elmo_model = elmo_model
        self.elmo_model.eval()
        for param in self.elmo_model.parameters():
            param.requires_grad = False

        self.proj_e0 = nn.Linear(e0_dim, combined_dim)
        self.proj_e1 = nn.Linear(e1_dim, combined_dim)
        self.proj_e2 = nn.Linear(e2_dim, combined_dim)
        
        # MLP to combine concatenated embeddings (3 * combined_dim -> combined_dim)
        self.combine_mlp = nn.Sequential(
            nn.Linear(3 * combined_dim, combined_dim),
            nn.ReLU(),
        )
        
        self.rnn = nn.LSTM(input_size=combined_dim, hidden_size=rnn_hidden_size, 
                           num_layers=rnn_num_layers, batch_first=True, bidirectional=bidirectional)
        final_dim = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.classifier = nn.Linear(final_dim, num_classes)

    def forward(self, x):
        e0, e1, e2, _, _ = self.elmo_model(x)
        e0, e1, e2 = e0.detach(), e1.detach(), e2.detach()
        
        proj_e0 = self.proj_e0(e0)
        proj_e1 = self.proj_e1(e1)
        proj_e2 = self.proj_e2(e2)
        
        # Concatenate along the feature dimension.
        cat = torch.cat([proj_e0, proj_e1, proj_e2], dim=-1)
        combined = self.combine_mlp(cat)
        
        rnn_out, (h_n, _) = self.rnn(combined)
        if self.rnn.bidirectional:
            final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final = h_n[-1]
        
        logits = self.classifier(final)
        return logits

# ------------------------
# News Dataset Definition
# ------------------------

class NewsDataset(Dataset):
    def __init__(self, csv_file, vocab):
        self.samples = []
        self.labels = []
        self.vocab = vocab
        
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                description = row["Description"]
                label = int(row["Class Index"]) - 1
                tokenized_sentences = custom_nlp_tokenizer(description)
                tokens = [token for sent in tokenized_sentences for token in sent]
                self.samples.append(tokens)
                self.labels.append(label)
        self.label2idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        indices = [self.vocab.get(token.lower(), self.vocab.get("<unk>")) for token in tokens]
        label_idx = self.label2idx[self.labels[idx]]
        return torch.tensor(indices, dtype=torch.long), label_idx

def collate_fn(batch):
    # Pad sequences in the batch.
    sequences, labels = zip(*batch)
    sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences, labels

# ------------------------
# Training Functions
# ------------------------

def train_classifier(classifier, dataloader, num_epochs=5, lr=0.001, device="cpu"):
    classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=lr)
    classifier.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total_samples = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for sequences, labels in progress_bar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = classifier(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy for the batch.
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_samples * 100
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pretrained ELMo model checkpoint from bilstm.pt.
    checkpoint = torch.load("./models/bilstm.pt", map_location=device)
    vocab = checkpoint["vocab"]
    vocab_size = len(vocab)
    
    # Instantiate the ELMo model with dimensions matching elmo.py and move it to device.
    elmo_model = ELMoModel(vocab_size, embed_size=100, hidden_size=256).to(device)
    elmo_model.load_state_dict(checkpoint["model_state_dict"])
    elmo_model.eval()

    # Assume number of classes based on your dataset (adjust as needed).
    num_classes = 4

    # Create the news dataset and dataloader.
    dataset = NewsDataset("./data/news_classification/train.csv", vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Train and save FrozenLambdasClassifier.
    frozen_classifier = FrozenLambdasClassifier(elmo_model, num_classes,e0_dim=100, e1_dim=512, e2_dim=512)
    print("Training FrozenLambdasClassifier...")
    train_classifier(frozen_classifier, dataloader, num_epochs=15, lr=0.0005, device=device)
    torch.save(frozen_classifier.state_dict(), "frozen_lambdas_classifier.pt")
    print("Saved FrozenLambdasClassifier to frozen_lambdas_classifier.pt")

    # Train and save TrainableLambdasClassifier.
    trainable_classifier = TrainableLambdasClassifier(elmo_model, num_classes,e0_dim=100, e1_dim=512, e2_dim=512)
    print("Training TrainableLambdasClassifier...")
    train_classifier(trainable_classifier, dataloader, num_epochs=15, lr=0.0005, device=device)
    torch.save(trainable_classifier.state_dict(), "trainable_lambdas_classifier.pt")
    print("Saved TrainableLambdasClassifier to trainable_lambdas_classifier.pt")

    # Train and save LearnableFunctionClassifier.
    learnable_classifier = LearnableFunctionClassifier(elmo_model, num_classes,e0_dim=100, e1_dim=512, e2_dim=512)
    print("Training LearnableFunctionClassifier...")
    train_classifier(learnable_classifier, dataloader, num_epochs=15, lr=0.0005, device=device)
    torch.save(learnable_classifier.state_dict(), "learnable_function_classifier.pt")
    print("Saved LearnableFunctionClassifier to learnable_function_classifier.pt")

if __name__ == "__main__":
    main()
