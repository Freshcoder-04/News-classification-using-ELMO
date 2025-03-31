import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import torch.nn as nn
import torch.optim as optim

from tokenizer import custom_nlp_tokenizer

# ------------------------
# Static Embedding Classifier Definition
# ------------------------

class StaticEmbeddingClassifier(nn.Module):
    """
    A classifier that uses a pretrained static embedding matrix (e.g. from CBOW, SkipGram, or SVD)
    as its embedding layer. The embedding size is expected to match (e.g., 100 dimensions).
    An LSTM is applied on the looked-up embeddings, and the final hidden state is used for classification.
    """
    def __init__(self, embedding_matrix, num_classes, rnn_hidden_size=128, rnn_num_layers=1, bidirectional=True):
        super(StaticEmbeddingClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Set the pretrained embeddings and freeze them.
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float), requires_grad=False)
        
        self.rnn = nn.LSTM(embedding_dim, rnn_hidden_size, num_layers=rnn_num_layers,
                           batch_first=True, bidirectional=bidirectional)
        final_dim = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.classifier = nn.Linear(final_dim, num_classes)
    
    def forward(self, x):
        embeds = self.embedding(x)  # (batch, seq_len, embedding_dim)
        rnn_out, (h_n, _) = self.rnn(embeds)
        if self.rnn.bidirectional:
            final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final = h_n[-1]
        logits = self.classifier(final)
        return logits

# Alias for clarity.
SVDEmbeddingClassifier = StaticEmbeddingClassifier
CBOWEmbeddingClassifier = StaticEmbeddingClassifier
SkipGramEmbeddingClassifier = StaticEmbeddingClassifier

# ------------------------
# News Dataset Definition (same as before)
# ------------------------

class NewsDataset(nn.Module, Dataset):
    """
    Assumes a CSV file with columns "Description" and "Class Index".
    The description is tokenized with the custom tokenizer.
    """
    def __init__(self, csv_file, vocab):
        self.samples = []
        self.labels = []
        self.vocab = vocab
        
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                description = row["Description"]
                label = int(row["Class Index"]) - 1
                # Tokenize description (flatten sentences into one sequence)
                tokenized_sentences = custom_nlp_tokenizer(description)
                tokens = [token for sent in tokenized_sentences for token in sent]
                self.samples.append(tokens)
                self.labels.append(label)
        # Create mapping for labels.
        self.label2idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # Look up token indices using the vocabulary specific to the embedding type.
        indices = [self.vocab.get(token.lower(), self.vocab.get("<unk>")) for token in tokens]
        label_idx = self.label2idx[self.labels[idx]]
        return torch.tensor(indices, dtype=torch.long), label_idx

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences, labels

# ------------------------
# Training Function
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
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_samples * 100
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# ------------------------
# Main Function: Train and Save Each Classifier with Its Corresponding Vocabulary
# ------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    news_csv = "./data/news_classification/train.csv"
    num_classes = 4

    # ------------------ CBOW Classifier ------------------
    cbow_data = torch.load("./models/static-embeddings/cbow.pt", map_location=device)
    cbow_embeddings = cbow_data["embeddings"]  # (vocab_size, 100)
    cbow_vocab = cbow_data["word_to_id"]
    print("Training CBOWEmbeddingClassifier...")
    dataset_cbow = NewsDataset(news_csv, cbow_vocab)
    dataloader_cbow = DataLoader(dataset_cbow, batch_size=32, shuffle=True, collate_fn=collate_fn)
    cbow_classifier = CBOWEmbeddingClassifier(cbow_embeddings, num_classes)
    train_classifier(cbow_classifier, dataloader_cbow, num_epochs=15, lr=0.0005, device=device)
    torch.save(cbow_classifier.state_dict(), "cbow_classifier.pt")
    print("Saved CBOWEmbeddingClassifier to cbow_classifier.pt")
    
    # ------------------ SkipGram Classifier ------------------
    skipgram_data = torch.load("./models/static-embeddings/skipgram.pt", map_location=device)
    skipgram_embeddings = skipgram_data["embeddings"]
    skipgram_vocab = skipgram_data["word_to_id"]
    print("Training SkipGramEmbeddingClassifier...")
    dataset_skipgram = NewsDataset(news_csv, skipgram_vocab)
    dataloader_skipgram = DataLoader(dataset_skipgram, batch_size=32, shuffle=True, collate_fn=collate_fn)
    skipgram_classifier = SkipGramEmbeddingClassifier(skipgram_embeddings, num_classes)
    train_classifier(skipgram_classifier, dataloader_skipgram, num_epochs=15, lr=0.0005, device=device)
    torch.save(skipgram_classifier.state_dict(), "skipgram_classifier.pt")
    print("Saved SkipGramEmbeddingClassifier to skipgram_classifier.pt")
    
    # ------------------ SVD Classifier ------------------
    svd_data = torch.load("./models/static-embeddings/svd.pt", map_location=device)
    svd_embeddings = svd_data["embeddings"]
    svd_vocab = svd_data["word_to_id"]
    print("Training SVDEmbeddingClassifier...")
    dataset_svd = NewsDataset(news_csv, svd_vocab)
    dataloader_svd = DataLoader(dataset_svd, batch_size=32, shuffle=True, collate_fn=collate_fn)
    svd_classifier = SVDEmbeddingClassifier(svd_embeddings, num_classes)
    train_classifier(svd_classifier, dataloader_svd, num_epochs=15, lr=0.0005, device=device)
    torch.save(svd_classifier.state_dict(), "svd_classifier.pt")
    print("Saved SVDEmbeddingClassifier to svd_classifier.pt")
    
if __name__ == "__main__":
    main()
