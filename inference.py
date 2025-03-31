import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Import your modules (adjust the import paths as necessary)
from tokenizer import custom_nlp_tokenizer
from classification import (FrozenLambdasClassifier, TrainableLambdasClassifier, LearnableFunctionClassifier, NewsDataset, collate_fn)
from static_classification import SVDEmbeddingClassifier, CBOWEmbeddingClassifier, SkipGramEmbeddingClassifier
from ELMO import ELMoModel

def load_elmo_model(bilstm_path, device):
    checkpoint = torch.load(bilstm_path, map_location=device)
    elmo_vocab = checkpoint["vocab"]
    vocab_size = len(elmo_vocab)
    model = ELMoModel(vocab_size, embed_size=100, hidden_size=256).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, elmo_vocab

def tokenize_description(description, vocab):
    tokens = [token for sent in custom_nlp_tokenizer(description) for token in sent]
    # Use <unk> if token not found.
    indices = [vocab.get(token.lower(), vocab.get("<unk>")) for token in tokens]
    return torch.tensor(indices, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Run inference on a description using a saved classifier.")
    parser.add_argument("model_path", type=str, help="Path to the saved model checkpoint")
    parser.add_argument("description", type=str, help="Description text to classify")
    args = parser.parse_args()

    test = pd.read_csv("./data/news_classification/test.csv")
    print(test['Class Index'].unique())
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model_path.lower()
    description = args.description

    num_classes = 4  # Adjust if needed

    # Determine model type based on the model_path string.
    # If the model_path string contains any of "frozen", "trainable", "learnable", we assume it is ELMo-based.
    if any(keyword in model_path for keyword in ["frozen", "trainable", "learnable"]):
        # Load the pretrained ELMo model.
        elmo_model, vocab = load_elmo_model("./models/bilstm.pt", device)
        # Instantiate the appropriate classifier.
        if "frozen" in model_path:
            classifier = FrozenLambdasClassifier(elmo_model, num_classes, e0_dim=100, e1_dim=512, e2_dim=512)
        elif "trainable" in model_path:
            classifier = TrainableLambdasClassifier(elmo_model, num_classes, e0_dim=100, e1_dim=512, e2_dim=512)
        elif "learnable" in model_path:
            classifier = LearnableFunctionClassifier(elmo_model, num_classes, e0_dim=100, e1_dim=512, e2_dim=512)
        else:
            sys.exit("Unknown ELMo classifier type.")
        classifier.to(device)
        classifier.load_state_dict(torch.load(args.model_path, map_location=device))
        classifier.eval()
    else:
        # Static embedding classifiers: determine type based on keywords.
        if "cbow" in model_path:
            static_type = "cbow"
            classifier_class = CBOWEmbeddingClassifier
            static_checkpoint = torch.load("./models/static-embeddings/cbow.pt", map_location=device)
        elif "skipgram" in model_path:
            static_type = "skipgram"
            classifier_class = SkipGramEmbeddingClassifier
            static_checkpoint = torch.load("./models/static-embeddings/skipgram.pt", map_location=device)
        elif "svd" in model_path:
            static_type = "svd"
            classifier_class = SVDEmbeddingClassifier
            static_checkpoint = torch.load("./models/static-embeddings/svd.pt", map_location=device)
        else:
            sys.exit("Unknown static embedding classifier type.")
        embedding_matrix = static_checkpoint["embeddings"]
        vocab = static_checkpoint["word_to_id"]
        classifier = classifier_class(embedding_matrix, num_classes)
        classifier.to(device)
        classifier.load_state_dict(torch.load(args.model_path, map_location=device))
        classifier.eval()

    # Tokenize the input description.
    input_indices = tokenize_description(description, vocab)
    # Add batch dimension.
    input_tensor = input_indices.unsqueeze(0).to(device)

    # Run inference.
    with torch.no_grad():
        logits = classifier(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    # Output probabilities for each class.
    for i, prob in enumerate(probabilities, start=1):
        print(f"class-{i} {prob:.2f}")

if __name__ == "__main__":
    main()
