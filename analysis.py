import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from tokenizer import custom_nlp_tokenizer
from classification import NewsDataset, collate_fn, FrozenLambdasClassifier, TrainableLambdasClassifier, LearnableFunctionClassifier
from static_classification import SVDEmbeddingClassifier, CBOWEmbeddingClassifier, SkipGramEmbeddingClassifier
from ELMO import ELMoModel


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            sequences = sequences.to(device)
            labels = labels.to(device)
            logits = model(sequences)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    conf = confusion_matrix(all_labels, all_preds)
    return acc, recall, f1, conf

# ------------------------
# Main Function: Load all 6 classifiers, evaluate on train and test data.
# ------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 4
    train_csv = "./data/news_classification/train.csv"
    test_csv = "./data/news_classification/test.csv"
    
    # -------------------- ELMo-based classifiers --------------------
    # Load the pretrained ELMo model from bilstm.pt.
    bilstm_checkpoint = torch.load("./models/bilstm.pt", map_location=device)
    elmo_vocab = bilstm_checkpoint["vocab"]
    vocab_size = len(elmo_vocab)
    elmo_model = ELMoModel(vocab_size, embed_size=100, hidden_size=256).to(device)
    elmo_model.load_state_dict(bilstm_checkpoint["model_state_dict"])
    elmo_model.eval()
    
    # Create NewsDataset instances for ELMo using elmo_vocab.
    train_dataset_elmo = NewsDataset(train_csv, elmo_vocab)
    test_dataset_elmo = NewsDataset(test_csv, elmo_vocab)
    train_loader_elmo = DataLoader(train_dataset_elmo, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader_elmo = DataLoader(test_dataset_elmo, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Load each ELMo classifier.
    frozen_model = FrozenLambdasClassifier(elmo_model, num_classes, e0_dim=100, e1_dim=512, e2_dim=512)
    frozen_model.load_state_dict(torch.load("./models/classifiers/elmo-classifiers/frozen_lambdas_classifier.pt", map_location=device))
    frozen_model.to(device)
    
    trainable_model = TrainableLambdasClassifier(elmo_model, num_classes, e0_dim=100, e1_dim=512, e2_dim=512)
    trainable_model.load_state_dict(torch.load("./models/classifiers/elmo-classifiers/trainable_lambdas_classifier.pt", map_location=device))
    trainable_model.to(device)
    
    learnable_model = LearnableFunctionClassifier(elmo_model, num_classes, e0_dim=100, e1_dim=512, e2_dim=512)
    learnable_model.load_state_dict(torch.load("./models/classifiers/elmo-classifiers/learnable_function_classifier.pt", map_location=device))
    learnable_model.to(device)
    
    # -------------------- Static embedding classifiers --------------------
    # CBOW classifier:
    cbow_data = torch.load("./models/static-embeddings/cbow.pt", map_location=device)
    cbow_embeddings = cbow_data["embeddings"]
    cbow_vocab = cbow_data["word_to_id"]
    train_dataset_cbow = NewsDataset(train_csv, cbow_vocab)
    test_dataset_cbow = NewsDataset(test_csv, cbow_vocab)
    train_loader_cbow = DataLoader(train_dataset_cbow, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader_cbow = DataLoader(test_dataset_cbow, batch_size=32, shuffle=False, collate_fn=collate_fn)
    cbow_model = CBOWEmbeddingClassifier(cbow_embeddings, num_classes)
    cbow_model.load_state_dict(torch.load("./models/classifiers/static-classifiers/cbow_classifier.pt", map_location=device))
    cbow_model.to(device)
    
    # SkipGram classifier:
    skipgram_data = torch.load("./models/static-embeddings/skipgram.pt", map_location=device)
    skipgram_embeddings = skipgram_data["embeddings"]
    skipgram_vocab = skipgram_data["word_to_id"]
    train_dataset_skipgram = NewsDataset(train_csv, skipgram_vocab)
    test_dataset_skipgram = NewsDataset(test_csv, skipgram_vocab)
    train_loader_skipgram = DataLoader(train_dataset_skipgram, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader_skipgram = DataLoader(test_dataset_skipgram, batch_size=32, shuffle=False, collate_fn=collate_fn)
    skipgram_model = SkipGramEmbeddingClassifier(skipgram_embeddings, num_classes)
    skipgram_model.load_state_dict(torch.load("./models/classifiers/static-classifiers/skipgram_classifier.pt", map_location=device))
    skipgram_model.to(device)
    
    # SVD classifier:
    svd_data = torch.load("./models/static-embeddings/svd.pt", map_location=device)
    svd_embeddings = svd_data["embeddings"]
    svd_vocab = svd_data["word_to_id"]
    train_dataset_svd = NewsDataset(train_csv, svd_vocab)
    test_dataset_svd = NewsDataset(test_csv, svd_vocab)
    train_loader_svd = DataLoader(train_dataset_svd, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader_svd = DataLoader(test_dataset_svd, batch_size=32, shuffle=False, collate_fn=collate_fn)
    svd_model = SVDEmbeddingClassifier(svd_embeddings, num_classes)
    svd_model.load_state_dict(torch.load("./models/classifiers/static-classifiers/svd_classifier.pt", map_location=device))
    svd_model.to(device)
    
    # Organize the six models and their corresponding dataloaders.
    models = {
        "ELMo Frozen": frozen_model,
        "ELMo Trainable": trainable_model,
        "ELMo Learnable": learnable_model,
        "CBOW": cbow_model,
        "SkipGram": skipgram_model,
        "SVD": svd_model,
    }
    
    train_loaders = {
        "ELMo Frozen": train_loader_elmo,
        "ELMo Trainable": train_loader_elmo,
        "ELMo Learnable": train_loader_elmo,
        "CBOW": train_loader_cbow,
        "SkipGram": train_loader_skipgram,
        "SVD": train_loader_svd,
    }
    test_loaders = {
        "ELMo Frozen": test_loader_elmo,
        "ELMo Trainable": test_loader_elmo,
        "ELMo Learnable": test_loader_elmo,
        "CBOW": test_loader_cbow,
        "SkipGram": test_loader_skipgram,
        "SVD": test_loader_svd,
    }
    
    results = {}
    for model_name, model in models.items():
        print(f"Evaluating {model_name} on TRAIN data...")
        acc_train, recall_train, f1_train, conf_train = evaluate_model(model, train_loaders[model_name], device)
        print(f"{model_name} TRAIN: Accuracy={acc_train:.2f}, Recall={recall_train:.2f}, F1={f1_train:.2f}")
        print("TRAIN Confusion Matrix:")
        print(conf_train)
        
        print(f"Evaluating {model_name} on TEST data...")
        acc_test, recall_test, f1_test, conf_test = evaluate_model(model, test_loaders[model_name], device)
        print(f"{model_name} TEST: Accuracy={acc_test:.2f}, Recall={recall_test:.2f}, F1={f1_test:.2f}")
        print("TEST Confusion Matrix:")
        print(conf_test)
        
        results[model_name] = {
            "train": {"accuracy": acc_train, "recall": recall_train, "f1": f1_train, "confusion": conf_train},
            "test": {"accuracy": acc_test, "recall": recall_test, "f1": f1_test, "confusion": conf_test},
        }
    
    # Save evaluation results.
    torch.save(results, "evaluation_results.pt")
    
if __name__ == "__main__":
    main()
