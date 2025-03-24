import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json

class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['req'].tolist()
        self.labels = dataframe.iloc[:, 1:].values
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[index])
        }

def train_and_evaluate():
    full_train_df = pd.read_excel('path/to/refined_core_classes.xlsx')
    test_df = pd.read_excel('path/to/test_data.xlsx')

    train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)

    MAX_LEN = 512
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 3e-5
    PATIENCE = 3  

    tokenizer = BertTokenizer.from_pretrained(r'path/to/bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained(
        r'path/to/bert-base-chinese',
        num_labels=18,
        problem_type="multi_label_classification"
    )

    train_dataset = MultiLabelDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = MultiLabelDataset(val_df, tokenizer, MAX_LEN)
    test_dataset = MultiLabelDataset(test_df, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()
            train_losses.append(loss.item())
            print(f"Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Train Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            print("Validation loss improved, saving model state.")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state for evaluation.")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss (per batch)')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss (per epoch)', marker='o')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('Validation Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(r'path/to/loss_curve.png')
    plt.show()

    model.eval()
    all_predictions = []
    all_labels = []
    all_reqs = test_df['req'].tolist()  
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()
            labels = labels.cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)
            print(f"Testing Batch {i+1}/{len(test_loader)} completed")

    all_predictions = np.array(all_predictions)
    all_predictions_binary = (all_predictions > 0.5).astype(int)
    all_labels = np.array(all_labels)

    predicted_df = pd.DataFrame({
        'req': all_reqs  
    })
    predicted_df = pd.concat([predicted_df, pd.DataFrame(all_predictions_binary, columns=test_df.columns[1:])], axis=1)

    predicted_df.to_excel(r'path/to/predicted_labels.xlsx', index=False)

    json_data = []
    label_names = test_df.columns[1:].tolist()  
    for i in range(len(all_reqs)):
        above_threshold_indices = np.where(all_predictions[i] > 0.5)[0]
        above_threshold_probs = all_predictions[i][above_threshold_indices]
        sorted_indices = above_threshold_indices[np.argsort(above_threshold_probs)[::-1]]
        pred_classes = [label_names[j] for j in sorted_indices]
        json_data.append({
            "req": all_reqs[i],
            "class": pred_classes 
        })

    with open(r'path/to/predicted_labels.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    train_and_evaluate()