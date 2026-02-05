import torch
import torch.nn as nn
import os
import datetime
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from models.LAMNet import LAMNet
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

plt.rcParams['axes.unicode_minus'] = False

# data augmentation (optional)
transform = transforms.Compose([
    # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),    
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = './datasets/Split_data'
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

full_dataset = ConcatDataset([train_dataset, val_dataset])

class_names = train_dataset.classes
new_order = ['High', 'Medium', 'Low']
new_order_indices = [class_names.index(cls) for cls in new_order]

k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_results = []
all_test_labels = []
all_test_preds = []

current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
main_log_dir = os.path.join('./runs', f"KFOLD_{current_time}")
os.makedirs(main_log_dir, exist_ok=True)

all_folds_train_loss = []
all_folds_val_loss = []
all_folds_train_acc = []
all_folds_val_acc = []


for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
    print(f"FOLD {fold + 1}")
    print("--------------------------------")
    
    # create fold log directory
    fold_log_dir = os.path.join(main_log_dir, f"fold_{fold+1}")
    os.makedirs(fold_log_dir, exist_ok=True)
    writer = SummaryWriter(fold_log_dir)
    
    # create model checkpoint save path
    model_dir = os.path.join(fold_log_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # create metrics save path
    metric_dir = os.path.join(fold_log_dir, 'metrics')
    os.makedirs(metric_dir, exist_ok=True)
    
    # create train and validation data subsets
    train_subsampler = Subset(full_dataset, train_ids)
    val_subsampler = Subset(full_dataset, val_ids)
    
    # create train and validation data loaders
    batch_size = 40
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # LAMNet init
    model = LAMNet(num_classes=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params}")
    
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # optional
    
    # visualize model structure
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    writer.add_graph(model, dummy_input)
    
    # train and validation
    num_epochs = 50
    best_val_accuracy = 0.0
    
    # initialize metrics record
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accs = []
    epoch_val_accs = []
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        total_train = 0
        all_labels = []
        all_preds = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_train += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        
        train_loss /= total_train
        train_accuracy = accuracy_score(all_labels, all_preds)
        
        # record training metrics
        epoch_train_losses.append(train_loss)
        epoch_train_accs.append(train_accuracy)
        
        model.eval()
        val_loss = 0.0
        total_val = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total_val += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        val_loss /= total_val
        val_accuracy = accuracy_score(all_labels, all_preds)
        
        # record validation metrics
        epoch_val_losses.append(val_loss)
        epoch_val_accs.append(val_accuracy)
        
        # update TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_fold{fold+1}.pth'))
        
        # update lr, optional
        # scheduler.step()
    
    writer.close()
    
    # =============== plot and save training curves ===============
    plt.figure(figsize=(12, 10))
    
    # Loss
    plt.subplot(2, 1, 1)
    plt.plot(epoch_train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epoch_val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.title(f'Fold {fold+1} - Training and Validation Loss Curves', fontsize=14)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epoch_train_accs, label='Training Accuracy', color='green', linewidth=2)
    plt.plot(epoch_val_accs, label='Validation Accuracy', color='orange', linewidth=2)
    plt.title(f'Fold {fold+1} - Training and Validation Accuracy Curves', fontsize=14)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(metric_dir, f'training_curves_fold{fold+1}.png'), dpi=300)
    plt.close()
    
    # save training metrics to CSV
    training_metrics = pd.DataFrame({
        'Epoch': range(1, num_epochs+1),
        'Train_Loss': epoch_train_losses,
        'Val_Loss': epoch_val_losses,
        'Train_Acc': epoch_train_accs,
        'Val_Acc': epoch_val_accs
    })
    training_metrics.to_csv(os.path.join(metric_dir, f'training_metrics_fold{fold+1}.csv'), index=False)
    
    # store metrics for overall comparison
    all_folds_train_loss.append(epoch_train_losses)
    all_folds_val_loss.append(epoch_val_losses)
    all_folds_train_acc.append(epoch_train_accs)
    all_folds_val_acc.append(epoch_val_accs)
    
    # load best model for testing
    model.load_state_dict(torch.load(os.path.join(model_dir, f'best_model_fold{fold+1}.pth')))
    model.eval()
    
    fold_labels = []
    fold_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            fold_labels.extend(labels.cpu().numpy())
            fold_preds.extend(preds.cpu().numpy())
    
    # calculate test metrics
    test_accuracy = accuracy_score(fold_labels, fold_preds)
    # test_precision = precision_score(fold_labels, fold_preds, average='weighted')
    # test_recall = recall_score(fold_labels, fold_preds, average='weighted')
    # test_f1 = f1_score(fold_labels, fold_preds, average='weighted')
    test_precision = precision_score(fold_labels, fold_preds, average='macro')
    test_recall = recall_score(fold_labels, fold_preds, average='macro')
    test_f1 = f1_score(fold_labels, fold_preds, average='macro')
    
    print(f'Fold {fold+1} Test Results:')
    print(f'Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')
    
    # save results for each fold
    fold_results.append({
        'Fold': fold+1,
        'Accuracy': test_accuracy,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1': test_f1,
        'Parameters': total_params
    })
    
    # save confusion matrix for each fold
    cm = confusion_matrix(fold_labels, fold_preds)
    cm = cm[new_order_indices, :]
    cm = cm[:, new_order_indices]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=new_order, yticklabels=new_order)
    plt.title(f'Confusion Matrix - Fold {fold+1}', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join(metric_dir, f'confusion_matrix_fold{fold+1}.png'), dpi=300)
    plt.close()
    
    # collect all predictions for overall evaluation
    all_test_labels.extend(fold_labels)
    all_test_preds.extend(fold_preds)

# =============== plot and save comparison plots ===============
# create comparison plot directory
comparison_dir = os.path.join(main_log_dir, 'comparison_plots')
os.makedirs(comparison_dir, exist_ok=True)

# calculate mean training curves
mean_train_loss = np.mean(all_folds_train_loss, axis=0)
mean_val_loss = np.mean(all_folds_val_loss, axis=0)
mean_train_acc = np.mean(all_folds_train_acc, axis=0)
mean_val_acc = np.mean(all_folds_val_acc, axis=0)

# plot loss comparison
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
for i, (train_loss, val_loss) in enumerate(zip(all_folds_train_loss, all_folds_val_loss)):
    plt.plot(train_loss, alpha=0.5, label=f'Fold {i+1} Train' if i == 0 else "")
    plt.plot(val_loss, alpha=0.5, label=f'Fold {i+1} Val' if i == 0 else "")

plt.plot(mean_train_loss, 'b-', linewidth=3, label='Mean Train Loss')
plt.plot(mean_val_loss, 'r-', linewidth=3, label='Mean Val Loss')
plt.title('5-Fold Cross-Validation Loss Comparison', fontsize=14)
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# plot accuracy comparison
plt.subplot(1, 2, 2)
for i, (train_acc, val_acc) in enumerate(zip(all_folds_train_acc, all_folds_val_acc)):
    plt.plot(train_acc, alpha=0.5, label=f'Fold {i+1} Train' if i == 0 else "")
    plt.plot(val_acc, alpha=0.5, label=f'Fold {i+1} Val' if i == 0 else "")

plt.plot(mean_train_acc, 'g-', linewidth=3, label='Mean Train Accuracy')
plt.plot(mean_val_acc, 'orange', linewidth=3, label='Mean Val Accuracy')
plt.title('5-Fold Cross-Validation Accuracy Comparison', fontsize=14)
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(comparison_dir, 'all_folds_training_comparison.png'), dpi=300)
plt.close()

# save all folds training metrics
all_folds_metrics = pd.DataFrame({
    'Epoch': np.tile(range(1, num_epochs+1), k_folds),
    'Fold': np.repeat(range(1, k_folds+1), num_epochs),
    'Train_Loss': np.concatenate(all_folds_train_loss),
    'Val_Loss': np.concatenate(all_folds_val_loss),
    'Train_Acc': np.concatenate(all_folds_train_acc),
    'Val_Acc': np.concatenate(all_folds_val_acc)
})
all_folds_metrics.to_csv(os.path.join(main_log_dir, 'all_folds_training_metrics.csv'), index=False)

# save all folds results to CSV
results_df = pd.DataFrame(fold_results)
results_df.to_csv(os.path.join(main_log_dir, 'fold_results.csv'), index=False)

# calculate mean metrics
mean_accuracy = results_df['Accuracy'].mean()
std_accuracy = results_df['Accuracy'].std()
mean_precision = results_df['Precision'].mean()
std_precision = results_df['Precision'].std()
mean_recall = results_df['Recall'].mean()
std_recall = results_df['Recall'].std()
mean_f1 = results_df['F1'].mean()
std_f1 = results_df['F1'].std()

# save overall results
summary_results = {
    'Mean Accuracy': mean_accuracy,
    'Std Accuracy': std_accuracy,
    'Mean Precision': mean_precision,
    'Std Precision': std_precision,
    'Mean Recall': mean_recall,
    'Std Recall': std_recall,
    'Mean F1': mean_f1,
    'Std F1': std_f1
}
summary_df = pd.DataFrame([summary_results])
summary_df.to_csv(os.path.join(main_log_dir, 'summary_results.csv'), index=False)

print("\nCross-Validation Summary:")
print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")
print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

# save overall confusion matrix
cm = confusion_matrix(all_test_labels, all_test_preds)
cm = cm[new_order_indices, :]
cm = cm[:, new_order_indices]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=new_order, yticklabels=new_order)
plt.title('Overall Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.savefig(os.path.join(main_log_dir, 'overall_confusion_matrix.png'), dpi=300)
plt.close()

# normalize confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=new_order, yticklabels=new_order)
plt.title('Normalized Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.savefig(os.path.join(main_log_dir, 'normalized_confusion_matrix.png'), dpi=300)
plt.close()

# plot performance metrics comparison bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
means = [mean_accuracy, mean_precision, mean_recall, mean_f1]
stds = [std_accuracy, std_precision, std_recall, std_f1]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, means, yerr=stds, capsize=10, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Performance Metrics Comparison', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# add numerical labels on top of bars
for bar, std in zip(bars, stds):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(main_log_dir, 'performance_metrics.png'), dpi=300)
plt.close()

print("Training completed! All results have been saved to:", main_log_dir)
