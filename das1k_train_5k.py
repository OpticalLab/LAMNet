import torch
import torch.nn as nn
import datetime
from torch.utils.data import Dataset, Subset
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from torchsummary import summary
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from models.DualLAMNet import Dual_LAMNet

plt.rcParams['font.sans-serif'] = ['AR PL UKai CN']
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class DualSoundDataset(Dataset):
    def __init__(self, intensity_root, phase_root, transform=None):
        self.intensity_root = intensity_root
        self.phase_root = phase_root
        self.transform = transform
        
        # 获取类别和文件列表
        self.classes = sorted(os.listdir(intensity_root))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 构建样本列表
        self.samples = []
        for cls_name in self.classes:
            cls_dir_intensity = os.path.join(intensity_root, cls_name)
            cls_dir_phase = os.path.join(phase_root, cls_name)
            
            for file in os.listdir(cls_dir_intensity):
                if file.endswith('.png'):
                    intensity_path = os.path.join(cls_dir_intensity, file)
                    phase_path = os.path.join(cls_dir_phase, file)
                    if os.path.exists(phase_path):
                        self.samples.append((intensity_path, phase_path, self.class_to_idx[cls_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        intensity_path, phase_path, label = self.samples[idx]
        
        intensity_img = Image.open(intensity_path).convert('RGB')
        phase_img = Image.open(phase_path).convert('RGB')
        
        if self.transform:
            intensity_img = self.transform(intensity_img)
            phase_img = self.transform(phase_img)
        
        return intensity_img, phase_img, label

if __name__ == '__main__':



    # 数据转换
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),  
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    dataset = DualSoundDataset(
        intensity_root="./data/DAS1K_data_augmentation_melspec/intensity",
        phase_root="./data/DAS1K_data_augmentation_melspec/phase",
        transform=transform
    )

    new_order = sorted(dataset.classes)  
    new_order_indices = [dataset.class_to_idx[cls_name] for cls_name in new_order]  

    # 创建日志目录
    current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
    log_dir = os.path.join(r'./runs', f"{current_time}")
    os.makedirs(log_dir, exist_ok=True)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 设置5折交叉验证
    num_epochs = 40
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)

    fold_results = []
    all_test_labels = []
    all_test_preds = []

    # 存储所有折的训练指标用于对比图
    all_folds_train_loss = []
    all_folds_val_loss = []
    all_folds_train_acc = []
    all_folds_val_acc = []

    # 存储所有折的结果
    all_fold_results = []

    # 开始交叉验证
    for fold, (train_ids, val_ids) in enumerate(kf.split(dataset)):
        print(f'FOLD {fold+1}/{k_folds}')
        print('-' * 50)
        
        # # 创建当前折的模型目录
        # fold_model_dir = os.path.join(model_dir, f'fold_{fold+1}')
        # os.makedirs(fold_model_dir, exist_ok=True)

            # 创建每折的日志目录
        fold_log_dir = os.path.join(log_dir, f"fold_{fold+1}")
        os.makedirs(fold_log_dir, exist_ok=True)
        writer = SummaryWriter(fold_log_dir)
        
        # 创建模型保存路径
        model_dir = os.path.join(fold_log_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建metrics保存路径
        metric_dir = os.path.join(fold_log_dir, 'metrics')
        os.makedirs(metric_dir, exist_ok=True)
        
        
        # 创建训练集和验证集
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=8, sampler=train_subsampler, num_workers=12)
        val_loader = DataLoader(dataset, batch_size=4, sampler=val_subsampler, num_workers=12)
        
        # 初始化模型
        model = Dual_LAMNet(num_classes=10).to(device)
        
        if fold == 0:
            model_summary = summary(model, [(3, 224, 224), (3, 224, 224)], device=str(device))
            total_params = sum(p.numel() for p in model.parameters())
            print("model params: ", total_params)
        
        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # # 可视化模型结构
        # dummy_input =  [(3, 224, 224), (3, 224, 224)]
        # writer.add_graph(model, dummy_input)
        
        # 训练和验证
        best_val_accuracy = 0.0

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
            
            for intensity_inputs, phase_inputs, labels in train_loader:
                intensity_inputs = intensity_inputs.to(device)
                phase_inputs = phase_inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                outputs = model(intensity_inputs, phase_inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * intensity_inputs.size(0)
                total_train += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            
            train_loss /= total_train
            train_accuracy = accuracy_score(all_labels, all_preds)

            # 记录训练指标
            epoch_train_losses.append(train_loss)
            epoch_train_accs.append(train_accuracy)
            
            model.eval()
            val_loss = 0.0
            total_val = 0
            all_labels = []
            all_preds = []
            
            with torch.no_grad():
                for intensity_inputs, phase_inputs, labels in val_loader:
                    intensity_inputs = intensity_inputs.to(device)
                    phase_inputs = phase_inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(intensity_inputs, phase_inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * intensity_inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
            
            val_loss /= total_val
            val_accuracy = accuracy_score(all_labels, all_preds)
            
            # 记录验证指标
            epoch_val_losses.append(val_loss)
            epoch_val_accs.append(val_accuracy)
            
            # 更新TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            
            print(f'Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            
            # 保存最好的模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_fold{fold+1}.pth'))
        writer.close()
        
        # =============== 绘制并保存每折的训练曲线 ===============
        plt.figure(figsize=(12, 10))
        
        # Loss曲线
        plt.subplot(2, 1, 1)
        plt.plot(epoch_train_losses, label='训练损失', color='blue', linewidth=2)
        plt.plot(epoch_val_losses, label='验证损失', color='red', linewidth=2)
        plt.title(f'Fold {fold+1} - 训练和验证损失曲线', fontsize=14)
        plt.xlabel('训练轮次', fontsize=12)
        plt.ylabel('损失值', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Accuracy曲线
        plt.subplot(2, 1, 2)
        plt.plot(epoch_train_accs, label='训练准确率', color='green', linewidth=2)
        plt.plot(epoch_val_accs, label='验证准确率', color='orange', linewidth=2)
        plt.title(f'Fold {fold+1} - 训练和验证准确率曲线', fontsize=14)
        plt.xlabel('训练轮次', fontsize=12)
        plt.ylabel('准确率', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(metric_dir, f'training_curves_fold{fold+1}.png'), dpi=300)
        plt.close()
        
        # 保存训练指标到CSV
        training_metrics = pd.DataFrame({
            'Epoch': range(1, num_epochs+1),
            'Train_Loss': epoch_train_losses,
            'Val_Loss': epoch_val_losses,
            'Train_Acc': epoch_train_accs,
            'Val_Acc': epoch_val_accs
        })
        training_metrics.to_csv(os.path.join(metric_dir, f'training_metrics_fold{fold+1}.csv'), index=False)
        
        # 存储当前折的指标用于整体对比
        all_folds_train_loss.append(epoch_train_losses)
        all_folds_val_loss.append(epoch_val_losses)
        all_folds_train_acc.append(epoch_train_accs)
        all_folds_val_acc.append(epoch_val_accs)
        
        # 加载最佳模型进行测试
        model.load_state_dict(torch.load(os.path.join(model_dir, f'best_model_fold{fold+1}.pth')))
        model.eval()
        
        # 在验证集上测试最佳模型
        # total_test = 0
        # all_test_labels = []
        # all_test_preds = []

        fold_labels = []
        fold_preds = []
        
        with torch.no_grad():
            for intensity_inputs, phase_inputs, labels in val_loader:
                intensity_inputs = intensity_inputs.to(device)
                phase_inputs = phase_inputs.to(device)
                labels = labels.to(device)
                outputs = model(intensity_inputs, phase_inputs)
                _, preds = torch.max(outputs, 1)
                # total_test += labels.size(0)
                # all_test_labels.extend(labels.cpu().numpy())
                # all_test_preds.extend(preds.cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())
                fold_preds.extend(preds.cpu().numpy())
        
        # test_accuracy = accuracy_score(all_test_labels, all_test_preds)
        # test_precision = precision_score(all_test_labels, all_test_preds, average='macro')
        # test_recall = recall_score(all_test_labels, all_test_preds, average='macro')
        # test_f1 = f1_score(all_test_labels, all_test_preds, average='macro')
        test_accuracy = accuracy_score(fold_labels, fold_preds)
        test_precision = precision_score(fold_labels, fold_preds, average='macro')
        test_recall = recall_score(fold_labels, fold_preds, average='macro')
        test_f1 = f1_score(fold_labels, fold_preds, average='macro')
        
        # print(f'Fold {fold+1} Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}')
        
        print(f'Fold {fold+1} Test Results:')
        print(f'Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')


        # 保存当前折的结果
        # 保存每折结果
        fold_results.append({
            'Fold': fold+1,
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1': test_f1,
            'Parameters': total_params
        })
        
        # 绘制混淆矩阵
        # 绘制混淆矩阵
        # new_order = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']
        # new_order_indices = [dataset.class_to_idx[cls_name] for cls_name in new_order]  # Changed from train_subsampler to dataset
        cm = confusion_matrix(fold_labels, fold_preds)
        cm = cm[new_order_indices, :]  # 重新排序行
        cm = cm[:, new_order_indices]  # 重新排序列
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        
        
        
        # 绘制混淆矩阵图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=new_order, yticklabels=new_order)
        plt.title(f'Fold {fold+1} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(metric_dir, f'confusion_matrix_fold_{fold+1}.png'))
        plt.close()
        
        # 绘制归一化混淆矩阵图
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=new_order, yticklabels=new_order)
        plt.title(f'Fold {fold+1} Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(metric_dir, f'normalized_confusion_matrix_fold_{fold+1}.png'))
        plt.close()
        all_test_labels.extend(fold_labels)
        all_test_preds.extend(fold_preds)


    # =============== 绘制所有折的指标对比图 ===============
    # 创建对比图目录
    comparison_dir = os.path.join(log_dir, 'comparison_plots')
    os.makedirs(comparison_dir, exist_ok=True)

    # 计算平均训练曲线
    mean_train_loss = np.mean(all_folds_train_loss, axis=0)
    mean_val_loss = np.mean(all_folds_val_loss, axis=0)
    mean_train_acc = np.mean(all_folds_train_acc, axis=0)
    mean_val_acc = np.mean(all_folds_val_acc, axis=0)

    # 损失曲线对比图
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for i, (train_loss, val_loss) in enumerate(zip(all_folds_train_loss, all_folds_val_loss)):
        plt.plot(train_loss, alpha=0.5, label=f'Fold {i+1} Train' if i == 0 else "")
        plt.plot(val_loss, alpha=0.5, label=f'Fold {i+1} Val' if i == 0 else "")

    plt.plot(mean_train_loss, 'b-', linewidth=3, label='平均训练损失')
    plt.plot(mean_val_loss, 'r-', linewidth=3, label='平均验证损失')
    plt.title('5折交叉验证损失曲线对比', fontsize=14)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 准确率曲线对比图
    plt.subplot(1, 2, 2)
    for i, (train_acc, val_acc) in enumerate(zip(all_folds_train_acc, all_folds_val_acc)):
        plt.plot(train_acc, alpha=0.5, label=f'Fold {i+1} Train' if i == 0 else "")
        plt.plot(val_acc, alpha=0.5, label=f'Fold {i+1} Val' if i == 0 else "")

    plt.plot(mean_train_acc, 'g-', linewidth=3, label='平均训练准确率')
    plt.plot(mean_val_acc, 'orange', linewidth=3, label='平均验证准确率')
    plt.title('5折交叉验证准确率曲线对比', fontsize=14)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'all_folds_training_comparison.png'), dpi=300)
    plt.close()

    # 保存所有折的训练指标
    all_folds_metrics = pd.DataFrame({
        'Epoch': np.tile(range(1, num_epochs+1), k_folds),
        'Fold': np.repeat(range(1, k_folds+1), num_epochs),
        'Train_Loss': np.concatenate(all_folds_train_loss),
        'Val_Loss': np.concatenate(all_folds_val_loss),
        'Train_Acc': np.concatenate(all_folds_train_acc),
        'Val_Acc': np.concatenate(all_folds_val_acc)
    })
    all_folds_metrics.to_csv(os.path.join(log_dir, 'all_folds_training_metrics.csv'), index=False)

    # 保存所有折的结果到CSV
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(log_dir, 'fold_results.csv'), index=False)

    # 计算平均指标
    mean_accuracy = results_df['Accuracy'].mean()
    std_accuracy = results_df['Accuracy'].std()
    mean_precision = results_df['Precision'].mean()
    std_precision = results_df['Precision'].std()
    mean_recall = results_df['Recall'].mean()
    std_recall = results_df['Recall'].std()
    mean_f1 = results_df['F1'].mean()
    std_f1 = results_df['F1'].std()

    # 保存整体结果
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
    summary_df.to_csv(os.path.join(log_dir, 'summary_results.csv'), index=False)

    print("\nCross-Validation Summary:")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # 保存整体混淆矩阵
    cm = confusion_matrix(all_test_labels, all_test_preds)
    cm = cm[new_order_indices, :]
    cm = cm[:, new_order_indices]
    # 获取类别名称
    # class_names = list(dataset.class_to_idx.keys())
    # class_indices = list(dataset.class_to_idx.values())
    # sorted_indices = np.argsort(class_indices)
    # sorted_class_names = [class_names[i] for i in sorted_indices]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=new_order, yticklabels=new_order)
    plt.title('整体混淆矩阵', fontsize=16)
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    plt.savefig(os.path.join(log_dir, 'overall_confusion_matrix.png'), dpi=300)
    plt.close()

    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=new_order, yticklabels=new_order)
    plt.title('归一化混淆矩阵', fontsize=16)
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    plt.savefig(os.path.join(log_dir, 'normalized_confusion_matrix.png'), dpi=300)
    plt.close()

    # 绘制指标对比柱状图
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    means = [mean_accuracy, mean_precision, mean_recall, mean_f1]
    stds = [std_accuracy, std_precision, std_recall, std_f1]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, means, yerr=stds, capsize=10, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('模型性能指标对比', fontsize=16)
    plt.ylabel('分数', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱子上方添加数值标签
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'performance_metrics.png'), dpi=300)
    plt.close()

    print("训练完成! 所有结果已保存至:", log_dir)