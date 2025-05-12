import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.font_manager as fm

try:
    # Windows系统字体
    font_path = "C:/Windows/Fonts/simhei.ttf"
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = font_prop.get_name()
except:
    # Linux/Mac系统备用方案
    try:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 思源黑体
    except:
        # 动态查找可用中文字体
        cjk_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name]
        if cjk_fonts:
            plt.rcParams['font.sans-serif'] = cjk_fonts[0]
            
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

 

# 配置路径
MODEL_PATH = "C:/iicnlp_roberta_backbone_lite_std"
TRAIN_PATH = r"C:/Users/DELL/Desktop/数据集素材/rhetoric_train.json"
VAL_PATH = r"C:/Users/DELL/Desktop/数据集素材/rhetoric_val.json"
OUTPUT_DIR = r"C:/Users/DELL/Desktop/实验数据"
# 在 OUTPUT_DIR 下按学习率创建子目录
def create_lr_dir(lr):
    lr_dir = os.path.join(OUTPUT_DIR, f"lr_{str(lr).replace('.', '_')}")
    os.makedirs(lr_dir, exist_ok=True)
    return lr_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)

# 标签编码器类
class ChineseLabelEncoder:
    def __init__(self):
        self.classes_ = None
        self._mapping = {}
        self._reverse_mapping = {}
    
    def fit(self, labels):
        self.classes_ = sorted(list(set(labels)))
        self._mapping = {label: idx for idx, label in enumerate(self.classes_)}
        self._reverse_mapping = {idx: label for label, idx in self._mapping.items()}
        return self
    
    def transform(self, labels):
        return np.array([self._mapping[label] for label in labels])
    
    def inverse_transform(self, indices):
        return [self._reverse_mapping[idx] for idx in indices]

# 数据集类
class RhetoricDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read().strip()
            self.data = json.loads(data) if data.startswith('[') else [json.loads(line) for line in data.split('\n') if line.strip()]
        
        # 初始化标签编码器
        self.main_encoder = ChineseLabelEncoder()
        self.main_encoder.fit([item["label_main"] for item in self.data])
        
        # 分词处理
        self.tokenizer = tokenizer
        self.encodings = tokenizer(
            [item["text"] for item in self.data],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(
                self.main_encoder.transform([self.data[idx]["label_main"]])[0],
                dtype=torch.long
            ),
            "original_labels": {
                "main": self.data[idx]["label_main"],
                "sub": self.data[idx]["label_sub"]
            }
        }

class RhetoricTrainer:
    def create_lr_dir(self, lr):
        """创建学习率专属目录"""
        lr_str = str(lr).replace('.', '_')
        lr_dir = os.path.join(OUTPUT_DIR, f"lr_{lr_str}")
        os.makedirs(lr_dir, exist_ok=True)
        return lr_dir
    
    def __init__(self, learning_rate=2e-5):
         # 初始化顺序优化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        
        # 先初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # 再加载数据集（确保tokenizer已初始化）
        self.train_set = RhetoricDataset(TRAIN_PATH, self.tokenizer)
        self.val_set = RhetoricDataset(VAL_PATH, self.tokenizer)
        
        # 初始化模型（使用数据集确定的类别数）
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=len(self.train_set.main_encoder.classes_)
        ).to(self.device)
        
       
        
        # 数据加载
        self.train_set = RhetoricDataset(TRAIN_PATH, self.tokenizer)
        self.val_set = RhetoricDataset(VAL_PATH, self.tokenizer)
        
        # 数据加载器
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=8,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=8,
            collate_fn=self._collate_fn
        )
        
        self.label_encoder = self.train_set.main_encoder
        self.optimizer = None  # 将在train方法中初始化

  
    def _collate_fn(self, batch):
            return {
                "input_ids": torch.stack([item["input_ids"] for item in batch]),
                 "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
                 "labels": torch.stack([item["labels"] for item in batch]),
                 "original_labels": [item["original_labels"] for item in batch]
            }
    
    def train(self, epochs=20, patience=3):
        logging.info(f"Starting training with learning rate: {self.learning_rate}")

        # 初始化优化器和调度器
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=len(self.train_loader)*epochs
        )

        best_f1 = 0
        no_improve_epochs = 0
        training_stats = []
        
        for epoch in range(1, epochs+1):
            logging.info(f"Epoch {epoch}/{epochs}")
            
            print(f"\nEpoch {epoch}/{epochs}")
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc="Training")
            for batch in progress_bar:
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "labels": batch["labels"].to(self.device)
                }
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                if torch.isnan(loss).any():
                    raise ValueError("NaN loss detected")
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 验证评估
            val_metrics = self.evaluate(epoch)
            training_stats.append({
                'epoch': epoch,
                'train_loss': total_loss/len(self.train_loader),
                **val_metrics
            })
            
            # 早停机制
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                no_improve_epochs = 0
                # 保存最佳模型
                self.save_checkpoint(epoch, total_loss/len(self.train_loader), self.learning_rate)
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        # 保存完整训练统计
        self.save_training_stats(training_stats)
        



    def evaluate(self, epoch):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_original = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device)
                }
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].cpu().numpy())
                all_original.extend(batch["original_labels"])

        # 转换回汉字标签
        pred_labels = self.label_encoder.inverse_transform(all_preds)
        true_labels = self.label_encoder.inverse_transform(all_labels)
        
        # 统一使用数字标签计算
        true_indices = self.label_encoder.transform(true_labels)
        pred_indices = self.label_encoder.transform(pred_labels)

        # 计算指标时转换为Python原生类型
        accuracy = float((np.array(true_indices) == np.array(pred_indices)).mean())
        f1 = float(f1_score(true_indices, pred_indices, average='macro'))
        precision = float(precision_score(true_indices, pred_indices, average='macro'))
        recall = float(recall_score(true_indices, pred_indices, average='macro'))
    
         # 处理分类报告中的numpy类型
        class_report = classification_report(
              true_indices, pred_indices,
             target_names=self.label_encoder.classes_,
             output_dict=True,
             zero_division=0  # 防止除以零错误
         )
    
         # 递归转换numpy类型为Python类型
        def convert_numpy_types(obj):
             if isinstance(obj, np.generic):
                 return obj.item()
             elif isinstance(obj, dict):
                 return {k: convert_numpy_types(v) for k, v in obj.items()}
             elif isinstance(obj, list):
                 return [convert_numpy_types(x) for x in obj]
             else:
                 return obj
    
        converted_report = convert_numpy_types(class_report)
    
         # 混淆矩阵转换为列表
        cm = confusion_matrix(true_indices, pred_indices).tolist()

        metrics = {
             'accuracy': accuracy,
             'f1': f1,
             'precision': precision,
             'recall': recall,
             'class_report': converted_report,
             'confusion_matrix': cm
         }
    
         # 保存混淆矩阵图片
        self.plot_confusion_matrix(cm, epoch, self.learning_rate)
        logging.info(f"Validation Metrics - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}， Macro Precision: {metrics['precision']:.4f}，Macro Recall: {metrics['recall']:.4f}")

        # 打印结果
        print(f"\nValidation Metrics:")
        print(f" - Accuracy: {metrics['accuracy']:.4f}")
        print(f" - Macro F1: {metrics['f1']:.4f}")
        print(f" - Macro Precision: {metrics['precision']:.4f}")
        print(f" - Macro Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def save_training_stats(self, stats):
    # 递归转换所有numpy类型为Python类型
         def convert_numpy(obj):
             if isinstance(obj, np.generic):
                 return obj.item()
             elif isinstance(obj, list):
                 return [convert_numpy(x) for x in obj]
             elif isinstance(obj, dict):
                 return {k: convert_numpy(v) for k, v in obj.items()}
             else:
                 return obj
    
         converted_stats = convert_numpy(stats)
    
         # 保存学习曲线
         self.plot_learning_curves(converted_stats)
         
    
         # 保存详细指标
         with open(os.path.join(OUTPUT_DIR, 'training_stats.json'), 'w') as f:
             json.dump(converted_stats, f, indent=2, ensure_ascii=False)  # 添加ensure_ascii=False支持中文
    
    def save_checkpoint(self, epoch, avg_loss, lr):
        
         # 创建学习率专属目录
        lr_dir = create_lr_dir(lr)

        checkpoint = {
            'epoch': epoch,
            'learning_rate': lr,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': avg_loss,
            'label_encoder': self.label_encoder
        }

        # 保存路径
        save_path = os.path.join(lr_dir, f'checkpoint_epoch{epoch}.pt')
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")  # 显示完整路径

    def plot_confusion_matrix(self, cm, epoch, lr):
        # 创建学习率专属目录
        lr_dir = create_lr_dir(lr)
    

        plt.figure(figsize=(15, 12))  # 根据类别数量调整
    
        # 动态调整字体大小
        num_classes = len(self.label_encoder.classes_)
        font_scale = 1.0 - num_classes*0.003  # 每增加一个类别缩小0.3%
    
        sns.set(font_scale=font_scale)
        ax = sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            cbar_kws={"shrink": 0.75},
            annot_kws={"size": 10},  # 调整标注字号
            linewidths=0.5  # 添加细线分隔
        )
    
        # 优化标签显示
        ax.set_xticklabels(
            self.label_encoder.classes_,
            rotation=45,
            ha='right',
            fontproperties=font_prop  # 使用自定义字体属性
        )
        ax.set_yticklabels(
            self.label_encoder.classes_,
            fontproperties=font_prop,
            rotation=0
        )
    
        plt.title(f'Epoch {epoch} 混淆矩阵', fontproperties=font_prop)
        plt.tight_layout()  # 自动调整布局

        # 修改保存路径
        save_path = os.path.join(lr_dir, f'confusion_matrix_epoch{epoch}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved confusion matrix to {save_path}")


    def plot_learning_curves(self, stats):
        plt.figure(figsize=(12, 8), dpi=120)
    
        # 设置全局字体
        plt.rcParams.update({'font.family': font_prop.get_name()})
    
        # 绘制训练损失
        plt.plot([s['epoch'] for s in stats], 
                [s['train_loss'] for s in stats],
                'r--', 
                label='训练损失')
    
        # 绘制验证指标
        plt.plot([s['epoch'] for s in stats], 
                [s['f1'] for s in stats],
                'b-', 
                label='验证F1')
    
        # 设置中文标签
        plt.xlabel('训练轮次', fontproperties=font_prop)
        plt.ylabel('指标值', fontproperties=font_prop)
        plt.title('训练过程监控', fontproperties=font_prop)
    
        # 解决图例中文显示
        plt.legend(prop=font_prop)
    
        # 保存图像
        save_path = os.path.join(OUTPUT_DIR, 'learning_curves.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    learning_rates = [1e-5, 3e-5, 5e-5]
    best_f1 = 0
    best_lr = None
 
    for lr in learning_rates:
        print(f"\n{'='*40}")
        print(f"Starting training with lr={lr}")
        logging.info(f"Starting training with lr={lr}")
        trainer = RhetoricTrainer(learning_rate=lr)
        trainer.train(epochs=20)
 
        # 加载训练统计，检查最佳 F1
        stats_path = os.path.join(OUTPUT_DIR, 'training_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                last_epoch_f1 = stats[-1]['f1']  # 假设最后一个 epoch 的 F1 是最佳的
                print(f"Final F1 for lr={lr}: {last_epoch_f1}")
                logging.info(f"Final F1 for lr={lr}: {last_epoch_f1}")
                if last_epoch_f1 > best_f1:
                    best_f1 = last_epoch_f1
                    best_lr = lr
 
    print(f"\nBest learning rate: {best_lr} with F1: {best_f1}")
    logging.info(f"Best learning rate: {best_lr} with F1: {best_f1}")