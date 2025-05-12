import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from modelscope import AutoModelForSequenceClassification
from tqdm import tqdm
import torch

# --------------------------
# 路径配置（保持不变）
# --------------------------
TRAIN_PATH = r"C:/Users/DELL/Desktop/数据集素材/rhetoric_train.json"
VAL_PATH = r"C:/Users/DELL/Desktop/数据集素材/rhetoric_val.json"
OUTPUT_DIR = r"C:/Users/DELL/Desktop/实验数据"

# --------------------------
# 数据加载与预处理（保持原结构）
# --------------------------
class RhetoricDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取整个文件内容
            file_content = f.read()
            
            # 检查是否是有效的JSON数组格式
            if file_content.strip().startswith('['):
                # 如果是数组格式，直接解析
                self.data = json.loads(file_content)
            else:
                # 如果是每行一个JSON对象，需要手动处理
                lines = [line.strip() for line in file_content.split('\n') if line.strip()]
                self.data = [json.loads(line) for line in lines]
        
        # 验证数据
        if not self.data:
            raise ValueError("数据集为空或格式不正确")
        if not all(['text' in item and 'label_main' in item for item in self.data]):
            raise ValueError("数据缺少必要字段（text或label_main）")
            
        # 主标签编码
        self.main_labels = sorted(list(set([d['label_main'] for d in self.data])))
        self.main_encoder = LabelEncoder().fit(self.main_labels)
        
    
        # 主-子标签映射（示例需补充完整）
        self.sub_label_map = {
             # 有子类的主标签
             "譬喻": ["明喻", "隐喻", "借喻"],
             "借代": ["旁借", "对代"],
             "映衬": ["反映", "对衬"],
             "摹状": ["摹声", "摹状其他"],
             "双关": ["表里双关", "彼此双关"],
             "引用": ["明引", "暗用"],
             "仿拟": ["拟句", "仿调"],
             "比拟": ["物拟人", "人拟物"],
             "讽喻": ["比方", "寓言"],
             "示现": ["追述", "预言", "悬想"],
             "夸张": ["普通夸张辞", "超前夸张辞"],
             "倒反": ["倒辞", "反语"],
             "婉转": ["不说本事型婉转", "隐约提示型婉转"],
             "避讳": ["公用避讳", "独用避讳"],
             "设问": ["提示下文型设问", "激发本意式设问"],
             "感叹": ["感叹词式感叹", "设问式感叹", "倒装式感叹"],
             "析字": ["化形式析字", "谐音式析字", "衍义式析字"],
             "飞白": ["记录式飞白", "援用式飞白"],
             "镶嵌": ["镶字", "嵌字"],
             "复叠": ["复辞", "叠字"],
             "节缩": ["缩合", "节短"],
             "顶真": ["连珠格", "连环体"],
             "跳脱": ["急收式跳脱", "突接式跳脱", "岔断式跳脱"],
    
             # 无子类的主标签（需保留空列表）
             "拈连": [],
             "移就": [],
             "藏词": [],
             "省略": [],
             "警策": [],
             "折绕": [],
             "转品": [],
             "回文": [],
             "反复": [],
             "对偶": [],
             "排比": [],
             "层递": [],
             "错综": [],
             "倒装": []
        }
        
        
        # 新增分词器
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
           "C:/iicnlp_roberta_backbone_lite_std"
           )



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "text": item["text"],
            "label_main": self.main_encoder.transform([item["label_main"]])[0],
            "label_sub": item["label_sub"]
        }

# --------------------------
# 评估模块
# --------------------------
class RhetoricEvaluator:
    def __init__(self, main_labels, sub_label_map):
        self.main_labels = main_labels
        self.sub_label_map = sub_label_map
        self.num_classes = len(main_labels)
        
        # 初始化存储结构
        self.confusion_matrices = {label: np.zeros((2,2)) for label in main_labels}
        self.metrics_history = []

    def _validate_sub(self, pred_sub, main_label):
        """验证子标签是否合法"""
        return pred_sub in self.sub_label_map.get(main_label, [])

    def update(self, y_true_main, y_pred_main, y_true_sub, y_pred_sub):
        """更新38个混淆矩阵"""
        for i in range(self.num_classes):
            main_label = self.main_labels[i]
            
            # 计算每个样本是否TP
            tp_mask = (y_true_main == i) & (y_pred_main == i) & \
                      [self._validate_sub(p_sub, main_label) and (t_sub == p_sub) 
                       for t_sub, p_sub in zip(y_true_sub, y_pred_sub)]
            
            fp_mask = (y_true_main != i) & (y_pred_main == i)
            fn_mask = (y_true_main == i) & (~tp_mask)
            tn_mask = (y_true_main != i) & (y_pred_main != i)
            
            self.confusion_matrices[main_label] += np.array([
                [tn_mask.sum(), fp_mask.sum()],
                [fn_mask.sum(), tp_mask.sum()]
            ])

    def get_metrics(self):
        """计算所有指标"""
        metrics = {}
        for label, cm in self.confusion_matrices.items():
            tn, fp = cm[0]
            fn, tp = cm[1]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn
            }
        return metrics

# --------------------------
# 训练模块（关键修改部分）
# --------------------------
class RhetoricTrainer:
    def __init__(self):
        # 初始化目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "confusion_matrix"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "pr_curves"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "tensorboard"), exist_ok=True)
        
        # 加载数据集（保持原有结构）
        self.train_dataset = RhetoricDataset(TRAIN_PATH)
        self.val_dataset = RhetoricDataset(VAL_PATH)
        
        # 初始化评估器
        self.evaluator = RhetoricEvaluator(
            self.train_dataset.main_labels,
            self.train_dataset.sub_label_map
        )
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            "C:/iicnlp_roberta_backbone_lite_std",
            revision='v1.0.0'
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "C:/iicnlp_roberta_backbone_lite_std",
            num_labels=len(self.train_dataset.main_labels),
            revision='v1.0.0'
        ).to(self.device)
        
        # 准备数据加载器
        self.train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_dataset, shuffle=False)

    def _create_dataloader(self, dataset, shuffle=False):
        labels = torch.as_tensor([item["label_main"] for item in dataset.data])
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        return DataLoader(
    dataset,
        batch_size=self.args.train_batch_size,
        sampler=sampler,
        collate_fn=self.data_collator,
        drop_last=self.args.dataloader_drop_last,
    )
    
class CustomDataset(Dataset):
    def __init__(self, encodings, labels_main, labels_sub):
        self.encodings = encodings
        self.labels_main = labels_main
        self.labels_sub = labels_sub

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label_main': self.labels_main[idx],
            'label_sub': self.labels_sub[idx]
        }

    def __len__(self):
        return len(self.labels_main)

# 将创建DataLoader的逻辑包裹在函数中
    def create_dataloader(dataset, encodings, shuffle=False):
        labels_main = torch.tensor([item["label_main"] for item in dataset.data])
        labels_sub = [item["label_sub"] for item in dataset.data]
    
        return DataLoader(
            CustomDataset(encodings, labels_main, labels_sub),
            batch_size=8,
            shuffle=shuffle
    )

# 使用示例
# dataloader = create_dataloader(your_dataset, your_encodings, shuffle=True)
    def train(self):
        """完全自定义的训练流程"""
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=len(self.train_loader)*12
        )
        
        for epoch in range(1, 13):
            print(f"\n{'='*40}")
            print(f"Epoch {epoch}/12")
            
            # 训练阶段
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(self.train_loader, desc="Training")
            
            for batch in progress_bar:
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'labels': batch['label_main'].to(self.device)
                }
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 评估阶段
            metrics = self._evaluate_epoch(epoch)
            
            # 打印摘要
            self._print_epoch_summary(epoch_loss/len(self.train_loader), metrics, epoch)
            
            # 保存模型
            torch.save(self.model.state_dict(), 
                      os.path.join(OUTPUT_DIR, f"model_epoch{epoch}.pt"))

    def _evaluate_epoch(self, epoch):
        """评估实现（保持原有评估逻辑）"""
        self.model.eval()
        all_preds_main = []
        all_labels_main = []
        all_labels_sub = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                outputs = self.model(**inputs)
                
                preds_main = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds_main.extend(preds_main)
                all_labels_main.extend(batch['label_main'].numpy())
                all_labels_sub.extend(batch['label_sub'])
        
        # 转换标签
        y_true_main = self.val_dataset.main_encoder.transform(all_labels_main)
        
        # 更新评估器（子标签预测暂用空字符串）
        self.evaluator.update(
            y_true_main,
            np.array(all_preds_main),
            all_labels_sub,
            [""]*len(all_preds_main)  # 子标签预测占位符
        )
        metrics = self.evaluator.get_metrics()
        
        # 保存报告和可视化（保持原有实现）
        with open(os.path.join(OUTPUT_DIR, f"epoch_{epoch}_report.txt"), "w", encoding='utf-8') as f:
            for label, vals in metrics.items():
                f.write(f"[{label}]\n")
                f.write(f"Precision: {vals['precision']:.3f} | Recall: {vals['recall']:.3f} | F1: {vals['f1']:.3f}\n\n")
        
        self._log_to_tensorboard(epoch, metrics)
        self._plot_confusion_matrices(epoch)
        self._plot_pr_curves(epoch)
        
        return metrics

    def _print_epoch_summary(self, metrics, epoch):
        """打印epoch摘要"""
        train_loss = getattr(self.trainer, 'train_outputs', {}).get('loss', 'N/A')
        
        macro_precision = np.mean([m['precision'] for m in metrics.values()])
        macro_recall = np.mean([m['recall'] for m in metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in metrics.values()])
        
        worst_performers = sorted(metrics.items(), key=lambda x: x[1]['f1'])[:3]
        
        print(f"\n[Epoch {epoch} Summary]")
        print(f"Train Loss: {train_loss if isinstance(train_loss, str) else f'{train_loss:.4f}'}")
        print("Eval Metrics:")
        print(f" - Macro Precision: {macro_precision:.4f}")
        print(f" - Macro Recall: {macro_recall:.4f}")
        print(f" - Macro F1: {macro_f1:.4f}")
        print("\nWorst Performing Classes:")
        for label, perf in worst_performers:
            print(f" - {label}: P={perf['precision']:.2f} R={perf['recall']:.2f} F1={perf['f1']:.2f}")

   

    def _log_to_tensorboard(self, epoch, metrics):
        """TensorBoard记录"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'tensorboard'))
            
            # 宏观指标
            writer.add_scalar('macro/precision', np.mean([m['precision'] for m in metrics.values()]), epoch)
            writer.add_scalar('macro/recall', np.mean([m['recall'] for m in metrics.values()]), epoch)
            writer.add_scalar('macro/f1', np.mean([m['f1'] for m in metrics.values()]), epoch)
            
            # 每个类别指标
            for label, vals in metrics.items():
                writer.add_scalar(f'precision/{label}', vals['precision'], epoch)
                writer.add_scalar(f'recall/{label}', vals['recall'], epoch)
                writer.add_scalar(f'f1/{label}', vals['f1'], epoch)
            
            writer.close()
        except ImportError:
            print("TensorBoard未安装，跳过日志记录")

    def _plot_confusion_matrices(self, epoch):
        """绘制混淆矩阵"""
        for label, cm in self.evaluator.confusion_matrices.items():
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {label} (Epoch {epoch})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix", f"epoch{epoch}_{label}.png"))
            plt.close()

    def _plot_pr_curves(self, epoch):
        """绘制PR曲线"""
        metrics = self.evaluator.get_metrics()
        for label, vals in metrics.items():
            plt.figure()
            plt.plot([0, 1], [vals['precision'], vals['recall']], marker='o')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR Curve - {label} (Epoch {epoch})")
            plt.savefig(os.path.join(OUTPUT_DIR, "pr_curves", f"epoch{epoch}_{label}.png"))
            plt.close()


if __name__ == "__main__":
    trainer = RhetoricTrainer()
    trainer.train()