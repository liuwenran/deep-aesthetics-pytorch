import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List
from tqdm import tqdm

# ------------------ 1. 加载 CLIP 模型 ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "/fs-computility/llm/shared/liuwenran/hf-models/clip-vit-large-patch14"
CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(device)
PROCESSOR = CLIPProcessor.from_pretrained(model_name)

# ------------------ 2. 为单个维度定义数据集 ------------------
class SingleDimensionDataset(Dataset):
    def __init__(self, image_dir: str, label_file: str):
        """
        Args:
            image_dir (str): 图片所在的根目录。
            label_file (str): 单个维度的标注文件路径。
        """
        self.image_dir = image_dir
        self.data = []
        self._load_data(label_file)
    
    def _load_data(self, label_file: str):
        """从单个标注文件加载数据"""
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                img_name, score_str = parts
                score = float(score_str)
                
                # 将分数从 [-1, 1] 映射到 [0, 10]
                normalized_score = ((score - (-1)) / (1 - (-1))) * 10  # (score + 1) / 2 * 10
                
                img_path = os.path.join(self.image_dir, img_name)
                if os.path.exists(img_path):
                    self.data.append((img_path, normalized_score))
                else:
                    print(f"Warning: Image {img_name} not found in directory {self.image_dir}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, score = self.data[idx]
        
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            pil_image = Image.new('RGB', (224, 224))

        # 获取处理后的张量
        image_inputs = PROCESSOR(images=pil_image, return_tensors="pt", padding=True)
        # 移除 batch 维度
        image_inputs = {k: v.squeeze(0) for k, v in image_inputs.items()}
        
        return image_inputs, torch.tensor(score, dtype=torch.float32)

# ------------------ 3. MLP 模型定义 ------------------
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)  # 输出单个维度的分数
        )

    def forward(self, x):
        return self.layers(x)

# ------------------ 4. 训练函数 ------------------
def train_mlp_for_dimension(train_dataset, val_dataset, dimension_name: str, device, input_size=768, epochs=50):
    """
    为单个维度训练一个 MLP 模型。
    现在接收的是完整的 Dataset 对象。
    """
    model = MLP(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 为每个维度创建独立的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Training {dimension_name}", leave=False)
        for batch in progress_bar:
            images = batch[0]['pixel_values'].squeeze(1)
            labels = batch[1].unsqueeze(1)  # shape: (batch_size, 1)
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                features = CLIP_MODEL.get_image_features(images)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 更新进度条的描述信息（可选）
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # 验证阶段
        avg_train_loss = train_loss / len(train_loader)
        print(f'avg_train_loss {avg_train_loss}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0]['pixel_values'].squeeze(1).to(device)
                labels = batch[1].unsqueeze(1).to(device)
                
                features = CLIP_MODEL.get_image_features(images)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Dimension: {dimension_name}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"mlp_model_{dimension_name}.pth")
            print(f"  Best model saved for {dimension_name}")
    
    return model

# ------------------ 5. 主程序 ------------------
if __name__ == "__main__":
    # 配置
    IMAGE_DIR = "/fs-computility/llm/liuwenran/datasets/datasetImages_originalSize"
    LABEL_DIR = "/fs-computility/llm/liuwenran/datasets/imgListFiles_label"
    DIMENSION_NAMES = [
        'BalacingElements', 'ColorHarmony', 'Content', 'DoF',
        'Light', 'MotionBlur', 'Object', 'Repetition',
        'RuleOfThirds', 'Symmetry', 'VividColor', 'score',
    ]

    import ipdb;ipdb.set_trace();

    for dim_name in DIMENSION_NAMES:
        print(f"\n{'='*50}")
        print(f"Processing dimension: {dim_name}")
        print(f"{'='*50}")

        # 为当前维度构建训练和验证数据集
        train_label_file = os.path.join(LABEL_DIR, f"imgListTrainRegression_{dim_name}.txt")
        val_label_file = os.path.join(LABEL_DIR, f"imgListTestRegression_{dim_name}.txt")

        train_dataset = SingleDimensionDataset(image_dir=IMAGE_DIR, label_file=train_label_file)
        val_dataset = SingleDimensionDataset(image_dir=IMAGE_DIR, label_file=val_label_file)

        print(f"  Dimension {dim_name}: {len(train_dataset)} training images, {len(val_dataset)} validation images")

        # 跳过空的数据集
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"  Skipping {dim_name} due to insufficient data.")
            continue

        # 训练该维度的 MLP
        trained_model = train_mlp_for_dimension(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dimension_name=dim_name,
            device=device,
            input_size=768,
            epochs=10
        ) 

    print("All dimensions processed.")