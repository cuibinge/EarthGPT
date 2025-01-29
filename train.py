import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from model import VisualProjection,CNNFeatureExtractor,ViTFeatureExtractor,EarthGPTUnified

class ImageTextDataset(Dataset):
    def __init__(self, image_dir, text_dir, transform=None):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.text_files = sorted(os.listdir(text_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 加载文本
        text_path = os.path.join(self.text_dir, self.text_files[idx])
        with open(text_path, "r") as f:
            text = f.read().strip()

        return image, text

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
dataset = ImageTextDataset(
    image_dir="./dataset/images",
    text_dir="./dataset/texts",
    transform=transform
)

# 划分训练集和验证集
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = EarthGPTUnified(visual_proj_dim=512, llm_name="huggyllama/llama-7b").to(device)
#加载本地llama模型用于训练
#model=EarthGPTUnified(visual_proj_dim=512, llm_local_path="./Llama-3.2-1B-Instruct").to(device)#换为自己的路径

# 初始化优化器
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 训练函数
def train(model, train_loader, val_loader, optimizer, epochs=10):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        # 训练阶段
        for images, texts in train_loader:
            images = images.to(device)

            # 提取视觉特征
            vit_feat = ViTFeatureExtractor()(images)
            cnn_feat = CNNFeatureExtractor()(images)
            visual_feat = VisualProjection(vit_feat.size(1), cnn_feat.size(1), 512)(vit_feat, cnn_feat)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            loss = model(visual_feat, texts)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader)

        # 记录历史
        history["train_loss"].append(epoch_train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc * 100:.2f}%")

    return history

# 验证函数
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(device)

            # 提取视觉特征
            vit_feat = ViTFeatureExtractor()(images)
            cnn_feat = CNNFeatureExtractor()(images)
            visual_feat = VisualProjection(vit_feat.size(1), cnn_feat.size(1), 512)(vit_feat, cnn_feat)

            # 计算损失
            loss = model(visual_feat, texts)
            total_loss += loss.item()

            # 生成预测
            vis_emb = model.vis_proj(visual_feat)
            outputs = model.llm.generate(
                inputs_embeds=vis_emb,
                max_length=20,
                num_return_sequences=1
            )

            # 解码并计算准确率
            preds = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
            correct += sum([1 if pred.strip() == tgt.strip() else 0 for pred, tgt in zip(preds, texts)])
            total += len(texts)

    return total_loss / len(val_loader), correct / total

# 可视化函数
def visualize_history(history):
    plt.figure(figsize=(12, 4))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# 开始训练
history = train(model, train_loader, val_loader, optimizer, epochs=10)

# 可视化训练结果
visualize_history(history)