from transformers import LlamaForCausalLM, LlamaTokenizer
import clip
import torch.nn as nn
from torchvision.models import convnext_large
import torch
import os
from tiktoken.load import load_tiktoken_bpe


dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vits14'):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = dinov2_vits14

    def forward(self, x):
        features = self.vit.get_intermediate_features(x)
        vit_features = torch.cat([features['blocks.{}'.format(i)] for i in range(12)], dim=1)
        return vit_features


class CNNFeatureExtractor(nn.Module):
    def __init__(self, model_name='convnext_large'):
        super(CNNFeatureExtractor, self).__init__()
        self.clip_model = convnext_large(pretrained=True)
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        cnn_features = self.clip_model(x)
        return cnn_features


class VisualProjection(nn.Module):
    def __init__(self, vit_dim, cnn_dim, output_dim):
        super(VisualProjection, self).__init__()
        self.proj = nn.Linear(vit_dim + cnn_dim, output_dim)

    def forward(self, vit_feat, cnn_feat):
        combined_feat = torch.cat([vit_feat, cnn_feat], dim=1)
        projected_feat = self.proj(combined_feat)
        return projected_feat


class EarthGPT(nn.Module):
    def __init__(self, visual_proj_dim, llm_local_path):
        super(EarthGPT, self).__init__()
        # 加载分词器并处理特殊token
        self.tokenizer = LlamaTokenizer.from_pretrained(
            llm_local_path,
            legacy=False,
            padding_side='left'  # 根据模型需求设置填充方向
        )

        # 处理缺失的pad token问题
        if self.tokenizer.pad_token is None:
            # 方案1：使用eos_token作为pad_token
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # 方案2：添加新token（需在词表中预留位置）
            # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # self.llm.resize_token_embeddings(len(self.tokenizer))

        # 加载语言模型
        self.llm = LlamaForCausalLM.from_pretrained(
            llm_local_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 解冻指定层（保持原有代码）
        for name, param in self.llm.named_parameters():
            param.requires_grad = 'self_attn' in name or 'rms_norm' in name

        # 投影层（保持原有代码）
        self.vis_proj = nn.Linear(visual_proj_dim, self.llm.config.hidden_size)

    def forward(self, visual_feat, lang_instr):
        # 确保输入设备一致
        device = visual_feat.device

        # 分词处理（显式设置padding参数）
        lang_tokens = self.tokenizer(
            lang_instr,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        ).to(device)

        # 投影视觉特征
        vis_emb = self.vis_proj(visual_feat)

        # 生成输入嵌入（处理不同长度的情况）
        input_emb = torch.cat([
            vis_emb.unsqueeze(1),  # 添加序列维度
            self.llm.model.embed_tokens(lang_tokens.input_ids)
        ], dim=1)

        # 前向传播（添加attention_mask）
        outputs = self.llm(
            inputs_embeds=input_emb,
            attention_mask=torch.cat([
                torch.ones(vis_emb.shape[0], 1).to(device),  # 视觉部分的mask
                lang_tokens.attention_mask
            ], dim=1),
            labels=lang_tokens.input_ids
        )

        return outputs.loss


class BiasTuningLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BiasTuningLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.alpha = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        y = self.linear(x)
        y = self.alpha * (y + self.beta)
        return y


class EarthGPTUnified(EarthGPT):
    def __init__(self, visual_proj_dim, llm_local_path):
        super(EarthGPTUnified, self).__init__(visual_proj_dim, llm_local_path)
        self.llm = self.replace_linear_layers(self.llm)

    def replace_linear_layers(self, module):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, nn.Linear):
                setattr(module, name, BiasTuningLinear(sub_module.in_features, sub_module.out_features, sub_module.bias is not None))
            else:
                self.replace_linear_layers(sub_module)
        return module


# 测试函数
def prepare_dummy_data(batch_size=2, image_size=(224, 224)):
    dummy_images = torch.rand(batch_size, 3, *image_size)
    dummy_lang_instr = ["Describe the scene."] * batch_size
    return dummy_images, dummy_lang_instr


def test_earthgpt_with_local_llama():
    # llm_local_path = "./2/Llama-2-7b"  # 替换为本地模型的路径
    llm_local_path = "./Llama-3.2-1B-Instruct"
    visual_proj_dim = 1024
    model = EarthGPT(visual_proj_dim, llm_local_path)
    dummy_images, dummy_lang_instr = prepare_dummy_data()
    vit_features = torch.rand(2, visual_proj_dim)
    try:
        loss = model(vit_features, dummy_lang_instr)
        print("Loss:", loss.item())
        assert loss.item() > 0, "Loss should be greater than 0!"
        print("EarthGPT with local LLaMA test passed!")
    except Exception as e:
        print("EarthGPT with local LLaMA test failed:", e)


if __name__ == "__main__":
    test_earthgpt_with_local_llama()

