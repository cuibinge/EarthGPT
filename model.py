from transformers import LlamaForCausalLM, LlamaTokenizer
import torch.nn as nn
from torchvision.models import convnext_large
import torch
import os


os.environ["HUGGINGFACE_HUB_TOKEN"] ='自己的huggingface token密钥'
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

#dinov2_vits14=torch.load('./dinov2/dinov2_vits14.pth')
#convnext=torch.load('./convnext/convnext_large.pth')

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vits14'):
        super(ViTFeatureExtractor, self).__init__()
        # self.vit = dinov2_vits14(pretrained=True)
        self.vit = dinov2_vits14


    def forward(self, x):
        features = self.vit.get_intermediate_features(x)
        vit_features = torch.cat([features['blocks.{}'.format(i)] for i in range(12)], dim=1)
        return vit_features


class CNNFeatureExtractor(nn.Module):
    def __init__(self, model_name='convnext_large'):
        super(CNNFeatureExtractor, self).__init__()
        # 加载 ConvNeXt Large 模型
        self.clip_model = convnext_large(pretrained=True)
        # 冻结 CNN 主干的参数
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 直接获取模型的输出特征
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
    def __init__(self, visual_proj_dim, llm_name='meta-llama/Llama-2-7b'):
        super(EarthGPT, self).__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(llm_name)
        self.llm = LlamaForCausalLM.from_pretrained(llm_name)
        # Unfreeze self-attention and RMSNorm layers
        for name, param in self.llm.named_parameters():
            if 'self_attn' in name or 'rms_norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.vis_proj = nn.Linear(visual_proj_dim, self.llm.config.hidden_size)

    def forward(self, visual_feat, lang_instr):
        vis_emb = self.vis_proj(visual_feat)
        lang_tokens = self.tokenizer(lang_instr, return_tensors='pt', padding=True)
        input_emb = torch.cat([vis_emb, self.llm.model.embed_tokens(lang_tokens.input_ids)], dim=1)
        outputs = self.llm(inputs_embeds=input_emb, labels=lang_tokens.input_ids)
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
    def __init__(self, visual_proj_dim, llm_name='meta-llama/Llama-2-7b'):
        super(EarthGPTUnified, self).__init__(visual_proj_dim, llm_name)
        self.llm = self.replace_linear_layers(self.llm)

    def replace_linear_layers(self, module):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, nn.Linear):
                setattr(module, name, BiasTuningLinear(sub_module.in_features, sub_module.out_features, sub_module.bias is not None))
            else:
                self.replace_linear_layers(sub_module)
        return module

