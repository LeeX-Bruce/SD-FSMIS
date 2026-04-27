import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline


model_id = ".cache/huggingface/hub/models--sd-legacy--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"  # TODO 
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# === 增加unet.conv_in_ref
unet_conv_in = pipe.unet.conv_in # Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

ref_conv_in = nn.Conv2d(
    in_channels=8,
    out_channels=unet_conv_in.out_channels, 
    kernel_size=unet_conv_in.kernel_size, 
    stride=unet_conv_in.stride, 
    padding=unet_conv_in.padding, 
    bias=True,
)

with torch.no_grad():
    ref_conv_in.weight = nn.Parameter((unet_conv_in.weight.repeat(1, 2, 1, 1) / 2))
    if unet_conv_in.bias is not None:
        ref_conv_in.bias = nn.Parameter(unet_conv_in.bias.clone())

pipe.unet.conv_in_ref = ref_conv_in

# ==== 改变unet.conv_in的输入通道数 4 -> 8
_weight = unet_conv_in.weight.clone()  # [320, 4, 3, 3]
_bias = unet_conv_in.bias.clone()  # [320]
_weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
# half the activation magnitude
_weight *= 0.5

# new conv_in channel
_new_conv_in = nn.Conv2d(
    8, unet_conv_in.out_channels, kernel_size=unet_conv_in.kernel_size, stride=unet_conv_in.stride,
    padding=unet_conv_in.padding
)
_new_conv_in.weight = nn.Parameter(_weight)
_new_conv_in.bias = nn.Parameter(_bias)
pipe.unet.conv_in = _new_conv_in
# replace config
pipe.unet.config["in_channels"] = 8

pipe.save_pretrained("weight/stable-diffusion-1-5-ref8inchannels-tag8inchannels")