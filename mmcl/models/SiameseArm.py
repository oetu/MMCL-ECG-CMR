import torch

from pl_bolts.utils.self_supervised import torchvision_ssl_encoder

from pl_bolts.models.self_supervised.byol.models import MLP, SiameseArm
import pl_bolts.models.self_supervised.byol.models

import models.ECGEncoder as ECGEncoder

class SiameseArm(pl_bolts.models.self_supervised.byol.models.SiameseArm):
    def __init__(self, encoder="resnet50", encoder_out_dim=2048, projector_hidden_size=4096, projector_out_dim=256, **kwargs):
        super(SiameseArm, self).__init__()

        print(f"KWARGS: {kwargs}")
        
        self.encoder_name = encoder

        if isinstance(encoder, str):
            if "vit":
                if kwargs["attention_pool"]:
                    kwargs["global_pool"] = "attention_pool"
                encoder = ECGEncoder.__dict__[kwargs["ecg_model"]](
                    img_size=kwargs["input_size"],
                    patch_size=kwargs["patch_size"],
                    num_classes=kwargs["num_classes"],
                    drop_path_rate=kwargs["drop_path"],
                    global_pool=kwargs["global_pool"])
                encoder.blocks[-1].attn.forward = self._attention_forward_wrapper(encoder.blocks[-1].attn) # required to read out the attention map of the last layer
                encoder_out_dim = encoder.embed_dim
            else:
                encoder = torchvision_ssl_encoder(encoder)
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(encoder_out_dim, projector_hidden_size, projector_out_dim)
        # Predictor
        self.predictor = MLP(projector_out_dim, projector_hidden_size, projector_out_dim)

    def _attention_forward_wrapper(self, attn_obj):
        """
        Modified version of def forward() of class Attention() in timm.models.vision_transformer
        """
        def my_forward(x):
            B, N, C = x.shape # C = embed_dim
            # (3, B, Heads, N, head_dim)
            qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            # (B, Heads, N, N)
            attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
            attn = attn.softmax(dim=-1)
            attn = attn_obj.attn_drop(attn)
            # (B, Heads, N, N)
            attn_obj.attn_map = attn # this was added 

            # (B, N, Heads*head_dim)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = attn_obj.proj(x)
            x = attn_obj.proj_drop(x)
            return x
        return my_forward

    def forward(self, x):
        if "vit" in self.encoder_name:
            y = self.encoder.forward_features(x).flatten(start_dim=1)
        else:
            y = self.encoder(x)[0]
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h