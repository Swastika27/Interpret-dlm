#%%
from training import train_sae_wo_model
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import StreamingActivationsStore
from config import get_default_cfg, post_init_cfg
import json
# from transformer_lens import HookedTransformer


# for sae_type in ['topk', 'batchtopk']:
#     # don't retrain *16
#     for dict_size in [768*4, 768*8, 768*32]:     

cfg = get_default_cfg()
cfg["sae_type"] = 'batchtopk'
# cfg["model_name"] = "gpt2-small"
cfg["layer"] = 8
cfg["num_tokens"] = 102400000
cfg["embedding_glob"] = "data/embeddings/train/layer_8/*.pt" 
cfg["aux_penalty"] = (1/32)
cfg["lr"] = 3e-4
cfg["input_unit_norm"] = True
cfg["top_k"] = 32
cfg["dict_size"] = 256*32
cfg['wandb_project'] = 'Interpret-dlm'
cfg['l1_coeff'] = 0.
cfg['act_size'] = 256
cfg['device'] = 'cuda'
cfg['bandwidth'] = 0.001
cfg['top_k'] = 32

sae = BatchTopKSAE(cfg)
print("created SAE")

cfg = post_init_cfg(cfg)
            
# model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
activations_store = StreamingActivationsStore(cfg)
print("Starting training")
train_sae_wo_model(sae, activations_store, cfg)



# sae cfg for future reference
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, "__str__") and "torch" in str(type(obj)):
        return str(obj)
    else:
        return obj

cfg_path = f"{cfg['run_dir']}/config.json"
with open(cfg_path, "w") as f:
    json.dump(make_json_serializable(cfg), f, indent=4)