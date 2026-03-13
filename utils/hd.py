import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

MODEL_ID = "LongSafari/hyenadna-large-1m-seqlen-hf"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        MODEL_ID,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device).eval()

    seq = "ACGT" * 2500 
    inputs = tok(seq, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True, return_dict=True)

    # Tuple: (embeddings_output, layer1, layer2, ..., layerN)
    # Each tensor: (batch, seq_len, hidden_dim)
    hidden_states = out.hidden_states

    print("num hidden_states:", len(hidden_states))
    print("each hidden state shape:", tuple(hidden_states[-1].shape))

    # Token-level embeddings per layer
    # hidden_states[i] is token embeddings for layer i
    token_emb_by_layer = hidden_states  # tuple of tensors

    # Sequence-level embedding per layer (mean pooling; change if you want CLS/last-token pooling)
    seq_emb_by_layer = tuple(h.mean(dim=1) for h in hidden_states)

    # Final layer (same as before)
    final_token_emb = hidden_states[-1]
    final_seq_emb = seq_emb_by_layer[-1]

    print("final token_emb:", tuple(final_token_emb.shape))
    print("final seq_emb:", tuple(final_seq_emb.shape))

    # Example: stack all sequence embeddings into one tensor
    # shape: (num_layers_plus_embed, batch, hidden_dim)
    seq_emb_stack = torch.stack(seq_emb_by_layer, dim=0)
    print("seq_emb_stack:", tuple(seq_emb_stack.shape))

    print(f"named modules: {list(model.named_children())}")
    # print(f"model hyena: {model.hyena} ")
    # print(f"backbone layers: {model.backbone.layers} ")
    # print(f"submodules of hyena {list(model.hyena)}")

if __name__ == "__main__":
    main()