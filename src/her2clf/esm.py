import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def load_esm_model(model_name):
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name)
    model.eval()

    return tokenizer, model


def embed_sequences(sequences, tokenizer, model, batch_size=512, pooling="mean"):
    all_embeddings=[]

    for start in range(0, len(sequences), batch_size):
        batch=sequences[start:start+batch_size]
        
        tokens=tokenizer(
            batch,
            padding=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**tokens)
        
        residue_embeddings = outputs.last_hidden_state[:, 1:-1, :]
        
        # Original approach:
        # embeddings = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
        
        if pooling == "mean":
            embeddings= residue_embeddings.mean(dim=1)
        
        elif pooling=="flatten":
            embeddings= residue_embeddings.flatten(start_dim=1)
        
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0)


def embed_sequences_to_disk(
    sequences,
    tokenizer,
    model,
    output_path,
    batch_size=512,
    pooling="flatten",
):
    if pooling == "mean":
        embedding_dim = model.config.hidden_size
    elif pooling == "flatten":
        seq_length = len(sequences[0])
        embedding_dim = seq_length * model.config.hidden_size
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

    embeddings = np.memmap(
        output_path,
        dtype="float32",
        mode="w+",
        shape=(len(sequences), embedding_dim),
    )

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]

        tokens = tokenizer(
            batch,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**tokens)

        residue_embeddings = outputs.last_hidden_state[:, 1:-1, :]

        if pooling == "mean":
            batch_embeddings = residue_embeddings.mean(dim=1)

        elif pooling == "flatten":
            batch_embeddings = residue_embeddings.flatten(start_dim=1)

        end = start + len(batch)

        embeddings[start:end] = batch_embeddings.cpu().numpy()

    embeddings.flush()

    return embeddings

def load_embeddings_from_disk(
    path,
    n_sequences,
    embedding_dim,
):
    return np.memmap(
        path,
        dtype="float32",
        mode="r",
        shape=(n_sequences, embedding_dim),
    )