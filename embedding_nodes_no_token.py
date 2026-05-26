import argparse
import os
import numpy as np
import json

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "")
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "output_no_token")


def get_node_embeddings(
    node_id_mapping,
    tokenizer,
    model,
    device,
    batch_size: int = 1024,
    max_length: int = 512,
):
    import torch
    content_to_id = {}
    for key in node_id_mapping.keys():
        for key2 in node_id_mapping[key].keys():
            node_id = node_id_mapping[key][key2]
            content_to_id[key2] = node_id

    node_num = max(content_to_id.values()) + 1
    print(f"Total number of nodes: {node_num}")

    id_to_content = {}
    for content, node_id in content_to_id.items():
        id_to_content[node_id] = content

    node_contents = []
    for i in range(node_num):
        if i in id_to_content:
            node_contents.append(id_to_content[i])
        else:
            print(f"Warning: Node ID {i} not found in mapping, using empty string")
            node_contents.append("")

    print(f"Processing {len(node_contents)} nodes...")

    node_embeddings = []

    with torch.no_grad():
        for i in range(0, len(node_contents), batch_size):
            batch_contents = node_contents[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(node_contents)-1)//batch_size + 1}")

            inputs = tokenizer(
                batch_contents,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)

            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']

            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = embeddings * mask_expanded
            embeddings = embeddings.sum(1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

            node_embeddings.append(embeddings.cpu())

    node_embeddings = torch.cat(node_embeddings, dim=0)

    print(f"Node embeddings shape: {node_embeddings.shape}")

    return node_embeddings.numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Compute node embeddings (no-token pipeline).")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing node_id_mapping.json.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="HuggingFace model path.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length for tokenizer truncation.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.model_path:
        raise SystemExit("[ERROR] --model-path is required (or set MODEL_PATH env var).")
    os.makedirs(args.data_dir, exist_ok=True)

    try:
        import torch
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Error: 'torch' is required for embedding scripts. Please install it first.") from e

    try:
        from transformers import AutoTokenizer, T5EncoderModel
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Error: 'transformers' is required for embedding scripts. Please install it first.") from e

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = T5EncoderModel.from_pretrained(args.model_path)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    with open(os.path.join(args.data_dir, "node_id_mapping.json"), "r") as f:
        node_id_mapping = json.load(f)
        print(f"{len(node_id_mapping)} node id mappings loaded")

    embeddings = get_node_embeddings(
        node_id_mapping,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    np.save(os.path.join(args.data_dir, "node_embeddings.npy"), embeddings)
    print("Node embeddings saved to 'node_embeddings.npy'")
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of nodes: {embeddings.shape[0]}")
