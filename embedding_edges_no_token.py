import argparse
import os
import numpy as np
import json

DEFAULT_MODEL_PATH = "/home/mengshi/embedding_model/models_hf/sentence-t5-base"
DEFAULT_DATA_DIR = "magellan_no_token"
DEFAULT_EDGE_LIST = [
    "cell_row",
    "cell_column",
    "row_table",
    "column_table",
    "entity_matching",
    "joinable_table_search",
    "schema_matching",
    "union_table_search",
]

def get_edge_embeddings(edge_list, tokenizer, model, device, batch_size: int = 1024, max_length: int = 512):
    import torch
    
    # 批量处理节点内容
    edge_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(edge_list), batch_size):
            batch_contents = edge_list[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(edge_list)-1)//batch_size + 1}")
            
            # Tokenize批次内容
            inputs = tokenizer(
                batch_contents,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            # 获取encoder输出
            outputs = model(**inputs)
            
            # 使用mean pooling获取句子嵌入
            # T5EncoderModel返回last_hidden_state
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = embeddings * mask_expanded
            embeddings = embeddings.sum(1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
            
            edge_embeddings.append(embeddings.cpu())
    
    # 合并所有批次的嵌入
    edge_embeddings = torch.cat(edge_embeddings, dim=0)
    
    print(f"Edge embeddings shape: {edge_embeddings.shape}")
    
    return edge_embeddings.numpy()

def parse_args():
    parser = argparse.ArgumentParser(description="Compute edge-type embeddings (no-token pipeline).")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory to save edge embeddings/mapping.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="HuggingFace model path.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length for tokenizer truncation.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()

# 使用示例
if __name__ == "__main__":
    args = parse_args()
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

    embeddings = get_edge_embeddings(
        DEFAULT_EDGE_LIST,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    edge_embedding_map = {edge: i + 1 for i, edge in enumerate(DEFAULT_EDGE_LIST)}
    # 保存嵌入结果
    np.save(os.path.join(args.data_dir, "edge_embeddings.npy"), embeddings)
    print("Edge embeddings saved to 'edge_embeddings.npy'")
    
    # 保存嵌入结果
    with open(os.path.join(args.data_dir, "edge_embedding_map.json"), "w") as f:
        json.dump(edge_embedding_map, f)
    

    # 打印一些统计信息
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of nodes: {embeddings.shape[0]}")
