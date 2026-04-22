from transformers import AutoTokenizer, T5EncoderModel
import torch
import numpy as np
import json

# 初始化tokenizer和encoder model (只用encoder部分)
tokenizer = AutoTokenizer.from_pretrained("/home/mengshi/embedding_model/models_hf/sentence-t5-base")
model = T5EncoderModel.from_pretrained("/home/mengshi/embedding_model/models_hf/sentence-t5-base")

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def get_edge_embeddings(edge_list):
    
    # 批量处理节点内容
    batch_size = 512  # 可以根据GPU内存调整
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
                max_length=512,
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

# 使用示例
if __name__ == "__main__":
    # data_dir = "magellan"
    # data_dir = "santos_benchmark"
    data_dir = "wikidbs_valentine"
    edge_list = [
            'token_cell', 'cell_row', 'cell_column', 'row_table', 'column_table',
            'entity_matching', 'joinable_table_search', 'schema_matching', 'union_table_search'
        ]
    
    embeddings = get_edge_embeddings(edge_list)
    edge_embedding_map = {edge: embedding.tolist() for edge, embedding in zip(edge_list, embeddings)}

    for i in range(len(edge_list)):
        edge_embedding_map[edge_list[i]] = i+1
    # 保存嵌入结果
    np.save(f"{data_dir}/edge_embeddings.npy", embeddings)
    print("Node embeddings saved to 'node_embeddings.npy'")
    
    # 保存嵌入结果
    with open(f"{data_dir}/edge_embedding_map.json", "w") as f:
        json.dump(edge_embedding_map, f)
    

    # 打印一些统计信息
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of nodes: {embeddings.shape[0]}")