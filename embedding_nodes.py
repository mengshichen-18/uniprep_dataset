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

def get_node_embeddings(node_id_mapping):
    """
    根据node_id_mapping获取节点嵌入
    
    Args:
        node_id_mapping: 嵌套字典，结构为 {key: {key2: node_id}}
                        其中key2是node content，node_id是从0到node_num的
    
    Returns:
        node_embeddings: 按node_id顺序排列的节点嵌入矩阵
    """
    
    # 打印节点映射信息
    # print("Node ID Mapping Keys:", node_id_mapping.keys())
    # for key in node_id_mapping.keys():
    #     print(f"Key: {key}")
    #     for key2 in node_id_mapping[key].keys():
    #         print(f"  Content: {key2}, Node ID: {node_id_mapping[key][key2]}")
    
    # 创建内容到node_id的映射
    content_to_id = {}
    for key in node_id_mapping.keys():
        for key2 in node_id_mapping[key].keys():
            node_id = node_id_mapping[key][key2]
            content_to_id[key2] = node_id
    
    # 确定节点数量
    node_num = max(content_to_id.values()) + 1
    print(f"Total number of nodes: {node_num}")
    
    # 创建按node_id顺序排列的内容列表
    id_to_content = {}
    for content, node_id in content_to_id.items():
        id_to_content[node_id] = content
    
    # 按顺序提取节点内容
    node_contents = []
    for i in range(node_num):
        if i in id_to_content:
            node_contents.append(id_to_content[i])
        else:
            print(f"Warning: Node ID {i} not found in mapping, using empty string")
            node_contents.append("")
    
    print(f"Processing {len(node_contents)} nodes...")
    
    # 批量处理节点内容
    batch_size = 512  # 可以根据GPU内存调整
    node_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(node_contents), batch_size):
            batch_contents = node_contents[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(node_contents)-1)//batch_size + 1}")
            
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
            
            node_embeddings.append(embeddings.cpu())
    
    # 合并所有批次的嵌入
    node_embeddings = torch.cat(node_embeddings, dim=0)
    
    print(f"Node embeddings shape: {node_embeddings.shape}")
    
    return node_embeddings.numpy()

# 使用示例
if __name__ == "__main__":
    # 假设您的node_id_mapping已经定义好了
    # node_id_mapping = {...}  # 您的实际数据
    # data_dir = "./magellan"
    # data_dir = "santos_benchmark"
    data_dir = "wikidbs_valentine"
    with open(f"{data_dir}/node_id_mapping.json", 'r') as f:
        node_id_mapping = json.load(f)
        print(f"{len(node_id_mapping)} node id mappings loaded")
    # 获取节点嵌入
    embeddings = get_node_embeddings(node_id_mapping)
    
    # 保存嵌入结果
    np.save(f"{data_dir}/node_embeddings.npy", embeddings)
    print("Node embeddings saved to 'node_embeddings.npy'")
    
    # 打印一些统计信息
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of nodes: {embeddings.shape[0]}")