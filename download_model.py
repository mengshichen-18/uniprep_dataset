from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel

# HF 仓库名（sentence-t5-base 的常用ID）
repo_id = "sentence-transformers/sentence-t5-base"

# 你的目标目录
local_dir = "/home/mengshi/embedding_model/models_hf/sentence-t5-base"

# 下载（local_dir_use_symlinks=False 可以避免软链接带来的权限/移动问题）
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    revision=None,        # 需要固定版本可以填具体commit或tag
    allow_patterns=None   # 需要精简文件时可指定包含的文件模式
)

# 之后就可以用本地路径加载
tok = AutoTokenizer.from_pretrained(local_dir)
model = AutoModel.from_pretrained(local_dir)   # 或者 SentenceTransformer(..) 见下方
