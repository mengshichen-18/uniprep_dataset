# file: test_tokenizer_only.py
from transformers import AutoTokenizer

LOCAL_DIR = "/home/mengshi/embedding_model/models_hf/sentence-t5-base"

tok = AutoTokenizer.from_pretrained(LOCAL_DIR, local_files_only=True)

text = "hello world"
ids = tok.encode(text, add_special_tokens=True)
back = tok.decode(ids, skip_special_tokens=True)

print("Vocab size:", tok.vocab_size if hasattr(tok, "vocab_size") else "N/A")
print("Encoded ids:", ids[:10], "... len =", len(ids))
print("Round-trip decode:", back)

assert isinstance(ids, list) and len(ids) > 0
assert isinstance(back, str) and len(back) > 0
print("✅ Tokenizer OK")
