from mlx_llm.model import create_model
from transformers import BertTokenizer
import mlx.core as mx

model = create_model("bert-base-uncased") # it will download weights from this repository
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

batch = ["This is an example of BERT working on MLX."]
tokens = tokenizer(batch, return_tensors="np", padding=True)
tokens = {key: mx.array(v) for key, v in tokens.items()}

output, pooled = model(**tokens)