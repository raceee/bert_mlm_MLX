from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import PreTrainedTokenizerFast
from transformers import BertForMaskedLM
from transformers import Trainer, TrainingArguments
import time
import torch

tokenizer = PreTrainedTokenizerFast(tokenizer_file="./eo_tokenizer.json")
tokenizer.mask_token = "[MASK]"
tokenizer.pad_token = "[PAD]"

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./eo.txt/eo_test.txt",
    block_size=128,  # This is the maximum length of a sequence. Adjust based on your model's limitations.
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
if torch.cuda.is_available():
    print("Moving model to cuda...")
    model.to("cuda")
# You might want to adjust the model configuration to better suit your needs.

training_args = TrainingArguments(
    output_dir="./hf_esperanto_bert",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

print("Starting training...")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


start = time.time()
trainer.train()
print("Training took: ", time.time() - start)
model.save_pretrained("./hf_esperanto_bert")
tokenizer.save_pretrained("./hf_esperanto_bert")
