from huggingface_hub.hf_api import HfFolder
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

HfFolder.save_token('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')

model_checkpoint = "facebook/bart-base"
model_name = model_checkpoint.split("/")[-1]
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_name}-finetuned-xsum")


def generate_summary(question, model):
    inputs = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


summaries_before_tuning = generate_summary(
    "Hi I'm XXXXXXX XXXXXXX I was told by a doctor I have either pneumonia or nodularity within the right lung upper lobe if idon't respond to antibiotics.Is that poosible and can you pneumni?Penelope or I have a mass and it's probably cancer",
    model)[1]
print(summaries_before_tuning)
