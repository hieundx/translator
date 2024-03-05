from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
os['WANDB_DISABLED'] = 'true'

if __name__ == "__main__":
    # Load and preprocess the dataset
    dataset = load_dataset('mt_eng_vietnamese', 'iwslt2015-en-vi')

    checkpoint = "Helsinki-NLP/opus-mt-en-vi"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    source_lang = "en"
    target_lang = "vi"

    def preprocess_function(examples):
        inputs =  [example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    mapped_data = dataset.map(preprocess_function, batched=True)


    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


    # Metrics
    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Training
    training_args = Seq2SeqTrainingArguments(
        output_dir="nlp_helsinki_en_vi",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=mapped_data["train"],
        eval_dataset=mapped_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()