%%writefile utils/metrics.py
import numpy as np
import evaluate

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def get_compute_metrics_fn(llama_tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        pred_ids = np.argmax(preds, axis=-1)
        labels = np.where(labels != -100, labels, llama_tokenizer.pad_token_id)
        
        decoded_preds = llama_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = llama_tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        decoded_labels_rouge = [label[0] for label in decoded_labels]
        rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels_rouge)
        
        exact_match = sum([1 if p == l[0] else 0 for p, l in zip(decoded_preds, decoded_labels)]) / max(len(decoded_preds), 1)

        return {
            "accuracy": exact_match,
            "bleu": bleu_result["bleu"],
            "rougeL": rouge_result["rougeL"]
        }
    return compute_metrics