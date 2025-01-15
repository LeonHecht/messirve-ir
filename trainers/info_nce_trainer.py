import torch
from transformers import Trainer

class InfoNCERetrievalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute InfoNCE loss for the retrieval task.

        Args:
            model: Hugging Face model instance.
            inputs: Input tensors (query, positive_doc, negative_docs).
            return_outputs (bool): Whether to return the model outputs.

        Returns:
            Loss or (Loss, Outputs)
        """
        # Extract query and document inputs from the dataset
        query_inputs = {
            "input_ids": inputs["query_input_ids"],
            "attention_mask": inputs["query_attention_mask"],
        }
        positive_inputs = {
            "input_ids": inputs["doc_input_ids"],
            "attention_mask": inputs["doc_attention_mask"],
        }

        # Optional: Handle negatives if present
        negative_inputs = None
        if "negative_doc_input_ids" in inputs:
            negative_inputs = {
                "input_ids": inputs["negative_doc_input_ids"],
                "attention_mask": inputs["negative_doc_attention_mask"],
            }
        
        query_embeds = model(query_inputs["input_ids"], query_inputs["attention_mask"]).logits[:, -1, :]
        positive_embeds = model(positive_inputs["input_ids"], positive_inputs["attention_mask"]).logits[:, -1, :]
        if negative_inputs is not None:
            negative_embeds = model(negative_inputs["input_ids"], negative_inputs["attention_mask"]).logits[:, -1, :]

        all_docs_embeds = torch.cat([positive_embeds, negative_embeds], dim=0) if negative_inputs else positive_embeds

        # Compute raw similarity scores (dot product)
        similarity_matrix = torch.matmul(query_embeds, all_docs_embeds.T)

        # Divide all values in the similarity matrix by 1e10
        similarity_matrix = similarity_matrix / 1000

        logits = similarity_matrix
        labels = torch.arange(query_embeds.size(0), device=query_embeds.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        return (loss, logits) if return_outputs else loss
