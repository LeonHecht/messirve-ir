import torch
from transformers import Trainer

class InfoNCERetrievalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute InfoNCE loss for the retrieval task.

        Args:
            model: Hugging Face model instance.
            inputs: Input tensors (query, positive_doc, negative_docs).
            return_outputs (bool): Whether to return the model outputs.

        Returns:
            Loss or (Loss, Outputs)
        """
        query_inputs = inputs['query']
        positive_inputs = inputs['positive_doc']
        negative_inputs = inputs.get('negative_docs')  # Optional
        
        query_embeds = model(**query_inputs).pooler_output
        positive_embeds = model(**positive_inputs).pooler_output
        if negative_inputs is not None:
            negative_embeds = model(**negative_inputs).pooler_output

        all_docs_embeds = torch.cat([positive_embeds, negative_embeds], dim=0) if negative_inputs else positive_embeds
        similarity_matrix = torch.matmul(query_embeds, all_docs_embeds.T)

        logits = similarity_matrix
        labels = torch.arange(query_embeds.size(0), device=query_embeds.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        return (loss, logits) if return_outputs else loss
