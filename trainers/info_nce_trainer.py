import torch
from transformers import Trainer
from torch.nn.utils import clip_grad_norm_

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
        if "neg_input_ids" in inputs:
            negative_inputs = {
                "input_ids": inputs["neg_input_ids"],
                "attention_mask": inputs["neg_attention_mask"],
            }
        
        outputs = model(query_inputs["input_ids"],
                             query_inputs["attention_mask"],
                             output_hidden_states=True)
        # query_embeds = outputs.hidden_states[-1][:, -1, :]  # Extract the last token's hidden state (EOS token)
        query_embeds = outputs.hidden_states[-1][:, 0, :]  # Extract the first token's hidden state (CLS token)
        
        outputs = model(positive_inputs["input_ids"],
                                positive_inputs["attention_mask"],
                                output_hidden_states=True)
        # positive_embeds = outputs.hidden_states[-1][:, -1, :]  # Extract the last token's hidden state (EOS token)
        positive_embeds = outputs.hidden_states[-1][:, 0, :]  # Extract the first token's hidden state (CLS token)
        if negative_inputs is not None:
            # negative_embeds = model(negative_inputs["input_ids"], negative_inputs["attention_mask"]).logits[:, -1, :]
            outputs = model(negative_inputs["input_ids"],
                                    negative_inputs["attention_mask"],
                                    output_hidden_states=True)
            # negative_embeds = outputs.hidden_states[-1][:, -1, :]
            negative_embeds = outputs.hidden_states[-1][:, 0, :]

        all_docs_embeds = torch.cat([positive_embeds, negative_embeds], dim=0) if negative_inputs else positive_embeds

        # Compute raw similarity scores (dot product)
        similarity_matrix = torch.matmul(query_embeds, all_docs_embeds.T)

        logits = similarity_matrix
        labels = torch.arange(query_embeds.size(0), device=query_embeds.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        return (loss, logits) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Custom prediction step for evaluation.
        """
        query_inputs = {
            "input_ids": inputs["query_input_ids"],
            "attention_mask": inputs["query_attention_mask"],
        }
        doc_inputs = {
            "input_ids": inputs["doc_input_ids"],
            "attention_mask": inputs["doc_attention_mask"],
        }

        with torch.no_grad():
            # query_embeds = model(**query_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
            query_embeds = model(**query_inputs, output_hidden_states=True).hidden_states[-1][:, 0, :]
            # doc_embeds = model(**doc_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
            doc_embeds = model(**doc_inputs, output_hidden_states=True).hidden_states[-1][:, 0, :]

        # Compute similarity matrix
        similarity_matrix = torch.matmul(query_embeds, doc_embeds.T)

        # Optionally return loss if labels are provided
        if prediction_loss_only:
            labels = torch.arange(query_embeds.size(0), device=query_embeds.device)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(similarity_matrix, labels)
            return (loss, None, None)

        return (None, similarity_matrix, None)


class InfoNCERetrievalTrainerHN(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute InfoNCE loss with hard negatives only.

        Args:
            model: Hugging Face model instance.
            inputs: Input tensors (query, positive_doc, hard_negatives).
            return_outputs (bool): Whether to return the model outputs.

        Returns:
            Loss or (Loss, Outputs)
        """
        # Extract query and document inputs
        query_inputs = {
            "input_ids": inputs["query_input_ids"],
            "attention_mask": inputs["query_attention_mask"],
        }
        positive_inputs = {
            "input_ids": inputs["doc_input_ids"],
            "attention_mask": inputs["doc_attention_mask"],
        }
        hard_negative_inputs = {
            "input_ids": inputs["neg_input_ids"],
            "attention_mask": inputs["neg_attention_mask"],
        }

        # Compute embeddings for queries and positive documents
        query_embeds = model(
            input_ids=query_inputs["input_ids"],
            attention_mask=query_inputs["attention_mask"],
            output_hidden_states=True,
        ).hidden_states[-1][:, 0, :]  # CLS token embedding

        positive_embeds = model(
            input_ids=positive_inputs["input_ids"],
            attention_mask=positive_inputs["attention_mask"],
            output_hidden_states=True,
        ).hidden_states[-1][:, 0, :]  # CLS token embedding

        # Compute embeddings for hard negatives
        hard_neg_embeds = model(
            input_ids=hard_negative_inputs["input_ids"].view(-1, hard_negative_inputs["input_ids"].size(-1)),
            attention_mask=hard_negative_inputs["attention_mask"].view(-1, hard_negative_inputs["attention_mask"].size(-1)),
            output_hidden_states=True,
        ).hidden_states[-1][:, 0, :]  # CLS token embedding

        # Reshape hard negative embeddings to match the batch structure
        hard_neg_embeds = hard_neg_embeds.view(
            query_embeds.size(0), -1, hard_neg_embeds.size(-1)
        )  # (batch_size, num_hard_negatives, embedding_dim)

        # Concatenate positive embeddings and hard negatives for similarity computation
        all_docs_embeds = torch.cat(
            [positive_embeds.unsqueeze(1), hard_neg_embeds], dim=1
        )  # (batch_size, 1 + num_hard_negatives, embedding_dim)

        # Compute similarity scores
        similarity_matrix = torch.matmul(
            query_embeds.unsqueeze(1), all_docs_embeds.transpose(1, 2)
        ).squeeze(1)  # (batch_size, 1 + num_hard_negatives)

        # Compute InfoNCE loss
        labels = torch.zeros(query_embeds.size(0), dtype=torch.long, device=query_embeds.device)  # Positive at index 0
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(similarity_matrix, labels)

        # Apply gradient clipping
        if self.args.fp16:
            # Prevent gradient scaling issues when using fp16
            self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        return (loss, similarity_matrix) if return_outputs else loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Custom prediction step for evaluation with hard negatives.
        """
        query_inputs = {
            "input_ids": inputs["query_input_ids"],
            "attention_mask": inputs["query_attention_mask"],
        }
        positive_inputs = {
            "input_ids": inputs["doc_input_ids"],
            "attention_mask": inputs["doc_attention_mask"],
        }
        hard_negative_inputs = {
            "input_ids": inputs["neg_input_ids"],
            "attention_mask": inputs["neg_attention_mask"],
        }

        with torch.no_grad():
            # Compute embeddings for queries
            query_embeds = model(
                input_ids=query_inputs["input_ids"],
                attention_mask=query_inputs["attention_mask"],
                output_hidden_states=True,
            ).hidden_states[-1][:, 0, :]  # CLS token embedding

            # Compute embeddings for positive documents
            positive_embeds = model(
                input_ids=positive_inputs["input_ids"],
                attention_mask=positive_inputs["attention_mask"],
                output_hidden_states=True,
            ).hidden_states[-1][:, 0, :]  # CLS token embedding

            # Compute embeddings for hard negatives
            hard_neg_embeds = model(
                input_ids=hard_negative_inputs["input_ids"].view(-1, hard_negative_inputs["input_ids"].size(-1)),
                attention_mask=hard_negative_inputs["attention_mask"].view(-1, hard_negative_inputs["attention_mask"].size(-1)),
                output_hidden_states=True,
            ).hidden_states[-1][:, 0, :]  # CLS token embedding

            # Reshape hard negative embeddings
            hard_neg_embeds = hard_neg_embeds.view(
                query_embeds.size(0), -1, hard_neg_embeds.size(-1)
            )  # (batch_size, num_hard_negatives, embedding_dim)

            # Concatenate positive embeddings and hard negatives
            all_docs_embeds = torch.cat(
                [positive_embeds.unsqueeze(1), hard_neg_embeds], dim=1
            )  # (batch_size, 1 + num_hard_negatives, embedding_dim)

            # Compute similarity scores
            similarity_matrix = torch.matmul(
                query_embeds.unsqueeze(1), all_docs_embeds.transpose(1, 2)
            ).squeeze(1)  # (batch_size, 1 + num_hard_negatives)

        # Optionally return loss if labels are provided
        if prediction_loss_only:
            labels = torch.zeros(query_embeds.size(0), dtype=torch.long, device=query_embeds.device)  # Positive at index 0
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(similarity_matrix, labels)
            return (loss, None, None)

        return (None, similarity_matrix, None)
