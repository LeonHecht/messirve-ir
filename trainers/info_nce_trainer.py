import torch
from transformers import Trainer
import torch.nn.functional as F


def get_eos_embeddings(model, input_ids, attention_mask, tokenizer):
    """
    Extract L2-normalized embeddings of exactly one EOS token per sequence.
    Assumes:
      - Each sequence in `input_ids` has exactly one EOS token.
      - That EOS token is `tokenizer.eos_token_id`.
    Raises an error if any sequence has zero or multiple EOS tokens.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)

    batch_size = input_ids.size(0)      # for hard negs it is batch_size*n_neg
    eos_id = tokenizer.eos_token_id

    # 1) Find positions of all eos_id tokens in the batch
    # eos_positions is (row_indices, col_indices), each shape (#eos_tokens_found,)
    eos_positions = (input_ids == eos_id).nonzero(as_tuple=True)
    row_indices, col_indices = eos_positions

    # 2) Assert that the number of EOS tokens found == batch_size
    if row_indices.size(0) != batch_size:
        raise ValueError(
            f"Expected exactly 1 EOS token per sequence, "
            f"but found {row_indices.size(0)} matches for {batch_size} sequences."
        )

    # 3) Check that each sequence index appears exactly once
    #    i.e., no sequence is missing an EOS or has multiple
    unique_rows, counts = row_indices.unique(return_counts=True)
    if unique_rows.size(0) < batch_size:
        raise ValueError("At least one sequence does not contain any EOS token.")
    if (counts > 1).any():
        raise ValueError("At least one sequence has multiple EOS tokens.")

    # 4) Gather the embeddings for the single EOS token per sequence
    eos_embeds = hidden_states[row_indices, col_indices, :]  # shape (batch_size, hidden_dim)

    # 5) Normalize embeddings to unit L2 norm
    eos_embeds = F.normalize(eos_embeds, p=2, dim=-1)

    return eos_embeds


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


class InfoNCERetrievalTrainerHNLLM(Trainer):
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
        num_negatives = inputs["neg_input_ids"].size(1)

        # Extract query and document inputs
        query_inputs = {
            "input_ids": inputs["query_input_ids"],
            "attention_mask": inputs["query_attention_mask"],
        }
        positive_inputs = {
            "input_ids": inputs["doc_input_ids"],
            "attention_mask": inputs["doc_attention_mask"],
        }

        query_embeds = get_eos_embeddings(model, query_inputs["input_ids"], query_inputs["attention_mask"], self.tokenizer)
        positive_embeds = get_eos_embeddings(model, positive_inputs["input_ids"], positive_inputs["attention_mask"], self.tokenizer)

        neg_ids = inputs["neg_input_ids"]        # (batch_size, 5, doc_seq_len)
        neg_att = inputs["neg_attention_mask"]   # (batch_size, 5, doc_seq_len)

        # Flatten for the forward pass
        neg_ids = neg_ids.view(-1, neg_ids.size(-1))           # (batch_size * 5, doc_seq_len)
        neg_att = neg_att.view(-1, neg_att.size(-1))           # (batch_size * 5, doc_seq_len)
        hard_neg_embeds = get_eos_embeddings(model,     # shape: (batch_size*5, emdeb_dim)
                                             neg_ids,
                                             neg_att,
                                             self.tokenizer)
        
        # Reshape to (batch_size, num_negatives, embed_dim)
        batch_size = query_embeds.size(0)
        embed_dim = query_embeds.size(-1)
        hard_neg_embeds = hard_neg_embeds.view(batch_size, num_negatives, embed_dim)

        # Concatenate positive embeddings and hard negatives for similarity computation
        all_docs_embeds = torch.cat(
            [positive_embeds.unsqueeze(1), hard_neg_embeds],
            dim=1
        )  # (batch_size, 1 + num_hard_negatives, embedding_dim)

        # Compute similarity scores
        similarity_matrix = torch.matmul(
            query_embeds.unsqueeze(1), all_docs_embeds.transpose(1, 2)
        ).squeeze(1)  # (batch_size, 1 + num_hard_negatives)

        # Compute InfoNCE loss
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_embeds.device)  # Positive at index 0
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(similarity_matrix, labels)

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
        num_negatives = inputs["neg_input_ids"].size(1)

        query_inputs = {
            "input_ids": inputs["query_input_ids"],
            "attention_mask": inputs["query_attention_mask"],
        }
        positive_inputs = {
            "input_ids": inputs["doc_input_ids"],
            "attention_mask": inputs["doc_attention_mask"],
        }

        with torch.no_grad():
            # Compute embeddings for queries
            query_embeds = get_eos_embeddings(model, query_inputs["input_ids"], query_inputs["attention_mask"], self.tokenizer)
            positive_embeds = get_eos_embeddings(model, positive_inputs["input_ids"], positive_inputs["attention_mask"], self.tokenizer)
            neg_ids = inputs["neg_input_ids"]        # (batch_size, 5, doc_seq_len)
            neg_att = inputs["neg_attention_mask"]   # (batch_size, 5, doc_seq_len)

            # Flatten for the forward pass
            neg_ids = neg_ids.view(-1, neg_ids.size(-1))           # (batch_size * 5, doc_seq_len)
            neg_att = neg_att.view(-1, neg_att.size(-1))           # (batch_size * 5, doc_seq_len)
            hard_neg_embeds = get_eos_embeddings(model,     # shape: (batch_size*5, emdeb_dim)
                                                neg_ids,
                                                neg_att,
                                                self.tokenizer)

            # Reshape to (batch_size, num_negatives, embed_dim)
            batch_size = query_embeds.size(0)
            embed_dim = query_embeds.size(-1)
            hard_neg_embeds = hard_neg_embeds.view(batch_size, num_negatives, embed_dim)

            # Concatenate positive embeddings and hard negatives
            all_docs_embeds = torch.cat(
                [positive_embeds.unsqueeze(1), hard_neg_embeds],
                dim=1
            )  # (batch_size, 1 + num_hard_negatives, embedding_dim)

            # Compute similarity scores
            similarity_matrix = torch.matmul(
                query_embeds.unsqueeze(1), all_docs_embeds.transpose(1, 2)
            ).squeeze(1)  # (batch_size, 1 + num_hard_negatives)

        # Optionally return loss if labels are provided
        if prediction_loss_only:
            labels = torch.zeros(batch_size, dtype=torch.long, device=query_embeds.device)  # Positive at index 0
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
        ).hidden_states[-1][:, 0, :]  # EOS token embedding

        positive_embeds = model(
            input_ids=positive_inputs["input_ids"],
            attention_mask=positive_inputs["attention_mask"],
            output_hidden_states=True,
        ).hidden_states[-1][:, 0, :]  # EOS token embedding

        # Compute embeddings for hard negatives
        hard_neg_embeds = model(
            input_ids=hard_negative_inputs["input_ids"].view(-1, hard_negative_inputs["input_ids"].size(-1)),
            attention_mask=hard_negative_inputs["attention_mask"].view(-1, hard_negative_inputs["attention_mask"].size(-1)),
            output_hidden_states=True,
        ).hidden_states[-1][:, 0, :]  # EOS token embedding

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
