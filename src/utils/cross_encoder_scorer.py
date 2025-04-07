import torch

class CrossEncoderScorer:
    """
    A wrapper class for scoring a single query–document pair using a cross-encoder.
    This class abstracts away the differences between models trained with different
    objectives (binary classification, fine-grained classification, RankNet, etc.).
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained cross-encoder model.
    tokenizer : transformers.PreTrainedTokenizer
        The corresponding tokenizer.
    head_type : str, optional
        The head type used during training. This can be:
          - "binary" classification,
          - "fine_grained" for fine-grained classification,
          - "ranknet" for pairwise ranking training.
        Default is "binary".
    """
    def __init__(self, model, tokenizer, head_type="binary"):
        self.model = model
        self.tokenizer = tokenizer
        self.head_type = head_type

    def score(self, query, doc, max_length=512):
        """
        Score a single query–document pair.
        
        Parameters
        ----------
        query : str
            The query text.
        doc : str
            The document text.
        max_length : int, optional
            Maximum sequence length for tokenization. Default is 512.
        
        Returns
        -------
        float
            A single numerical score representing the relevance of the document to the query.
        """
        # Tokenize and prepare input (batch size=1)
        inputs = self.tokenizer(
            query, doc, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits  # shape: [1, num_labels]
        
        if self.head_type == "fine_grained":
            # For fine-grained classification, compute weighted average of class indices.
            probs = torch.softmax(logits, dim=-1)
            indices = torch.arange(logits.size(-1), device=logits.device, dtype=torch.float)
            score = (probs * indices).sum().item()
        elif self.head_type == "binary":
            # For binary classification, assume two classes; return the positive class probability.
            if logits.size(-1) == 2:
                score = torch.softmax(logits, dim=-1)[0, 1].item()
            else:
                # If only one logit is present, use it directly.
                score = logits.item()
        elif self.head_type == "ranknet":
            # For RankNet training, the model should be callable in single-input mode,
            # so we return the raw logit score (the higher, the more relevant).
            score = logits[:, 0].item()
        else:
            # Default: use the argmax label.
            score = logits.argmax(dim=-1).item()
        
        return score
