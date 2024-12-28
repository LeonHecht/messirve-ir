from transformers import AutoTokenizer
from data.dataset_preparation import prepare_dataset


def get_dataset():
    # Example dataset 
    raw_data = {
        "query": ["What is AI?", "Define ML", "Explain Deep Learning", "What is Natural Language Processing (NLP)?"], 
        "positive_doc": ["AI stands for Artificial Intelligence.", "ML is Machine Learning.", "Deep Learning is a subset of ML.", "NLP is a subfield of AI that focuses on processing human language."],
        "negative_docs": [
            ["AI is a hardware device.", "AI is a movie."],
            ["ML is a programming language.", "ML is a dataset."],
            ["Deep Learning is a type of neural network.", "Deep Learning is a type of computer."],
            ["NLP is a type of programming language.", "NLP is a type of dataset."]
        ]
    }

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_dataset = prepare_dataset(raw_data, tokenizer)
    return tokenized_dataset


def main():
    dataset = get_dataset()
    print(dataset)


if __name__ == "__main__":
    main()