import os
import sys
import pandas as pd
import string

def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Apply the path tweak before any project imports
configure_python_path()

from utils.retrieval_utils import get_legal_dataset


class LLMAnnotationQC:
    def __init__(self, qrels_path_in, qrels_path_out, corpus_path):
        self.qrels_dev_df = pd.read_csv(
                qrels_path_in,
                sep="\t",                # TREC qrels are usually tab-separated
                names=["query_id", "iteration", "doc_id", "relevance", "evidence"],
                header=None,            # There's no header in qrels files
                dtype={"query_id": str, "iteration": int, "doc_id": str, "relevance": int, "evidence": str}
            )
        
        # Ensure 'evidence' is a string column with no NaNs
        self.qrels_dev_df['evidence'] = (
            self.qrels_dev_df['evidence']
            .fillna("")
            .astype(str)
        )

        dids, docs = get_legal_dataset(corpus_path)
        self.doc_dict = dict(zip(dids, docs))

        self.qrels_path_out = qrels_path_out

        self.set_zero_lexical = 0
        self.set_zero_counter_exact = 0
    
    def filter_by_lexical_match(self):
        # iterate over df
        for index, row in self.qrels_dev_df.iterrows():
            if row["relevance"] == 0:
                continue
            evidence = row["evidence"].lower()
            doc_text = self.doc_dict[row["doc_id"]].lower()
            
            found_counter = 0
            # 1. Check individual word match
            words = evidence.split()
            for word in words:
                token = word.strip(string.punctuation)
                if token and token in doc_text:
                    found_counter += 1
            
            lexical_match = found_counter / len(words)

            if lexical_match < 1.0:
                # set relevance to 0
                self.qrels_dev_df.at[index, "relevance"] = 0
                self.qrels_dev_df.at[index, "evidence"] = f"Set to 0 due to low lexical match: {lexical_match}"
                print(f"Set relevance to 0 for {row['query_id']} - {row['doc_id']} due to low lexical match: {lexical_match:.2f}")
                self.set_zero_lexical += 1
        print(f"Set relevance to 0 for {self.set_zero_lexical} pairs due to low lexical match.")
            
    def filter_by_exact_match(self):
        # iterate over df
        for index, row in self.qrels_dev_df.iterrows():
            if row["relevance"] == 0:
                continue
            evidence = row["evidence"].lower()
            words = evidence.split()
            words = [word.strip(string.punctuation) for word in words]
            evidence = " ".join(words)

            doc_text = self.doc_dict[row["doc_id"]].lower()
            
            if evidence not in doc_text:
                # set relevance to 0
                self.qrels_dev_df.at[index, "relevance"] = 0
                self.qrels_dev_df.at[index, "evidence"] = f"Set to 0 due to no exact match"
                print(f"Set relevance to 0 for {row['query_id']} - {row['doc_id']} due to no exact match")
                self.set_zero_counter_exact += 1
        print(f"Set relevance to 0 for {self.set_zero_counter_exact} pairs due to no exact match.")

    def save_qrels(self):
        self.qrels_dev_df.to_csv(self.qrels_path_out, sep="\t", index=False, header=False)
        print(f"Saved filtered qrels to {self.qrels_path_out}")


if __name__ == "__main__":
    # Paths
    qrels_path_in = "/media/discoexterno/leon/legal_ir/data/annotations/qrels_mistral-small-2501_v7.tsv"
    qrels_path_out = "/media/discoexterno/leon/legal_ir/data/annotations/qrels_mistral-small-2501_v7_QC.tsv"
    corpus_path = "/media/discoexterno/leon/legal_ir/data/corpus/corpus_py.csv"

    llm_qc = LLMAnnotationQC(qrels_path_in, qrels_path_out, corpus_path)
    llm_qc.filter_by_lexical_match()
    # llm_qc.filter_by_exact_match()
    llm_qc.save_qrels()