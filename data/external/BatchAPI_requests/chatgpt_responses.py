"""
Script to create the jsonl file for the ChatGPT Batch API requests.
Idea: Generate responses for 50k queries from the MS MARCO dataset.
Then finetune using InfoNCE loss with the responses instead of the queries.
"""
import sys
sys.path.append("src")

import json
from utils.train_utils import get_msmarco_hard_negatives, get_msmarco_queries
import pickle
from tqdm import tqdm
import codecs
import ir_datasets
from utils.retrieval_utils import get_legal_dataset
import os
from config.config import STORAGE_DIR


def create_jsonl_file():
    # qid_to_query = get_msmarco_queries()

    dataset = ir_datasets.load("msmarco-passage/dev/small")

    # num_negs = 15
    # negs_ds = get_msmarco_hard_negatives(num_negs, reload=True)
    # negs_ds = negs_ds.select(range(50_000))

    # qid_to_query_100k = {qid: qid_to_query[qid] for qid in negs_ds["query"]}
    # print(len(qid_to_query_100k))

    # with open("qid_to_query_226.json", "r", encoding='utf-8') as f:
    #     qid_to_query_226 = json.load(f)

    sys_instruct = "Given any user query, analyze and describe the user's underlying intent in a concise and semantically enriched manner in no more than 150 tokens."
    # sys_instruct = "Given any user query, generate a concise, informative, and well-structured answer in no more than 150 tokens."

    # Step 1: Prepare the .jsonl file
    requests = [
        {
            "custom_id": str(qid),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": sys_instruct},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                "max_tokens": 150
            }
        }
        for qid, query in dataset.queries_iter()
    ]

    print(len(requests))

    # assert len(requests) == len(qid_to_query_226), "Number of requests should match the number of queries"

    # Write requests to a .jsonl file
    with open('batch_requests_dev_small_intent.jsonl', 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')


def create_jsonl_file_legal():
    doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus_raw_google_ocr.csv"))
    sys_instruct = "I have an OCR output of a legal case document that contains a lot of repeated header and footer information. Could you please remove these headers, footers, and any extraneous lines (like lines that only have digits or are very short) from the text? Other than that, keep the text as it is. Only include the cleaned text in your answer."

    requests = [
        {
            "custom_id": str(id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": sys_instruct},
                    {"role": "user", "content": f"Here is the case:\n {doc}"}
                ],
            }
        }
        for id, doc in zip(doc_ids, docs)
    ]
    print(len(requests))

    limit = 200
    counter = 0

    for i in range(len(requests) // limit):
        # Write requests to a .jsonl file
        with open(f'batch_requests_legal_cleanup_{i}.jsonl', 'w', encoding="utf-8") as f:
            r = i * limit
            while r < len(requests):
                if counter >= limit:
                    counter = 0
                    break
                f.write(json.dumps(requests[r], ensure_ascii=False) + '\n')
                counter += 1
                r += 1


def process_response_file(in_path, out_path):
    """ Read json file containing responses from ChatGPT Batch API.
        Then iterate over the responses and return a list of responses. 
    """
    with open(in_path, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
    
    id_to_response = {}
    for item in items:
        id = item["custom_id"]
        response = item["response"]["body"]["choices"][0]["message"]["content"]
        id_to_response[id] = response
    
    # dump json to file with \n
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(id_to_response, f, indent=4, ensure_ascii=False)


def load_and_write_to_file(in_path, out_path):
    """Read JSON file containing responses from ChatGPT Batch API and write a specific response as plain text,
    ensuring that newline characters become actual line breaks.
    """
    with open(in_path, 'r', encoding='utf-8') as f:
        id_to_response = json.load(f)

    # Instead of json.dump, just write the response directly.
    # If necessary, replace literal "\n" sequences with actual newlines.
    response_text = id_to_response["85629"]
    
    # If your JSON load returns a string with proper newlines, this line may be unnecessary.
    response_text = response_text.replace("\\n", "\n")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(response_text)


def write_original_text_to_file():
    doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus_raw_google_ocr.csv"))

    doc_id = 85629
    doc = docs[doc_ids.index(doc_id)]
    with open("85629_original.txt", 'w', encoding='utf-8') as f:
        f.write(doc)


def create_new_corpus(filename):
    num = 25
    id_to_response = {}

    for i in range(num):
        with open(f"cleanup_{i}_output.json", 'r', encoding='utf-8') as f:
            id_to_response.update(json.load(f))
    
    # Write the combined responses to a new JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(id_to_response, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # process_response_file('gpt_responses_dev_small.jsonl', 'gpt_responses_dev_small.json')
    # create_jsonl_file()
    # create_jsonl_file_legal()
    for i in range(3, 25):
        process_response_file(f"cleanup_{i}_output.jsonl", f"cleanup_{i}_output.json")
    # load_and_write_to_file("cleanup_1_output.json", "85629_4o_mini.txt")
    # write_original_text_to_file()