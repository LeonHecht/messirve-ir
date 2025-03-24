"""
Script to create the jsonl file for the ChatGPT Batch API requests.
Idea: Generate responses for 50k queries from the MS MARCO dataset.
Then finetune using InfoNCE loss with the responses instead of the queries.
"""

import json
from utils.train_utils import get_msmarco_hard_negatives, get_msmarco_queries
import pickle
from tqdm import tqdm
import codecs
import ir_datasets


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


def process_response_file(in_path, out_path):
    """ Read json file containing responses from ChatGPT Batch API.
        Then iterate over the responses and return a list of responses. 
    """
    with open(in_path, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
    
    qid_to_response = {}
    for item in items:
        qid = item["custom_id"]
        response = item["response"]["body"]["choices"][0]["message"]["content"]
        qid_to_response[qid] = response
    
    # dump json to file with \n
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(qid_to_response, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    process_response_file('gpt_responses_dev_small.jsonl', 'gpt_responses_dev_small.json')
    # create_jsonl_file()