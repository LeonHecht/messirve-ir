"""
Script to create the jsonl file for the ChatGPT Batch API requests.
Idea: Generate responses for 50k queries from the MS MARCO dataset.
Then finetune using InfoNCE loss with the responses instead of the queries.
"""
import sys
sys.path.append("home/leon/tesis/messirve-ir")

import json
from src.utils.train_utils import get_msmarco_hard_negatives, get_msmarco_queries
import pickle
from tqdm import tqdm
import codecs
import ir_datasets
from src.utils.retrieval_utils import get_legal_dataset, get_legal_queries
import os
from config.config import STORAGE_DIR
from pathlib import Path


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


def test_request(request):
    from openai import OpenAI

    # 1) Initialize client
    client = OpenAI()

    response = client.responses.create(
        model = request["body"]["model"],
        input = request["body"]["messages"],
        text = request["body"]["text"],
    )

    # 5) Parse and print the JSON
    output = response.output_text
    annotation = json.loads(output)
    print(json.dumps(annotation, ensure_ascii=False, indent=2))


def create_jsonl_file_legal():
    from pydantic import BaseModel

    class Annotation(BaseModel):
        relevant: str
        evidence: str

    doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_py.csv"))
    query_ids, queries = get_legal_queries(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "queries_57.csv"))

    doc_dict = {str(doc_id): doc for doc_id, doc in zip(doc_ids, docs)}

    print(doc_dict["37854"])

    with open(Path("data") / "processed" / "doctag_runs.json", "r") as f:
        doctag_runs = json.load(f)["run"]
    
    # Structure of doctag_runs.json:
    #   {
    #   "run": [
    #     {
    #       "topic_id": "1:Hurto",
    #       "documents": [
    #         {
    #           "document_id": "37854"
    #         },
    #         {
    #           "document_id": "39988"
    #         },
    qid_range = range(1, 58)

    preranked_doc_ids = {}
    for topic in doctag_runs:
        qid, _ = topic["topic_id"].split(":")
        assert int(qid) in qid_range, f"qid {qid} is not in the range {qid_range}"
        doc_ids_preranked = [dict_["document_id"] for dict_ in topic["documents"]]
        preranked_doc_ids[qid] = doc_ids_preranked
    
    # sys_instruct = "I have an OCR output of a legal case document that contains a lot of repeated header and footer information. Could you please remove these headers, footers, and any extraneous lines (like lines that only have digits or are very short) from the text? Other than that, keep the text as it is. Only include the cleaned text in your answer."
    sys_instruct = (
    "Eres un abogado penalista paraguayo. "
    "Debes decidir si un DOCUMENTO es relevante o no a una CONSULTA. "
    '{"relevant":"yes|no","evidence":"..."} '
    "donde 'evidence' es la cita textual que justifica tu decisión en caso 'yes'. En caso 'no' pon 'None' como evidence."
    )

    assert type(sys_instruct) == str, "System instruction should be a string"

    user_template = (
        "CONSULTA: {query}\n\n"
        "DOCUMENTO: {doc}\n\n"
        "¿El documento es relevante?"
    )

    schema = {
        "type": "object",
        "properties": {
            "relevant": {
                "type": "string",
                "enum": ["yes", "no"]
            },
            "evidence": {
                "type": "string"
            }
        },
        "required": ["relevant", "evidence"],
        "additionalProperties": False
    }

    requests = []

    for qid, query in zip(query_ids, queries):
        for did in preranked_doc_ids[qid]:
            requests.append({
                    "custom_id": f"{str(qid)}_{str(did)}",
                    "body": {
                        "model": "gpt-4.1-2025-04-14",
                        "messages": [
                            {"role": "system", "content": sys_instruct},
                            {"role": "user", "content": user_template.format(query=query, doc=doc_dict[did])}
                        ],
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "relevance_annotation",
                                "schema": schema,
                                "strict": True
                            }
                        }
                    }
                })
            
    print(len(requests))
    
    # Write to JSONL
    out_path = "annotation_requests.jsonl"
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
        
    return requests

    # limit = 200
    # counter = 0

    # for i in range(len(requests) // limit):
    #     # Write requests to a .jsonl file
    #     with open(f'batch_requests_legal_cleanup_{i}.jsonl', 'w', encoding="utf-8") as f:
    #         r = i * limit
    #         while r < len(requests):
    #             if counter >= limit:
    #                 counter = 0
    #                 break
    #             f.write(json.dumps(requests[r], ensure_ascii=False) + '\n')
    #             counter += 1
    #             r += 1


def make_mandatory_list(query: str):
    # split by comma and strip
    parts = [p.strip() for p in query.split(",")]
    # lower‑case for matching
    return parts


# build the per‑request user prompt on the fly, inserting
# a bullet list of mandatory terms for that query
def make_user_msg(query, mandatory_terms, doc):
    bullets = "\n".join(f"- {t}" for t in mandatory_terms)
    return (
        f"CONSULTA: {query}\n\n"
        "CONDICIONES:\n"
        f"{bullets}\n\n"
        "DOCUMENTO:\n"
        f"{doc}\n\n"
        "¿Cumple TODAS las condiciones?  "
        "Responde SOLO en JSON como se indicó."
    )


def create_jsonl_annotation_mistral():
    doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_py.csv"))
    query_ids, queries = get_legal_queries(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "queries_57.csv"))

    mandatory_list = [make_mandatory_list(query) for query in queries]

    doc_dict = {str(doc_id): doc for doc_id, doc in zip(doc_ids, docs)}

    print(doc_dict["37854"])

    with open(Path("data") / "processed" / "doctag_runs.json", "r") as f:
        doctag_runs = json.load(f)["run"]
    
    # Structure of doctag_runs.json:
    #   {
    #   "run": [
    #     {
    #       "topic_id": "1:Hurto",
    #       "documents": [
    #         {
    #           "document_id": "37854"
    #         },
    #         {
    #           "document_id": "39988"
    #         },
    qid_range = range(1, 58)

    preranked_doc_ids = {}
    for topic in doctag_runs:
        qid, _ = topic["topic_id"].split(":")
        assert int(qid) in qid_range, f"qid {qid} is not in the range {qid_range}"
        doc_ids_preranked = [dict_["document_id"] for dict_ in topic["documents"]]
        preranked_doc_ids[qid] = doc_ids_preranked
    
    # sys_instruct = "I have an OCR output of a legal case document that contains a lot of repeated header and footer information. Could you please remove these headers, footers, and any extraneous lines (like lines that only have digits or are very short) from the text? Other than that, keep the text as it is. Only include the cleaned text in your answer."
    
    # Sys Instruct 1 for LLM Annotation: 
    #  sys_instruct = (
    # "Eres un abogado penalista paraguayo. "
    # "Debes decidir si un DOCUMENTO es relevante o no a una CONSULTA. "
    # '{"relevant":"yes|no","evidence":"..."} '
    # "donde 'evidence' es la cita textual que justifica tu decisión en caso 'yes'. En caso 'no' pon 'None' como evidence."
    # )

    # Sys Instruct 2 for LLM Annotation:
    sys_instruct = (
        "Eres un abogado penalista paraguayo experto en análisis jurisprudencial. "
        "Tu tarea es decidir si un DOCUMENTO es relevante o no a una CONSULTA. "
        "Las consultas pueden ser SIMPLES (sin coma) o COMPUESTAS (con una coma ',').\n\n"
        "- CONSULTA SIMPLE: (por ej. 'Hurto') el documento es relevante si es claramente relevante a la consulta (Se estricto).\n"
        "- CONSULTA COMPUESTA ('consulta general, subtema'): el documento SOLO es relevante si cumple exactamente con el subtema especificado después de la coma. Por ejemplo, si la consulta es 'Derecho a la defensa, Doble instancia', solo documentos que mencionen claramente el concepto 'Doble instancia' dentro del contexto de 'Derecho a la defensa' serán relevantes.\n\n"
        "Debes responder siempre en ESPAÑOL y en este formato JSON estricto:\n"
        '{"relevant":"yes|no","evidence":"..."}\n'
        "- 'relevant': 'yes' o 'no' según tu decisión.\n"
        "- 'evidence': si respondes 'yes', incluye ÚNICAMENTE una cita textual EXACTA del documento que justifique claramente tu decisión. Si respondes 'no', coloca 'None'."
    )

    # User Template for LLM Annotation v1, v3:
    user_template = (
        "CONSULTA: {query}\n\n"
        "DOCUMENTO: {doc}\n\n"
        "¿El documento es relevante?"
    )
    
    requests = []

    for qid, query, mand in zip(query_ids, queries, mandatory_list):
        for did in preranked_doc_ids[str(qid)]:
            requests.append({
                    "custom_id": f"{str(qid)}_{str(did)}",
                    "body": {
                        "messages": [
                            {"role": "system", "content": sys_instruct},
                            {"role": "user",
                            #  "content": make_user_msg(query, mand, doc_dict[str(did)])}
                             "content": user_template.format(query=query, doc=doc_dict[str(did)])}
                        ],
                        "response_format": {
                            "type": "json_object",
                        },
                    }
                })
            
    print(len(requests))

    # write requests to jsonl file
    file_path = f"batch_requests_mistral_v7_sintetic.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    return requests


def create_jsonl_inpars_mistral():
    doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_py.csv"))

    # v1
    sys_instruct = (
        "Eres un abogado penalista paraguayo experto en análisis jurisprudencial y sistemas de búsqueda.\n"
        "A continuación, recibirás un documento legal completo. Genera una o más consultas (máximo 3) breves pero específicas "
        "que un abogado podría ingresar en un buscador jurídico para encontrar exactamente este documento.\n"
        "Las consultas deben mencionar claramente los aspectos que caracterizan el caso y que permiten distinguirlo "
        "contra otros casos. No menciones nombres de personas en tus consultas.\n"
        "Devuelve tu respuesta únicamente en formato de lista de strings:\n"
        "['consulta1', 'consulta2', ...]\n"
        "No incluyas ningún otro texto o explicación adicional."
    )

    # v2
    sys_instruct_v2 = (
        "Eres un abogado penalista paraguayo, experto en análisis jurisprudencial y sistemas de búsqueda.\n"
        "A continuación, recibirás un documento legal completo. Genera una o más consultas breves pero específicas de tres tipos distintos, "
        "que un abogado podría ingresar en un buscador jurídico para encontrar exactamente este documento.\n"
        "Las consultas deben mencionar claramente los aspectos que caracterizan el caso y que permiten distinguirlo de otros casos. "
        "Los tres tipos de consulta se especifican a continuación:\n"
        "1) Consulta corta: Usa muy pocas palabras para describir el caso. Ejemplos: 'Hurto agravado', 'Violación de derechos de adolescentes'.\n"
        "2) Consulta compleja: Describe el caso lo más detalladamente posible usando más palabras. No incluyas nombres de personas. "
        "Ejemplo: 'casación directa extemporánea apelación especial previa robo agravado 7 años central'.\n"
        "3) Consulta personalizada: Describe el caso con el mayor detalle posible, incluyendo los nombres de las personas involucradas.\n"
        "Devuelve tu respuesta únicamente en formato JSON:\n"
        "{'Consulta corta': ['consulta1', 'consulta2', ...], 'Consulta compleja': ['consulta1', 'consulta2', ...], 'Consulta personalizada': ['consulta1', 'consulta2', ...]}\n"
        "No incluyas ningún otro texto o explicación adicional."
    )


    # User Template for LLM Annotation v1, v3:
    user_template = (
        "DOCUMENTO:\n"
        "{doc}"
    )
    
    requests = []

    for did, doc_text in zip(doc_ids, docs):
        requests.append({
                "custom_id": f"{str(did)}",
                "body": {
                    "max_tokens": 256,
                    "messages": [
                        {"role": "system", "content": sys_instruct},
                        {"role": "user",
                            "content": user_template.format(doc=doc_text)}
                    ],
                    "response_format": {
                            "type": "json_object",
                    },
                }
            })
            
    print(len(requests))

    # write requests to jsonl file
    file_path = f"batch_requests_mistral_inpars.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    return requests


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


def create_qrels_from_response(in_path, out_path):
    with open(in_path, 'r', encoding='utf-8') as f:
        response = json.load(f)
    
    # Write to qrels file
    with open(out_path, 'a', encoding='utf-8') as f:
        for custom_id, content in response.items():
            query_id, doc_id = custom_id.split("_")
            
            # Parse the content
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for custom_id {custom_id}: {content}")
                continue

            # Extract relevant information
            relevant = content.get("relevant")
            evidence = content.get("evidence")

            if relevant == "yes":
                label = 1
            elif relevant == "no":
                label = 0
            else:
                print(f"Invalid label for custom_id {custom_id}: {relevant}")
                continue
            # Write to qrels file
            f.write(f"{query_id}\t0\t{doc_id}\t{label}\t{evidence}\n")


def create_qrels_from_inpars_response(in_path, out_path_queries, out_path_qrels):
    with open(in_path, 'r', encoding='utf-8') as f:
        response = json.load(f)
    
    # Write to qrels file
    with open(out_path_queries, 'w', encoding='utf-8') as f_queries, open(out_path_qrels, 'w', encoding='utf-8') as f_qrels:
        for doc_id, queries in response.items():
            # if not queries.strip().endswith('"]'):
            #     print(f"queries before appending ']' for doc_id {doc_id}: {queries}")
            #     queries += '"]'
            #     print(f'Appended "] to queries for doc_id {doc_id}.')
            # Parse the queries
            try:
                queries = json.loads(queries)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for doc_id {doc_id}: {e}")
                print(f"Error decoding JSON for custom_id {doc_id}: {queries}")
                continue
            
            assert isinstance(queries, list), (
                f"Value for doc_id '{doc_id}' is not a list after parsing."
            )
            assert len(queries) >= 3, (
                f"Expected min. 3 elements for doc_id '{doc_id}', but got {len(queries)}."
            )
            
            for i, query in enumerate(queries):
                qid = f"{doc_id}_Q{i+1}"
                # Write to qrels file
                f_queries.write(f"{qid}\t{query}\n")
                # Write to qrels file (qid  0   doc_id  label)
                f_qrels.write(f"{qid}\t0\t{doc_id}\t1\n")

    print(f"Queries written to {out_path_queries} and qrels written to {out_path_qrels}")


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
    # requests = create_jsonl_annotation_mistral()
    # requests = create_jsonl_inpars_mistral()

    # process_response_file(
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "external" / "mistral" / "BatchAPI_outputs" / "inpars_mistral-small-2501.jsonl",
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "external" / "mistral" / "BatchAPI_outputs" / "inpars_mistral-small-2501_processed.json"
    # )

    create_qrels_from_inpars_response(
        in_path=Path(STORAGE_DIR) / "legal_ir" / "data" / "external" / "mistral" / "BatchAPI_outputs" / "inpars_mistral-small-2501_processed.json",
        out_path_queries=Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "inpars_mistral-small-2501_queries.tsv",
        out_path_qrels=Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "inpars_mistral-small-2501_qrels.tsv"
    )

    # create_qrels_from_response(
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "external" / "mistral" / "BatchAPI_outputs" / "annotation_57_mistral-large-2411_v7_processed.json",
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "qrels_mistral-large-2411_v7.tsv"
    # )