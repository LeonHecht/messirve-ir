"""
Script to create the jsonl file for the ChatGPT Batch API requests.
Idea: Generate responses for 50k queries from the MS MARCO dataset.
Then finetune using InfoNCE loss with the responses instead of the queries.
"""
import sys
import os

def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Apply the path tweak before any project imports
configure_python_path()

import json
from src.utils.train_utils import get_msmarco_hard_negatives, get_msmarco_queries
import pickle
from tqdm import tqdm
import codecs
import ir_datasets
from src.utils.retrieval_utils import get_legal_dataset, get_legal_queries
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
    doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_py.csv"))
    query_ids, queries = get_legal_queries(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "queries_57.csv"))

    doc_dict = {str(doc_id): doc for doc_id, doc in zip(doc_ids, docs)}

    summary_len_dict = {}
    for doc_id, doc in zip(doc_ids, docs):
        len_doc = len(doc.split(" "))
        tokens_resumen = min(512, max(128, int(len_doc * 0.08)))
        summary_len_dict[doc_id] = tokens_resumen

    # print(doc_dict["37854"])

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
    
    sys_instruct = "I have an OCR output of a legal case document that contains a lot of repeated header and footer information. Could you please remove these headers, footers, and any extraneous lines (like lines that only have digits or are very short) from the text? Other than that, keep the text as it is. Only include the cleaned text in your answer."
    sys_instruct = (
    "Eres un abogado penalista paraguayo. "
    "Debes decidir si un DOCUMENTO es relevante o no a una CONSULTA. "
    '{"relevant":"yes|no","evidence":"..."} '
    "donde 'evidence' es la cita textual que justifica tu decisión en caso 'yes'. En caso 'no' pon 'None' como evidence."
    )

    assert type(sys_instruct) == str, "System instruction should be a string"

    user_template = [
        "### EXPEDIENTE COMPLETO",
        "{doc}",
        "### FIN DEL EXPEDIENTE",
        "",
        "Instrucciones finales:",
        "1. Resume en máximo {max_tokens} tokens.",
        "2. Sigue exactamente la plantilla indicada.",
        "3. Responde solo con el resumen en Markdown, sin comentarios extra."
    ]
    user_template = "\n".join(user_template)

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
        for did in doc_ids:
            requests.append({
                    "custom_id": f"{str(qid)}_{str(did)}",
                    "body": {
                        "model": "gpt-4.1-2025-04-14",
                        "messages": [
                            {"role": "system", "content": sys_instruct},
                            {"role": "user", "content": user_template.format(doc=doc_dict[did])}
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


def create_jsonl_original_annotation_mistral():
    doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_py.csv"))
    # query_ids, queries = get_legal_queries(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "queries_57.csv"))
    query_ids, queries = get_legal_queries(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "consultas_sinteticas_380.tsv"), header=0)

    # mandatory_list = [make_mandatory_list(query) for query in queries]

    doc_dict = {str(doc_id): doc for doc_id, doc in zip(doc_ids, docs)}

    with open(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "llm_annotation_runs.json"), "r") as f:
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
    qid_range = range(58, 505)

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

    for qid, query in zip(query_ids, queries):
        for did in preranked_doc_ids[str(qid)]:
            requests.append({
                    "custom_id": f"{str(qid)}_{str(did)}",
                    "body": {
                        "max_tokens": 256,
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
    file_path = f"batch_requests_mistral_v7_synthetic.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    return requests


def create_jsonl_inpars_mistral():
    # doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_py.csv"))
    doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_NEW.jsonl"))

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
    
    sys_instruct_v2_corpus_NEW = (
        "Eres un abogado penalista paraguayo, experto en análisis jurisprudencial y sistemas de búsqueda.\n"
        "A continuación, recibirás un documento legal completo. Genera tres consultas de tres tipos distintos, "
        "que un abogado podría ingresar en un buscador jurídico para encontrar exactamente este documento.\n"
        "Los tres tipos de consulta se especifican a continuación:\n"
        "1) Consulta corta: Usa muy pocas palabras para describir el caso. Ejemplos: 'Hurto agravado', 'Violación de derechos de adolescentes'.\n"
        "2) Consulta compleja: Incluya más detalles clave del caso. La consulta debe mencionar claramente los aspectos que caracterizan el caso y que permiten distinguirlo "
        "contra otros casos. No incluyas nombres de personas. Ejemplo: 'Recurso extraordinario de casación inadmisible por falta de cédula de notificación en tentativa de homicidio en Encarnación'.\n"
        "3) Consulta tipo pregunta: Formula una pregunta que este caso responde. Debe ser una pregunta que un abogado podría ingresar en un RAG Chat.\n"
        "Devuelve tu respuesta únicamente en formato JSON:\n"
        "{'Consulta corta': 'consulta', 'Consulta compleja': 'consulta', 'Consulta tipo pregunta': 'consulta'}\n"
        "No incluyas ningún otro texto o explicación adicional."
    )

    sys_instruct_v3 = (
        "Eres un abogado penalista paraguayo experto en análisis jurisprudencial y sistemas de búsqueda.\n"
        "A continuación, recibirás un documento legal completo. Genera una consulta que un abogado podría ingresar "
        "en un buscador jurídico para encontrar este documento.\n"
        "Las consultas deben mencionar claramente los aspectos que caracterizan el caso y que permiten distinguirlo "
        "contra otros casos. No menciones nombres de personas en tus consultas.\n"
        "Devuelve tu respuesta únicamente en formato de lista de strings:\n"
        "['consulta1', 'consulta2', ...]\n"
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
                        {"role": "system", "content": sys_instruct_v2_corpus_NEW},
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
    file_path = f"batch_requests_mistral_inpars_v2_corpus_NEW.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    return requests


def create_jsonl_summary_mistral():
    doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus.jsonl"))
    # query_ids, queries = get_legal_queries(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "queries_57.csv"))

    doc_dict = {str(doc_id): doc for doc_id, doc in zip(doc_ids, docs)}

    # summary_len_dict = {}
    # for doc_id, doc in zip(doc_ids, docs):
    #     len_doc = len(doc.split(" "))
    #     tokens_resumen = min(512, max(128, int(len_doc * 0.08)))
    #     summary_len_dict[doc_id] = tokens_resumen

    sys_instruct = [
      "Eres un relator jurídico experto de la Corte Suprema de Justicia del Paraguay.",
      "Tu tarea es elaborar un resumen FIEL, PRECISO y SIN ALUCINACIONES del expediente que recibirás.",
      "Lenguaje: español jurídico formal (neutro).",
      "Prohibido inventar hechos, fechas o citas. Si algo es ambiguo, indícalo como «[AMBIGUO]».",
      "Devuelve la salida **en formato Markdown** (no JSON) y sigue ESTRICTAMENTE la plantilla:",
      "",
      "### Identificación",
        "- Órgano / Sala: …",
        "- Expediente: …",
        "- Fecha de resolución: …",
        "- Tipo de proceso: …",
        "",
        "### Hechos clave",
        "Breve descripción cronológica (3–4 líneas).",
        "",
        "### Pretensiones y agravios",
        "Demandas principales y argumentos de cada parte.",
        "",
        "### Cuestiones sometidas",
        "Lista de preguntas jurídicas que el tribunal debía resolver.",
        "",
        "### Decisión",
        "- Admisibilidad: …",
        "- Fondo: …",
        "",
        "### Fundamentos esenciales",
        "Párrafo conciso con las razones jurídicas decisivas (citas de artículos, precedentes, etc.).",
        "",
        "### Resultado y efectos",
        "Qué se confirma, revoca o anula; órdenes posteriores y costas.",
        "",
        "### Citas relevantes",
        "Artículos del CPP, doctrina o jurisprudencia citados (máx. 5).",
        "",
      "Máximo 512 tokens en total para la respuesta.",
      "Usa tono desapasionado y evita opiniones.",
      "Si excedes el límite, recorta primero detalles menores de los fundamentos, nunca los hechos."
    ]
    sys_instruct = "\n".join(sys_instruct)
    print(sys_instruct)

    user_template = [
        "### EXPEDIENTE COMPLETO",
        "{doc}",
        "### FIN DEL EXPEDIENTE",
        "",
        "Instrucciones finales:",
        "1. Resume en máximo 512 tokens.",
        "2. Sigue exactamente la plantilla indicada.",
        "3. Responde solo con el resumen en Markdown, sin comentarios extra."
    ]
    user_template = "\n".join(user_template)

    requests = []
    for did in doc_ids:
        max_tokens = 1024
        requests.append({
                "custom_id": str(did),
                "body": {
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": sys_instruct.format(max_tokens=max_tokens)},
                        {"role": "user",
                        "content": user_template.format(doc=doc_dict[str(did)], max_tokens=max_tokens)}
                    ],
                    # "response_format": {
                    #     "type": "json_object",
                    # },
                }
            })
            
    assert len(requests) == 5000

        # write requests to jsonl file
    file_path = f"batch_requests_mistral_summary.jsonl"
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
            # evidence = None

            if relevant == "yes":
                label = 1
            elif relevant == "no":
                label = 0
            else:
                print(f"Invalid label for custom_id {custom_id}: {relevant}")
                continue
            # Write to qrels file
            if evidence is not None:
                f.write(f"{query_id}\t0\t{doc_id}\t{label}\t{evidence}\n")
            else:
                f.write(f"{query_id}\t0\t{doc_id}\t{label}\n")


def create_qrels_from_inpars_response(in_path, out_path_queries, out_path_qrels):
    """
    Convert an InPars-style response file into queries and qrels files.

    This function reads a JSON file containing document IDs mapped to
    serialized query dictionaries. For each document, it parses the queries
    and generates two output files:
    
    - A queries file: stores query IDs and their corresponding text.
    - A qrels file: stores query-document relevance judgments in TREC format.

    Each document is expected to have at least three queries, which are mapped
    to the suffixes "corta", "compleja", and "pregunta".

    Parameters
    ----------
    in_path : str
        Path to the input JSON file containing InPars-style responses. 
        The JSON should map each document ID (str) to a JSON-encoded dict of queries.
    out_path_queries : str
        Path to the output file where query IDs and their texts will be written.
        Each line has the format: "<qid>\\t<query_text>".
    out_path_qrels : str
        Path to the output file where query-document relevance judgments
        will be written. Each line has the format: "<qid>\\t0\\t<doc_id>\\t1".

    Notes
    -----
    - The function assumes that each document has at least 3 queries.
    - Query IDs are constructed as "<doc_id>_<mapping>", where `mapping` is one of:
        ["corta", "compleja", "pregunta"].
    - Lines with invalid or undecodable JSON queries are skipped with an error message.
    - Output files are overwritten if they already exist.

    Example
    -------
    Suppose the input JSON (`in_path`) contains:
        {
            "13995": "{\\"query1\\": \\"Hurto agravado\\", \\"query2\\": \\"Caso complejo\\", \\"query3\\": \\"¿Qué pena corresponde?\\"}"
        }

    The function will produce:
        Queries file (`out_path_queries`):
            13995_corta    Hurto agravado
            13995_compleja Caso complejo
            13995_pregunta ¿Qué pena corresponde?

        Qrels file (`out_path_qrels`):
            13995_corta    0    13995    1
            13995_compleja 0    13995    1
            13995_pregunta 0    13995    1
    """
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
            
            assert isinstance(queries, dict), (
                f"Value for doc_id '{doc_id}' is not a dict after parsing."
            )
            assert len(queries) >= 3, (
                f"Expected min. 3 elements for doc_id '{doc_id}', but got {len(queries)}."
            )

            mapping = ["corta", "compleja", "pregunta"]
            
            for i, query in enumerate(queries):
                qid = f"{doc_id}_{mapping[i]}"
                # Write to qrels file
                f_queries.write(f"{qid}\t{queries[query]}\n")
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


import pandas as pd
def filter_qrels_without_relevant_docs(in_path):
    qrels_df = pd.read_csv(
                in_path,
                sep="\t",                # TREC qrels are usually tab-separated
                names=["query_id", "iteration", "doc_id", "relevance", "evidence"],
                header=None,            # There's no header in qrels files
                dtype={"query_id": str, "iteration": int, "doc_id": str, "relevance": int, "evidence": str}
            )
    # 2. Find all query IDs that have at least one row with relevance > 0
    #    Here `> 0` is plain Python comparison, no .gt() needed.
    relevant_queries = qrels_df.loc[qrels_df["relevance"] > 0, "query_id"].unique()

    # 3. Keep only the rows whose query_id is in that list
    filtered_df = qrels_df[qrels_df["query_id"].isin(relevant_queries)]

    # 4. Write out to a new file
    out_path = Path(in_path).with_name(Path(in_path).stem + "_filtered.tsv")
    filtered_df.to_csv(out_path, sep="\t", index=False, header=False)


def filter_queries_without_relevant_docs(qrels_path, queries_path):
    filtered_qrels_df = pd.read_csv(
                qrels_path,
                sep="\t",                # TREC qrels are usually tab-separated
                names=["query_id", "iteration", "doc_id", "relevance"],
                header=None,            # There's no header in qrels files
                dtype={"query_id": str, "iteration": int, "doc_id": str, "relevance": int}
            )
    
    query_ids, queries = get_legal_queries(str(queries_path))
    query_dict = dict(zip(query_ids, queries))

    # Filter out query_ids and queries that do not appear in the qrels
    filtered_query_ids = filtered_qrels_df["query_id"].unique()
    filtered_query_dict = {qid: query for qid, query in query_dict.items() if qid in filtered_query_ids}

    # Save the filtered queries to a new TSV file
    with open(str(queries_path).replace(".tsv", "_filtered.tsv"), 'w', encoding='utf-8') as f:
        f.write("topic_id\tQuery\n")
        for qid, query in filtered_query_dict.items():
            f.write(f"{qid}\t{query}\n")


def create_corpus_from_summaries(summaries_file, output_file):
    with open(summaries_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, "w") as f:
        for doc_id, text in data.items():
            row = {"id": doc_id, "text": text}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    print(f"Corpus created with {len(data)} documents in {output_file}")


def split_inpars_v2_queries(in_path, out_path_corta, out_path_compleja, out_path_pregunta):
    """
    inpars v2 query file looks like this:
        13995_corta	Recurso de casación inadmisible
        13995_compleja	Recurso de casación inadmisible por falta de cédula de notificación en caso de tentativa de homicidio en Encarnación
        13995_pregunta	¿Por qué fue inadmisible el recurso de casación en un caso de tentativa de homicidio en Encarnación por falta de cédula de notificación?

    Goal of this function is to split it into three files:
        queries_corta.tsv
        queries_compleja.tsv
        queries_pregunta.tsv

    """
    with open(in_path, "r", encoding="utf-8") as in_queries, \
         open(out_path_corta, "w", encoding="utf-8") as out_corta, \
         open(out_path_compleja, "w", encoding="utf-8") as out_compleja, \
         open(out_path_pregunta, "w", encoding="utf-8") as out_pregunta:

        for line in in_queries:
            qid, query = line.strip().split("\t", 1)  # split only once
            if qid.endswith("_corta"):
                out_corta.write(f"{qid}\t{query}\n")
            elif qid.endswith("_compleja"):
                out_compleja.write(f"{qid}\t{query}\n")
            elif qid.endswith("_pregunta"):
                out_pregunta.write(f"{qid}\t{query}\n")
            else:
                print(f"⚠️ Unknown query id format: {qid}")


def split_inpars_v2_qrels(in_path, out_path_corta, out_path_compleja, out_path_pregunta):
    """
    inpars v2 qrel file looks like this:
        13995_corta	0	13995	1
        13995_compleja	0	13995	1
        13995_pregunta	0	13995	1

    Goal of this function is to split it into three files:
        qrels_corta.tsv
        qrels_compleja.tsv
        qrels_pregunta.tsv
    """
    with open(in_path, "r", encoding="utf-8") as in_qrels, \
         open(out_path_corta, "w", encoding="utf-8") as out_corta, \
         open(out_path_compleja, "w", encoding="utf-8") as out_compleja, \
         open(out_path_pregunta, "w", encoding="utf-8") as out_pregunta:

        for line in in_qrels:
            qid, zero, doc_id, relevance = line.strip().split("\t")
            if qid.endswith("_corta"):
                out_corta.write(f"{qid}\t{zero}\t{doc_id}\t{relevance}\n")
            elif qid.endswith("_compleja"):
                out_compleja.write(f"{qid}\t{zero}\t{doc_id}\t{relevance}\n")
            elif qid.endswith("_pregunta"):
                out_pregunta.write(f"{qid}\t{zero}\t{doc_id}\t{relevance}\n")
            else:
                print(f"⚠️ Unknown query id format: {qid}")


def dedup_inpars_queries(in_path, out_path):
    qids, queries = get_legal_queries(in_path)

    # map query → first qid
    query_to_qid = {}
    for qid, query in zip(qids, queries):
        if query not in query_to_qid:
            query_to_qid[query] = qid

    print(f"Original: {len(queries)}")
    print(f"Deduped: {len(query_to_qid)}")

    with open(out_path, "w", encoding="utf-8") as f:
        for query, qid in query_to_qid.items():
            f.write(f"{qid}\t{query}\n")


import pandas as pd

def filter_qrels_by_deduped_queries(dedup_queries_path, qrels_path, out_path):
    """
    Filter a qrels.tsv file so it only contains entries whose qid
    appears in the deduplicated queries file.

    Parameters
    ----------
    dedup_queries_path : str
        Path to deduplicated queries TSV (qid \t query).
    qrels_path : str
        Path to qrels TSV (qid \t run \t doc_id \t label).
    out_path : str
        Destination TSV with filtered qrels.
    """
    # load deduped queries
    df_queries = pd.read_csv(dedup_queries_path, sep="\t", names=["qid", "query"], dtype=str)
    keep_qids = set(df_queries["qid"].tolist())

    # load qrels
    df_qrels = pd.read_csv(qrels_path, sep="\t", names=["qid", "run", "doc_id", "label"], dtype=str)

    # filter
    df_filtered = df_qrels[df_qrels["qid"].isin(keep_qids)]

    print(f"Original qrels: {len(df_qrels)}")
    print(f"Filtered qrels: {len(df_filtered)}")

    # save
    df_filtered.to_csv(out_path, sep="\t", index=False, header=False)
    print(f"[✓] Written filtered qrels to {out_path}")


from typing import Optional
import random

def filter_inpars_v2_with_extra(
    in_queries: str,
    in_qrels: str,
    out_queries: str,
    out_qrels: str,
    extra_ratio: float = 1.0,   # add this fraction of non-penal qids relative to penal count
    seed: Optional[int] = 42,
):
    """
    Keep all penal qids (base id in penal corpus), and add a random sample of
    non-penal qids equal to round(len(penal_qids) * extra_ratio).

    Assumes `in_queries` is TSV: qid \t query
            `in_qrels`   is TSV: qid \t run \t doc_id \t label
    """
    if seed is not None:
        random.seed(seed)

    # 1) Penal corpus doc_ids (base ids)
    penal_doc_ids, _ = get_legal_dataset(
        os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus.jsonl")
    )
    penal_doc_ids = set(map(str, penal_doc_ids))

    # 2) Load queries & qrels
    df_queries = pd.read_csv(in_queries, sep="\t", names=["qid", "query"], dtype=str)
    df_qrels   = pd.read_csv(in_qrels,   sep="\t", names=["qid", "run", "doc_id", "label"], dtype=str)

    # 3) Split qids into penal vs non-penal by base id (before first "_")
    def base_id(qid: str) -> str:
        return qid.split("_", 1)[0]

    all_qids = set(df_queries["qid"].tolist())
    penal_qids = {qid for qid in all_qids if base_id(qid) in penal_doc_ids}
    nonpenal_qids = list(all_qids - penal_qids)  # list for sampling

    # 4) Decide how many non-penal to add
    add_n = int(round(len(penal_qids) * float(extra_ratio)))
    add_n = max(0, min(add_n, len(nonpenal_qids)))  # clamp

    # 5) Sample non-penal qids that actually exist in both queries *and* qrels (safer)
    nonpenal_qids_with_qrels = list(set(nonpenal_qids) & set(df_qrels["qid"].tolist()))
    if add_n > len(nonpenal_qids_with_qrels):
        add_n = len(nonpenal_qids_with_qrels)
    extra_qids = set(random.sample(nonpenal_qids_with_qrels, add_n)) if add_n > 0 else set()

    # 6) Final keep set
    keep_qids = penal_qids | extra_qids

    # 7) Filter frames
    df_queries_out = df_queries[df_queries["qid"].isin(keep_qids)]
    df_qrels_out   = df_qrels[df_qrels["qid"].isin(keep_qids)]

    # (Optional) If you also want to ensure qrels doc_ids are inside penal corpus, uncomment:
    # df_qrels_out = df_qrels_out[df_qrels_out["doc_id"].isin(penal_doc_ids)]

    # 8) Log stats
    print(f"Penal qids: {len(penal_qids)}")
    print(f"Non-penal available: {len(nonpenal_qids)}  | sampled: {len(extra_qids)} (ratio={extra_ratio})")
    print(f"→ Final queries: {len(df_queries_out)} / {len(df_queries)}")
    print(f"→ Final qrels:   {len(df_qrels_out)} / {len(df_qrels)}")

    # 9) Save
    df_queries_out.to_csv(out_queries, sep="\t", index=False, header=False)
    print(f"[✓] Written filtered queries to {out_queries}")
    df_qrels_out.to_csv(out_qrels, sep="\t", index=False, header=False)
    print(f"[✓] Written filtered qrels to {out_qrels}")



if __name__ == "__main__":
    # dedup_inpars_queries(
    #     os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "mistral_inpars_v2_corpus_NEW_queries.tsv"),
    #     os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "mistral_inpars_v2_corpus_NEW_queries_dedup.tsv")
    # )

    # filter_qrels_by_deduped_queries(
    #     dedup_queries_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "mistral_inpars_v2_corpus_NEW_queries_dedup.tsv"),
    #     qrels_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "mistral_inpars_v2_corpus_NEW_qrels.tsv"),
    #     out_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "mistral_inpars_v2_corpus_NEW_qrels_dedup.tsv")
    # )

    filter_inpars_v2_with_extra(
        in_queries=os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "mistral_inpars_v2_corpus_NEW_queries_corta_dedup.tsv"),
        in_qrels=os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "mistral_inpars_v2_corpus_NEW_qrels_corta_dedup.tsv"),
        out_queries=os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "mistral_inpars_v2_corpus_NEW_queries_corta_dedup_penal_mixed_1-1.tsv"),
        out_qrels=os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "mistral_inpars_v2_corpus_NEW_qrels_corta_dedup_penal_mixed_1-1.tsv"),
        extra_ratio=1.0
    )

    # requests = create_jsonl_original_annotation_mistral()
    # requests = create_jsonl_inpars_mistral()
    # requests = create_jsonl_summary_mistral()

    # process_response_file(
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "external" / "mistral" / "BatchAPI_outputs" / "mistral_inpars_v2_corpus_NEW_out.jsonl",
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "external" / "mistral" / "BatchAPI_outputs" / "mistral_inpars_v2_corpus_NEW_out_processed.json"
    # )

    # create_corpus_from_summaries(
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "external" / "mistral" / "BatchAPI_outputs" / "mistral_summary_1024_processed.json",
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "corpus_mistral_summaries_1024.jsonl"
    # )

    # create_qrels_from_inpars_response(
    #     in_path=Path(STORAGE_DIR) / "legal_ir" / "data" / "external" / "mistral" / "BatchAPI_outputs" / "mistral_inpars_v2_corpus_NEW_out_processed.json",
    #     out_path_queries=Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "mistral_inpars_v2_corpus_NEW_queries.tsv",
    #     out_path_qrels=Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "mistral_inpars_v2_corpus_NEW_qrels.tsv"
    # )

    # split_inpars_v2_queries(
    #     in_path=Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "mistral_inpars_v2_corpus_NEW_queries.tsv",
    #     out_path_corta=Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "mistral_inpars_v2_corpus_NEW_queries_corta.tsv",
    #     out_path_compleja=Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "mistral_inpars_v2_corpus_NEW_queries_compleja.tsv",
    #     out_path_pregunta=Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "mistral_inpars_v2_corpus_NEW_queries_pregunta.tsv"
    # )

    # split_inpars_v2_qrels(
    #     in_path=Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "mistral_inpars_v2_corpus_NEW_qrels.tsv",
    #     out_path_corta=Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "mistral_inpars_v2_corpus_NEW_qrels_corta.tsv",
    #     out_path_compleja=Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "mistral_inpars_v2_corpus_NEW_qrels_compleja.tsv",
    #     out_path_pregunta=Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "mistral_inpars_v2_corpus_NEW_qrels_pregunta.tsv"
    # )

    # create_qrels_from_response(
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "external" / "mistral" / "BatchAPI_outputs" / "annotation_synthetic_mistral-small-2501_processed.json",
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "qrels_synthetic_mistral-small-2501_raw_evidence.tsv"
    # )

    # filter_qrels_without_relevant_docs(
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "qrels_synthetic_mistral-small-2501_raw_evidence.tsv"
    # )

    # filter_queries_without_relevant_docs(
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "qrels_synthetic_mistral-small-2501_filtered.tsv",
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "consultas_sinteticas_380_unfiltered.tsv"
    # )