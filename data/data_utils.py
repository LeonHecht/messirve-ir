import os
import sys
import json
import pandas as pd

def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    print(f"Adding {project_root} to sys.path")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()

from config.config import STORAGE_DIR


def corpus_json_to_jsonl(input_path, output_path):
    """
    Lee un JSON de mapeo y escribe un JSONL con campos 'id' y 'text'.

    Parameters
    ----------
    input_path : str
        Ruta al fichero JSON de entrada.
    output_path : str
        Ruta al fichero JSONL de salida.

    Raises
    ------
    ValueError
        Si el JSON de entrada no es un diccionario.
    """
    with open(input_path, encoding="utf-8") as f_in:
        data = json.load(f_in)

    if not isinstance(data, dict):
        raise ValueError("El JSON de entrada debe ser un objeto (dict) "
                         f"pero se leyó: {type(data)}")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for key, value in data.items():
            record = {"id": key, "text": value}
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")



def merge_and_shuffle(input_files, output_file):
    """
    Read multiple JSONL files, shuffle their contents, and write to an output file.

    Parameters
    ----------
    input_files : list of str
        Paths to the input JSONL files.
    output_file : str
        Path to the output JSONL file where shuffled lines will be saved.

    Returns
    -------
    None
    """
    import random
    
    all_lines = []
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as infile:
            all_lines.extend(infile.readlines())

    random.shuffle(all_lines)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(all_lines)


def add_chunk_headings_to_corpus(input_path, output_path, chunk_size=256):
    """
    Chunk texts from a JSONL corpus into fixed-size word chunks and
    write out a new JSONL where each entry’s `text` contains all
    chunk sections annotated with a header.

    Parameters
    ----------
    input_path : str
        Path to the input JSONL file. Each line must be a JSON object
        with keys "id" and "text".
    output_path : str
        Path to the output JSONL file to write chunked entries.
    chunk_size : int, optional
        Number of words per chunk (default is 512).

    Returns
    -------
    None
        Writes the chunked corpus to `output_path` in JSONL format.
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            doc = json.loads(line)
            doc_id = str(doc['id'])
            words = doc['text'].split()
            total = len(words)
            sections = []

            for idx, start in enumerate(range(0, total, chunk_size), start=1):
                end = min(start + chunk_size, total)
                chunk_words = words[start:start + chunk_size]
                header = (
                    f"## CHUNK_{idx:03d} "
                    f"(words {start + 1}-{end})"
                )
                chunk_text = ' '.join(chunk_words)
                sections.append(f"{header}\n\n{chunk_text}")

            new_doc = {
                'id': doc_id,
                'text': '\n\n'.join(sections)
            }
            outfile.write(json.dumps(new_doc, ensure_ascii=False) + '\n')


import os
from tqdm import tqdm
import numpy as np

from config.config import STORAGE_DIR

from src.utils.retrieval_utils import (
    get_legal_dataset,
    get_legal_queries,
)


def compute_mean_jaccard(queries, docs):
    """
    Calcula la superposición tipo Jaccard (|intersección| / |unión|)
    entre un conjunto de consultas y un conjunto de documentos.

    Parameters
    ----------
    queries : list[str]
        Lista de cadenas, cada una es una consulta.
    docs : list[str]
        Lista de cadenas, cada una es un documento completo o resumen.

    Returns
    -------
    float
        Valor medio de la similitud Jaccard entre todas las combinaciones
        (query, doc). Si hay Q consultas y D documentos, son Q*D pares.
    """
    # Tokenizar consultas y documentos una sola vez
    # query_sets = [tokenize_simple(q) for q in queries]
    # doc_sets = [tokenize_simple(d) for d in docs]
    query_sets = [set(q.split()) for q in queries]
    doc_sets = [set(d.split()) for d in docs]

    total = 0.0
    count = 0

    # Bucle anidado, pero cada operación de intersección/ unión es O(min(|A|, |B|))
    for qset in tqdm(query_sets, desc="Iterando consultas"):
        for dset in doc_sets:
            # Calcular intersección y unión de conjuntos
            inter = qset & dset
            union = qset | dset
            if union:
                total += len(inter) / len(union)
            else:
                total += 0.0
            count += 1

    return total / count if count else 0.0


# def download_msmarco_doc():
#     import ir_datasets
#     dataset = ir_datasets.load("msmarco-document/train")
#     for doc in dataset.docs_iter():
#         doc # namedtuple<doc_id, url, title, body>


def main2():
    # Ajusta estas rutas según tu estructura
    base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data")

    # Carga de consultas
    qids_54, queries_54 = get_legal_queries(
        os.path.join(base_dir, "corpus", "queries_54.tsv")
    )
    qids_inpars, queries_inpars = get_legal_queries(
        os.path.join(
            base_dir, "corpus", "inpars_mistral-small-2501_queries_Q1.tsv"
        )
    )
    qids_synthetic, queries_synthetic = get_legal_queries(
        os.path.join(base_dir, "corpus", "consultas_sinteticas_380_filtered.tsv")
    )

    # Carga de documentos completos y resumidos
    _, docs_full = get_legal_dataset(
        os.path.join(base_dir, "corpus", "corpus.jsonl")
    )
    _, docs_summary = get_legal_dataset(
        os.path.join(base_dir, "corpus", "corpus_mistral_summaries_1024.jsonl")
    )

    # Calculamos la media Jaccard para cada par (queries_54, docs_full) y (queries_54, docs_summary)
    mean_54_full = compute_mean_jaccard(queries_54, docs_full)
    mean_54_sum = compute_mean_jaccard(queries_54, docs_summary)
    print(
        f"Media Jaccard ds-54 | full: {mean_54_full:.4f}, summary: {mean_54_sum:.4f}"
    )

    # Para InPars
    mean_inpars_full = compute_mean_jaccard(queries_inpars, docs_full)
    mean_inpars_sum = compute_mean_jaccard(queries_inpars, docs_summary)
    print(
        f"Media Jaccard InPars | full: {mean_inpars_full:.4f}, summary: {mean_inpars_sum:.4f}"
    )

    # Para consultas sintéticas
    mean_synth_full = compute_mean_jaccard(queries_synthetic, docs_full)
    mean_synth_sum = compute_mean_jaccard(queries_synthetic, docs_summary)
    print(
        f"Media Jaccard Synthetic | full: {mean_synth_full:.4f}, summary: {mean_synth_sum:.4f}"
    )


def tokenize_simple(text):
    """
    Tokeniza un texto en palabras sencillas, eliminando puntuación básica
    y pasando a minúsculas.

    Parameters
    ----------
    text : str
        Texto a tokenizar.

    Returns
    -------
    set[str]
        Conjunto de tokens únicos.
    """
    tokens = (
        text.lower()
        .replace("¿", " ")
        .replace("?", " ")
        .replace("¡", " ")
        .replace("!", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .split()
    )
    return set(tokens)


def load_qrels(qrels_path):
    """
    Carga un archivo TSV de qrels con formato qid,docid,label.

    Parameters
    ----------
    qrels_path : str
        Ruta al archivo TSV con las columnas: qid, docid, label.

    Returns
    -------
    dict[str, list[tuple[str, int]]]
        Diccionario que mapea cada qid a una lista de tuplas (docid, label).
    """
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                print(f"Formato incorrecto en línea: {line.strip()}")
                continue
            qid, run, docid, label = parts
            label = int(label)
            qrels.setdefault(qid, []).append((docid, label))
    return qrels


def query_token_recall(qset, dset):
    return len(qset & dset) / len(qset) if qset else 0.0


def compute_jaccard(qset, dset):
    """
    Calcula la similitud Jaccard entre dos conjuntos de tokens.

    Parameters
    ----------
    qset : set[str]
        Conjunto de tokens de la consulta.
    dset : set[str]
        Conjunto de tokens del documento.

    Returns
    -------
    float
        Valor Jaccard = |intersección| / |unión|, o 0 si la unión está vacía.
    """
    inter = qset & dset
    union = qset | dset
    return len(inter) / len(union) if union else 0.0


def main3():
    # Ajusta estas rutas según tu estructura
    base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data")

    # Carga de consultas (solo textos, usaremos qids en qrels)
    qids_54, queries_54 = get_legal_queries(
        os.path.join(base_dir, "corpus", "queries_54.tsv")
    )
    qids_inpars, queries_inpars = get_legal_queries(
        os.path.join(base_dir, "corpus", "inpars_mistral-small-2501_queries_Q1.tsv")
    )
    qids_synthetic, queries_synthetic = get_legal_queries(
        os.path.join(base_dir, "corpus", "consultas_sinteticas_380_filtered.tsv")
    )

    # Carga de documentos completos y resumidos
    dids_full, docs_full = get_legal_dataset(
        os.path.join(base_dir, "corpus", "corpus.jsonl")
    )
    dids_summary, docs_summary = get_legal_dataset(
        os.path.join(base_dir, "corpus", "corpus_mistral_summaries_1024.jsonl")
    )

    # Crear mapas id->texto e id->token_set
    doc_texts_full = dict(zip(dids_full, docs_full))
    doc_texts_summary = dict(zip(dids_summary, docs_summary))

    # Tokenizar todos los documentos una sola vez
    doc_tokens_full = {did: tokenize_simple(text) for did, text in doc_texts_full.items()}
    doc_tokens_summary = {did: tokenize_simple(text) for did, text in doc_texts_summary.items()}

    # Tokenizar consultas
    query_tokens_54 = {qid: tokenize_simple(q) for qid, q in zip(qids_54, queries_54)}
    query_tokens_inpars = {qid: tokenize_simple(q) for qid, q in zip(qids_inpars, queries_inpars)}
    query_tokens_synth = {qid: tokenize_simple(q) for qid, q in zip(qids_synthetic, queries_synthetic)}

    # Cargar qrels (archivo TSV con columnas qid,docid,label)
    qrels_54 = load_qrels(os.path.join(base_dir, "annotations", "qrels_54.tsv"))
    qrels_inpars = load_qrels(os.path.join(base_dir, "annotations", "inpars_mistral-small-2501_qrels_Q1.tsv"))
    qrels_synth = load_qrels(os.path.join(base_dir, "annotations", "qrels_synthetic_mistral-small-2501_filtered.tsv"))

    # Lista para acumular resultados
    records = []

    # Función auxiliar para procesar un dataset dado
    def process_dataset(dataset_name, qrels, query_tokens_map):
        for qid, rel_list in tqdm(qrels.items(), desc=f"Procesando {dataset_name}"):
            q_tokens = query_tokens_map.get(qid, set())
            for docid, label in rel_list:
                full_tokens = doc_tokens_full.get(docid, set())
                sum_tokens = doc_tokens_summary.get(docid, set())

                j_full = query_token_recall(q_tokens, full_tokens)
                j_sum = query_token_recall(q_tokens, sum_tokens)

                records.append({
                    "dataset": dataset_name,
                    "qid": qid,
                    "docid": docid,
                    "label": label,
                    "jaccard_full": j_full,
                    "jaccard_summary": j_sum,
                })

    # Procesar los 3 datasets
    process_dataset("ds_54", qrels_54, query_tokens_54)
    process_dataset("inpars", qrels_inpars, query_tokens_inpars)
    process_dataset("synthetic", qrels_synth, query_tokens_synth)

    # Convertir a DataFrame para análisis posterior
    df = pd.DataFrame.from_records(records)

    # Calcular la media Jaccard por dataset (tanto full como summary)
    mean_by_dataset = df.groupby("dataset")[["jaccard_full", "jaccard_summary"]].mean().reset_index()

    # Calcular la media Jaccard por dataset y label
    mean_by_dataset_label = (
        df.groupby(["dataset", "label"])[["jaccard_full", "jaccard_summary"]]
        .mean()
        .reset_index()
    )

    print("Media Jaccard por dataset:")
    print(mean_by_dataset)
    print("\nMedia Jaccard por dataset y label:")
    print(mean_by_dataset_label)

    # Guardar DataFrame completo para graficar luego
    output_path = os.path.join(base_dir, "results", "jaccard_analysis.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nRegistro guardado en: {output_path}")


def main():
    # """
    # Convierte un JSON de mapeo a un JSONL con campos 'id' y 'text'.
    # """
    # input_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_Gpt4o-mini_cleaned.json")
    # output_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_Gpt4o-mini_cleaned.jsonl")

    # corpus_json_to_jsonl(input_path, output_path)
    # print(f"Converted {input_path} to {output_path}")
    
    # Hard-coded input filenames
    inputs = [
        os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_summary_1024_train_baai.jsonl"),
        os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_summary_1024_dev_baai.jsonl"),
        os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_summary_1024_test_baai.jsonl"),
        # os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_synthetic_train_summary_1024_baai.jsonl"),
        # os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_synthetic_dev_summary_1024_baai.jsonl"),
        # os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_synthetic_test_summary_1024_baai.jsonl"),
    ]
    # Hard-coded output filename
    output = os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_summary_1024_baai.jsonl")

    merge_and_shuffle(inputs, output)
    print(f"✅ Merged {len(inputs)} files and wrote shuffled output to '{output}'")

    # add_chunk_headings_to_corpus(
    #     os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus.jsonl"),
    #     os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_chunk_headers.jsonl"),
    #     chunk_size=512
    # )

    

if __name__ == "__main__":
    main()