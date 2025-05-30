import os
import sys
import json


def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Apply the path tweak before any project imports
configure_python_path()

from config.config import STORAGE_DIR


def get_annotations_by_username(data, username):
    """
    Return annotations of a specific user.

    This function filters annotation records by username
    and returns all matching entries.

    Parameters
    ----------
    data : list of dict
        List of annotation records.
    username : str
        Username to filter annotations.

    Returns
    -------
    list of dict
        Annotation records for the given user.

    Examples
    --------
    >>> annotations = [
    ...     {"username": "cristian", "labels": [...], ...},
    ...     {"username": "maximiliano9", "labels": [...], ...},
    ... ]
    >>> get_annotations_by_username(annotations, "cristian")
    [{'username': 'cristian', 'labels': [...], ...}]
    """
    filtered = [
        record for record in data
        if record.get("username", "").lower() == username.lower()
    ]
    return filtered


def write_qrel_file(annotations, output_path, label_map=None):
    """
    Write a TREC-style qrel file from annotation records.

    Parameters
    ----------
    annotations : list of dict
        List of annotation records, each with keys:
        - 'topic_id': str (e.g. "3:Robo Agravado")
        - 'id_report': str or int (document ID)
        - 'labels': list of dicts, first dict has key 'label'
    output_path : str
        File path where the qrel lines will be written.
    label_map : dict, optional
        Mapping from textual labels to numeric scores.
        Default:
            {'Altamente relevante': 3,
             'Relevante': 2,
             'Parcialmente relevante': 1,
             'No relevante': 0}

    Notes
    -----
    The qrel file will have lines of the form:
        <topic> 0 <doc_id> <relevance>
    where <topic> is derived from the part of 'topic_id'
    before the colon.
    """
    if label_map is None:
        label_map = {
            'Altamente relevante': 3,
            'Relevante': 2,
            'Parcialmente relevante': 1,
            'No relevante': 0
        }

    with open(output_path, 'w', encoding='utf-8') as fh:
        for rec in annotations:
            # Extract topic number (before the colon)
            if isinstance(rec['topic_id'], str) and ':' in rec['topic_id']:
                topic = rec['topic_id'].split(':', 1)[0]
            else:
                topic = str(rec['topic_id'])
            doc_id = rec['id_report']
            label_text = rec['labels'][0]['label']
            relevance = label_map.get(label_text, 0)
            fh.write(f"{topic}\t0\t{doc_id}\t{relevance}\n")


def filter_full_qrel_by_meta(full_qrel_path, meta_qrel_path, output_path):
    """
    Filter a full qrel file by the (query, doc) pairs in a meta-qrel file.

    Parameters
    ----------
    full_qrel_path : str
        Path to the complete qrel file. Each line must be of the form:
        <query_id> <iter> <doc_id> <relevance>
    meta_qrel_path : str
        Path to the meta qrel file (Christian’s). Same format as full qrel.
    output_path : str
        Path where the filtered qrel lines will be written.

    Notes
    -----
    The function reads the meta qrel to build a set of (query_id, doc_id)
    tuples, then writes only those lines from the full qrel that match.

    Examples
    --------
    >>> filter_full_qrel_by_meta(
    ...     "full.qrel",
    ...     "cristian.qrel",
    ...     "filtered.qrel"
    ... )
    """
    # Read meta pairs
    meta_pairs = set()
    with open(meta_qrel_path, "r", encoding="utf-8") as meta_file:
        for line in meta_file:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id, _, doc_id = parts[:3]
                meta_pairs.add((query_id, doc_id))

    # Filter full qrel
    with open(full_qrel_path, "r", encoding="utf-8") as full_file, \
         open(output_path, "w", encoding="utf-8") as out_file:
        for line in full_file:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id, _, doc_id = parts[:3]
                if (query_id, doc_id) in meta_pairs:
                    out_file.write(line)


if __name__ == "__main__":
    # # Load your JSON list from a file (or assign it directly to `annotations`)
    # path = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "all_json_ground_truth.json")
    # with open(path, "r", encoding="utf-8") as f:
    #     annotations = json.load(f)

    # # Get only the records for user "cristian"
    # cristian_annotations = get_annotations_by_username(annotations, "cristian")

    # write_qrel_file(
    #     cristian_annotations,
    #     output_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "meta_qrels.tsv")
    # )

    filter_full_qrel_by_meta(
        full_qrel_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_54.tsv"),
        meta_qrel_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "meta_qrels.tsv"),
        output_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "filtered_qrels_54.tsv")
    )