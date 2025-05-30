import os
import sys
import json

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
        os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_inpars_train_chunked_baai.jsonl"),
        os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_inpars_dev_chunked_baai.jsonl"),
        os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_inpars_test_chunked_baai.jsonl"),
        os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_synthetic_train_chunked_baai.jsonl"),
        os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_synthetic_dev_chunked_baai.jsonl"),
        os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_synthetic_test_chunked_baai.jsonl"),
    ]
    # Hard-coded output filename
    output = os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_inpars_synthetic_chunked_baai.jsonl")

    merge_and_shuffle(inputs, output)
    print(f"✅ Merged {len(inputs)} files and wrote shuffled output to '{output}'")

if __name__ == "__main__":
    main()