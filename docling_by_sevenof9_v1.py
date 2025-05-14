import json
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict
from multiprocessing import get_context
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extract_clean_table_data(table):
    cells = table.get("data", {}).get("table_cells", [])
    if not cells:
        return None

    max_row = max(cell["end_row_offset_idx"] for cell in cells)
    max_col = max(cell["end_col_offset_idx"] for cell in cells)
    table_matrix = [["" for _ in range(max_col)] for _ in range(max_row)]

    for cell in cells:
        row = cell["start_row_offset_idx"]
        col = cell["start_col_offset_idx"]
        table_matrix[row][col] = cell.get("text", "").strip()

    column_headers = table_matrix[0]
    data_rows = table_matrix[1:]

    structured_rows = []
    for row in data_rows:
        row_data = {
            column_headers[i]: row[i] for i in range(len(column_headers)) if column_headers[i]
        }
        structured_rows.append(row_data)

    return {
        "num_rows": len(data_rows),
        "num_columns": len(column_headers),
        "columns": column_headers,
        "data": structured_rows,
    }

def process_single_pdf(pdf_path: Path, accelerator_options: AcceleratorOptions):
    logging.info(f"Verarbeite: {pdf_path.name}")
    output_dir = pdf_path.parent

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend,
                pipeline_options=pipeline_options,
            )
        }
    )

    doc = converter.convert(pdf_path).document
    doc_dict = doc.export_to_dict()

    page_texts = defaultdict(list)
    page_tables = defaultdict(list)

    for text_item in doc_dict.get("texts", []):
        if "text" in text_item and "prov" in text_item:
            for prov in text_item["prov"]:
                page = prov.get("page_no")
                if page is not None:
                    page_texts[page].append(text_item["text"])

    for table_item in doc_dict.get("tables", []):
        prov = table_item.get("prov", [])
        if not prov:
            continue
        page = prov[0].get("page_no")
        clean_table = extract_clean_table_data(table_item)
        if clean_table:
            page_tables[page].append(clean_table)

    output_txt_path = output_dir / f"{pdf_path.stem}_extracted.txt"
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for page_no in sorted(set(page_texts.keys()).union(page_tables.keys())):
            f.write(f"=== Page {page_no} ===\n\n")

            texts = page_texts.get(page_no, [])
            if texts:
                f.write("\n")
                f.write("\n".join(texts))
                f.write("\n\n")

            tables = page_tables.get(page_no, [])
            if tables:
                f.write("table:\n")
                for i, table in enumerate(tables, 1):
                    table_entry = {
                        "table_index": i,
                        **table,
                    }
                    f.write(json.dumps(table_entry, ensure_ascii=False, indent=1))
                    f.write("\n\n")

    logging.info(f"Fertig: {pdf_path.name} → {output_txt_path.name}")


def main():
    base_dir = Path(__file__).resolve().parent
    pdf_files = list(base_dir.glob("*.pdf"))

    if not pdf_files:
        print("Keine PDF-Dateien im aktuellen Ordner gefunden.")
        return

    print(f"{len(pdf_files)} PDF-Dateien gefunden. Starte Verarbeitung.")

    # Manuell festgelegter VRAM in GB
    vram_gb = 16  # YOUR GPU VRAM, Dedicated RAM

    # Anzahl paralleler Prozesse basierend auf VRAM
    max_subprocesses = int(vram_gb / 1.3)
    print(f"Maximale Anzahl paralleler Subprozesse: {max_subprocesses}")

    accelerator_options = AcceleratorOptions(num_threads=1, device=AcceleratorDevice.AUTO)

    ctx = get_context("spawn")

    # Verteile PDFs auf Prozesse – jeweils eine ganze PDF pro Subprozess
    with ctx.Pool(processes=min(max_subprocesses, len(pdf_files))) as pool:
        pool.starmap(process_single_pdf, [(pdf_path, accelerator_options) for pdf_path in pdf_files])

    sys.exit(">>> STOP <<<")

if __name__ == "__main__":
    main()
