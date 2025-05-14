import os
import sys
import tkinter as tk # internal
from tkinter import filedialog, messagebox # internal
import subprocess
import threading
import tempfile
import shutil
import json
import logging
import pdfplumber
from pdfplumber.utils import get_bbox_overlap, obj_to_bbox
from pdfplumber.utils.exceptions import PdfminerException
from joblib import delayed, cpu_count, parallel_backend, Parallel
import multiprocessing # intternal
from multiprocessing import Pool # internal


# ========================
# Parser Configuration
# ========================

TEXT_EXTRACTION_SETTINGS = {
    "x_tolerance": 1,
    "y_tolerance": 3,
    "keep_blank_chars": False,
    "use_text_flow": True
}

if sys.platform == "win32":
    sys.stderr = open(os.devnull, 'w')

PARALLEL_THRESHOLD = 16

def suppress_pdfminer_logging():
    for logger_name in [
        "pdfminer",
        "pdfminer.pdfparser",
        "pdfminer.pdfdocument",
        "pdfminer.pdfpage",
        "pdfminer.converter",
        "pdfminer.layout",
        "pdfminer.cmapdb",
        "pdfminer.utils"
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

def clean_cell_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("-\n", "").replace("\n", " ")
    return " ".join(text.split())

def safe_join(row):
    return [clean_cell_text(str(cell)) if cell is not None else "" for cell in row]

def clamp_bbox(bbox, page_width, page_height):
    x0, top, x1, bottom = bbox
    x0 = max(0, min(x0, page_width))
    x1 = max(0, min(x1, page_width))
    top = max(0, min(top, page_height))
    bottom = max(0, min(bottom, page_height))
    return (x0, top, x1, bottom)

def process_page(args):
    suppress_pdfminer_logging()
    try:
        page_number, pdf_path, text_settings = args
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number]
            output = f"Page {page_number + 1}\n"
            width, height = page.width, page.height

            filtered_page = page
            table_bboxes = []
            table_json_outputs = []

            for table in page.find_tables():
                bbox = clamp_bbox(table.bbox, width, height)
                table_bboxes.append(bbox)

                if not page.crop(bbox).chars:
                    continue

                filtered_page = filtered_page.filter(
                    lambda obj: get_bbox_overlap(obj_to_bbox(obj), bbox) is None
                )

                table_data = table.extract()
                if table_data and len(table_data) >= 1:
                    headers = safe_join(table_data[0])
                    rows = [safe_join(row) for row in table_data[1:]]
                    json_table = [dict(zip(headers, row)) for row in rows]
                    table_json_outputs.append(json.dumps(json_table, indent=1, ensure_ascii=False))

            words_outside_tables = [
                word for word in page.extract_words(**text_settings)
                if not any(
                    bbox[0] <= float(word['x0']) <= bbox[2] and
                    bbox[1] <= float(word['top']) <= bbox[3]
                    for bbox in table_bboxes
                )
            ]

            current_y = None
            line = []
            text_content = ""

            for word in words_outside_tables:
                if current_y is None or abs(word['top'] - current_y) > 10:
                    if line:
                        text_content += " ".join(line) + "\n"
                    line = [word['text']]
                    current_y = word['top']
                else:
                    line.append(word['text'])
            if line:
                text_content += " ".join(line) + "\n"

            output += text_content.strip() + "\n"

            for idx, table in enumerate(table_json_outputs, start=1):
                output += f'"table {idx}":\n{table}\n'

            return page_number, output

    except Exception as e:
        return args[0], f"[ERROR] Page {args[0]+1} ({args[1]}): {str(e)}"

def process_pdf(pdf_path):
    suppress_pdfminer_logging()
    try:
        if not os.path.exists(pdf_path):
            return f"[ERROR] File not found: {pdf_path}"

        print(f"[INFO] Starting processing: {pdf_path}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
        except PdfminerException as e:
            return f"[ERROR] Cannot open PDF: {pdf_path} – {str(e)}"
        except Exception as e:
            return f"[ERROR] General error opening PDF: {pdf_path} – {str(e)}"

        pages = [(i, pdf_path, TEXT_EXTRACTION_SETTINGS) for i in range(num_pages)]

        try:
            results = run_serial(pages) if num_pages <= PARALLEL_THRESHOLD else run_parallel(pages)
        except (EOFError, BrokenPipeError, KeyboardInterrupt):
            return "[INFO] Processing was interrupted."

        sorted_results = sorted(results, key=lambda x: x[0])
        final_output = "\n".join(text for _, text in sorted_results)

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.dirname(pdf_path)
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(output_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(final_output)

        print(f"[INFO] Processing complete: {output_path}")

    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        return "[INFO] Processing interrupted by user."
    except Exception as e:
        return f"[ERROR] Unexpected error with '{pdf_path}': {str(e)}"

def run_serial(pages):
    return [process_page(args) for args in pages]

def run_parallel(pages):
    available_cores = max(1, cpu_count() - 2)
    num_cores = min(available_cores, len(pages))
    print(f"Starting parallel processing with {num_cores} cores...")
    with Pool(processes=num_cores) as pool:
        return pool.map(process_page, pages)

def process_pdfs_main():
    suppress_pdfminer_logging()
    pdf_files = sys.argv[1:]
    if not pdf_files:
        print("No PDF files provided.")
        return

    small_pdfs = []
    large_pdfs = []

    for path in pdf_files:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        try:
            with pdfplumber.open(path) as pdf:
                if len(pdf.pages) <= PARALLEL_THRESHOLD:
                    small_pdfs.append(path)
                else:
                    large_pdfs.append(path)
        except PdfminerException:
            print(f"[ERROR] Password-protected PDF skipped: {path}")
        except Exception as e:
            print(f"[ERROR] Error opening {path}: {str(e)}")

    if small_pdfs:
        available_cores = max(1, cpu_count() - 2)
        num_cores = min(available_cores, len(small_pdfs))
        print(f"\n[Phase 1] Starting parallel processing of small PDFs with {num_cores} cores...")
        results = Parallel(n_jobs=num_cores)(
            delayed(process_pdf)(path) for path in small_pdfs
        )
        for r in results:
            print(r)

    for path in large_pdfs:
        print(f"\n[Phase 2] Processing large PDF: {os.path.basename(path)}")
        print(process_pdf(path))


# ========================
# GUI Class
# ========================

class FileManager:
    def __init__(self, master):
        self.master = master
        self.master.title("Parser-Sevenof9")
        self.files = []
        self.last_selected_index = None

        self.label = tk.Label(master, text="Selected PDF files:")
        self.label.pack(pady=5)

        listbox_frame = tk.Frame(master)
        listbox_frame.pack(pady=5)

        scrollbar_listbox = tk.Scrollbar(listbox_frame)
        self.listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, width=80, height=6, yscrollcommand=scrollbar_listbox.set)
        scrollbar_listbox.config(command=self.listbox.yview)

        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_listbox.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox.bind("<<ListboxSelect>>", self.show_text_file)
        self.listbox.bind("<Button-1>", self.on_listbox_click)
        self.listbox.bind("<Shift-Button-1>", self.on_listbox_shift_click)

        self.context_menu = tk.Menu(master, tearoff=0)
        self.context_menu.add_command(label="Remove selected", command=self.remove_file)
        self.listbox.bind("<Button-3>", self.show_context_menu)

        self.frame = tk.Frame(master)
        self.frame.pack(pady=10)

        tk.Button(self.frame, text="Add Folder", command=self.add_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame, text="Select Files", command=self.add_file).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame, text="Remove Selected", command=self.remove_file).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame, text="Remove All", command=self.remove_all).pack(side=tk.LEFT, padx=5)
        tk.Button(master, text="Stop", command=self.stop_parser).pack(pady=5)
        self.parser_process = None  # Will be stored in thread

        tk.Button(master, text="Start Parser", command=self.start_parser).pack(pady=10)

        text_frame = tk.Frame(master)
        text_frame.pack(padx=10, pady=5)

        scrollbar_text = tk.Scrollbar(text_frame)
        self.text_widget = tk.Text(text_frame, height=15, width=100, wrap=tk.WORD, yscrollcommand=scrollbar_text.set)
        scrollbar_text.config(command=self.text_widget.yview)

        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_text.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(master, text="Progress:").pack()

        progress_frame = tk.Frame(master)
        progress_frame.pack(padx=10, pady=5)

        scrollbar_progress = tk.Scrollbar(progress_frame)
        self.progress_text = tk.Text(progress_frame, height=8, width=100, state=tk.DISABLED, yscrollcommand=scrollbar_progress.set)
        scrollbar_progress.config(command=self.progress_text.yview)

        self.progress_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_progress.pack(side=tk.RIGHT, fill=tk.Y)

    def on_listbox_click(self, event):
        index = self.listbox.nearest(event.y)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.last_selected_index = index
        self.show_text_file(None)
        return "break"

    def on_listbox_shift_click(self, event):
        index = self.listbox.nearest(event.y)
        if self.last_selected_index is None:
            self.last_selected_index = index
        start, end = sorted((self.last_selected_index, index))
        self.listbox.selection_clear(0, tk.END)
        for i in range(start, end + 1):
            self.listbox.selection_set(i)
        return "break"

    def show_context_menu(self, event):
        if self.listbox.curselection():
            self.context_menu.tk_popup(event.x_root, event.y_root)

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select Folder")
        if not folder:
            return
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(".pdf"):
                    path = os.path.join(root, file)
                    if path not in self.files:
                        self.files.append(path)
                        self.listbox.insert(tk.END, path)

    def add_file(self):
        paths = filedialog.askopenfilenames(title="Select PDF Files", filetypes=[("PDF Files", "*.pdf")])
        for path in paths:
            if path not in self.files:
                self.files.append(path)
                self.listbox.insert(tk.END, path)

    def remove_file(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Notice", "Please select an entry to remove.")
            return
        for index in reversed(selection):
            self.listbox.delete(index)
            del self.files[index]
        self.text_widget.delete(1.0, tk.END)

    def remove_all(self):
        self.listbox.delete(0, tk.END)
        self.files.clear()
        self.text_widget.delete(1.0, tk.END)

    def start_parser(self):
        if not self.files:
            messagebox.showinfo("No Files", "Please select at least one file.")
            return
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.insert(tk.END, "Starting parser...\n")
        self.progress_text.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.run_parser)
        thread.start()

    def stop_parser(self):
        if self.parser_process and self.parser_process.poll() is None:
            self.parser_process.terminate()
            self.append_progress_text("Parser process was stopped.\n")
        else:
            self.append_progress_text("No active parser process to stop.\n")

    def run_parser(self):
        try:
            self.parser_process = subprocess.Popen(
                [sys.executable, __file__] + self.files,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='ignore',
                bufsize=4096
            )
            for line in self.parser_process.stdout:
                self.append_progress_text(line)
            self.parser_process.stdout.close()
            self.parser_process.wait()

            if self.parser_process.returncode == 0:
                self.append_progress_text("\nParser finished successfully.\n")
                self.show_messagebox_threadsafe("Parser Done", "The parser was executed successfully.")
            else:
                self.append_progress_text("\nError while running the parser.\n")
                self.show_messagebox_threadsafe("Error", "Error while running the parser.")
        except Exception as e:
            self.append_progress_text(f"Error: {e}\n")
            self.show_messagebox_threadsafe("Error", f"Error during execution:\n{e}")
        finally:
            self.parser_process = None

    def append_progress_text(self, text):
        self.progress_text.after(0, lambda: self._insert_text(text))

    def _insert_text(self, text):
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.insert(tk.END, text)
        self.progress_text.see(tk.END)
        self.progress_text.config(state=tk.DISABLED)

    def show_messagebox_threadsafe(self, title, message):
        self.master.after(0, lambda: messagebox.showinfo(title, message))

    def show_text_file(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
        index = selection[0]
        path = self.files[index]
        txt_path = os.path.splitext(path)[0] + ".txt"
        self.text_widget.delete(1.0, tk.END)
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    self.text_widget.insert(tk.END, f.read())
            except Exception as e:
                self.text_widget.insert(tk.END, f"Error loading text file:\n{e}")
        else:
            self.text_widget.insert(tk.END, "[No corresponding .txt file found]")

# ========================
# Entry Point
# ========================

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Must be first in main for compatibility with multiprocessing on Windows

    if len(sys.argv) > 1:
        process_pdfs_main()
    else:
        root = tk.Tk()
        app = FileManager(root)
        root.mainloop()

