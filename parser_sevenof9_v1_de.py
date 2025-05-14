import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
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
import multiprocessing  # Wichtig für frozen support
from multiprocessing import Pool


# ========================
# Parser-Konfiguration
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
            page_output = f"Page {page_number + 1}\n"
            page_width = page.width
            page_height = page.height

            filtered_page = page
            table_bbox_list = []
            table_json_outputs = []

            for table in page.find_tables():
                bbox = clamp_bbox(table.bbox, page_width, page_height)
                table_bbox_list.append(bbox)

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

            chars_outside_tables = [
                word for word in page.extract_words(**text_settings)
                if not any(
                    bbox[0] <= float(word['x0']) <= bbox[2] and
                    bbox[1] <= float(word['top']) <= bbox[3]
                    for bbox in table_bbox_list
                )
            ]

            current_y = None
            line = []
            text_content = ""

            for word in chars_outside_tables:
                if current_y is None or abs(word['top'] - current_y) > 10:
                    if line:
                        text_content += " ".join(line) + "\n"
                    line = [word['text']]
                    current_y = word['top']
                else:
                    line.append(word['text'])
            if line:
                text_content += " ".join(line) + "\n"

            page_output += text_content.strip() + "\n"

            for idx, table in enumerate(table_json_outputs, start=1):
                page_output += f'"tabelle {idx}":\n{table}\n'

            return page_number, page_output

    except Exception as e:
        return args[0], f"[FEHLER] Seite {args[0]+1} ({args[1]}): {str(e)}"

def verarbeite_pdf(pdf_path):
    suppress_pdfminer_logging()
    try:
        if not os.path.exists(pdf_path):
            return f"[FEHLER] Datei nicht gefunden: {pdf_path}"

        print(f"[INFO] Beginne Verarbeitung: {pdf_path}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
        except PdfminerException as e:
            return f"[FEHLER] PDF kann nicht geöffnet werden: {pdf_path} – {str(e)}"
        except Exception as e:
            return f"[FEHLER] Allgemeiner Fehler beim Öffnen: {pdf_path} – {str(e)}"

        pages = [(i, pdf_path, TEXT_EXTRACTION_SETTINGS) for i in range(num_pages)]

        try:
            results = run_serial(pages) if num_pages <= PARALLEL_THRESHOLD else run_parallel(pages)
        except (EOFError, BrokenPipeError, KeyboardInterrupt):
            return "[INFO] Verarbeitung wurde abgebrochen."

        sorted_results = sorted(results, key=lambda x: x[0])
        final_output = "\n".join(text for _, text in sorted_results)

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.dirname(pdf_path)
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(output_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(final_output)

        print(f"[INFO] Verarbeitung abgeschlossen: {output_path}")

    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        return "[INFO] Verarbeitung durch Benutzer abgebrochen."
    except Exception as e:
        return f"[FEHLER] Unerwarteter Fehler bei '{pdf_path}': {str(e)}"

def run_serial(pages):
    return [process_page(args) for args in pages]

def run_parallel(pages):
    available_cores = max(1, cpu_count() - 2)  # Mindestens 1 Kern
    num_cores = min(available_cores, len(pages))
    print(f"Starte Parallelverarbeitung mit {num_cores} -2 Kernen...")
    with Pool(processes=num_cores) as pool:
        return pool.map(process_page, pages)

def verarbeite_pdfs_main():
    suppress_pdfminer_logging()
    pdf_dateien = sys.argv[1:]
    if not pdf_dateien:
        print("Keine PDF-Dateien übergeben.")
        return

    kleine_pdfs = []
    grosse_pdfs = []

    for pfad in pdf_dateien:
        if not os.path.exists(pfad):
            print(f"Datei nicht gefunden: {pfad}")
            continue
        try:
            with pdfplumber.open(pfad) as pdf:
                if len(pdf.pages) <= PARALLEL_THRESHOLD:
                    kleine_pdfs.append(pfad)
                else:
                    grosse_pdfs.append(pfad)
        except PdfminerException:
            print(f"[FEHLER] Passwortgeschützte PDF-Datei: {pfad} – wird übersprungen.")
        except Exception as e:
            print(f"[FEHLER] Fehler beim Öffnen von {pfad}: {str(e)}")

    if kleine_pdfs:
        available_cores = max(1, cpu_count() - 2)
        num_cores = min(available_cores, len(kleine_pdfs))
        print(f"\n[Phase 1] Starte Parallelverarbeitung kleiner PDFs mit {num_cores} -2 Kernen...")
        results = Parallel(n_jobs=num_cores)(
            delayed(verarbeite_pdf)(pfad) for pfad in kleine_pdfs
        )
        for r in results:
            print(r)

    for pfad in grosse_pdfs:
        print(f"\n[Phase 2] Verarbeitung großer PDFs: {os.path.basename(pfad)}")
        print(verarbeite_pdf(pfad))



# ========================
# GUI-Klasse
# ========================

class DateiManager:
    def __init__(self, master):
        self.master = master
        self.master.title("Parser-Sevenof9")
        self.dateien = []
        self.last_selected_index = None

        self.label = tk.Label(master, text="Ausgewählte PDF-Dateien:")
        self.label.pack(pady=5)

        listbox_frame = tk.Frame(master)
        listbox_frame.pack(pady=5)

        scrollbar_listbox = tk.Scrollbar(listbox_frame)
        self.listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, width=80, height=6, yscrollcommand=scrollbar_listbox.set)
        scrollbar_listbox.config(command=self.listbox.yview)

        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_listbox.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox.bind("<<ListboxSelect>>", self.zeige_textdatei)
        self.listbox.bind("<Button-1>", self.on_listbox_click)
        self.listbox.bind("<Shift-Button-1>", self.on_listbox_shift_click)

        self.context_menu = tk.Menu(master, tearoff=0)
        self.context_menu.add_command(label="Ausgewählte entfernen", command=self.datei_entfernen)
        self.listbox.bind("<Button-3>", self.show_context_menu)

        self.frame = tk.Frame(master)
        self.frame.pack(pady=10)

        tk.Button(self.frame, text="Ordner hinzufügen", command=self.ordner_hinzufuegen).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame, text="Dateien auswählen", command=self.datei_hinzufuegen).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame, text="Ausgewählte entfernen", command=self.datei_entfernen).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame, text="Alle entfernen", command=self.alle_entfernen).pack(side=tk.LEFT, padx=5)
        tk.Button(master, text="Stop", command=self.parser_stoppen).pack(pady=5)
        self.parser_process = None  # Wird im Thread gespeichert

        tk.Button(master, text="Parser starten", command=self.parser_starten).pack(pady=10)

        textfeld_frame = tk.Frame(master)
        textfeld_frame.pack(padx=10, pady=5)

        scrollbar_textfeld = tk.Scrollbar(textfeld_frame)
        self.textfeld = tk.Text(textfeld_frame, height=15, width=100, wrap=tk.WORD, yscrollcommand=scrollbar_textfeld.set)
        scrollbar_textfeld.config(command=self.textfeld.yview)

        self.textfeld.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_textfeld.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(master, text="Fortschritt:").pack()

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
        self.zeige_textdatei(None)
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

    def ordner_hinzufuegen(self):
        ordner = filedialog.askdirectory(title="Ordner auswählen")
        if not ordner:
            return
        for root, _, files in os.walk(ordner):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pfad = os.path.join(root, file)
                    if pfad not in self.dateien:
                        self.dateien.append(pfad)
                        self.listbox.insert(tk.END, pfad)
    
    def datei_hinzufuegen(self):
        pfade = filedialog.askopenfilenames(title="PDF-Dateien auswählen", filetypes=[("PDF-Dateien", "*.pdf")])
        for pfad in pfade:
            if pfad not in self.dateien:
                self.dateien.append(pfad)
                self.listbox.insert(tk.END, pfad)

    def datei_entfernen(self):
        selektion = self.listbox.curselection()
        if not selektion:
            messagebox.showwarning("Hinweis", "Bitte wählen Sie einen Eintrag zum Entfernen.")
            return
        for index in reversed(selektion):
            self.listbox.delete(index)
            del self.dateien[index]
        self.textfeld.delete(1.0, tk.END)

    def alle_entfernen(self):
        self.listbox.delete(0, tk.END)
        self.dateien.clear()
        self.textfeld.delete(1.0, tk.END)
        

    def parser_starten(self):
        if not self.dateien:
            messagebox.showinfo("Keine Dateien", "Bitte wählen Sie mindestens eine Datei aus.")
            return
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.insert(tk.END, "Starte Parser...\n")
        self.progress_text.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.parser_ausfuehren)
        thread.start()

    def parser_stoppen(self):
        if self.parser_process and self.parser_process.poll() is None:
            self.parser_process.terminate()
            self.progress_text_einfuegen("Parser-Prozess wurde gestoppt.\n")
        else:
            self.progress_text_einfuegen("Kein laufender Parser-Prozess zum Stoppen.\n")

    def parser_ausfuehren(self):
        try:
            self.parser_process = subprocess.Popen(
                [sys.executable, __file__] + self.dateien,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='ignore',
                bufsize=4096
            )
            for line in self.parser_process.stdout:
                self.progress_text_einfuegen(line)
            self.parser_process.stdout.close()
            self.parser_process.wait()

            if self.parser_process.returncode == 0:
                self.progress_text_einfuegen("\nParser abgeschlossen.\n")
                self.show_messagebox_threadsafe("Parser abgeschlossen", "Der Parser wurde erfolgreich ausgeführt.")
            else:
                self.progress_text_einfuegen("\nFehler beim Ausführen des Parsers.\n")
                self.show_messagebox_threadsafe("Fehler", "Fehler beim Ausführen des Parsers.")
        except Exception as e:
            self.progress_text_einfuegen(f"Fehler: {e}\n")
            self.show_messagebox_threadsafe("Fehler", f"Fehler beim Ausführen:\n{e}")
        finally:
            self.parser_process = None

    def progress_text_einfuegen(self, text):
        self.progress_text.after(0, lambda: self._text_einfuegen(text))

    def _text_einfuegen(self, text):
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.insert(tk.END, text)
        self.progress_text.see(tk.END)
        self.progress_text.config(state=tk.DISABLED)

    def show_messagebox_threadsafe(self, titel, nachricht):
        self.master.after(0, lambda: messagebox.showinfo(titel, nachricht))

    def zeige_textdatei(self, event):
        selektion = self.listbox.curselection()
        if not selektion:
            return
        index = selektion[0]
        pfad = self.dateien[index]
        txt_pfad = os.path.splitext(pfad)[0] + ".txt"
        self.textfeld.delete(1.0, tk.END)
        if os.path.exists(txt_pfad):
            try:
                with open(txt_pfad, "r", encoding="utf-8", errors="ignore") as f:
                    self.textfeld.insert(tk.END, f.read())
            except Exception as e:
                self.textfeld.insert(tk.END, f"Fehler beim Laden der Textdatei:\n{e}")
        else:
            self.textfeld.insert(tk.END, "[Keine zugehörige .txt-Datei vorhanden]")

# ========================
# Einstiegspunkt
# ========================


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Muss als erstes im main stehen

    if len(sys.argv) > 1:
        verarbeite_pdfs_main()
    else:
        root = tk.Tk()
        app = DateiManager(root)
        root.mainloop()
