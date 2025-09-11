import os
import sys
import time
import json
import wx
import re
import platform
import subprocess
import threading
import concurrent.futures
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pdfplumber
import psutil
import logging
from pdfminer.pdfparser import PDFParser, PDFSyntaxError
from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError, PDFPasswordIncorrect
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager


# -------------------- Konfiguration --------------------
PARALLEL_THRESHOLD = 16

TEXT_EXTRACTION_SETTINGS = {
    "x_tolerance": 1.5,
    "y_tolerance": 2.5,
    "keep_blank_chars": False,
    "use_text_flow": False,
}



# GUi update intervall
def throttle_callback(callback, interval_ms=1):
    last_called = 0

    def wrapper(status):
        nonlocal last_called
        now = time.time() * 1000  # Zeit in ms
        if now - last_called >= interval_ms:
            last_called = now
            callback(status)
    return wrapper



# Function to suppress PDFMiner logging, reducing verbosity
def suppress_pdfminer_logging():
    for logger_name in [
        "pdfminer",  # Various pdfminer modules to suppress logging from
        "pdfminer.pdfparser",
        "pdfminer.pdfdocument",
        "pdfminer.pdfpage",
        "pdfminer.converter",
        "pdfminer.layout",
        "pdfminer.cmapdb",
        "pdfminer.utils"
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)  # Set logging level to ERROR to suppress lower levels


EUROPEAN_PRINTABLES_PATTERN =  re.compile(r"[^\u0000-\uFFFF]", re.DOTALL)
CID_PATTERN = re.compile(r"\(cid:\d+\)")

def clean_cell_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("-\n", "").replace("\n", " ")
    text = CID_PATTERN.sub("", text)
    return EUROPEAN_PRINTABLES_PATTERN.sub("", text)

def clamp_bbox(bbox, page_width, page_height, p=3):
    x0, top, x1, bottom = bbox
    x0 = max(0, min(x0, page_width))
    x1 = max(0, min(x1, page_width))
    top = max(0, min(top, page_height))
    bottom = max(0, min(bottom, page_height))
    return round(x0, p), round(top, p), round(x1, p), round(bottom, p)

def get_physical_cores():
    count = psutil.cpu_count(logical=False)
    return max(1, count if count else 1)  # fallback = 1
cores = get_physical_cores()


def is_valid_cell(cell):
    """Prüft, ob eine Zelle mehr als nur Leerzeichen oder ein einzelnes Zeichen enthält."""
    if cell is None:
        return False
    content = str(cell).strip()
    return len(content) > 1


def block_area(block):
    x0 = min(w["x0"] for w in block)
    x1 = max(w["x1"] for w in block)
    top = min(w["top"] for w in block)
    bottom = max(w["bottom"] for w in block)
    return (x1 - x0) * (bottom - top)


suppress_pdfminer_logging()

# -------------------- Status-Tracking --------------------
class StatusTracker:
    def __init__(self, total_pages):
        self.start_time = time.time()
        self.total_pages = total_pages
        self.processed_pages = 0

    def update(self, n=1):
        self.processed_pages += n

    def get_status(self):
        elapsed = time.time() - self.start_time
        pages_per_sec = round(self.processed_pages / elapsed) if elapsed > 0 else 0
        remaining_pages = self.total_pages - self.processed_pages
        est_time = (remaining_pages / pages_per_sec) / 60 if pages_per_sec > 0 else float('inf')
        return {
            "processed_pages": self.processed_pages,
            "total_pages": self.total_pages,
            "pages_per_sec": pages_per_sec,
            "elapsed_time": round(elapsed / 60, 1),
            "est_time": round(est_time, 1)
        }


# -------------------- PDF Verarbeitung --------------------
def process_page_worker(args):
    suppress_pdfminer_logging()
    try:
        page_number, path = args
        with pdfplumber.open(path) as pdf:
            page = pdf.pages[page_number]
            width, height = page.width, page.height
            
            # Unabhängige Ränder definieren (z. B. 4 % links/rechts, 5 % oben, 7 % unten)
            margin_x_percent = 0.06
            top_margin_percent = 0.06
            bottom_margin_percent = 0.04

            margin_x = width * margin_x_percent
            top_margin = height * top_margin_percent
            bottom_margin = height * bottom_margin_percent

            # crop(left, top, right, bottom)
            cropped_page = page.crop((
                margin_x,
                top_margin,
                width - margin_x,
                height - bottom_margin
            ))            
            '''
            dpi = 150  # gleiche Auflösung wie to_image
            pixel_per_point = dpi / 85.5
            
            # Originalgröße in Punkten
            print(f"Originalgröße: {width:.2f}pt x {height:.2f}pt")
            print(f"Originalgröße: {width * pixel_per_point:.0f}px x {height * pixel_per_point:.0f}px")

            # Cropped-Größe berechnen
            cropped_width = width - 2 * margin_x
            cropped_height = height - top_margin - bottom_margin

            print(f"Cropped-Größe: {cropped_width:.2f}pt x {cropped_height:.2f}pt")
            print(f"Cropped-Größe: {cropped_width * pixel_per_point:.0f}px x {cropped_height * pixel_per_point:.0f}px")            
            '''
            #margin_x, margin_y = width * 0.04, height * 0.04

            #cropped_page = page.crop((margin_x, margin_y, width - margin_x, height - margin_y))
            table_bboxes = [clamp_bbox(t.bbox, width, height) for t in cropped_page.find_tables()]
            extracted_tables = cropped_page.extract_tables({"text_x_tolerance": 1.5})
            tables_json = []

            for raw_table in extracted_tables:
                if not raw_table or len(raw_table) < 2:
                    continue  # Weniger als 2 Zeilen

                # Prüfe auf mindestens 2 Spalten
                if all(len(row) < 2 for row in raw_table if row):
                    continue

                # Leere oder fast leere Tabellen (nur Leerzeichen oder 1 Zeichen pro Zelle) ausschließen
                if all(all(not is_valid_cell(cell) for cell in row) for row in raw_table):
                    continue

                cleaned_table = [[clean_cell_text(c) for c in row] for row in raw_table]
                header_row = cleaned_table[0]
                is_corner_empty = header_row[0].strip() == ""

                if is_corner_empty:
                    col_headers = cleaned_table[0][1:]
                    row_headers = [row[0] for row in cleaned_table[1:]]
                    data_rows = cleaned_table[1:]

                    table_data = {}
                    for row_header, row in zip(row_headers, data_rows):
                        row_dict = {}
                        for col_header, cell in zip(col_headers, row[1:]):
                            row_dict[col_header] = cell
                        table_data[row_header] = row_dict
                else:
                    headers = header_row
                    data_rows = cleaned_table[1:]
                    table_data = []
                    for row in data_rows:
                        if len(row) == len(headers):
                            table_data.append(dict(zip(headers, row)))

                tables_json.append(json.dumps(table_data, indent=1, ensure_ascii=False))


            words = []
            for w in cropped_page.extract_words(**TEXT_EXTRACTION_SETTINGS):
                x0, top = float(w["x0"]), float(w["top"])
                if any(bx0 <= x0 <= bx2 and by0 <= top <= by3 for bx0, by0, bx2, by3 in table_bboxes):
                    continue
                if EUROPEAN_PRINTABLES_PATTERN.search(w["text"]):
                    continue
                words.append(w)

            def is_bold(fontname: str) -> bool:
                fontname = fontname.lower()
                return "bold" in fontname or "bd" in fontname or "black" in fontname

            word_info = []
            font_sizes = []
            for w in words:
                x0 = float(w["x0"])
                x1 = float(w["x1"])
                top = float(w["top"])
                bottom = float(w["bottom"])
                text = w["text"]
                #cropped_chars = cropped_page.chars

                chars = [c for c in cropped_page.chars if x0 <= float(c["x0"]) <= x1 and top <= float(c["top"]) <= bottom]
                sizes = [float(c.get("size", 0)) for c in chars if c.get("text", "").strip()]
                fonts = [c.get("fontname", "") for c in chars]
                bold_flags = [is_bold(c.get("fontname", "")) for c in chars]

                font_size = max(sizes) if sizes else 0
                font_sizes.append(font_size)
                font_name = fonts[0] if fonts else "Unknown"
                bold_flag = any(bold_flags)

                word_info.append({
                    "text": text,
                    "top": round(top, 1),
                    "bottom": round(bottom, 1),
                    "font_size": font_size,
                    "font_name": font_name,
                    "bold_flag": bold_flag,
                    "x0": round(x0, 1),
                    "x1": round(x1, 1),
                })

                             

            avg_fontsize = sum(font_sizes) / len(font_sizes) if font_sizes else 0

            # Abstandsschwellen
            MAX_DIST_X = 12
            MAX_DIST_Y = 10

            def are_words_close(w1, w2):
                # Prüfe, ob Wörter räumlich nah beieinander liegen
                dx = max(0, max(w1["x0"], w2["x0"]) - min(w1["x1"], w2["x1"]))
                dy = max(0, max(w1["top"], w2["top"]) - min(w1["bottom"], w2["bottom"]))
                return dx <= MAX_DIST_X and dy <= MAX_DIST_Y

            def group_into_blocks(words):
                blocks = []
                unvisited = set(range(len(words)))
                while unvisited:
                    idx = unvisited.pop()
                    block = {idx}
                    to_visit = {idx}
                    while to_visit:
                        current = to_visit.pop()
                        for other in list(unvisited):
                            if are_words_close(words[current], words[other]):
                                block.add(other)
                                to_visit.add(other)
                                unvisited.remove(other)
                    blocks.append([words[i] for i in block])
                return blocks

            def group_block_into_lines(block, line_tolerance=2.5):
                # Gruppiere Wörter innerhalb eines Blocks in Zeilen (nach Y-Koordinate)
                sorted_words = sorted(block, key=lambda w: w["top"])
                lines = []
                #lines = [sorted(block, key=lambda w: w["x0"])]
                current_line = [sorted_words[0]]
                current_top = sorted_words[0]["top"]

                for word in sorted_words[1:]:
                    if abs(word["top"] - current_top) <= line_tolerance:
                        current_line.append(word)
                    else:
                        lines.append(sorted(current_line, key=lambda w: w["x0"]))
                        current_line = [word]
                        current_top = word["top"]
                if current_line:
                    lines.append(sorted(current_line, key=lambda w: w["x0"]))
                return lines

           
            blocks = group_into_blocks(word_info)

            SORT_TOLERANCE = 1  # e.g. 1 point distance

            def round_to_nearest(value, tolerance):
                return round(value / tolerance) * tolerance

            def get_block_reference(block):
                min_x0 = min(w["x0"] for w in block)
                min_top = min(w["top"] for w in block)
                return (
                    round_to_nearest(min_x0, SORT_TOLERANCE),
                    round_to_nearest(min_top, SORT_TOLERANCE),
                )

            # Sort blocks first by x0, then by top (row beginning)
            sorted_blocks = sorted(blocks, key=get_block_reference)
            
            '''
            # Visualisierung: Blocks als Rechtecke zeichnen
            im = page.to_image(resolution=150)  # ggf. Auflösung anpassen

            # Zeichne roten Rahmen für die Cropped-Region
            im.draw_rect(
                (
                    margin_x,
                    top_margin,
                    width - margin_x,
                    height - bottom_margin
                ),
                stroke="red",
                stroke_width=2
            )

            for block in blocks:
                # Grenzen berechnen
                x0 = min(w["x0"] for w in block)
                top = min(w["top"] for w in block)
                x1 = max(w["x1"] for w in block)
                bottom = max(w["bottom"] for w in block)
                
                # Rechteck zeichnen (blauer Rahmen, Dicke 1)
                im.draw_rect((x0, top, x1, bottom), stroke="blue", stroke_width=1)

            # Bild speichern – Dateiname z. B. mit Seitenzahl
            im.save(f"page_{page_number + 1}_blocks.png")
            '''

            output_lines = []
            #output_lines.append(f"\nPage {page_number + 1}, Seite {page_number + 1}, Página {page_number + 1}\n")  # Seitenzahl

            for block_idx, block in enumerate(sorted_blocks, 1):
                lines = group_block_into_lines(block)

                chapter_hits = 0
                important_hits = 0
                block_label = None  # Initialisierung hier

                # Regel 1: Nur Wörter mit mehr als 3 Zeichen und keine reinen Zahlen
                for w in block:
                    text = w["text"]
                    if len(text) <= 5 or text.isdigit():
                        continue  # Regel 1 – alle anderen Regeln überspringen

                    size_ratio = w["font_size"] / avg_fontsize if avg_fontsize else 0
                    bold_flag = w["bold_flag"]

                    # Regel 2 – Vorrangig
                    if size_ratio >= 1.15:
                        chapter_hits += 1
                    # Regel 3 – Wenn Regel 2 nicht greift
                    elif bold_flag and size_ratio >= 1:
                        important_hits += 1

                total_hits = chapter_hits + important_hits

                # Regel 4 – Entscheidung auf Basis der Anzahl Treffer
                if total_hits > 1:
                    block_label = "IMPORTANT"
                elif total_hits == 1:
                    if chapter_hits == 1:
                        block_label = "CHAPTER"
                    elif important_hits == 1:
                        block_label = "IMPORTANT"

                output_lines.append("")  # Leerzeile vor Block
            
                for line_idx, line in enumerate(lines):
                    line_text = " ".join(w["text"] for w in line)
                    if line_idx == 0 and block_label:
                        line_text = f"[{block_label}] {line_text}"
                    output_lines.append(line_text)

                    

            # Tabellen anhängen (wie gehabt)
            for idx, tbl in enumerate(tables_json, 1):
                output_lines.append(f'"table {idx}":\n{tbl}')

            return page_number, "\n".join(output_lines)


    except Exception as e:
        msg = str(e).strip() or f"{type(e).__name__} (no message)"
        return args[0], f"[ERROR] Seite {args[0]+1}: {msg}"



def run_serial(path, page_number, tracker=None, progress_callback=None, stop_flag=None):
    results = []
    for i in range(page_number):
        if stop_flag and stop_flag.is_set():
            break
        result = process_page_worker((i, path,))
        results.append(result)
        if tracker is not None:
            tracker.update()
        if progress_callback and tracker is not None:
            report_status(tracker, progress_callback)
    return results




def run_parallel(path, page_number, tracker=None, progress_callback=None, stop_flag=None):
    args = [(i, path) for i in range(page_number)]  # stop_flag entfernt
    results = [None] * page_number

    def callback(result):
        if result is None:
            return
        page, _ = result
        results[page] = result
        if tracker is not None:
            tracker.update()
        if progress_callback and tracker is not None:
            report_status(tracker, progress_callback)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(page_number, get_physical_cores())
    ) as executor:
        futures = {executor.submit(process_page_worker, arg): arg for arg in args}
        for future in concurrent.futures.as_completed(futures):
            # stop_flag nicht hier prüfen, sondern im Hauptthread
            callback(future.result())

    return [r for r in results if r]


def report_status(tracker, progress_callback=None):
    status = tracker.get_status()
    if progress_callback:
        progress_callback(status)
    else:
        print(f"[STATUS] {status['processed_pages']}/{status['total_pages']} Seiten "
              f"({status['pages_per_sec']:} Seiten/s, "
              f"Elapsed: {status['elapsed_time']} Sek.)"
              f"Est Time: {status['est_time']} Sek.)")


def save_pdf(path, page_number, tracker=None, parallel=False, progress_callback=None, stop_flag=None):
    if stop_flag and stop_flag.is_set():
        return 0

    if parallel:
        results = run_parallel(path, page_number, tracker, progress_callback, stop_flag)
    else:
        results = run_serial(path, page_number, tracker, progress_callback, stop_flag)

    results = [r for r in results if r]  # Filter None (bei Stop)

    results.sort(key=lambda x: x[0])
    text_output = "\n".join(text for _, text in results)

    out_path = os.path.splitext(path)[0] + ".txt"
    with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text_output)

    return page_number



def _process_single_pdf(path):
    suppress_pdfminer_logging()
    try:
        with open(path, "rb") as f:
            parser = PDFParser(f)
            document = PDFDocument(parser)

            if not document.is_extractable:
                raise PDFTextExtractionNotAllowed("Text-Extraktion nicht erlaubt")

            pages = list(PDFPage.create_pages(document))
            return (path, len(pages), None)

    except (PDFEncryptionError, PDFPasswordIncorrect) as e:
        return (path, 0, f"[ERROR] Datei passwortgeschützt: {path} ({type(e).__name__}: {e})\n")
    except PDFSyntaxError as e:
        return (path, 0, f"[ERROR] Ungültige PDF-Syntax: {path} ({type(e).__name__}: {e})\n")
    except PDFTextExtractionNotAllowed as e:
        return (path, 0, f"[ERROR] Text-Extraktion nicht erlaubt: {path} ({type(e).__name__}: {e})\n")
    except Exception as e:
        return (path, 0, f"[ERROR] Fehler bei Datei {path}: {type(e).__name__}: {e}\n")

def get_total_pages(pdf_files, error_callback=None, progress_callback=None):
    suppress_pdfminer_logging()
    total = 0
    page_info = []

    def handle_result(path, count, error):
        nonlocal total
        if error:
            if error_callback:
                error_callback(error)
            else:
                print(error, end="")
        else:
            page_info.append((path, count))
            total += count
            if progress_callback:
                progress_callback(total)  # Rückmeldung an GUI

    if len(pdf_files) > 14:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            results = executor.map(_process_single_pdf, pdf_files)
            for path, count, error in results:
                handle_result(path, count, error)
    else:
        for path in pdf_files:
            path, count, error = _process_single_pdf(path)
            handle_result(path, count, error)

    return page_info, total




# -------------------- GUI --------------------
class FileManager(wx.Frame):
    def __init__(self, parent):
        super().__init__(parent, title="PDF Parser - Sevenof9_v7d", size=(1000, 800))
        self.files = []
        self.InitUI()
        self.stop_flag = threading.Event()

    def InitUI(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox_lbl1 = wx.BoxSizer(wx.HORIZONTAL)

        lbl1 = wx.StaticText(panel, label="Filed PDF files: (with right mouse you can remove and open)")
        hbox_lbl1.Add(lbl1, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=10)

        hbox_lbl1.AddStretchSpacer()  # <== schiebt den Button ganz nach rechts

        help_btn = wx.Button(panel, label="? HELP ?", size=(60, 25))
        help_btn.Bind(wx.EVT_BUTTON, self.ShowHelpText)
        hbox_lbl1.Add(help_btn, flag=wx.RIGHT, border=10)

        vbox.Add(hbox_lbl1, flag=wx.EXPAND | wx.TOP, border=10)


        self.listbox = wx.ListBox(panel, style=wx.LB_EXTENDED)
        self.listbox.Bind(wx.EVT_RIGHT_DOWN, self.OnRightClick)
        self.listbox.Bind(wx.EVT_LISTBOX, self.ShowText)
        vbox.Add(self.listbox, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        self.popup_menu = wx.Menu()
        self.popup_menu.Append(1, "Remove selected")
        self.popup_menu.Append(2, "Open in default PDF app")
        self.popup_menu.Append(3, "Copy File Location")
        self.popup_menu.Append(4, "Open File Location")
        self.Bind(wx.EVT_MENU, self.RemoveFile, id=1)
        self.Bind(wx.EVT_MENU, self.OpenPDF, id=2)
        self.Bind(wx.EVT_MENU, self.CopyFileLocation, id=3)
        self.Bind(wx.EVT_MENU, self.OpenFileLocation, id=4)


        btn_panel = wx.Panel(panel)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        for label, handler in [
            ("Add Folder", self.AddFolder),
            ("Select Files", self.AddFile),
            ("Remove Selected", self.RemoveFile),
            ("Remove All", self.RemoveAll),
            ("Stop Parser", self.StopParser),
            ("Start Parser", self.StartParser)
        ]:
            btn = wx.Button(btn_panel, label=label)
            btn.Bind(wx.EVT_BUTTON, handler)
            if label == "Start Parser":
                self.start_btn = btn  # <-- Referenz merken
            btn_sizer.Add(btn, proportion=1, flag=wx.ALL, border=5)
        btn_panel.SetSizer(btn_sizer)
        vbox.Add(btn_panel, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)


        lbl2 = wx.StaticText(panel, label="Text Frame: (choose PDF to see converted text)")
        vbox.Add(lbl2, flag=wx.LEFT, border=10)

        self.text_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.ShowHelpText(None)
        vbox.Add(self.text_ctrl, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        # Statusanzeige
        stat_grid = wx.FlexGridSizer(1, 5, 5, 55)
        self.lbl_processed_pages = wx.StaticText(panel, label="Processed pages: 0")
        self.lbl_total_pages = wx.StaticText(panel, label="Total pages: 0")
        self.lbl_pages_per_sec = wx.StaticText(panel, label="Pages/sec: 0")
        self.lbl_est_time = wx.StaticText(panel, label="Estimated time (min): 0.0")
        self.lbl_elapsed_time = wx.StaticText(panel, label="Elapsed time: 0.0")
        
        for lbl in [self.lbl_processed_pages, self.lbl_total_pages, self.lbl_pages_per_sec, self.lbl_est_time, self.lbl_elapsed_time]:
            stat_grid.Add(lbl)
        vbox.Add(stat_grid, flag=wx.LEFT | wx.TOP, border=10)

        self.prog_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        vbox.Add(self.prog_ctrl, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        panel.SetSizer(vbox)


    def ShowHelpText(self, event):
        help_text = (
            "	This is a small help\n\n"
            "	• PRE ALPHA version (for ever) •\n"
            "• The generated TXT file has the same name as the PDF file\n"
            "• The TXT file is created in the same directory as the PDF\n"
            "• Older TXT files will be overwritten without prompting\n"
            "• When selecting a folder, subfolders are also selected\n"
            "If:\n"
            "[INFO] File completed: TEST.pdf (X pages)!\n"
            "[INFO] Processing completed\n"
            "-> This only means that all pages have been processed; it does not mean that the quality is good.\n"
            "• An attempt is made to reproduce the layout of the page in columns from left to right and in blocks from top to bottom\n"
            "• An attempt is made to detect regular tables with lines; headers (top or top and left) are assigned to the cells and stored in JSON format in the text file\n"
            "• Adds the label “Page X” at the beginning of every page (absdlute number)\n"
            "• Adds the label “Chapter” for large font and/or “important” for bold font\n"
            "\n"
            "Stop function becomes effective only after the currently processed file\n"
            "When processing large amounts of data, the following should be noted:\n"
            "First, all PDFs are opened once to determine the number of pages:\n"
            "Then, all small PDFs are processed in parallel:\n"
            "Then, each large PDF is processed page by page in parallel:\n"
        )
        self.text_ctrl.SetValue(help_text)
        
        
    def AddFolder(self, event):
        dlg = wx.DirDialog(self, "Select Folder")
        if dlg.ShowModal() == wx.ID_OK:
            for root, _, files in os.walk(dlg.GetPath()):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        path = os.path.normpath(os.path.join(root, f))
                        if path not in self.files:
                            self.files.append(path)
                            self.listbox.Append(path)
        dlg.Destroy()

    def AddFile(self, event):
        with wx.FileDialog(self, "Select PDF Files", wildcard="PDF files (*.pdf)|*.pdf",
                           style=wx.FD_OPEN | wx.FD_MULTIPLE) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                for path in dlg.GetPaths():
                    if path not in self.files:
                        self.files.append(path)
                        self.listbox.Append(path)

    def RemoveFile(self, event):
        for i in reversed(self.listbox.GetSelections()):
            self.listbox.Delete(i)
            del self.files[i]
        self.text_ctrl.Clear()

    def RemoveAll(self, event):
        self.listbox.Clear()
        self.files.clear()
        self.text_ctrl.Clear()

    def OpenPDF(self, event):
        i = self.listbox.GetSelections()
        if i:
            path = self.files[i[0]]
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.call(["open", path])
            else:
                subprocess.call(["xdg-open", path])
                
    def CopyFileLocation(self, event):
        sel = self.listbox.GetSelections()
        if sel:
            path = self.files[sel[0]]
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(wx.TextDataObject(path))
                wx.TheClipboard.Close()

    def OpenFileLocation(self, event):
        sel = self.listbox.GetSelections()
        if sel:
            folder = os.path.dirname(self.files[sel[0]])
            if platform.system() == "Windows":
                subprocess.Popen(f'explorer "{folder}"')
            elif platform.system() == "Darwin":
                subprocess.call(["open", folder])
            else:
                subprocess.call(["xdg-open", folder])


    def OnRightClick(self, event):
        if self.listbox.GetSelections():
            self.PopupMenu(self.popup_menu, event.GetPosition())

    def StartParser(self, event):
        if not self.files:
            wx.MessageBox("Please select files first.", "Hinweis", wx.OK | wx.ICON_INFORMATION)
            wx.CallAfter(self.start_btn.Enable)  # <-- wieder aktivieren
            return


        self.start_btn.Disable()
        self.stop_flag.clear()
        self.prog_ctrl.Clear()

        def error_callback(msg):
            wx.CallAfter(self.AppendProg, msg)
        
        def update_total_pages_live(new_total):
            wx.CallAfter(self.lbl_total_pages.SetLabel, f"Total pages: {new_total}")


        page_info, total_pages = get_total_pages(
            self.files,
            error_callback=error_callback,
            progress_callback=update_total_pages_live
        )

        if total_pages == 0:
            self.AppendProg("[INFO] No pages found.\n")
            wx.CallAfter(self.start_btn.Enable)  # <-- wieder aktivieren
            return

        tracker = StatusTracker(total_pages)

        def gui_progress_callback(status):
            wx.CallAfter(self.lbl_processed_pages.SetLabel, f"Processed pages: {status['processed_pages']}")
            wx.CallAfter(self.lbl_total_pages.SetLabel, f"Total pages: {status['total_pages']}")
            wx.CallAfter(self.lbl_pages_per_sec.SetLabel, f"Pages/sec: {status['pages_per_sec']:}")
            wx.CallAfter(self.lbl_est_time.SetLabel, f"Estimated time (min): {status['est_time']:}")
            wx.CallAfter(self.lbl_elapsed_time.SetLabel, f"Elapsed time: {status['elapsed_time']}")

        throttled_gui_callback = throttle_callback(gui_progress_callback, 100)

        def background():
            small = [p for p in page_info if p[1] <= PARALLEL_THRESHOLD]
            large = [p for p in page_info if p[1] > PARALLEL_THRESHOLD]

            # Verarbeite kleine Dateien je in einem eigenen Prozess
            if small:
                max_workers = max(1, min(len(small), get_physical_cores()))
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for path, count in small:
                        if self.stop_flag.is_set():
                            break
                        future = executor.submit(save_pdf, path, count, None, False, None)
                        futures[future] = (path, count)

                    for future in concurrent.futures.as_completed(futures):
                        if self.stop_flag.is_set():
                            break
                        path, count = futures[future]
                        try:
                            pages_processed = future.result()
                            tracker.update(pages_processed)
                            throttled_gui_callback(tracker.get_status())
                            wx.CallAfter(self.AppendProg, f"[INFO] File ready: {path} ({pages_processed} Seiten)\n")
                        except Exception as e:
                            wx.CallAfter(self.AppendProg, f"[ERROR] File {path}: {str(e)}\n")

            # Verarbeite große Dateien Seite für Seite parallel
            for path, count in large:
                if self.stop_flag.is_set():
                    break

                try:
                    pages_processed = save_pdf(
                        path,
                        count,
                        tracker,
                        parallel=True,
                        progress_callback=throttled_gui_callback,
                        stop_flag=self.stop_flag
                    )
                    if pages_processed:
                        wx.CallAfter(
                            self.AppendProg,
                            f"[INFO] File ready: {path} ({pages_processed} Seiten)\n"
                        )
                    else:
                        wx.CallAfter(
                            self.AppendProg,
                            f"[INFO] Stopped: {path}\n"
                        )
                except Exception as e:
                    wx.CallAfter(
                        self.AppendProg,
                        f"[ERROR] File {path}: {str(e)}\n"
                    )



            wx.CallAfter(self.AppendProg, "\n[INFO] Processing completed.\n")
            wx.CallAfter(self.start_btn.Enable)  # <-- wieder aktivieren
            self.stop_flag.clear()

        threading.Thread(target=background, daemon=True).start()


    def StopParser(self, event):
        self.stop_flag.set()
        self.AppendProg("[INFO] Processing Stopped...\n")

    
    def ShowText(self, event):
        sel = self.listbox.GetSelections()
        if not sel:
            return
        txt_path = os.path.splitext(self.files[sel[0]])[0] + ".txt"
        self.text_ctrl.Clear()
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                self.text_ctrl.SetValue(f.read())
        else:
            self.text_ctrl.SetValue("[No .txt file found]")

    def AppendProg(self, text):
        self.prog_ctrl.AppendText(text)


# -------------------- Einstiegspunkt --------------------
def main():
    if len(sys.argv) > 1:
        pdf_files = sys.argv[1:]
        page_info, total_pages = get_total_pages(pdf_files)
        tracker = StatusTracker(total_pages)

        def cli_callback(status):
            print(json.dumps(status))

        for path, count in page_info:
            save_pdf(path, count, tracker, parallel=(count > PARALLEL_THRESHOLD), progress_callback=cli_callback)
    else:
        app = wx.App(False)
        frame = FileManager(None)
        frame.Show()
        app.MainLoop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
