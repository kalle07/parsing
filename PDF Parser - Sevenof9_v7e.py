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
from concurrent.futures import ThreadPoolExecutor, as_completed
import pdfplumber
import psutil
import logging
from pdfminer.pdfparser import PDFParser, PDFSyntaxError
from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError, PDFPasswordIncorrect
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from rtree import index
import numpy as np
from typing import Any, Dict, Iterable, List, Sequence, Tuple, ClassVar
from dataclasses import dataclass, field, replace
import math

# --------------------------------------------------------------
#   1. Configuration & compiled regexes
# --------------------------------------------------------------
PARALLEL_THRESHOLD = 16


@dataclass(frozen=True)
class Config:
    PARALLEL_THRESHOLD: int = 16          # pages per file before we switch to parallel mode
    
    # Class‑level constant – accessible via Config.TEXT_EXTRACT_SETTINGS
    TEXT_EXTRACT_SETTINGS: ClassVar[Dict[str, Any]] = {
        "x_tolerance": 1.5,
        "y_tolerance": 2.5,
        "keep_blank_chars": False,
        "use_text_flow": False,
    }
    
    LEFT_RIGHT_MARGIN_PCT: float = 5.3
    TOP_BOTTOM_MARGIN_PCT: float = 6.0



#CID_PATTERN = re.compile(r"\$cid:\d+$")  # Fixed: removed incorrect trailing $
CID_PATTERN = re.compile(r"\(cid:\d+\)")
# NON_PRINTABLE_RE pattern
NON_PRINTABLE_RE = re.compile(r"[^\u0000-\uFFFF]", re.DOTALL)

def clean_cell_text(text):
    if not isinstance(text, str):
        return ""
    # Remove hyphenated line endings first (only at end of lines)
    #text = re.sub(r'-(?=\s*$)', '', text)
    #text = re.sub(r'-\s+', '', text)
    #text = re.sub(r'-(?:\s+)?$', '', text)
    #text = re.sub(r'-(?=\s*$)', '', text)  # Remove trailing hyphens 
    #text = re.sub(r'-\s+', ' ', text)      # Replace hyphen + spaces with space
    #text = text.replace("-\n", "")
    # Remove CID patterns like (cid:79), (cid:111), etc.
    text = CID_PATTERN.sub("", text)
    # Remove other non-printable characters
    text = NON_PRINTABLE_RE.sub("", text)
    return text.strip()

def clamp_bbox(bbox, page_width, page_height, p=3):
    x0, top, x1, bottom = bbox
    x0 = max(0, min(x0, page_width))
    x1 = max(0, min(x1, page_width))
    top = max(0, min(top, page_height))
    bottom = max(0, min(bottom, page_height))
    return round(x0, p), round(top, p), round(x1, p), round(bottom, p)


# Regexes – compile once for speed
#CID_RE = re.compile(r"^cid:\d+$")

# --------------------------------------------------------------
#   2. Small utilities
# --------------------------------------------------------------

def get_physical_cores():
    count = psutil.cpu_count(logical=False)
    return max(1, count if count else 1)  # fallback = 1
cores = get_physical_cores()

# GUI update interval
def throttle_callback(callback, interval_ms=1):
    last_called = 0

    def wrapper(status):
        nonlocal last_called
        now = time.time() * 1000  # Time in ms
        if now - last_called >= interval_ms:
            last_called = now
            callback(status)
    return wrapper


def clamp_bbox(bbox: Tuple[float, float, float, float], w: float, h: float) -> Tuple[int, int, int, int]:
    """Clamp a bbox to the page dimensions and round to nearest integer."""
    x0, top, x1, bottom = bbox
    return (
        round(max(0, min(x0, w))),
        round(max(0, min(top, h))),
        round(min(x1, w)),
        round(min(bottom, h)),
    )


def is_valid_cell(cell: Any) -> bool:
    """Return True if a cell contains something meaningful."""
    return bool(str(cell).strip() and len(str(cell).strip()) > 1)



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

suppress_pdfminer_logging()

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

# --------------------------------------------------------------
#   3. Data models
# --------------------------------------------------------------

@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    font_name: str
    bold: bool


@dataclass
class Block:
    words: List[Word] = field(default_factory=list)

    def bbox(self) -> Tuple[float, float, float, float]:
        if not self.words:
            return 0.0, 0.0, 0.0, 0.0
        x0 = min(w.x0 for w in self.words)
        y0 = min(w.y0 for w in self.words)
        x1 = max(w.x1 for w in self.words)
        y1 = max(w.y1 for w in self.words)
        return (x0, y0, x1, y1)


@dataclass
class ImageInfo:
    bbox: Tuple[float, float, float, float]
    obj: Any  # raw image dictionary from pdfplumber


# --------------------------------------------------------------
#   4. Union‑Find clustering
# --------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# --------------------------------------------------------------
#   4. Union‑Find clustering
# --------------------------------------------------------------

def cluster_words(words: Sequence[Word], max_dx: int, max_dy: int) -> List[Block]:
    """Group words into blocks based on proximity using optimized neighbor search."""
    n = len(words)
    if n == 0:
        return []

    uf = _UnionFind(n)

    def is_neighbor(word1: Word, word2: Word) -> bool:
        dx = max(0.0, max(word1.x0 - word2.x1, word2.x0 - word1.x1))
        dy = max(0.0, max(word1.y0 - word2.y1, word2.y0 - word1.y0))
        return dx <= max_dx and dy <= max_dy

    # Track which words have already been processed (4 neighbors found)
    processed = [False] * n
    
    for i in range(n):
        if processed[i]:
            continue
            
        neighbor_count = 0
        neighbors_found = []
        
        # Check against ALL other words - the key optimization is to stop early
        for j in range(n):
            if i == j:
                continue
                
            word1, word2 = words[i], words[j]
            
            if is_neighbor(word1, word2):
                neighbors_found.append(j)
                neighbor_count += 1
                
                # Early stopping as per your requirements:
                # 1. If we have at least 2 neighbors, the word belongs to a text block
                # 2. If we already have 4 neighbors (max possible in 2D), stop processing this word
                if neighbor_count >= 2: 
                    # Union with all found neighbors so far
                    for k in neighbors_found:
                        uf.union(i, k)
                    
                    # Second early stop - no need to check further when 4 neighbors found
                    if neighbor_count >= 4:
                        processed[i] = True
                        break
                        
        # Continue processing other words even if current word had < 2 neighbors

    # Build clusters
    clusters: Dict[int, List[Word]] = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(words[idx])

    # Return as list of Blocks
    return [Block(wlist) for wlist in clusters.values()]






# --------------------------------------------------------------
#   5. Character index (vectorised)
# --------------------------------------------------------------

@dataclass
class CharIndex:
    xs0: np.ndarray
    xs1: np.ndarray
    tops: np.ndarray
    bottoms: np.ndarray
    texts: List[str]
    fonts: List[str]
    sizes: np.ndarray

    @classmethod
    def build(cls, chars: Sequence[Dict[str, Any]]) -> "CharIndex":
        return cls(
            xs0=np.array([float(c["x0"]) for c in chars]),
            xs1=np.array([float(c["x1"]) for c in chars]),
            tops=np.array([float(c["top"]) for c in chars]),
            bottoms=np.array([float(c["bottom"]) for c in chars]),
            texts=[c.get("text", "") for c in chars],
            fonts=[c.get("fontname", "") for c in chars],
            sizes=np.array([float(c.get("size", 0)) for c in chars]),
        )

    def inside(self, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        return (
            (self.xs0 >= x0)
            & (self.xs1 <= x1)
            & (self.tops >= y0)
            & (self.bottoms <= y1)
        )


# --------------------------------------------------------------
#   6. Core extraction helpers
# --------------------------------------------------------------

def _extract_tables(page: pdfplumber.page.Page) -> List[Tuple[str, Any]]:
    """Return a list of JSON strings representing tables."""
    suppress_pdfminer_logging()
    raw_tables = page.extract_tables({"text_x_tolerance": Config.TEXT_EXTRACT_SETTINGS["x_tolerance"]})
    jsons = []

    for tbl in raw_tables:
        if not tbl or len(tbl) < 2:  # ignore tiny tables
            continue

        # filter out empty tables
        if all(all(not is_valid_cell(cell) for cell in row if row) for row in tbl):
            continue

        cleaned = [[clean_cell_text(c) for c in row] for row in tbl]
        header = cleaned[0]

        if header[0].strip() == "":
            # corner‑empty table
            col_headers = header[1:]
            row_headers = [row[0] for row in cleaned[1:]]
            data_rows = cleaned[1:]

            table_dict = {}
            for rh, row in zip(row_headers, data_rows):
                table_dict[rh] = dict(zip(col_headers, row[1:]))
        else:
            # normal header‑table
            headers = header
            data_rows = cleaned[1:]
            table_dict = [dict(zip(headers, row)) for row in data_rows if len(row) == len(headers)]

        jsons.append(json.dumps(table_dict, indent=1, ensure_ascii=False))
    return jsons


def _filter_words(
    words: List[Dict[str, Any]],
    tables_bboxes: List[Tuple[int, int, int, int]],
) -> List[Dict[str, Any]]:
    """Remove words that overlap a table or contain non‑printable chars."""
    filtered = []
    for w in words:
        x0, top = float(w["x0"]), float(w["top"])
        if any(bx0 <= x0 <= bx2 and by0 <= top <= by3 for bx0, by0, bx2, by3 in tables_bboxes):
            continue
        clean_text = clean_cell_text(w["text"])
        if NON_PRINTABLE_RE.search(w["text"]):
            continue
        w["text"] = clean_text
        filtered.append(w)
    return filtered


def _build_word_info(
    words: List[Dict[str, Any]],
    char_index: CharIndex,
) -> List[Word]:
    """Convert raw pdfplumber words into Word dataclass instances."""
    def is_bold(name: str) -> bool:
        n = name.lower()
        return "bold" in n or "bd" in n or "black" in n

    word_objs: List[Word] = []
    for w in words:
        x0, y0, x1, y1 = map(float, (w["x0"], w["top"], w["x1"], w["bottom"]))
        mask = char_index.inside(x0, x1, y0, y1)
        sizes = char_index.sizes[mask]
        fonts = [char_index.fonts[i] for i in np.nonzero(mask)[0]]
        bolds = [is_bold(f) for f in fonts]

        font_size = float(sizes.max()) if sizes.size else 0.0
        word_objs.append(
            Word(
                text=w["text"],
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                font_size=font_size,
                font_name=fonts[0] if fonts else "Unknown",
                bold=bool(bolds),
            )
        )
    return word_objs

'''
def _group_blocks(
    words: List[Word],
    page_width: float,
    page_height: float,
) -> List[Block]:
    """Cluster words into logical blocks using Union-Find."""
    # thresholds in pixel – derived from percentages
    max_dx = int(round(page_width * 0.0129))   # 1.29 %, ~15px
    max_dy = int(round(page_height * 0.0143))  # 1.43 %, ~25px

    blocks = cluster_words(words, max_dx, max_dy)
    
    # Filter out empty blocks and single-character printable blocks
    filtered_blocks = []
    for block in blocks:
        # Combine all text from words in this block
        combined_text = " ".join(w.text for w in block.words)
        
        # Check if the block is not empty after stripping whitespace
        stripped_text = combined_text.strip()
        
        # Filter out blocks that are:
        # 1. Empty (only whitespace)
        # 2. Contain only one printable character
        if stripped_text and len(stripped_text) > 1:
            # Additional check for single printable characters
            # Remove all whitespace characters to count actual content
            printable_chars = ''.join(char for char in stripped_text if not char.isspace())
            
            # Only keep blocks with more than one printable character
            if len(printable_chars) > 1:
                filtered_blocks.append(block)
    
    return filtered_blocks
'''

def _group_blocks(
    words: List[Word],
    page_width: float,
    page_height: float,
) -> List[Block]:
    """Cluster words into logical blocks using Union-Find, cleaning text and merging hyphen-split words."""

    merged_words = []
    skip_next = False

    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue

        text = word.text.strip()

        # If word ends with a hyphen (possibly at a line break), merge with next
        if text.endswith('-') and i + 1 < len(words):
            next_word = words[i + 1]
            merged_text = re.sub(r'-\s*$', '', text) + next_word.text.lstrip()
            merged_word = replace(word, text=merged_text)
            merged_words.append(merged_word)
            skip_next = True
        else:
            # Clean trailing hyphens and extra spaces (no merge)
            cleaned_text = re.sub(r'-\s*$', '', text).strip()
            if cleaned_text != text:
                word = replace(word, text=cleaned_text)
            merged_words.append(word)

    # thresholds in pixel – derived from percentages
    max_dx = int(round(page_width * 0.0151))   # 1.51 %, ~9px
    max_dy = int(round(page_height * 0.0143))  # 1.43 %, ~12px

    blocks = cluster_words(merged_words, max_dx, max_dy)
    
    # Filter out empty blocks and single-character printable blocks
    filtered_blocks = []
    for block in blocks:
        combined_text = " ".join(w.text for w in block.words)
        stripped_text = combined_text.strip()

        if stripped_text and len(stripped_text) > 1:
            printable_chars = ''.join(c for c in stripped_text if not c.isspace())
            if len(printable_chars) > 1:
                filtered_blocks.append(block)
    
    return filtered_blocks






# --------------------------------------------------------------
#   7. Page worker – orchestrator
# --------------------------------------------------------------

def process_page_worker(args: Tuple[int, str]) -> Tuple[int, str]:
    """Process a single page; returns (page_number, rendered_text)."""
    try:
        page_no, path = args

        with pdfplumber.open(path) as pdf:
            page = pdf.pages[page_no]
            w, h = page.width, page.height

            # Crop margins
            margin_x = w * Config.LEFT_RIGHT_MARGIN_PCT / 100.0
            margin_y = h * Config.TOP_BOTTOM_MARGIN_PCT / 100.0
            cropped_page = page.crop((margin_x, margin_y, w - margin_x, h - margin_y))

            # ---------- Tables ----------
            tables_json = _extract_tables(cropped_page)

            # ---------- Words ----------
            table_bboxes = [clamp_bbox(t.bbox, w, h) for t in cropped_page.find_tables()]
            raw_words = cropped_page.extract_words(**Config.TEXT_EXTRACT_SETTINGS)
            # Clean line break artifacts from PDF text extraction
            filtered_raw = _filter_words(raw_words, table_bboxes)
            char_index = CharIndex.build(cropped_page.chars)

            words = _build_word_info(filtered_raw, char_index)
            avg_font_size = float(np.mean([w.font_size for w in words])) if words else 0.0
            
            # ---------- Blocks ----------
            blocks = _group_blocks(words, w, h)

            # ---------- Sorting (reading order) ----------
            def reading_score(block: Block) -> Tuple[float, float]:
                x0, y0, x1, y1 = block.bbox()
                height = y1 - y0
                width = x1 - x0
                area_log = math.log1p(width * height)
                return (y0 * 0.7 + x0 * 0.3 - area_log * 0.05, y0)

            blocks.sort(key=reading_score)

            # ---------- Images ----------
            images: List[ImageInfo] = []
            for im in cropped_page.images:
                img_bbox = (
                    float(im["x0"]),
                    h - float(im["y1"]),
                    float(im["x1"]),
                    h - float(im["y0"]),
                )
                images.append(ImageInfo(bbox=img_bbox, obj=im))

            # ---------- Assemble output ----------
            lines: List[str] = [f"\n\n--- Page {page_no + 1} ---\n\n"]

            # ---------- Identify small blocks near large blocks ----------
            large_blocks: List[Block] = []
            small_blocks: List[Block] = []

            for block in blocks:
                x0, y0, x1, y1 = block.bbox()
                area = (x1 - x0) * (y1 - y0)
                
                if area > 3000:
                    large_blocks.append(block)
                else:
                    small_blocks.append(block)

            # Group small blocks near large blocks
            nearby_small_blocks: List[Tuple[Block, List[List[Word]]]] = []
            
            for small_block in small_blocks:
                x0_s, y0_s, x1_s, y1_s = small_block.bbox()
                small_area = (x1_s - x0_s) * (y1_s - y0_s)
                
                # Only process small blocks under 7000 pixels
                if small_area >= 3000:
                    continue
                    
                # Check proximity to large blocks
                for large_block in large_blocks:
                    x0_l, y0_l, x1_l, y1_l = large_block.bbox()
                    large_area = (x1_l - x0_l) * (y1_l - y0_l)
                    
                    # Only consider large blocks over 7000 pixels
                    if large_area <= 3000:
                        continue
                        
                    # Calculate distance between blocks
                    dx = max(0, max(x0_s, x0_l) - min(x1_s, x1_l))
                    dy = max(0, max(y0_s, y0_l) - min(y1_s, y1_l))
                    
                    # If within 25 pixels proximity
                    if dx >= 25 and dy >= 25:
                        # Sort words in the block for consistent output
                        sorted_words = sorted(small_block.words, key=lambda w: (w.y0, w.x0))
                        nearby_small_blocks.append((small_block, [sorted_words]))
                        break

            # ---------- Process regular blocks ----------
            for block in blocks:
                # Skip already processed small blocks that are near large ones
                if any(block is small_block for small_block, _ in nearby_small_blocks):
                    continue
                    
                # ------------------------------------------------------------------
                #   One‑line per block (preserve any internal \n or \r)
                # ------------------------------------------------------------------
                sorted_words = sorted(block.words, key=lambda w: (w.y0, w.x0))
                combined_text = " ".join(w.text for w in sorted_words)

                # ------------------------------------------------------------------
                #   Labeling heuristics (unchanged from your original logic)
                # ------------------------------------------------------------------
                chapter_hits = 0
                important_hits = 0
                for wobj in block.words:
                    # Skip words with fewer than 4 letters and all numbers (no alphabetic characters)
                    if len(wobj.text) < 4 and not any(c.isalpha() for c in wobj.text):
                        continue
                    size_ratio = (
                        wobj.font_size / avg_font_size if avg_font_size else 0.0
                    )
                    if size_ratio >= 1.15:
                        chapter_hits += 1
                    elif wobj.bold and size_ratio >= 1.0:
                        important_hits += 1

                label: str | None = None
                hits = chapter_hits + important_hits
                if hits > 1 or (hits == 1 and chapter_hits):
                    label = "CHAPTER" if chapter_hits else "IMPORTANT"

                # ------------------------------------------------------------------
                #   Append block text (single line) and an empty line afterwards
                # ------------------------------------------------------------------
                if label:
                    line_text = f"[{label}] {combined_text}"
                else:
                    line_text = combined_text

                lines.append(line_text)
                lines.append("")          # <‑ blank line after every text block

            # ---------- Tables ----------
            for idx, tbl_json in enumerate(tables_json, 1):
                lines.append(f'"table {idx}":\n{tbl_json}')

            # ---------- Nearby small blocks (near large blocks) ----------
            if nearby_small_blocks:
                lines.append("\n--- Blocks with unsorted small text snippets far away from large blocks. ---")
                for i, (blk, lns) in enumerate(nearby_small_blocks, 1):
                    lines.append(f"Block {i}:")
                    for j, line_words in enumerate(lns):
                        txt = " ".join(w.text for w in line_words)
                        lines.append(txt)

            return page_no, "\n".join(lines)


    except Exception as exc:  # pragma: no cover
        err_msg = f"[ERROR] Seite {page_no + 1}: {exc.__class__.__name__}: {exc}"
        logging.exception(err_msg)
        return page_no, err_msg


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
    args = [(i, path) for i in range(page_number)]
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

    max_workers = min(page_number, get_physical_cores())
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_page_worker, arg): arg for arg in args}
        for future in concurrent.futures.as_completed(futures):
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
        super().__init__(parent, title="PDF Parser - Sevenof9_v7e", size=(1000, 800))
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



