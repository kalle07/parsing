import os  # OS module for interacting with the operating system (file management, etc.)
import sys  # Provides access to system-specific parameters and functions
import tkinter as tk  # GUI module for creating desktop applications
from tkinter import filedialog, messagebox  # Additional tkinter components for file dialogs and message boxes
import subprocess  # Module to run system commands
import threading  # Threading module to run tasks concurrently
import tempfile  # Module to create temporary files and directories
import shutil  # Module for file operations like copy, move, and delete
import json  # JSON module for working with JSON data
import logging  # Logging module for tracking events and errors
import pdfplumber  # Library for extracting text and tables from PDFs
from pdfplumber.utils import get_bbox_overlap, obj_to_bbox  # Helper functions from pdfplumber for working with bounding boxes
from pdfplumber.utils.exceptions import PdfminerException  # Exception related to PDF processing
from joblib import delayed, cpu_count, parallel_backend, Parallel  # Joblib for parallel processing and optimization
import multiprocessing  # Module for parallel processing using multiple CPU cores
from multiprocessing import Pool  # Pool class for parallelizing tasks across multiple processes


# ========================
# Parser Configuration
# ========================

TEXT_EXTRACTION_SETTINGS = {
    "x_tolerance": 1,  # Horizontal tolerance for text extraction
    "y_tolerance": 3,  # Vertical tolerance for text extraction
    "keep_blank_chars": False,  # Option to retain blank characters in the extracted text
    "use_text_flow": True  # Option to use text flow for better structure
}

# Suppress stderr output on Windows platform to avoid cluttering the console
if sys.platform == "win32":
    sys.stderr = open(os.devnull, 'w')  # Redirect stderr to null

PARALLEL_THRESHOLD = 16  # Number of pages to use for deciding between serial or parallel processing

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

# Function to clean up text by removing unwanted hyphenations and newlines
def clean_cell_text(text):
    if not isinstance(text, str):  # If text is not a string, return empty string
        return ""
    text = text.replace("-\n", "").replace("\n", " ")  # Remove hyphenated line breaks and replace newlines with space
    return " ".join(text.split())  # Split text into words and join with single spaces

# Function to safely clean and join row cell data
def safe_join(row):
    return [clean_cell_text(str(cell)) if cell is not None else "" for cell in row]  # Clean each cell in the row, or return empty if None

# Function to clamp bounding box coordinates within page boundaries
def clamp_bbox(bbox, page_width, page_height):
    x0, top, x1, bottom = bbox  # Extract bounding box coordinates
    # Ensure each coordinate is within the page width and height limits
    x0 = max(0, min(x0, page_width))
    x1 = max(0, min(x1, page_width))
    top = max(0, min(top, page_height))
    bottom = max(0, min(bottom, page_height))
    return (x0, top, x1, bottom)  # Return the adjusted bounding box

# Function to process a single PDF page
def process_page(args):
    suppress_pdfminer_logging()  # Suppress unnecessary PDFMiner logging
    try:
        page_number, pdf_path, text_settings = args  # Extract page number, PDF path, and text extraction settings
        with pdfplumber.open(pdf_path) as pdf:  # Open the PDF using pdfplumber
            page = pdf.pages[page_number]  # Get the specific page
            output = f"Page {page_number + 1}\n"  # Add page number to the output
            width, height = page.width, page.height  # Get page dimensions

            filtered_page = page  # Initialize filtered page
            table_bboxes = []  # List to hold bounding boxes of tables
            table_json_outputs = []  # List to hold JSON output of tables

            # Iterate through all tables found on the page
            for table in page.find_tables():
                bbox = clamp_bbox(table.bbox, width, height)  # Adjust the bounding box to fit within the page
                table_bboxes.append(bbox)  # Add the bounding box to the list

                if not page.crop(bbox).chars:  # Skip tables that have no characters
                    continue

                # Filter out any elements that overlap with the table's bounding box
                filtered_page = filtered_page.filter(
                    lambda obj: get_bbox_overlap(obj_to_bbox(obj), bbox) is None
                )

                # Extract the table data and structure it
                table_data = table.extract()
                if table_data and len(table_data) >= 1:  # Ensure there is data in the table
                    headers = safe_join(table_data[0])  # Clean and join the headers
                    rows = [safe_join(row) for row in table_data[1:]]  # Clean and join the table rows
                    json_table = [dict(zip(headers, row)) for row in rows]  # Create a JSON object from headers and rows
                    table_json_outputs.append(json.dumps(json_table, indent=1, ensure_ascii=False))  # Convert table data to JSON

            # Extract words outside the tables
            words_outside_tables = [
                word for word in page.extract_words(**text_settings)  # Extract words from the page using the settings
                if not any(
                    bbox[0] <= float(word['x0']) <= bbox[2] and
                    bbox[1] <= float(word['top']) <= bbox[3]
                    for bbox in table_bboxes  # Ensure word is not inside any table bounding box
                )
            ]

            current_y = None  # Track vertical position of words
            line = []  # List to hold words for the current line
            text_content = ""  # Store the extracted text content

            # Iterate through words and group them into lines
            for word in words_outside_tables:
                if current_y is None or abs(word['top'] - current_y) > 10:  # Start a new line if Y position changes significantly
                    if line:  # If there's a previous line, join and add it to text content
                        text_content += " ".join(line) + "\n"
                    line = [word['text']]  # Start a new line with the current word
                    current_y = word['top']  # Update the current Y position
                else:
                    line.append(word['text'])  # Append the word to the current line
            if line:  # Add the last line to the text content
                text_content += " ".join(line) + "\n"

            output += text_content.strip() + "\n"  # Add the final text content for the page

            # Add table JSON outputs to the page output
            for idx, table in enumerate(table_json_outputs, start=1):
                output += f'"table {idx}":\n{table}\n'

            return page_number, output  # Return the processed page number and output content

    except Exception as e:
        return args[0], f"[ERROR] Page {args[0]+1} ({args[1]}): {str(e)}"  # Return an error message if an exception occurs

# Function to process the entire PDF document
def process_pdf(pdf_path):
    suppress_pdfminer_logging()  # Suppress unnecessary logging
    try:
        if not os.path.exists(pdf_path):  # Check if the file exists
            return f"[ERROR] File not found: {pdf_path}"  # Return error message if file does not exist

        print(f"[INFO] Starting processing: {pdf_path}")  # Log the start of processing
        try:
            with pdfplumber.open(pdf_path) as pdf:  # Open the PDF using pdfplumber
                num_pages = len(pdf.pages)  # Get the number of pages in the PDF
        except PdfminerException as e:
            return f"[ERROR] Cannot open PDF: {pdf_path} – {str(e)}"  # Return error if the PDF cannot be opened
        except Exception as e:
            return f"[ERROR] General error opening PDF: {pdf_path} – {str(e)}"  # Return general error if any exception occurs

        pages = [(i, pdf_path, TEXT_EXTRACTION_SETTINGS) for i in range(num_pages)]  # Prepare the pages for processing

        try:
            results = run_serial(pages) if num_pages <= PARALLEL_THRESHOLD else run_parallel(pages)  # Run serial or parallel processing
        except (EOFError, BrokenPipeError, KeyboardInterrupt):
            return "[INFO] Processing was interrupted."  # Handle interruptions during processing

        sorted_results = sorted(results, key=lambda x: x[0])  # Sort results by page number
        final_output = "\n".join(text for _, text in sorted_results)  # Combine all page results into a single string

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]  # Get the base name of the PDF file
        output_dir = os.path.dirname(pdf_path)  # Get the directory of the PDF file
        output_path = os.path.join(output_dir, f"{base_name}.txt")  # Generate the output file path

        with open(output_path, "w", encoding="utf-8", errors="ignore") as f:  # Open the output file for writing
            f.write(final_output)  # Write the final output to the file

        print(f"[INFO] Processing complete: {output_path}")  # Log the successful processing completion

    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        return "[INFO] Processing interrupted by user."  # Handle user interruptions
    except Exception as e:
        return f"[ERROR] Unexpected error with '{pdf_path}': {str(e)}"  # Handle unexpected errors during processing

# Function to run the PDF processing serially (one page at a time)
def run_serial(pages):
    return [process_page(args) for args in pages]  # Process each page in sequence

# Function to run the PDF processing in parallel (across multiple cores)
def run_parallel(pages):
    available_cores = max(1, cpu_count() - 2)  # Calculate the number of available CPU cores, leaving 2 for system processes
    num_cores = min(available_cores, len(pages))  # Limit the number of cores based on the number of pages
    print(f"Starting parallel processing with {num_cores} cores...")  # Log the number of cores used
    with Pool(processes=num_cores) as pool:  # Create a pool of processes
        return pool.map(process_page, pages)  # Distribute the page processing across the available cores

# Main function to process a list of PDFs
def process_pdfs_main():
    suppress_pdfminer_logging()  # Suppress unnecessary logging
    pdf_files = sys.argv[1:]  # Get PDF file paths from command-line arguments
    if not pdf_files:  # Check if any PDFs are provided
        print("No PDF files provided.")  # Log message if no PDFs are provided
        return

    small_pdfs = []  # List to store small PDFs (less than the parallel threshold)
    large_pdfs = []  # List to store large PDFs (greater than the parallel threshold)

    # Categorize PDFs into small and large based on the number of pages
    for path in pdf_files:
        if not os.path.exists(path):  # Check if the file exists
            print(f"File not found: {path}")  # Log error if file does not exist
            continue
        try:
            with pdfplumber.open(path) as pdf:  # Open the PDF
                if len(pdf.pages) <= PARALLEL_THRESHOLD:  # If the PDF has fewer pages than the threshold
                    small_pdfs.append(path)  # Add to small PDFs list
                else:
                    large_pdfs.append(path)  # Add to large PDFs list
        except PdfminerException:
            print(f"[ERROR] Password-protected PDF skipped: {path}")  # Log if the PDF is password-protected
        except Exception as e:
            print(f"[ERROR] Error opening {path}: {str(e)}")  # Log any other errors when opening the PDF

    # Process small PDFs in parallel (if there are any)
    if small_pdfs:
        available_cores = max(1, cpu_count() - 2)  # Determine the number of available cores
        num_cores = min(available_cores, len(small_pdfs))  # Use the lesser of available cores or small PDFs count
        print(f"\n[Phase 1] Starting parallel processing of small PDFs with {num_cores} cores...")  # Log processing start
        results = Parallel(n_jobs=num_cores)(  # Run parallel processing for small PDFs
            delayed(process_pdf)(path) for path in small_pdfs
        )
        for r in results:
            print(r)  # Print the results for each small PDF

    # Process large PDFs one by one (in serial)
    for path in large_pdfs:
        print(f"\n[Phase 2] Processing large PDF: {os.path.basename(path)}")  # Log processing of large PDF
        print(process_pdf(path))  # Process the large PDF


# GUI

class FileManager:
    def __init__(self, master):
        # Initialize the main window and title
        self.master = master
        self.master.title("Parser-Sevenof9")

        # Internal list to track selected PDF files
        self.files = []
        self.last_selected_index = None  # Stores the last clicked index for shift-selection

        # Label for file list
        self.label = tk.Label(master, text="Selected PDF files:")
        self.label.pack(pady=5)

        # Frame to contain the listbox and its scrollbar
        listbox_frame = tk.Frame(master)
        listbox_frame.pack(pady=5)

        # Scrollbar for the listbox
        scrollbar_listbox = tk.Scrollbar(listbox_frame)
        self.listbox = tk.Listbox(
            listbox_frame, selectmode=tk.MULTIPLE, width=80, height=6,
            yscrollcommand=scrollbar_listbox.set
        )
        scrollbar_listbox.config(command=self.listbox.yview)

        # Pack listbox and scrollbar side by side
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_listbox.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection and click events for the listbox
        self.listbox.bind("<<ListboxSelect>>", self.show_text_file)
        self.listbox.bind("<Button-1>", self.on_listbox_click)
        self.listbox.bind("<Shift-Button-1>", self.on_listbox_shift_click)

        # Create a context menu for right-click actions
        self.context_menu = tk.Menu(master, tearoff=0)
        self.context_menu.add_command(label="Remove selected", command=self.remove_file)
        self.listbox.bind("<Button-3>", self.show_context_menu)

        # Frame for action buttons (Add/Remove)
        self.frame = tk.Frame(master)
        self.frame.pack(pady=10)

        # Action buttons
        tk.Button(self.frame, text="Add Folder", command=self.add_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame, text="Select Files", command=self.add_file).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame, text="Remove Selected", command=self.remove_file).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame, text="Remove All", command=self.remove_all).pack(side=tk.LEFT, padx=5)
        tk.Button(master, text="Stop", command=self.stop_parser).pack(pady=5)

        # Placeholder for the parser process (used in threading)
        self.parser_process = None

        # Start button for parsing process
        tk.Button(master, text="Start Parser", command=self.start_parser).pack(pady=10)

        # Text frame to display the contents of the selected .txt file
        text_frame = tk.Frame(master)
        text_frame.pack(padx=10, pady=5)

        scrollbar_text = tk.Scrollbar(text_frame)
        self.text_widget = tk.Text(
            text_frame, height=15, width=100, wrap=tk.WORD,
            yscrollcommand=scrollbar_text.set
        )
        scrollbar_text.config(command=self.text_widget.yview)

        # Pack text viewer and scrollbar
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_text.pack(side=tk.RIGHT, fill=tk.Y)

        # Label for progress section
        tk.Label(master, text="Progress:").pack()

        # Frame for progress output
        progress_frame = tk.Frame(master)
        progress_frame.pack(padx=10, pady=5)

        scrollbar_progress = tk.Scrollbar(progress_frame)
        self.progress_text = tk.Text(
            progress_frame, height=8, width=100, state=tk.DISABLED,
            yscrollcommand=scrollbar_progress.set
        )
        scrollbar_progress.config(command=self.progress_text.yview)

        self.progress_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_progress.pack(side=tk.RIGHT, fill=tk.Y)

    def on_listbox_click(self, event):
        # Handle single left-click selection; clear previous selection
        index = self.listbox.nearest(event.y)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.last_selected_index = index
        self.show_text_file(None)
        return "break"  # Prevent default event propagation

    def on_listbox_shift_click(self, event):
        # Handle shift-click for range selection
        index = self.listbox.nearest(event.y)
        if self.last_selected_index is None:
            self.last_selected_index = index
        start, end = sorted((self.last_selected_index, index))
        self.listbox.selection_clear(0, tk.END)
        for i in range(start, end + 1):
            self.listbox.selection_set(i)
        return "break"

    def show_context_menu(self, event):
        # Show right-click context menu if any item is selected
        if self.listbox.curselection():
            self.context_menu.tk_popup(event.x_root, event.y_root)

    def add_folder(self):
        # Add all PDFs from a selected folder
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
        # Add selected individual PDF files
        paths = filedialog.askopenfilenames(title="Select PDF Files", filetypes=[("PDF Files", "*.pdf")])
        for path in paths:
            if path not in self.files:
                self.files.append(path)
                self.listbox.insert(tk.END, path)

    def remove_file(self):
        # Remove selected files from list and internal storage
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Notice", "Please select an entry to remove.")
            return
        for index in reversed(selection):  # Reverse to avoid index shifting
            self.listbox.delete(index)
            del self.files[index]
        self.text_widget.delete(1.0, tk.END)

    def remove_all(self):
        # Remove all files from the list
        self.listbox.delete(0, tk.END)
        self.files.clear()
        self.text_widget.delete(1.0, tk.END)

    def start_parser(self):
        # Validate input and launch parser in separate thread
        if not self.files:
            messagebox.showinfo("No Files", "Please select at least one file.")
            return
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.insert(tk.END, "Starting parser...\n")
        self.progress_text.config(state=tk.DISABLED)

        # Launch parsing in background to avoid UI freeze
        thread = threading.Thread(target=self.run_parser)
        thread.start()

    def stop_parser(self):
        # Terminate running parser process if active
        if self.parser_process and self.parser_process.poll() is None:
            self.parser_process.terminate()
            self.append_progress_text("Parser process was stopped.\n")
        else:
            self.append_progress_text("No active parser process to stop.\n")

    def run_parser(self):
        # Internal method to run the external parser script
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
        # Thread-safe method to append text to the progress view
        self.progress_text.after(0, lambda: self._insert_text(text))

    def _insert_text(self, text):
        # Append text and scroll to bottom
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.insert(tk.END, text)
        self.progress_text.see(tk.END)
        self.progress_text.config(state=tk.DISABLED)

    def show_messagebox_threadsafe(self, title, message):
        # Display a messagebox from a background thread
        self.master.after(0, lambda: messagebox.showinfo(title, message))

    def show_text_file(self, event):
        # Load and show the content of the corresponding .txt file (if available)
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


# MAIN

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows compatibility with multiprocessing

    if len(sys.argv) > 1:
        # If called with file arguments, execute parsing logic (e.g., from subprocess)
        process_pdfs_main()
    else:
        # Otherwise, launch the GUI application
        root = tk.Tk()
        app = FileManager(root)
        root.mainloop()
