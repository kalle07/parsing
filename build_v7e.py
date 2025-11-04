import sys
import subprocess

# Hier wird der Pfad zum Entry-Point (dein Skript) definiert
entry_point = "PDF Parser - Sevenof9_v7e.py"  # Passe dies nach Bedarf an


# Der Befehl, der an PyInstaller übergeben wird
cmd = [
    sys.executable,
    "-m", "PyInstaller",
    "--onefile",
    "--noconfirm",
    "--clean",
    "--noconsole",  # Keine Konsole anzeigen (wichtig für GUI-Programme)
    
    # External dependencies that need explicit hidden imports
    "--hidden-import", "pdfminer.six",
    "--hidden-import", "joblib",
    "--hidden-import", "joblib.externals.loky.backend.resource_tracker",
    "--hidden-import", "pdfplumber.utils.exceptions",
    "--hidden-import", "pdfminer.layout",
    "--hidden-import", "pdfminer.pdfpage",
    "--hidden-import", "psutil",
    "--hidden-import", "multiprocessing",
    "--hidden-import", "rtree",
    "--hidden-import", "numpy",
    "--hidden-import", "concurrent.futures",
    "--hidden-import", "wx",  # This is the correct import for wxPython
    
    entry_point
]

# Der Befehl wird ausgeführt
try:
    subprocess.run(cmd, check=True)
    print("Kompilierung abgeschlossen.")
except subprocess.CalledProcessError as e:
    print(f"Fehler bei der Kompilierung: {e}")
