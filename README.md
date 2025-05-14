# parsing
only for advanced users

# <b>PDF to TXT converter ready to chunck for your RAG</b>
<b>ONLY WINDOWS</b><br>
<b>EXE and PY available (en and german)</b><br>

<b>&#x21e8;</b> give me a ❤️, if you like  ;)<br><br>

Most LLM applications only convert your PDF simple to txt, nothing more, its like you save your PDF as txt file. Often textblocks are mixed and tables not readable.
Therefore its better to convert it with some help of a <b>parser</b>.<br>
I work with "<b>pdfplumber/pdfminer</b>" none OCR, so its very fast!<br>
<ul style="line-height: 1.05;">
<li>Works with single and multi pdf list, works with folder</li>
<li>Intelligent multiprocessing</li>
<li>Error tolerant, that means if your PDF is not convertible, it will be skipped, no special handling</li>
<li>Instant view of the result, hit one pdf on top of the list</li>
<li>Converts some common tables as json inside the txt file</li>
<li>It adds the absolute PAGE number to each page</li>
<li>All txt files will be created in original folder of PDF</li>
<li>All txt files will be overwritten</li>
</ul>
exe files aviable here: <br>
https://huggingface.co/kalle07/pdf2txt_parser_converter
<br>
This I have created with my brain and the help of chatGPT, Iam not a coder... sorry so I will not fulfill any wishes unless there are real errors.<br>
It is really hard for me with GUI and the Function and in addition to compile it.<br>
For the python-file oc you need to import missing libraries.<br>
<br>
I also have a "<b>docling</b>" parser with OCR (GPU is need for fast processing), its only be a python-file, not compiled.<br>
You have to download all libs, and if you start (first time) internal also OCR models are downloaded. At the moment i have prepared a kind of multi docling, 
the number of parallel processes depend on VRAM and if you use OCR only for tables or for all. I have set VRAM = 16GB (my GPU RAM, you should set yours) and the multiple calls for docling are VRAM/1.3, 
so it uses ~12GB (in my version) and processes 12 PDFs at once, only txt and tables are converted, so no images no diagrams. For now all PDFs must be same folder like the python file. 
If you change OCR for all the VRAM consum is rasing you hace to set 1.3 to 2 or more.
<br><br>

<b>now have fun and leave a comment if you like  ;)</b><br>
on discord "sevenof9"
<br>
my embedder collection:<br>
https://huggingface.co/kalle07/embedder_collection

<br>
<br>
I am not responsible for any errors or crashes on your system. If you use it, you take full responsibility!
