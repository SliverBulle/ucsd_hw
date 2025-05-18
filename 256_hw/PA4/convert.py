import nbformat
from nbconvert import HTMLExporter, PDFExporter

# 读取notebook
with open("./PA4_CSE256_FA24.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

# pdf
pdf_exporter = PDFExporter()
pdf_data, resources = pdf_exporter.from_notebook_node(nb)

# 保存PDF
with open("PA4_CSE256_FA24.pdf", "w") as f:
    f.write(pdf_data)
