import nbformat
from nbconvert import PythonExporter

# 读取 .ipynb 文件
with open('Midterm_stub.ipynb') as f:
    nb_content = nbformat.read(f, as_version=4)

# 创建 Python 导出器
exporter = PythonExporter()
body, resources = exporter.from_notebook_node(nb_content)

# 保存为 .py 文件
with open('midterm.py', 'w') as f:
    f.write(body)
