import nbformat
from nbconvert import PythonExporter

# 读取 notebook
with open('homework1_stub.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

# 转换为 Python
exporter = PythonExporter()
python_code, _ = exporter.from_notebook_node(nb)

# 保存为 .py 文件
with open('homework1.py', 'w') as f:
    f.write(python_code)