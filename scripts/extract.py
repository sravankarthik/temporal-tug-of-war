import json
import sys

def extract_code(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
    source = []
    for cell in code_cells:
        source.append(''.join(cell.get('source', [])))
    return '\n\n'.join(source)

with open('notebooks_code.py', 'w', encoding='utf-8') as out:
    for f in ['evalaute.ipynb', 'evaluate_dream.ipynb']:
        out.write(f'# --- {f} ---\n')
        out.write(extract_code(f))
        out.write('\n\n')
