import json
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Focus on the main evaluate_dream_new.ipynb - get all result cells
notebook_path = os.path.join('notebooks', 'dream', 'evaluate_dream_new.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"NOTEBOOK: {os.path.basename(notebook_path)}")
print(f"Total cells: {len(nb.get('cells', []))}")

cells = nb.get('cells', [])
for i, cell in enumerate(cells):
    cell_type = cell.get('cell_type', 'unknown')
    source = ''.join(cell.get('source', []))
    outputs = cell.get('outputs', [])
    
    if cell_type != 'code' or not outputs:
        continue
    
    # Collect all text output
    all_text = []
    has_image = False
    for out in outputs:
        otype = out.get('output_type', '')
        if otype == 'stream':
            text = ''.join(out.get('text', []))
            all_text.append(text)
        elif otype == 'execute_result':
            data = out.get('data', {})
            if 'text/plain' in data:
                all_text.append(''.join(data['text/plain']))
            if 'text/html' in data:
                # Check for dataframe tables
                html = ''.join(data['text/html'])
                if '<table' in html.lower():
                    all_text.append('[HTML TABLE - DataFrame output]')
        elif otype == 'display_data':
            data = out.get('data', {})
            if 'text/plain' in data:
                all_text.append(''.join(data['text/plain']))
            if 'image/png' in data:
                has_image = True
        elif otype == 'error':
            all_text.append(f"[ERROR] {out.get('ename','')}: {out.get('evalue','')}")
    
    combined = '\n'.join(all_text)
    
    # Filter: only show cells with meaningful results
    keywords = ['accuracy', 'auroc', 'AUROC', 'Accuracy', 'RESULTS', 'Classification Report',
                'precision', 'recall', 'f1-score', 'Shape', 'COMPLETE', 'SPLIT', 
                'FINAL', 'SUCCESS', 'BREAKDOWN', 'Verified', 'False Negative',
                'FILTERING', 'Discarded', 'Total Time', 'Data Shape', 'DETECTOR']
    
    is_result = any(k in combined for k in keywords)
    
    if not is_result:
        continue
    
    source_lines = source.split('\n')
    header = source_lines[1].strip().strip('#').strip() if len(source_lines) > 1 else source_lines[0][:80]
    
    print(f"\n{'='*60}")
    print(f"Cell {i}: {header}")
    print(f"{'='*60}")
    
    # Filter out noise
    clean_lines = []
    for line in combined.split('\n'):
        if any(skip in line for skip in ['%|', 'B/s]', 'it/s]', 'WARNING:', 'huggingface_hub',
                                           'config.json', 'model.safetensors', 'tokenizer',
                                           'vocab.txt', 'modules.json', 'sentence_bert',
                                           'special_tokens', 'README.md', 'Loading weights',
                                           '0%|', 'pip install', 'pip uninstall', '/kaggle/',
                                           'Downloading', 'Generating']):
            continue
        clean_lines.append(line)
    
    result = '\n'.join(clean_lines).strip()
    if len(result) > 3000:
        result = result[:1500] + '\n...[TRUNCATED]...\n' + result[-1500:]
    
    print(result)
    if has_image:
        print("[Contains matplotlib figure]")

# Also check the evalaute_dream_new_models.ipynb for more model comparisons
print("\n\n" + "="*80)
print("NOW CHECKING: evalaute_dream_new_models.ipynb")
print("="*80)

notebook_path2 = os.path.join('notebooks', 'dream', 'evalaute_dream_new_models.ipynb')
with open(notebook_path2, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

cells2 = nb2.get('cells', [])
for i, cell in enumerate(cells2):
    cell_type = cell.get('cell_type', 'unknown')
    source = ''.join(cell.get('source', []))
    outputs = cell.get('outputs', [])
    
    if cell_type != 'code' or not outputs:
        continue
    
    all_text = []
    has_image = False
    for out in outputs:
        otype = out.get('output_type', '')
        if otype == 'stream':
            all_text.append(''.join(out.get('text', [])))
        elif otype == 'execute_result':
            data = out.get('data', {})
            if 'text/plain' in data:
                all_text.append(''.join(data['text/plain']))
        elif otype == 'display_data':
            data = out.get('data', {})
            if 'text/plain' in data:
                all_text.append(''.join(data['text/plain']))
            if 'image/png' in data:
                has_image = True
    
    combined = '\n'.join(all_text)
    
    keywords = ['accuracy', 'auroc', 'AUROC', 'Accuracy', 'RESULTS', 'FINAL',
                'ABLATION', 'Seeds', 'BREAKDOWN', 'TRACEDET', 'TraceDet']
    
    if not any(k in combined for k in keywords):
        continue
    
    source_lines = source.split('\n')
    header = source_lines[1].strip().strip('#').strip() if len(source_lines) > 1 else source_lines[0][:80]
    
    print(f"\n{'='*60}")
    print(f"Cell {i}: {header}")
    print(f"{'='*60}")
    
    clean_lines = []
    for line in combined.split('\n'):
        if any(skip in line for skip in ['%|', 'B/s]', 'it/s]', 'WARNING:', 'huggingface_hub',
                                           'config.json', 'model.safetensors', '0%|']):
            continue
        clean_lines.append(line)
    
    result = '\n'.join(clean_lines).strip()
    if len(result) > 3000:
        result = result[:1500] + '\n...[TRUNCATED]...\n' + result[-1500:]
    
    print(result)
    if has_image:
        print("[Contains matplotlib figure]")
