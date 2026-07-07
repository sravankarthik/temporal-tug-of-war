import json
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def extract_key_results(notebook_path):
    """Extract only key result cells from a notebook - skip progress bars and setup noise."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"NOTEBOOK: {os.path.basename(notebook_path)}")
    print(f"{'='*80}")
    
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
                    'Mean Accuracy', 'TRACEDET', 'TraceDet']
        
        is_result = any(k in combined for k in keywords) or has_image
        
        if not is_result:
            continue
        
        source_lines = source.split('\n')
        # Find comment header
        header = ""
        for line in source_lines[:5]:
            if '#' in line and ('CELL' in line or 'EVALUATION' in line or 'RESULT' in line or 
                              'SPLIT' in line or 'REPORT' in line or 'FINAL' in line or
                              'VISUALIZATION' in line or 'BREAKDOWN' in line or 'DYNAMICS' in line):
                header = line.strip().strip('#').strip()
                break
        
        if not header:
            header = source_lines[0][:80] if source_lines else "Unknown"
        
        print(f"\n--- Cell {i}: {header} ---")
        
        # Filter out progress bars and download noise
        clean_lines = []
        for line in combined.split('\n'):
            # Skip progress bar lines and download noise
            if any(skip in line for skip in ['%|', 'B/s]', 'it/s]', 'WARNING:', 'huggingface_hub',
                                               'config.json', 'model.safetensors', 'tokenizer',
                                               'vocab.txt', 'modules.json', 'sentence_bert',
                                               'special_tokens', 'README.md', 'Loading weights',
                                               '0%|', 'pip install', 'pip uninstall']):
                continue
            clean_lines.append(line)
        
        result = '\n'.join(clean_lines).strip()
        if len(result) > 3000:
            result = result[:1500] + '\n...[TRUNCATED]...\n' + result[-1500:]
        
        print(result)
        if has_image:
            print("[Contains matplotlib figure]")

# Process all dream notebooks
dream_dir = os.path.join('notebooks', 'dream')
for fname in sorted(os.listdir(dream_dir)):
    if fname.endswith('.ipynb'):
        extract_key_results(os.path.join(dream_dir, fname))
