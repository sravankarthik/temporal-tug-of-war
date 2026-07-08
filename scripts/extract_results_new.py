import json
import sys

# Force UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

for nb_path in ['notebooks/llada/evalaute_llada_new.ipynb', 'notebooks/dream/evaluate_dream_new.ipynb']:
    print(f"=== {nb_path} ===")
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for c in nb['cells']:
            if c.get('cell_type') == 'code':
                for out in c.get('outputs', []):
                    if out.get('output_type') == 'stream':
                        print("".join(out.get('text', [])))
                    elif out.get('output_type') == 'execute_result':
                        print("".join(out.get('data', {}).get('text/plain', [])))
                    elif out.get('output_type') == 'display_data':
                        print("".join(out.get('data', {}).get('text/plain', [])))
    except Exception as e:
        print(f"Error reading {nb_path}: {e}")
