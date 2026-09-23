import re
import os

def clean_latex(text):
    # Remove comments
    text = re.sub(r'%.*', '', text)
    # Remove figures
    text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', text, flags=re.DOTALL)
    # Remove tables
    text = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\input\{tables/.*?\}', '', text)
    # Remove display math
    text = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{align\}.*?\\end\{align\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    # Remove inline math
    text = re.sub(r'\$.*?\$', '', text)
    # Remove footnotes
    text = re.sub(r'\\footnote\{.*?\}', '', text, flags=re.DOTALL)
    # Remove citations and refs
    text = re.sub(r'\\cite[pt]?\{.*?\}', '', text)
    text = re.sub(r'\\ref\{.*?\}', '', text)
    text = re.sub(r'\\label\{.*?\}', '', text)
    # Remove common formatting commands but keep inner text
    text = re.sub(r'\\(chapter|section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}', r'\2', text)
    text = re.sub(r'\\(textbf|textit|emph|textsc|texttt)\{([^}]*)\}', r'\2', text)
    # Remove remaining backslash commands
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    # Remove curly braces left over
    text = re.sub(r'[{}]', '', text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

budget = {
    'Chap 1 (Introduction)': {'path': 'output/thesis/chapters/01_introduction.tex', 'min': 6000, 'max': 10000},
    'Chap 2 (Literature Review)': {'path': 'output/thesis/chapters/02_literature_review.tex', 'min': 20000, 'max': 30000},
    'Chap 3 (Data & Study Area)': {'path': 'output/thesis/chapters/03_data_and_study_area.tex', 'min': 8000, 'max': 16000},
}

all_budget = {
    'Chap 1 (Introduction)': (6000, 10000),
    'Chap 2 (Literature Review)': (20000, 30000),
    'Chap 3 (Data & Study Area)': (8000, 16000),
    'Chap 4 (Methodology)': (12000, 25000),
    'Chap 5 (Empirical Results)': (14000, 30000),
    'Chap 6 (Discussion)': (7000, 13000),
    'Chap 7 (Conclusions)': (3000, 6000),
}

print(f"{'Capitolo':<28} | {'Caratteri (con spazi)':<22} | {'Parole':<8} | {'Budget Min':<10} | {'Budget Max':<10} | {'Stato vs Budget'}")
print("-" * 105)

total_chars = 0
total_words = 0

for name, info in budget.items():
    with open(info['path'], 'r', encoding='utf-8') as f:
        raw = f.read()
    clean = clean_latex(raw)
    chars = len(clean)
    words = len(clean.split())
    total_chars += chars
    total_words += words
    status = ""
    if chars < info['min']:
        status = f"Sotto min ({chars - info['min']:+d})"
    elif chars > info['max']:
        status = f"SOPRA MAX ({chars - info['max']:+d})"
    else:
        status = "Nel target [OK]"
    print(f"{name:<28} | {chars:<22} | {words:<8} | {info['min']:<10} | {info['max']:<10} | {status}")

print("-" * 105)
print(f"{'TOTALE ATTUALE (Cap 1-3)':<28} | {total_chars:<22} | {total_words:<8} | {'34.000':<10} | {'56.000':<10} | {'Nel target parziale'}")

print("\n" + "=" * 90)
print("DETTAGLIO SEZIONI CAPITOLO 3 (Data and Study Area)")
print("=" * 90)

with open('output/thesis/chapters/03_data_and_study_area.tex', 'r', encoding='utf-8') as f:
    c3_raw = f.read()

parts = re.split(r'(\\section\{[^}]+\})', c3_raw)
intro_clean = clean_latex(parts[0])
print(f"{'Intro capitolo':<55} | {len(intro_clean):>8} car. | {len(intro_clean.split()):>6} parole")

for i in range(1, len(parts), 2):
    sec_match = re.search(r'\\section\{([^}]+)\}', parts[i])
    sec_title = sec_match.group(1) if sec_match else f"Sezione {i}"
    sec_content = clean_latex(parts[i] + parts[i+1])
    print(f"\n[SEC] {sec_title:<50} | {len(sec_content):>8} car. | {len(sec_content.split()):>6} parole")
    
    subparts = re.split(r'(\\subsection\{[^}]+\})', parts[i+1])
    if len(subparts) > 1:
        for j in range(1, len(subparts), 2):
            sub_match = re.search(r'\\subsection\{([^}]+)\}', subparts[j])
            sub_title = "  |-- " + (sub_match.group(1) if sub_match else f"Sub {j}")
            sub_content = clean_latex(subparts[j] + subparts[j+1])
            print(f"{sub_title:<55} | {len(sub_content):>8} car. | {len(sub_content.split()):>6} parole")

print("\n" + "=" * 90)
print("QUADRO COMPLESSIVO RISPETTO AL VINCOLO ATENEO (MAX 100.000 CARATTERI)")
print("=" * 90)
print(f"Vincolo Ateneo: max 100.000 caratteri (spazi inclusi, escluse tabelle/figure/formule/note/bibliografia)")
print(f"Caratteri attuali scritti (Cap 1-3): {total_chars:,} caratteri ({total_chars/100000*100:.1f}% del limite di 100.000)")
print(f"Spazio residuo per Cap 4-7: {100000 - total_chars:,} caratteri")
