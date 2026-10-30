"""Count manuscript prose under the author's explicit exclusion rules.

Include headings, in-text citations and cross-references. Exclude entire figures
and tables, including captions/sources/notes, all math, footnotes, indices,
bibliography and appendices. Normalize source whitespace to single spaces.
The guide (p. 4) explicitly excludes footnotes, indices, bibliography, images and
appendices; table/caption/math exclusions are the author's instructions.
This internal check supports the thesis's LaTeX subset, not arbitrary TeX.
"""

from pathlib import Path
import re
import unicodedata

THESIS_DIR = Path(__file__).resolve().parents[2] / "output/thesis"
CHARACTER_LIMIT = 100_000
EXCLUDED_ENVIRONMENTS = (
    "figure", "table", "tabular", "tabularx", "longtable", "equation", "align",
    "alignat", "gather", "multline", "displaymath", "math", "thebibliography",
    "titlepage", "comment",
)
HEADINGS = {"chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph"}
FORMATTING = {"textbf", "textit", "emph", "textsc", "texttt", "textrm", "textsf", "textnormal", "mbox"}


def read_group(text, position, opening="{", closing="}"):
    """Read a balanced argument, preserving nested braces and escaped symbols."""
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != opening:
        raise ValueError(f"Expected {opening!r} near {text[position:position + 40]!r}")
    start, depth = position + 1, 1
    position += 1
    while position < len(text):
        if text[position] == "\\":
            position += 2
            continue
        if text[position] == opening:
            depth += 1
        elif text[position] == closing:
            depth -= 1
            if depth == 0:
                return text[start:position], position + 1
        position += 1
    raise ValueError("Unclosed LaTeX argument")


def strip_comments(text):
    # An escaped percent is visible text; a percent after \\ starts a comment.
    return re.sub(r"\\(?:[A-Za-z]+|[^\n])|%[^\n]*(?:\n|$)",
                  lambda m: "" if m[0].startswith("%") else m[0], text)


def clean_latex(text, citations=None, references=None):
    """Return countable text; unsupported commands/references fail explicitly."""
    citations = {} if citations is None else citations
    references = {} if references is None else references
    text = re.split(r"\\appendix\b", strip_comments(text), maxsplit=1)[0]
    environments = "|".join(EXCLUDED_ENVIRONMENTS)
    text = re.sub(r"\\begin\{(" + environments + r")(\*?)\}.*?\\end\{\1\2\}",
                  " ", text, flags=re.DOTALL)
    # Escaped dollars are preserved. Display delimiters precede inline dollars.
    text = re.sub(r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$|\\\[.*?\\\]|\\\(.*?\\\)",
                  "", text, flags=re.DOTALL)
    output, position = [], 0
    while position < len(text):
        char = text[position]
        if char == "{":
            content, position = read_group(text, position)
            output.append(clean_latex(content, citations, references))
            continue
        if char != "\\":
            if char in "}$":
                raise ValueError(f"Unmatched LaTeX delimiter: {char}")
            output.append(" " if char == "~" else char)
            position += 1
            continue
        match = re.match(r"\\([A-Za-z]+\*?|.)", text[position:], re.DOTALL)
        if match is None:
            raise ValueError("Incomplete LaTeX command")
        command = match[1]
        name = command.rstrip("*")
        position += len(match[0])
        if command in {"%", "&", "_", "$", "#", "{", "}"}:
            output.append(command)
        elif command in {"'", '"', "`", "^", "~"}:
            if text[position:position + 1] == "{":
                letter, position = read_group(text, position)
            else:
                letter, position = text[position:position + 1], position + 1
            accent = {"'": "\u0301", '"': "\u0308", "`": "\u0300", "^": "\u0302", "~": "\u0303"}[command]
            output.append(unicodedata.normalize("NFC", letter + accent))
        elif name in HEADINGS | FORMATTING:
            # Short running titles do not contribute to the main prose.
            if text[position:position + 1] == "[":
                _, position = read_group(text, position, "[", "]")
            content, position = read_group(text, position)
            rendered = clean_latex(content, citations, references)
            output.append(f" {rendered} " if name in HEADINGS else rendered)
        elif name in {"footnote", "footnotetext", "label", "index", "bibliography", "bibliographystyle", "color"}:
            if text[position:position + 1] == "[":
                _, position = read_group(text, position, "[", "]")
            _, position = read_group(text, position)
        elif name in {"input", "include"}:
            path, position = read_group(text, position)
            if not path.startswith(("tables/", "figures/", "appendices/")):
                raise ValueError(f"Unsupported prose inclusion: {path}")
            output.append(" ")
        elif name in {"citet", "citep", "citealt", "citealp", "cite"}:
            prenote, postnote = "", ""
            if text[position:position + 1] == "[":
                opt1, position = read_group(text, position, "[", "]")
                if text[position:position + 1] == "[":
                    opt2, position = read_group(text, position, "[", "]")
                    prenote, postnote = opt1.strip() + " ", ", " + opt2.strip()
                else:
                    postnote = ", " + opt1.strip()
            keys, position = read_group(text, position)
            labels = []
            for key in keys.split(","):
                key = key.strip()
                if key not in citations:
                    raise ValueError(f"Missing citation: {key}")
                author, year = citations[key]
                if name == "citet":
                    labels.append(f"{author} ({year})")
                else:
                    labels.append(f"{author}{' ' if name == 'citealt' else ', '}{year}")
            rendered = f"{prenote}{'; '.join(labels)}{postnote}".strip()
            output.append(f"({rendered})" if name in {"citep", "cite"} else rendered)
        elif name in {"ref", "eqref"}:
            key, position = read_group(text, position)
            if key not in references:
                raise ValueError(f"Missing reference: {key}")
            value = references[key]
            output.append(f"({value})" if name == "eqref" else value)
        elif name in {"begin", "end"}:
            environment, position = read_group(text, position)
            if environment not in {"itemize", "enumerate", "description", "document"}:
                raise ValueError(f"Unsupported or unclosed environment: {environment}")
            output.append(" ")
        elif command in {"item", "\\", "quad", "qquad", ",", ";", ":", " ", "newline", "par",
                         "tableofcontents", "listoffigures", "listoftables", "clearpage", "newpage", "noindent"}:
            output.append(" ")
        else:
            raise ValueError(f"Unsupported LaTeX command: {command}")
    rendered = "".join(output).replace("---", "—").replace("--", "–")
    rendered = rendered.replace("``", "“").replace("''", "”")
    return re.sub(r"\s+", " ", rendered).strip()


def load_rendered_references(thesis_dir):
    """Use actual author-year labels and numbers from existing compiled files."""
    bibliography = (thesis_dir / "main.bbl").read_text(encoding="utf-8")
    citations = {}
    for match in re.finditer(r"\\bibitem\[", bibliography):
        label, position = read_group(bibliography, match.end() - 1, "[", "]")
        key, _ = read_group(bibliography, position)
        # Handle natbib format "Author(Year)..." or standard "Author, Year"
        if "(" in label and label.endswith(")"):
            author_part, year_part = label[:-1].split("(", 1)
            citations[key] = (clean_latex(author_part), clean_latex(year_part))
        elif "," in label:
            author, year = label.rsplit(",", 1)
            citations[key] = (clean_latex(author), clean_latex(year))
        else:
            citations[key] = (clean_latex(label), "")
    references = {}
    for path in [thesis_dir / "main.aux", *sorted((thesis_dir / "chapters").glob("*.aux"))]:
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\newlabel\{", source):
            key, position = read_group(source, match.end() - 1)
            fields, _ = read_group(source, position)
            number, _ = read_group(fields, 0)
            references[key] = number
    return citations, references


def main():
    citations, references = load_rendered_references(THESIS_DIR)
    source = strip_comments((THESIS_DIR / "main.tex").read_text(encoding="utf-8"))
    source = re.split(r"\\appendix\b", source, maxsplit=1)[0]
    paths = [THESIS_DIR / (name + ".tex") for name in re.findall(r"\\include\{(chapters/[^}]+)\}", source)]
    if not paths:
        raise ValueError("No manuscript chapters found in main.tex")
    print("Character count including spaces; source whitespace normalized to one space.")
    print("Included: chapter/section titles, prose, in-text citations and cross-references.")
    print("Excluded: complete figures/tables, captions, sources, notes, math, footnotes,")
    print("          indices, bibliography, appendices and title page.")
    print("Citations and reference numbers use existing main.bbl and chapter .aux files.")
    if any(p.stat().st_mtime > p.with_suffix(".aux").stat().st_mtime for p in paths):
        print("NOTE: chapter sources are newer than .aux files. Recompile if reference numbering changed.")
    if (THESIS_DIR / "references.bib").stat().st_mtime > (THESIS_DIR / "main.bbl").stat().st_mtime:
        raise ValueError("Bibliography is newer than main.bbl: recompile before counting citations")
    print(f"\n{'Chapter':<42} {'Characters':>12} {'Words':>10}")
    total = 0
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        prose = clean_latex(raw, citations, references)
        total += len(prose)
        title = re.search(r"\\chapter\{([^}]+)\}", raw)[1]
        print(f"{title:<42} {len(prose):>12,} {len(prose.split()):>10,}")
    print(f"{'TOTAL':<42} {total:>12,}")
    print(f"\nLimit: {CHARACTER_LIMIT:,} characters. No tolerance applied.")
    if total <= CHARACTER_LIMIT:
        print(f"WITHIN LIMIT: {CHARACTER_LIMIT - total:,} characters remaining.")
    else:
        print(f"OVER LIMIT: {total - CHARACTER_LIMIT:,} characters to remove.")


if __name__ == "__main__":
    main()
