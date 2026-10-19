"""Regression tests for the internal manuscript character counter."""

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "thesis_notes/count_thesis_characters.py"
spec = importlib.util.spec_from_file_location("character_count", SCRIPT)
counter = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(counter)


class CharacterCountTests(unittest.TestCase):
    def test_percentage_does_not_remove_prose(self):
        self.assertEqual(counter.clean_latex(r"Gain 10\% remains. % comment"), "Gain 10% remains.")

    def test_comments_do_not_insert_spaces(self):
        self.assertEqual(counter.clean_latex("fore% comment\ncast"), "forecast")

    def test_comment_after_line_break_command(self):
        self.assertEqual(counter.clean_latex("A\\\\% comment\nB"), "A B")

    def test_figure_including_caption_source_and_notes(self):
        source = r"A \begin{figure*}[ht]\caption{Nested \textbf{caption}} Source: X. Notes: Y.\end{figure*} B"
        self.assertEqual(counter.clean_latex(source), "A B")

    def test_table_including_content_caption_and_source(self):
        source = r"A \begin{table}\caption{Caption}\begin{tabular}{ll}Cell & 10\%\end{tabular} Source: X.\end{table} B"
        self.assertEqual(counter.clean_latex(source), "A B")

    def test_starred_table_and_standalone_longtable(self):
        for environment in ["table*", "longtable"]:
            with self.subTest(environment=environment):
                source = "A \\begin{" + environment + "}Excluded \\end{" + environment + "} B"
                self.assertEqual(counter.clean_latex(source), "A B")

    def test_external_table_and_figure_inputs(self):
        self.assertEqual(counter.clean_latex(r"A \input{tables/test} \input{figures/test} B"), "A B")

    def test_nested_footnote(self):
        self.assertEqual(counter.clean_latex(r"A\footnote{Note with \emph{nested text} and $x$.} B"), "A B")

    def test_math_delimiters(self):
        for math in [r"$x$", r"$$x$$", r"\(x\)", r"\[x\]", r"\begin{align*}x&=1\end{align*}"]:
            with self.subTest(math=math):
                self.assertEqual(counter.clean_latex("A " + math + " B"), "A B")

    def test_formatting_titles_and_visible_symbols(self):
        source = r"\section{A \emph{nested} heading} Cost \& risk: 10\%, \$5, a\_b."
        self.assertEqual(counter.clean_latex(source), "A nested heading Cost & risk: 10%, $5, a_b.")

    def test_formatting_does_not_split_words(self):
        self.assertEqual(counter.clean_latex(r"pre\emph{season}al"), "preseasonal")

    def test_revision_color_does_not_change_character_count(self):
        self.assertEqual(counter.clean_latex(r"{\color{red} Revised prose.}"), "Revised prose.")

    def test_whitespace_and_typeset_punctuation(self):
        self.assertEqual(counter.clean_latex("A~B\n\n C---D  E--F"), "A B C—D E–F")

    def test_appendix_is_excluded(self):
        self.assertEqual(counter.clean_latex(r"Body. \appendix \chapter{Extra} Appendix text."), "Body.")

    def test_bibliography_and_indices_are_excluded(self):
        source = r"Body. \tableofcontents \listoffigures \listoftables \begin{thebibliography}{9}Entry\end{thebibliography}"
        self.assertEqual(counter.clean_latex(source), "Body.")

    def test_citations_and_cross_references_are_counted(self):
        text = r"\citet{Paper} agrees \citep{Paper}; Figure~\ref{fig:test}, Eq.~\eqref{eq:test}."
        result = counter.clean_latex(text, {"Paper": ("Smith et al.", "2020")}, {"fig:test": "3.1", "eq:test": "4.2"})
        self.assertEqual(result, "Smith et al. (2020) agrees (Smith et al., 2020); Figure 3.1, Eq. (4.2).")

    def test_missing_citation_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Missing citation"):
            counter.clean_latex(r"\citep{Missing}")

    def test_missing_reference_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            counter.clean_latex(r"\ref{Missing}")

    def test_unclosed_excluded_environment_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "unclosed environment"):
            counter.clean_latex(r"Body \begin{figure}Unclosed caption")

    def test_unknown_command_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            counter.clean_latex(r"A \unknown{hidden} B")


if __name__ == "__main__":
    unittest.main()
