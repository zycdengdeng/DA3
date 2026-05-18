# LAD-DA3 — Paper

CVPR-style write-up for the LiDAR-Anchored Depth project. Source lives in
[`main.tex`](main.tex); the bibliography in [`references.bib`](references.bib).
Figure placeholders + the exact commands that produce them are listed in
[`figures/README.md`](figures/README.md) — fill those in and re-build.

## Build

```bash
cd paper/
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or, with `latexmk`:

```bash
latexmk -pdf main.tex
```

The CVPR style file (`cvpr.sty`, `ieee_fullname.bst`, etc.) is **not vendored**
in this repo. Drop the official author kit into this directory before building:

```bash
# From the CVPR author kit (https://cvpr.thecvf.com/Conferences/<year>/AuthorGuidelines)
cp cvpr_template/cvpr.sty paper/
cp cvpr_template/ieee_fullname.bst paper/
```

(If you don't care about the exact CVPR ruler/two-column visual and just want
to read the content, the document falls back to `\documentclass{article}` —
see the `\IfFileExists{cvpr.sty}` block in `main.tex`.)

## Figures

Every `\includegraphics` call in `main.tex` points at a PNG in
[`figures/`](figures/). The figure list in [`figures/README.md`](figures/README.md)
spells out the *exact* `cd …` + `lad …` command that produces each
artefact, and the post-processing (cropping / overlaying / side-by-side) that
turns the raw output into a paper-ready figure. Run the commands, drop the
PNGs into `figures/`, rebuild.

## Status

- Method section: complete.
- Experiments section: structure complete; numbers are placeholders
  (`\todo{…}`) until the formal evaluation lands.
- Related work: covers the main families; add citations as needed.
- 9 figures, 3 tables — all with capture commands documented.
