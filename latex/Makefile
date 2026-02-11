# Makefile for NeurIPS paper compilation

LATEX = /Library/TeX/texbin/pdflatex
BIBTEX = /Library/TeX/texbin/bibtex
MAIN = main
PDF = $(MAIN).pdf

.PHONY: all clean view

all: $(PDF)

$(PDF): $(MAIN).tex references.bib neurips_2025.sty
	@echo "Compiling LaTeX (first pass)..."
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	@echo "Running BibTeX..."
	$(BIBTEX) $(MAIN)
	@echo "Compiling LaTeX (second pass)..."
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	@echo "Compiling LaTeX (third pass)..."
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	@echo "Done! Generated $(PDF)"

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf

view: $(PDF)
	open $(PDF)

check:
	@echo "Checking for overfull/underfull boxes..."
	@grep -i "overfull\|underfull" $(MAIN).log || echo "No warnings found"

quick:
	@echo "Quick compile (single pass)..."
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
