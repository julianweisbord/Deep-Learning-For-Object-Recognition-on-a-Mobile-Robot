filename = ProblemStatement

all: pdf 

pdf: ps
	ps2pdf ${filename}.ps

ps: dvi
	dvips -t letter ${filename}.dvi

dvi:
	latex ${filename}
	bibtex ${filename}||true
	latex ${filename}
	latex ${filename}

clean:
	rm -f *.ps *.dvi *.out *.log *.aux *.bbl *.blg *.pyg *.o *.out *.exe
