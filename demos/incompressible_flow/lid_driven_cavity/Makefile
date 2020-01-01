all: compile clean

compile:
	pdflatex -shell-escape test_radau.tex
	#biber test_radau
	pdflatex -shell-escape test_radau.tex

clean:
	rm -f test_radau.nav  test_radau.snm test_radau.aux  test_radau.out
	rm -f test_radau.toc test_radau.log test_radau.vrb test_radau.bcf 
	rm -f test_radau.run.xml test_radau.blg test_radau.bbl
	rm -rf _minted-test_radau/
