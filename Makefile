TARGET = paper-corekick
TARGET2 = dag

# needed for latexmk to handle the dependencies
.PHONY: $(TARGET).pdf


all: $(TARGET).pdf

$(TARGET).pdf : $(TARGET).tex
	latexmk -pdf -pdflatex="pdflatex -interaction=errorstopmode -file-line-error" $(TARGET).tex

figures/$(TARGET2).pdf : diagrams/$(TARGET2).tex tikzlibrarybayesnet.code.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=errorstopmode -file-line-error" diagrams/$(TARGET2).tex
	mv $(TARGET2).pdf figures/
	mv $(TARGET2).* diagrams/


#delete all temporary files and the output .pdf file
clean :
	latexmk -C
