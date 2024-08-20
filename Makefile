.PHONY: html nb latex help import test nbclean

html:
	sphinx-build -Ea . _build/html -b html

nb:
	@cd demo; for i in `ls *.py`; do bash ../bin/py2html.sh $$i; done; cd ..
	
latex:
	sphinx-build . _build/latex -b latex
	
import:	
	ghp-import -n -o -p -f _build/html

test:
	@cd demo; for i in `ls *.py`; do bash ../bin/test_py.sh $$i; done; echo "Tests finished."

nbclean:
	@cd demo; rm ../_static/*.py.html; for i in `ls *.py`; do bash ../bin/py2html.sh $$i; done; cd ..
			
help:
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@echo "  html    - Build HTML documentation"
	@echo "  nb      - Convert updated Python notebooks to HTML"
	@echo "  nbclean - Convert all Python notebooks to HTML (deletes old HTML files first)"
	@echo "  nbhtml  - Both nb and html and upload by ghp-import"
	@echo "  latex   - Build LaTeX documentation"
	@echo "  import  - Upload using ghp-import to github pages"
	@echo "  test    - Run all py files in demo directory and report possible problems"
	@echo "  help    - Show this help message"
