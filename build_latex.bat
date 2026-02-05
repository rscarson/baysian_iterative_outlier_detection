@echo off
set src=latex-src\0_main.tex
set dst=bayesian_iterative_outlier_detection.draft.pdf
set tmp=tmp

rem First LuaLaTeX run
lualatex --halt-on-error --c-style-errors -output-directory=%tmp% %src%
if errorlevel 1 goto :err

rem Run BibTeX on the auxiliary file
bibtex %tmp%\0_main.aux
if errorlevel 1 goto :err

rem Second LuaLaTeX run
lualatex --halt-on-error --c-style-errors -output-directory=%tmp% %src%
if errorlevel 1 goto :err

rem Third LuaLaTeX run (to fix cross-references)
lualatex --halt-on-error --c-style-errors -output-directory=%tmp% %src%
if errorlevel 1 goto :err

rem Move final PDF to destination
move /Y %tmp%\0_main.pdf %dst%
if errorlevel 1 goto :err

echo Build succeeded: %dst%
goto :eof

:err
echo Build failed.

:eof
