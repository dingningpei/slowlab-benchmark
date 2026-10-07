# Figure 2 (agent interface) source

`crop.tex` builds `../../fig2_framework.pdf` from `figpre.tex` (colours, styles) and `fig.tex` (the TikZ picture).
It needs `icons/*.png`: Freepik flat icons from flaticon.com (free licence, attribution required), listed with
their icon pages in `output/icons/flaticon/CREDITS.json`; they are not redistributed in this repository.
Build: copy the icons to `icons/` here, then `pdflatex crop.tex` and copy `crop.pdf` to `../../fig2_framework.pdf`.
