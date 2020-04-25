(TeX-add-style-hook
 "ms"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("physics" "arrowdel") ("subfig" "caption=false") ("cleveref" "capitalise")))
   (TeX-run-style-hooks
    "latex2e"
    "abstract"
    "chapter/intro"
    "chapter/lit_review"
    "chapter/methods"
    "chapter/results"
    "chapter/conclusion"
    "chapter/acknowledgments"
    "chapter/figures"
    "ws-ijbc"
    "ws-ijbc10"
    "physics"
    "lmodern"
    "siunitx"
    "amsmath"
    "amssymb"
    "textcomp"
    "graphicx"
    "subfig"
    "tikz"
    "mhchem"
    "tipa"
    "hyperref"
    "cleveref")
   (TeX-add-symbols
    '("mean" 1)
    "crefpairconjunction"
    "etal"
    "hrx"
    "hry"
    "hrz"
    "hra"
    "hrb"
    "chimera"
    "meta"
    "ordparam"
    "phase")
   (LaTeX-add-labels
    "sec:intro"
    "sec:lit_review"
    "sec:methods"
    "sec:results"
    "sec:conclusion"
    "sec:acknowledgements"
    "sec:figures")
   (LaTeX-add-bibliographies
    "ms.bib"))
 :latex)

