# ML_Project

## Cerințe și setup
- Folosește `.venv` deja prezent (nu se instalează global): `source .venv/bin/activate`.
- Dependențe Python (în `.venv`): `pip install -r requirements.txt`.
- Date: `data/ap_dataset.csv`.

## Rulări principale
- 2.1 Crazy Sauce (doar bonuri cu Crazy Schnitzel):  
  `python main.py --experiment crazy_sauce`
- 2.2 Per-sauce + recomandare Top-K:  
  `python main.py --experiment all_sauces --top-k 3`
- Ranking upsell (coș parțial, Hit@K vs popularitate/revenit):  
  `python main.py --experiment ranking --ranking-topk 1,3,5 --ranking-min-occ 20`

Opțiuni utile:
- `--split-mode temporal|random`, `--test-size 0.2`
- `--binary-counts` (0/1 în loc de număr de apariții)
- `--lr --l2 --max-iter --tol` (pentru logreg GD)
- `--ranking-alpha` (smoothing Naive Bayes), `--random-state`

## Structură
- `crazy_sauce/` – data prep, modele custom, evaluări
- `main.py` – CLI pentru experimentele 2.1, 2.2, ranking
- `docs/report.tex` – raport LaTeX (vezi mai jos)

## Raport
Compilează cu `pdflatex docs/report.tex` (de 2 ori pentru referințe). Output: `docs/report.pdf`.
