# ReLU vs ELU Neural Network Comparison

Бұл жоба ReLU және ELU активация функцияларын салыстырып, олардың нейрондық желі оқу процесіне әсерін зерттейді. Нәтижелер CSV файлдарға сақталып, графиктер арқылы салыстырылады.

## Орнату

Жобаны виртуалды ортада іске қосу үшін:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Қолдану
Бір модельді оқу (Single run)
bash
Копировать код
python -m src.train --activation elu --optimizer adam --bn on --dropout 0.2 --subset_size 10000 --config config.yaml
Параметрлер торын тексеру (Grid search)
bash
Копировать код
python -m src.experiments --config config.yaml --activations elu relu leakyrelu --optimizers adam sgd --bn on off --dropouts 0.0 0.2 0.5 --subset_sizes 5000 10000
Нәтижелерді жинақтау (Summary)
bash
Копировать код
python -m src.report_utils --glob "results/*/metrics.csv" --out results/summary
Нәтижелер
ELU және ReLU активацияларының оқу динамикасын салыстыру

Validation Accuracy, Train Loss және басқа метрикаларды визуалдау

Efficiency (Accuracy per second) және орташа көрсеткіштер

Барлық нәтижелер results/ папкасында сақталады.
