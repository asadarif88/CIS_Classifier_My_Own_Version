# CIS Classifier

A Django web application that classifies uploaded images as **CIS** or **non-CIS** using
four trained Keras/TensorFlow models and compares their predictions side by side.

## Models

| Model                       | Validation accuracy |
| --------------------------- | ------------------- |
| Custom CNN                  | 62.9%               |
| Pretrained V1 (frozen)      | 92.1%               |
| Pretrained V2 (fine-tuned)  | 83.9%               |
| Pretrained V3 (augmented)   | 84.1%               |

Trained model files live in `cis_project/classifier/models/*.h5`.
Training code is in `cis_project/CIS classifier.ipynb`.

## Project layout

```
cis_project/
├── cis_project/          # Django project settings
├── classifier/           # App: views, forms, model loading, templates
│   └── models/           # Trained .h5 models
├── CIS classifier.ipynb  # Model training / evaluation notebook
├── manage.py
└── requirements.txt
```

> The training dataset (~3.7 GB of images) is **not** included in this repo and is
> excluded via `.gitignore`. Place it under `cis_project/dataset/` locally to retrain.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r cis_project/requirements.txt

cd cis_project
python manage.py migrate
python manage.py runserver
```

Then open http://127.0.0.1:8000/ and upload an image.

## Notes

- `settings.py` ships with `DEBUG = True` and a development `SECRET_KEY`.
  Set a real secret key and disable debug before deploying.
- Uploaded images are saved to `cis_project/media/` (gitignored).
