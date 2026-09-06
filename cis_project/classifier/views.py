from django.shortcuts import render
from django.apps import apps
from .forms import ImageUploadForm
from PIL import Image
import numpy as np
import tensorflow as tf

CLASS_NAMES = ['CIS', 'non_CIS']  # match your class_names order from training

# Report these from your evaluation step so the UI shows each model's known accuracy
MODEL_ACCURACIES = {
    "Custom CNN": 0.6289,
    "Pretrained V1 (frozen)": 0.9207,
    "Pretrained V2 (fine-tuned)": 0.8385,
    "Pretrained V3 (augmented)": 0.8414,
}

def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(request):
    results = None
    uploaded_image_url = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            img_array = preprocess_image(image_file)

            config = apps.get_app_config('classifier')
            models_dict = config.models_dict

            results = []
            for name, model in models_dict.items():
                pred = model.predict(img_array, verbose=0)[0][0]
                predicted_class = CLASS_NAMES[1] if pred > 0.5 else CLASS_NAMES[0]
                confidence = pred if pred > 0.5 else 1 - pred
                results.append({
                    "model_name": name,
                    "prediction": predicted_class,
                    "confidence": round(float(confidence) * 100, 2),
                    "model_accuracy": round(MODEL_ACCURACIES[name] * 100, 2),
                })

            # Save uploaded file for display
            from django.core.files.storage import FileSystemStorage
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            uploaded_image_url = fs.url(filename)
    else:
        form = ImageUploadForm()

    return render(request, 'classifier/predict.html', {
        'form': form,
        'results': results,
        'uploaded_image_url': uploaded_image_url,
    })