from django.apps import AppConfig
import os

class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'
    models_dict = {}

    def ready(self):
        import tensorflow as tf
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        ClassifierConfig.models_dict = {
            "Custom CNN": tf.keras.models.load_model(os.path.join(model_dir, "custom_model.h5")),
            "Pretrained V1 (frozen)": tf.keras.models.load_model(os.path.join(model_dir, "pretrained_v1_frozen.h5")),
            "Pretrained V2 (fine-tuned)": tf.keras.models.load_model(os.path.join(model_dir, "pretrained_v2_finetuned.h5")),
            "Pretrained V3 (augmented)": tf.keras.models.load_model(os.path.join(model_dir, "pretrained_v3_augmented.h5")),
        }