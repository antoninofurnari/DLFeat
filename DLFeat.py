# DLFeat: Deep Learning Feature Extraction Library
# Inspired by VLFeat for ease of use and modularity in the modern deep learning era.
# Version: 0.4.9 
# Author: Antonino Furnari
# Date: 2025-06-01

"""
DLFeat: Deep Learning Feature Extraction Library
================================================

**DLFeat** is a Python library designed for easy and modular feature extraction 
from various data modalities including images, videos, audio, and text. 
It leverages powerful pre-trained models from libraries like PyTorch, torchvision, 
Transformers, Sentence-Transformers, and TIMM. 
The goal is to provide a "black box" tool suitable for educational and research purposes, 
allowing users to quickly extract meaningful features for data analysis tasks 
without needing to delve into the complexities of each model's architecture or training.

This version is compatible with the scikit-learn transformer API, includes an expanded
model zoo, and a callable self-test function with a summary report.

Core Features
-------------
- **Unified API**: Consistent `DLFeatExtractor` class for all modalities.
- **Scikit-learn Compatible**: Implements `BaseEstimator` and `TransformerMixin` for easy pipeline integration.
- **Multi-Modal Support**: Extract features from images, videos, audio, and text.
- **Extensive Model Zoo**: Access to a wide range of pre-trained models (see Model Zoo section).
- **Automatic Handling**: Manages model loading, preprocessing, and device placement (CPU/GPU).
- **Single-File Library**: Easy to distribute and integrate (though dependencies must be installed).
- **Self-Testing**: Built-in function `run_self_tests()` to verify model availability and basic functionality.

Installation
------------
DLFeat itself is a single Python file. However, it relies on several external 
libraries. You'll need to install them first.

.. code-block:: bash

    pip install torch torchvision torchaudio scikit-learn Pillow numpy scipy
    pip install transformers sentence-transformers timm requests

For optimal performance and access to all models, ensure these libraries, especially `transformers` and `torchvision`, are up-to-date:

.. code-block:: bash

    pip install --upgrade torch torchvision torchaudio transformers sentence-transformers timm requests

Getting Started
---------------
Here's how to quickly get started with `DLFeat` for various modalities.

**1. Import necessary components:**

.. code-block:: python

    from DLFeat import DLFeatExtractor, list_available_models

    # You can list models for a specific task
    # print(list_available_models(task_type="image"))

**2. Image Feature Extraction:**

.. code-block:: python

    # Initialize for an image model
    img_extractor = DLFeatExtractor(model_name="resnet18", task_type="image")
    
    # Provide a list of image paths or PIL.Image objects
    # Replace with your actual image paths
    try:
        # Create a dummy image for the snippet if Pillow is available
        from PIL import Image as PILImage, ImageDraw as PILImageDraw
        dummy_img_path = "temp_dummy_image_snippet.png"
        img = PILImage.new('RGB', (224, 224), color = 'skyblue')
        d = PILImageDraw.Draw(img)
        d.text((10,10), "Sample", fill=(0,0,0))
        img.save(dummy_img_path)
        
        image_paths = [dummy_img_path, dummy_img_path] 
        
        # Fit (no-op for pre-trained models) and transform
        img_extractor.fit(image_paths) 
        image_features = img_extractor.transform(image_paths)
        print(f"Image features shape: {image_features.shape}") # e.g., (2, 512) for resnet18
        
        if os.path.exists(dummy_img_path): os.remove(dummy_img_path)
    except ImportError:
        print("Pillow not found, skipping image snippet. Install Pillow to run.")
    except Exception as e:
        print(f"Error in image snippet: {e}")

**3. Text Feature Extraction:**

.. code-block:: python

    text_extractor = DLFeatExtractor(model_name="sentence-bert", task_type="text")
    texts = ["This is the first sentence for feature extraction.", 
             "DLFeat makes it easy to get embeddings."]
    text_features = text_extractor.transform(texts)
    print(f"Text features shape: {text_features.shape}") # e.g., (2, 384) for sentence-bert

**4. Audio Feature Extraction:**
   Requires `torchaudio` and `scipy` for dummy audio creation in this snippet.

.. code-block:: python

    try:
        # Create a dummy audio file for the snippet
        import numpy as np_audio
        import scipy.io.wavfile as scipy_wav_audio
        dummy_audio_path = "temp_dummy_audio_snippet.wav"
        sample_rate = 16000; duration = 1; frequency = 440
        t = np_audio.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        data = np_audio.sin(2 * np_audio.pi * frequency * t) * 0.5
        data_int16 = (data * 32767).astype(np_audio.int16)
        scipy_wav_audio.write(dummy_audio_path, sample_rate, data_int16)

        audio_extractor = DLFeatExtractor(model_name="wav2vec2_base", task_type="audio")
        audio_paths = [dummy_audio_path, dummy_audio_path]
        audio_features = audio_extractor.transform(audio_paths)
        print(f"Audio features shape: {audio_features.shape}") # e.g., (2, 768) for wav2vec2_base

        if os.path.exists(dummy_audio_path): os.remove(dummy_audio_path)
    except ImportError:
        print("Scipy or Numpy not found, skipping audio snippet. Install them to run.")
    except Exception as e:
        print(f"Error in audio snippet: {e}")


**5. Video Feature Extraction:**
   Note: Video processing can be slow. For self-tests, `DLFeat` downloads a small sample video.
   You'll need `requests` library installed for this download in tests.
   Replace `"your_video.mp4"` with actual video paths for your use.

.. code-block:: python

    # Using a torchvision model for this example
    # Replace "path/to/your/video1.mp4" with a real video file for actual use.
    # The self-test function (run_self_tests) attempts to download a sample video.
    try:
        video_extractor = DLFeatExtractor(model_name="r2plus1d_18", task_type="video")
        # This will likely fail or be skipped if dummy_video.mp4 isn't a real video
        # For a real test, provide a list of actual video paths:
        # video_paths = ["path/to/your/video1.mp4", "path/to/your/video2.mp4"]
        # video_features = video_extractor.transform(video_paths)
        # print(f"Video features shape: {video_features.shape}") 
        print("Video Extractor for r2plus1d_18 initialized. Test with real video paths.")
    except Exception as e:
        print(f"Could not initialize video extractor or run snippet: {e}")
        print("Video models require their dependencies (torchvision, transformers) to be correctly installed.")


**6. Multimodal (Image-Text) Feature Extraction:**

.. code-block:: python

    try:
        # Re-create dummy image for this snippet
        from PIL import Image as PILImageMM, ImageDraw as PILImageDrawMM
        dummy_img_path_mm = "temp_dummy_image_mm_snippet.png"
        img_mm = PILImageMM.new('RGB', (224, 224), color = 'lightgreen')
        d_mm = PILImageDrawMM.Draw(img_mm)
        d_mm.text((10,10), "MM Sample", fill=(0,0,0))
        img_mm.save(dummy_img_path_mm)

        clip_extractor = DLFeatExtractor(model_name="clip_vit_b32", task_type="multimodal_image_text")
        multimodal_data = [
            (dummy_img_path_mm, "A light green square with text."),
            (dummy_img_path_mm, "Another description of the same image.")
        ]
        multimodal_features = clip_extractor.transform(multimodal_data)
        print(f"CLIP Image features shape: {multimodal_features['image_features'].shape}") # e.g., (2, 512)
        print(f"CLIP Text features shape: {multimodal_features['text_features'].shape}")   # e.g., (2, 512)

        if os.path.exists(dummy_img_path_mm): os.remove(dummy_img_path_mm)
    except ImportError:
        print("Pillow not found, skipping multimodal image-text snippet.")
    except Exception as e:
        print(f"Error in multimodal (image-text) snippet: {e}")


API Reference
-------------

.. automodule:: DLFeat
   :members: DLFeatExtractor, list_available_models, run_self_tests

**DLFeatExtractor Class**

.. autoclass:: DLFeat.DLFeatExtractor
   :members: __init__, fit, transform, get_feature_dimension, get_model_config

**Utility Functions**

.. autofunction:: DLFeat.list_available_models
.. autofunction:: DLFeat.run_self_tests

Model Zoo
---------
The following table provides an overview of the models available in DLFeat.
Performance metrics (Acc., mAP, R@1, etc.) are typically reported on standard benchmarks 
(e.g., ImageNet, Kinetics-400, GLUE, MSCOCO, MSR-VTT). FLOPS and Speed are indicative and 
can vary significantly based on hardware, batch size, input resolution, and specific implementation. 
"SSL" denotes Self-Supervised Learning. "Multimodal" models are trained on multiple data types.

.. list-table:: DLFeat Model Zoo
   :widths: 12 28 8 20 8 10 12 12
   :header-rows: 1

   * - Modality
     - Model Name (Identifier)
     - Feat. Dim
     - Performance (Benchmark)
     - FLOPS (G)
     - Speed
     - Supervision
     - Source
   * - **Image**
     - 
     - 
     - 
     - 
     - 
     - 
     - 
   * - Image
     - `resnet18`
     - 512
     - 69.76% (ImageNet Top-1)
     - 1.8
     - Fast
     - Supervised
     - torchvision
   * - Image
     - `resnet34`
     - 512
     - 73.30% (ImageNet Top-1)
     - 3.6
     - Fast
     - Supervised
     - torchvision
   * - Image
     - `resnet50`
     - 2048
     - 80.86% (ImageNet Top-1, tv)
     - 4.1
     - Medium
     - Supervised
     - torchvision_or_timm
   * - Image
     - `resnet101`
     - 2048
     - 81.88% (ImageNet Top-1, tv)
     - 7.8
     - Medium
     - Supervised
     - torchvision_or_timm
   * - Image
     - `resnet152`
     - 2048
     - 82.28% (ImageNet Top-1, tv)
     - 11.5
     - Slower
     - Supervised
     - torchvision_or_timm
   * - Image
     - `efficientnet_b0`
     - 1280
     - 77.69% (ImageNet Top-1)
     - 0.39
     - Very Fast
     - Supervised
     - timm
   * - Image
     - `efficientnet_b2`
     - 1408
     - 80.51% (ImageNet Top-1)
     - 1.0
     - Fast
     - Supervised
     - timm
   * - Image
     - `efficientnet_b4`
     - 1792
     - 83.37% (ImageNet Top-1)
     - 4.4
     - Medium
     - Supervised
     - timm
   * - Image
     - `mobilenet_v2`
     - 1280
     - 71.88% (ImageNet Top-1)
     - 0.3
     - Very Fast
     - Supervised
     - torchvision
   * - Image
     - `mobilenet_v3_small`
     - 576
     - 67.67% (ImageNet Top-1)
     - 0.06
     - Very Fast
     - Supervised
     - torchvision
   * - Image
     - `mobilenet_v3_large`
     - 960
     - 74.04% (ImageNet Top-1)
     - 0.22
     - Very Fast
     - Supervised
     - torchvision
   * - Image
     - `vit_tiny_patch16_224`
     - 192
     - 75.4% (ImageNet Top-1, DeiT)
     - 1.3
     - Fast
     - Supervised (DeiT)
     - timm
   * - Image
     - `vit_small_patch16_224`
     - 384
     - 81.2% (ImageNet Top-1, DeiT)
     - 4.6
     - Medium
     - Supervised (DeiT)
     - timm
   * - Image
     - `vit_base_patch16_224`
     - 768
     - 85.2% (ImageNet Top-1, MAE FT)
     - 17.6
     - Medium
     - SSL (MAE)
     - timm
   * - Image
     - `dinov2_base`
     - 768
     - 82.8% (ImageNet k-NN, ViT-B/14)
     - ~33 (ViT-B/14)
     - Medium
     - SSL (DINOv2)
     - Transformers
   * - **Video**
     - 
     - 
     - 
     - 
     - 
     - 
     - 
   * - Video
     - `r2plus1d_18`
     - 512
     - 65.2% (K400 Top-1)
     - 31.6 (16f)
     - Medium
     - Supervised
     - torchvision
   * - Video
     - `video_swin_t`
     - 768
     - 78.8% (K400 Top-1)
     - 48 (32x224^2)
     - Medium
     - Supervised
     - torchvision
   * - Video
     - `video_swin_s`
     - 768
     - 81.6% (K400 Top-1)
     - 92 (32x224^2)
     - Slower
     - Supervised
     - torchvision
   * - Video
     - `video_swin_b`
     - 1024
     - 82.7% (K400 Top-1)
     - 199 (32x224^2)
     - Slower
     - Supervised
     - torchvision
   * - Video
     - `videomae_base_k400_pt`
     - 768
     - 81.2% (K400 Top-1, ViT-B)
     - ~168 (16x224^2)
     - Medium
     - Supervised (PT+FT)
     - transformers
   * - **Audio**
     - 
     - 
     - 
     - 
     - 
     - 
     - 
   * - Audio
     - `wav2vec2_base`
     - 768
     - ~6.9% (LibriSpeech WER, no LM)
     - 94.5M Params
     - Fast
     - SSL (Wav2Vec2)
     - Transformers
   * - Audio
     - `ast_vit_base_patch16_224`
     - 768
     - 0.459 (AudioSet mAP)
     - 87M Params
     - Medium
     - Supervised
     - Transformers
   * - **Text**
     - 
     - 
     - 
     - 
     - 
     - 
     - 
   * - Text
     - `sentence-bert`
     - 384
     - 85.3 (STS-B Spearman)
     - N/A
     - Very Fast
     - SSL (SBERT)
     - sentence-transformers
   * - Text
     - `bert_base_uncased`
     - 768
     - 79.6 (GLUE Avg.)
     - 110M Params
     - Medium
     - SSL (BERT)
     - Transformers
   * - **Multimodal**
     - 
     - 
     - 
     - 
     - 
     - 
     - 
   * - Image-Text
     - `clip_vit_b32`
     - 512
     - 63.3% (ImageNet zero-shot)
     - N/A
     - Fast
     - Multimodal SSL
     - Transformers
   * - Video-Text
     - `xclip_base_patch16`
     - 512
     - 46.7% (MSR-VTT R@1)
     - N/A
     - Medium
     - Multimodal SSL
     - Transformers

*Note on FLOPS/Speed: These are highly approximate and depend on input size, hardware, and batching. "Fast" might mean >100 FPS for images on a modern GPU. "N/A" indicates data not readily found or highly variable. For parameter counts, "M Params" refers to millions of parameters.*

"""

__version__ = "0.4.9" 

import torch
import torchvision 
import torchvision.transforms as T
import torchvision.models as tv_models 
import torchvision.models.video as tv_video_models 
from PIL import Image, ImageDraw 
import numpy as np
import warnings
import os
import textwrap 
import sys 
import traceback # For printing full tracebacks in tests

try:
    import requests
except ImportError:
    warnings.warn(
        "The 'requests' library is not installed. "
        "DLFeat self-tests will use a placeholder for video models instead of downloading a real sample. "
        "To enable real video testing, please install it: pip install requests"
    )
    class requests: 
        @staticmethod
        def get(*args, **kwargs):
            raise ImportError("requests library is not installed.")
        class exceptions: 
            class RequestException(Exception): pass


try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    warnings.warn(
        "scikit-learn not found. DLFeatExtractor will not be scikit-learn compatible. "
        "Please install it: pip install scikit-learn"
    )
    class BaseEstimator: pass
    class TransformerMixin: pass

try:
    from transformers import (
        AutoProcessor, AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor,
        AutoImageProcessor, 
        CLIPProcessor, CLIPModel, BlipProcessor, BlipModel,
        VideoMAEImageProcessor, VideoMAEModel, 
        XCLIPProcessor, XCLIPModel,
        ASTFeatureExtractor,
        Dinov2Model
    )
except ImportError:
    warnings.warn(
        "Transformers library not found or key components are missing. "
        "Text, some audio, DINOv2, VideoMAE and multimodal models may not be available. "
        "Please install or upgrade transformers: pip install --upgrade transformers"
    )
    class AutoProcessor: pass
    class AutoModel: pass
    class AutoTokenizer: pass
    class Wav2Vec2FeatureExtractor: pass
    class AutoImageProcessor: pass
    class CLIPProcessor: pass
    class CLIPModel: pass
    class BlipProcessor: pass
    class BlipModel: pass
    class VideoMAEImageProcessor: pass 
    class VideoMAEModel: pass
    class XCLIPProcessor: pass
    class XCLIPModel: pass
    class ASTFeatureExtractor: pass
    class Dinov2Model: pass


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    warnings.warn(
        "Sentence-Transformers library not found. `sentence-bert` model will not be available. "
        "Please install it: pip install sentence-transformers"
    )
    class SentenceTransformer: pass

try:
    import timm
except ImportError:
    warnings.warn(
        "TIMM library not found. Some image models (ViT, EfficientNet, ConvNeXt) will not be available. " 
        "Please install it: pip install timm"
    )
    class timm:
        @staticmethod
        def create_model(model_name, pretrained=True, num_classes=0):
            raise ImportError("TIMM library is not installed.")

try:
    import torchaudio
    import torchaudio.transforms as TA
    import scipy.io.wavfile as scipy_wav
except ImportError:
    warnings.warn(
        "Torchaudio or Scipy library not found. Audio processing or self-tests for audio might be limited. "
        "Please install them: pip install torchaudio scipy"
    )
    class torchaudio: pass 
    class TA: pass 
    class scipy_wav: 
        @staticmethod
        def write(*args, **kwargs): raise ImportError("Scipy not installed, cannot write dummy audio.")

# PyTorchVideo related imports and mocks are removed as it's no longer a direct dependency.

MODEL_CONFIGS = {
    # --- Image Models ---
    "resnet18": {"task": "image", "dim": 512, "input_size": 224, "source": "torchvision"},
    "resnet34": {"task": "image", "dim": 512, "input_size": 224, "source": "torchvision"},
    "resnet50": {"task": "image", "dim": 2048, "input_size": 224, "source": "torchvision_or_timm"},
    "resnet101": {"task": "image", "dim": 2048, "input_size": 224, "source": "torchvision_or_timm"},
    "resnet152": {"task": "image", "dim": 2048, "input_size": 224, "source": "torchvision_or_timm"},
    "efficientnet_b0": {"task": "image", "dim": 1280, "input_size": 224, "source": "timm", "timm_name": "efficientnet_b0"},
    "efficientnet_b2": {"task": "image", "dim": 1408, "input_size": 260, "source": "timm", "timm_name": "efficientnet_b2"},
    "efficientnet_b4": {"task": "image", "dim": 1792, "input_size": 380, "source": "timm", "timm_name": "efficientnet_b4"},
    "mobilenet_v2": {"task": "image", "dim": 1280, "input_size": 224, "source": "torchvision"},
    "mobilenet_v3_small": {"task": "image", "dim": 576, "input_size": 224, "source": "torchvision"}, 
    "mobilenet_v3_large": {"task": "image", "dim": 960, "input_size": 224, "source": "torchvision"}, 
    "vit_tiny_patch16_224": {"task": "image", "dim": 192, "input_size": 224, "source": "timm", "timm_name": "vit_tiny_patch16_224.augreg_in21k_ft_in1k"}, 
    "vit_small_patch16_224": {"task": "image", "dim": 384, "input_size": 224, "source": "timm", "timm_name": "vit_small_patch16_224.augreg_in21k_ft_in1k"},
    "vit_base_patch16_224": {"task": "image", "dim": 768, "input_size": 224, "source": "timm", "timm_name": "vit_base_patch16_224.mae"}, 
    "dinov2_base": {"task": "image", "dim": 768, "input_size": 224, "source": "transformers", "hf_name": "facebook/dinov2-base"},

    # --- Video Models (Refactored - MViT removed) ---
    # torchvision models
    "r2plus1d_18": {"task": "video", "dim": 512, "source": "torchvision", "tv_model_name":"r2plus1d_18", "clip_len": 16, "frame_rate": 15, "input_size": 112}, 
    "video_swin_t": {"task": "video", "dim": 768, "source": "torchvision", "tv_model_name": "swin3d_t", "clip_len": 32, "input_size": 224}, 
    "video_swin_s": {"task": "video", "dim": 768, "source": "torchvision", "tv_model_name": "swin3d_s", "clip_len": 32, "input_size": 224}, 
    "video_swin_b": {"task": "video", "dim": 1024, "source": "torchvision", "tv_model_name": "swin3d_b", "clip_len": 32, "input_size": 224},
    
    # transformers models
    "videomae_base_k400_pt": {"task": "video", "dim": 768, "source": "transformers", "hf_name": "MCG-NJU/videomae-base-finetuned-kinetics", "num_frames": 16, "input_size": 224},
    
    # --- Audio Models ---
    "wav2vec2_base": {"task": "audio", "dim": 768, "source": "transformers", "hf_name": "facebook/wav2vec2-base-960h", "sampling_rate": 16000},
    "ast_vit_base_patch16_224": {"task": "audio", "dim": 768, "source": "transformers", "hf_name": "MIT/ast-finetuned-audioset-10-10-0.4593", "sampling_rate": 16000, "num_mel_bins": 128, "max_length_s": 10.24},

    # --- Text Models ---
    "sentence-bert": {"task": "text", "dim": 384, "source": "sentence-transformers", "st_name": "all-MiniLM-L6-v2"},
    "bert_base_uncased": {"task": "text", "dim": 768, "source": "transformers", "hf_name": "bert-base-uncased"},

    # --- Multimodal Models ---
    "clip_vit_b32": {"task": "multimodal_image_text", "dim": 512, "source": "transformers", "hf_name": "openai/clip-vit-base-patch32"},
    "xclip_base_patch16": {"task": "multimodal_video_text", "dim": 512, "source": "transformers", "hf_name": "microsoft/xclip-base-patch16", "num_frames": 8}
}

DEFAULT_MODELS_TO_TEST = [ 
    "resnet18", "efficientnet_b0", "mobilenet_v2", "vit_tiny_patch16_224", "dinov2_base",
    "r2plus1d_18", "videomae_base_k400_pt", "video_swin_t",
    "wav2vec2_base", "sentence-bert", 
    "clip_vit_b32", "xclip_base_patch16"
]


def list_available_models(task_type=None):
    if task_type:
        return [name for name, config in MODEL_CONFIGS.items() if config["task"] == task_type]
    return list(MODEL_CONFIGS.keys())

class DLFeatExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model_name, task_type=None, device="auto"):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {list_available_models()}"
            )

        self.model_name = model_name 
        self.config = MODEL_CONFIGS[self.model_name]
        self.task_type = self.config["task"]

        if task_type and task_type != self.task_type:
            raise ValueError(
                f"Provided task_type '{task_type}' does not match model '{self.model_name}'s task '{self.task_type}'."
            )

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): 
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.processor = None 
        self.tokenizer = None 
        self.image_transform = None 
        self.audio_resampler = None 
        self.video_frame_transform = None 
        self.target_sr = None 

        self._load_model()

    def get_feature_dimension(self):
        return self.config["dim"]

    def get_model_config(self):
        return self.config.copy()
    
    def _load_image_model_torchvision(self, model_name_tv):
        # ... (same as v0.4.7)
        model_fn = getattr(tv_models, model_name_tv)
        
        weights_enum_name_to_try = None
        if model_name_tv.startswith("resnet"):
            name_part = model_name_tv[len("resnet"):] 
            weights_enum_name_to_try = f"ResNet{name_part}_Weights" 
        elif model_name_tv.startswith("vgg"): 
            weights_enum_name_to_try = f"{model_name_tv.upper()}_Weights"
        elif model_name_tv == "mobilenet_v2": 
             weights_enum_name_to_try = "MobileNet_V2_Weights"
        elif model_name_tv == "mobilenet_v3_large": 
             weights_enum_name_to_try = "MobileNet_V3_Large_Weights"
        elif model_name_tv == "mobilenet_v3_small": 
             weights_enum_name_to_try = "MobileNet_V3_Small_Weights"
        elif model_name_tv.startswith("efficientnet_b"): 
            name_part = model_name_tv[len("efficientnet_"):] 
            weights_enum_name_to_try = f"EfficientNet_{name_part.upper()}_Weights" 
        elif model_name_tv.startswith("convnext_"): 
            name_part = model_name_tv[len("convnext_"):] 
            weights_enum_name_to_try = f"ConvNeXt_{name_part.capitalize()}_Weights" 
        else:
            weights_enum_name_to_try = model_name_tv[0].upper() + model_name_tv[1:] + "_Weights"

        try:
            weights_class = getattr(tv_models, weights_enum_name_to_try)
            
            if hasattr(weights_class, 'DEFAULT'):
                weights_obj = weights_class.DEFAULT
            elif hasattr(weights_class, 'IMAGENET1K_V1'): 
                weights_obj = weights_class.IMAGENET1K_V1
            else:
                available_enum_members = [m for m in dir(weights_class) if not m.startswith('_') and m.isupper()]
                if available_enum_members:
                    first_available_weight_name = available_enum_members[0]
                    weights_obj = getattr(weights_class, first_available_weight_name)
                    warnings.warn(f"DLFeatExtractor: Using first available weight '{first_available_weight_name}' for {model_name_tv} as DEFAULT/IMAGENET1K_V1 not found in its enum.")
                else:
                    raise AttributeError(f"No DEFAULT, IMAGENET1K_V1, or other suitable weights found in {weights_enum_name_to_try}")
            
            self.model = model_fn(weights=weights_obj)
            if hasattr(weights_obj, 'transforms') and callable(weights_obj.transforms):
                self.image_transform = weights_obj.transforms()

        except (AttributeError, ValueError) as e_new_api_img: 
            warnings.warn(
                f"DLFeatExtractor: Failed to use new 'weights' API for image model {model_name_tv} (Error: {type(e_new_api_img).__name__}: {e_new_api_img}). "
                f"Falling back to legacy 'pretrained=True'. Torchvision warnings may follow."
            )
            self.model = model_fn(pretrained=True)
        
        if model_name_tv.startswith("mobilenet_v3"):
            self.model.classifier = torch.nn.Identity() 
        elif hasattr(self.model, 'classifier') and isinstance(self.model.classifier, torch.nn.Sequential) and len(self.model.classifier) > 0 :
             self.model.classifier[-1] = torch.nn.Identity()
        elif hasattr(self.model, 'fc'): 
            self.model.fc = torch.nn.Identity()
        elif hasattr(self.model, 'classifier') and isinstance(self.model.classifier, torch.nn.Linear): 
            self.model.classifier = torch.nn.Identity()

        self.model.eval().to(self.device)
        
        if self.image_transform is None:
            input_size = self.config.get("input_size", 224)
            self.image_transform = T.Compose([
                T.Resize(256 if input_size == 224 else int(input_size / (224.0/256.0))), 
                T.CenterCrop(input_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif hasattr(self.image_transform, 'crop_size') and isinstance(self.image_transform.crop_size, int) and self.image_transform.crop_size != self.config.get("input_size", 224):
            pass
    
    def _load_image_model_timm(self, timm_model_name):
        if not hasattr(timm, 'create_model'): 
             raise ImportError("TIMM library is not installed. Please install it: pip install timm")
        self.model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
        self.model.eval().to(self.device)
        data_config = timm.data.resolve_data_config({}, model=self.model)
        self.image_transform = timm.data.create_transform(**data_config)
        if "input_size" not in self.config: 
            self.config["input_size"] = data_config['input_size'][1] 

    def _load_image_model_dinov2(self):
        if not hasattr(Dinov2Model, 'from_pretrained') or not hasattr(AutoImageProcessor, 'from_pretrained'):
            raise ImportError(
                "Transformers components (Dinov2Model or AutoImageProcessor) not available or are dummy classes. "
                "Update transformers: pip install --upgrade transformers"
            )
        hf_name = self.config["hf_name"]
        self.processor = AutoImageProcessor.from_pretrained(hf_name) 
        self.model = Dinov2Model.from_pretrained(hf_name)
        self.model.eval().to(self.device)

    def _load_model(self):
        source = self.config["source"]
        self.video_frame_transform = None 

        if self.task_type == "image":
            if self.model_name == "dinov2_base": 
                if not hasattr(Dinov2Model, 'from_pretrained'): 
                    raise ImportError("Transformers (Dinov2Model) dummy class detected or not installed.")
                self._load_image_model_dinov2() 
            elif source == "torchvision": 
                self._load_image_model_torchvision(self.model_name)
            elif source == "torchvision_or_timm":
                try:
                    if not hasattr(timm, 'create_model'): raise ImportError("TIMM not available")
                    timm_model_id = self.config.get("timm_name", self.model_name)
                    self._load_image_model_timm(timm_model_id)
                except Exception: 
                    self._load_image_model_torchvision(self.model_name) 
            elif source == "timm":
                if not hasattr(timm, 'create_model'): raise ImportError("TIMM not available")
                self._load_image_model_timm(self.config["timm_name"])
            else:
                raise ValueError(f"Unsupported image model source for {self.model_name}: {source}")

        elif self.task_type == "text":
            if source == "sentence-transformers":
                if not hasattr(SentenceTransformer, 'encode'): 
                     raise ImportError("Sentence-Transformers dummy class detected or not installed.")
                self.model = SentenceTransformer(self.config["st_name"], device=self.device)
            elif source == "transformers":
                if not hasattr(AutoTokenizer, 'from_pretrained'): 
                    raise ImportError("Transformers (AutoTokenizer) dummy class detected or not installed.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config["hf_name"], use_fast=True) # use_fast=True
                if not hasattr(AutoModel, 'from_pretrained'):
                     raise ImportError("Transformers (AutoModel) dummy class detected or not installed.")
                self.model = AutoModel.from_pretrained(self.config["hf_name"])
                self.model.eval().to(self.device)
            else:
                raise ValueError(f"Unsupported text model source: {source}")

        elif self.task_type == "video":
            if source == "torchvision":
                actual_tv_model_name = self.config.get("tv_model_name") 
                if not actual_tv_model_name:
                    raise ValueError(f"Configuration for torchvision video model {self.model_name} is missing 'tv_model_name'.")
                if not hasattr(tv_video_models, actual_tv_model_name): 
                    raise ImportError(f"Video model '{actual_tv_model_name}' not found in torchvision.models.video.")
                model_fn = getattr(tv_video_models, actual_tv_model_name)
                weights_enum_name_map = { 
                    "r2plus1d_18": "R2Plus1D_18_Weights",
                    "swin3d_t": "Swin3D_T_Weights", 
                    "swin3d_s": "Swin3D_S_Weights", 
                    "swin3d_b": "Swin3D_B_Weights", 
                }
                lookup_key_for_map = actual_tv_model_name 
                weights_enum_name = weights_enum_name_map.get(lookup_key_for_map)
                weights_value_name = "DEFAULT" 
                if weights_enum_name:
                    try:
                        weights_enum_class = getattr(tv_video_models, weights_enum_name, None)
                        if weights_enum_class:
                            if hasattr(weights_enum_class, weights_value_name): weights_obj = getattr(weights_enum_class, weights_value_name)
                            elif hasattr(weights_enum_class, 'KINETICS400_V1'): weights_obj = weights_enum_class.KINETICS400_V1
                            elif hasattr(weights_enum_class, 'KINETICS400_IMAGENET1K_V1'): weights_obj = weights_enum_class.KINETICS400_IMAGENET1K_V1
                            else: 
                                available_weights = [w for w in dir(weights_enum_class) if w.isupper() and not w.startswith('_')]
                                if available_weights:
                                    weights_obj = getattr(weights_enum_class, available_weights[0])
                                    warnings.warn(f"DLFeat: Using first available weight '{available_weights[0]}' for {actual_tv_model_name}")
                                else: raise AttributeError(f"No suitable weights found in {weights_enum_name}")
                            
                            self.model = model_fn(weights=weights_obj)
                            if hasattr(weights_obj, 'transforms') and callable(weights_obj.transforms): self.video_frame_transform = weights_obj.transforms()
                        else: raise AttributeError(f"{weights_enum_name} enum not found for {actual_tv_model_name}.")
                    except (AttributeError, ValueError) as e_new_api:
                        warnings.warn(
                            f"DLFeatExtractor: Failed to use new 'weights' API for {actual_tv_model_name} (Error: {type(e_new_api).__name__}: {e_new_api}). "
                            f"Falling back to legacy 'pretrained=True' if applicable."
                        )
                        import inspect
                        sig = inspect.signature(model_fn)
                        if 'pretrained' in sig.parameters:
                            self.model = model_fn(pretrained=True)
                        else: 
                            raise ImportError(f"New weights API failed for {actual_tv_model_name} and legacy 'pretrained' not supported or model init failed. Error: {e_new_api}")
                else: 
                    warnings.warn(
                        f"DLFeatExtractor: No specific 'weights' API logic for {actual_tv_model_name} (lookup key: {lookup_key_for_map}). Attempting legacy 'pretrained=True'."
                    )
                    self.model = model_fn(pretrained=True) 

                if hasattr(self.model, 'head'): 
                    if isinstance(self.model.head, torch.nn.Linear): self.model.head = torch.nn.Identity()
                    elif isinstance(self.model.head, torch.nn.Sequential) and len(self.model.head)>0 and isinstance(self.model.head[-1], torch.nn.Linear):
                        self.model.head[-1] = torch.nn.Identity()
                elif hasattr(self.model, 'fc'): self.model.fc = torch.nn.Identity() 
                
                self.model.eval().to(self.device)

                if self.video_frame_transform is None: 
                    input_size = self.config.get("input_size", 224)
                    warnings.warn(f"DLFeat: Using manual basic per-frame transforms for {actual_tv_model_name} as weights.transforms() was not available/used for video_frame_transform.")
                    self.video_frame_transform = T.Compose([ 
                        T.ConvertImageDtype(torch.float32), 
                        T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]), 
                        T.Resize([input_size, input_size], antialias=True) 
                    ])
            
            elif source == "transformers": 
                hf_model_name = self.config["hf_name"]
                hf_processor_name = self.config.get("hf_name_processor", hf_model_name) 

                if self.model_name.startswith("videomae"):
                    if not hasattr(VideoMAEImageProcessor, 'from_pretrained') or not hasattr(VideoMAEModel, 'from_pretrained'):
                        raise ImportError("VideoMAE components from Transformers not available or are dummy classes.")
                    self.processor = VideoMAEImageProcessor.from_pretrained(hf_processor_name)
                    self.model = VideoMAEModel.from_pretrained(hf_model_name)
                else: 
                    raise ValueError(f"Unsupported transformers video model: {self.model_name}")
                
                self.model.eval().to(self.device)
            else:
                raise ValueError(f"Unsupported video model source: {source}")

        elif self.task_type == "audio":
            ProcessorCheckClass = ASTFeatureExtractor if self.model_name.startswith("ast") else Wav2Vec2FeatureExtractor
            if not hasattr(ProcessorCheckClass, 'from_pretrained'): 
                raise ImportError(f"Transformers ({ProcessorCheckClass.__name__}) dummy class detected or not installed.")
            if not hasattr(TA, 'Resample'):  
                raise ImportError("Torchaudio (transforms.Resample) dummy class detected or not installed.")
            hf_name = self.config["hf_name"]
            ProcessorClassToUse = ASTFeatureExtractor if self.model_name.startswith("ast") else Wav2Vec2FeatureExtractor
            self.processor = ProcessorClassToUse.from_pretrained(hf_name)
            self.model = AutoModel.from_pretrained(hf_name) 
            self.model.eval().to(self.device)
            self.target_sr = self.config["sampling_rate"] 

        elif self.task_type == "multimodal_image_text":
            ProcessorCheckClass = BlipProcessor if self.model_name.startswith("blip") else CLIPProcessor
            ModelCheckClass = BlipModel if self.model_name.startswith("blip") else CLIPModel
            if not hasattr(ProcessorCheckClass, 'from_pretrained') or not hasattr(ModelCheckClass, 'from_pretrained'):
                raise ImportError(f"Transformers ({ProcessorCheckClass.__name__} or {ModelCheckClass.__name__}) dummy class detected or not installed.")
            hf_name = self.config["hf_name"]
            self.processor = ProcessorCheckClass.from_pretrained(hf_name)
            self.model = ModelCheckClass.from_pretrained(hf_name)
            self.model.eval().to(self.device)

        elif self.task_type == "multimodal_video_text":
            if not hasattr(XCLIPProcessor, 'from_pretrained') or not hasattr(XCLIPModel, 'from_pretrained'): 
                raise ImportError("Transformers (XCLIP components) dummy class detected or not installed.")
            hf_name = self.config["hf_name"]
            self.processor = XCLIPProcessor.from_pretrained(hf_name)
            self.model = XCLIPModel.from_pretrained(hf_name)
            self.model.eval().to(self.device)
            
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
    def _preprocess_image(self, image_input):
        if isinstance(image_input, str):
            if not os.path.exists(image_input): raise FileNotFoundError(f"Image file: {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image): img = image_input.convert("RGB")
        else: raise TypeError("Image input must be a file path or PIL Image.")
        
        if self.model_name == "dinov2_base": 
            return self.processor(images=img, return_tensors="pt")
        elif self.image_transform: 
            return self.image_transform(img).unsqueeze(0) 
        else: 
            raise RuntimeError(f"No image transform or processor available for {self.model_name}")

    def _preprocess_text_transformers(self, text_input):
        return self.tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")

    def _preprocess_video_torchvision(self, video_path):
        if not os.path.exists(video_path): raise FileNotFoundError(f"Video file: {video_path}")
        try:
            # Reverted to default output_format
            frames, _, info = torchvision.io.read_video(video_path, pts_unit='sec') 
        except Exception as e: raise RuntimeError(f"Failed to read {video_path} using torchvision.io: {e}.")
        if frames.numel() == 0: raise ValueError(f"No frames from {video_path}.")
        
        # Assuming default output is THWC (Time, Height, Width, Channel)
        frames = frames.permute(0, 3, 1, 2) # T, C, H, W uint8

        total_frames = frames.shape[0]
        clip_len = self.config.get("clip_len", 32) 
        
        if total_frames < clip_len: 
            padding_count = clip_len - total_frames
            padding_frames = frames[-1:].repeat(padding_count, 1, 1, 1) 
            sampled_frames = torch.cat((frames, padding_frames), dim=0)
        else: 
            indices = torch.linspace(0, total_frames - 1, steps=clip_len).long()
            sampled_frames = frames[indices] 

        if self.video_frame_transform:
            try:
                processed_clip = self.video_frame_transform(sampled_frames) 
                if not (processed_clip.ndim == 4 and processed_clip.shape[0] == 3 and processed_clip.shape[1] == clip_len):
                    warnings.warn(f"DLFeat: Video transform for {self.model_name} output shape {processed_clip.shape} "
                                  f"unexpected. Expected (3, {clip_len}, H, W). Permuting if T,C,H,W was output.")
                    if processed_clip.ndim == 4 and processed_clip.shape[1] == 3 and processed_clip.shape[0] == clip_len: # T,C,H,W
                        processed_clip = processed_clip.permute(1,0,2,3) # C,T,H,W
            except Exception as e_transform:
                 raise RuntimeError(f"Error applying self.video_frame_transform for {self.model_name}: {e_transform}. "
                                    f"Input shape to transform was {sampled_frames.shape}, transform type: {type(self.video_frame_transform)}")
        else: 
            input_size = self.config.get("input_size", 224)
            warnings.warn(f"DLFeat: Using manual basic per-frame transforms for {self.model_name} as weights.transforms() was not available/used for video_frame_transform.")
            manual_per_frame_transform = T.Compose([ 
                 T.ToPILImage(), 
                 T.Resize([input_size, input_size], antialias=True),
                 T.ToTensor(), 
                 T.ConvertImageDtype(torch.float32), 
                 T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            ])
            processed_frames_list = [manual_per_frame_transform(frame) for frame in sampled_frames] 
            processed_clip_temp = torch.stack(processed_frames_list) 
            processed_clip = processed_clip_temp.permute(1,0,2,3) 
            
        return processed_clip.unsqueeze(0) 

    def _preprocess_video_transformers(self, video_path): 
        if not os.path.exists(video_path): raise FileNotFoundError(f"Video file: {video_path}")
        try:
            # Reverted to default output_format
            frames_tensor, _, _ = torchvision.io.read_video(video_path, pts_unit='sec') 
        except Exception as e: raise RuntimeError(f"Failed to read {video_path}: {e}.")
        if frames_tensor.numel() == 0: raise ValueError(f"No frames from {video_path}.")
        
        # Assuming default output is THWC (Time, Height, Width, Channel)
        
        total_frames = frames_tensor.shape[0]
        num_sample_frames = self.config.get("num_frames", 16) 
        
        if total_frames == 0: raise ValueError(f"Video {video_path} has 0 frames.")
        actual_samples_to_take = min(num_sample_frames, total_frames)
        if total_frames < num_sample_frames:
            warnings.warn(f"Video {video_path} has {total_frames} frames, less than configured {num_sample_frames}. Using all {total_frames} frames and padding.")
            indices = np.arange(total_frames)
            padding_needed = num_sample_frames - total_frames
            # frames_tensor is (T,H,W,C) assuming default THWC
            video_frames_list = [Image.fromarray(frames_tensor[i].numpy()) for i in indices]
            for _ in range(padding_needed): video_frames_list.append(video_frames_list[-1])
        else:
            indices = np.linspace(0, total_frames - 1, num_sample_frames, dtype=int)
            video_frames_list = [Image.fromarray(frames_tensor[i].numpy()) for i in indices] 

        if self.model_name.startswith("xclip"): 
            return video_frames_list
        else: # VideoMAE
            return self.processor(images=video_frames_list, return_tensors="pt") 
    
    def _preprocess_audio(self, audio_input):
        if not hasattr(TA, 'Resample'): 
             raise ImportError("Torchaudio (transforms.Resample) dummy class detected or not installed.")
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input): raise FileNotFoundError(f"Audio file: {audio_input}")
            try: waveform, sr = torchaudio.load(audio_input) 
            except Exception as e: raise RuntimeError(f"Failed to load {audio_input}: {e}")
        else: raise TypeError("Audio input must be a file path.")
        if sr != self.target_sr:
            if self.audio_resampler is None or self.audio_resampler.orig_freq != sr:
                ResamplerClass = TA.Resample 
                self.audio_resampler = ResamplerClass(orig_freq=sr, new_freq=self.target_sr,dtype=waveform.dtype).to(waveform.device) 
            waveform = self.audio_resampler(waveform)
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True) 
        processed_input_data = waveform.squeeze(0).numpy() 
        if self.model_name.startswith("ast"): processed_input_data = waveform.squeeze(0) 
        return self.processor(processed_input_data, sampling_rate=self.target_sr, return_tensors="pt", padding=True)


    @torch.no_grad()
    def transform(self, X, batch_size=32, **kwargs):
        if not isinstance(X, list): raise TypeError("Input X for transform must be a list of items.")
        if not X: 
            if self.task_type.startswith("multimodal"): return {k: np.array([]) for k in ["image_features", "text_features", "video_features"] if k.split('_')[0] in self.task_type}
            return np.array([])
        all_features_batches = [] 

        if self.task_type == "image":
            for i in range(0, len(X), batch_size):
                batch_items = X[i:i+batch_size]
                if self.model_name == "dinov2_base":
                    pil_images = [Image.open(item).convert("RGB") if isinstance(item, str) else item.convert("RGB") for item in batch_items]
                    inputs = self.processor(images=pil_images, return_tensors="pt") 
                    inputs = {k: v.to(self.device) for k, v in inputs.items()} 
                    outputs = self.model(**inputs)
                    features = outputs.pooler_output 
                else: 
                    processed_batch_tensors = [self._preprocess_image(item).squeeze(0) for item in batch_items]
                    if not processed_batch_tensors: continue
                    final_batch_tensor = torch.stack(processed_batch_tensors).to(self.device)
                    features = self.model(final_batch_tensor)
                all_features_batches.append(features.cpu().numpy())

        elif self.task_type == "text":
            if self.config["source"] == "sentence-transformers":
                features = self.model.encode(X, convert_to_numpy=True, batch_size=batch_size, device=self.device)
                all_features_batches.append(features)
            else: 
                for i in range(0, len(X), batch_size):
                    batch_texts = X[i:i+batch_size]
                    inputs = self._preprocess_text_transformers(batch_texts)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :].cpu().numpy() 
                    all_features_batches.append(features)
        
        elif self.task_type == "video":
            for i in range(0, len(X), batch_size):
                batch_video_paths = X[i:i+batch_size]
                processed_clips_tensors = []
                if self.config["source"] == "transformers":
                    pixel_values_list = []
                    for video_path in batch_video_paths:
                        processed_output = self._preprocess_video_transformers(video_path)
                        pixel_values_list.append(processed_output['pixel_values'])
                    if not pixel_values_list: continue
                    batch_tensor = torch.cat(pixel_values_list, dim=0).to(self.device)
                else: # torchvision
                    for video_path in batch_video_paths:
                        clip_tensor = self._preprocess_video_torchvision(video_path) 
                        processed_clips_tensors.append(clip_tensor)
                    if not processed_clips_tensors: continue
                    batch_tensor = torch.cat(processed_clips_tensors, dim=0).to(self.device)
                
                outputs = self.model(pixel_values=batch_tensor) if self.config["source"] == "transformers" else self.model(batch_tensor)
                
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None: 
                    features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'): 
                    features = outputs.last_hidden_state.mean(dim=1) 
                else: 
                    features = outputs 
                all_features_batches.append(features.cpu().numpy())

        elif self.task_type == "audio":
            for i in range(0, len(X), batch_size):
                batch_audio_paths = X[i:i+batch_size]
                batch_item_features_list = []
                for audio_path in batch_audio_paths:
                    inputs = self._preprocess_audio(audio_path) 
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    item_features = outputs.last_hidden_state.mean(dim=1) 
                    batch_item_features_list.append(item_features)
                if batch_item_features_list:
                    stacked_features = torch.cat(batch_item_features_list, dim=0)
                    all_features_batches.append(stacked_features.cpu().numpy())

        elif self.task_type == "multimodal_image_text":
            img_feats_list, text_feats_list = [], []
            for i in range(0, len(X), batch_size):
                batch_tuples = X[i:i+batch_size]
                pil_images = [Image.open(item[0]).convert("RGB") if isinstance(item[0], str) else item[0].convert("RGB") for item in batch_tuples]
                str_texts = [item[1] for item in batch_tuples]
                inputs = self.processor(text=str_texts, images=pil_images, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if self.model_name.startswith("clip"):
                    outputs = self.model(**inputs)
                    img_feats_list.append(outputs.image_embeds.cpu().numpy())
                    text_feats_list.append(outputs.text_embeds.cpu().numpy())
                else: raise NotImplementedError 
            final_output_dict = {"image_features": np.concatenate(img_feats_list, axis=0) if img_feats_list else np.array([]),
                                 "text_features": np.concatenate(text_feats_list, axis=0) if text_feats_list else np.array([])}
            return final_output_dict


        elif self.task_type == "multimodal_video_text": # XCLIP
            vid_feats_list, txt_feats_list = [], []
            for i in range(0, len(X), batch_size): 
                batch_tuples = X[i:i+batch_size]
                batch_video_pil_frames = [self._preprocess_video_transformers(video_path) for video_path, _ in batch_tuples] 
                text_queries = [item[1] for item in batch_tuples]
                inputs = self.processor(text=text_queries, videos=batch_video_pil_frames, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs) 
                
                # Corrected XCLIP feature extraction:
                # Directly use the documented final embeddings from XCLIPModelOutput
                # These should be (batch_size, output_dim)
                video_f = outputs.video_embeds
                text_f = outputs.text_embeds
                
                # If by any chance they are still 3D (e.g., (batch, seq_len, dim)), reduce them.
                # This is a safeguard based on the previous error, though ideally not needed if docs are accurate.
                if video_f.ndim == 3 and video_f.shape[0] == len(batch_tuples): 
                    video_f = video_f.mean(dim=1) # Mean pool over video tokens/frames
                
                if text_f.ndim == 3 and text_f.shape[0] == len(batch_tuples): 
                    text_f = text_f[:, 0, :] # Take CLS token embedding

                vid_feats_list.append(video_f.cpu().numpy())
                txt_feats_list.append(text_f.cpu().numpy())   

            final_output_dict = {"video_features": np.concatenate(vid_feats_list, axis=0) if vid_feats_list else np.array([]),
                                 "text_features": np.concatenate(txt_feats_list, axis=0) if txt_feats_list else np.array([])}
            return final_output_dict

        else:
            raise ValueError(f"Transform not implemented for task type: {self.task_type}")

        if not all_features_batches: return np.array([])
        final_features_np = np.concatenate(all_features_batches, axis=0)
        return final_features_np

# --- Self-Test Suite ---
DUMMY_VIDEO_URL = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
DUMMY_VIDEO_FILENAME = "dlfeat_sample_video.mp4"

def _create_dummy_image(path="dummy_image_dlfeat.png"):
    try:
        img = Image.new('RGB', (224, 224), color = 'red')
        d = ImageDraw.Draw(img)
        d.text((10,10), "Hello DLFeat", fill=(0,0,0))
        img.save(path)
        return path
    except ImportError: warnings.warn("Pillow with ImageDraw not available to create dummy image for tests.")
    except Exception as e: warnings.warn(f"Could not create dummy image: {e}")
    return None

def _create_dummy_audio(path="dummy_audio_dlfeat.wav"):
    try:
        if not hasattr(scipy_wav, 'write') or scipy_wav.__name__.startswith('dummy_'): raise ImportError("Scipy.io.wavfile not available or is a dummy.")
        sample_rate = 16000; duration = 1; frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        data = np.sin(2 * np.pi * frequency * t) * 0.5
        data_int16 = (data * 32767).astype(np.int16)
        scipy_wav.write(path, sample_rate, data_int16)
        return path
    except ImportError: warnings.warn("Scipy.io.wavfile not available to create dummy audio for tests.")
    except Exception as e: warnings.warn(f"Could not create dummy audio: {e}")
    return None

def _create_dummy_video(path=DUMMY_VIDEO_FILENAME, download_real_video=True):
    if download_real_video and hasattr(requests, 'get'):
        try:
            response = requests.get(DUMMY_VIDEO_URL, stream=True, timeout=10)
            response.raise_for_status() 
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            return path, True 
        except requests.exceptions.RequestException as e: warnings.warn(f"Failed to download real sample video: {e}. Using placeholder.")
        except Exception as e: warnings.warn(f"An error occurred during video download or saving: {e}. Using placeholder.")
    try:
        with open(path, 'wb') as f: f.write(b"This is a placeholder for DLFeat video tests.")
        return path, False 
    except Exception as e: warnings.warn(f"Could not create placeholder dummy video file: {e}")
    return None, False


def run_self_tests(models_to_test='all', device='cpu', verbose=True, attempt_real_video_test=True):
    print("="*70 + f"\n DLFeat Self-Test Suite (v{__version__})\n" + "="*70)
    actual_models_to_test = list(MODEL_CONFIGS.keys()) if (isinstance(models_to_test, str) and models_to_test.lower() == 'all') else models_to_test
    if not isinstance(actual_models_to_test, list): actual_models_to_test = list(MODEL_CONFIGS.keys())
    print(f"Running tests for: {actual_models_to_test if models_to_test=='all' else models_to_test}")

    dummy_image_path = _create_dummy_image(); dummy_audio_path = _create_dummy_audio()
    dummy_video_file_path, is_real_video = _create_dummy_video(download_real_video=attempt_real_video_test)
    text_sample = ["This is a test sentence.", "Another sentence for testing."]
    test_results_list = []

    for model_name in actual_models_to_test:
        if model_name not in MODEL_CONFIGS:
            test_results_list.append({"model_name": model_name, "task": "N/A", "source": "N/A", "availability": "✗", "test_status": "SKIPPED", "notes": "Not in MODEL_CONFIGS"})
            continue
        config = MODEL_CONFIGS[model_name]; task = config["task"]; source = config["source"]
        current_result = {"model_name": model_name, "task": task, "source": source, "availability": "✗", "test_status": "SKIPPED", "notes": ""}
        notes_collector = []
        if verbose: print(f"\n--- Testing: {model_name} (Task: {task}, Source: {source}) ---")
        
        try:
            extractor = DLFeatExtractor(model_name=model_name, device=device)
            current_result["availability"] = "✓"
            dummy_input_data, expected_batch_size = None, 0
            
            video_related_task = task == "video" or task == "multimodal_video_text"
            skip_video_transform = video_related_task and (not (dummy_video_file_path and os.path.exists(dummy_video_file_path)) or not is_real_video)
            if video_related_task and not (dummy_video_file_path and os.path.exists(dummy_video_file_path)): notes_collector.append("No dummy video file.")
            elif video_related_task and not is_real_video: notes_collector.append("Using placeholder video.")
            
            if task == "image":
                if dummy_image_path and os.path.exists(dummy_image_path): dummy_input_data, expected_batch_size = [dummy_image_path]*2, 2
                else: notes_collector.append("No dummy image.")
            elif task == "text": dummy_input_data, expected_batch_size = text_sample, len(text_sample)
            elif task == "audio":
                if dummy_audio_path and os.path.exists(dummy_audio_path): dummy_input_data, expected_batch_size = [dummy_audio_path]*2, 2
                else: notes_collector.append("No dummy audio.")
            elif task == "video":
                if not skip_video_transform: dummy_input_data, expected_batch_size = [dummy_video_file_path]*2, 2
            elif task == "multimodal_image_text":
                if dummy_image_path and os.path.exists(dummy_image_path): dummy_input_data, expected_batch_size = [(dummy_image_path, text_sample[0]), (dummy_image_path, text_sample[1])], 2
                else: notes_collector.append("No dummy image for multimodal.")
            elif task == "multimodal_video_text":
                if not skip_video_transform: dummy_input_data, expected_batch_size = [(dummy_video_file_path, text_sample[0]), (dummy_video_file_path, text_sample[1])], 2

            if skip_video_transform and current_result["availability"] == "✓":
                current_result["test_status"] = "SKIPPED (Transform)"
            elif dummy_input_data:
                features = extractor.transform(dummy_input_data)
                if task.startswith("multimodal"):
                    if not isinstance(features, dict): raise AssertionError(f"Multimodal features not dict (got {type(features)})")
                    key_map = {"multimodal_image_text": ["image_features", "text_features"], "multimodal_video_text": ["video_features", "text_features"]}
                    if not all(k in features for k in key_map[task]): raise AssertionError(f"Missing keys in multimodal output. Expected {key_map[task]}, Got {list(features.keys())}")
                    for key_feat, val_feat in features.items():
                        if not isinstance(val_feat, np.ndarray): raise AssertionError(f"Feature '{key_feat}' not np.ndarray.")
                        expected_dim_mm = extractor.get_feature_dimension() 
                        if val_feat.shape != (expected_batch_size, expected_dim_mm): raise AssertionError(f"Feature '{key_feat}' shape mismatch. Got {val_feat.shape}, expected ({expected_batch_size}, {expected_dim_mm})")
                else:
                    if not isinstance(features, np.ndarray): raise AssertionError(f"Features not np.ndarray (got {type(features)})")
                    expected_dim_uni = extractor.get_feature_dimension()
                    if features.shape != (expected_batch_size, expected_dim_uni): raise AssertionError(f"Feature shape mismatch. Got {features.shape}, expected ({expected_batch_size}, {expected_dim_uni})")
                current_result["test_status"] = "PASSED"
            else: current_result["test_status"] = "SKIPPED (No data)"
        except ImportError as ie: 
            current_result.update({"availability": "✗ (Import)", "test_status": "FAILED"})
            notes_collector.append(f"{type(ie).__name__}: {textwrap.shorten(str(ie).splitlines()[0],60)}")
            if verbose: print(f"[{model_name}] FAILED (Initialization ImportError):\n{traceback.format_exc()}")
        except Exception as e: 
            current_result.update({"test_status": "FAILED (Runtime)" if current_result["availability"] == "✓" else "FAILED (Init)"})
            notes_collector.append(f"{type(e).__name__}: {textwrap.shorten(str(e).splitlines()[0],60)}")
            if verbose: print(f"[{model_name}] FAILED ({type(e).__name__}):\n{traceback.format_exc()}")
        
        current_result["notes"] = "; ".join(notes_collector) if notes_collector else current_result.get("notes", "") # Preserve existing notes from exceptions
        test_results_list.append(current_result)

    if dummy_image_path and os.path.exists(dummy_image_path): os.remove(dummy_image_path)
    if dummy_audio_path and os.path.exists(dummy_audio_path): os.remove(dummy_audio_path)
    if dummy_video_file_path and os.path.exists(dummy_video_file_path): os.remove(dummy_video_file_path) 
    
    print("\n\n" + "="*80 + "\n DLFeat Self-Test Summary Report".center(80) + "\n" + "="*80)
    cols = {"Model": 30, "Task": 20, "Source": 15, "Available": 10, "Test Status": 20, "Notes": 40}
    header = "| " + " | ".join([col_name.ljust(cols[col_name]) for col_name in cols]) + " |"
    separator = "|" + "".join(["-"*(cols[col_name]+2) + "|" for col_name in cols])
    print(separator + "\n" + header + "\n" + separator)
    for r in test_results_list:
        row = f"| {textwrap.shorten(r['model_name'], width=cols['Model']).ljust(cols['Model'])} " \
              f"| {r['task'].ljust(cols['Task'])} " \
              f"| {r['source'].ljust(cols['Source'])} " \
              f"| {r['availability'].center(cols['Available'])} " \
              f"| {r['test_status'].ljust(cols['Test Status'])} " \
              f"| {textwrap.shorten(r['notes'], width=cols['Notes']).ljust(cols['Notes'])} |"
        print(row)
    print(separator + "\n" + "="*80)
    return test_results_list

if __name__ == '__main__':
    results = run_self_tests(attempt_real_video_test=True, verbose=True) 
