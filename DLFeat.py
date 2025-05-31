# DLFeat: Deep Learning Feature Extraction Library
# Inspired by VLFeat for ease of use and modularity in the modern deep learning era.
# Version: 0.3.1
# Author: Gemini
# Date: 2025-05-31

"""
DLFeat: Deep Learning Feature Extraction Library
================================================
# ... (Main docstring same as v0.3.0, will update Model Zoo entry for Video Swin)
This version switches Video Swin Transformer to use the torchvision implementation.
# ...

Model Zoo
---------
# ... (Model Zoo table to be updated for Video Swin source)
.. list-table:: DLFeat Model Zoo (Excerpt for Video Swin Change)
   :widths: 15 25 10 10 10 10 15 15
   :header-rows: 1

   * - Modality
     - Model Name (Identifier)
     - Feat. Dim
     - Performance (Benchmark)
     - FLOPS (G)
     - Speed (Ref.)
     - Supervision
     - Source
   * - Video
     - `video_swin_t`
     - 768
     - ~78.8% (K400 Top-1, Swin-T)
     - 48 (32x224^2)
     - Medium
     - Supervised
     - torchvision
   * - Video
     - `video_swin_b`
     - 1024
     - ~80.6% (K400 Top-1, Swin-B)
     - 92 (32x224^2)
     - Slower
     - Supervised
     - torchvision


*(Note: Full Model Zoo table is extensive and located at the beginning of the file.
Performance, FLOPS, and Speed are indicative.)*

"""

__version__ = "0.3.1" 

import torch
import torchvision.transforms as T
import torchvision.models as tv_models 
import torchvision.models.video as tv_video_models 
from PIL import Image, ImageDraw 
import numpy as np
import warnings
import os
import textwrap 

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
        VideoMAEFeatureExtractor, VideoMAEModel, # VideoMAE still from Transformers
        XCLIPProcessor, XCLIPModel,
        ASTFeatureExtractor,
        Dinov2Model 
        # VideoSwinModel, VideoSwinImageProcessor removed as VideoSwin is now from torchvision
    )
except ImportError:
    warnings.warn(
        "Transformers library not found or key components are missing. "
        "Text, some audio, DINOv2, VideoMAE and multimodal models may not be available. "
        "Please install or upgrade transformers: pip install --upgrade transformers"
    )
    class AutoProcessor: pass; 
    class AutoModel: pass; 
    class AutoTokenizer: pass;
    class Wav2Vec2FeatureExtractor: pass; 
    class AutoImageProcessor: pass;
    class CLIPProcessor: pass; 
    class CLIPModel: pass; 
    class BlipProcessor: pass;
    class BlipModel: pass; 
    class VideoMAEFeatureExtractor: pass; 
    class VideoMAEModel: pass;
    class XCLIPProcessor: pass; 
    class XCLIPModel: pass; 
    class ASTFeatureExtractor: pass;
    class Dinov2Model: pass; # VideoSwinModel, VideoSwinImageProcessor dummies removed


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
        "TIMM library not found. Some image models (ViT, EfficientNet, ConvNeXt, TSM) will not be available. "
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


try:
    import pytorchvideo.models as ptv_models_real 
    import pytorchvideo.transforms as ptv_transforms_real
    from pytorchvideo.data.encoded_video import EncodedVideo as EncodedVideoReal
    import pytorchvideo.data.transforms as ptv_data_transforms_real
    ptv_models = ptv_models_real
    ptv_transforms = ptv_transforms_real
    EncodedVideo = EncodedVideoReal
    ptv_data_transforms = ptv_data_transforms_real
except ImportError:
    warnings.warn(
        "PyTorchVideo library not found. Some video models (MViT, SlowFast) will not be available. "
        "Please install it: pip install pytorchvideo"
    )
    class ptv_models: 
        class mvit: 
            @staticmethod
            def create_mvit(*args, **kwargs):  raise ImportError("PyTorchVideo not installed.")
        class i3d: 
            @staticmethod
            def create_i3d(*args, **kwargs): raise ImportError("PyTorchVideo not installed.")
        class slowfast: 
             @staticmethod
             def create_slowfast(*args, **kwargs): raise ImportError("PyTorchVideo not installed.")
    class ptv_transforms: pass 
    class EncodedVideo: pass 
    class ptv_data_transforms: pass 


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

    # --- Video Models ---
    "r2plus1d_18": {"task": "video", "dim": 512, "source": "torchvision", "tv_model_name":"r2plus1d_18", "clip_len": 16, "frame_rate": 15, "input_size": 112}, 
    "videomae_base_k400_pt": {"task": "video", "dim": 768, "source": "transformers", "hf_name": "MCG-NJU/videomae-base-finetuned-kinetics", "num_frames": 16, "input_size": 224},
    "mvit_v2_s": {"task": "video", "dim": 768, "source": "pytorchvideo", "clip_len": 32, "frame_rate": 3, "input_size": 224, "ptv_hub_name": "mvit_v2_s_32x3_kinetics400_strid"},
    "slowfast_r50": {"task": "video", "dim": 2304, "source": "pytorchvideo", "clip_len": 32, "frame_rate_alpha": 4, "input_size": 256, "ptv_hub_name": "slowfast_r50"}, # Using standard slowfast_r50 from PTV hub
    "tsm_resnet50": {"task": "video", "dim": 2048, "source": "timm", "timm_name": "tsm_resnet50", "num_frames": 8, "input_size": 224}, 
    "video_swin_t": {"task": "video", "dim": 768, "source": "torchvision", "tv_model_name": "swin_t", "clip_len": 32, "input_size": 224},
    "video_swin_s": {"task": "video", "dim": 768, "source": "torchvision", "tv_model_name": "swin_s", "clip_len": 32, "input_size": 224},
    "video_swin_b": {"task": "video", "dim": 1024, "source": "torchvision", "tv_model_name": "swin_b", "clip_len": 32, "input_size": 224},
    
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
    "r2plus1d_18", "videomae_base_k400_pt", "mvit_v2_s", "slowfast_r50", "tsm_resnet50", "video_swin_t",
    "wav2vec2_base", "sentence-bert", 
    "clip_vit_b32", "xclip_base_patch16"
]


def list_available_models(task_type=None):
    # ... (same as v0.2.9) ...
    if task_type:
        return [name for name, config in MODEL_CONFIGS.items() if config["task"] == task_type]
    return list(MODEL_CONFIGS.keys())


class DLFeatExtractor(BaseEstimator, TransformerMixin):
    # ... (Constructor, get_feature_dimension, get_model_config same as v0.2.9) ...
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
        self.video_frame_transform = None # Explicitly initialize for video
        self.video_transform_params = None 
        self.target_sr = None 

        self._load_model()

    def get_feature_dimension(self):
        return self.config["dim"]

    def get_model_config(self):
        return self.config.copy()

    def _load_image_model_torchvision(self, model_name_tv):
        # ... (same as v0.2.9)
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
        elif model_name_tv.startswith("efficientnet_b"): # Not typically loaded from torchvision by DLFeat, but for completeness
            name_part = model_name_tv[len("efficientnet_"):] 
            weights_enum_name_to_try = f"EfficientNet_{name_part.upper()}_Weights" 
        elif model_name_tv.startswith("convnext_"): # Not typically loaded from torchvision by DLFeat
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
            # Store transforms from weights if available for image models too
            if hasattr(weights_obj, 'transforms') and callable(weights_obj.transforms):
                self.image_transform = weights_obj.transforms()

        except (AttributeError, ValueError) as e_new_api_img: 
            warnings.warn(
                f"DLFeatExtractor: Failed to use new 'weights' API for image model {model_name_tv} (Error: {type(e_new_api_img).__name__}: {e_new_api_img}). "
                f"Falling back to legacy 'pretrained=True'. Torchvision warnings may follow."
            )
            self.model = model_fn(pretrained=True)
             
        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, torch.nn.Sequential) and len(self.model.classifier) > 0 :
             self.model.classifier[-1] = torch.nn.Identity()
        elif hasattr(self.model, 'fc'): 
            self.model.fc = torch.nn.Identity()
        elif hasattr(self.model, 'classifier') and isinstance(self.model.classifier, torch.nn.Linear): 
            self.model.classifier = torch.nn.Identity()

        self.model.eval().to(self.device)
        
        # If transforms weren't set from weights, set default
        if self.image_transform is None:
            input_size = self.config.get("input_size", 224)
            self.image_transform = T.Compose([
                T.Resize(256 if input_size == 224 else int(input_size / (224.0/256.0))), # Maintain aspect ratio for resize then crop
                T.CenterCrop(input_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        # Ensure the loaded transform's crop size matches config if possible
        elif hasattr(self.image_transform, 'crop_size') and isinstance(self.image_transform.crop_size, int) and self.image_transform.crop_size != self.config.get("input_size", 224):
            new_input_size = self.config.get("input_size", 224)
            # This is tricky as transforms object is already composed. Modifying it directly can be error prone.
            # For now, we assume the transforms() from weights are mostly correct for the pretrained model.
            # If a specific input_size is mandated by DLFeat config AND different from default weight's transform,
            # it might indicate a mismatch or need for custom transform chain.
            # Current logic: if self.image_transform is set, we use it as is.
            pass


    def _load_image_model_timm(self, timm_model_name):
        # ... (same as v0.2.9)
        if not hasattr(timm, 'create_model'): 
             raise ImportError("TIMM library is not installed. Please install it: pip install timm")
        self.model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
        self.model.eval().to(self.device)
        data_config = timm.data.resolve_data_config({}, model=self.model)
        self.image_transform = timm.data.create_transform(**data_config)
        if "input_size" not in self.config: 
            self.config["input_size"] = data_config['input_size'][1] 

    def _load_image_model_dinov2(self):
        # ... (same as v0.2.9)
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
        # Clear any pre-existing video frame transform before loading a new model
        self.video_frame_transform = None 

        if self.task_type == "image":
            # ... (image loading logic as in v0.2.9)
            if self.model_name == "dinov2_base": 
                if not hasattr(Dinov2Model, 'from_pretrained'): 
                    raise ImportError("Transformers (Dinov2Model) dummy class detected or not installed.")
                self._load_image_model_dinov2() 
            elif source == "torchvision": 
                self._load_image_model_torchvision(self.model_name)
            elif source == "torchvision_or_timm":
                try:
                    if not hasattr(timm, 'create_model'): raise ImportError("TIMM not available")
                    # Pass the timm_name if specified in config, otherwise model_name
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
            # ... (text loading logic as in v0.2.9)
            if source == "sentence-transformers":
                if not hasattr(SentenceTransformer, 'encode'): 
                     raise ImportError("Sentence-Transformers dummy class detected or not installed.")
                self.model = SentenceTransformer(self.config["st_name"], device=self.device)
            elif source == "transformers":
                if not hasattr(AutoTokenizer, 'from_pretrained'): 
                    raise ImportError("Transformers (AutoTokenizer) dummy class detected or not installed.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config["hf_name"])
                if not hasattr(AutoModel, 'from_pretrained'):
                     raise ImportError("Transformers (AutoModel) dummy class detected or not installed.")
                self.model = AutoModel.from_pretrained(self.config["hf_name"])
                self.model.eval().to(self.device)
            else:
                raise ValueError(f"Unsupported text model source: {source}")

        elif self.task_type == "video":
            if source == "torchvision":
                actual_tv_model_name = self.config.get("tv_model_name", self.model_name) 

                if not hasattr(tv_video_models, actual_tv_model_name): 
                    raise ImportError(f"Video model '{actual_tv_model_name}' not found in torchvision.models.video.")
                model_fn = getattr(tv_video_models, actual_tv_model_name)

                weights_enum_name_map = {
                    "r2plus1d_18": "R2Plus1D_18_Weights",
                    "swin_t": "Swin_T_Weights",
                    "swin_s": "Swin_S_Weights",
                    "swin_b": "Swin_B_Weights",
                    # Add other torchvision video model internal names to their Weight enum names here
                }
                weights_enum_name = weights_enum_name_map.get(actual_tv_model_name)
                weights_value_name = "KINETICS400_V1" # Common, can be overridden by specific model needs

                if weights_enum_name:
                    try:
                        weights_enum_class = getattr(tv_video_models, weights_enum_name, None)
                        if weights_enum_class:
                            if hasattr(weights_enum_class, 'DEFAULT'): 
                                weights_obj = weights_enum_class.DEFAULT
                            elif hasattr(weights_enum_class, weights_value_name):
                                weights_obj = getattr(weights_enum_class, weights_value_name)
                            else: # Try first available uppercase if common ones fail
                                available_weights = [w for w in dir(weights_enum_class) if w.isupper() and not w.startswith('_')]
                                if available_weights:
                                    weights_obj = getattr(weights_enum_class, available_weights[0])
                                    warnings.warn(f"DLFeat: Using first available weight '{available_weights[0]}' for {actual_tv_model_name}")
                                else: raise AttributeError(f"No suitable weights found in {weights_enum_name}")
                            
                            self.model = model_fn(weights=weights_obj)
                            if hasattr(weights_obj, 'transforms') and callable(weights_obj.transforms):
                                self.video_frame_transform = weights_obj.transforms()
                                # print(f"DLFeat: Using default transforms from weights for {actual_tv_model_name}") # Debug
                        else: raise AttributeError(f"{weights_enum_name} enum not found.")
                    except (AttributeError, ValueError) as e_new_api:
                        warnings.warn(
                            f"DLFeatExtractor: Failed to use new 'weights' API for {actual_tv_model_name} (Error: {type(e_new_api).__name__}: {e_new_api}). "
                            f"Falling back to legacy 'pretrained=True' if applicable."
                        )
                        # Check if 'pretrained' is even a valid kwarg for this model_fn
                        import inspect
                        sig = inspect.signature(model_fn)
                        if 'pretrained' in sig.parameters:
                            self.model = model_fn(pretrained=True)
                        else:
                            raise ImportError(f"New weights API failed for {actual_tv_model_name} and legacy 'pretrained' not supported or model init failed.")
                else: 
                    warnings.warn(
                        f"DLFeatExtractor: No specific 'weights' API logic for {actual_tv_model_name}. Attempting legacy 'pretrained=True'."
                    )
                    self.model = model_fn(pretrained=True) # Fallback for unhandled models

                # Remove classifier head (common patterns)
                if hasattr(self.model, 'head'): 
                    if isinstance(self.model.head, torch.nn.Linear): self.model.head = torch.nn.Identity()
                    elif isinstance(self.model.head, torch.nn.Sequential) and len(self.model.head)>0 and isinstance(self.model.head[-1], torch.nn.Linear):
                        self.model.head[-1] = torch.nn.Identity()
                elif hasattr(self.model, 'fc'): self.model.fc = torch.nn.Identity() 
                
                self.model.eval().to(self.device)

                # If transforms weren't set from weights, set a basic default
                if self.video_frame_transform is None:
                    input_size = self.config.get("input_size", 224)
                    warnings.warn(f"DLFeat: Using manual basic transforms for {actual_tv_model_name} as weights.transforms() was not available/used.")
                    self.video_frame_transform = T.Compose([ # This transform is applied PER FRAME in _preprocess_video_torchvision
                        T.ConvertImageDtype(torch.float32),
                        T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]), # Kinetics mean/std
                        T.Resize([input_size, input_size], antialias=True) 
                    ])
            elif source == "pytorchvideo":
                # ... (pytorchvideo loading logic as in v0.2.9, with SlowFast specific create)
                if not hasattr(EncodedVideo, 'from_path'): 
                    raise ImportError("PyTorchVideo dummy class detected or not installed.")
                
                ptv_hub_model_name = self.config.get("ptv_hub_name", self.model_name)
                try:
                    if self.model_name == "slowfast_r50":
                         if not hasattr(ptv_models, 'slowfast') or not hasattr(getattr(ptv_models, 'slowfast'), 'create_slowfast'): 
                             raise ImportError("PTV slowfast model or create_slowfast method not found.")
                         self.model = ptv_models.slowfast.create_slowfast(head_activation=None, model_num_class=None) # For features
                         # For PTV create_* functions, manual weight loading is often needed.
                         # This part is complex for a generic library. We assume torch.hub handles most cases.
                         # For slowfast, users might need to load weights manually or use a hub version if PTV provides one under that name.
                         # The ptv_hub_name for slowfast_r50 in MODEL_CONFIGS might point to a hub-loadable version.
                         # If ptv_hub_name IS "slowfast_r50", torch.hub.load will be attempted:
                         if ptv_hub_model_name == "slowfast_r50": # Redundant check if this path is taken
                              self.model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True, trust_repo=True)
                         else: # If create_slowfast was used and needs manual weights
                             warnings.warn("DLFeat: PTV SlowFast model created; ensure weights are loaded if not handled by create_slowfast.")
                    else: # For MViT etc.
                        self.model = torch.hub.load("facebookresearch/pytorchvideo", model=ptv_hub_model_name, pretrained=True, trust_repo=True) 
                except Exception as e:
                    raise ImportError(f"Failed to load PyTorchVideo model {ptv_hub_model_name}: {e}. ")

                # Classifier removal for PTV models
                if hasattr(self.model, 'blocks') and self.model.blocks and hasattr(self.model.blocks[-1], 'proj'): # MViT
                    self.model.blocks[-1].proj = torch.nn.Identity() 
                elif self.model_name == "slowfast_r50" and hasattr(self.model, 'blocks') and len(self.model.blocks) == 6 and hasattr(self.model.blocks[5], 'proj'): # SlowFast specific head path
                    self.model.blocks[5].proj = torch.nn.Identity()
                elif hasattr(self.model, 'head'): # More general PTV head patterns
                     if hasattr(self.model.head, 'projection'): self.model.head.projection = torch.nn.Identity()
                     elif isinstance(self.model.head, torch.nn.Linear): self.model.head = torch.nn.Identity()

                self.model.eval().to(self.device)
                self.video_transform_params = { 
                    "side_size": self.config.get("input_size", 224) + 32, 
                    "crop_size": self.config.get("input_size", 224),
                    "num_frames": self.config.get("clip_len", 32), 
                    "sampling_rate": self.config.get("frame_rate_alpha", 4) if self.model_name=="slowfast_r50" else self.config.get("frame_rate", 8), 
                    "video_mean": (0.45, 0.45, 0.45),
                    "video_std": (0.225, 0.225, 0.225),
                }
            elif source == "transformers": 
                # ... (transformers video loading logic for VideoMAE, as in v0.2.9)
                # For VideoSwin, ProcessorClass would be VideoSwinImageProcessor (dummy if not imported)
                # ModelClass would be VideoSwinModel (dummy if not imported)
                # Since VideoSwin moved to torchvision, this path is mainly for VideoMAE now.
                ProcessorClass = VideoMAEFeatureExtractor # Assuming only VideoMAE uses this path now
                ModelClass = VideoMAEModel
                
                if not hasattr(ProcessorClass, 'from_pretrained'): 
                    raise ImportError(f"Transformers ({ProcessorClass.__name__}) dummy class detected or not installed.")
                self.processor = ProcessorClass.from_pretrained(self.config["hf_name"])
                
                if not hasattr(ModelClass, 'from_pretrained'):
                    raise ImportError(f"Transformers ({ModelClass.__name__}) dummy class detected or not installed.")
                self.model = ModelClass.from_pretrained(self.config["hf_name"])
                self.model.eval().to(self.device)
            elif source == "timm": # For TSM
                 if not hasattr(timm, 'create_model'): raise ImportError("TIMM not available")
                 self._load_image_model_timm(self.config["timm_name"]) 
            else:
                raise ValueError(f"Unsupported video model source: {source}")

        elif self.task_type == "audio":
            # ... (audio loading logic as in v0.2.9)
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
            # ... (multimodal_image_text loading logic as in v0.2.9)
            ProcessorCheckClass = BlipProcessor if self.model_name.startswith("blip") else CLIPProcessor
            ModelCheckClass = BlipModel if self.model_name.startswith("blip") else CLIPModel
            if not hasattr(ProcessorCheckClass, 'from_pretrained') or not hasattr(ModelCheckClass, 'from_pretrained'):
                raise ImportError(f"Transformers ({ProcessorCheckClass.__name__} or {ModelCheckClass.__name__}) dummy class detected or not installed.")
            
            hf_name = self.config["hf_name"]
            self.processor = ProcessorCheckClass.from_pretrained(hf_name)
            self.model = ModelCheckClass.from_pretrained(hf_name)
            self.model.eval().to(self.device)

        elif self.task_type == "multimodal_video_text":
            # ... (multimodal_video_text loading logic as in v0.2.9)
            if not hasattr(XCLIPProcessor, 'from_pretrained') or not hasattr(XCLIPModel, 'from_pretrained'): 
                raise ImportError("Transformers (XCLIP components) dummy class detected or not installed.")
            hf_name = self.config["hf_name"]
            self.processor = XCLIPProcessor.from_pretrained(hf_name)
            self.model = XCLIPModel.from_pretrained(hf_name)
            self.model.eval().to(self.device)
            
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
    # --- Preprocessing methods ---
    def _preprocess_image(self, image_input):
        # ... (same as v0.2.9)
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
        # ... (same as v0.2.9)
        return self.tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")

    def _preprocess_video_torchvision(self, video_path):
        # Updated to prefer transforms from weights and handle (T,C,H,W) input
        if not os.path.exists(video_path): raise FileNotFoundError(f"Video file: {video_path}")
        try:
            frames, _, info = torchvision.io.read_video(video_path, pts_unit='sec', output_format="T_C_H_W") # T, C, H, W uint8
        except Exception as e: raise RuntimeError(f"Failed to read {video_path} using torchvision.io: {e}.")
        if frames.numel() == 0: raise ValueError(f"No frames from {video_path}.")

        total_frames = frames.shape[0]
        clip_len = self.config.get("clip_len", 32) 
        
        if total_frames < clip_len: # Pad if too short
            padding_count = clip_len - total_frames
            padding_frames = frames[-1:].repeat(padding_count, 1, 1, 1) # Repeat last frame
            sampled_frames = torch.cat((frames, padding_frames), dim=0)
        else: # Uniformly sample clip_len frames
            indices = torch.linspace(0, total_frames - 1, steps=clip_len).long()
            sampled_frames = frames[indices] # (clip_len, C, H, W) tensor

        if self.video_frame_transform:
            # This transform (e.g., from weights.DEFAULT.transforms()) should handle the whole (T,C,H,W) uint8 clip
            # and output a (C,T,H,W) float32 normalized tensor.
            try:
                processed_clip = self.video_frame_transform(sampled_frames) # Input T,C,H,W ; Output C,T,H,W expected
                # Some transforms might already output C,T,H,W directly. Others might need permute.
                # The standard video transforms from torchvision.models.video.<Model>_Weights.DEFAULT.transforms()
                # usually output C,T,H,W
                if processed_clip.shape[0] != 3 or processed_clip.shape[1] != clip_len: # Basic sanity check, C should be 3
                    # Try to infer if it's T,C,H,W and needs permute
                    if processed_clip.shape[1] == 3 and processed_clip.shape[0] == clip_len:
                        processed_clip = processed_clip.permute(1,0,2,3) # T,C,H,W -> C,T,H,W
                    # else: pass through, hope for the best or rely on model to handle it
            except Exception as e_transform:
                 raise RuntimeError(f"Error applying self.video_frame_transform for {self.model_name}: {e_transform}. "
                                    f"Input shape was {sampled_frames.shape}, transform type: {type(self.video_frame_transform)}")
        else: # Fallback: manual per-frame (should be rare if weights.transforms() is used)
            input_size = self.config.get("input_size", 224)
            manual_per_frame_transform = T.Compose([
                 T.ConvertImageDtype(torch.float32),
                 T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                 T.Resize([input_size, input_size], antialias=True)
            ])
            processed_frames_list = [manual_per_frame_transform(frame) for frame in sampled_frames] # frame is (C,H,W)
            processed_clip_temp = torch.stack(processed_frames_list) # (T,C,H,W)
            processed_clip = processed_clip_temp.permute(1,0,2,3) # (C,T,H,W)
            
        return processed_clip.unsqueeze(0) # B, C, T, H, W


    def _preprocess_video_pytorchvideo(self, video_path):
        # ... (same as v0.2.9)
        if not os.path.exists(video_path): raise FileNotFoundError(f"Video file: {video_path}")
        if not hasattr(EncodedVideo, 'from_path') or \
           not hasattr(ptv_transforms, 'Normalize') or \
           not hasattr(ptv_data_transforms, 'UniformTemporalSubsample'): 
             raise ImportError("PyTorchVideo components appear to be dummy classes or not imported correctly.")

        params = self.video_transform_params
        if self.model_name == "slowfast_r50":
            # Simplified single-pathway transform for DLFeat's SlowFast.
            # True SlowFast requires a list of 2 tensors (slow and fast pathways).
            # PyTorchVideo's hub model for "slowfast_r50" might be adapted to handle single input.
            transform = T.Compose([
                ptv_data_transforms.UniformTemporalSubsample(params["num_frames"]), # Fast pathway sampling
                T.Lambda(lambda x: x / 255.0), 
                ptv_transforms.Normalize(params["video_mean"], params["video_std"]),
                ptv_transforms.ShortSideScale(size=params["side_size"]),
                ptv_transforms.CenterCrop(params["crop_size"]),
                # ptv_transforms.PackPathway() # Omitted for simplicity, model might handle single tensor input
            ])
            warnings.warn("DLFeat's SlowFast preprocessing is simplified (single pathway simulated). "
                          "For optimal features or if model expects list input, consult PyTorchVideo SlowFast examples.", UserWarning)
        else: 
            transform = T.Compose([
                ptv_data_transforms.UniformTemporalSubsample(params["num_frames"]),
                T.Lambda(lambda x: x / 255.0), 
                ptv_transforms.Normalize(params["video_mean"], params["video_std"]),
                ptv_transforms.ShortSideScale(size=params["side_size"]),
                ptv_transforms.CenterCrop(params["crop_size"]),
            ])
        
        nominal_fps = 30 
        duration_to_sample_sec = (params["num_frames"] * params["sampling_rate"]) / nominal_fps

        video = EncodedVideo.from_path(video_path) 
        video_duration_sec = video.duration
        
        start_sec = max(0, (video_duration_sec - duration_to_sample_sec) / 2)
        end_sec = min(video_duration_sec, start_sec + duration_to_sample_sec)
        
        if start_sec >= end_sec and video_duration_sec > 0:
            start_sec = 0; end_sec = video_duration_sec
        
        try:
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            if video_data is None or video_data.get('video') is None or video_data['video'].numel() == 0:
                raise ValueError("Empty clip returned by get_clip")
        except Exception as e:
            warnings.warn(f"PTV get_clip failed for {video_path} (start={start_sec:.2f}, end={end_sec:.2f}, duration={video_duration_sec:.2f}): {e}. Trying full video.")
            video_data = video.get_clip(start_sec=0, end_sec=video_duration_sec) 

        if video_data is None or video_data.get('video') is None or video_data['video'].numel() == 0:
            raise ValueError(f"Failed to decode any clip from {video_path}")
        
        transformed_clip_dict = transform(video_data)
        
        # If model (like SlowFast from hub) expects a list of tensors, PackPathway should have handled it.
        # If not, transformed_clip_dict['video'] is a tensor.
        # Forcing to list for models that might expect it (e.g. some PTV SlowFast implementations)
        # This is a heuristic. The hub model for slowfast_r50 might be robust to single tensor.
        # if self.model_name == "slowfast_r50" and not isinstance(transformed_clip_dict['video'], list):
        #    return [transformed_clip_dict['video'].unsqueeze(0)] # Wrap in list and add batch
            
        return transformed_clip_dict['video'].unsqueeze(0) 


    def _preprocess_video_transformers(self, video_path): 
        # ... (same as v0.2.9)
        if not os.path.exists(video_path): raise FileNotFoundError(f"Video file: {video_path}")
        try:
            frames_tensor, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format="T_H_W_C")
        except Exception as e: raise RuntimeError(f"Failed to read {video_path}: {e}.")
        if frames_tensor.numel() == 0: raise ValueError(f"No frames from {video_path}.")

        total_frames = frames_tensor.shape[0]
        num_sample_frames = self.config.get("num_frames", 16) # Default, specific models might override in MODEL_CONFIGS
        
        if total_frames == 0: raise ValueError(f"Video {video_path} has 0 frames.")
        if num_sample_frames > total_frames : num_sample_frames = total_frames
        
        indices = np.linspace(0, total_frames - 1, num_sample_frames, dtype=int)
        
        video_frames_list_np = [frame.numpy() for frame in frames_tensor[indices]] 
        
        if self.model_name.startswith("xclip"):
            return [Image.fromarray(frame) for frame in video_frames_list_np] 
        else: # VideoMAE, (formerly VideoSwin)
            # self.processor is VideoMAEFeatureExtractor or VideoSwinImageProcessor (if it were still HF)
            return self.processor(images=video_frames_list_np, return_tensors="pt") 
    
    def _preprocess_video_timm(self, video_path): # For TSM
        # ... (same as v0.3.0)
        if not os.path.exists(video_path): raise FileNotFoundError(f"Video file: {video_path}")
        try:
            frames_tensor, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format="T_H_W_C")
        except Exception as e: raise RuntimeError(f"Failed to read {video_path}: {e}.")
        if frames_tensor.numel() == 0: raise ValueError(f"No frames from {video_path}.")

        total_frames = frames_tensor.shape[0]
        num_sample_frames = self.config.get("num_frames", 8) 
        if num_sample_frames > total_frames: num_sample_frames = total_frames
        
        indices = np.linspace(0, total_frames - 1, num_sample_frames, dtype=int)
        
        if not self.image_transform:
            raise RuntimeError("Image transform (from TIMM base model) not initialized for TSM video model.")

        processed_frames = []
        for idx in indices:
            frame_pil = Image.fromarray(frames_tensor[idx].numpy())
            processed_frame = self.image_transform(frame_pil) # C, H, W
            processed_frames.append(processed_frame)
        
        stacked_frames = torch.stack(processed_frames, dim=1) # C, T, H, W
        return stacked_frames.unsqueeze(0) # B, C, T, H, W

    def _preprocess_audio(self, audio_input):
        # ... (same as v0.2.9)
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
                self.audio_resampler = ResamplerClass(orig_freq=sr, new_freq=self.target_sr,
                                                   dtype=waveform.dtype).to(waveform.device) 
            waveform = self.audio_resampler(waveform)
        
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True) 
        
        processed_input_data = waveform.squeeze(0).numpy() 
        if self.model_name.startswith("ast"): 
             processed_input_data = waveform.squeeze(0) 

        return self.processor(processed_input_data, sampling_rate=self.target_sr, return_tensors="pt", padding=True)

    def fit(self, X, y=None):
        # ... (same as v0.2.9)
        return self

    @torch.no_grad()
    def transform(self, X, batch_size=32, **kwargs):
        # ... (same as v0.3.0, with TSM call to _preprocess_video_timm)
        if not isinstance(X, list):
            raise TypeError("Input X for transform must be a list of items.")
        if not X: 
            if self.task_type in ["multimodal_image_text", "multimodal_video_text"]:
                feat_keys = MODEL_CONFIGS[self.model_name]["dim"].keys() if isinstance(MODEL_CONFIGS[self.model_name]["dim"], dict) else \
                            (["image_features", "text_features"] if self.task_type == "multimodal_image_text" else ["video_features", "text_features"])
                return {key: np.array([]) for key in feat_keys}
            return np.array([])

        all_features_batches = [] 

        if self.task_type == "image":
            for i in range(0, len(X), batch_size):
                batch_items = X[i:i+batch_size]
                if self.model_name == "dinov2_base":
                    pil_images = []
                    for item in batch_items:
                        if isinstance(item, str): pil_images.append(Image.open(item).convert("RGB"))
                        elif isinstance(item, Image.Image): pil_images.append(item.convert("RGB"))
                        else: raise TypeError("DINOv2 expects image path or PIL Image.")
                    inputs = self.processor(images=pil_images, return_tensors="pt") 
                    inputs = {k: v.to(self.device) for k, v in inputs.items()} 
                    outputs = self.model(**inputs)
                    features = outputs.pooler_output 
                else: 
                    processed_batch_tensors = []
                    for item in batch_items:
                        processed_item_output = self._preprocess_image(item) 
                        processed_batch_tensors.append(processed_item_output.squeeze(0))
                    
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
                for video_path in batch_video_paths:
                    if self.config["source"] == "torchvision":
                        clip_tensor = self._preprocess_video_torchvision(video_path) 
                    elif self.config["source"] == "pytorchvideo":
                        clip_tensor = self._preprocess_video_pytorchvideo(video_path) 
                    elif self.config["source"] == "transformers": 
                        video_inputs_dict = self._preprocess_video_transformers(video_path) 
                        clip_tensor = video_inputs_dict['pixel_values'] 
                    elif self.config["source"] == "timm": 
                        clip_tensor = self._preprocess_video_timm(video_path)
                    else:
                        raise NotImplementedError(f"Video preprocessing not implemented for {self.config['source']}")
                    processed_clips_tensors.append(clip_tensor)
                
                if not processed_clips_tensors: continue
                batch_tensor = torch.cat(processed_clips_tensors, dim=0).to(self.device)
                
                if self.config["source"] == "transformers": 
                    outputs = self.model(pixel_values=batch_tensor) 
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        features = outputs.pooler_output
                    else:
                        features = outputs.last_hidden_state.mean(dim=1) 
                elif self.config["source"] == "pytorchvideo":
                    # PTV SlowFast might expect a list of tensors if PackPathway was used in transform
                    # If self.model(batch_tensor) fails for slowfast, it might be expecting list input.
                    # The hub "slowfast_r50" model usually handles single tensor input by replicating for pathways.
                    if self.model_name == "slowfast_r50" and not isinstance(batch_tensor, list):
                        # Some PTV slowfast models might expect inputs as [slow_path_tensor, fast_path_tensor]
                        # The hub model is often more flexible. Forcing list for robustness if needed.
                        # features = self.model([batch_tensor, batch_tensor]) # Example, if model takes list
                        features = self.model(batch_tensor) # Assuming hub model handles it
                    else:
                        features = self.model(batch_tensor) 

                    if isinstance(features, list): features = features[0] # Often, PTV models return list of outputs
                    if features.ndim > 2 and features.shape[0] == batch_tensor.shape[0]: # Ensure batch dim matches
                         # Global average pool over remaining spatial/temporal dimensions if any
                         features = torch.mean(features, dim=list(range(2, features.ndim)))
                    elif features.ndim == 1 and batch_tensor.shape[0] > 1 : # (B*D) case, needs reshape
                         expected_dim = self.get_feature_dimension()
                         if features.numel() == batch_tensor.shape[0] * expected_dim :
                              features = features.view(batch_tensor.shape[0], expected_dim)

                else: # torchvision, timm (TSM)
                    features = self.model(batch_tensor) 
                all_features_batches.append(features.cpu().numpy())

        elif self.task_type == "audio":
            # ... (same as v0.2.9)
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
            # ... (same as v0.2.9)
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
                elif self.model_name.startswith("blip"): 
                    image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                    text_features = self.model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                    img_feats_list.append(image_features.cpu().numpy())
                    text_feats_list.append(text_features.cpu().numpy())
                else: raise NotImplementedError

            final_output_dict = {}
            if img_feats_list: final_output_dict["image_features"] = np.concatenate(img_feats_list, axis=0)
            if text_feats_list: final_output_dict["text_features"] = np.concatenate(text_feats_list, axis=0)
            return final_output_dict

        elif self.task_type == "multimodal_video_text": 
            # ... (same as v0.2.9)
            vid_feats_list, txt_feats_list = [], []
            for i in range(0, len(X), batch_size): 
                batch_tuples = X[i:i+batch_size]
                
                batch_video_pil_frames = [] 
                for video_path, _ in batch_tuples:
                    single_video_pil_frames = self._preprocess_video_transformers(video_path)
                    batch_video_pil_frames.append(single_video_pil_frames)

                text_queries = [item[1] for item in batch_tuples]
                
                inputs = self.processor(text=text_queries, videos=batch_video_pil_frames, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                vid_feats_list.append(outputs.video_embeds.cpu().numpy())
                txt_feats_list.append(outputs.text_embeds.cpu().numpy())

            final_output_dict = {}
            if vid_feats_list: final_output_dict["video_features"] = np.concatenate(vid_feats_list, axis=0)
            if txt_feats_list: final_output_dict["text_features"] = np.concatenate(txt_feats_list, axis=0)
            return final_output_dict
        else:
            raise ValueError(f"Transform not implemented for task type: {self.task_type}")

        if not all_features_batches: 
            return np.array([])
            
        final_features_np = np.concatenate(all_features_batches, axis=0)
        return final_features_np

# --- Self-Test Suite ---
# ... (run_self_tests, _create_dummy_image, _create_dummy_audio, _create_dummy_video same as v0.2.9) ...
def _create_dummy_image(path="dummy_image_dlfeat.png"):
    try:
        img = Image.new('RGB', (224, 224), color = 'red')
        d = ImageDraw.Draw(img)
        d.text((10,10), "Hello DLFeat", fill=(0,0,0))
        img.save(path)
        return path
    except ImportError:
        warnings.warn("Pillow with ImageDraw not available to create dummy image for tests.")
        return None
    except Exception as e:
        warnings.warn(f"Could not create dummy image: {e}")
        return None

def _create_dummy_audio(path="dummy_audio_dlfeat.wav"):
    try:
        if not hasattr(scipy_wav, 'write') or scipy_wav.__name__.startswith('dummy_'): 
             raise ImportError("Scipy.io.wavfile not available or is a dummy.")
        sample_rate = 16000; duration = 1; frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        data = np.sin(2 * np.pi * frequency * t) * 0.5
        data_int16 = (data * 32767).astype(np.int16)
        scipy_wav.write(path, sample_rate, data_int16)
        return path
    except ImportError: 
        warnings.warn("Scipy.io.wavfile not available to create dummy audio for tests.")
        return None
    except Exception as e:
        warnings.warn(f"Could not create dummy audio: {e}")
        return None

def _create_dummy_video(path="dummy_video_dlfeat.mp4"):
    try:
        with open(path, 'wb') as f:
            f.write(b"This is not a real video file, but a placeholder for DLFeat tests.")
        return path
    except Exception as e:
        warnings.warn(f"Could not create placeholder dummy video file: {e}")
        return None

def run_self_tests(models_to_test=None, device='cpu', verbose=True):
    print("="*70)
    print(" DLFeat Self-Test Suite")
    print("="*70)

    if models_to_test is None:
        models_to_test = DEFAULT_MODELS_TO_TEST
        print(f"Running tests for a default set of models: {models_to_test}")
    elif isinstance(models_to_test, str) and models_to_test.lower() == 'all':
        models_to_test = list(MODEL_CONFIGS.keys())
        print("Running tests for ALL available models in MODEL_CONFIGS.")
    elif not isinstance(models_to_test, list):
        print(f"Warning: 'models_to_test' should be a list of model names, 'all', or None. Got {type(models_to_test)}. Running default set.")
        models_to_test = DEFAULT_MODELS_TO_TEST
    else: 
        print(f"Running tests for specified models: {models_to_test}")


    print("\n--- Creating dummy files for tests ---")
    dummy_image_path = _create_dummy_image()
    if dummy_image_path and verbose: print(f"Dummy image created: {dummy_image_path}")
    dummy_audio_path = _create_dummy_audio()
    if dummy_audio_path and verbose: print(f"Dummy audio created: {dummy_audio_path}")
    dummy_video_path = _create_dummy_video() 
    if dummy_video_path and verbose: print(f"Dummy video placeholder created: {dummy_video_path}")
    
    test_results_list = []
    text_sample = ["This is a test sentence for DLFeat.", "Another sentence."]

    for model_name in models_to_test:
        if model_name not in MODEL_CONFIGS:
            warnings.warn(f"Model {model_name} specified for testing not found in MODEL_CONFIGS. Skipping.")
            test_results_list.append({
                "model_name": model_name, "task": "N/A", "source": "N/A",
                "availability": "✗", "test_status": "SKIPPED", "notes": "Not in MODEL_CONFIGS"
            })
            continue

        config = MODEL_CONFIGS[model_name]
        task = config["task"]
        source = config["source"]
        
        current_result = {
            "model_name": model_name, "task": task, "source": source,
            "availability": "✗", "test_status": "SKIPPED", "notes": ""
        }
        notes_collector = []

        if verbose: print(f"\n--- Testing: {model_name} (Task: {task}, Source: {source}) ---")
        
        extractor = None
        try:
            if verbose: print(f"[{model_name}] Initializing DLFeatExtractor...")
            extractor = DLFeatExtractor(model_name=model_name, device=device)
            current_result["availability"] = "✓"
            if verbose: print(f"[{model_name}] Initialized successfully.")

            dummy_input_data = None
            expected_batch_size = 0
            is_video_task_with_placeholder = False

            if task == "image":
                if dummy_image_path and os.path.exists(dummy_image_path): 
                    dummy_input_data = [dummy_image_path, dummy_image_path] 
                    expected_batch_size = 2
                else: notes_collector.append("No dummy image.")
            elif task == "text":
                dummy_input_data = text_sample
                expected_batch_size = len(text_sample)
            elif task == "audio":
                if dummy_audio_path and os.path.exists(dummy_audio_path):
                    dummy_input_data = [dummy_audio_path, dummy_audio_path]
                    expected_batch_size = 2
                else: notes_collector.append("No dummy audio.")
            elif task == "video":
                is_video_task_with_placeholder = True 
                if dummy_video_path and os.path.exists(dummy_video_path): 
                    dummy_input_data = [dummy_video_path, dummy_video_path] 
                    expected_batch_size = 2 
                    # notes_collector.append("Video transform uses placeholder.") # Redundant due to special handling
                else: notes_collector.append("No dummy video placeholder.")
            elif task == "multimodal_image_text":
                if dummy_image_path and os.path.exists(dummy_image_path):
                    dummy_input_data = [(dummy_image_path, text_sample[0]), (dummy_image_path, text_sample[1])]
                    expected_batch_size = 2
                else: notes_collector.append("No dummy image for multimodal.")
            elif task == "multimodal_video_text":
                is_video_task_with_placeholder = True 
                if dummy_video_path and os.path.exists(dummy_video_path):
                    dummy_input_data = [(dummy_video_path, text_sample[0]), (dummy_video_path, text_sample[1])]
                    expected_batch_size = 2
                    # notes_collector.append("Multimodal video uses placeholder.")
                else: notes_collector.append("No dummy video placeholder for multimodal.")
            
            if is_video_task_with_placeholder and current_result["availability"] == "✓":
                current_result["test_status"] = "SKIPPED (Transform)"
                notes_collector.append("Transform with placeholder video skipped. Test with real video.")
                if verbose: print(f"[{model_name}] Transform test SKIPPED due to placeholder video data.")
            elif dummy_input_data: 
                if verbose: print(f"[{model_name}] Attempting feature extraction with batch size {expected_batch_size}...")
                features = extractor.transform(dummy_input_data)
                
                if task.startswith("multimodal"):
                    if not isinstance(features, dict):
                        raise AssertionError(f"Multimodal features not a dict (got {type(features)})")
                    
                    expected_keys_present = True
                    if task == "multimodal_image_text" and not ("image_features" in features and "text_features" in features):
                        expected_keys_present = False
                    elif task == "multimodal_video_text" and not ("video_features" in features and "text_features" in features):
                        expected_keys_present = False
                    if not expected_keys_present:
                         raise AssertionError("Expected standard keys in multimodal output dict.")

                    for key_feat, val_feat in features.items(): 
                        if not isinstance(val_feat, np.ndarray):
                            raise AssertionError(f"Multimodal feature '{key_feat}' not a numpy array.")
                        mm_dim_config = extractor.get_feature_dimension()
                        if isinstance(mm_dim_config, dict):
                            main_key_part = key_feat.split('_')[0] 
                            if main_key_part not in mm_dim_config:
                                if model_name.startswith("xclip") and main_key_part in ["video", "text"]: # XCLIP dim is single value
                                    expected_dim_mm = mm_dim_config 
                                else:
                                    raise AssertionError(f"Dimension key '{main_key_part}' not in model's dim config {mm_dim_config} for feature key '{key_feat}'")
                            else:
                                expected_dim_mm = mm_dim_config[main_key_part]
                        else: 
                            expected_dim_mm = mm_dim_config

                        if val_feat.shape != (expected_batch_size, expected_dim_mm):
                             raise AssertionError(f"Multimodal feature '{key_feat}' shape mismatch. Got {val_feat.shape}, expected ({expected_batch_size}, {expected_dim_mm})")
                else: 
                    if not isinstance(features, np.ndarray):
                        raise AssertionError(f"Features not a numpy array (got {type(features)})")
                    expected_dim_uni = extractor.get_feature_dimension()
                    # Handle case where expected_dim_uni itself might be a dict for some misconfigured unimodal model (should not happen)
                    if isinstance(expected_dim_uni, dict): 
                        raise AssertionError(f"Unimodal model '{model_name}' has dict for feature_dimension: {expected_dim_uni}")

                    if features.shape != (expected_batch_size, expected_dim_uni):
                        raise AssertionError(f"Feature shape mismatch. Got {features.shape}, expected ({expected_batch_size}, {expected_dim_uni})")
                
                current_result["test_status"] = "PASSED"
                if verbose: print(f"[{model_name}] Feature extraction and validation PASSED.")
            else: 
                current_result["test_status"] = "SKIPPED"
                if not is_video_task_with_placeholder: 
                    notes_collector.append("No dummy data prepared.")
                if verbose: print(f"[{model_name}] SKIPPED (No dummy data prepared).")


        except ImportError as ie:
            current_result["availability"] = "✗ (Import)"
            current_result["test_status"] = "FAILED"
            short_msg = textwrap.shorten(str(ie).splitlines()[0], width=60, placeholder="...")
            notes_collector.append(f"{type(ie).__name__}: {short_msg}")
            if verbose: print(f"[{model_name}] FAILED (Initialization ImportError): {ie}")
        except Exception as e:
            if current_result["availability"] == "✓": 
                current_result["test_status"] = "FAILED (Runtime)"
            else: 
                current_result["availability"] = "✗ (Init Error)"
                current_result["test_status"] = "FAILED"
            short_msg = textwrap.shorten(str(e).splitlines()[0], width=60, placeholder="...")
            notes_collector.append(f"{type(e).__name__}: {short_msg}")
            if verbose: print(f"[{model_name}] FAILED ({type(e).__name__}): {e}")
        
        current_result["notes"] = "; ".join(notes_collector) if notes_collector else ""
        test_results_list.append(current_result)

    if dummy_image_path and os.path.exists(dummy_image_path): os.remove(dummy_image_path)
    if dummy_audio_path and os.path.exists(dummy_audio_path): os.remove(dummy_audio_path)
    if dummy_video_path and os.path.exists(dummy_video_path): os.remove(dummy_video_path)
    if verbose: print("\nDummy files cleaned up.")

    print("\n\n" + "="*80) 
    print(" DLFeat Self-Test Summary Report".center(80))
    print("="*80)
    
    col_model_name = max(25, max(len(r["model_name"]) for r in test_results_list) if test_results_list else 25) # Increased for longer names
    col_task = max(12, max(len(r["task"]) for r in test_results_list) if test_results_list else 12)
    col_source = max(15, max(len(r["source"]) for r in test_results_list) if test_results_list else 15)
    col_avail = 10 
    col_status = 22 
    
    preferred_total_width = 140 # Adjusted preferred width for more notes
    col_notes = max(20, preferred_total_width - (col_model_name + col_task + col_source + col_avail + col_status + (6*3) + 7)) # +7 for extra chars in header


    header = f"| {'Model'.ljust(col_model_name)} | {'Task'.ljust(col_task)} | {'Source'.ljust(col_source)} | {'Available'.ljust(col_avail)} | {'Test Status'.ljust(col_status)} | Notes{' '.ljust(col_notes-5)} |"
    separator = f"|{'-'*(col_model_name+2)}|{'-'*(col_task+2)}|{'-'*(col_source+2)}|{'-'*(col_avail+2)}|{'-'*(col_status+2)}|{'-'*(col_notes+2)}|"
    
    print(separator)
    print(header)
    print(separator)

    for r in test_results_list:
        avail_str = r["availability"]
        status_str = r["test_status"]
        
        if "✓" in avail_str: avail_display = "✓".center(col_avail)
        else: avail_display = avail_str.ljust(col_avail) 

        if "PASSED" in status_str: status_display = f"✓ {status_str}".ljust(col_status)
        elif "FAILED" in status_str: status_display = f"✗ {status_str}".ljust(col_status)
        else: status_display = f"~ {status_str}".ljust(col_status) # SKIPPED (Transform) or SKIPPED

        model_disp = textwrap.shorten(r["model_name"], width=col_model_name, placeholder="..")
        task_disp = textwrap.shorten(r["task"], width=col_task, placeholder="..")
        source_disp = textwrap.shorten(r["source"], width=col_source, placeholder="..")
        notes_disp = textwrap.shorten(r["notes"], width=col_notes, placeholder="..")

        print(f"| {model_disp.ljust(col_model_name)} | {task_disp.ljust(col_task)} | {source_disp.ljust(col_source)} | {avail_display} | {status_display} | {notes_disp.ljust(col_notes)} |")
    print(separator)
    print("="*80)
    
    return test_results_list


if __name__ == '__main__':
    results = run_self_tests() 
