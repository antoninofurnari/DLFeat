# Getting Started with DLFeat
This guide will help you quickly get started with DLFeat to extract features from various modalities.

# Installation
First, ensure you have DLFeat.py in your project directory or installed as a package. Then, you need to install its core dependencies.

```
pip install torch torchvision torchaudio scikit-learn Pillow numpy scipy
pip install transformers sentence-transformers timm requests
```

For the best experience and access to all models, it's recommended to keep these libraries, especially transformers and torchvision, up-to-date:

```pip install --upgrade torch torchvision torchaudio transformers sentence-transformers timm requests
```

Basic Usage
Here's how to use DLFeatExtractor for different types of data:

## 1. Import DLFeat Components

```python
from DLFeat import DLFeatExtractor, list_available_models
import os # For dummy file creation/cleanup in examples
from PIL import Image, ImageDraw # For dummy image creation
import numpy as np_example # Renamed to avoid conflict if DLFeat itself uses np
import scipy.io.wavfile as scipy_wav_example # Renamed

# You can list available models (optional)
# print("All available models:", list_available_models())
# print("Image models:", list_available_models(task_type="image"))
```

## 2. Image Feature Extraction

```python
# Initialize for an image model
try:
    img_extractor = DLFeatExtractor(model_name="resnet18", task_type="image")
    print(f"Successfully initialized resnet18 extractor for images.")

    # Create a dummy image for this snippet
    dummy_img_path_gs = "temp_dummy_image_gs.png"
    try:
        img_gs = Image.new('RGB', (224, 224), color = 'skyblue')
        d_gs = ImageDraw.Draw(img_gs)
        d_gs.text((10,10), "Sample Img", fill=(0,0,0))
        img_gs.save(dummy_img_path_gs)
        
        # DLFeat expects a list of inputs for the transform method
        image_paths_gs = [dummy_img_path_gs, dummy_img_path_gs] 
        
        # Fit (no-op for pre-trained models) and transform
        img_extractor.fit(image_paths_gs) 
        image_features_gs = img_extractor.transform(image_paths_gs)
        print(f"Image features shape: {image_features_gs.shape}") 
        # Example output for resnet18 with 2 images: (2, 512)
        
    except ImportError:
        print("Pillow (for ImageDraw) not found, skipping image creation part of snippet.")
    except Exception as e_img:
        print(f"Error in image feature extraction snippet: {e_img}")
    finally:
        if os.path.exists(dummy_img_path_gs):
            os.remove(dummy_img_path_gs)

except Exception as e_init_img:
    print(f"Could not initialize image extractor: {e_init_img}")
    print("Ensure dependencies for image models are installed.")
```

## 3. Text Feature Extraction

```python
try:
    text_extractor = DLFeatExtractor(model_name="sentence-bert", task_type="text")
    print("Successfully initialized sentence-bert extractor for text.")
    
    texts_gs = ["This is the first sentence for DLFeat feature extraction.", 
                "DLFeat aims to make embeddings easy to obtain."]
    text_features_gs = text_extractor.transform(texts_gs)
    print(f"Text features shape: {text_features_gs.shape}") 
    # Example output for sentence-bert with 2 sentences: (2, 384)
except Exception as e_text:
    print(f"Error in text feature extraction snippet: {e_text}")
    print("Ensure dependencies for text models (like sentence-transformers) are installed.")
```

## 4. Audio Feature Extraction

Requires torchaudio and scipy for creating the dummy audio in this snippet.

```python
try:
    # Create a dummy audio file for this snippet
    dummy_audio_path_gs = "temp_dummy_audio_gs.wav"
    try:
        sample_rate_gs = 16000; duration_gs = 1; frequency_gs = 440
        t_gs = np_example.linspace(0, duration_gs, int(sample_rate_gs * duration_gs), endpoint=False)
        data_gs = np_example.sin(2 * np_example.pi * frequency_gs * t_gs) * 0.5
        data_int16_gs = (data_gs * 32767).astype(np_example.int16)
        scipy_wav_example.write(dummy_audio_path_gs, sample_rate_gs, data_int16_gs)

        audio_extractor = DLFeatExtractor(model_name="wav2vec2_base", task_type="audio")
        print("Successfully initialized wav2vec2_base extractor for audio.")
        
        audio_paths_gs = [dummy_audio_path_gs, dummy_audio_path_gs]
        audio_features_gs = audio_extractor.transform(audio_paths_gs)
        print(f"Audio features shape: {audio_features_gs.shape}") 
        # Example output for wav2vec2_base with 2 audio files: (2, 768)
        
    except ImportError:
        print("Scipy or Numpy not found, skipping audio creation/processing part of snippet.")
    except Exception as e_audio:
        print(f"Error in audio feature extraction snippet: {e_audio}")
    finally:
        if os.path.exists(dummy_audio_path_gs):
            os.remove(dummy_audio_path_gs)
            
except Exception as e_init_audio:
    print(f"Could not initialize audio extractor: {e_init_audio}")
    print("Ensure dependencies for audio models (like transformers, torchaudio) are installed.")
```

## 5. Video Feature Extraction

Note: Video processing can be resource-intensive and slow. The example below uses r2plus1d_18. For actual use, replace placeholder paths with paths to your real video files. The self-test function run_self_tests(attempt_real_video_test=True) attempts to download a small sample video if the requests library is installed.

```python
try:
    video_extractor = DLFeatExtractor(model_name="r2plus1d_18", task_type="video")
    print("Successfully initialized r2plus1d_18 extractor for video.")
    
    # For actual use, create a list of paths to your video files:
    # video_paths = ["path/to/your/video1.mp4", "path/to/your/video2.avi"]
    # video_features = video_extractor.transform(video_paths)
    # print(f"Video features shape: {video_features.shape}")
    print("To test video feature extraction, please provide a list of actual video file paths to the .transform() method.")
    print("Example: video_features = video_extractor.transform(['my_video.mp4'])")
    
except Exception as e_video:
    print(f"Could not initialize video extractor or run snippet: {e_video}")
    print("Ensure dependencies for video models (like torchvision, transformers, and potentially ffmpeg for video reading) are correctly installed.")
```

## 6. Multimodal (Image-Text) Feature Extraction

```python
try:
    # Create a dummy image for this snippet
    dummy_img_path_mm_gs = "temp_dummy_image_mm_gs.png"
    try:
        img_mm_gs = Image.new('RGB', (224, 224), color = 'lightcoral')
        d_mm_gs = ImageDraw.Draw(img_mm_gs)
        d_mm_gs.text((10,10), "Multimodal", fill=(0,0,0))
        img_mm_gs.save(dummy_img_path_mm_gs)

        clip_extractor = DLFeatExtractor(model_name="clip_vit_b32", task_type="multimodal_image_text")
        print("Successfully initialized clip_vit_b32 extractor for multimodal image-text.")
        
        multimodal_data_gs = [
            (dummy_img_path_mm_gs, "A light coral square with text."),
            (dummy_img_path_mm_gs, "Another description of the same image example.")
        ]
        multimodal_features_gs = clip_extractor.transform(multimodal_data_gs)
        
        print(f"CLIP Image features shape: {multimodal_features_gs['image_features'].shape}") 
        # Example output for clip_vit_b32 with 2 image-text pairs: (2, 512)
        print(f"CLIP Text features shape: {multimodal_features_gs['text_features'].shape}")   
        # Example output: (2, 512)
        
    except ImportError:
        print("Pillow (for ImageDraw) not found, skipping image creation part of multimodal snippet.")
    except Exception as e_mm_imgtxt:
        print(f"Error in multimodal (image-text) snippet: {e_mm_imgtxt}")
    finally:
        if os.path.exists(dummy_img_path_mm_gs):
            os.remove(dummy_img_path_mm_gs)

except Exception as e_init_mm_imgtxt:
    print(f"Could not initialize multimodal (image-text) extractor: {e_init_mm_imgtxt}")
    print("Ensure dependencies for multimodal models (like transformers) are installed.")
```

## Next Steps

Explore the Model Zoo in the full documentation for a list of all available models, their capabilities, and source libraries.

Refer to the API Reference for detailed information on DLFeatExtractor methods and other utilities.

Run the self-tests using from DLFeat import run_self_tests; run_self_tests() to check model availability in your environment.

