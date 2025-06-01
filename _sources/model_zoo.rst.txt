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
