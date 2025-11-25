#!/usr/bin/env python3
"""
Training script for YOLOv12 Triple Input with DINOv3 backbone - GREYSCALE VERSION.

This script trains YOLOv12 Triple Input models with DINOv3 feature extraction
for enhanced performance in civil engineering applications.

GREYSCALE VERSION: Uses 3 channels total (1 greyscale channel per image)
instead of 9 channels (3 RGB channels per image).

Usage:
    # P0 DINOv3 input preprocessing only (like reference repository)
    python "train_grey scale.py" --data test_triple_dataset.yaml --integrate initial

    # No DINOv3 integration (standard triple input greyscale)
    python "train_grey scale.py" --data test_triple_dataset.yaml --integrate nodino

    # P3 DINOv3 feature enhancement only (after P3 stage)
    python "train_grey scale.py" --data test_triple_dataset.yaml --integrate p3

    # Dual DINOv3 integration (P0 preprocessing + P3 enhancement)
    python "train_grey scale.py" --data test_triple_dataset.yaml --integrate p0p3

    # With different DINOv3 sizes
    python "train_grey scale.py" --data test_triple_dataset.yaml --dinov3-size base --freeze-dinov3
    python "train_grey scale.py" --data test_triple_dataset.yaml --pretrained yolov12n.pt --dinov3-size small
"""

import argparse
import torch
from pathlib import Path
import warnings
from ultralytics import YOLO

def train_triple_dinov3_greyscale(
    data_config: str,
    dinov3_size: str = "small",
    freeze_dinov3: bool = True,
    use_triple_branches: bool = False,
    pretrained_path: str = None,
    epochs: int = 100,
    batch_size: int = 8,  # Smaller default due to DINOv3 memory usage
    imgsz: int = 640,     # YOLO default size (can use 224 for DINOv3 or 640 for YOLO)
    patience: int = 50,
    name: str = "yolov12_triple_dinov3_greyscale",
    device: str = "0",
    integrate: str = "initial",  # New parameter: "initial", "nodino", "p3"
    variant: str = "s",  # YOLOv12 model variant: n, s, m, l, x
    save_period: int = -1,  # Save weights every N epochs (-1 = only best/last)
    **kwargs
):
    """
    Train YOLOv12 Triple Input with DINOv3 backbone - GREYSCALE VERSION.

    Args:
        data_config: Path to dataset configuration
        dinov3_size: DINOv3 model size (small, base, large, giant)
        freeze_dinov3: Whether to freeze DINOv3 backbone
        use_triple_branches: Whether to use separate DINOv3 branches
        pretrained_path: Path to pretrained YOLOv12 model (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Image size
        patience: Early stopping patience
        name: Experiment name
        device: Device to use
        integrate: DINOv3 integration strategy
            - "initial": P0 input preprocessing only (3-channel greyscale → enhanced input)
            - "nodino": No DINOv3 integration (standard triple input greyscale)
            - "p3": P3 feature enhancement only (after P3 stage)
            - "p0p3": Dual integration (P0 preprocessing + P3 enhancement)
        variant: YOLOv12 model variant (n, s, m, l, x)
        save_period: Save weights every N epochs (-1 = only best/last, saves disk space)
        **kwargs: Additional training arguments

    Returns:
        Training results
    """

    print("YOLOv12 Triple Input with DINOv3 Training - GREYSCALE VERSION")
    print("=" * 60)
    print(f"Data Config: {data_config}")
    print(f"YOLOv12 Variant: {variant}")
    print(f"DINOv3 Size: {dinov3_size}")
    print(f"DINOv3 Integration: {integrate}")
    print(f"Freeze DINOv3: {freeze_dinov3}")
    print(f"Triple Branches: {use_triple_branches}")
    print(f"Pretrained: {pretrained_path or 'None'}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {imgsz}")
    print(f"Device: {device}")
    print(f"Input Channels: 3 (greyscale: 1 channel per image)")
    print(f"Save Period: {'Best/Last only' if save_period == -1 else f'Every {save_period} epochs'}")
    print("-" * 60)

    # Determine actual device to use - Auto-detect CPU if CUDA not available
    actual_device = 'cpu'
    if device != 'cpu' and torch.cuda.is_available():
        try:
            # Convert device string to proper format
            if device.isdigit():
                test_device = f"cuda:{device}"
            else:
                test_device = device

            # Test if the device is valid
            torch.zeros(1).to(test_device)
            actual_device = test_device
            print(f"✓ Using GPU device: {actual_device}")
        except Exception as e:
            print(f"⚠️ Device '{device}' not available ({e}), falling back to CPU")
            actual_device = 'cpu'
    else:
        print(f"✓ Using CPU device")

    print("-" * 60)

    # Step 1: Setup requirements based on integration strategy
    print(f"\n🔧 Step 1: Setting up requirements for integration strategy: {integrate}...")

    if integrate == "nodino":
        print("No DINOv3 integration - using standard triple input greyscale model")
        # Skip DINOv3 setup for nodino mode
    else:
        try:
            import transformers
            import timm
            print("✓ DINOv3 packages available")
        except ImportError as e:
            print(f"❌ Missing required packages: {e}")
            print("Install with: pip install transformers timm huggingface_hub")
            return None

    # Step 2: Download DINOv3 model if needed
    if integrate != "nodino":
        print(f"\n📥 Step 2: Preparing DINOv3 {dinov3_size} model...")
        print("🔐 Setting up HuggingFace authentication...")

        # Check HuggingFace authentication
        try:
            from ultralytics.nn.modules.dinov3 import setup_huggingface_auth
            auth_success, auth_source = setup_huggingface_auth()

            if not auth_success:
                print("⚠️ HuggingFace authentication not configured.")
                print("DINOv3 models may not be accessible without authentication.")
                print("To set up authentication:")
                print("  1. Get token from: https://huggingface.co/settings/tokens")
                print("  2. Set environment variable: export HUGGINGFACE_HUB_TOKEN='your_token'")
                print("  3. Or run: huggingface-cli login")
        except Exception as e:
            print(f"⚠️ Authentication setup warning: {e}")

        try:
            from download_dinov3 import DINOv3Downloader
            downloader = DINOv3Downloader()
            success, _ = downloader.download_model(dinov3_size, method="auto")
            if success:
                print(f"✓ DINOv3 {dinov3_size} model ready")
            else:
                print(f"⚠️ Failed to download DINOv3 {dinov3_size}, proceeding anyway...")
        except Exception as e:
            print(f"⚠️ DINOv3 download warning: {e}")
            print("Proceeding with training, model will be downloaded automatically if needed")
    else:
        print("\n📥 Step 2: Skipping DINOv3 download (nodino mode)")

    # Step 3: Create model configuration based on integration strategy
    print(f"\n🏗️ Step 3: Creating model configuration for integration: {integrate}...")

    # Select appropriate model configuration
    if integrate == "dinov3":
        model_config = "ultralytics/cfg/models/v12/yolov12_triple_greyscale.yaml"
        print("Using standard triple input greyscale model (DINOv3)")
    elif integrate == "initial":
        model_config = "ultralytics/cfg/models/v12/yolov12_triple_dinov3_p0_only.yaml"
        print("Using DINOv3 P0-only preprocessing integration (input preprocessing before backbone)")
    elif integrate == "p3":
        model_config = "ultralytics/cfg/models/v12/yolov12_triple_dinov3_p3.yaml"
        print("Using DINOv3 integration after P3 stage")
    elif integrate == "p0p3":
        model_config = "ultralytics/cfg/models/v12/yolov12_triple_dinov3_p0p3_adapter.yaml"
        print("Using adapter-based dual DINOv3 integration (before backbone + after P3) - all variants")
    else:
        print(f"❌ Unknown integration strategy: {integrate}")
        print("Available options: dinov3, initial, p3, p0p3")
        return None

    if not Path(model_config).exists():
        print(f"❌ Model config not found: {model_config}")
        return None

    # Step 4: Initialize model
    print(f"\n🚀 Step 4: Initializing model...")
    try:
        if pretrained_path:
            print(f"Loading with pretrained weights: {pretrained_path}")
            # For DINOv3 integration, we'll need custom weight loading
            from load_pretrained_triple import load_pretrained_weights_to_triple_model
            model = load_pretrained_weights_to_triple_model(
                pretrained_path=pretrained_path,
                triple_model_config=model_config
            )
        else:
            print("Training from scratch with DINOv3 features (greyscale)")
            # For nodino integration, use the YAML as-is
            if integrate == "nodino":
                # Set model scale to prevent scale warning and ensure proper architecture
                import yaml

                # Load the base config
                with open(model_config, 'r') as f:
                    config = yaml.safe_load(f)

                # GREYSCALE: Set input channels to 3 for triple greyscale input (1 channel per image)
                config['ch'] = 3  # Changed from 9 to 3 for greyscale

                # NO DINOv3: Replace DINOv3Backbone with standard Conv layer
                if config.get('backbone') and len(config['backbone']) > 0:
                    first_layer = config['backbone'][0]
                    # Check if first layer is DINOv3Backbone
                    if len(first_layer) >= 3 and first_layer[2] == 'DINOv3Backbone':
                        print("⚠️ Removing DINOv3Backbone from config (nodino mode)")
                        print(f"   Original first layer: {first_layer}")
                        # Replace DINOv3Backbone with standard Conv layer
                        # Calculate output channels based on variant
                        width_scaling = {'n': 0.25, 's': 0.5, 'm': 1.0, 'l': 1.0, 'x': 1.5}
                        scale_factor = width_scaling.get(variant, 1.0)
                        base_channels = 64
                        output_channels = int(base_channels * scale_factor)
                        # Replace with Conv layer: [from, repeats, module, [out_channels, kernel, stride]]
                        config['backbone'][0] = [-1, 1, 'Conv', [output_channels, 3, 1]]  # P0 - standard Conv for greyscale input
                        print(f"   Replaced with Conv layer: output_channels={output_channels} (variant '{variant}', {scale_factor}x scaling)")

                # Set scale BEFORE creating model
                config['scale'] = variant  # Set model scale (n, s, m, l, x)

                # Create temporary config file with correct settings and variant
                # IMPORTANT: Scale letter MUST come immediately after version number for guess_model_scale() regex
                temp_config_path = f"temp_yolov12{variant}_triple_nodino_greyscale.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                # DEBUG: Print temp config path and first few lines
                print(f"\n[DEBUG] Temporary config path: {temp_config_path}")
                print(f"[DEBUG] Config contents - ch: {config.get('ch')}, scale: {config.get('scale')}")
                print(f"[DEBUG] First backbone layer: {config.get('backbone', [[]])[0]}")

                # Create model with variant scaling
                print(f"[DEBUG] Creating model with scale={variant} (set in config)")
                model = YOLO(temp_config_path)

                # Don't clean up temporary file yet - keep for debugging
                # Path(temp_config_path).unlink(missing_ok=True)
            else:
                # For DINOv3 integration, modify the model configuration dynamically
                import yaml

                # Load the base config
                with open(model_config, 'r') as f:
                    config = yaml.safe_load(f)

                # DEBUG: Check what's in the config before processing
                if config.get('backbone') and len(config['backbone']) > 0:
                    first_layer_before = config['backbone'][0]
                    print(f"\n[DEBUG] First backbone layer BEFORE processing: {first_layer_before}")
                    print(f"[DEBUG] Module name BEFORE: {first_layer_before[2] if len(first_layer_before) >= 3 else 'N/A'}")

                # GREYSCALE: Set input channels to 3 for triple greyscale input (1 channel per image)
                config['ch'] = 3  # Changed from 9 to 3 for greyscale

                # Update DINOv3 model size and configuration
                if integrate in ["dinov3", "initial", "p3", "p0p3"]:
                    # Map model sizes to HuggingFace model names
                    model_name_map = {
                        "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
                        "small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                        "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                        "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                        "huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                        "giant": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                        "sat_large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                        "sat_giant": "facebook/dinov3-vit7b16-pretrain-sat493m"
                    }

                    dino_model_name = model_name_map.get(dinov3_size, model_name_map["small"])

                    # Update DINOv3Backbone layers in config if they exist (for any mode)
                    if config.get('backbone'):
                        print(f"\n[DEBUG] Checking backbone layers for DINOv3Backbone...")
                        dino_found = False
                        for i, layer in enumerate(config['backbone']):
                            if len(layer) >= 3:
                                module_name = layer[2]
                                print(f"[DEBUG] Layer {i}: module = {module_name} (type: {type(module_name)})")
                                if module_name == 'DINOv3Backbone' or str(module_name) == 'DINOv3Backbone':
                                    dino_found = True
                                    if len(layer) >= 4 and isinstance(layer[3], list) and len(layer[3]) > 0:
                                        old_model_name = layer[3][0] if isinstance(layer[3][0], str) else None
                                        layer[3][0] = dino_model_name  # Update model name
                                        if old_model_name and old_model_name != dino_model_name:
                                            print(f"✓ Updated DINOv3Backbone layer {i}: {old_model_name} → {dino_model_name}")
                                        else:
                                            print(f"✓ DINOv3Backbone layer {i} using model: {dino_model_name}")
                                    break  # Found it, break out
                        
                        if not dino_found:
                            print(f"⚠️ [DEBUG] DINOv3Backbone NOT found in backbone! First layer is: {config['backbone'][0] if len(config['backbone']) > 0 else 'N/A'}")
                            # For dinov3 mode, we MUST have DINOv3Backbone - add it if missing
                            if integrate == "dinov3":
                                print(f"⚠️ [DEBUG] Adding DINOv3Backbone as first layer for dinov3 mode...")
                                # Calculate output channels based on variant
                                width_scaling = {'n': 0.25, 's': 0.5, 'm': 1.0, 'l': 1.0, 'x': 1.5}
                                scale_factor = width_scaling.get(variant, 1.0)
                                base_channels = 64
                                output_channels = int(base_channels * scale_factor)
                                # Insert DINOv3Backbone as first layer: [from, repeats, module, [model_name, input_channels, output_channels, freeze]]
                                dinov3_layer = [-1, 1, 'DINOv3Backbone', [dino_model_name, 3, output_channels, freeze_dinov3]]
                                config['backbone'].insert(0, dinov3_layer)
                                print(f"✓ [DEBUG] Added DINOv3Backbone layer: {dinov3_layer}")

                    if integrate == "initial":
                        # P0-only integration: P0 preprocessing happens outside backbone
                        # No backbone modifications needed - preprocessing will be handled separately
                        # The backbone starts with regular Conv layer that receives P0 preprocessed input
                        pass  # No backbone config changes needed
                    elif integrate == "p3":
                        # Update the P3 feature enhancement configuration (using conv-based approach)
                        # P3FeatureEnhancer has simpler args: [input_channels, output_channels]

                        # Calculate scaled channels for P3 enhancement
                        width_scaling = {'n': 0.25, 's': 0.5, 'm': 1.0, 'l': 1.0, 'x': 1.5}
                        scale_factor = width_scaling.get(variant, 1.0)

                        # P3 enhancer: receives scaled input from P3 stage, outputs base channels that will be scaled
                        # Base 256 ensures compatibility: 256 * 0.25 = 64, 256 * 0.5 = 128, etc.
                        # Note: YOLO parse_model will automatically set input_channels from previous layer
                        config['backbone'][5][-1][1] = 256  # Target output channels (base, will be scaled)
                    elif integrate == "p0p3":
                        # P0P3 with adapter pattern - update model names and freeze settings
                        config['backbone'][0][-1][0] = dino_model_name  # P0 DINOv3 model name
                        config['backbone'][0][-1][4] = freeze_dinov3    # P0 DINOv3 freeze setting
                        config['backbone'][5][-1][0] = dino_model_name  # P3 DINOv3 model name
                        config['backbone'][5][-1][4] = freeze_dinov3    # P3 DINOv3 freeze setting

                        # Calculate scaled channels for adapters
                        width_scaling = {'n': 0.25, 's': 0.5, 'm': 1.0, 'l': 1.0, 'x': 1.5}
                        scale_factor = width_scaling.get(variant, 1.0)

                        # P0 adapter: always outputs the base channel count that will be scaled by YOLO
                        # The adapter will handle 3→64→64 (greyscale), then YOLO will scale the 64 to match variant
                        config['backbone'][0][-1][3] = 64   # P0 adapter target channels (base, will be scaled)

                        # P3 adapter: receives input from layer 4, outputs base channels that will be scaled
                        # Ensure base channels result in multiples of 32 after scaling for ABlock compatibility
                        # For all variants: 128 * 0.25 = 32, 128 * 0.5 = 64, 128 * 1.0 = 128, 128 * 1.5 = 192
                        # All are multiples of 32, so 128 is a good base
                        config['backbone'][5][-1][1] = 128   # Placeholder - will be replaced by YOLO parse_model
                        config['backbone'][5][-1][3] = 128  # P3 adapter target channels (base, will be scaled)

                    print(f"Using DINOv3 model: {dino_model_name}")
                    print(f"DINOv3 frozen: {freeze_dinov3}")
                    if integrate == "dinov3":
                        width_scaling = {'n': 0.25, 's': 0.5, 'm': 1.0, 'l': 1.0, 'x': 1.5}
                        scale_factor = width_scaling.get(variant, 1.0)
                        print(f"Direct DINOv3Backbone integration (GREYSCALE): 3-channel input → {dino_model_name}, variant '{variant}' ({scale_factor}x scaling)")
                    elif integrate == "p3":
                        width_scaling = {'n': 0.25, 's': 0.5, 'm': 1.0, 'l': 1.0, 'x': 1.5}
                        scale_factor = width_scaling.get(variant, 1.0)
                        print(f"P3 Feature Enhancement: after P3 stage (512*{scale_factor}→256*{scale_factor}), variant '{variant}' ({scale_factor}x scaling)")
                    elif integrate == "p0p3":
                        width_scaling = {'n': 0.25, 's': 0.5, 'm': 1.0, 'l': 1.0, 'x': 1.5}
                        scale_factor = width_scaling.get(variant, 1.0)
                        print(f"Adapter-based Dual DINOv3 (GREYSCALE): P0 (3→64→64*{scale_factor}) + P3 (auto→64→128*{scale_factor}), variant '{variant}' ({scale_factor}x scaling)")

                # Set scale BEFORE creating model
                config['scale'] = variant  # Set model scale (n, s, m, l, x)

                # Create temporary config file with variant
                # IMPORTANT: Scale letter MUST come immediately after version number for guess_model_scale() regex
                temp_config_path = f"temp_yolov12{variant}_triple_dinov3_{dinov3_size}_greyscale.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                # DEBUG: Print temp YAML file contents to verify scale is written
                print(f"\n[DEBUG] === Contents of {temp_config_path} ===")
                with open(temp_config_path, 'r') as f:
                    yaml_contents = f.read()
                    # Print first 50 lines to see nc, ch, scale, scales section
                    lines = yaml_contents.split('\n')[:50]
                    for i, line in enumerate(lines, 1):
                        print(f"{i:3d}: {line}")
                print(f"[DEBUG] === End of YAML contents ===\n")

                # DEBUG: Print first backbone layer to verify DINOv3Backbone is still there
                if config.get('backbone') and len(config['backbone']) > 0:
                    first_layer = config['backbone'][0]
                    print(f"\n[DEBUG] First backbone layer in config: {first_layer}")
                    print(f"[DEBUG] Module name: {first_layer[2] if len(first_layer) >= 3 else 'N/A'}")
                    
                    # CRITICAL: Ensure DINOv3Backbone is accessible in parse_model's globals
                    # parse_model uses globals()[m] which looks in function scope, so we need to
                    # make DINOv3Backbone available in the parse_model function's globals
                    from ultralytics.nn.modules.dinov3 import DINOv3Backbone, DINOv3TripleBackbone, DINOv3BackboneWithAdapter, P3FeatureEnhancer
                    import ultralytics.nn.tasks as tasks_module
                    import sys
                    import types
                    
                    # Patch parse_model to have access to DINOv3 modules in its globals
                    # We need to add them to the module's __dict__ before parse_model is called
                    tasks_module.DINOv3Backbone = DINOv3Backbone
                    tasks_module.DINOv3TripleBackbone = DINOv3TripleBackbone
                    tasks_module.DINOv3BackboneWithAdapter = DINOv3BackboneWithAdapter
                    tasks_module.P3FeatureEnhancer = P3FeatureEnhancer
                    
                    # Also add to sys.modules to ensure they're accessible everywhere
                    import ultralytics.nn.modules
                    if not hasattr(ultralytics.nn.modules, 'DINOv3Backbone'):
                        ultralytics.nn.modules.DINOv3Backbone = DINOv3Backbone
                    
                    print(f"[DEBUG] ✓ DINOv3Backbone added to tasks module globals")
                    
                    # Verify it's accessible
                    try:
                        test_module = getattr(tasks_module, 'DINOv3Backbone')
                        print(f"[DEBUG] ✓ DINOv3Backbone accessible: {test_module}")
                    except AttributeError as e:
                        print(f"[DEBUG] ❌ DINOv3Backbone NOT accessible: {e}")

                # Create model with variant scaling - simple approach for all variants
                print(f"\n[DEBUG] Creating model from: {temp_config_path}")
                
                # CRITICAL FIX: Patch parse_model to find DINOv3 modules
                # parse_model uses globals()[m] which looks in function scope, not module scope
                # We need to monkey-patch the module lookup to also check ultralytics.nn.modules
                from ultralytics.nn.modules.dinov3 import DINOv3Backbone, DINOv3TripleBackbone, DINOv3BackboneWithAdapter, P3FeatureEnhancer
                import ultralytics.nn.tasks as tasks_module
                import ultralytics.nn.modules as modules_module
                import inspect
                
                # Store original parse_model
                original_parse_model = tasks_module.parse_model
                
                # Get the function's code object to patch globals lookup
                # We'll create a wrapper that modifies the globals() result
                def patched_parse_model(d, ch, verbose=True):
                    """Wrapper that ensures DINOv3 modules are in globals before calling parse_model."""
                    # Get the module where parse_model is defined
                    parse_model_module = inspect.getmodule(original_parse_model)
                    
                    # Ensure DINOv3 modules are in module's __dict__
                    parse_model_module.__dict__.setdefault('DINOv3Backbone', DINOv3Backbone)
                    parse_model_module.__dict__.setdefault('DINOv3TripleBackbone', DINOv3TripleBackbone)
                    parse_model_module.__dict__.setdefault('DINOv3BackboneWithAdapter', DINOv3BackboneWithAdapter)
                    parse_model_module.__dict__.setdefault('P3FeatureEnhancer', P3FeatureEnhancer)
                    
                    # Also ensure they're in the imported modules namespace
                    modules_module.DINOv3Backbone = DINOv3Backbone
                    modules_module.DINOv3TripleBackbone = DINOv3TripleBackbone
                    modules_module.DINOv3BackboneWithAdapter = DINOv3BackboneWithAdapter
                    modules_module.P3FeatureEnhancer = P3FeatureEnhancer
                    
                    # Call original parse_model (it will now see DINOv3 modules in globals)
                    return original_parse_model(d, ch, verbose)
                
                # Replace parse_model temporarily
                tasks_module.parse_model = patched_parse_model
                print(f"[DEBUG] ✓ Patched parse_model to include DINOv3 modules in globals")
                
                try:
                    print(f"[DEBUG] Creating model with scale={variant} (set in config)")
                    model = YOLO(temp_config_path)
                finally:
                    # Restore original parse_model
                    tasks_module.parse_model = original_parse_model

                # DEBUG: Check if DINOv3Backbone was actually loaded
                if hasattr(model, 'model') and hasattr(model.model, 'model'):
                    first_module = model.model.model[0]
                    print(f"\n[DEBUG] First module type in loaded model: {type(first_module).__name__}")
                    if hasattr(first_module, 'conv'):
                        print(f"[DEBUG] First module has conv attribute (not DINOv3Backbone)")
                    if 'dinov3' in str(type(first_module)).lower():
                        print(f"[DEBUG] ✓ DINOv3Backbone was loaded successfully!")
                    else:
                        print(f"[DEBUG] ⚠️ DINOv3Backbone was NOT loaded - module type: {type(first_module)}")

                # Don't clean up temporary file yet - keep for debugging
                # Path(temp_config_path).unlink(missing_ok=True)

        # Step 4.5: Setup P0 preprocessing if needed
        if integrate == "initial":
            print(f"\n🔄 Step 4.5: Setting up P0 DINOv3 preprocessing (greyscale)...")

            # Initialize P0 DINOv3 preprocessor
            from ultralytics.nn.modules.dinov3 import DINOv3Backbone

            # Calculate scaled output channels for P0 preprocessor
            width_scaling = {'n': 0.25, 's': 0.5, 'm': 1.0, 'l': 1.0, 'x': 1.5}
            scale_factor = width_scaling.get(variant, 1.0)
            p0_output_channels = int(64 * scale_factor)  # Base 64 channels scaled by variant

            # Create P0 preprocessor on CPU first, then move to target device
            print(f"Creating P0 DINOv3 preprocessor (greyscale 3 channels)...")
            p0_preprocessor = DINOv3Backbone(
                model_name=dino_model_name,
                input_channels=3,  # GREYSCALE: Changed from 9 to 3
                output_channels=p0_output_channels,
                freeze=freeze_dinov3,
                image_size=imgsz  # Use user-specified image size (default: 640)
            )

            # Move to target device after creation
            print(f"Moving P0 preprocessor to {actual_device}...")
            p0_preprocessor = p0_preprocessor.to(actual_device)

            # Assign to model after successful device movement
            model.p0_preprocessor = p0_preprocessor

            # Update model's first Conv layer to expect P0 preprocessor output
            first_conv = model.model.model[0]
            if hasattr(first_conv, 'conv'):
                # Update input channels of first conv layer
                original_weight = first_conv.conv.weight.data
                first_conv.conv = torch.nn.Conv2d(
                    p0_output_channels,
                    first_conv.conv.out_channels,
                    first_conv.conv.kernel_size,
                    first_conv.conv.stride,
                    first_conv.conv.padding,
                    bias=first_conv.conv.bias is not None
                )

                # Move to target device
                if actual_device != 'cpu':
                    first_conv.conv = first_conv.conv.to(actual_device)

                # Initialize new weights (simple approach)
                torch.nn.init.kaiming_normal_(first_conv.conv.weight, mode='fan_out', nonlinearity='relu')

            print(f"✓ P0 DINOv3 preprocessor configured: 3 → {p0_output_channels} channels (greyscale)")
            print(f"✓ First backbone Conv updated: {p0_output_channels} → {first_conv.conv.out_channels} channels")

        print("✓ Model initialized successfully")

    except Exception as e:
        import traceback
        print(f"❌ Model initialization failed: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Full traceback:")
        print(traceback.format_exc())
        print("This might be due to DINOv3 integration complexity")
        return None

    # Step 5: Configure training parameters
    print(f"\n⚙️ Step 5: Configuring training parameters...")

    # Auto-adjust batch size and epochs based on dataset size and training strategy
    # For small datasets: smaller batch → more epochs (more training steps)
    # For large epochs: smaller batch → more gradient updates per epoch
    # For small epochs: larger batch → faster training
    
    # Calculate effective batch size and epochs
    original_batch_size = batch_size
    original_epochs = epochs
    
    # Adjust batch size based on epochs
    # More epochs = smaller batch (more training steps), Fewer epochs = larger batch (faster convergence)
    if epochs >= 200:
        # Very long training: use smaller batch for more gradient updates
        effective_batch_size = max(batch_size, 2)  # Keep minimum 2
        if batch_size >= 8:
            effective_batch_size = max(4, batch_size // 2)  # Reduce batch if too large
            print(f"  ℹ️ Long training ({epochs} epochs): Reducing batch size for more gradient updates")
    elif epochs >= 100:
        # Medium training: moderate batch size
        effective_batch_size = max(batch_size, 2)
    else:
        # Short training: use larger batch if possible (faster convergence)
        effective_batch_size = max(batch_size, 2)
        if batch_size < 4 and epochs < 50:
            print(f"  ℹ️ Short training ({epochs} epochs): Consider using larger batch size for faster convergence")
    
    # Adjust epochs based on batch size for small datasets
    # Smaller batch → need more epochs for same number of training steps
    # Standard: batch_size * epochs should be roughly constant for same amount of learning
    base_training_steps = 75 * 100  # 75 images * 100 epochs = 7500 training steps (baseline)
    effective_epochs = epochs
    if effective_batch_size < 4 and epochs < 100:
        # Very small batch: increase epochs to compensate
        steps_ratio = 4 / effective_batch_size  # How many more steps needed
        suggested_epochs = int(epochs * steps_ratio)
        if suggested_epochs <= epochs * 2:  # Don't more than double
            effective_epochs = suggested_epochs
            print(f"  ℹ️ Small batch size ({effective_batch_size}): Suggesting {effective_epochs} epochs for better convergence")
        else:
            print(f"  ℹ️ Small batch size ({effective_batch_size}): Consider increasing epochs for better convergence")
    
    # Force minimum batch size of 2 to avoid BatchNorm issues with batch=1
    effective_batch_size = max(effective_batch_size, 2)
    
    # Use adjusted values
    final_batch_size = effective_batch_size
    final_epochs = effective_epochs
    train_args = {
        'data': data_config,
        'epochs': final_epochs,  # Use adjusted epochs
        'batch': final_batch_size,  # Use adjusted batch size
        'imgsz': imgsz,
        'patience': patience,
        'save': True,
        'save_period': save_period,  # Save weights every N epochs (-1 = only best/last)
        'name': name,
        'verbose': True,
        'val': True,
        'plots': True,
        'cache': False,  # Disable caching for triple input
        'device': actual_device,  # Use auto-detected device (CPU if CUDA unavailable)

        # Learning rate configuration for VERY SMALL DATASET (66 images) - aggressive learning
        # Much higher initial LR to force learning on limited data
        'lr0': 0.01 if freeze_dinov3 else 0.005,  # HIGH initial LR for aggressive learning
        'lrf': 0.1,  # Higher final LR ratio to maintain learning throughout
        'momentum': 0.937,  # Standard momentum
        'weight_decay': 0.0001,  # REDUCED - Less regularization to allow overfitting on small dataset
        'warmup_epochs': 5,  # Longer warmup for stability with high LR
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,

        # Optimizer (AdamW often works better with transformers)
        'optimizer': 'AdamW',

        # Mixed precision (helpful for memory with DINOv3) - disabled to avoid BatchNorm issues with small batch
        'amp': False,

        # Augmentation for VERY SMALL DATASET (66 images) - STRONG approach to improve generalization
        # Strategy: Use heavy augmentation to force model to generalize better
        # Goal: Reduce overfitting, improve Test mAP (currently only 5.5%)
        'mixup': 0.0,  # Keep disabled - too confusing for dental X-rays
        'copy_paste': 0.0,  # Keep disabled - not suitable for this task
        'mosaic': 0.0,  # Keep disabled - too difficult with only 66 images
        'hsv_h': 0.0,  # Disable HSV hue augmentation (greyscale has no color)
        'hsv_s': 0.0,  # Disable HSV saturation augmentation (greyscale has no color)
        'hsv_v': 0.4,  # INCREASED - Stronger brightness variation to improve generalization
        'auto_augment': None,  # Disable auto augmentation (ToGray incompatible with greyscale)
        'erasing': 0.2,  # ENABLED - Random erasing to force robustness
        'plots': False,  # Disable plots (visualization might be incompatible with 3-channel greyscale input)

        # Geometric augmentations for greyscale - STRONG settings to reduce overfitting
        'degrees': 15.0,  # INCREASED - Stronger rotation (±15 degrees)
        'translate': 0.15,  # INCREASED - Stronger translation (±15% of image size)
        'scale': 0.6,  # INCREASED - Stronger scaling variation (0.4x-1.6x)
        'shear': 5.0,  # ENABLED - Add shearing for more variation
        'perspective': 0.0,  # Keep disabled - might distort dental features too much
        'flipud': 0.3,  # ENABLED - Vertical flip for more variation
        'fliplr': 0.5,  # Keep enabled - horizontal flip

        # Additional settings for small dataset
        'close_mosaic': 10,  # Mosaic already disabled, but set early closure anyway
        'bgr': 0.0,  # Disable BGR channel shuffling (not applicable to greyscale)
        'workers': 0,  # Disable multiprocessing workers to avoid DataLoader issues
        
        # Additional training tricks for small dataset
        'cls': 0.5,  # Classification loss weight
        'box': 7.5,  # Box loss weight (increase for better localization)
        'dfl': 1.5,  # Distribution focal loss weight
        'pose': 12.0,  # Pose loss weight (if applicable)
        'kobj': 1.0,  # Keypoint object loss weight
        'label_smoothing': 0.0,  # Label smoothing (can help with small dataset)

        # Additional arguments
        **kwargs
    }

    # Display key training parameters
    print(f"\n  📊 Training Configuration (Auto-adjusted for dataset size):")
    if final_batch_size != original_batch_size:
        print(f"  Batch size: {original_batch_size} → {final_batch_size} (adjusted)")
    else:
        print(f"  Batch size: {final_batch_size}")
    
    if final_epochs != original_epochs:
        print(f"  Epochs: {original_epochs} → {final_epochs} (adjusted for batch size)")
    else:
        print(f"  Epochs: {final_epochs}")
    
    # Calculate approximate training steps
    # Assuming ~75 training images
    approx_training_steps = 75 * final_epochs // final_batch_size
    print(f"  Approx. training steps: ~{approx_training_steps:,} (75 images × {final_epochs} epochs ÷ {final_batch_size} batch)")
    
    print(f"\n  🔧 Hyperparameters:")
    print(f"  Learning rate: {train_args['lr0']} → {train_args['lr0'] * train_args['lrf']:.6f} (final)")
    print(f"  Optimizer: {train_args['optimizer']}")
    print(f"  Mixed precision: {train_args['amp']}")
    print(f"  Warmup epochs: {train_args['warmup_epochs']}")
    print(f"\n  📈 Augmentation for SMALL DATASET:")
    print(f"    - Mosaic: {train_args['mosaic']}")
    print(f"    - MixUp: {train_args['mixup']}")
    print(f"    - CopyPaste: {train_args['copy_paste']}")
    print(f"    - Rotation: ±{train_args['degrees']}°")
    print(f"    - Translation: ±{train_args['translate']*100}%")
    print(f"    - Scaling: {1-train_args['scale']}x-{1+train_args['scale']}x")
    print(f"    - Horizontal Flip: {train_args['fliplr']*100}%")
    print(f"    - Brightness (HSV-V): {train_args['hsv_v']}")
    print(f"    - Random Erasing: {train_args['erasing']}")

    # Step 6: Memory and performance warnings
    print(f"\n⚠️ Step 6: Performance considerations...")
    print("DINOv3 backbone requires significant memory and compute:")
    print(f"  - Recommended batch size: 4-8 (effective: {effective_batch_size})")
    print(f"  - Recommended image size: 224-384 (current: {imgsz})")
    print(f"  - DINOv3 frozen: {freeze_dinov3} (recommended for initial training)")
    print(f"  - Using GREYSCALE input: 3 channels (1 per image)")

    if effective_batch_size > 8:
        print("⚠️ Large batch size may cause OOM with DINOv3")

    if imgsz > 384:
        print("⚠️ Large image size may cause OOM with DINOv3")

    # Step 7: Start training
    print(f"\n🎯 Step 7: Starting training...")
    print("Note: First epoch may be slow due to DINOv3 model download/loading")

    try:
        # For P0 integration, we need to wrap the model to handle preprocessing
        if integrate == "initial" and hasattr(model, 'p0_preprocessor'):
            print("Setting up persistent P0 preprocessing wrapper (greyscale)...")

            # Store original forward method
            original_forward = model.model.forward

            def wrapped_forward(x, *args, **kwargs):
                # Apply P0 preprocessing if x has 3 channels (greyscale)
                if x.shape[1] == 3:
                    x = model.p0_preprocessor(x)
                return original_forward(x, *args, **kwargs)

            # Replace forward method for the entire training session
            model.model.forward = wrapped_forward
            model._original_forward = original_forward  # Store for later restoration

            # Run training with wrapped forward but catch final validation errors
            try:
                results = model.train(**train_args)
            except RuntimeError as e:
                if "expected input" in str(e) and "to have 3 channels" in str(e):
                    print("⚠️ Final validation failed due to channel mismatch (expected behavior for P0 integration)")
                    print("✅ Training completed successfully despite validation error")
                    results = None  # Training was successful, just validation failed
                else:
                    raise e
        else:
            # Standard training without P0 preprocessing
            results = model.train(**train_args)

        print(f"\n✅ Training completed successfully!")
        print(f"Best model saved to: runs/detect/{name}/weights/best.pt")
        print(f"Last model saved to: runs/detect/{name}/weights/last.pt")

        # Step 8: Post-training evaluation on all splits
        print(f"\n📊 Step 8: Post-training evaluation on all splits...")

        # Helper function to evaluate on a specific split
        def evaluate_split(split_name, model, data_config):
            try:
                print(f"\n  Evaluating on {split_name} set...")
                if integrate == "initial" and hasattr(model, 'p0_preprocessor'):
                    print(f"    ⚠️ Skipping {split_name} evaluation for P0 integration to avoid channel mismatch")
                    return None
                else:
                    val_results = model.val(data=data_config, split=split_name)
                    if hasattr(val_results, 'box'):
                        map50 = val_results.box.map50
                        map50_95 = val_results.box.map
                    else:
                        # Fallback for different result formats
                        map50 = getattr(val_results, 'map50', 0.0)
                        map50_95 = getattr(val_results, 'map', 0.0)

                    print(f"    {split_name.capitalize()} mAP50: {map50:.4f}")
                    print(f"    {split_name.capitalize()} mAP50-95: {map50_95:.4f}")
                    return {'map50': map50, 'map50_95': map50_95}
            except Exception as e:
                print(f"    ⚠️ {split_name.capitalize()} evaluation failed: {e}")
                return None

        # Evaluate on all splits
        results_summary = {}

        # Validation set
        val_results = evaluate_split('val', model, data_config)
        if val_results:
            results_summary['validation'] = val_results

        # Test set (if available)
        test_results = evaluate_split('test', model, data_config)
        if test_results:
            results_summary['test'] = test_results

        # Training set evaluation (sample to avoid long computation)
        try:
            print(f"\n  Evaluating on training set (sample)...")
            if integrate == "initial" and hasattr(model, 'p0_preprocessor'):
                print(f"    ⚠️ Skipping train evaluation for P0 integration to avoid channel mismatch")
            else:
                # For training set, we might want to use a smaller sample for efficiency
                train_results = model.val(data=data_config, split='train')
                if hasattr(train_results, 'box'):
                    train_map50 = train_results.box.map50
                    train_map50_95 = train_results.box.map
                else:
                    train_map50 = getattr(train_results, 'map50', 0.0)
                    train_map50_95 = getattr(train_results, 'map', 0.0)

                print(f"    Train mAP50: {train_map50:.4f}")
                print(f"    Train mAP50-95: {train_map50_95:.4f}")
                results_summary['train'] = {'map50': train_map50, 'map50_95': train_map50_95}
        except Exception as e:
            print(f"    ⚠️ Train evaluation failed: {e}")

        # Display summary
        print(f"\n📈 Final mAP Summary (GREYSCALE):")
        print("=" * 50)
        if integrate == "initial" and hasattr(model, 'p0_preprocessor'):
            print("⚠️ P0 integration model - evaluations skipped due to channel mismatch")
            print("✓ Training completed successfully. Check training logs for metrics.")
        else:
            for split, metrics in results_summary.items():
                if metrics:
                    print(f"{split.capitalize():>12}: mAP50={metrics['map50']:.4f}, mAP50-95={metrics['map50_95']:.4f}")

            if not results_summary:
                print("No evaluation results available")
        print("=" * 50)

        # Step 9: DINOv3 feature analysis (detailed check)
        print(f"\n🔍 Step 9: DINOv3 integration analysis...")
        print("=" * 60)
        try:
            # Check first module in backbone
            if hasattr(model, 'model') and hasattr(model.model, 'model'):
                first_module = model.model.model[0]
                print(f"\n📋 First backbone layer analysis:")
                print(f"  Module type: {type(first_module).__name__}")
                print(f"  Module class: {type(first_module)}")
                
                # Check if it's DINOv3Backbone
                is_dinov3 = False
                dinov3_type = None
                
                if 'DINOv3Backbone' in str(type(first_module)):
                    is_dinov3 = True
                    dinov3_type = "DINOv3Backbone"
                elif 'DINOv3TripleBackbone' in str(type(first_module)):
                    is_dinov3 = True
                    dinov3_type = "DINOv3TripleBackbone"
                elif 'DINOv3BackboneWithAdapter' in str(type(first_module)):
                    is_dinov3 = True
                    dinov3_type = "DINOv3BackboneWithAdapter"
                elif hasattr(first_module, 'dino_model'):
                    is_dinov3 = True
                    dinov3_type = "DINOv3 (via dino_model attribute)"
                elif 'dinov3' in str(type(first_module)).lower():
                    is_dinov3 = True
                    dinov3_type = "DINOv3 (generic)"
                
                if is_dinov3:
                    print(f"  ✓ DINOv3Backbone detected! Type: {dinov3_type}")
                    
                    # Check DINOv3 model name if available
                    if hasattr(first_module, 'model_name'):
                        print(f"  DINOv3 model: {first_module.model_name}")
                    elif hasattr(first_module, 'dino_model'):
                        print(f"  DINOv3 model found in dino_model attribute")
                    
                    # Check if frozen
                    if hasattr(first_module, 'freeze'):
                        print(f"  Freeze setting: {first_module.freeze}")
                    
                    # Count DINOv3 parameters
                    if hasattr(first_module, 'parameters'):
                        dinov3_params = sum(p.numel() for p in first_module.parameters())
                        print(f"  DINOv3 parameters: {dinov3_params:,}")
                else:
                    print(f"  ❌ DINOv3Backbone NOT detected!")
                    print(f"  ⚠️ First layer is {type(first_module).__name__} (should be DINOv3Backbone)")
            
            # Check all DINOv3 layers in model
            print(f"\n📊 Full DINOv3 layer scan:")
            dinov3_layers = []
            dinov3_params = 0
            dinov3_frozen = 0
            
            for name, module in model.model.named_modules():
                module_type = str(type(module))
                if 'DINOv3' in module_type or 'dinov3' in name.lower():
                    dinov3_layers.append(name)
                    # Count parameters
                    for param_name, param in module.named_parameters(recurse=False):
                        dinov3_params += param.numel()
                        if not param.requires_grad:
                            dinov3_frozen += param.numel()
            
            if dinov3_layers:
                print(f"  ✓ DINOv3 layers found: {len(dinov3_layers)}")
                print(f"  Layer names:")
                for layer_name in dinov3_layers[:10]:  # Show first 10
                    print(f"    - {layer_name}")
                if len(dinov3_layers) > 10:
                    print(f"    ... and {len(dinov3_layers) - 10} more")
                
                print(f"\n  📈 DINOv3 Statistics:")
                print(f"    Total DINOv3 parameters: {dinov3_params:,}")
                print(f"    Frozen DINOv3 parameters: {dinov3_frozen:,}")
                print(f"    Trainable DINOv3 parameters: {dinov3_params - dinov3_frozen:,}")
                if dinov3_params > 0:
                    frozen_percent = (dinov3_frozen / dinov3_params) * 100
                    print(f"    Frozen ratio: {frozen_percent:.1f}%")
            else:
                print(f"  ❌ No DINOv3 layers detected in entire model!")
                print(f"  ⚠️ Model is NOT using DINOv3")
            
            # Check DINOv3-specific attributes
            print(f"\n🔎 DINOv3 Attribute Check:")
            if hasattr(model, 'model') and hasattr(model.model, 'model'):
                first_module = model.model.model[0]
                dinov3_attrs = ['dino_model', 'model_name', 'input_channels', 'output_channels', 'freeze', 'pretrained']
                found_attrs = []
                for attr in dinov3_attrs:
                    if hasattr(first_module, attr):
                        found_attrs.append(attr)
                        value = getattr(first_module, attr)
                        print(f"    ✓ {attr}: {value}")
                
                if len(found_attrs) >= 3:
                    print(f"  ✓ First module has {len(found_attrs)} DINOv3 attributes - likely DINOv3Backbone")
                else:
                    print(f"  ⚠️ First module has only {len(found_attrs)} DINOv3 attributes - might not be DINOv3")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Could not analyze DINOv3 integration: {e}")
            import traceback
            traceback.print_exc()

        # Instructions for further use
        print(f"\n🎯 Next steps:")
        print(f"1. Evaluate on test set:")
        print(f"   python test_model_evaluation.py --model runs/detect/{name}/weights/best.pt --data {data_config} --split test")

        if freeze_dinov3:
            print(f"2. Consider unfreezing DINOv3 for fine-tuning:")
            print(f"   # Load model and unfreeze DINOv3, then continue training with lower LR")

        return results

    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Common causes:")
        print("  - Out of memory (reduce batch size or image size)")
        print("  - DINOv3 model download issues (check internet connection)")
        print("  - Incompatible model configuration")
        import traceback
        traceback.print_exc()
        return None

def print_dinov3_info():
    """Print detailed DINOv3 model information."""
    print("\n📊 DINOv3 Model Size Comparison:")
    print("=" * 80)
    print(f"{'Model':<12} {'Params':<8} {'Dimensions':<11} {'Memory':<10} {'Speed':<8} {'Use Case'}")
    print("-" * 80)
    print(f"{'small':<12} {'21M':<8} {'384':<11} {'~2GB':<10} {'Fast':<8} {'Quick experiments, prototyping'}")
    print(f"{'base':<12} {'86M':<8} {'768':<11} {'~4GB':<10} {'Medium':<8} {'Recommended for most tasks'}")
    print(f"{'large':<12} {'304M':<8} {'1024':<11} {'~8GB':<10} {'Slow':<8} {'High accuracy requirements'}")
    print(f"{'giant':<12} {'1.1B':<8} {'1536':<11} {'~16GB':<10} {'Slowest':<8} {'Research, maximum quality'}")
    print(f"{'sat_large':<12} {'304M':<8} {'1024':<11} {'~8GB':<10} {'Slow':<8} {'Satellite/aerial imagery'}")
    print(f"{'sat_giant':<12} {'1.1B':<8} {'1536':<11} {'~16GB':<10} {'Slowest':<8} {'Max satellite performance'}")
    print("=" * 80)
    print("\n💡 Recommendations:")
    print("  • Start with 'base' for most applications")
    print("  • Use 'small' for quick testing or limited GPU memory")
    print("  • Use 'large' for production where accuracy is critical")
    print("  • Use 'sat_*' variants specifically for satellite/aerial images")
    print("  • Consider 'giant' only if you have ample compute resources\n")

def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv12 Triple Input with DINOv3 backbone - GREYSCALE VERSION',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  # Quick training with small model (greyscale):\n"
               "  python \"train_grey scale.py\" --data dataset.yaml --dinov3-size small --variant s\n\n"
               "  # Production training with large model:\n"
               "  python \"train_grey scale.py\" --data dataset.yaml --dinov3-size large --variant x --epochs 300\n\n"
               "  # Satellite imagery with specialized model:\n"
               "  python \"train_grey scale.py\" --data dataset.yaml --dinov3-size sat_large --integrate p0p3\n\n"
               "For DINOv3 model details, run: python -c \"from train_grey_scale import print_dinov3_info; print_dinov3_info()\""
    )
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset configuration (.yaml file)')
    parser.add_argument('--dinov3-size', type=str,
                       choices=['small', 'base', 'large', 'giant', 'sat_large', 'sat_giant'],
                       default='small',
                       help='DINOv3 model size (default: small)')
    parser.add_argument('--freeze-dinov3', action='store_true', default=True,
                       help='Freeze DINOv3 backbone during training (default: True)')
    parser.add_argument('--unfreeze-dinov3', action='store_true',
                       help='Unfreeze DINOv3 backbone for fine-tuning')
    parser.add_argument('--triple-branches', action='store_true',
                       help='Use separate DINOv3 branches for each input')
    parser.add_argument('--pretrained', type=str,
                       help='Path to pretrained YOLOv12 model (.pt file)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--batch', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640, use 224 for DINOv3 optimal or 640 for YOLO standard)')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience (default: 100)')
    parser.add_argument('--name', type=str, default='yolov12_triple_dinov3_greyscale',
                       help='Experiment name (default: yolov12_triple_dinov3_greyscale)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use (default: 0, use "cpu" for CPU)')
    parser.add_argument('--variant', type=str, choices=['n', 's', 'm', 'l', 'x'], default='s',
                       help='YOLOv12 model variant (default: s)')
    parser.add_argument('--save-period', type=int, default=-1,
                       help='Save weights every N epochs (-1 = only best/last)')
    parser.add_argument('--download-only', action='store_true',
                       help='Only download DINOv3 models without training')
    parser.add_argument('--integrate', type=str, choices=['dinov3', 'initial', 'nodino', 'p3', 'p0p3'],
                       default='initial',
                       help='DINOv3 integration strategy (default: initial)')

    args = parser.parse_args()

    # Handle freeze/unfreeze logic
    freeze_dinov3 = args.freeze_dinov3 and not args.unfreeze_dinov3

    # Validate input files
    if not args.download_only and not Path(args.data).exists():
        raise FileNotFoundError(f"Data config not found: {args.data}")

    if args.pretrained and not Path(args.pretrained).exists():
        raise FileNotFoundError(f"Pretrained model not found: {args.pretrained}")

    # Download only mode
    if args.download_only:
        print("Downloading DINOv3 models...")
        from download_dinov3 import DINOv3Downloader
        downloader = DINOv3Downloader()
        success, _ = downloader.download_model(args.dinov3_size, method="auto")
        if success:
            print(f"✅ Downloaded DINOv3 {args.dinov3_size}")
            # Test integration
            downloader.test_integration(args.dinov3_size)
        return

    # Run training
    train_triple_dinov3_greyscale(
        data_config=args.data,
        dinov3_size=args.dinov3_size,
        freeze_dinov3=freeze_dinov3,
        use_triple_branches=args.triple_branches,
        pretrained_path=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        name=args.name,
        device=args.device,
        integrate=args.integrate,
        variant=args.variant,
        save_period=args.save_period
    )

if __name__ == "__main__":
    main()
