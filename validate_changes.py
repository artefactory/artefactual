#!/usr/bin/env python
"""Validation script to test the asset loading changes."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_assets_dir():
    """Test that assets directory exists and contains files."""
    from artefactual.utils.io import _get_assets_dir
    
    assets_dir = _get_assets_dir()
    print(f"✓ Assets directory: {assets_dir}")
    
    if not assets_dir.exists():
        print(f"✗ Assets directory does not exist!")
        return False
    
    json_files = list(assets_dir.glob("*.json"))
    print(f"✓ Found {len(json_files)} JSON files")
    
    return True

def test_load_weights():
    """Test loading weights from built-in models."""
    from artefactual.utils.io import load_weights, MODEL_WEIGHT_MAP
    
    for model_name in MODEL_WEIGHT_MAP.keys():
        try:
            weights = load_weights(model_name)
            print(f"✓ Loaded weights for {model_name}")
        except Exception as e:
            print(f"✗ Failed to load weights for {model_name}: {e}")
            return False
    
    return True

def test_load_calibration():
    """Test loading calibration from built-in models."""
    from artefactual.utils.io import load_calibration, MODEL_CALIBRATION_MAP
    
    for model_name in MODEL_CALIBRATION_MAP.keys():
        try:
            calibration = load_calibration(model_name)
            print(f"✓ Loaded calibration for {model_name}")
        except Exception as e:
            print(f"✗ Failed to load calibration for {model_name}: {e}")
            return False
    
    return True

def test_data_api():
    """Test that data model is exposed."""
    try:
        import artefactual.data
        
        # Check __all__
        expected = ["TokenLogprob", "Completion", "Result", "Dataset"]
        for name in expected:
            if name not in artefactual.data.__all__:
                print(f"✗ {name} not in artefactual.data.__all__")
                return False
        
        print(f"✓ Data API exposes: {', '.join(expected)}")
        return True
    except Exception as e:
        print(f"✗ Failed to test data API: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Validating asset loading changes")
    print("=" * 60)
    
    tests = [
        ("Assets directory", test_assets_dir),
        ("Load weights", test_load_weights),
        ("Load calibration", test_load_calibration),
        ("Data API", test_data_api),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
