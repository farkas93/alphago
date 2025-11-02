#!/usr/bin/env python3
"""
Test MLflow artifact upload with different file sizes
"""
import mlflow
import tempfile
import os
import random
import string
from config import MLFLOW_DEFAULT_URI

def test_mlflow_multipart_upload():
    """Test MLflow artifact upload with different file sizes"""
    
    # Configuration
    MLFLOW_URI = MLFLOW_DEFAULT_URI
    MODEL_PATH = "/mnt/Projects/machine_learning/rl/alphago/checkpoints/dummy_artifact_test.pth"  # Your model path
    
    print(f"Testing MLflow multipart upload to: {MLFLOW_URI}")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at: {MODEL_PATH}")
        print("Please update the MODEL_PATH variable with the correct path to your .pth file")
        return False
    
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Model file size: {model_size_mb:.2f} MB")
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(MLFLOW_URI)
        print(f"✓ Set tracking URI: {MLFLOW_URI}")
        
        # Set experiment
        experiment_name = "mlflow-tests"
        mlflow.set_experiment(experiment_name)
        print(f"✓ Set experiment: {experiment_name}")
        
        # Start a run
        with mlflow.start_run(run_name="multipart_test") as run:
            print(f"✓ Started run: {run.info.run_id}")
            print(f"Current Run Artifact URI: {mlflow.get_artifact_uri()}")
            print(f"AWS_ACCESS_KEY_ID: {os.environ.get('AWS_ACCESS_KEY_ID', 'Not Set')}")
            print(f"AWS_SECRET_ACCESS_KEY: {'Set' if os.environ.get('AWS_SECRET_ACCESS_KEY') else 'Not Set'}")
            print(f"AWS_SESSION_TOKEN: {'Set' if os.environ.get('AWS_SESSION_TOKEN') else 'Not Set'}")
            
            # 1. Create and upload a small test file (10KB)
            print("\nTest 1: Small file upload (10KB)")
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(''.join(random.choices(string.ascii_letters, k=10240)))  # 10KB of random text
                small_file = f.name
            
            print(f"  Created small test file: {small_file}")
            print("  Uploading small file...")
            mlflow.log_artifact(small_file, artifact_path="test_small")
            print("✓ Successfully uploaded small file!")
            os.unlink(small_file)
            
            # 2. Create and upload a large test file (100MB)
            print("\nTest 2: Large file upload (100MB)")
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
                # Write 100MB of random bytes (in chunks to avoid memory issues)
                chunk_size = 1024 * 1024  # 1MB chunks
                for _ in range(100):  # 100 chunks of 1MB = 100MB
                    f.write(os.urandom(chunk_size))
                large_file = f.name
            
            large_file_size_mb = os.path.getsize(large_file) / (1024 * 1024)
            print(f"  Created large test file: {large_file} ({large_file_size_mb:.2f} MB)")
            print("  Uploading large file (this may take a while)...")
            mlflow.log_artifact(large_file, artifact_path="test_large")
            print("✓ Successfully uploaded large file!")
            os.unlink(large_file)
            
            # 3. Upload the model file
            print(f"\nTest 3: Model file upload ({model_size_mb:.2f} MB)")
            print(f"  Uploading model from: {MODEL_PATH}")
            mlflow.log_artifact(MODEL_PATH, artifact_path="model")
            print("✓ Successfully uploaded model file!")
            
            # Get run info
            print(f"\n{'=' * 70}")
            print(f"✅ ALL TESTS PASSED!")
            print(f"{'=' * 70}")
            print(f"🏃 View run at: {MLFLOW_URI}#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
            
    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"❌ TEST FAILED!")
        print(f"{'=' * 70}")
        print(f"Error: {type(e).__name__}")
        print(f"Message: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_mlflow_multipart_upload()
    exit(0 if success else 1)