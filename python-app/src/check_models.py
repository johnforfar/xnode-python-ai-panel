#!/usr/bin/env python3
import os
import sys

def check_models():
    models_dir = "/models"
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory '{models_dir}' does not exist")
        return False
    
    # Check for Llama model
    llama_model_dirs = [
        os.path.join(models_dir, "meta-llama", "Llama-3.2-1B"),
        os.path.join(models_dir, "Llama-3.2-1B")
    ]
    
    llama_found = False
    for llama_dir in llama_model_dirs:
        if os.path.exists(llama_dir):
            print(f"✅ Found Llama-3.2-1B model at: {llama_dir}")
            llama_found = True
            break
    
    if not llama_found:
        print("❌ Llama-3.2-1B model not found")
    
    # Check for CSM model
    csm_model_dirs = [
        os.path.join(models_dir, "sesame", "csm-1b"),
        os.path.join(models_dir, "csm-1b")
    ]
    
    csm_found = False
    for csm_dir in csm_model_dirs:
        if os.path.exists(csm_dir):
            print(f"✅ Found CSM-1B model at: {csm_dir}")
            csm_found = True
            break
    
    if not csm_found:
        print("❌ CSM-1B model not found")
    
    # Check for mimi model
    mimi_dirs = [
        os.path.join(models_dir, "kyutai", "mimi"),
        os.path.join(models_dir, "mimi")
    ]
    
    mimi_found = False
    for mimi_dir in mimi_dirs:
        if os.path.exists(mimi_dir):
            print(f"✅ Found Mimi model at: {mimi_dir}")
            mimi_found = True
            break
    
    if not mimi_found:
        print("❌ Mimi model not found")
    
    return llama_found and csm_found and mimi_found

if __name__ == "__main__":
    print("Checking for required models...")
    if check_models():
        print("All required models found!")
        sys.exit(0)
    else:
        print("Some required models are missing. Please check the model directory.")
        sys.exit(1) 