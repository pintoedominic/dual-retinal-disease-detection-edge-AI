# clean_aptos_transfer_learning.py
# Clean implementation of APTOS DR to ROP transfer learning

import os
import shutil
import pandas as pd
import re
from ultralytics import YOLO
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("ðŸš€ Starting APTOSâ†’ROP Transfer Learning Pipeline")
print("=" * 60)

# -----------------------------
# Configuration
# -----------------------------
# APTOS dataset paths (updated structure)
APTOS_ROOT = r"C:\Dominic\Datasets Project\APTOS DR\archive"
APTOS_TRAIN = os.path.join(APTOS_ROOT, "train_images")
APTOS_VAL = os.path.join(APTOS_ROOT, "val_images")
APTOS_TEST = os.path.join(APTOS_ROOT, "test_images")
APTOS_TRAIN_CSV = os.path.join(APTOS_ROOT, "train_1.csv")
APTOS_VAL_CSV = os.path.join(APTOS_ROOT, "valid.csv")
APTOS_TEST_CSV = os.path.join(APTOS_ROOT, "test.csv")

# Output directories
APTOS_OUT = r"C:\Dominic\Datasets Project\APTOS_PROCESSED"
MIXED_OUT = r"C:\Dominic\Datasets Project\MIXED_APTOS_ROP"

# Existing ROP datasets
BIN_OUT = r"C:\Dominic\Datasets Project\ROP_TWO_STAGE\STAGE_A_BIN"
SEV_OUT = r"C:\Dominic\Datasets Project\ROP_TWO_STAGE\STAGE_B_SEVERITY"

# Training settings
DEVICE = 0 if torch.cuda.is_available() else "cpu"
IMGSZ = 224
BATCH = 32
WORKERS = 0

print(f"ðŸ”§ Configuration:")
print(f"   Device: {DEVICE}")
print(f"   Image size: {IMGSZ}")
print(f"   Batch size: {BATCH}")

# -----------------------------
# Step 1: Verify APTOS Dataset
# -----------------------------
def verify_aptos_dataset():
    """Verify APTOS dataset structure and files"""
    print("\nðŸ” Step 1: Verifying APTOS Dataset")
    print("-" * 40)
    
    # Check root directory
    if not os.path.exists(APTOS_ROOT):
        print(f"âŒ APTOS root directory not found: {APTOS_ROOT}")
        return False
    
    print(f"âœ… Root directory found: {APTOS_ROOT}")
    
    # Check image directories
    image_dirs = [
        ("Train", APTOS_TRAIN),
        ("Validation", APTOS_VAL),
        ("Test", APTOS_TEST)
    ]
    
    dir_status = {}
    for name, path in image_dirs:
        exists = os.path.exists(path)
        print(f"   {name} images: {'âœ…' if exists else 'âŒ'} ({path})")
        if exists:
            images = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"      Found {len(images)} images")
            dir_status[name.lower()] = len(images)
        else:
            dir_status[name.lower()] = 0
    
    # Check CSV files
    csv_files = [
        ("Train CSV", APTOS_TRAIN_CSV),
        ("Val CSV", APTOS_VAL_CSV),
        ("Test CSV", APTOS_TEST_CSV)
    ]
    
    csv_status = {}
    for name, path in csv_files:
        exists = os.path.exists(path)
        print(f"   {name}: {'âœ…' if exists else 'âŒ'} ({path})")
        if exists:
            try:
                df = pd.read_csv(path)
                print(f"      {len(df)} rows, columns: {list(df.columns)}")
                csv_status[name.lower().replace(' csv', '')] = len(df)
            except Exception as e:
                print(f"      âŒ Error reading: {e}")
                csv_status[name.lower().replace(' csv', '')] = 0
        else:
            csv_status[name.lower().replace(' csv', '')] = 0
    
    # Summary
    total_images = sum(dir_status.values())
    total_csv_rows = sum(csv_status.values())
    
    print(f"\nðŸ“Š Summary:")
    print(f"   Total images: {total_images}")
    print(f"   Total CSV rows: {total_csv_rows}")
    
    if total_images > 0 and csv_status.get('train', 0) > 0:
        print("âœ… APTOS dataset verification passed!")
        return True
    else:
        print("âŒ APTOS dataset verification failed!")
        return False

# -----------------------------
# Step 2: Process APTOS Dataset
# -----------------------------
def process_aptos_dataset():
    """Process APTOS dataset into YOLO format"""
    print("\nðŸ”„ Step 2: Processing APTOS Dataset")
    print("-" * 40)
    
    def map_dr_grade(grade):
        """Map DR grades to class names"""
        if grade == 0:
            return "no_DR"
        elif grade == 1:
            return "mild_DR"
        elif grade == 2:
            return "moderate_DR"
        elif grade >= 3:  # Combine severe (3) and proliferative (4)
            return "severe_DR"
    
    # Process each available split
    splits_to_process = []
    
    # Check train split
    if os.path.exists(APTOS_TRAIN_CSV) and os.path.exists(APTOS_TRAIN):
        splits_to_process.append(("train", APTOS_TRAIN_CSV, APTOS_TRAIN))
    
    # Check val split
    if os.path.exists(APTOS_VAL_CSV) and os.path.exists(APTOS_VAL):
        splits_to_process.append(("val", APTOS_VAL_CSV, APTOS_VAL))
    
    # Check test split
    if os.path.exists(APTOS_TEST_CSV) and os.path.exists(APTOS_TEST):
        splits_to_process.append(("test", APTOS_TEST_CSV, APTOS_TEST))
    
    if not splits_to_process:
        print("âŒ No valid APTOS splits found to process!")
        return False
    
    total_processed = 0
    
    for split_name, csv_path, img_dir in splits_to_process:
        print(f"\nðŸ“‚ Processing {split_name} split...")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} entries from CSV")
        
        # Determine column names (handle different CSV formats)
        id_col = None
        label_col = None
        
        for col in df.columns:
            if 'id' in col.lower() or 'image' in col.lower():
                id_col = col
            if 'diagnosis' in col.lower() or 'level' in col.lower():
                label_col = col
        
        if id_col is None or label_col is None:
            print(f"   âŒ Could not find ID or label columns in {csv_path}")
            print(f"   Available columns: {list(df.columns)}")
            continue
        
        print(f"   Using ID column: {id_col}, Label column: {label_col}")
        
        # Map grades to classes
        df['class'] = df[label_col].apply(map_dr_grade)
        class_counts = df['class'].value_counts()
        print(f"   Class distribution: {class_counts.to_dict()}")
        
        # Create class directories
        for class_name in df['class'].unique():
            class_dir = os.path.join(APTOS_OUT, split_name, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        # Copy images
        split_counts = {cls: 0 for cls in df['class'].unique()}
        
        for _, row in df.iterrows():
            # Try to find image with different extensions
            base_name = str(row[id_col])
            found_image = False
            
            for ext in ['.png', '.jpg', '.jpeg']:
                src_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(src_path):
                    dst_dir = os.path.join(APTOS_OUT, split_name, row['class'])
                    dst_path = os.path.join(dst_dir, base_name + ext)
                    shutil.copy2(src_path, dst_path)
                    split_counts[row['class']] += 1
                    total_processed += 1
                    found_image = True
                    break
            
            if not found_image:
                print(f"   âš ï¸  Image not found: {base_name}")
        
        print(f"   Processed: {split_counts}")
    
    print(f"\nâœ… APTOS processing complete: {total_processed} images processed")
    print(f"   Output directory: {APTOS_OUT}")
    
    # Create test split if missing
    if not os.path.exists(os.path.join(APTOS_OUT, "test")):
        print("\nðŸ“‚ Creating test split from validation...")
        create_test_from_val()
    
    return True

def create_test_from_val():
    """Create test split by moving half of validation images"""
    val_dir = os.path.join(APTOS_OUT, "val")
    test_dir = os.path.join(APTOS_OUT, "test")
    
    if not os.path.exists(val_dir):
        print("   âš ï¸  No validation directory found")
        return
    
    for class_name in os.listdir(val_dir):
        class_val_dir = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_val_dir):
            continue
        
        class_test_dir = os.path.join(test_dir, class_name)
        os.makedirs(class_test_dir, exist_ok=True)
        
        # Move every other image to test
        val_images = [f for f in os.listdir(class_val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        test_images = val_images[::2]  # Every other image
        
        for img in test_images:
            src = os.path.join(class_val_dir, img)
            dst = os.path.join(class_test_dir, img)
            shutil.move(src, dst)
        
        print(f"   {class_name}: moved {len(test_images)} to test")

# -----------------------------
# Step 3: Create Mixed Dataset
# -----------------------------
def create_mixed_dataset():
    """Create mixed APTOS+ROP binary dataset"""
    print("\nðŸ”„ Step 3: Creating Mixed APTOS+ROP Dataset")
    print("-" * 40)
    
    for split in ["train", "val", "test"]:
        print(f"\nðŸ“‚ Processing {split} split...")
        
        # Create binary class directories
        no_path_dir = os.path.join(MIXED_OUT, split, "no_pathology")
        pathology_dir = os.path.join(MIXED_OUT, split, "pathology")
        os.makedirs(no_path_dir, exist_ok=True)
        os.makedirs(pathology_dir, exist_ok=True)
        
        counts = {"no_pathology": 0, "pathology": 0}
        
        # Process APTOS data
        aptos_split_dir = os.path.join(APTOS_OUT, split)
        if os.path.exists(aptos_split_dir):
            for class_name in os.listdir(aptos_split_dir):
                src_class_dir = os.path.join(aptos_split_dir, class_name)
                if not os.path.isdir(src_class_dir):
                    continue
                
                # Map to binary classes
                dst_dir = no_path_dir if class_name == "no_DR" else pathology_dir
                binary_class = "no_pathology" if class_name == "no_DR" else "pathology"
                
                for img_file in os.listdir(src_class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = os.path.join(src_class_dir, img_file)
                        dst_path = os.path.join(dst_dir, f"APTOS_{img_file}")
                        shutil.copy2(src_path, dst_path)
                        counts[binary_class] += 1
        
        # Process ROP data
        rop_split_dir = os.path.join(BIN_OUT, split)
        if os.path.exists(rop_split_dir):
            for class_name in os.listdir(rop_split_dir):
                src_class_dir = os.path.join(rop_split_dir, class_name)
                if not os.path.isdir(src_class_dir):
                    continue
                
                # Map to binary classes
                dst_dir = no_path_dir if class_name == "no_ROP" else pathology_dir
                binary_class = "no_pathology" if class_name == "no_ROP" else "pathology"
                
                for img_file in os.listdir(src_class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = os.path.join(src_class_dir, img_file)
                        dst_path = os.path.join(dst_dir, f"ROP_{img_file}")
                        shutil.copy2(src_path, dst_path)
                        counts[binary_class] += 1
        
        print(f"   {split}: {counts}")
    
    print(f"âœ… Mixed dataset created at: {MIXED_OUT}")
    return True

# -----------------------------
# Step 4: Transfer Learning Training
# -----------------------------
def train_stage1_aptos_pretrain():
    """Stage 1: Pre-train on APTOS DR dataset"""
    print("\nðŸŽ¯ Step 4a: Stage 1 - APTOS Pre-training")
    print("-" * 40)
    
    model = YOLO("yolo11m-cls.pt")
    
    print("   Starting APTOS pre-training...")
    results = model.train(
        data=APTOS_OUT,
        epochs=30,
        patience=10,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        optimizer="AdamW",
        lr0=1e-3,
        dropout=0.2,
        augment=True,
        save=True,
        project="runs/classify",
        name="APTOS_pretrain",
        exist_ok=True,
        verbose=True
    )
    
    weights_path = "runs/classify/APTOS_pretrain/weights/best.pt"
    print(f"âœ… Stage 1 complete. Weights: {weights_path}")
    return weights_path

def train_stage2_mixed():
    """Stage 2: Mixed APTOS+ROP training"""
    print("\nðŸŽ¯ Step 4b: Stage 2 - Mixed Training")
    print("-" * 40)
    
    stage1_weights = "runs/classify/APTOS_pretrain/weights/best.pt"
    model = YOLO(stage1_weights)
    
    print("   Starting mixed APTOS+ROP training...")
    results = model.train(
        data=MIXED_OUT,
        epochs=25,
        patience=8,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        optimizer="AdamW",
        lr0=5e-4,
        dropout=0.3,
        augment=True,
        save=True,
        project="runs/classify",
        name="Mixed_APTOS_ROP",
        exist_ok=True,
        verbose=True
    )
    
    weights_path = "runs/classify/Mixed_APTOS_ROP/weights/best.pt"
    print(f"âœ… Stage 2 complete. Weights: {weights_path}")
    return weights_path

def train_stage3_rop_binary():
    """Stage 3: ROP binary fine-tuning"""
    print("\nðŸŽ¯ Step 4c: Stage 3 - ROP Binary Fine-tuning")
    print("-" * 40)
    
    stage2_weights = "runs/classify/Mixed_APTOS_ROP/weights/best.pt"
    model = YOLO(stage2_weights)
    
    print("   Starting ROP binary fine-tuning...")
    results = model.train(
        data=BIN_OUT,
        epochs=40,
        patience=12,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        optimizer="AdamW",
        lr0=1e-4,
        dropout=0.2,
        augment=True,
        save=True,
        project="runs/classify",
        name="ROP_binary_transfer",
        exist_ok=True,
        verbose=True
    )
    
    weights_path = "runs/classify/ROP_binary_transfer/weights/best.pt"
    print(f"âœ… Stage 3 complete. Weights: {weights_path}")
    return weights_path

def train_stage4_rop_severity():
    """Stage 4: ROP severity transfer"""
    print("\nðŸŽ¯ Step 4d: Stage 4 - ROP Severity Transfer")
    print("-" * 40)
    
    stage3_weights = "runs/classify/ROP_binary_transfer/weights/best.pt"
    model = YOLO(stage3_weights)
    
    print("   Starting ROP severity transfer learning...")
    results = model.train(
        data=SEV_OUT,
        epochs=35,
        patience=10,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        optimizer="AdamW",
        lr0=8e-5,
        dropout=0.3,
        augment=True,
        save=True,
        project="runs/classify",
        name="ROP_severity_transfer",
        exist_ok=True,
        verbose=True
    )
    
    weights_path = "runs/classify/ROP_severity_transfer/weights/best.pt"
    print(f"âœ… Stage 4 complete. Weights: {weights_path}")
    return weights_path

# -----------------------------
# Step 5: Evaluation
# -----------------------------
def evaluate_models():
    """Evaluate and compare models"""
    print("\nðŸ“Š Step 5: Model Evaluation")
    print("-" * 40)
    
    def eval_single_model(weights_path, data_root, model_name):
        """Evaluate single model"""
        print(f"\nðŸ” Evaluating {model_name}...")
        
        model = YOLO(weights_path)
        names = model.names
        idx2name = [names[i] for i in sorted(names.keys())]
        
        y_true, y_pred = [], []
        test_root = os.path.join(data_root, "test")
        
        for cls_idx, cls_name in enumerate(idx2name):
            cls_dir = os.path.join(test_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(cls_dir, fname)
                res = model(img_path, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
                pred_idx = int(res.probs.top1)
                y_true.append(cls_idx)
                y_pred.append(pred_idx)
        
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        rep = classification_report(y_true, y_pred, target_names=idx2name, digits=4)
        
        print(f"   Accuracy: {acc*100:.2f}%")
        print(f"   Confusion Matrix:\n{cm}")
        print(f"   Report:\n{rep}")
        
        return acc, cm, rep
    
    # Compare binary classification
    print("\nðŸ†š Binary Classification Comparison:")
    print("="*50)
    
    print("Original Model (No Transfer):")
    orig_acc, _, _ = eval_single_model("runs/classify/ROP_stageA_bin/weights/best.pt", BIN_OUT, "Original Binary")
    
    print("Transfer Learning Model:")
    transfer_acc, _, _ = eval_single_model("runs/classify/ROP_binary_transfer/weights/best.pt", BIN_OUT, "Transfer Binary")
    
    binary_improvement = (transfer_acc - orig_acc) * 100
    print(f"\nðŸ“ˆ Binary Improvement: {binary_improvement:+.2f} percentage points")
    
    # Compare severity classification
    print("\nðŸ†š Severity Classification Comparison:")
    print("="*50)
    
    print("Original Model (No Transfer):")
    orig_sev_acc, _, _ = eval_single_model("runs/classify/ROP_stageB_severity/weights/best.pt", SEV_OUT, "Original Severity")
    
    print("Transfer Learning Model:")
    transfer_sev_acc, _, _ = eval_single_model("runs/classify/ROP_severity_transfer/weights/best.pt", SEV_OUT, "Transfer Severity")
    
    severity_improvement = (transfer_sev_acc - orig_sev_acc) * 100
    print(f"\nðŸ“ˆ Severity Improvement: {severity_improvement:+.2f} percentage points")
    
    # Summary
    print(f"\nðŸŽ¯ TRANSFER LEARNING SUMMARY:")
    print(f"Binary: {orig_acc*100:.2f}% â†’ {transfer_acc*100:.2f}% ({binary_improvement:+.2f}%)")
    print(f"Severity: {orig_sev_acc*100:.2f}% â†’ {transfer_sev_acc*100:.2f}% ({severity_improvement:+.2f}%)")

# -----------------------------
# Main Execution
# -----------------------------
def main():
    """Execute complete pipeline"""
    print("ðŸš€ APTOSâ†’ROP Transfer Learning Pipeline")
    print("=" * 60)
    
    # Step 1: Verify dataset
    if not verify_aptos_dataset():
        print("âŒ Dataset verification failed. Please check your APTOS setup.")
        return False
    
    # Step 2: Process APTOS dataset
    if not process_aptos_dataset():
        print("âŒ APTOS processing failed.")
        return False
    
    # Step 3: Create mixed dataset
    if not create_mixed_dataset():
        print("âŒ Mixed dataset creation failed.")
        return False
    
    # Step 4: Transfer learning (4 stages)
    print("\nðŸŽ¯ Starting 4-Stage Transfer Learning...")
    stage1_weights = train_stage1_aptos_pretrain()
    stage2_weights = train_stage2_mixed()
    stage3_weights = train_stage3_rop_binary()
    stage4_weights = train_stage4_rop_severity()
    
    # Step 5: Evaluation
    evaluate_models()
    
    print("\nâœ… Transfer Learning Pipeline Complete!")
    print(f"Final models saved:")
    print(f"  Binary: {stage3_weights}")
    print(f"  Severity: {stage4_weights}")
    
    return True

if __name__ == "__main__":
    # Execute main pipeline
    success = main()
    if success:
        print("\nðŸŽ‰ All done! Check the results above.")
    else:
        print("\nâŒ Pipeline failed. Check error messages above.")