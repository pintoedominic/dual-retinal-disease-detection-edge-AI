# verify_aptos_setup.py - Quick verification of APTOS dataset
import os
import pandas as pd

# Your APTOS directory structure
APTOS_ROOT = r"C:\Dominic\Datasets Project\APTOS DR\archive"
APTOS_TRAIN = os.path.join(APTOS_ROOT, "train_images")
APTOS_VAL = os.path.join(APTOS_ROOT, "val_images")
APTOS_TEST = os.path.join(APTOS_ROOT, "test_images")
APTOS_TRAIN_CSV = os.path.join(APTOS_ROOT, "train_1.csv")
APTOS_VAL_CSV = os.path.join(APTOS_ROOT, "valid.csv")
APTOS_TEST_CSV = os.path.join(APTOS_ROOT, "test.csv")

print("ðŸ” APTOS Dataset Verification")
print("=" * 50)

# Check directories
print(f"ðŸ“‚ Root directory: {APTOS_ROOT}")
print(f"   Exists: {'âœ…' if os.path.exists(APTOS_ROOT) else 'âŒ'}")

directories = [
    ("Train Images", APTOS_TRAIN),
    ("Val Images", APTOS_VAL), 
    ("Test Images", APTOS_TEST)
]

csv_files = [
    ("Train CSV", APTOS_TRAIN_CSV),
    ("Val CSV", APTOS_VAL_CSV),
    ("Test CSV", APTOS_TEST_CSV)
]

print("\nðŸ“‚ Checking directories:")
for name, path in directories:
    exists = os.path.exists(path)
    print(f"   {name}: {'âœ…' if exists else 'âŒ'} ({path})")
    if exists:
        files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"      Images found: {len(files)}")
        if files:
            print(f"      Sample: {files[0]}")

print("\nðŸ“„ Checking CSV files:")
for name, path in csv_files:
    exists = os.path.exists(path)
    print(f"   {name}: {'âœ…' if exists else 'âŒ'} ({path})")
    if exists:
        try:
            df = pd.read_csv(path)
            print(f"      Rows: {len(df)}")
            print(f"      Columns: {list(df.columns)}")
            
            # Check for different possible column names
            if 'diagnosis' in df.columns:
                print(f"      DR distribution: {df['diagnosis'].value_counts().sort_index().to_dict()}")
            elif 'level' in df.columns:
                print(f"      DR distribution: {df['level'].value_counts().sort_index().to_dict()}")
            
        except Exception as e:
            print(f"      âŒ Error: {e}")

print("\n" + "=" * 50)
print("âœ… Verification complete!")
print("If everything looks good, you can run the transfer learning pipeline.")