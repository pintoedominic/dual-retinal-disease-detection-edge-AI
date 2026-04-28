import os, re
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- paths: edit if your run names/locations differ
BIN_OUT = r"C:\Dominic\Datasets Project\ROP_TWO_STAGE\STAGE_A_BIN"
wA = r"runs\classify\ROP_stageA_bin\weights\best.pt"        # Stage A weights
wB = r"runs\classify\ROP_stageB_severity\weights\best.pt"   # Stage B weights

# --- map DG -> 4 classes
def four_class_from_filename(fname):
    m = re.search(r"DG(\d+)", fname)
    if not m:
        return None
    dg = int(m.group(1))
    if 0 <= dg <= 3:   return "no_ROP"
    if 4 <= dg <= 6:   return "mild"
    if 7 <= dg <= 10:  return "moderate"
    if 11 <= dg <= 13: return "severe"
    return None

# canonical class order
four_classes = ["no_ROP", "mild", "moderate", "severe"]
cls_to_idx = {c:i for i,c in enumerate(four_classes)}

# --- load models
A = YOLO(wA)  # binary gate
B = YOLO(wB)  # severity on ROP

# --- iterate Stage-A TEST set (contains both no_ROP and ROP images)
test_root = os.path.join(BIN_OUT, "test")

y_true, y_pred = [], []

for sub in os.listdir(test_root):
    sub_dir = os.path.join(test_root, sub)
    if not os.path.isdir(sub_dir): 
        continue
    for fname in os.listdir(sub_dir):
        if not fname.lower().endswith((".jpg",".jpeg",".png")):
            continue

        fpath = os.path.join(sub_dir, fname)

        # ground truth (from filename DG code)
        gt = four_class_from_filename(fname)
        if gt is None: 
            continue
        y_true.append(cls_to_idx[gt])

        # --- Stage A: No ROP vs ROP
        rA = A(fpath, imgsz=224, verbose=False)[0]
        predA_idx = int(rA.probs.top1)
        predA_name = A.names[predA_idx]  # expect {"no_ROP","ROP"} (check A.names if unsure)

        if predA_name == "no_ROP":
            pred_final = "no_ROP"
        else:
            # --- Stage B: Mild/Moderate/Severe
            rB = B(fpath, imgsz=224, verbose=False)[0]
            predB_idx = int(rB.probs.top1)
            pred_final = B.names[predB_idx]  # expect {"mild","moderate","severe"}

        y_pred.append(cls_to_idx[pred_final])

# --- metrics
acc = accuracy_score(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(four_classes))))
print(f"\nEnd-to-end 4-class Test Accuracy: {acc*100:.2f}%")
print("\nConfusion Matrix (rows=true, cols=pred):")
print(four_classes)
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=four_classes, digits=4))

