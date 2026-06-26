import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models import FCModel
from train_utils import train, train_perlayer_bound, test
from kdtrain import train_knowledge_distillation, kd_train_perlayer_bound

#for saving model wieghts
import os, re

def _safe(s: str) -> str:
    # filesystem-safe slug
    s = str(s)
    s = s.replace(" ", "")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]", "", s)
    return s


def train_models(architectures, activations, degree, last_bern, learning_rate,
                 weight_decay, train_loader, val_loader, test_loader,epochs,seed,header = False, file_name = None):
    all_results = []

    for layer_sizes in architectures:
        for act in activations:
            print("=" * 80)
            print(f"Architecture={layer_sizes}, act={act}")

            torch.manual_seed(seed)
            
            model = FCModel(
                layer_sizes=layer_sizes,
                degree=degree,
                act=act,
                last_bern=last_bern,
                dropout=0.0,
            ).to(device)
            
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            

            # -----------------------------
            # Bernstein: staged calibration
            # -----------------------------
            if act == "bern":
                print("→ Running per-layer Bernstein calibration")

                train_perlayer_bound(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    label_smoothing=0.0,
                    device=device,
                    warmup_epochs=1,
                    stage_epochs=3,
                    max_calib_batches=1000,
                )

            # -----------------------------
            # Full training (your function)
            # -----------------------------
            print("→ Training model")
            val_acc, train_loss, val_loss = train(
                model,
                train_loader,
                val_loader,
                test_loader,
                epochs=epochs,
                device=device,
                label_smoothing=0.0,
                optimizer=optimizer,
                patience=20,
                act = act
            )

            # -----------------------------
            # Final testing (your function)
            # -----------------------------
            test_loss, test_acc = test(
                model,
                test_loader,
                device=device
            )
            # ---- SAVE WEIGHTS (one file per model) ----
            save_dir = "ct_saved_models"
            os.makedirs(save_dir, exist_ok=True)

            arch_str = "x".join(map(str, layer_sizes))  # e.g., 54x12x7
            deg_str  = f"deg{degree}" if act == "bern" else "degNA"
            last_str = f"lastbern{int(bool(last_bern))}"
            lr_str   = f"lr{learning_rate:g}"
            wd_str   = f"wd{weight_decay:g}"
            seed_str = f"seed{seed}"

            pth_name = _safe(f"fc_{arch_str}_{act}_{deg_str}_{last_str}_{lr_str}_{wd_str}_{seed_str}.pth")
            pth_path = os.path.join(save_dir, pth_name)

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": layer_sizes,
                    "activation": act,
                    "degree": degree if act == "bern" else None,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "val_acc": float(val_acc),
                    "test_acc": float(test_acc),
                },
                pth_path
            )
            results = []
            results.append({
                "activation": act,
                "layer_sizes": str(layer_sizes),
                "degree": degree if act == "bern" else None,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "ce_train_loss": train_loss,
                "val_accuracy_%": float(val_acc),
                "val_loss": float(val_loss),
                "test_loss": float(test_loss),
                "test_accuracy_%": float(test_acc),
            })
            df = pd.DataFrame(results)
            df.to_csv(file_name,index=False, mode='a', header=header)
            
            all_results.append({
                "activation": act,
                "layer_sizes": str(layer_sizes),
                "degree": degree if act == "bern" else None,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "ce_train_loss": train_loss,
                "val_accuracy_%": float(val_acc),
                "val_loss": float(val_loss),
                "test_loss": float(test_loss),
                "test_accuracy_%": float(test_acc),
            })
            
    all_df = pd.DataFrame(all_results)
    print("\n===== Bernstein vs ReLU =====")
    print(all_df.to_string(index=False))  
            
    return all_df

def kd_train_models(teacher, architectures, activations, alphas, temps, degree, 
                    last_bern, learning_rate, weight_decay, train_loader, val_loader, test_loader,
                    epochs, seed, header = False, file_name = None):
    all_results = []
    for layer_sizes in architectures:
        print("=" * 80)
        for alpha in alphas:
            Ts = [1.0] if alpha == 0.0 else temps  # T irrelevant if no KD
            for T in Ts:
                for act in activations:
                    print(f"Architecture={layer_sizes}")
                    print(f"\n--- act={act}, alpha={alpha}, T={T} ---")

                    torch.manual_seed(seed)

                    student = FCModel(
                        layer_sizes=layer_sizes,
                        degree=degree,
                        act=act,
                        last_bern=last_bern
                    ).to(device)
                    optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=weight_decay)

                    # -----------------------------
                    # Bernstein: staged calibration
                    # -----------------------------
                    if act == "bern":
                        print("→ Running per-layer Bernstein calibration")
                        kd_train_perlayer_bound(
                            student=student,
                            teacher=teacher,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            label_smoothing=0.0,
                            T=T,
                            alpha=alpha,
                            device=device,
                            warmup_epochs=2,
                            stage_epochs=5,
                            max_calib_batches=1000,
                        )
                        

                    # -----------------------------
                    # Full training (your function)
                    # -----------------------------
                    print("→ Training model")
                    val_acc, train_loss, val_loss = train_knowledge_distillation(
                        teacher,
                        student,
                        train_loader,
                        val_loader,
                        test_loader,
                        epochs=epochs,
                        T=T,
                        alpha=alpha,
                        device=device,
                        optimizer=optimizer,
                        label_smoothing=0.0,
                        patience=20,
                        act = act
                    )

                    # -----------------------------
                    # Final testing (your function)
                    # -----------------------------
                    
                    test_loss, test_acc = test(
                        student,
                        test_loader,
                        device=device
                    )
                    
                    # ---- SAVE WEIGHTS (one file per model) ----
                    save_dir = "ct_saved_models"
                    os.makedirs(save_dir, exist_ok=True)

                    arch_str = "x".join(map(str, layer_sizes))  # e.g., 54x12x7
                    deg_str  = f"deg{degree}" if act == "bern" else "degNA"
                    alpha_str = f"alpha{alpha:g}"
                    temp_str  = f"T{T:g}"
                    lr_str   = f"lr{learning_rate:g}"
                    wd_str   = f"wd{weight_decay:g}"
                    seed_str = f"seed{seed}"

                    pth_name = _safe(f"fc_{arch_str}_{act}_{deg_str}_{alpha_str}_{temp_str}_{lr_str}_{wd_str}_{seed_str}.pth")
                    pth_path = os.path.join(save_dir, pth_name)

                    torch.save(
                        {
                            "state_dict": student.state_dict(),
                            "arch": layer_sizes,
                            "activation": act,
                            "degree": degree if act == "bern" else None,
                            "alpha": alpha,
                            "T": T,
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "seed": seed,
                            "val_acc": float(val_acc),
                            "test_acc": float(test_acc),
                        },
                        pth_path
                    )
                    results=[]
                    results.append({
                        "activation": act,
                        "layer_sizes": str(layer_sizes),
                        "degree": degree if act == "bern" else None,
                        "alpha": alpha,
                        "T": T,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "ce_train_loss": float(train_loss),
                        "val_accuracy_%": float(val_acc),
                        "val_loss": float(val_loss),
                        "test_loss": float(test_loss),
                        "test_accuracy_%": float(test_acc),
                    })
                    df = pd.DataFrame(results)
                    df.to_csv(file_name, index=False, mode='a', header=header)
                    
                    all_results.append({
                        "activation": act,
                        "layer_sizes": str(layer_sizes),
                        "degree": degree if act == "bern" else None,
                        "alpha": alpha,
                        "T": T,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "ce_train_loss": float(train_loss),
                        "val_accuracy_%": float(val_acc),
                        "val_loss": float(val_loss),
                        "test_loss": float(test_loss),
                        "test_accuracy_%": float(test_acc),
                    })
                    
    all_df = pd.DataFrame(all_results)
    print("\n===== Bernstein vs ReLU =====")
    print(all_df.to_string(index=False))          
    return all_df

def save_results_to_csv(df, filename, index=True,mode = 'w', header=True):
        df.to_csv(filename, index=index, mode=mode, header=header)

        