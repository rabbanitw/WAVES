"""Score the regen-aug R18 (s55, best_regen_aug_r18.pt) on every attack
image we've generated so far: regen ladders (00001, 00000) and darkroom
flips (00001).

Compares baseline_r50's P(natural) to regen-aug s55's P(natural). If s55
holds where baseline_r50 flipped, the regen-augmented training generalized
beyond in-distribution photoreal edits."""

import io
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data import to_tensor_neg1_1
from models import BinaryClassifier

TARGETS = [
    ("baseline_r50",  "/home/trabbani/WAVES/detector/best_ood_balanced_r50.pt",  "resnet50"),
    ("regen_aug_s55", "/home/trabbani/WAVES/detector/best_regen_aug_r18.pt",     "resnet18"),
]
IMAGE_SIZE = 384

ATTACK_DIRS = [
    ("regen_00001",    Path("/home/trabbani/WAVES/regen_attack_flips")),
    ("regen_00000",    Path("/home/trabbani/WAVES/regen_attack_flips_00000")),
    ("darkroom_00001", Path("/home/trabbani/WAVES/darkroom_attack_flips")),
]


@torch.no_grad()
def score(model, pil, device):
    pil2 = pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    pil2.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    pil3 = Image.open(buf).convert("RGB")
    x = to_tensor_neg1_1(pil3).unsqueeze(0).to(device)
    p = F.softmax(model(x), dim=1)[0]
    return float(p[0].item()), float(p[1].item())


def main():
    device = "cuda"
    models = {}
    for name, ckpt, bb in TARGETS:
        m = BinaryClassifier(pretrained=False, backbone=bb).to(device).eval()
        ck = torch.load(ckpt, map_location=device)
        m.load_state_dict(ck["state_dict"])
        models[name] = m
        print(f"loaded {name}: {ckpt}")

    all_results = {}
    for attack_name, ddir in ATTACK_DIRS:
        print(f"\n{'='*80}\n  {attack_name}   ({ddir})\n{'='*80}")
        files = sorted(f for f in ddir.iterdir() if f.suffix == ".jpg")
        rows = []
        for f in files:
            pil = Image.open(f).convert("RGB")
            r = {"file": f.name}
            for name, model in models.items():
                p_nat, p_synth = score(model, pil, device)
                r[f"{name}_p_natural"] = p_nat
                r[f"{name}_p_synth"] = p_synth
            rows.append(r)
        all_results[attack_name] = rows

        # print table
        print(f"  {'file':<28s} | {'baseline_r50 P(nat)':>20s} | "
              f"{'regen_aug_s55 P(nat)':>22s}")
        print(f"  {'-'*28} | {'-'*20} | {'-'*22}")
        for r in rows:
            b_marker = "*" if r["baseline_r50_p_natural"] > 0.5 else " "
            s_marker = "*" if r["regen_aug_s55_p_natural"] > 0.5 else " "
            print(f"  {r['file']:<28s} | "
                  f"{r['baseline_r50_p_natural']:>18.4f}{b_marker} | "
                  f"{r['regen_aug_s55_p_natural']:>20.4f}{s_marker}")

    OUT = Path("/home/trabbani/WAVES/detector/runs/regen_aug_vs_attacks.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
