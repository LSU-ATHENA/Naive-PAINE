import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import torch
from PIL import Image

SHARED_ROOT = "/home/jkim5/Shared"
NOISEAR_DPO = True

MODELS = {
    "sdxl": {
        "paine_ckpt": "weights/sdxl.pth",
        "golden_id": "SDXL", "golden_ckpt": "golden_noise/sdxl.pth",
        "noisear_pipe": "SDXL", "noisear_dir": "sdxl_and_dreamshaper", "cond_dim": 2048,
        "steps": 50, "guidance": 5.5,
    },
    "dreamshaper": {
        "paine_ckpt": "weights/dreamshaper.pth",
        "golden_id": "DreamShaper", "golden_ckpt": "golden_noise/dreamshaper.pth",
        "noisear_pipe": "DreamShaper", "noisear_dir": "sdxl_and_dreamshaper", "cond_dim": 2048,
        "steps": 8, "guidance": 3.5,
    },
    "hunyuan_dit": {
        "paine_ckpt": "weights/hunyuandit.pth",
        "golden_id": "DiT", "golden_ckpt": "golden_noise/dit.pth",
        "noisear_pipe": "DiT", "noisear_dir": "dit", "cond_dim": 1024,
        "steps": 50, "guidance": 5.0,
    },
}

ROOT = Path(__file__).resolve().parent
for _p in (str(ROOT), str(ROOT / "inference"), str(ROOT / "gen_dataset")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate import load_pipeline, encode_prompt_for_model
from loader import load_predictor
from selection import generate_noise_candidates, select_top_k_noise
from model.config import get_dims
from metrics.scorer import MultiMetricScorer


def load_npnet(cfg, device):
    gi = f"{SHARED_ROOT}/golden_noise/inference"
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "model" or k.startswith("model.")}
    sys.path.insert(0, gi)
    try:
        import npnet_pipeline
        net = npnet_pipeline.NPNet(cfg["golden_id"], f"{SHARED_ROOT}/{cfg['golden_ckpt']}", device=device)
    finally:
        if gi in sys.path:
            sys.path.remove(gi)
        for k in list(sys.modules):
            if k == "model" or k.startswith("model.") or k == "npnet_pipeline":
                sys.modules.pop(k, None)
        sys.modules.update(saved)
    net._alpha = net._alpha.to(device)
    net._beta = net._beta.to(device)
    return net


def load_noisear(cfg, device):
    npath = f"{SHARED_ROOT}/noisear"
    sys.path.insert(0, npath)
    try:
        from noisear_net import NoiseARNet
    finally:
        if npath in sys.path:
            sys.path.remove(npath)
    sub = cfg["noisear_dir"] + ("_dpo" if NOISEAR_DPO else "")
    ckpt = f"{SHARED_ROOT}/noisear/pretrained_models/{sub}/model.pth"
    return NoiseARNet(pretrained_path=ckpt, pipeline=cfg["noisear_pipe"]).to(device).eval()


def load_prompts(args):
    if args.prompt:
        return [args.prompt]
    f = Path(args.prompts_file)
    suf = f.suffix.lower()
    if suf == ".txt":
        ps = [x.strip() for x in f.read_text().splitlines() if x.strip()]
    elif suf == ".csv":
        with f.open(encoding="utf-8-sig", newline="") as fh:
            r = csv.DictReader(fh)
            col = next((c for c in r.fieldnames if c and c.lower() in ("prompt", "prompts", "caption", "text")),
                       r.fieldnames[0])
            ps = [row[col].strip() for row in r if row.get(col) and row[col].strip()]
    elif suf == ".jsonl":
        ps = []
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                o = json.loads(line)
                ps.append(o if isinstance(o, str) else (o.get("prompt") or o.get("caption")))
    elif suf == ".json":
        data = json.loads(f.read_text())
        ps = [o if isinstance(o, str) else (o.get("prompt") or o.get("caption")) for o in data]
    else:
        raise ValueError(f"unsupported prompts_file: {f}")
    if args.max_prompts > 0:
        ps = ps[: args.max_prompts]
    return ps


def make_latents(method, B, latent_shape, cond_embeds, pred_embeds, pred_mask,
                 predictor, npnet, noisear, N, device, dtype):
    if method == "standard":
        return torch.randn(B, *latent_shape, device=device, dtype=dtype)
    if method == "paine":
        cands = generate_noise_candidates(N, latent_shape, device=device, dtype=dtype)
        sel, _ = select_top_k_noise(predictor, cands, pred_embeds, pred_mask, num_select=B, head_index=0)
        return sel
    ce = cond_embeds.expand(B, -1, -1).float()
    if method == "golden":
        base = torch.randn(B, *latent_shape, device=device, dtype=torch.float32)
        with torch.no_grad():
            return npnet(base, ce).to(dtype)
    if method == "noisear":
        with torch.no_grad():
            return noisear(ce).to(dtype)
    raise ValueError(method)


def expand_kwargs(gen_kwargs, B):
    out = {}
    for k, v in gen_kwargs.items():
        if torch.is_tensor(v) and v.dim() >= 2:
            out[k] = v.expand(B, *[-1] * (v.dim() - 1))
        else:
            out[k] = v
    return out


def encode_prompt_any(pipe, prompt, model, device):
    if model == "hunyuan_dit":
        pe0, npe0, pam0, npam0 = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1,
                                                    do_classifier_free_guidance=True, text_encoder_index=0)
        pe1, npe1, pam1, npam1 = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1,
                                                    do_classifier_free_guidance=True, text_encoder_index=1)
        gen_kwargs = {"prompt_embeds": pe0, "negative_prompt_embeds": npe0,
                      "prompt_attention_mask": pam0, "negative_prompt_attention_mask": npam0,
                      "prompt_embeds_2": pe1, "negative_prompt_embeds_2": npe1,
                      "prompt_attention_mask_2": pam1, "negative_prompt_attention_mask_2": npam1}
        return pe1, pam1, gen_kwargs, pe0
    pred_embeds, pred_mask, gen_kwargs = encode_prompt_for_model(pipe, prompt, model, device)
    return pred_embeds, pred_mask, gen_kwargs, gen_kwargs.get("prompt_embeds")


def summarize(rows, metrics):
    agg = {}
    for r in rows:
        key = f"{r['model']}/{r['method']}"
        agg.setdefault(key, {m: [] for m in metrics})
        for m in metrics:
            agg[key][m].append(r[m])
    return {k: {m: sum(v[m]) / len(v[m]) for m in metrics} for k, v in agg.items()}


def write_results(out, rows, metrics, args):
    out.mkdir(parents=True, exist_ok=True)
    summ = summarize(rows, metrics)
    (out / "results.json").write_text(json.dumps({
        "config": {"model": args.model, "methods": args.methods, "num_images": args.num_images,
                   "N": args.N, "noisear_dpo": NOISEAR_DPO, "metrics": metrics},
        "summary": summ, "rows": rows,
    }, indent=2))
    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "method", "prompt_idx", "image", *metrics])
        for r in rows:
            w.writerow([r["model"], r["method"], r["prompt_idx"], r["image"], *[r[m] for m in metrics]])
    print("\n" + "=" * 72)
    for model in args.model:
        present = [m for m in args.methods if f"{model}/{m}" in summ]
        if not present:
            continue
        n = sum(1 for r in rows if r["model"] == model and r["method"] == present[0])
        print(f"\n{model}   (mean over {n} images/method)")
        print("  " + f"{'method':<10}" + "  ".join(f"{m:>12}" for m in metrics))
        for method in present:
            vals = summ[f"{model}/{method}"]
            print("  " + f"{method:<10}" + "  ".join(f"{vals[m]:>12.4f}" for m in metrics))
    print("\nwrote", out / "results.json", "and", out / "results.csv")


def main():
    ap = argparse.ArgumentParser(description="Compare Standard / PAINE / Golden Noise / NoiseAR on reward scores.")
    ap.add_argument("--model", nargs="+", default=["sdxl"], choices=list(MODELS))
    ap.add_argument("--methods", nargs="+", default=["standard", "paine", "golden", "noisear"],
                    choices=["standard", "paine", "golden", "noisear"])
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--max_prompts", type=int, default=-1)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--num_images", type=int, default=4)
    ap.add_argument("--metrics", nargs="+", default=["hpsv2", "hpsv3", "image_reward", "pick_score"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance_scale", type=float, default=None)
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--output_dir", default="compare_out")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.prompt and not args.prompts_file:
        ap.error("provide --prompt or --prompts_file")

    prompts = load_prompts(args)
    rows = []
    out = Path(args.output_dir)
    B = args.num_images

    for model in args.model:
        cfg = MODELS[model]
        steps = args.steps or cfg["steps"]
        guidance = args.guidance_scale or cfg["guidance"]
        latent_shape = get_dims(model)["latent_shape"]
        print(f"\n=== generate {model}  steps={steps}  cfg={guidance}  prompts={len(prompts)} ===", flush=True)

        pipe = load_pipeline(model, device=args.device, base_model=args.base_model)
        dtype = pipe.unet.dtype if hasattr(pipe, "unet") else pipe.transformer.dtype

        predictor = load_predictor(f"{SHARED_ROOT}/{cfg['paine_ckpt']}", device=args.device)[0] if "paine" in args.methods else None
        npnet = load_npnet(cfg, args.device) if "golden" in args.methods else None
        noisear = load_noisear(cfg, args.device) if "noisear" in args.methods else None
        need_cond = any(m in ("golden", "noisear") for m in args.methods)

        for pidx, prompt in enumerate(prompts):
            pred_embeds, pred_mask, gen_kwargs, cond_embeds = encode_prompt_any(pipe, prompt, model, args.device)
            if need_cond:
                if cond_embeds is None:
                    raise RuntimeError(f"{model}: encode returned no prompt_embeds; golden/noisear need conditioning embeds.")
                if cond_embeds.shape[-1] != cfg["cond_dim"]:
                    raise RuntimeError(f"{model}: expected {cfg['cond_dim']}-d cond embeds, got {tuple(cond_embeds.shape)}.")

            for method in args.methods:
                latents = make_latents(method, B, latent_shape, cond_embeds, pred_embeds, pred_mask,
                                       predictor, npnet, noisear, args.N, args.device, dtype)
                gk = expand_kwargs(gen_kwargs, B)
                images = pipe(prompt=None, **gk, latents=latents, num_images_per_prompt=1,
                              num_inference_steps=steps, guidance_scale=guidance).images
                d = out / model / method
                d.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(images):
                    p = d / f"p{pidx:03d}_{i:02d}.png"
                    img.save(p)
                    rows.append({"model": model, "method": method, "prompt_idx": pidx, "prompt": prompt, "image": str(p)})
            print(f"  [{model}] {pidx + 1}/{len(prompts)}  {prompt[:60]}", flush=True)

        del pipe, predictor, npnet, noisear
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n=== score {len(rows)} images  metrics={args.metrics} ===", flush=True)
    scorer = MultiMetricScorer(metrics=args.metrics, device=args.device)
    for n, r in enumerate(rows, 1):
        img = Image.open(r["image"]).convert("RGB")
        s = scorer.score(img, r["prompt"], image_path=r["image"])
        for m in scorer.metrics:
            r[m] = float(s[m])
        if n % 25 == 0 or n == len(rows):
            print(f"  scored {n}/{len(rows)}", flush=True)

    write_results(out, rows, scorer.metrics, args)


if __name__ == "__main__":
    main()
