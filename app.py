"""DreamForge AI: sketch -> image with Stable Diffusion 1.5 + ControlNet (Scribble).

Run locally / on Hugging Face Spaces:  python app.py
"""
import json, os, random, time

import gradio as gr
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image, ImageDraw, ImageOps

BASE_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/control_v11p_sd15_scribble"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32   # fp16 on GPU, fp32 on CPU

controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=DTYPE)
pipe = StableDiffusionControlNetPipeline.from_pretrained(BASE_ID, controlnet=controlnet, torch_dtype=DTYPE)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(DEVICE)


# ---------- starter sketches ----------
from PIL import Image, ImageDraw, ImageOps

def make_castle_sketch(size: int = 512, line_px: int = 6) -> Image.Image:
    """Return a white-line-on-black castle sketch as an RGB image."""
    img = Image.new("L", (size, size), 0)          # black background
    d = ImageDraw.Draw(img)

    def closed(points):                             # draw a closed outline with lines
        d.line(points + [points[0]], fill=255, width=line_px, joint="curve")

    closed([(180, 250), (340, 250), (340, 420), (180, 420)])   # central keep
    closed([(110, 210), (180, 210), (180, 420), (110, 420)])   # left tower
    closed([(340, 210), (410, 210), (410, 420), (340, 420)])   # right tower
    closed([(95, 210), (145, 120), (195, 210)])                # left roof
    closed([(325, 210), (375, 120), (425, 210)])               # right roof
    closed([(170, 250), (260, 160), (350, 250)])               # central roof
    d.line([(260, 160), (260, 100)], fill=255, width=line_px)  # flag pole
    closed([(260, 100), (300, 115), (260, 130)])               # flag
    d.arc([225, 340, 295, 410], 180, 360, fill=255, width=line_px)  # door arch
    d.line([(225, 375), (225, 420)], fill=255, width=line_px)
    d.line([(295, 375), (295, 420)], fill=255, width=line_px)
    for x in (60, 200, 340):                                   # cloud bumps below
        d.arc([x, 410, x + 110, 480], 180, 360, fill=255, width=line_px)
    return img.convert("RGB")

def _closed(d, pts, w):
    d.line(pts + [pts[0]], fill=0, width=w, joint="curve")

def make_house_sketch(size=512, w=8):
    img = Image.new("RGB", (size, size), "white"); d = ImageDraw.Draw(img)
    _closed(d, [(130, 260), (382, 260), (382, 430), (130, 430)], w)   # walls
    _closed(d, [(100, 260), (256, 130), (412, 260)], w)               # roof
    _closed(d, [(230, 340), (282, 340), (282, 430), (230, 430)], w)   # door
    _closed(d, [(160, 300), (210, 300), (210, 350), (160, 350)], w)   # left window
    _closed(d, [(302, 300), (352, 300), (352, 350), (302, 350)], w)   # right window
    return img

def make_apple_sketch(size=512, w=10):
    img = Image.new("RGB", (size, size), "white"); d = ImageDraw.Draw(img)
    d.ellipse([140, 180, 372, 430], outline=0, width=w)               # body
    d.line([(256, 185), (262, 120)], fill=0, width=w)                 # stem
    _closed(d, [(262, 130), (320, 100), (335, 135), (290, 150)], w)   # leaf
    return img

TEST_SKETCHES = {   # all black-on-white, as the UI would hand them over
    "castle": (ImageOps.invert(make_castle_sketch()),
               "A floating fantasy castle above the clouds, cinematic lighting, detailed concept art"),
    "house":  (make_house_sketch(),
               "a cozy wooden cabin in a snowy forest, warm light in the windows"),
    "apple":  (make_apple_sketch(),
               "a shiny red apple with a stem and a leaf, studio photo"),
}

# ---------- generation ----------
STYLE_PRESETS = {
    "None": "",
    "Cinematic concept art": ", cinematic lighting, detailed concept art, highly detailed",
    "Watercolor": ", soft watercolor painting, pastel colors, paper texture",
    "Pixel art": ", pixel art, 16-bit video game style",
    "Photorealistic": ", photorealistic, natural lighting, 35mm photo",
}
DEFAULT_NEGATIVE = "blurry, low quality, distorted, deformed"

def prepare_control_image(sketch, size: int = 512, dark_threshold: int = 128) -> Image.Image:
    """Turn a user sketch (dark strokes on light canvas) into white strokes on black, size x size RGB."""
    if isinstance(sketch, dict):                       # Gradio Sketchpad/ImageEditor value
        sketch = sketch.get("composite")
    if sketch is None:
        raise ValueError("Please draw a sketch or upload one first.")
    img = Image.fromarray(sketch) if isinstance(sketch, np.ndarray) else sketch
    if img.mode in ("RGBA", "LA", "P"):                # flatten transparency onto white
        img = img.convert("RGBA")
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(canvas, img)
    gray = np.asarray(img.convert("L").resize((size, size), Image.LANCZOS))
    strokes = gray < dark_threshold
    if strokes.mean() < 0.002:
        raise ValueError("The sketch looks empty. Draw darker or thicker lines on a light background.")
    return Image.fromarray(np.where(strokes, 255, 0).astype(np.uint8)).convert("RGB")

def generate(sketch, prompt: str, style: str, steps: int, control_scale: float,
             seed: int, guidance: float = 7.5, out_dir=None):
    """Run sketch+text generation. Returns (image, control_image, settings dict)."""
    if not prompt or not prompt.strip():
        raise ValueError("Please enter a text prompt.")
    if not 10 <= int(steps) <= 50:
        raise ValueError("Inference steps must be between 10 and 50.")
    if not 0.0 <= float(control_scale) <= 2.0:
        raise ValueError("Control strength must be between 0.0 and 2.0.")
    if style not in STYLE_PRESETS:
        raise ValueError(f"Unknown style: {style}")

    control = prepare_control_image(sketch)
    seed = int(seed)
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)            # seed -1 means "pick one for me"
    full_prompt = prompt.strip() + STYLE_PRESETS[style]

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        out = pipe(
            prompt=full_prompt, negative_prompt=DEFAULT_NEGATIVE, image=control,
            num_inference_steps=int(steps), guidance_scale=float(guidance),
            controlnet_conditioning_scale=float(control_scale),
            height=512, width=512, generator=generator,
        )
    elapsed = time.time() - t0
    if out.nsfw_content_detected and out.nsfw_content_detected[0]:
        raise RuntimeError("The safety checker blocked this result. Try a different seed or prompt.")

    settings = {
        "prompt": full_prompt, "negative_prompt": DEFAULT_NEGATIVE, "style": style,
        "seed": seed, "steps": int(steps), "guidance_scale": float(guidance),
        "controlnet_conditioning_scale": float(control_scale), "size": "512x512",
        "seconds": round(elapsed, 2),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if DEVICE == "cuda" else None,
        "base_model": BASE_ID, "controlnet": CONTROLNET_ID,
    }
    if out_dir:                                        # only save when asked (private use)
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out.images[0].save(f"{out_dir}/{stamp}_seed{seed}.png")
        with open(f"{out_dir}/{stamp}_seed{seed}.json", "w") as f:
            json.dump(settings, f, indent=2)
    return out.images[0], control, settings

# ---------- UI ----------
SAVE_OUTPUTS = False                   # keep False on public demos: do not store strangers' images

STYLE_CHOICES = [("🎬 Cinematic", "Cinematic concept art"), ("📷 Photoreal", "Photorealistic"),
                 ("🎨 Watercolor", "Watercolor"), ("👾 Pixel art", "Pixel art"), ("⚪ No style", "None")]
assert {v for _, v in STYLE_CHOICES} == set(STYLE_PRESETS)

PROMPT_IDEAS = {"🏰 Sky castle": "A floating fantasy castle above the clouds",
                "🌃 Neon city": "A neon cyberpunk city at night, rain on the streets",
                "🏡 Snow cabin": "A cozy wooden cabin in a snowy forest, warm light in the windows",
                "🍎 Studio apple": "A shiny red apple with a stem and a leaf, studio photo",
                "🌋 Volcano": "A glowing volcano erupting under a starry sky"}

HERO_HTML = """
<div id="df-hero">
  <div class="df-badge">✦ Sketch → Image &nbsp;·&nbsp; Stable Diffusion 1.5 + ControlNet</div>
  <h1>Dream<span>Forge</span> AI</h1>
  <p>Scribble a rough shape. Describe a world. Watch it get forged.</p>
  <div class="df-steps"><span><b>1</b> Sketch</span><i></i><span><b>2</b> Describe</span><i></i><span><b>3</b> Forge</span></div>
</div>"""

HOW_HTML = """
<div class="df-how">
  <div class="df-flow"><span>✏️ Your sketch</span><i>→</i><span>🧼 Clean &amp; invert to white-on-black</span><i>→</i><span>🧭 ControlNet (spatial hints)</span></div>
  <div class="df-flow"><span>💬 Your prompt</span><i>→</i><span>🔤 CLIP tokenizer + text encoder</span><i>→</i><span>🧠 U-Net denoises 25 steps</span><i>→</i><span>🖼️ VAE decoder</span></div>
  <p>The sketch decides <b>where</b> shapes go. The prompt decides <b>what</b> they are. Higher control strength follows your lines more strictly, but can look flatter.</p>
</div>"""

FOOTER_HTML = """
<div id="df-footer">
  <div class="df-models"><span>🤗 stable-diffusion-v1-5</span><span>🤗 control_v11p_sd15_scribble</span><span>🤗 Diffusers</span><span>🔥 PyTorch</span><span>⚡ Gradio</span></div>
  <p>Models: Stable Diffusion v1.5 (CreativeML OpenRAIL-M) and ControlNet v1.1 Scribble (OpenRAIL). Output follows your sketch's rough structure; it is guidance, not a guarantee. A safety checker is enabled.</p>
</div>"""

def card_title(number, text, hint=""):
    hint_html = f"<small>{hint}</small>" if hint else ""
    return f'<div class="df-card-title"><b>{number}</b><span>{text}</span>{hint_html}</div>'

THEME = gr.themes.Base(primary_hue="violet", secondary_hue="cyan", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])

CSS = """
:root, .gradio-container, .dark {
  --body-background-fill: #07050f; --background-fill-primary: #0d0a1c; --background-fill-secondary: #120e26;
  --block-background-fill: rgba(255,255,255,0.035); --block-border-color: rgba(255,255,255,0.09);
  --border-color-primary: rgba(255,255,255,0.12); --body-text-color: #ece9ff; --body-text-color-subdued: #a79fd6;
  --block-title-text-color: #cfc8ff; --block-label-text-color: #b9b0ee; --block-label-background-fill: transparent;
  --input-background-fill: rgba(255,255,255,0.05); --input-border-color: rgba(255,255,255,0.14);
  --input-border-color-focus: #8b5cf6; --color-accent: #8b5cf6; --slider-color: #8b5cf6;
  --shadow-drop: none; --shadow-drop-lg: none;
}
.gradio-container { max-width: 1280px !important; margin: 0 auto !important;
  background: radial-gradient(1100px 560px at 8% -8%, rgba(124,58,237,.38), transparent 60%),
              radial-gradient(900px 520px at 96% 2%, rgba(6,182,212,.28), transparent 58%),
              radial-gradient(900px 600px at 50% 110%, rgba(236,72,153,.16), transparent 60%), #07050f !important; }
#df-hero { text-align: center; padding: 34px 10px 10px; }
#df-hero .df-badge { display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: .82rem; color: #cbbcff;
  background: rgba(139,92,246,.14); border: 1px solid rgba(139,92,246,.4); letter-spacing: .02em; }
#df-hero h1 { margin: 16px 0 6px; font-weight: 800; letter-spacing: -0.035em; line-height: 1.02;
  font-size: clamp(2.6rem, 7vw, 4.8rem); color: #fff; }
#df-hero h1 span { background: linear-gradient(90deg, #a78bfa, #22d3ee, #f472b6, #a78bfa); background-size: 300% 100%;
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; animation: df-shift 8s linear infinite; }
#df-hero p { color: #b8b0e6; font-size: 1.12rem; margin: 0 0 18px; }
.df-steps { display: inline-flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: center; }
.df-steps span { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px 6px 6px; border-radius: 999px;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1); color: #e6e1ff; font-size: .92rem; }
.df-steps span b { display: inline-grid; place-items: center; width: 24px; height: 24px; border-radius: 50%;
  background: linear-gradient(135deg, #7c3aed, #06b6d4); color: #fff; font-size: .8rem; }
.df-steps i { width: 26px; height: 1px; background: linear-gradient(90deg, #7c3aed, #06b6d4); }
@keyframes df-shift { to { background-position: 300% 0; } }
.df-card { background: rgba(255,255,255,.04) !important; border: 1px solid rgba(255,255,255,.09) !important;
  border-radius: 22px !important; padding: 16px !important; backdrop-filter: blur(14px);
  box-shadow: 0 10px 40px rgba(0,0,0,.35) !important; gap: 12px !important; }
.df-card-title { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.05rem; color: #fff; }
.df-card-title b { display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 8px;
  background: linear-gradient(135deg, #7c3aed, #06b6d4); font-size: .85rem; }
.df-card-title small { margin-left: auto; color: #9d94d6; font-weight: 500; font-size: .8rem; }
.df-chips { gap: 8px !important; flex-wrap: wrap !important; }
.df-chip, .df-starter { border-radius: 999px !important; border: 1px solid rgba(255,255,255,.14) !important;
  background: rgba(255,255,255,.05) !important; color: #e6e1ff !important; font-size: .85rem !important;
  min-width: 0 !important; flex: 0 0 auto !important; transition: transform .15s, border-color .15s, background .15s; }
.df-chip:hover, .df-starter:hover { transform: translateY(-2px); border-color: #8b5cf6 !important; background: rgba(139,92,246,.18) !important; }
.df-styles label { border-radius: 999px !important; border: 1px solid rgba(255,255,255,.14) !important;
  background: rgba(255,255,255,.05) !important; padding: 7px 14px !important; cursor: pointer; transition: transform .15s, background .15s; }
.df-styles label:hover { transform: translateY(-1px); }
.df-styles label.selected { background: linear-gradient(135deg, #7c3aed, #0891b2) !important; border-color: transparent !important; color: #fff !important; }
.df-styles input[type=radio] { display: none !important; }
#df-forge-btn { min-height: 58px; font-size: 1.15rem !important; font-weight: 700 !important; letter-spacing: .02em;
  border: none !important; color: #fff !important; border-radius: 16px !important;
  background: linear-gradient(120deg, #7c3aed, #06b6d4 60%, #ec4899) !important; background-size: 200% 100% !important;
  box-shadow: 0 0 0 0 rgba(139,92,246,.6), 0 10px 30px rgba(124,58,237,.45) !important;
  animation: df-pulse 2.6s ease-in-out infinite; transition: transform .15s; }
#df-forge-btn:hover { transform: translateY(-2px) scale(1.01); background-position: 100% 0 !important; }
@keyframes df-pulse { 50% { box-shadow: 0 0 0 10px rgba(139,92,246,0), 0 10px 40px rgba(6,182,212,.5); } }
.df-how { color: #c9c2f0; font-size: .92rem; line-height: 1.6; }
.df-flow { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.df-flow span { padding: 6px 12px; border-radius: 12px; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1); }
.df-flow i { color: #22d3ee; font-style: normal; }
#df-footer { text-align: center; padding: 14px 6px 28px; color: #8f87c4; font-size: .82rem; }
.df-models { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-bottom: 10px; }
.df-models span { padding: 5px 12px; border-radius: 999px; background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.1); color: #cfc8ff; font-size: .8rem; }
footer { opacity: .5; }
"""

STARTERS = {name: sk for name, (sk, _) in TEST_SKETCHES.items()}

def run_generation(sketch, prompt, style, steps, strength, seed, guidance):
    return generate(sketch, prompt, style, steps, strength, seed, guidance,
                    out_dir="outputs" if SAVE_OUTPUTS else None)

def forge(sketch, prompt, style, steps, strength, guidance, random_seed, seed, history,
          progress=gr.Progress(track_tqdm=True)):
    use_seed = -1 if random_seed else int(seed)
    try:
        image, control, settings = run_generation(sketch, prompt, style, steps, strength, use_seed, guidance)
    except (ValueError, RuntimeError) as e:
        raise gr.Error(str(e))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise gr.Error("The GPU ran out of memory. Lower the steps and try again.")
    caption = f"{settings['seed']} · {settings['steps']} steps · control {settings['controlnet_conditioning_scale']}"
    history = ([(image, caption)] + (history or []))[:12]
    before = ImageOps.invert(control.convert("L")).convert("RGB")      # dark lines on white, for the slider
    return (before, image), control, settings, history, history

with gr.Blocks(title="DreamForge AI · sketch to image") as demo:
    gr.HTML(HERO_HTML)
    history_state = gr.State([])
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="df-card"):
            gr.HTML(card_title("1", "Sketch", "draw dark lines, or try a starter"))
            with gr.Row(elem_classes="df-chips"):
                b_castle = gr.Button("🏰 Castle", size="sm", elem_classes="df-starter")
                b_house = gr.Button("🏠 Cabin", size="sm", elem_classes="df-starter")
                b_apple = gr.Button("🍎 Apple", size="sm", elem_classes="df-starter")
            sketch = gr.Sketchpad(label="Canvas", show_label=False)
            with gr.Accordion("…or upload a line drawing", open=False):
                gr.Markdown("Dark lines on a light background work best. Photos are not sketches.")
                uploaded = gr.Image(label="Upload", type="numpy", show_label=False)

            gr.HTML(card_title("2", "Describe"))
            prompt = gr.Textbox(show_label=False, lines=2, placeholder="Describe what your sketch should become…",
                                value=PROMPT_IDEAS["🏰 Sky castle"])
            with gr.Row(elem_classes="df-chips"):
                idea_buttons = [(gr.Button(label, size="sm", elem_classes="df-chip"), text)
                                for label, text in PROMPT_IDEAS.items()]
            style = gr.Radio(STYLE_CHOICES, value="Cinematic concept art", label="Style", elem_classes="df-styles")

            with gr.Accordion("⚙️ Fine-tune", open=False):
                strength = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Sketch strictness (control strength)",
                                     info="0 ignores the sketch. About 0.5 to 1.2 works well.")
                steps = gr.Slider(10, 50, value=25, step=1, label="Quality steps",
                                  info="More steps cost more time; gains flatten around 25.")
                guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Prompt strength (guidance)")
                random_seed = gr.Checkbox(value=True, label="🎲 Random seed")
                seed = gr.Number(value=42, precision=0, label="Seed (used when random is off)")

            gr.HTML(card_title("3", "Forge"))
            forge_btn = gr.Button("✦  Forge my image", variant="primary", elem_id="df-forge-btn")

        with gr.Column(scale=6, elem_classes="df-card"):
            gr.HTML(card_title("✦", "Your creation", "drag the slider: sketch ⇄ result"))
            out_slider = gr.ImageSlider(show_label=False, height=520)
            with gr.Tabs():
                with gr.Tab("🧭 AI view"):
                    out_control = gr.Image(show_label=False, interactive=False, height=300)
                with gr.Tab("🧾 Settings"):
                    out_settings = gr.JSON(show_label=False)
                with gr.Tab("🧠 How"):
                    gr.HTML(HOW_HTML)
            gallery = gr.Gallery(label="Forge history (compare settings)", columns=4, rows=1, height=220,
                                 object_fit="cover")
    gr.HTML(FOOTER_HTML)

    b_castle.click(lambda: STARTERS["castle"], None, sketch)
    b_house.click(lambda: STARTERS["house"], None, sketch)
    b_apple.click(lambda: STARTERS["apple"], None, sketch)
    for btn, text in idea_buttons:
        btn.click(lambda t=text: t, None, prompt)

    forge_btn.click(
        lambda drawn, up, *rest: forge(up if up is not None else drawn, *rest),
        inputs=[sketch, uploaded, prompt, style, steps, strength, guidance, random_seed, seed, history_state],
        outputs=[out_slider, out_control, out_settings, gallery, history_state])

demo.queue(max_size=8).launch(theme=THEME, css=CSS)
