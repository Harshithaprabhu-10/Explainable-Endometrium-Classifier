"""
Explainable ensemble (ConvNeXt + ResNet) with Grad-CAM and a Gradio interface.

Run:  pip install gradio opencv-python
      python app.py
Put this file in the same folder as convnext_model.pth and resnet_model.pth.

Matches ensemble.py exactly: timm convnext_tiny + torchvision resnet50,
224x224 input, ToTensor only (NO ImageNet normalisation), 0.5/0.5 weights.
"""
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from torchvision import models, transforms
import timm
from PIL import Image

# ---------------- CONFIG (match train.py) ----------------
CLASSES = ['EA', 'EH', 'EP', 'NE']          # same order as your training folders
FULL_NAMES = {                               # edit if your dataset means something else
    'EA': 'Endometrial Adenocarcinoma',
    'EH': 'Endometrial Hyperplasia',
    'EP': 'Endometrial Polyp',
    'NE': 'Normal Endometrium',
}
IMG_SIZE = 224                               # same as ensemble.py (83.43% was measured at 224)
W_CONVNEXT, W_RESNET = 0.5, 0.5              # best weights from your search
CONVNEXT_PATH, RESNET_PATH = 'convnext_model.pth', 'resnet_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_convnext():
    m = timm.create_model('convnext_tiny', pretrained=False, num_classes=len(CLASSES))
    m.load_state_dict(torch.load(CONVNEXT_PATH, map_location=DEVICE))
    return m.to(DEVICE).eval()


def load_resnet():
    m = models.resnet50()
    m.fc = nn.Sequential(nn.Dropout(0.75), nn.Linear(m.fc.in_features, len(CLASSES)))
    ckpt = torch.load(RESNET_PATH, map_location=DEVICE)
    m.load_state_dict(ckpt['model'])
    return m.to(DEVICE).eval()


convnext = load_convnext()
resnet = load_resnet()

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                   # no Normalize: models were trained without it
])


# ---------------- Grad-CAM ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.acts, self.grads = model, None, None
        target_layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        if not out.requires_grad:            # plain inference under no_grad: nothing to hook
            return
        self.acts = out.detach()
        out.register_hook(lambda g: setattr(self, 'grads', g.detach()))

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear',
                            align_corners=False)[0, 0].cpu().numpy()
        cam -= cam.min()
        return cam / (cam.max() + 1e-8)


cam_convnext = GradCAM(convnext, convnext.stages[-1].blocks[-1])
cam_resnet = GradCAM(resnet, resnet.layer4[-1])


DISPLAY = 448                                # heatmaps are drawn larger than the 224 model input


def grid_name(cx, cy):
    v = 'upper' if cy < IMG_SIZE / 3 else 'lower' if cy > 2 * IMG_SIZE / 3 else 'middle'
    h = 'left' if cx < IMG_SIZE / 3 else 'right' if cx > 2 * IMG_SIZE / 3 else 'centre'
    return 'centre' if (v, h) == ('middle', 'centre') else f'{v}-{h}'


def find_regions(cam_f, cam_c, cam_r, gray, max_regions=3, thr=0.5):
    """Split the fused heatmap into separate hot regions and describe each one."""
    mask = (cam_f > thr).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = cam_f[mask > 0].sum() + 1e-8
    regions = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.005 * IMG_SIZE ** 2:     # ignore specks
            continue
        m = labels == i
        mc, mr = cam_c[m].mean(), cam_r[m].mean()
        regions.append(dict(
            cx=cents[i][0], cy=cents[i][1],
            area=100 * area / IMG_SIZE ** 2,
            share=100 * cam_f[m].sum() / total,
            peak=cam_f[m].max(),
            driver=('mostly ConvNeXt' if mc > 1.3 * mr else
                    'mostly ResNet' if mr > 1.3 * mc else 'both models'),
            gray=gray[m].mean(),
        ))
    regions.sort(key=lambda r: r['share'], reverse=True)
    regions = regions[:max_regions]
    for k, r in enumerate(regions, 1):
        r['id'] = k
        r['loc'] = grid_name(r['cx'], r['cy'])
    return regions


def overlay(img_rgb, cam, alpha=0.45, regions=None):
    base = cv2.resize(img_rgb, (DISPLAY, DISPLAY), interpolation=cv2.INTER_CUBIC)
    cam_big = cv2.resize(cam, (DISPLAY, DISPLAY), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((np.clip(cam_big, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (alpha * heat + (1 - alpha) * base).astype(np.uint8)
    k = DISPLAY / IMG_SIZE
    for r in regions or []:
        c = (int(r['cx'] * k), int(r['cy'] * k))
        cv2.circle(out, c, 17, (255, 255, 255), -1)
        cv2.circle(out, c, 17, (20, 20, 20), 2)
        cv2.putText(out, str(r['id']), (c[0] - 7, c[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def describe_regions(regions, gray_mean):
    if not regions:
        return ('### Heatmap regions\nNo single strong region: the evidence is weak or '
                'spread thinly over the whole image.')
    rows = ['| # | Location | Image area | Share of attention | Driven by | Appearance |',
            '|---|---|---|---|---|---|']
    warn = []
    for r in regions:
        ratio = r['gray'] / gray_mean
        look = ('darker than average (cell-dense / heavily stained)' if ratio < 0.85 else
                'paler than average (e.g. gland lumen, stroma or empty space)' if ratio > 1.15 else
                'typical staining')
        if r['gray'] > 200:
            warn.append(f"Region {r['id']} is very bright, so it may be empty space or "
                        "background rather than tissue. Check that the model is not "
                        "relying on a non-tissue cue.")
        rows.append(f"| **{r['id']}** | {r['loc']} | {r['area']:.1f}% | {r['share']:.0f}% "
                    f"| {r['driver']} | {look} |")
    top = regions[0]
    lines = [
        '### Heatmap regions',
        'Numbered circles on the ensemble heatmap mark the strongest regions, ranked by '
        'how much of the total attention they hold.',
        '', *rows, '',
        f"- **Main driver:** region 1 ({top['loc']}) carries {top['share']:.0f}% of the "
        f"high-attention area, so the prediction depends mostly on that spot.",
    ]
    if len(regions) > 1:
        rest = ', '.join(f"{r['id']} ({r['loc']})" for r in regions[1:])
        lines.append(f"- **Supporting regions:** {rest} add further evidence.")
    lines += [f'- **Warning:** {w}' for w in warn]
    lines.append('- **Reading the colours:** red / yellow = strongest influence on the '
                 'result, blue = little influence.')
    return '\n'.join(lines)


# ---------------- Explanation text ----------------
def where(cam):
    ys, xs = np.where(cam > 0.5)
    if len(xs) == 0:
        return 'no clear region'
    cy, cx = ys.mean() / IMG_SIZE, xs.mean() / IMG_SIZE
    v = 'upper' if cy < 0.4 else 'lower' if cy > 0.6 else 'middle'
    h = 'left' if cx < 0.4 else 'right' if cx > 0.6 else 'centre'
    return 'centre of the image' if (v, h) == ('middle', 'centre') else f'{v}-{h} part of the image'


def build_explanation(probs, p_cn, p_rn, cam_c, cam_r, cam_f):
    order = np.argsort(probs)[::-1]
    top, second = order[0], order[1]
    conf, margin = probs[top], probs[top] - probs[second]
    agree = int(np.argmax(p_cn)) == int(np.argmax(p_rn))
    mc, mr = cam_c > 0.5, cam_r > 0.5
    iou = (mc & mr).sum() / max((mc | mr).sum(), 1)
    area = (cam_f > 0.5).mean() * 100

    lines = [
        f"### Prediction: **{CLASSES[top]}** – {FULL_NAMES[CLASSES[top]]} ({conf*100:.1f}%)",
        f"- **Confidence:** runner-up is {CLASSES[second]} ({probs[second]*100:.1f}%), "
        f"a margin of {margin*100:.1f} points. "
        + ("The decision is clear-cut." if margin > 0.4 else
           "The decision is fairly close, so treat it with caution." if margin > 0.15 else
           "**The model is uncertain between these two classes.**"),
        f"- **Model agreement:** ConvNeXt says {CLASSES[int(np.argmax(p_cn))]} "
        f"({p_cn.max()*100:.0f}%), ResNet says {CLASSES[int(np.argmax(p_rn))]} "
        f"({p_rn.max()*100:.0f}%). "
        + ("Both models agree, which strengthens the result." if agree else
           "The models disagree; the ensemble result is a compromise."),
        f"- **Where the model looked:** the strongest evidence is in the {where(cam_f)}; "
        f"about {area:.0f}% of the image drives the decision "
        + ("(focused on a localized region)." if area < 20 else
           "(spread over a moderate area)." if area < 45 else
           "(spread broadly across the tissue)."),
        f"- **Attention overlap:** the two models' hot regions overlap by {iou*100:.0f}% (IoU). "
        + ("They rely on similar tissue regions." if iou > 0.4 else
           "They focus on different regions, so each is seeing different cues."),
        "",
        "> The heatmap shows **where** the network found evidence for this class, "
        "not a pathological reason. A pathologist should confirm that the highlighted "
        "region is actually clinically meaningful. This tool is for research, "
        "not for diagnosis.",
    ]
    return '\n'.join(lines)


# ---------------- Inference ----------------
def predict(pil_img):
    if pil_img is None:
        return None, 'Upload an image first.', '', None, None, None
    pil_img = pil_img.convert('RGB')
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    img_rgb = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    gray = img_rgb.mean(axis=2)

    with torch.no_grad():
        p_cn = F.softmax(convnext(x), 1)[0].cpu().numpy()
        p_rn = F.softmax(resnet(x), 1)[0].cpu().numpy()
    probs = W_CONVNEXT * p_cn + W_RESNET * p_rn
    pred = int(np.argmax(probs))

    cam_c = cam_convnext(x.clone().requires_grad_(True), pred)
    cam_r = cam_resnet(x.clone().requires_grad_(True), pred)
    cam_f = W_CONVNEXT * cam_c + W_RESNET * cam_r
    cam_f /= (cam_f.max() + 1e-8)

    regions = find_regions(cam_f, cam_c, cam_r, gray)
    label = {f'{c} – {FULL_NAMES[c]}': float(p) for c, p in zip(CLASSES, probs)}
    summary = build_explanation(probs, p_cn, p_rn, cam_c, cam_r, cam_f)
    region_md = describe_regions(regions, gray.mean())
    return (label, summary, region_md,
            overlay(img_rgb, cam_f, regions=regions),
            overlay(img_rgb, cam_c),
            overlay(img_rgb, cam_r))


LEGEND = (
    '<div style="display:flex;align-items:center;gap:10px;margin:2px 0 8px 0;font-size:13px">'
    '<span>Low influence</span>'
    '<div style="flex:1;max-width:320px;height:12px;border-radius:6px;'
    'background:linear-gradient(90deg,#00007f,#0000ff,#00ffff,#ffff00,#ff0000,#7f0000)"></div>'
    '<span>High influence</span></div>'
)
H = 380                                      # same height for every image

with gr.Blocks(title='Explainable Histopathology Classifier') as demo:
    gr.Markdown('# Explainable Histopathology Classifier\n'
                'ConvNeXt + ResNet ensemble with Grad-CAM explanations')
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            inp = gr.Image(type='pil', label='Microscopic image', height=H)
            btn = gr.Button('Analyze', variant='primary')
        with gr.Column(scale=1):
            out_label = gr.Label(num_top_classes=len(CLASSES), label='Class probabilities')
            out_summary = gr.Markdown()

    gr.Markdown('## Why this result? Grad-CAM heatmaps')
    gr.HTML(LEGEND)
    with gr.Row(equal_height=True):
        out_ens = gr.Image(label='Ensemble (final decision)', height=H, interactive=False)
        out_cn = gr.Image(label='ConvNeXt only', height=H, interactive=False)
        out_rn = gr.Image(label='ResNet only', height=H, interactive=False)
    out_regions = gr.Markdown()

    btn.click(predict, inp, [out_label, out_summary, out_regions, out_ens, out_cn, out_rn])

if __name__ == '__main__':
    demo.launch()
