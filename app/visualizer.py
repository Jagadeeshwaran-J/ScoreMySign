import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim_fn
from skimage.morphology import skeletonize
from scipy.spatial.distance import directed_hausdorff
import os


def save_all_visualizations(b1: np.ndarray, b2: np.ndarray,
                             img1_raw_bytes: bytes, img2_raw_bytes: bytes,
                             scores: dict, output_dir: str = "outputs") -> list:
    """
    Generate and save visualizations for all preprocessing steps and methods.
    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # Decode raw images for display
    arr1 = np.frombuffer(img1_raw_bytes, np.uint8)
    arr2 = np.frombuffer(img2_raw_bytes, np.uint8)
    img1_raw = cv2.imdecode(arr1, cv2.IMREAD_COLOR)
    img2_raw = cv2.imdecode(arr2, cv2.IMREAD_COLOR)

    saved += [_save_preprocessing(img1_raw, img2_raw, b1, b2, output_dir)]
    saved += [_save_ssim(b1, b2, scores["ssim"], output_dir)]
    saved += [_save_phash(b1, b2, scores["phash"], output_dir)]
    saved += [_save_contour(b1, b2, scores["contour"], output_dir)]
    saved += [_save_ink_density(b1, b2, scores["ink"], output_dir)]
    saved += [_save_hausdorff(b1, b2, scores["hausdorff"], output_dir)]
    saved += [_save_dashboard(img1_raw, img2_raw, b1, b2, scores, output_dir)]

    return saved


# ── 1. Preprocessing Pipeline ─────────────────────────────────────────────────
def _preprocess_steps(img_bgr, target_size=(200, 200)) -> dict:
    steps = {}
    steps["1. Original"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    steps["2. Grayscale"] = gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    steps["3. Otsu Binarize"] = binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    steps["4. Morph Close"] = closed
    coords = cv2.findNonZero(closed)
    cropped = closed.copy()
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 5
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(img_bgr.shape[1] - x, w + 2 * pad)
        h = min(img_bgr.shape[0] - y, h + 2 * pad)
        cropped = closed[y:y + h, x:x + w]
    steps["5. Tight Crop"] = cropped
    steps["6. Resized 200x200"] = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    return steps


def _save_preprocessing(img1_raw, img2_raw, b1, b2, out_dir) -> str:
    steps1 = _preprocess_steps(img1_raw)
    steps2 = _preprocess_steps(img2_raw)
    fig, axes = plt.subplots(2, 6, figsize=(22, 7))
    fig.suptitle("Preprocessing Pipeline — Step by Step", fontsize=15, fontweight='bold')
    for row, (steps, label) in enumerate(zip([steps1, steps2], ["PAN Signature", "Sample Signature"])):
        for col, (name, img) in enumerate(steps.items()):
            ax = axes[row][col]
            ax.imshow(img if len(img.shape) == 3 else img, cmap=None if len(img.shape) == 3 else 'gray')
            ax.set_title(name, fontsize=9, fontweight='bold', pad=4)
            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight='bold', color='navy')
            ax.axis('off')
    plt.tight_layout()
    path = os.path.join(out_dir, "01_preprocessing_pipeline.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ── 2. SSIM ───────────────────────────────────────────────────────────────────
def _save_ssim(b1, b2, score, out_dir) -> str:
    _, diff_map = ssim_fn(b1, b2, gaussian_weights=True, sigma=1.5,
                          use_sample_covariance=False, full=True)
    diff_viz = ((1 - diff_map) * 255).astype(np.uint8)
    overlay  = cv2.addWeighted(b1, 0.5, b2, 0.5, 0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"Method 1 — SSIM (Structural Similarity)  |  Score: {score*100:.1f}%",
                 fontsize=13, fontweight='bold')
    axes[0].imshow(b1, cmap='gray');      axes[0].set_title("PAN (binarized)")
    axes[1].imshow(b2, cmap='gray');      axes[1].set_title("Sample (binarized)")
    im = axes[2].imshow(diff_viz, cmap='hot')
    axes[2].set_title("Difference Map\n(bright = structurally different)")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(overlay, cmap='gray'); axes[3].set_title("Pixel Overlay\n(both superimposed)")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    path = os.path.join(out_dir, "02_method_ssim.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ── 3. pHash ──────────────────────────────────────────────────────────────────
def _phash_internals(img):
    r   = cv2.resize(img, (32, 32))
    dct = cv2.dct(np.float32(r))
    low = dct[:8, :8]
    avg = low.mean()
    bits = np.array([[1 if v > avg else 0 for v in row] for row in low])
    return bits, r, dct[:8, :8]


def _save_phash(b1, b2, score, out_dir) -> str:
    bits1, r1, dct1 = _phash_internals(b1)
    bits2, r2, dct2 = _phash_internals(b2)
    hamming   = int(np.sum(bits1 != bits2))
    diff_bits = (bits1 != bits2).astype(np.uint8)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        f"Method 2 — pHash (Perceptual Hash)  |  Score: {score*100:.1f}%  |  Hamming Distance: {hamming}/64",
        fontsize=13, fontweight='bold')
    for row, (img, r, dct, bits, lbl) in enumerate([
            (b1, r1, dct1, bits1, "PAN"),
            (b2, r2, dct2, bits2, "Sample")]):
        axes[row][0].imshow(img,  cmap='gray');   axes[row][0].set_title(f"{lbl} — Binary Input")
        axes[row][1].imshow(r,    cmap='gray');   axes[row][1].set_title("Resized to 32×32")
        axes[row][2].imshow(dct,  cmap='viridis');axes[row][2].set_title("DCT Low-Freq 8×8")
        axes[row][3].imshow(bits, cmap='gray', vmin=0, vmax=1)
        axes[row][3].set_title(f"{lbl} Hash Bits")
    # Overlay diff on last cell of row 1
    im = axes[1][3].imshow(diff_bits, cmap='Reds', vmin=0, vmax=1)
    axes[1][3].set_title(f"Bit Differences ({hamming} bits differ)")
    plt.colorbar(im, ax=axes[1][3], fraction=0.046, pad=0.04)
    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    path = os.path.join(out_dir, "03_method_phash.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ── 4. Contour ────────────────────────────────────────────────────────────────
def _save_contour(b1, b2, score, out_dir) -> str:
    c1, _ = cv2.findContours(b1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c2, _ = cv2.findContours(b2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis1 = cv2.cvtColor(b1, cv2.COLOR_GRAY2BGR)
    vis2 = cv2.cvtColor(b2, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis1, c1, -1, (0, 220, 80),  1)
    cv2.drawContours(vis2, c2, -1, (255, 120, 0), 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f"Method 3 — Contour Density  |  Score: {score*100:.1f}%  "
        f"|  PAN: {len(c1)} contours   Sample: {len(c2)} contours",
        fontsize=13, fontweight='bold')
    axes[0].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"PAN — {len(c1)} contours (green)"); axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Sample — {len(c2)} contours (orange)"); axes[1].axis('off')

    ax = axes[2]
    bars = ax.bar(["PAN", "Sample"], [len(c1), len(c2)],
                  color=["#2196F3", "#FF9800"], width=0.4, edgecolor='black')
    ax.set_ylabel("Number of Contours", fontsize=11)
    ax.set_title(f"Contour Count  |  Similarity: {score*100:.1f}%", fontsize=11)
    ax.set_ylim(0, max(len(c1), len(c2)) * 1.35)
    for bar, v in zip(bars, [len(c1), len(c2)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha='center', fontsize=13, fontweight='bold')
    ax.set_facecolor('#F9F9F9')
    plt.tight_layout()
    path = os.path.join(out_dir, "04_method_contour.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ── 5. Ink Density ────────────────────────────────────────────────────────────
def _save_ink_density(b1, b2, score, out_dir) -> str:
    r1 = np.sum(b1 > 127) / b1.size
    r2 = np.sum(b2 > 127) / b2.size

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Method 4 — Ink Density Ratio  |  Score: {score*100:.1f}%",
                 fontsize=13, fontweight='bold')
    axes[0].imshow(b1, cmap='gray')
    axes[0].set_title(f"PAN — Ink density: {r1*100:.2f}%"); axes[0].axis('off')
    axes[1].imshow(b2, cmap='gray')
    axes[1].set_title(f"Sample — Ink density: {r2*100:.2f}%"); axes[1].axis('off')

    ax = axes[2]
    bars = ax.bar(["PAN", "Sample"], [r1*100, r2*100],
                  color=["#4CAF50", "#2196F3"], width=0.4, edgecolor='black')
    ax.set_ylabel("Ink Pixel Ratio (%)", fontsize=11)
    ax.set_title(f"Ink Density Comparison  |  Similarity: {score*100:.1f}%", fontsize=11)
    ax.set_ylim(0, max(r1, r2)*100 * 1.4)
    ax.set_facecolor('#F9F9F9')
    for bar, v in zip(bars, [r1*100, r2*100]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.2f}%", ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, "05_method_ink_density.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ── 6. Hausdorff Skeleton ─────────────────────────────────────────────────────
def _get_skeleton(binary, canvas=200):
    skel = skeletonize(binary > 127)
    pts  = np.column_stack(np.where(skel)).astype(np.float32)
    if len(pts) == 0:
        return skel, pts
    mn, mx = pts.min(0), pts.max(0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    return skel, (pts - mn) / rng * (canvas - 1)


def _save_hausdorff(b1, b2, score, out_dir) -> str:
    skel1_img, pts1 = _get_skeleton(b1)
    skel2_img, pts2 = _get_skeleton(b2)
    np.random.seed(42)
    s1 = pts1[np.random.choice(len(pts1), min(500, len(pts1)), replace=False)]
    s2 = pts2[np.random.choice(len(pts2), min(500, len(pts2)), replace=False)]
    hd = max(directed_hausdorff(s1, s2)[0], directed_hausdorff(s2, s1)[0])

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(
        f"Method 5 — Hausdorff Skeleton Distance  |  Score: {score*100:.1f}%  |  HD: {hd:.1f}px",
        fontsize=13, fontweight='bold')
    axes[0].imshow(b1, cmap='gray');        axes[0].set_title("PAN Binary");             axes[0].axis('off')
    axes[1].imshow(skel1_img, cmap='gray'); axes[1].set_title("PAN Skeleton (1px)");     axes[1].axis('off')
    axes[2].imshow(skel2_img, cmap='gray'); axes[2].set_title("Sample Skeleton (1px)");  axes[2].axis('off')

    ax = axes[3]
    ax.scatter(pts1[:, 1], -pts1[:, 0], s=1, c='#2196F3', alpha=0.6, label=f"PAN ({len(pts1)} pts)")
    ax.scatter(pts2[:, 1], -pts2[:, 0], s=1, c='#F44336', alpha=0.6, label=f"Sample ({len(pts2)} pts)")
    ax.set_title("Normalized Stroke Point Clouds\n(same 200×200 canvas)")
    ax.legend(loc='upper right', markerscale=6, fontsize=9)
    ax.set_aspect('equal'); ax.set_facecolor('#F0F0F0')
    plt.tight_layout()
    path = os.path.join(out_dir, "06_method_hausdorff.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ── 7. Final Dashboard ────────────────────────────────────────────────────────
def _save_dashboard(img1_raw, img2_raw, b1, b2, scores, out_dir) -> str:
    skel1_img, pts1 = _get_skeleton(b1)
    skel2_img, pts2 = _get_skeleton(b2)
    composite = scores["composite"]
    verdict   = scores["verdict"]
    color_map = {"MATCH": "#4CAF50", "LIKELY_MISMATCH": "#FF9800", "FORGERY_OR_MISMATCH": "#F44336"}
    verdict_color = color_map.get(verdict, "#888888")

    method_names  = ["SSIM", "pHash", "Contour\nDensity", "Ink\nDensity", "Hausdorff\nSkeleton"]
    method_keys   = ["ssim", "phash", "contour", "ink", "hausdorff"]
    method_scores = [scores[k] for k in method_keys]
    weights_list  = [0.20, 0.20, 0.15, 0.15, 0.30]
    bar_colors    = ["#5C6BC0", "#26A69A", "#EF5350", "#FFA726", "#66BB6A"]

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('#FAFAFA')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    # Top-left: original images
    ax_imgs = fig.add_subplot(gs[0, 0])
    p = cv2.cvtColor(img1_raw, cv2.COLOR_BGR2RGB)
    s = cv2.cvtColor(img2_raw, cv2.COLOR_BGR2RGB)
    gap = 20
    h = max(p.shape[0], s.shape[0])
    canvas = np.ones((h, p.shape[1] + gap + s.shape[1], 3), dtype=np.uint8) * 240
    canvas[:p.shape[0], :p.shape[1]] = p
    canvas[:s.shape[0], p.shape[1]+gap:] = s
    ax_imgs.imshow(canvas)
    ax_imgs.set_title("Input Images\n← PAN                         Sample →",
                      fontsize=10, fontweight='bold')
    ax_imgs.axis('off')

    # Top-middle: binary overlay
    ax_over = fig.add_subplot(gs[0, 1])
    ov = np.zeros((200, 200, 3), dtype=np.uint8)
    ov[:, :, 2] = b1; ov[:, :, 0] = b2
    ax_over.imshow(ov)
    ax_over.set_title("Preprocessed Overlay\nBlue=PAN  Red=Sample  Purple=Match",
                      fontsize=10, fontweight='bold')
    ax_over.axis('off')

    # Top-right: skeleton overlay
    ax_skel = fig.add_subplot(gs[0, 2])
    sk = np.zeros((200, 200, 3), dtype=np.uint8)
    sk[:, :, 2] = (skel1_img * 255).astype(np.uint8)
    sk[:, :, 0] = (skel2_img * 255).astype(np.uint8)
    ax_skel.imshow(sk)
    ax_skel.set_title("Skeleton Overlay\nBlue=PAN strokes  Red=Sample strokes",
                      fontsize=10, fontweight='bold')
    ax_skel.axis('off')

    # Bottom-left+middle: horizontal bar chart
    ax_bar = fig.add_subplot(gs[1, 0:2])
    y_pos = list(range(len(method_names)))
    bars  = ax_bar.barh(y_pos, [s * 100 for s in method_scores],
                        color=bar_colors, edgecolor='white', height=0.55)
    ax_bar.set_xlim(0, 120)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(method_names, fontsize=11)
    ax_bar.set_xlabel("Similarity Score (%)", fontsize=11)
    ax_bar.set_title("Per-Method Similarity Scores", fontsize=12, fontweight='bold')
    ax_bar.axvline(scores["thresholds"]["match"] * 100, color='gray',
               linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Match threshold ({scores["thresholds"]["match"]*100:.0f}%)')
    ax_bar.legend(fontsize=9)
    ax_bar.set_facecolor('#F5F5F5')
    for bar, sc, w in zip(bars, method_scores, weights_list):
        ax_bar.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                    f"{sc*100:.1f}%  (weight={w})", va='center', fontsize=10, fontweight='bold')

    # Bottom-right: gauge
    ax_g = fig.add_subplot(gs[1, 2])
    theta = np.linspace(0, np.pi, 300)
    for start, end, col in [(0, 0.38, '#F44336'), (0.38, 0.55, '#FF9800'), (0.55, 1.0, '#4CAF50')]:
        mask = (theta >= start * np.pi) & (theta <= end * np.pi)
        ax_g.fill_between(np.cos(theta[mask]), np.zeros(mask.sum()),
                          np.sin(theta[mask]), color=col, alpha=0.22)
        ax_g.plot(np.cos(theta[mask]), np.sin(theta[mask]), color=col, linewidth=6)
    angle = composite * np.pi
    ax_g.annotate('', xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)), xytext=(0, 0),
                  arrowprops=dict(arrowstyle='->', color='black', lw=3))
    ax_g.text(0, -0.22, f"{composite*100:.1f}%", ha='center', fontsize=22,
              fontweight='bold', color=verdict_color)
    ax_g.text(0, -0.48, verdict.replace("_", " "), ha='center', fontsize=12,
              fontweight='bold', color=verdict_color,
              bbox=dict(boxstyle='round,pad=0.3', facecolor=verdict_color, alpha=0.15))
    ax_g.text(-0.95, 0.05, '0%',   fontsize=9, color='#F44336', fontweight='bold')
    ax_g.text( 0.78, 0.05, '100%', fontsize=9, color='#4CAF50', fontweight='bold')
    ax_g.text(-0.07, 1.06, '50%',  fontsize=9, color='#FF9800', fontweight='bold')
    ax_g.set_xlim(-1.15, 1.15); ax_g.set_ylim(-0.65, 1.25)
    ax_g.set_aspect('equal'); ax_g.axis('off')
    ax_g.set_title("Composite Score", fontsize=12, fontweight='bold')

    fig.suptitle("Signature Similarity — Full Analysis Dashboard",
                 fontsize=15, fontweight='bold', y=1.01)
    path = os.path.join(out_dir, "07_final_dashboard.png")
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    return path