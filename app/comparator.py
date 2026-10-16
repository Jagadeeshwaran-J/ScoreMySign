import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.morphology import skeletonize
from scipy.spatial.distance import directed_hausdorff


def preprocess(image_bytes: bytes, target_size=(200, 200)) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure it is a valid PNG/JPG.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 5
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)
        binary = binary[y:y + h, x:x + w]
    return cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)


def ssim_score(b1, b2) -> float:
    s, _ = ssim_fn(b1, b2, gaussian_weights=True, sigma=1.5,
                   use_sample_covariance=False, full=True)
    return float(max(0.0, s))


def phash_score(b1, b2) -> float:
    def _hash(img):
        r   = cv2.resize(img, (32, 32))
        dct = cv2.dct(np.float32(r))
        low = dct[:8, :8]
        avg = low.mean()
        return "".join("1" if v > avg else "0" for v in low.flatten())
    h1, h2 = _hash(b1), _hash(b2)
    return float(1.0 - sum(a != b for a, b in zip(h1, h2)) / len(h1))


def contour_score(b1, b2) -> float:
    c1, _ = cv2.findContours(b1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c2, _ = cv2.findContours(b2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n1, n2 = len(c1), len(c2)
    if n1 == 0 and n2 == 0:
        return 1.0
    return float(1.0 - abs(n1 - n2) / max(n1, n2))


def ink_density_score(b1, b2) -> float:
    r1 = np.sum(b1 > 127) / b1.size
    r2 = np.sum(b2 > 127) / b2.size
    return float(1.0 - abs(r1 - r2) / max(r1, r2, 1e-5))


def _skeleton_points(binary, canvas=200):
    skel = skeletonize(binary > 127)
    pts  = np.column_stack(np.where(skel)).astype(np.float32)
    if len(pts) == 0:
        return pts
    mn, mx = pts.min(0), pts.max(0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    return (pts - mn) / rng * (canvas - 1)


def hausdorff_score(b1, b2) -> float:
    p1 = _skeleton_points(b1)
    p2 = _skeleton_points(b2)
    if len(p1) == 0 or len(p2) == 0:
        return 0.0
    np.random.seed(42)
    s1 = p1[np.random.choice(len(p1), min(500, len(p1)), replace=False)]
    s2 = p2[np.random.choice(len(p2), min(500, len(p2)), replace=False)]
    hd = max(directed_hausdorff(s1, s2)[0], directed_hausdorff(s2, s1)[0])
    return float(max(0.0, 1.0 - hd / 100.0))


# Only this function signature changes, nothing else

WEIGHTS = {"ssim": 0.20, "phash": 0.20, "contour": 0.15, "ink": 0.15, "hausdorff": 0.30}

# Default thresholds — change these values anytime
DEFAULT_MATCH_THRESHOLD    = 0.55   # ← above this = MATCH
DEFAULT_MISMATCH_THRESHOLD = 0.50   # ← below this = FORGERY_OR_MISMATCH


def compute_similarity(
    b1: np.ndarray,
    b2: np.ndarray,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    mismatch_threshold: float = DEFAULT_MISMATCH_THRESHOLD,
) -> dict:
    scores = {
        "ssim":      ssim_score(b1, b2),
        "phash":     phash_score(b1, b2),
        "contour":   contour_score(b1, b2),
        "ink":       ink_density_score(b1, b2),
        "hausdorff": hausdorff_score(b1, b2),
    }
    composite = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    scores["composite"] = round(composite, 4)
    scores["verdict"] = (
        "MATCH" if composite >= match_threshold else
        "LIKELY_MISMATCH" if composite >= mismatch_threshold else
        "FORGERY_OR_MISMATCH"
    )
    scores["thresholds"] = {
        "match":    match_threshold,
        "mismatch": mismatch_threshold,
    }
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores.items()}




