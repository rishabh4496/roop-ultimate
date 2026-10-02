"""Face-recognition model registry and asset resolver.

One table says, for every recogniser the app can load: where the ONNX comes from, the
SHA-256 it must have, and how a crop must be preprocessed before it is fed in
(colour order, mean/std, input size, output width). Nothing here imports onnxruntime
or touches a session -- it only answers "where is the file, and is it the right file".

Provenance rule: an entry exists ONLY if its URL answered and its hash was taken from
the host's own metadata (Hugging Face LFS oid / git-LFS pointer) and re-checked by
downloading it. A registry full of guessed hashes fails at first use with a
"mismatch" that reads like corruption, so MagFace, CosFace and GhostFaceNet are
deliberately NOT registered: no ONNX export of any of them could be found (MagFace
and CosFace ship PyTorch checkpoints, GhostFaceNet Keras ones). Add them here once
an export exists and its hash has been measured -- ``RecognitionModelSpec`` already
carries ``extract_quality_score`` for the MagFace case.

Two registry facts that are easy to miss:
  * ``antelopev2`` and ``glintr100`` are the SAME network. The antelopev2 pack's
    recogniser is ``glintr100.onnx``; there is no separate "glacr100". Both keys
    resolve to one file on disk so it is stored (and downloaded) once.
  * ``default`` resolves to ``buffalo_l/w600k_r50.onnx`` -- the path the rest of the
    app already reads -- so an existing install is reused, not downloaded again.
"""

import hashlib
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm


@dataclass(frozen=True)
class RecognitionModelSpec:
    name: str                                   # unique registry key
    display_name: str                           # human-readable UI title
    url: str                                    # direct download endpoint
    filename: str                               # path under models_dir ('/' allowed)
    sha256: str                                 # lowercase hex digest of the file
    input_size: Tuple[int, int] = (112, 112)    # (height, width)
    color_space: str = "RGB"                    # channel order the network expects
    mean: Tuple[float, float, float] = (127.5, 127.5, 127.5)   # per-channel, 0-255 scale
    std: Tuple[float, float, float] = (127.5, 127.5, 127.5)
    output_dim: int = 512
    extract_quality_score: bool = False         # True: embedding norm is a quality score
    description: str = ""                      # one or two plain sentences for the UI


_HF = "https://huggingface.co"

RECOGNITION_REGISTRY: Dict[str, RecognitionModelSpec] = {
    # InsightFace buffalo_l recogniser. The same bytes feed the swapper's identity
    # vector, so this entry is also a statement of what "the" w600k file is.
    "default": RecognitionModelSpec(
        name="default",
        display_name="ArcFace ResNet-50 (w600k_r50)",
        description='InsightFace buffalo_l ResNet-50 trained on WebFace600K. Its vector also drives the swapper, so it is the reference identity space.',
        url=f"{_HF}/immich-app/buffalo_l/resolve/main/recognition/model.onnx",
        filename="buffalo_l/w600k_r50.onnx",
        sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        color_space="RGB",
    ),
    # Same file recognizer_adaface.py loads (WebFace4M; the 12M weights are
    # non-commercial). BGR in, (x/255 - 0.5)/0.5; output is NOT unit-norm.
    "adaface": RecognitionModelSpec(
        name="adaface",
        display_name="AdaFace IR-101 (WebFace4M)",
        description='AdaFace IR-101 trained on WebFace4M; a quality-adaptive margin makes it steadier on blurred and low-light faces. It has its own distance scale: do not reuse w600k thresholds.',
        url=f"{_HF}/Evn9172/cvlface_adaface_ir101_webface4m_onnx/resolve/main/adaface_ir101.onnx",
        filename="adaface_ir101.onnx",
        sha256="d177da5864546e761579af1a91e308d7c868a32670f37e7b8b04628a78d7e5b5",
        color_space="BGR",
    ),
    "glintr100": RecognitionModelSpec(
        name="glintr100",
        display_name="ArcFace Glint-R100 (Glint360k)",
        description="ArcFace ResNet-100 trained on Glint360K, the largest backbone offered here and the recogniser in InsightFace's antelopev2 pack.",
        url=f"{_HF}/DIAMONIK7777/antelopev2/resolve/main/glintr100.onnx",
        filename="glintr100.onnx",
        sha256="4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf",
        color_space="RGB",
    ),
    # Alias of glintr100 -- see the module docstring. Same file, same hash.
    "antelopev2": RecognitionModelSpec(
        name="antelopev2",
        display_name="Antelopev2 (Glint-R100)",
        description="The antelopev2 pack's recogniser. It is the same file as Glint-R100 and is listed under both names.",
        url=f"{_HF}/DIAMONIK7777/antelopev2/resolve/main/glintr100.onnx",
        filename="glintr100.onnx",
        sha256="4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf",
        color_space="RGB",
    ),
    # buffalo_s recogniser.
    "mobilefacenet": RecognitionModelSpec(
        name="mobilefacenet",
        display_name="MobileFaceNet (w600k_mbf)",
        description='MobileFaceNet trained on WebFace600K (InsightFace buffalo_s). The smallest backbone and the one to prefer on CPU-only machines.',
        url=f"{_HF}/immich-app/buffalo_s/resolve/main/recognition/model.onnx",
        filename="w600k_mbf.onnx",
        sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
        color_space="RGB",
    ),
    # OpenCV's FaceRecognizerSF takes a raw 0-255 BGR crop (no mean/std) and
    # returns a 128-d feature.
    "facerecognizersf": RecognitionModelSpec(
        name="facerecognizersf",
        display_name="OpenCV FaceRecognizerSF (SFace)",
        description='OpenCV Zoo SFace (2021 Dec): 128-d features from raw BGR 112x112 crops. The graph has a fixed batch of 1.',
        url=("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx"),
        filename="face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        color_space="BGR",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        output_dim=128,
    ),
}


def get_registered_models() -> List[str]:
    """Keys of every supported recognition model."""
    return list(RECOGNITION_REGISTRY)


def get_model_spec(model_name: str) -> RecognitionModelSpec:
    """Specification for one model key; ValueError names the valid keys."""
    try:
        return RECOGNITION_REGISTRY[model_name]
    except KeyError:
        raise ValueError(
            f"Unsupported recognition model: '{model_name}'. "
            f"Valid options: {get_registered_models()}"
        ) from None


def compute_file_sha256(filepath: str, block_size: int = 1 << 20) -> str:
    """SHA-256 of a local file, lowercase hex."""
    digest = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


# (abs path) -> (size, mtime_ns, sha256) of a file already proven to match its spec.
# Re-hashing a 260 MB model on every resolve costs ~0.3 s; stat is free.
_verified: Dict[str, Tuple[int, int, str]] = {}
_lock = threading.Lock()


def _is_verified(path: str, expected: str) -> bool:
    """True when `path` exists and its bytes hash to `expected`."""
    st = os.stat(path)
    stamp = (st.st_size, st.st_mtime_ns, expected)
    if _verified.get(path) == stamp:
        return True
    if compute_file_sha256(path) != expected:
        return False
    _verified[path] = stamp
    return True


def _download(spec: RecognitionModelSpec, destination: str) -> None:
    """Stream spec.url to `destination`, verifying before it becomes visible.

    The bytes land in a private temp file and are renamed over `destination` only
    after the hash matches, so a failed or interrupted download never leaves a
    truncated file where a model is expected -- and a mismatching file already
    sitting at `destination` is replaced, never deleted first.
    """
    temp = f"{destination}.{uuid.uuid4().hex[:8]}.part"
    digest = hashlib.sha256()
    try:
        try:
            response = requests.get(spec.url, stream=True, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not download {spec.filename} from {spec.url}: {exc}") from exc

        total = int(response.headers.get("content-length", 0))
        received = 0
        with response, open(temp, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True,
            desc=f"Fetching {spec.filename}", ncols=80,
        ) as bar:
            try:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                        bar.update(len(chunk))
            except requests.RequestException as exc:
                raise RuntimeError(f"Download of {spec.filename} was interrupted: {exc}") from exc

        if total and received != total:
            raise RuntimeError(
                f"Truncated download of {spec.filename}: got {received} of {total} bytes.")
        actual = digest.hexdigest()
        if actual != spec.sha256.lower():
            raise RuntimeError(
                f"Cryptographic hash mismatch for {spec.filename}! "
                f"Expected: {spec.sha256}, calculated: {actual}.")
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


def resolve_model_path(model_name: str, models_dir: str) -> str:
    """Absolute path of the recogniser's ONNX, downloading + verifying it if needed.

    An existing file is reused only if its SHA-256 matches the registry; otherwise it
    is replaced by a verified download. Raises RuntimeError on a download failure or a
    hash mismatch (nothing is left at the destination in that case beyond what was
    already there), ValueError for an unknown `model_name`.
    """
    spec = get_model_spec(model_name)
    expected = spec.sha256.lower()
    destination = os.path.abspath(os.path.join(models_dir, spec.filename))
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with _lock:
        if os.path.isfile(destination) and _is_verified(destination, expected):
            return destination
        _download(spec, destination)
        st = os.stat(destination)
        _verified[destination] = (st.st_size, st.st_mtime_ns, expected)
    return destination
