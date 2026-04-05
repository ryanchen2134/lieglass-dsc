from dataclasses import dataclass, field


GESTURE_COLUMNS = [
    "OtherGestures", "Smile", "Laugh", "Scowl", "otherEyebrowMovement",
    "Frown", "Raise", "OtherEyeMovements", "Close-R", "X-Open", "Close-BE",
    "gazeInterlocutor", "gazeDown", "gazeUp", "otherGaze", "gazeSide",
    "openMouth", "closeMouth", "lipsDown", "lipsUp", "lipsRetracted",
    "lipsProtruded", "SideTurn", "downR", "sideTilt", "backHead",
    "otherHeadM", "sideTurnR", "sideTiltR", "waggle", "forwardHead",
    "downRHead", "singleHand", "bothHands", "otherHandM", "complexHandM",
    "sidewaysHand", "downHands", "upHands",
]


@dataclass
class RealLifeConfig:
    # Paths (relative to project root)
    annotation_csv: str = (
        "Data/Real-life_Deception_Detection_2016/Annotation/"
        "All_Gestures_Deceptive and Truthful.csv"
    )
    transcript_dir: str = "Data/Real-life_Deception_Detection_2016/Transcription"
    checkpoint_dir: str = "checkpoints_real_life"

    # Text encoder
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_max_length: int = 128
    d_text: int = 384

    # Annotation encoder
    d_annot_hidden: int = 128
    d_annot_out: int = 64
    n_annotations: int = 39  # number of gesture columns (id + class = 2 non-feature cols)

    # Classifier head
    d_fused: int = 448  # d_text + d_annot_out = 384 + 64
    d_hidden: int = 128
    dropout: float = 0.4

    # Training
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_epochs: int = 200
    patience: int = 40
    grad_clip: float = 1.0
    n_folds: int = 5
    seed: int = 42

    # Hardware
    device: str = "cuda"
