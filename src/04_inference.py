import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config
from utils import get_logger

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*encoder_attention_mask.*",
    category=FutureWarning,
)


# --------------------------------------------------
# Helper: single-text inference
# --------------------------------------------------
def predict_text(text: str, model, tokenizer, device):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

    pred_class = torch.argmax(probs, dim=1).item()
    pred_class = int(pred_class)
    confidence = probs[0, pred_class].item()


    return pred_class + 1, confidence, probs.squeeze().cpu().numpy()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    logger = get_logger("inference")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("=== INFERENCE STARTED ===")

    # --------------------------------------------------
    # Load model & tokenizer
    # --------------------------------------------------
    logger.info(f"Loading model from: {config.MODEL_BEST_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_BEST_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_BEST_DIR
    ).to(device)

    model.eval()

    logger.info("Model and tokenizer loaded successfully")

    # --------------------------------------------------
    # Demo texts (example inputs)
    # --------------------------------------------------
    demo_texts = [
        "A felhasználó bármikor lemondhatja a szolgáltatást.",
        "A szolgáltatás megszüntetése a jelen szerződésben meghatározott feltételek szerint történik.",
        "Ezeket a tartalmakat csak az adott személy vagy szervezet engedélyével, illetve a törvény által másképp megengedett módon használhatja fel. A más emberek vagy szervezetek tartalmaiban kifejezésre juttatott nézetek az ő saját nézeteik, és nem feltétlenül tükrözik a Google nézeteit."
    ]


    # --------------------------------------------------
    # Run inference
    # --------------------------------------------------
    for idx, text in enumerate(demo_texts, start=1):
        label, conf, probs = predict_text(text, model, tokenizer, device)

        logger.info(f"\n--- Example {idx} ---")
        logger.info(f"Text: {text}")
        logger.info(f"Predicted understandability: {label} / 5")
        logger.info(f"Confidence: {conf:.4f}")
        logger.info(
            "Class probabilities (1–5): "
            + ", ".join(f"{p:.3f}" for p in probs)
        )

    logger.info("=== INFERENCE FINISHED ===")


if __name__ == "__main__":
    main()
