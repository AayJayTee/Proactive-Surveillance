# captioning/caption_frames.py

import torch
import cv2
import json
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BlipProcessor,
    BlipForConditionalGeneration
)


class SceneCaptioner:

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pegasus_available = False

        # ===============================
        # 1️⃣ GIT MODEL
        # ===============================
        print("Loading GIT-large captioning model...")
        self.git_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
        self.git_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/git-large-coco"
        ).to(self.device)
        self.git_model.eval()
        print("✓ GIT loaded")

        # ===============================
        # 2️⃣ BLIP MODEL
        # ===============================
        print("Loading BLIP captioning model...")
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.blip_model.eval()
        print("✓ BLIP loaded")

        # ===============================
        # 3️⃣ PEGASUS SUMMARIZER
        # ===============================
        try:
            print("Loading PEGASUS summarizer...")
            self.sum_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
            self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/pegasus-xsum"
            ).to(self.device)
            self.sum_model.eval()
            self.pegasus_available = True
            print("✓ PEGASUS loaded")
        except Exception as e:
            print(f"⚠️ PEGASUS failed: {e}")
            self.pegasus_available = False

    # =========================================================
    # 🔥 FRAME CAPTIONING (GIT + BLIP ENSEMBLE)
    # =========================================================
    @torch.no_grad()
    def caption_video_frames(self, frames):

        captions = []
        generic_phrases = [
            "all images are copyrighted",
            "image",
            "photo",
            "picture",
            "no caption"
        ]

        for frame in frames:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            # ---- GIT caption ----
            git_inputs = self.git_processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

            git_ids = self.git_model.generate(**git_inputs, max_length=40)
            git_caption = self.git_processor.batch_decode(
                git_ids,
                skip_special_tokens=True
            )[0].strip()

            # ---- BLIP caption ----
            blip_inputs = self.blip_processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

            blip_ids = self.blip_model.generate(**blip_inputs)
            blip_caption = self.blip_processor.decode(
                blip_ids[0],
                skip_special_tokens=True
            ).strip()

            # ---- Decide which caption to keep ----
            git_clean = git_caption.lower()

            if (
                len(git_clean.split()) < 4 or
                any(p in git_clean for p in generic_phrases)
            ):
                final_caption = blip_caption
            else:
                final_caption = git_caption

            captions.append(final_caption)

        return captions

    # =========================================================
    # 🧠 SUMMARIZATION
    # =========================================================
    @torch.no_grad()
    def build_final_caption(
        self,
        frame_captions,
        action_label=None,
        action_confidence=None
    ):

        if not frame_captions:
            return "No visual description available."

        unique = list(dict.fromkeys(frame_captions))
        text_block = " ".join(unique)

        # ---- PEGASUS summarization ----
        if self.pegasus_available:
            try:
                inputs = self.sum_tokenizer(
                    text_block,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                summary_ids = self.sum_model.generate(
                    **inputs,
                    max_length=60,
                    num_beams=4,
                    early_stopping=True
                )

                summary = self.sum_tokenizer.decode(
                    summary_ids[0],
                    skip_special_tokens=True
                ).strip()

                # fallback if bad output
                if len(summary.split()) < 5:
                    summary = text_block

            except Exception:
                summary = text_block
        else:
            summary = text_block

        # ---- Add action ----
        if action_label:
            summary += (
                f" The recognized activity is {action_label} "
                f"(confidence {action_confidence:.2f})."
            )

        return summary.strip()


def save_captions(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✓ Captions saved at: {path}")