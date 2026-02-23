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
import time


class SceneCaptioner:

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pegasus_available = False
        self.batch_size = 4
        self.use_fp16 = True if torch.cuda.is_available() else False

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
            "no caption",
            "images courtesy of afp",
            "getty images",
            "reuters",
            "epa"
        ]

        if not frames:
            return []

        # Convert BGR -> RGB PIL once
        images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        batch_size = getattr(self, "batch_size", 4) or 4

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            start = time.time()

            # ---- GIT batch caption ----
            try:
                git_inputs = self.git_processor(
                    images=batch,
                    return_tensors="pt"
                ).to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_fp16 and self.device.startswith("cuda")):
                    git_ids = self.git_model.generate(**git_inputs, max_length=40, num_beams=1)

                git_captions = self.git_processor.batch_decode(git_ids, skip_special_tokens=True)
            except Exception as e:
                print(f"⚠️ GIT generation failed for batch starting at {i}: {e}")
                git_captions = [""] * len(batch)

            # ---- BLIP batch caption ----
            try:
                blip_inputs = self.blip_processor(
                    images=batch,
                    return_tensors="pt"
                ).to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_fp16 and self.device.startswith("cuda")):
                    blip_ids = self.blip_model.generate(**blip_inputs, max_length=40)

                blip_captions = [self.blip_processor.decode(x, skip_special_tokens=True).strip() for x in blip_ids]
            except Exception as e:
                print(f"⚠️ BLIP generation failed for batch starting at {i}: {e}")
                blip_captions = [""] * len(batch)

            # ---- Decide per item ----
            for idx in range(len(batch)):
                git_caption = git_captions[idx].strip() if idx < len(git_captions) else ""
                blip_caption = blip_captions[idx].strip() if idx < len(blip_captions) else ""

                git_clean = git_caption.lower()

                if (
                    len(git_clean.split()) < 4 or
                    any(p in git_clean for p in generic_phrases)
                ):
                    final_caption = blip_caption or git_caption
                else:
                    final_caption = git_caption

                captions.append(final_caption)

            duration = time.time() - start
            print(f"Processed batch {i // batch_size + 1} ({len(batch)} frames) in {duration:.2f}s")

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

        # Filter out generic/bad captions before summarization
        generic_phrases = [
            "all images are copyrighted",
            "image",
            "photo",
            "picture",
            "no caption",
            "images courtesy of afp",
            "getty images",
            "reuters",
            "epa"
        ]
        filtered = [
            c for c in frame_captions
            if c and not any(p in c.lower() for p in generic_phrases)
        ]
        if not filtered:
            filtered = frame_captions  # fallback if all filtered out

        unique = list(dict.fromkeys(filtered))
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