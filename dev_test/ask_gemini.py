"""Send one image to Gemini and ask whether SynthID is present.
Appends the prompt + response to a log file.
"""

import argparse
import os
import sys
from datetime import datetime

import dotenv
from google import genai
from google.genai import types
from PIL import Image

PROMPT = (
    "I am inspecting this image for the presence of a SynthID watermark "
    "(Google DeepMind's invisible image watermark for AI-generated content). "
    "Please answer the following:\n"
    "1) Is a SynthID watermark present in this image? (yes / no / uncertain)\n"
    "2) What is your confidence (low / medium / high)?\n"
    "3) Briefly explain your reasoning, including whether you can actually "
    "detect SynthID or whether you are inferring from visual content alone.\n"
    "Respond concisely."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image to inspect")
    ap.add_argument("--log", default="dev_test/gemini_synthid_log.txt")
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    dotenv.load_dotenv("/home/trabbani/WAVES/.env", override=False)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    img = Image.open(args.image)
    print(f"image: {args.image}  size={img.size}  mode={img.mode}")
    print(f"model: {args.model}")

    resp = client.models.generate_content(
        model=args.model,
        contents=[PROMPT, img],
    )
    text = resp.text or "(empty response)"

    print("\n--- Gemini response ---")
    print(text)
    print("-----------------------")

    log_entry = (
        f"=== {datetime.now().isoformat(timespec='seconds')} ===\n"
        f"image: {args.image}\n"
        f"model: {args.model}\n"
        f"prompt:\n{PROMPT}\n"
        f"response:\n{text}\n\n"
    )
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "a") as f:
        f.write(log_entry)
    print(f"appended to: {args.log}")


if __name__ == "__main__":
    main()
