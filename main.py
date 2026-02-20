"""
BrosephTech — Automated YouTube Description Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW:
  1. Drop a video file path into the script
  2. ElevenLabs transcribes it and returns an SRT file
  3. Claude reads the SRT and generates timestamps + description
  4. Final description is printed and saved to a .txt file
"""

import os
import sys
import requests
import anthropic
from dotenv import load_dotenv

# ── CONFIG — LOADS KEYS FROM .env FILE ────────────────────────────────────────

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")

if not ELEVENLABS_API_KEY or not ANTHROPIC_API_KEY:
    print("Error: API keys not found. Make sure your .env file exists and has both keys.")
    sys.exit(1)

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a YouTube description generator for a TFT (Teamfight Tactics) content
channel called BrosephTech. Follow all rules and guidelines below exactly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESCRIPTION FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every description must follow this exact structure:

#TFT #TeamfightTactics #TFTGuide #TFTMeta #CompetitiveTFT #TFTComps #TFTStrategy #TFTPatch

[Two sentence description]

Slides: https://docs.google.com/presentation/...

---

[Timestamps]

---

Follow BrosephTech:
Twitter:   / brosephtech  
TikTok:   / brosephtech

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESCRIPTION WRITING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The two sentence description follows this formula:
  Sentence 1 — Bold claim or tension hook about the patch or comp
  Sentence 2 — Promise of what the viewer will gain

Examples of good descriptions:
  "Asol is the hardest to play but also the most broken comp in 16.3!
   In this video, I'll walk you through how to execute this line smoothly."

  "Riot is taking a jab at making a big meta swing for patch 16.5.
   Let's go over these changes and come up with an early meta strategy together."

  "Patch 16.4 has more playable comps than people think.
   Here's every strong line outside the top five so you always have something
   to pilot no matter what your opener looks like."

Rules:
  - Short and punchy — never list the content of the video
  - The timestamps handle the content breakdown
  - For series episodes, reference the previous episode in sentence 2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIMESTAMP RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read exact timestamps directly from the SRT. Never estimate.

Hard transition signals (always catch these):
  - "Next comp" / "Next is" / "First off" / "Next"
  - "All right, [topic]"
  - Exclamatory name drops e.g. "Ruined King!"
  - "Last comp" / "That's it for"
  - "Let's start with" / "Let us get started with"

Soft transition signals (read content not just keywords):
  - Any pivot from one named subject to another even mid-flow
  - "The other version" / "level-up version" / "there's two ways"
  - Topic shifts without hard keywords
  - Community comp names won't always match transcript — identify the concept

Timestamp labeling rules:
  - Every new distinct topic = new timestamp regardless of size
  - Never merge intro with first content section
  - Sub-sections are their own timestamps
  - Use carry/champion names when more informative than trait names
  - Hero augment comps: "Champion Name (Hero)" e.g. "Viego (Hero)"

Timestamp format:
  0:00 Intro
  0:15 [Section Name]
  3:31 [Section Name]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERIES EPISODE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the video is part of a series:
  - Reference the previous episode in the description
  - Keep the same format and hashtags

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT TERMINOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Comps           = Team compositions
  Low/High Cost   = Unit gold costs
  Augments        = Mid-game power-up choices
  Reroll          = Rolling at low level for 3-star units
  Hero Augments   = Augments tied to specific champions
  Meta Playbook   = Guide covering strongest comps
  Meta Prediction = Analysis of upcoming patch changes
  Opener          = Early game board setup
  Cap             = Strongest possible late game board
"""

# ── STEP 1: TRANSCRIBE VIDEO WITH ELEVENLABS ───────────────────────────────────

def transcribe_video_to_srt(video_path: str) -> str:
    """
    Sends the video file to ElevenLabs Speech-to-Text API
    and returns the transcript as an SRT string.
    """
    print(f"[1/3] Transcribing video: {video_path}")

    url = "https://api.elevenlabs.io/v1/speech-to-text"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }

    with open(video_path, "rb") as video_file:
        files = {
            "file": (os.path.basename(video_path), video_file, "video/mp4")
        }
        data = {
            "output_format": "srt"   # Request SRT format directly
        }
        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        raise Exception(
            f"ElevenLabs transcription failed: {response.status_code} — {response.text}"
        )

    srt_content = response.text
    print("[1/3] Transcription complete.")
    return srt_content


# ── STEP 2: GENERATE DESCRIPTION WITH CLAUDE ──────────────────────────────────

def generate_description(srt_content: str) -> str:
    """
    Sends the SRT transcript to Claude with the BrosephTech system prompt
    and returns the fully formatted YouTube description.
    """
    print("[2/3] Sending SRT to Claude for description generation...")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Here is the SRT transcript for the video. "
                    "Please generate the full YouTube description with timestamps "
                    "following all the rules in your instructions.\n\n"
                    f"{srt_content}"
                )
            }
        ]
    )

    description = message.content[0].text
    print("[2/3] Description generated.")
    return description


# ── STEP 3: SAVE OUTPUT ────────────────────────────────────────────────────────

def save_output(description: str, video_path: str) -> str:
    """
    Saves the final description to a .txt file
    in the same folder as the video.
    """
    base = os.path.splitext(video_path)[0]
    output_path = f"{base}_description.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(description)

    print(f"[3/3] Description saved to: {output_path}")
    return output_path


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def run(video_path: str):
    """
    Full automated pipeline:
      Video file → ElevenLabs SRT → Claude Description → Saved .txt file
    """
    if not os.path.exists(video_path):
        print(f"Error: File not found — {video_path}")
        sys.exit(1)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  BrosephTech Description Generator")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # Step 1 — Transcribe
    srt_content = transcribe_video_to_srt(video_path)

    # Step 2 — Generate description
    description = generate_description(srt_content)

    # Step 3 — Save and print
    output_path = save_output(description, video_path)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  FINAL DESCRIPTION:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    print(description)
    print(f"\n✓ Saved to: {output_path}")


if __name__ == "__main__":
    # ── DROP YOUR VIDEO PATH HERE ──────────────────────────────────────────────
    # Option A: hardcode the path
    VIDEO_PATH = "your_video.mp4"

    # Option B: pass it as a command line argument
    # Run with: python main.py your_video.mp4
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]

    run(VIDEO_PATH)
