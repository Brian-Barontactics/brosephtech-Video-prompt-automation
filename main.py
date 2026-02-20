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

IDENTIFYING TRANSITIONS
━━━━━━━━━━━━━━━━━━━━━━
Hard transition signals — speaker explicitly signals a new section:
  - "Next comp is..." / "Next up..." / "First up..."
  - "All right, let's talk about [name]"
  - Exclamatory champion name drop e.g. "Graves!" or "Miss Fortune!"
  - "Last comp" / "That's it for [topic]"
  - "Let's start with" / "Let's move on to"
  - Sponsor callouts e.g. "Shoutout to MetaTFT"

Soft transition signals — content shifts without explicit keywords:
  - Speaker stops discussing one comp and begins explaining a different one
  - A new trait, strategy, or playstyle is introduced and becomes the main focus
  - The board, items, or rolldown described clearly belong to a different comp
  - Level or econ strategy changes e.g. fast 8 vs reroll signals a new comp section
  - "There's another version..." / "You can also play..."

STEP 1 — IDENTIFY VIDEO TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read the full transcript first and classify the video into ONE of these types:

  TYPE A — PATCH UPDATE
  Signals: speaker discusses buffs, nerfs, balance changes, meta impact of specific
  traits or champions. Words like "nerfed", "buffed", "changed", "this patch" appear
  frequently. Sections revolve around what changed, not how to play a comp.

  TYPE B — COMP GUIDE / META PLAYBOOK
  Signals: speaker explains how to execute specific comps with boards, rolldowns,
  item holders, and augment choices. Deep tactical breakdown of one or more comps.
  Words like "rolldown", "item holders", "fast 8", "reroll", "opener" appear frequently.

  TYPE C — TIER LIST / META PREDICTION
  Signals: speaker ranks comps or previews an upcoming patch. Covers many comps
  briefly rather than going deep on any one. Words like "tier", "ranked", "prediction",
  "best comps" appear frequently.

STEP 2 — APPLY VIDEO TYPE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TYPE A — PATCH UPDATE TIMESTAMP RULES:
  - Name each section after the TRAIT or CHAMPION being discussed
  - Always append "Changes" to the label e.g. "Ixtal Changes" / "Kai'Sa Changes"
  - Absorb ALL sub-topics (augments, boards, meta impact) into one parent timestamp
  - Do NOT create sub-timestamps within a patch section
  - Group minor changes together e.g. "WW, Bel & Sera Boards"
  - Example output:
      0:00 Intro
      0:14 Tierlist
      0:43 Ixtal Changes
      1:46 Kai'Sa Changes
      3:12 Aurelion Sol Changes
      4:46 Bilgewater Changes
      5:40 WW, Bel & Sera Boards
      6:10 Outro

  TYPE B — COMP GUIDE TIMESTAMP RULES:
  - Name each section after the COMP CARRY or PLAYSTYLE
  - Use star emoji for 3-star reroll comps e.g. "Graves 3⭐" / "TF 3⭐"
  - Include playstyle in name where relevant e.g. "MF + Fizz Fast 8" / "Illaoi Fast 9"
  - Sub-sections each get their own timestamp e.g. early board, rolldown, augments
  - Group fundamentals sub-topics each with their own timestamp
  - Example output:
      0:00 Intro
      0:44 Bilgewater Fundamentals
      1:17 Feeding Tahm Kench
      1:42 Bilgewater Shop Economy
      2:37 Toggling Bilgewater Tiers
      3:12 When to Commit to Bilgewater
      3:36 Captain's Brew
      3:56 Positioning
      4:18 MetaTFT Sponsor
      4:56 Graves 3⭐
      6:36 MF + Fizz Fast 8
      8:16 TF 3⭐
      10:06 Illaoi Fast 9
      11:26 Augments
      13:06 Final Boards
      14:16 Outro

  TYPE C — TIER LIST TIMESTAMP RULES:
  - Keep timestamps broad and high level
  - Name after tier or comp group e.g. "Tierlist" / "S Tier" / "A Tier Comps"
  - Individual comp breakdowns get their own timestamp if discussed in depth
  - Example output:
      0:00 Intro
      0:30 Tierlist
      1:00 S Tier Comps
      3:00 A Tier Comps
      5:00 Outro

STEP 3 — UNIVERSAL RULES FOR ALL VIDEO TYPES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Read transcript carefully — never guess or invent champion/trait names
  - Use the closest match from the actual transcript text
  - Never merge intro with first content section
  - Sponsor segments always get their own short label e.g. "MetaTFT Sponsor"
  - Hero augment comps labeled as "Champion Name (Hero)" e.g. "Viego (Hero)"
  - Hard transition signals: "Next", "All right", "Let's talk about", exclamatory
    name drops, "Last comp", "That's it for", sponsor callouts
  - Soft transition signals: content shifts to a new comp or trait being discussed,
    new strategy or playstyle introduced, board/items clearly belong to different comp

Timestamp format example:
  0:00 Intro
  0:44 Bilgewater Fundamentals
  1:17 Feeding Tahm Kench
  1:42 Bilgewater Shop Economy
  2:37 Toggling Bilgewater Tiers
  3:12 When to Commit to Bilgewater
  3:36 Captain's Brew
  3:56 Positioning
  4:18 MetaTFT Sponsor
  4:56 Graves 3⭐
  6:36 MF + Fizz Fast 8
  8:16 TF 3⭐
  10:06 Illaoi Fast 9
  11:26 Augments
  13:06 Final Boards
  14:16 Outro

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
            "model_id": "scribe_v1",
            "output_format": "srt"
        }
        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        raise Exception(
            f"ElevenLabs transcription failed: {response.status_code} — {response.text}"
        )

    srt_content = response.text
    print("[1/3] Transcription complete.")
    return srt_content


# ── STEP 2: TRIM SRT ───────────────────────────────────────────────────────────

def trim_srt(srt_content: str, max_chars: int = 80000) -> str:
    """Trims SRT content to fit within Claude's token limit."""
    if len(srt_content) > max_chars:
        print(f"[!] SRT too long ({len(srt_content)} chars), trimming to {max_chars}...")
        return srt_content[:max_chars]
    return srt_content


# ── STEP 3: GENERATE DESCRIPTION WITH CLAUDE ──────────────────────────────────

def generate_description(srt_content: str) -> str:
    """
    Sends the SRT transcript to Claude with the BrosephTech system prompt
    and returns the fully formatted YouTube description.
    """
    print("[2/3] Sending SRT to Claude for description generation...")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
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


# ── STEP 4: SAVE OUTPUT ────────────────────────────────────────────────────────

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

    # Step 2 — Trim and generate description
    srt_content = trim_srt(srt_content)
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
    VIDEO_PATH = "your_video.mp4"

    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]

    run(VIDEO_PATH)
