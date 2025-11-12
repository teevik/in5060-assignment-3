# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: hello-world-dev-env
#     language: python
#     name: python3
# ---

# %% Imports
from typing import cast

import pandas as pd
import simdjson
from pydantic import BaseModel
from rich import print

from xdf_types import XDFData


# %% Define types for questionnaire data
class AnswerRound(BaseModel):
    round_number: int
    latency_applied: int  # Actual latency in milliseconds from LatencyMarkers stream
    blocks_moved: int  # Number of blocks moved during this round (from BoxBlockMarkers)
    delays_experienced: int  # Answer to "Did you experience delays..."
    task_difficulty: int  # Answer to "How difficult was it..."
    felt_controlling: int  # Answer to "I felt like I was controlling..."
    felt_part_of_body: int  # Answer to "It felt like the robot was part of my body"


class Participant(BaseModel):
    submission_id: int
    created: str
    participant_number: int
    gender: str
    age: int
    dominant_hand: str  # Can be "Right hand", "Left hand", "Ambidextrous", etc.
    robotics_experience: int
    answer_time_ms: float
    rounds: list[AnswerRound]


ignored_participants = [4]


# %% Load questionnaire
questionnaire_df = pd.read_csv(
    "data/questionnaire_data-561422-2025-11-11-1622.csv", sep=";"
)

# %% Parse repeating questionnaire columns
# Static columns (demographics)
static_cols = [
    "$submission_id",
    "$created",
    "Participant number",
    "What is your gender",
    "How old are you?",
    "What is your dominant hand?",
    "How experienced are you with robotic systems?",
    "$answer_time_ms",
]

# The repeating question columns
repeating_questions = [
    "Did you experience delays between your actions and the robot&#39;s movements?",
    "How difficult was it to perform the task?",
    "I felt like I was controlling the movement of the robot",
    "It felt like the robot was part of my body",
]

# Count how many rounds we have
all_cols = questionnaire_df.columns.tolist()
# Remove static columns to get only repeating ones
repeating_cols = [col for col in all_cols if col not in static_cols]
num_rounds = len(repeating_cols) // len(repeating_questions)

print(f"Number of rounds: {num_rounds}")

# %% Create typed list of participants
participants: list[Participant] = []

for _, row in questionnaire_df.iterrows():
    participant_number = int(row["Participant number"])
    if participant_number in ignored_participants:
        continue

    data_file = f"./data/sub-{participant_number:03d}/sub-{participant_number:03d}_ses-_task-_run-001.json"
    json_parser = simdjson.Parser()
    data: XDFData = cast(XDFData, json_parser.load(data_file))

    # Extract latency markers from the data
    latency_markers_stream = next(
        (
            stream
            for stream in data["streams"]
            if stream["info"]["name"] == "LatencyMarkers"
        ),
        None,
    )

    if latency_markers_stream is None:
        raise ValueError(
            f"LatencyMarkers stream not found for participant {participant_number}"
        )

    # Extract latencies from condition_advance markers
    # Format: ["condition_advance|rep_1|200ms|condition_1"]
    latencies_by_round = []
    for marker in latency_markers_stream["time_series"]:
        marker_str = marker[0]
        if marker_str.startswith("condition_advance|"):
            parts = marker_str.split("|")
            if len(parts) >= 3:
                latency_str = parts[2]  # e.g., "200ms"
                latency_ms = int(latency_str.replace("ms", ""))
                latencies_by_round.append(latency_ms)

    # Extract ExpMarkers stream to count block_moved events
    # Note: Some participants have duplicate ExpMarkers streams (identical events)
    # We only need to process one of them
    exp_markers_stream = None
    for stream in data["streams"]:
        if stream["info"]["name"] == "ExpMarkers":
            exp_markers_stream = stream
            break

    if exp_markers_stream is None:
        raise ValueError("No ExpMarkers stream found")

    # Count blocks moved per round
    blocks_moved_by_round = []
    current_round_blocks = 0
    in_boxblock = False

    for i, marker in enumerate(exp_markers_stream["time_series"]):
        marker_str = marker[0]

        # Handle both practice and regular boxblock sessions
        if marker_str in ["boxblock_start", "practice_boxblock_start"]:
            # If we were already in a session, save the count first
            if in_boxblock and marker_str == "boxblock_start":
                blocks_moved_by_round.append(current_round_blocks)
            in_boxblock = True
            current_round_blocks = 0
        elif marker_str in [
            "boxblock_stop",
            "practice_boxblock_stop",
            "boxblock_end",
        ]:
            if in_boxblock:
                blocks_moved_by_round.append(current_round_blocks)
                in_boxblock = False
                current_round_blocks = 0
        elif marker_str == "block_moved" and in_boxblock:
            current_round_blocks += 1

    # Ensure we have the right number of block counts
    while len(blocks_moved_by_round) < num_rounds:
        blocks_moved_by_round.append(0)

    # Parse answer rounds
    rounds = []
    for round_idx in range(num_rounds):
        # Column names have suffixes like .1, .2, etc. (pandas duplicate column naming)
        # First occurrence has no suffix, then .1, .2, .3, etc.
        suffix = f".{round_idx}" if round_idx > 0 else ""

        round_data = AnswerRound(
            round_number=round_idx + 1,
            latency_applied=latencies_by_round[round_idx]
            if round_idx < len(latencies_by_round)
            else 0,
            blocks_moved=blocks_moved_by_round[round_idx]
            if round_idx < len(blocks_moved_by_round)
            else 0,
            delays_experienced=int(row[f"{repeating_questions[0]}{suffix}"]),
            task_difficulty=int(row[f"{repeating_questions[1]}{suffix}"]),
            felt_controlling=int(row[f"{repeating_questions[2]}{suffix}"]),
            felt_part_of_body=int(row[f"{repeating_questions[3]}{suffix}"]),
        )
        rounds.append(round_data)

    # Create participant with all rounds
    participant = Participant(
        submission_id=int(row["$submission_id"]),
        created=str(row["$created"]),
        participant_number=participant_number,
        gender=str(row["What is your gender"]),
        age=int(row["How old are you?"]),
        dominant_hand=str(row["What is your dominant hand?"]),
        robotics_experience=int(row["How experienced are you with robotic systems?"]),
        answer_time_ms=float(row["$answer_time_ms"]),
        rounds=rounds,
    )
    participants.append(participant)

# %% Print summary statistics
print(participants)
