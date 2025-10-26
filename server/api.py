import os
import sys
from typing import Optional
from supabase import create_client


def check_validation_status(username: str) -> bool:
    """Check if a participant has passed validation."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)
    existing = (
        client.table("ai2_leaderboard")
        .select("validation_status")
        .eq("username", username)
        .execute()
    )
    rows = []
    if hasattr(existing, "data") and isinstance(existing.data, list):
        rows = existing.data
    if rows:
        return bool(rows[0]["validation_status"])
    return False


def validate_battle(username1, username2) -> bool:
    """Validate two participants for a battle."""
    return check_validation_status(username1) and check_validation_status(username2)


def update_validation_status(username: str, status: bool) -> None:
    """Update the validation status for a participant."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)
    resp = (
        client.table("ai2_leaderboard")
        .update({"validation_status": status})
        .eq("username", username)
        .execute()
    )
    if hasattr(resp, "error") and resp.error:
        # supabase-py may return error object/message
        raise RuntimeError(str(resp.error))


def create_participant(username: str) -> None:
    """Create a participant in the ai2_leaderboard table."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)

    # Exit early if user already exists
    existing = (
        client.table("ai2_leaderboard")
        .select("username")
        .eq("username", username)
        .execute()
    )
    rows = []
    if hasattr(existing, "data") and isinstance(existing.data, list):
        rows = existing.data
    elif hasattr(existing, "data") and isinstance(existing.data, dict):
        # Some client versions may return a dict when using RPCs/single; normalize
        rows = [existing.data]

    if rows:
        return

    resp = (
        client.table("ai2_leaderboard")
        .insert({"username": username, "elo": 1000})
        .execute()
    )
    if hasattr(resp, "error") and resp.error:
        # supabase-py may return error object/message
        raise RuntimeError(str(resp.error))


def main(argv: list[str]) -> int:
    """
    CLI: python server/elo.py <username1> <username2>
    Prints ELO for two users on separate lines (for Actions step usage).
    """
    u1 = argv[1]

    create_participant(u1)
    is_valid = check_validation_status(u1)
    if not is_valid:
        print(f"User {u1} has not passed validation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
