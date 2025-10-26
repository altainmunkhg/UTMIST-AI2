import os
from typing import Optional
from supabase import create_client


def elo_update(elo1, elo2, result, k=32):
    """
    Update two ELO scores based on result.
    :param elo1: Player 1 ELO
    :param elo2: Player 2 ELO
    :param result: 1 if player 1 wins, 0 if player 1 loses, 0.5 for draw
    :param k: K-factor (default 32)
    :return: (new_elo1, new_elo2)
    """
    # Calculate expected scores
    expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    expected2 = 1 / (1 + 10 ** ((elo1 - elo2) / 400))
    # Update ratings
    new_elo1 = elo1 + k * (result - expected1)
    new_elo2 = elo2 + k * ((1 - result) - expected2)
    return new_elo1, new_elo2


def get_participant_elo(username: str) -> int:
    """Read ELO for a participant from the ai2_leaderboard table."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    if not key:
        raise RuntimeError(
            "Missing Supabase key in environment (SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY)"
        )

    client = create_client(url, key)

    resp = (
        client.table("ai2_leaderboard")
        .select("elo")
        .eq("username", username)
        .single()
        .execute()
    )
    if getattr(resp, "data", None) and "elo" in resp.data:
        return resp.data["elo"]
    raise ValueError(f"No ELO found for user: {username}")


def update_participant_elo(
    username: str, elo: int, match_result: Optional[str] = None
) -> None:
    """Update ELO for a participant in the ai2_leaderboard table.

    match_result is accepted for compatibility but not used here.
    """
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)
    resp = (
        client.table("ai2_leaderboard")
        .update({"elo": elo})
        .eq("username", username)
        .execute()
    )

    if hasattr(resp, "error") and resp.error:
        # supabase-py may return error object/message
        raise RuntimeError(str(resp.error))
