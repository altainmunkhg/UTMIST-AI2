"""
double_elim_tourn.py

Brief description:
------------------
This module implements a double-elimination tournament system.
Each competitor must lose twice to be eliminated. Matches are tracked
using `Match` and `Participant` classes, and the tournament logic
handles both winners and losers brackets.

Classes:
--------
- Participant: Represents a placeholder for a competitor.
- Match: Represents a match between two participants.
- Tournament: Manages the bracket creation and match progression.

Usage:
------
Create a Tournament instance with a list of competitors (strings or objects).
Iterate over the matches, use `set_winner()` to progress the tournament.
Call `get_active_matches()` to find matches ready to play.

Example:
--------
    competitors = ["Alice", "Bob", "Charlie", "Dave"]
    tournament = Tournament(competitors)

    for match in tournament.get_active_matches():
        # Simulate a winner (in practice, you'd get this from user input)
        match.set_winner(match.get_participants()[0].get_competitor())

Author: Ambrose Ling
Date: 2025-05-25
"""


class Participant:

    def __init__(self, competitor=None):
        self.competitor = competitor

    def __repr__(self) -> str:
        return f"<Participant {self.competitor}>"

    def get_competitor(self):
        """
        Return the competitor that was set,
        or None if it hasn't been decided yet
        """
        return self.competitor

    def set_competitor(self, competitor):
        """
        Set competitor after you've decided who it will be,
        after a previous match is completed.
        """
        self.competitor = competitor


class Match:
    """
    A match represents a single match in a tournament, between 2 participants.
    It adds empty participants as placeholders for the winner and loser,
    so they can be accessed as individual object pointers.
    """

    def __init__(self, left_participant, right_participant):
        self.__left_participant = left_participant
        self.__right_participant = right_participant
        self.__winner = Participant()
        self.__loser = Participant()

    def __repr__(self) -> str:
        left = self.__left_participant
        right = self.__right_participant
        winner = self.__winner
        loser = self.__loser
        return f"<Match left={left} right={right} winner={winner} loser={loser}>"

    def set_winner(self, competitor):
        """
        When the match is over, set the winner competitor here and the loser will be set too.
        """
        if competitor == self.__left_participant.get_competitor():
            self.__winner.set_competitor(competitor)
            self.__loser.set_competitor(self.__right_participant.get_competitor())
        elif competitor == self.__right_participant.get_competitor():
            self.__winner.set_competitor(competitor)
            self.__loser.set_competitor(self.__left_participant.get_competitor())
        else:
            raise Exception("Invalid competitor")

    def get_winner_participant(self):
        """
        If the winner is set, get it here. Otherwise this return None.
        """
        return self.__winner

    def get_loser_participant(self):
        """
        If the winner is set, you can get the loser here. Otherwise this return None.
        """
        return self.__loser

    def get_participants(self):
        """
        Get the left and right participants in a list.
        """
        return [self.__left_participant, self.__right_participant]

    def is_ready_to_start(self):
        """
        This returns True if both of the participants coming in have their competitors "resolved".
        This means that the match that the participant is coming from is finished.
        It also ensure that the winner hasn't been set yet.
        """
        is_left_resolved = self.__left_participant.get_competitor() is not None
        is_right_resolved = self.__right_participant.get_competitor() is not None
        is_winner_resolved = self.__winner.get_competitor() is not None
        return is_left_resolved and is_right_resolved and not is_winner_resolved


class Tournament:
    """
    This is a double-elimination tournament where each match is between 2 competitors.
    When a competitor loses they are sent to the losers bracket where they'll play until
    they lose again or they make it to the final match against the winner of the winners bracket.
    It does not handle a second "grand finals" match, that should be handled outside of this object.
    It takes in a list of competitors, which can be strings or any type of Python object,
    but they should be unique. They should be ordered by a seed, with the first entry being the most
    skilled and the last being the least. They can also be randomized before creating the instance.
    Optional options dict fields:
    """

    def __init__(self, competitors_list, bracket_reset_finals=True):
        assert len(competitors_list) > 1
        self.__matches = []
        self.__bracket_reset_finals = bracket_reset_finals
        next_higher_power_of_two = int(
            math.pow(2, math.ceil(math.log2(len(competitors_list))))
        )

        winners_number_of_byes = next_higher_power_of_two - len(competitors_list)

        incoming_participants = list(map(Participant, competitors_list))
        incoming_participants.extend([None] * winners_number_of_byes)
        last_winner = None
        last_loser = None

        losers_by_round = []
        while len(incoming_participants) > 1:
            losers = []
            half_length = int(len(incoming_participants) / 2)
            first = incoming_participants[0:half_length]
            last = incoming_participants[half_length:]
            last.reverse()

            next_round_participants = []
            for participant_pair in zip(first, last):
                if participant_pair[1] is None:
                    next_round_participants.append(participant_pair[0])
                elif participant_pair[0] is None:
                    next_round_participants.append(participant_pair[1])
                else:
                    match = Match(participant_pair[0], participant_pair[1])
                    next_round_participants.append(match.get_winner_participant())
                    last_winner = match.get_winner_participant()
                    losers.append(match.get_loser_participant())
                    self.__matches.append(match)
            if len(losers) > 0:
                losers_by_round.append(losers)
            incoming_participants = next_round_participants
        if winners_number_of_byes > 0 and len(losers_by_round) > 1:
            losers_by_round[1].extend(losers_by_round[0])
            losers_by_round = losers_by_round[1:]

        empty_by_round = []
        for __ in losers_by_round:
            empty_by_round.append([])
        losers_by_round = list(itertools.chain(*zip(losers_by_round, empty_by_round)))
        if len(losers_by_round) > 2:
            new_losers = [losers_by_round[0]]
            new_losers.extend(losers_by_round[2:])
            losers_by_round = new_losers

        for loser_round in range(0, len(losers_by_round), 4):
            losers_by_round[loser_round].reverse()

        index = 0
        incoming_participants = []
        for losers in losers_by_round:
            incoming_participants = losers

            if len(incoming_participants) > 1:
                next_higher_power_of_two = int(
                    math.pow(2, math.ceil(math.log2(len(incoming_participants))))
                )
                number_of_byes = next_higher_power_of_two - len(incoming_participants)
                incoming_participants.extend([None] * number_of_byes)
                half_length = math.ceil(len(incoming_participants) / 2)
                first = incoming_participants[0:half_length]
                last = incoming_participants[half_length:]
                last.reverse()

                incoming_participants = []
                for participant_pair in zip(first, last):
                    if participant_pair[0] is None:
                        incoming_participants.append(participant_pair[1])
                    elif participant_pair[1] is None:
                        incoming_participants.append(participant_pair[0])
                    else:
                        match = Match(participant_pair[0], participant_pair[1])
                        incoming_participants.append(match.get_winner_participant())
                        self.__matches.append(match)
                if len(incoming_participants) > 0:
                    if len(losers_by_round) <= index + 1:
                        losers_by_round.append(incoming_participants)
                    else:
                        losers_by_round[index + 1].extend(incoming_participants)

            elif len(losers_by_round) > index + 1:
                losers_by_round[index + 1].extend(incoming_participants)

            if len(incoming_participants) == 1:
                last_loser = incoming_participants[0]
            index += 1

        # Finals match
        finals_match = Match(last_winner, last_loser)
        self.__matches.append(finals_match)
        self.__finals_match = finals_match

        if bracket_reset_finals:
            bracket_reset_finals_match = Match(
                finals_match.get_winner_participant(),
                finals_match.get_loser_participant(),
            )
            self.__matches.append(bracket_reset_finals_match)

            self.__winner = bracket_reset_finals_match.get_winner_participant()
            self.__bracket_reset_finals_match = bracket_reset_finals_match
        else:
            self.__winner = finals_match.get_winner_participant()

    def __iter__(self):
        return iter(self.__matches)

    def __repr__(self) -> str:
        winner = self.__winner
        num_matches = len(self.__matches)
        return f"<Tournament winner={winner} num_matches={num_matches}>"

    def get_active_matches(self):
        """
        Returns a list of all matches that are ready to be played.
        """
        return [match for match in self.get_matches() if match.is_ready_to_start()]

    def get_matches(self):
        """
        Returns a list of all matches for the tournament.
        """
        return self.__matches

    def get_active_matches_for_competitor(self, competitor):
        """
        Given the string or object of the competitor that was supplied
        when creating the tournament instance,
        returns a list of Match's that they are currently playing in.
        """
        matches = []
        for match in self.get_active_matches():
            competitors = [
                participant.get_competitor() for participant in match.get_participants()
            ]
            if competitor in competitors:
                matches.append(match)
        return matches

    def get_winners(self):
        """
        Returns None if the tournament is done, otherwise
        returns list of the one victor.
        """
        if len(self.get_active_matches()) > 0:
            return None
        return [self.__winner.get_competitor()]

    def add_win(self, match, competitor):
        """
        Set the victor of a match, given the competitor string/object and match.
        """
        match.set_winner(competitor)
        if self.__bracket_reset_finals:
            finals = self.__finals_match
            bracket_reset = self.__bracket_reset_finals_match
            if finals.get_winner_participant().get_competitor() is not None:
                if bracket_reset.get_winner_participant().get_competitor() is None:
                    if (
                        finals.get_winner_participant().get_competitor()
                        is finals.get_participants()[0].get_competitor()
                    ):
                        self.add_win(
                            bracket_reset,
                            finals.get_winner_participant().get_competitor(),
                        )
