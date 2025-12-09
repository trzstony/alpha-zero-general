import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class for single-player games.
    
    For single-player games, this evaluates how well a player performs
    (solved vs not solved) rather than pitting two players against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player1: function that takes board as input, returns action (primary player)
            player2: function that takes board as input, returns action (for compatibility, 
                     but not used in single-player games)
            game: Game object (single-player game)
            display: a function that takes board as input and prints it (e.g.
                     display in RewritePuzzleGame). Is necessary for verbose mode.

        Note: For single-player games, player2 is kept for interface compatibility
        but only player1 is actually used.
        """
        self.player1 = player1
        self.player2 = player2  # Kept for compatibility, not used in single-player
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a single-player game.

        Returns:
            int: Result of the game
                - 1 if the puzzle was solved (WIN)
                - -1 if max steps reached without solving (LOSS)
                - 0 if game ended in a draw (should not happen in single-player)
        """
        # For single-player games, we only use player1
        player = self.player1
        board = self.game.getInitBoard()
        it = 0

        # Notify player of game start
        if hasattr(player, "startGame"):
            player.startGame()

        # For single-player games, player is always 1
        curPlayer = 1

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Step ", str(it))
                self.display(board)

            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            action = player(canonicalBoard)

            valids = self.game.getValidMoves(canonicalBoard, curPlayer)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        # Notify player of game end
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            result = self.game.getGameEnded(board, curPlayer)
            result_str = "SOLVED" if result == 1 else "LOST (max steps)"
            print("Game over: Step ", str(it), "Result: ", result_str)
            self.display(board)
        
        # For single-player games, return the result directly (1 for win, -1 for loss)
        return self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games with the single player.

        For single-player games, this evaluates how many puzzles were solved
        vs how many were lost (max steps reached).

        Returns:
            solved: number of games where the puzzle was solved (result == 1)
            lost: number of games where max steps were reached without solving (result == -1)
            draws: number of games that ended in a draw (should be 0 for single-player)
        """
        solved = 0
        lost = 0
        draws = 0
        
        for _ in tqdm(range(num), desc="Arena.playGames"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                solved += 1
            elif gameResult == -1:
                lost += 1
            else:
                draws += 1

        return solved, lost, draws
