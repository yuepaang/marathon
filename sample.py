import argparse
import marathon
import random


class RealGame(marathon.Game):
    def __init__(self, match_id):
        super().__init__(match_id=match_id)

    def on_game_start(self, data):
        pass

    def on_game_state(self, data: marathon.MessageGameState):
        action = {}
        direction = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        for k, v in data.get_states().items():
            # print(k,v.get_pos())
            # print(k,v.get_role())
            # print(k,v)
            action[k] = random.choice(direction)
        return action

    def on_game_end(self, data):
        pass
        # print(data)

    def on_game_over(self, data):
        pass
        # print(data)


if __name__ == "__main__":
    # room =sys.argv[1]
    ps = argparse.ArgumentParser()
    ps.add_argument("-room", type=str, required=True)
    args = ps.parse_args()
    room = args.room
    g = RealGame(match_id=room)
    # 给一个同步的run
    g.sync_run()
