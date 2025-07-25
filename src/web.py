from flask import Flask, jsonify
from .mcts import MCTS
from .network import OtrioNet, policy_value
from .otrio import GameState

app = Flask(__name__)
model = OtrioNet()

@app.route("/move")
def ai_move():
    state = GameState()
    mcts = MCTS(lambda s: policy_value(model, s), num_simulations=10)
    move, _ = mcts.run(state)
    return jsonify({"row": move.row, "col": move.col, "size": move.size})

if __name__ == "__main__":
    app.run()
