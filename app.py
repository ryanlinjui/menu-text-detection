from flask import (
    Flask,
    request,
    jsonify,
    make_response,
    send_from_directory
)
from flask_cors import CORS
from datetime import datetime

from Menu import MenuInstance
from Menu.engine import (
    Vertex2TextPromptBiDir
)

app = Flask(__name__)
CORS(app)

# Path for our main Svelte page
@app.route("/")
def index():
    return send_from_directory("./web/build", "index.html")

# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def base(path):
    return send_from_directory("./web/build", path)

@app.route("/api/menu", methods=["POST"])
def menu_detection():
    image = request.files["image"].read()
    menu = MenuInstance(image)
    payload = menu.run(Vertex2TextPromptBiDir)
    menu.save(datetime.now().strftime("%Y%m%d-%H%M%S"))
    return make_response(jsonify(payload), 200)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)