"""Development entry point for the Flask API."""

import os

from app import create_app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        # The reloader would initialise the 85 MB dual checkpoint twice.
        use_reloader=False,
    )
