"""Command-line entry point for EgoGaze-Thor (live / demo / replay / benchmark)."""

from __future__ import annotations

import argparse
import logging
import uvicorn

from egogaze.core.config import load_config
from egogaze.pipeline import EgoGazePipeline
from egogaze.web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="EgoGaze-Thor Multimodal Perception System")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--mode", choices=["live", "demo", "replay"], default=None, help="Run mode")
    parser.add_argument("--host", type=str, default=None, help="Web server host")
    parser.add_argument("--port", type=int, default=None, help="Web server port")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_level = cfg.get("system", {}).get("log_level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = EgoGazePipeline(cfg, mode=args.mode)
    app = create_app(pipeline)

    host = args.host or cfg.get("web", {}).get("host", "0.0.0.0")
    port = args.port or int(cfg.get("web", {}).get("port", 8080))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
