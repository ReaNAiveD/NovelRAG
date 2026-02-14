import asyncio
import logging
from argparse import ArgumentParser
from pathlib import Path

from pyaml_env import parse_config as parse_config_with_env

from novelrag.cli.session import Session
from novelrag.config.novel_rag import NovelRagConfig
from novelrag.cli.shell import NovelShell
from novelrag.tracer import Tracer, YAMLExporter

logger = logging.getLogger(__name__)


async def run(config_path: str, verbosity: int, request: str | None = None):
    azure_logger = logging.getLogger('azure')
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
        azure_logger.setLevel(logging.WARNING)
    else:
        level = logging.DEBUG
    logging.basicConfig(level=level)

    # Initialize the tracer before any LLM models are built so that
    # ChatLLMFactory.build() can attach the callback handler.
    log_dir = Path(config_path).parent / "logs"
    exporter = YAMLExporter(output_dir=log_dir)
    tracer = Tracer(exporter=exporter)
    tracer_token = tracer.activate()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = parse_config_with_env(data=f, tag=None)
            config = NovelRagConfig.model_validate(config)
            logger.debug(f"Loaded config: {config}")
            session = await Session.from_config(config)
            shell = NovelShell(session)
            if request:
                await shell.handle_command(request)
            else:
                await shell.run()
    finally:
        tracer.deactivate(tracer_token)


if __name__ == "__main__":
    parser = ArgumentParser('Novel RAG')
    parser.add_argument('--config', required=True, help="Path to the configuration file")
    parser.add_argument('-v', action='count', help="Verbosity level. -v for INFO, -vv for DEBUG")
    parser.add_argument('request', nargs='?', help="Optional request to execute")
    ns = parser.parse_args()
    asyncio.run(run(ns.config, ns.v, ns.request))
