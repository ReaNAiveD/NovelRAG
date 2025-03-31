import asyncio
import logging
from argparse import ArgumentParser

from pyaml_env import parse_config as parse_config_with_env

from novelrag.config.novel_rag import NovelRagConfig
from novelrag.shell import NovelShell

logger = logging.getLogger(__name__)


async def run(config_path: str, verbosity: int):
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logging.basicConfig(level=level)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = parse_config_with_env(data=f, tag=None)
        config = NovelRagConfig.model_validate(config)
        logger.debug(f"Loaded config: {config}")
        shell = await NovelShell.create(config)
        await shell.run()


if __name__ == "__main__":
    parser = ArgumentParser('Novel RAG')
    parser.add_argument('--config', required=True, help="Path to the configuration file")
    parser.add_argument('-v', action='count', help="Verbosity level. -v for INFO, -vv for DEBUG")
    ns = parser.parse_args()
    asyncio.run(run(ns.config, ns.v))
