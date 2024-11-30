import logging

from pyaml_env import parse_config as parse_config_with_env

from novelrag.core.storage import NovelStorage
from novelrag.editors import default_aspects
from novelrag.core.config import AspectsConfig, NovelRagConfig
from novelrag.core.shell import NovelShell

logger = logging.getLogger(__name__)


def load_aspects(config: AspectsConfig, storage: NovelStorage):
    aspect_contexts = {}
    for name in config:
        # TODO: handle custom aspects
        aspect = default_aspects[name](storage, config[name])
        aspect_contexts[name] = aspect
    return aspect_contexts


async def run(config_path: str, verbosity: int):
    logging.basicConfig(level=logging.DEBUG if verbosity > 1 else logging.INFO)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = parse_config_with_env(data=f, tag=None)
        config = NovelRagConfig.model_validate(config)
        logger.debug(f"Loaded config: {config}")
    storage = NovelStorage(config.storage)
    aspect_contexts = load_aspects(config.aspects, storage)
    shell = NovelShell(aspect_contexts)
    await shell.run()
