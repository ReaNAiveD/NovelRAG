from pyaml_env import parse_config as parse_config_with_env

from novelrag.editors import default_aspects
from novelrag.config import AspectsConfig
from novelrag.shell import NovelShell


def load_aspects(config: AspectsConfig):
    aspect_contexts = {}
    for name in config.root:
        # TODO: handle custom aspects
        aspect = default_aspects[name](config.root[name].config)
        aspect_contexts[name] = aspect
    return aspect_contexts


async def run(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = parse_config_with_env(data=f, tag=None)
        config = AspectsConfig(config)
    aspect_contexts = load_aspects(config)
    shell = NovelShell(aspect_contexts)
    await shell.run()
