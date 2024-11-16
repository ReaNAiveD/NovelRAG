import os

import yaml


def load_ideas() -> list[str] | None:
    if not os.path.exists('output/premise.yml'):
        return None
    with open('output/premise.yml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def store_ideas(ideas: list[str]):
    os.makedirs('output', exist_ok=True)
    with open('output/premise.yml', 'w', encoding='utf-8') as f:
        yaml.safe_dump({
            "ideas": ideas
        }, f, allow_unicode=True)
