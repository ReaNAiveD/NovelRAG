import asyncio
from argparse import ArgumentParser

from novelrag.run import run

if __name__ == "__main__":
    parser = ArgumentParser('Novel RAG')
    parser.add_argument('--config', required=True, help="Path to the configuration file")
    ns = parser.parse_args()
    asyncio.run(run(ns.config))
