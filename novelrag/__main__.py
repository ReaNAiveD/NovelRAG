import asyncio
from argparse import ArgumentParser

from novelrag.run import run

if __name__ == "__main__":
    parser = ArgumentParser('Novel RAG')
    parser.add_argument('--config', required=True, help="Path to the configuration file")
    parser.add_argument('-v', action='count', help="Verbosity level. -v for INFO, -vv for DEBUG")
    ns = parser.parse_args()
    asyncio.run(run(ns.config, ns.v))
