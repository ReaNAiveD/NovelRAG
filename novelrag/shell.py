import logging
import sys

import yaml

from novelrag.config.novel_rag import NovelRagConfig
from novelrag.config.resource import AspectConfig
from novelrag.exceptions import SessionQuitError, NovelRagError
from novelrag.intent import DictionaryIntentFactory
from novelrag.llm.factory import ChatLLMFactory, EmbeddingLLMFactory
from novelrag.resource import LanceDBResourceRepository
from novelrag.session import Session, Command, ConfigBasedIntentScopeFactory


logger = logging.getLogger(__name__)


class NovelShell:
    """Interactive shell for NovelRAG"""

    def __init__(self, session: Session):
        self.session = session
        self.running = False

    @classmethod
    async def create(cls, config: NovelRagConfig) -> 'NovelShell':
        """Create a new shell instance with configured components"""
        embedder = EmbeddingLLMFactory.build(config.embedding)
        chat_llm_factory = ChatLLMFactory(ChatLLMFactory.build(config.chat_llm) if config.chat_llm else None)
        embedding_factory = EmbeddingLLMFactory(embedder)

        # Initialize repository
        repository = await LanceDBResourceRepository.from_config(
            config.resource_config,
            config.vector_store,
            embedder,
            config.default_resource_dir,
        )

        # Create aspect factory with configured intents
        aspect_factory = ConfigBasedIntentScopeFactory(
            resource_repository=repository,
            scope_configurations=config.scopes,
        )

        # Create session with configured intents
        session = Session(
            aspect_factory=aspect_factory,
            resource_repository=repository,
            default_lang=config.template_lang,
            intents=DictionaryIntentFactory.from_config(config.intents),
            chat_llm_factory=chat_llm_factory,
            embedding_factory=embedding_factory,
        )

        return cls(session)

    @staticmethod
    def parse_command(line: str) -> Command | None:
        """Parse user input into a Command object"""
        if not line.strip():
            return None

        parts = line.split(' ')
        aspect = None
        intent = None
        message = line

        # Parse aspect (@aspect)
        if parts[0].startswith('@'):
            aspect = parts[0][1:]
            message = ' '.join(parts[1:])
            parts = parts[1:]

        # Parse intent (/intent)
        if parts and parts[0].startswith('/'):
            intent = parts[0][1:]
            message = ' '.join(parts[1:])

        return Command(
            raw=line,
            aspect=aspect,
            intent=intent,
            message=message if message else None
        )

    async def handle_command(self, line: str):
        """Process a single command line"""
        command = self.parse_command(line)
        if command:
            current_command = command
            while current_command:
                response = await self.session.invoke(current_command)
                for message in response.messages:
                    print(message)
                if response.redirect:
                    logger.info(f"Redirecting to command: {response.redirect.text}")
                current_command = response.redirect

    async def run(self):
        """Start the interactive shell"""
        self.running = True
        print("NovelRAG Shell (Ctrl+C to exit)")
        
        while self.running:
            try:
                if self.session.context.current_scope:
                    prompt = f"{self.session.context.current_scope.name}> "
                else:
                    prompt = "> "
                
                line = input(prompt)
                await self.handle_command(line)

            except SessionQuitError:
                print("\nQuitting...")
                self.running = False
            except NovelRagError as e:
                print(f"Error: {e.msg}")
                logger.debug(e, exc_info=e)
            except KeyboardInterrupt:
                print("\nExiting...")
                self.running = False
            except EOFError:
                print("\nExiting...")
                self.running = False
            except UnicodeDecodeError:
                print(f"\nUnicode Error\nExiting...")
                self.running = False
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=e)
