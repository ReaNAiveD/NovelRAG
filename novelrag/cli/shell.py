import logging

from novelrag.exceptions import SessionQuitError, NovelRagError
from novelrag.cli import Session, Command
from novelrag.tracer import trace_session


logger = logging.getLogger(__name__)


class NovelShell:
    """Interactive shell for NovelRAG"""

    def __init__(self, session: Session):
        self.session = session
        self.running = False

    @staticmethod
    def parse_command(line: str) -> Command | None:
        """Parse user input into a Command object"""
        if not line.strip():
            return None

        parts = line.split(' ')
        handler = None
        message = line

        # Parse handler (/handler)
        if parts and parts[0].startswith('/'):
            handler = parts[0][1:]
            message = ' '.join(parts[1:])

        return Command(
            raw=line,
            handler=handler,
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

    @trace_session("shell_session")
    async def run(self):
        """Start the interactive shell"""
        self.running = True
        print("NovelRAG Shell (Ctrl+C to exit)")
        
        while self.running:
            try:
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
