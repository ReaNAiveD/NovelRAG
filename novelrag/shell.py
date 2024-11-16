from novelrag.action import Action, ActionResult, QuitResult, MessageResult, OperationResult
from novelrag.aspect import AspectContext
from novelrag.exceptions import (
    AspectNotFoundError,
    ActionNotFoundError,
    NoAspectSelectedError
)


class NovelShell:
    def __init__(self, aspects: dict[str, AspectContext]):
        self.aspect: AspectContext | None = None
        self.action: Action | None = None
        self.aspects = aspects

    def get_prompt(self) -> str:
        prompt = '>> '
        if self.action and self.action.name != 'default':
            prompt = f'/{self.action.name} {prompt}'
        if self.aspect:
            prompt = f'@{self.aspect.name} {prompt}'
        return prompt

    async def handle_aspect_switch(self, input_text: str) -> None:
        """Handle @aspect commands"""
        input_parts = input_text.split(maxsplit=1)
        aspect_name = input_parts[0][1:]  # Remove @ prefix
        remaining_input = input_parts[1] if len(input_parts) > 1 else ''
        
        if aspect_name not in self.aspects:
            raise AspectNotFoundError(aspect_name, list(self.aspects.keys()))
            
        self.aspect = self.aspects[aspect_name]
        self.action = None

        if remaining_input.startswith('/'):
            await self.switch_action(remaining_input)
        elif remaining_input:
            await self.switch_to_default(remaining_input)

    async def switch_action(self, input_text: str) -> None:
        """Switch to a new action"""
        if not self.aspect:
            raise NoAspectSelectedError()

        input_parts = input_text.split(maxsplit=1)
        action_name = input_parts[0][1:]  # Remove / prefix
        message = input_parts[1] if len(input_parts) > 1 else None

        try:
            self.action, message = await self.aspect.act(action_name, message)
            await self.process_action(message)
        except Exception as e:
            raise ActionNotFoundError(action_name, self.aspect.name) from e

    async def switch_to_default(self, message: str) -> None:
        """Switch to default action with message"""
        if not self.aspect:
            raise NoAspectSelectedError()

        self.action, message = await self.aspect.act(None, message)
        await self.process_action(message)

    async def handle_command(self, input_text: str) -> None:
        """Handle /command inputs"""
        if not self.aspect:
            raise NoAspectSelectedError()

        input_parts = input_text.split(maxsplit=1)
        command = input_parts[0][1:]  # Remove / prefix
        message = input_parts[1] if len(input_parts) > 1 else None

        if not self.action or self.action.name == 'default':
            await self.switch_action(input_text)
        else:
            try:
                result = await self.action.handle_command(command, message)
                if isinstance(result, QuitResult):
                    self.action = None
                    print(result.message or "Quited.")
            except Exception as e:
                print(f"Error handling command '{command}': {str(e)}")

    async def process_action(self, message: str | None) -> None:
        """Process action with message and handle result"""
        if not self.action or not self.aspect:
            return

        try:
            result = await self.aspect.handle_action(self.action, message)
            if isinstance(result, QuitResult):
                self.action = None
                print(result.message or "Quited.")
            elif isinstance(result, MessageResult):
                print(result.message)
            else:
                raise Exception(f"Unrecognized Action Result: {result}")
        except Exception as e:
            print(f"Error processing action: {str(e)}")
            self.action = None

    async def get_input(self) -> str | None:
        """Get user input, handling interrupts gracefully"""
        try:
            return input(self.get_prompt())
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return None
        except UnicodeDecodeError:
            # Handle interrupted input stream
            print("\nGoodbye!")
            return None

    async def run(self):
        while True:
            try:
                user_input = await self.get_input()
                if user_input is None:  # Exit condition
                    return

                if user_input.startswith('@'):
                    await self.handle_aspect_switch(user_input)
                elif user_input.startswith('/'):
                    await self.handle_command(user_input)
                elif self.aspect:
                    if not self.action:
                        await self.switch_to_default(user_input)
                    else:
                        await self.process_action(user_input)
                
            except (AspectNotFoundError, ActionNotFoundError, NoAspectSelectedError) as e:
                print(str(e))
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                print("Continuing...")
