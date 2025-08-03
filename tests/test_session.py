import unittest
from unittest.mock import AsyncMock, Mock

from novelrag.conversation import ConversationHistory
from novelrag.intent.factory import DictionaryIntentFactory, IntentFactory
from novelrag.intent.action import Action, UpdateAction
from novelrag.exceptions import IntentNotFoundError, NoItemToSubmitError, NoItemToUndoError
from novelrag.intent import Intent, IntentContext
from novelrag.pending_queue import PendingUpdateItem
from novelrag.resource import ResourceRepository, ResourceAspect, Operation
from novelrag.session import Session, Command, IntentScopeFactory, IntentScope


# Helper functions for creating common mock objects
def create_mock_repository():
    """Create a mock repository with async methods"""
    repo = Mock(spec=ResourceRepository)
    repo.vector_search = AsyncMock()
    repo.apply = AsyncMock()
    return repo


def create_mock_intent(name: str):
    """Create a mock intent with async handle method"""
    intent = Mock(spec=Intent)
    intent.name = name
    intent.handle = AsyncMock()
    # By default, return an Action with no special actions
    intent.handle.return_value = Action()
    return intent


def create_mock_aspect():
    """Create a mock aspect with data and intent registry"""
    mock_data = Mock(spec=ResourceAspect)
    mock_intents = Mock(spec=IntentFactory)
    mock_intents.get_intent = AsyncMock()
    aspect = Mock(spec=IntentScope)
    aspect.name = 'mock'
    aspect.data = mock_data
    aspect.intents = mock_intents
    return aspect


class SessionTestCase(unittest.IsolatedAsyncioTestCase):
    """Test cases for Session class functionality"""

    async def asyncSetUp(self):
        """Setup test environment before each test"""
        # Create mock components
        self.aspect_factory = Mock(spec=IntentScopeFactory)
        self.aspect_factory.get = AsyncMock()
        self.repository = Mock(spec=ResourceRepository)
        
        # Setup session-level intent
        self.session_intent = create_mock_intent("session_intent")
        self.intent_factory = DictionaryIntentFactory([
            self.session_intent
        ])

        self.history = Mock(spec=ConversationHistory)

        # Create session instance
        self.session = Session(
            aspect_factory=self.aspect_factory,
            resource_repository=self.repository,
            intents=self.intent_factory,
            conversation=self.history,
        )

    async def test_invoke_with_aspect_switch(self):
        """Test switching to a new aspect without any intent but with a default intent available"""
        # Create a default intent that returns an empty Action
        default_intent = create_mock_intent("_default")
        default_intent.handle.return_value = Action()
        
        # Add the default intent to the session registry
        self.intent_factory = DictionaryIntentFactory([
            self.session_intent,
            default_intent
        ])
        
        # Recreate session with the updated intent factory
        self.session = Session(
            aspect_factory=self.aspect_factory,
            resource_repository=self.repository,
            intents=self.intent_factory,
            conversation=self.history,
        )
        
        command = Command(
            aspect="test_aspect",
            raw="@test_aspect"
        )
        mock_aspect = create_mock_aspect()
        self.aspect_factory.get.return_value = mock_aspect
        # Mock the intent factory to return None for no intent specified
        mock_aspect.intents.get_intent.return_value = None

        response = await self.session.invoke(command)

        self.aspect_factory.get.assert_called_once_with('test_aspect')
        self.assertEqual(self.session.context.current_scope, mock_aspect)
        self.assertEqual([], response.messages)

    async def test_invoke_unregistered_intent(self):
        """Test error handling when invoking an unknown intent"""
        command = Command(
            intent="unknown_intent",
            message="test message",
            raw="/unknown_intent test message"
        )
        
        with self.assertRaises(IntentNotFoundError):
            await self.session.invoke(command)

    async def test_invoke_with_updates(self):
        """Test handling of updates from intent execution
        
        Verifies:
        1. Aspect switching
        2. Intent handling
        3. Update queueing
        4. Response message generation
        """
        mock_aspect = create_mock_aspect()
        self.aspect_factory.get.return_value = mock_aspect

        mock_intent = create_mock_intent("test_intent")
        # Setup mock intent that produces updates
        mock_update = Mock(spec=PendingUpdateItem)
        mock_update.ops = [Mock(spec=Operation)]
        mock_output = Action(update=UpdateAction(item=mock_update))
        mock_intent.handle.return_value = mock_output
        mock_aspect.intents.get_intent.return_value = mock_intent

        command = Command(
            aspect="test_aspect",
            intent="test_intent",
            message="test message",
            raw="@test_aspect /test_intent test message"
        )
        response = await self.session.invoke(command)

        self.aspect_factory.get.assert_called_once_with('test_aspect')
        self.assertEqual(self.session.context.current_scope, mock_aspect)
        mock_intent.handle.assert_called_once_with(
            "test message",
            context=IntentContext(
                current_element=None,
                current_aspect=mock_aspect.data,
                raw_command="@test_aspect /test_intent test message",
                pending_update=self.session.update_queue,
                resource_repository=self.repository,
                chat_llm_factory=self.session.chat_llm_factory,
                embedding_factory=self.session.embedding_factory,
                conversation_history=self.history,
                template_env=self.session.template_env,
            )
        )
        self.assertEqual(self.session.update_queue.pop(), mock_update)
        self.assertEqual(self.session.update_queue.pop(), None)
        self.session_intent.handle.assert_not_called()
        self.assertEqual([], response.messages)

    async def test_invoke_with_submit(self):
        """Test submitting queued updates to repository
        
        Verifies:
        1. Update application
        2. Undo operation creation
        3. Queue state management
        4. Response messages
        """
        mock_aspect = create_mock_aspect()
        self.aspect_factory.get.return_value = mock_aspect

        mock_intent = create_mock_intent("test_intent")
        mock_output = Action(submit=True)
        mock_intent.handle.return_value = mock_output
        mock_aspect.intents.get_intent.return_value = mock_intent

        # Add a mock update to the queue
        mock_update = Mock(spec=PendingUpdateItem)
        mock_op = Mock(spec=Operation)
        mock_undo_op = Mock(spec=Operation)
        self.repository.apply.return_value = mock_undo_op
        mock_update.ops = [mock_op]
        self.session.update_queue.push(mock_update)

        command = Command(
            aspect="test_aspect",
            intent="test_intent",
            message="test message",
            raw="@test_aspect /test_intent test message"
        )
        response = await self.session.invoke(command)

        self.aspect_factory.get.assert_called_once_with('test_aspect')
        self.assertEqual(self.session.context.current_scope, mock_aspect)
        mock_intent.handle.assert_called_once_with(
            "test message",
            context=IntentContext(
                current_element=None,
                current_aspect=mock_aspect.data,
                raw_command="@test_aspect /test_intent test message",
                pending_update=self.session.update_queue,
                resource_repository=self.repository,
                chat_llm_factory=self.session.chat_llm_factory,
                embedding_factory=self.session.embedding_factory,
                conversation_history=self.history,
                template_env=self.session.template_env,
            )
        )
        self.repository.apply.assert_called_once_with(mock_op)
        self.session_intent.handle.assert_not_called()
        self.assertEqual([], response.messages)

        # Verify undo queue state
        undo_item = self.session.undo.pop()
        self.assertIsNotNone(undo_item)
        self.assertEqual(undo_item.ops, [mock_undo_op]) # type: ignore
        self.assertEqual(undo_item.redo, mock_update) # type: ignore
        self.assertIsNone(self.session.undo.pop())  # Queue should be empty now

        # Verify update queue is empty
        self.assertIsNone(self.session.update_queue.pop())

    async def test_invoke_with_undo(self):
        """Test undoing previous operations
        
        Verifies:
        1. Undo operation application
        2. Redo operation queueing
        3. Queue state management
        4. Response messages
        """
        mock_aspect = create_mock_aspect()
        self.aspect_factory.get.return_value = mock_aspect

        mock_intent = create_mock_intent("test_intent")
        mock_output = Action(undo=True)
        mock_intent.handle.return_value = mock_output
        mock_aspect.intents.get_intent.return_value = mock_intent

        # Add something to undo queue first
        mock_undo_op = Mock(spec=Operation)
        mock_update = Mock(spec=PendingUpdateItem)
        mock_update.ops = [Mock(spec=Operation)]
        self.session.undo.push([mock_undo_op], mock_update)

        command = Command(
            aspect="test_aspect",
            intent="test_intent",
            message="test message",
            raw="@test_aspect /test_intent test message"
        )
        response = await self.session.invoke(command)

        self.aspect_factory.get.assert_called_once_with('test_aspect')
        self.assertEqual(self.session.context.current_scope, mock_aspect)
        mock_intent.handle.assert_called_once_with(
            "test message",
            context=IntentContext(
                current_element=None,
                current_aspect=mock_aspect.data,
                raw_command="@test_aspect /test_intent test message",
                pending_update=self.session.update_queue,
                resource_repository=self.repository,
                chat_llm_factory=self.session.chat_llm_factory,
                embedding_factory=self.session.embedding_factory,
                conversation_history=self.history,
                template_env=self.session.template_env,
            )
        )
        self.repository.apply.assert_called_once_with(mock_undo_op)
        self.session_intent.handle.assert_not_called()
        self.assertEqual([], response.messages)

        # Verify undo queue is empty
        self.assertIsNone(self.session.undo.pop())

        # Verify update queue contains the redo operation
        queued_update = self.session.update_queue.pop()
        self.assertEqual(queued_update, mock_update)
        self.assertIsNone(self.session.update_queue.pop())

    async def test_invoke_with_empty_undo(self):
        """Test handling of undo request with empty undo queue
        
        Verifies:
        1. No operations applied
        2. Proper error message
        3. Queue states remain empty
        """
        mock_aspect = create_mock_aspect()
        self.aspect_factory.get.return_value = mock_aspect

        mock_intent = create_mock_intent("test_intent")
        mock_output = Action(undo=True)
        mock_intent.handle.return_value = mock_output
        mock_aspect.intents.get_intent.return_value = mock_intent

        command = Command(
            aspect="test_aspect",
            intent="test_intent",
            message="test message",
            raw="@test_aspect /test_intent test message"
        )
        with self.assertRaises(NoItemToUndoError):
            await self.session.invoke(command)

        # Verify nothing was applied
        self.repository.apply.assert_not_called()

        # Verify queues remain empty
        self.assertIsNone(self.session.undo.pop())
        self.assertIsNone(self.session.update_queue.pop())

    async def test_invoke_with_empty_submit(self):
        """Test handling of submit request with empty update queue
        
        Verifies:
        1. No operations applied
        2. Proper error message
        3. Queue states remain empty
        """
        mock_aspect = create_mock_aspect()
        self.aspect_factory.get.return_value = mock_aspect

        mock_intent = create_mock_intent("test_intent")
        mock_output = Action(submit=True)
        mock_intent.handle.return_value = mock_output
        mock_aspect.intents.get_intent.return_value = mock_intent

        command = Command(
            aspect="test_aspect",
            intent="test_intent",
            message="test message",
            raw="@test_aspect /test_intent test message"
        )
        with self.assertRaises(NoItemToSubmitError):
            await self.session.invoke(command)

        # Verify nothing was applied
        self.repository.apply.assert_not_called()

        # Verify queues remain empty
        self.assertIsNone(self.session.undo.pop())
        self.assertIsNone(self.session.update_queue.pop())

    async def test_invoke_with_session_intent(self):
        """Test session-level intent without aspect context
        
        Verifies:
        1. No aspect switching
        2. Session intent handling
        3. Update queueing
        4. Response messages
        """
        # Setup session intent
        mock_update = Mock(spec=PendingUpdateItem)
        mock_update.ops = [Mock(spec=Operation)]
        mock_output = Action(update=UpdateAction(item=mock_update))
        self.session_intent.handle.return_value = mock_output

        command = Command(
            intent="session_intent",
            message="test message",
            raw="/session_intent test message"
        )
        response = await self.session.invoke(command)

        # Verify aspect was not switched
        self.aspect_factory.get.assert_not_called()
        self.assertIsNone(self.session.context.current_scope)

        # Verify session intent was called
        self.session_intent.handle.assert_called_once_with(
            "test message",
            context=IntentContext(
                current_element=None,
                current_aspect=None,
                raw_command="/session_intent test message",
                pending_update=self.session.update_queue,
                resource_repository=self.repository,
                chat_llm_factory=self.session.chat_llm_factory,
                embedding_factory=self.session.embedding_factory,
                conversation_history=self.history,
                template_env=self.session.template_env,
            )
        )

        # Verify update was queued
        self.assertEqual(self.session.update_queue.pop(), mock_update)
        self.assertEqual(self.session.update_queue.pop(), None)

        # Verify response messages
        self.assertEqual([], response.messages)

    async def test_invoke_with_session_intent_after_aspect(self):
        """Test session-level intent with existing aspect context
        
        Verifies:
        1. Aspect context preservation
        2. Session intent handling with aspect
        3. Update queueing
        4. Response messages
        """
        # Setup aspect context first
        mock_aspect = create_mock_aspect()
        self.aspect_factory.get.return_value = mock_aspect
        mock_aspect.intents.get_intent.return_value = None
        
        # Create a default intent for the aspect that returns an empty Action
        default_intent = create_mock_intent("_default")
        default_intent.handle.return_value = Action()
        
        # Add the default intent to the session registry
        self.intent_factory = DictionaryIntentFactory([
            self.session_intent,
            default_intent
        ])
        
        # Recreate session with the updated intent factory
        self.session = Session(
            aspect_factory=self.aspect_factory,
            resource_repository=self.repository,
            intents=self.intent_factory,
            conversation=self.history,
        )
        
        # First switch to aspect (this will use the default intent)
        await self.session.invoke(Command(aspect="test_aspect", raw="@test_aspect"))

        # Setup session intent
        mock_update = Mock(spec=PendingUpdateItem)
        mock_update.ops = [Mock(spec=Operation)]
        mock_output = Action(update=UpdateAction(item=mock_update))
        self.session_intent.handle.return_value = mock_output

        # Invoke session intent
        command = Command(
            intent="session_intent",
            message="test message",
            raw="/session_intent test message"
        )
        response = await self.session.invoke(command)

        # Verify aspect wasn't switched again
        self.aspect_factory.get.assert_called_once_with("test_aspect")  # Only from first call
        self.assertEqual(self.session.context.current_scope, mock_aspect)

        # Verify session intent was called with existing aspect
        self.session_intent.handle.assert_called_once_with(
            "test message",
            context=IntentContext(
                current_element=None,
                current_aspect=mock_aspect.data,
                raw_command="/session_intent test message",
                pending_update=self.session.update_queue,
                resource_repository=self.repository,
                chat_llm_factory=self.session.chat_llm_factory,
                embedding_factory=self.session.embedding_factory,
                conversation_history=self.history,
                template_env=self.session.template_env,
            )
        )

        # Verify update was queued
        self.assertEqual(self.session.update_queue.pop(), mock_update)
        self.assertEqual(self.session.update_queue.pop(), None)

        # Verify response messages
        self.assertEqual([], response.messages)

    async def asyncTearDown(self):
        """Cleanup after each test"""
        self.repository.reset_mock()
        self.aspect_factory.reset_mock()
