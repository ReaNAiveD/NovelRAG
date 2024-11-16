import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch, Mock

from novelrag.core.shell import NovelShell
from novelrag.editors.premise.aspect import PremiseAspectContext
from novelrag.llm import AzureAIClient


class TestShellPremiseE2E(IsolatedAsyncioTestCase):
    def setUp(self):
        # Mock LLM response
        self.mock_oai_response = Mock()
        self.mock_oai_response.choices = [
            Mock(message=Mock(
                role='assistant',
                content='Mock LLM Response'
            ))
        ]
        
        # Mock file content
        self.mock_yaml_content = {
            'premises': [
                'Premise 1: A story about a hero',
                'Premise 2: A story about a villain'
            ]
        }

        # Create shell with mocked premise aspect
        self.config = {
            'file_path': 'mock_premises.yaml',
            'oai_config': {
                'api_key': 'test',
                'api_version': '2024-02-15-preview',
                'azure_endpoint': 'https://mock-endpoint.openai.azure.com',
                'azure_deployment': 'mock-deployment'
            },
            'chat_params': {}
        }
        
        # Patch PremiseAspectContext.load_file before creating the instance
        with patch('novelrag.editors.premise.aspect.PremiseAspectContext.load_file', 
                  return_value=self.mock_yaml_content):
            # Create shell with premise aspect
            premise_aspect = PremiseAspectContext(**self.config)
            # Now patch instance method for subsequent calls
            premise_aspect.load_file = Mock(return_value=self.mock_yaml_content)
            premise_aspect.dump_file = Mock()
            
            self.shell = NovelShell({
                'premise': premise_aspect
            })

    @patch.object(AzureAIClient, 'chat')
    async def test_premise_update_flow(self, mock_chat):
        # Setup mocks
        mock_chat.return_value = self.mock_oai_response

        # Switch to premise aspect
        await self.shell.handle_aspect_switch('@premise')
        self.assertEqual(self.shell.aspect.name, 'premise')
        
        # Start update action
        await self.shell.handle_command('/update 0 Make it more exciting')
        self.assertEqual(self.shell.action.name, 'update')
        
        # Verify LLM was called with correct prompt
        mock_chat.assert_called()
        
        # Submit the update
        await self.shell.handle_command('/submit')
        
        # Verify the file was updated
        self.shell.aspect.dump_file.assert_called()
        
        # Verify action was cleared
        self.assertIsNone(self.shell.action)

    @patch.object(AzureAIClient, 'chat')
    async def test_premise_create_flow(self, mock_chat):
        # Setup mocks
        mock_chat.return_value = self.mock_oai_response

        # Switch to premise aspect and create action
        await self.shell.handle_aspect_switch('@premise /create 0')
        self.assertEqual(self.shell.action.name, 'create')
        
        # Send message to LLM
        await self.shell.process_action('Create a new premise about space')
        
        # Verify LLM was called
        mock_chat.assert_called()
        
        # Submit the new premise
        await self.shell.handle_command('/submit New space premise')
        
        # Verify file was updated
        self.shell.aspect.dump_file.assert_called()
        
        # Verify action was cleared
        self.assertIsNone(self.shell.action)

    async def test_premise_list_flow(self):
        # Switch to premise aspect and list action
        await self.shell.handle_aspect_switch('@premise /list')
        
        # Verify action was cleared (list is immediate)
        self.assertIsNone(self.shell.action)
        
        # No file updates should have occurred
        self.shell.aspect.dump_file.assert_not_called()

    async def test_premise_delete_flow(self):
        # Switch to premise aspect and delete action
        await self.shell.handle_aspect_switch('@premise /delete 0')
        
        # Verify file was updated
        self.shell.aspect.dump_file.assert_called()
        
        # Verify action was cleared
        self.assertIsNone(self.shell.action)


if __name__ == '__main__':
    unittest.main()
