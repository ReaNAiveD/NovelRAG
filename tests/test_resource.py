import os
import shutil
import unittest
from typing import Any

from novelrag.config.resource import ResourceConfig, VectorStoreConfig
from novelrag.resource import ResourceRepository
from novelrag.resource.operation import ElementOperation, PropertyOperation, AspectLocation


class TestData:
    """Test data fixtures"""
    @staticmethod
    def create_character(name: str, age: int) -> dict[str, Any]:
        return {
            'name': name,
            'age': age,
            'description': f'A character named {name} who is {age} years old'
        }

    @staticmethod
    def create_event(name: str, characters: list[str]) -> dict[str, Any]:
        return {
            'name': name,
            'mainCharacters': characters,
            'description': f'An event involving {", ".join(characters)}'
        }


async def create_test_repository():
    """Helper to create a test repository with standard config"""
    resource_config = {
        'character': ResourceConfig(
            path='resource/characters.yml',
            children_keys=['relationships']
        ),
        'event': ResourceConfig(
            path='resource/events.yml',
            children_keys=['subEvents']
        )
    }
    vector_store_config = VectorStoreConfig(
        lancedb_uri='resource/lancedb',
        table_name='test_vectors',
        overwrite=True,  # Clean state for tests
    )
    embedding_config = {
        'endpoint': 'https://novel-rag.openai.azure.com',
        'deployment': 'text-embedding-3-large',
        'api_version': '2024-08-01-preview',
        'api_key': os.environ['OPENAI_API_KEY'],
        'model': 'text-embedding-3-large',
    }
    return await ResourceRepository.from_config(
        resource_config, vector_store_config, embedding_config
    )


class RepositoryTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repository = await create_test_repository()
        self.test_data = TestData()

    async def test_add_elements(self):
        """Test adding elements to an aspect"""
        # Add characters
        await self.repository.apply(ElementOperation.new(
            location=AspectLocation.new(aspect='character'),
            data=[
                self.test_data.create_character("Alice", 25),
                self.test_data.create_character("Bob", 30)
            ]
        ))

        # Verify characters were added
        self.assertEqual(len(self.repository.resource_aspects['character'].root_elements), 2)
        self.assertEqual(
            self.repository.resource_aspects['character'].root_elements[0].inner.props()['name'],
            "Alice"
        )

    async def test_modify_element(self):
        """Test modifying an existing element"""
        # First add an element
        await self.repository.apply(ElementOperation.new(
            location=AspectLocation.new(aspect='character'),
            data=[self.test_data.create_character("Alice", 25)]
        ))

        # Get the element's ID
        element_id = str(self.repository.resource_aspects['character'].root_elements[0].inner.id)

        # Modify the element
        await self.repository.apply(PropertyOperation(
            target='property',
            element_id=element_id,
            data={'age': 26, 'description': 'Updated description'}
        ))

        # Verify changes
        modified_element = self.repository.lut.find_by_id(element_id)
        self.assertEqual(modified_element.inner.props()['age'], 26)
        self.assertEqual(modified_element.inner.props()['description'], 'Updated description')

    async def test_vector_search(self):
        """Test vector search functionality"""
        # Add test data
        await self.repository.apply(ElementOperation.new(
            location=AspectLocation.new(aspect='event'),
            data=[
                self.test_data.create_event("Birthday Party", ["Alice", "Bob"]),
                self.test_data.create_event("Wedding", ["Charlie", "Diana"]),
                self.test_data.create_event("Graduation", ["Alice", "Eve"])
            ]
        ))

        # Search for events involving Alice
        results = await self.repository.vector_search("Find events with Alice")

        # Verify search results
        self.assertGreater(len(results), 0)
        found_alice = False
        for result in results[:2]:  # Check top 2 results
            if "Alice" in result.element.inner.props()['mainCharacters']:
                found_alice = True
                break
        self.assertTrue(found_alice, "Vector search failed to find relevant results")

    async def test_nested_elements(self):
        """Test handling of nested elements"""
        # Create an event with sub-events
        main_event = self.test_data.create_event("Main Event", ["Alice"])
        sub_event1 = self.test_data.create_event("Sub Event 1", ["Bob"])
        sub_event2 = self.test_data.create_event("Sub Event 2", ["Charlie"])
        main_event['subEvents'] = [sub_event1, sub_event2]

        await self.repository.apply(ElementOperation.new(
            location=AspectLocation.new(aspect='event'),
            data=[main_event]
        ))

        # Verify the structure
        root_element = self.repository.resource_aspects['event'].root_elements[0]
        self.assertEqual(len(root_element.children['subEvents']), 2)
        self.assertEqual(
            root_element.children['subEvents'][0].inner.props()['name'],
            "Sub Event 1"
        )

    async def test_lookup_table(self):
        """Test lookup table functionality"""
        # Add elements
        await self.repository.apply(ElementOperation.new(
            location=AspectLocation.new(aspect='character'),
            data=[
                self.test_data.create_character("Alice", 25),
                self.test_data.create_character("Bob", 30)
            ]
        ))

        # Get IDs
        alice_id = str(self.repository.resource_aspects['character'].root_elements[0].inner.id)
        bob_id = str(self.repository.resource_aspects['character'].root_elements[1].inner.id)

        # Test lookup
        alice = self.repository.lut.find_by_id(alice_id)
        bob = self.repository.lut.find_by_id(bob_id)

        self.assertEqual(alice.inner.props()['name'], "Alice")
        self.assertEqual(bob.inner.props()['name'], "Bob")

    async def asyncTearDown(self):
        """Clean up test resources"""
        del self.repository
        if os.path.exists('resource'):
            shutil.rmtree('resource')
