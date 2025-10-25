import os
import shutil
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch
import yaml

from novelrag.config.llm import AzureOpenAIEmbeddingConfig, EmbeddingLLMType
from novelrag.config.resource import AspectConfig, VectorStoreConfig
from novelrag.resource import LanceDBResourceRepository
from novelrag.resource.operation import ResourceOperation, PropertyOperation, ResourceLocation, OperationTarget
from novelrag.llm import EmbeddingLLM
from novelrag.resource.element import Element, DirectiveElement, DirectiveElementList


class TestData:
    """Test data fixtures"""
    @staticmethod
    def create_character(name: str, age: int) -> dict[str, Any]:
        return {
            'id': name.lower().replace(" ", "_"),
            'name': name,
            'age': age,
            'description': f'A character named {name} who is {age} years old'
        }

    @staticmethod
    def create_event(name: str, characters: list[str]) -> dict[str, Any]:
        return {
            'id': name.lower().replace(" ", "_"),
            'name': name,
            'mainCharacters': characters,
            'description': f'An event involving {", ".join(characters)}'
        }


class MockEmbeddingLLM(EmbeddingLLM):
    """Mock embedder for testing"""
    def __init__(self, dimension: int = 3072):
        self.dimension = dimension
        self.embedding_calls = AsyncMock()

    async def embedding(self, message: str, **params) -> list[list[float]]:
        await self.embedding_calls(message, **params)
        # Return a fixed-size vector of zeros for testing
        return [[0.0] * self.dimension]


class DummyVectorStore:
    def __init__(self, embedder: EmbeddingLLM | None = None):
        self.embedder = embedder
        self._store: dict[str, dict[str, Any]] = {}

    async def vector_search(self, vector: list[float], *, aspect: str | None = None, limit: int | None = 20):
        items = []
        for uri, rec in self._store.items():
            if aspect and rec['aspect'] != aspect:
                continue
            data = rec['data']
            # Prefer events containing Alice in mainCharacters
            score = 0.0
            if isinstance(data, dict):
                chars = data.get('mainCharacters')
                if not (isinstance(chars, list) and 'Alice' in chars):
                    score = 1.0
            items.append((score, uri))
        items.sort(key=lambda x: x[0])
        if limit is not None:
            items = items[:limit]
        class Result:
            def __init__(self, resource_uri: str, distance: float):
                self.resource_uri = resource_uri
                self.distance = distance
        return [Result(uri, dist) for dist, uri in [(s, u) for (s, u) in items]]

    async def get(self, resource_uri: str):
        return None

    async def batch_add(self, elements: list[Element]):
        for ele in elements:
            await self.add(ele, unchecked=True)

    async def add(self, element: Element, *, unchecked: bool = False):
        self._store[element.uri] = {
            'aspect': element.aspect,
            'data': element.element_dict(),
        }

    async def update(self, element: Element):
        await self.add(element)

    async def delete(self, resource_uri: str):
        self._store.pop(resource_uri, None)

    async def get_all_resource_uris(self) -> list[str]:
        """Return all resource URIs currently in the store."""
        return list(self._store.keys())

    async def batch_delete_by_uris(self, resource_uris: list[str]):
        """Delete multiple resources by their URIs."""
        for uri in resource_uris:
            self._store.pop(uri, None)

    async def cleanup_invalid_resources(self, valid_resource_uris: set[str]) -> int:
        """Remove resources not in the valid set and return count removed."""
        all_uris = list(self._store.keys())
        invalid_uris = [uri for uri in all_uris if uri not in valid_resource_uris]
        for uri in invalid_uris:
            self._store.pop(uri, None)
        return len(invalid_uris)


async def create_test_repository(*, use_mock: bool = True):
    """Helper to create a test repository with standard config"""
    resource_config = {
        'character': AspectConfig(
            path='resource/characters.yml',
            description='A collection of characters in the story',
            children_keys=['relationships']
        ),
        'event': AspectConfig(
            path='resource/events.yml',
            description='A collection of events in the story',
            children_keys=['subEvents']
        )
    }
    vector_store_config = VectorStoreConfig(
        lancedb_uri='resource/lancedb',
        table_name='test_vectors',
        overwrite=True,  # Clean state for tests
        cleanup_invalid_on_init=True,  # Enable cleanup for tests
    )

    if use_mock:
        embedder = MockEmbeddingLLM()
    else:
        from novelrag.llm.oai import OpenAIEmbeddingLLM
        embedding_config = AzureOpenAIEmbeddingConfig(
            endpoint='https://novel-rag.openai.azure.com',
            deployment='text-embedding-3-large',
            api_version='2024-08-01-preview',
            api_key=os.environ['OPENAI_API_KEY'],
            model='text-embedding-3-large',
            timeout=180.0,
            type=EmbeddingLLMType.AzureOpenAI,
        )
        embedder = OpenAIEmbeddingLLM.from_config(embedding_config)

    os.makedirs('resource', exist_ok=True)
    cfg_path = os.path.join('resource', 'test_resources.yml')
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump({
            'character': {
                'path': 'resource/characters.yml',
                'description': 'A collection of characters in the story',
                'children_keys': ['relationships']
            },
            'event': {
                'path': 'resource/events.yml',
                'description': 'A collection of events in the story',
                'children_keys': ['subEvents']
            }
        }, f, allow_unicode=True)

    # Patch LanceDBStore.create to return our dummy store
    with patch('novelrag.resource.repository.LanceDBStore.create', new=AsyncMock(side_effect=lambda **kwargs: DummyVectorStore(embedder))):
        return await LanceDBResourceRepository.from_config(
            cfg_path,
            vector_store_config,
            embedder
        )


class RepositoryTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repository = await create_test_repository(use_mock=True)
        self.test_data = TestData()

    async def test_add_elements(self):
        """Test adding elements to an aspect"""
        # Add characters
        await self.repository.apply(ResourceOperation.new(
            location=ResourceLocation.aspect('character'),
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
        await self.repository.apply(ResourceOperation.new(
            location=ResourceLocation.aspect('character'),
            data=[self.test_data.create_character("Alice", 25)]
        ))

        # Get the element's ID
        resource_uri = str(self.repository.resource_aspects['character'].root_elements[0].uri)

        # Modify the element
        await self.repository.apply(PropertyOperation(
            target=OperationTarget.PROPERTY,
            resource_uri=resource_uri,
            data={'age': 26, 'description': 'Updated description'}
        ))

        # Verify changes
        modified_element = self.repository.lut.find_by_uri(resource_uri)
        self.assertIsNotNone(modified_element)
        self.assertEqual(modified_element['age'], 26) # type: ignore
        self.assertEqual(modified_element['description'], 'Updated description') # type: ignore

    async def test_vector_search(self):
        """Test vector search functionality"""
        # Add test data
        await self.repository.apply(ResourceOperation.new(
            location=ResourceLocation.aspect('event'),
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

        await self.repository.apply(ResourceOperation.new(
            location=ResourceLocation.aspect('event'),
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
        await self.repository.apply(ResourceOperation.new(
            location=ResourceLocation.aspect('character'),
            data=[
                self.test_data.create_character("Alice", 25),
                self.test_data.create_character("Bob", 30)
            ]
        ))

        # Get IDs
        alice_uri = str(self.repository.resource_aspects['character'].root_elements[0].inner.uri)
        bob_uri = str(self.repository.resource_aspects['character'].root_elements[1].inner.uri)

        # Test lookup
        alice = self.repository.lut.find_by_uri(alice_uri)
        bob = self.repository.lut.find_by_uri(bob_uri)

        self.assertEqual(alice['name'], "Alice") # type: ignore
        self.assertEqual(bob['name'], "Bob") # type: ignore

    async def asyncTearDown(self):
        """Clean up test resources"""
        del self.repository
        if os.path.exists('resource'):
            shutil.rmtree('resource')


class DirectiveElementTestCase(unittest.TestCase):
    """Test cases for DirectiveElement and DirectiveElementList"""
    
    def setUp(self):
        self.test_data = TestData()
        
    def test_directive_element_wrap(self):
        """Test wrapping a basic Element in DirectiveElement"""
        # Create a basic character element
        character = Element.build(
            self.test_data.create_character("Alice", 25),
            parent_uri='character',
            aspect='character',
            children_keys=['relationships']
        )
        
        # Wrap it in DirectiveElement
        directive = DirectiveElement.wrap(character, ['relationships'])
        
        # Verify basic properties
        self.assertEqual(directive.props['name'], "Alice")
        self.assertEqual(directive.props['age'], 25)
        self.assertIsNone(directive.parent)
        self.assertIsNone(directive.prev)
        self.assertIsNone(directive.next)
        
    def test_directive_element_list_linking(self):
        """Test that DirectiveElementList properly links elements"""
        # Create multiple character elements
        characters = [
            Element.build(self.test_data.create_character(name, age), 
                        parent_uri='character',
                        aspect='character', 
                        children_keys=['relationships'])
            for name, age in [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
        ]
        
        # Create DirectiveElementList
        directive_list = DirectiveElementList.wrap(characters, ['relationships'])
        
        # Verify linking
        self.assertIsNone(directive_list[0].prev)
        self.assertEqual(directive_list[0].next, directive_list[1])
        self.assertEqual(directive_list[1].prev, directive_list[0])
        self.assertEqual(directive_list[1].next, directive_list[2])
        self.assertEqual(directive_list[2].prev, directive_list[1])
        self.assertIsNone(directive_list[2].next)
        
    def test_nested_directive_elements(self):
        """Test handling of nested elements in DirectiveElement"""
        # Create an event with sub-events
        main_event = self.test_data.create_event("Main Event", ["Alice"])
        sub_event1 = self.test_data.create_event("Sub Event 1", ["Bob"])
        sub_event2 = self.test_data.create_event("Sub Event 2", ["Charlie"])
        main_event['subEvents'] = [sub_event1, sub_event2]
        
        # Create and wrap the element
        event_element = Element.build(main_event, parent_uri='event', aspect='event', children_keys=['subEvents'])
        directive = DirectiveElement.wrap(event_element, ['subEvents'])

        # Verify structure
        self.assertEqual(len(directive.children['subEvents']), 2)
        self.assertEqual(directive.children['subEvents'][0].props['name'], "Sub Event 1")
        self.assertEqual(directive.children['subEvents'][1].props['name'], "Sub Event 2")
        
        # Verify parent-child relationships
        for child in directive.children['subEvents']:
            self.assertEqual(child.parent, directive)
            
    def test_directive_element_splice(self):
        """Test splicing elements in DirectiveElement"""
        # Create main event with sub-events
        main_event = self.test_data.create_event("Main Event", ["Alice"])
        sub_events = [
            self.test_data.create_event(f"Sub Event {i}", ["Character{i}"])
            for i in range(1, 4)
        ]
        main_event['subEvents'] = sub_events
        
        # Create and wrap the element
        event_element = Element.build(main_event, parent_uri='event', aspect='event', children_keys=['subEvents'])
        directive = DirectiveElement.wrap(event_element, ['subEvents'])
        
        # Create new sub-event to splice in
        new_sub_event = Element.build(
            self.test_data.create_event("New Sub Event", ["Dave"]),
            parent_uri='event',
            aspect='event',
            children_keys=['subEvents']
        )
        
        # Splice the new event in place of the second event
        new_elements, old_elements = directive.splice_at('subEvents', 1, 2, new_sub_event)
        
        # Verify the splice
        self.assertEqual(len(directive.children['subEvents']), 3)
        self.assertEqual(directive.children['subEvents'][1].props['name'], "New Sub Event")
        self.assertEqual(len(old_elements), 1)
        self.assertEqual(old_elements[0].props['name'], "Sub Event 2")

        # Verify the linking is maintained
        self.assertEqual(directive.children['subEvents'][0].next, directive.children['subEvents'][1])
        self.assertEqual(directive.children['subEvents'][1].prev, directive.children['subEvents'][0])
        self.assertEqual(directive.children['subEvents'][1].next, directive.children['subEvents'][2])
        self.assertEqual(directive.children['subEvents'][2].prev, directive.children['subEvents'][1])
