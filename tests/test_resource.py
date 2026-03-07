import os
import shutil
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch
import yaml

from novelrag.config.llm import AzureOpenAIEmbeddingConfig, EmbeddingLLMType
from novelrag.config.resource import AspectConfig, VectorStoreConfig
from novelrag.storage.local.resource import LanceDBResourceRepository
from novelrag.resource.operation import ResourceOperation, PropertyOperation, ResourceLocation, OperationTarget
from langchain_core.embeddings import Embeddings
from novelrag.resource.element import Element


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


class MockEmbeddingLLM(Embeddings):
    """Mock embedder for testing"""
    def __init__(self, dimension: int = 3072):
        self.dimension = dimension
        self.embedding_calls = AsyncMock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dimension

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            await self.embedding_calls(text)
        return [[0.0] * self.dimension for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        await self.embedding_calls(text)
        return [0.0] * self.dimension


class DummyVectorStore:
    def __init__(self, embedder: Embeddings | None = None):
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
            'data': element.element_dict,
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

    async def cleanup_invalid_resources(self, valid_uris: set[str]) -> int:
        """Remove resources not in the valid set and return count removed."""
        all_uris = list(self._store.keys())
        invalid_uris = [uri for uri in all_uris if uri not in valid_uris]
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
        from novelrag.llm.factory import EmbeddingLLMFactory
        embedding_config = AzureOpenAIEmbeddingConfig(
            endpoint='https://novel-rag.openai.azure.com',
            deployment='text-embedding-3-large',
            api_version='2024-08-01-preview',
            api_key=os.environ['OPENAI_API_KEY'],
            model='text-embedding-3-large',
            timeout=180.0,
            type=EmbeddingLLMType.AzureOpenAI,
        )
        embedder = EmbeddingLLMFactory.build(embedding_config)

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
    # Create empty aspect files so load_from_disk can read them
    for aspect_file in ['resource/characters.yml', 'resource/events.yml']:
        with open(aspect_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump([], f)

    # Patch LanceDBStore.create to return our dummy store
    with patch('novelrag.storage.local.resource.LanceDBStore.create', new=AsyncMock(side_effect=lambda **kwargs: DummyVectorStore(embedder))):
        return await LanceDBResourceRepository.load_from_disk(
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

        # root_elements is list[str] of element names
        self.assertEqual(len(self.repository.resource_aspects['character'].root_element_names), 2)
        # Verify via lut
        alice = self.repository.lut.find_by_uri('/character/alice')
        self.assertIsNotNone(alice)
        self.assertEqual(alice.props['name'], "Alice")

    async def test_modify_element(self):
        """Test modifying an existing element"""
        # First add an element
        await self.repository.apply(ResourceOperation.new(
            location=ResourceLocation.aspect('character'),
            data=[self.test_data.create_character("Alice", 25)]
        ))

        # root_elements is list[str]; construct URI from aspect name + element id
        root_name = self.repository.resource_aspects['character'].root_element_names[0]
        resource_uri = f'/character/{root_name}'

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
            if "Alice" in result.element.props['mainCharacters']:
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

        # root_elements is list[str]
        root_name = self.repository.resource_aspects['event'].root_element_names[0]
        root_element = self.repository.lut.find_by_uri(f'/event/{root_name}')
        self.assertIsNotNone(root_element)

        # children_names_of returns list[str]
        sub_event_names = root_element.children_names_of('subEvents')
        self.assertEqual(len(sub_event_names), 2)

        # Look up actual child elements via lut
        child1 = self.repository.lut.find_by_uri(f'/event/{root_name}/{sub_event_names[0]}')
        self.assertIsNotNone(child1)
        self.assertEqual(child1.props['name'], "Sub Event 1")

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

        # root_elements is list[str]; construct URIs
        root_names = self.repository.resource_aspects['character'].root_element_names
        alice_uri = f'/character/{root_names[0]}'
        bob_uri = f'/character/{root_names[1]}'

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


class ElementTreeTestCase(unittest.TestCase):
    """Test cases for Element flat structure and children_names"""
    
    def setUp(self):
        self.test_data = TestData()
        
    def test_element_properties(self):
        """Test basic Element property access"""
        # Create a basic character element
        character = Element.build(
            self.test_data.create_character("Alice", 25),
            parent_uri='character',
            aspect='character',
            children_keys=['relationships']
        )
        
        # Verify basic properties
        self.assertEqual(character.props['name'], "Alice")
        self.assertEqual(character.props['age'], 25)
        self.assertEqual(character.uri, 'character/alice')
        self.assertEqual(character.aspect, 'character')
        
    def test_nested_element_children_names(self):
        """Test that Element.build normalises nested children to name lists"""
        # Create an event with sub-events
        main_event = self.test_data.create_event("Main Event", ["Alice"])
        sub_event1 = self.test_data.create_event("Sub Event 1", ["Bob"])
        sub_event2 = self.test_data.create_event("Sub Event 2", ["Charlie"])
        main_event['subEvents'] = [sub_event1, sub_event2]

        # Element.build normalises children dicts to list[str]
        event_element = Element.build(main_event, parent_uri='event', aspect='event', children_keys=['subEvents'])

        # children_names_of returns list[str] (just the ids)
        sub_names = event_element.children_names_of('subEvents')
        self.assertEqual(len(sub_names), 2)
        self.assertEqual(sub_names[0], 'sub_event_1')
        self.assertEqual(sub_names[1], 'sub_event_2')

        # children_names returns dict[str, list[str]]
        all_children = event_element.children_names
        self.assertIn('subEvents', all_children)
        self.assertEqual(len(all_children['subEvents']), 2)
            
    def test_element_set_children_names(self):
        """Test setting and modifying children name lists"""
        # Create main event with sub-events
        main_event = self.test_data.create_event("Main Event", ["Alice"])
        sub_events = [
            self.test_data.create_event(f"Sub Event {i}", [f"Character{i}"])
            for i in range(1, 4)
        ]
        main_event['subEvents'] = sub_events
        
        # Create the element
        event_element = Element.build(main_event, parent_uri='event', aspect='event', children_keys=['subEvents'])
        
        # Verify initial children
        initial_names = event_element.children_names_of('subEvents')
        self.assertEqual(len(initial_names), 3)
        
        # Set new children names (simulating a splice: replace index 1 with new name)
        new_names = initial_names[:1] + ['new_sub_event'] + initial_names[2:]
        event_element.set_children_names('subEvents', new_names)
        
        # Verify the update
        updated_names = event_element.children_names_of('subEvents')
        self.assertEqual(len(updated_names), 3)
        self.assertEqual(updated_names[1], 'new_sub_event')
        
        # Test add_child_names
        event_element.add_child_names('subEvents', ['extra_event'])
        final_names = event_element.children_names_of('subEvents')
        self.assertEqual(len(final_names), 4)
        self.assertEqual(final_names[3], 'extra_event')
