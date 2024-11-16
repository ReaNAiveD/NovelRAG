from unittest import TestCase

from novelrag.operation import Operation, OperationType, apply_operation


class TestOperation(TestCase):
    def test_root_new_undo(self):
        op = Operation(type=OperationType.NEW, path='', data='root string')
        ndata, undo = apply_operation(None, op)
        self.assertIsInstance(ndata, str)
        self.assertEqual(ndata, 'root string')
        self.assertEqual(undo, Operation(type=OperationType.DELETE, path='', data=None))
        cleaned, redo = apply_operation(ndata, undo)
        self.assertIsNone(cleaned)
        self.assertEqual(redo, op)

    def test_update_dict_undo(self):
        origin = {'data': {'field1': 'data1', 'field2': 'data2'}}
        op = Operation(type=OperationType.UPDATE, path='data', data={'field2': 'ndata2', 'field3': 'ndata3'})
        ndata, undo = apply_operation(origin, op)
        self.assertDictEqual(ndata['data'], {'field1': 'data1', 'field2': 'ndata2', 'field3': 'ndata3'})
        self.assertEqual(undo.type, OperationType.PUT)
        self.assertDictEqual(undo.data, {'field1': 'data1', 'field2': 'data2'})
        cleaned, redo = apply_operation(ndata, undo)
        self.assertDictEqual(cleaned['data'], origin['data'])
        self.assertEqual(redo.type, OperationType.PUT)
        self.assertDictEqual(redo.data, {'field1': 'data1', 'field2': 'ndata2', 'field3': 'ndata3'})

    def test_append_list_undo(self):
        origin = {'data': ['item1', 'item2']}
        op = Operation(type=OperationType.NEW, path='data[2]', data='item3')
        ndata, undo = apply_operation(origin, op)
        self.assertEqual(ndata['data'], ['item1', 'item2', 'item3'])
        self.assertEqual(undo.type, OperationType.DELETE)
        self.assertEqual(undo.path, 'data[2]')
        cleaned, redo = apply_operation(ndata, undo)
        self.assertEqual(cleaned['data'], origin['data'])
        self.assertEqual(redo, op)

    def test_new_dict_key_undo(self):
        origin = {'data': {'field1': 'value1'}}
        op = Operation(type=OperationType.NEW, path='data.field2', data='value2')
        ndata, undo = apply_operation(origin, op)
        self.assertDictEqual(ndata['data'], {'field1': 'value1', 'field2': 'value2'})
        self.assertEqual(undo.type, OperationType.DELETE)
        self.assertEqual(undo.path, 'data.field2')
        cleaned, redo = apply_operation(ndata, undo)
        self.assertDictEqual(cleaned['data'], origin['data'])
        self.assertEqual(redo, op)
