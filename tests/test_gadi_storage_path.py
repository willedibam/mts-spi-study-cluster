"""Historical path lookup must not silently redirect deleted data or siblings."""
import unittest
from scripts.gadi_storage_path import resolve_path


class StoragePathTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {'paths': {
            '/old': {'path': '/new', 'status': 'live'},
            '/old/run': {'path': '/gdata/run', 'status': 'live'},
            '/old/deleted': {'path': '/gone', 'status': 'deleted'},
            '/old/archive': {'path': '/bank.tar.zst', 'status': 'archived'},
        }}

    def test_longest_prefix_and_suffix(self):
        self.assertEqual(resolve_path('/old/run/seed1/meta.json', self.mapping), '/gdata/run/seed1/meta.json')

    def test_path_component_boundary(self):
        self.assertEqual(resolve_path('/old/run2/file', self.mapping), '/new/run2/file')
        self.assertEqual(resolve_path('/older/run', self.mapping), '/older/run')

    def test_later_archive_applies_through_old_alias(self):
        self.mapping['paths']['/gdata/run'] = {'path': '/bank.tar.zst', 'status': 'archived'}
        with self.assertRaises(ValueError):
            resolve_path('/old/run/meta.json', self.mapping)

    def test_cycle_is_reported(self):
        self.mapping['paths']['/new'] = {'path': '/old', 'status': 'live'}
        with self.assertRaises(ValueError):
            resolve_path('/old/file', self.mapping)

    def test_retired_data_requires_explicit_action(self):
        for path in ['/old/deleted/input.npy', '/old/archive/meta.json']:
            with self.assertRaises(ValueError):
                resolve_path(path, self.mapping)


if __name__ == '__main__':
    unittest.main()
