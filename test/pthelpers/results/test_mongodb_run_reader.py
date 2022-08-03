import unittest

from src.pthelpers.results.mongodb_run_reader import MongoDbRunReader


class MongoDBRunReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reader = MongoDbRunReader()


    def tearDown(self) -> None:
        pass


    def test_compare_configs_empty(self):
        def get_run_mock(id):
            return {"config": {}}


        self.reader.get_run = get_run_mock

        added, removed, modified, same = self.reader.compare_run_dict(1, 2, "config")

        self.assertEqual(0, len(added))
        self.assertEqual(0, len(removed))
        self.assertEqual(0, len(modified))
        self.assertEqual(0, len(same))


    def test_compare_configs_filled(self):
        def get_run_mock(id):
            return {"config": {"epochs": 5, "inner": {"veggie": "carrot"}}}


        self.reader.get_run = get_run_mock

        added, removed, modified, same = self.reader.compare_run_dict(1, 2, "config")

        self.assertEqual(0, len(added))
        self.assertEqual(0, len(removed))
        self.assertEqual(0, len(modified))
        self.assertEqual(2, len(same))


    def test_compare_configs_modified(self):
        def get_run_mock(id):
            if id == 1:
                return {"config": {"epochs": 5, "loss": "BCE", "inner": {"veggie": "carrot", "fruit": "apple"}}}
            return {"config": {"epochs": 6, "loss": "BCE", "inner": {"veggie": "potato", "fruit": "apple"}}}


        self.reader.get_run = get_run_mock

        added, removed, modified, same = self.reader.compare_run_dict(1, 2, "config")

        self.assertEqual(0, len(added))
        self.assertEqual(0, len(removed))
        self.assertEqual(2, len(modified))
        self.assertEqual(["epochs", "inner.veggie"], list(modified.keys()))
        self.assertEqual([(5, 6), ("carrot", "potato")], list(modified.values()))
        self.assertEqual(2, len(same))
        self.assertEqual(["loss", "inner.fruit"], list(same.keys()))
        self.assertEqual(["BCE", "apple"], list(same.values()))


    def test_compare_configs_removed(self):
        def get_run_mock(id):
            if id == 1:
                return {"config": {"epochs": 5, "loss": "BCE", "inner": {"veggie": "carrot", "fruit": "apple"}}}
            return {"config": {"epochs": 5, "inner": {"veggie": "carrot"}}}


        self.reader.get_run = get_run_mock

        added, removed, modified, same = self.reader.compare_run_dict(1, 2, "config")

        self.assertEqual(0, len(added))
        self.assertEqual(2, len(removed))
        self.assertEqual(["loss", "inner.fruit"], list(removed.keys()))
        self.assertEqual(["BCE", "apple"], list(removed.values()))
        self.assertEqual(0, len(modified))
        self.assertEqual(2, len(same))
        self.assertEqual(["epochs", "inner.veggie"], list(same.keys()))
        self.assertEqual([5, "carrot"], list(same.values()))


    def test_compare_configs_added(self):
        def get_run_mock(id):
            if id == 1:
                return {"config": {"epochs": 5, "inner": {"veggie": "carrot"}}}
            return {"config": {"epochs": 5, "loss": "BCE", "inner": {"veggie": "carrot", "fruit": "apple"}}}


        self.reader.get_run = get_run_mock

        added, removed, modified, same = self.reader.compare_run_dict(1, 2, "config")

        self.assertEqual(2, len(added))
        self.assertEqual(["loss", "inner.fruit"], list(added.keys()))
        self.assertEqual(["BCE", "apple"], list(added.values()))
        self.assertEqual(0, len(removed))
        self.assertEqual(0, len(modified))
        self.assertEqual(2, len(same))
        self.assertEqual(["epochs", "inner.veggie"], list(same.keys()))
        self.assertEqual([5, "carrot"], list(same.values()))


    def test_mix(self):
        def get_run_mock(id):
            if id == 1:
                return {"config": {"epochs": 5, "inner": {"veggie": "carrot"}}}


        self.reader.get_run = get_run_mock

        other_dict = {"epochs": 5, "loss": "BCE", "inner": {"veggie": "carrot", "fruit": "apple"}}
        added, removed, modified, same = self.reader.compare_run_dict(1, other_dict, "config")

        self.assertEqual(2, len(added))
        self.assertEqual(0, len(removed))

        added, removed, modified, same = self.reader.compare_run_dict(other_dict, 1, "config")
        self.assertEqual(0, len(added))
        self.assertEqual(2, len(removed))


    def test_dicts(self):
        first_dict = {"epochs": 5, "inner": {"veggie": "carrot"}}
        other_dict = {"epochs": 5, "loss": "BCE", "inner": {"veggie": "carrot", "fruit": "apple"}}
        added, removed, modified, same = self.reader.compare_run_dict(first_dict, other_dict, "config")

        self.assertEqual(2, len(added))
        self.assertEqual(0, len(removed))

        added, removed, modified, same = self.reader.compare_run_dict(other_dict, first_dict, "config")
        self.assertEqual(0, len(added))
        self.assertEqual(2, len(removed))


if __name__ == '__main__':
    unittest.main()
