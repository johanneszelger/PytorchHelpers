from typing import Union, List, Tuple

import pymongo


class MongoDbRunReader:
    def __init__(self, url: str = "mongodb://localhost:27017/", db: str = "sacred"):
        myclient = pymongo.MongoClient(url)
        mydb = myclient[db]
        self.runs = mydb["runs"]


    def get_run(self, runId: int):

        myquery = {"_id": runId}

        mydoc = self.runs.find(myquery)

        return mydoc[0]


    def __get_nested_value__(self, dct, *keys):
        for key in keys:
            try:
                dct = dct[key]
            except KeyError:
                return None
        return dct


    def compare_run_dict(self, run_id_or_dict_1: Union[int, dict], run_id_or_dict_2: Union[int, dict],
                         dict_names: Union[str, List[str], Tuple[str]] = None,
                         print_result=True):
        if isinstance(run_id_or_dict_1, int):
            dict1 = self.get_run(run_id_or_dict_1)
            if isinstance(dict_names, str): dict_names = [dict_names]
            dict1 = self.__get_nested_value__(dict1, *dict_names)
        else:
            dict1 = run_id_or_dict_1

        if isinstance(run_id_or_dict_2, int):
            dict2 = self.get_run(run_id_or_dict_2)
            if isinstance(dict_names, str): dict_names = [dict_names]
            dict2 = self.__get_nested_value__(dict2, *dict_names)
        else:
            dict2 = run_id_or_dict_2

        return self.compare_dicts(dict1, dict2, print_result)


    def compare_dicts(self, dict1, dict2, print_result=True):
        added, removed, modified, same = self.__dict_compare__(dict1, dict2)

        if print_result:
            for (k, v) in added.items(): print(f"Added: {k} ({v})")
            for (k, v) in removed.items(): print(f"Removed: {k} ({v})")
            for (k, v) in modified.items(): print(f"Changed: {k} ({v[0]} to {v[1]})")

        return added, removed, modified, same


    def __dict_compare__(self, d1, d2, prefix="", added=None, removed=None, modified=None, same=None):
        if not added: added = dict()
        if not removed: removed = dict()
        if not modified: modified = dict()
        if not same: same = dict()

        d1_keys = set(d1.keys()) if d1 else set()
        d2_keys = set(d2.keys()) if d2 else set()
        shared_keys = d1_keys.intersection(d2_keys)
        added.update({prefix + x: d2[x] for x in d2_keys - d1_keys})
        removed.update({prefix + x: d1[x] for x in d1_keys - d2_keys})

        modified_new = set(k for k in shared_keys if d1[k] != d2[k])
        modified_dict = set(x for x in modified_new if isinstance(d1[x], dict))
        modified.update({prefix + o: (d1[o], d2[o]) for o in modified_new - modified_dict})

        same.update({prefix + o: d1[o] for o in shared_keys if d1[o] == d2[o]})

        for md in modified_dict:
            added, removed, modified, same = self.__dict_compare__(d1[md], d2[md], prefix + md + ".", added, removed, modified, same)
        return added, removed, modified, same


# r = MongoDbRunReader()
# r.getRun(8)
#
# r.compare_configs(10, 11)
