import pymongo


class MongoDbRunReader:
    def __init__(self, url: str = "mongodb://localhost:27017/", db: str = "sacred"):
        myclient = pymongo.MongoClient(url)
        mydb = myclient[db]
        self.runs = mydb["runs"]


    def getRun(self, runId: int):

        myquery = {"_id": runId}

        mydoc = self.runs.find(myquery)

        return mydoc[0]


    def compare_configs(self, run_id1: int, run_id2: int):
        config1 = self.getRun(run_id1)["config"]
        config2 = self.getRun(run_id2)["config"]

        added, removed, modified, same = self.__dict_compare__(config1, config2)

        for (k, v) in added.items(): print(f"Added: {k} ({v})")
        for (k, v) in removed.items(): print(f"Removed: {k} ({v})")
        for (k, v) in modified.items(): print(f"Changed: {k} ({v[0]} to {v[1]})")


    def __dict_compare__(self, d1, d2, prefix="", added=None, removed=None, modified=None, same=None):
        if not added: added = dict()
        if not removed: removed = dict()
        if not modified: modified = dict()
        if not same: same = dict()

        d1_keys = set(d1.keys())
        d2_keys = set(d2.keys())
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
