import os, pickle, sys
from cache import LFUCache


dirname = os.path.dirname(__file__)
capacity = 400
lfuCachePath = os.path.join(dirname, "cache2.txt")

cache = LFUCache(capacity)
cache.set("1", "oi")
with open(lfuCachePath, "wb") as f:
	print(pickle.dump(cache, f))