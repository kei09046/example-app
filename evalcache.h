#ifndef EVALCACHE_H
#define EVALCACHE_H

#include <vector>
#include <mutex>
#include <unordered_map>
#include <list>
#include "consts.h"

template <typename T>
class EvalCache { // must be thread safe
    struct Shard {
        std::mutex m;
        std::list<std::pair<HashValue, T>> lru;
        std::unordered_map<HashValue,
            typename std::list<std::pair<HashValue,T>>::iterator> map;
    };

    std::vector<Shard> shards;

    Shard& getShard(HashValue h) { return shards[h % shardCount]; }

public:
    EvalCache() : shards(shardCount) {}

    bool get(HashValue h, T& out) {
        auto& s = getShard(h);
        std::lock_guard<std::mutex> lock(s.m);
        auto it = s.map.find(h);
        if(it == s.map.end()) return false;
        s.lru.splice(s.lru.begin(), s.lru, it->second);
        out = it->second->second;
        return true;
    }

    void insert(HashValue h, const T& e) {
        auto& s = getShard(h);
        std::lock_guard<std::mutex> lock(s.m);

        auto it = s.map.find(h);
        if(it != s.map.end()) {
            it->second->second = e;
            s.lru.splice(s.lru.begin(), s.lru, it->second);
            return;
        }

        s.lru.emplace_front(h, e);
        s.map[h] = s.lru.begin();

        if(s.map.size() > capPerShard) {
            auto last = s.lru.end(); --last;
            s.map.erase(last->first);
            s.lru.pop_back();
        }
    }

    void clear() {
        for (auto& s : shards) {
            std::lock_guard<std::mutex> lock(s.m);
            s.lru.clear();
            s.map.clear();
        }
    }
};

#endif