#ifndef PMCTS_H
#define PMCTS_H

#include "gamerules.h"
#include "policyvalue.h"
#include "random.h"
#include "memorypool.h"
#include "hash.h"
#include "evalcache.h"
#include <vector>
#include <utility>
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
#include <unordered_map>


using MoveData = std::tuple<std::pair<int, int>, std::array<float, outputSize> >; // move + move possibility

class alignas(64) Node{
private:
    const Game game; // includes position, territory, valid moves etc. for heuristic
    float N, W, P, initQ; // N : # of visits, W : total action-value Q : mean action-value P : prior evaluation from nn
    color turn;

    std::vector<Node*> child;
    std::vector<std::pair<int, int> > legal;
    std::pair<int, int> winmove;

    void expand();

    static std::vector<float> softmax(std::vector<float>& logit);

public:
    uint refCount; // counts how many parents share this node
    const HashValue hashValue; // hash value needed for transition table and evaluation hash, for each dihedral transformation
    
    Node(const Game& g);

    Node(const Game& g, const HashValue hashValue); // prevent computing hashValue twice.

    //void deletetree();

    float searchandPropagate(PolicyValueNet& net);

    std::pair<int, int> selectMove(float temperature);

    MoveData selectMoveProb(float temperature);

    Node* jump(std::pair<int, int> move);
};

class alignas(64) MCTS{
private:
    Node* root;
    int playout;
    PolicyValueNet* net;

public:
    MCTS(int playout, PolicyValueNet* net);

    void runSimulation();

    std::pair<int, int> getMove(float temperature);

    MoveData getMoveProb(float temperature);

    bool jump(std::pair<int, int> move);

    void reset();

    void updateModel(); // what to do when model gets updated. 

    #ifdef measureTime
    std::vector<int> getTimeStats() const;
    
    void resetTimeStats();
    #endif
};

#endif