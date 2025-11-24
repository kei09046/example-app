#include "rotation.h"

inputMatrix rotate90(const inputMatrix mat) {
    inputMatrix res;
    int cnt = 0, dnt = 0;

    for(int k = 0; k < inputDepth; ++k){
        for (int i = 0; i < inputRow; ++i)
            for (int j = 0; j < inputCol; ++j)
                res[cnt++] = mat[dnt + (inputRow - 1 - j) * inputCol + i];
        
        dnt += inputSize;
    }
    return res;
}

inputMatrix reflectHorizontal(const inputMatrix mat) {
    inputMatrix res;
    int cnt = 0, dnt = 0;

    for(int k=0; k<inputDepth; ++k){
        for (int i = 0; i < inputRow; ++i)
            for (int j = 0; j < inputCol; ++j)
                res[cnt++] = mat[dnt + (inputRow - 1 - i) * inputCol + j];

        dnt += inputSize;
    }
    return res;
}

std::vector<inputMatrix> generateDihedralTransformations(const inputMatrix mat) {
    std::vector<inputMatrix> transforms;
    
    // Original inputMatrix
    transforms.push_back(mat);

    // Rotations
    inputMatrix rot90 = rotate90(mat);
    inputMatrix rot180 = rotate90(rot90);
    inputMatrix rot270 = rotate90(rot180);

    transforms.push_back(rot90);
    transforms.push_back(rot180);
    transforms.push_back(rot270);

    // Reflections
    inputMatrix reflH = reflectHorizontal(mat);
    transforms.push_back(reflH);
    transforms.push_back(rotate90(reflH));
    transforms.push_back(rotate90(rotate90(reflH)));
    transforms.push_back(rotate90(rotate90(rotate90(reflH))));

    return transforms;
}


outputMatrix rotate90(const outputMatrix mat) {
    outputMatrix res;
    int cnt = 0;

    for (int i = 0; i < outputRow; ++i)
        for (int j = 0; j < outputCol; ++j)
            res[cnt++] = mat[(outputRow - 1 - j) * outputCol + i];

    res[cnt] = mat[cnt];
    return res;
}

outputMatrix reflectHorizontal(const outputMatrix mat) {
    outputMatrix res;
    int cnt = 0;

    for (int i = 0; i < outputRow; ++i)
        for (int j = 0; j < outputCol; ++j)
            res[cnt++] = mat[(outputRow - 1 - i) * outputCol + j];

    res[cnt] = mat[cnt];
    return res;
}

std::vector<outputMatrix> generateDihedralTransformations(const outputMatrix mat) {
    std::vector<outputMatrix> transforms;
    
    // Original outputMatrix
    transforms.push_back(mat);

    // Rotations
    outputMatrix rot90 = rotate90(mat);
    outputMatrix rot180 = rotate90(rot90);
    outputMatrix rot270 = rotate90(rot180);

    transforms.push_back(rot90);
    transforms.push_back(rot180);
    transforms.push_back(rot270);

    // Reflections
    outputMatrix reflH = reflectHorizontal(mat);
    transforms.push_back(reflH);
    transforms.push_back(rotate90(reflH));
    transforms.push_back(rotate90(rotate90(reflH)));
    transforms.push_back(rotate90(rotate90(rotate90(reflH))));

    return transforms;
}

std::vector<TrainData*> generateDihedralTransformations(const TrainData& data) {
    std::vector<TrainData*> transformed_data;
    
    auto rotatedStates = generateDihedralTransformations(std::get<0>(data));
    auto rotatedMoves = generateDihedralTransformations(std::get<1>(data));
    auto value = std::get<2>(data);
    auto del_flag = std::get<3>(data);

    for(int i=0; i<rotatedStates.size(); ++i){
        transformed_data.push_back(new TrainData(rotatedStates[i], rotatedMoves[i], value, del_flag));
    }

    return transformed_data;
}

PolicyValueOutput rotateNNOutput(const PolicyValueOutput& original,
                                 const std::vector<std::pair<int,int>>& legal,
                                 int s, int N) 
{
    const auto& policy = original.first;
    float value = original.second;
    size_t L = legal.size();

    // Compute rotated legal positions
    std::vector<std::pair<int,int>> rotated(L);
    for (size_t i = 0; i < L; ++i)
        rotated[i] = rot(s, legal[i].first, legal[i].second, N);

    // Create index array
    std::vector<size_t> idx(L);
    for (size_t i = 0; i < L; ++i) idx[i] = i;

    // Sort indices based on rotated legal positions
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b){ return rotated[a] < rotated[b]; });

    // Build new policy vector in sorted order
    std::vector<float> new_policy(L);
    for (size_t i = 0; i < L; ++i)
        new_policy[i] = policy[idx[i]];

    return {new_policy, value};
}

// Returns rotated PolicyValueOutput AND rotated legal moves
std::pair<PolicyValueOutput, std::vector<std::pair<int,int>>> rotateNNOutputandLegal(const PolicyValueOutput& original,
               const std::vector<std::pair<int,int>>& legal, int N, int s) 
{
    const auto& policy = original.first;
    float value = original.second;
    size_t L = legal.size();

    // Compute rotated legal positions
    std::vector<std::pair<int,int>> rotated_legal(L);
    for (size_t i = 0; i < L; ++i)
        rotated_legal[i] = rot(s, legal[i].first, legal[i].second, N);

    // Create index array
    std::vector<size_t> idx(L);
    for (size_t i = 0; i < L; ++i) idx[i] = i;

    // Sort indices based on rotated legal positions
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b){ return rotated_legal[a] < rotated_legal[b]; });

    // Build new policy vector in sorted order
    std::vector<float> new_policy(L);
    for (size_t i = 0; i < L; ++i)
        new_policy[i] = policy[idx[i]];

    // Reorder legal in the same way
    std::vector<std::pair<int,int>> new_legal(L);
    for (size_t i = 0; i < L; ++i)
        new_legal[i] = rotated_legal[idx[i]];

    return {{new_policy, value}, new_legal};
}


std::vector<PolicyValueOutput> rotateAllNNOutputs(
    const PolicyValueOutput& original,
    const std::vector<std::pair<int,int>>& legal,
    int N)
{
    std::vector<PolicyValueOutput> outputs;
    outputs.reserve(8);

    for (int s = 0; s < 8; ++s) {
        outputs.push_back(rotateNNOutput(original, legal, s, N));
    }

    return outputs;
}
