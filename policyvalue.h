#ifndef POLICYVALUE_H
#define POLICYVALUE_H

#include <torch/torch.h>
#include "gamerules.h"
#include "consts.h"

class NetImpl : public torch::nn::Module{
public:
	NetImpl(bool use_gpu);
	std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& state);
	torch::nn::Conv2d cv1;
	torch::nn::BatchNorm2d bn1;
	torch::nn::Conv2d cv2;
	torch::nn::BatchNorm2d bn2;
	torch::nn::Conv2d cv3;
	torch::nn::BatchNorm2d bn3;
	torch::nn::Conv2d cv4;
	torch::nn::BatchNorm2d bn4;
	torch::nn::Conv2d at_cv3;
	torch::nn::BatchNorm2d at_bn3;
	torch::nn::Linear at_fc1;
	torch::nn::Conv2d v_cv3;
	torch::nn::BatchNorm2d v_bn3;
	torch::nn::Linear v_fc1;
	torch::nn::Linear v_fc2;

	torch::Device device;
};
TORCH_MODULE(Net);

class PolicyValueNet {
private:
	bool use_gpu;
	float l2_const = 0.0001f;
	torch::optim::Adam* optimizer;

public:
	Net policy_value_net;

	PolicyValueNet(const std::string& model_file, bool use_gpu);

	static std::array<float, inputDepth * inputSize> getData(const Game* game);

	static std::vector<float> getData(std::vector<const Game*> gameBatch);

	std::pair< std::vector<std::vector<float>>, std::vector<float> > batchEvaluate(std::vector<const Game*> gameBatch, 
		const std::vector<std::vector<std::pair<int, int>>*> legalBatch);

	PolicyValueOutput evaluate(const Game* game, const std::vector<std::pair<int, int> > legal);

	void train_step(std::array<float, inputDepth * batchSize * inputSize>& state_batch, std::array<float, batchSize * outputSize>& mcts_probs,
		std::array<float, batchSize>& winner_batch, float lr);

	void save_model(const std::string& model_file) const;

	void load_model(const std::string& model_file);
};

#endif