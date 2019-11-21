# Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

This repository is a modified version of https://github.com/deepmind/scalable_agent to support the Pommerman environment.
Modifications to the code include multi-agent support and population-based training.

## Running the Code

### Prerequisites

* [TensorFlow][tensorflow] >= 1.9.0 - code uses deprecated functionality, but still works in 1.13.1
* [DeepMind Sonnet][sonnet]
* [Pommerman][pommerman]


We include a [Dockerfile][dockerfile] that serves as a reference for the
prerequisites and commands needed to run the code.

### Single Machine Training

```sh
python experiment.py --num_agents=4 --batch_size=32 --logdir=agents/test
```

### Distributed Training

See the [PBS script][pbs]

#### Test

```sh
python agent.py
```

[arxiv]: https://arxiv.org/abs/1802.01561
[sonnet]: https://github.com/deepmind/sonnet
[tensorflow]: https://github.com/tensorflow/tensorflow
[dockerfile]: Dockerfile
[pommerman]: https://github.com/MultiAgentLearning/playground
[pbs]: train_distributed.pbs