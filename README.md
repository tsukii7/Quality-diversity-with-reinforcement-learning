# Quality-diversity-with-reinforcement-learning
### 创新实践课题《基于强化学习和质量多样性算法的多样策略生成》 ###
**任务目标**：
质量多样性强化学习算法文章 [3] 提出了 PGA-Map-Elites 算法，它将基于 actor-critic 架构的面向连续动作空间的深度强化学习算法 TD3 与质量多样性算法 Map-Elites 相结合, 成功地阐明了搜索空间中性能与关心的维度之间的关系，并在整个可能的行为空间内找到了高性能的解决方案。本学期的最终目标为复现文章 [3]，并结合已有研究，对当前算法框架提出改进，更换其他 DRL 算法或根据实现细节改进现有算法中的一些不足 [4]。计划选择基于 Mujoco 机器人控制模拟环境的 QDGym 作为测试所需的强化学习环境。

**学期工作总结**:
1、项目初期（1-7 周）：
* 对于质量多样性优化，我们首先研读了提出Map-ELites 算法的文章 [2]，对质量多样性算法的目的和内容有了初步认识。其次，我们阅读了项目目标需要复现的算法的文章 [3]，了解 PGA-Map-Elites 算法的大致框架，并尝试配置论文中提到的的开源环境 QDgym(https://github.com/ollenilsson19/QDgym)。
* 对于强化学习，我们首先学习并复现了的 Q-learning、Sarsa 算法，并调整参数进行测试。
  
2、项目中期（8-12 周）：
* 沿着既定的算法学习路线，学习复现与 TD3 相关的 DQN、DDPG 算法，并完成 PGA-Map-Elites 算法中所需的 TD3 算法的复现，对算法进行调参测试。
* 结合 TD3 与 MAP-Elites 最终初步复现“Policygradient assisted MAP-Elites”，将其应用于QDgym 环境进行训练模型测试并与原论文 [3]的实验结果进行比较。

3、项目后期（13-16 周）：
* 对 不 完 善 的 初 步 复 现 版 本 进 行 不 断修 正， 测 试， 最 终 实 现 了 表 现 较 好 的PGA-MAP-Elites 算 法 复 现 （代 码 见Github 仓 库https://github.com/tsukii7/Quality-diversity-with-reinforcement-learning）。
* 对比复现版本与源码的实验结果，总结不足和改进方向。

**评价与改进**:
    在本学期的项目中，我们学习了强化学习以及质量多样性的相关课题背景，较好地完成了 PGA-Map-Elites算法的复现，验证了其同时具备产生高性能解决方案以及高效地产生多样性策略的能力，但未能对其做出很好的改进。在原论文 [3] 的实现中，采用了添加高斯噪声的方式对 Actor 神经网络参数进行变异，产生新的个体，然而，论文 [1] 指出，随机突变通常适用于低纬度，对于大型深度神经网络，数千或数百万个权重的随机扰动可能会破坏现有的功能，而 PGA-MAP-Elites 涉及的 TD3中的神经网络正需要巨量参数。因而一个可改进的方向是采用 [1] 中的安全变异 (SM) 算子，探索其能否提高基于遗传算法进化的神经网络在需要深度高维域中寻找新解决方案的能力。

**参考**：
[1] Joel Lehman, Jay Chen, Jeff Clune, and Kenneth OStanley. Safe mutations for deep and recurrentneural networks through output gradients. pages117–124, 2018.
[2] J.-B. Mouret and J. Clune. Illuminating searchspaces by mapping elites. arXiv preprintarXiv:1504.04909, 2015.
[3] O. Nilsson and A. Cully. Policy gradient assistedmap-elites. Genetic and Evolutionary ComputationConference, page 866–875, 2021.
[4] 王子祺. 基于强化学习和质量多样性算法的多样策略生成. 创新实践课题介绍, 2022