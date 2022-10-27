# Quality-diversity-with-reinforcement-learning
### 创新实践课题《基于强化学习和质量多样性算法的多样策略生成》 ###
**任务目标**：复现质量多样性强化学习算法文章$^{[1]}$提出的PGA-Map-Elites算法
**任务进度**：
 * 研读了提出 Map-ELites 算法的文章，对质量多样性算法的目的和内容有了初步认识
 * 阅读了项目目标需要复现的算法的文章，了解PGA-Map-Elites 算法的大致框架，并尝试配置论文中提到的的开源环境QDgym (https://github.com/ollenilsson19/QDgym)
 * 我们在Github的qlearning_robot项目（https://github.com/nd009/qlearning_robot） 基础上完成Q-Learning代码的实现，并进行迷宫测试。随后将其运用于Gym库中的出租车调度环境中调整参数，再次测试。除此之外，我们还实现了Sarsa算法并将其与Q-Learning进行测试结果对比以比较强化学习算法分类中Off-policy与on-policy的差异。
* 研究了 Policy Gradient 算法原理，对其具体实现有了一定了解
