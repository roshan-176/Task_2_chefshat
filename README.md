assigned variant: ROBUSTNESS AND GENERALISATION VARIANT

how to run the code: 
install the required libraries 
pip install -r requirements.txt

Experiments Conducted
In this project, several experiments were carried out to evaluate the robustness and generalisation ability of the Deep Q-Network (DQN) agent in the Chef’s Hat multi-agent card game environment.
First, the DQN agent was trained in a competitive setting against random opponent agents. The purpose was not only to achieve high performance but to observe how well the agent could learn and behave under uncertain and stochastic conditions. A custom wrapper architecture was used to ensure compatibility with the Chef’s Hat Gym interface.
The second experiment focused on random seed variability. Multiple independent runs were conducted using different random seeds while keeping the agent architecture exactly the same. This helped determine whether performance differences were due to randomness in the environment or instability in the learning process.
Another key experiment involved robustness testing without further training. The agent was evaluated against random opponents across multiple executions.Finally, behavioural stability was analysed
Overall, the experiments confirmed that the DQN agent was robust, generalised well to stochastic opponents, and maintained stable performance across different testing conditions.

The results should be interpreted in terms of robustness, stability, and generalisation, rather than just overall performance or win rate.

