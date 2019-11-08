## Aim : 
To develop a bitcoin trading Agent using tensortrade 

You can get started testing on Google Colab or your local machine, by viewing our [many examples](https://github.com/notadamking/tensortrade/tree/master/examples)

---

## Prerequisites

TensorTrade requires Python >= 3.5 for all functionality to work as expected.

You can install the package from PyPi via pip or from the Github repo.

```bash
pip install tensortrade
```

OR

```bash
pip install git+https://github.com/notadamking/tensortrade.git
```

Some functionality included in TensorTrade is optional. To install all optional dependencies, run the following command:

```bash
pip install tensortrade[tf,tensorforce,baselines,ccxt,fbm]
```

OR

```bash
pip install git+https://github.com/notadamking/tensortrade.git[tf,tensorforce,baselines,ccxt,fbm]
```

 The following is the list of steps we are going to take in other to develop a fully functional bitcoin trading bot.
======================================================================================================================

(1) how to define a trading environment  for the agent.
(2) how to Define the Agent
(3) how to Training the Agent to learn a Strategy
(4) how to Save and Restore the agent learnt model.
(5) how to Tuning Your Strategy
(6) how to put the agent into Live Trading




## Work Flow
Creating an Environment :
    The Agent Achitechture
![image](_static/agent.png)

1) How to define a trading environment for the Agent
=====================================================
The tensortrade library provides some Trading environments that are fully configurable gym environments which allows you to compose your own  `InstrumentExchange`, `FeaturePipeline`, `ActionStrategy`, and `RewardStrategy` components.
Lets take some time to examine what this compoments are really providing for us to implement our own bitcoin trading agent.

Below are the codes for defining our Agents Trading Enviroment.

```python
import sys
import os
import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.simplefilter(action='ignore', category=FutureWarning)

import gym
import numpy as np

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

sys.path.append(os.path.dirname(os.path.abspath('')))

from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.live import CCXTExchange
from tensortrade.actions import DiscreteActionStrategy
from tensortrade.rewards import SimpleProfitStrategy
# Defining the Exchange
coinbase = ccxt.coinbasepro()
exchange = CCXTExchange(exchange=coinbase, base_instrument='USD')
# exchange = FBMExchange(times_to_generate=100000)

# defining the action strategy
action_strategy = DiscreteActionStrategy()

# Defining the reward strategy
reward_strategy = SimpleProfitStrategy()

# configuring the trading environment
env = TradingEnvironment(exchange=exchange,
                         action_strategy=action_strategy,
                         reward_strategy=reward_strategy,
                         feature_pipeline=None)


```

`The InstrumentExchange :`
===========================
 This component is part of the environment defination and it provides acess to the crypto exchanges that are supported by the ccxt library and it provides a two way comunication with the exchange in other to recieve live market price data from the Exchange and also place our trade automatically using the exchange live trade execution engine, there are two types of instrument exchange which are live and simulated.


```python
import ccxt
from tensortrade.exchanges.live import CCXTExchange
coinbase = ccxt.coinbasepro()
exchange = CCXTExchange(exchange=coinbase, base_instrument='USD')
```
The above is a simple use case for the live instrument exchange, we are going to be making use of the the live exchange in our implementation.

`The FeaturePipeline :`
======================
Feature pipelines are used for transforming noisy observations from the exchange into meaningful features for an agent to learn from.
For this tutorial we are going to be making use of the the featurePipeline, and set it up with the coinbase exchange so that we can perform some feature transformation on the live trading data before it is returned to the agent.  

```python
from tensortrade.features import FeaturePipeline
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.indicators import SimpleMovingAverage
price_columns = ["open", "high", "low", "close"]
normalize_price = MinMaxNormalizer(price_columns)
moving_averages = SimpleMovingAverage(price_columns)
difference_all = FractionalDifference(difference_order=0.6)
feature_pipeline = FeaturePipeline(steps=[normalize_price,
                                          moving_averages,
                                          difference_all])

# After defining the FeaturePipeline we add it to an existing exchange defined above.   
exchange.feature_pipeline = feature_pipeline

```

`Action Strategy :`
=====================
The Action strategy is a vital part or our environment, this converts the agent’s actions into executable trades.
For the purpose of this tutorial we are going to use the discrete action space of 3 actions (0 = hold, 1 = buy 100%, 2 = sell 100%), our learning agent does not need to know that returning an action of 1 is equivalent to buying an instrument. Rather, our agent is only concered with the reward for returning an action of 1 in specific circumstances, and can leave the implementation details of converting actions to trades to the ActionStrategy.
The following is the code that defines our action strategy for the crypto trading agent.

```python
from tensortrade.actions import DiscreteActionStrategy
action_strategy = DiscreteActionStrategy(n_actions=20, 
                                         instrument_symbol='BTC')

```



`Reward Strategies :`
=====================

Reward strategies receive the trade taken at each time step and return a float, corresponding to the benefit of that specific action. For example, if the action taken this step was a sell that resulted in positive profits, our RewardStrategy could return a positive number to encourage more trades like this. On the other hand, if the action was a sell that resulted in a loss, the strategy could return a negative reward to teach the agent not to make similar actions in the future.
A version of this example algorithm is implemented in SimpleProfitStrategy, however more complex strategies can obviously be used instead.
Each reward strategy has a get_reward method, which takes in the trade executed at each time step and returns a float corresponding to the value of that action. As with action strategies, it is often necessary to store additional state within a reward strategy for various reasons. This state should be reset each time the reward strategy’s reset method is called, which is done automatically when the parent TradingEnvironment is reset.

The following Code snippets are used to define the reward strategy.

```python
from tensortrade.rewards import SimpleProfitStrategy
reward_strategy = SimpleProfitStrategy()

```


2) How to Define the Agent
===========================
We are going to define our agent for a particular trading environment, then observations will be passed through the FeaturePipeline before being output to the agent.
