# Heroes RL

Should a hero attack an enemy town today or wait a turn for reinforcements? Waiting gives the defender time to recruit, too. The value of that decision may only become clear days later, when the town changes hands. Heroes RL explores how a policy can learn to make those choices by playing *Heroes of Might and Magic III* against itself.

![A Heroes III battle beside terminal windows showing self-play and policy training.](docs/media/heroes-self-play.webp)

## Learning through self-play

A fresh policy starts with randomly initialized weights. Two copies of the same network play Red and Blue in the VCMI engine. Each sees its own side of the map and remembers what it has seen across turns. The policy chooses where to move, when to recruit, what to build and when to attack. VCMI’s BattleAI resolves the tactical battles.

We design the reward and let the policy work out how to earn it. A win earns a positive reward, a loss earns a penalty, and both sides receive a penalty if neither wins within 365 days. Each bot also has a budget of 327,680 decisions per game. A smaller penalty discourages repeatedly opening and closing an unchanged window. Opening a town to recruit is free. The reward leaves the choice of army, route and timing to the policy.

After each batch of games, Proximal Policy Optimization (PPO) uses the rewards to estimate how to adjust the network’s weights. Choices that led to better outcomes than expected tend to become more likely in similar situations. Both sides use the updated policy in the next batch, so a successful opening can eventually face an opponent that knows how to answer it.

## What strategies can emerge?

Hiring a second hero to scout costs gold that could have bought troops for the first. Under fog of war, when is the extra information worth fielding a smaller army? By letting the policy try different answers across many games, we can look for openings, army compositions and attack timings that even experienced players might overlook.

We test a new policy against a frozen earlier version on fresh seeds, playing each seed from both colors. The engine records the actions and changes in town ownership, so we can trace a win back to the turns that set it up. [Research journal](docs/research-log.md) · [Architecture](docs/architecture.md) · [Scenario and action restrictions](docs/temporary-restrictions.md)

## Run it

The default training map is 36×36 with four neutral gold mines. The bot can open the artifact panel only when equipment is enabled. These commands inspect the selected checkpoint and collect 100 usable games for one PPO update. The launcher resumes the most recent local checkpoint. An empty default state starts from the published checkpoint 2025.

```sh
./train.command check
./train.command 100
```

For a new checkout, the training guide covers building the patched engine, supplying your own game data, and starting from the published checkpoint or random weights. [Training guide](docs/training.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)
