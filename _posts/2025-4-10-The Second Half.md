---
layout: post
title: The Second Half
---

The Second Half

tldr: We’re at AI’s halftime. 

For decades, AI has largely been about developing new methods, models, and algorithms that could show improvements in major benchmarks. And it worked: from beating world champions at chess and Go, surpassing most humans on the SAT and bar exams, to earning IMO and IOI gold medals. Behind these historic milestones—DeepBlue, AlphaGo, GPT-4, and the o-series—are fundamental innovations in AI methods: search, deep RL, scaling, and reasoning. 

What’s different now? In three words: RL finally works. More precisely: RL finally generalizes. After several major detours, we’ve landed on a working recipe to solve a wide range of RL tasks using language and reasoning. Even a year ago, if you told most AI researchers that a single recipe could tackle software engineering, creative writing, IMO-level math, mouse-and-keyboard manipulation, and long-form question answering — they’d laugh at your hallucinations. Each of these tasks is incredibly difficult and many researchers spend their entire PhDs focused on just one narrow slice. Yet it happened.

So what comes next? The second half of AI—starting now—will shift focus from solving well-defined problems to defining the right problems. In this new era, evaluation takes center stage. Instead of just asking, “Can we train a model to solve X?”, we’re asking, “What should we be training AI to do, and how do we measure real progress?” To thrive in this second half, we’ll need a timely shift in mindset and skill set, ones perhaps closer to a product manager than to a scientist or hacker.


## The first half 

To make sense of the first half, let’s make sense of its winners. What do you consider to be the most impactful AI papers so far?

I tried the quiz in Stanford 224N, and the answers were not surprising: Transformer, AlexNet, GPT-3, etc. What’s common about these papers? They propose some fundamental breakthroughs to train better models. But also, they managed to publish their papers by showing some (significant) improvements on some benchmarks.

There is a latent commonality though: these “winners” are all models or training methods, not benchmarks. Even arguably the most impactful benchmark of all, ImageNet, has less than one third of the citation of AlexNet. The contrast of method vs benchmark is even more drastic anywhere else —- for example, the main benchmark of Transformer is WMT’14, whose workshop report has ~1300 citations, while Transformer had >160000.

That’s the game of the first half: focus on building new models and methods, and evaluation and benchmark are secondary (although necessary to make the paper system work).

Why? A big reason is that, in the first half of AI, methods were harder and more exciting than tasks. Creating a new algorithm or model architecture from scratch – think of breakthroughs like the backpropagation algorithm, convolutional networks (AlexNet), or the Transformer used in GPT-3 – required remarkable insight and engineering. In contrast, defining tasks for AI often felt more straightforward: we simply took tasks humans already do (like translation, image recognition, or chess) and turned them into benchmarks. Not much insight or even engineering.

Methods also tended to be more general and widely applicable than individual tasks, making them especially valuable. For example, the Transformer architecture ended up powering progress in CV, NLP, RL, and many other domains – far beyond the single dataset (WMT’14 translation) where it first proved itself. A great new method can hillclimb many different benchmarks because it’s simple and general, thus the impact tends to go beyond an individual task. 

This game has worked for decades and sparked world-changing ideas and breakthroughs, which manifested themselves by ever-increasing benchmark performances in various domains. Why would the game change at all? Because the cumulation of these ideas and breakthroughs have made a qualitative difference in creating a working recipe in solving general tasks.


## The recipe

What’s the recipe? Its ingredients, not surprisingly, include massive language pre-training, scale (in data and compute), and the idea of reasoning and acting. These might sound like buzzwords that you hear everyday in Bay Area… So how do they form a recipe?

We can understand this by looking through the lens of reinforcement learning (RL), which is often thought of as the “end game” of AI — after all, RL is theoretically guaranteed to win games, and empirically it’s hard to imagine any superhuman systems (e.g. AlphaGo) without RL and relying purely on imitation learning (IL).

In RL, there are three key components: the algorithm, the environment, and the agent’s priors. For a long time, RL researchers focused mostly on the algorithm – the intellectual core of how an agent learns – while treating the environment and priors as fixed or minimal. For example, Sutton and Barto’s classical textbook is all about algorithms and almost nothing about environments or priors.

However, in the era of deep RL, it became clear that environments matter a lot: an algorithm’s performance is often highly specific to the environment it was developed and tested in. If you ignore the environment, you risk building an “optimal” algorithm that only excels in toy settings. 

So why don’t we first figure out the environment, then find the algorithm best suited for it? That’s exactly the plan of OpenAI at its beginning. It built gym, a standard RL environment for various games, then the World of Bits and Universe projects, trying to make the Internet or computer a game to be solved. A good plan, isn’t it? Once we turn these digital worlds into an environment, solve these environments with smart RL algorithms, we have digital AGI.

A good plan, but not working. OpenAI made tremendous progress down the path, using RL to solve Dota, robotic hands, etc. But it never came close to solving computer use or web navigation, and the RL agents working in one domain do not transfer to another. Something is missing.

Only after GPT-2 or GPT-3, it turned out that the missing piece is priors. You need powerful language pre-training to distill general commonsense and language knowledge into models, which then can be fine-tuned to become web (WebGPT) or chat (ChatGPT) agents (and change the world). It turned out the most important part of RL might not even be the RL algorithm or environment, but the priors (which can be obtained in a way totally unrelated from RL).


Language pre-training created a good priors for chatting, but not for everything. For playing a game with drastically different 



## The second half
