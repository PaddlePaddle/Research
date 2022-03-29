# Long Time No See! Open-Domain Conversation with Long-Term Persona Memory

Data and codes of the paper: [Long Time No See! Open-Domain Conversation with Long-Term Persona Memory](https://arxiv.org/pdf/2203.05797.pdf). Findings of ACL 2022 (Long Paper)

<img src=./images/fig1.png width=500>

## Abstract

Most of the open-domain dialogue models tend to perform poorly in the setting of long-term human-bot conversations. The possible reason is that they lack the capability of understanding and memorizing long-term dialogue history information. To address this issue, we present a novel task of Long-term Memory Conversation (LeMon) and then build a new dialogue dataset DuLeMon and a dialogue generation framework with Long-Term Memory (LTM) mechanism (called PLATO-LTM). This LTM mechanism enables our system to accurately extract and continuously update long-term persona memory without requiring multiple-session dialogue datasets for model training. To our knowledge, this is the first attempt to conduct real-time dynamic management of persona information of both parties, including the user and the bot. Results on DuLeMon indicate that PLATO-LTM can significantly outperform baselines in terms of long-term dialogue consistency, leading to better dialogue engagingness.


## Dataset: DuLeMon

Bai**du** **L**ong-t**e**rm **M**emory C**on**versation

![](./images/fig3.png)

Download:

```sh
cd ./data
bash download_dulemon.sh
```

Example of a conversation session:
```
{
   "bot_persona":[
      "B1: 我 喜欢 做饭 ， 但是 不 爱 刷 碗",
      "B2: 我 喜欢 去 不同 的 国家 旅行",
      "B3: 我 25 岁"
   ],
   "user_said_persona":[
      "U1: 我 在 学校 的 图书馆 打工",
      "U2: 每次 有 机会 我 都 会 去 游泳",
      "U3: 我 喜欢 雪",
      "U4: 我 最 喜欢 的 演员 是 胡歌",
      "U5: 我 叫 曾鹏"
   ],
   "user_no_said_persona":[
      "U6: 我 在 上 大学"
   ],
   "conversation":[
      "Usr: 我 的 大学 生活 真 是 太 轻松 了 。",
      "Bot: 嗯嗯 ， 看起来 是 挺 轻松 的 。 你 还在 图书馆 里 打工 呢 。\tU1",
      "Usr: 是 的 ， 平时 也 没有 什么 事情 ， 闲 的 时间 很多 。",
      "Bot: 那 挺好 的 ， 你 也 有 很多 时间 去 游泳 了 。\tU2",
      "Usr: 是 呀 ， 我 现在 每天 过 的 都 很 充实 。",
      "Bot: 嗯嗯 ， 最近 胡歌 要 出演 电视 剧 了 ， 你 知道 吗 ？\tU4",
      "Usr: 当然 了 ， 我 可是 他的 铁粉 ， 他的 每个 电视 剧 我 都 会 看 。",
      "Bot: 我 也 很 喜欢 他 演 的 电视 ， 他的 演技 我 觉得 特别 好 。",
      "Usr: 是 的 。 每次 他 演 的 角色 都 很 能 打动 我 。",
      "Bot: 嗯嗯 ， 都 会 引起 很多 人 的 触动 。",
      "Usr: 对 的 ， 你 最近 在 干嘛 呀 ？",
      "Bot: 我 打算 去 不同 的 国家 旅行 ， 我 真 的 太 喜欢 出去 了 。\tB2",
      "Usr: 那 这 是 一 个 大 工程 呀 ， 也 得 具备 很多 钱 。",
      "Bot: 嗯嗯 ， 我 知道 。 我 也 在 为 之 付出 努力 认真 的 工作 呢 。",
      "Usr: 真好 呀 ， 为 自己 喜欢 的 事情 努力 也 会 很 有 动力 。",
      "Bot: 是 的 。"
   ]
}
```


## Framework: PLATO-LTM

![](./images/fig2.png)

Coming soon.


## Citations
If you find our paper and code useful, please cite the following paper:
```
@article{xu2022long,
  title={Long Time No See! Open-Domain Conversation with Long-Term Persona Memory},
  author={Xu, Xinchao and Gou, Zhibin and Wu, Wenquan and Niu, Zheng-Yu and Wu, Hua and Wang, Haifeng and Wang, Shihang},
  journal={arXiv preprint arXiv:2203.05797},
  year={2022}
}
```
