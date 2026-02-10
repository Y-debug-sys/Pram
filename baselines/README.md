# Baselines of Pram

This directory contains the implementation of the baseline methods used in our experimental evaluation.

```
.baselines
├── 📁 LP                       # Implementation of linear programming (LP)
├── 📁 ML                       # Implementation of learning-based methods
│   ├── 📁 HARP                 # Implementation of HARP (SIGCOMM '24)
│   └── 📁 Aether               # Implementation of Aether (INFOCOM '25)
├── 📁 RL                       # Implementation of deep RL (PPO)
└── 📁 pop-ncflow-lptop         # Implementations of POP (SOSP '21) and LP-top (HotNets '23)
```

If you want to use the code, please cite the following papers:

```bibtex
@inproceedings{yuan2026divide,
  title={Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models},
  author={Xinyu Yuan and Yan Qiao and Zonghui Wang and Wenzhi Chen},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=kL9nYFvs6O}
}

@inproceedings{teal,
    title={Teal: Learning-Accelerated Optimization of WAN Traffic Engineering},
    author={Xu, Zhiying and Yan, Francis Y. and Singh, Rachee and Chiu, Justin T. and Rush, Alexander M. and Yu, Minlan},
    booktitle={Proceedings of the ACM SIGCOMM 2023 Conference},
    pages={378--393},
    month=sep,
    year={2023}
}

@inproceedings{narayanan2021solving,
  title={Solving large-scale granular resource allocation problems efficiently with pop},
  author={Narayanan, Deepak and Kazhamiaka, Fiodar and Abuzaid, Firas and Kraft, Peter and Agrawal, Akshay and Kandula, Srikanth and Boyd, Stephen and Zaharia, Matei},
  booktitle={Proceedings of the ACM SIGOPS 28th Symposium on Operating Systems Principles (SOSP 21)},
  pages={521--537},
  year={2021}
}

@inproceedings{fan2025aether,
  title={Aether: Toward Generalized Traffic Engineering with Elastic Multi-agent Graph Transformers},
  author={Fan, Yu and Liu, Jingyao and Xie, Pengjin and Liu, Liang},
  booktitle={IEEE INFOCOM 2025-IEEE Conference on Computer Communications},
  pages={1--10},
  year={2025},
  organization={IEEE}
}
```
