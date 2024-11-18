# 使用 Pytorch 复现 DOMR

由于部分文件实在太大，data/ 文件夹没有上传，以下是实验时的文件结构

```
.
├── ReadME.md
├── data
│    └── processed
│    └── raw
│        └── cicmaldroid2020.csv
├── experiments
│   └── plots
│       ├── experiment_maf_bar_chart.png
│       ├── experiment_waf_bar_chart.png
│       ├── parameter_heatmap.png
│       └── sample_ratio_line_chart.png
├── requirements.txt
└── src
    ├── main.py
    ├── models
    │   ├── __pycache__
    │   │   ├── classifier.cpython-310.pyc
    │   │   ├── meta_learning.cpython-310.pyc
    │   │   └── representation.cpython-310.pyc
    │   ├── _init_.py
    │   ├── classifier.pth
    │   ├── classifier.py
    │   ├── feature_extractor.pth
    │   ├── meta_learning.py
    │   └── representation.py
    └── utils
        ├── __pycache__
        │   ├── dataset.cpython-310.pyc
        │   ├── evaluation.cpython-310.pyc
        │   ├── feature_extraction.cpython-310.pyc
        │   └── visualization.cpython-310.pyc
        ├── dataset.py
        ├── evaluation.py
        ├── feature_extraction.py
        └── visualization.py
```

Drebin 不能够自由下载，所以目前只用了 CICMaldroid2020 数据集，后续想着使用其他数据集

要运行代码需要现按上面的项目结构创建好 data 文件夹部分

然后下载好库（建议运行时使用虚拟环境）

```python
pip install -r requirements.txt
```

然后运行 main.py 文件即可

还在施工中，欢迎提 issue 和 PR

