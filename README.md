[![Documentation Status](https://readthedocs.org/projects/python-anesthesia-simulator/badge/?version=latest)](https://python-anesthesia-simulator.readthedocs.io/en/latest/?badge=latest)
<img src ="https://img.shields.io/github/last-commit/AnesthesiaSimulation/Python_Anesthesia_Simulator" alt="GitHub last commit"> 

# Python_Anesthesia_Simulator

The Python Anesthesia Simulator (PAS) models the effect of drugs on physiological variables during total intravenous anesthesia.（Python 麻醉模拟器（PAS）用于建模全静脉麻醉过程中药物对生理变量的影响。）

It is particularly dedicated to the control community, to be used as a benchmark for the design of multidrug controllers.（该项目特别面向控制领域，可作为多药物控制器设计的基准平台。）

The available drugs are propofol, remifentanil, norepinephrine, and atracurium.（当前支持的药物包括丙泊酚、瑞芬太尼、去甲肾上腺素和阿曲库铵。）

The outputs are the Bispectral Index (BIS), tolerance to laryngoscopy (TOL), loss of consciousness (LOC), mean arterial pressure (MAP), cardiac output (CO), total peripheral resistence (TPR), stroke volume (SV), heart rate (HR), and Train of four level (TOF).（输出变量包括双频指数（BIS）、喉镜耐受度（TOL）、意识消失（LOC）、平均动脉压（MAP）、心输出量（CO）、总外周阻力（TPR）、每搏量（SV）、心率（HR）和四个成串刺激水平（TOF）。）

PAS includes different well-known models along with their uncertainties to simulate inter-patient variability.（PAS 集成了多种经典模型及其不确定性，用于模拟患者间差异。）

Blood loss can also be simulated to assess the controller's performance in a shock scenario.（同时支持失血场景模拟，以评估控制器在休克情境下的表现。）

Finally, PAS includes disturbance profiles calibrated on clinical data to facilitate the evaluation of the controller's performances in realistic condition.（最后，PAS 还提供了基于临床数据校准的扰动曲线，便于在更贴近真实临床条件下评估控制器性能。）

- **Documentation and examples（文档与示例）:** <https://python-anesthesia-simulator-doc.readthedocs.io>
- **Associated paper（相关论文）:** <https://joss.theoj.org/papers/10.21105/joss.05480>

## Installation（安装）

Use pip to install the package:（使用 `pip` 安装软件包：）

```bash
pip install python-anesthesia-simulator
```

Or, to get the latest version, clone this repository and install the package with:（或者，为了获取最新版本，先克隆本仓库后再执行：）

```bash
pip install .
```
## Citation（引用）

To cite PAS in your work, cite this paper:（如果你在研究或项目中使用 PAS，请引用以下论文：）

```
Aubouin-Pairault et al., (2023). PAS: a Python Anesthesia Simulator for drug control. Journal of Open Source Software, 8(88), 5480, https://doi.org/10.21105/joss.05480
```

## Guidelines（贡献指南）

Contribution and discussions are welcomed!（欢迎贡献和讨论！）Please feel free to use the [issue tracker](https://github.com/AnesthesiaSimulation/Python_Anesthesia_Simulator/issues) ensuring that you follow our [contribution guide](https://python-anesthesia-simulator-doc.readthedocs.io/latest/contributing.html) and our [Code of Conduct](./CODE_OF_CONDUCT.md).（请通过 [Issue Tracker](https://github.com/AnesthesiaSimulation/Python_Anesthesia_Simulator/issues) 反馈问题，并确保遵循我们的[贡献指南](https://python-anesthesia-simulator-doc.readthedocs.io/latest/contributing.html)和[行为准则](./CODE_OF_CONDUCT.md)。）

## Structure（项目结构）

    .
    ├─── src
    |   ├─── python_anesthesia_simulator           # Simulator library + metrics function（模拟器库 + 指标计算函数）
    |
    ├── tests              # files for testing the package（包测试文件）
    |
    ├── docs               # files for generating the docs（文档生成文件）
    | 
    ├── LICENSE
    ├── pyproject.toml      # packaging file（打包配置文件）
    ├── requirements.txt
    ├── README.md
    └── .gitignore          

## License（许可证）

_GNU General Public License 3.0_

## Authors（作者）

Bob Aubouin--Pairault, Michele Schiavo, Erhan Yumuk
