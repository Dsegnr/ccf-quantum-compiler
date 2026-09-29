# 噪声自适应量子编译器

**2026 CCF 量子计算编程挑战赛（量旋杯）· 算法赛道** 参赛作品

面向 NISQ 硬件的量子电路编译器：把两比特门的错误率编码成硬件图上的边权重，让 SABRE 在选 SWAP 时同时优化**路径长度**与**边的质量**，并配合多策略初始布局与正反向交替路由。

## 结果

| 指标 | 分数 |
| --- | --- |
| **Final Score**（0.75 × Track A + 0.25 × Track B） | **0.7550** |
| Track A　　通用 NISQ 线路（8 题） | 0.8472 |
| Track B　　Clifford-only 压力测试（2 题） | 0.4785 |
| 对比竞赛发布包中的最优 baseline `paper_greedy_vertex`（0.6971） | **+8.3%** |

10 道公开题全部通过**合法性检查、等价性验证、无噪语义检查**三关。

## 方法

1. **多策略初始布局**：可靠性评估、度数匹配、星型 / 链式 / 社区分簇结构检测，外加随机扰动，生成候选布局后择优。
2. **噪声感知路由**：把两比特门错误率写进硬件图的边权，SWAP 选择时在路径长度与边质量之间做权衡。
3. **正反向交替路由**：用反向遍历迭代修正布局。
4. **自适应搜索参数**：按电路规模调节搜索强度，并引入 Turbo 搜索模式。

## 文件

| 文件 | 说明 |
| --- | --- |
| `compiler.py` | 编译器主入口（读取 `circuit.json` 与 `backend.json`，输出物理线路与 layout sidecar） |
| `noise_adaptive_sabre.py` | 算法核心：布局生成、噪声感知 SABRE、反向精炼、门抵消 |
| `evaluate_all.py` | 批量评测脚本，跑全部公开 benchmark |
| `analyze.py` | 结果统计与对比图 |
| `generate_figures.py` | 说明文档插图生成 |
| `评测结果/public_summary.json` | 公开集逐题结果（success rate、深度、两比特门数、编译耗时） |
| `docs/作品说明文档.pdf` | 完整作品说明文档（21 页） |

## 运行

```bash
# 单题编译
python compiler.py --circuit circuit.json --backend backend.json --outdir output/

# 批量评测
python evaluate_all.py --root /path/to/release_zh --compiler "$(pwd)/compiler.py" --outdir results/
```

依赖：Python 3、`cirq`、`networkx`、`numpy`（生成图表另需 `matplotlib`）。

## 说明

赛事已结束，作品在此存档公开，供交流参考。
