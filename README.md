# Unlearning Architecture: 拡散言語モデルから自己回帰モデルへの逆変換

本リポジトリは、「アーキテクチャを変更せず、Unlearning(忘却学習)のみによって、拡散言語モデル(Diffusion Language Model、以下 DLM)を自己回帰モデル(Autoregressive Model、以下 AR)の挙動に戻すことが可能か」という問いを実証的に検討した研究プロジェクトの実装と結果を収録している。

## 研究の問い

既存研究 (Gong et al. 2024 [3] 等) は、AR モデルから DLM への変換を **追加学習またはアーキテクチャ拡張** によって実現してきた。本研究はこれと逆向きの問いを扱う。すなわち、すでに DLM 化されたモデルから、勾配計算と既存パラメータの選択的更新のみで、AR の挙動を復元できるか。

この問いを解明することは、次の二点で意義がある。

1. **DLM が獲得した「双方向参照能力」が、どのような内部表現に支えられているか** を機構的に分析するための介入手段を与える。
2. **同一の重みが AR モードと DLM モードを行き来できるか** を検証することで、両モードの本質的な差異(アーキテクチャか、表現か)を切り分ける。

## 既存研究との対比

| 方向 | 代表研究 | 手段 |
|---|---|---|
| AR → DLM | Gong et al. 2024 [3] 等 | 追加学習、双方向 attention 化、MDM 損失導入 |
| **DLM → AR (本研究)** | (本研究) | **勾配計算による Unlearning のみ。アーキテクチャ非改変** |

## 実験設計

研究は段階的に組み立てた。各 Phase は前 Phase の結果を踏まえて目的を更新している。

### Phase 1: AR ベースラインと五軸評価フレームワーク

最初に、AR モデル(Pythia-160M、410M、1B)を双方向 attention 化と MDM 損失で DLM 化し、その後 Unlearning で AR 挙動を復元できるかを検証した。同時に、両モードを公平に評価するための五軸評価フレームワークを確立した。

**五軸評価の構成**:

| 観点 | 計測内容 |
|---|---|
| AR 性能 | 強制 causal mask 下の next-token Perplexity |
| DLM 性能 | 強制 bidirectional 下の Masked Diffusion Modeling NELBO |
| Reverse-context probe (causal) | 先頭 5 トークンを mask、causal モードで top-1 復元精度 |
| Reverse-context probe (bidirectional) | 同条件、bidirectional モードでの top-1 復元精度 |
| Reverse gap | 上記二項の差(DLM 固有の双方向参照能力の指標) |

**主結果(Pythia-160M、B4 構成: lora_only + KL-to-AR + α=0.5)**:

| 指標 | AR baseline | DLM-adapted | **Unlearned (B4)** |
|---|---|---|---|
| AR Perplexity | 81.7 | 644 | **81.7** |
| DLM NELBO | 19.2 | 10.18 | 16.15 |
| Reverse causal acc | 0.16% | 1.09% | 1.09% |
| Reverse bidir acc | 0.66% | 6.72% | 1.05% |
| Reverse gap | +0.50 pp | +5.63 pp | **−0.04 pp** |

![Phase 1 五軸評価サマリ](figs/summary_viz.png)

すなわち、Pythia-160M を一旦 DLM 化(双方向化 + MDM 適応)した後、勾配計算のみで AR ベースラインと統計的に区別できない水準まで復元できた。同型の現象は Pythia-410M、Pythia-1B、LLaMA-3.1-8B でも再現を確認している。

![モデル規模スケーリング](figs/scaling_viz.png)

### Phase 2: LLaDA-8B-Base を対象とする拡張

Phase 1 は「自分で DLM 化した AR モデル」を AR に戻す実験であり、対象モデルが本来 AR 由来であった点で予備的である。Phase 2 では、はじめから DLM として事前学習された LLaDA-8B-Base (Nie et al. 2025 [4]) を対象とすることで、より厳密な検証に進んだ。

**評価データセット reversal_v1**:

500 個の架空エンティティ対(例: "Aelirius Vondrek is paired with Quorn-7.")を生成し、forward 方向と reverse 方向の cand@1 を測定する。架空エンティティを用いるのは、事前学習データに存在しない関係を新規に学習させ、リークの影響を排除するためである。

**Phase 2 の SFT 後の挙動**:

| 評価 | LLaDA-8B-Base (Phase 2 SFT 後) | Pythia-160M AR (同データ SFT 後) |
|---|---|---|
| forward cand@1 | **0.995** | 0.91 |
| reverse cand@1 | **0.980** | 0.16 |
| forward−reverse gap | **+1.0 pp** | +75 pp |

LLaDA は順方向と逆方向で対称な性能を示し、reversal curse (Berglund et al. 2023 [1]) を持たない。これに対し AR である Pythia-160M は典型的な逆向き劣化を示す。両者の +33.5 pp という gap 差は、本研究の Unlearning が削減を試みる対象である。

![Phase 2 LLaDA SFT 訓練曲線](figs/wandb_p2_llada_sft.png)

### Phase 2.5: 推論時 causal bias 注入による参照実験

Unlearning に進む前に、「LLaDA に inference 時の attention bias として causal mask を強制したら何が起きるか」を確認した。これはアーキテクチャ的には「LLaDA の重みのまま AR 風 attention で動かす」操作であり、Unlearning の上限線として参考になる。

| Mode | forward cand@1 | reverse cand@1 |
|---|---|---|
| natural (LLaDA 既定) | 0.995 | 0.980 |
| causal bias 強制 | **0.240** | **0.155** |

結果は mode collapse(両方向とも壊滅的に劣化)であり、「同一重みのまま attention だけを causal にする」操作では AR 化は達成されない。LLaDA は推論時の双方向 attention を前提に重みを学習しており、attention 構造を切り替えるだけでは出力分布全体が崩壊する。これは Unlearning が「attention 構造を維持したまま、重みの方を AR 寄りに調整する」必要があることを示唆した。

### Phase 3 v2: 残差ストリーム右側文脈不変性損失 (L_rs)

Mech-Interp(機構解釈)文献に基づき、「DLM の双方向参照能力は、各層の残差ストリームに右側文脈情報が統合されて流れる挙動として現れる」という仮説を立てた。これに基づき、自己蒸留型の損失 **L_rs** を導入した。

**L_rs の定義**: 入力位置 *i* で、左側のみ可視 (`[0, i)` の元 token + `[i, seq)` を `[MASK]` で潰した入力)と全文脈可視(元の入力)で、各層の hidden state が一致するように forward 側を引き寄せる平均二乗誤差。

```
L_rs = Σ_layer || h_layer^full(i) − h_layer^left_masked(i) ||² / N
```

これを αL_rs + retain·CE の合算として AdamW8bit で 1000 step 最小化する。retain 損失は WikiText-103 上の通常 MDM CE で、一般言語能力の保持を担う。

**Phase 3 v2 の結果(LLaDA-8B-Base、reversal_v1 評価)**:

| 評価 | Phase 2 | Phase 3 v2 |
|---|---|---|
| forward cand@1 (eval_memory) | 0.995 | 0.865 |
| reverse cand@1 (eval_logic) | 0.980 | 0.755 |
| forward−reverse gap | +1.0 pp | **+11.0 pp** |

Pythia-160M AR の参照点 (+33.5 pp) には届かないが、勾配計算のみによって reverse 側を選択的に劣化させ、forward−reverse 非対称性を生じさせることに成功した。

### Phase 3 v3: 層スライス + マルチポジション化

Transformer 系列の階層別機能分業に関する先行研究 (Tenney et al. 2019 [5]、Clark et al. 2019 [2]) は、表層的な構文処理が浅層に、意味的統合が深層に局在することを示している。本研究の予備観察でも、LLaDA の双方向情報統合は深層 (層 20-32 周辺) に集中する傾向が認められた。Phase 3 v2 は 33 層(embedding + 32 層)を一様平均しており、深層への勾配が約 1/3 に希釈されていた可能性がある。Phase 3 v3 では二点を改良した。

1. **層スライス**: L_rs を `hidden_states[20:]` のみに制限。
2. **マルチポジション**: バッチごとに 4 個の位置 *i* を一様サンプリングして平均(full forward は共有、left-masked forward のみ no_grad で 4 回回す)。

**Phase 3 v3 の結果**:

| 評価 | Phase 2 | Phase 3 v2 | Phase 3 v3 |
|---|---|---|---|
| forward cand@1 | 0.995 | 0.865 | **0.845** |
| reverse cand@1 | 0.980 | 0.755 | **0.705** |
| forward−reverse gap | +1.0 pp | +11.0 pp | **+14.0 pp** |

![Phase 2.5 mode collapse と Phase 3 進行](figs/phase3_progress.png)

gap は v2 より拡大したが、forward 側の劣化(collateral damage、後述)も同時に進行している。v2 と v3 の訓練曲線比較から、v3 は深層に勾配を集中させることで forget 値が断続的にスパイクしやすく、retain CE もやや高水準で推移することが確認できる。

![Phase 3 v2 vs v3 訓練曲線](figs/wandb_phase3_compare.png)

## 主要な解釈

Phase 3 で得られた最も重要な知見は、**「DLM の右側文脈参照能力は、attention 経路と重み記憶経路という少なくとも二つの独立した経路で実現されており、勾配計算による L_rs 最小化は前者を選択的に削減できる」** ことである。これを次の二つの追加実験で示した。

### 並列生成能力の定量化 (parallel_gen_probe)

WikiText-103 validation 上で、各位置 *i* の元 token を、三つの context ablation 条件下で予測した:

- **full**: 位置 *i* のみ mask
- **left_only**: 位置 `[i, seq)` を mask(AR 等価の左側文脈のみ)
- **right_only**: 位置 `[0, i]` を mask(右側文脈のみ)

「右側文脈の寄与」を `log p_full − log p_left_only` で定義する。これは「位置 *i* で元 token を当てるとき、右側文脈が左側文脈にどれだけ追加情報を与えているか」を nats で測る指標である。

| ckpt | 右側文脈寄与 (nats) |
|---|---|
| Phase 2 (双方向 fact-tuned LLaDA) | **3.20** |
| Phase 3 v3 final (L_rs Unlearned) | **0.19** |

**右側文脈の寄与が約 94% 削減** された。これは「並列にトークンを生成する能力」の主要な物理基盤(右側にある masked 位置からの情報還流)が実質的に消失したことを示す。

![並列生成能力 (右側文脈寄与)](figs/parallel_gen_contrib.png)

### 記憶経路と attention 経路の切り分け (reversal_memorization_probe)

reverse query "[MASK] is paired with Y." について、入力を三つの ablation regime に分けて評価した。

- **full**: 通常入力(先頭 [MASK]、Y は visible)
- **no_Y**: 先頭 [MASK] は保持、Y span を mask で潰す
- **no_context**: 先頭 [MASK] のみ、右側全削除

`full − no_Y` の差分は「Y への attention 経路の寄与」を、`no_Y − no_context` の差分は「重み記憶経路の寄与」を、それぞれ定量化する。

| 経路 | Phase 2 寄与 | Phase 3 v3 寄与 | 削減率 |
|---|---|---|---|
| Y attention 経路 (full − no_Y) | +0.18 | +0.055 | **約 70%** |
| no_Y 水準 (重み記憶 + prior) | 0.18 | 0.185 | ほぼ変化なし |

first-token cand@1 では「Y を可視にしたときの上昇分」が attention 経路、「Y を mask しても残る予測精度」が重みベースの寄与に対応する。L_rs は前者を 70% 削減した一方、後者の水準はほぼ動いていない。重要なのは、span 単位の reverse cand@1 (主実験) は 0.705 と高水準にとどまる点で、これは「主体名 X 全体 (複数トークン) の重み記憶」が L_rs では削減されないことを意味する。**L_rs は attention 経路 (Y → X) を主に削減し、X の固有名詞自体の重み記憶は無傷で残す** という二重経路像が確認された。

![記憶経路と attention 経路の分解](figs/memorization_paths.png)

## 限界

本研究は、L_rs による「右側文脈 attention 経路」の選択的削減を示した一方、次の限界がある。

1. **完全な AR 復元には至らない**: Phase 3 v3 の +14 pp は AR 参照点 +33.5 pp の 4 割程度であり、forward−reverse 非対称性は部分的にしか再現されない。
2. **forward 側の collateral damage**: forward cand@1 も 0.995 → 0.845 に劣化。`log p_full` は Phase 2 の −0.86 から v3 では −5.91 まで下がり、left_only でさえ Phase 2 baseline (−4.06) を下回る。これは「左側文脈のみで予測する能力」自体が同時に弱まっていることを示す。
3. **重み記憶経路の Unlearning は未達**: L_rs は attention 経路を消すが、weights に焼き込まれた reverse 関係は残る。これは reversal curse の真の機構解明には別の介入が必要であることを示唆する。
4. **実応用との接続が不明瞭**: 「右側文脈参照を消す」操作の実用的価値が明確でない。本研究で得た「勾配計算による経路選択的削減」の知見は、後続研究の出発点として保持される。

## コード構成

```
src/unlearning_architecture/
    adapt.py            # AR → DLM 適応 (双方向 attention 化、MDM 損失)
    unlearn.py          # DLM → AR Unlearning (selector × forget_loss)
    eval.py             # 五軸評価の中核 (force_attention_mode、ar_ppl、dlm_nelbo)
    data.py             # WikiText-103 streaming loader
    native_dlm.py       # LLaDA 等のネイティブ DLM ロード補助

scripts/
    verify_adapt.py                    # 実装検証 harness
    make_reversal_v1_dataset.py        # data/reversal_v1/ の生成
    reverse_probe.py                   # 五軸の reverse probe
    relational_bidir_probe.py          # span PLL ベースの reversal_v1 評価
    parallel_gen_probe.py              # 並列生成能力の定量化 (右側文脈寄与)
    reversal_memorization_probe.py     # 記憶経路 vs attention 経路の切り分け
    phase25_llada_causal_probe.py      # 推論時 causal bias 注入実験
    phase3_path_viz.py                 # Phase 2.5/3 の results/*.jsonl 集約可視化
    wandb_curves_viz.py                # wandb から主要 run の訓練曲線を pull
    summary_viz.py                     # Phase 1 五軸評価サマリ図の生成
    scaling_viz.py                     # Phase 1 スケーリング図の生成

data/
    reversal_v1/        # 500 fictional entity pairs の forward/reverse 評価データ

figs/                   # README に埋め込まれた可視化結果
```

## 環境

- Python 3.x、依存管理は `uv`
- PyTorch `2.11.0+cu130`(NVIDIA Blackwell sm_120 対応)
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q ×7
- 主要モデル: Pythia 160M/410M/1B、LLaMA-3.1-8B、LLaDA-8B-Base
- 評価データ: WikiText-103 + 自作の合成データセット reversal_v1
- 実験管理: Weights & Biases (`udon0729-shizuoka-university/unlearning-architecture`)

## 再現方法

代表的なコマンド例を以下に示す。GPU は `CUDA_VISIBLE_DEVICES` で個別指定、長時間 run は `nohup` で起動する。

```bash
# 実装検証 (実装変更後に必ず実行)
CUDA_VISIBLE_DEVICES=4 uv run python scripts/verify_adapt.py

# Phase 1: AR → DLM 適応 (Pythia 系の例)
CUDA_VISIBLE_DEVICES=2 nohup uv run python -m unlearning_architecture.adapt \
  --model EleutherAI/pythia-410m \
  --steps 15000 --batch_size 16 --seq_len 512 --lr 3e-4 --warmup 600 \
  --use_lora --lora_r 16 --lora_alpha 32 \
  --out checkpoints/dlm-pythia410m-lora-15k \
  > logs/adapt_pythia410m_lora.log 2>&1 &

# Phase 1: DLM → AR Unlearning (B4 構成: lora_only + KL→AR)
CUDA_VISIBLE_DEVICES=5 nohup uv run python -m unlearning_architecture.unlearn \
  --adapted_ckpt checkpoints/dlm-pythia410m-lora-15k \
  --selector lora_only --ar_base EleutherAI/pythia-410m \
  --steps 2000 --warmup 200 --lr 5e-5 --alpha 0.5 --forget_loss kl_to_ar \
  --out checkpoints/unlearn-pythia410m-loraonly-kl-2k \
  > logs/unlearn_pythia410m_kl_2k.log 2>&1 &

# Phase 3 v3: LLaDA への L_rs Unlearning
CUDA_VISIBLE_DEVICES=6 nohup uv run python -m unlearning_architecture.unlearn \
  --adapted_ckpt checkpoints/dlm-llada-reversal-2k \
  --selector delta_topk --top_pct 0.10 --ar_base GSAI-ML/LLaDA-8B-Base \
  --forget_loss right_context_invariance --native_dlm \
  --lrs_layer_start 20 --lrs_n_positions 4 \
  --alpha 0.7 --lr 3e-5 --steps 1500 --warmup 100 \
  --batch_size 2 --seq_len 256 --use_8bit_optim \
  --out checkpoints/unlearn-llada-reversal-lrs-v3-1500 \
  > logs/unlearn_llada_v3.log 2>&1 &

# 五軸評価
CUDA_VISIBLE_DEVICES=4 uv run python -m unlearning_architecture.eval --ckpt <ckpt>
CUDA_VISIBLE_DEVICES=4 uv run python scripts/reverse_probe.py --ckpt <ckpt>

# Phase 3 の追加診断
CUDA_VISIBLE_DEVICES=4 uv run python scripts/parallel_gen_probe.py \
  --ckpt v2=checkpoints/dlm-llada-reversal-2k \
  --ckpt v3=checkpoints/unlearn-llada-reversal-lrs-v3-1500
CUDA_VISIBLE_DEVICES=4 uv run python scripts/reversal_memorization_probe.py \
  --ckpt v2=checkpoints/dlm-llada-reversal-2k \
  --ckpt v3=checkpoints/unlearn-llada-reversal-lrs-v3-1500

# 可視化 (results/*.jsonl と wandb から figs/*.png を生成)
uv run python scripts/phase3_path_viz.py
uv run python scripts/wandb_curves_viz.py
```


## 補足図

### 訓練曲線 (wandb)

各 Phase の主要 run について、loss / forget / retain / gradient norm の推移を `wandb` から pull した図を `figs/` に保存している。

- [`figs/wandb_p1_dlm_lora25k.png`](figs/wandb_p1_dlm_lora25k.png): Phase 1 の DLM 化訓練 (Pythia-160M、LoRA、25k step)
- [`figs/wandb_p1_unlearn_b4.png`](figs/wandb_p1_unlearn_b4.png): Phase 1 Unlearning B4 構成 (lora_only + KL→AR) の loss / forget / retain
- [`figs/wandb_p2_llada_sft.png`](figs/wandb_p2_llada_sft.png): Phase 2 LLaDA-8B-Base の reversal_v1 SFT (2k step)
- [`figs/wandb_p3v2_lrs.png`](figs/wandb_p3v2_lrs.png): Phase 3 v2 の L_rs Unlearning (uniform 33 層、step 860 付近で gnorm スパイクあり)
- [`figs/wandb_p3v3_deep.png`](figs/wandb_p3v3_deep.png): Phase 3 v3 の L_rs Unlearning (deep 層スライス、4 positions 平均)
- [`figs/wandb_phase3_compare.png`](figs/wandb_phase3_compare.png): v2 と v3 の forget / retain 直接比較

## 参考文献

[1] Berglund, L., Tong, M., Kaufmann, M., Balesni, M., Cooper Stickland, A., Korbak, T., & Evans, O. (2023). *The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"*. arXiv:2309.12288.

[2] Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). *What Does BERT Look At? An Analysis of BERT's Attention*. Proceedings of the 2019 ACL Workshop BlackboxNLP. arXiv:1906.04341.

[3] Gong, S., Agarwal, S., Zhang, Y., Ye, J., Zheng, L., Li, M., An, C., Zhao, P., Bi, W., Han, J., Peng, H., & Kong, L. (2024). *Scaling Diffusion Language Models via Adaptation from Autoregressive Models*. arXiv:2410.17891.

[4] Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., & Li, C. (2025). *Large Language Diffusion Models* (LLaDA). arXiv:2502.09992.

[5] Tenney, I., Das, D., & Pavlick, E. (2019). *BERT Rediscovers the Classical NLP Pipeline*. Proceedings of ACL 2019. arXiv:1905.05950.
