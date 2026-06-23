# 拡散言語モデルからARモデルへのアンラーニング

拡散言語モデル(DLM)をアーキテクチャ変更なしにアンラーニングだけでAR(自己回帰)モデルの挙動に戻せるか検討した研究リポジトリ。

## 研究の問い

Gong et al. 2024 [3] はARからDLMへの変換を追加学習やアーキテクチャ拡張で実現した。
本研究は逆方向を扱う。
DLM化されたモデルから、勾配計算と既存パラメータの選択的更新のみでARの挙動を復元できるか。

意義は二点ある。

- DLMの双方向参照能力を支える内部表現を分析する介入手段になる
- 同一の重みがARモードとDLMモードを行き来できるか検証し、両モードの差異がアーキテクチャ由来か表現由来かを切り分けられる

## 既存研究との対比

| 方向 | 代表研究 | 手段 |
|---|---|---|
| AR → DLM | Gong et al. 2024 [3] 等 | 追加学習、双方向アテンション化、MDM損失導入 |
| DLM → AR | 本研究 | 勾配計算によるアンラーニングのみ、アーキテクチャ非改変 |

## 実験設計

研究は段階的に進めた。
各段階は前段階の結果を踏まえて目的を更新している。

### ARベースラインと五軸評価

ARモデル(Pythia-160M, 410M, 1B)を双方向アテンション化とMDM損失でDLM化し、アンラーニングでAR挙動を復元できるか検証した。
同時にARモードとDLMモードを公平に評価する五軸評価フレームワークを確立した。

五軸評価の構成:

| 観点 | 計測内容 |
|---|---|
| AR性能 | 強制因果マスク下の next-token perplexity |
| DLM性能 | 強制双方向下のMDM NELBO |
| Reverse-context probe (因果) | 先頭5トークンをマスクし因果モードでtop-1復元精度 |
| Reverse-context probe (双方向) | 先頭5トークンをマスクし双方向モードでtop-1復元精度 |
| Reverse gap | 因果プローブと双方向プローブの差、DLM固有の双方向参照能力の指標 |

主結果(Pythia-160M, B4構成: lora_only + KL-to-AR + α=0.5):

| 指標 | AR baseline | DLM-adapted | Unlearned (B4) |
|---|---|---|---|
| AR Perplexity | 81.7 | 644 | 81.7 |
| DLM NELBO | 19.2 | 10.18 | 16.15 |
| Reverse causal acc | 0.16% | 1.09% | 1.09% |
| Reverse bidir acc | 0.66% | 6.72% | 1.05% |
| Reverse gap | +0.50 pp | +5.63 pp | -0.04 pp |

![Phase 1 五軸評価サマリ](figs/summary_viz.png)

Pythia-160Mを一旦DLM化した後、勾配計算のみでARベースラインと統計的に区別できない水準まで復元できた。
同じ現象はPythia-410M, Pythia-1B, LLaMA-3.1-8Bでも確認している。

![モデル規模スケーリング](figs/scaling_viz.png)

### ネイティブDLMへの拡張

前段階は自分でDLM化したARモデルをARに戻す実験で、対象が本来AR由来という点で予備的だった。
ここではDLMとして事前学習されたLLaDA-8B-Base (Nie et al. 2025 [4]) を対象にし、より厳密な検証に進んだ。

評価にはreversal_v1データセットを用いる。
500個の架空エンティティ対を生成し、順方向と逆方向のcand@1を測定する。
架空エンティティを使う理由は事前学習データに存在しない関係を新規に学習させ、データリークを排除するため。

SFT後の挙動:

| 評価 | LLaDA-8B-Base (SFT後) | Pythia-160M AR (同データSFT後) |
|---|---|---|
| forward cand@1 | 0.995 | 0.91 |
| reverse cand@1 | 0.980 | 0.16 |
| forward−reverse gap | +1.5 pp | +75 pp |

LLaDAは順方向と逆方向で対称な性能を示し、逆転呪い(Berglund et al. 2023 [1])を持たない。
ARであるPythia-160Mは典型的な逆方向の劣化を示す。
両者の順逆差は+73.5 ppで、アンラーニングが削減を試みる対象になる。

![LLaDA SFT訓練曲線](figs/wandb_p2_llada_sft.png)

### 推論時の因果バイアス注入

アンラーニングに進む前に、LLaDAの推論時アテンションバイアスとして因果マスクを強制した場合の挙動を確認した。
重みを変えずにAR風アテンションで動かす操作であり、アンラーニングの上限線として参照できる。

| モード | forward cand@1 | reverse cand@1 |
|---|---|---|
| natural (LLaDA既定) | 0.995 | 0.980 |
| 因果バイアス強制 | 0.240 | 0.155 |

結果はモード崩壊だった。
同一の重みのままアテンションだけを因果的にしてもAR化は達成されない。
LLaDAは推論時の双方向アテンションを前提に重みを学習しており、アテンション構造を切り替えるだけでは出力分布全体が崩壊する。
アンラーニングはアテンション構造を維持したまま重みをAR寄りに調整する必要がある。

### L_rs: 残差ストリーム右側文脈不変性損失

機構解釈の文献に基づき、DLMの双方向参照能力は各層の残差ストリームに右側文脈情報が統合されて流れる挙動として現れるという仮説を立てた。
この仮説から自己蒸留型の損失L_rsを導入した。

L_rsは、入力位置iにおける左側のみ可視の入力と全文脈可視の入力について、各層の隠れ状態が一致するように全文脈側を引き寄せる平均二乗誤差。

```
L_rs = Σ_layer || h_layer^full(i) − h_layer^left_masked(i) ||² / N
```

この損失をαL_rs + (1-α)·CEの形でAdamW8bitにより1000ステップ最小化する。
保持損失はWikiText-103上の因果的next-token CEで、言語能力の維持を担う。

結果(LLaDA-8B-Base, reversal_v1評価):

| 評価 | SFT後 | L_rs適用後 |
|---|---|---|
| forward cand@1 (eval_memory) | 0.995 | 0.865 |
| reverse cand@1 (eval_logic) | 0.980 | 0.755 |
| forward−reverse gap | +1.5 pp | +11.0 pp |

Pythia-160M ARの参照点(+73.5 pp)には届かないが、勾配計算のみで逆方向側を選択的に劣化させ順逆の非対称性を生じさせることに成功した。

### 層スライスとマルチポジション化

Transformerの階層別機能分業に関する先行研究(Tenney et al. 2019 [5], Clark et al. 2019 [2])は、構文処理が浅層に、意味的統合が深層に局在することを示している。
本研究の予備観察でもLLaDAの双方向情報統合は第20層から第32層の深層に集中する傾向があった。
前段階は33個の隠れ状態を一様平均しており、深層への勾配が約四割に希釈されていた可能性がある。
ここでは二点を改良した。

- 層スライス: L_rsをhidden_states[20:]のみに制限
- マルチポジション: バッチごとに4個の位置を一様サンプリングして平均。全文脈の順伝播は共有し、左側マスク順伝播のみ勾配なしで4回実行

結果:

| 評価 | SFT後 | L_rs全層 | L_rs深層スライス |
|---|---|---|---|
| forward cand@1 | 0.995 | 0.865 | 0.845 |
| reverse cand@1 | 0.980 | 0.755 | 0.705 |
| forward−reverse gap | +1.5 pp | +11.0 pp | +14.0 pp |

![モード崩壊とL_rsの進行](figs/phase3_progress.png)

順逆差は全層版より拡大したが、順方向側の劣化も同時に進行している。
訓練曲線を比較すると、深層スライス版は忘却損失が断続的にスパイクしやすく、保持CEもやや高水準で推移する。

![全層版と深層スライス版の訓練曲線](figs/wandb_phase3_compare.png)

## 二重経路の分析

DLMの右側文脈参照能力はアテンション経路と重み記憶経路という少なくとも二つの独立した経路で実現されている。
L_rsの最小化はアテンション経路を選択的に削減できる。
以下の二つの追加実験で確認した。

### 右側文脈寄与の定量化

WikiText-103の検証分割上で、各位置iの元トークンを三つの文脈除去条件下で予測した。

- full: 位置iのみマスク
- left_only: 位置[i, seq)をマスクし、AR等価の左側文脈のみ使用
- right_only: 位置[0, i]をマスクし、右側文脈のみ使用

右側文脈の寄与をlog p(full) − log p(left_only)で定義する。
右側文脈が左側文脈に対してどれだけ追加情報を与えているかをnats単位で測る指標になる。

| チェックポイント | 右側文脈寄与 (nats) |
|---|---|
| SFT後 (双方向事実微調整済みLLaDA) | 3.20 |
| L_rs適用後 | 0.19 |

右側文脈の寄与が約94%削減された。
並列トークン生成の物理基盤である右側マスク位置からの情報還流が実質的に消失したことを意味する。

![右側文脈寄与](figs/parallel_gen_contrib.png)

### 記憶経路とアテンション経路の分離

逆方向クエリ "[MASK] is paired with Y." について、入力を三つの除去条件に分けて評価した。

- full: 通常入力、先頭[MASK]とYの両方が可視
- no_Y: 先頭[MASK]は保持し、Yのスパンをマスクで置換
- no_context: 先頭[MASK]のみ、右側をすべて削除

fullとno_Yの差分はYへのアテンション経路の寄与、no_Yとno_contextの差分は重み記憶経路の寄与を定量化する。

| 経路 | SFT後の寄与 | L_rs適用後の寄与 | 削減率 |
|---|---|---|---|
| Yアテンション経路 (full − no_Y) | +0.18 | +0.055 | 約70% |
| no_Y水準 (重み記憶 + 事前分布) | 0.18 | 0.185 | ほぼ変化なし |

先頭トークンのcand@1ではYを可視にしたときの上昇分がアテンション経路、Yをマスクしても残る予測精度が重みベースの寄与に対応する。
L_rsはYへのアテンション経路を約70%削減した一方、重み記憶の水準はほぼ動いていない。
スパン単位の逆方向cand@1が0.705と高水準にとどまるのは、主体名X全体の重み記憶がL_rsでは削減されないことを意味する。
L_rsはアテンション経路(Y→X)を主に削減し、X固有名詞の重み記憶は無傷で残す。

![記憶経路とアテンション経路の分解](figs/memorization_paths.png)

## 限界

- AR復元は部分的にとどまる。+14 ppはAR参照点+73.5 ppの約二割
- 順方向側の副次的劣化が生じる。順方向cand@1が0.995から0.845に下がった。log p(full)はSFT後の-0.86からL_rs適用後は-5.91まで低下し、left_onlyでもSFT後ベースラインの-4.06を下回る
- 重み記憶経路のアンラーニングは未達。L_rsはアテンション経路を削減するが重みに焼き込まれた逆方向関係は残る。逆転呪いの機構解明には別の介入が必要
- 右側文脈参照を消す操作の実用的価値は不明瞭。勾配計算による経路選択的削減の知見は後続研究の出発点として保持する

## コード構成

```
src/unlearning_architecture/
    adapt.py            # AR→DLM適応 (双方向アテンション化、MDM損失)
    unlearn.py          # DLM→ARアンラーニング (selector × forget_loss)
    eval.py             # 五軸評価の中核
    data.py             # WikiText-103 / 合成関係データのストリーミングローダー
    native_dlm.py       # LLaDA等のネイティブDLMロード補助

scripts/
    verify_adapt.py                    # 実装検証ハーネス
    make_reversal_v1_dataset.py        # data/reversal_v1/の生成
    reverse_probe.py                   # 五軸の逆方向プローブ
    relational_bidir_probe.py          # スパン擬似対数尤度ベースのreversal_v1評価
    parallel_gen_probe.py              # 右側文脈寄与の定量化
    reversal_memorization_probe.py     # 記憶経路とアテンション経路の分離
    phase25_llada_causal_probe.py      # 推論時因果バイアス注入実験
    phase3_path_viz.py                 # Phase 2.5/3の結果集約可視化
    wandb_curves_viz.py                # wandbから主要実行の訓練曲線を取得
    summary_viz.py                     # Phase 1五軸評価サマリ図の生成
    scaling_viz.py                     # Phase 1スケーリング図の生成

data/
    reversal_v1/        # 500の架空エンティティ対からなる順方向/逆方向評価データ

figs/                   # 本文書に埋め込まれた可視化結果
```

## 環境

- Python ≥3.12、依存管理はuv
- PyTorch 2.11.0+cu130
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q × 7
- 主要モデル: Pythia 160M/410M/1B, LLaMA-3.1-8B, LLaDA-8B-Base
- 評価データ: WikiText-103 + reversal_v1
- 実験管理: Weights & Biases (udon0729-shizuoka-university/unlearning-architecture)

## 再現方法

代表的なコマンド例。
GPUはCUDA_VISIBLE_DEVICESで個別指定し、長時間の実行はnohupで起動する。

```bash
# 実装検証 (実装変更後に必ず実行)
CUDA_VISIBLE_DEVICES=4 uv run python scripts/verify_adapt.py

# AR→DLM適応 (Pythia系の例)
CUDA_VISIBLE_DEVICES=2 nohup uv run python -m unlearning_architecture.adapt \
  --model EleutherAI/pythia-410m \
  --steps 15000 --batch_size 16 --seq_len 512 --lr 3e-4 --warmup 600 \
  --use_lora --lora_r 16 --lora_alpha 32 \
  --out checkpoints/dlm-pythia410m-lora-15k \
  > logs/adapt_pythia410m_lora.log 2>&1 &

# DLM→ARアンラーニング (B4構成: lora_only + KL→AR)
CUDA_VISIBLE_DEVICES=5 nohup uv run python -m unlearning_architecture.unlearn \
  --adapted_ckpt checkpoints/dlm-pythia410m-lora-15k \
  --selector lora_only --ar_base EleutherAI/pythia-410m \
  --steps 2000 --warmup 200 --lr 5e-5 --alpha 0.5 --forget_loss kl_to_ar \
  --out checkpoints/unlearn-pythia410m-loraonly-kl-2k \
  > logs/unlearn_pythia410m_kl_2k.log 2>&1 &

# LLaDAへのL_rsアンラーニング (深層スライス版)
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

# 追加診断
CUDA_VISIBLE_DEVICES=4 uv run python scripts/parallel_gen_probe.py \
  --ckpt v2=checkpoints/dlm-llada-reversal-2k \
  --ckpt v3=checkpoints/unlearn-llada-reversal-lrs-v3-1500
CUDA_VISIBLE_DEVICES=4 uv run python scripts/reversal_memorization_probe.py \
  --ckpt v2=checkpoints/dlm-llada-reversal-2k \
  --ckpt v3=checkpoints/unlearn-llada-reversal-lrs-v3-1500

# 可視化 (results/*.jsonlとwandbからfigs/*.pngを生成)
uv run python scripts/phase3_path_viz.py
uv run python scripts/wandb_curves_viz.py
```

## 補足図

### wandbからの訓練曲線

各段階の主要な実行について損失/忘却損失/保持損失/勾配ノルムの推移をwandbから取得した図をfigs/に保存している。

- [figs/wandb_p1_dlm_lora25k.png](figs/wandb_p1_dlm_lora25k.png): DLM化訓練 (Pythia-160M, LoRA, 25000ステップ)
- [figs/wandb_p1_unlearn_b4.png](figs/wandb_p1_unlearn_b4.png): アンラーニングB4構成の損失推移
- [figs/wandb_p2_llada_sft.png](figs/wandb_p2_llada_sft.png): LLaDA-8B-BaseのSFT (2000ステップ)
- [figs/wandb_p3v2_lrs.png](figs/wandb_p3v2_lrs.png): L_rsアンラーニング全層版
- [figs/wandb_p3v3_deep.png](figs/wandb_p3v3_deep.png): L_rsアンラーニング深層スライス版
- [figs/wandb_phase3_compare.png](figs/wandb_phase3_compare.png): 全層版と深層スライス版の忘却損失/保持損失の直接比較

## 参考文献

[1] Berglund, L., Tong, M., Kaufmann, M., Balesni, M., Cooper Stickland, A., Korbak, T., & Evans, O. (2023). *The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"*. arXiv:2309.12288.

[2] Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). *What Does BERT Look At? An Analysis of BERT's Attention*. Proceedings of the 2019 ACL Workshop BlackboxNLP. arXiv:1906.04341.

[3] Gong, S., Agarwal, S., Zhang, Y., Ye, J., Zheng, L., Li, M., An, C., Zhao, P., Bi, W., Han, J., Peng, H., & Kong, L. (2024). *Scaling Diffusion Language Models via Adaptation from Autoregressive Models*. arXiv:2410.17891.

[4] Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., & Li, C. (2025). *Large Language Diffusion Models* (LLaDA). arXiv:2502.09992.

[5] Tenney, I., Das, D., & Pavlick, E. (2019). *BERT Rediscovers the Classical NLP Pipeline*. Proceedings of ACL 2019. arXiv:1905.05950.
