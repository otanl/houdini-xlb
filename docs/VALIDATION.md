# XLB 外部風検証プロトコル

この文書は、Houdini上の見た目や単一解像度の結果ではなく、XLB計算の数値誤差と
実験との差を分けて確認するための手順です。最初の対象には、建築分野で再利用しやすい
AIJ UWE Benchmark Case A（単体建物、幅:奥行:高さ = 1:1:2）を採用します。

## 参照データ

- データ: [AIJ UWE Benchmark Dataset – Case A (112)](https://doi.org/10.5281/zenodo.15050148)
- ライセンス: CC BY 4.0
- 模型: b = 0.08 m、H = 0.16 m
- Reynolds数: U_H と H に基づき約24,000
- 比較対象: 流入鉛直分布24点、建物周辺の平均速度3成分186点
- 原論文: [Meng & Hibi (1998)](https://doi.org/10.5359/jawe.1998.76_55)

実行時にCSVとREADMEだけをZenodoから
`artifacts/validation/aij_case_a/reference` へ取得し、公開MD5で検査します。
外部データをリポジトリへ複製しません。

AIJの関連する単体建物LES研究では、建物幅を20分割した格子で平均流と二次統計を
再現できたと報告されています。ただし、これは同じ1:1:2形状を用いたCase Hの研究で、
Case Aと同一の実験ではありません。本スイートではこの20 cells/bを「参照解像度へ
到達したか」の最低条件として使い、精度を事前保証する値にはしません。
[Kikumoto et al. (2021)](https://doi.org/10.1016/j.buildenv.2021.108021)

本スイートの計算領域は12.5H × 6H × 5Hとし、建物風上面から2H、風下面から
10Hを確保しています。これは参照CSVそのものに規定された領域ではなく、関連LESと
風洞断面を踏まえた明示的な初期設定です。上流距離、側方・上面境界、blockageの
感度を確認するまでは固定された正解条件として扱いません。

## まず計画だけ確認する

次のコマンドはGPU計算を開始しません。参照データを検査し、格子、step数、平均化時間、
概算population-buffer容量をJSONへ書きます。

```powershell
$env:PYTHONUTF8=1
.venv\Scripts\houdini-xlb-validate.exe --cells-per-b 8,12,16
```

既定のscreening格子は次の通りです。step数は一定値ではなく、
`steps × U_lattice / grid_x` が同じ流下回数になるよう解像度とともに増えます。

| cells/b | grid (x,y,z) | lattice cells | steps | 用途 |
|---:|---:|---:|---:|---|
| 8 | 200 × 96 × 80 | 1.54 M | 6,000 | 粗いscreening |
| 12 | 300 × 144 × 120 | 5.18 M | 9,000 | 中間格子 |
| 16 | 400 × 192 × 160 | 12.29 M | 12,000 | screening最細格子 |
| 20 | 500 × 240 × 200 | 24.00 M | 15,000 | 参照解像度の最低線 |

表のstep数は既定の1.5 flow-through、lattice wind 0.05の場合です。実際のGPU使用量は
2本のpopulation buffer以外の作業領域、境界mask、JAX変換、時間平均も含むため、
計画JSONの単純見積もりより大きくなります。

## GPUでscreeningを実行する

```powershell
$env:PYTHONUTF8=1
.venv\Scripts\houdini-xlb-validate.exe --run --cells-per-b 8,12,16 --time-check
```

既定の衝突モデルはKBCです。XLB組込みのSmagorinsky LES-BGKを感度試験する場合は
`--collision-model SmagorinskyLESBGK` を明示します。衝突モデルはcache keyとreportに含まれるため、
異なるモデルの場が混ざることはありません。検証caseは静止場からの長い立上げを避けるため、
指定流入の基準高さ速度で平衡初期化します。通常のHoudini profileは従来どおりKBC・静止初期化です。

同じ設定の3D平均速度場は
`artifacts/validation/aij_case_a/cache` にNPZキャッシュされます。中断後の再実行では
完了済み格子を読み、未完了の格子だけを計算します。`--force` で再計算できます。
最細格子では建物なしの空領域も1回解き、指定したべき乗則が床・側面・上面の影響を
受けた後、将来の建物中心位置でどの鉛直分布になるかを確認します。
`--skip-empty-domain-check` で省略できますが、その場合statusは `incomplete` です。

本評価に進む候補コマンドは次です。

```powershell
$env:PYTHONUTF8=1
.venv\Scripts\houdini-xlb-validate.exe --run --cells-per-b 10,15,20 --time-check --strict
```

出力:

- `outputs/validation/aij_case_a_report.json`: 条件、来歴、誤差、合否、制約
- `outputs/validation/aij_case_a_predictions.csv`: 全186測点の実験値と各格子の予測値
- `outputs/validation/aij_case_a_inlet_profile.csv`: 空領域の実到達流入とAIJ値
- validation cache: 3D平均速度 `(component,z,y,x)`

## 判定

暫定gateは次の5項目です。

1. 最細2格子の正規化平均Uの相対L2差が3%以下
2. 最細格子で平均時間を2倍にしたときの相対L2差が3%以下
3. 空領域の建物中心位置へ実際に届いた流入分布とAIJ値のrelative RMSEが10%以下
4. AIJ全測点に対する正規化平均Uの相対L2誤差が15%以下
5. 最細格子が20 cells/b以上

`--time-check` または空領域checkがなければstatusは `incomplete` です。指定した
べき乗則自体とAIJ値の近似誤差も別に記録しますが、実到達流入の代用にはしません。
全gateを通ってもstatusは `provisional_pass` であり、「工学的に検証済み」とは
表現しません。

## 2026-07-20 初期screening

まず計算可能性を確認するため、参照最低解像度20 cells/bより粗い6・8 cells/bで実行しました。
したがって、以下はモデル選定と失敗要因の診断値であり、精度検証結果ではありません。

| 衝突モデル | cells/b | U relative L2 | U correlation | 結果 |
|---|---:|---:|---:|---|
| KBC | 6 | 98.4% | -0.008 | 完走 |
| KBC | 8 | — | — | 非有限値となり停止 |
| Smagorinsky LES-BGK | 6 | 42.3% | 0.816 | 完走 |
| Smagorinsky LES-BGK | 8 | 43.7% | 0.846 | 完走 |

Smagorinskyの6→8 cells/bにおける測点Uの格子間driftは23.5%、8 cells/bの時間窓driftは
2.34%でした。空領域で建物中心へ実際に到達した流入分布のrelative RMSEは14.8%です。
時間窓だけは暫定3%基準内ですが、格子3%、流入10%、実験U 15%、最低20 cells/bのgateを
満たさないためstatusは `provisional_fail` です。

再現コマンド:

```powershell
$env:PYTHONUTF8=1
.venv\Scripts\houdini-xlb-validate.exe --run --cells-per-b 6,8 --collision-model SmagorinskyLESBGK --time-check
```

この結果から言えるのは、Smagorinskyがこの粗格子・高Re条件でKBCより安定かつ実験との傾向が
近いことまでです。衝突モデルの採用や街区設計値の信頼性は、流入・境界条件を改善し、20 cells/b
以上を含む格子列で再評価するまで確定しません。

## 現在わかっているモデル差

- 流入はAIJ平均速度からfitしたべき乗則だけで、実測された乱流変動を注入していません。
- 現在は床・上面・側面がno-slip、出口がextrapolationです。境界条件感度が必要です。
- 建物はuniform lattice上のvoxel/full-way bounce-backです。
- 通常profileの既定はKBCです。検証CLIではKBCとSmagorinsky LES-BGKを比較できますが、
  初期screeningだけで既定を変更しません。
- 今回比較するのは平均速度だけです。AIJの乱流統計は次のsynthetic-inflow段階で扱います。
- 独立ソルバー（OpenFOAM等）との同条件比較はまだありません。

したがって、この段階の目的は「XLBが正しい」と先に結論づけることではなく、
数値誤差、時間平均誤差、流入近似誤差、実験差のどれが支配的かを可視化することです。
街区最適化は、設計差がこの不確かさの少なくとも2倍あることを確認してから再開します。

## 次の段階

Case Aで支配的な誤差を特定した後、順に進めます。

1. 上面・側面境界、domain padding、衝突モデルの感度を安価な格子で切り分ける
2. AIJ流入時系列または整合するsynthetic turbulenceを導入する
3. 20 cells/b以上を含む格子列でCase Aを再実行する
4. street canyonの換気指標を検証する
5. Mokumitsu代表街区を同じ格子・時間プロトコルで再評価する
