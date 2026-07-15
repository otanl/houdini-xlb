# Houdini × XLB

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Houdiniで編集・アニメーションした建物や街区形状を、プロジェクト側の
Python 3.12環境でXLB（GPU Lattice-Boltzmann）へ送り、風速場をHoudiniへ
戻すための連携層です。停止中のフレームは自動解析し、タイムライン再生中は
ベイク済みキャッシュだけを即時表示します。
Houdini同梱PythonへWarp/XLBを直接インストールしないことが重要な設計判断です。

![Timeline-driven Houdini massing study with cached XLB fields](docs/assets/houdini_xlb_demo.gif)

## 対応環境

- Windows native（WSL不要）
- NVIDIA CUDA GPU
- Houdini Indie 20.5 / Python 3.11（20.5.684で確認済み）
- 外部Python 3.12
- uvとGit

CPUのみ、AMD GPU、Linux、macOSは現時点では未検証です。

## GitHubからcloneした場合のインストール

リポジトリ直下で実行します。

    uv venv .venv --python 3.12
    $env:PYTHONUTF8=1
    uv pip install --python .venv\Scripts\python.exe -e ".[xlb,dev]"
    .venv\Scripts\python.exe scripts\smoke_worker.py

XLBは動作確認済みcommitを指定し、warp-langは1.10.0に固定します。
Houdini側へ必要なのはnumpyと軽量クライアントだけで、GPU workerは上記環境です。

このモノレポ内で開発する場合は、ルートのPython 3.12環境を使用できます。

    uv pip install --python .venv\Scripts\python.exe -e "projects\houdini-xlb[xlb,dev]"

モノレポの既存環境では external/XLB のeditable installも利用できます。

## サンプルHIP

最小サンプルは [examples/houdini_xlb_demo.hip](examples/houdini_xlb_demo.hip) です。
`/obj/houdini_xlb`内のbuilding Box SOPにはフレーム1、12、24、36の
キーフレームがあり、各フレームが別の設計案になっています。

1. `xlb_confirmation` SOPを選択します（既定はdraft／Auto Analyze on Pause）。
2. タイムラインを停止してスクラブすると、0.75秒後に現在案を自動解析します。
3. `Bake Range`を押すと、1–36の未解析形状をバックグラウンドで順番に解析します。
4. 再生中はベイク済みフレームだけが表示され、XLBジョブは新規起動しません。

`Run Now (Current Frame)`は、自動解析を待たず現在案を直ちにキューへ入れる
任意の操作です。通常の形態スタディで毎回押す必要はありません。

HIP内のPython SOPには、生成時の外部Pythonとソースパスが保存されます。
別のPCへcloneした後は、次のコマンドでそのPC用に一度再生成してください。

    $HYTHON = "C:\Program Files\Side Effects Software\Houdini 20.5.xxx\bin\hython.exe"
    & $HYTHON houdini\build_demo_hip.py --run-xlb-smoke

Steam版などHoudiniの場所が異なる場合は、`$HYTHON`だけ変更します。
仮想環境をプロジェクト直下以外に置いた場合:

    & $HYTHON houdini\build_demo_hip.py --python-executable C:\path\to\python.exe

生成先の既定値は`examples/houdini_xlb_demo.hip`です。GUIで開くと、停止中の
現在フレームをデバウンス後に自動解析します。自動起動したくない場合は
`Auto Analyze on Pause`をオフにできます。HythonにはUIイベントループがないため、
`--run-xlb-smoke`だけは同期的に1案を検査します。

READMEのGIFも、実際のHoudini geometryとXLB結果からCLIで再生成できます。

    uv pip install --python .venv\Scripts\python.exe -e ".[xlb,demo]"
    .venv\Scripts\python.exe scripts\make_readme_gif.py --hython $HYTHON

キーフレーム4案を実際にdraft解析するため、未キャッシュ時は時間がかかります。中間PNGと
XLBキャッシュは`artifacts/readme-demo/`に保存され、Gitには含まれません。

## タイムラインとキャッシュ

このサンプルのフレーム番号は流体の時間刻みではなく、設計案のインデックスです。
キャッシュキーはフレーム番号ではなく、正規化高さマップ・解析設定・キャッシュ
versionのSHA-256です。同じ形状が複数フレームに現れる場合、解析結果を共有します。

- 停止／スクラブ: 最新の形状だけを残し、デバウンス後に非同期解析
- 再生: exact cache hitだけを表示し、未ベイク案は灰色の`not-baked`表示
- Bake Range: 未キャッシュのユニーク形状を単一workerで順次解析
- Cancel Bake: 実行中の1件は完了させ、残りの投入を停止

SOPのdetail attribute `xlb_status`、`xlb_job_state`、`xlb_frame`、
`xlb_bake_done`、`xlb_bake_total`で状態を確認できます。

## 処理の流れ

    Houdini timeline / geometry
        → connected-piece rasterization
        → height-map request
        → debounce / latest-only scheduler
        → persistent Python 3.12 worker
        → XLB / NVIDIA Warp
        → atomic NPZ cache
        → speed field + metadata
        → Houdini visualization

ワーカーを常駐させるので、Python・Warpの起動費用は最初の1回だけです。
同じ形状・設定はSHA-256キーでキャッシュされます。

## 解析プロファイル

| profile | grid | steps | 用途 |
|---|---:|---:|---|
| draft | 96 × 96 × 40 | 300 | 停止・スクラブ時の自動確認（既定） |
| preview | 128 × 128 × 48 | 600 | 対話的な比較 |
| quality | 256 × 256 × 64 | 2500 | 候補案の確認 |
| custom validation | CLIで明示 | CLIで明示 | 報告・再現実験 |

XLBの所要時間はGPU、格子、収束・平均化条件に依存します。したがって
draftであっても「再生速度でCFDを解く」のではなく、「停止中に非同期解析し、
再生時はキャッシュを読む」方式です。
滑らかな編集中表示には、既存のFNOプレビューを併用します。

## CLIで高さマップを解析

.npy、または高さマップを含む.npzを入力できます。

    $env:PYTHONUTF8=1
    .venv\Scripts\houdini-xlb.exe input_heightmap.npy --profile preview --cache artifacts\houdini\cache\xlb --out outputs\houdini_xlb_preview.npz

格子・step・Reynolds数・平均化条件は、各profileを基準にCLI optionで上書きできます。

結果には速度場、正規化高さマップ、解析設定、キャッシュキー、実行時間が
含まれます。

## Houdini Pythonから呼ぶ

Houdini側は軽量クライアントだけを読み込みます。入力ジオメトリには
connected pieceごとの整数point attribute class が必要です。

    from houdini_xlb.houdini import analyze_geometry

    node = hou.pwd()
    result = analyze_geometry(
        node.inputs()[0].geometry(),
        profile="preview",
        cache_dir=hou.text.expandString("$HIP/../artifacts/cache/xlb"),
    )
    speed = result.speed

XlbWorkerClient.analyze_async() を使えば、Houdini UIを止めずに要求できます。
表示更新はHoudiniのメインスレッドへ戻して行ってください。

最小デモHIPを再生成:

    & $HYTHON houdini\build_demo_hip.py

初回の実XLB接続まで検査する場合は末尾に --run-xlb-smoke を付けます。

生成された`examples/houdini_xlb_demo.hip`を開き、タイムラインを停止して
建物またはフレームを編集します。未キャッシュ形状は灰色表示になり、デバウンス後に
`queued → running → current`へ更新されます。先に`Bake Range`を完了すれば、
タイムラインをそのまま設計案比較として再生できます。

Houdiniを介さず、同じ常駐workerと実XLBを確認:

    $env:PYTHONUTF8=1
    .venv\Scripts\python.exe projects\houdini-xlb\scripts\smoke_worker.py

## 公開境界

このプロジェクトとして切り出す対象:

- src/houdini_xlb
- package内の高さマップXLB backend
- 最小のHoudiniサンプルHIP／Python SOP
- worker/cache/timeline scheduler/rasterizationのCPUテスト

FNOの学習実験、木密更新ロジック、OpenFOAM比較はこの配布物へ含めません。
このリポジトリはMIT Licenseです。XLB本体はApache-2.0です。
