# Houdini sample

`houdini_xlb_demo.hip`は、中央広場の風環境と中央通風帯を同時に扱う、
制約付き配置最適化の最小サンプルです。

    2 movable + 2 fixed building Box SOPs
        → merge / connectivity
        → xlb_init
        → xlb_solver (Prev_Frame → xlb_step → OUT)
        → xlb_result
        → study_display (plaza / ventilation measurement guides)
        → pause debounce / latest-only request
        → external Python 3.12 worker
        → XLB / Warp / CUDA
        → exact SHA-256 cache
        → windspeed visualization

青い2棟の中心位置4変数を各2水準とし、固定容積・敷地内・非重複の16候補を
すべて実XLBで評価します。目的は、黄色の広場で
`0.55 <= U/Uin <= 1.20`を外れるセルを最小化することです。シアンの中央通風帯は、
初期案の95%以上を保つハード制約です。study解析の大域最良解は評価6で見つかり、
広場の快適域内セルは30.8%から100.0%へ増え、通風帯平均は初期案の190.2%になります。解析は96 × 96 × 38、2400 step、
結果高さ1.5 mです。これはUIと制約付き探索を示す再現可能デモであり、
格子独立性や外部風の境界条件を検証した工学解析ではありません。

タイムラインのフレーム1、31、61、91、120は、評価1、3、5、11、16時点の
best-so-farです。配置はconstant keyで切り替わり、フレームは物理時間でも
建物の移動アニメーションでもありません。青が可動棟、灰色が固定棟です。

最適化を再実行してから、ローカル環境用のHIPを生成できます。

    $env:PYTHONUTF8=1
    .venv\Scripts\python.exe scripts\optimize_demo.py
    & $HYTHON houdini\build_demo_hip.py --run-xlb-smoke

`xlb_solver`を選択し、停止中にフレームを移動すると未キャッシュ案を自動解析します。
`Bake Range`後の再生ではキャッシュ済み風速場だけを表示します。最適化の全評価は
`houdini_xlb_demo_optimization.json`、タイムラインには解の改善履歴だけを保存します。

HIPは生成したPCの絶対パスを含みません。標準cloneではsourceと`.venv`をHIP位置から
自動検出し、標準外の配置は`HOUDINI_XLB_SOURCE`、`HOUDINI_XLB_PYTHON`、
`HOUDINI_XLB_CACHE`で指定します。詳しい前提条件は
[プロジェクトREADME](../README.md)を参照してください。
