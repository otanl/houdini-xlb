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
初期案の95%以上を保つハード制約です。draft解析の大域最良解は評価11で見つかり、
広場の快適域内セルは39.6%から83.5%へ増え、通風帯は95.2%残ります。

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

HIPのSolver／Python SOPには生成したPCのパスが保存されるため、clone後は再生成して
ください。詳しい前提条件とセットアップは[プロジェクトREADME](../README.md)を
参照してください。
