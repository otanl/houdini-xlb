# Houdini sample

`houdini_xlb_demo.hip`は、次の最小ループを確認するためのサンプルです。

    animated building Box SOPs (frames 1–36)
        → merge / connectivity
        → xlb_confirmation
        → pause debounce / latest-only request
        → external Python 3.12 worker
        → XLB / Warp / CUDA
        → exact SHA-256 cache
        → windspeed visualization

停止中にフレームを移動すると自動解析されます。`Bake Range`で範囲を先読み
した後は、タイムライン再生中にキャッシュ済み風速場だけを即時表示します。
フレームは物理時間ではなく、形態案のインデックスです。

HIPのPython SOPには生成したPCのパスが保存されるため、clone後はリポジトリ
直下で次を実行し、ローカル環境用に再生成してください。

    & $HYTHON houdini\build_demo_hip.py --run-xlb-smoke

詳しい前提条件とセットアップは[プロジェクトREADME](../README.md)を参照してください。
