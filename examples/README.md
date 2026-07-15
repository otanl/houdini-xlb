# Houdini sample

`houdini_xlb_demo.hip`は、次の最小ループを確認するためのサンプルです。

    building Box SOPs
        → merge / connectivity
        → xlb_confirmation
        → external Python 3.12 worker
        → XLB / Warp / CUDA
        → windspeed visualization

HIPのPython SOPには生成したPCのパスが保存されるため、clone後はリポジトリ
直下で次を実行し、ローカル環境用に再生成してください。

    & $HYTHON houdini\build_demo_hip.py --run-xlb-smoke

詳しい前提条件とセットアップは[プロジェクトREADME](../README.md)を参照してください。
