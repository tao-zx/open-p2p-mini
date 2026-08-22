"""官方 Open P2P 的 protobuf schema（自动生成的 _pb2 代码，未改动）。

来源：https://github.com/elefant-ai/open-p2p 的 `elefant/data/proto/`
- `video_annotation_pb2.py`：`VideoAnnotation` / `FrameAnnotation` / `LowLevelAction` 等消息
- `shared_pb2.py`：`Vec2Int` / `Vec2Float` 等共享类型

复制进本仓库是为了让 `preprocess.py` 独立复现、不依赖仓库外的官方代码路径。
仅用于解析 `annotation.proto`，未做任何修改。
"""
