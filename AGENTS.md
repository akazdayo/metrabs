# AGENTS ガイドライン（metrabs）

このファイルは本リポジトリ全体に適用されます。
用途は、エージェントが作業を始める際の実行手順とコーディング規約の整理です。

## リポジトリ概要

- 主言語は Python です。
- 依存管理は `uv`（`pyproject.toml` + `uv.lock`）を想定しています。
- Nix 用の `flake.nix` があり、`nix develop` / `nix run` が利用できます。
- 主要な実行スクリプトは `main.py` と `metrabs_pytorch/scripts/*` です。
- モデルは `metrabs_eff2l_384px_800k_28ds_pytorch` を参照します。
- `DATA_ROOT` 環境変数がデータ参照の基準になります。

## セットアップ（依存関係）

- `uv sync` で依存関係をインストールします。
- Python バージョンは `>=3.10,<3.12` です。
- Nix を使う場合は `nix develop` で開発シェルに入ります。
- GPU を使う実行は CUDA が必要です。

## 実行コマンド（ビルド/ラン）

- 通常実行: `uv run main.py`
- Nix から実行: `nix run .`（`metrabs-run` ラッパー）
- 画像デモ: `uv run python -m metrabs_pytorch.scripts.demo_image --model-dir <dir> --image-path <path-or-url>`
- TF 変換: `uv run python -m metrabs_pytorch.convert_model_from_tf ...`

## ビルド/リント/テスト

- ビルド工程はありません（スクリプト実行が主）。
- 公式のリント/フォーマット設定は存在しません。
- 公式のテストスイートは存在しません。
- 新規テストを追加する場合は `pytest` を前提にして構いません。
- 単体テスト例（新規追加時のみ）:
  - `python -m pytest path/to/test_file.py::TestClass::test_name`
  - `python -m pytest path/to/test_file.py -k test_name`
- リントを追加する場合は `ruff` または `black` を提案できますが、既存は未設定です。

## コーディング規約（全体）

- 既存の書式に合わせ、4スペースインデントを使います。
- 既存コードは型ヒントがほぼ無いため、新規も最小限にします。
- 1行の長さ制限は特に設定されていません。
- フォーマッタは未設定なので、既存ファイルの見た目に合わせます。
- 行末空白は避けます。

## インポート順

1. 標準ライブラリ
2. サードパーティ
3. ローカル（`metrabs_pytorch` など）

- グループ間は1行空けます。
- `import ... as ...` の短縮は既存に合わせます。
- `from x import y` を使う場合も整理して並べます。

## 命名規則

- 関数/変数: `snake_case`
- クラス: `PascalCase`
- 定数: `UPPER_SNAKE_CASE`
- モジュール: `lower_snake_case`
- Hydra 設定は `FLAGS` のような大文字名が使われます。

## 文字列・フォーマット

- 既存はシングルクォートとダブルクォートが混在しています。
- 追加箇所はファイル内の既存スタイルに合わせます。
- 長い行は無理に折り返さず、既存と同等の可読性を優先します。

## 型とテンソル取り扱い

- 推論時は `torch.inference_mode()` を使います。
- `torch.device(...)` でデバイスを明示します。
- ループ内の CPU/GPU 移動は最小限にします。
- 可能な限り `torch` のテンソル操作で完結させます。

## 設定とパス

- Hydra 設定取得は `metrabs_pytorch.util.get_config` を使います。
- パスは `Path` or `os.path` で統一し、既存ファイルの流儀に合わせます。
- `DATA_ROOT` を使う関数は相対パスを許容します。
- `model_dir` は `config.yaml`/`ckpt.pt`/`joint_info.npz` を前提にします。

## エラーハンドリング

- 既存は `RuntimeError` / `ValueError` / `FileNotFoundError` を利用します。
- 例外は具体的なメッセージを含めます。
- 想定外の例外は握りつぶさず再送出します。
- `try/except` は限定的に使用し、既存パターンを踏襲します。

## I/O と依存関係

- 外部モデルのダウンロードは `urllib.request` を使います。
- ファイル展開は `tarfile` を使用します。
- 画像読み込みは `torchvision.io` や `cv2` を使います。
- 依存追加が必要な場合は `pyproject.toml` に追記します。

## 実行時の注意

- 推論は GPU を前提にしている箇所が多いです。
- CPU 実行に切り替える場合は `torch.device('cpu')` を明示します。
- モデルロード前に `config.yaml` を読み込む必要があります。
- `joint_info.npz` / `skeleton_infos.pkl` などが揃っているか確認します。
- `DATA_ROOT` が未設定の場合は `data/` 直下を使う設計です。
- デモ画像/動画はパスまたは URL を受け付けます。
- 実行に時間がかかる処理は進捗ログを出します。
- 推論中の `cv2.imshow` は GUI 環境が必要です。
- Nix 実行時は `LD_LIBRARY_PATH` を設定済みの前提です。
- 依存追加時は CUDA/CPU 両方の互換を意識します。

## 変更時の注意

- 既存 API のシグネチャはできるだけ変えません。
- 速度劣化が疑われる変更は避けます。
- `simplepyutils` の既存フロー（`spu.argparse`）を尊重します。
- デバイスや dtype を暗黙に変えないよう注意します。

## デバッグ/ログ

- CLI からの実行は `print` を中心に簡潔に出力します。
- 例外時はスタックトレースが追えるように握りつぶしを避けます。
- デバッグ用の一時ログは恒久化前に削除します。

## コード追加時の推奨パターン

- CLI スクリプトは `metrabs_pytorch/scripts` 配下に置きます。
- `main.py` は軽量な起動スクリプトに留めます。
- 再利用可能な処理は `metrabs_pytorch` モジュールに切り出します。

## Cursor / Copilot ルール

- `.cursor/rules/` や `.cursorrules` は存在しません。
- `.github/copilot-instructions.md` は存在しません。

## 参考コマンドまとめ

- 依存同期: `uv sync`
- 通常実行: `uv run main.py`
- Nix 実行: `nix run .`
- 画像デモ: `uv run python -m metrabs_pytorch.scripts.demo_image --model-dir <dir> --image-path <path-or-url>`
- 単体テスト（追加時のみ）: `python -m pytest path/to/test_file.py::TestClass::test_name`

## 追記の目安

- 本ガイドラインに新たなツールを追加した場合は必ず更新します。
- 実行コマンドの差分が出たら最新版へ反映します。
- コーディング規約が変化した場合はここに明示します。
