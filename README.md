# Anime lines parser

## install

```bash
uv sync
uv pip install --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu124
uv pip install --pre paddlepaddle-gpu --force-reinstall --index-url https://www.paddlepaddle.org.cn/packages/stable/cu123/
python .\parse_file_cli.py -i "file or dir path" -o "dir path"
```
