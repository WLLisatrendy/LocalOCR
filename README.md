# 本地截图识字（LocalOCR）

macOS 菜单栏截图 OCR 小工具：按 Control+Q 框选屏幕区域，识别中英文并自动复制。图片只在本机处理，免费，无需账号或 API Key。

- 识别引擎：PaddleOCR v4（通过 [RapidOCR](https://github.com/RapidAI/RapidOCR) + ONNX Runtime 在本机运行），常驻后台；不可用时自动退回 Apple Vision。结果窗口底部会显示本次使用的引擎。
- 保留原有换行、段落空行和中英文标点；表格、分栏按从左到右、从上到下的阅读顺序输出。
- 结果窗口提供「合并成一段」按钮，把被换行拆开的句子拼成一段，再点一下可恢复。
- Control+Option+Q 识别剪贴板中的图片。

## 使用

1. 打开应用，按 Control+Q（或点菜单栏「文」）框选区域，Esc 取消。
2. 首次使用需要在「系统设置 → 隐私与安全性 → 屏幕与系统音频录制」中允许本应用。
3. 识别结果自动复制，并显示在可编辑的结果窗口里，窗口需要手动关闭。
4. 如果 Control+Q 被其他工具占用，应用会改用 Control+Shift+Q；解绑后可在菜单栏「文 → 重新绑定 Control+Q」。

不保存截图历史，截图临时文件在识别后删除。

已知局限：无衬线字体里小写 l 和大写 I 字形几乎相同，任何 OCR 都难以区分；中英文之间的空格按字距推断，个别位置可能不准；手写、模糊的小字需要人工检查。

## 安装

需要 macOS 13+、Xcode Command Line Tools、[uv](https://github.com/astral-sh/uv)。项目需要放在 `~/LocalOCR`（应用从这里启动 PaddleOCR 识别进程）。

```sh
cd ~/LocalOCR

# 1. PaddleOCR 环境（首次识别时会自动下载模型）
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. 一次性创建本机签名证书。必须用固定证书签名：临时签名（--sign -）
#    每次重新编译都会让屏幕录制授权失效，需要重新授权。
D=$(mktemp -d) && P=$(uuidgen)
printf '[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n[dn]\nCN=LocalOCR Self Signed\n[ext]\nbasicConstraints=critical,CA:false\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=critical,codeSigning\n' > "$D/cert.cnf"
/usr/bin/openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -config "$D/cert.cnf" -keyout "$D/key.pem" -out "$D/cert.pem"
/usr/bin/openssl pkcs12 -export -inkey "$D/key.pem" -in "$D/cert.pem" -out "$D/id.p12" -passout pass:$P
security import "$D/id.p12" -k ~/Library/Keychains/login.keychain-db -P "$P" -T /usr/bin/codesign
rm -rf "$D"

# 3. 编译、签名
APP=~/Applications/本地截图识字.app
mkdir -p "$APP/Contents/MacOS"
xcrun swiftc -O -swift-version 5 -module-cache-path /tmp/local-ocr-module-cache -framework AppKit -framework Vision -framework Carbon -framework CoreServices LocalOCR.swift -o "$APP/Contents/MacOS/LocalOCR"
cp Info.plist "$APP/Contents/Info.plist"
codesign --force --sign 'LocalOCR Self Signed' "$APP"
```

## 测试

```sh
APP=~/Applications/本地截图识字.app/Contents/MacOS/LocalOCR
"$APP" --self-test                 # 合成中英文、数字、空白图和段落合并
"$APP" --recognize /path/image.png # 识别图片并输出到终端，不改剪贴板
LOCALOCR_ENGINE=vision "$APP" --recognize /path/image.png  # 强制使用 Apple Vision 对比
```
