import AppKit
import Vision
import Carbon
import CoreServices

enum OCR {
    private struct Line {
        var text: String
        let box: CGRect  // pixels, origin bottom-left
    }

    // PaddleOCR reads Chinese, letter case, and digits more accurately than
    // Apple Vision in testing; Vision remains the fallback when the helper is
    // unavailable. `file` is the image on disk, if there is one already.
    static func recognize(_ image: CGImage, file: URL? = nil) throws -> (text: String, engine: String) {
        if ProcessInfo.processInfo.environment["LOCALOCR_ENGINE"] != "vision",
           let lines = PaddleEngine.shared.recognize(image, file: file) {
            let converted = lines.map { Line(text: normalizePunctuation($0.text),
                                             box: CGRect(x: $0.box.minX, y: CGFloat(image.height) - $0.box.maxY,
                                                         width: $0.box.width, height: $0.box.height)) }
            return (layout(converted), "PaddleOCR")
        }
        return (try recognizeWithVision(image), "Apple Vision")
    }

    // Recognize with the Chinese+English model, then re-read lines that contain
    // no Chinese with the English-only model, which keeps ASCII punctuation and
    // reads IDs/code better. Automatic language detection is off on purpose: it
    // misreads mixed Chinese/English lines as other languages.
    private static func recognizeWithVision(_ image: CGImage) throws -> String {
        let lines = try run(image, languages: ["zh-Hans", "en-US"], correction: true).map { line -> Line in
            var line = line
            if !hasHan(line.text), let crop = crop(image, line.box),
               let english = try? run(crop, languages: ["en-US"], correction: false),
               english.count == 1, !english[0].text.isEmpty, !hasHan(english[0].text),
               keepsSymbols(of: line.text, in: english[0].text) {
                line.text = english[0].text
            }
            line.text = normalizePunctuation(fixCapitalI(line.text))
            return line
        }
        return layout(recoverLinePrefixes(image, lines))
    }

    // Vision sometimes drops a short prefix such as "3." when the rest of the
    // line starts further right than its neighbours. Re-read the empty strip on
    // the left of such lines and put anything found back in front.
    private static let listMarker = try! NSRegularExpression(pattern: "^((?:[0-9]{1,3}|[A-Za-z]|[一二三四五六七八九十]{1,3})[.)、）]|[(（][0-9]{1,3}[)）]|[•·\\-*])")

    private static func padOnBackground(_ image: CGImage, margin: Int) -> CGImage? {
        let space = CGColorSpaceCreateDeviceRGB(), info = CGImageAlphaInfo.premultipliedLast.rawValue
        var pixel = [UInt8](repeating: 255, count: 4)
        guard let probe = CGContext(data: &pixel, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4, space: space, bitmapInfo: info),
              let canvas = CGContext(data: nil, width: image.width + margin * 2, height: image.height + margin * 2,
                                     bitsPerComponent: 8, bytesPerRow: 0, space: space, bitmapInfo: info) else { return nil }
        // The top-left pixel of the strip is background (the strip starts left of all text).
        probe.draw(image, in: CGRect(x: 0, y: 1 - image.height, width: image.width, height: image.height))
        canvas.setFillColor(red: CGFloat(pixel[0]) / 255, green: CGFloat(pixel[1]) / 255, blue: CGFloat(pixel[2]) / 255, alpha: 1)
        canvas.fill(CGRect(x: 0, y: 0, width: canvas.width, height: canvas.height))
        canvas.draw(image, in: CGRect(x: margin, y: margin, width: image.width, height: image.height))
        return canvas.makeImage()
    }

    private static func recoverLinePrefixes(_ image: CGImage, _ lines: [Line]) -> [Line] {
        guard lines.count >= 2 else { return lines }
        let leftEdge = lines.map(\.box.minX).min()!
        return lines.map { line in
            let width = line.box.minX - leftEdge
            guard width > line.box.height * 0.6,
                  !lines.contains(where: { $0.box.maxX <= line.box.minX && abs($0.box.midY - line.box.midY) < line.box.height * 0.5 }) else { return line }
            // Re-read the blank strip together with the first few characters
            // of the line; Vision ignores a lone tiny strip but reads it with
            // context. A margin of background keeps the crop away from edges.
            let strip = CGRect(x: leftEdge - line.box.height * 0.3, y: line.box.minY,
                               width: width + line.box.height * 3.3, height: line.box.height)
                .intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height))
            guard !strip.isEmpty, let first = line.text.first,
                  let crop = image.cropping(to: CGRect(x: strip.minX, y: CGFloat(image.height) - strip.maxY,
                                                       width: strip.width, height: strip.height).integral),
                  let padded = padOnBackground(crop, margin: 40),
                  let found = try? run(padded, languages: ["zh-Hans", "en-US"], correction: false).first else { return line }
            // Accept a list marker only, and only if the line's own text follows it.
            let text = normalizePunctuation(found.text)
            guard let match = listMarker.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
                  let range = Range(match.range(at: 1), in: text),
                  text[range.upperBound...].trimmingCharacters(in: .whitespaces).first == first else { return line }
            return Line(text: String(text[range]) + " " + line.text,
                        box: line.box.union(CGRect(x: leftEdge, y: line.box.minY, width: 1, height: line.box.height)))
        }
    }

    private static func run(_ image: CGImage, languages: [String], correction: Bool) throws -> [Line] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = languages
        request.usesLanguageCorrection = correction
        request.automaticallyDetectsLanguage = false
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        let w = CGFloat(image.width), h = CGFloat(image.height)
        return (request.results ?? []).compactMap { observation in
            guard let text = observation.topCandidates(1).first?.string.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else { return nil }
            let b = observation.boundingBox
            return Line(text: text, box: CGRect(x: b.minX * w, y: b.minY * h, width: b.width * w, height: b.height * h))
        }
    }

    private static func crop(_ image: CGImage, _ box: CGRect) -> CGImage? {
        let padded = box.insetBy(dx: -box.height * 0.5, dy: -box.height * 0.35)
            .intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard !padded.isEmpty else { return nil }
        // CGImage cropping uses a top-left origin.
        return image.cropping(to: CGRect(x: padded.minX, y: CGFloat(image.height) - padded.maxY,
                                         width: padded.width, height: padded.height).integral)
    }

    // Vision returns text blocks in no particular reading order. Group blocks
    // into visual rows (top to bottom, then left to right), keep a blank line
    // where the original has a paragraph gap, and restore leading indentation
    // of code-like lines.
    private static func layout(_ lines: [Line]) -> String {
        var rows: [[Line]] = []
        for line in lines.sorted(by: { $0.box.midY > $1.box.midY }) {
            if let last = rows.indices.last {
                let top = rows[last].map(\.box.maxY).max()!, bottom = rows[last].map(\.box.minY).min()!
                let overlap = min(top, line.box.maxY) - max(bottom, line.box.minY)
                if overlap > 0.5 * min(top - bottom, line.box.height) { rows[last].append(line); continue }
            }
            rows.append([line])
        }
        guard !rows.isEmpty else { return "" }
        let tops = rows.map { $0.map(\.box.maxY).max()! }, bottoms = rows.map { $0.map(\.box.minY).min()! }
        let heights = zip(tops, bottoms).map { $0 - $1 }.sorted()
        let lineHeight = heights[heights.count / 2]
        let gaps = (1..<rows.count).map { bottoms[$0 - 1] - tops[$0] }
        let normalGap = gaps.count >= 2 ? gaps.sorted()[gaps.count / 4] : lineHeight * 0.4
        let leftEdge = rows.flatMap { $0 }.map(\.box.minX).min()!
        let latin = rows.flatMap { $0 }.filter { !hasHan($0.text) && $0.text.count >= 8 }
        let charWidth = latin.isEmpty ? 0 : latin.map { $0.box.width / CGFloat($0.text.count) }.sorted()[latin.count / 2]

        var output: [String] = []
        for (index, row) in rows.enumerated() {
            if index > 0, gaps[index - 1] > normalGap + lineHeight * 0.8 { output.append("") }
            var text = row.sorted { $0.box.minX < $1.box.minX }.map(\.text).joined(separator: " ")
            if charWidth > 0, !hasHan(text) {
                let indent = Int(((row.map(\.box.minX).min()! - leftEdge) / charWidth).rounded())
                if (2...24).contains(indent) { text = String(repeating: " ", count: indent) + text }
            }
            output.append(text)
        }
        return output.joined(separator: "\n")
    }

    // The English model reads symbols such as ¥ as other characters; keep the
    // Chinese model's line when the English re-read loses one of them.
    private static func keepsSymbols(of original: String, in english: String) -> Bool {
        let symbols = original.replacingOccurrences(of: "￥", with: "¥").unicodeScalars.filter {
            (0xA0...0xFF).contains($0.value) || (0x20A0...0x20CF).contains($0.value)
        }
        return symbols.allSatisfy { english.unicodeScalars.contains($0) }
    }

    static func hasHan(_ text: String) -> Bool {
        text.unicodeScalars.contains { (0x3400...0x9FFF).contains($0.value) }
    }

    private static func isChineseContext(_ c: Character) -> Bool {
        guard let v = c.unicodeScalars.first?.value else { return false }
        return (0x3400...0x9FFF).contains(v) || (0x3000...0x303F).contains(v) || "“”‘’—…·、。《》「」".contains(c)
    }

    // "APl" -> "API": a lowercase l inside an otherwise upper-case word is almost always I.
    private static let capitalIPattern = try! NSRegularExpression(pattern: "(?<![A-Za-z])([A-Z]{2,}l[A-Z]*|[A-Z]+l[A-Z]+)(?![a-z])")

    // "ControL" -> "Control": a capital L ending an otherwise lower-case word.
    private static let trailingCapitalL = try! NSRegularExpression(pattern: "(?<=[A-Za-z][a-z]{2})L(?![A-Za-z])")

    private static func fixCapitalI(_ text: String) -> String {
        var result = trailingCapitalL.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "l")
        let text = result
        for match in capitalIPattern.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            let range = Range(match.range, in: text)!
            result.replaceSubrange(range, with: text[range].replacingOccurrences(of: "l", with: "I"))
        }
        return result
    }

    private static let alwaysHalfWidth: [Character: Character] = [
        "＄": "$", "％": "%", "＋": "+", "＝": "=", "＜": "<", "＞": ">", "＠": "@", "＃": "#",
        "＆": "&", "＊": "*", "｜": "|", "＼": "\\", "／": "/", "～": "~", "＾": "^", "＿": "_", "｀": "`", "⋯": "…", "．": "."
    ]
    private static let contextHalfWidth: [Character: Character] = [
        "（": "(", "）": ")", "：": ":", "；": ";", "｛": "{", "｝": "}", "！": "!", "？": "?",
        "，": ",", "［": "[", "］": "]", "＂": "\"", "＇": "'"
    ]

    // The Chinese model tends to emit full-width punctuation everywhere. Keep it
    // next to Chinese text and switch back to ASCII inside English/code.
    private static func normalizePunctuation(_ text: String) -> String {
        var chars = Array(text)
        let hasChinese = hasHan(text)
        func neighbor(_ i: Int, _ step: Int) -> Character? {
            var j = i + step
            while chars.indices.contains(j) { if chars[j] != " " { return chars[j] }; j += step }
            return nil
        }
        for i in chars.indices {
            let c = chars[i]
            if let half = alwaysHalfWidth[c] { chars[i] = half; continue }
            if let v = c.unicodeScalars.first?.value, c.unicodeScalars.count == 1,
               (0xFF10...0xFF19).contains(v) || (0xFF21...0xFF3A).contains(v) || (0xFF41...0xFF5A).contains(v) {
                chars[i] = Character(UnicodeScalar(v - 0xFEE0)!); continue
            }
            // Chinese sentences keep full-width commas even after English
            // words, so in Chinese lines only brackets switch back to ASCII.
            if let half = contextHalfWidth[c], !hasChinese || "（）｛｝［］".contains(c) {
                let left = neighbor(i, -1), right = neighbor(i, 1)
                if (left != nil || right != nil), !(left.map(isChineseContext) ?? false), !(right.map(isChineseContext) ?? false) {
                    chars[i] = half
                }
            }
            // In Chinese text, ASCII punctuation right after (or, for opening
            // brackets, right before) a Chinese character is a misread full-width mark.
            if hasChinese, let full = fullWidth[chars[i]] {
                let before = i > 0 ? chars[i - 1] : nil, after = i + 1 < chars.count ? chars[i + 1] : nil
                let hanBefore = before.map { hasHan(String($0)) } ?? false, hanAfter = after.map { hasHan(String($0)) } ?? false
                if hanBefore || ("(".contains(chars[i]) && hanAfter) { chars[i] = full }
            }
        }
        return String(chars)
    }

    private static let fullWidth: [Character: Character] = [
        ",": "，", ":": "：", ";": "；", "!": "！", "?": "？", "(": "（", ")": "）"
    ]

    // Join wrapped lines into one paragraph: Chinese joins directly, English
    // words get a single space, and a trailing hyphen inside a word is removed.
    static func mergeIntoParagraph(_ text: String) -> String {
        let lines = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        var result = ""
        for line in lines {
            guard let last = result.last, let first = line.first else { result = line; continue }
            let wide = { (c: Character) in isChineseContext(c) || (c.unicodeScalars.first.map { (0xFF00...0xFFEF).contains($0.value) } ?? false) }
            if last == "-", first.isLetter, result.dropLast().last?.isLetter == true {
                result.removeLast()
            } else if !wide(last) && !wide(first) {
                result += " "
            }
            result += line
        }
        return result
    }

    static func recognizeFile(_ url: URL) throws -> (text: String, engine: String) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NSError(domain: "LocalOCR", code: 1, userInfo: [NSLocalizedDescriptionKey: "无法读取图片"])
        }
        return try recognize(image, file: url)
    }
}

// Long-running PaddleOCR helper (paddle_ocr_helper.py in the project's .venv).
// Requests are serialized; a missing, failing, or slow helper returns nil so the
// caller falls back to Apple Vision.
final class PaddleEngine {
    struct Result { let text: String; let box: CGRect }  // pixels, origin top-left

    static let shared = PaddleEngine()
    static let root = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("LocalOCR")

    private let queue = DispatchQueue(label: "local-ocr.paddle")
    private let lock = NSLock()
    private let lineReady = DispatchSemaphore(value: 0)
    private var process: Process?
    private var input: FileHandle?
    private var buffer = Data()
    private var pending: [String] = []
    private var lastFailure: Date?

    func warmUp() { queue.async { _ = self.startIfNeeded() } }

    func stop() { queue.sync { terminate() } }

    func recognize(_ image: CGImage, file: URL?) -> [Result]? {
        queue.sync {
            guard startIfNeeded() else { return nil }
            var path = file
            var temporary: URL?
            if path == nil {
                let url = FileManager.default.temporaryDirectory.appendingPathComponent("local-ocr-\(UUID().uuidString).png")
                guard let data = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]),
                      (try? data.write(to: url)) != nil else { return nil }
                path = url
                temporary = url
            }
            defer { if let temporary { try? FileManager.default.removeItem(at: temporary) } }
            guard let input, let request = (path!.path + "\n").data(using: .utf8) else { return nil }
            do { try input.write(contentsOf: request) } catch { fail(); return nil }
            guard let reply = nextLine(timeout: 30),
                  let json = try? JSONSerialization.jsonObject(with: Data(reply.utf8)) as? [String: Any],
                  let lines = json["lines"] as? [[String: Any]] else { fail(); return nil }
            return lines.compactMap { line in
                guard let text = line["text"] as? String, let b = line["box"] as? [Double], b.count == 4 else { return nil }
                return Result(text: text, box: CGRect(x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1]))
            }
        }
    }

    private func startIfNeeded() -> Bool {
        if let process, process.isRunning { return true }
        // After a failure, wait before paying the start-up cost again.
        if let lastFailure, Date().timeIntervalSince(lastFailure) < 60 { return false }
        let python = Self.root.appendingPathComponent(".venv/bin/python")
        let helper = Self.root.appendingPathComponent("paddle_ocr_helper.py")
        guard FileManager.default.isExecutableFile(atPath: python.path),
              FileManager.default.fileExists(atPath: helper.path) else { lastFailure = Date(); return false }
        let process = Process()
        process.executableURL = python
        process.arguments = ["-I", helper.path]
        let stdin = Pipe(), stdout = Pipe()
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = FileHandle.nullDevice
        lock.lock(); buffer = Data(); pending = []; lock.unlock()
        while lineReady.wait(timeout: .now()) == .success {}
        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let self else { return }
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil; return }
            self.lock.lock()
            self.buffer.append(data)
            while let newline = self.buffer.firstIndex(of: 0x0A) {
                self.pending.append(String(decoding: self.buffer[..<newline], as: UTF8.self))
                self.buffer.removeSubrange(...newline)
                self.lineReady.signal()
            }
            self.lock.unlock()
        }
        do { try process.run() } catch { lastFailure = Date(); return false }
        self.process = process
        self.input = stdin.fileHandleForWriting
        guard let ready = nextLine(timeout: 60), ready.contains("\"ready\"") else { fail(); return false }
        lastFailure = nil
        return true
    }

    private func nextLine(timeout: TimeInterval) -> String? {
        guard lineReady.wait(timeout: .now() + timeout) == .success else { return nil }
        lock.lock(); defer { lock.unlock() }
        return pending.isEmpty ? nil : pending.removeFirst()
    }

    // A late reply must never be read as the answer to the next request.
    private func fail() {
        terminate()
        lastFailure = Date()
    }

    private func terminate() {
        process?.terminate()
        process = nil
        input = nil
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var window: NSWindow!
    var textView: NSTextView!
    var statusLabel: NSTextField!
    var captureButton: NSButton!
    var mergeButton: NSButton!
    var unmergedText: String?
    var captureMenu: NSMenuItem!
    var hotKey: EventHotKeyRef?
    var clipboardHotKey: EventHotKeyRef?
    var eventHandler: EventHandlerRef?
    var shortcut = "Control+Q"
    var shortcutNote = ""
    var busy = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        PaddleEngine.shared.warmUp()
        configureHotKey()
        configureMenu()
        configureWindow()
        showWindow()
    }

    func configureHotKey() {
        var event = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let installed = InstallEventHandler(GetApplicationEventTarget(), { _, event, pointer in
            guard let pointer, let event else { return OSStatus(eventNotHandledErr) }
            var identifier = EventHotKeyID()
            let result = GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID), nil, MemoryLayout<EventHotKeyID>.size, nil, &identifier)
            guard result == noErr, identifier.signature == 0x4C4F4352 else { return OSStatus(eventNotHandledErr) }
            let app = Unmanaged<AppDelegate>.fromOpaque(pointer).takeUnretainedValue()
            DispatchQueue.main.async { identifier.id == 2 ? app.recognizeClipboard() : app.capture() }
            return noErr
        }, 1, &event, Unmanaged.passUnretained(self).toOpaque(), &eventHandler)
        guard installed == noErr else {
            shortcut = "菜单栏启动"
            shortcutNote = "快捷键注册失败，请点击菜单栏的「文」图标截图。"
            return
        }
        let identifier = EventHotKeyID(signature: 0x4C4F4352, id: 1)
        var result = RegisterEventHotKey(UInt32(kVK_ANSI_Q), UInt32(controlKey), identifier, GetApplicationEventTarget(), 0, &hotKey)
        if result != noErr {
            shortcut = "Control+Shift+Q"
            result = RegisterEventHotKey(UInt32(kVK_ANSI_Q), UInt32(controlKey | shiftKey), identifier, GetApplicationEventTarget(), 0, &hotKey)
            shortcutNote = "Control+Q 已被占用，暂用 Control+Shift+Q。取消 uTools 的旧绑定后，点击菜单栏「重新绑定 Control+Q」。"
        }
        // Control+Option+Q recognizes an image already in the clipboard.
        let clipboardIdentifier = EventHotKeyID(signature: 0x4C4F4352, id: 2)
        _ = RegisterEventHotKey(UInt32(kVK_ANSI_Q), UInt32(controlKey | optionKey), clipboardIdentifier, GetApplicationEventTarget(), 0, &clipboardHotKey)
        if result != noErr {
            shortcut = "菜单栏启动"
            shortcutNote = "两个快捷键均不可用，请点击菜单栏的「文」图标截图。"
        }
    }

    func configureMenu() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "文"
        statusItem.button?.toolTip = "本地截图识字 · \(shortcut)"
        let menu = NSMenu()
        captureMenu = NSMenuItem(title: "截图识字（\(shortcut)）", action: #selector(capture), keyEquivalent: "")
        captureMenu.target = self
        menu.addItem(captureMenu)
        let clipboardItem = NSMenuItem(title: "识别剪贴板图片（Control+Option+Q）", action: #selector(recognizeClipboard), keyEquivalent: "")
        clipboardItem.target = self
        menu.addItem(clipboardItem)
        let startup = NSMenuItem(title: "开机启动", action: #selector(toggleLaunchAtLogin), keyEquivalent: "")
        startup.target = self
        startup.state = launchAtLoginEnabled ? .on : .off
        menu.addItem(startup)
        for (title, action) in [("显示识别结果", #selector(showWindow)), ("重新绑定 Control+Q", #selector(rebind)), ("屏幕录制权限设置…", #selector(openScreenSettings))] {
            let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
            item.target = self
            menu.addItem(item)
        }
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "退出本地截图识字", action: #selector(quit), keyEquivalent: "")
        quit.target = self
        menu.addItem(quit)
        statusItem.menu = menu
    }

    func configureWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 700, height: 470), styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "本地截图识字"
        window.isReleasedWhenClosed = false
        window.minSize = NSSize(width: 560, height: 370)
        window.center()
        let content = window.contentView!
        captureButton = NSButton(title: "截图识字 · \(shortcut)", target: self, action: #selector(capture))
        captureButton.bezelStyle = .rounded
        let copy = NSButton(title: "复制文字", target: self, action: #selector(copyText))
        copy.bezelStyle = .rounded
        mergeButton = NSButton(title: "合并成一段", target: self, action: #selector(toggleMerge))
        mergeButton.bezelStyle = .rounded
        let buttons = NSStackView(views: [captureButton, copy, mergeButton])
        buttons.orientation = .horizontal
        buttons.spacing = 10
        let intro = NSTextField(wrappingLabelWithString: "框选后自动识别并复制。中英文均可，图片在本机处理，无需账户。按 Esc 取消截图。")
        intro.textColor = .secondaryLabelColor
        statusLabel = NSTextField(wrappingLabelWithString: shortcutNote.isEmpty ? "点击截图开始；首次使用需要允许屏幕录制。关闭窗口后，仍可用快捷键或菜单栏「文」。" : shortcutNote)
        statusLabel.textColor = .secondaryLabelColor
        // scrollableTextView() wires the text view to the clip view's width; a
        // hand-sized text view drifted sideways and clipped the first characters.
        let scroll = NSTextView.scrollableTextView()
        scroll.borderType = .bezelBorder
        textView = scroll.documentView as? NSTextView
        textView.isRichText = false
        textView.font = .systemFont(ofSize: 16)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.isContinuousSpellCheckingEnabled = false
        textView.textContainerInset = NSSize(width: 10, height: 10)
        let views: [NSView] = [buttons, intro, statusLabel, scroll]
        for view in views {
            view.translatesAutoresizingMaskIntoConstraints = false
            content.addSubview(view)
        }
        NSLayoutConstraint.activate([
            buttons.topAnchor.constraint(equalTo: content.topAnchor, constant: 20),
            buttons.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            intro.topAnchor.constraint(equalTo: buttons.bottomAnchor, constant: 12),
            intro.leadingAnchor.constraint(equalTo: buttons.leadingAnchor),
            intro.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            scroll.topAnchor.constraint(equalTo: intro.bottomAnchor, constant: 14),
            scroll.leadingAnchor.constraint(equalTo: buttons.leadingAnchor),
            scroll.trailingAnchor.constraint(equalTo: intro.trailingAnchor),
            scroll.bottomAnchor.constraint(equalTo: statusLabel.topAnchor, constant: -12),
            statusLabel.leadingAnchor.constraint(equalTo: buttons.leadingAnchor),
            statusLabel.trailingAnchor.constraint(equalTo: intro.trailingAnchor),
            statusLabel.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -16)
        ])
    }

    @objc func showWindow() {
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }

    @objc func rebind() {
        guard !busy else { return }
        if let hotKey { UnregisterEventHotKey(hotKey) }
        if let clipboardHotKey { UnregisterEventHotKey(clipboardHotKey) }
        if let eventHandler { RemoveEventHandler(eventHandler) }
        hotKey = nil
        eventHandler = nil
        shortcut = "Control+Q"
        shortcutNote = ""
        configureHotKey()
        captureButton.title = "截图识字 · \(shortcut)"
        captureMenu.title = "截图识字（\(shortcut)）"
        statusItem.button?.toolTip = "本地截图识字 · \(shortcut)"
        statusLabel.stringValue = shortcutNote.isEmpty ? "已绑定 Control+Q。" : shortcutNote
        showWindow()
    }

    var launchAgentURL: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/LaunchAgents/local.yc.screenshot-ocr.plist")
    }

    var launchAtLoginEnabled: Bool { FileManager.default.fileExists(atPath: launchAgentURL.path) }

    @objc func toggleLaunchAtLogin() {
        let fm = FileManager.default
        if launchAtLoginEnabled {
            runLaunchctl(["bootout", "gui/\(getuid())", launchAgentURL.path])
            try? fm.removeItem(at: launchAgentURL)
        } else {
            let appPath = Bundle.main.bundlePath
            let xml = """
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
            <plist version="1.0"><dict>
            <key>Label</key><string>local.yc.screenshot-ocr</string>
            <key>ProgramArguments</key><array><string>\(appPath.replacingOccurrences(of: "&", with: "&amp;"))</string></array>
            <key>RunAtLoad</key><true/>
            <key>ProcessType</key><string>Interactive</string>
            </dict></plist>
            """
            do {
                try fm.createDirectory(at: launchAgentURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                try xml.write(to: launchAgentURL, atomically: true, encoding: .utf8)
                runLaunchctl(["bootstrap", "gui/\(getuid())", launchAgentURL.path])
            } catch { statusLabel.stringValue = "开机启动设置失败：\(error.localizedDescription)" }
        }
        if let item = statusItem.menu?.item(withTitle: "开机启动") { item.state = launchAtLoginEnabled ? .on : .off }
        statusLabel.stringValue = launchAtLoginEnabled ? "已设置开机启动。" : "已关闭开机启动。"
    }

    func runLaunchctl(_ arguments: [String]) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
        process.waitUntilExit()
    }

    @objc func openScreenSettings() {
        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!)
    }

    private func showScreenCapturePermissionAlert() {
        statusLabel.stringValue = "截图没有生成。请在「隐私与安全性 → 屏幕与系统音频录制」中允许本地截图识字，然后重试。"
        showWindow()
        let alert = NSAlert()
        alert.messageText = "需要截图权限"
        alert.informativeText = "请在「系统设置 → 隐私与安全性 → 屏幕与系统音频录制」中允许「本地截图识字」。授权后不需要重新配置快捷键。"
        alert.addButton(withTitle: "打开设置")
        alert.addButton(withTitle: "稍后")
        if alert.runModal() == .alertFirstButtonReturn { openScreenSettings() }
    }

    @objc func capture() {
        guard !busy else { return }
        busy = true
        captureButton.isEnabled = false
        let wasVisible = window.isVisible
        window.orderOut(nil)
        NSApp.hide(nil)
        statusItem.button?.title = "…"
        DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 0.3) {
            let file = FileManager.default.temporaryDirectory.appendingPathComponent("local-ocr-\(UUID().uuidString).png")
            defer { try? FileManager.default.removeItem(at: file) }
            do {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
                process.arguments = ["-i", "-s", "-x", "-t", "png", file.path]
                process.standardError = FileHandle.nullDevice
                try process.run()
                process.waitUntilExit()
                guard process.terminationStatus == 0, FileManager.default.fileExists(atPath: file.path) else {
                    DispatchQueue.main.async {
                        self.finish()
                        // A canceled interactive screencapture and a denied
                        // capture have the same exit status. Only show the
                        // permission prompt when the system still reports no
                        // access; otherwise this is just a normal cancel.
                        if !CGPreflightScreenCaptureAccess() {
                            self.showScreenCapturePermissionAlert()
                        } else {
                            self.statusLabel.stringValue = "截图已取消或未完成。"
                            if wasVisible { self.showWindow() }
                        }
                    }
                    return
                }
                let (text, engine) = try OCR.recognizeFile(file)
                DispatchQueue.main.async {
                    self.finish()
                    self.showResult(text)
                    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        self.statusLabel.stringValue = "没有识别到文字，请尝试框选更清晰或更大的区域。剪贴板保持原内容。"
                    } else {
                        let copied = self.writeClipboard(text)
                        self.statusLabel.stringValue = copied ? "已复制 \(text.count) 个字符（\(engine)），可直接粘贴。\(self.shortcut) 再次截图。" : "识别完成，但复制失败，请点击「复制文字」重试。"
                    }
                    self.showWindow()
                }
            } catch {
                DispatchQueue.main.async {
                    self.finish()
                    self.statusLabel.stringValue = "识别失败：\(error.localizedDescription)"
                    self.showWindow()
                }
            }
        }
    }

    @objc func recognizeClipboard() {
        guard !busy else { return }
        let pasteboard = NSPasteboard.general
        guard let image = pasteboard.readObjects(forClasses: [NSImage.self], options: nil)?.first as? NSImage else {
            showWindow()
            statusLabel.stringValue = "剪贴板里没有图片。先截图并复制图片，再按 Control+Option+Q。"
            return
        }
        var rect = NSRect(origin: .zero, size: image.size)
        guard let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
            showWindow()
            statusLabel.stringValue = "无法读取剪贴板图片。"
            return
        }
        busy = true
        captureButton.isEnabled = false
        NSApp.hide(nil)
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let (text, engine) = try OCR.recognize(cgImage)
                DispatchQueue.main.async {
                    self.finish()
                    self.showResult(text)
                    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        self.statusLabel.stringValue = "剪贴板图片里没有识别到文字。"
                    } else {
                        self.statusLabel.stringValue = self.writeClipboard(text) ? "已识别剪贴板图片，并复制文字（\(engine)）。" : "识别完成，但复制失败。"
                    }
                    self.showWindow()
                }
            } catch {
                DispatchQueue.main.async {
                    self.finish()
                    self.statusLabel.stringValue = "识别失败：\(error.localizedDescription)"
                    self.showWindow()
                }
            }
        }
    }

    func finish() {
        busy = false
        captureButton.isEnabled = true
        statusItem.button?.title = "文"
    }

    func writeClipboard(_ text: String) -> Bool {
        NSPasteboard.general.clearContents()
        return NSPasteboard.general.setString(text, forType: .string)
    }

    func showResult(_ text: String) {
        textView.string = text
        unmergedText = nil
        mergeButton.title = "合并成一段"
    }

    @objc func toggleMerge() {
        guard !textView.string.isEmpty else { return }
        if let original = unmergedText {
            textView.string = original
            unmergedText = nil
            mergeButton.title = "合并成一段"
        } else {
            unmergedText = textView.string
            textView.string = OCR.mergeIntoParagraph(textView.string)
            mergeButton.title = "恢复原换行"
        }
        let state = unmergedText == nil ? "已恢复原换行" : "已合并成一段"
        statusLabel.stringValue = writeClipboard(textView.string) ? "\(state)，并已复制。" : "\(state)，但复制失败，请点击「复制文字」。"
    }

    @objc func copyText() {
        guard !textView.string.isEmpty else { return }
        statusLabel.stringValue = writeClipboard(textView.string) ? "已复制。" : "复制失败，请重试。"
    }

    @objc func quit() { NSApp.terminate(nil) }

    func applicationWillTerminate(_ notification: Notification) {
        PaddleEngine.shared.stop()
        if let hotKey { UnregisterEventHotKey(hotKey) }
        if let clipboardHotKey { UnregisterEventHotKey(clipboardHotKey) }
        if let eventHandler { RemoveEventHandler(eventHandler) }
    }
}

func selfTest() throws {
    let image = NSImage(size: NSSize(width: 1100, height: 220))
    image.lockFocus()
    NSColor.white.setFill()
    NSRect(x: 0, y: 0, width: 1100, height: 220).fill()
    let attributes: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 42), .foregroundColor: NSColor.black]
    ("本地截图识字 中文测试" as NSString).draw(at: NSPoint(x: 35, y: 130), withAttributes: attributes)
    ("Hello OCR 12345" as NSString).draw(at: NSPoint(x: 35, y: 45), withAttributes: attributes)
    image.unlockFocus()
    var rect = NSRect(origin: .zero, size: image.size)
    guard let cg = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else { fatalError("Fixture creation failed") }
    let (text, engine) = try OCR.recognize(cg)
    guard text.contains("本地截图识字"), text.contains("Hello OCR 12345") else { fatalError("OCR self-test failed: \(text)") }
    let blank = NSImage(size: NSSize(width: 400, height: 150))
    blank.lockFocus()
    NSColor.white.setFill()
    NSRect(x: 0, y: 0, width: 400, height: 150).fill()
    blank.unlockFocus()
    var blankRect = NSRect(origin: .zero, size: blank.size)
    let blankText = try OCR.recognize(blank.cgImage(forProposedRect: &blankRect, context: nil, hints: nil)!).text
    guard blankText.isEmpty else { fatalError("Blank image produced text") }
    let merged = OCR.mergeIntoParagraph("当网关向上游转发请求时，\n客户端会断开。\n\nThe gateway returns\nHTTP 503.")
    guard merged == "当网关向上游转发请求时，客户端会断开。The gateway returns HTTP 503." else { fatalError("Merge failed: \(merged)") }
    print("PASS (\(engine)): Chinese, English, digits, blank-image recognition, and paragraph merge")
    print(text)
}

if CommandLine.arguments.contains("--self-test") {
    do { try selfTest() } catch { fputs("\(error)\n", stderr); exit(1) }
} else if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--recognize" {
    do { print(try OCR.recognizeFile(URL(fileURLWithPath: CommandLine.arguments[2])).text) }
    catch { fputs("\(error.localizedDescription)\n", stderr); exit(1) }
} else {
    let app = NSApplication.shared
    let delegate = AppDelegate()
    app.delegate = delegate
    app.run()
}
